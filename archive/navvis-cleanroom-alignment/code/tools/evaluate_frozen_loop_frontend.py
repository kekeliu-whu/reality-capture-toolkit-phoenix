#!/usr/bin/env python3
"""Evaluate the clean-room loop frontend on frozen node/submap scan data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.surveyor_frontend import (  # noqa: E402
    FrontendNode,
    FrontendSubmap,
    detect_loop_constraints,
    load_trajectory_node_clouds,
    parse_submap_hybrid_grid,
    parse_submap_surfel_cloud,
)
from navvis_recon.surveyor_slam import (  # noqa: E402
    LoopConstraint,
    load_loop_constraints,
    load_optimization_imu,
    load_submaps,
    load_trajectory_nodes,
    online_fast_loop_initial_pose,
    replay_online_fast_pose_graph,
)


def _statistics(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _pair(loop: LoopConstraint) -> tuple[int, int]:
    return loop.submap_id.index, loop.node_id.index


def evaluate(
    dataset: Path,
    *,
    online_solver: Path | None = None,
    online_work_dir: Path | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    nodes_path = dataset / "internal/nodes/trajectory_node/trajectory_node_00000000.zip"
    node_clouds_path = (
        dataset
        / "internal/nodes/trajectory_node_clouds/trajectory_node_clouds_00000000.zip"
    )
    submaps_path = dataset / "internal/submaps/submap/submap_00000000.zip"
    submap_clouds_path = (
        dataset / "internal/submaps/submap_clouds/submap_clouds_00000000.zip"
    )
    loops_path = (
        dataset
        / "internal/constraints_inter_dataset"
        / dataset.name
        / "constraints/constraint_data/constraint_data_00000000.zip"
    )
    optimization_data_path = dataset / "artifacts/optimization_data.pb"
    inputs = (
        nodes_path,
        node_clouds_path,
        submaps_path,
        submap_clouds_path,
        loops_path,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen loop inputs: " + ", ".join(missing))

    frozen_nodes = load_trajectory_nodes(nodes_path)
    frozen_submaps = load_submaps(submaps_path)
    frozen_loops = tuple(
        loop for loop in load_loop_constraints(loops_path) if loop.valid
    )
    node_clouds = load_trajectory_node_clouds(node_clouds_path)
    frontend_nodes = tuple(
        FrontendNode(
            node.node_id,
            node.timestamp_ns,
            node.local_pose,
            node_clouds[node.node_id.index],
            np.zeros_like(node_clouds[node.node_id.index]),
            None,
            node.gravity_observation,
        )
        for node in frozen_nodes
    )

    frontend_submaps: list[FrontendSubmap] = []
    try:
        with zipfile.ZipFile(submap_clouds_path) as archive:
            for submap in frozen_submaps:
                payload = archive.read(f"submap_{submap.submap_id.index:08d}.pb")
                frontend = FrontendSubmap(
                    submap.submap_id,
                    submap.local_pose,
                    submap.start_timestamp_ns,
                    0.0,
                    list(submap.node_indices),
                    submap.end_timestamp_ns,
                    submap.finished,
                )
                frontend._cached_levels = [
                    parse_submap_surfel_cloud(payload, level) for level in range(3)
                ]
                frontend._cached_points, frontend._cached_normals = (
                    frontend._cached_levels[0]
                )
                frontend._hybrid_grid.close()
                frontend._hybrid_grid = parse_submap_hybrid_grid(payload)
                frontend_submaps.append(frontend)

        online_snapshots = ()
        initial_pose_for_pair = None
        if online_solver is not None:
            if online_work_dir is None:
                raise ValueError("online work directory is required with the solver")
            online_snapshots = replay_online_fast_pose_graph(
                frozen_nodes,
                frozen_submaps,
                load_optimization_imu(optimization_data_path),
                online_solver,
                online_work_dir,
            )

            def initial_pose_for_pair(submap, node):
                return online_fast_loop_initial_pose(
                    online_snapshots,
                    frozen_nodes,
                    frozen_submaps,
                    submap.submap_id,
                    node.node_id,
                )

        match_start = time.perf_counter()
        actual_loops = detect_loop_constraints(
            frontend_nodes,
            frontend_submaps,
            initial_pose_for_pair=initial_pose_for_pair,
        )
        match_seconds = time.perf_counter() - match_start
        expected_by_pair = {_pair(loop): loop for loop in frozen_loops}
        actual_by_pair = {_pair(loop): loop for loop in actual_loops if loop.valid}
        expected_pairs = set(expected_by_pair)
        actual_pairs = set(actual_by_pair)
        common = sorted(expected_pairs & actual_pairs)
        translation = np.asarray(
            [
                np.linalg.norm(
                    actual_by_pair[pair].submap_from_node.translation
                    - expected_by_pair[pair].submap_from_node.translation
                )
                for pair in common
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(
            [
                np.degrees(
                    (
                        actual_by_pair[pair].submap_from_node.rotation.inv()
                        * expected_by_pair[pair].submap_from_node.rotation
                    ).magnitude()
                )
                for pair in common
            ],
            dtype=np.float64,
        )
        return {
            "dataset": str(dataset),
            "counts": {
                "nodes": len(frontend_nodes),
                "submaps": len(frontend_submaps),
                "sampled_candidates": 41,
                "expected_loops": len(expected_pairs),
                "actual_loops": len(actual_pairs),
            },
            "online_pose_graph": {
                "enabled": online_solver is not None,
                "solve_node_counts": [
                    snapshot.node_count for snapshot in online_snapshots
                ],
            },
            "pairs": {
                "expected": [list(pair) for pair in sorted(expected_pairs)],
                "actual": [list(pair) for pair in sorted(actual_pairs)],
                "false_positive": [list(pair) for pair in sorted(actual_pairs - expected_pairs)],
                "false_negative": [list(pair) for pair in sorted(expected_pairs - actual_pairs)],
                "exact": actual_pairs == expected_pairs,
            },
            "matched_measurement_error": {
                "translation_m": _statistics(translation),
                "rotation_deg": _statistics(rotation),
            },
            "timing_seconds": {
                "constraint_detection": match_seconds,
                "total": time.perf_counter() - start,
            },
        }
    finally:
        for submap in frontend_submaps:
            submap.hybrid_grid.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--online-solver", type=Path)
    parser.add_argument("--online-work-dir", type=Path)
    args = parser.parse_args()
    report = evaluate(
        args.dataset.resolve(),
        online_solver=(
            args.online_solver.resolve() if args.online_solver is not None else None
        ),
        online_work_dir=(
            args.online_work_dir.resolve()
            if args.online_work_dir is not None
            else None
        ),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
