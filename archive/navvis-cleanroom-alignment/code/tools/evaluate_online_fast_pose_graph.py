#!/usr/bin/env python3
"""Evaluate the complete online Fast-IMU pose-graph schedule.

The clean solve finishes before the official optimization state is loaded.
Reference data is therefore used only to report numerical alignment, never to
initialize or alter the generated trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_complete_slam import load_frontend_state  # noqa: E402
from navvis_recon.surveyor_slam import (  # noqa: E402
    NodeId,
    Rigid3,
    Submap,
    finish_online_fast_pose_graph,
    load_loop_constraints,
    load_optimization_imu,
    load_optimization_state,
    load_submaps,
    load_trajectory_nodes,
    replay_online_fast_pose_graph,
)
from slam_state_io import ONLINE_STAGE1_ARRAY_NAMES, online_stage1_arrays  # noqa: E402


def _dataset_paths(dataset: Path) -> dict[str, Path]:
    return {
        "nodes": dataset
        / "internal/nodes/trajectory_node/trajectory_node_00000000.zip",
        "submaps": dataset / "internal/submaps/submap/submap_00000000.zip",
        "loops": dataset
        / "internal/constraints_inter_dataset"
        / dataset.name
        / "constraints/constraint_data/constraint_data_00000000.zip",
        "optimization_data": dataset / "artifacts/optimization_data.pb",
        "optimization_state": dataset / "artifacts/optimization_state.pb",
    }


def _load_transient_submap(path: Path) -> Submap:
    with np.load(path, allow_pickle=False) as state:
        return Submap(
            NodeId(0, 2),
            int(state["start_timestamp_ns"]),
            int(state["end_timestamp_ns"]),
            Rigid3(state["translation"], state["quaternion_xyzw"]),
            tuple(int(index) for index in state["node_indices"]),
            True,
            np.asarray(state["gravity_observation"], dtype=np.float64),
        )


def _pose_errors(
    generated: tuple[Rigid3, ...], reference: tuple[Rigid3, ...]
) -> dict[str, float | int]:
    if len(generated) != len(reference):
        raise ValueError("generated and reference pose counts differ")
    translations = np.asarray(
        [
            np.linalg.norm(left.translation - right.translation)
            for left, right in zip(generated, reference)
        ],
        dtype=np.float64,
    )
    rotations = np.asarray(
        [
            np.degrees((left.rotation.inv() * right.rotation).magnitude())
            for left, right in zip(generated, reference)
        ],
        dtype=np.float64,
    )
    return {
        "translation_mean_m": float(np.mean(translations)),
        "translation_max_m": float(np.max(translations)),
        "translation_max_index": int(np.argmax(translations)),
        "rotation_mean_deg": float(np.mean(rotations)),
        "rotation_max_deg": float(np.max(rotations)),
        "rotation_max_index": int(np.argmax(rotations)),
    }


def _official_costs(path: Path) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    expression = re.compile(
        r"CERES_RETURN call=(\d+).*initial_cost=([^ ]+) final_cost=([^ ]+)"
    )
    calls: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = expression.search(line)
        if match:
            calls[int(match.group(1))] = (
                float(match.group(2)),
                float(match.group(3)),
            )
    periodic = [calls[index] for index in range(1, 16, 2)]
    return periodic, calls[17]


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    dataset = args.dataset.resolve()
    paths = _dataset_paths(dataset)
    frozen_nodes = load_trajectory_nodes(paths["nodes"])
    frozen_submaps = load_submaps(paths["submaps"])
    state = load_frontend_state(args.frontend_state.resolve(), frozen_nodes, frozen_submaps)
    samples = load_optimization_imu(paths["optimization_data"])
    loops = (
        state.loops
        if args.loop_source == "frontend"
        else load_loop_constraints(paths["loops"])
    )
    submaps = state.submaps
    if args.transient_submap is not None:
        submaps += (_load_transient_submap(args.transient_submap.resolve()),)

    periodic = replay_online_fast_pose_graph(
        state.nodes,
        state.submaps,
        samples,
        args.ceres_solver.resolve(),
        args.work_dir.resolve() / "periodic",
        optimize_every_n_nodes=args.optimize_every,
        max_iterations=args.periodic_iterations,
        num_threads=args.solver_threads,
    )
    finish = finish_online_fast_pose_graph(
        state.nodes,
        submaps,
        samples,
        loops,
        periodic,
        args.ceres_solver.resolve(),
        args.work_dir.resolve() / "finish",
        max_iterations=args.finish_iterations,
        num_threads=args.solver_threads,
    )

    generated_state_output = None
    if args.stage1_state_output is not None:
        output = args.stage1_state_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with np.load(args.frontend_state.resolve(), allow_pickle=False) as source:
            output_arrays = {
                name: np.asarray(source[name])
                for name in source.files
                if name not in ONLINE_STAGE1_ARRAY_NAMES
            }
        output_arrays["state_schema_version"] = np.asarray(3, dtype=np.int64)
        output_arrays.update(online_stage1_arrays(finish, state.submaps))
        np.savez_compressed(output, **output_arrays)
        generated_state_output = str(output)

    periodic_rows: list[dict[str, object]] = [
        {
            "nodes": snapshot.node_count,
            "submaps": len(snapshot.submaps),
            "initial_cost": snapshot.result.initial_cost,
            "final_cost": snapshot.result.final_cost,
            "iterations": snapshot.result.iterations,
        }
        for snapshot in periodic
    ]
    finish_row: dict[str, object] = {
        "nodes": finish.node_count,
        "submaps": len(finish.submaps),
        "initial_cost": finish.result.initial_cost,
        "final_cost": finish.result.final_cost,
        "iterations": finish.result.iterations,
    }
    if args.vendor_trace is not None:
        official_periodic, official_finish = _official_costs(args.vendor_trace.resolve())
        if len(official_periodic) != len(periodic_rows):
            raise ValueError("vendor trace periodic solve count differs")
        for row, official in zip(periodic_rows, official_periodic):
            row["official_initial_cost"] = official[0]
            row["official_final_cost"] = official[1]
            row["initial_cost_error"] = float(row["initial_cost"]) - official[0]
            row["final_cost_error"] = float(row["final_cost"]) - official[1]
        finish_row["official_initial_cost"] = official_finish[0]
        finish_row["official_final_cost"] = official_finish[1]
        finish_row["initial_cost_error"] = (
            float(finish_row["initial_cost"]) - official_finish[0]
        )
        finish_row["final_cost_error"] = (
            float(finish_row["final_cost"]) - official_finish[1]
        )

    # Load the official output only after every clean-room solve is complete.
    reference = load_optimization_state(paths["optimization_state"])
    node_count = len(state.nodes)
    retained_submap_count = len(reference.submap_poses)
    return {
        "scope": "online Fast-IMU schedule plus FinishTrajectory",
        "loop_source": args.loop_source,
        "generated_stage1_state": generated_state_output,
        "periodic": periodic_rows,
        "finish": finish_row,
        "node_alignment": _pose_errors(
            finish.result.poses[:node_count], reference.poses
        ),
        "retained_submap_alignment": _pose_errors(
            finish.result.poses[node_count : node_count + retained_submap_count],
            reference.submap_poses,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frontend-state", type=Path, required=True)
    parser.add_argument("--transient-submap", type=Path)
    parser.add_argument("--ceres-solver", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--vendor-trace", type=Path)
    parser.add_argument(
        "--stage1-state-output",
        type=Path,
        help="Write a schema-v3 frontend state containing this clean Stage1 result",
    )
    parser.add_argument(
        "--loop-source", choices=("frontend", "archive"), default="frontend"
    )
    parser.add_argument("--optimize-every", type=int, default=321)
    parser.add_argument("--periodic-iterations", type=int, default=10)
    parser.add_argument("--finish-iterations", type=int, default=200)
    parser.add_argument("--solver-threads", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
