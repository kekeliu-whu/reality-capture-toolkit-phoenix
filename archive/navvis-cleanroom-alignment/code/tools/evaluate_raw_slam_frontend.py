#!/usr/bin/env python3
"""Run raw dual-lidar/IMU frontend regression against serialized NavVis nodes."""

from __future__ import annotations

import argparse
import faulthandler
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.surveyor_frontend import (  # noqa: E402
    RawConstantVelocityPosePredictor,
    RawImuTracker,
    SlamScanArchive,
    SurveyorFrontend,
    iter_archive_at_node_times,
)
from navvis_recon.surveyor_slam import (  # noqa: E402
    Rigid3,
    load_imu_rosbag,
    load_submaps,
    load_trajectory_nodes,
)
from slam_state_io import online_stage1_arrays  # noqa: E402


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def relative_pose_statistics(
    estimate,
    reference,
    pairs: list[tuple[int, int]],
) -> dict[str, object]:
    translation_mm: list[float] = []
    rotation_deg: list[float] = []
    for first, second in pairs:
        estimate_relative = estimate[first].local_pose.between(
            estimate[second].local_pose
        )
        reference_relative = reference[first].local_pose.between(
            reference[second].local_pose
        )
        error = reference_relative.inverse().compose(estimate_relative)
        translation_mm.append(float(np.linalg.norm(error.translation) * 1000.0))
        rotation_deg.append(float(np.degrees(error.rotation.magnitude())))
    if not pairs:
        return {"pair_count": 0}
    return {
        "pair_count": len(pairs),
        "translation_mm": statistics(np.asarray(translation_mm)),
        "rotation_deg": statistics(np.asarray(rotation_deg)),
    }


def main() -> int:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--imu-bag", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--submaps", type=Path)
    parser.add_argument("--node-limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--state-output",
        type=Path,
        help="Write the generated node/submap poses and topology to a compressed NPZ",
    )
    parser.add_argument("--find-loops", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress after every N retained nodes (0 disables it)",
    )
    parser.add_argument("--online-pose-graph-solver", type=Path)
    parser.add_argument("--online-pose-graph-work-dir", type=Path)
    args = parser.parse_args()

    reference = load_trajectory_nodes(
        args.nodes,
        limit=args.node_limit if args.node_limit > 0 else None,
    )
    reference_submaps = load_submaps(args.submaps) if args.submaps else ()
    if len(reference) < 2:
        raise ValueError("at least two reference nodes are required")
    imu = load_imu_rosbag(
        args.imu_bag,
        end_timestamp_ns=(
            reference[-1].timestamp_ns if args.node_limit > 0 else None
        ),
    )

    with SlamScanArchive(args.archive) as archive:
        first_by_sensor: dict[int, int] = {}
        for record in archive.records:
            first_by_sensor.setdefault(
                record.sensor, (record.timestamp_ns // 1000) * 1000
            )
        if len(first_by_sensor) != 2:
            raise ValueError("a dual-laser archive is required")
        first_all_sources_ns = max(first_by_sensor.values())
        tracker = RawImuTracker(imu)
        first_node_ns = reference[0].timestamp_ns
        # A trajectory node may lie a few milliseconds inside the first
        # collated laser revolution, so its timestamp can precede the scan
        # header chosen as the dual-sensor start.  Advance the tracker in
        # chronological order instead of assuming the scan header is first.
        if first_node_ns < first_all_sources_ns:
            tracker_at_first_node = tracker.advance(first_node_ns)
            tracker_at_start = tracker.advance(first_all_sources_ns)
        else:
            tracker_at_start = tracker.advance(first_all_sources_ns)
            tracker_at_first_node = tracker.advance(first_node_ns)
        initial_pose = Rigid3(
            reference[0].local_pose.translation.copy(),
            (
                reference[0].local_pose.rotation
                * (tracker_at_start.inv() * tracker_at_first_node).inv()
            ).as_quat(),
        )
        predictor = RawConstantVelocityPosePredictor(
            imu, initial_pose=initial_pose
        )
        scans = iter(
            iter_archive_at_node_times(
            archive,
            [node.timestamp_ns for node in reference],
            imu_pose_predictor=predictor,
            )
        )
        # Range deskew advances the raw IMU tracker at every unique ray stamp.
        # Its gravity blend is intentionally interval-sensitive, so a direct
        # start-to-node advance differs slightly from the exact ray-query
        # sequence.  Consume the first batch, then close the local trajectory
        # gauge at the first retained node.  This changes no relative motion
        # or scan data; it only removes an otherwise artificial initial-frame
        # rotation from every regression error.
        first_scan = next(scans)
        predictor.correct(first_scan.timestamp_ns, reference[0].local_pose)
        scans = itertools.chain((first_scan,), scans)
        if args.progress_every > 0:
            source_scans = scans

            def scans_with_progress():
                retained = 0
                total = 0
                for value in source_scans:
                    total += 1
                    if value.retain_node:
                        retained += 1
                        if retained % args.progress_every == 0:
                            print(
                                f"SLAM frontend progress: {retained}/{len(reference)} "
                                f"retained nodes ({total} collated batches)",
                                flush=True,
                            )
                    yield value

            scans = scans_with_progress()
        start = time.perf_counter()
        result = SurveyorFrontend().process(
            scans,
            imu_pose_predictor=predictor,
            find_loops=args.find_loops,
            online_pose_graph_solver=args.online_pose_graph_solver,
            online_pose_graph_work_directory=args.online_pose_graph_work_dir,
        )
        elapsed = time.perf_counter() - start

    translation_mm = np.asarray(
        [
            np.linalg.norm(node.local_pose.translation - expected.local_pose.translation)
            * 1000.0
            for node, expected in zip(result.nodes, reference)
        ]
    )
    rotation_deg = np.asarray(
        [
            np.degrees(
                (node.local_pose.rotation.inv() * expected.local_pose.rotation).magnitude()
            )
            for node, expected in zip(result.nodes, reference)
        ]
    )
    adjacent_pairs = [(index - 1, index) for index in range(1, len(reference))]
    timestamps_ns = np.asarray(
        [node.timestamp_ns for node in reference], dtype=np.int64
    )
    one_second_pairs: list[tuple[int, int]] = []
    for first, timestamp_ns in enumerate(timestamps_ns):
        second = int(np.searchsorted(timestamps_ns, timestamp_ns + 1_000_000_000))
        if second >= len(timestamps_ns):
            continue
        if abs(int(timestamps_ns[second] - timestamp_ns) - 1_000_000_000) <= 30_000_000:
            one_second_pairs.append((first, second))
    blocks = []
    for begin in range(0, len(reference), 100):
        end = min(begin + 100, len(reference))
        blocks.append(
            {
                "begin": begin,
                "end": end,
                "translation_mm": statistics(translation_mm[begin:end]),
                "rotation_deg": statistics(rotation_deg[begin:end]),
            }
        )
    submap_details = []
    for index, submap in enumerate(result.submaps):
        surfel_level_counts = [
            int(len(submap.cloud_level(level)[0])) for level in range(3)
        ]
        detail = {
            "index": submap.submap_id.index,
            "start_timestamp_ns": submap.start_timestamp_ns,
            "end_timestamp_ns": submap.end_timestamp_ns,
            "member_count": len(submap.node_indices),
            "first_node": submap.node_indices[0] if submap.node_indices else None,
            "last_node": submap.node_indices[-1] if submap.node_indices else None,
            "surfel_level_counts": surfel_level_counts,
            "hybrid_grid_cells": submap.hybrid_grid.cell_count,
        }
        if index < len(reference_submaps):
            expected = reference_submaps[index]
            detail["reference"] = {
                "start_timestamp_exact": (
                    submap.start_timestamp_ns == expected.start_timestamp_ns
                ),
                "end_timestamp_exact": (
                    submap.end_timestamp_ns == expected.end_timestamp_ns
                ),
                "members_exact": tuple(submap.node_indices) == expected.node_indices,
                "translation_mm": float(
                    np.linalg.norm(
                        submap.local_pose.translation
                        - expected.local_pose.translation
                    )
                    * 1000.0
                ),
                "rotation_deg": float(
                    np.degrees(
                        (
                            submap.local_pose.rotation.inv()
                            * expected.local_pose.rotation
                        ).magnitude()
                    )
                ),
            }
        submap_details.append(detail)
    report = {
        "archive": str(args.archive.resolve()),
        "imu_bag": str(args.imu_bag.resolve()),
        "nodes": str(args.nodes.resolve()),
        "node_count": len(reference),
        "elapsed_seconds": elapsed,
        "nodes_per_second": len(reference) / elapsed,
        "translation_mm": statistics(translation_mm),
        "rotation_deg": statistics(rotation_deg),
        "ate": {
            "translation_mm": statistics(translation_mm),
            "rotation_deg": statistics(rotation_deg),
        },
        "rpe_adjacent": relative_pose_statistics(
            result.nodes, reference, adjacent_pairs
        ),
        "rpe_1s": relative_pose_statistics(
            result.nodes, reference, one_second_pairs
        ),
        "submap_count": len(result.submaps),
        "submap_memberships": sum(len(value.node_indices) for value in result.submaps),
        "loop_count": len(result.loops),
        "loops": [
            {
                "submap": loop.submap_id.index,
                "node": loop.node_id.index,
                "translation_m": loop.submap_from_node.translation.tolist(),
                "quaternion_xyzw": loop.submap_from_node.quaternion_xyzw.tolist(),
            }
            for loop in result.loops
            if loop.valid
        ],
        "node_errors": [
            {
                "index": index,
                "timestamp_ns": int(node.timestamp_ns),
                "translation_mm": float(translation_mm[index]),
                "rotation_deg": float(rotation_deg[index]),
                "icp_correspondences": (
                    int(node.scan_match.correspondences)
                    if node.scan_match is not None
                    else 0
                ),
                "icp_iterations": (
                    int(node.scan_match.iterations)
                    if node.scan_match is not None
                    else 0
                ),
                "icp_fitness_m": (
                    float(node.scan_match.fitness_m)
                    if node.scan_match is not None
                    else None
                ),
            }
            for index, node in enumerate(result.nodes)
        ],
        "submaps": submap_details,
        "blocks": blocks,
    }
    encoded = json.dumps(report, indent=2)
    if args.state_output:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        _, backend_submaps, _ = result.backend_inputs()
        if len(backend_submaps) != len(result.submaps):
            raise RuntimeError("frontend/backend Submap count mismatch")
        membership_offsets = [0]
        membership_indices: list[int] = []
        for submap in result.submaps:
            membership_indices.extend(submap.node_indices)
            membership_offsets.append(len(membership_indices))
        loop_rows = np.asarray(
            [
                (
                    loop.submap_id.index,
                    loop.node_id.index,
                    *loop.submap_from_node.translation,
                    *loop.submap_from_node.quaternion_xyzw,
                    loop.translation_weight,
                    loop.rotation_weight,
                    float(loop.valid),
                    loop.tag,
                )
                for loop in result.loops
            ],
            dtype=np.float64,
        ).reshape((-1, 13))
        surfel_points: list[np.ndarray] = []
        surfel_normals: list[np.ndarray] = []
        surfel_offsets = [0]
        hybrid_indices: list[np.ndarray] = []
        hybrid_values: list[np.ndarray] = []
        hybrid_offsets = [0]
        for submap in result.submaps:
            for level in range(3):
                points, normals = submap.cloud_level(level)
                surfel_points.append(points)
                surfel_normals.append(normals)
                surfel_offsets.append(surfel_offsets[-1] + len(points))
            indices, values = submap.hybrid_grid.export_cells()
            hybrid_indices.append(indices)
            hybrid_values.append(values)
            hybrid_offsets.append(hybrid_offsets[-1] + len(indices))
        online_arrays: dict[str, np.ndarray] = {}
        state_schema_version = 2
        if result.online_fast_pose_graph is not None:
            online_arrays = online_stage1_arrays(
                result.online_fast_pose_graph, backend_submaps
            )
            state_schema_version = 3
        np.savez_compressed(
            args.state_output,
            state_schema_version=np.asarray(state_schema_version, dtype=np.int64),
            node_timestamps_ns=np.asarray(
                [node.timestamp_ns for node in result.nodes], dtype=np.int64
            ),
            node_translations=np.vstack(
                [node.local_pose.translation for node in result.nodes]
            ),
            node_quaternions_xyzw=np.vstack(
                [node.local_pose.quaternion_xyzw for node in result.nodes]
            ),
            submap_start_timestamps_ns=np.asarray(
                [submap.start_timestamp_ns for submap in result.submaps],
                dtype=np.int64,
            ),
            submap_end_timestamps_ns=np.asarray(
                [submap.end_timestamp_ns for submap in result.submaps], dtype=np.int64
            ),
            submap_translations=np.vstack(
                [submap.local_pose.translation for submap in result.submaps]
            ),
            submap_quaternions_xyzw=np.vstack(
                [submap.local_pose.quaternion_xyzw for submap in result.submaps]
            ),
            submap_membership_offsets=np.asarray(membership_offsets, dtype=np.int64),
            submap_membership_indices=np.asarray(membership_indices, dtype=np.int64),
            submap_finished=np.asarray(
                [submap.finished for submap in backend_submaps], dtype=np.bool_
            ),
            submap_gravity_observations=np.vstack(
                [submap.gravity_observation for submap in backend_submaps]
            ),
            submap_hybrid_grid_cell_counts=np.asarray(
                [submap.hybrid_grid.cell_count for submap in result.submaps],
                dtype=np.int64,
            ),
            submap_hybrid_grid_offsets=np.asarray(hybrid_offsets, dtype=np.int64),
            submap_hybrid_grid_indices=np.concatenate(hybrid_indices),
            submap_hybrid_grid_values=np.concatenate(hybrid_values),
            submap_surfel_offsets=np.asarray(surfel_offsets, dtype=np.int64),
            submap_surfel_points=np.concatenate(surfel_points),
            submap_surfel_normals=np.concatenate(surfel_normals),
            loops=loop_rows,
            **online_arrays,
        )
        report["state_output"] = str(args.state_output)
        report["state_output_schema"] = {
            "version": state_schema_version,
            "generated_submap_fields": [
                "submap_finished",
                "submap_gravity_observations",
                "submap_hybrid_grid_cell_counts",
                "submap_hybrid_grid_offsets",
                "submap_hybrid_grid_indices",
                "submap_hybrid_grid_values",
            ],
            "generated_online_stage1": bool(online_arrays),
            "online_stage1_fields": sorted(online_arrays),
        }
        encoded = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
