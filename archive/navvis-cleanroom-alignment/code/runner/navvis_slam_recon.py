#!/usr/bin/env python3
"""Run generated Frontend -> loop closure -> Stage1 -> Stage2 SLAM.

Reference artifacts are optional and are loaded only after both clean-room
solves complete.  They can therefore report alignment but cannot initialize
or alter the generated trajectory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.autonomous_slam import load_native_frontend_state  # noqa: E402
from navvis_recon.surveyor_frontend import (  # noqa: E402
    FrontendNode,
    FrontendSubmap,
    detect_loop_constraints,
)
from navvis_recon.surveyor_slam import (  # noqa: E402
    ImuCalibration,
    ImuIntrinsics,
    LoopConstraint,
    OptimizationState,
    Rigid3,
    Submap,
    TrajectoryNode,
    build_imu_pose_graph,
    build_pose_graph,
    finish_online_fast_pose_graph,
    load_imu_rosbag,
    load_loop_constraints,
    load_optimization_state,
    online_fast_loop_initial_pose,
    optimize_imu_pose_graph_ceres,
    replay_online_fast_pose_graph,
)


def _statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _stage1_initial_state(
    nodes: Sequence[TrajectoryNode],
    retained_submaps: Sequence[Submap],
    snapshot,
) -> OptimizationState:
    node_count = len(nodes)
    if snapshot.node_count != node_count:
        raise ValueError("Stage1 result node count differs from frontend")
    submap_pose_by_id = {
        submap.submap_id: snapshot.result.poses[node_count + index]
        for index, submap in enumerate(snapshot.submaps)
    }
    missing = [
        submap.submap_id
        for submap in retained_submaps
        if submap.submap_id not in submap_pose_by_id
    ]
    if missing:
        raise ValueError(f"Stage1 result is missing retained Submaps: {missing}")
    calibration: ImuCalibration = snapshot.result.calibration
    return OptimizationState(
        timestamps_ns=np.asarray(
            [node.timestamp_ns for node in nodes], dtype=np.int64
        ),
        poses=tuple(snapshot.result.poses[:node_count]),
        velocities=np.zeros((node_count, 3), dtype=np.float64),
        submap_poses=tuple(
            submap_pose_by_id[submap.submap_id] for submap in retained_submaps
        ),
        gravity_magnitude=calibration.gravity_magnitude,
        imu_from_tracking=calibration.imu_from_tracking,
        linear_acceleration_intrinsics=ImuIntrinsics(
            calibration.linear_acceleration_bias.copy(),
            calibration.linear_acceleration_scaling.copy(),
            calibration.linear_acceleration_cross_axis.copy(),
        ),
        angular_velocity_intrinsics=ImuIntrinsics(
            calibration.angular_velocity_bias.copy(),
            calibration.angular_velocity_scaling.copy(),
            calibration.angular_velocity_cross_axis.copy(),
        ),
    )


def _interpolate_reference(
    reference: OptimizationState, timestamps_ns: np.ndarray
) -> tuple[Rigid3, ...]:
    source_ns = np.asarray(reference.timestamps_ns, dtype=np.int64)
    if len(source_ns) < 2 or np.any(np.diff(source_ns) <= 0):
        raise ValueError("reference timestamps must be strictly increasing")
    if timestamps_ns[0] < source_ns[0] or timestamps_ns[-1] > source_ns[-1]:
        raise ValueError("generated trajectory extends outside reference support")
    origin_ns = int(source_ns[0])
    source_s = (source_ns - origin_ns).astype(np.float64) / 1.0e9
    query_s = (timestamps_ns - origin_ns).astype(np.float64) / 1.0e9
    translations = np.column_stack(
        [
            np.interp(
                query_s,
                source_s,
                np.asarray([pose.translation[axis] for pose in reference.poses]),
            )
            for axis in range(3)
        ]
    )
    rotations = Slerp(
        source_s,
        Rotation.from_quat(
            np.vstack([pose.quaternion_xyzw for pose in reference.poses])
        ),
    )(query_s).as_quat()
    return tuple(
        Rigid3(translation, quaternion)
        for translation, quaternion in zip(translations, rotations)
    )


def _gauge_align(
    estimate: Sequence[Rigid3], reference: Sequence[Rigid3]
) -> tuple[tuple[Rigid3, ...], Rigid3]:
    if len(estimate) != len(reference) or not estimate:
        raise ValueError("trajectory alignment requires equal non-empty sequences")
    gauge = reference[0].compose(estimate[0].inverse())
    return tuple(gauge.compose(pose) for pose in estimate), gauge


def _trajectory_metrics(
    estimate: Sequence[Rigid3],
    reference: Sequence[Rigid3],
    timestamps_ns: np.ndarray,
    *,
    rpe_seconds: float = 1.0,
    rpe_tolerance_ms: float = 30.0,
) -> dict[str, object]:
    aligned, gauge = _gauge_align(estimate, reference)
    translation = np.asarray(
        [
            np.linalg.norm(actual.translation - expected.translation)
            for actual, expected in zip(aligned, reference)
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [
            np.degrees((actual.rotation.inv() * expected.rotation).magnitude())
            for actual, expected in zip(aligned, reference)
        ],
        dtype=np.float64,
    )
    interval_ns = int(round(rpe_seconds * 1.0e9))
    tolerance_ns = int(round(rpe_tolerance_ms * 1.0e6))
    relative_translation: list[float] = []
    relative_rotation: list[float] = []
    for first, timestamp in enumerate(timestamps_ns):
        second = int(np.searchsorted(timestamps_ns, int(timestamp) + interval_ns))
        if second >= len(timestamps_ns):
            continue
        if abs(int(timestamps_ns[second] - timestamp) - interval_ns) > tolerance_ns:
            continue
        estimate_relative = aligned[first].between(aligned[second])
        reference_relative = reference[first].between(reference[second])
        error = reference_relative.inverse().compose(estimate_relative)
        relative_translation.append(float(np.linalg.norm(error.translation)))
        relative_rotation.append(float(np.degrees(error.rotation.magnitude())))
    return {
        "gauge": {
            "translation_m": gauge.translation.tolist(),
            "quaternion_xyzw": gauge.quaternion_xyzw.tolist(),
        },
        "ate": {
            "pose_count": len(aligned),
            "translation_m": _statistics(translation),
            "rotation_deg": _statistics(rotation),
        },
        "rpe": {
            "interval_seconds": rpe_seconds,
            "tolerance_ms": rpe_tolerance_ms,
            "pair_count": len(relative_translation),
            "translation_m": _statistics(np.asarray(relative_translation)),
            "rotation_deg": _statistics(np.asarray(relative_rotation)),
        },
    }


def _loop_metrics(
    generated: Sequence[LoopConstraint], reference: Sequence[LoopConstraint]
) -> dict[str, object]:
    generated_pairs = {
        (loop.submap_id.index, loop.node_id.index)
        for loop in generated
        if loop.valid
    }
    reference_pairs = {
        (loop.submap_id.index, loop.node_id.index)
        for loop in reference
        if loop.valid
    }
    return {
        "generated_valid": len(generated_pairs),
        "reference_valid": len(reference_pairs),
        "pair_true_positive": len(generated_pairs & reference_pairs),
        "pair_false_positive": len(generated_pairs - reference_pairs),
        "pair_false_negative": len(reference_pairs - generated_pairs),
    }


def _write_trajectory(
    path: Path, timestamps_ns: np.ndarray, poses: Sequence[Rigid3]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"))
        for index, (timestamp_ns, pose) in enumerate(zip(timestamps_ns, poses)):
            writer.writerow(
                (
                    index,
                    int(timestamp_ns),
                    *[format(float(value), ".17g") for value in pose.translation],
                    *[
                        format(float(value), ".17g")
                        for value in pose.quaternion_xyzw
                    ],
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run autonomous clean-room loop/Stage1/Stage2 SLAM"
    )
    parser.add_argument("--frontend-state", type=Path, required=True)
    parser.add_argument("--imu-bag", type=Path, required=True)
    parser.add_argument("--stage1-solver", type=Path, required=True)
    parser.add_argument("--stage2-solver", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--reference-loops", type=Path)
    parser.add_argument("--skip-loop-closures", action="store_true")
    parser.add_argument("--periodic-stage1", action="store_true")
    parser.add_argument("--stage1-period", type=int, default=321)
    parser.add_argument("--stage1-periodic-iterations", type=int, default=10)
    parser.add_argument("--stage1-finish-iterations", type=int, default=200)
    parser.add_argument("--stage2-iterations", type=int, default=200)
    parser.add_argument("--solver-threads", type=int, default=7)
    parser.add_argument("--maximum-ate-mean-mm", type=float, default=5.0)
    parser.add_argument("--maximum-ate-p95-mm", type=float, default=10.0)
    parser.add_argument("--maximum-ate-rotation-mean-deg", type=float, default=0.10)
    parser.add_argument("--maximum-ate-rotation-p95-deg", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.solver_threads < 1 or args.stage1_period < 1:
        raise ValueError("solver threads and Stage1 period must be positive")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "scope": "generated native frontend -> generated loops -> Stage1 -> Stage2",
        "reference_used_for_solve": False,
    }
    total_started = time.perf_counter()

    load_started = time.perf_counter()
    state = load_native_frontend_state(args.frontend_state)
    samples = load_imu_rosbag(args.imu_bag)
    frontend_nodes: tuple[FrontendNode, ...] = state.frontend_nodes
    frontend_submaps: tuple[FrontendSubmap, ...] = state.frontend_submaps
    backend_nodes: tuple[TrajectoryNode, ...] = state.backend_nodes
    backend_submaps: tuple[Submap, ...] = state.backend_submaps
    report["counts"] = {
        "nodes": len(backend_nodes),
        "submaps": len(backend_submaps),
        "memberships": sum(len(submap.node_indices) for submap in backend_submaps),
        "imu_samples": len(samples),
    }
    report["timing_seconds"] = {"load_frontend_and_imu": time.perf_counter() - load_started}

    periodic_started = time.perf_counter()
    periodic = ()
    if args.periodic_stage1:
        periodic = replay_online_fast_pose_graph(
            backend_nodes,
            backend_submaps,
            samples,
            args.stage1_solver,
            work / "stage1_periodic",
            optimize_every_n_nodes=args.stage1_period,
            max_iterations=args.stage1_periodic_iterations,
            num_threads=args.solver_threads,
        )
    report["timing_seconds"]["stage1_periodic"] = (
        time.perf_counter() - periodic_started
    )

    loop_started = time.perf_counter()
    loops: tuple[LoopConstraint, ...] = ()
    if not args.skip_loop_closures:
        initial_pose_for_pair = None
        if periodic:
            def initial_pose_for_pair(submap, node):
                return online_fast_loop_initial_pose(
                    periodic,
                    backend_nodes,
                    backend_submaps,
                    submap.submap_id,
                    node.node_id,
                )
        loops = detect_loop_constraints(
            frontend_nodes,
            frontend_submaps,
            initial_pose_for_pair=initial_pose_for_pair,
        )
    report["timing_seconds"]["loop_closure"] = time.perf_counter() - loop_started
    report["counts"]["generated_valid_loops"] = sum(loop.valid for loop in loops)

    stage1_started = time.perf_counter()
    stage1 = finish_online_fast_pose_graph(
        backend_nodes,
        backend_submaps,
        samples,
        loops,
        periodic,
        args.stage1_solver,
        work / "stage1_finish",
        max_iterations=args.stage1_finish_iterations,
        num_threads=args.solver_threads,
    )
    if not stage1.result.success:
        raise RuntimeError(f"Stage1 solve failed: {stage1.result.message}")
    stage1_state = _stage1_initial_state(backend_nodes, backend_submaps, stage1)
    report["timing_seconds"]["stage1_finish"] = time.perf_counter() - stage1_started
    report["stage1"] = {
        "initial_cost": stage1.result.initial_cost,
        "final_cost": stage1.result.final_cost,
        "iterations": stage1.result.iterations,
        "success": stage1.result.success,
    }

    # Loop matching is complete. Release high-resolution frontend geometry
    # before constructing the full Stage2 IMU problem.
    for submap in frontend_submaps:
        submap.hybrid_grid.close()
    del frontend_nodes, frontend_submaps, state
    gc.collect()

    stage2_started = time.perf_counter()
    pose_graph = build_pose_graph(backend_nodes, backend_submaps, loops)
    stage2_problem = build_imu_pose_graph(
        pose_graph,
        backend_nodes,
        samples,
        stage1.result.calibration,
        initial_state=stage1_state,
    )
    stage2 = optimize_imu_pose_graph_ceres(
        stage2_problem,
        args.stage2_solver,
        work / "stage2",
        max_iterations=args.stage2_iterations,
        num_threads=args.solver_threads,
    )
    if not stage2.success:
        raise RuntimeError(f"Stage2 solve failed: {stage2.message}")
    report["timing_seconds"]["stage2"] = time.perf_counter() - stage2_started
    report["stage2"] = {
        "initial_cost": stage2.initial_cost,
        "final_cost": stage2.final_cost,
        "iterations": stage2.iterations,
        "success": stage2.success,
    }

    timestamps_ns = np.asarray(
        [node.timestamp_ns for node in backend_nodes], dtype=np.int64
    )
    optimized_nodes = tuple(stage2.poses[: len(backend_nodes)])
    _write_trajectory(args.output_trajectory.resolve(), timestamps_ns, optimized_nodes)

    engineering_aligned = None
    if args.reference_state is not None:
        # Reference output is loaded only after both clean solves finish.
        reference = load_optimization_state(args.reference_state.resolve())
        reference_poses = _interpolate_reference(reference, timestamps_ns)
        metrics = _trajectory_metrics(
            optimized_nodes, reference_poses, timestamps_ns
        )
        report["alignment"] = metrics
        ate = metrics["ate"]
        translation = ate["translation_m"]
        rotation = ate["rotation_deg"]
        engineering_aligned = bool(
            translation["mean"] * 1000.0 <= args.maximum_ate_mean_mm
            and translation["p95"] * 1000.0 <= args.maximum_ate_p95_mm
            and rotation["mean"] <= args.maximum_ate_rotation_mean_deg
            and rotation["p95"] <= args.maximum_ate_rotation_p95_deg
        )
        report["acceptance"] = {
            "engineering_aligned": engineering_aligned,
            "thresholds": {
                "ate_translation_mean_mm": args.maximum_ate_mean_mm,
                "ate_translation_p95_mm": args.maximum_ate_p95_mm,
                "ate_rotation_mean_deg": args.maximum_ate_rotation_mean_deg,
                "ate_rotation_p95_deg": args.maximum_ate_rotation_p95_deg,
            },
        }
    if args.reference_loops is not None:
        # As with the trajectory, reference loops are metrics-only.
        report["loop_alignment"] = _loop_metrics(
            loops, load_loop_constraints(args.reference_loops.resolve())
        )

    report["timing_seconds"]["total"] = time.perf_counter() - total_started
    output_report = args.output_report.resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if engineering_aligned is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
