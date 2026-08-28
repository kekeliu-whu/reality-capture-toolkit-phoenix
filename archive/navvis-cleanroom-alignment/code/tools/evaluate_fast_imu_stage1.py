#!/usr/bin/env python3
"""Evaluate the clean-room pose-only Stage1 IMU backend.

The vendor output is read only after the clean solve.  It is used exclusively
for ATE/RPE and calibration metrics, never as a solver input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.surveyor_slam import (  # noqa: E402
    build_fast_imu_pose_graph,
    build_pose_graph,
    imu_calibration_from_state,
    load_loop_constraints,
    load_optimization_imu,
    load_optimization_state,
    load_submaps,
    load_trajectory_nodes,
    optimize_fast_imu_pose_graph,
    optimize_fast_imu_pose_graph_ceres,
)

from evaluate_complete_slam import (  # noqa: E402
    _reference_poses_for_timestamps,
    ate_errors,
    gauge_align,
    rpe_errors,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_paths(dataset: Path) -> dict[str, Path]:
    return {
        "nodes": dataset
        / "internal/nodes/trajectory_node/trajectory_node_00000000.zip",
        "submaps": dataset / "internal/submaps/submap/submap_00000000.zip",
        "loops": dataset
        / "internal/constraints_inter_dataset"
        / dataset.name
        / "constraints/constraint_data/constraint_data_00000000.zip",
        "optimization_data": dataset / "artifacts/optimization_data.pb",
        "initial_state": dataset / "artifacts/optimization_state.pb",
    }


def _compare_captured_measurements(
    path: Path, problem
) -> dict[str, object]:
    acceleration = []
    rotation = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["ACCELERATION_MEASUREMENT"]:
            acceleration.append([float(value) for value in fields[2:5]])
        elif fields[:1] == ["DELTA_ROTATION_MEASUREMENT"]:
            rotation.append([float(value) for value in fields[2:6]])
    clean_acceleration = np.asarray(
        [factor.delta_velocity for factor in problem.acceleration_factors]
    )
    clean_rotation = np.asarray(
        [factor.delta_rotation_xyzw for factor in problem.rotation_factors]
    )
    captured_acceleration = np.asarray(acceleration)
    captured_rotation = np.asarray(rotation)
    if captured_acceleration.shape != clean_acceleration.shape:
        raise ValueError("captured Stage1 acceleration count differs")
    if captured_rotation.shape != clean_rotation.shape:
        raise ValueError("captured Stage1 rotation count differs")

    def metrics(clean: np.ndarray, captured: np.ndarray) -> dict[str, object]:
        difference = clean - captured
        return {
            "shape": list(clean.shape),
            "exact_scalar_count": int(np.count_nonzero(difference == 0.0)),
            "exact_row_count": int(np.count_nonzero(np.all(difference == 0.0, axis=1))),
            "mean_absolute_error": float(np.mean(np.abs(difference))),
            "max_absolute_error": float(np.max(np.abs(difference))),
            "max_row_norm": float(np.max(np.linalg.norm(difference, axis=1))),
        }

    return {
        "capture": str(path),
        "acceleration": metrics(clean_acceleration, captured_acceleration),
        "rotation": metrics(clean_rotation, captured_rotation),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    dataset = args.dataset.resolve()
    defaults = _default_paths(dataset)
    paths = {
        key: (getattr(args, key) or default).resolve()
        for key, default in defaults.items()
    }
    paths["reference_state"] = args.reference_state.resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Stage1 inputs: " + ", ".join(missing))

    load_start = time.perf_counter()
    nodes = load_trajectory_nodes(paths["nodes"])
    submaps = load_submaps(paths["submaps"])
    loops = load_loop_constraints(paths["loops"])
    samples = load_optimization_imu(paths["optimization_data"])
    initial = load_optimization_state(paths["initial_state"])
    load_seconds = time.perf_counter() - load_start

    build_start = time.perf_counter()
    pose_graph = build_pose_graph(nodes, submaps, loops)
    problem = build_fast_imu_pose_graph(
        pose_graph,
        nodes,
        samples,
        imu_calibration_from_state(initial),
        initial_state=initial,
    )
    build_seconds = time.perf_counter() - build_start
    measurement_alignment = None
    if args.vendor_measurements:
        measurement_alignment = _compare_captured_measurements(
            args.vendor_measurements.resolve(), problem
        )

    optimize_start = time.perf_counter()
    if args.ceres_solver:
        result = optimize_fast_imu_pose_graph_ceres(
            problem,
            args.ceres_solver,
            args.ceres_work_dir,
            max_iterations=args.max_iterations,
            num_threads=args.solver_threads,
        )
        solver_backend = "native_ceres"
    else:
        result = optimize_fast_imu_pose_graph(
            problem,
            max_iterations=args.max_iterations,
        )
        solver_backend = "scipy_numerical_jacobian"
    optimize_seconds = time.perf_counter() - optimize_start

    # The reference is deliberately loaded only after the clean solve.
    reference = load_optimization_state(paths["reference_state"])
    reference_nodes = _reference_poses_for_timestamps(
        reference, [node.timestamp_ns for node in nodes]
    )
    node_count = len(nodes)
    node_estimate = result.poses[:node_count]
    aligned_nodes, gauge = gauge_align(node_estimate, reference_nodes)
    metrics = {
        "gauge_transform": {
            "translation_m": gauge.translation.tolist(),
            "quaternion_xyzw": gauge.quaternion_xyzw.tolist(),
        },
        "gauge_aligned_ate_nodes": ate_errors(aligned_nodes, reference_nodes),
        "gauge_aligned_rpe_nodes": rpe_errors(
            aligned_nodes,
            reference_nodes,
            np.asarray([node.timestamp_ns for node in nodes], dtype=np.int64),
            interval_seconds=args.rpe_seconds,
            tolerance_ms=args.rpe_tolerance_ms,
        ),
    }
    clean_calibration = result.calibration
    reference_calibration = imu_calibration_from_state(reference)
    calibration = {
        "gravity_error": (
            clean_calibration.gravity_magnitude
            - reference_calibration.gravity_magnitude
        ),
        "imu_orientation_error_deg": float(
            np.degrees(
                (
                    clean_calibration.imu_from_tracking.rotation.inv()
                    * reference_calibration.imu_from_tracking.rotation
                ).magnitude()
            )
        ),
    }
    ate = metrics["gauge_aligned_ate_nodes"]
    rpe = metrics["gauge_aligned_rpe_nodes"]
    factor_measurements_aligned = bool(
        measurement_alignment
        and measurement_alignment["acceleration"]["max_absolute_error"] <= 1e-12
        and measurement_alignment["rotation"]["max_absolute_error"] <= 1e-12
    )
    trajectory_aligned = bool(
        ate["translation_m"]["max"] <= 1e-9
        and ate["rotation_deg"]["max"] <= 1e-6
        and rpe["translation_m"]["max"] <= 1e-8
        and rpe["rotation_deg"]["max"] <= 1e-6
    )
    calibration_aligned = bool(
        abs(calibration["gravity_error"]) <= 1e-9
        and calibration["imu_orientation_error_deg"] <= 1e-6
    )
    precise_cost_error = None
    cost_aligned = False
    if args.official_final_cost_precise is not None:
        precise_cost_error = result.final_cost - args.official_final_cost_precise
        cost_aligned = abs(precise_cost_error) <= 1e-5
    stage1_result_aligned = bool(
        result.success
        and factor_measurements_aligned
        and trajectory_aligned
        and calibration_aligned
        and cost_aligned
    )
    vendor = args.vendor_binary.resolve()
    observed_vendor_sha256 = _sha256(vendor) if vendor.is_file() else None
    return {
        "scope": "Stage1 pose-only IMU backend isolation",
        "input_provenance": {
            key: str(path) for key, path in paths.items()
        },
        "vendor_reference": {
            "binary": str(vendor),
            "binary_available": vendor.is_file(),
            "sha256_expected": args.vendor_sha256,
            "sha256_observed": observed_vendor_sha256,
            "sha256_matches_expected": (
                observed_vendor_sha256 == args.vendor_sha256
                if observed_vendor_sha256 is not None
                else None
            ),
            "reference_state_sha256": _sha256(paths["reference_state"]),
            "measurement_capture_sha256": (
                _sha256(args.vendor_measurements.resolve())
                if args.vendor_measurements
                else None
            ),
            "build_id": args.vendor_build_id,
            "official_initial_cost_displayed": args.official_initial_cost,
            "official_final_cost_displayed": args.official_final_cost,
            "official_final_cost_precise": args.official_final_cost_precise,
        },
        "counts": {
            "nodes": len(nodes),
            "submaps": len(submaps),
            "memberships": sum(len(submap.node_indices) for submap in submaps),
            "valid_loops": sum(loop.valid for loop in loops),
            "imu_samples": len(samples),
            "acceleration_factors": len(problem.acceleration_factors),
            "rotation_factors": len(problem.rotation_factors),
        },
        "timing_seconds": {
            "load": load_seconds,
            "build": build_seconds,
            "optimize": optimize_seconds,
        },
        "solver": {
            "backend": solver_backend,
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
            "function_evaluations": result.iterations,
            "success": result.success,
            "message": result.message,
            "final_cost_error_vs_precise_reference": precise_cost_error,
        },
        "metrics": metrics,
        "calibration": calibration,
        "factor_measurement_alignment": measurement_alignment,
        "acceptance": {
            "thresholds": {
                "factor_measurement_max_absolute_error": 1e-12,
                "ate_translation_max_m": 1e-9,
                "ate_rotation_max_deg": 1e-6,
                "rpe_translation_max_m": 1e-8,
                "rpe_rotation_max_deg": 1e-6,
                "gravity_absolute_error": 1e-9,
                "imu_orientation_error_deg": 1e-6,
                "final_cost_absolute_error": 1e-5,
            },
            "factor_topology_exact": (
                len(problem.acceleration_factors) == len(nodes) - 2
                and len(problem.rotation_factors) == len(nodes) - 1
            ),
            "factor_measurements_double_precision_aligned": (
                factor_measurements_aligned
            ),
            "trajectory_result_floating_precision_aligned": trajectory_aligned,
            "calibration_result_floating_precision_aligned": calibration_aligned,
            "cost_result_floating_precision_aligned": cost_aligned,
            "complete_stage1_backend_result_aligned": stage1_result_aligned,
            "complete_stage1_exact": stage1_result_aligned,
            "protobuf_byte_exact": False,
            "exact_semantics": (
                "complete Stage1 backend result alignment at the declared "
                "floating-point tolerances for this frozen input and vendor "
                "build; it does not mean byte-identical protobuf output"
            ),
            "reason": (
                "factor measurements, final objective, trajectory ATE/RPE, "
                "gravity and IMU orientation all meet the frozen Stage1 "
                "backend acceptance thresholds"
                if stage1_result_aligned
                else "one or more frozen Stage1 backend acceptance thresholds failed"
            ),
        },
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate clean-room Stage1 IMU factors against a frozen vendor rerun."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--nodes", type=Path)
    parser.add_argument("--submaps", type=Path)
    parser.add_argument("--loops", type=Path)
    parser.add_argument("--optimization-data", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--solver-threads", type=int, default=7)
    parser.add_argument("--ceres-solver", type=Path)
    parser.add_argument(
        "--ceres-work-dir",
        type=Path,
        default=ROOT / "build-release/stage1_acceptance_work",
    )
    parser.add_argument("--vendor-measurements", type=Path)
    parser.add_argument("--rpe-seconds", type=float, default=1.0)
    parser.add_argument("--rpe-tolerance-ms", type=float, default=30.0)
    parser.add_argument(
        "--vendor-binary",
        type=Path,
        default=Path("/opt/NavVis/slam/lib/surveyor_ros/compute_trajectories"),
    )
    parser.add_argument(
        "--vendor-build-id",
        default="67b4a8b2a22cd09e2b22a9036579e1f4c6a66ea3",
    )
    parser.add_argument(
        "--vendor-sha256",
        default="1bda604e3e9b151b7e497bc9d5f2e8c37d849d147cb678c868655e69587750dc",
    )
    parser.add_argument("--official-initial-cost", type=float, default=401619.9)
    parser.add_argument("--official-final-cost", type=float, default=401551.6)
    parser.add_argument(
        "--official-final-cost-precise",
        type=float,
        help=(
            "precise vendor objective evaluated with the clean residual model; "
            "required for complete Stage1 backend acceptance"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = argument_parser()
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.solver_threads < 1:
        parser.error("--solver-threads must be positive")
    payload = evaluate(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
