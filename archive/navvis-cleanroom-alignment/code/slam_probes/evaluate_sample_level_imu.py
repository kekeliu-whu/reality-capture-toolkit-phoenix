#!/usr/bin/env python3
"""Narrow same-input regression for the clean-room IMU pose graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navvis_recon.surveyor_slam import (
    Rigid3,
    build_imu_pose_graph,
    build_pose_graph,
    imu_calibration_from_state,
    load_loop_constraints,
    load_optimization_imu,
    load_optimization_state,
    load_submaps,
    load_trajectory_nodes,
    optimize_imu_pose_graph,
)


def pose_errors(estimate: tuple[Rigid3, ...], reference: tuple[Rigid3, ...]):
    translation = np.asarray(
        [np.linalg.norm(a.translation - b.translation) for a, b in zip(estimate, reference)]
    )
    rotation = np.asarray(
        [(a.rotation.inv() * b.rotation).magnitude() * 180.0 / np.pi for a, b in zip(estimate, reference)]
    )
    return {
        "translation_mean_m": float(np.mean(translation)),
        "translation_p95_m": float(np.percentile(translation, 95)),
        "translation_max_m": float(np.max(translation)),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_p95_deg": float(np.percentile(rotation, 95)),
        "rotation_max_deg": float(np.max(rotation)),
    }


def rpe_errors(
    estimate: tuple[Rigid3, ...], reference: tuple[Rigid3, ...], timestamps_ns: np.ndarray
):
    translation: list[float] = []
    rotation: list[float] = []
    for first, timestamp in enumerate(timestamps_ns):
        second = int(np.searchsorted(timestamps_ns, timestamp + 1_000_000_000))
        if second >= len(timestamps_ns):
            continue
        if abs(int(timestamps_ns[second]) - int(timestamp) - 1_000_000_000) > 30_000_000:
            continue
        estimate_relative = estimate[first].between(estimate[second])
        reference_relative = reference[first].between(reference[second])
        error = reference_relative.inverse().compose(estimate_relative)
        translation.append(float(np.linalg.norm(error.translation)))
        rotation.append(float(error.rotation.magnitude() * 180.0 / np.pi))
    return {
        "pair_count": len(translation),
        "translation_mean_m": float(np.mean(translation)),
        "translation_p95_m": float(np.percentile(translation, 95)),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_p95_deg": float(np.percentile(rotation, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-iterations", type=int, default=200)
    args = parser.parse_args()
    root = args.dataset
    nodes = load_trajectory_nodes(
        root / "internal/nodes/trajectory_node/trajectory_node_00000000.zip"
    )
    submaps = load_submaps(root / "internal/submaps/submap/submap_00000000.zip")
    loops = load_loop_constraints(
        root
        / "internal/constraints_inter_dataset"
        / root.name
        / "constraints/constraint_data/constraint_data_00000000.zip"
    )
    samples = load_optimization_imu(root / "artifacts/optimization_data.pb")
    initial = load_optimization_state(root / "artifacts/optimization_state.pb")
    reference = load_optimization_state(
        root / "internal/anchors/optimization/optimization_state.pb"
    )
    problem = build_imu_pose_graph(
        build_pose_graph(nodes, submaps, loops),
        nodes,
        samples,
        imu_calibration_from_state(reference),
        initial,
    )
    result = optimize_imu_pose_graph(problem, max_iterations=args.max_iterations)
    node_estimate = result.poses[: len(nodes)]
    payload = {
        "input": str(root),
        "counts": {
            "imu_samples": len(samples),
            "nodes": len(nodes),
            "submaps": len(submaps),
            "memberships": sum(len(submap.node_indices) for submap in submaps),
            "loops": len(loops),
            "imu_factors": len(problem.preintegrations),
            "retained_problem_samples": len(problem.samples),
        },
        "solver": {
            "iterations": result.iterations,
            "success": result.success,
            "message": result.message,
            "initial_cost": result.initial_cost,
            "final_cost": result.final_cost,
        },
        "absolute": pose_errors(node_estimate, reference.poses),
        "rpe_1s": rpe_errors(node_estimate, reference.poses, reference.timestamps_ns),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
