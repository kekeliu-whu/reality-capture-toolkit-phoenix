#!/usr/bin/env python3
"""Replay one frozen local ICP call and compare every native solver step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import navvis_recon.surveyor_frontend as frontend  # noqa: E402
from navvis_recon.surveyor_frontend import FrontendConfig  # noqa: E402
from navvis_recon.surveyor_slam import Rigid3  # noqa: E402


def load_pose(path: Path) -> Rigid3:
    values = np.fromfile(path, dtype="<f8")
    if values.shape != (8,):
        raise ValueError(f"unexpected pose payload in {path}")
    return Rigid3(values[:3], values[4:8])


def pose_values(pose: Rigid3) -> np.ndarray:
    return np.concatenate((pose.translation, pose.quaternion_xyzw))


def trace_pose(payload: dict[str, object]) -> np.ndarray:
    return np.asarray(
        payload["translation"] + payload["quaternion_xyzw"],
        dtype=np.float64,
    )


def coefficient_report(clean: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    difference = clean - reference
    clean_bits = clean.view(np.uint64)
    reference_bits = reference.view(np.uint64)
    return {
        "bit_exact": bool(np.array_equal(clean_bits, reference_bits)),
        "different_fields": int(np.count_nonzero(clean_bits != reference_bits)),
        "max_abs": float(np.max(np.abs(difference))),
        "difference": difference.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-capture", type=Path, required=True)
    parser.add_argument("--solver-trace", type=Path, required=True)
    parser.add_argument("--call-index", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--loop-constraint",
        action="store_true",
        help="Use the six-iteration inter-dataset constraint matcher options",
    )
    args = parser.parse_args()

    prefix = f"call_{args.call_index:03d}"
    rays = np.fromfile(
        args.vendor_capture / f"{prefix}_source.bin", dtype="<f4"
    ).reshape((-1, 6))
    target_levels = tuple(
        np.fromfile(
            args.vendor_capture / f"{prefix}_target_level_{level}.bin",
            dtype="<f4",
        ).reshape((-1, 3))
        for level in range(3)
    )
    flattened_normals = np.fromfile(
        args.vendor_capture / f"{prefix}_target_normals.bin", dtype="<f4"
    ).reshape((-1, 3))
    normal_levels = []
    begin = 0
    for target in target_levels:
        end = begin + len(target)
        normal_levels.append(flattened_normals[begin:end])
        begin = end
    initial = load_pose(args.vendor_capture / f"{prefix}_initial.bin")
    reference_result = load_pose(
        args.vendor_capture / f"{prefix}_result.bin"
    )
    vendor = json.loads(args.solver_trace.read_text())

    native_step = frontend._binary_point_plane_step
    native_compose = frontend._compose_pose_binary
    clean_steps: list[
        tuple[int, Rigid3, np.ndarray, float, np.ndarray, np.ndarray]
    ] = []
    clean_poses: list[Rigid3] = []

    def capture_step(*values, **keywords):
        increment, delta, scale, normal_matrix, right_hand_side = (
            frontend._binary_point_plane_step_diagnostics(*values, **keywords)
        )
        clean_steps.append((
            len(values[0]), increment, delta.copy(), scale,
            normal_matrix.copy(), right_hand_side.copy(),
        ))
        return increment, delta, scale

    def capture_compose(lhs: Rigid3, rhs: Rigid3) -> Rigid3:
        output = native_compose(lhs, rhs)
        if len(clean_poses) < len(clean_steps):
            clean_poses.append(output)
        return output

    frontend._binary_point_plane_step = capture_step
    frontend._compose_pose_binary = capture_compose
    config = FrontendConfig()
    maximum_iterations = 6 if args.loop_constraint else config.icp_iterations
    minimum_iterations = 6 if args.loop_constraint else config.icp_min_iterations
    initial_plane_distance = (
        0.20 if args.loop_constraint else config.icp_initial_plane_distance_m
    )
    contracted_plane_distance = (
        0.03 if args.loop_constraint else config.icp_contracted_plane_distance_m
    )
    maximum_incidence_angle = (
        88.0 if args.loop_constraint else config.icp_max_incidence_angle_deg
    )
    try:
        result = frontend.point_to_plane_icp(
            rays[:, 3:],
            target_levels,
            tuple(normal_levels),
            initial,
            source_origins=rays[:, :3],
            binary_compatible=True,
            max_correspondence_m=config.icp_max_correspondence_m,
            huber_m=config.icp_huber_m,
            max_iterations=maximum_iterations,
            min_iterations=minimum_iterations,
            correspondence_levels_m=config.icp_correspondence_levels_m,
            initial_plane_distance_m=initial_plane_distance,
            contracted_plane_distance_m=contracted_plane_distance,
            contraction_iterations=config.icp_contraction_iterations,
            min_correspondences=config.icp_min_correspondences,
            max_incidence_angle_deg=maximum_incidence_angle,
            num_threads=config.icp_num_threads,
        )
    finally:
        frontend._binary_point_plane_step = native_step
        frontend._compose_pose_binary = native_compose

    step_reports = []
    for index, (
        (count, increment, delta, scale, normal_matrix, right_hand_side),
        pose,
        reference,
    ) in enumerate(
        zip(clean_steps, clean_poses, vendor["steps"])
    ):
        kernel = vendor["kernels"][index]
        step_reports.append({
            "iteration": index,
            "correspondences": count,
            "increment": coefficient_report(
                pose_values(increment), trace_pose(reference["increment"])
            ),
            "delta": coefficient_report(
                np.asarray(delta, dtype=np.float64),
                np.asarray(kernel["delta"], dtype=np.float64),
            ),
            "normal_matrix": coefficient_report(
                np.asarray(normal_matrix, dtype=np.float64).reshape(-1),
                np.asarray(kernel["hessian"], dtype=np.float64),
            ),
            "right_hand_side": coefficient_report(
                np.asarray(right_hand_side, dtype=np.float64),
                np.asarray(kernel["rhs"], dtype=np.float64),
            ),
            "scale": float(scale),
            "pose_after": coefficient_report(
                pose_values(pose), trace_pose(reference["pose_after"])
            ),
        })

    clean_result = pose_values(result.target_from_source)
    reference_values = pose_values(reference_result)
    if np.dot(clean_result[3:], reference_values[3:]) < 0.0:
        reference_values[3:] *= -1.0
    report = {
        "call_index": args.call_index,
        "iterations": int(result.iterations),
        "correspondences": int(result.correspondences),
        "vendor_step_count": len(vendor["steps"]),
        "result": coefficient_report(clean_result, reference_values),
        "steps": step_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
