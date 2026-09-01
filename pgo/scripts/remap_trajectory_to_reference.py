#!/usr/bin/env python3
"""Resample a reference text trajectory onto an existing traj.dat timeline.

The input traj.dat supplies the exact one-pose-per-scan timestamps and gravity
fields expected by slam_post.  Position and orientation are interpolated from
the reference trajectory, producing another PoseMsgList that can be paired
with the original lidar_undist.dat without dropping or duplicating scans.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-traj", type=Path, required=True)
    parser.add_argument("--reference-trajectory", type=Path, required=True)
    parser.add_argument("--output-traj", type=Path, required=True)
    parser.add_argument("--output-text", type=Path)
    parser.add_argument(
        "--proto-python-dir",
        type=Path,
        required=True,
        help="Directory containing the generated sensors_pb2.py binding",
    )
    return parser.parse_args()


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = [float(value) for value in stripped.split()]
            if len(values) < 11:
                raise ValueError(f"Expected at least 11 columns, got {len(values)}: {line!r}")
            rows.append(values)

    if len(rows) < 2:
        raise ValueError(f"Reference trajectory contains only {len(rows)} poses")

    data = np.asarray(rows, dtype=np.float64)
    timestamps = data[:, 10]
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("Reference trajectory timestamps must be strictly increasing")
    positions = data[:, 0:3]
    quaternions_xyzw = data[:, 6:10]
    quaternions_xyzw /= np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True)
    return timestamps, positions, quaternions_xyzw


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    result = (
        math.sin((1.0 - alpha) * theta) / sin_theta * q0
        + math.sin(alpha * theta) / sin_theta * q1
    )
    return result / np.linalg.norm(result)


def quaternion_to_rpy_xyzw(q: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.proto_python_dir.resolve()))
    import sensors_pb2  # type: ignore[import-not-found]  # pylint: disable=import-error,import-outside-toplevel

    source = sensors_pb2.PoseMsgList()
    source.ParseFromString(args.source_traj.read_bytes())
    if not source.pose_msgs:
        raise ValueError("Source traj.dat contains no poses")

    ref_t, ref_p, ref_q = load_reference(args.reference_trajectory)
    source_t = np.asarray([pose.timestamp for pose in source.pose_msgs], dtype=np.float64)
    if source_t[0] < ref_t[0] or source_t[-1] > ref_t[-1]:
        raise ValueError(
            f"Source time range [{source_t[0]:.6f}, {source_t[-1]:.6f}] is outside "
            f"reference range [{ref_t[0]:.6f}, {ref_t[-1]:.6f}]"
        )

    output_rows: list[tuple[float, ...]] = []
    for pose, timestamp in zip(source.pose_msgs, source_t):
        right = int(np.searchsorted(ref_t, timestamp, side="left"))
        if right == 0:
            left = right = 0
            alpha = 0.0
        elif right == len(ref_t):
            left = right = len(ref_t) - 1
            alpha = 0.0
        else:
            left = right - 1
            alpha = float((timestamp - ref_t[left]) / (ref_t[right] - ref_t[left]))

        position = ref_p[left] if left == right else (1.0 - alpha) * ref_p[left] + alpha * ref_p[right]
        quaternion = ref_q[left] if left == right else slerp_xyzw(ref_q[left], ref_q[right], alpha)

        pose.tx, pose.ty, pose.tz = (float(value) for value in position)
        pose.rx, pose.ry, pose.rz, pose.rw = (float(value) for value in quaternion)
        roll, pitch, yaw = quaternion_to_rpy_xyzw(quaternion)
        output_rows.append(
            (*position, roll, pitch, yaw, *quaternion, float(timestamp))
        )

    args.output_traj.parent.mkdir(parents=True, exist_ok=True)
    args.output_traj.write_bytes(source.SerializeToString())

    if args.output_text:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        with args.output_text.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("#x y z roll pitch yaw qx qy qz qw timestamp\n")
            for row in output_rows:
                stream.write(" ".join(f"{value:.12f}" for value in row) + "\n")

    print(
        f"Wrote {len(source.pose_msgs)} poses; source range "
        f"[{source_t[0]:.6f}, {source_t[-1]:.6f}], reference range "
        f"[{ref_t[0]:.6f}, {ref_t[-1]:.6f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
