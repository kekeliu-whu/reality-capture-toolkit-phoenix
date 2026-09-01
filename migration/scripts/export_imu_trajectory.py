#!/usr/bin/env python3
"""Export odometry trajectory.txt as validated IMU-frame TUM trajectories.

The odometry file stores T_world_imu as:
  x y z roll pitch yaw qx qy qz qw timestamp

Two products are written:
  * trajectory_imu_world_tum.txt:          T_world_imu(t)
  * trajectory_imu_start_relative_tum.txt: inv(T_world_imu(0)) T_world_imu(t)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def save_tum(path: Path, timestamps: np.ndarray, positions: np.ndarray,
             quaternions: np.ndarray) -> None:
    values = np.column_stack((timestamps, positions, quaternions))
    np.savetxt(path, values, fmt="%.12f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = np.loadtxt(args.input, comments="#", dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 11:
        raise ValueError(f"Expected Nx11 odometry trajectory, got {source.shape}")

    positions = source[:, 0:3]
    quaternions = source[:, 6:10]
    timestamps = source[:, 10]
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if not np.isfinite(source).all():
        raise ValueError("Trajectory contains non-finite values")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Trajectory timestamps are not strictly increasing")
    if np.max(np.abs(quaternion_norms - 1.0)) > 1e-6:
        raise ValueError("Trajectory contains non-unit quaternions")

    rotations = Rotation.from_quat(quaternions)
    initial_rotation_inverse = rotations[0].inv()
    relative_positions = initial_rotation_inverse.apply(positions - positions[0])
    relative_rotations = initial_rotation_inverse * rotations
    relative_quaternions = relative_rotations.as_quat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_path = args.output_dir / "trajectory_imu_world_tum.txt"
    relative_path = args.output_dir / "trajectory_imu_start_relative_tum.txt"
    save_tum(world_path, timestamps, positions, quaternions)
    save_tum(relative_path, timestamps, relative_positions, relative_quaternions)

    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    delta_time = np.diff(timestamps)
    speeds = steps / delta_time
    metrics = {
        "source": str(args.input),
        "pose_semantics": {
            "source": "T_world_imu(t)",
            "world_output": "T_world_imu(t)",
            "start_relative_output": "inverse(T_world_imu(0)) * T_world_imu(t)",
        },
        "tum_columns": "timestamp tx ty tz qx qy qz qw",
        "pose_count": int(source.shape[0]),
        "start_timestamp": float(timestamps[0]),
        "end_timestamp": float(timestamps[-1]),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "path_length_m": float(steps.sum()),
        "end_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "position_range_m": np.ptp(positions, axis=0).tolist(),
        "step_m": {
            "median": float(np.median(steps)),
            "p95": float(np.quantile(steps, 0.95)),
            "p99": float(np.quantile(steps, 0.99)),
            "max": float(steps.max()),
        },
        "speed_mps": {
            "median": float(np.median(speeds)),
            "p95": float(np.quantile(speeds, 0.95)),
            "p99": float(np.quantile(speeds, 0.99)),
            "max": float(speeds.max()),
        },
        "quaternion_norm": {
            "min": float(quaternion_norms.min()),
            "max": float(quaternion_norms.max()),
        },
        "checksums_sha256": {
            world_path.name: sha256(world_path),
            relative_path.name: sha256(relative_path),
        },
    }
    metrics_path = args.output_dir / "trajectory_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
