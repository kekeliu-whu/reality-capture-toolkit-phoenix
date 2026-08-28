#!/usr/bin/env python3
"""Compare the clean Fast IMU native kernel with a frozen measurement trace."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.surveyor_slam import (  # noqa: E402
    load_optimization_imu,
    load_trajectory_nodes,
)


def _pointer(array: np.ndarray, scalar: type[ctypes._SimpleCData]):
    return array.ctypes.data_as(ctypes.POINTER(scalar))


def _metrics(clean: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    difference = clean - reference
    return {
        "shape": list(clean.shape),
        "exact_rows": int(np.count_nonzero(np.all(difference == 0.0, axis=1))),
        "exact_scalars": int(np.count_nonzero(difference == 0.0)),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    dataset = args.dataset.resolve()
    nodes = sorted(
        load_trajectory_nodes(
            dataset / "internal/nodes/trajectory_node/trajectory_node_00000000.zip"
        ),
        key=lambda node: node.timestamp_ns,
    )
    samples = load_optimization_imu(dataset / "artifacts/optimization_data.pb")
    timestamps = np.ascontiguousarray(
        [sample.timestamp_ns for sample in samples], dtype=np.int64
    )
    acceleration = np.ascontiguousarray(
        [sample.linear_acceleration for sample in samples], dtype=np.float64
    )
    angular_velocity = np.ascontiguousarray(
        [sample.angular_velocity for sample in samples], dtype=np.float64
    )

    library = ctypes.CDLL(str(args.library.resolve()))
    integrate = library.navvis_recon_slam_fast_imu_integrate
    integrate.argtypes = (
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    )
    integrate.restype = ctypes.c_int
    acceleration_measurement = (
        library.navvis_recon_slam_fast_imu_acceleration_measurement
    )
    acceleration_measurement.argtypes = (
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
    )
    acceleration_measurement.restype = ctypes.c_int

    timestamp_pointer = _pointer(timestamps, ctypes.c_int64)
    acceleration_pointer = _pointer(acceleration, ctypes.c_double)
    angular_velocity_pointer = _pointer(angular_velocity, ctypes.c_double)
    clean_rotation = np.empty((len(nodes) - 1, 4), dtype=np.float64)
    scratch_velocity = np.empty(3, dtype=np.float64)
    for index, (first, second) in enumerate(zip(nodes, nodes[1:])):
        status = integrate(
            len(timestamps),
            timestamp_pointer,
            acceleration_pointer,
            angular_velocity_pointer,
            first.timestamp_ns,
            second.timestamp_ns,
            _pointer(clean_rotation[index], ctypes.c_double),
            _pointer(scratch_velocity, ctypes.c_double),
        )
        if status:
            raise RuntimeError(f"native Fast IMU integration failed at {index}")

    clean_acceleration = np.empty((len(nodes) - 2, 3), dtype=np.float64)
    for index, (first, second, third) in enumerate(
        zip(nodes, nodes[1:], nodes[2:])
    ):
        status = acceleration_measurement(
            len(timestamps),
            timestamp_pointer,
            acceleration_pointer,
            angular_velocity_pointer,
            first.timestamp_ns,
            second.timestamp_ns,
            third.timestamp_ns,
            _pointer(clean_acceleration[index], ctypes.c_double),
        )
        if status:
            raise RuntimeError(
                f"native Fast IMU acceleration measurement failed at {index}"
            )

    reference_acceleration: list[list[float]] = []
    reference_rotation: list[list[float]] = []
    for line in args.reference.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["ACCELERATION_MEASUREMENT"]:
            reference_acceleration.append([float(value) for value in fields[2:5]])
        elif fields[:1] == ["DELTA_ROTATION_MEASUREMENT"]:
            reference_rotation.append([float(value) for value in fields[2:6]])
    reference_acceleration_array = np.asarray(reference_acceleration)
    reference_rotation_array = np.asarray(reference_rotation)
    if reference_acceleration_array.shape != clean_acceleration.shape:
        raise ValueError("reference acceleration measurement count differs")
    if reference_rotation_array.shape != clean_rotation.shape:
        raise ValueError("reference rotation measurement count differs")

    return {
        "scope": "Fast IMU native factor measurement bit alignment",
        "dataset": str(dataset),
        "library": str(args.library.resolve()),
        "reference": str(args.reference.resolve()),
        "acceleration": _metrics(
            clean_acceleration, reference_acceleration_array
        ),
        "rotation": _metrics(clean_rotation, reference_rotation_array),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
