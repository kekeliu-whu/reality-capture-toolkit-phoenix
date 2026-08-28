#!/usr/bin/env python3
"""Compare the first clean-room local ICP call with a frozen binary capture."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import navvis_recon.surveyor_frontend as frontend  # noqa: E402
from navvis_recon.surveyor_frontend import (  # noqa: E402
    RawConstantVelocityPosePredictor,
    RawImuTracker,
    SlamScanArchive,
    SurveyorFrontend,
    iter_archive_at_node_times,
)
from navvis_recon.surveyor_slam import Rigid3, load_imu_rosbag, load_trajectory_nodes  # noqa: E402


def distance_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def compare_cloud(clean: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    clean32 = np.ascontiguousarray(clean, dtype=np.float32)
    reference32 = np.ascontiguousarray(reference, dtype=np.float32)
    result: dict[str, object] = {
        "clean_count": int(len(clean32)),
        "reference_count": int(len(reference32)),
        "same_shape": clean32.shape == reference32.shape,
    }
    if clean32.shape == reference32.shape:
        delta = np.linalg.norm(clean32.astype(np.float64) - reference32, axis=1)
        result["same_order_bit_exact"] = bool(np.array_equal(clean32, reference32))
        result["same_order_distance_m"] = distance_statistics(delta)
    if len(clean32) and len(reference32):
        clean_to_reference = cKDTree(reference32).query(clean32, workers=-1)[0]
        reference_to_clean = cKDTree(clean32).query(reference32, workers=-1)[0]
        result["clean_to_reference_m"] = distance_statistics(clean_to_reference)
        result["reference_to_clean_m"] = distance_statistics(reference_to_clean)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--imu-bag", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--vendor-capture", type=Path, required=True)
    parser.add_argument("--clean-capture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    reference_nodes = load_trajectory_nodes(args.nodes)[:2]
    imu = load_imu_rosbag(args.imu_bag)
    captured: dict[str, object] = {}
    original_icp = frontend.point_to_plane_icp
    original_extract = frontend.extract_valid_split_surfels

    def capture_icp(source_points, target_points, target_normals, initial=None, **kwargs):
        if "source_points" not in captured:
            captured["source_points"] = np.asarray(source_points).copy()
            if kwargs.get("source_origins") is not None:
                captured["source_origins"] = np.asarray(
                    kwargs["source_origins"]
                ).copy()
            captured["target_points"] = tuple(
                np.asarray(value).copy() for value in target_points
            )
            captured["target_normals"] = tuple(
                np.asarray(value).copy() for value in target_normals
            )
            captured["initial"] = initial
        return original_icp(
            source_points, target_points, target_normals, initial, **kwargs
        )

    def capture_extract(statistics, voxel_size):
        if len(captured.setdefault("surfel_statistics", [])) < 3:
            captured["surfel_statistics"].append(
                frontend.SplitSurfelStatistics(
                    statistics.keys.copy(),
                    statistics.weights.copy(),
                    statistics.counts.copy(),
                    statistics.means.copy(),
                    statistics.covariances.copy(),
                )
            )
        return original_extract(statistics, voxel_size)

    frontend.point_to_plane_icp = capture_icp
    frontend.extract_valid_split_surfels = capture_extract
    try:
        with SlamScanArchive(args.archive) as archive:
            first_by_sensor: dict[int, int] = {}
            for record in archive.records:
                first_by_sensor.setdefault(
                    record.sensor, (record.timestamp_ns // 1000) * 1000
                )
            first_all_sources_ns = max(first_by_sensor.values())
            tracker = RawImuTracker(imu)
            tracker_at_start = tracker.advance(first_all_sources_ns)
            tracker_at_first_node = tracker.advance(reference_nodes[0].timestamp_ns)
            initial_pose = Rigid3(
                reference_nodes[0].local_pose.translation.copy(),
                (
                    reference_nodes[0].local_pose.rotation
                    * (tracker_at_start.inv() * tracker_at_first_node).inv()
                ).as_quat(),
            )
            predictor = RawConstantVelocityPosePredictor(imu, initial_pose=initial_pose)
            scans = iter(
                iter_archive_at_node_times(
                    archive,
                    [node.timestamp_ns for node in reference_nodes],
                    imu_pose_predictor=predictor,
                )
            )
            first_scan = next(scans)
            predictor.correct(
                first_scan.timestamp_ns, reference_nodes[0].local_pose
            )
            result = SurveyorFrontend().process(
                itertools.chain((first_scan,), scans),
                imu_pose_predictor=predictor,
                find_loops=False,
            )
    finally:
        frontend.point_to_plane_icp = original_icp
        frontend.extract_valid_split_surfels = original_extract

    if not captured:
        raise RuntimeError("the two-node run did not execute local ICP")

    clean_source = np.asarray(captured["source_points"], dtype=np.float32)
    clean_levels = tuple(
        np.asarray(value, dtype=np.float32) for value in captured["target_points"]
    )
    clean_normal_levels = tuple(
        np.asarray(value, dtype=np.float32) for value in captured["target_normals"]
    )
    initial = captured["initial"]
    assert isinstance(initial, Rigid3)
    transformed_clean_source = np.asarray(
        initial.rotation.apply(clean_source) + initial.translation,
        dtype=np.float32,
    )

    vendor_levels = tuple(
        np.fromfile(
            args.vendor_capture / f"correspondence_finder_{level}_points.bin",
            dtype="<f4",
        ).reshape((-1, 3))
        for level in range(3)
    )
    vendor_points = np.fromfile(
        args.vendor_capture / "solver_target_points.bin", dtype="<f4"
    ).reshape((-1, 3))
    vendor_normals = np.fromfile(
        args.vendor_capture / "solver_target_normals.bin", dtype="<f4"
    ).reshape((-1, 3))
    vendor_source = np.fromfile(
        args.vendor_capture / "iteration_00_source.bin", dtype="<f4"
    ).reshape((-1, 3))

    clean_points = np.concatenate(clean_levels)
    clean_normals = np.concatenate(clean_normal_levels)
    level_definitions = ((0.10, 0.05), (0.30, 0.15), (0.60, 0.0))
    key_differences = []
    for (resolution, offset), clean, vendor in zip(
        level_definitions, clean_levels, vendor_levels
    ):
        clean_keys = np.floor(
            (clean.astype(np.float64) - offset) / resolution
        ).astype(np.int64)
        vendor_keys = np.floor(
            (vendor.astype(np.float64) - offset) / resolution
        ).astype(np.int64)
        clean_set = {tuple(key) for key in clean_keys}
        vendor_set = {tuple(key) for key in vendor_keys}
        key_differences.append(
            {
                "clean_only": [
                    [int(value) for value in key]
                    for key in sorted(clean_set - vendor_set)
                ],
                "vendor_only": [
                    [int(value) for value in key]
                    for key in sorted(vendor_set - clean_set)
                ],
            }
        )
    report = {
        "level_points": [
            compare_cloud(clean, vendor)
            for clean, vendor in zip(clean_levels, vendor_levels)
        ],
        "flattened_points": compare_cloud(clean_points, vendor_points),
        "flattened_normals": compare_cloud(clean_normals, vendor_normals),
        "transformed_source": compare_cloud(
            transformed_clean_source, vendor_source
        ),
        "level_key_differences": key_differences,
        "clean_icp": {
            "correspondences": int(result.nodes[1].scan_match.correspondences),
            "iterations": int(result.nodes[1].scan_match.iterations),
            "fitness_m": float(result.nodes[1].scan_match.fitness_m),
            "translation": result.nodes[1].local_pose.translation.tolist(),
            "quaternion_xyzw": result.nodes[1].local_pose.quaternion_xyzw.tolist(),
        },
    }

    args.clean_capture.parent.mkdir(parents=True, exist_ok=True)
    capture_arrays = dict(
        source_points=clean_source,
        source_origins=np.asarray(
            captured.get("source_origins", np.zeros_like(clean_source)),
            dtype=np.float32,
        ),
        transformed_source=transformed_clean_source,
        target_points=clean_points,
        target_normals=clean_normals,
        target_level_offsets=np.cumsum([0, *[len(value) for value in clean_levels]]),
        initial_translation=initial.translation,
        initial_quaternion_xyzw=initial.quaternion_xyzw,
    )
    for level, statistics in enumerate(captured["surfel_statistics"]):
        capture_arrays[f"surfel_{level}_keys"] = statistics.keys
        capture_arrays[f"surfel_{level}_weights"] = statistics.weights
        capture_arrays[f"surfel_{level}_counts"] = statistics.counts
        capture_arrays[f"surfel_{level}_means"] = statistics.means
        capture_arrays[f"surfel_{level}_covariances"] = statistics.covariances
    np.savez_compressed(args.clean_capture, **capture_arrays)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
