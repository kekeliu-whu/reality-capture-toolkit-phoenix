#!/usr/bin/env python3
"""Compare local ICP calls while forcing frozen binary results between calls.

Forcing only the returned pose keeps every later submap insertion in the
binary trajectory gauge.  Each clean ICP invocation can therefore be compared
independently without allowing an earlier numerical residual to move all later
source and target geometry.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import navvis_recon.surveyor_frontend as frontend  # noqa: E402
from navvis_recon.surveyor_frontend import (  # noqa: E402
    IcpResult,
    RawConstantVelocityPosePredictor,
    RawImuTracker,
    SlamScanArchive,
    SurveyorFrontend,
    iter_archive_at_node_times,
)
from navvis_recon.surveyor_slam import (  # noqa: E402
    Rigid3,
    load_imu_rosbag,
    load_trajectory_nodes,
)


def load_rigid(path: Path) -> Rigid3:
    values = np.fromfile(path, dtype="<f8")
    if values.shape != (8,):
        raise ValueError(f"unexpected rigid transform size in {path}")
    return Rigid3(values[:3], values[4:8])


def pose_error(clean: Rigid3, reference: Rigid3) -> dict[str, float]:
    return {
        "translation_mm": float(
            np.linalg.norm(clean.translation - reference.translation) * 1000.0
        ),
        "rotation_deg": float(
            np.degrees((clean.rotation.inv() * reference.rotation).magnitude())
        ),
    }


def pose_coefficient_error(
    clean: Rigid3, reference: Rigid3
) -> dict[str, object]:
    """Compare raw coefficients after choosing the equivalent quaternion sign."""

    clean_values = np.concatenate((
        clean.translation, clean.quaternion_xyzw
    )).astype(np.float64)
    reference_quaternion = reference.quaternion_xyzw.copy()
    if np.dot(clean.quaternion_xyzw, reference_quaternion) < 0.0:
        reference_quaternion = -reference_quaternion
    reference_values = np.concatenate((
        reference.translation, reference_quaternion
    )).astype(np.float64)
    difference = clean_values - reference_values
    return {
        "bit_exact_up_to_quaternion_sign": bool(
            np.array_equal(clean_values.view(np.uint64),
                           reference_values.view(np.uint64))
        ),
        "different_fields": int(np.count_nonzero(
            clean_values.view(np.uint64) != reference_values.view(np.uint64)
        )),
        "max_abs": float(np.max(np.abs(difference))),
        "difference": difference.tolist(),
    }


def same_order_error(clean: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    clean32 = np.ascontiguousarray(clean, dtype=np.float32)
    reference32 = np.ascontiguousarray(reference, dtype=np.float32)
    result: dict[str, object] = {
        "clean_count": int(len(clean32)),
        "reference_count": int(len(reference32)),
        "same_shape": clean32.shape == reference32.shape,
    }
    if clean32.shape == reference32.shape:
        distance = np.linalg.norm(
            clean32.astype(np.float64) - reference32.astype(np.float64), axis=1
        )
        mismatched = np.flatnonzero(np.any(clean32 != reference32, axis=1))
        result.update({
            "bit_exact": bool(np.array_equal(clean32, reference32)),
            "mean": float(np.mean(distance)),
            "p95": float(np.percentile(distance, 95)),
            "max": float(np.max(distance)),
            "mismatched_rows": int(len(mismatched)),
            "mismatch_indices_first": mismatched[:8].tolist(),
            "clean_mismatches_first": (
                clean32[mismatched[:8]].astype(np.float64).tolist()
            ),
            "reference_mismatches_first": (
                reference32[mismatched[:8]].astype(np.float64).tolist()
            ),
        })
    elif clean32.ndim == 2 and reference32.ndim == 2:
        shared = min(len(clean32), len(reference32))
        prefix = 0
        while prefix < shared and np.array_equal(
            clean32[prefix], reference32[prefix]
        ):
            prefix += 1
        suffix = 0
        while (
            suffix < shared - prefix
            and np.array_equal(clean32[-1 - suffix], reference32[-1 - suffix])
        ):
            suffix += 1
        row_dtype = np.dtype((np.void, clean32.dtype.itemsize * clean32.shape[1]))
        clean_rows = np.ascontiguousarray(clean32).view(row_dtype).reshape(-1)
        reference_rows = np.ascontiguousarray(reference32).view(row_dtype).reshape(-1)
        clean_only = clean32[~np.isin(clean_rows, reference_rows)]
        reference_only = reference32[~np.isin(reference_rows, clean_rows)]
        result.update({
            "common_prefix": prefix,
            "common_suffix": suffix,
            "clean_only_count": int(len(clean_only)),
            "reference_only_count": int(len(reference_only)),
            "clean_only_first": clean_only[:8].astype(np.float64).tolist(),
            "reference_only_first": (
                reference_only[:8].astype(np.float64).tolist()
            ),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--imu-bag", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--vendor-capture", type=Path, required=True)
    parser.add_argument("--node-limit", type=int, default=20)
    parser.add_argument(
        "--vendor-call-start",
        type=int,
        default=0,
        help=(
            "Run earlier clean ICP calls autonomously and start comparing at "
            "this absolute call index"
        ),
    )
    parser.add_argument(
        "--use-vendor-initial",
        action="store_true",
        help="Isolate ICP by replacing only each clean initial pose",
    )
    parser.add_argument(
        "--keep-clean-result",
        action="store_true",
        help=(
            "Continue autonomously with each clean ICP result instead of "
            "forcing the captured vendor result"
        ),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--clean-capture",
        type=Path,
        help="Optionally save each clean source cloud as little-endian float32",
    )
    parser.add_argument(
        "--clean-filter-capture-index",
        type=int,
        help=(
            "Limit raw/filtered scan capture to this absolute filter-call "
            "index; omitted preserves the legacy all-call capture"
        ),
    )
    parser.add_argument(
        "--clean-map-capture",
        type=Path,
        help="Optionally save transformed ray batches inserted into submap zero",
    )
    parser.add_argument(
        "--clean-map-capture-index",
        type=int,
        action="append",
        help=(
            "Save only the selected trajectory-node insertion; may be "
            "repeated. Omitted preserves the legacy all-node capture."
        ),
    )
    parser.add_argument(
        "--clean-map-capture-submap-index",
        type=int,
        action="append",
        help=(
            "Save insertions for the selected submap; may be repeated. "
            "Omitted preserves the legacy submap-zero capture."
        ),
    )
    parser.add_argument(
        "--clean-deskew-capture",
        type=Path,
        help="Optionally save the first point-wise IMU deskew operands",
    )
    parser.add_argument(
        "--clean-deskew-capture-index",
        type=int,
        default=0,
        help="Zero-based relative-motion call saved by --clean-deskew-capture",
    )
    parser.add_argument(
        "--replace-deskew-timestamp-ns",
        type=int,
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Probe one hypothesized raw-ray timestamp representation",
    )
    parser.add_argument(
        "--discarded-batch-timestamp-ns",
        type=int,
        action="append",
        default=None,
        help=(
            "Replay a binary-captured motion-filtered range batch so its IMU "
            "predictor state is preserved; may be repeated"
        ),
    )
    parser.add_argument(
        "--clean-tracker-state-capture",
        type=Path,
        help="Optionally save the longest clean IMU tracker state sequence",
    )
    parser.add_argument(
        "--clean-predictor-capture",
        type=Path,
        help="Optionally save one clean constant-velocity prediction stage",
    )
    parser.add_argument(
        "--clean-predictor-timestamp-ns",
        type=int,
        help="Timestamp selected by --clean-predictor-capture",
    )
    parser.add_argument(
        "--clean-surfel-state-capture",
        type=Path,
        help="Optionally save every incremental split-surfel state",
    )
    parser.add_argument(
        "--clean-surfel-state-capture-index",
        type=int,
        action="append",
        help=(
            "Save only the selected zero-based split-surfel update; may be "
            "repeated. Omitted preserves the legacy all-update capture."
        ),
    )
    parser.add_argument(
        "--trace-surfel-key",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Record one level-0 split-surfel cell after every retained-node "
            "insertion"
        ),
    )
    args = parser.parse_args()
    if args.clean_capture is not None:
        args.clean_capture.mkdir(parents=True, exist_ok=True)
    if args.clean_map_capture is not None:
        args.clean_map_capture.mkdir(parents=True, exist_ok=True)
    if args.clean_deskew_capture is not None:
        args.clean_deskew_capture.mkdir(parents=True, exist_ok=True)
    if args.clean_surfel_state_capture is not None:
        args.clean_surfel_state_capture.mkdir(parents=True, exist_ok=True)

    reference_nodes = load_trajectory_nodes(args.nodes)[: args.node_limit]
    imu = load_imu_rosbag(args.imu_bag)
    calls: list[dict[str, object]] = []
    original_icp = frontend.point_to_plane_icp
    original_insert = frontend.FrontendSubmap.insert
    original_quaternion_transform = frontend._quaternion_transform_vector
    original_float_quaternion_transform = frontend._transform_points_float_quaternion
    original_float_matrix_transform = frontend._transform_points_float_matrix
    original_relative_motion = frontend.RawConstantVelocityPosePredictor.relative_motion
    original_predict_from_tracker = (
        frontend.RawConstantVelocityPosePredictor
        ._prediction_from_tracker_quaternion
    )
    original_tracker_advance = frontend.RawImuTracker.advance
    original_filter_scan = frontend.filter_scan
    original_update_split_surfels = frontend.update_split_surfel_statistics
    deskew_calls = 0
    deskew_motion_calls = 0
    float_quaternion_calls = 0
    float_matrix_calls = 0
    deskew_batch_active = False
    filter_calls = 0
    surfel_update_calls = 0
    surfel_key_trace: list[dict[str, object]] = []
    tracker_states: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    predictor_capture: dict[str, object] | None = None
    icp_calls_seen = 0

    def vendor_prefix(call_index: int) -> str:
        """Accept both legacy two-digit and targeted three-digit captures."""

        three_digit = f"call_{call_index:03d}"
        if (args.vendor_capture / f"{three_digit}_initial.bin").exists():
            return three_digit
        return f"call_{call_index:02d}"

    def capture_tracker_advance(self, timestamp_ns):
        orientation = original_tracker_advance(self, timestamp_ns)
        if args.clean_tracker_state_capture is not None:
            tracker_states.setdefault(id(self), []).append((
                int(timestamp_ns),
                self.orientation_xyzw.copy(),
                self._gravity_vector.copy(),
            ))
        return orientation

    def capture_predict_from_tracker(self, timestamp_ns, tracker_quaternion):
        nonlocal predictor_capture
        if (
            predictor_capture is None
            and args.clean_predictor_capture is not None
            and timestamp_ns == args.clean_predictor_timestamp_ns
            and self._corrections
        ):
            anchor_time, anchor_pose, anchor_tracker_quaternion = (
                self._corrections[-1]
            )
            delta_quaternion = frontend._quaternion_product_binary(
                frontend._quaternion_inverse_xyzw(anchor_tracker_quaternion),
                tracker_quaternion,
            )
            prediction = original_predict_from_tracker(
                self, timestamp_ns, tracker_quaternion
            )
            predictor_capture = {
                "timestamp_ns": int(timestamp_ns),
                "anchor_timestamp_ns": int(anchor_time),
                "anchor_translation": anchor_pose.translation.tolist(),
                "anchor_quaternion_xyzw": anchor_pose.quaternion_xyzw.tolist(),
                "anchor_tracker_quaternion_xyzw": (
                    anchor_tracker_quaternion.tolist()
                ),
                "tracker_quaternion_xyzw": tracker_quaternion.tolist(),
                "increment_quaternion_xyzw": delta_quaternion.tolist(),
                "prediction_translation": prediction.translation.tolist(),
                "prediction_quaternion_xyzw": (
                    prediction.quaternion_xyzw.tolist()
                ),
            }
            return prediction
        return original_predict_from_tracker(
            self, timestamp_ns, tracker_quaternion
        )

    def capture_update_split_surfels(
        previous, points, origins, voxel_size, offset
    ):
        nonlocal surfel_update_calls
        state = original_update_split_surfels(
            previous, points, origins, voxel_size, offset
        )
        if args.trace_surfel_key is not None and np.isclose(voxel_size, 0.10):
            wanted = np.asarray(args.trace_surfel_key, dtype=np.int64)
            matches = np.flatnonzero(np.all(state.keys == wanted, axis=1))
            record: dict[str, object] = {
                "update_call": surfel_update_calls,
                "retained_insertion": surfel_update_calls // 3,
                "present": bool(len(matches)),
            }
            if len(matches):
                index = int(matches[0])
                record.update({
                    "state_index": index,
                    "weight": float(state.weights[index]),
                    "count": int(state.counts[index]),
                    "mean": state.means[index].astype(np.float64).tolist(),
                    "covariance": state.covariances[index].astype(
                        np.float64
                    ).tolist(),
                    "viewpoint": state.viewpoints[index].astype(
                        np.float64
                    ).tolist(),
                    "secondary_weight": float(
                        state.secondary_weights[index]
                    ),
                    "secondary_count": int(
                        state.secondary_counts[index]
                    ),
                    "secondary_mean": state.secondary_means[index].astype(
                        np.float64
                    ).tolist(),
                    "secondary_covariance": (
                        state.secondary_covariances[index]
                        .astype(np.float64)
                        .tolist()
                    ),
                    "secondary_viewpoint": (
                        state.secondary_viewpoints[index]
                        .astype(np.float64)
                        .tolist()
                    ),
                    "is_split": int(state.is_split[index]),
                    "split_normal": state.split_normals[index].astype(
                        np.float64
                    ).tolist(),
                    "primary_dirty": int(state.primary_dirty[index]),
                    "secondary_dirty": int(state.secondary_dirty[index]),
                })
            surfel_key_trace.append(record)
        capture_surfel_state = (
            args.clean_surfel_state_capture is not None
            and (
                args.clean_surfel_state_capture_index is None
                or surfel_update_calls
                in args.clean_surfel_state_capture_index
            )
        )
        if capture_surfel_state:
            np.savez_compressed(
                args.clean_surfel_state_capture
                / f"update_{surfel_update_calls:03d}_{voxel_size:.2f}.npz",
                input_points=np.asarray(points, dtype=np.float32),
                input_origins=np.asarray(origins, dtype=np.float32),
                keys=state.keys,
                weights=state.weights,
                counts=state.counts,
                means=state.means,
                covariances=state.covariances,
                secondary_weights=state.secondary_weights,
                secondary_counts=state.secondary_counts,
                secondary_means=state.secondary_means,
                secondary_covariances=state.secondary_covariances,
                is_split=state.is_split,
                split_normals=state.split_normals,
                viewpoints=state.viewpoints,
                secondary_viewpoints=state.secondary_viewpoints,
                primary_dirty=state.primary_dirty,
                secondary_dirty=state.secondary_dirty,
            )
        surfel_update_calls += 1
        return state

    def capture_filter_scan(scan, config):
        nonlocal filter_calls
        capture_this_call = (
            args.clean_capture is not None
            and (
                args.clean_filter_capture_index is None
                or filter_calls == args.clean_filter_capture_index
            )
        )
        if capture_this_call:
            np.asarray(scan.points, dtype="<f4").tofile(
                args.clean_capture / f"node_{filter_calls:02d}_accumulated_points.bin"
            )
            if scan.ray_origins is not None:
                np.asarray(scan.ray_origins, dtype="<f4").tofile(
                    args.clean_capture
                    / f"node_{filter_calls:02d}_accumulated_origins.bin"
                )
            if scan.point_timestamps_ns is not None:
                np.asarray(scan.point_timestamps_ns, dtype="<i8").tofile(
                    args.clean_capture
                    / f"node_{filter_calls:02d}_accumulated_timestamps_ns.bin"
                )
        filtered = original_filter_scan(scan, config)
        if capture_this_call:
            np.asarray(filtered.points, dtype="<f4").tofile(
                args.clean_capture / f"node_{filter_calls:02d}_filtered_points.bin"
            )
            if filtered.ray_origins is not None:
                np.asarray(filtered.ray_origins, dtype="<f4").tofile(
                    args.clean_capture / f"node_{filter_calls:02d}_filtered_origins.bin"
                )
        filter_calls += 1
        return filtered

    def capture_relative_motion(self, timestamps_ns, target_timestamp_ns):
        nonlocal deskew_motion_calls, deskew_batch_active
        nonlocal float_quaternion_calls, float_matrix_calls
        motion_timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
        if args.replace_deskew_timestamp_ns is not None:
            old_timestamp_ns, new_timestamp_ns = args.replace_deskew_timestamp_ns
            replace = motion_timestamps_ns == old_timestamp_ns
            if np.any(replace):
                motion_timestamps_ns = motion_timestamps_ns.copy()
                motion_timestamps_ns[replace] = new_timestamp_ns
        quaternions, translations = original_relative_motion(
            self, motion_timestamps_ns, target_timestamp_ns
        )
        if (
            args.clean_deskew_capture is not None
            and deskew_motion_calls == args.clean_deskew_capture_index
        ):
            np.asarray(motion_timestamps_ns, dtype="<i8").tofile(
                args.clean_deskew_capture / "relative_timestamp_ns.bin"
            )
            np.asarray(quaternions, dtype="<f8").tofile(
                args.clean_deskew_capture / "relative_quaternion.bin"
            )
            np.asarray(translations, dtype="<f8").tofile(
                args.clean_deskew_capture / "relative_translation.bin"
            )
            np.asarray(self.tracker.orientation_xyzw, dtype="<f8").tofile(
                args.clean_deskew_capture / "tracker_end_quaternion.bin"
            )
            np.asarray(self.tracker._gravity_vector, dtype="<f8").tofile(
                args.clean_deskew_capture / "tracker_end_gravity.bin"
            )
            np.asarray([target_timestamp_ns], dtype="<i8").tofile(
                args.clean_deskew_capture / "target_timestamp_ns.bin"
            )
            float_quaternion_calls = 0
            float_matrix_calls = 0
            deskew_batch_active = True
        deskew_motion_calls += 1
        return quaternions, translations

    def capture_quaternion_transform(quaternion, values):
        nonlocal deskew_calls
        transformed = original_quaternion_transform(quaternion, values)
        quaternion_values = np.asarray(quaternion)
        point_values = np.asarray(values)
        if (
            args.clean_deskew_capture is not None
            and deskew_calls < 2
            and quaternion_values.ndim == 2
            and point_values.ndim == 2
            and quaternion_values.shape == (len(point_values), 4)
        ):
            np.asarray(quaternion_values, dtype="<f8").tofile(
                args.clean_deskew_capture / f"call_{deskew_calls:02d}_quaternion.bin"
            )
            np.asarray(point_values, dtype="<f8").tofile(
                args.clean_deskew_capture / f"call_{deskew_calls:02d}_input.bin"
            )
            np.asarray(transformed, dtype="<f8").tofile(
                args.clean_deskew_capture / f"call_{deskew_calls:02d}_output.bin"
            )
            deskew_calls += 1
        return transformed

    def capture_float_quaternion_transform(points, quaternions, translations):
        nonlocal float_quaternion_calls
        transformed = original_float_quaternion_transform(
            points, quaternions, translations
        )
        if (
            args.clean_deskew_capture is not None
            and deskew_batch_active
            and float_quaternion_calls < 2
        ):
            prefix = f"float_quaternion_call_{float_quaternion_calls:02d}"
            np.asarray(points, dtype="<f4").tofile(
                args.clean_deskew_capture / f"{prefix}_input.bin"
            )
            np.asarray(quaternions, dtype="<f8").tofile(
                args.clean_deskew_capture / f"{prefix}_quaternion.bin"
            )
            np.asarray(translations, dtype="<f8").tofile(
                args.clean_deskew_capture / f"{prefix}_translation.bin"
            )
            np.asarray(transformed, dtype="<f4").tofile(
                args.clean_deskew_capture / f"{prefix}_output.bin"
            )
        float_quaternion_calls += 1
        return transformed

    def capture_float_matrix_transform(points, pose):
        nonlocal float_matrix_calls, deskew_batch_active
        transformed = original_float_matrix_transform(points, pose)
        if (
            args.clean_deskew_capture is not None
            and deskew_batch_active
            and float_matrix_calls < 2
        ):
            prefix = f"float_matrix_call_{float_matrix_calls:02d}"
            np.asarray(points, dtype="<f4").tofile(
                args.clean_deskew_capture / f"{prefix}_input.bin"
            )
            np.asarray(
                np.concatenate((pose.translation, pose.quaternion_xyzw)),
                dtype="<f8",
            ).tofile(args.clean_deskew_capture / f"{prefix}_pose.bin")
            np.asarray(transformed, dtype="<f4").tofile(
                args.clean_deskew_capture / f"{prefix}_output.bin"
            )
        float_matrix_calls += 1
        if deskew_batch_active and float_matrix_calls >= 2:
            deskew_batch_active = False
        return transformed

    def capture_map_insert(self, node, **kwargs):
        capture_node = (
            args.clean_map_capture_index is None
            or node.node_id.index in args.clean_map_capture_index
        )
        capture_submap = (
            self.submap_id.index == 0
            if args.clean_map_capture_submap_index is None
            else self.submap_id.index in args.clean_map_capture_submap_index
        )
        if (
            args.clean_map_capture is not None
            and capture_submap
            and capture_node
        ):
            source_points = kwargs.get("map_points")
            source_normals = kwargs.get("map_normals")
            source_origins = kwargs.get("map_origins")
            if source_points is None:
                source_points = node.points
            if source_normals is None:
                source_normals = node.normals
            if source_origins is None:
                source_origins = np.zeros_like(source_points)
            batch = node.node_id.index
            prefix = f"submap_{self.submap_id.index:02d}_batch_{batch:04d}"
            np.asarray(
                np.hstack((source_origins, source_points)), dtype="<f4"
            ).tofile(args.clean_map_capture / f"{prefix}_input.bin")
            np.asarray(
                np.concatenate((
                    node.local_pose.translation,
                    node.local_pose.quaternion_xyzw,
                    self.local_pose.translation,
                    self.local_pose.quaternion_xyzw,
                )),
                dtype="<f8",
            ).tofile(args.clean_map_capture / f"{prefix}_poses.bin")
            points, _, origins, _ = frontend._transform_range_data_to_submap(
                node.local_pose,
                self.local_pose,
                source_points,
                source_normals,
                source_origins,
            )
            np.asarray(np.hstack((origins, points)), dtype="<f4").tofile(
                args.clean_map_capture / f"{prefix}.bin"
            )
        original_insert(self, node, **kwargs)

    def capture_and_force(
        source_points, target_points, target_normals, initial=None, **kwargs
    ):
        nonlocal icp_calls_seen
        call_index = icp_calls_seen
        icp_calls_seen += 1
        if initial is None:
            raise RuntimeError("binary-compatible local ICP requires an initial pose")
        if call_index < args.vendor_call_start:
            return original_icp(
                source_points,
                target_points,
                target_normals,
                initial,
                **kwargs,
            )
        prefix = vendor_prefix(call_index)
        vendor_initial = load_rigid(
            args.vendor_capture / f"{prefix}_initial.bin"
        )
        vendor_result = load_rigid(
            args.vendor_capture / f"{prefix}_result.bin"
        )
        clean_initial = vendor_initial if args.use_vendor_initial else initial
        clean_result = original_icp(
            source_points,
            target_points,
            target_normals,
            clean_initial,
            **kwargs,
        )
        clean_levels = tuple(np.asarray(value) for value in target_points)
        clean_normals = np.concatenate(tuple(np.asarray(value) for value in target_normals))
        level_reports = []
        for level, clean_level in enumerate(clean_levels):
            vendor_level = np.fromfile(
                args.vendor_capture
                / f"{prefix}_target_level_{level}.bin",
                dtype="<f4",
            ).reshape((-1, 3))
            level_reports.append(same_order_error(clean_level, vendor_level))
        vendor_normals = np.fromfile(
            args.vendor_capture / f"{prefix}_target_normals.bin",
            dtype="<f4",
        ).reshape((-1, 3))
        transformed_source = np.asarray(
            initial.rotation.apply(source_points) + initial.translation,
            dtype=np.float32,
        )
        if args.clean_capture is not None:
            np.asarray(source_points, dtype="<f4").tofile(
                args.clean_capture / f"call_{call_index:02d}_source.bin"
            )
            clean_origins_to_save = kwargs.get("source_origins")
            if clean_origins_to_save is not None:
                np.asarray(clean_origins_to_save, dtype="<f4").tofile(
                    args.clean_capture / f"call_{call_index:02d}_origins.bin"
                )
            for level, (clean_level, clean_normal_level) in enumerate(
                zip(clean_levels, target_normals)
            ):
                np.asarray(clean_level, dtype="<f4").tofile(
                    args.clean_capture
                    / f"call_{call_index:02d}_target_level_{level}.bin"
                )
                np.asarray(clean_normal_level, dtype="<f4").tofile(
                    args.clean_capture
                    / f"call_{call_index:02d}_normal_level_{level}.bin"
                )
            np.asarray(
                np.concatenate((
                    clean_initial.translation,
                    np.zeros(1, dtype=np.float64),
                    clean_initial.quaternion_xyzw,
                )),
                dtype="<f8",
            ).tofile(args.clean_capture / f"call_{call_index:02d}_initial.bin")
        vendor_rays = np.fromfile(
            args.vendor_capture / f"{prefix}_source.bin",
            dtype="<f4",
        ).reshape((-1, 6))
        clean_origins = kwargs.get("source_origins")
        calls.append({
            "call": call_index,
            "initial": pose_error(initial, vendor_initial),
            "initial_coefficients": pose_coefficient_error(
                initial, vendor_initial
            ),
            "source_raw": same_order_error(source_points, vendor_rays[:, 3:]),
            "source_origins": (
                same_order_error(clean_origins, vendor_rays[:, :3])
                if clean_origins is not None
                else None
            ),
            "source_transformed": same_order_error(
                transformed_source, vendor_rays[:, 3:]
            ),
            "target_levels": level_reports,
            "target_normals": same_order_error(clean_normals, vendor_normals),
            "clean_result": {
                **pose_error(clean_result.target_from_source, vendor_result),
                "coefficients": pose_coefficient_error(
                    clean_result.target_from_source, vendor_result
                ),
                "correspondences": int(clean_result.correspondences),
                "iterations": int(clean_result.iterations),
                "fitness_m": float(clean_result.fitness_m),
            },
        })
        if args.keep_clean_result:
            return clean_result
        return IcpResult(
            vendor_result,
            clean_result.fitness_m,
            clean_result.overlap,
            clean_result.correspondences,
            clean_result.iterations,
            clean_result.converged,
            clean_result.euclidean_fitness_m,
        )

    frontend.point_to_plane_icp = capture_and_force
    frontend.filter_scan = capture_filter_scan
    frontend.update_split_surfel_statistics = capture_update_split_surfels
    frontend.FrontendSubmap.insert = capture_map_insert
    frontend._quaternion_transform_vector = capture_quaternion_transform
    frontend._transform_points_float_quaternion = (
        capture_float_quaternion_transform
    )
    frontend._transform_points_float_matrix = capture_float_matrix_transform
    frontend.RawConstantVelocityPosePredictor.relative_motion = (
        capture_relative_motion
    )
    frontend.RawConstantVelocityPosePredictor._prediction_from_tracker_quaternion = (
        capture_predict_from_tracker
    )
    frontend.RawImuTracker.advance = capture_tracker_advance
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
            scans = iter(iter_archive_at_node_times(
                archive,
                [node.timestamp_ns for node in reference_nodes],
                imu_pose_predictor=predictor,
                discarded_batch_timestamps_ns=(
                    args.discarded_batch_timestamp_ns
                ),
            ))
            first_scan = next(scans)
            predictor.correct(
                first_scan.timestamp_ns, reference_nodes[0].local_pose
            )
            SurveyorFrontend().process(
                itertools.chain((first_scan,), scans),
                imu_pose_predictor=predictor,
                find_loops=False,
            )
    finally:
        frontend.point_to_plane_icp = original_icp
        frontend.filter_scan = original_filter_scan
        frontend.update_split_surfel_statistics = original_update_split_surfels
        frontend.FrontendSubmap.insert = original_insert
        frontend._quaternion_transform_vector = original_quaternion_transform
        frontend._transform_points_float_quaternion = (
            original_float_quaternion_transform
        )
        frontend._transform_points_float_matrix = original_float_matrix_transform
        frontend.RawConstantVelocityPosePredictor.relative_motion = (
            original_relative_motion
        )
        frontend.RawConstantVelocityPosePredictor._prediction_from_tracker_quaternion = (
            original_predict_from_tracker
        )
        frontend.RawImuTracker.advance = original_tracker_advance

    if args.clean_predictor_capture is not None:
        args.clean_predictor_capture.parent.mkdir(parents=True, exist_ok=True)
        args.clean_predictor_capture.write_text(
            json.dumps(predictor_capture, indent=2) + "\n"
        )

    if args.clean_tracker_state_capture is not None and tracker_states:
        sequence = max(tracker_states.values(), key=len)
        args.clean_tracker_state_capture.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.clean_tracker_state_capture,
            timestamps_ns=np.asarray([state[0] for state in sequence], dtype="<i8"),
            quaternions=np.vstack([state[1] for state in sequence]).astype("<f8"),
            gravity=np.vstack([state[2] for state in sequence]).astype("<f8"),
        )

    report = {
        "node_limit": len(reference_nodes),
        "calls": calls,
        "surfel_key": args.trace_surfel_key,
        "surfel_key_trace": surfel_key_trace,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
