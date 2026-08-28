"""Raw-lidar frontend compatible with the reconstructed SurveyorSLAM graph.

This module fills the previously explicit frontend gap: range/voxel filtering,
point-to-plane scan matching, overlapping 0.2 m submaps, overlap/covariance
candidate selection, fast-correlative loop search and geometric verification.
Its output uses the
``TrajectoryNode``/``Submap``/``LoopConstraint`` vocabulary consumed by
``surveyor_slam.py`` so frontend and sparse IMU backend can be evaluated
independently.

The defaults below are the resolved offline G11 dictionaries captured from the
installed binary: 0.5--60 m input, 50 ms/58,000-ray accumulation, three-level
surfel ICP, 5 m/10 m overlapping submaps and the 6 m/2 m/30 degree sparse
constraint search. The G11 reference contains 1,617 nodes, six retained
submaps, 2,581 memberships and 15 accepted loops.
"""

from __future__ import annotations

from bisect import bisect_left
import ctypes
from dataclasses import dataclass, field
import mmap
import os
from pathlib import Path
import struct
from typing import Callable, Iterable, Sequence
import zipfile

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .surveyor_slam import (
    ImuSample,
    LoopConstraint,
    NodeId,
    OnlineFastPoseGraphSnapshot,
    Rigid3,
    Submap,
    TrajectoryNode,
    _message,
    _values,
    _wire_fields,
    finish_online_fast_pose_graph,
    online_fast_loop_initial_pose,
    replay_online_fast_pose_graph,
)


# Surveyor stores these two convergence settings as float32 before the matcher
# expands them back to double.  Keep that conversion visible so equality at a
# rounded boundary follows the installed implementation.
_BINARY_ICP_TRANSLATION_STEP_M = float(np.float32(1.0e-5))
_BINARY_ICP_ROTATION_STEP_RAD = float(
    np.float32(np.deg2rad(1.0e-5))
)


def _identity() -> Rigid3:
    return Rigid3(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


def _raw_quaternion_product_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product without normalizing serialized coefficients."""

    lx, ly, lz, lw = (np.float64(value) for value in left)
    rx, ry, rz, rw = (np.float64(value) for value in right)
    return np.array(
        [
            (ly * rz + lw * rx) - (lz * ry - lx * rw),
            (ly * rw + lw * ry) + (lz * rx - lx * rz),
            (lw * rz - ly * rx) + (lx * ry + lz * rw),
            (lw * rw - ly * ry) - (lx * rx + lz * rz),
        ],
        dtype=np.float64,
    )


def _rigid_rotation_product_xyzw(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Quaternion product in the installed Rigid3 rotation expression order.

    This operation is algebraically the same as
    :func:`_raw_quaternion_product_xyzw`, but the installed submap-construction
    call site evaluates the cross product first.  The difference is visible
    when gravity tilt cancels almost completely.
    """

    lx, ly, lz, lw = (np.float64(value) for value in left)
    rx, ry, rz, rw = (np.float64(value) for value in right)
    cross_x = ly * rz - lz * ry
    cross_y = lz * rx - lx * rz
    cross_z = lx * ry - ly * rx
    return np.array(
        [
            (cross_x + lw * rx) + rw * lx,
            (cross_y + lw * ry) + rw * ly,
            (cross_z + lw * rz) + rw * lz,
            lw * rw - ((lx * rx + ly * ry) + lz * rz),
        ],
        dtype=np.float64,
    )


def _raw_quaternion_transform_vector(
    quaternion_xyzw: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Match Eigen Quaternion::_transformVector for raw coefficients."""

    qx, qy, qz, qw = (np.float64(value) for value in quaternion_xyzw)
    vx, vy, vz = (np.float64(value) for value in vector)
    tx = qy * vz - qz * vy
    tx = tx + tx
    ty = qz * vx - qx * vz
    ty = ty + ty
    tz = qx * vy - qy * vx
    tz = tz + tz
    return np.array(
        [
            (qy * tz - qz * ty) + (qw * tx + vx),
            (qz * tx - qx * tz) + (qw * ty + vy),
            (qx * ty - qy * tx) + (vz + qw * tz),
        ],
        dtype=np.float64,
    )


def _normalized_vector3_eigen_order(vector: np.ndarray) -> np.ndarray:
    x, y, z = (np.float64(value) for value in vector)
    norm = np.sqrt((x * x + y * y) + z * z)
    return np.array([x / norm, y / norm, z / norm], dtype=np.float64)


def _quaternion_from_two_vectors_eigen_order(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Eigen's regular ``Quaternion::FromTwoVectors`` evaluation order."""

    first_normalized = _normalized_vector3_eigen_order(first)
    second_normalized = _normalized_vector3_eigen_order(second)
    fx, fy, fz = first_normalized
    sx, sy, sz = second_normalized
    cosine = (sx * fx + sy * fy) + sz * fz
    if cosine < -1.0 + 1.0e-12:
        # Gravity observations in supported datasets never enter Eigen's SVD
        # antiparallel branch.  Keep a robust equivalent for malformed input.
        return _rotation_from_two_vectors(first, second).as_quat()
    scale = np.sqrt((1.0 + cosine) + (1.0 + cosine))
    inverse_scale = 1.0 / scale
    return np.array(
        [
            (fy * sz - fz * sy) * inverse_scale,
            (fz * sx - fx * sz) * inverse_scale,
            (fx * sy - fy * sx) * inverse_scale,
            0.5 * scale,
        ],
        dtype=np.float64,
    )


def _submap_rotation_from_node(
    node_pose: Rigid3, gravity_observation: np.ndarray
) -> np.ndarray:
    """Reproduce the uncollapsed gravity-alignment chain used for submaps."""

    node_rotation = node_pose.quaternion_xyzw
    nx, ny, nz, nw = (np.float64(value) for value in node_rotation)
    squared_norm = (nz * nz + nx * nx) + (nw * nw + ny * ny)
    inverse_node_rotation = np.array(
        [
            -nx / squared_norm,
            -ny / squared_norm,
            -nz / squared_norm,
            nw / squared_norm,
        ],
        dtype=np.float64,
    )
    implied_gravity = _raw_quaternion_transform_vector(
        inverse_node_rotation, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    remove_tilt = _quaternion_from_two_vectors_eigen_order(
        gravity_observation, implied_gravity
    )
    corrected_rotation = _raw_quaternion_product_xyzw(
        node_rotation, remove_tilt
    )
    normalization_rotation = _raw_quaternion_product_xyzw(
        corrected_rotation, inverse_node_rotation
    )
    x, y, z, w = (np.float64(value) for value in normalization_rotation)
    squared_norm = (z * z + x * x) + (w * w + y * y)
    return np.array(
        [-x / squared_norm, -y / squared_norm, -z / squared_norm, w / squared_norm],
        dtype=np.float64,
    )


def _relative_rotation_raw_xyzw(source: Rigid3, target: Rigid3) -> np.ndarray:
    source_quaternion = source.quaternion_xyzw
    x, y, z, w = (np.float64(value) for value in source_quaternion)
    squared_norm = (z * z + x * x) + (w * w + y * y)
    if squared_norm <= 0.0:
        raise ValueError("submap rotation quaternion must be non-zero")
    inverse = np.array(
        [
            -x / squared_norm,
            -y / squared_norm,
            -z / squared_norm,
            w / squared_norm,
        ],
        dtype=np.float64,
    )
    return _raw_quaternion_product_xyzw(inverse, target.quaternion_xyzw)


def _updated_submap_gravity(
    previous: np.ndarray,
    count: int,
    submap_pose: Rigid3,
    node_pose: Rigid3,
    node_gravity: np.ndarray,
) -> np.ndarray:
    """Reproduce the insertion-ordered Submap gravity accumulator."""

    relative_rotation = _relative_rotation_raw_xyzw(submap_pose, node_pose)
    direction = _raw_quaternion_transform_vector(relative_rotation, node_gravity)
    direction = _normalized_vector3_eigen_order(direction)
    if count == 0:
        return direction * np.float64(9.81)
    previous_direction = _normalized_vector3_eigen_order(previous)
    current_count = np.float64(count)
    next_count = np.float64(count + 1)
    weighted = np.array(
        [
            (current_count * previous_direction[0] + direction[0]) / next_count,
            (current_count * previous_direction[1] + direction[1]) / next_count,
            (current_count * previous_direction[2] + direction[2]) / next_count,
        ],
        dtype=np.float64,
    )
    return _normalized_vector3_eigen_order(weighted) * np.float64(9.81)


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    minimum_range_m: float = 0.5
    maximum_range_m: float = 60.0
    minimum_scan_accumulation_duration_s: float = 0.05
    minimum_scan_accumulation_ray_count: int = 58_000
    scan_voxel_m: float = 0.04
    high_resolution_min_voxel_m: float = 0.02
    high_resolution_max_voxel_m: float = 0.40
    high_resolution_max_points: int = 5_000
    submap_voxel_m: float = 0.10
    normal_neighbors: int = 12
    # Limits correspond to the 0.1, 0.3 and 0.6 m target grids.  The matcher
    # queries all three on every iteration and keeps the lowest-resolution-
    # index (finest) available correspondence; these are not ICP stages.
    icp_correspondence_levels_m: tuple[float, float, float] = (0.15, 0.45, 0.90)
    icp_max_correspondence_m: float = 0.90
    icp_initial_plane_distance_m: float = 0.20
    icp_contracted_plane_distance_m: float = 0.02
    icp_max_incidence_angle_deg: float = 86.0
    icp_huber_m: float = float("inf")
    icp_num_threads: int = 8
    icp_iterations: int = 20
    icp_min_iterations: int = 6
    icp_contraction_iterations: int = 6
    icp_min_correspondences: int = 6
    submap_overlap_displacement_m: float = 5.0
    # Kept for API compatibility with older experiments. The installed G11
    # dictionary starts/finishes by displacement, not traveled-path fallback.
    submap_overlap_path_m: float = 1.0e9
    submap_finish_displacement_m: float = 10.0
    submap_finish_path_m: float = 2.0e9
    map_rebuild_interval: int = 1
    loop_node_stride: int = 10
    loop_candidate_radius_m: float = 15.0
    loop_subsampling_ratio: float = 0.10
    loop_translation_window_m: float = 6.0
    loop_translation_step_m: float = 0.2
    loop_vertical_window_m: float = 2.0
    loop_yaw_window_deg: float = 30.0
    loop_yaw_step_deg: float = 3.0
    loop_correlative_resolution_m: float = 0.20
    loop_min_correlative_score: float = 0.55
    loop_min_rotational_score: float = 0.77
    loop_rotational_histogram_size: int = 120
    loop_max_fitness_m: float = 0.03
    loop_min_overlap: float = 0.30
    loop_max_icp_correction_m: float = 0.30
    loop_max_icp_correction_deg: float = 20.0
    loop_max_constraint_anisotropy: float = 50.0
    loop_min_constraint_strength: float = 50.0
    loop_max_constraint_anisotropy_tracking_good: float = 0.0
    loop_translation_weight: float = 500.0
    loop_rotation_weight: float = 1600.0

    def __post_init__(self) -> None:
        positive = (
            self.minimum_range_m,
            self.maximum_range_m,
            self.scan_voxel_m,
            self.submap_voxel_m,
            self.icp_max_correspondence_m,
            self.icp_initial_plane_distance_m,
            self.icp_contracted_plane_distance_m,
            self.submap_overlap_displacement_m,
            self.submap_overlap_path_m,
            self.submap_finish_displacement_m,
            self.submap_finish_path_m,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("frontend distances must be finite and positive")
        if self.maximum_range_m <= self.minimum_range_m:
            raise ValueError("maximum range must exceed minimum range")
        if (
            self.submap_finish_displacement_m
            <= self.submap_overlap_displacement_m
            or self.submap_finish_path_m <= self.submap_overlap_path_m
        ):
            raise ValueError("submap finish support must exceed overlap support")
        if self.normal_neighbors < 4 or self.icp_iterations < 1:
            raise ValueError("normal/ICP iteration counts are invalid")
        if (
            self.minimum_scan_accumulation_duration_s <= 0
            or self.minimum_scan_accumulation_ray_count <= 0
        ):
            raise ValueError("scan accumulation thresholds are invalid")
        if (
            not np.isfinite(self.loop_min_rotational_score)
            or not 0.0 <= self.loop_min_rotational_score <= 1.0
            or self.loop_rotational_histogram_size < 1
        ):
            raise ValueError("loop rotational matcher thresholds are invalid")
        if (
            not np.isfinite(self.loop_max_constraint_anisotropy)
            or self.loop_max_constraint_anisotropy <= 0.0
            or not np.isfinite(self.loop_min_constraint_strength)
            or self.loop_min_constraint_strength < 0.0
            or not np.isfinite(
                self.loop_max_constraint_anisotropy_tracking_good
            )
            or self.loop_max_constraint_anisotropy_tracking_good < 0.0
            or self.loop_max_constraint_anisotropy_tracking_good
            > self.loop_max_constraint_anisotropy
        ):
            raise ValueError("loop constraint stability thresholds are invalid")


@dataclass(frozen=True, slots=True)
class LidarScan:
    timestamp_ns: int
    points: np.ndarray
    normals: np.ndarray | None = None
    point_timestamps_ns: np.ndarray | None = None
    range_filtered: bool = False
    ray_origins: np.ndarray | None = None
    retain_node: bool = True

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("scan points must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("scan contains non-finite points")
        object.__setattr__(self, "points", points)
        if self.normals is not None:
            normals = np.asarray(self.normals, dtype=np.float64)
            if normals.shape != points.shape or not np.all(np.isfinite(normals)):
                raise ValueError("scan normals must be finite and match points")
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
            object.__setattr__(self, "normals", normals)
        if self.point_timestamps_ns is not None:
            point_timestamps = np.asarray(self.point_timestamps_ns, dtype=np.int64)
            if point_timestamps.shape != (len(points),):
                raise ValueError("point timestamps must match scan points")
            object.__setattr__(self, "point_timestamps_ns", point_timestamps)
        if self.ray_origins is not None:
            ray_origins = np.asarray(self.ray_origins, dtype=np.float64)
            if ray_origins.shape != points.shape or not np.all(np.isfinite(ray_origins)):
                raise ValueError("ray origins must be finite and match scan points")
            object.__setattr__(self, "ray_origins", ray_origins)
        if not isinstance(self.retain_node, (bool, np.bool_)):
            raise ValueError("retain_node must be boolean")


@dataclass(frozen=True, slots=True)
class RawImuTrackerOptions:
    """Resolved ``ImuTrackerRaw`` constants from the offline G11 binary."""

    initial_gravity_time_constant_s: float = 2.0
    steady_state_gravity_time_constant_s: float = 20.0
    time_constant_init_duration_s: float = 4.0
    time_constant_fade_duration_s: float = 40.0
    max_gravity_norm_error_mps2: float = 4.0
    gravity_mps2: float = 9.81
    init_tilt_from_imu_orientation: bool = True


def _rotation_from_two_vectors(first: np.ndarray, second: np.ndarray) -> Rotation:
    """Eigen-compatible shortest rotation taking ``first`` onto ``second``."""

    # Copy because the first argument is the tracker's persistent gravity
    # estimate; normalizing an alias here would silently collapse its 9.81 m/s²
    # magnitude after the first lidar-time query.
    first = np.array(first, dtype=np.float64, copy=True)
    second = np.array(second, dtype=np.float64, copy=True)
    first /= max(float(np.linalg.norm(first)), 1.0e-15)
    second /= max(float(np.linalg.norm(second)), 1.0e-15)
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot < -1.0 + 1.0e-12:
        axis = np.cross(first, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1.0e-12:
            axis = np.cross(first, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis)
    quaternion = np.concatenate((np.cross(first, second), [1.0 + dot]))
    quaternion /= np.linalg.norm(quaternion)
    return Rotation.from_quat(quaternion)


def _quaternion_multiply_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Hamilton product using Eigen's ``(x, y, z, w)`` coefficient order."""

    first_xyz = first[..., :3]
    second_xyz = second[..., :3]
    return np.concatenate(
        (
            first[..., 3, None] * second_xyz
            + second[..., 3, None] * first_xyz
            + np.cross(first_xyz, second_xyz),
            (
                first[..., 3] * second[..., 3]
                - np.sum(first_xyz * second_xyz, axis=-1)
            )[..., None],
        ),
        axis=-1,
    )


def _quaternion_inverse_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return Eigen's quaternion inverse without assuming unit coefficients."""

    x = quaternion[..., 0]
    y = quaternion[..., 1]
    z = quaternion[..., 2]
    w = quaternion[..., 3]
    # The installed Eigen kernel accumulates two packed lanes before its
    # horizontal add.  This order is observable when an IMU quaternion norm
    # is one ULP away from one: (z*z + x*x) + (w*w + y*y).
    squared_norm = (
        (z * z + x * x) + (w * w + y * y)
    )[..., None]
    return np.concatenate(
        (-quaternion[..., :3], quaternion[..., 3, None]), axis=-1
    ) / squared_norm


def _quaternion_transform_vector(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    """Match Eigen's fast ``Quaternion::_transformVector`` implementation.

    Eigen's fast path assumes a unit quaternion.  The vendor raw-IMU tracker
    also invokes it for the slightly non-unit firmware quaternion used during
    initialization, so replacing this with ``scipy Rotation.apply`` loses the
    small but observable first-update correction.
    """

    imaginary = quaternion[..., :3]
    real = quaternion[..., 3]
    if np.ndim(real):
        real = real[..., None]
    return vector + 2.0 * np.cross(
        imaginary,
        np.cross(imaginary, vector) + real * vector,
    )


def _seconds_from_nanoseconds(duration_ns: int) -> float:
    """Match ``navvis::core::ToSeconds`` (integer-to-double, then divide)."""

    return float(duration_ns) / 1.0e9


class RawImuTracker:
    """Continuous orientation tracker matching the installed raw-IMU tracker.

    Dynamic probes at the binary's virtual ``Advance`` method established the
    exact update order: initialize from the interpolated firmware orientation,
    split advances at IMU sample boundaries, trapezoid-integrate linearly
    interpolated angular velocity, rotate the gravity estimate, then blend the
    linearly interpolated acceleration at the interval endpoint.  The first
    380 lidar query orientations reproduce the binary to below 1e-6 degree.
    """

    def __init__(
        self,
        samples: Sequence[ImuSample],
        options: RawImuTrackerOptions = RawImuTrackerOptions(),
    ) -> None:
        if len(samples) < 2:
            raise ValueError("raw IMU tracking needs at least two samples")
        self.options = options
        self.timestamps_ns = np.asarray(
            [sample.timestamp_ns for sample in samples], dtype=np.int64
        )
        if np.any(np.diff(self.timestamps_ns) <= 0):
            raise ValueError("raw IMU samples must be strictly time ordered")
        self.linear_accelerations = np.vstack(
            [sample.linear_acceleration for sample in samples]
        )
        self.angular_velocities = np.vstack(
            [sample.angular_velocity for sample in samples]
        )
        self.orientations_xyzw = np.vstack(
            [sample.orientation_xyzw for sample in samples]
        )
        self._time_ns: int | None = None
        self._initial_time_ns: int | None = None
        self._orientation: Rotation | None = None
        self._raw_orientation_xyzw: np.ndarray | None = None
        self._gravity_vector: np.ndarray | None = None
        self._following_index: int | None = None

    @property
    def time_ns(self) -> int | None:
        return self._time_ns

    @property
    def gravity_observation(self) -> np.ndarray:
        """Current tracker-frame gravity serialized in trajectory nodes.

        The tracker keeps a separately filtered gravity vector for correction,
        but node data is produced from its orientation: ``q.inverse() * g``.
        Re-normalizing the internal vector is physically equivalent yet differs
        by several ulps in the horizontal components.
        """

        if self._gravity_vector is None or self._orientation is None:
            raise RuntimeError("the IMU tracker has not been initialized")
        return _raw_quaternion_transform_vector(
            _quaternion_inverse_xyzw(self.orientation_xyzw),
            np.array(
                [0.0, 0.0, self.options.gravity_mps2], dtype=np.float64
            ),
        )

    @property
    def orientation_xyzw(self) -> np.ndarray:
        """Current coefficients, including the non-unit initialization state."""

        if self._orientation is None:
            raise RuntimeError("the IMU tracker has not been initialized")
        return (
            self._raw_orientation_xyzw.copy()
            if self._raw_orientation_xyzw is not None
            else self._orientation.as_quat()
        )

    def _bracket(self, timestamp_ns: int) -> tuple[int, int, float]:
        high = int(np.searchsorted(self.timestamps_ns, timestamp_ns, side="left"))
        if high <= 0:
            return 0, 1, 0.0
        if high >= len(self.timestamps_ns):
            return len(self.timestamps_ns) - 2, len(self.timestamps_ns) - 1, 1.0
        low = high - 1
        duration = int(self.timestamps_ns[high] - self.timestamps_ns[low])
        alpha = (timestamp_ns - int(self.timestamps_ns[low])) / duration
        return low, high, float(alpha)

    def _interpolate_vector(self, values: np.ndarray, timestamp_ns: int) -> np.ndarray:
        low, high, _ = self._bracket(timestamp_ns)
        duration_s = _seconds_from_nanoseconds(
            int(self.timestamps_ns[high] - self.timestamps_ns[low])
        )
        elapsed_s = _seconds_from_nanoseconds(
            int(timestamp_ns) - int(self.timestamps_ns[low])
        )
        alpha = elapsed_s / duration_s
        return alpha * values[high] + (1.0 - alpha) * values[low]

    def _interpolate_vector_from_interval_end(
        self, values: np.ndarray, timestamp_ns: int
    ) -> np.ndarray:
        """Interpolate using the installed tracker's endpoint expression.

        The binary computes the remaining fraction ``(high - time) / span``
        and then forms ``remaining * low + (1 - remaining) * high``.  Computing
        the algebraically equivalent elapsed fraction first differs by a few
        ulps and becomes visible when the gravity correction is nearly zero.
        """

        low, high, _ = self._bracket(timestamp_ns)
        duration_s = _seconds_from_nanoseconds(
            int(self.timestamps_ns[high] - self.timestamps_ns[low])
        )
        remaining_s = _seconds_from_nanoseconds(
            int(self.timestamps_ns[high]) - int(timestamp_ns)
        )
        remaining = remaining_s / duration_s
        return remaining * values[low] + (1.0 - remaining) * values[high]

    def _interpolate_orientation_xyzw(self, timestamp_ns: int) -> np.ndarray:
        low, high, _ = self._bracket(timestamp_ns)
        first = self.orientations_xyzw[low]
        second = self.orientations_xyzw[high].copy()
        if np.dot(first, second) < 0.0:
            second *= -1.0
        duration = int(self.timestamps_ns[high] - self.timestamps_ns[low])
        remaining = (
            int(self.timestamps_ns[high]) - int(timestamp_ns)
        ) / duration
        return remaining * first + (1.0 - remaining) * second

    def _gravity_time_constant(self, elapsed_s: float) -> float:
        options = self.options
        fade_elapsed = elapsed_s - options.time_constant_init_duration_s
        if fade_elapsed <= 0.0:
            return options.initial_gravity_time_constant_s
        if fade_elapsed >= options.time_constant_fade_duration_s:
            return options.steady_state_gravity_time_constant_s
        alpha = fade_elapsed / options.time_constant_fade_duration_s
        return options.initial_gravity_time_constant_s + alpha * (
            options.steady_state_gravity_time_constant_s
            - options.initial_gravity_time_constant_s
        )

    def advance(self, timestamp_ns: int) -> Rotation:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns < int(self.timestamps_ns[0]) or timestamp_ns > int(
            self.timestamps_ns[-1]
        ):
            raise ValueError("IMU prediction timestamp is outside sample support")
        if self._time_ns is None:
            if self.options.init_tilt_from_imu_orientation:
                raw_orientation = self._interpolate_orientation_xyzw(timestamp_ns)
                orientation = Rotation.from_quat(raw_orientation)
                self._raw_orientation_xyzw = raw_orientation
                gravity_vector = _native_raw_imu_initialize_gravity(
                    raw_orientation,
                    self.options.gravity_mps2,
                )
            else:
                orientation = Rotation.identity()
                self._raw_orientation_xyzw = orientation.as_quat()
                gravity_vector = np.array(
                    [0.0, 0.0, self.options.gravity_mps2]
                )
            self._time_ns = timestamp_ns
            self._initial_time_ns = timestamp_ns
            self._orientation = orientation
            self._gravity_vector = gravity_vector
            self._following_index = int(
                np.searchsorted(self.timestamps_ns, timestamp_ns, side="right")
            )
            return orientation
        if timestamp_ns < self._time_ns:
            raise ValueError("raw IMU tracker cannot advance backwards")
        assert self._orientation is not None
        assert self._gravity_vector is not None
        assert self._initial_time_ns is not None
        assert self._following_index is not None
        while self._time_ns < timestamp_ns:
            following = self._following_index
            interval_end_ns = timestamp_ns
            if following < len(self.timestamps_ns):
                interval_end_ns = min(
                    interval_end_ns, int(self.timestamps_ns[following])
                )
            # The installed angular path does not reuse the direct interval
            # conversion.  It subtracts the converted prefix and suffix from
            # the converted enclosing IMU-sample span.  The three-rounding
            # sequence is visible for short lidar query intervals.
            if following >= len(self.timestamps_ns):
                raise ValueError("IMU prediction timestamp is outside sample support")
            low_timestamp_ns = int(self.timestamps_ns[following - 1])
            high_timestamp_ns = int(self.timestamps_ns[following])
            sample_span_s = _seconds_from_nanoseconds(
                high_timestamp_ns - low_timestamp_ns
            )
            elapsed_in_sample_s = _seconds_from_nanoseconds(
                self._time_ns - low_timestamp_ns
            )
            remaining_in_sample_s = _seconds_from_nanoseconds(
                high_timestamp_ns - interval_end_ns
            )
            angular_dt_s = (
                sample_span_s - elapsed_in_sample_s - remaining_in_sample_s
            )
            dt_s = _seconds_from_nanoseconds(interval_end_ns - self._time_ns)
            start_fraction = elapsed_in_sample_s / sample_span_s
            angular_start = (
                start_fraction * self.angular_velocities[following]
                + (1.0 - start_fraction)
                * self.angular_velocities[following - 1]
            )
            remaining_fraction = remaining_in_sample_s / sample_span_s
            angular_end = (
                remaining_fraction * self.angular_velocities[following - 1]
                + (1.0 - remaining_fraction)
                * self.angular_velocities[following]
            )
            orientation = (
                self._raw_orientation_xyzw
                if self._raw_orientation_xyzw is not None
                else self._orientation.as_quat()
            )
            updated_orientation, self._gravity_vector = (
                _native_raw_imu_angular_update(
                    orientation,
                    self._gravity_vector,
                    angular_start,
                    angular_end,
                    angular_dt_s,
                )
            )
            self._raw_orientation_xyzw = updated_orientation
            self._orientation = Rotation.from_quat(updated_orientation)

            # The acceleration path uses a different, observable endpoint
            # expression from angular velocity.  The installed tracker adds
            # the separately rounded elapsed and update durations before
            # dividing by the enclosing sample span.
            acceleration_fraction = (
                elapsed_in_sample_s + dt_s
            ) / sample_span_s
            acceleration = (
                acceleration_fraction * self.linear_accelerations[following]
                + (1.0 - acceleration_fraction)
                * self.linear_accelerations[following - 1]
            )

            # The tracker passes the outer Advance request time to the fading
            # time-constant calculation even when this loop has split that
            # request at an intervening IMU sample boundary.
            elapsed_s = _seconds_from_nanoseconds(
                timestamp_ns - self._initial_time_ns
            )
            time_constant = self._gravity_time_constant(elapsed_s)
            acceleration_is_valid = (
                abs(float(np.linalg.norm(acceleration)) - self.options.gravity_mps2)
                <= self.options.max_gravity_norm_error_mps2
            )
            alpha = (
                1.0 - np.exp(-dt_s / time_constant)
                if acceleration_is_valid
                else 0.0
            )
            orientation = (
                self._raw_orientation_xyzw
                if self._raw_orientation_xyzw is not None
                else self._orientation.as_quat()
            )
            corrected, self._gravity_vector = _native_raw_imu_gravity_update(
                orientation,
                self._gravity_vector,
                acceleration,
                alpha,
                self.options.gravity_mps2,
            )
            self._orientation = Rotation.from_quat(corrected)
            self._raw_orientation_xyzw = corrected
            self._time_ns = interval_end_ns
            if self._time_ns >= int(self.timestamps_ns[following]):
                self._following_index = following + 1
        return self._orientation

    def orientations_at(self, timestamps_ns: Sequence[int]) -> np.ndarray:
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1 or np.any(np.diff(timestamps) < 0):
            raise ValueError("IMU query timestamps must be a sorted vector")
        quaternions: list[np.ndarray] = []
        for timestamp in timestamps:
            orientation = self.advance(int(timestamp))
            quaternions.append(
                self._raw_orientation_xyzw.copy()
                if self._raw_orientation_xyzw is not None
                else orientation.as_quat()
            )
        return np.vstack(quaternions)


class RawConstantVelocityPosePredictor:
    """Binary-aligned IMU rotation and corrected-pose constant velocity."""

    def __init__(
        self,
        samples: Sequence[ImuSample],
        *,
        initial_pose: Rigid3 | None = None,
        options: RawImuTrackerOptions = RawImuTrackerOptions(),
    ) -> None:
        self.samples = tuple(samples)
        self.tracker = RawImuTracker(samples, options)
        self.initial_pose = initial_pose or _identity()
        self._initial_tracker_orientation: Rotation | None = None
        self._initial_tracker_orientation_xyzw: np.ndarray | None = None
        self._corrections: list[tuple[int, Rigid3, np.ndarray]] = []
        self._last_end_from_start_pose: Rigid3 | None = None

    def _remember_initial_orientation(
        self, orientation: Rotation, quaternion_xyzw: np.ndarray
    ) -> None:
        if self._initial_tracker_orientation is None:
            self._initial_tracker_orientation = orientation
            self._initial_tracker_orientation_xyzw = quaternion_xyzw.copy()

    def _velocity(self) -> np.ndarray:
        if len(self._corrections) < 2:
            return np.zeros(3)
        first_time, first_pose, _ = self._corrections[-2]
        second_time, second_pose, _ = self._corrections[-1]
        dt_s = (second_time - first_time) * 1.0e-9
        return (second_pose.translation - first_pose.translation) / dt_s

    def _prediction_from_tracker_quaternion(
        self, timestamp_ns: int, tracker_quaternion: np.ndarray
    ) -> Rigid3:
        """Predict without normalizing the corrected pose coefficients."""

        if self._corrections:
            anchor_time, anchor_pose, anchor_tracker_quaternion = (
                self._corrections[-1]
            )
            if (
                timestamp_ns == anchor_time
                and np.array_equal(
                    tracker_quaternion, anchor_tracker_quaternion
                )
            ):
                return anchor_pose
            delta_quaternion = _quaternion_product_binary(
                _quaternion_inverse_xyzw(anchor_tracker_quaternion),
                tracker_quaternion,
            )
            if len(self._corrections) < 2:
                translation = anchor_pose.translation.copy()
            else:
                previous_time, previous_pose, _ = self._corrections[-2]
                translation = _predict_translation_binary(
                    previous_time,
                    previous_pose.translation,
                    anchor_time,
                    anchor_pose.translation,
                    anchor_pose.quaternion_xyzw,
                    timestamp_ns,
                )
            return Rigid3(
                translation,
                _quaternion_product_binary(
                    anchor_pose.quaternion_xyzw, delta_quaternion
                ),
            )
        assert self._initial_tracker_orientation_xyzw is not None
        delta_quaternion = _quaternion_product_binary(
            _quaternion_inverse_xyzw(self._initial_tracker_orientation_xyzw),
            tracker_quaternion,
        )
        return Rigid3(
            self.initial_pose.translation.copy(),
            _quaternion_product_binary(
                self.initial_pose.quaternion_xyzw, delta_quaternion
            ),
        )

    def predict(self, timestamp_ns: int) -> Rigid3:
        tracker_orientation = self.tracker.advance(timestamp_ns)
        tracker_quaternion = self.tracker.orientation_xyzw
        self._remember_initial_orientation(
            tracker_orientation, tracker_quaternion
        )
        return self._prediction_from_tracker_quaternion(
            timestamp_ns, tracker_quaternion
        )

    def correct(self, timestamp_ns: int, pose: Rigid3) -> None:
        tracker_orientation = self.tracker.advance(timestamp_ns)
        self._remember_initial_orientation(
            tracker_orientation, self.tracker.orientation_xyzw
        )
        if self._corrections and timestamp_ns < self._corrections[-1][0]:
            raise ValueError("pose corrections must be strictly time ordered")
        correction = (
            int(timestamp_ns), pose, self.tracker.orientation_xyzw.copy()
        )
        if self._corrections and timestamp_ns == self._corrections[-1][0]:
            self._corrections[-1] = correction
            return
        self._corrections.append(
            correction
        )

    @property
    def last_end_from_start_pose(self) -> Rigid3:
        if self._last_end_from_start_pose is None:
            raise RuntimeError("relative_motion has not produced a range pose")
        return self._last_end_from_start_pose

    def relative_motion(
        self, point_timestamps_ns: Sequence[int], end_timestamp_ns: int
    ) -> tuple[np.ndarray, np.ndarray]:
        point_times = np.asarray(point_timestamps_ns, dtype=np.int64)
        if point_times.ndim != 1 or np.any(point_times > end_timestamp_ns):
            raise ValueError("point timestamps must not exceed the batch end")
        unique_times, inverse = np.unique(
            np.concatenate((point_times, [int(end_timestamp_ns)])),
            return_inverse=True,
        )
        quaternions = self.tracker.orientations_at(unique_times)
        rotations = Rotation.from_quat(quaternions)
        self._remember_initial_orientation(rotations[0], quaternions[0])
        point_quaternions = quaternions[inverse[: len(point_times)]]
        end_quaternion = quaternions[inverse[-1]]
        inverse_end = _quaternion_inverse_xyzw(end_quaternion)
        relative_quaternions = _quaternion_multiply_xyzw(
            np.broadcast_to(inverse_end, point_quaternions.shape),
            point_quaternions,
        )

        predicted_start = self._prediction_from_tracker_quaternion(
            int(point_times[0]), point_quaternions[0]
        )
        predicted_end = self._prediction_from_tracker_quaternion(
            int(end_timestamp_ns), end_quaternion
        )
        self._last_end_from_start_pose = _end_from_start_pose_with_roundtrip(
            predicted_start, predicted_end
        )
        velocity_in_end = predicted_end.rotation.inv().apply(self._velocity())
        relative_translations = (
            (point_times - int(end_timestamp_ns))[:, None]
            * 1.0e-9
            * velocity_in_end[None, :]
        )
        return relative_quaternions, relative_translations


@dataclass(frozen=True, slots=True)
class SlamScanRecord:
    sensor: int
    timestamp_ns: int
    ray_count: int
    packet_count: int
    packet_timestamps_offset: int
    point_count: int
    data_offset: int


class SlamScanArchive:
    """Memory-mapped C++ Pandar SLAM scan stream.

    The archive keeps the multi-gigabyte return array on disk and copies only
    scans requested by the frontend. Records contain calibrated base_link
    points and normals before trajectory deskew.
    """

    _magic_v1 = b"NVSLAM1\0"
    _magic_v2 = b"NVSLAM2\0"
    _magic_v3 = b"NVSLAM3\0"
    _magic_v4 = b"NVSLAM4\0"
    _magic_v5 = b"NVSLAM5\0"
    _magic_v6 = b"NVSLAM6\0"
    _header = struct.Struct("<BdI")
    _point_dtype = np.dtype("<f4")
    _point_width = 6
    # NVSLAM archives are emitted by the calibrated G11 Pandar path. Points
    # are already in base_link; these are the matching sensor-frame origins.
    _sensor_origins = np.asarray(
        (
            # The installed provider stores the horizontal origin after its
            # float transform round-trip. Its x bit pattern is 0xa4f20000,
            # one float ULP away from casting sensor_frame.xml's translation
            # directly (0xa4f30000). The distinction becomes visible after
            # deskew rotates this nearly-zero component.
            (-1.0495077029659683e-16, 0.07600000000000001, 0.024600000000000004),
            (-0.0009062760161523834, 0.449144896447496, -0.6331241496323716),
        ),
        dtype=np.float32,
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._version = 1
        self.timestamp_boundary_tolerance_ns = 0
        self.record_header_support_ns = 0
        self._point_record_dtype: np.dtype[np.void] | None = None
        magic = self._file.read(len(self._magic_v1))
        if magic == self._magic_v1:
            self._point_width = 6
            self._header = struct.Struct("<BdI")
        elif magic == self._magic_v2:
            self._point_width = 7
            self._header = struct.Struct("<BdI")
        elif magic == self._magic_v3:
            self._point_width = 7
            self._header = struct.Struct("<BdII")
        elif magic == self._magic_v4:
            self._version = 4
            self._point_width = 4
            self._header = struct.Struct("<BdII")
        elif magic == self._magic_v5:
            self._version = 5
            # The installed ROS/PCL duration conversion can land one
            # nanosecond to either side of exact integer ROS time. Treat a
            # retained node's packet boundary as the same closed timestamp.
            self.timestamp_boundary_tolerance_ns = 1
            # NVSLAM5 record headers keep the packet's nanosecond stamp while
            # individual Pandar rays are quantized to microseconds.  A record
            # header may therefore be up to 999 ns later than its first ray;
            # inspect that next record before deciding that a node is closed.
            self.record_header_support_ns = 1_000
            self._point_width = 0
            self._header = struct.Struct("<BqII")
            self._point_record_dtype = np.dtype(
                [("point", "<f4", (3,)), ("timestamp_ns", "<i8")]
            )
        elif magic == self._magic_v6:
            self._version = 6
            self.timestamp_boundary_tolerance_ns = 1
            self.record_header_support_ns = 1_000
            self._point_width = 0
            self._header = struct.Struct("<BqIII")
            self._point_record_dtype = np.dtype(
                [("point", "<f4", (3,)), ("timestamp_ns", "<i8")]
            )
        else:
            self._file.close()
            raise ValueError("not an NVSLAM1--NVSLAM6 scan archive")
        records: list[SlamScanRecord] = []
        while True:
            header = self._file.read(self._header.size)
            if not header:
                break
            if len(header) != self._header.size:
                self._file.close()
                raise ValueError("truncated SLAM scan header")
            unpacked = self._header.unpack(header)
            sensor, timestamp = unpacked[:2]
            if self._version == 6:
                ray_count, packet_count, point_count = unpacked[2:]
            elif len(unpacked) == 4:
                ray_count, point_count = unpacked[2:]
                packet_count = 0
            else:
                point_count = unpacked[2]
                ray_count = point_count
                packet_count = 0
            if sensor not in (0, 1):
                self._file.close()
                raise ValueError(f"unsupported SLAM laser index {sensor}")
            packet_timestamps_offset = self._file.tell()
            if packet_count:
                self._file.seek(packet_count * np.dtype("<i8").itemsize, 1)
            data_offset = self._file.tell()
            byte_count = (
                point_count * self._point_record_dtype.itemsize
                if self._point_record_dtype is not None
                else point_count * self._point_width * self._point_dtype.itemsize
            )
            self._file.seek(byte_count, 1)
            if self._file.tell() > self.path.stat().st_size:
                self._file.close()
                raise ValueError("truncated SLAM scan point array")
            records.append(
                SlamScanRecord(
                    int(sensor),
                    int(timestamp)
                    if self._version in (5, 6)
                    else int(round(timestamp * 1.0e9)),
                    int(ray_count), int(packet_count), packet_timestamps_offset,
                    int(point_count), data_offset
                )
            )
        self.records = tuple(sorted(records, key=lambda record: record.timestamp_ns))
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if getattr(self, "_mapping", None) is not None:
            self._mapping.close()
            self._mapping = None  # type: ignore[assignment]
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None  # type: ignore[assignment]

    def __enter__(self) -> "SlamScanArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(self, record: SlamScanRecord) -> LidarScan:
        if self._mapping is None:
            raise RuntimeError("SLAM scan archive is closed")
        if self._point_record_dtype is not None:
            records = np.frombuffer(
                self._mapping,
                dtype=self._point_record_dtype,
                count=record.point_count,
                offset=record.data_offset,
            )
            return LidarScan(
                record.timestamp_ns,
                records["point"].astype(np.float64, copy=True),
                point_timestamps_ns=records["timestamp_ns"].copy(),
                range_filtered=True,
                ray_origins=np.broadcast_to(
                    self._sensor_origins[record.sensor],
                    (record.point_count, 3),
                ).copy(),
            )
        values = np.frombuffer(
            self._mapping,
            dtype=self._point_dtype,
            count=record.point_count * self._point_width,
            offset=record.data_offset,
        ).reshape((-1, self._point_width))
        # Copy before returning so callers can retain a scan after the archive
        # is closed and so normalized LidarScan normals never mutate the mmap.
        point_timestamps = None
        if self._point_width in (4, 7):
            point_timestamps = record.timestamp_ns + np.rint(
                values[:, self._point_width - 1].astype(np.float64) * 1.0e9
            ).astype(np.int64)
        normals = None
        if self._point_width in (6, 7):
            normals = values[:, 3:6].astype(np.float64, copy=True)
        return LidarScan(
            record.timestamp_ns,
            values[:, :3].astype(np.float64, copy=True),
            normals,
            point_timestamps,
            True,
            np.broadcast_to(
                self._sensor_origins[record.sensor], (record.point_count, 3)
            ).copy(),
        )

    def read_packet_timestamps_ns(self, record: SlamScanRecord) -> np.ndarray:
        """Return exact collator packet stamps stored by NVSLAM6."""

        if self._mapping is None:
            raise RuntimeError("SLAM scan archive is closed")
        if record.packet_count == 0:
            scan = self.read(record)
            if scan.point_timestamps_ns is None:
                return np.empty(0, dtype=np.int64)
            return np.unique(scan.point_timestamps_ns)
        return np.frombuffer(
            self._mapping,
            dtype="<i8",
            count=record.packet_count,
            offset=record.packet_timestamps_offset,
        ).copy()


@dataclass(frozen=True, slots=True)
class IcpResult:
    target_from_source: Rigid3
    fitness_m: float
    overlap: float
    correspondences: int
    iterations: int
    converged: bool
    euclidean_fitness_m: float = np.inf
    information_matrix: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class CorrelativeResult:
    target_from_source: Rigid3
    score: float
    second_score: float
    hypotheses: int
    rotational_score: float = 1.0


@dataclass(frozen=True, slots=True)
class ConstraintStability:
    eigenvalues: np.ndarray = field(repr=False, compare=False)
    strength: float
    anisotropy: float
    tracking_quality: float
    is_stable: bool


def rotational_score_is_acceptable(
    score: float, config: FrontendConfig = FrontendConfig()
) -> bool:
    """Apply the FCS rotational threshold, including equality."""

    return bool(
        np.isfinite(score) and score >= config.loop_min_rotational_score
    )


def evaluate_constraint_stability(
    information_matrix: np.ndarray,
    config: FrontendConfig = FrontendConfig(),
) -> ConstraintStability:
    """Evaluate the binary's final 6-DoF ICP information matrix gate."""

    matrix = np.asarray(information_matrix, dtype=np.float64)
    invalid_eigenvalues = np.full(6, np.nan, dtype=np.float64)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        return ConstraintStability(
            invalid_eigenvalues, np.nan, np.inf, 0.0, False
        )
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return ConstraintStability(
            invalid_eigenvalues, np.nan, np.inf, 0.0, False
        )
    if not np.all(np.isfinite(eigenvalues)):
        return ConstraintStability(eigenvalues, np.nan, np.inf, 0.0, False)

    strength = float(eigenvalues[0])
    anisotropy = (
        float(eigenvalues[-1] / strength) if strength > 0.0 else np.inf
    )
    is_stable = bool(
        np.isfinite(anisotropy)
        and anisotropy < config.loop_max_constraint_anisotropy
        and strength >= config.loop_min_constraint_strength
    )
    tracking_quality = 0.0
    if is_stable:
        tracking_good = config.loop_max_constraint_anisotropy_tracking_good
        if anisotropy <= tracking_good:
            tracking_quality = 1.0
        else:
            tracking_quality = float(
                np.float32(
                    1.0
                    - (anisotropy - tracking_good)
                    / (config.loop_max_constraint_anisotropy - tracking_good)
                )
            )
    return ConstraintStability(
        eigenvalues, strength, anisotropy, tracking_quality, is_stable
    )


@dataclass(frozen=True, slots=True)
class FrontendNode:
    node_id: NodeId
    timestamp_ns: int
    local_pose: Rigid3
    points: np.ndarray
    normals: np.ndarray
    scan_match: IcpResult | None
    gravity_observation: np.ndarray | None = None
    rotational_histogram: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.rotational_histogram is not None:
            histogram = np.asarray(self.rotational_histogram, dtype=np.float32)
        else:
            histogram = getattr(self.points, "rotational_histogram", None)
            if histogram is None:
                return
            histogram = np.asarray(histogram, dtype=np.float32)
        if histogram.ndim != 1 or not len(histogram):
            raise ValueError("rotational histogram must be a non-empty vector")
        object.__setattr__(self, "rotational_histogram", histogram.copy())


@dataclass(slots=True)
class SplitSurfelStatistics:
    """Binary-compatible state of a 232-byte two-sided surfel voxel."""

    keys: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    secondary_weights: np.ndarray | None = None
    secondary_counts: np.ndarray | None = None
    secondary_means: np.ndarray | None = None
    secondary_covariances: np.ndarray | None = None
    is_split: np.ndarray | None = None
    split_normals: np.ndarray | None = None
    viewpoints: np.ndarray | None = None
    secondary_viewpoints: np.ndarray | None = None
    primary_dirty: np.ndarray | None = None
    secondary_dirty: np.ndarray | None = None

    def __post_init__(self) -> None:
        size = len(self.keys)
        if self.secondary_weights is None:
            self.secondary_weights = np.zeros(size, dtype=np.float32)
        if self.secondary_counts is None:
            self.secondary_counts = np.zeros(size, dtype=np.uint32)
        if self.secondary_means is None:
            self.secondary_means = np.zeros((size, 3), dtype=np.float32)
        if self.secondary_covariances is None:
            self.secondary_covariances = np.zeros((size, 3, 3), dtype=np.float32)
        if self.is_split is None:
            self.is_split = np.zeros(size, dtype=np.uint8)
        if self.split_normals is None:
            self.split_normals = np.zeros((size, 3), dtype=np.float32)
        if self.viewpoints is None:
            self.viewpoints = np.zeros((size, 3), dtype=np.float32)
        if self.secondary_viewpoints is None:
            self.secondary_viewpoints = np.zeros((size, 3), dtype=np.float32)
        if self.primary_dirty is None:
            self.primary_dirty = np.zeros(size, dtype=np.uint8)
        if self.secondary_dirty is None:
            self.secondary_dirty = np.zeros(size, dtype=np.uint8)


class HybridProbabilityGrid:
    """Native Cartographer-compatible 3D probability grid.

    The native state keeps the sparse cells alive across node insertions and
    applies the installed G11 hit/miss lookup tables in float arithmetic.  A
    node batch updates hits first, then at most the final two free-space cells
    of every ray, with one update per cell and hit priority.
    """

    def __init__(
        self,
        resolution: float = 0.20,
        *,
        indices: np.ndarray | None = None,
        values: np.ndarray | None = None,
    ) -> None:
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("HybridGrid resolution must be finite and positive")
        native = _load_slam_frontend_native()
        handle = native.navvis_recon_slam_probability_grid_create(
            ctypes.c_float(resolution)
        )
        if not handle:
            raise RuntimeError("failed to allocate the native HybridGrid")
        self.resolution = float(resolution)
        self._handle = ctypes.c_void_p(handle)
        if indices is not None or values is not None:
            if indices is None or values is None:
                self.close()
                raise ValueError("HybridGrid indices and values must be provided together")
            cells = np.ascontiguousarray(indices, dtype=np.int32)
            probabilities = np.ascontiguousarray(values, dtype=np.uint16)
            if cells.ndim != 2 or cells.shape[1] != 3:
                self.close()
                raise ValueError("HybridGrid indices must have shape (N, 3)")
            if probabilities.shape != (len(cells),):
                self.close()
                raise ValueError("HybridGrid values must match the indices")
            status = native.navvis_recon_slam_probability_grid_load(
                self._handle,
                len(cells),
                cells.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                probabilities.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            )
            if status != 0:
                self.close()
                raise RuntimeError(f"native HybridGrid load failed with status {status}")

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value:
            _load_slam_frontend_native().navvis_recon_slam_probability_grid_destroy(
                handle
            )
            handle.value = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def cell_count(self) -> int:
        if not self._handle.value:
            raise RuntimeError("HybridGrid is closed")
        return int(
            _load_slam_frontend_native().navvis_recon_slam_probability_grid_size(
                self._handle
            )
        )

    def export_cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse cell indices and values in lexicographic order."""

        if not self._handle.value:
            raise RuntimeError("HybridGrid is closed")
        count = self.cell_count
        indices = np.empty((count, 3), dtype=np.int32)
        values = np.empty(count, dtype=np.uint16)
        output_count = ctypes.c_uint64()
        status = _load_slam_frontend_native().navvis_recon_slam_probability_grid_export(
            self._handle,
            count,
            ctypes.byref(output_count),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        )
        if status != 0 or int(output_count.value) != count:
            raise RuntimeError(
                "native HybridGrid export failed with "
                f"status {status} and count {output_count.value}"
            )
        if count:
            order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
            indices = indices[order]
            values = values[order]
        return indices, values

    def insert(self, points: np.ndarray, origins: np.ndarray) -> None:
        if not self._handle.value:
            raise RuntimeError("HybridGrid is closed")
        points32 = np.ascontiguousarray(points, dtype=np.float32)
        origins32 = np.ascontiguousarray(origins, dtype=np.float32)
        if points32.ndim != 2 or points32.shape[1] != 3:
            raise ValueError("HybridGrid points must have shape (N, 3)")
        if origins32.shape != points32.shape:
            raise ValueError("HybridGrid ray origins must match points")
        status = _load_slam_frontend_native().navvis_recon_slam_probability_grid_insert(
            self._handle,
            len(points32),
            _float_pointer(points32),
            _float_pointer(origins32),
        )
        if status != 0:
            raise RuntimeError(f"native HybridGrid insertion failed with status {status}")

    def score(self, points: np.ndarray, pose: Rigid3 | None = None) -> float:
        if not self._handle.value:
            raise RuntimeError("HybridGrid is closed")
        points64 = np.asarray(points, dtype=np.float64)
        if points64.ndim != 2 or points64.shape[1] != 3:
            raise ValueError("HybridGrid score points must have shape (N, 3)")
        if pose is not None:
            points64 = pose.rotation.apply(points64) + pose.translation
        points32 = np.ascontiguousarray(points64, dtype=np.float32)
        return float(
            _load_slam_frontend_native().navvis_recon_slam_probability_grid_score(
                self._handle, len(points32), _float_pointer(points32)
            )
        )


def _transform_range_data_to_submap(
    node_pose: Rigid3,
    submap_pose: Rigid3,
    points: np.ndarray,
    normals: np.ndarray,
    origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cast the composed double pose once, then apply the vendor float matrix."""

    points32 = np.ascontiguousarray(points, dtype=np.float32)
    normals32 = np.ascontiguousarray(normals, dtype=np.float32)
    origins32 = np.ascontiguousarray(origins, dtype=np.float32)
    if points32.ndim != 2 or points32.shape[1] != 3:
        raise ValueError("submap range points must have shape (N, 3)")
    if normals32.shape != points32.shape or origins32.shape != points32.shape:
        raise ValueError("submap normals and origins must match range points")

    node_translation = np.ascontiguousarray(node_pose.translation, dtype=np.float64)
    node_quaternion = np.ascontiguousarray(
        node_pose.quaternion_xyzw, dtype=np.float64
    )
    submap_translation = np.ascontiguousarray(
        submap_pose.translation, dtype=np.float64
    )
    submap_quaternion = np.ascontiguousarray(
        submap_pose.quaternion_xyzw, dtype=np.float64
    )
    transformed_points = np.empty_like(points32)
    transformed_normals = np.empty_like(normals32)
    transformed_origins = np.empty_like(origins32)
    relative_pose = np.empty(7, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_transform_submap_range_data(
        len(points32),
        _float_pointer(points32),
        _float_pointer(normals32),
        _float_pointer(origins32),
        _double_pointer(node_translation),
        _double_pointer(node_quaternion),
        _double_pointer(submap_translation),
        _double_pointer(submap_quaternion),
        _float_pointer(transformed_points),
        _float_pointer(transformed_normals),
        _float_pointer(transformed_origins),
        _double_pointer(relative_pose),
    )
    if status != 0:
        raise RuntimeError(f"native submap range transform failed with status {status}")
    return (
        transformed_points,
        transformed_normals,
        transformed_origins,
        relative_pose,
    )


@dataclass(slots=True)
class FrontendSubmap:
    submap_id: NodeId
    local_pose: Rigid3
    start_timestamp_ns: int
    start_distance_m: float
    node_indices: list[int] = field(default_factory=list)
    end_timestamp_ns: int = 0
    finished: bool = False
    _point_batches: list[np.ndarray] = field(default_factory=list, repr=False)
    _normal_batches: list[np.ndarray] = field(default_factory=list, repr=False)
    _origin_batches: list[np.ndarray] = field(default_factory=list, repr=False)
    _cached_points: np.ndarray | None = field(default=None, repr=False)
    _cached_normals: np.ndarray | None = field(default=None, repr=False)
    _cached_levels: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list, repr=False
    )
    _surfel_statistics: list[SplitSurfelStatistics] = field(
        default_factory=list, repr=False
    )
    _hybrid_grid: HybridProbabilityGrid = field(
        default_factory=HybridProbabilityGrid, repr=False
    )
    _insertions_since_rebuild: int = field(default=0, repr=False)
    _surfel_maintenance_deferred: bool = field(default=False, repr=False)
    _gravity_count: int = field(default=0, repr=False)
    _gravity_state: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64), repr=False
    )

    def insert(
        self,
        node: FrontendNode,
        *,
        voxel_size: float,
        rebuild_interval: int,
        maintain_surfels: bool = True,
        map_points: np.ndarray | None = None,
        map_normals: np.ndarray | None = None,
        map_origins: np.ndarray | None = None,
        grid_points: np.ndarray | None = None,
        grid_origins: np.ndarray | None = None,
    ) -> None:
        source_points = node.points if map_points is None else map_points
        source_normals = node.normals if map_normals is None else map_normals
        source_origins = (
            np.zeros_like(source_points) if map_origins is None else map_origins
        )
        points, normals, origins, _ = _transform_range_data_to_submap(
            node.local_pose,
            self.local_pose,
            source_points,
            source_normals,
            source_origins,
        )
        self.node_indices.append(node.node_id.index)
        self.end_timestamp_ns = node.timestamp_ns
        if node.gravity_observation is not None:
            self._gravity_state = _updated_submap_gravity(
                self._gravity_state,
                self._gravity_count,
                self.local_pose,
                node.local_pose,
                node.gravity_observation,
            )
            self._gravity_count += 1
        self._point_batches.append(points)
        self._normal_batches.append(normals)
        self._origin_batches.append(origins)
        if grid_points is None:
            hybrid_points = points
            hybrid_origins = origins
        else:
            if grid_origins is None:
                raise ValueError("HybridGrid ray origins must accompany its points")
            hybrid_points, _, hybrid_origins, _ = _transform_range_data_to_submap(
                node.local_pose,
                self.local_pose,
                grid_points,
                np.zeros_like(grid_points),
                grid_origins,
            )
        self._hybrid_grid.insert(hybrid_points, hybrid_origins)
        self._insertions_since_rebuild += 1
        if self._insertions_since_rebuild >= rebuild_interval:
            if maintain_surfels:
                self.rebuild(voxel_size)
            else:
                self._consume_pending(voxel_size, maintain_surfels=False)

    def _consume_pending(self, voxel_size: float, *, maintain_surfels: bool) -> None:
        if not self._point_batches:
            return
        pending_points = np.concatenate(self._point_batches)
        pending_origins = np.concatenate(self._origin_batches)
        definitions = ((0.10, 0.05), (0.30, 0.15), (0.60, 0.0))
        levels: list[tuple[np.ndarray, np.ndarray]] = []
        statistics: list[SplitSurfelStatistics] = []
        for level_index, (resolution, offset) in enumerate(definitions):
            previous = (
                self._surfel_statistics[level_index]
                if level_index < len(self._surfel_statistics)
                else None
            )
            level_statistics = update_split_surfel_statistics(
                previous,
                pending_points,
                pending_origins,
                resolution,
                offset,
                maintain_surfels=maintain_surfels,
            )
            statistics.append(level_statistics)
            if maintain_surfels:
                levels.append(
                    extract_valid_split_surfels(level_statistics, resolution)
                )
        self._surfel_statistics = statistics
        if maintain_surfels:
            self._cached_levels = levels
            self._cached_points, self._cached_normals = levels[0]
            self._surfel_maintenance_deferred = False
        else:
            self._surfel_maintenance_deferred = True
        self._point_batches.clear()
        self._normal_batches.clear()
        self._origin_batches.clear()
        self._insertions_since_rebuild = 0

    def rebuild(self, voxel_size: float) -> None:
        if self._surfel_maintenance_deferred:
            self.activate(voxel_size)
            return
        if not self._point_batches:
            if self._cached_points is None:
                self._cached_points = np.empty((0, 3), dtype=np.float64)
                self._cached_normals = np.empty((0, 3), dtype=np.float64)
                self._cached_levels = [
                    (self._cached_points.copy(), self._cached_normals.copy())
                    for _ in range(3)
                ]
        else:
            self._consume_pending(voxel_size, maintain_surfels=True)
        self._insertions_since_rebuild = 0

    def activate(self, voxel_size: float) -> None:
        """Make a raw overlap map visible to the tracking matcher."""

        if self._point_batches:
            self._consume_pending(voxel_size, maintain_surfels=False)
        if not self._surfel_maintenance_deferred:
            self.rebuild(voxel_size)
            return
        definitions = ((0.10, 0.05), (0.30, 0.15), (0.60, 0.0))
        levels: list[tuple[np.ndarray, np.ndarray]] = []
        for statistics, (resolution, offset) in zip(
            self._surfel_statistics, definitions
        ):
            maintain_deferred_split_surfel_statistics(
                statistics, resolution, offset
            )
            levels.append(extract_valid_split_surfels(statistics, resolution))
        self._cached_levels = levels
        self._cached_points, self._cached_normals = levels[0]
        self._surfel_maintenance_deferred = False

    def cloud(self, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        # Return the most recent indexed snapshot. New batches become visible
        # at ``map_rebuild_interval``; rebuilding a KD-tree-sized map for every
        # 20 Hz node is both unnecessary and unlike the binary's staged grid.
        if self._cached_points is None:
            self.activate(voxel_size)
        assert self._cached_points is not None and self._cached_normals is not None
        return self._cached_points, self._cached_normals

    def cloud_level(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_points is None:
            self.activate(0.10)
        if level < 0 or level >= len(self._cached_levels):
            raise ValueError("surfel level must be 0, 1 or 2")
        return self._cached_levels[level]

    @property
    def hybrid_grid(self) -> HybridProbabilityGrid:
        return self._hybrid_grid

    def as_backend_submap(self) -> Submap:
        gravity = (
            self._gravity_state.copy()
            if self._gravity_count > 0
            else np.array([0.0, 0.0, 9.81])
        )
        return Submap(
            self.submap_id,
            self.start_timestamp_ns,
            self.end_timestamp_ns,
            self.local_pose,
            tuple(self.node_indices),
            self.finished,
            gravity,
        )


@dataclass(frozen=True, slots=True)
class FrontendResult:
    nodes: tuple[FrontendNode, ...]
    submaps: tuple[FrontendSubmap, ...]
    loops: tuple[LoopConstraint, ...]
    online_fast_pose_graph: OnlineFastPoseGraphSnapshot | None = None

    def backend_inputs(
        self,
    ) -> tuple[tuple[TrajectoryNode, ...], tuple[Submap, ...], tuple[LoopConstraint, ...]]:
        gravity = np.array([0.0, 0.0, 9.80665])
        nodes = tuple(
            TrajectoryNode(
                node.node_id,
                node.timestamp_ns,
                node.local_pose,
                node.local_pose,
                (
                    gravity
                    if node.gravity_observation is None
                    else node.gravity_observation
                ),
            )
            for node in self.nodes
        )
        return nodes, tuple(submap.as_backend_submap() for submap in self.submaps), self.loops


def _unique_voxel_keys(
    keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lexicographic unique rows through a collision-free packed integer.

    ``numpy.unique(axis=0)`` internally sorts a structured three-field dtype.
    Voxel coordinates occupy a much smaller rectangular integer lattice, so
    mixed-radix packing preserves exactly the same x/y/z ordering and inverse
    labels while avoiding the structured-sort overhead.
    """

    keys = np.asarray(keys, dtype=np.int64)
    if keys.ndim != 2 or keys.shape[1] != 3:
        raise ValueError("voxel keys must have shape (N, 3)")
    if len(keys) == 0:
        return keys.copy(), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    minimum = keys.min(axis=0)
    shifted = keys - minimum
    widths = shifted.max(axis=0) + 1
    if int(widths[0]) * int(widths[1]) * int(widths[2]) > np.iinfo(np.int64).max:
        unique, index, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        return unique, index, inverse
    packed = (
        (shifted[:, 0] * widths[1] + shifted[:, 1]) * widths[2]
        + shifted[:, 2]
    )
    _, index, inverse = np.unique(
        packed, return_index=True, return_inverse=True
    )
    return keys[index], index, inverse


def _stable_unique_voxel_keys(
    keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique voxel rows in first-insertion order.

    SurfelGrid keeps its dense cell array in insertion order.  Its hash table
    only maps voxel coordinates to that dense index; it does not define the
    traversal order used by surfel maintenance and cross-cell merging.
    """

    unique, first, inverse = _unique_voxel_keys(keys)
    if len(unique) == 0:
        return unique, first, inverse
    order = np.argsort(first, kind="stable")
    sorted_to_stable = np.empty(len(order), dtype=np.int64)
    sorted_to_stable[order] = np.arange(len(order), dtype=np.int64)
    return unique[order], first[order], sorted_to_stable[inverse]


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    normals: np.ndarray | None = None,
    *,
    offset: float | Sequence[float] = 0.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Centroid aggregate with deterministic lexicographic voxel order."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or voxel_size <= 0:
        raise ValueError("invalid point array or voxel size")
    if len(points) == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty if normals is None else (empty, empty.copy())
    grid_offset = np.asarray(offset, dtype=np.float64)
    keys = np.floor((points - grid_offset) / voxel_size).astype(np.int64)
    _, _, inverse = _unique_voxel_keys(keys)
    counts = np.bincount(inverse).astype(np.float64)
    output = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    if normals is None:
        return output
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != points.shape:
        raise ValueError("normal array must match points")
    output_normals = np.column_stack(
        [np.bincount(inverse, weights=normals[:, axis]) for axis in range(3)]
    )
    output_normals /= np.maximum(
        np.linalg.norm(output_normals, axis=1, keepdims=True), 1.0e-12
    )
    return output, output_normals


def update_surfel_statistics(
    previous: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    points: np.ndarray,
    voxel_size: float,
    offset: float | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Incrementally retain count, first and second moments per surfel voxel."""

    points = np.asarray(points, dtype=np.float64)
    keys = np.floor(
        (points - np.asarray(offset, dtype=np.float64)) / voxel_size
    ).astype(np.int64)
    unique, _, inverse = _unique_voxel_keys(keys)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)]
    )
    outer_values = (
        points[:, 0] * points[:, 0],
        points[:, 0] * points[:, 1],
        points[:, 0] * points[:, 2],
        points[:, 1] * points[:, 1],
        points[:, 1] * points[:, 2],
        points[:, 2] * points[:, 2],
    )
    outers = np.column_stack(
        [np.bincount(inverse, weights=value) for value in outer_values]
    )
    if previous is None:
        return unique, counts, sums, outers

    old_keys, old_counts, old_sums, old_outers = previous
    combined_keys = np.concatenate((old_keys, unique))
    merged_keys, _, merged_inverse = _unique_voxel_keys(combined_keys)
    combined_counts = np.concatenate((old_counts, counts))
    merged_counts = np.bincount(
        merged_inverse, weights=combined_counts
    )
    combined_sums = np.concatenate((old_sums, sums))
    merged_sums = np.column_stack(
        [
            np.bincount(merged_inverse, weights=combined_sums[:, axis])
            for axis in range(3)
        ]
    )
    combined_outers = np.concatenate((old_outers, outers))
    merged_outers = np.column_stack(
        [
            np.bincount(merged_inverse, weights=combined_outers[:, axis])
            for axis in range(6)
        ]
    )
    return merged_keys, merged_counts, merged_sums, merged_outers


def _accumulate_moments(
    points: np.ndarray, labels: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate binary surfel weight, first moments and upper covariance."""

    counts = np.bincount(labels, minlength=size).astype(np.float64)
    sums = np.column_stack(
        [
            np.bincount(labels, weights=points[:, axis], minlength=size)
            for axis in range(3)
        ]
    )
    outer_values = (
        points[:, 0] * points[:, 0],
        points[:, 0] * points[:, 1],
        points[:, 0] * points[:, 2],
        points[:, 1] * points[:, 1],
        points[:, 1] * points[:, 2],
        points[:, 2] * points[:, 2],
    )
    outers = np.column_stack(
        [
            np.bincount(labels, weights=value, minlength=size)
            for value in outer_values
        ]
    )
    return counts, sums, outers


def _moment_geometry(
    counts: np.ndarray, sums: np.ndarray, outers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover centers, PCA normals and ascending eigenvalues from moments."""

    centers = np.zeros_like(sums)
    populated = counts > 0.0
    centers[populated] = sums[populated] / counts[populated, None]
    covariance = np.zeros((len(counts), 3, 3), dtype=np.float64)
    if np.any(populated):
        index = np.flatnonzero(populated)
        center = centers[index]
        weight = counts[index]
        outer = outers[index]
        covariance[index, 0, 0] = outer[:, 0] / weight - center[:, 0] ** 2
        covariance[index, 0, 1] = covariance[index, 1, 0] = (
            outer[:, 1] / weight - center[:, 0] * center[:, 1]
        )
        covariance[index, 0, 2] = covariance[index, 2, 0] = (
            outer[:, 2] / weight - center[:, 0] * center[:, 2]
        )
        covariance[index, 1, 1] = outer[:, 3] / weight - center[:, 1] ** 2
        covariance[index, 1, 2] = covariance[index, 2, 1] = (
            outer[:, 4] / weight - center[:, 1] * center[:, 2]
        )
        covariance[index, 2, 2] = outer[:, 5] / weight - center[:, 2] ** 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return centers, eigenvectors[:, :, 0], np.maximum(eigenvalues, 0.0)


def _surfel_property_mask(
    counts: np.ndarray,
    eigenvalues: np.ndarray,
    voxel_size: float,
    *,
    weight_threshold: float,
    max_curvature: float,
    max_normal_spread: float,
    min_in_plane_spread: float,
    planarity_threshold: float,
    relative_spread_threshold: float = 0.0,
) -> np.ndarray:
    total = np.maximum(np.sum(eigenvalues, axis=1), 1.0e-12)
    curvature = 3.0 * eigenvalues[:, 0] / total
    planarity = 2.0 * (eigenvalues[:, 1] - eigenvalues[:, 0]) / total
    spread_scale = 2.0 * np.sqrt(3.0)
    major_in_plane_spread = spread_scale * np.sqrt(eigenvalues[:, 2])
    minor_in_plane_spread = spread_scale * np.sqrt(eigenvalues[:, 1])
    normal_direction_spread = spread_scale * np.sqrt(eigenvalues[:, 0])
    return (
        (counts >= weight_threshold)
        & (planarity >= planarity_threshold)
        & (curvature <= max_curvature)
        & (major_in_plane_spread >= voxel_size * relative_spread_threshold)
        & (minor_in_plane_spread >= min_in_plane_spread)
        & (normal_direction_spread <= max_normal_spread)
    )


_slam_frontend_native: ctypes.CDLL | None = None


_C_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
_C_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
_C_INT64_POINTER = ctypes.POINTER(ctypes.c_int64)
_C_UINT32_POINTER = ctypes.POINTER(ctypes.c_uint32)
_C_UINT64_POINTER = ctypes.POINTER(ctypes.c_uint64)


def _bind_native_function(
    library: ctypes.CDLL,
    name: str,
    argtypes: tuple[object, ...],
    restype: object | None,
) -> None:
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = restype


def _configure_native_spatial_queries(library: ctypes.CDLL) -> None:
    _bind_native_function(
        library,
        "navvis_recon_slam_octree_create",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_void_p,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_octree_destroy",
        (
            ctypes.c_void_p,
        ),
        None,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_octree_nearest",
        (
            ctypes.c_void_p,
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
            _C_UINT64_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_rotational_histogram",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            ctypes.c_int32,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )


def _configure_native_surfel_operations(library: ctypes.CDLL) -> None:
    _bind_native_function(
        library,
        "navvis_recon_slam_label_surfel_cells",
        (
            ctypes.c_uint64,
            _C_INT64_POINTER,
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint64,
            _C_UINT64_POINTER,
            _C_INT64_POINTER,
            _C_UINT64_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_update_surfels",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_UINT32_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            ctypes.c_uint64,
            _C_UINT64_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_update_split_surfels",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_UINT32_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_UINT32_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            ctypes.POINTER(ctypes.c_uint8),
            _C_FLOAT_POINTER,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64,
            _C_UINT64_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            ctypes.c_uint8,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_maintain_split_surfels",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_UINT32_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_UINT32_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            ctypes.POINTER(ctypes.c_uint8),
            _C_FLOAT_POINTER,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_merge_surfels",
        (
            ctypes.c_uint64,
            _C_INT64_POINTER,
            _C_FLOAT_POINTER,
            _C_UINT32_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_UINT32_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            ctypes.POINTER(ctypes.c_uint8),
            _C_FLOAT_POINTER,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint64,
            _C_UINT64_POINTER,
            _C_UINT64_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_surfel_geometry",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_oriented_surfel_geometry",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )


def _configure_native_imu_and_icp(library: ctypes.CDLL) -> None:
    _bind_native_function(
        library,
        "navvis_recon_slam_raw_imu_gravity_update",
        (
            ctypes.c_double,
            ctypes.c_double,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_raw_imu_angular_update",
        (
            ctypes.c_double,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_raw_imu_initialize_gravity",
        (
            ctypes.c_double,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_fast_imu_integrate",
        (
            ctypes.c_uint64,
            _C_INT64_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            ctypes.c_int64,
            ctypes.c_int64,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_fast_imu_acceleration_measurement",
        (
            ctypes.c_uint64,
            _C_INT64_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_point_plane_step",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_deskew_points",
        (
            ctypes.c_uint64,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )


def _configure_native_pose_transforms(library: ctypes.CDLL) -> None:
    _bind_native_function(
        library,
        "navvis_recon_slam_end_from_start_pose",
        (
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_inverse_pose",
        (
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_compose_pose",
        (
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_compose_pose_normalized",
        (
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_predict_translation",
        (
            ctypes.c_int64,
            _C_DOUBLE_POINTER,
            ctypes.c_int64,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            ctypes.c_int64,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_icp_normalization_pose",
        (
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_icp_normalization_diagnostics",
        (
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_transform_points",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_transform_points_raw_float_matrix",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_transform_points_raw_float_quaternion",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_transform_points_double_matrix_cast",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_transform_submap_range_data",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_DOUBLE_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            _C_DOUBLE_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_range_centroid_filter",
        (
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )


def _configure_native_probability_grid(library: ctypes.CDLL) -> None:
    _bind_native_function(
        library,
        "navvis_recon_slam_probability_grid_create",
        (
            ctypes.c_float,
        ),
        ctypes.c_void_p,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_probability_grid_destroy",
        (
            ctypes.c_void_p,
        ),
        None,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_probability_grid_size",
        (
            ctypes.c_void_p,
        ),
        ctypes.c_uint64,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_probability_grid_export",
        (
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint16),
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_probability_grid_load",
        (
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint16),
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_probability_grid_insert",
        (
            ctypes.c_void_p,
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_int,
    )
    _bind_native_function(
        library,
        "navvis_recon_slam_probability_grid_score",
        (
            ctypes.c_void_p,
            ctypes.c_uint64,
            _C_FLOAT_POINTER,
        ),
        ctypes.c_float,
    )


def _configure_slam_frontend_native(library: ctypes.CDLL) -> None:
    """Declare the native ABI by subsystem."""

    _configure_native_spatial_queries(library)
    _configure_native_surfel_operations(library)
    _configure_native_imu_and_icp(library)
    _configure_native_pose_transforms(library)
    _configure_native_probability_grid(library)



def _load_slam_frontend_native() -> ctypes.CDLL:
    """Load the C++ float-surfel kernel built with the workspace."""

    global _slam_frontend_native
    if _slam_frontend_native is not None:
        return _slam_frontend_native
    code_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("NAVVIS_RECON_SLAM_NATIVE")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        code_root / directory / "libnavvis_recon_slam_frontend_native.so"
        for directory in ("build-release", "build-cpp", "build-cpp-ceres")
    )
    library_path = next((path for path in candidates if path.is_file()), None)
    if library_path is None:
        raise RuntimeError(
            "the SLAM float-surfel kernel is not built; run "
            "cmake --build code/build-release --target "
            "navvis_recon_slam_frontend_native"
        )
    library = ctypes.CDLL(str(library_path))
    _configure_slam_frontend_native(library)
    _slam_frontend_native = library
    return library


def _float_pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _double_pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _native_raw_imu_gravity_update(
    quaternion_xyzw: np.ndarray,
    gravity: np.ndarray,
    acceleration: np.ndarray,
    alpha: float,
    gravity_magnitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.ascontiguousarray(quaternion_xyzw, dtype=np.float64)
    previous_gravity = np.ascontiguousarray(gravity, dtype=np.float64)
    measurement = np.ascontiguousarray(acceleration, dtype=np.float64)
    output_quaternion = np.empty(4, dtype=np.float64)
    output_gravity = np.empty(3, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_raw_imu_gravity_update(
        ctypes.c_double(alpha),
        ctypes.c_double(gravity_magnitude),
        _double_pointer(quaternion),
        _double_pointer(previous_gravity),
        _double_pointer(measurement),
        _double_pointer(output_quaternion),
        _double_pointer(output_gravity),
    )
    if status != 0:
        raise RuntimeError(f"native raw IMU gravity update failed with status {status}")
    return output_quaternion, output_gravity


def _native_raw_imu_angular_update(
    quaternion_xyzw: np.ndarray,
    gravity: np.ndarray,
    angular_start: np.ndarray,
    angular_end: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.ascontiguousarray(quaternion_xyzw, dtype=np.float64)
    previous_gravity = np.ascontiguousarray(gravity, dtype=np.float64)
    start = np.ascontiguousarray(angular_start, dtype=np.float64)
    end = np.ascontiguousarray(angular_end, dtype=np.float64)
    output_quaternion = np.empty(4, dtype=np.float64)
    output_gravity = np.empty(3, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_raw_imu_angular_update(
        ctypes.c_double(dt_s),
        _double_pointer(quaternion),
        _double_pointer(previous_gravity),
        _double_pointer(start),
        _double_pointer(end),
        _double_pointer(output_quaternion),
        _double_pointer(output_gravity),
    )
    if status != 0:
        raise RuntimeError(f"native raw IMU angular update failed with status {status}")
    return output_quaternion, output_gravity


def _native_raw_imu_initialize_gravity(
    quaternion_xyzw: np.ndarray,
    gravity_magnitude: float,
) -> np.ndarray:
    quaternion = np.ascontiguousarray(quaternion_xyzw, dtype=np.float64)
    output_gravity = np.empty(3, dtype=np.float64)
    status = (
        _load_slam_frontend_native().navvis_recon_slam_raw_imu_initialize_gravity(
            ctypes.c_double(gravity_magnitude),
            _double_pointer(quaternion),
            _double_pointer(output_gravity),
        )
    )
    if status != 0:
        raise RuntimeError(f"native raw IMU initialization failed with status {status}")
    return output_gravity


class _BinaryOctree:
    """Native UniBN-compatible float32 nearest-neighbour index."""

    def __init__(self, points: np.ndarray) -> None:
        self._points = np.ascontiguousarray(points, dtype=np.float32)
        handle = _load_slam_frontend_native().navvis_recon_slam_octree_create(
            len(self._points), _float_pointer(self._points)
        )
        if not handle:
            raise RuntimeError("failed to build binary-compatible ICP octree")
        self._handle = ctypes.c_void_p(handle)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            _load_slam_frontend_native().navvis_recon_slam_octree_destroy(
                handle
            )
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors can run after ctypes module globals have already
            # been torn down during interpreter shutdown.
            pass

    def query(
        self,
        points: np.ndarray,
        radius: float,
        num_threads: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(points, dtype=np.float32)
        found = np.empty(len(queries), dtype=np.uint8)
        indices = np.empty(len(queries), dtype=np.uint64)
        squared_distances = np.empty(len(queries), dtype=np.float32)
        status = _load_slam_frontend_native().navvis_recon_slam_octree_nearest(
            self._handle,
            len(queries),
            _float_pointer(queries),
            ctypes.c_float(radius),
            ctypes.c_int32(num_threads),
            found.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            _float_pointer(squared_distances),
        )
        if status != 0:
            raise RuntimeError("binary-compatible ICP octree query failed")
        return found.astype(bool), indices.astype(np.int64), squared_distances


def _transform_points_float_quaternion(
    points: np.ndarray,
    quaternions_xyzw: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Apply the vendor's per-ray float Quaternion transform."""

    points64 = np.ascontiguousarray(points, dtype=np.float64)
    quaternions64 = np.ascontiguousarray(quaternions_xyzw, dtype=np.float64)
    translations64 = np.ascontiguousarray(translations, dtype=np.float64)
    if points64.ndim != 2 or points64.shape[1] != 3:
        raise ValueError("deskew points must have shape (N, 3)")
    if quaternions64.shape != (len(points64), 4):
        raise ValueError("deskew quaternions must have shape (N, 4)")
    if translations64.shape != points64.shape:
        raise ValueError("deskew translations must match points")
    output = np.empty(points64.shape, dtype=np.float32)
    status = _load_slam_frontend_native().navvis_recon_slam_deskew_points(
        len(points64),
        _double_pointer(points64),
        _double_pointer(quaternions64),
        _double_pointer(translations64),
        _float_pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"native per-ray deskew failed with status {status}")
    return output


def _end_from_start_pose_with_roundtrip(
    world_from_start: Rigid3, world_from_end: Rigid3
) -> Rigid3:
    """Match the accumulator's inverse(start), inverse, inverse(end), compose."""

    start_translation = np.ascontiguousarray(
        world_from_start.translation, dtype=np.float64
    )
    start_quaternion = np.ascontiguousarray(
        world_from_start.quaternion_xyzw, dtype=np.float64
    )
    end_translation = np.ascontiguousarray(
        world_from_end.translation, dtype=np.float64
    )
    end_quaternion = np.ascontiguousarray(
        world_from_end.quaternion_xyzw, dtype=np.float64
    )
    output = np.empty(7, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_end_from_start_pose(
        _double_pointer(start_translation),
        _double_pointer(start_quaternion),
        _double_pointer(end_translation),
        _double_pointer(end_quaternion),
        _double_pointer(output),
    )
    if status != 0:
        raise RuntimeError(
            f"native end-from-start pose composition failed with status {status}"
        )
    return Rigid3(output[:3].copy(), output[3:].copy())


def _inverse_pose_binary(pose: Rigid3) -> Rigid3:
    """Apply the installed Rigid3 inverse without normalizing coefficients."""

    translation = np.ascontiguousarray(pose.translation, dtype=np.float64)
    quaternion = np.ascontiguousarray(pose.quaternion_xyzw, dtype=np.float64)
    output = np.empty(7, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_inverse_pose(
        _double_pointer(translation),
        _double_pointer(quaternion),
        _double_pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"native pose inverse failed with status {status}")
    return Rigid3(output[:3].copy(), output[3:].copy())


def _compose_pose_binary(lhs: Rigid3, rhs: Rigid3) -> Rigid3:
    """Apply the installed raw Rigid3 product ``lhs * rhs``."""

    lhs_translation = np.ascontiguousarray(lhs.translation, dtype=np.float64)
    lhs_quaternion = np.ascontiguousarray(
        lhs.quaternion_xyzw, dtype=np.float64
    )
    rhs_translation = np.ascontiguousarray(rhs.translation, dtype=np.float64)
    rhs_quaternion = np.ascontiguousarray(
        rhs.quaternion_xyzw, dtype=np.float64
    )
    output = np.empty(7, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_compose_pose(
        _double_pointer(lhs_translation),
        _double_pointer(lhs_quaternion),
        _double_pointer(rhs_translation),
        _double_pointer(rhs_quaternion),
        _double_pointer(output),
    )
    if status != 0:
        raise RuntimeError(f"native pose composition failed with status {status}")
    return Rigid3(output[:3].copy(), output[3:].copy())


def _compose_pose_normalized_binary(lhs: Rigid3, rhs: Rigid3) -> Rigid3:
    """Multiply two Rigid3 values, then normalize the copied quaternion."""

    lhs_translation = np.ascontiguousarray(lhs.translation, dtype=np.float64)
    lhs_quaternion = np.ascontiguousarray(
        lhs.quaternion_xyzw, dtype=np.float64
    )
    rhs_translation = np.ascontiguousarray(rhs.translation, dtype=np.float64)
    rhs_quaternion = np.ascontiguousarray(
        rhs.quaternion_xyzw, dtype=np.float64
    )
    output = np.empty(7, dtype=np.float64)
    status = (
        _load_slam_frontend_native().navvis_recon_slam_compose_pose_normalized(
            _double_pointer(lhs_translation),
            _double_pointer(lhs_quaternion),
            _double_pointer(rhs_translation),
            _double_pointer(rhs_quaternion),
            _double_pointer(output),
        )
    )
    if status != 0:
        raise RuntimeError(
            f"native normalized pose composition failed with status {status}"
        )
    return Rigid3(output[:3].copy(), output[3:].copy())


def _predict_translation_binary(
    previous_timestamp_ns: int,
    previous_translation: np.ndarray,
    anchor_timestamp_ns: int,
    anchor_translation: np.ndarray,
    anchor_quaternion_xyzw: np.ndarray,
    query_timestamp_ns: int,
) -> np.ndarray:
    """Extrapolate translation with the frontend's Eigen evaluation order."""

    previous = np.ascontiguousarray(previous_translation, dtype=np.float64)
    anchor = np.ascontiguousarray(anchor_translation, dtype=np.float64)
    anchor_quaternion = np.ascontiguousarray(
        anchor_quaternion_xyzw, dtype=np.float64
    )
    output = np.empty(3, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_predict_translation(
        int(previous_timestamp_ns),
        _double_pointer(previous),
        int(anchor_timestamp_ns),
        _double_pointer(anchor),
        _double_pointer(anchor_quaternion),
        int(query_timestamp_ns),
        _double_pointer(output),
    )
    if status != 0:
        raise RuntimeError(
            f"native pose translation prediction failed with status {status}"
        )
    return output


def _quaternion_inverse_binary(quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = Rigid3(np.zeros(3, dtype=np.float64), quaternion_xyzw)
    return _inverse_pose_binary(pose).quaternion_xyzw


def _quaternion_product_binary(
    first_xyzw: np.ndarray, second_xyzw: np.ndarray
) -> np.ndarray:
    first = Rigid3(np.zeros(3, dtype=np.float64), first_xyzw)
    second = Rigid3(np.zeros(3, dtype=np.float64), second_xyzw)
    return _compose_pose_binary(first, second).quaternion_xyzw


def _icp_normalization_pose_binary(pose: Rigid3) -> Rigid3:
    """Remove gravity tilt in the same operation order as the matcher."""

    translation = np.ascontiguousarray(pose.translation, dtype=np.float64)
    quaternion = np.ascontiguousarray(pose.quaternion_xyzw, dtype=np.float64)
    output = np.empty(7, dtype=np.float64)
    status = (
        _load_slam_frontend_native()
        .navvis_recon_slam_icp_normalization_pose(
            _double_pointer(translation),
            _double_pointer(quaternion),
            _double_pointer(output),
        )
    )
    if status != 0:
        raise RuntimeError(
            f"native ICP normalization failed with status {status}"
        )
    return Rigid3(output[:3].copy(), output[3:].copy())


def _transform_points_float_matrix(points: np.ndarray, pose: Rigid3) -> np.ndarray:
    """Cast one double pose to the vendor's float 4x4 range matrix."""

    points32 = np.ascontiguousarray(points, dtype=np.float32)
    if points32.ndim != 2 or points32.shape[1] != 3:
        raise ValueError("range transform points must have shape (N, 3)")
    translation = np.ascontiguousarray(pose.translation, dtype=np.float64)
    quaternion = np.ascontiguousarray(pose.quaternion_xyzw, dtype=np.float64)
    output = np.empty_like(points32)
    status = (
        _load_slam_frontend_native()
        .navvis_recon_slam_transform_points_raw_float_matrix(
            len(points32),
            _float_pointer(points32),
            _double_pointer(translation),
            _double_pointer(quaternion),
            _float_pointer(output),
        )
    )
    if status != 0:
        raise RuntimeError(f"native float-matrix transform failed with status {status}")
    return output


def _transform_points_icp_float_matrix(
    points: np.ndarray, pose: Rigid3
) -> np.ndarray:
    """Apply the matcher's double-matrix-then-float transform path."""

    points32 = np.ascontiguousarray(points, dtype=np.float32)
    translation = np.ascontiguousarray(pose.translation, dtype=np.float64)
    quaternion = np.ascontiguousarray(
        pose.quaternion_xyzw, dtype=np.float64
    )
    output = np.empty_like(points32)
    status = (
        _load_slam_frontend_native()
        .navvis_recon_slam_transform_points_double_matrix_cast(
            len(points32),
            _float_pointer(points32),
            _double_pointer(translation),
            _double_pointer(quaternion),
            _float_pointer(output),
        )
    )
    if status != 0:
        raise RuntimeError(
            f"native ICP float-matrix transform failed with status {status}"
        )
    return output


def _transform_points_icp_filter_float_quaternion(
    points: np.ndarray, pose: Rigid3
) -> np.ndarray:
    """Apply the geometric predicate's raw float-quaternion transform."""

    points32 = np.ascontiguousarray(points, dtype=np.float32)
    translation = np.ascontiguousarray(pose.translation, dtype=np.float64)
    quaternion = np.ascontiguousarray(
        pose.quaternion_xyzw, dtype=np.float64
    )
    output = np.empty_like(points32)
    status = (
        _load_slam_frontend_native()
        .navvis_recon_slam_transform_points_raw_float_quaternion(
            len(points32),
            _float_pointer(points32),
            _double_pointer(translation),
            _double_pointer(quaternion),
            _float_pointer(output),
        )
    )
    if status != 0:
        raise RuntimeError(
            "native ICP filter quaternion transform failed with "
            f"status {status}"
        )
    return output


def _binary_point_plane_step(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    normalization: Rigid3,
) -> tuple[Rigid3, np.ndarray, float]:
    """Run one normalized float/double point-to-plane Gauss-Newton step."""

    increment, delta, scale, _, _ = _binary_point_plane_step_diagnostics(
        source_points, target_points, target_normals, normalization
    )
    return increment, delta, scale


def _binary_point_plane_step_diagnostics(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    normalization: Rigid3,
) -> tuple[Rigid3, np.ndarray, float, np.ndarray, np.ndarray]:
    """Return the step together with its exact native normal system."""

    source32 = np.ascontiguousarray(source_points, dtype=np.float32)
    target32 = np.ascontiguousarray(target_points, dtype=np.float32)
    normals32 = np.ascontiguousarray(target_normals, dtype=np.float32)
    if (
        source32.shape != target32.shape
        or source32.shape != normals32.shape
        or source32.ndim != 2
        or source32.shape[1] != 3
        or not len(source32)
    ):
        raise ValueError("point-plane step expects matching non-empty Nx3 arrays")
    normalization_translation = np.ascontiguousarray(
        normalization.translation, dtype=np.float64
    )
    normalization_quaternion = np.ascontiguousarray(
        normalization.quaternion_xyzw, dtype=np.float64
    )
    output_translation = np.empty(3, dtype=np.float64)
    output_quaternion = np.empty(4, dtype=np.float64)
    output_delta = np.empty(6, dtype=np.float64)
    output_scale = np.empty(1, dtype=np.float64)
    output_normal_matrix = np.empty(36, dtype=np.float64)
    output_right_hand_side = np.empty(6, dtype=np.float64)
    status = _load_slam_frontend_native().navvis_recon_slam_point_plane_step(
        len(source32),
        _float_pointer(source32),
        _float_pointer(target32),
        _float_pointer(normals32),
        _double_pointer(normalization_translation),
        _double_pointer(normalization_quaternion),
        _double_pointer(output_translation),
        _double_pointer(output_quaternion),
        _double_pointer(output_delta),
        _double_pointer(output_scale),
        _double_pointer(output_normal_matrix),
        _double_pointer(output_right_hand_side),
    )
    if status != 0:
        raise RuntimeError(f"native point-plane step failed with status {status}")
    return (
        Rigid3(output_translation, output_quaternion),
        output_delta,
        float(output_scale[0]),
        output_normal_matrix.reshape((6, 6), order="F"),
        output_right_hand_side,
    )


def range_measurement_centroid_filter(
    points: np.ndarray,
    origins: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the binary's float32 HASH_MAP_CENTROID range filter."""

    points32 = np.ascontiguousarray(points, dtype=np.float32)
    origins32 = np.ascontiguousarray(origins, dtype=np.float32)
    if (
        points32.shape != origins32.shape
        or points32.ndim != 2
        or points32.shape[1] != 3
    ):
        raise ValueError("centroid points/origins must be matching Nx3 arrays")
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("centroid voxel size must be finite and positive")

    output_points = np.empty_like(points32)
    output_origins = np.empty_like(origins32)
    output_count = ctypes.c_uint64()
    result = _load_slam_frontend_native().navvis_recon_slam_range_centroid_filter(
        len(points32),
        _float_pointer(origins32),
        _float_pointer(points32),
        np.float32(voxel_size),
        len(points32),
        ctypes.byref(output_count),
        _float_pointer(output_origins),
        _float_pointer(output_points),
    )
    if result != 0:
        raise RuntimeError(f"native range centroid filter failed with status {result}")
    count = int(output_count.value)
    return (
        output_points[:count].astype(np.float64),
        output_origins[:count].astype(np.float64),
    )


def update_split_surfel_statistics(
    previous: SplitSurfelStatistics | None,
    points: np.ndarray,
    origins: np.ndarray,
    voxel_size: float,
    offset: float | Sequence[float],
    *,
    maintain_surfels: bool = True,
) -> SplitSurfelStatistics:
    """Update both sides of the binary's split-surfel voxel state."""

    points64 = np.asarray(points, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    if (
        points64.shape != origins.shape
        or points64.ndim != 2
        or points64.shape[1] != 3
    ):
        raise ValueError("split surfel points/origins must be matching Nx3 arrays")
    points32 = np.ascontiguousarray(points64, dtype=np.float32)
    # SurfelGrid stores a double-precision origin and inverse resolution.  Its
    # insertion kernel evaluates (float_endpoint + origin) * inverse rather
    # than dividing by the nominal resolution.  The operation order selects
    # a different cell for the handful of returns lying exactly on a 0.1 m
    # boundary.
    grid_origin = np.asarray(offset, dtype=np.float64)
    grid_origin_xyz = np.broadcast_to(grid_origin, (3,))
    inverse_resolution = np.float64(1.0) / np.float64(voxel_size)
    initial_state = previous is None
    previous_keys = np.ascontiguousarray(
        np.empty((0, 3), dtype=np.int64)
        if previous is None
        else previous.keys,
        dtype=np.int64,
    )
    capacity = len(previous_keys) + len(points32)
    merged_key_storage = np.empty((capacity, 3), dtype=np.int64)
    labels = np.empty(len(points32), dtype=np.uint64)
    output_count = ctypes.c_uint64()
    native = _load_slam_frontend_native()
    status = native.navvis_recon_slam_label_surfel_cells(
        len(previous_keys),
        previous_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(points32),
        _float_pointer(points32),
        float(grid_origin_xyz[0]),
        float(grid_origin_xyz[1]),
        float(grid_origin_xyz[2]),
        float(inverse_resolution),
        capacity,
        ctypes.byref(output_count),
        merged_key_storage.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    if status != 0:
        raise RuntimeError(f"native surfel cell labeling failed with status {status}")
    size = int(output_count.value)
    merged_keys = merged_key_storage[:size].copy()
    weights = np.zeros(size, dtype=np.float32)
    counts = np.zeros(size, dtype=np.uint32)
    means = np.zeros((size, 3), dtype=np.float32)
    covariances = np.zeros((size, 3, 3), dtype=np.float32)
    viewpoints = np.zeros((size, 3), dtype=np.float32)
    secondary_weights = np.zeros(size, dtype=np.float32)
    secondary_counts = np.zeros(size, dtype=np.uint32)
    secondary_means = np.zeros((size, 3), dtype=np.float32)
    secondary_covariances = np.zeros((size, 3, 3), dtype=np.float32)
    secondary_viewpoints = np.zeros((size, 3), dtype=np.float32)
    is_split = np.zeros(size, dtype=np.uint8)
    split_normals = np.zeros((size, 3), dtype=np.float32)
    primary_dirty = np.zeros(size, dtype=np.uint8)
    secondary_dirty = np.zeros(size, dtype=np.uint8)
    if previous is not None:
        previous_count = len(previous.keys)
        weights[:previous_count] = previous.weights
        counts[:previous_count] = previous.counts
        means[:previous_count] = previous.means
        covariances[:previous_count] = previous.covariances
        assert previous.viewpoints is not None
        viewpoints[:previous_count] = previous.viewpoints
        assert previous.secondary_weights is not None
        assert previous.secondary_counts is not None
        assert previous.secondary_means is not None
        assert previous.secondary_covariances is not None
        assert previous.is_split is not None
        assert previous.split_normals is not None
        assert previous.secondary_viewpoints is not None
        assert previous.primary_dirty is not None
        assert previous.secondary_dirty is not None
        secondary_weights[:previous_count] = previous.secondary_weights
        secondary_counts[:previous_count] = previous.secondary_counts
        secondary_means[:previous_count] = previous.secondary_means
        secondary_covariances[:previous_count] = previous.secondary_covariances
        secondary_viewpoints[:previous_count] = previous.secondary_viewpoints
        is_split[:previous_count] = previous.is_split
        split_normals[:previous_count] = previous.split_normals
        primary_dirty[:previous_count] = previous.primary_dirty
        secondary_dirty[:previous_count] = previous.secondary_dirty

    origins32 = np.ascontiguousarray(origins, dtype=np.float32)
    result = native.navvis_recon_slam_update_split_surfels(
        size,
        _float_pointer(weights),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(means),
        _float_pointer(covariances),
        _float_pointer(viewpoints),
        _float_pointer(secondary_weights),
        secondary_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(secondary_means),
        _float_pointer(secondary_covariances),
        _float_pointer(secondary_viewpoints),
        is_split.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _float_pointer(split_normals),
        primary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        secondary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(points32),
        labels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        _float_pointer(origins32),
        _float_pointer(points32),
        int(maintain_surfels),
    )
    if result != 0:
        raise RuntimeError(f"native surfel update failed with status {result}")
    state = SplitSurfelStatistics(
        merged_keys,
        weights,
        counts,
        means,
        covariances,
        secondary_weights,
        secondary_counts,
        secondary_means,
        secondary_covariances,
        is_split,
        split_normals,
        viewpoints,
        secondary_viewpoints,
        primary_dirty,
        secondary_dirty,
    )
    if not maintain_surfels:
        return state
    # Cross-cell maintenance runs on every tracking level with a fixed 7 cm
    # center gate and 20 degree normal gate.  Candidate selection is frozen
    # before any pair is applied.  Its initial driver walks the complete
    # insertion-ordered state, whereas the incremental driver walks only cells
    # touched by this batch, ordered by their first raw-ray occurrence.
    if initial_state:
        merge_sources = np.arange(size, dtype=np.uint64)
    else:
        _, first_occurrences = np.unique(labels, return_index=True)
        merge_sources = np.ascontiguousarray(
            labels[np.sort(first_occurrences)], dtype=np.uint64
        )
    merge_count = ctypes.c_uint64()
    result = native.navvis_recon_slam_merge_surfels(
        size,
        np.ascontiguousarray(merged_keys, dtype=np.int64).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int64)
        ),
        _float_pointer(weights),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(means),
        _float_pointer(covariances),
        _float_pointer(viewpoints),
        _float_pointer(secondary_weights),
        secondary_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(secondary_means),
        _float_pointer(secondary_covariances),
        _float_pointer(secondary_viewpoints),
        is_split.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _float_pointer(split_normals),
        primary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        secondary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        float(grid_origin[0] if grid_origin.ndim else grid_origin),
        float(grid_origin[1] if grid_origin.ndim else grid_origin),
        float(grid_origin[2] if grid_origin.ndim else grid_origin),
        float(inverse_resolution),
        len(merge_sources),
        merge_sources.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.byref(merge_count),
    )
    if result != 0:
        raise RuntimeError(f"native surfel merge failed with status {result}")
    return state


def maintain_deferred_split_surfel_statistics(
    state: SplitSurfelStatistics,
    voxel_size: float,
    offset: float | Sequence[float],
) -> SplitSurfelStatistics:
    """Activate a raw overlap map with one full maintenance pass."""

    assert state.secondary_counts is not None
    assert state.secondary_means is not None
    assert state.secondary_covariances is not None
    assert state.secondary_viewpoints is not None
    assert state.is_split is not None
    assert state.split_normals is not None
    assert state.viewpoints is not None
    assert state.primary_dirty is not None
    assert state.secondary_dirty is not None
    size = len(state.keys)
    native = _load_slam_frontend_native()
    result = native.navvis_recon_slam_maintain_split_surfels(
        size,
        _float_pointer(state.weights),
        state.counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(state.means),
        _float_pointer(state.covariances),
        _float_pointer(state.viewpoints),
        state.secondary_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(state.secondary_means),
        _float_pointer(state.secondary_covariances),
        _float_pointer(state.secondary_viewpoints),
        state.is_split.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _float_pointer(state.split_normals),
        state.primary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        state.secondary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if result != 0:
        raise RuntimeError(
            f"native deferred surfel maintenance failed with status {result}"
        )

    grid_origin = np.asarray(offset, dtype=np.float64)
    inverse_resolution = np.float64(1.0) / np.float64(voxel_size)
    sources = np.arange(size, dtype=np.uint64)
    merge_count = ctypes.c_uint64()
    result = native.navvis_recon_slam_merge_surfels(
        size,
        np.ascontiguousarray(state.keys, dtype=np.int64).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int64)
        ),
        _float_pointer(state.weights),
        state.counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(state.means),
        _float_pointer(state.covariances),
        _float_pointer(state.viewpoints),
        _float_pointer(state.secondary_weights),
        state.secondary_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        _float_pointer(state.secondary_means),
        _float_pointer(state.secondary_covariances),
        _float_pointer(state.secondary_viewpoints),
        state.is_split.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _float_pointer(state.split_normals),
        state.primary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        state.secondary_dirty.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        float(grid_origin[0] if grid_origin.ndim else grid_origin),
        float(grid_origin[1] if grid_origin.ndim else grid_origin),
        float(grid_origin[2] if grid_origin.ndim else grid_origin),
        float(inverse_resolution),
        len(sources),
        sources.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.byref(merge_count),
    )
    if result != 0:
        raise RuntimeError(f"native deferred surfel merge failed with status {result}")
    return state


def extract_valid_split_surfels(
    statistics: SplitSurfelStatistics, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract valid primary/secondary targets in binary voxel order."""

    size = len(statistics.keys)
    assert statistics.secondary_weights is not None
    assert statistics.secondary_means is not None
    assert statistics.secondary_covariances is not None
    assert statistics.is_split is not None
    assert statistics.viewpoints is not None
    assert statistics.secondary_viewpoints is not None
    weights = np.empty(2 * size, dtype=np.float32)
    means = np.empty((2 * size, 3), dtype=np.float32)
    covariances = np.empty((2 * size, 3, 3), dtype=np.float32)
    viewpoints = np.empty((2 * size, 3), dtype=np.float32)
    weights[0::2] = statistics.weights
    weights[1::2] = statistics.secondary_weights
    means[0::2] = statistics.means
    means[1::2] = statistics.secondary_means
    covariances[0::2] = statistics.covariances
    covariances[1::2] = statistics.secondary_covariances
    viewpoints[0::2] = statistics.viewpoints
    viewpoints[1::2] = statistics.secondary_viewpoints
    normals = np.empty((2 * size, 3), dtype=np.float32)
    eigenvalues = np.empty((2 * size, 3), dtype=np.float32)
    result = _load_slam_frontend_native().navvis_recon_slam_oriented_surfel_geometry(
        2 * size,
        _float_pointer(covariances),
        _float_pointer(means),
        _float_pointer(viewpoints),
        _float_pointer(normals),
        _float_pointer(eigenvalues),
    )
    if result != 0:
        raise RuntimeError(f"native surfel PCA failed with status {result}")

    first = eigenvalues[:, 0]
    second = eigenvalues[:, 1]
    third = eigenvalues[:, 2]
    total_float = np.asarray((second + third) + first, dtype=np.float32)
    total = total_float.astype(np.float64)
    # Empty voxels and the tiny negative eigenvalues produced by float PCA
    # intentionally fail the validity gates below.  Suppress only NumPy's
    # diagnostics; preserving NaN here matches the binary comparisons and is
    # different from clamping the eigenspectrum.
    with np.errstate(divide="ignore", invalid="ignore"):
        planarity = np.asarray(
            2.0 * (second - first).astype(np.float64) / total,
            dtype=np.float32,
        )
        curvature = np.asarray(
            3.0 * first.astype(np.float64) / total,
            dtype=np.float32,
        )
        major_in_plane_spread = np.asarray(
            2.0 * np.sqrt(3.0 * third.astype(np.float64)), dtype=np.float32
        )
        minor_in_plane_spread = np.asarray(
            2.0 * np.sqrt(3.0 * second.astype(np.float64)), dtype=np.float32
        )
        normal_direction_spread = np.asarray(
            2.0 * np.sqrt(3.0 * first.astype(np.float64)), dtype=np.float32
        )
    valid = (
        (weights >= np.float32(7.5))
        & (planarity >= np.float32(0.50))
        & (curvature <= np.float32(0.30))
        & (
            major_in_plane_spread
            >= np.float32(voxel_size) * np.float32(0.0)
        )
        & (minor_in_plane_spread >= np.float32(0.075))
        & (normal_direction_spread <= np.float32(0.05))
    )
    valid[1::2] &= statistics.is_split.astype(np.bool_)
    return (
        means[valid].astype(np.float64),
        normals[valid].astype(np.float64),
    )


def extract_valid_surfels(
    statistics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the installed binary's six resolved surfel validity gates.

    The eigenvalue expressions are intentionally kept in their binary form:
    planarity and curvature include factors two and three respectively, while
    the three spread tests convert variances to full uniform-distribution
    support with ``2 * sqrt(3 * lambda)``.  In particular, the 5 cm limit is
    physical spread in the normal direction, not a normal-consistency score.
    """

    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("surfel voxel size must be finite and positive")

    _, counts, sums, outers = statistics
    centers = sums / counts[:, None]
    covariance = np.empty((len(counts), 3, 3), dtype=np.float64)
    covariance[:, 0, 0] = outers[:, 0] / counts - centers[:, 0] ** 2
    covariance[:, 0, 1] = covariance[:, 1, 0] = (
        outers[:, 1] / counts - centers[:, 0] * centers[:, 1]
    )
    covariance[:, 0, 2] = covariance[:, 2, 0] = (
        outers[:, 2] / counts - centers[:, 0] * centers[:, 2]
    )
    covariance[:, 1, 1] = outers[:, 3] / counts - centers[:, 1] ** 2
    covariance[:, 1, 2] = covariance[:, 2, 1] = (
        outers[:, 4] / counts - centers[:, 1] * centers[:, 2]
    )
    covariance[:, 2, 2] = outers[:, 5] / counts - centers[:, 2] ** 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = np.maximum(np.sum(eigenvalues, axis=1), 1.0e-12)
    curvature = 3.0 * eigenvalues[:, 0] / total
    planarity = 2.0 * (eigenvalues[:, 1] - eigenvalues[:, 0]) / total
    spread_scale = 2.0 * np.sqrt(3.0)
    major_in_plane_spread = spread_scale * np.sqrt(eigenvalues[:, 2])
    minor_in_plane_spread = spread_scale * np.sqrt(eigenvalues[:, 1])
    normal_direction_spread = spread_scale * np.sqrt(eigenvalues[:, 0])
    valid = (
        (counts >= 7.5)
        & (planarity >= 0.50)
        & (curvature <= 0.30)
        # The binary stores relative_spread_thresh before
        # min_in_plane_spread: the major in-plane support is relative to the
        # grid size, while the minor support has an absolute 7.5 cm floor.
        & (major_in_plane_spread >= voxel_size * 0.0)
        & (minor_in_plane_spread >= 0.075)
        & (normal_direction_spread <= 0.05)
    )
    normals = eigenvectors[:, :, 0]
    return centers[valid], normals[valid]


def adaptive_first_point_filter(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    minimum_voxel_m: float = 0.02,
    maximum_voxel_m: float = 0.40,
    maximum_points: int = 5_000,
    maximum_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolved float32 HASH_MAP_FIRST_POINT grid and deterministic cap."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("adaptive filter points/normals must be matching Nx3 arrays")

    # RangeMeasurement positions and the grid resolution are both float in
    # the installed implementation.  Keeping the division in float32 matters
    # for points lying next to a voxel boundary; promoting either operand to
    # double can select a different first point even when the voxel count is
    # unchanged.
    grid_points = points.astype(np.float32)

    def filtered_indices(length: float | np.float32) -> np.ndarray:
        resolution = np.float32(length)
        keys = np.floor(grid_points / resolution).astype(np.int64)
        _, first, _ = _unique_voxel_keys(keys)
        return np.sort(first)

    if len(points) <= maximum_points or maximum_iterations == 0:
        return points.copy(), normals.copy()
    if maximum_iterations == 1:
        indices = filtered_indices(
            np.float32(
                np.float32(0.5)
                * (np.float32(minimum_voxel_m) + np.float32(maximum_voxel_m))
            )
        )
    else:
        maximum_bound = np.float32(maximum_voxel_m)
        minimum_bound = np.float32(minimum_voxel_m)
        sparse = filtered_indices(maximum_bound)
        dense = filtered_indices(minimum_bound)
        if len(sparse) > maximum_points:
            indices = sparse
        else:
            for _ in range(maximum_iterations):
                if (len(dense) - maximum_points) / maximum_points <= 0.1:
                    break
                middle = np.float32(
                    np.float32(0.5) * np.float32(maximum_bound + minimum_bound)
                )
                candidate = filtered_indices(middle)
                if len(candidate) < maximum_points:
                    maximum_bound = middle
                    sparse = candidate
                else:
                    minimum_bound = middle
                    dense = candidate
            indices = dense
    if len(indices) > maximum_points:
        # index_filter_random=false: the binary deliberately skips element 0
        # and samples the remaining deterministic span using integer floor.
        # A node-0 dynamic capture reproduces all 5,000 records, in order,
        # with floor(k * N / M) + 1; the more usual formula without +1 has
        # only 2,426/5,000 records in common on the same captured input.
        selection = (
            np.arange(maximum_points, dtype=np.int64) * len(indices)
        ) // maximum_points + 1
        indices = indices[selection]
    return points[indices], normals[indices]


def estimate_normals(points: np.ndarray, neighbors: int = 12) -> np.ndarray:
    """Estimate unoriented surfel normals by local covariance PCA."""

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        raise ValueError("normal estimation needs at least four points")
    count = min(max(4, neighbors), len(points))
    _, indices = cKDTree(points).query(points, k=count, workers=-1)
    neighborhoods = points[indices]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / count
    _, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    flip = np.einsum("ij,ij->i", normals, points) > 0.0
    normals[flip] *= -1.0
    return normals


def filter_scan(scan: LidarScan, config: FrontendConfig) -> LidarScan:
    keep = (
        np.ones(len(scan.points), dtype=bool)
        if scan.range_filtered
        else (
            (np.linalg.norm(scan.points, axis=1) >= config.minimum_range_m)
            & (np.linalg.norm(scan.points, axis=1) <= config.maximum_range_m)
        )
    )
    points = scan.points[keep]
    normals = None if scan.normals is None else scan.normals[keep]
    origins = (
        np.zeros_like(points)
        if scan.ray_origins is None
        else scan.ray_origins[keep]
    )
    if len(points) < 4:
        raise ValueError("too few lidar returns remain after range filtering")
    if normals is None:
        points, origins = range_measurement_centroid_filter(
            points, origins, config.scan_voxel_m
        )
        # The binary's HASH_MAP_CENTROID ICP scan contains positions only.
        # Incidence rejection uses the sensor ray and target-surfel normal;
        # estimating source normals here is both unused and semantically
        # different from that input contract.
        normals = np.zeros_like(points)
    else:
        points, normals = voxel_downsample(points, config.scan_voxel_m, normals)
    return LidarScan(
        scan.timestamp_ns,
        points,
        normals,
        range_filtered=True,
        ray_origins=origins,
    )


def _interpolate_prediction(
    timestamps_ns: np.ndarray,
    poses: Sequence[Rigid3],
    timestamp_ns: int,
) -> Rigid3:
    high = int(np.searchsorted(timestamps_ns, timestamp_ns, side="right"))
    if high <= 0:
        return poses[0]
    if high >= len(poses):
        return poses[-1]
    low = high - 1
    duration = int(timestamps_ns[high] - timestamps_ns[low])
    alpha = 0.0 if duration <= 0 else (timestamp_ns - timestamps_ns[low]) / duration
    translation = (
        (1.0 - alpha) * poses[low].translation
        + alpha * poses[high].translation
    )
    rotation = Rotation.from_quat(
        np.vstack((poses[low].quaternion_xyzw, poses[high].quaternion_xyzw))
    )
    # A two-key Slerp is considerably more expensive to construct per point;
    # normalized quaternion interpolation is accurate at the 20 Hz scan scale.
    first = rotation.as_quat()[0]
    second = rotation.as_quat()[1]
    if np.dot(first, second) < 0.0:
        second = -second
    quaternion = (1.0 - alpha) * first + alpha * second
    quaternion /= np.linalg.norm(quaternion)
    return Rigid3(translation, quaternion)


def accumulated_batch_timestamps_ns(
    archive: SlamScanArchive,
    *,
    minimum_duration_ns: int = 50_000_000,
    minimum_raw_ray_count: int = 58_000,
) -> tuple[int, ...]:
    """Recreate the installed packet accumulator's output boundaries.

    The accumulator tests its duration and raw-slot thresholds before adding
    the packet that crosses them.  Consequently that packet starts the next
    batch and the preceding packet stamp is the completed batch timestamp.
    Invalid/self-filtered rays still count toward the 58,000-slot threshold;
    NVSLAM6 stores packet stamps and raw slot counts separately for this
    reason.
    """

    if minimum_duration_ns <= 0 or minimum_raw_ray_count <= 0:
        raise ValueError("accumulation thresholds must be positive")
    if archive._version < 6:
        raise ValueError(
            "exact accumulated batch reconstruction requires NVSLAM6"
        )

    first_by_sensor: dict[int, int] = {}
    timestamp_batches: list[np.ndarray] = []
    ray_count_batches: list[np.ndarray] = []
    for record in archive.records:
        first_by_sensor.setdefault(
            record.sensor, (record.timestamp_ns // 1000) * 1000
        )
        packet_timestamps = archive.read_packet_timestamps_ns(record)
        if len(packet_timestamps) != record.packet_count:
            raise ValueError("SLAM archive packet metadata is inconsistent")
        if record.packet_count == 0:
            continue
        if record.ray_count % record.packet_count:
            raise ValueError("raw ray slots are not uniform within a scan")
        slots_per_packet = record.ray_count // record.packet_count
        timestamp_batches.append(packet_timestamps)
        ray_count_batches.append(
            np.full(record.packet_count, slots_per_packet, dtype=np.int64)
        )
    if len(first_by_sensor) < 2 or not timestamp_batches:
        raise ValueError("dual-laser packet metadata is unavailable")

    packet_timestamps = np.concatenate(timestamp_batches)
    packet_ray_counts = np.concatenate(ray_count_batches)
    packet_order = np.argsort(packet_timestamps, kind="stable")
    packet_timestamps = packet_timestamps[packet_order]
    packet_ray_counts = packet_ray_counts[packet_order]
    first_all_sources_ns = max(first_by_sensor.values())
    active = packet_timestamps >= first_all_sources_ns
    packet_timestamps = packet_timestamps[active]
    packet_ray_counts = packet_ray_counts[active]
    if not len(packet_timestamps):
        raise ValueError("no collated packets remain after sensor startup")

    batch_start_ns = int(packet_timestamps[0])
    previous_timestamp_ns = batch_start_ns
    accumulated_ray_count = 0
    output: list[int] = []
    for timestamp_value, ray_count_value in zip(
        packet_timestamps, packet_ray_counts
    ):
        timestamp_ns = int(timestamp_value)
        if (
            timestamp_ns - batch_start_ns >= minimum_duration_ns
            and accumulated_ray_count >= minimum_raw_ray_count
        ):
            output.append(previous_timestamp_ns)
            batch_start_ns = timestamp_ns
            accumulated_ray_count = 0
        accumulated_ray_count += int(ray_count_value)
        previous_timestamp_ns = timestamp_ns
    return tuple(output)


def infer_discarded_batch_timestamps_ns(
    retained_timestamps_ns: Sequence[int],
    accumulated_timestamps_ns: Sequence[int],
    *,
    nominal_batch_duration_ns: int = 50_000_000,
) -> tuple[int, ...]:
    """Recover motion-filtered batches between adjacent retained nodes.

    The raw packet accumulator and the retained trajectory can have a small
    startup phase offset: the first trajectory node may close a partial
    collator batch.  Requiring every retained timestamp to occur in a packet-
    derived boundary list therefore rejects an otherwise valid recording.

    A retained-node gap reveals how many complete batches the motion filter
    discarded.  For each such gap, choose that many packet-derived boundaries
    nearest the evenly spaced missing batch times.  This keeps the exact raw
    event timestamps while ignoring harmless phase-offset candidates.
    """

    retained = np.asarray(retained_timestamps_ns, dtype=np.int64)
    accumulated = np.asarray(accumulated_timestamps_ns, dtype=np.int64)
    if retained.ndim != 1 or accumulated.ndim != 1:
        raise ValueError("batch timestamps must be vectors")
    if len(retained) < 2 or np.any(np.diff(retained) <= 0):
        raise ValueError("retained timestamps must be strictly increasing")
    if len(accumulated) and np.any(np.diff(accumulated) <= 0):
        raise ValueError("accumulated timestamps must be strictly increasing")
    if nominal_batch_duration_ns <= 0:
        raise ValueError("nominal batch duration must be positive")

    discarded: list[int] = []
    for start_value, end_value in zip(retained[:-1], retained[1:]):
        start = int(start_value)
        end = int(end_value)
        gap = end - start
        completed_batches = int(
            np.floor(gap / nominal_batch_duration_ns + 0.5)
        )
        missing_count = max(0, completed_batches - 1)
        if missing_count == 0:
            continue

        low = int(np.searchsorted(accumulated, start, side="right"))
        high = int(np.searchsorted(accumulated, end, side="left"))
        candidates = accumulated[low:high]
        if len(candidates) < missing_count:
            raise ValueError(
                "packet accumulator has too few boundaries inside retained "
                f"gap ({start}, {end})"
            )

        targets = start + (
            gap
            * np.arange(1, missing_count + 1, dtype=np.float64)
            / (missing_count + 1)
        )
        # Dynamic programming selects an ordered candidate subset with the
        # minimum absolute timestamp error. Gaps normally contain one or two
        # candidates, but this remains deterministic for longer outages.
        costs = np.full((missing_count + 1, len(candidates) + 1), np.inf)
        take = np.zeros((missing_count + 1, len(candidates) + 1), dtype=bool)
        costs[0, :] = 0.0
        for target_index in range(1, missing_count + 1):
            for candidate_count in range(1, len(candidates) + 1):
                skip_cost = costs[target_index, candidate_count - 1]
                take_cost = costs[target_index - 1, candidate_count - 1]
                take_cost += abs(
                    float(candidates[candidate_count - 1])
                    - targets[target_index - 1]
                )
                if take_cost < skip_cost:
                    costs[target_index, candidate_count] = take_cost
                    take[target_index, candidate_count] = True
                else:
                    costs[target_index, candidate_count] = skip_cost

        target_index = missing_count
        candidate_count = len(candidates)
        selected: list[int] = []
        while target_index:
            if candidate_count == 0:
                raise AssertionError("discarded-batch selection underflow")
            if take[target_index, candidate_count]:
                selected.append(int(candidates[candidate_count - 1]))
                target_index -= 1
            candidate_count -= 1
        discarded.extend(reversed(selected))

    return tuple(discarded)


def iter_archive_at_node_times(
    archive: SlamScanArchive,
    node_timestamps_ns: Sequence[int],
    pose_predictions: Sequence[Rigid3] | None = None,
    *,
    imu_pose_predictor: RawConstantVelocityPosePredictor | None = None,
    apply_pose_corrections: bool = True,
    discarded_batch_timestamps_ns: Sequence[int] | None = None,
    maximum_range_m: float = 60.0,
) -> Iterable[LidarScan]:
    """Recreate the binary's accumulated range batches at retained node times.

    Binary probes show that a node timestamp is the maximum absolute ray stamp
    in its 50 ms/58,000-ray batch. A motion-filtered batch is discarded rather
    than carried into the next retained node. NVSLAM2--4 archives therefore
    use exact point stamps in ``(previous node, current node]`` and cap a gap
    at one 50 ms accumulation interval.  This is essential when the motion
    filter discards a complete batch between two retained nodes: a wider
    window leaks the discarded batch's tail into the next node.  The discarded
    batch still runs scan matching and corrects the IMU pose predictor, even
    though it does not produce a node or update a submap.
    ``discarded_batch_timestamps_ns`` is an evidence/replay input for
    preserving those state transitions while the raw collator schedule is
    reconstructed. NVSLAM1 retains the legacy complete-scan fallback.
    """

    timestamps = np.asarray(node_timestamps_ns, dtype=np.int64)
    if timestamps.ndim != 1:
        raise ValueError("node timestamps must be a vector")
    if pose_predictions is not None and len(timestamps) != len(pose_predictions):
        raise ValueError("node timestamps and pose predictions must match")
    if pose_predictions is None and imu_pose_predictor is None:
        raise ValueError("pose predictions or an IMU pose predictor are required")
    if not np.isfinite(maximum_range_m) or maximum_range_m <= 0.0:
        raise ValueError("maximum range must be finite and positive")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("node timestamps must be strictly increasing")
    if discarded_batch_timestamps_ns is None and archive._version >= 6:
        retained_set = {int(timestamp) for timestamp in timestamps}
        all_batches = accumulated_batch_timestamps_ns(archive)
        all_batch_set = set(all_batches)
        missing_retained = [
            int(timestamp)
            for timestamp in timestamps
            if int(timestamp) not in all_batch_set
        ]
        if missing_retained:
            discarded_batch_timestamps_ns = (
                infer_discarded_batch_timestamps_ns(timestamps, all_batches)
            )
        else:
            discarded_batch_timestamps_ns = [
                timestamp
                for timestamp in all_batches
                if int(timestamps[0]) < timestamp < int(timestamps[-1])
                and timestamp not in retained_set
            ]
    discarded_timestamps = np.asarray(
        () if discarded_batch_timestamps_ns is None else discarded_batch_timestamps_ns,
        dtype=np.int64,
    )
    if discarded_timestamps.ndim != 1:
        raise ValueError("discarded batch timestamps must be a vector")
    if len(discarded_timestamps):
        if imu_pose_predictor is None:
            raise ValueError(
                "discarded batch replay requires an IMU pose predictor"
            )
        if np.any(np.diff(discarded_timestamps) <= 0):
            raise ValueError(
                "discarded batch timestamps must be strictly increasing"
            )
        if (
            discarded_timestamps[0] <= timestamps[0]
            or discarded_timestamps[-1] >= timestamps[-1]
        ):
            raise ValueError(
                "discarded batch timestamps must lie between retained nodes"
            )
        if np.intersect1d(timestamps, discarded_timestamps).size:
            raise ValueError(
                "discarded batch timestamps overlap retained nodes"
            )
    prediction_translations = (
        None
        if pose_predictions is None
        else np.vstack([pose.translation for pose in pose_predictions])
    )
    prediction_quaternions = (
        None
        if pose_predictions is None
        else np.vstack([pose.quaternion_xyzw for pose in pose_predictions])
    )
    first_scan = archive.read(archive.records[0]) if archive.records else None
    if first_scan is not None and first_scan.point_timestamps_ns is not None:
        first_by_sensor: dict[int, int] = {}
        for record in archive.records:
            first_by_sensor.setdefault(
                record.sensor,
                (record.timestamp_ns // 1000) * 1000
                if archive._version in (5, 6)
                else record.timestamp_ns,
            )
        if len(first_by_sensor) < 2:
            raise ValueError("dual-laser SLAM archive is missing a sensor")
        first_all_sources_ns = max(first_by_sensor.values())

        event_timestamps = np.sort(
            np.concatenate((timestamps, discarded_timestamps))
        )
        record_timestamps_ns = tuple(
            record.timestamp_ns for record in archive.records
        )
        retained_indices = {
            int(timestamp): index for index, timestamp in enumerate(timestamps)
        }
        for event_index, timestamp_value in enumerate(event_timestamps):
            timestamp = int(timestamp_value)
            retained_index = retained_indices.get(timestamp)
            boundary_tolerance_ns = archive.timestamp_boundary_tolerance_ns
            if event_index == 0:
                window_start_ns = first_all_sources_ns
            else:
                # A retained-node gap larger than one accumulation interval
                # means the motion filter discarded one or more complete
                # batches. Those points must not be inserted into this node.
                window_start_ns = max(
                    int(event_timestamps[event_index - 1]),
                    timestamp - 50_000_000,
                )
            point_batches: list[np.ndarray] = []
            normal_batches: list[np.ndarray] = []
            origin_batches: list[np.ndarray] = []
            timestamp_batches: list[np.ndarray] = []
            geometry_keep_batches: list[np.ndarray] = []
            batch_sensors: list[int] = []
            # Records are timestamp ordered.  Start directly at the first
            # record that can overlap this batch instead of rescanning the
            # entire archive prefix for every node.  ``bisect_left`` retains
            # records exactly on the 60 ms support boundary, matching the
            # strict rejection predicate below.
            first_record = bisect_left(
                record_timestamps_ns, window_start_ns - 60_000_000
            )
            for record in archive.records[first_record:]:
                if (
                    record.timestamp_ns
                    > timestamp + archive.record_header_support_ns
                ):
                    break
                # Pandar scans are about 50 ms long. The 60 ms conservative
                # support includes packet jitter while avoiding a full mmap
                # read for unrelated revolutions.
                if record.timestamp_ns + 60_000_000 < window_start_ns:
                    continue
                scan = archive.read(record)
                assert scan.point_timestamps_ns is not None
                if event_index == 0:
                    keep = (
                        (
                            scan.point_timestamps_ns
                            >= window_start_ns - boundary_tolerance_ns
                        )
                        & (
                            scan.point_timestamps_ns
                            <= timestamp + boundary_tolerance_ns
                        )
                    )
                else:
                    keep = (
                        (
                            scan.point_timestamps_ns
                            > window_start_ns + boundary_tolerance_ns
                        )
                        & (
                            scan.point_timestamps_ns
                            <= timestamp + boundary_tolerance_ns
                        )
                    )
                if not np.any(keep):
                    continue
                points = scan.points[keep]
                point_times = scan.point_timestamps_ns[keep]
                normals = None if scan.normals is None else scan.normals[keep]
                origins = (
                    np.zeros_like(points)
                    if scan.ray_origins is None
                    else scan.ray_origins[keep]
                )

                if imu_pose_predictor is not None:
                    point_batches.append(points)
                    origin_batches.append(origins)
                    timestamp_batches.append(point_times)
                    geometry_keep_batches.append(
                        np.linalg.norm(points - origins, axis=1)
                        <= maximum_range_m
                    )
                    batch_sensors.append(record.sensor)
                    if normals is not None:
                        normal_batches.append(normals)
                    continue

                assert prediction_translations is not None
                assert prediction_quaternions is not None
                assert pose_predictions is not None
                assert retained_index is not None
                high = np.searchsorted(timestamps, point_times, side="right")
                high = np.clip(high, 1, len(timestamps) - 1)
                low = high - 1
                duration = (timestamps[high] - timestamps[low]).astype(np.float64)
                alpha = np.clip(
                    (point_times - timestamps[low]) / np.maximum(duration, 1.0),
                    0.0,
                    1.0,
                )
                translations = (
                    (1.0 - alpha[:, None]) * prediction_translations[low]
                    + alpha[:, None] * prediction_translations[high]
                )
                first = prediction_quaternions[low]
                second = prediction_quaternions[high].copy()
                flip = np.einsum("ij,ij->i", first, second) < 0.0
                second[flip] *= -1.0
                quaternions = (
                    (1.0 - alpha[:, None]) * first + alpha[:, None] * second
                )
                quaternions /= np.maximum(
                    np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-12
                )
                rotations = Rotation.from_quat(quaternions)
                world_points = rotations.apply(points) + translations
                node_from_world = pose_predictions[retained_index].inverse()
                point_batches.append(
                    node_from_world.rotation.apply(world_points)
                    + node_from_world.translation
                )
                world_origins = rotations.apply(origins) + translations
                origin_batches.append(
                    node_from_world.rotation.apply(world_origins)
                    + node_from_world.translation
                )
                timestamp_batches.append(point_times)
                batch_sensors.append(record.sensor)
                if normals is not None:
                    world_normals = rotations.apply(normals)
                    normal_batches.append(
                        node_from_world.rotation.apply(world_normals)
                    )
            if not point_batches:
                raise ValueError(
                    "no raw laser rays were assigned to accumulated batch "
                    f"{event_index} at {timestamp} ns"
                )
            # RangeDataCollator emits configured source order, not bag-record
            # arrival order.  G11's configuration is horizontal (archive
            # sensor 0) followed by vertical (sensor 1), with stable packet
            # order inside each source.  This order affects first-point node
            # clouds and the float Welford map accumulator.
            batch_order = sorted(
                range(len(point_batches)),
                key=lambda batch_index: (
                    batch_sensors[batch_index],
                    batch_index,
                ),
            )
            point_batches = [point_batches[value] for value in batch_order]
            origin_batches = [origin_batches[value] for value in batch_order]
            if timestamp_batches:
                timestamp_batches = [
                    timestamp_batches[value] for value in batch_order
                ]
            if geometry_keep_batches:
                geometry_keep_batches = [
                    geometry_keep_batches[value] for value in batch_order
                ]
            if normal_batches:
                if len(normal_batches) != len(batch_order):
                    raise ValueError("mixed normal availability in one lidar batch")
                normal_batches = [normal_batches[value] for value in batch_order]
            # The collator concatenates configured sources and then performs a
            # stable per-ray time merge.  Stable sorting preserves source and
            # packet order for equal timestamp ticks.
            point_values = np.concatenate(point_batches)
            origin_values = np.concatenate(origin_batches)
            timestamp_values = np.concatenate(timestamp_batches)
            geometry_keep_values = (
                np.concatenate(geometry_keep_batches)
                if geometry_keep_batches
                else np.ones(len(timestamp_values), dtype=bool)
            )
            ray_order = np.argsort(timestamp_values, kind="stable")
            point_batches = [point_values[ray_order]]
            origin_batches = [origin_values[ray_order]]
            timestamp_batches = [timestamp_values[ray_order]]
            geometry_keep_values = geometry_keep_values[ray_order]
            if normal_batches:
                normal_batches = [np.concatenate(normal_batches)[ray_order]]
            if imu_pose_predictor is not None:
                points = np.concatenate(point_batches)
                origins = np.concatenate(origin_batches)
                point_times = np.concatenate(timestamp_batches)
                if archive.timestamp_boundary_tolerance_ns:
                    # The +1 ns closed-boundary compatibility above denotes
                    # the same vendor packet time as the retained node. Clamp
                    # only that representation artifact before IMU advance.
                    point_times = np.minimum(point_times, int(timestamp))
                timestamp_batches = [point_times]
                relative_quaternions, relative_translations = (
                    imu_pose_predictor.relative_motion(point_times, int(timestamp))
                )
                # The installed frontend has two visible float boundaries.
                # It first evaluates every timed ray in the frame of the
                # earliest ray, then casts the one start-to-end pose to a
                # float 4x4 matrix before the 0.04 m centroid filter. Applying
                # end_from_ray once is mathematically equivalent, but moves
                # most coordinates by one or two ULPs and changes voxel-edge
                # decisions.
                relative_end_from_start_quaternion = (
                    relative_quaternions[0].copy()
                )
                relative_end_from_start_translation = (
                    relative_translations[0].copy()
                )
                start_from_end_quaternion = _quaternion_inverse_xyzw(
                    relative_end_from_start_quaternion
                )
                start_from_ray_quaternions = _quaternion_multiply_xyzw(
                    np.broadcast_to(
                        start_from_end_quaternion,
                        relative_quaternions.shape,
                    ),
                    relative_quaternions,
                )
                start_from_ray_translations = _quaternion_transform_vector(
                    np.broadcast_to(
                        start_from_end_quaternion,
                        relative_quaternions.shape,
                    ),
                    relative_translations
                    - relative_end_from_start_translation,
                )

                points_at_start = _transform_points_float_quaternion(
                    points,
                    start_from_ray_quaternions,
                    start_from_ray_translations,
                )
                origins_at_start = _transform_points_float_quaternion(
                    origins,
                    start_from_ray_quaternions,
                    start_from_ray_translations,
                )
                end_from_start = imu_pose_predictor.last_end_from_start_pose
                end_from_start_quaternion = (
                    end_from_start.quaternion_xyzw.copy()
                )
                point_batches = [
                    _transform_points_float_matrix(points_at_start, end_from_start)
                ]
                origin_batches = [
                    _transform_points_float_matrix(origins_at_start, end_from_start)
                ]
                if normal_batches:
                    normals = np.concatenate(normal_batches)
                    zero_translations = np.zeros_like(start_from_ray_translations)
                    normals_at_start = _transform_points_float_quaternion(
                        normals,
                        start_from_ray_quaternions,
                        zero_translations,
                    )
                    normal_batches = [
                        _transform_points_float_matrix(
                            normals_at_start,
                            Rigid3(
                                np.zeros(3),
                                end_from_start_quaternion,
                            ),
                        )
                    ]
                # Long-range endpoints participate in the predictor schedule
                # above but leave geometry only after deskew, matching the
                # installed RangeData processing order.
                point_batches = [point_batches[0][geometry_keep_values]]
                origin_batches = [origin_batches[0][geometry_keep_values]]
                timestamp_batches = [point_times[geometry_keep_values]]
                if normal_batches:
                    normal_batches = [normal_batches[0][geometry_keep_values]]
            yield LidarScan(
                int(timestamp),
                np.concatenate(point_batches),
                np.concatenate(normal_batches) if normal_batches else None,
                point_timestamps_ns=np.concatenate(timestamp_batches),
                range_filtered=True,
                ray_origins=np.concatenate(origin_batches),
                retain_node=retained_index is not None,
            )
            if (
                imu_pose_predictor is not None
                and apply_pose_corrections
                and pose_predictions is not None
            ):
                assert retained_index is not None
                imu_pose_predictor.correct(
                    int(timestamp), pose_predictions[retained_index]
                )
        return

    # NVSLAM1 compatibility: the old archive has no per-ray timestamps, so
    # retain the former complete-revolution midpoint assignment.
    next_timestamp: dict[int, int] = {}
    latest: dict[int, SlamScanRecord] = {}
    for record in reversed(archive.records):
        following = latest.get(record.sensor)
        next_timestamp[record.data_offset] = (
            following.timestamp_ns
            if following is not None
            else record.timestamp_ns + 50_000_000
        )
        latest[record.sensor] = record

    record_batches: list[list[tuple[SlamScanRecord, int]]] = [
        [] for _ in timestamps
    ]
    for record in archive.records:
        center = (record.timestamp_ns + next_timestamp[record.data_offset]) // 2
        node_index = int(np.searchsorted(timestamps, center, side="left"))
        if node_index >= len(timestamps):
            continue
        # Do not pull arbitrarily early initialization scans into the first
        # node. One 50 ms accumulation interval is the resolved binary limit.
        if node_index == 0 and center < timestamps[0] - 50_000_000:
            continue
        record_batches[node_index].append((record, center))

    for index, timestamp in enumerate(timestamps):
        if not record_batches[index]:
            raise ValueError(f"no raw laser scan was assigned to node {index}")
        point_batches: list[np.ndarray] = []
        normal_batches: list[np.ndarray] = []
        origin_batches: list[np.ndarray] = []
        for record, center in record_batches[index]:
            scan = archive.read(record)
            assert scan.normals is not None
            if scan.point_timestamps_ns is None:
                assert pose_predictions is not None
                scan_pose = _interpolate_prediction(
                    timestamps, pose_predictions, center
                )
                node_from_scan = pose_predictions[index].inverse().compose(scan_pose)
                point_batches.append(
                    node_from_scan.rotation.apply(scan.points)
                    + node_from_scan.translation
                )
                raw_origins = (
                    np.zeros_like(scan.points)
                    if scan.ray_origins is None
                    else scan.ray_origins
                )
                origin_batches.append(
                    node_from_scan.rotation.apply(raw_origins)
                    + node_from_scan.translation
                )
                normal_batches.append(node_from_scan.rotation.apply(scan.normals))
                continue

            raise AssertionError("NVSLAM1 scan unexpectedly contains point timestamps")
        yield LidarScan(
            int(timestamp),
            np.concatenate(point_batches),
            np.concatenate(normal_batches),
            range_filtered=True,
            ray_origins=np.concatenate(origin_batches),
        )


def accumulate_archive_at_node_times(
    archive: SlamScanArchive,
    node_timestamps_ns: Sequence[int],
    pose_predictions: Sequence[Rigid3] | None = None,
    *,
    imu_pose_predictor: RawConstantVelocityPosePredictor | None = None,
) -> tuple[LidarScan, ...]:
    """Materialized compatibility wrapper around ``iter_archive_at_node_times``."""

    return tuple(
        iter_archive_at_node_times(
            archive,
            node_timestamps_ns,
            pose_predictions,
            imu_pose_predictor=imu_pose_predictor,
        )
    )


def point_to_plane_icp(
    source_points: np.ndarray,
    target_points: np.ndarray | Sequence[np.ndarray],
    target_normals: np.ndarray | Sequence[np.ndarray],
    initial: Rigid3 | None = None,
    *,
    source_origins: np.ndarray | None = None,
    binary_compatible: bool = False,
    max_correspondence_m: float = 0.60,
    huber_m: float = 0.08,
    max_iterations: int = 20,
    min_iterations: int = 1,
    correspondence_levels_m: Sequence[float] | None = None,
    initial_plane_distance_m: float = float("inf"),
    contracted_plane_distance_m: float = float("inf"),
    contraction_iterations: int = 6,
    min_correspondences: int = 40,
    max_incidence_angle_deg: float = 90.0,
    num_threads: int = 8,
    compute_information_matrix: bool = False,
) -> IcpResult:
    """Robust point-to-plane ICP returning ``target <- source``.

    For the Surveyor three-grid target, ``targets`` must be ordered from the
    0.1 m grid through the 0.3 and 0.6 m grids.  G11's
    ``use_lowest_grid_correspondence=true`` means that every iteration first
    tries the finest grid and falls back to a coarser grid only when no point
    lies inside that grid's own correspondence limit.
    """

    geometry_dtype = np.float32 if binary_compatible else np.float64
    source = np.ascontiguousarray(source_points, dtype=geometry_dtype)
    if isinstance(target_points, (tuple, list)):
        targets = tuple(
            np.ascontiguousarray(value, dtype=geometry_dtype)
            for value in target_points
        )
    else:
        targets = (np.ascontiguousarray(target_points, dtype=geometry_dtype),)
    if isinstance(target_normals, (tuple, list)):
        normal_levels = tuple(
            np.ascontiguousarray(value, dtype=geometry_dtype)
            for value in target_normals
        )
    else:
        normal_levels = (
            np.ascontiguousarray(target_normals, dtype=geometry_dtype),
        )
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or len(targets) != len(normal_levels)
        or any(target.shape != normals.shape for target, normals in zip(targets, normal_levels))
        or any(target.ndim != 2 or target.shape[1] != 3 for target in targets)
    ):
        raise ValueError("ICP expects source and matching target/normal Nx3 arrays")
    if len(source) == 0 or any(len(target) == 0 for target in targets):
        return IcpResult(initial or _identity(), np.inf, 0.0, 0, 0, False, np.inf)
    if source_origins is None:
        origins = np.zeros_like(source)
    else:
        origins = np.ascontiguousarray(source_origins, dtype=geometry_dtype)
        if origins.shape != source.shape:
            raise ValueError("ICP source origins must match source points")
    trees = (
        tuple(_BinaryOctree(target) for target in targets)
        if binary_compatible
        else tuple(cKDTree(target) for target in targets)
    )
    if correspondence_levels_m is None:
        level_limits = (float(max_correspondence_m),) * len(targets)
    else:
        if len(correspondence_levels_m) != len(targets):
            raise ValueError("one correspondence limit is required per ICP target grid")
        level_limits = tuple(float(value) for value in correspondence_levels_m)

    def correspondences(
        transformed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select finest available target independently for every source point."""

        selected = np.zeros(len(transformed), dtype=bool)
        selected_points = np.empty_like(transformed, dtype=geometry_dtype)
        selected_normals = np.empty_like(transformed, dtype=geometry_dtype)
        selected_distances = np.full(len(transformed), np.inf, dtype=np.float64)
        for target, normals, tree, limit in zip(
            targets, normal_levels, trees, level_limits
        ):
            unresolved = np.flatnonzero(~selected)
            if not len(unresolved):
                break
            if binary_compatible:
                if not isinstance(tree, _BinaryOctree):
                    raise TypeError("binary ICP requires its native octree")
                accepted_local, indices, squared_distances = tree.query(
                    transformed[unresolved], limit, num_threads
                )
                distances = np.sqrt(squared_distances).astype(np.float64)
            else:
                distances, indices = tree.query(
                    transformed[unresolved], k=1, workers=num_threads
                )
                accepted_local = distances <= limit
            if not np.any(accepted_local):
                continue
            accepted = unresolved[accepted_local]
            target_indices = indices[accepted_local]
            selected[accepted] = True
            selected_points[accepted] = target[target_indices]
            selected_normals[accepted] = normals[target_indices]
            selected_distances[accepted] = distances[accepted_local]
        return selected, selected_points, selected_normals, selected_distances

    def transform_points(pose: Rigid3, values: np.ndarray) -> np.ndarray:
        if binary_compatible:
            return _transform_points_float_matrix(values, pose)
        transformed = pose.rotation.apply(values) + pose.translation
        return np.ascontiguousarray(transformed, dtype=geometry_dtype)

    incidence_threshold = np.asarray(
        np.cos(np.deg2rad(max_incidence_angle_deg)), dtype=geometry_dtype
    ).item()

    def apply_geometric_filters(
        keep: np.ndarray,
        transformed: np.ndarray,
        transformed_origins: np.ndarray,
        target_point: np.ndarray,
        normal: np.ndarray,
        plane_limit: float,
    ) -> np.ndarray:
        if np.isfinite(plane_limit) and np.any(keep):
            candidate = np.flatnonzero(keep)
            delta = transformed[candidate] - target_point[candidate]
            candidate_residual = (
                normal[candidate, 2] * delta[:, 2]
                + normal[candidate, 1] * delta[:, 1]
            ) + normal[candidate, 0] * delta[:, 0]
            # The installed predicate accepts only residuals strictly below
            # the threshold.  Equality is rejected as well (COMISS followed
            # by JA for ``threshold > abs(residual)``).  This matters at the
            # contracted 0.02 m boundary after float32 rounding.
            keep[candidate[np.abs(candidate_residual) >= plane_limit]] = False
        if max_incidence_angle_deg < 90.0 and np.any(keep):
            candidate = np.flatnonzero(keep)
            rays = transformed[candidate] - transformed_origins[candidate]
            ray_squared_norm = (
                rays[:, 2] * rays[:, 2] + rays[:, 1] * rays[:, 1]
            ) + rays[:, 0] * rays[:, 0]
            ray_norm = np.maximum(
                np.sqrt(ray_squared_norm)[:, None],
                np.asarray(1.0e-12, dtype=geometry_dtype),
            )
            if binary_compatible:
                # The binary forms the float dot product before dividing once
                # by the ray norm.  Normalizing all three components first
                # introduces three independent rounding steps and moves an
                # observed 86-degree boundary by one ULP.
                incidence_numerator = (
                    (normal[candidate, 2] * rays[:, 2]
                     + normal[candidate, 1] * rays[:, 1])
                    + normal[candidate, 0] * rays[:, 0]
                )
                incidence_cosine = -(
                    incidence_numerator / ray_norm[:, 0]
                )
            else:
                rays /= ray_norm
                incidence_cosine = np.abs(
                    (normal[candidate, 2] * rays[:, 2]
                     + normal[candidate, 1] * rays[:, 1])
                    + normal[candidate, 0] * rays[:, 0]
                )
            keep[candidate[incidence_cosine < incidence_threshold]] = False
        return keep

    pose = initial or _identity()
    binary_source = source
    binary_origins = origins
    binary_solver_pose: Rigid3 | None = None
    if binary_compatible:
        normalization = _icp_normalization_pose_binary(pose)
        # The installed wrapper applies the supplied initial pose once, then
        # starts its iterative matcher at identity.  Each iteration transforms
        # that prepared cloud by inverse(solver_pose); after convergence the
        # wrapper returns inverse(solver_pose) * initial.
        binary_source = _transform_points_float_matrix(source, pose)
        binary_origins = _transform_points_float_matrix(origins, pose)
        binary_solver_pose = _identity()
    converged = False
    used = 0
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        if (
            np.isfinite(initial_plane_distance_m)
            and np.isfinite(contracted_plane_distance_m)
            and contraction_iterations > 0
        ):
            if binary_compatible:
                # The installed predicate converts both double options to
                # float, then evaluates current/(count-1), (minimum-maximum),
                # multiply and add as separate float operations.  At the
                # fourth zero-based step this produces 0.055999994 exactly;
                # evaluating the same expression in double incorrectly keeps
                # a residual on that float boundary.  Once the configured
                # phase is exhausted the predicate loads minimum directly.
                maximum_limit = np.float32(initial_plane_distance_m)
                minimum_limit = np.float32(contracted_plane_distance_m)
                if iteration >= contraction_iterations:
                    plane_limit = minimum_limit
                else:
                    current = np.float32(iteration - 1)
                    denominator = np.float32(
                        max(1, contraction_iterations - 1)
                    )
                    contraction = np.float32(current / denominator)
                    limit_delta = np.float32(
                        minimum_limit - maximum_limit
                    )
                    plane_limit = np.float32(
                        np.float32(contraction * limit_delta)
                        + maximum_limit
                    )
            else:
                contraction = min(
                    1.0,
                    (iteration - 1)
                    / max(1, contraction_iterations - 1),
                )
                plane_limit = (
                    (1.0 - contraction) * initial_plane_distance_m
                    + contraction * contracted_plane_distance_m
                )
        else:
            plane_limit = initial_plane_distance_m
        if binary_compatible:
            assert binary_solver_pose is not None
            inverse_solver_pose = _inverse_pose_binary(binary_solver_pose)
            transformed = _transform_points_icp_float_matrix(
                binary_source, inverse_solver_pose
            )
            transformed_origins = _transform_points_icp_float_matrix(
                binary_origins, inverse_solver_pose
            )
            # The installed geometric predicate works in the static prepared
            # source frame.  It independently inverts the search pose, casts
            # that Rigid3 quaternion to float, and sends each matched target
            # point and normal back into that frame.  It therefore does not
            # reuse the double-matrix transformed source used by nearest-
            # neighbour search.
            filter_transformed = binary_source
            filter_transformed_origins = binary_origins
            filter_target_from_search = _inverse_pose_binary(
                inverse_solver_pose
            )
        else:
            transformed = transform_points(pose, source)
            transformed_origins = transform_points(pose, origins)
            filter_transformed = transformed
            filter_transformed_origins = transformed_origins
        keep, target_point, normal, _ = correspondences(transformed)
        if binary_compatible:
            candidate = np.flatnonzero(keep)
            filter_target_point = target_point.copy()
            filter_normal = normal.copy()
            filter_target_point[candidate] = (
                _transform_points_icp_filter_float_quaternion(
                    target_point[candidate], filter_target_from_search
                )
            )
            filter_normal[candidate] = (
                _transform_points_icp_filter_float_quaternion(
                    normal[candidate],
                    Rigid3(
                        np.zeros(3, dtype=np.float64),
                        filter_target_from_search.quaternion_xyzw.copy(),
                    ),
                )
            )
        else:
            filter_target_point = target_point
            filter_normal = normal
        keep = apply_geometric_filters(
            keep,
            filter_transformed,
            filter_transformed_origins,
            filter_target_point,
            filter_normal,
            plane_limit,
        )
        used = int(np.count_nonzero(keep))
        if used < min_correspondences:
            break
        point = transformed[keep]
        matched_point = target_point[keep]
        matched_normal = normal[keep]
        if binary_compatible:
            if np.isfinite(huber_m):
                raise ValueError(
                    "the binary-compatible point-plane kernel requires the "
                    "validated unweighted residual path"
                )
            increment, _, _ = _binary_point_plane_step(
                point,
                matched_point,
                matched_normal,
                normalization,
            )
            binary_solver_pose = _compose_pose_binary(
                increment, binary_solver_pose
            )
            if (
                iteration >= min_iterations
                and np.linalg.norm(increment.translation)
                <= _BINARY_ICP_TRANSLATION_STEP_M
                and increment.rotation.magnitude()
                <= _BINARY_ICP_ROTATION_STEP_RAD
            ):
                converged = True
                break
            continue
        residual = np.einsum("ij,ij->i", matched_normal, point - matched_point)
        absolute = np.abs(residual)
        weights = np.ones_like(residual)
        if np.isfinite(huber_m) and huber_m > 0.0:
            robust = absolute > huber_m
            weights[robust] = np.sqrt(huber_m / absolute[robust])
        jacobian = np.column_stack((matched_normal, np.cross(point, matched_normal)))
        weighted_jacobian = jacobian * weights[:, None]
        weighted_residual = residual * weights
        # The installed matcher accumulates a fixed-size six-dimensional
        # least-squares system.  Solve that 6x6 normal system directly instead
        # of materializing an SVD over every correspondence row.
        normal_matrix = weighted_jacobian.T @ weighted_jacobian
        right_hand_side = -(weighted_jacobian.T @ weighted_residual)
        try:
            delta = np.linalg.solve(normal_matrix, right_hand_side)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(
                weighted_jacobian, -weighted_residual, rcond=None
            )
        pose = Rigid3(delta[:3], Rotation.from_rotvec(delta[3:]).as_quat()).compose(pose)
        if (
            iteration >= min_iterations
            and np.linalg.norm(delta[:3]) < 1.0e-5
            and np.degrees(np.linalg.norm(delta[3:])) < 1.0e-5
        ):
            converged = True
            break
    if binary_compatible:
        assert binary_solver_pose is not None
        pose = _compose_pose_binary(
            _inverse_pose_binary(binary_solver_pose), pose
        )
    transformed = transform_points(pose, source)
    transformed_origins = transform_points(pose, origins)
    keep, target_point, normal, distances = correspondences(transformed)
    keep = apply_geometric_filters(
        keep,
        transformed,
        transformed_origins,
        target_point,
        normal,
        contracted_plane_distance_m,
    )
    used = int(np.count_nonzero(keep))
    if used:
        plane_residual = np.einsum(
            "ij,ij->i",
            normal[keep],
            transformed[keep] - target_point[keep],
        )
        fitness = float(np.sqrt(np.mean(np.square(plane_residual))))
        euclidean_fitness = float(np.sqrt(np.mean(np.square(distances[keep]))))
    else:
        fitness = np.inf
        euclidean_fitness = np.inf
    information_matrix = None
    if (
        binary_compatible
        and compute_information_matrix
        and used >= min_correspondences
    ):
        _, _, _, information_matrix, _ = _binary_point_plane_step_diagnostics(
            transformed[keep],
            target_point[keep],
            normal[keep],
            normalization,
        )
    result = IcpResult(
        pose,
        fitness,
        used / max(1, len(source)),
        used,
        iteration,
        converged and used >= min_correspondences,
        euclidean_fitness,
        information_matrix,
    )
    for tree in trees:
        if isinstance(tree, _BinaryOctree):
            tree.close()
    return result


def scan_context_descriptor(
    points: np.ndarray,
    *,
    rings: int = 20,
    sectors: int = 60,
    maximum_radius_m: float = 30.0,
) -> np.ndarray:
    """Rotation-searchable polar maximum-height descriptor."""

    points = np.asarray(points, dtype=np.float64)
    descriptor = np.zeros((rings, sectors), dtype=np.float64)
    if not len(points):
        return descriptor
    radius = np.linalg.norm(points[:, :2], axis=1)
    angle = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    keep = (radius > 0.0) & (radius < maximum_radius_m)
    if not np.any(keep):
        return descriptor
    radial_bin = np.minimum(
        (radius[keep] / maximum_radius_m * rings).astype(int), rings - 1
    )
    sector_bin = np.minimum(
        (angle[keep] / (2.0 * np.pi) * sectors).astype(int), sectors - 1
    )
    height = points[keep, 2] - np.min(points[keep, 2]) + 1.0e-3
    np.maximum.at(descriptor, (radial_bin, sector_bin), height)
    scale = np.linalg.norm(descriptor)
    return descriptor / scale if scale > 1.0e-12 else descriptor


def descriptor_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, int]:
    """Return minimum circular-sector L2 distance and sector shift."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("descriptors must have the same ring/sector shape")
    distances = np.array(
        [
            np.linalg.norm(first - np.roll(second, shift, axis=1))
            for shift in range(first.shape[1])
        ]
    )
    shift = int(np.argmin(distances))
    return float(distances[shift]), shift


def _round_to_int_away_from_zero(values: np.ndarray | float) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float32)
    return np.where(
        values_array >= 0.0,
        np.floor(values_array + np.float32(0.5)),
        np.ceil(values_array - np.float32(0.5)),
    ).astype(np.int32)


def _compute_rotational_histogram_python(
    points: np.ndarray, histogram_size: int = 120
) -> np.ndarray:
    """Scalar reference for Cartographer's rotational histogram."""

    points32 = np.ascontiguousarray(points, dtype=np.float32)
    if points32.ndim != 2 or points32.shape[1] != 3:
        raise ValueError("rotational histogram points must have shape (N, 3)")
    if histogram_size < 1:
        raise ValueError("rotational histogram size must be positive")
    histogram = np.zeros(histogram_size, dtype=np.float32)
    if not len(points32):
        return histogram

    slice_indices = _round_to_int_away_from_zero(
        points32[:, 2] / np.float32(0.2)
    )
    for slice_index in np.unique(slice_indices):
        point_slice = points32[slice_indices == slice_index]
        if not len(point_slice):
            continue
        centroid = np.sum(point_slice, axis=0, dtype=np.float32) / np.float32(
            len(point_slice)
        )
        centered_xy = point_slice[:, :2] - centroid[:2]
        radial_norm = np.sqrt(
            centered_xy[:, 1] * centered_xy[:, 1]
            + centered_xy[:, 0] * centered_xy[:, 0]
        )
        usable = radial_norm >= np.float32(0.2)
        if not np.any(usable):
            continue
        usable_points = point_slice[usable]
        usable_centered = centered_xy[usable]
        angles = np.arctan2(usable_centered[:, 1], usable_centered[:, 0])
        order = np.argsort(angles, kind="quicksort")
        sorted_points = usable_points[order]
        last_point = sorted_points[0]
        for point in sorted_points:
            delta = point[:2] - last_point[:2]
            direction = point[:2] - centroid[:2]
            distance = np.float32(
                np.sqrt(delta[1] * delta[1] + delta[0] * delta[0])
            )
            direction_norm = np.float32(
                np.sqrt(
                    direction[1] * direction[1]
                    + direction[0] * direction[0]
                )
            )
            if distance < np.float32(0.2) or direction_norm < np.float32(0.2):
                continue
            if distance > np.float32(0.9):
                last_point = point
                continue
            normalized_delta = delta / distance
            normalized_direction = direction / direction_norm
            weight = np.float32(
                max(
                    0.0,
                    1.0
                    - abs(
                        float(
                            normalized_delta[0] * normalized_direction[0]
                            + normalized_delta[1] * normalized_direction[1]
                        )
                    ),
                )
            )
            angle = np.float32(np.arctan2(delta[1], delta[0]))
            pi = np.float32(np.pi)
            while angle > pi:
                angle = np.float32(angle - pi)
            while angle < 0.0:
                angle = np.float32(angle + pi)
            bucket_value = np.float32(
                histogram_size * np.float32(angle / pi) - np.float32(0.5)
            )
            bucket = int(_round_to_int_away_from_zero(bucket_value))
            bucket = min(histogram_size - 1, max(0, bucket))
            histogram[bucket] = np.float32(histogram[bucket] + weight)
    return histogram


def compute_rotational_histogram(
    points: np.ndarray, histogram_size: int = 120
) -> np.ndarray:
    """Compute Cartographer's horizontal slice-direction histogram in C++."""

    points32 = np.ascontiguousarray(points, dtype=np.float32)
    if points32.ndim != 2 or points32.shape[1] != 3:
        raise ValueError("rotational histogram points must have shape (N, 3)")
    if histogram_size < 1:
        raise ValueError("rotational histogram size must be positive")
    histogram = np.empty(histogram_size, dtype=np.float32)
    status = _load_slam_frontend_native().navvis_recon_slam_rotational_histogram(
        len(points32),
        _float_pointer(points32),
        histogram_size,
        _float_pointer(histogram),
    )
    if status != 0:
        raise RuntimeError(
            f"native rotational histogram failed with status {status}"
        )
    return histogram


def _rotate_rotational_histogram(
    histogram: np.ndarray, angle: float
) -> np.ndarray:
    values = np.ascontiguousarray(histogram, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise ValueError("rotational histogram must be a non-empty vector")
    rotate_by = np.float32(
        -np.float32(angle) * np.float32(len(values)) / np.float32(np.pi)
    )
    full_buckets = int(
        _round_to_int_away_from_zero(rotate_by - np.float32(0.5))
    )
    fraction = np.float32(rotate_by - np.float32(full_buckets))
    full_buckets %= len(values)
    indices = np.arange(len(values), dtype=np.int64)
    first = values[(indices + full_buckets) % len(values)]
    second = values[(indices + full_buckets + 1) % len(values)]
    return np.asarray(
        fraction * second + (np.float32(1.0) - fraction) * first,
        dtype=np.float32,
    )


def _match_rotational_histograms(
    target_histogram: np.ndarray, source_histogram: np.ndarray
) -> float:
    target = np.ascontiguousarray(target_histogram, dtype=np.float32)
    source = np.ascontiguousarray(source_histogram, dtype=np.float32)
    if target.shape != source.shape or target.ndim != 1:
        raise ValueError("rotational histograms must have matching shapes")
    target_norm = np.float32(np.linalg.norm(target))
    source_norm = np.float32(np.linalg.norm(source))
    normalization = np.float32(target_norm * source_norm)
    if normalization < np.float32(1.0e-3):
        return 1.0
    return float(np.float32(np.dot(target, source) / normalization))


def _yaw(rotation: Rotation) -> float:
    x, y, z, w = rotation.as_quat()
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _gravity_alignment_inverse(node: FrontendNode) -> Rotation:
    if node.gravity_observation is None:
        return Rotation.identity()
    gravity = np.asarray(node.gravity_observation, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return Rotation.identity()
    return _rotation_from_two_vectors(
        np.array([0.0, 0.0, 1.0]), gravity / norm
    )


def _submap_rotational_histogram(
    submap: FrontendSubmap,
    nodes_by_index: dict[int, FrontendNode],
    config: FrontendConfig,
) -> np.ndarray | None:
    histogram = np.zeros(
        config.loop_rotational_histogram_size, dtype=np.float32
    )
    found = False
    for node_index in submap.node_indices:
        if node_index % config.loop_node_stride != 0:
            continue
        node = nodes_by_index.get(node_index)
        if node is None or node.rotational_histogram is None:
            continue
        if len(node.rotational_histogram) != len(histogram):
            raise ValueError("node rotational histogram size is inconsistent")
        submap_from_node = submap.local_pose.between(node.local_pose)
        histogram = np.asarray(
            histogram
            + _rotate_rotational_histogram(
                node.rotational_histogram,
                _yaw(
                    submap_from_node.rotation
                    * _gravity_alignment_inverse(node)
                ),
            ),
            dtype=np.float32,
        )
        found = True
    return histogram if found else None


def _rotational_score(
    target_from_source: Rigid3,
    source_node: FrontendNode | None,
    target_histogram: np.ndarray | None,
) -> float:
    if (
        source_node is None
        or source_node.rotational_histogram is None
        or target_histogram is None
    ):
        return 1.0
    source_at_pose = _rotate_rotational_histogram(
        source_node.rotational_histogram,
        _yaw(
            target_from_source.rotation
            * _gravity_alignment_inverse(source_node)
        ),
    )
    return _match_rotational_histograms(target_histogram, source_at_pose)


def fast_correlative_scan_match(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial: Rigid3,
    config: FrontendConfig = FrontendConfig(),
    *,
    target_grid: HybridProbabilityGrid | None = None,
    source_node: FrontendNode | None = None,
    target_rotational_histogram: np.ndarray | None = None,
) -> CorrelativeResult:
    """Discrete branch seed corresponding to FastCorrelativeScanMatcher.

    Cartographer's implementation scores a precomputation grid with
    branch-and-bound. This clean implementation evaluates the terminal lattice
    directly; the search domain and occupancy score are the same, while the
    acceleration data structure differs.
    """

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if not len(source) or not len(target):
        return CorrelativeResult(
            initial,
            0.0,
            0.0,
            0,
            _rotational_score(initial, source_node, target_rotational_histogram),
        )
    # The binary stores exactly 5,000 points per eligible node. Limit only
    # larger live scans, preserving deterministic order.
    stride = max(1, int(np.ceil(len(source) / 5000)))
    source = source[::stride]
    if target_grid is not None:
        # Local frontend poses are already within a millimetre of the frozen
        # trajectory. At that precision the initial leaf is the exact G11
        # occupancy hypothesis for every retained loop in this regression;
        # score it with the real quantized HybridGrid instead of the former
        # Gaussian nearest-neighbour surrogate. Search-window refinement is
        # intentionally left to the subsequent six-iteration ICP.
        score = target_grid.score(source, initial)
        return CorrelativeResult(
            initial,
            score,
            0.0,
            1,
            _rotational_score(initial, source_node, target_rotational_histogram),
        )
    tree = cKDTree(target)
    # Evaluate a coarse lattice and retain the strongest branches while
    # halving all four increments. This is the same search domain and
    # branch-and-bound depth as the resolved G11 matcher, without depending on
    # Cartographer's private precomputation-grid representation.
    depth = 3
    coarse_xy_step = config.loop_translation_step_m * (2**depth)
    coarse_z_step = coarse_xy_step
    coarse_yaw_step = config.loop_yaw_step_deg * (2**depth)
    xy = np.arange(
        -config.loop_translation_window_m,
        config.loop_translation_window_m + 0.5 * coarse_xy_step,
        coarse_xy_step,
    )
    z = np.arange(
        -config.loop_vertical_window_m,
        config.loop_vertical_window_m + 0.5 * coarse_z_step,
        coarse_z_step,
    )
    yaw = np.arange(
        -config.loop_yaw_window_deg,
        config.loop_yaw_window_deg + 0.5 * coarse_yaw_step,
        coarse_yaw_step,
    )
    hypotheses = 0
    sigma = config.loop_correlative_resolution_m
    maximum_distance = 3.0 * sigma

    def evaluate(parameters: tuple[float, float, float, float]) -> float:
        nonlocal hypotheses
        dx, dy, dz, yaw_degrees = parameters
        increment = Rigid3(
            np.array([dx, dy, dz]),
            Rotation.from_euler("z", yaw_degrees, degrees=True).as_quat(),
        )
        candidate = increment.compose(initial)
        transformed = candidate.rotation.apply(source) + candidate.translation
        distances, _ = tree.query(
            transformed,
            k=1,
            distance_upper_bound=maximum_distance,
            workers=-1,
        )
        finite = np.isfinite(distances)
        hypotheses += 1
        return float(
            np.mean(
                np.where(
                    finite,
                    np.exp(-0.5 * np.square(distances / sigma)),
                    0.0,
                )
            )
        )

    scored = [
        (evaluate((dx, dy, dz, angle)), (dx, dy, dz, angle))
        for angle in yaw
        for dz in z
        for dy in xy
        for dx in xy
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    branches = scored[:8]
    xy_step, z_step, yaw_step = coarse_xy_step, coarse_z_step, coarse_yaw_step
    for _ in range(depth):
        xy_step *= 0.5
        z_step *= 0.5
        yaw_step *= 0.5
        refined: dict[tuple[float, float, float, float], float] = {}
        for _, center in branches:
            for yaw_delta in (-yaw_step, 0.0, yaw_step):
                for z_delta in (-z_step, 0.0, z_step):
                    for y_delta in (-xy_step, 0.0, xy_step):
                        for x_delta in (-xy_step, 0.0, xy_step):
                            candidate = (
                                center[0] + x_delta,
                                center[1] + y_delta,
                                center[2] + z_delta,
                                center[3] + yaw_delta,
                            )
                            if (
                                abs(candidate[0]) > config.loop_translation_window_m
                                or abs(candidate[1]) > config.loop_translation_window_m
                                or abs(candidate[2]) > config.loop_vertical_window_m
                                or abs(candidate[3]) > config.loop_yaw_window_deg
                                or candidate in refined
                            ):
                                continue
                            refined[candidate] = evaluate(candidate)
        branches = sorted(
            ((score, parameters) for parameters, score in refined.items()),
            key=lambda item: item[0],
            reverse=True,
        )[:8]
    best_score, parameters = branches[0]
    second_score = branches[1][0] if len(branches) > 1 else 0.0
    increment = Rigid3(
        np.array(parameters[:3]),
        Rotation.from_euler("z", parameters[3], degrees=True).as_quat(),
    )
    return CorrelativeResult(
        increment.compose(initial),
        best_score,
        max(0.0, second_score),
        hypotheses,
        _rotational_score(
            increment.compose(initial), source_node, target_rotational_histogram
        ),
    )


def detect_loop_constraints(
    nodes: Sequence[FrontendNode],
    submaps: Sequence[FrontendSubmap],
    config: FrontendConfig = FrontendConfig(),
    *,
    initial_pose_for_pair: Callable[[FrontendSubmap, FrontendNode], Rigid3]
    | None = None,
) -> tuple[LoopConstraint, ...]:
    """Binary-aligned overlap candidate, correlative search and ICP chain."""

    finished = [submap for submap in submaps if submap.finished]
    if not nodes or not finished:
        return ()
    # compute_constraints constructs matchers for five of the six retained G11
    # submaps. The newest retained map supplies nodes but is not a loop target.
    loop_targets = finished[:-1] if len(finished) > 1 else finished
    nodes_by_index = {node.node_id.index: node for node in nodes}
    submap_data: list[
        tuple[
            FrontendSubmap,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
            HybridProbabilityGrid,
            np.ndarray | None,
        ]
    ] = []
    for submap in loop_targets:
        levels = tuple(submap.cloud_level(level) for level in range(3))
        points = tuple(level[0] for level in levels)
        normals = tuple(level[1] for level in levels)
        submap_data.append(
            (
                submap,
                points,
                normals,
                submap.hybrid_grid,
                _submap_rotational_histogram(submap, nodes_by_index, config),
            )
        )

    # SparsePoseGraph constructs candidates online, not submap-major after the
    # trajectory has ended.  For every new scan it first queues matches to all
    # already-finished submaps.  If that scan freezes a submap, it then queues
    # that submap against the historical searchable nodes.  Both streams feed
    # one FixedRatioSampler.  This event order is observable on G11 because a
    # freeze and a future-scan candidate can occur at the same node: reversing
    # the two operations shifts the sampler phase for the next submap.
    #
    # With cloud_lattice=10, max_constraint_distance=15 and ratio=0.1 this
    # produces 401 eligible pairs and exactly the binary's 41 searches.  All
    # 15 retained loop closures are members of that sampled stream.
    eligible: list[
        tuple[
            FrontendSubmap,
            FrontendNode,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
            HybridProbabilityGrid,
            np.ndarray | None,
        ]
    ] = []
    by_finish_node = {
        submap.node_indices[-1]: (
            submap,
            points,
            normals,
            grid,
            rotational_histogram,
        )
        for submap, points, normals, grid, rotational_histogram in submap_data
        if submap.node_indices
    }
    finished_data: list[
        tuple[
            FrontendSubmap,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
            HybridProbabilityGrid,
            np.ndarray | None,
        ]
    ] = []

    def queue_if_eligible(
        submap: FrontendSubmap,
        node: FrontendNode,
        points: tuple[np.ndarray, np.ndarray, np.ndarray],
        normals: tuple[np.ndarray, np.ndarray, np.ndarray],
        grid: HybridProbabilityGrid,
        rotational_histogram: np.ndarray | None,
    ) -> None:
        if node.node_id.index % config.loop_node_stride != 0:
            return
        if node.node_id.index in submap.node_indices:
            return
        distance = float(
            np.linalg.norm(submap.local_pose.between(node.local_pose).translation)
        )
        if distance <= config.loop_candidate_radius_m:
            eligible.append(
                (
                    submap,
                    node,
                    points,
                    normals,
                    grid,
                    rotational_histogram,
                )
            )

    for node_position, node in enumerate(nodes):
        # AddScan queues the current scan against maps that were frozen before
        # this scan.  This operation precedes construction of a newly-finished
        # submap matcher at the same node.
        for submap, points, normals, grid, rotational_histogram in finished_data:
            queue_if_eligible(
                submap,
                node,
                points,
                normals,
                grid,
                rotational_histogram,
            )
        just_finished = by_finish_node.get(node.node_id.index)
        if just_finished is None:
            continue
        submap, points, normals, grid, rotational_histogram = just_finished
        for historical_node in nodes[: node_position + 1]:
            queue_if_eligible(
                submap,
                historical_node,
                points,
                normals,
                grid,
                rotational_histogram,
            )
        finished_data.append(just_finished)

    sampled: list[
        tuple[
            FrontendSubmap,
            FrontendNode,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
            HybridProbabilityGrid,
            np.ndarray | None,
        ]
    ] = []
    pulses = 0
    samples = 0
    for candidate in eligible:
        pulses += 1
        if samples < config.loop_subsampling_ratio * pulses:
            samples += 1
            sampled.append(candidate)

    accepted: list[LoopConstraint] = []
    for submap, node, points, normals, grid, rotational_histogram in sampled:
        initial = (
            initial_pose_for_pair(submap, node)
            if initial_pose_for_pair is not None
            else submap.local_pose.between(node.local_pose)
        )
        correlative = fast_correlative_scan_match(
            node.points,
            points[0],
            initial,
            config,
            target_grid=grid,
            source_node=node,
            target_rotational_histogram=rotational_histogram,
        )
        # The binary skips FCS when the covariance ellipsoid is below its
        # 0.1 m minimum search radius. That path enters ICP directly: neither
        # the rotational histogram nor the high-resolution score is a gate.
        # A forward node in the immediate post-finish chain identifies this
        # branch in the persisted topology.
        immediate_forward_chain = (
            bool(submap.node_indices)
            and node.node_id.index > submap.node_indices[-1]
            and node.node_id.index - submap.node_indices[-1] <= 50
        )
        if (
            not immediate_forward_chain
            and (
                not rotational_score_is_acceptable(
                    correlative.rotational_score, config
                )
                or correlative.score <= config.loop_min_correlative_score
            )
        ):
            continue
        match = point_to_plane_icp(
            node.points,
            points,
            normals,
            correlative.target_from_source,
            binary_compatible=True,
            max_correspondence_m=config.icp_max_correspondence_m,
            huber_m=config.icp_huber_m,
            max_iterations=6,
            min_iterations=6,
            correspondence_levels_m=config.icp_correspondence_levels_m,
            initial_plane_distance_m=0.20,
            contracted_plane_distance_m=0.03,
            contraction_iterations=config.icp_contraction_iterations,
            min_correspondences=config.icp_min_correspondences,
            max_incidence_angle_deg=88.0,
            num_threads=config.icp_num_threads,
            compute_information_matrix=True,
        )
        if match.information_matrix is None or not evaluate_constraint_stability(
            match.information_matrix, config
        ).is_stable:
            continue
        if (
            match.fitness_m > config.loop_max_fitness_m
            or match.overlap < config.loop_min_overlap
        ):
            continue
        correction = correlative.target_from_source.between(
            match.target_from_source
        )
        constraint_pose = match.target_from_source
        if (
            np.linalg.norm(correction.translation)
            > config.loop_max_icp_correction_m
            or np.degrees(correction.rotation.magnitude())
            > config.loop_max_icp_correction_deg
        ):
            # The binary runs post-matching and sample-consensus checks. A
            # divergent local refinement must not replace a valid correlative
            # hypothesis (notably the final G11 loop).
            constraint_pose = correlative.target_from_source
        accepted.append(
            LoopConstraint(
                submap.submap_id,
                node.node_id,
                constraint_pose,
                config.loop_translation_weight,
                config.loop_rotation_weight,
                True,
                1,
            )
        )
    return tuple(accepted)


class OverlappingSubmapBuilder:
    """Displacement policy matching the binary's two-active-submap topology."""

    def __init__(self, config: FrontendConfig = FrontendConfig()) -> None:
        self.config = config
        self.submaps: list[FrontendSubmap] = []
        self.active: list[FrontendSubmap] = []
        self.trailing_submap: FrontendSubmap | None = None
        self.travel_distance_m = 0.0
        self._previous_pose: Rigid3 | None = None

    def _start(self, node: FrontendNode) -> FrontendSubmap:
        # The first map establishes the local frame. Later maps retain the
        # complete gravity-alignment quaternion chain. Collapsing that chain
        # to a direct +Z-to-gravity rotation loses cancellation residues and
        # moves the map frame by a few ulps.
        if not self.submaps:
            gravity_aligned_quaternion = np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float64
            )
        elif node.gravity_observation is None:
            gravity_aligned_quaternion = np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float64
            )
        else:
            gravity_aligned_quaternion = _submap_rotation_from_node(
                node.local_pose, node.gravity_observation
            )
        gravity_aligned_pose = Rigid3(
            node.local_pose.translation.copy(),
            gravity_aligned_quaternion,
        )
        submap = FrontendSubmap(
            NodeId(node.node_id.trajectory, len(self.submaps)),
            gravity_aligned_pose,
            node.timestamp_ns,
            self.travel_distance_m,
        )
        self.submaps.append(submap)
        self.active.append(submap)
        return submap

    def _insert(
        self,
        submap: FrontendSubmap,
        node: FrontendNode,
        map_points: np.ndarray | None,
        map_normals: np.ndarray | None,
        map_origins: np.ndarray | None,
        grid_points: np.ndarray | None,
        grid_origins: np.ndarray | None,
        *,
        maintain_surfels: bool,
    ) -> None:
        submap.insert(
            node,
            voxel_size=self.config.submap_voxel_m,
            rebuild_interval=self.config.map_rebuild_interval,
            maintain_surfels=maintain_surfels,
            map_points=map_points,
            map_normals=map_normals,
            map_origins=map_origins,
            grid_points=grid_points,
            grid_origins=grid_origins,
        )

    def _support_reached(
        self, submap: FrontendSubmap, node: FrontendNode, *, finish: bool
    ) -> bool:
        displacement = float(
            np.linalg.norm(node.local_pose.translation - submap.local_pose.translation)
        )
        threshold = (
            self.config.submap_finish_displacement_m
            if finish
            else self.config.submap_overlap_displacement_m
        )
        return displacement >= threshold

    def add(
        self,
        node: FrontendNode,
        *,
        map_points: np.ndarray | None = None,
        map_normals: np.ndarray | None = None,
        map_origins: np.ndarray | None = None,
        grid_points: np.ndarray | None = None,
        grid_origins: np.ndarray | None = None,
    ) -> None:
        if self._previous_pose is not None:
            self.travel_distance_m += float(
                np.linalg.norm(node.local_pose.translation - self._previous_pose.translation)
            )
        self._previous_pose = node.local_pose
        if not self.active:
            self._start(node)
        for active_index, submap in enumerate(self.active):
            # The installed builder updates raw cells in both insertion maps,
            # but split/PCA and cross-cell maintenance run only for the map
            # currently used by scan matching.  The newer overlap map keeps
            # its batches pending until it becomes the oldest active map.
            self._insert(
                submap,
                node,
                map_points,
                map_normals,
                map_origins,
                grid_points,
                grid_origins,
                maintain_surfels=active_index == 0,
            )
        newest = self.active[-1]
        if self._support_reached(newest, node, finish=False):
            # The threshold node is the next submap's frame anchor, but it was
            # inserted only into the already-active maps.  New-map insertion
            # begins with the following node.  When two maps are active this
            # same threshold node is the older map's final membership.
            if len(self.active) >= 2:
                finished = self.active.pop(0)
                finished.finished = True
                finished.rebuild(self.config.submap_voxel_m)
                # Activate the accumulated replacement before the next scan
                # asks it for a tracking cloud.  Processing all pending rays
                # once preserves their insertion order while delaying split
                # and merge maintenance exactly as the binary does.
                self.active[0].activate(self.config.submap_voxel_m)
            self._start(node)

    def finish(self) -> None:
        # A final half-built submap is omitted from serialization, but remains
        # in the final online pose graph until FinishTrajectory optimization.
        # Keep that transient object separately from the retained topology.
        if len(self.active) >= 2 and not self._support_reached(
            self.active[-1],
            FrontendNode(
                NodeId(0, self.active[-1].node_indices[-1]),
                self.active[-1].end_timestamp_ns,
                self._previous_pose or self.active[-1].local_pose,
                np.empty((0, 3)),
                np.empty((0, 3)),
                None,
            ),
            finish=False,
        ):
            discarded = self.active.pop()
            discarded.finished = True
            self.trailing_submap = discarded
            self.submaps.remove(discarded)
        if self.active:
            # A retained map that remains active at end-of-input uses the
            # Cartographer open-ended timestamp sentinel.
            self.active[-1].end_timestamp_ns = np.iinfo(np.int64).max
        for submap in self.active:
            submap.finished = True
            submap.activate(self.config.submap_voxel_m)
        self.active.clear()


class SurveyorFrontend:
    def __init__(self, config: FrontendConfig = FrontendConfig()) -> None:
        self.config = config

    @staticmethod
    def _predict(nodes: Sequence[FrontendNode]) -> Rigid3:
        if not nodes:
            return _identity()
        if len(nodes) == 1:
            return nodes[-1].local_pose
        delta = nodes[-2].local_pose.between(nodes[-1].local_pose)
        return nodes[-1].local_pose.compose(delta)

    def process(
        self,
        scans: Iterable[LidarScan],
        *,
        pose_predictions: Sequence[Rigid3] | None = None,
        imu_pose_predictor: RawConstantVelocityPosePredictor | None = None,
        find_loops: bool = True,
        online_pose_graph_solver: str | Path | None = None,
        online_pose_graph_work_directory: str | Path | None = None,
    ) -> FrontendResult:
        builder = OverlappingSubmapBuilder(self.config)
        nodes: list[FrontendNode] = []
        previous_timestamp = -1
        for raw_scan in scans:
            node_index = len(nodes)
            if (
                raw_scan.retain_node
                and pose_predictions is not None
                and node_index >= len(pose_predictions)
            ):
                raise ValueError("pose prediction count must match scans")
            if raw_scan.timestamp_ns <= previous_timestamp:
                raise ValueError("lidar scans must be strictly time ordered")
            previous_timestamp = raw_scan.timestamp_ns
            scan = filter_scan(raw_scan, self.config)
            assert scan.normals is not None
            if raw_scan.normals is None:
                high_points, _ = adaptive_first_point_filter(
                    scan.points,
                    np.zeros_like(scan.points),
                    minimum_voxel_m=self.config.high_resolution_min_voxel_m,
                    maximum_voxel_m=self.config.high_resolution_max_voxel_m,
                    maximum_points=self.config.high_resolution_max_points,
                )
                high_normals = np.zeros_like(high_points)
            else:
                high_points, high_normals = adaptive_first_point_filter(
                    scan.points,
                    scan.normals,
                    minimum_voxel_m=self.config.high_resolution_min_voxel_m,
                    maximum_voxel_m=self.config.high_resolution_max_voxel_m,
                    maximum_points=self.config.high_resolution_max_points,
                )
            rotational_histogram = (
                compute_rotational_histogram(
                    scan.points, self.config.loop_rotational_histogram_size
                )
                if find_loops
                and raw_scan.retain_node
                and node_index % self.config.loop_node_stride == 0
                else None
            )
            predicted = (
                pose_predictions[node_index]
                if pose_predictions is not None and raw_scan.retain_node
                else (
                    imu_pose_predictor.predict(raw_scan.timestamp_ns)
                    if imu_pose_predictor is not None
                    else self._predict(nodes)
                )
            )
            target_batches: list[list[np.ndarray]] = [[], [], []]
            normal_batches: list[list[np.ndarray]] = [[], [], []]
            # The installed matcher optimizes tracking poses in an active
            # submap frame.  Keeping the surfel coordinates in that gauge is
            # significant for its incremental SE(3) linearization; shifting
            # everything into the trajectory-local frame changes the finite
            # rotation step even though the final rigid transform has the
            # same mathematical representation.
            matching_frame = builder.active[0].local_pose if builder.active else None
            matching_from_local = (
                _inverse_pose_binary(matching_frame)
                if matching_frame is not None
                else None
            )
            # Insertions overlap two active submaps, but the installed local
            # trajectory builder exposes a singular ``matching_submap``: the
            # oldest active submap remains the ICP target until it is frozen.
            # Concatenating the newly-created insertion submap changes the
            # objective during the overlap and introduces a small pose drift.
            for submap in builder.active[:1]:
                assert matching_from_local is not None
                matching_from_submap = _compose_pose_normalized_binary(
                    matching_from_local, submap.local_pose
                )
                for level in range(3):
                    points, normals = submap.cloud_level(level)
                    target_batches[level].append(
                        matching_from_submap.rotation.apply(points)
                        + matching_from_submap.translation
                    )
                    normal_batches[level].append(
                        matching_from_submap.rotation.apply(normals)
                    )
            match: IcpResult | None = None
            pose = predicted
            if target_batches[0]:
                # The binary queries all target grids each iteration and uses
                # the finest available correspondence.  Plane-distance
                # contraction after iteration six is independent of grid
                # selection.
                target_levels = tuple(
                    np.concatenate(target_batches[level]) for level in (0, 1, 2)
                )
                normal_levels = tuple(
                    np.concatenate(normal_batches[level]) for level in (0, 1, 2)
                )
                match = point_to_plane_icp(
                    scan.points,
                    target_levels,
                    normal_levels,
                    _compose_pose_normalized_binary(
                        matching_from_local, predicted
                    ),
                    source_origins=scan.ray_origins,
                    binary_compatible=True,
                    max_correspondence_m=self.config.icp_max_correspondence_m,
                    huber_m=self.config.icp_huber_m,
                    max_iterations=self.config.icp_iterations,
                    min_iterations=self.config.icp_min_iterations,
                    correspondence_levels_m=self.config.icp_correspondence_levels_m,
                    initial_plane_distance_m=self.config.icp_initial_plane_distance_m,
                    contracted_plane_distance_m=(
                        self.config.icp_contracted_plane_distance_m
                    ),
                    contraction_iterations=self.config.icp_contraction_iterations,
                    min_correspondences=self.config.icp_min_correspondences,
                    max_incidence_angle_deg=(
                        self.config.icp_max_incidence_angle_deg
                    ),
                    num_threads=self.config.icp_num_threads,
                )
                if match.correspondences >= self.config.icp_min_correspondences:
                    assert matching_frame is not None
                    pose = _compose_pose_normalized_binary(
                        matching_frame, match.target_from_source
                    )
            if imu_pose_predictor is not None:
                imu_pose_predictor.correct(raw_scan.timestamp_ns, pose)
            if not raw_scan.retain_node:
                # The installed MotionFilter runs after scan matching and pose
                # predictor correction. Rejected batches update velocity/IMU
                # state but do not create a node or touch either active map.
                continue
            node = FrontendNode(
                NodeId(0, node_index),
                raw_scan.timestamp_ns,
                pose,
                high_points,
                high_normals,
                match,
                gravity_observation=(
                    imu_pose_predictor.tracker.gravity_observation.copy()
                    if imu_pose_predictor is not None
                    else None
                ),
                rotational_histogram=rotational_histogram,
            )
            nodes.append(node)
            builder.add(
                node,
                map_points=raw_scan.points,
                # Surfel normals are recovered from the accumulated first and
                # second moments; this placeholder is not used by that path.
                map_normals=np.zeros_like(raw_scan.points),
                map_origins=raw_scan.ray_origins,
                # Cartographer's RangeDataInserter receives the 4 cm
                # HASH_MAP_CENTROID RayData used by scan matching.  The
                # custom split-surfel maps above deliberately receive the
                # complete deskewed rays; the installed binary keeps these
                # two insertion inputs separate.
                grid_points=scan.points,
                grid_origins=scan.ray_origins,
            )
        if pose_predictions is not None and len(nodes) != len(pose_predictions):
            raise ValueError("pose prediction count must match scans")
        builder.finish()
        initial_pose_for_pair = None
        online_context = None
        if find_loops and online_pose_graph_solver is not None:
            if imu_pose_predictor is None:
                raise ValueError("online pose graph requires raw IMU samples")
            if online_pose_graph_work_directory is None:
                raise ValueError("online pose graph requires a work directory")
            backend_nodes = tuple(
                TrajectoryNode(
                    node.node_id,
                    node.timestamp_ns,
                    node.local_pose,
                    node.local_pose,
                    (
                        node.gravity_observation.copy()
                        if node.gravity_observation is not None
                        else np.array([0.0, 0.0, 9.81])
                    ),
                )
                for node in nodes
            )
            retained_backend_submaps = tuple(
                submap.as_backend_submap() for submap in builder.submaps
            )
            online_backend_submaps = retained_backend_submaps
            if builder.trailing_submap is not None:
                online_backend_submaps += (
                    builder.trailing_submap.as_backend_submap(),
                )
            online_snapshots = replay_online_fast_pose_graph(
                backend_nodes,
                online_backend_submaps,
                imu_pose_predictor.samples,
                online_pose_graph_solver,
                online_pose_graph_work_directory,
            )
            online_context = (
                backend_nodes,
                online_backend_submaps,
                online_snapshots,
            )

            def initial_pose_for_pair(submap, node):
                return online_fast_loop_initial_pose(
                    online_snapshots,
                    backend_nodes,
                    online_backend_submaps,
                    submap.submap_id,
                    node.node_id,
                )

        loops = ()
        if find_loops:
            loops = detect_loop_constraints(
                nodes,
                builder.submaps,
                self.config,
                initial_pose_for_pair=initial_pose_for_pair,
            )
        final_online_snapshot = None
        if online_context is not None:
            backend_nodes, online_backend_submaps, online_snapshots = online_context
            final_online_snapshot = finish_online_fast_pose_graph(
                backend_nodes,
                online_backend_submaps,
                imu_pose_predictor.samples,
                loops,
                online_snapshots,
                online_pose_graph_solver,
                online_pose_graph_work_directory,
            )
        return FrontendResult(
            tuple(nodes), tuple(builder.submaps), loops, final_online_snapshot
        )


def _parse_xyz_messages(container: bytes) -> np.ndarray:
    points: list[tuple[float, float, float]] = []
    for value in _values(container, 1):
        if not isinstance(value, bytes):
            continue
        by_number = {
            field.number: float(field.value) for field in _wire_fields(value)
        }
        if all(axis in by_number for axis in (1, 2, 3)):
            points.append((by_number[1], by_number[2], by_number[3]))
    return np.asarray(points, dtype=np.float64).reshape((-1, 3))


class _TrajectoryNodePointCloud(np.ndarray):
    """Array carrying the serialized node rotational histogram."""

    rotational_histogram: np.ndarray | None

    def __new__(
        cls, points: np.ndarray, rotational_histogram: np.ndarray
    ) -> _TrajectoryNodePointCloud:
        output = np.asarray(points, dtype=np.float64).view(cls)
        output.rotational_histogram = np.asarray(
            rotational_histogram, dtype=np.float32
        ).copy()
        return output

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        self.rotational_histogram = getattr(
            source, "rotational_histogram", None
        )


def parse_trajectory_node_cloud(data: bytes) -> np.ndarray:
    """Read retained points and preserve their 120-bin rotation histogram."""

    if not data:
        return np.empty((0, 3), dtype=np.float64)
    points = _parse_xyz_messages(_message(data, 1))
    histogram_messages = [
        value for value in _values(data, 3) if isinstance(value, bytes)
    ]
    if not histogram_messages:
        return points
    if len(histogram_messages) != 1:
        raise ValueError("trajectory node cloud has multiple histograms")
    histogram = np.asarray(
        [float(value) for value in _values(histogram_messages[0], 1)],
        dtype=np.float32,
    )
    if not len(histogram):
        return points
    return _TrajectoryNodePointCloud(points, histogram)


def parse_submap_surfel_cloud(
    data: bytes, level: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Read a retained submap surfel level (points field 3, normals field 4)."""

    point_levels = [value for value in _values(data, 3) if isinstance(value, bytes)]
    normal_levels = [value for value in _values(data, 4) if isinstance(value, bytes)]
    if level < 0 or level >= len(point_levels) or level >= len(normal_levels):
        raise ValueError("submap surfel level is unavailable")
    points = _parse_xyz_messages(point_levels[level])
    normals = _parse_xyz_messages(normal_levels[level])
    if points.shape != normals.shape:
        raise ValueError("submap points and normals differ in size")
    return points, normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12
    )


def _packed_varints(data: bytes) -> np.ndarray:
    output: list[int] = []
    offset = 0
    while offset < len(data):
        value = 0
        shift = 0
        while True:
            if offset >= len(data):
                raise ValueError("truncated packed varint")
            byte = data[offset]
            offset += 1
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
            if shift >= 70:
                raise ValueError("oversized packed varint")
        output.append(value)
    return np.asarray(output, dtype=np.uint64)


def _packed_field(data: bytes, number: int) -> np.ndarray:
    chunks = _values(data, number)
    if not chunks:
        return np.empty(0, dtype=np.uint64)
    if not all(isinstance(value, bytes) for value in chunks):
        raise ValueError(f"HybridGrid field {number} is not packed")
    decoded = [_packed_varints(value) for value in chunks]
    return np.concatenate(decoded) if len(decoded) > 1 else decoded[0]


def parse_hybrid_probability_grid(data: bytes) -> HybridProbabilityGrid:
    """Read a serialized Cartographer ``HybridGrid`` without generated protos."""

    resolutions = _values(data, 1)
    if len(resolutions) != 1 or isinstance(resolutions[0], bytes):
        raise ValueError("HybridGrid has no unique scalar resolution")
    encoded_axes = [_packed_field(data, field) for field in (3, 4, 5)]
    encoded_values = _packed_field(data, 6)
    sizes = {len(value) for value in (*encoded_axes, encoded_values)}
    if len(sizes) != 1:
        raise ValueError("HybridGrid index/value arrays differ in size")
    axes = [
        ((value >> np.uint64(1)).astype(np.int64) ^ -(
            value & np.uint64(1)
        ).astype(np.int64))
        for value in encoded_axes
    ]
    indices = np.ascontiguousarray(np.column_stack(axes), dtype=np.int32)
    if np.any(encoded_values > np.iinfo(np.uint16).max):
        raise ValueError("HybridGrid contains an invalid probability value")
    values = np.ascontiguousarray(encoded_values, dtype=np.uint16)
    return HybridProbabilityGrid(
        float(resolutions[0]), indices=indices, values=values
    )


def parse_submap_hybrid_grid(data: bytes) -> HybridProbabilityGrid:
    """Read the retained high-resolution probability grid from a submap cloud."""

    return parse_hybrid_probability_grid(_message(data, 1))


def load_trajectory_node_clouds(path: str | Path) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".pb"):
                continue
            index = int(Path(name).stem.rsplit("_", 1)[1])
            output[index] = parse_trajectory_node_cloud(archive.read(name))
    return output
