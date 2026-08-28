"""Trajectory-side reconstruction of the SurveyorSLAM output chain.

This module does not pretend that an existing trajectory is raw-lidar SLAM.
It reconstructs the part that can be evaluated independently from the
recording: combine the online global SLAM track (``map <- base``) with the
higher-coverage local track (``odom <- base``), interpolate the slowly varying
``map <- odom`` correction, and upsample every graph interval by the factor of
five used by ``compute_trajectories --trajectory-upsampling=5``.

Offline scan matching, submap construction and loop-closure search now live in
``surveyor_frontend``; the recovered node/submap/IMU graph is implemented in
``surveyor_slam``.  This file remains the recorded-track adapter and evaluator,
keeping that fallback measurable instead of silently calling local odometry
"SLAM".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True, slots=True)
class Trajectory:
    timestamps: np.ndarray
    translations: np.ndarray
    quaternions_xyzw: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        translations = np.asarray(self.translations, dtype=np.float64)
        quaternions = np.asarray(self.quaternions_xyzw, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("trajectory needs at least two timestamps")
        if translations.shape != (len(timestamps), 3):
            raise ValueError("translations must have shape (N, 3)")
        if quaternions.shape != (len(timestamps), 4):
            raise ValueError("quaternions must have shape (N, 4)")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("trajectory timestamps must be strictly increasing")
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(norms < 1.0e-12):
            raise ValueError("trajectory contains a zero quaternion")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "quaternions_xyzw", quaternions / norms[:, None])


def interpolate(trajectory: Trajectory, timestamps: np.ndarray) -> Trajectory:
    """Interpolate a trajectory, holding its endpoint poses outside its span."""

    requested = np.asarray(timestamps, dtype=np.float64)
    if requested.ndim != 1 or len(requested) < 2 or np.any(np.diff(requested) <= 0):
        raise ValueError("requested timestamps must be a strictly increasing vector")
    clipped = np.clip(requested, trajectory.timestamps[0], trajectory.timestamps[-1])
    translations = np.column_stack(
        [
            np.interp(clipped, trajectory.timestamps, trajectory.translations[:, axis])
            for axis in range(3)
        ]
    )
    rotations = Slerp(
        trajectory.timestamps, Rotation.from_quat(trajectory.quaternions_xyzw)
    )(clipped)
    return Trajectory(requested, translations, rotations.as_quat())


def upsample_timestamps(timestamps: np.ndarray, factor: int = 5) -> np.ndarray:
    """Subdivide each support-pose interval exactly ``factor`` times."""

    source = np.asarray(timestamps, dtype=np.float64)
    if factor < 1:
        raise ValueError("upsampling factor must be positive")
    if source.ndim != 1 or len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    fractions = np.arange(factor, dtype=np.float64) / factor
    intervals = source[:-1, None] + np.diff(source)[:, None] * fractions[None, :]
    return np.concatenate((intervals.reshape(-1), source[-1:]))


def fuse_global_and_local(
    global_slam: Trajectory,
    local_odometry: Trajectory,
    *,
    target_timestamps: np.ndarray | None = None,
    upsampling_factor: int = 5,
) -> Trajectory:
    """Apply interpolated ``map <- odom`` corrections to local poses.

    At every online SLAM support timestamp we compute

    ``map_from_odom = map_from_base * inverse(odom_from_base)``.

    The correction is interpolated independently from the local motion.  This
    preserves the online SLAM poses at their support points while extending the
    global frame to the local trajectory's leading/trailing support and avoids
    treating local odometry as a map-frame trajectory.
    """

    if target_timestamps is None:
        target_timestamps = upsample_timestamps(
            local_odometry.timestamps, upsampling_factor
        )
    target_timestamps = np.asarray(target_timestamps, dtype=np.float64)

    local_at_global = interpolate(local_odometry, global_slam.timestamps)
    global_rotation = Rotation.from_quat(global_slam.quaternions_xyzw)
    local_rotation = Rotation.from_quat(local_at_global.quaternions_xyzw)
    correction_rotation = global_rotation * local_rotation.inv()
    correction_translation = (
        global_slam.translations
        - correction_rotation.apply(local_at_global.translations)
    )
    correction = Trajectory(
        global_slam.timestamps,
        correction_translation,
        correction_rotation.as_quat(),
    )

    local_target = interpolate(local_odometry, target_timestamps)
    correction_target = interpolate(correction, target_timestamps)
    correction_target_rotation = Rotation.from_quat(
        correction_target.quaternions_xyzw
    )
    output_rotation = correction_target_rotation * Rotation.from_quat(
        local_target.quaternions_xyzw
    )
    output_translation = (
        correction_target_rotation.apply(local_target.translations)
        + correction_target.translations
    )
    return Trajectory(
        target_timestamps, output_translation, output_rotation.as_quat()
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def evaluate_relative_trajectory(
    estimated: Trajectory,
    reference: Trajectory,
    *,
    delta_seconds: float = 1.0,
) -> dict[str, object]:
    """Evaluate gauge-invariant relative-pose drift over a fixed interval.

    Absolute trajectory error mixes the global gauge with local SLAM shape.
    The vendor pose graph is anchored independently, so this companion metric
    compares ``pose(t)^-1 * pose(t + delta)`` and exposes scan-matching/IMU
    drift without rewarding a post-hoc rigid alignment.
    """

    if not np.isfinite(delta_seconds) or delta_seconds <= 0.0:
        raise ValueError("delta_seconds must be finite and positive")
    overlap_start = max(estimated.timestamps[0], reference.timestamps[0])
    overlap_end = min(estimated.timestamps[-1], reference.timestamps[-1])
    starts = estimated.timestamps[
        (estimated.timestamps >= overlap_start)
        & (estimated.timestamps + delta_seconds <= overlap_end)
    ]
    if len(starts) < 2:
        raise ValueError("trajectories have no usable relative-pose intervals")
    ends = starts + delta_seconds

    estimated_start = interpolate(estimated, starts)
    estimated_end = interpolate(estimated, ends)
    reference_start = interpolate(reference, starts)
    reference_end = interpolate(reference, ends)

    estimated_start_rotation = Rotation.from_quat(
        estimated_start.quaternions_xyzw
    )
    reference_start_rotation = Rotation.from_quat(
        reference_start.quaternions_xyzw
    )
    estimated_relative_rotation = estimated_start_rotation.inv() * Rotation.from_quat(
        estimated_end.quaternions_xyzw
    )
    reference_relative_rotation = reference_start_rotation.inv() * Rotation.from_quat(
        reference_end.quaternions_xyzw
    )
    estimated_relative_translation = estimated_start_rotation.inv().apply(
        estimated_end.translations - estimated_start.translations
    )
    reference_relative_translation = reference_start_rotation.inv().apply(
        reference_end.translations - reference_start.translations
    )
    translation_error = np.linalg.norm(
        estimated_relative_translation - reference_relative_translation, axis=1
    )
    rotation_error = np.degrees(
        (estimated_relative_rotation.inv() * reference_relative_rotation).magnitude()
    )
    return {
        "delta_seconds": float(delta_seconds),
        "sample_count": int(len(starts)),
        "translation_error_m": _summary(translation_error),
        "rotation_error_deg": _summary(rotation_error),
    }


def evaluate_trajectory(
    estimated: Trajectory, reference: Trajectory
) -> dict[str, object]:
    """Compare positions and orientations at estimated support timestamps."""

    mask = (estimated.timestamps >= reference.timestamps[0]) & (
        estimated.timestamps <= reference.timestamps[-1]
    )
    timestamps = estimated.timestamps[mask]
    if len(timestamps) < 2:
        raise ValueError("trajectories have no usable time overlap")
    estimate_overlap = Trajectory(
        timestamps,
        estimated.translations[mask],
        estimated.quaternions_xyzw[mask],
    )
    reference_overlap = interpolate(reference, timestamps)
    position_error = np.linalg.norm(
        estimate_overlap.translations - reference_overlap.translations, axis=1
    )
    rotation_error = np.degrees(
        (
            Rotation.from_quat(estimate_overlap.quaternions_xyzw).inv()
            * Rotation.from_quat(reference_overlap.quaternions_xyzw)
        ).magnitude()
    )
    return {
        "sample_count": int(len(timestamps)),
        "start": float(timestamps[0]),
        "end": float(timestamps[-1]),
        "position_error_m": _summary(position_error),
        "rotation_error_deg": _summary(rotation_error),
        "relative_pose_error": evaluate_relative_trajectory(
            estimate_overlap, reference_overlap, delta_seconds=1.0
        ),
    }
