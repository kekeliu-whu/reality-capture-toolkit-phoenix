"""Serialization helpers for generated clean-room SLAM states."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from navvis_recon.surveyor_slam import OnlineFastPoseGraphSnapshot, Submap


ONLINE_STAGE1_ARRAY_NAMES = frozenset(
    {
        "online_node_translations",
        "online_node_quaternions_xyzw",
        "online_submap_translations",
        "online_submap_quaternions_xyzw",
        "online_gravity_magnitude",
        "online_imu_from_tracking_translation",
        "online_imu_from_tracking_quaternion_xyzw",
        "online_linear_acceleration_bias",
        "online_linear_acceleration_scaling",
        "online_linear_acceleration_cross_axis",
        "online_angular_velocity_bias",
        "online_angular_velocity_scaling",
        "online_angular_velocity_cross_axis",
    }
)


def online_stage1_arrays(
    snapshot: OnlineFastPoseGraphSnapshot,
    retained_submaps: Sequence[Submap],
) -> dict[str, np.ndarray]:
    """Encode a finished online Fast-IMU solution for Stage2 initialization.

    ``FinishTrajectory`` optimizes the retained Submaps and one possible
    trailing in-memory Submap.  Only retained Submaps are serialized by the
    frontend, so select their optimized poses by ID rather than truncating the
    pose array implicitly.
    """

    node_count = snapshot.node_count
    expected_pose_count = node_count + len(snapshot.submaps)
    if len(snapshot.result.poses) != expected_pose_count:
        raise ValueError(
            "online Stage1 pose count differs from its node/Submap topology"
        )

    submap_pose_by_id = {
        submap.submap_id: snapshot.result.poses[node_count + index]
        for index, submap in enumerate(snapshot.submaps)
    }
    if len(submap_pose_by_id) != len(snapshot.submaps):
        raise ValueError("online Stage1 contains duplicate Submap IDs")
    missing = [
        submap.submap_id
        for submap in retained_submaps
        if submap.submap_id not in submap_pose_by_id
    ]
    if missing:
        raise ValueError(f"online Stage1 is missing retained Submaps: {missing}")

    node_poses = snapshot.result.poses[:node_count]
    retained_submap_poses = tuple(
        submap_pose_by_id[submap.submap_id] for submap in retained_submaps
    )
    calibration = snapshot.result.calibration
    return {
        "online_node_translations": np.vstack(
            [pose.translation for pose in node_poses]
        ),
        "online_node_quaternions_xyzw": np.vstack(
            [pose.quaternion_xyzw for pose in node_poses]
        ),
        "online_submap_translations": np.vstack(
            [pose.translation for pose in retained_submap_poses]
        ),
        "online_submap_quaternions_xyzw": np.vstack(
            [pose.quaternion_xyzw for pose in retained_submap_poses]
        ),
        "online_gravity_magnitude": np.asarray(
            calibration.gravity_magnitude, dtype=np.float64
        ),
        "online_imu_from_tracking_translation": (
            calibration.imu_from_tracking.translation.copy()
        ),
        "online_imu_from_tracking_quaternion_xyzw": (
            calibration.imu_from_tracking.quaternion_xyzw.copy()
        ),
        "online_linear_acceleration_bias": (
            calibration.linear_acceleration_bias.copy()
        ),
        "online_linear_acceleration_scaling": (
            calibration.linear_acceleration_scaling.copy()
        ),
        "online_linear_acceleration_cross_axis": (
            calibration.linear_acceleration_cross_axis.copy()
        ),
        "online_angular_velocity_bias": calibration.angular_velocity_bias.copy(),
        "online_angular_velocity_scaling": (
            calibration.angular_velocity_scaling.copy()
        ),
        "online_angular_velocity_cross_axis": (
            calibration.angular_velocity_cross_axis.copy()
        ),
    }
