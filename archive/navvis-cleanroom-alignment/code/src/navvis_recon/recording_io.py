"""Shared read-only access to rec-v4 calibration and raw laser bags."""

from __future__ import annotations

import heapq
import logging
from pathlib import Path
import re
from typing import Iterable, Iterator
import xml.etree.ElementTree as ET

import numpy as np
import rosbag
from scipy.spatial.transform import Rotation


LOG = logging.getLogger(__name__)


def _child_text(element: ET.Element, path: str) -> str:
    child = element.find(path)
    if child is None or child.text is None:
        raise ValueError(f"missing XML field {path}")
    return child.text.strip()


def _xml_pose(
    element: ET.Element,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pose = element.find("Pose")
    if pose is None:
        raise ValueError("sensor has no Pose")
    translation = np.array(
        [float(_child_text(pose, f"position/{axis}")) for axis in "xyz"]
    )
    q_wxyz = [
        float(_child_text(pose, f"orientation/{axis}")) for axis in "wxyz"
    ]
    return translation, (q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0])


def read_laser_poses(path: Path) -> dict[str, str]:
    """Read the calibrated head-frame laser poses expected by C++ workers."""
    root = ET.parse(path).getroot()
    result: dict[str, str] = {}
    head_from_cam_head_translation = np.array(
        [0.002770609359137844, 0.07619045530812281, -0.03438298002198831]
    )
    head_from_cam_head_rotation = Rotation.from_euler(
        "xyz",
        [0.0008889239100979543, -0.0015862031285254787, 1.5803277823298127],
    )
    for model in root.iter("VelodyneLaserModel"):
        name = _child_text(model, "SensorName")
        translation, quaternion = _xml_pose(model)
        if name == "laser_vert":
            # The vertical trolley box is defined directly in cam_head.
            result["laser_vert_box"] = ",".join(
                map(str, (*translation, *quaternion))
            )
        head_translation = (
            head_from_cam_head_translation
            + head_from_cam_head_rotation.apply(translation)
        )
        head_rotation = head_from_cam_head_rotation * Rotation.from_quat(quaternion)
        result[name] = ",".join(
            map(str, (*head_translation, *head_rotation.as_quat()))
        )
    expected = {"laser_horiz", "laser_vert", "laser_vert_box"}
    if set(result) != expected:
        raise ValueError(f"expected horizontal/vertical lasers, found {sorted(result)}")
    return result


def numeric_laser_bags(dataset: Path, sensor: str) -> list[Path]:
    """Return one sensor's split bags in numeric suffix order."""
    pattern = re.compile(rf"bag_laser_{re.escape(sensor)}_(\d+)\.bag$")
    found: list[tuple[int, Path]] = []
    for path in (dataset / "internal" / "bags").glob(
        f"bag_laser_{sensor}_*.bag"
    ):
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


# Backwards-compatible name used by existing diagnostic scripts.
numeric_bags = numeric_laser_bags


def chronological_laser_scans(
    dataset: Path, sensors: Iterable[tuple[int, str]]
) -> Iterator[tuple[int, str, Path, object]]:
    """Merge complete laser messages from all split bags by header timestamp."""
    opened: list[rosbag.Bag] = []
    heap: list[tuple[int, int, int, int, str, Path, object, object]] = []
    serial = 0
    try:
        entries = [
            (sensor_id, sensor_name, bag_path)
            for sensor_id, sensor_name in sensors
            for bag_path in numeric_laser_bags(dataset, sensor_name)
        ]
        for bag_index, (sensor_id, sensor_name, bag_path) in enumerate(entries):
            bag = rosbag.Bag(str(bag_path))
            opened.append(bag)
            topic = f"/laser_{sensor_name}/packets"
            iterator = iter(bag.read_messages(topics=[topic]))
            try:
                _, message, _ = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (
                    message.header.stamp.to_nsec(),
                    sensor_id,
                    bag_index,
                    serial,
                    sensor_name,
                    bag_path,
                    message,
                    iterator,
                ),
            )
            serial += 1

        while heap:
            (
                _,
                sensor_id,
                bag_index,
                _,
                sensor_name,
                bag_path,
                message,
                iterator,
            ) = heapq.heappop(heap)
            yield sensor_id, sensor_name, bag_path, message
            try:
                _, next_message, _ = next(iterator)
            except StopIteration:
                continue
            heapq.heappush(
                heap,
                (
                    next_message.header.stamp.to_nsec(),
                    sensor_id,
                    bag_index,
                    serial,
                    sensor_name,
                    bag_path,
                    next_message,
                    iterator,
                ),
            )
            serial += 1
    finally:
        for bag in opened:
            bag.close()


def world_builder_active_windows(
    dataset: Path, trajectory_start: float, trajectory_end: float
) -> list[tuple[float, float]]:
    """Recover cloud_builder's add_scans state from recorded control events."""
    events: list[tuple[float, bool]] = []
    bag_directory = dataset / "internal" / "bags"
    for path in sorted(bag_directory.glob("bag_*.bag")):
        if path.name.startswith("bag_laser_"):
            continue
        try:
            with rosbag.Bag(str(path)) as bag:
                if "/user_interaction" not in bag.get_type_and_topic_info().topics:
                    continue
                messages = bag.read_messages(topics=["/user_interaction"])
                for _, message, bag_time in messages:
                    if (
                        message.target != "world_builder"
                        or message.feature != "add_scans"
                    ):
                        continue
                    stamp = message.header.stamp.to_sec()
                    events.append(
                        (
                            stamp if stamp > 0 else bag_time.to_sec(),
                            bool(message.val_b),
                        )
                    )
        except Exception as error:
            LOG.warning("Could not read scan-control events from %s: %s", path, error)
    events.sort()
    windows: list[tuple[float, float]] = []
    active = True
    active_start = trajectory_start
    for timestamp, enabled in events:
        timestamp = min(max(timestamp, trajectory_start), trajectory_end)
        if active and not enabled:
            if timestamp > active_start:
                windows.append((active_start, timestamp))
            active = False
        elif not active and enabled:
            active_start = timestamp
            active = True
    if active and trajectory_end > active_start:
        windows.append((active_start, trajectory_end))
    return windows
