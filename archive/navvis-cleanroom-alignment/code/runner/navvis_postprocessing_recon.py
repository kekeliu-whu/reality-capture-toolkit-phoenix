#!/usr/bin/env python3
"""Independent post-processing runner for NavVis rec-v4 G10/G11 recordings.

It accepts the important options of navvis-postprocessing, reconstructs an
upsampled map-frame trajectory from the recording's online SLAM and local
odometry tracks (or uses a supplied optimized trajectory), streams laser
packets to the C++ workers, and writes the standard artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import rosbag
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

RECON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECON_ROOT / "src"))

from navvis_recon.slam_reconstruction import (  # noqa: E402
    Trajectory as SlamTrajectory,
    evaluate_trajectory,
    fuse_global_and_local,
)
from navvis_recon.floor_estimator import (  # noqa: E402
    TRACE_MINIMUM_INTERVAL_NS,
    TraceSample,
    refined_floor_estimator,
    to_official_json,
)
from navvis_recon.recording_io import (  # noqa: E402
    chronological_laser_scans,
    numeric_bags,
    read_laser_poses,
    world_builder_active_windows,
)


LOG = logging.getLogger("navvis-recon")


@dataclass(frozen=True)
class Pose:
    timestamp: float
    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    timestamp_ns: int | None = None


@dataclass(frozen=True)
class OCam:
    name: str
    translation: np.ndarray
    rotation: np.ndarray
    width: int
    height: int
    c: float
    d: float
    e: float
    cx: float
    cy: float
    world2cam: np.ndarray


def _add_general_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--bagplayer-args", default="")
    parser.add_argument("--proc-base-dir", type=Path, required=True)
    parser.add_argument("--caller", default="standalone")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--res", "--resolution", type=float, default=0.01)
    parser.add_argument("--cloud-format", choices=("ply",), default="ply")
    parser.add_argument("--preset", choices=("standard", "outdoors", "high-confidence"), default="standard")
    parser.add_argument("--num-threads-panos", type=int, default=8)
    parser.add_argument(
        "--panorama-sigfpe-retries",
        type=int,
        default=1,
        help="Retry a capture this many times after SIGFPE (default: 1)",
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Default: PROC_BASE/DATASET_ID/recon")


def _add_trajectory_arguments(parser: argparse.ArgumentParser) -> None:
    trajectory_input = parser.add_mutually_exclusive_group()
    trajectory_input.add_argument(
        "--trajectory-bag",
        type=Path,
        help="Use an existing SLAM/output trajectory bag as immutable input (topic: tf_trajectory)",
    )
    trajectory_input.add_argument(
        "--trajectory-csv",
        type=Path,
        help=(
            "Use a generated optimized trajectory CSV as immutable input; accepts the "
            "navvis_slam_recon.py timestamp_ns,tx,ty,tz,qx,qy,qz,qw schema"
        ),
    )
    parser.add_argument(
        "--slam-mode",
        choices=("recorded-global", "local-only"),
        default="recorded-global",
        help=(
            "Without --trajectory-bag/--trajectory-csv, fuse recorded online map-frame SLAM "
            "with local odometry and upsample 5x (default), or retain the legacy local-only "
            "trajectory"
        ),
    )
    parser.add_argument(
        "--slam-reference-bag",
        type=Path,
        help=(
            "Optional frozen trajectory.bag used only to report absolute and 1-second "
            "relative-pose SLAM errors; it never changes the generated trajectory"
        ),
    )


def _add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pandar-worker", type=Path, help="Path to the compiled navvis_recon_pandar")
    parser.add_argument("--surface-worker", type=Path, help="Path to navvis_recon_shard_surface_filter")
    parser.add_argument(
        "--quality-worker",
        type=Path,
        help="Path to navvis_recon_mapped_space_quality",
    )
    parser.add_argument("--quality-grid-resolution", type=float, default=0.16666666666666669)
    parser.add_argument("--quality-min-rays-per-voxel", type=int, default=36)
    parser.add_argument("--quality-use-every-nth-point", type=int, default=1)
    parser.add_argument(
        "--surface-tile-threads",
        type=int,
        default=8,
        help=(
            "Concurrent 5 m surface tiles; the current byte-exact kernel defaults to "
            "8 outer workers and derives 4 point threads from a 32-thread budget"
        ),
    )
    parser.add_argument(
        "--surface-preprocess-threads",
        type=int,
        default=8,
        help="Concurrent 10 m free-space shards (default: 8, matching the captured worker pool)",
    )
    parser.add_argument("--colorizer-worker", type=Path, help="Path to navvis_recon_surface_colorizer")
    parser.add_argument(
        "--color-backend",
        choices=("auto", "original", "recon"),
        default="recon",
        help="Point-coloring backend: clean-room recon by default; original must be explicitly selected",
    )
    parser.add_argument(
        "--original-colorizer-worker",
        type=Path,
        default=Path("/opt/NavVis/pointcloud-coloring/bin/nv_colorcloud"),
        help="Path to the installed NavVis C++ point-cloud colorizer",
    )
    parser.add_argument("--panorama-worker", type=Path, help="Path to the compiled navvis_recon_ocam_panorama")


def _add_run_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet-stride", type=int, default=1, help="Decode every Nth packet")
    parser.add_argument(
        "--start-offset",
        type=float,
        default=0.0,
        help="Start this many seconds after the trajectory begins (validation mode)",
    )
    parser.add_argument(
        "--time-window",
        action="append",
        default=[],
        metavar="START:END",
        help="Repeatable trajectory-relative time window; overrides start-offset/max-duration",
    )
    parser.add_argument("--max-duration", type=float, help="Only process this many seconds (validation mode)")
    parser.add_argument(
        "--cloud-roi",
        help="Optional world ROI minx,miny,minz,maxx,maxy,maxz for ray-history validation",
    )
    parser.add_argument("--max-panos", type=int, help="Only create the first N panoramas")
    parser.add_argument("--pano-width", type=int, default=8192)
    parser.add_argument(
        "--aligned-standard",
        action="store_true",
        help="Use the validated raw-shard, binary-parameter surface, direct-camera color pipeline",
    )
    parser.add_argument("--skip-cloud", action="store_true")
    parser.add_argument("--skip-panos", action="store_true")
    parser.add_argument("--skip-coloring", action="store_true")


def _add_cloud_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-active-voxels", type=int, default=8_000_000)
    parser.add_argument("--ray-origin-cell", type=float, default=0.50)
    parser.add_argument("--ray-angular-bin-deg", type=float, default=0.12)
    parser.add_argument("--free-space-mode", choices=("sparse", "directional"), default="sparse")
    parser.add_argument("--free-space-traversal-resolution", type=float, default=0.02)
    parser.add_argument("--free-space-ray-radius", type=float, default=0.012)
    parser.add_argument("--free-space-ray-stride", type=int, default=16)
    parser.add_argument("--free-space-endpoint-margin", type=float, default=0.08)
    parser.add_argument(
        "--free-space-min-intersections",
        type=int,
        help="Minimum traversing rays (auto: G10/VLP16=6, G11/Pandar=3)",
    )
    parser.add_argument(
        "--free-space-intersection-hit-ratio",
        type=float,
        help="Traversal/endpoint-hit ratio (auto: G10/VLP16=3.0, G11/Pandar=1.5)",
    )
    parser.add_argument(
        "--surface-min-neighbors",
        type=int,
        help="Override preset surface-neighbor threshold; 0 is useful for raw-decoder validation",
    )
    parser.add_argument(
        "--unvoxelized-cloud",
        type=Path,
        help="Optional diagnostic PLY of filtered returns before 1 cm endpoint voxelization",
    )
    parser.add_argument(
        "--scan-stats",
        type=Path,
        help="Optional per-scan CSV with decoded return counts and trajectory motion metrics",
    )
    parser.add_argument(
        "--scan-stats-only",
        action="store_true",
        help="Decode scans and write scan statistics without normal/fringe/point processing",
    )
    parser.add_argument(
        "--no-multilayer-fringe",
        action="store_true",
        help="Diagnostic: estimate normals but do not remove multilayer fringe returns",
    )
    parser.add_argument(
        "--no-vertical-foot-filter",
        action="store_true",
        help="Diagnostic: disable the G11 vertical trolley-foot PlaneFilter",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NavVis rec-v4 post-processing reconstruction")
    _add_general_arguments(parser)
    _add_trajectory_arguments(parser)
    _add_worker_arguments(parser)
    _add_run_selection_arguments(parser)
    _add_cloud_tuning_arguments(parser)
    return parser


def setup_logging(output_dir: Path, requested_log: Path | None) -> None:
    output_log = output_dir / "logs" / "proc" / "postprocessing-recon.log"
    output_log.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(), logging.FileHandler(output_log)]
    if requested_log and requested_log.resolve() != output_log.resolve():
        requested_log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(requested_log))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def quaternion_matrix_xyzw(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = np.asarray(tuple(q), dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def child_text(element: ET.Element, path: str) -> str:
    child = element.find(path)
    if child is None or child.text is None:
        raise ValueError(f"missing XML field {path}")
    return child.text.strip()


def xml_pose(element: ET.Element) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    pose = element.find("Pose")
    if pose is None:
        raise ValueError("sensor has no Pose")
    translation = np.array([float(child_text(pose, f"position/{axis}")) for axis in "xyz"])
    q_wxyz = [float(child_text(pose, f"orientation/{axis}")) for axis in "wxyz"]
    return translation, (q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0])


def read_sensor_frame(path: Path) -> tuple[dict[str, str], list[OCam]]:
    root = ET.parse(path).getroot()
    laser_poses = read_laser_poses(path)

    cameras: list[OCam] = []
    for model in root.iter("CameraModel"):
        name = child_text(model, "SensorName")
        translation, q = xml_pose(model)
        ocam = model.find("OCamModel")
        if ocam is None:
            continue
        cameras.append(
            OCam(
                name=name,
                translation=translation,
                rotation=quaternion_matrix_xyzw(q),
                width=int(child_text(model, "ImageSize/Width")),
                height=int(child_text(model, "ImageSize/Height")),
                c=float(child_text(ocam, "c")),
                d=float(child_text(ocam, "d")),
                e=float(child_text(ocam, "e")),
                cx=float(child_text(ocam, "cx")),
                cy=float(child_text(ocam, "cy")),
                world2cam=np.array([float(x.text) for x in ocam.findall("world2cam/coeff")]),
            )
        )
    if len(cameras) != 4:
        raise ValueError(f"expected four cameras, found {len(cameras)}")
    cameras.sort(key=lambda camera: camera.name)
    return laser_poses, cameras


def capture_poses(dataset: Path) -> list[Pose]:
    result = []
    for path in sorted((dataset / "info").glob("*-info.json")):
        data = json.loads(path.read_text())
        if str(data.get("valid", "true")).lower() != "true":
            continue
        head = data["cam_head"]
        # Capture metadata stores quaternions as WXYZ (matching sensor_frame.xml),
        # while Pose and quaternion_matrix_xyzw use XYZW internally.
        q_wxyz = tuple(map(float, head["quaternion"]))
        result.append(
            Pose(
                float(data["timestamp"]),
                tuple(map(float, head["position"])),
                (q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]),
            )
        )
    return result


def read_trajectory_bag(path: Path) -> list[Pose]:
    result: list[Pose] = []
    if not path.exists():
        return result

    def collect(candidate: Path, allow_unindexed: bool) -> None:
        with rosbag.Bag(str(candidate), allow_unindexed=allow_unindexed) as bag:
            for _, message, _ in bag.read_messages(topics=["tf_trajectory"]):
                for transform in message.transforms:
                    if transform.child_frame_id != "base_link":
                        continue
                    t = transform.transform.translation
                    q = transform.transform.rotation
                    result.append(
                        Pose(
                            transform.header.stamp.to_sec(),
                            (t.x, t.y, t.z),
                            (q.x, q.y, q.z, q.w),
                            transform.header.stamp.to_nsec(),
                        )
                    )

    try:
        # rosbag's allow_unindexed mode can silently return no messages for a
        # bag whose final index was never written. Try it first because it is
        # read-only and cheap.
        collect(path, allow_unindexed=True)
        if not result:
            # Never modify the recording. Reindex a small temporary copy and
            # consume that copy instead.
            with tempfile.TemporaryDirectory(prefix="navvis-recon-trajectory-") as temp_dir:
                copied = Path(temp_dir) / path.name
                shutil.copy2(path, copied)
                subprocess.run(
                    ["rosbag", "reindex", "--quiet", str(copied)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                collect(copied, allow_unindexed=False)
                LOG.info("Recovered %d poses by reindexing a temporary trajectory copy", len(result))
    except Exception as error:
        LOG.warning("Could not read trajectory bag %s: %s", path, error)
    result.sort(key=lambda pose: pose.timestamp)
    return result


def read_trajectory_csv(path: Path) -> list[Pose]:
    """Read the autonomous SLAM trajectory without changing its nanosecond clock."""
    if not path.is_file():
        raise FileNotFoundError(path)
    result: list[Pose] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"}
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(
                f"trajectory CSV {path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row_index, row in enumerate(reader, start=2):
            try:
                timestamp_ns = int(row["timestamp_ns"])
                translation = tuple(float(row[name]) for name in ("tx", "ty", "tz"))
                quaternion = tuple(float(row[name]) for name in ("qx", "qy", "qz", "qw"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid trajectory CSV value at {path}:{row_index}: {error}"
                ) from error
            values = np.asarray((*translation, *quaternion), dtype=np.float64)
            quaternion_norm = float(np.linalg.norm(quaternion))
            if (
                not np.all(np.isfinite(values))
                or abs(quaternion_norm - 1.0) > 1.0e-3
            ):
                raise ValueError(f"invalid finite pose at {path}:{row_index}")
            if result and timestamp_ns <= _pose_timestamp_ns(result[-1]):
                raise ValueError(
                    f"trajectory timestamps must be strictly increasing at {path}:{row_index}"
                )
            result.append(
                Pose(timestamp_ns * 1.0e-9, translation, quaternion, timestamp_ns)
            )
    return result


def _slam_trajectory(poses: list[Pose]) -> SlamTrajectory:
    return SlamTrajectory(
        np.asarray([pose.timestamp for pose in poses], dtype=np.float64),
        np.asarray([pose.translation for pose in poses], dtype=np.float64),
        np.asarray([pose.quaternion_xyzw for pose in poses], dtype=np.float64),
    )


def _runner_poses(trajectory: SlamTrajectory) -> list[Pose]:
    return [
        Pose(
            float(timestamp),
            tuple(map(float, translation)),
            tuple(map(float, quaternion)),
        )
        for timestamp, translation, quaternion in zip(
            trajectory.timestamps,
            trajectory.translations,
            trajectory.quaternions_xyzw,
        )
    ]


def combined_trajectory(
    dataset: Path,
    supplied_bag: Path | None = None,
    slam_mode: str = "recorded-global",
    supplied_csv: Path | None = None,
) -> tuple[list[Pose], dict]:
    if supplied_csv is not None:
        supplied_csv = supplied_csv.resolve()
        supplied = read_trajectory_csv(supplied_csv)
        if len(supplied) < 2:
            raise RuntimeError(f"No usable trajectory in supplied CSV: {supplied_csv}")
        return supplied, {
            "source": "generated autonomous optimized trajectory CSV",
            "source_path": str(supplied_csv),
            "supplied_pose_count": len(supplied),
            "combined_pose_count": len(supplied),
            "start": supplied[0].timestamp,
            "end": supplied[-1].timestamp,
            "tail_fallback": False,
            "slam_mode": "autonomous-optimized",
            "offline_frontend_recomputed": True,
            "loop_closures_recomputed": True,
            "imu_pose_graph_recomputed": True,
        }
    if supplied_bag is not None:
        supplied_bag = supplied_bag.resolve()
        supplied = read_trajectory_bag(supplied_bag)
        if len(supplied) < 2:
            raise RuntimeError(f"No usable base_link trajectory in supplied bag: {supplied_bag}")
        return supplied, {
            "source": "supplied optimized trajectory bag",
            "source_path": str(supplied_bag),
            "supplied_pose_count": len(supplied),
            "combined_pose_count": len(supplied),
            "start": supplied[0].timestamp,
            "end": supplied[-1].timestamp,
            "tail_fallback": False,
            "slam_mode": "supplied-optimized",
        }

    local_path = dataset / "internal" / "artifacts" / "trajectory_local.bag"
    local = read_trajectory_bag(local_path)
    global_path = dataset / "internal" / "trajectory_slam.bag"
    global_slam = read_trajectory_bag(global_path)
    if slam_mode == "recorded-global" and len(global_slam) >= 2 and len(local) >= 2:
        fused = fuse_global_and_local(
            _slam_trajectory(global_slam),
            _slam_trajectory(local),
            upsampling_factor=5,
        )
        result = _runner_poses(fused)
        return result, {
            "source": "recorded online SLAM fused with local odometry",
            "global_source_path": str(global_path),
            "local_source_path": str(local_path),
            "slam_mode": "recorded-global",
            "online_slam_pose_count": len(global_slam),
            "local_pose_count": len(local),
            "combined_pose_count": len(result),
            "trajectory_upsampling": 5,
            "start": result[0].timestamp,
            "end": result[-1].timestamp,
            "map_odom_fusion": True,
            "offline_frontend_recomputed": False,
            "loop_closures_recomputed": False,
            "imu_pose_graph_recomputed": False,
        }
    if slam_mode == "recorded-global" and len(global_slam) >= 2:
        return global_slam, {
            "source": "recorded online SLAM (local trajectory unavailable)",
            "source_path": str(global_path),
            "slam_mode": "recorded-global",
            "online_slam_pose_count": len(global_slam),
            "combined_pose_count": len(global_slam),
            "trajectory_upsampling": 1,
            "start": global_slam[0].timestamp,
            "end": global_slam[-1].timestamp,
            "map_odom_fusion": False,
            "offline_frontend_recomputed": False,
            "loop_closures_recomputed": False,
            "imu_pose_graph_recomputed": False,
        }

    captures = capture_poses(dataset)
    metadata = {
        "source": "recording-local trajectory with capture-tail fallback",
        "source_path": str(local_path),
        "slam_mode": "local-only" if slam_mode == "local-only" else "local-fallback",
        "local_pose_count": len(local),
        "capture_pose_count": len(captures),
        "tail_fallback": False,
    }
    if local:
        last_local = local[-1].timestamp
        tail = [pose for pose in captures if pose.timestamp > last_local]
        metadata["tail_fallback"] = bool(tail)
        result = local + tail
    else:
        result = captures
        metadata["capture_only_fallback"] = True
    result.sort(key=lambda pose: pose.timestamp)
    deduplicated: list[Pose] = []
    for pose in result:
        if not deduplicated or pose.timestamp > deduplicated[-1].timestamp + 1e-8:
            deduplicated.append(pose)
    if len(deduplicated) < 2:
        raise RuntimeError("No usable supplied, online-SLAM, local, or capture trajectory")
    metadata.update(
        {
            "combined_pose_count": len(deduplicated),
            "start": deduplicated[0].timestamp,
            "end": deduplicated[-1].timestamp,
        }
    )
    return deduplicated, metadata


def write_trajectory(path: Path, trajectory: list[Pose]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["# timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for pose in trajectory:
            if pose.timestamp_ns is None:
                timestamp_text = f"{pose.timestamp:.9f}"
            else:
                seconds, nanoseconds = divmod(pose.timestamp_ns, 1_000_000_000)
                timestamp_text = f"{seconds}.{nanoseconds:09d}"
            writer.writerow((timestamp_text, *pose.translation, *pose.quaternion_xyzw))


def _pose_timestamp_ns(pose: Pose) -> int:
    return pose.timestamp_ns if pose.timestamp_ns is not None else round(pose.timestamp * 1e9)


def _floor_trace_events(dataset: Path) -> tuple[list[tuple[int, tuple[float, float, float]]], list[Path]]:
    """Read the magnetic-field clock used by the native trace splitter."""
    bag_directory = dataset / "internal" / "bags"
    candidates = sorted(
        path
        for path in bag_directory.glob("bag_*.bag")
        if not path.name.startswith("bag_laser_")
    )
    events: list[tuple[int, tuple[float, float, float]]] = []
    sources: list[Path] = []
    for path in candidates:
        with rosbag.Bag(str(path), allow_unindexed=True) as bag:
            topics = bag.get_type_and_topic_info()[1]
            if "/imu/magnetic_field" not in topics:
                continue
            sources.append(path)
            for _, message, _ in bag.read_messages(topics=["/imu/magnetic_field"]):
                vector = message.magnetic_field
                magnetic_field = (float(vector.x), float(vector.y), float(vector.z))
                if not all(math.isfinite(value) for value in magnetic_field):
                    raise ValueError(f"non-finite magnetic-field sample in {path}")
                events.append((message.header.stamp.to_nsec(), magnetic_field))
    if not sources:
        raise FileNotFoundError(
            f"no /imu/magnetic_field source bag below {bag_directory}; "
            "cannot create the official Floor trace"
        )
    events.sort(key=lambda item: item[0])
    if not events:
        raise RuntimeError("/imu/magnetic_field exists but contains no messages")
    return events, sources


def _interpolate_floor_trace(
    trajectory: list[Pose],
    events: Iterable[tuple[int, tuple[float, float, float]]],
) -> tuple[list[TraceSample], list[tuple[float, float, float, float]], list[tuple[float, float, float]]]:
    if len(trajectory) < 2:
        raise ValueError("Floor trace interpolation requires at least two trajectory poses")
    trajectory_ns = np.asarray([_pose_timestamp_ns(pose) for pose in trajectory], dtype=np.int64)
    if np.any(trajectory_ns[1:] <= trajectory_ns[:-1]):
        raise ValueError("trajectory timestamps must be strictly increasing for Floor")
    translations = np.asarray([pose.translation for pose in trajectory], dtype=np.float64)
    quaternions = np.asarray([pose.quaternion_xyzw for pose in trajectory], dtype=np.float64)
    if not np.all(np.isfinite(translations)) or not np.all(np.isfinite(quaternions)):
        raise ValueError("trajectory contains non-finite poses")

    selected: list[tuple[int, tuple[float, float, float]]] = []
    last_timestamp: int | None = None
    for timestamp_ns, magnetic_field in events:
        if timestamp_ns < trajectory_ns[0] or timestamp_ns > trajectory_ns[-1]:
            continue
        if (
            last_timestamp is not None
            and timestamp_ns - last_timestamp <= TRACE_MINIMUM_INTERVAL_NS
        ):
            continue
        selected.append((timestamp_ns, magnetic_field))
        last_timestamp = timestamp_ns
    if not selected:
        raise RuntimeError("no magnetic-field timestamps overlap the trajectory")

    event_ns = np.asarray([item[0] for item in selected], dtype=np.int64)
    high = np.searchsorted(trajectory_ns, event_ns, side="right")
    high = np.clip(high, 1, len(trajectory_ns) - 1)
    low = high - 1
    alpha = (event_ns - trajectory_ns[low]) / (trajectory_ns[high] - trajectory_ns[low])
    xyz = translations[low] + alpha[:, None] * (translations[high] - translations[low])

    relative_pose_seconds = (trajectory_ns - trajectory_ns[0]).astype(np.float64) * 1e-9
    relative_event_seconds = (event_ns - trajectory_ns[0]).astype(np.float64) * 1e-9
    orientations_xyzw = Slerp(
        relative_pose_seconds,
        Rotation.from_quat(quaternions),
    )(relative_event_seconds).as_quat()
    # The native trace converts rotation matrices back through Eigen's
    # Quaternion constructor. Its sign branch is deterministic but is not the
    # same as SciPy's continuous quaternion sign. Canonicalize the equivalent
    # rotations using Eigen's positive selected-component convention.
    for quaternion in orientations_xyzw:
        if quaternion[3] * quaternion[3] > 0.25:
            selected_component = 3
        else:
            selected_component = int(np.argmax(np.abs(quaternion[:3])))
        if quaternion[selected_component] < 0.0:
            quaternion *= -1.0
    samples = [
        TraceSample(int(timestamp_ns), float(point[0]), float(point[1]), float(point[2]))
        for timestamp_ns, point in zip(event_ns, xyz)
    ]
    orientations_wxyz = [
        (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
        for q in orientations_xyzw
    ]
    magnetic_fields = [item[1] for item in selected]
    return samples, orientations_wxyz, magnetic_fields


def _write_floor_trace(
    path: Path,
    samples: list[TraceSample],
    orientations_wxyz: list[tuple[float, float, float, float]],
    magnetic_fields: list[tuple[float, float, float]],
) -> None:
    if not (len(samples) == len(orientations_wxyz) == len(magnetic_fields)):
        raise ValueError("Floor trace columns have inconsistent lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        stream.write("nsecs,  x, y, z,  ori_w, ori_x, ori_y, ori_z,  mag_x, mag_y, mag_z\n")
        for sample, orientation, magnetic_field in zip(
            samples, orientations_wxyz, magnetic_fields
        ):
            values = (
                sample.x,
                sample.y,
                sample.z,
                *orientation,
                *magnetic_field,
            )
            text = [format(value, ".6g") for value in values]
            stream.write(
                f"{sample.timestamp_ns}, {', '.join(text[:7])},  {', '.join(text[7:])}\n"
            )


def estimate_and_write_floors(
    dataset: Path, trajectory: list[Pose], artifacts_directory: Path
) -> dict:
    events, sources = _floor_trace_events(dataset)
    samples, orientations, magnetic_fields = _interpolate_floor_trace(
        trajectory, events
    )
    # The native trace is serialized to six significant digits before Floor
    # consumes it. Reparse that representation so rounding decisions match.
    rounded_samples = [
        TraceSample(
            sample.timestamp_ns,
            float(format(sample.x, ".6g")),
            float(format(sample.y, ".6g")),
            float(format(sample.z, ".6g")),
        )
        for sample in samples
    ]
    floors = refined_floor_estimator(rounded_samples)
    trace_path = artifacts_directory / "trace.csv"
    floors_path = artifacts_directory / "floors.json"
    _write_floor_trace(trace_path, rounded_samples, orientations, magnetic_fields)
    floors_path.write_text(json.dumps(to_official_json(floors), indent=4) + "\n")
    return {
        "trace_sources": [str(path) for path in sources],
        "trace_sample_count": len(rounded_samples),
        "floor_count": len(floors),
        "trace_path": str(trace_path),
        "floors_path": str(floors_path),
        "schema": "official top-level floor array with nanosecond time_ranges",
    }


def find_worker(explicit: Path | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("NAVVIS_RECON_PANDAR"):
        candidates.append(Path(os.environ["NAVVIS_RECON_PANDAR"]))
    root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            root / "build-release" / "navvis_recon_pandar",
            root / "build-cpp" / "navvis_recon_pandar",
            root / "cpp" / "build" / "navvis_recon_pandar",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("navvis_recon_pandar not found; build cpp/ or pass --pandar-worker")


def find_panorama_worker(explicit: Path | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("NAVVIS_RECON_PANORAMA"):
        candidates.append(Path(os.environ["NAVVIS_RECON_PANORAMA"]))
    root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            root / "build-release" / "navvis_recon_ocam_panorama",
            root / "build-cpp" / "navvis_recon_ocam_panorama",
            root / "cpp" / "build" / "navvis_recon_ocam_panorama",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "navvis_recon_ocam_panorama not found; build cpp/ or pass --panorama-worker"
    )


def find_stage_worker(explicit: Path | None, executable: str, environment: str) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get(environment):
        candidates.append(Path(os.environ[environment]))
    root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            root / "build-release" / executable,
            root / "build-cpp" / executable,
            root / "cpp" / "build" / executable,
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(f"{executable} not found; build cpp/ or pass its worker option")


def stream_cloud(
    args: argparse.Namespace,
    dataset: Path,
    output: Path,
    trajectory_path: Path,
    laser_poses: dict[str, str],
    trajectory_start: float,
    trajectory_end: float,
) -> dict:
    worker = find_worker(args.pandar_worker)
    surface_neighbors = (
        args.surface_min_neighbors
        if args.surface_min_neighbors is not None
        else {"standard": 3, "outdoors": 4, "high-confidence": 6}[args.preset]
    )
    command = [
        str(worker),
        "--frame-timestamps-ns",
        "--trajectory", str(trajectory_path),
        "--output", str(output),
        "--resolution", str(args.res),
        "--min-range", "0.4",
        "--max-range", "30.0",
        "--max-active-voxels", str(args.max_active_voxels),
        "--surface-min-neighbors", str(surface_neighbors),
        "--ray-origin-cell", str(args.ray_origin_cell),
        "--horiz-pose", laser_poses["laser_horiz"],
        "--vert-pose", laser_poses["laser_vert"],
        "--vert-box-pose", laser_poses["laser_vert_box"],
    ]
    retained_shards = output.parent / "raw_shards"
    if args.cloud_roi:
        command.extend(["--world-roi", args.cloud_roi])
    if args.unvoxelized_cloud:
        command.extend(["--unvoxelized-output", str(args.unvoxelized_cloud.resolve())])
    if args.scan_stats:
        command.extend(["--scan-stats", str(args.scan_stats.resolve())])
    if args.scan_stats_only:
        command.append("--scan-stats-only")
    if args.no_multilayer_fringe:
        command.append("--no-multilayer-fringe")
    if args.no_vertical_foot_filter:
        command.append("--no-vertical-foot-filter")
    if args.aligned_standard:
        command.extend(["--retain-shards", str(retained_shards), "--shards-only"])
    LOG.info("Starting C++ laser cloud worker: %s", " ".join(command))
    started = time.time()
    packet_count = 0
    motion_state_packet_count = 0
    packet_size_counts: dict[int, int] = {}
    scan_count = 0
    motion_state_scan_count = 0
    windows: list[tuple[float, float]] = []
    if args.time_window:
        for specification in args.time_window:
            start_text, separator, end_text = specification.partition(":")
            if not separator:
                raise ValueError(f"invalid time window {specification!r}; expected START:END")
            start_offset, end_offset = float(start_text), float(end_text)
            if start_offset < 0 or end_offset <= start_offset:
                raise ValueError(f"invalid time window {specification!r}")
            windows.append(
                (
                    min(trajectory_end, trajectory_start + start_offset),
                    min(trajectory_end, trajectory_start + end_offset),
                )
            )
    else:
        window_start = min(trajectory_end, trajectory_start + args.start_offset)
        window_end = trajectory_end
        if args.max_duration is not None:
            window_end = min(window_end, window_start + args.max_duration)
        windows.append((window_start, window_end))
    windows = [(start, end) for start, end in windows if end > start]
    if not windows:
        raise ValueError("no processing time remains inside the trajectory")
    control_windows = world_builder_active_windows(dataset, trajectory_start, trajectory_end)
    LOG.info(
        "Applying recorded world_builder/add_scans windows: %s",
        ", ".join(f"{start:.6f}:{end:.6f}" for start, end in control_windows),
    )
    cutoff = max(end for _, end in windows)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        bag_names = [
            path.name
            for sensor_name in ("horiz", "vert")
            for path in numeric_bags(dataset, sensor_name)
        ]
        LOG.info(
            "Reading %d laser bags in chronological scan order: %s",
            len(bag_names), ", ".join(bag_names),
        )
        for sensor_id, _, _, message in chronological_laser_scans(
            dataset, enumerate(("horiz", "vert"))
        ):
            scan_stamp = message.header.stamp.to_sec()
            if scan_stamp > cutoff:
                break
            valid_packets = [
                packet for packet in message.packets
                if packet.stamp.to_sec() > 0 and len(packet.data) in (820, 1206)
            ]
            if not valid_packets:
                continue
            scan_end = max(packet.stamp.to_sec() for packet in valid_packets)
            if not any(
                start <= scan_stamp and scan_end <= end
                for start, end in windows
            ):
                continue
            insertion_enabled = any(
                start <= scan_stamp < end for start, end in control_windows
            )
            motion_state_scan_count += 1
            scan_count += int(insertion_enabled)
            process.stdin.write(struct.pack(
                "<BqH",
                sensor_id | (0 if insertion_enabled else 0x80),
                message.header.stamp.to_nsec(),
                0,
            ))
            for packet in valid_packets:
                payload = bytes(packet.data)
                motion_state_packet_count += 1
                if insertion_enabled:
                    packet_count += 1
                    packet_size_counts[len(payload)] = (
                        packet_size_counts.get(len(payload), 0) + 1
                    )
                if (motion_state_packet_count - 1) % args.packet_stride:
                    continue
                process.stdin.write(struct.pack(
                    "<BqH", sensor_id, packet.stamp.to_nsec(), len(payload)
                ))
                process.stdin.write(payload)
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"laser worker exited with status {return_code}")
    except Exception:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.terminate()
        process.wait()
        raise
    if args.aligned_standard:
        shard_files = list(retained_shards.glob("*.raytile"))
        return {
            "worker": str(worker),
            "packets_seen": packet_count,
            "packets_decoded": (packet_count + args.packet_stride - 1) // args.packet_stride,
            "motion_state_packets_seen": motion_state_packet_count,
            "packet_sizes": {str(size): count for size, count in sorted(packet_size_counts.items())},
            "scans": scan_count,
            "motion_state_scans": motion_state_scan_count,
            "seconds": time.time() - started,
            "raw_shards": len(shard_files),
            "raw_shard_bytes": sum(path.stat().st_size for path in shard_files),
            "raw_shard_directory": str(retained_shards),
            "ray_history": "endpoint voxel plus sensor-origin cell",
            "ray_origin_cell_m": args.ray_origin_cell,
            "time_windows_seconds": [
                [start - trajectory_start, end - trajectory_start] for start, end in windows
            ],
            "add_scans_windows_seconds": [
                [start - trajectory_start, end - trajectory_start]
                for start, end in control_windows
            ],
            "world_roi": args.cloud_roi,
            "minimum_range_m": 0.4,
            "maximum_range_m": 30.0,
            "intensity_filter": "Pandar standard profile or VLP16 horiz>=2/vert>=1 raw units",
            "vertical_foot_plane_filter": not args.no_vertical_foot_filter,
        }
    points, _ = open_reconstructed_ply(output)
    output_points = len(points)
    del points
    return {
        "worker": str(worker),
        "packets_seen": packet_count,
        "packets_decoded": (packet_count + args.packet_stride - 1) // args.packet_stride,
        "motion_state_packets_seen": motion_state_packet_count,
        "packet_sizes": {str(size): count for size, count in sorted(packet_size_counts.items())},
        "scans": scan_count,
        "motion_state_scans": motion_state_scan_count,
        "seconds": time.time() - started,
        "output_bytes": output.stat().st_size,
        "output_points": output_points,
        "surface_minimum_neighbors": surface_neighbors,
        "minimum_range_m": 0.4,
        "maximum_range_m": 30.0,
        "intensity_filter": "Pandar standard profile or VLP16 horiz>=2/vert>=1 raw units",
        "vertical_foot_plane_filter": not args.no_vertical_foot_filter,
        "add_scans_windows_seconds": [
            [start - trajectory_start, end - trajectory_start] for start, end in control_windows
        ],
    }


def panorama_rays(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    longitude = ((x + 0.5) / width - 0.5) * (2 * np.pi)
    latitude = (0.5 - (y + 0.5) / height) * np.pi
    cos_latitude = np.cos(latitude)
    # The recording/head frame is ROS-like: x/y span the floor plane and z is
    # vertical. Place +x at the panorama center and increase longitude toward
    # +y.
    return np.stack(
        [cos_latitude * np.cos(longitude), cos_latitude * np.sin(longitude), np.sin(latitude)], axis=-1
    ).astype(np.float32)


def ocam_map(camera: OCam, rays_head: np.ndarray, thumbnail_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays_camera = np.einsum("ij,hwj->hwi", camera.rotation.T, rays_head)
    x, y, z = [rays_camera[..., i] for i in range(3)]
    norm = np.sqrt(x * x + y * y)
    theta = np.arctan2(z, np.maximum(norm, 1e-9))
    rho = np.polynomial.polynomial.polyval(theta, camera.world2cam)
    normalized_x = x / np.maximum(norm, 1e-9)
    normalized_y = y / np.maximum(norm, 1e-9)
    row = normalized_x * rho * camera.c + normalized_y * rho * camera.d + camera.cx
    column = normalized_x * rho * camera.e + normalized_y * rho + camera.cy
    thumb_width, thumb_height = thumbnail_size
    map_x = (column * thumb_width / camera.width).astype(np.float32)
    map_y = (row * thumb_height / camera.height).astype(np.float32)
    valid = (map_x >= 0) & (map_y >= 0) & (map_x < thumb_width - 1) & (map_y < thumb_height - 1)
    return map_x, map_y, valid.astype(np.uint8)


def read_dng_thumbnail(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        orientation = image.getexif().get(274, 1)
        rgb = np.asarray(image.convert("RGB"))
    # The TIFF/DNG decoder has already applied the EXIF orientation to this
    # embedded preview, although Image.size still reports the stored raster
    # dimensions. Undo that display transform so the pixels again use the
    # same row/column convention as the full-resolution OCam calibration.
    if orientation == 8:
        rgb = np.rot90(rgb, k=-1)
    elif orientation == 6:
        rgb = np.rot90(rgb, k=1)
    elif orientation == 3:
        rgb = np.rot90(rgb, k=2)
    rgb = np.ascontiguousarray(rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build_panorama_maps(dataset: Path, cameras: list[OCam], width: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    first_images = [read_dng_thumbnail(dataset / "cam" / f"00000-{camera.name}.dng") for camera in cameras]
    rays = panorama_rays(width, width // 2)
    maps = [ocam_map(camera, rays, (image.shape[1], image.shape[0])) for camera, image in zip(cameras, first_images)]
    return maps


def feather_weights(maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    distances = [cv2.distanceTransform(valid, cv2.DIST_L2, 3) + valid * 1e-3 for _, _, valid in maps]
    total = np.maximum(np.sum(distances, axis=0), 1e-6)
    return [distance / total for distance in distances]


def stitch_one_panorama(
    dataset: Path,
    output_dir: Path,
    index: int,
    cameras: list[OCam],
    maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    weights: list[np.ndarray],
) -> None:
    blended = np.zeros((*weights[0].shape, 3), np.float32)
    valid_union = np.zeros(weights[0].shape, np.uint8)
    for camera, (map_x, map_y, valid), weight in zip(cameras, maps, weights):
        image = read_dng_thumbnail(dataset / "cam" / f"{index:05d}-{camera.name}.dng")
        image = cv2.detailEnhance(image, sigma_s=8, sigma_r=0.15)
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        blended += warped.astype(np.float32) * weight[..., None]
        valid_union |= valid
    missing = valid_union == 0
    result = np.clip(blended, 0, 255).astype(np.uint8)
    if np.any(missing):
        result = cv2.inpaint(result, missing.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    cv2.imwrite(str(output_dir / f"{index:05d}.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 92])


def create_panoramas(args: argparse.Namespace, dataset: Path, output_dir: Path, cameras: list[OCam]) -> dict:
    started = time.time()
    dngs = sorted((dataset / "cam").glob("*-cam0.dng"))
    count = len(dngs) if args.max_panos is None else min(len(dngs), args.max_panos)
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = build_panorama_maps(dataset, cameras, args.pano_width)
    weights = feather_weights(maps)
    # 32 full 2K panorama buffers can exceed memory; preserve the requested
    # parallelism up to a safe bound.
    workers = max(1, min(args.num_threads_panos, 8))
    LOG.info("Creating %d panoramas at %dx%d with %d workers", count, args.pano_width, args.pano_width // 2, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda index: stitch_one_panorama(dataset, output_dir, index, cameras, maps, weights), range(count)))
    return {
        "count": count,
        "width": args.pano_width,
        "height": args.pano_width // 2,
        "source": "embedded DNG thumbnails",
        "seconds": time.time() - started,
    }


def create_panoramas_cpp(args: argparse.Namespace, dataset: Path, output_dir: Path) -> dict:
    started = time.time()
    panorama_worker = find_panorama_worker(args.panorama_worker)
    captures = sorted((dataset / "cam").glob("*-cam0.dng"))
    count = len(captures) if args.max_panos is None else min(len(captures), args.max_panos)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_camera_dir = output_dir.parent / "cam"
    if args.aligned_standard:
        processed_camera_dir.mkdir(parents=True, exist_ok=True)
    # Captures use disjoint processes and output paths.
    # The captured G11 run starts all 24 camera images and all six panoramas as
    # outer tasks and peaks at about 42.7 GiB during image post-processing.
    # Six reconstructed captures stay well below that envelope on the target
    # 125 GiB host while matching the original's short-dataset outer topology.
    capture_workers = max(1, min(6, args.num_threads_panos // 5 or 1, count))
    opencv_threads_per_capture = max(1, args.num_threads_panos // capture_workers)
    LOG.info(
        "Creating %d full-resolution C++ panoramas at %dx%d with %d concurrent "
        "captures and %d OpenCV threads per capture",
        count,
        args.pano_width,
        args.pano_width // 2,
        capture_workers,
        opencv_threads_per_capture,
    )

    def process_capture(index: int) -> None:
        capture = f"{index:05d}"
        for camera in range(4):
            source = dataset / "cam" / f"{capture}-cam{camera}.dng"
            if not source.is_file():
                raise FileNotFoundError(source)
        command = [
            str(panorama_worker),
            "--sensor-frame",
            str(dataset / "sensor_frame.xml"),
            "--input-dir",
            str(dataset / "cam"),
            "--metadata-dir",
            str(dataset / "cam"),
            "--capture",
            capture,
            "--width",
            str(args.pano_width),
            "--opencv-threads",
            str(opencv_threads_per_capture),
        ]
        if args.aligned_standard:
            command.extend(["--decoded-dir", str(processed_camera_dir)])
            camera_masks = Path("/opt/NavVis/panorama-rendering/res/g8")
            operator_mask = Path(
                "/opt/NavVis/panorama-rendering/res/g8_operator_mask.png"
            )
            if camera_masks.is_dir():
                command.extend(["--camera-mask-dir", str(camera_masks)])
            if operator_mask.is_file():
                command.extend(["--operator-mask", str(operator_mask)])
            surface_cloud = output_dir.parent / "pointcloud_surface.ply"
            panorama_info = output_dir.parent / "info" / f"{capture}-info.json"
            if surface_cloud.is_file() and panorama_info.is_file():
                command.extend(
                    [
                        "--surface-cloud",
                        str(surface_cloud),
                        "--panorama-info",
                        str(panorama_info),
                    ]
                )
        final_output = output_dir / f"{capture}.jpg"
        for attempt in range(args.panorama_sigfpe_retries + 1):
            attempt_output = output_dir / f".{capture}.attempt-{attempt}.jpg"
            attempt_output.unlink(missing_ok=True)
            attempt_command = [*command, "--output", str(attempt_output)]
            completed = subprocess.run(
                attempt_command,
                check=False,
                stdout=subprocess.DEVNULL,
            )
            if completed.returncode == 0:
                attempt_output.replace(final_output)
                return
            attempt_output.unlink(missing_ok=True)
            can_retry = (
                completed.returncode == -signal.SIGFPE
                and attempt < args.panorama_sigfpe_retries
            )
            if can_retry:
                LOG.warning(
                    "Panorama capture %s terminated with SIGFPE; retrying (%d/%d)",
                    capture,
                    attempt + 1,
                    args.panorama_sigfpe_retries,
                )
                continue
            raise subprocess.CalledProcessError(
                completed.returncode,
                attempt_command,
            )

    with ThreadPoolExecutor(max_workers=capture_workers) as pool:
        list(pool.map(process_capture, range(count)))
    return {
        "count": count,
        "width": args.pano_width,
        "height": args.pano_width // 2,
        "source": "full-resolution DNG; direct LibRaw/DNG-SDK decode; C++ OCam/GraphCut/multiband",
        "worker": str(panorama_worker),
        "capture_workers": capture_workers,
        "opencv_threads_per_capture": opencv_threads_per_capture,
        "sigfpe_retries": args.panorama_sigfpe_retries,
        "processed_camera_directory": str(processed_camera_dir) if args.aligned_standard else None,
        "seconds": time.time() - started,
    }


def pose_at(trajectory: list[Pose], timestamp: float) -> Pose:
    times = np.fromiter((pose.timestamp for pose in trajectory), dtype=np.float64)
    high = int(np.searchsorted(times, timestamp, side="right"))
    if high <= 0:
        return trajectory[0]
    if high >= len(trajectory):
        return trajectory[-1]
    low = high - 1
    alpha = float((timestamp - times[low]) / max(times[high] - times[low], 1e-12))
    translation = tuple(
        (1.0 - alpha) * np.asarray(trajectory[low].translation) +
        alpha * np.asarray(trajectory[high].translation)
    )
    rotation = Slerp(
        [0.0, 1.0],
        Rotation.from_quat([trajectory[low].quaternion_xyzw, trajectory[high].quaternion_xyzw]),
    )([alpha])[0]
    return Pose(timestamp, translation, tuple(rotation.as_quat()))


def pose_json(translation: np.ndarray, rotation: Rotation) -> dict:
    qx, qy, qz, qw = rotation.as_quat()
    return {
        "position": [float(value) for value in translation],
        "quaternion": [float(qw), float(qx), float(qy), float(qz)],
    }


def write_aligned_capture_infos(
    dataset: Path,
    output_directory: Path,
    trajectory: list[Pose],
    cameras: list[OCam],
    count: int,
) -> None:
    """Compose optimized immutable trajectory poses with calibrated camera extrinsics."""
    source_infos = sorted((dataset / "info").glob("*-info.json"))[:count]
    if len(source_infos) != count:
        raise RuntimeError(f"expected {count} source capture infos, found {len(source_infos)}")
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(source_infos):
        source = json.loads(source_path.read_text())
        timestamp = float(source["timestamp"])
        base = pose_at(trajectory, timestamp)
        base_translation = np.asarray(base.translation, dtype=np.float64)
        base_rotation = Rotation.from_quat(base.quaternion_xyzw)
        # tf_trajectory is world_from_base_link/head.  Camera calibration is
        # serialized in cam_head, linked by the fixed transform also used by
        # the validated Pandar path.
        head_from_cam_head_translation = np.array(
            [0.002770609359137844, 0.07619045530812281, -0.03438298002198831]
        )
        head_from_cam_head_rotation = Rotation.from_euler(
            "xyz", [0.0008889239100979543, -0.0015862031285254787, 1.5803277823298127]
        )
        head_translation = base_translation + base_rotation.apply(head_from_cam_head_translation)
        head_rotation = base_rotation * head_from_cam_head_rotation
        data: dict[str, object] = {}
        for camera in cameras:
            camera_rotation = head_rotation * Rotation.from_matrix(camera.rotation)
            camera_translation = head_translation + head_rotation.apply(camera.translation)
            data[camera.name] = pose_json(camera_translation, camera_rotation)
        data["cam_head"] = pose_json(head_translation, head_rotation)
        data["timestamp"] = timestamp
        data["valid"] = "true"
        data["capture_mode"] = source.get("capture_mode", "Manual")
        (output_directory / f"{index:05d}-info.json").write_text(
            json.dumps(data, indent=4) + "\n", encoding="utf-8"
        )


def write_pano_compatibility(output_dir: Path, count: int) -> None:
    """Add the reference filename/layout without duplicating 8K image bytes."""
    panorama_directory = output_dir / "pano"
    panorama_directory.mkdir(parents=True, exist_ok=True)
    pose_lines = [
        "# pano poses v1.0: ID; filename; timestamp; pano_pos_x; pano_pos_y; pano_pos_z; "
        "pano_ori_w; pano_ori_x; pano_ori_y; pano_ori_z"
    ]
    info_names: list[str] = []
    for index in range(count):
        capture = f"{index:05d}"
        source = output_dir / "panoramas" / f"{capture}.jpg"
        destination = panorama_directory / f"{capture}-pano.jpg"
        if destination.exists():
            destination.unlink()
        os.link(source, destination)
        info_name = f"{capture}-info.json"
        info_names.append(info_name)
        info = json.loads((output_dir / "info" / info_name).read_text())
        head = info["cam_head"]
        position = head["position"]
        quaternion = head["quaternion"]
        pose_lines.append(
            f"{index}; {destination.name}; {float(info['timestamp']):.6f}; "
            f"{position[0]:.6f}; {position[1]:.6f}; {position[2]:.6f}; "
            f"{quaternion[0]:.6f}; {quaternion[1]:.6f}; {quaternion[2]:.6f}; "
            f"{quaternion[3]:.6f}"
        )
    (panorama_directory / "pano-poses.csv").write_text(
        "\n".join(pose_lines) + "\n", encoding="utf-8"
    )
    (panorama_directory / "info.list").write_text(
        "\n".join(info_names) + "\n", encoding="utf-8"
    )


def mapped_space_quality_cpp(args: argparse.Namespace, output_dir: Path) -> dict:
    worker = find_stage_worker(
        args.quality_worker,
        "navvis_recon_mapped_space_quality",
        "NAVVIS_RECON_QUALITY",
    )
    mapped_space = output_dir / "mapped_space"
    command = [
        str(worker),
        "--input-shards", str(output_dir / "raw_shards"),
        "--output-dir", str(mapped_space),
        "--grid-resolution", str(args.quality_grid_resolution),
        "--min-num-rays-per-voxel", str(args.quality_min_rays_per_voxel),
        "--use-every-nth-point", str(args.quality_use_every_nth_point),
        "--max-ray-length", "50",
        "--brotli-quality", "5",
    ]
    started = time.time()
    LOG.info("Starting mapped-space quality worker: %s", " ".join(command))
    subprocess.run(command, check=True)
    binary = mapped_space / "quality_voxels.bin"
    sidecar_path = mapped_space / "quality_voxels_sidecar.json"
    pcd = mapped_space / "mapped_space.pcd"
    missing = [str(path) for path in (binary, sidecar_path, pcd) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "mapped-space quality worker returned success without required outputs: "
            + ", ".join(missing)
        )
    sidecar = json.loads(sidecar_path.read_text())
    expected_raw_bytes = int(sidecar["num_voxels"]) * 13
    if sidecar.get("quality_grid_format_version") != 2:
        raise RuntimeError("mapped-space quality sidecar is not format version 2")
    if int(sidecar.get("bytes_uncompressed", -1)) != expected_raw_bytes:
        raise RuntimeError("mapped-space quality sidecar has inconsistent 13-byte record count")
    if int(sidecar.get("bytes_compressed", -1)) != binary.stat().st_size:
        raise RuntimeError("mapped-space quality sidecar has inconsistent compressed byte count")
    return {
        "worker": str(worker),
        "output_directory": str(mapped_space),
        "format_version": 2,
        "voxel_size_m": float(sidecar["voxel_size"]),
        "num_voxels": int(sidecar["num_voxels"]),
        "bytes_uncompressed": expected_raw_bytes,
        "bytes_compressed": binary.stat().st_size,
        "minimum_rays_per_voxel": args.quality_min_rays_per_voxel,
        "use_every_nth_point": args.quality_use_every_nth_point,
        "max_ray_length_m": 50.0,
        "max_ray_length_semantics": "partition bound; rays are not clipped",
        "seconds": time.time() - started,
    }


def filter_surface_cpp(args: argparse.Namespace, output_dir: Path) -> dict:
    worker = find_stage_worker(
        args.surface_worker, "navvis_recon_shard_surface_filter", "NAVVIS_RECON_SURFACE"
    )
    work_directory = output_dir / "surface_work"
    command = [
        str(worker),
        "--input-shards", str(output_dir / "raw_shards"),
        "--output", str(output_dir / "pointcloud_surface.ply"),
        "--work-directory", str(work_directory),
        "--resolution", str(args.res),
        "--output-cell", str(args.res),
        "--free-space-carving",
        "--free-space-mode", args.free_space_mode,
        "--tile-threads", str(args.surface_tile_threads),
        "--preprocess-threads", str(args.surface_preprocess_threads),
    ]
    if args.free_space_mode == "directional":
        # The directional implementation is a diagnostic/non-standard path;
        # its public tuning controls remain available.  Standard sparse mode
        # deliberately uses the constants captured from the original G11
        # FreespaceOctreeOptions constructor inside the C++ worker.
        command.extend([
            "--ray-origin-cell", str(args.ray_origin_cell),
            "--ray-angular-bin-deg", str(args.ray_angular_bin_deg),
            "--free-space-endpoint-margin", str(args.free_space_endpoint_margin),
            "--free-space-min-intersections", str(args.free_space_min_intersections),
            "--free-space-intersection-hit-ratio", str(args.free_space_intersection_hit_ratio),
        ])
    started = time.time()
    surface_environment = os.environ.copy()
    point_threads = max(1, args.num_threads_panos // args.surface_tile_threads)
    surface_environment["OMP_NUM_THREADS"] = str(point_threads)
    surface_environment["OMP_MAX_ACTIVE_LEVELS"] = "1"
    surface_environment["OMP_DYNAMIC"] = "FALSE"
    LOG.info(
        "Starting adaptive C++ surface filter (%d tile workers x %d point threads): %s",
        args.surface_tile_threads,
        point_threads,
        " ".join(command),
    )
    subprocess.run(command, check=True, env=surface_environment)
    return {
        "worker": str(worker),
        "output": str(output_dir / "pointcloud_surface.ply"),
        "output_bytes": (output_dir / "pointcloud_surface.ply").stat().st_size,
        "output_cell_m": args.res,
        "density_filter": "fixed worker defaults; no point-count-derived 8M switch",
        "parallelism": {
            "preprocess_threads": args.surface_preprocess_threads,
            "tile_threads": args.surface_tile_threads,
            "point_threads_per_tile": point_threads,
            "cpu_budget": args.num_threads_panos,
        },
        "free_space_carving": {
            "mode": args.free_space_mode,
            **(
                {
                    "occupancy_resolution_m": 0.02,
                    "minimum_ray_distance_m": 0.5,
                    "maximum_ray_distance_m": 15.0,
                    "ray_to_centroid_m": 0.006,
                    "maximum_incidence_degrees": 85.0,
                    "ray_stride": 1,
                    "endpoint_margin_m": 0.05,
                    "minimum_intersections": 1,
                    "intersection_hit_ratio": 1.0,
                    "global_ray_history": True,
                }
                if args.free_space_mode == "sparse"
                else {
                    "origin_cell_m": args.ray_origin_cell,
                    "angular_bin_degrees": args.ray_angular_bin_deg,
                    "endpoint_margin_m": args.free_space_endpoint_margin,
                    "minimum_intersections": args.free_space_min_intersections,
                    "intersection_hit_ratio": args.free_space_intersection_hit_ratio,
                }
            ),
        },
        "seconds": time.time() - started,
    }


def write_processed_dataset_compatibility(
    args: argparse.Namespace, dataset: Path, output_dir: Path
) -> None:
    """Make the reconstructed output recognizable as a minimal proc-v4 dataset."""
    metadata = json.loads((dataset / "dataset.json").read_text())
    metadata["root"] = {"dataset_type": "proc", "dataset_version_layout": "proc-v4"}
    artifacts = metadata.setdefault("artifacts", {})
    artifacts.update(
        {
            "depth_maps_version": "2",
            "panorama_format": "equirectangular",
            "processed_panoramas": len(list((output_dir / "info").glob("*-info.json"))),
        }
    )
    processing = metadata.setdefault("processing_properties", {})
    processing.update(
        {
            "point_cloud_resolution": args.res,
            "point_cloud_preset": args.preset,
        }
    )
    (output_dir / "dataset.json").write_text(json.dumps(metadata, indent=4))
    shutil.copy2(dataset / "sensor_frame.xml", output_dir / "sensor_frame.xml")
    signature = dataset / "sensor_frame.xml.sig"
    if signature.is_file():
        shutil.copy2(signature, output_dir / signature.name)


def colorize_surface_cpp(args: argparse.Namespace, dataset: Path, output_dir: Path) -> dict:
    use_original = args.color_backend == "original" or (
        args.color_backend == "auto"
        and args.original_colorizer_worker.is_file()
        and os.access(args.original_colorizer_worker, os.X_OK)
    )
    if use_original:
        write_processed_dataset_compatibility(args, dataset, output_dir)
        worker = args.original_colorizer_worker
        if not worker.is_file() or not os.access(worker, os.X_OK):
            raise FileNotFoundError(f"original colorizer is not executable: {worker}")
        command = [
            str(worker),
            f"--input={output_dir / 'pointcloud_surface.ply'}",
            f"--info-dir={output_dir / 'info'}",
            f"--sensor-frame={dataset / 'sensor_frame.xml'}",
            f"--output={output_dir / 'pointcloud.ply'}",
            f"--res={args.res}",
            "--output-type=rgbi",
            "--color-extrapolation=fill",
            "--exposure=global",
            "--reweighting-method=abs",
            "--disable-rolling-shutter-compensation",
        ]
        started = time.time()
        LOG.info("Starting installed NavVis C++ cloud colorizer: %s", " ".join(command))
        subprocess.run(command, check=True)
        return {
            "backend": "original-installed-cpp",
            "worker": str(worker),
            "method": (
                "24 fisheye depth maps, camera masks, five-view voxel ranking, global exposure, "
                "ABS reweighting, rolling shutter disabled, KNN fill"
            ),
            "output": str(output_dir / "pointcloud.ply"),
            "output_bytes": (output_dir / "pointcloud.ply").stat().st_size,
            "seconds": time.time() - started,
        }
    if args.color_backend == "original":
        raise FileNotFoundError(f"original colorizer is not executable: {args.original_colorizer_worker}")
    worker = find_stage_worker(
        args.colorizer_worker, "navvis_recon_surface_colorizer", "NAVVIS_RECON_COLORIZER"
    )
    command = [
        str(worker),
        "--input", str(output_dir / "pointcloud_surface.ply"),
        "--output", str(output_dir / "pointcloud.ply"),
        "--panorama-dir", str(output_dir / "panoramas"),
        "--info-dir", str(output_dir / "info"),
        "--camera-dir", str(output_dir / "cam"),
        "--sensor-frame", str(dataset / "sensor_frame.xml"),
        "--depth-width", "684",
        "--depth-views", "24",
        "--depth-splat-radius", "1",
        "--color-views", "24",
        "--visibility-tolerance", "0.50",
        "--visibility-patch-radius", "1",
        "--visibility-min-fraction", "0.56",
        "--view-max-dist", "30",
        "--direct-blend", "robust",
        "--exposure", "global",
        "--color-extrapolation", "fill",
        "--chunk-points", "1000000",
        "--image-cache", "32",
        "--camera-cache", "6",
    ]
    # The standalone worker defaults to G11 when called without dataset
    # metadata.  The unified runner does have dataset.json, so select the
    # installed mask family explicitly and avoid applying G11 masks to G10
    # recordings.
    metadata = json.loads((dataset / "dataset.json").read_text())
    serial = str(metadata.get("device", {}).get("serial", ""))
    camera_family = "g10" if serial.startswith("G10-") else "g11"
    camera_mask_dir = Path("/opt/NavVis/pointcloud-coloring/res") / camera_family
    if camera_mask_dir.is_dir():
        command.extend(["--camera-mask-dir", str(camera_mask_dir)])
    started = time.time()
    LOG.info("Starting direct-camera C++ cloud colorizer: %s", " ".join(command))
    subprocess.run(command, check=True)
    return {
        "backend": "clean-room-recon-cpp",
        "worker": str(worker),
        "method": (
            "24 binary-aligned 684x456 fisheye depth maps, G11 masks/vignetting, "
            "surface-patch visibility, top-5 robust blending, GammaModel response, "
            "exact five-neighbor geometry-weighted fill"
        ),
        "output": str(output_dir / "pointcloud.ply"),
        "output_bytes": (output_dir / "pointcloud.ply").stat().st_size,
        "seconds": time.time() - started,
    }


PLY_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("intensity", "<f4"),
    ]
)


def open_reconstructed_ply(path: Path) -> tuple[np.memmap, int]:
    header = bytearray()
    with path.open("rb") as stream:
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 64 * 1024:
                raise ValueError(f"invalid PLY header: {path}")
            header.extend(line)
    match = re.search(rb"^element vertex (\d+)$", bytes(header), re.MULTILINE)
    if not match:
        raise ValueError(f"PLY has no vertex count: {path}")
    count = int(match.group(1))
    expected = len(header) + count * PLY_POINT_DTYPE.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"unexpected PLY record layout: expected {expected}, got {path.stat().st_size}")
    points = np.memmap(path, dtype=PLY_POINT_DTYPE, mode="r+", offset=len(header), shape=(count,))
    return points, len(header)


def colorize_cloud_from_panoramas(
    pointcloud: Path,
    panorama_dir: Path,
    captures: list[Pose],
    panorama_count: int,
    workers: int,
) -> dict:
    """Color PLY records in place using their nearest capture panorama.

    This adapter keeps memory bounded and gives the output a useful photo-color
    channel. The standalone C++ point-coloring library contains the more exact
    depth-map/patch/multi-view implementation; this runner uses nearest capture
    equirectangular lookup because only embedded DNG previews are decodable on
    the current host.
    """
    started = time.time()
    selected = captures[:panorama_count]
    if not selected:
        return {"colored_points": 0, "seconds": 0.0, "reason": "no panoramas"}
    tree = cKDTree(np.asarray([pose.translation for pose in selected], dtype=np.float64))
    rotations = np.asarray([quaternion_matrix_xyzw(pose.quaternion_xyzw) for pose in selected])
    translations = np.asarray([pose.translation for pose in selected], dtype=np.float64)
    points, header_bytes = open_reconstructed_ply(pointcloud)
    cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def panorama(index: int) -> np.ndarray:
        cached = cache.pop(index, None)
        if cached is None:
            cached = cv2.imread(str(panorama_dir / f"{index:05d}.jpg"), cv2.IMREAD_COLOR)
            if cached is None:
                raise FileNotFoundError(panorama_dir / f"{index:05d}.jpg")
        cache[index] = cached
        while len(cache) > 8:
            cache.popitem(last=False)
        return cached

    chunk_size = 500_000
    colored = 0
    query_workers = max(1, min(workers, 8))
    for start in range(0, len(points), chunk_size):
        stop = min(len(points), start + chunk_size)
        view = points[start:stop]
        xyz = np.column_stack((view["x"], view["y"], view["z"])).astype(np.float64, copy=False)
        _, nearest = tree.query(xyz, k=1, workers=query_workers)
        rgb = np.empty((len(view), 3), dtype=np.uint8)
        for capture_index in np.unique(nearest):
            mask = nearest == capture_index
            # Pose rotation maps head coordinates to world coordinates. For
            # row vectors, multiplying by R performs R^T * (world - t).
            direction = (xyz[mask] - translations[capture_index]) @ rotations[capture_index]
            horizontal = np.hypot(direction[:, 0], direction[:, 1])
            longitude = np.arctan2(direction[:, 1], direction[:, 0])
            latitude = np.arctan2(direction[:, 2], np.maximum(horizontal, 1e-12))
            image = panorama(int(capture_index))
            height, width = image.shape[:2]
            u = np.mod(np.rint((longitude / (2.0 * np.pi) + 0.5) * width).astype(np.int64), width)
            v = np.clip(np.rint((0.5 - latitude / np.pi) * height).astype(np.int64), 0, height - 1)
            rgb[mask] = image[v, u, ::-1]
        view["red"] = rgb[:, 0]
        view["green"] = rgb[:, 1]
        view["blue"] = rgb[:, 2]
        colored += len(view)
    points.flush()
    return {
        "colored_points": colored,
        "header_bytes": header_bytes,
        "method": "nearest capture equirectangular projection",
        "occlusion": "not depth-tested in the streaming adapter",
        "seconds": time.time() - started,
    }


def validate_dataset(dataset: Path) -> dict:
    required = [dataset / "dataset.json", dataset / "sensor_frame.xml", dataset / "internal" / "bags"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing rec-v4 inputs: " + ", ".join(missing))
    metadata = json.loads((dataset / "dataset.json").read_text())
    if metadata.get("root", {}).get("dataset_version_layout") != "rec-v4":
        raise ValueError("only rec-v4 recordings are supported")
    return metadata


def select_device_profile(metadata: dict) -> dict:
    """Choose packet and carving defaults without overriding explicit CLI values."""
    device = metadata.get("device", {})
    serial = str(device.get("serial", ""))
    if serial.startswith("G10-"):
        return {
            "serial": serial,
            "laser_packet_profile": "Velodyne VLP16",
            "free_space_min_intersections": 6,
            "free_space_intersection_hit_ratio": 3.0,
        }
    return {
        "serial": serial,
        "laser_packet_profile": "Pandar XTM",
        "free_space_min_intersections": 3,
        "free_space_intersection_hit_ratio": 1.5,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.res <= 0
        or args.packet_stride <= 0
        or args.pano_width < 64
        or args.num_threads_panos <= 0
        or args.panorama_sigfpe_retries < 0
        or args.surface_tile_threads <= 0
        or args.surface_preprocess_threads <= 0
        or args.quality_grid_resolution <= 0
        or args.quality_min_rays_per_voxel < 0
        or args.quality_use_every_nth_point <= 0
    ):
        raise ValueError("resolution/packet stride/panorama width/thread counts are invalid")
    if args.start_offset < 0 or (args.max_duration is not None and args.max_duration <= 0):
        raise ValueError("start offset must be non-negative and max duration must be positive")
    dataset = args.dataset.resolve()
    metadata = validate_dataset(dataset)
    device_profile = select_device_profile(metadata)
    if args.free_space_min_intersections is None:
        args.free_space_min_intersections = device_profile["free_space_min_intersections"]
    if args.free_space_intersection_hit_ratio is None:
        args.free_space_intersection_hit_ratio = device_profile["free_space_intersection_hit_ratio"]
    if (
        args.ray_origin_cell <= 0
        or args.ray_angular_bin_deg <= 0
        or args.free_space_traversal_resolution <= 0
        or args.free_space_ray_radius < 0
        or args.free_space_ray_stride <= 0
        or args.free_space_endpoint_margin < 0
        or args.free_space_min_intersections < 0
        or args.free_space_intersection_hit_ratio < 0
    ):
        raise ValueError("free-space carving parameters are invalid")
    if args.cloud_roi:
        values = [float(value) for value in args.cloud_roi.split(",")]
        if len(values) != 6 or any(values[index + 3] <= values[index] for index in range(3)):
            raise ValueError("cloud ROI must be minx,miny,minz,maxx,maxy,maxz")
    if args.surface_min_neighbors is not None and args.surface_min_neighbors < 0:
        raise ValueError("surface minimum neighbors must be non-negative")
    if args.aligned_standard and args.preset != "standard":
        raise ValueError("--aligned-standard is calibrated only for --preset=standard")
    dataset_id = metadata["dataset"]["dataset_id"]
    output_dir = (args.output_dir or args.proc_base_dir / dataset_id / "recon").resolve()
    marker = output_dir / "processing_report.json"
    if marker.exists() and not args.force:
        raise FileExistsError(f"{output_dir} already processed; pass --force")
    if args.aligned_standard and args.force:
        # These names are owned exclusively by this runner.  Keep unrelated
        # files in a user-supplied output directory intact.
        for name in (
            "raw_shards",
            "surface_work",
            "mapped_space",
            "panoramas",
            "pano",
            "cam",
            "info",
        ):
            generated = output_dir / name
            if generated.is_dir():
                shutil.rmtree(generated)
        for name in (
            "pointcloud.ply",
            "pointcloud_surface.ply",
            "pointcloud_raw.ply",
            "floors.json",
        ):
            generated = output_dir / name
            if generated.is_file():
                generated.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, args.log_file)
    started = time.time()
    LOG.info("Starting independent processing for %s", dataset)
    if args.free_space_mode == "sparse":
        LOG.info(
            "Device %s uses %s packets; standard sparse free-space uses captured G11 "
            "thresholds intersections=1, ratio=1, occupancy=0.02m, ray radius=0.006m",
            device_profile["serial"] or "unknown",
            device_profile["laser_packet_profile"],
        )
    else:
        LOG.info(
            "Device %s uses %s packets; diagnostic directional free-space thresholds are "
            "intersections=%d, ratio=%.3g",
            device_profile["serial"] or "unknown",
            device_profile["laser_packet_profile"],
            args.free_space_min_intersections,
            args.free_space_intersection_hit_ratio,
        )
    laser_poses, cameras = read_sensor_frame(dataset / "sensor_frame.xml")
    trajectory, trajectory_metadata = combined_trajectory(
        dataset, args.trajectory_bag, args.slam_mode, args.trajectory_csv
    )
    trajectory_path = output_dir / "trajectory.csv"
    write_trajectory(trajectory_path, trajectory)
    floor_report = estimate_and_write_floors(
        dataset, trajectory, output_dir / "artifacts"
    )

    report = {
        "implementation": "navvis_postprocessing_reconstruction",
        "dataset": dataset_id,
        "caller": args.caller,
        "preset": args.preset,
        "resolution": args.res,
        "cloud_format": args.cloud_format,
        "device": {
            "serial": device_profile["serial"],
            "calibration_version": metadata.get("device", {}).get("calibration_version_rec"),
            "laser_packet_profile": device_profile["laser_packet_profile"],
        },
        "trajectory": trajectory_metadata,
        "floor": floor_report,
        "slam": {
            "mode": trajectory_metadata.get("slam_mode"),
            "source": trajectory_metadata.get("source"),
            "offline_frontend_recomputed": trajectory_metadata.get(
                "offline_frontend_recomputed", False
            ),
            "loop_closures_recomputed": trajectory_metadata.get(
                "loop_closures_recomputed", False
            ),
            "imu_pose_graph_recomputed": trajectory_metadata.get(
                "imu_pose_graph_recomputed", False
            ),
        },
    }
    if args.slam_reference_bag:
        reference_poses = read_trajectory_bag(args.slam_reference_bag.resolve())
        if len(reference_poses) < 2:
            raise RuntimeError(
                f"No usable base_link trajectory in SLAM reference bag: "
                f"{args.slam_reference_bag}"
            )
        report["slam"]["reference_bag"] = str(args.slam_reference_bag.resolve())
        report["slam"]["evaluation"] = evaluate_trajectory(
            _slam_trajectory(trajectory), _slam_trajectory(reference_poses)
        )
    if not args.skip_cloud:
        report["cloud"] = stream_cloud(
            args,
            dataset,
            output_dir / ("pointcloud_raw.ply" if args.aligned_standard else "pointcloud.ply"),
            trajectory_path,
            laser_poses,
            trajectory[0].timestamp,
            trajectory[-1].timestamp,
        )
        if args.aligned_standard:
            report["mapped_space_quality"] = mapped_space_quality_cpp(args, output_dir)
            report["surface"] = filter_surface_cpp(args, output_dir)
    if args.aligned_standard and not args.skip_panos:
        panorama_count = len(list((dataset / "cam").glob("*-cam0.dng")))
        if args.max_panos is not None:
            panorama_count = min(panorama_count, args.max_panos)
        write_aligned_capture_infos(
            dataset, output_dir / "info", trajectory, cameras, panorama_count
        )
    if not args.skip_panos:
        try:
            report["panoramas"] = create_panoramas_cpp(args, dataset, output_dir / "panoramas")
        except FileNotFoundError as error:
            LOG.warning("Full-resolution C++ panorama path unavailable (%s); using thumbnail fallback", error)
            report["panoramas"] = create_panoramas(args, dataset, output_dir / "panoramas", cameras)
    if args.aligned_standard and not args.skip_panos:
        write_pano_compatibility(output_dir, report["panoramas"]["count"])
    if not args.skip_cloud and not args.skip_panos and not args.skip_coloring:
        if args.aligned_standard:
            report["cloud_coloring"] = colorize_surface_cpp(args, dataset, output_dir)
        else:
            report["cloud_coloring"] = colorize_cloud_from_panoramas(
                output_dir / "pointcloud.ply",
                output_dir / "panoramas",
                capture_poses(dataset),
                report["panoramas"]["count"],
                args.num_threads_panos,
            )
    report["elapsed_seconds"] = time.time() - started
    report["completed"] = True
    marker.write_text(json.dumps(report, indent=2))
    if not (output_dir / "dataset.json").is_file():
        write_processed_dataset_compatibility(args, dataset, output_dir)
    LOG.info("Completed reconstructed processing in %.1f seconds: %s", report["elapsed_seconds"], output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        raise SystemExit(130)
    except Exception as error:
        LOG.exception("Processing failed: %s", error)
        raise SystemExit(1)
