#!/usr/bin/env python3
"""Fuse/evaluate recorded SLAM trajectories against a frozen reference bag."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rosbag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.slam_reconstruction import (  # noqa: E402
    Trajectory,
    evaluate_trajectory,
    fuse_global_and_local,
)


def read_bag(path: Path) -> Trajectory:
    rows: list[tuple[float, list[float], list[float]]] = []
    with rosbag.Bag(str(path)) as bag:
        topics = bag.get_type_and_topic_info().topics
        topic = "trajectory" if "trajectory" in topics else "tf_trajectory"
        for _, message, _ in bag.read_messages(topics=[topic]):
            if topic == "trajectory":
                stamp = message.header.stamp
                translation = message.pose.position
                rotation = message.pose.orientation
            else:
                transforms = [
                    transform
                    for transform in message.transforms
                    if transform.child_frame_id == "base_link"
                ]
                if not transforms:
                    continue
                transform = transforms[0]
                stamp = transform.header.stamp
                translation = transform.transform.translation
                rotation = transform.transform.rotation
            rows.append(
                (
                    stamp.to_sec(),
                    [translation.x, translation.y, translation.z],
                    [rotation.x, rotation.y, rotation.z, rotation.w],
                )
            )
    rows.sort(key=lambda row: row[0])
    rows = [row for index, row in enumerate(rows) if index == 0 or row[0] > rows[index - 1][0]]
    return Trajectory(
        np.asarray([row[0] for row in rows]),
        np.asarray([row[1] for row in rows]),
        np.asarray([row[2] for row in rows]),
    )


def write_csv(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["# timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for timestamp, translation, quaternion in zip(
            trajectory.timestamps,
            trajectory.translations,
            trajectory.quaternions_xyzw,
        ):
            writer.writerow((f"{timestamp:.9f}", *translation, *quaternion))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-bag", type=Path, required=True)
    parser.add_argument("--local-bag", type=Path)
    parser.add_argument("--reference-bag", type=Path)
    parser.add_argument("--upsampling-factor", type=int, default=5)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    global_slam = read_bag(args.global_bag)
    if args.local_bag:
        trajectory = fuse_global_and_local(
            global_slam,
            read_bag(args.local_bag),
            upsampling_factor=args.upsampling_factor,
        )
        method = "recorded online SLAM + interpolated map<-odom correction"
    else:
        trajectory = global_slam
        method = "recorded online SLAM"
    report: dict[str, object] = {
        "method": method,
        "pose_count": len(trajectory.timestamps),
        "start": float(trajectory.timestamps[0]),
        "end": float(trajectory.timestamps[-1]),
        "upsampling_factor": args.upsampling_factor if args.local_bag else 1,
        "offline_frontend_recomputed": False,
        "loop_closures_recomputed": False,
        "imu_pose_graph_recomputed": False,
    }
    if args.reference_bag:
        report["reference"] = str(args.reference_bag)
        report["evaluation"] = evaluate_trajectory(
            trajectory, read_bag(args.reference_bag)
        )
    if args.output_csv:
        write_csv(args.output_csv, trajectory)
        report["output_csv"] = str(args.output_csv)
    encoded = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
