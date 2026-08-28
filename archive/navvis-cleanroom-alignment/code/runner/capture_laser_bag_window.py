#!/usr/bin/env python3
"""Freeze the same complete Pandar scans into small per-sensor ROS bags.

The CloudBuilder command-line start/stop bounds and the clean-room framed
stream intentionally use different boundary semantics.  A frozen bag avoids
that ambiguity when an installed binary is used as a small, read-only oracle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import rosbag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.recording_io import numeric_bags  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--stop", type=float, required=True)
    args = parser.parse_args()
    if args.stop <= args.start:
        parser.error("--stop must be greater than --start")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for sensor_name in ("horiz", "vert"):
        topic = f"/laser_{sensor_name}/packets"
        output_path = args.output_directory / f"laser_{sensor_name}.bag"
        scan_count = 0
        packet_count = 0
        with rosbag.Bag(str(output_path), "w") as output:
            for input_path in numeric_bags(args.dataset, sensor_name):
                with rosbag.Bag(str(input_path)) as source:
                    if source.get_end_time() < args.start:
                        continue
                    if source.get_start_time() > args.stop:
                        break
                    for _, message, bag_time in source.read_messages(topics=[topic]):
                        scan_start = message.header.stamp.to_sec()
                        valid_packets = [
                            packet
                            for packet in message.packets
                            if packet.stamp.to_sec() > 0 and len(packet.data) in (820, 1206)
                        ]
                        if not valid_packets:
                            continue
                        scan_end = max(packet.stamp.to_sec() for packet in valid_packets)
                        if args.start <= scan_start and scan_end <= args.stop:
                            output.write(topic, message, bag_time)
                            scan_count += 1
                            packet_count += len(valid_packets)
        print(
            f"{sensor_name}: {scan_count} scans, {packet_count} packets -> {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
