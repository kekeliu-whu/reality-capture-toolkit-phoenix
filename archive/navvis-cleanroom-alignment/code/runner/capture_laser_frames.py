#!/usr/bin/env python3
"""Capture the deterministic framed laser stream consumed by navvis_recon_pandar.

This turns bag I/O into a one-time benchmark setup cost.  Replaying the resulting
file isolates CloudBuilder CPU and shard-writing performance from rosbag/Python
overhead while preserving scan boundaries and packet order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.recording_io import (  # noqa: E402
    chronological_laser_scans,
    world_builder_active_windows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trajectory-start", type=float, required=True)
    parser.add_argument("--trajectory-end", type=float, required=True)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument(
        "--ignore-control-events",
        action="store_true",
        help="Capture raw SLAM input even while world_builder/add_scans is disabled",
    )
    parser.add_argument(
        "--preserve-control-state",
        action="store_true",
        help=(
            "Include disabled scans with the frame-marker high bit set so the "
            "CloudBuilder no-motion state advances without emitting their points"
        ),
    )
    parser.add_argument(
        "--timestamps-ns",
        action="store_true",
        help="Write exact int64 ROS nanoseconds instead of float64 seconds",
    )
    parser.add_argument(
        "--sensor",
        choices=("both", "horiz", "vert"),
        default="both",
        help="Capture both laser topics or isolate one while preserving sensor IDs",
    )
    args = parser.parse_args()
    if args.ignore_control_events and args.preserve_control_state:
        parser.error(
            "--ignore-control-events and --preserve-control-state are mutually exclusive"
        )

    window_start = min(args.trajectory_end, args.trajectory_start + args.start_offset)
    window_end = min(args.trajectory_end, window_start + args.duration)
    control_windows = (
        [(args.trajectory_start, args.trajectory_end)]
        if args.ignore_control_events
        else world_builder_active_windows(
            args.dataset, args.trajectory_start, args.trajectory_end
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    packet_count = 0
    scan_count = 0
    state_packet_count = 0
    state_scan_count = 0
    with args.output.open("wb") as output:
        sensors = tuple(
            (sensor_id, sensor_name)
            for sensor_id, sensor_name in enumerate(("horiz", "vert"))
            if args.sensor in ("both", sensor_name)
        )
        for sensor_id, _, _, message in chronological_laser_scans(args.dataset, sensors):
            scan_stamp = message.header.stamp.to_sec()
            if scan_stamp > window_end:
                break
            packets = [
                packet
                for packet in message.packets
                if packet.stamp.to_sec() > 0 and len(packet.data) in (820, 1206)
            ]
            if not packets:
                continue
            scan_end = max(packet.stamp.to_sec() for packet in packets)
            if not (window_start <= scan_stamp and scan_end <= window_end):
                continue
            insertion_enabled = any(
                start <= scan_stamp < end for start, end in control_windows
            )
            if not insertion_enabled and not args.preserve_control_state:
                continue
            scan_timestamp = (
                int(message.header.stamp.secs) * 1_000_000_000
                + int(message.header.stamp.nsecs)
                if args.timestamps_ns
                else scan_stamp
            )
            frame_format = "<BqH" if args.timestamps_ns else "<BdH"
            marker_sensor_id = sensor_id | (0 if insertion_enabled else 0x80)
            output.write(struct.pack(frame_format, marker_sensor_id, scan_timestamp, 0))
            state_scan_count += 1
            scan_count += int(insertion_enabled)
            for packet in packets:
                stamp = packet.stamp.to_sec()
                packet_timestamp = (
                    int(packet.stamp.secs) * 1_000_000_000
                    + int(packet.stamp.nsecs)
                    if args.timestamps_ns
                    else stamp
                )
                payload = bytes(packet.data)
                output.write(
                    struct.pack(frame_format, sensor_id, packet_timestamp, len(payload))
                )
                output.write(payload)
                state_packet_count += 1
                packet_count += int(insertion_enabled)

    print(
        f"captured {scan_count} enabled scans and {packet_count} enabled packets; "
        f"motion state saw {state_scan_count} scans and {state_packet_count} packets "
        f"to {args.output} ({args.output.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
