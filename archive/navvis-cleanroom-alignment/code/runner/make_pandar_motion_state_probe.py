#!/usr/bin/env python3
"""Create a small CloudBuilder oracle input with complete no-motion history.

Every selected Pandar scan remains present with its original header stamp, but
only the first valid packet is retained.  CloudBuilder's no-motion decision is
scan-pose based, so this preserves the state machine while keeping PLY oracle
outputs small.  The script writes both a ROS bag for the installed reference
and the exact-nanosecond framed stream used by navvis_recon_pandar.
"""

from __future__ import annotations

import argparse
import copy
import gzip
from pathlib import Path
import struct
import sys

import rosbag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.recording_io import (  # noqa: E402
    numeric_bags,
    world_builder_active_windows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--sensor", choices=("horiz", "vert"), required=True)
    parser.add_argument("--trajectory-start", type=float, required=True)
    parser.add_argument("--trajectory-end", type=float, required=True)
    parser.add_argument("--stop-offset", type=float, required=True)
    parser.add_argument(
        "--preserve-packet-timing",
        action="store_true",
        help=(
            "Keep every packet/timestamp but zero ranges after the first packet; "
            "this preserves converter timing while making oracle PLY output small"
        ),
    )
    parser.add_argument(
        "--edge-packets",
        action="store_true",
        help="Keep the first and last packet of each scan instead of only the first",
    )
    args = parser.parse_args()
    if args.preserve_packet_timing and args.edge_packets:
        parser.error("--preserve-packet-timing and --edge-packets are mutually exclusive")

    stop = min(args.trajectory_end, args.trajectory_start + args.stop_offset)
    active_windows = world_builder_active_windows(
        args.dataset, args.trajectory_start, args.trajectory_end
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    bag_path = args.output_directory / f"laser_{args.sensor}.bag"
    frames_path = args.output_directory / (
        f"frames_{args.sensor}_timed_ns.bin.gz"
        if args.preserve_packet_timing
        else f"frames_{args.sensor}_ns.bin"
    )
    topic = f"/laser_{args.sensor}/packets"
    sensor_id = 0 if args.sensor == "horiz" else 1

    scans = 0
    packets = 0
    enabled_scans = 0
    compression = (
        rosbag.Compression.BZ2
        if args.preserve_packet_timing
        else rosbag.Compression.NONE
    )
    frames_context = (
        gzip.open(frames_path, "wb")
        if args.preserve_packet_timing
        else frames_path.open("wb")
    )
    with rosbag.Bag(str(bag_path), "w", compression=compression) as output_bag, frames_context as frames:
        for input_path in numeric_bags(args.dataset, args.sensor):
            with rosbag.Bag(str(input_path)) as source:
                if source.get_end_time() < args.trajectory_start:
                    continue
                if source.get_start_time() > stop:
                    break
                for _, message, bag_time in source.read_messages(topics=[topic]):
                    scan_start = message.header.stamp.to_sec()
                    if scan_start > stop:
                        break
                    valid_packets = [
                        packet
                        for packet in message.packets
                        if packet.stamp.to_sec() > 0 and len(packet.data) == 820
                    ]
                    if not valid_packets:
                        continue
                    scan_end = max(packet.stamp.to_sec() for packet in valid_packets)
                    if not (args.trajectory_start <= scan_start and scan_end <= stop):
                        continue

                    thinned = copy.deepcopy(message)
                    if args.preserve_packet_timing:
                        thinned.packets = copy.deepcopy(valid_packets)
                        for packet_index, packet in enumerate(thinned.packets):
                            if packet_index == 0:
                                continue
                            payload = bytearray(packet.data)
                            for block in range(6):
                                block_offset = 12 + block * 130
                                for ring in range(32):
                                    distance_offset = block_offset + 2 + ring * 4
                                    payload[distance_offset] = 0
                                    payload[distance_offset + 1] = 0
                            packet.data = bytes(payload)
                    else:
                        # Real edge packets guarantee a non-empty decoded scan.
                        # Keeping the last packet also preserves converters
                        # that derive the output cloud stamp from scan extent.
                        selected_packets = (
                            [valid_packets[0], valid_packets[-1]]
                            if args.edge_packets and len(valid_packets) > 1
                            else [valid_packets[0]]
                        )
                        thinned.packets = copy.deepcopy(selected_packets)
                    output_bag.write(topic, thinned, bag_time)

                    insertion_enabled = any(
                        begin <= scan_start < end for begin, end in active_windows
                    )
                    marker = sensor_id | (0 if insertion_enabled else 0x80)
                    scan_ns = (
                        int(message.header.stamp.secs) * 1_000_000_000
                        + int(message.header.stamp.nsecs)
                    )
                    frames.write(struct.pack("<BqH", marker, scan_ns, 0))
                    for packet in thinned.packets:
                        packet_ns = (
                            int(packet.stamp.secs) * 1_000_000_000
                            + int(packet.stamp.nsecs)
                        )
                        payload = bytes(packet.data)
                        frames.write(
                            struct.pack("<BqH", sensor_id, packet_ns, len(payload))
                        )
                        frames.write(payload)
                    scans += 1
                    packets += len(thinned.packets)
                    enabled_scans += int(insertion_enabled)

    print(
        f"{args.sensor}: {scans} state scans ({enabled_scans} enabled), "
        f"{packets} packets -> {bag_path} and {frames_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
