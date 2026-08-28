#!/usr/bin/env python3
"""Split an exact scan prefix from a Pandar CloudBuilder motion probe.

The reference ROS bag and the clean-room framed stream must contain precisely
the same scan set.  Using cloud_builder's ``--stop-time`` is not sufficient for
this purpose because its bag playback boundary is not the scan header boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

import rosbag


FRAME_HEADER = struct.Struct("<BqH")


def copy_bag_prefix(source: Path, destination: Path, scan_limit: int) -> tuple[int, str]:
    copied = 0
    topic_name = ""
    with rosbag.Bag(str(source)) as input_bag, rosbag.Bag(
        str(destination), "w", compression=rosbag.Compression.NONE
    ) as output_bag:
        for topic, message, bag_time in input_bag.read_messages():
            if copied >= scan_limit:
                break
            output_bag.write(topic, message, bag_time)
            copied += 1
            if not topic_name:
                topic_name = topic
            elif topic != topic_name:
                raise RuntimeError(
                    f"motion probe bag contains multiple topics: {topic_name}, {topic}"
                )
    return copied, topic_name


def read_frame(stream) -> bytes | None:
    header = stream.read(FRAME_HEADER.size)
    if not header:
        return None
    if len(header) != FRAME_HEADER.size:
        raise RuntimeError("truncated framed-stream header")
    _, _, payload_size = FRAME_HEADER.unpack(header)
    payload = stream.read(payload_size)
    if len(payload) != payload_size:
        raise RuntimeError("truncated framed-stream payload")
    return header + payload


def write_scan(output_stream, frames: list[bytes], marker_timestamp: str) -> int:
    if marker_timestamp != "original" and len(frames) > 1:
        marker_sensor, _, _ = FRAME_HEADER.unpack_from(frames[0])
        packet_index = 1 if marker_timestamp == "first-packet" else -1
        _, packet_timestamp, _ = FRAME_HEADER.unpack_from(frames[packet_index])
        frames[0] = FRAME_HEADER.pack(marker_sensor, packet_timestamp, 0)
    output_stream.writelines(frames)
    return len(frames) - 1


def copy_frame_prefix(
    source: Path, destination: Path, scan_limit: int, marker_timestamp: str
) -> tuple[int, int]:
    copied_scans = 0
    copied_packets = 0
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        pending_scan: list[bytes] = []
        while True:
            frame = read_frame(input_stream)
            if frame is None:
                if pending_scan and copied_scans < scan_limit:
                    copied_packets += write_scan(
                        output_stream, pending_scan, marker_timestamp
                    )
                    copied_scans += 1
                break

            _, _, payload_size = FRAME_HEADER.unpack_from(frame)
            if payload_size == 0:
                if pending_scan:
                    copied_packets += write_scan(
                        output_stream, pending_scan, marker_timestamp
                    )
                    copied_scans += 1
                    if copied_scans >= scan_limit:
                        break
                pending_scan = [frame]
            else:
                if not pending_scan:
                    raise RuntimeError("packet frame encountered before the first scan marker")
                pending_scan.append(frame)

    return copied_scans, copied_packets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_bag", type=Path)
    parser.add_argument("source_frames", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--scans", type=int, required=True)
    parser.add_argument(
        "--marker-timestamp",
        choices=("original", "first-packet", "last-packet"),
        default="original",
        help="Diagnostic override for the clean-room scan-pose timestamp",
    )
    args = parser.parse_args()
    if args.scans <= 0:
        parser.error("--scans must be positive")
    if args.source_frames.suffix == ".gz":
        parser.error("gzip framed streams are not supported by exact prefix splitting")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_bag = args.output_directory / args.source_bag.name
    output_frames = args.output_directory / args.source_frames.name

    bag_scans, topic = copy_bag_prefix(args.source_bag, output_bag, args.scans)
    frame_scans, frame_packets = copy_frame_prefix(
        args.source_frames, output_frames, args.scans, args.marker_timestamp
    )
    if bag_scans != args.scans or frame_scans != args.scans:
        raise RuntimeError(
            f"requested {args.scans} scans, copied bag={bag_scans}, frames={frame_scans}"
        )
    print(
        f"copied {bag_scans} scans on {topic}; framed packets={frame_packets}; "
        f"bag={output_bag}; frames={output_frames}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
