#!/usr/bin/env python3
"""Build the raw SLAM laser stream from a warm-up prefix and control capture.

CloudBuilder captures mark scans outside ``add_scans`` windows by setting the
high bit of the scan-marker sensor byte.  SLAM consumes those scans regardless
of CloudBuilder insertion state.  This filter prepends complete warm-up scans
that precede the main capture and clears that control bit while streaming the
result to stdout.  Packet payloads and nanosecond timestamps are not changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
from typing import BinaryIO, Iterator


FRAME_HEADER = struct.Struct("<BqH")


def frames(stream: BinaryIO) -> Iterator[tuple[int, int, bytes]]:
    while True:
        header = stream.read(FRAME_HEADER.size)
        if not header:
            return
        if len(header) != FRAME_HEADER.size:
            raise ValueError("truncated Pandar frame header")
        sensor, timestamp_ns, payload_size = FRAME_HEADER.unpack(header)
        payload = stream.read(payload_size)
        if len(payload) != payload_size:
            raise ValueError("truncated Pandar frame payload")
        yield sensor, timestamp_ns, payload


def write_frame(
    output: BinaryIO, sensor: int, timestamp_ns: int, payload: bytes
) -> None:
    output.write(FRAME_HEADER.pack(sensor, timestamp_ns, len(payload)))
    output.write(payload)


def write_warmup_scans(
    source: Path, output: BinaryIO, cutoff_ns: int
) -> tuple[int, int]:
    scan_count = 0
    frame_count = 0
    include_scan = False
    with source.open("rb") as stream:
        for sensor, timestamp_ns, payload in frames(stream):
            if not payload:
                include_scan = timestamp_ns < cutoff_ns
                if not include_scan:
                    break
                scan_count += 1
            if include_scan:
                write_frame(output, sensor & 0x7F, timestamp_ns, payload)
                frame_count += 1
    return scan_count, frame_count


def write_main_stream(source: Path, output: BinaryIO) -> tuple[int, int]:
    scan_count = 0
    frame_count = 0
    with source.open("rb") as stream:
        for sensor, timestamp_ns, payload in frames(stream):
            if not payload:
                scan_count += 1
            write_frame(output, sensor & 0x7F, timestamp_ns, payload)
            frame_count += 1
    return scan_count, frame_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument(
        "--main-first-scan-ns",
        type=int,
        required=True,
        help="Exclude warm-up scans at or after this main-stream scan marker",
    )
    args = parser.parse_args()

    output = sys.stdout.buffer
    warmup_scans, warmup_frames = write_warmup_scans(
        args.warmup, output, args.main_first_scan_ns
    )
    main_scans, main_frames = write_main_stream(args.main, output)
    print(
        f"warmup scans={warmup_scans}, frames={warmup_frames}; "
        f"main scans={main_scans}, frames={main_frames}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
