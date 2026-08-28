#!/usr/bin/env python3
"""Stream raw rec-v4 Pandar scans into the clean-room NVSLAM6 archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.recording_io import (  # noqa: E402
    chronological_laser_scans,
    read_laser_poses,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an NVSLAM6 frontend archive from the two raw laser topics"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pandar-worker", type=Path, required=True)
    parser.add_argument(
        "--max-scans",
        type=int,
        help="Stop after this many chronological laser messages (regression only)",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    worker = args.pandar_worker.resolve()
    if args.max_scans is not None and args.max_scans < 1:
        raise ValueError("--max-scans must be positive")
    if not worker.is_file():
        raise FileNotFoundError(worker)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        if not output.is_file():
            raise ValueError(f"refusing to replace non-file output: {output}")
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    laser_poses = read_laser_poses(dataset / "sensor_frame.xml")
    command = [
        str(worker),
        "--frame-timestamps-ns",
        "--slam-scans-output",
        str(output),
        "--horiz-pose",
        laser_poses["laser_horiz"],
        "--vert-pose",
        laser_poses["laser_vert"],
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("could not open Pandar worker stdin")

    scan_count = 0
    packet_count = 0
    try:
        for sensor_id, _, _, message in chronological_laser_scans(
            dataset, enumerate(("horiz", "vert"))
        ):
            if args.max_scans is not None and scan_count >= args.max_scans:
                break
            packets = [
                packet
                for packet in message.packets
                if packet.stamp.to_nsec() > 0 and len(packet.data) == 820
            ]
            if not packets:
                continue
            process.stdin.write(
                struct.pack("<BqH", sensor_id, message.header.stamp.to_nsec(), 0)
            )
            scan_count += 1
            for packet in packets:
                payload = bytes(packet.data)
                process.stdin.write(
                    struct.pack("<BqH", sensor_id, packet.stamp.to_nsec(), len(payload))
                )
                process.stdin.write(payload)
                packet_count += 1
        process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"Pandar worker exited with status {return_code}")
    except BaseException:
        try:
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        process.terminate()
        process.wait()
        raise

    print(
        f"generated {output}: chronological_scans={scan_count}, packets={packet_count}, "
        f"bytes={output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
