#!/usr/bin/env python3
"""Extract selected full Pandar scans using an indexed probe as the key set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import rosbag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.recording_io import numeric_bags  # noqa: E402


def parse_indices(text: str) -> list[int]:
    indices = sorted({int(value) for value in text.split(",") if value})
    if not indices or indices[0] <= 0:
        raise ValueError("scan indices must be positive and one-based")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("selection_bag", type=Path)
    parser.add_argument("output_bag", type=Path)
    parser.add_argument("--sensor", choices=("horiz", "vert"), required=True)
    parser.add_argument(
        "--indices", required=True, help="comma-separated one-based indices in selection_bag"
    )
    args = parser.parse_args()
    try:
        selected_indices = parse_indices(args.indices)
    except ValueError as error:
        parser.error(str(error))

    topic = f"/laser_{args.sensor}/packets"
    selected_timestamps: set[int] = set()
    wanted = set(selected_indices)
    with rosbag.Bag(str(args.selection_bag)) as selection:
        for index, (_, message, _) in enumerate(
            selection.read_messages(topics=[topic]), start=1
        ):
            if index in wanted:
                selected_timestamps.add(message.header.stamp.to_nsec())
    if len(selected_timestamps) != len(selected_indices):
        raise RuntimeError(
            f"selected {len(selected_timestamps)} of {len(selected_indices)} requested scans"
        )

    args.output_bag.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    packets = 0
    with rosbag.Bag(str(args.output_bag), "w") as output:
        for input_path in numeric_bags(args.dataset, args.sensor):
            with rosbag.Bag(str(input_path)) as source:
                for _, message, bag_time in source.read_messages(topics=[topic]):
                    if message.header.stamp.to_nsec() not in selected_timestamps:
                        continue
                    output.write(topic, message, bag_time)
                    copied += 1
                    packets += len(message.packets)
    if copied != len(selected_timestamps):
        raise RuntimeError(
            f"found {copied} of {len(selected_timestamps)} selected scans in raw bags"
        )
    print(f"copied {copied} scans/{packets} packets -> {args.output_bag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
