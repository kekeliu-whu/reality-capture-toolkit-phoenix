#!/usr/bin/env python3
"""Extract byte-exact records while preserving a NavVis binary PLY header."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_count(line: bytes, prefix: bytes, count: int) -> bytes:
    match = re.fullmatch(re.escape(prefix) + rb"([0-9]+)(\r?\n)", line)
    if not match:
        return line
    width = len(match.group(1))
    return prefix + f"{count:0{width}d}".encode() + match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-bytes", type=int, required=True)
    parser.add_argument("indices", nargs="+", type=int)
    args = parser.parse_args()

    with args.input.open("rb") as source:
        header: list[bytes] = []
        vertex_count: int | None = None
        while True:
            line = source.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            match = re.fullmatch(rb"element vertex ([0-9]+)\r?\n", line)
            if match:
                vertex_count = int(match.group(1))
            line = replace_count(line, b"element vertex ", len(args.indices))
            line = replace_count(line, b"obj_info num_cols ", len(args.indices))
            header.append(line)
            if line.rstrip() == b"end_header":
                break

        if vertex_count is None:
            raise ValueError("PLY header has no vertex element")
        data_offset = source.tell()
        if any(index < 0 or index >= vertex_count for index in args.indices):
            raise IndexError("point index outside vertex array")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output:
            output.writelines(header)
            for index in args.indices:
                source.seek(data_offset + index * args.record_bytes)
                record = source.read(args.record_bytes)
                if len(record) != args.record_bytes:
                    raise EOFError(index)
                output.write(record)


if __name__ == "__main__":
    main()
