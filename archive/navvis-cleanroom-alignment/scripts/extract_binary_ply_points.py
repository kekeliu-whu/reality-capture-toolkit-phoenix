#!/usr/bin/env python3
"""Extract selected records from a binary little-endian PLY without changing bytes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-bytes", type=int, required=True)
    parser.add_argument("indices", nargs="+", type=int)
    args = parser.parse_args()

    with args.input.open("rb") as source:
        header_lines: list[bytes] = []
        vertex_count = None
        while True:
            line = source.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            match = re.fullmatch(rb"element vertex ([0-9]+)\r?\n", line)
            if match:
                vertex_count = int(match.group(1))
                line = f"element vertex {len(args.indices)}\n".encode()
            header_lines.append(line)
            if line.rstrip() == b"end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY header has no vertex element")
        data_offset = source.tell()
        for index in args.indices:
            if index < 0 or index >= vertex_count:
                raise IndexError(index)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output:
            output.writelines(header_lines)
            for index in args.indices:
                source.seek(data_offset + index * args.record_bytes)
                record = source.read(args.record_bytes)
                if len(record) != args.record_bytes:
                    raise EOFError(index)
                output.write(record)


if __name__ == "__main__":
    main()
