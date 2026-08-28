#!/usr/bin/env python3
"""Extract selected binary-PLY records without changing any property bits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PROPERTY_BYTES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


def parse_header(data: bytes) -> tuple[bytes, int, int]:
    marker = b"end_header\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end]
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("only binary_little_endian PLY is supported")

    vertex_match = re.search(r"^element vertex\s+(\d+)\s*$", text, re.MULTILINE)
    if not vertex_match:
        raise ValueError("missing vertex element")
    vertex_count = int(vertex_match.group(1))

    in_vertex = False
    stride = 0
    for line in text.splitlines():
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex ")
        elif in_vertex and line.startswith("property "):
            fields = line.split()
            if fields[1] == "list":
                raise ValueError("list properties are not supported")
            stride += PROPERTY_BYTES[fields[1]]
    return header, header_end, vertex_count, stride


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("indices", nargs="+", type=int)
    args = parser.parse_args()

    data = args.input.read_bytes()
    header, body_offset, count, stride = parse_header(data)
    if len(data) < body_offset + count * stride:
        raise ValueError("truncated PLY body")
    for index in args.indices:
        if index < 0 or index >= count:
            raise IndexError(index)

    replacement = str(len(args.indices)).encode("ascii")
    header = re.sub(
        rb"(?m)^(element vertex\s+)\d+(\s*)$",
        rb"\g<1>" + replacement + rb"\g<2>",
        header,
        count=1,
    )
    header = re.sub(
        rb"(?m)^(obj_info num_cols\s+)\d+(\s*)$",
        rb"\g<1>" + replacement + rb"\g<2>",
        header,
        count=1,
    )
    records = b"".join(
        data[body_offset + index * stride : body_offset + (index + 1) * stride]
        for index in args.indices
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + records)
    print(
        f"wrote {args.output}: {len(args.indices)} records, stride={stride}, "
        f"source_count={count}"
    )


if __name__ == "__main__":
    main()
