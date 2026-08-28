#!/usr/bin/env python3
"""Convert a NavVis colored binary PLY to its eight-float Surface layout.

The conversion preserves the bit patterns of x/y/z, intensity, normal and
curvature fields.  RGB and alpha are deliberately omitted, which makes it
possible to prove that a colorizer regression does not inherit reference
colors from its input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


COLORED_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("alpha", "u1"),
        ("intensity", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("curvature", "<f4"),
    ],
    align=False,
)
SURFACE_DTYPE = np.dtype(
    [(name, "<f4") for name in ("x", "y", "z", "intensity", "nx", "ny", "nz", "curvature")],
    align=False,
)
SURFACE_FIELDS = tuple(SURFACE_DTYPE.names or ())


def parse_header(stream) -> tuple[int, int]:
    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY header has no end_header")
        text = line.decode("ascii").strip()
        if text == "format binary_little_endian 1.0":
            continue
        if text.startswith("element vertex "):
            vertex_count = int(text.rsplit(" ", 1)[1])
        elif text.startswith("property "):
            _, field_type, name = text.split()
            properties.append((field_type, name))
        elif text == "end_header":
            break

    expected = [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("uchar", "red"),
        ("uchar", "green"),
        ("uchar", "blue"),
        ("uchar", "alpha"),
        ("float", "intensity"),
        ("float", "nx"),
        ("float", "ny"),
        ("float", "nz"),
        ("float", "curvature"),
    ]
    if vertex_count is None or properties != expected:
        raise ValueError("input is not the expected 36-byte NavVis colored PLY")
    return vertex_count, stream.tell()


def surface_header(vertex_count: int) -> bytes:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment RGB and alpha intentionally removed for color-independence regression",
        f"element vertex {vertex_count}",
    ]
    lines.extend(f"property float {name}" for name in SURFACE_FIELDS)
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def convert(source: Path, destination: Path, chunk_points: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream:
        vertex_count, body_offset = parse_header(input_stream)
        expected_size = body_offset + vertex_count * COLORED_DTYPE.itemsize
        if source.stat().st_size != expected_size:
            raise ValueError("input PLY body is truncated or has trailing bytes")

        with destination.open("xb") as output_stream:
            output_stream.write(surface_header(vertex_count))
            remaining = vertex_count
            while remaining:
                count = min(remaining, chunk_points)
                colored = np.fromfile(input_stream, dtype=COLORED_DTYPE, count=count)
                if colored.size != count:
                    raise ValueError("short read while converting PLY body")
                surface = np.empty(count, dtype=SURFACE_DTYPE)
                for name in SURFACE_FIELDS:
                    surface[name] = colored[name]
                surface.tofile(output_stream)
                remaining -= count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-points", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.chunk_points <= 0:
        parser.error("--chunk-points must be positive")
    convert(args.source, args.destination, args.chunk_points)


if __name__ == "__main__":
    main()
