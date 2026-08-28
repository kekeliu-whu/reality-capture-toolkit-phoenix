#!/usr/bin/env python3
"""Compare raw top-origin 24-bit TGA files without image-codec limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def read_tga(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(18)
        if len(header) != 18:
            raise ValueError(f"truncated TGA header: {path}")
        fields = struct.unpack("<BBBHHBHHHHBB", header)
        id_length, color_map_type, image_type = fields[:3]
        width, height, depth, descriptor = fields[8:12]
        if color_map_type != 0 or image_type != 2 or depth != 24:
            raise ValueError(f"unsupported TGA layout: {path}")
        stream.seek(id_length, 1)
        payload = stream.read(width * height * 3)
        if len(payload) != width * height * 3:
            raise ValueError(f"truncated TGA payload: {path}")
    image = np.frombuffer(payload, np.uint8).reshape(height, width, 3)
    if descriptor & 0x20 == 0:
        image = image[::-1]
    return image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = read_tga(args.reference)
    candidate = read_tga(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")

    absolute = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    result: dict[str, object] = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "shape": list(reference.shape),
        "reference_sha256": sha256(args.reference),
        "candidate_sha256": sha256(args.candidate),
        "absolute_error_sum": int(absolute.sum()),
        "different_channels": int(np.count_nonzero(absolute)),
        "different_pixels": int(np.count_nonzero(np.any(absolute != 0, axis=2))),
        "mae": float(absolute.mean()),
        "max_absolute_error": int(absolute.max()),
    }
    if args.mask is not None:
        with args.mask.open("rb") as stream:
            if stream.readline().strip() != b"P5":
                raise ValueError("mask is not a binary PGM")
            width, height = map(int, stream.readline().split())
            if int(stream.readline()) != 255:
                raise ValueError("mask is not 8-bit")
            mask = np.frombuffer(stream.read(width * height), np.uint8).reshape(height, width)
        if mask.shape != reference.shape[:2]:
            raise ValueError("mask shape does not match images")
        for name, selection in (("valid", mask != 0), ("hole", mask == 0)):
            selected = absolute[selection]
            result[name] = {
                "pixels": int(selection.sum()),
                "absolute_error_sum": int(selected.sum()),
                "different_channels": int(np.count_nonzero(selected)),
                "mae": float(selected.mean()) if selected.size else 0.0,
                "max_absolute_error": int(selected.max()) if selected.size else 0,
            }

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
