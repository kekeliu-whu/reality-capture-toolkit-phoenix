#!/usr/bin/env python3
"""Regression for the frozen panorama floor-mask and JPEG final stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    difference = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    pixel_difference = (
        np.any(difference != 0, axis=2) if difference.ndim == 3 else difference != 0
    )
    return {
        "shape": list(reference.shape),
        "absolute_error_sum": int(difference.sum(dtype=np.int64)),
        "mae": float(difference.mean()),
        "max_absolute_error": int(difference.max(initial=0)),
        "different_values": int(np.count_nonzero(difference)),
        "different_pixels": int(np.count_nonzero(pixel_difference)),
        "exact": bool(not np.any(difference)),
    }


def encode_jpeg(path: Path, rgb: np.ndarray, optimize: bool) -> None:
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    parameters = [
        cv2.IMWRITE_JPEG_QUALITY,
        95,
        cv2.IMWRITE_JPEG_OPTIMIZE,
        int(optimize),
    ]
    if not cv2.imwrite(str(path), bgr, parameters):
        raise RuntimeError(f"could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-mask", action="append", type=Path, required=True)
    parser.add_argument("--binary-mask-png", type=Path, required=True)
    parser.add_argument("--floor-mask", type=Path, required=True)
    parser.add_argument("--blend-output", type=Path, required=True)
    parser.add_argument("--no-floor-jpeg", type=Path, required=True)
    parser.add_argument("--floor-input", type=Path, required=True)
    parser.add_argument("--floor-output", type=Path, required=True)
    parser.add_argument("--filled-jpeg", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpp-candidate-directory", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    if len(args.projection_mask) < 2:
        parser.error("at least two --projection-mask arguments are required")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)

    projection_masks = [read_mask(path) for path in args.projection_mask]
    if any(mask.shape != projection_masks[0].shape for mask in projection_masks):
        raise ValueError("projection-mask dimensions do not match")
    reconstructed_floor_mask = np.logical_or.reduce(
        [mask != 0 for mask in projection_masks]
    )
    vendor_floor_mask = read_mask(args.floor_mask) != 0
    floor_mask_metrics = image_metrics(
        vendor_floor_mask.astype(np.uint8),
        reconstructed_floor_mask.astype(np.uint8),
    )
    reconstructed_floor_mask_u8 = reconstructed_floor_mask.astype(np.uint8)
    reconstructed_floor_mask_pgm = (
        args.output_directory / "reconstructed-floor-mask.pgm"
    )
    reconstructed_floor_mask_png = (
        args.output_directory / "reconstructed-binary-mask-compression9.png"
    )
    with reconstructed_floor_mask_pgm.open("wb") as output:
        output.write(
            f"P5\n{reconstructed_floor_mask.shape[1]} "
            f"{reconstructed_floor_mask.shape[0]}\n255\n".encode("ascii")
        )
        output.write(reconstructed_floor_mask_u8.tobytes())
    if not cv2.imwrite(
        str(reconstructed_floor_mask_png),
        reconstructed_floor_mask_u8,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    ):
        raise RuntimeError(f"could not write {reconstructed_floor_mask_png}")

    blend_output = read_rgb(args.blend_output)
    floor_input = read_rgb(args.floor_input)
    floor_output = read_rgb(args.floor_output)
    no_floor_decoded = read_rgb(args.no_floor_jpeg)
    filled_decoded = read_rgb(args.filled_jpeg)

    reconstructed_no_floor = args.output_directory / "no-floor-q95-opt0.jpg"
    reconstructed_filled = args.output_directory / "filled-q95-opt1.jpg"
    encode_jpeg(reconstructed_no_floor, blend_output, optimize=False)
    encode_jpeg(reconstructed_filled, floor_output, optimize=True)

    result = {
        "opencv_version": cv2.__version__,
        "inputs": {
            "projection_masks": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.projection_mask
            ],
            "binary_mask_png": {
                "path": str(args.binary_mask_png),
                "sha256": sha256(args.binary_mask_png),
            },
            "floor_mask": {"path": str(args.floor_mask), "sha256": sha256(args.floor_mask)},
            "blend_output": {"path": str(args.blend_output), "sha256": sha256(args.blend_output)},
            "no_floor_jpeg": {"path": str(args.no_floor_jpeg), "sha256": sha256(args.no_floor_jpeg)},
            "floor_input": {"path": str(args.floor_input), "sha256": sha256(args.floor_input)},
            "floor_output": {"path": str(args.floor_output), "sha256": sha256(args.floor_output)},
            "filled_jpeg": {"path": str(args.filled_jpeg), "sha256": sha256(args.filled_jpeg)},
        },
        "floor_mask": {
            "algorithm": "logical OR of all nonzero full-resolution projection masks",
            "vendor_valid_pixels": int(np.count_nonzero(vendor_floor_mask)),
            "vendor_hole_pixels": int(vendor_floor_mask.size - np.count_nonzero(vendor_floor_mask)),
            "reconstructed_valid_pixels": int(np.count_nonzero(reconstructed_floor_mask)),
            "reconstructed_pgm": {
                "path": str(reconstructed_floor_mask_pgm),
                "sha256": sha256(reconstructed_floor_mask_pgm),
                "byte_exact": (
                    reconstructed_floor_mask_pgm.read_bytes()
                    == args.floor_mask.read_bytes()
                ),
            },
            "reconstructed_binary_mask_png": {
                "path": str(reconstructed_floor_mask_png),
                "parameters": {"png_compression": 9},
                "sha256": sha256(reconstructed_floor_mask_png),
                "reference_sha256": sha256(args.binary_mask_png),
                "byte_exact": (
                    reconstructed_floor_mask_png.read_bytes()
                    == args.binary_mask_png.read_bytes()
                ),
            },
            "metrics": floor_mask_metrics,
        },
        "stage_order": {
            "blend_output_vs_floor_input": image_metrics(blend_output, floor_input),
            "no_floor_jpeg_decoded_vs_floor_input": image_metrics(no_floor_decoded, floor_input),
            "floor_output_vs_filled_jpeg_decoded": image_metrics(floor_output, filled_decoded),
        },
        "jpeg": {
            "no_floor": {
                "parameters": {"quality": 95, "optimize": False},
                "candidate": str(reconstructed_no_floor),
                "candidate_sha256": sha256(reconstructed_no_floor),
                "reference_sha256": sha256(args.no_floor_jpeg),
                "byte_exact": reconstructed_no_floor.read_bytes() == args.no_floor_jpeg.read_bytes(),
                "decoded_metrics": image_metrics(
                    read_rgb(args.no_floor_jpeg), read_rgb(reconstructed_no_floor)
                ),
            },
            "filled": {
                "parameters": {"quality": 95, "optimize": True},
                "candidate": str(reconstructed_filled),
                "candidate_sha256": sha256(reconstructed_filled),
                "reference_sha256": sha256(args.filled_jpeg),
                "byte_exact": reconstructed_filled.read_bytes() == args.filled_jpeg.read_bytes(),
                "decoded_metrics": image_metrics(
                    read_rgb(args.filled_jpeg), read_rgb(reconstructed_filled)
                ),
            },
        },
    }
    if args.cpp_candidate_directory is not None:
        candidate_references = {
            "floor-mask.pgm": args.floor_mask,
            "binary-mask.png": args.binary_mask_png,
            "no-floor.jpg": args.no_floor_jpeg,
            "floor-input.tga": args.floor_input,
            "floor-output.tga": args.floor_output,
            "filled.jpg": args.filled_jpeg,
        }
        result["cpp_probe"] = {}
        for filename, reference in candidate_references.items():
            candidate = args.cpp_candidate_directory / filename
            result["cpp_probe"][filename] = {
                "candidate": str(candidate),
                "candidate_sha256": sha256(candidate),
                "reference": str(reference),
                "reference_sha256": sha256(reference),
                "byte_exact": candidate.read_bytes() == reference.read_bytes(),
            }
    result["exact"] = bool(
        floor_mask_metrics["exact"]
        and result["floor_mask"]["reconstructed_pgm"]["byte_exact"]
        and result["floor_mask"]["reconstructed_binary_mask_png"]["byte_exact"]
        and result["stage_order"]["no_floor_jpeg_decoded_vs_floor_input"]["exact"]
        and result["jpeg"]["no_floor"]["byte_exact"]
        and result["jpeg"]["filled"]["byte_exact"]
        and result["jpeg"]["no_floor"]["decoded_metrics"]["exact"]
        and result["jpeg"]["filled"]["decoded_metrics"]["exact"]
        and (
            "cpp_probe" not in result
            or all(item["byte_exact"] for item in result["cpp_probe"].values())
        )
    )

    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
