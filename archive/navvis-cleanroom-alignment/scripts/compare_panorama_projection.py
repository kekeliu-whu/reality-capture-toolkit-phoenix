#!/usr/bin/env python3
"""Compare clean-room projected camera inputs with frozen vendor matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-dir",
        required=True,
        type=Path,
        help="directory containing metadata.json and frozen image_N.bin/mask_N.bin",
    )
    parser.add_argument(
        "--clean-debug-dir",
        required=True,
        type=Path,
        help="pipeline debug directory containing exposure-input-*-camN.png",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for the machine-readable result",
    )
    return parser.parse_args()


def load_frozen_matrix(entry: dict[str, object], directory: Path) -> np.ndarray:
    depth = int(entry["depth"])
    if depth != 0:
        raise ValueError(f"only frozen CV_8U matrices are supported, got depth={depth}")
    rows = int(entry["rows"])
    cols = int(entry["cols"])
    channels = int(entry["channels"])
    step = int(entry["step0"])
    path = directory / Path(str(entry["file"])).name
    packed_width = cols * channels
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != rows * step:
        raise ValueError(
            f"{path}: expected {rows * step} bytes from metadata, got {raw.size}"
        )
    matrix = raw.reshape(rows, step)[:, :packed_width]
    if channels == 1:
        return matrix.reshape(rows, cols)
    return matrix.reshape(rows, cols, channels)


def read_clean(path: Path, flags: int) -> np.ndarray:
    matrix = cv2.imread(str(path), flags)
    if matrix is None:
        raise FileNotFoundError(path)
    return matrix


def compare_camera(
    index: int,
    vendor_image: np.ndarray,
    vendor_mask: np.ndarray,
    clean_directory: Path,
) -> dict[str, object]:
    clean_image = read_clean(
        clean_directory / f"exposure-input-image-cam{index}.png", cv2.IMREAD_COLOR
    )
    clean_mask = read_clean(
        clean_directory / f"exposure-input-mask-cam{index}.png", cv2.IMREAD_GRAYSCALE
    )
    if clean_image.shape != vendor_image.shape:
        raise ValueError(
            f"cam{index}: image shape mismatch {clean_image.shape} != {vendor_image.shape}"
        )
    if clean_mask.shape != vendor_mask.shape:
        raise ValueError(
            f"cam{index}: mask shape mismatch {clean_mask.shape} != {vendor_mask.shape}"
        )

    vendor_valid = vendor_mask > 0
    clean_valid = clean_mask > 0
    intersection = vendor_valid & clean_valid
    union = vendor_valid | clean_valid
    intersection_count = int(np.count_nonzero(intersection))
    union_count = int(np.count_nonzero(union))
    if union_count == 0 or intersection_count == 0:
        raise ValueError(f"cam{index}: comparison has no valid support")

    absolute = np.abs(clean_image.astype(np.int16) - vendor_image.astype(np.int16))
    common_values = absolute[intersection]
    result: dict[str, object] = {
        "camera": index,
        "mask_iou": intersection_count / union_count,
        "vendor_valid": int(np.count_nonzero(vendor_valid)),
        "clean_valid": int(np.count_nonzero(clean_valid)),
        "intersection": intersection_count,
        "union": union_count,
        "false_positive": int(np.count_nonzero(clean_valid & ~vendor_valid)),
        "false_negative": int(np.count_nonzero(vendor_valid & ~clean_valid)),
        "common_valid_mae_255": float(np.mean(common_values)),
        "common_valid_exact_channel_fraction": float(np.mean(common_values == 0)),
        "common_valid_max_channel_error": int(np.max(common_values)),
    }
    return result


def main() -> int:
    args = parse_args()
    metadata_path = args.vendor_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    images = metadata.get("images", [])
    masks = metadata.get("masks", [])
    if len(images) != 4 or len(masks) != 4:
        raise ValueError(f"{metadata_path}: expected four images and four masks")

    cameras = []
    for index in range(4):
        cameras.append(
            compare_camera(
                index,
                load_frozen_matrix(images[index], args.vendor_dir),
                load_frozen_matrix(masks[index], args.vendor_dir),
                args.clean_debug_dir,
            )
        )
    result = {"vendor_dir": str(args.vendor_dir), "cameras": cameras}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
