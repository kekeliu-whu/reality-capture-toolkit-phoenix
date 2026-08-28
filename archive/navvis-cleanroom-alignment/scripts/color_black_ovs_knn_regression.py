#!/usr/bin/env python3
"""Regress black-OVS filtering and the final KNN seed partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


POINT_RECORD_BYTES = 36
OVS_RECORD_BYTES = 40


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ovs(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size % OVS_RECORD_BYTES:
        raise ValueError(f"{path}: size is not a multiple of {OVS_RECORD_BYTES}")
    return data.reshape(-1, 5, 8)


def load_ply_body(path: Path, point_count: int) -> np.ndarray:
    with path.open("rb") as source:
        while True:
            line = source.readline()
            if not line:
                raise ValueError(f"{path}: missing end_header")
            if line.rstrip() == b"end_header":
                break
        body = source.read()
    expected = point_count * POINT_RECORD_BYTES
    if len(body) != expected:
        raise ValueError(f"{path}: expected {expected} body bytes, got {len(body)}")
    return np.frombuffer(body, dtype=np.uint8).reshape(point_count, POINT_RECORD_BYTES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-ovs", type=Path, required=True)
    parser.add_argument("--candidate-direct-mask", type=Path, required=True)
    parser.add_argument("--vendor-colored-indices", type=Path, required=True)
    parser.add_argument("--vendor-uncolored-indices", type=Path, required=True)
    parser.add_argument("--reference-ply", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-index", type=int)
    args = parser.parse_args()

    ovs = load_ovs(args.candidate_ovs)
    point_count = ovs.shape[0]
    quality = ovs[:, :, 6].astype(np.uint16) | (ovs[:, :, 7].astype(np.uint16) << 8)
    valid = quality > 0
    black = valid & np.all(ovs[:, :, 3:6] == 0, axis=2)
    selected_count = valid.sum(axis=1)
    nonblack_count = (valid & ~black).sum(axis=1)
    raw_direct = selected_count > 0
    effective_colored = (nonblack_count > 0) & (
        (selected_count == 1) | (nonblack_count >= 2)
    )

    candidate_colored = np.flatnonzero(effective_colored).astype("<i4")
    candidate_uncolored = np.flatnonzero(~effective_colored).astype("<i4")
    vendor_colored = np.fromfile(args.vendor_colored_indices, dtype="<i4")
    vendor_uncolored = np.fromfile(args.vendor_uncolored_indices, dtype="<i4")

    direct_mask = np.fromfile(args.candidate_direct_mask, dtype=np.uint8)
    if direct_mask.size != point_count:
        raise ValueError("direct-mask point count does not match OVS")

    reference = load_ply_body(args.reference_ply, point_count)
    candidate = load_ply_body(args.candidate_ply, point_count)
    geometry_columns = np.r_[0:12, 15:POINT_RECORD_BYTES]
    rgb_delta = np.abs(
        candidate[:, 12:15].astype(np.int16) - reference[:, 12:15].astype(np.int16)
    )
    rgb_squared_error = np.square(
        candidate[:, 12:15].astype(np.float64) - reference[:, 12:15].astype(np.float64)
    )
    mse = float(rgb_squared_error.mean())

    result: dict[str, object] = {
        "inputs": {
            "candidate_ovs": str(args.candidate_ovs),
            "candidate_ovs_sha256": file_hash(args.candidate_ovs),
            "reference_ply": str(args.reference_ply),
            "candidate_ply": str(args.candidate_ply),
        },
        "partition": {
            "points": point_count,
            "raw_direct": int(raw_direct.sum()),
            "effective_colored": int(effective_colored.sum()),
            "effective_uncolored": int((~effective_colored).sum()),
            "rejected_all_black": int((raw_direct & (nonblack_count == 0)).sum()),
            "rejected_multi_to_single": int(
                ((selected_count > 1) & (nonblack_count == 1)).sum()
            ),
            "allowed_original_single_nonblack": int(
                ((selected_count == 1) & (nonblack_count == 1)).sum()
            ),
            "direct_mask_exact_raw_ovs": bool(
                np.array_equal(direct_mask, raw_direct.astype(np.uint8))
            ),
            "vendor_colored_ordered_exact": bool(
                np.array_equal(candidate_colored, vendor_colored)
            ),
            "vendor_uncolored_ordered_exact": bool(
                np.array_equal(candidate_uncolored, vendor_uncolored)
            ),
            "candidate_only_colored": int(
                np.setdiff1d(candidate_colored, vendor_colored, assume_unique=True).size
            ),
            "vendor_only_colored": int(
                np.setdiff1d(vendor_colored, candidate_colored, assume_unique=True).size
            ),
            "candidate_only_uncolored": int(
                np.setdiff1d(candidate_uncolored, vendor_uncolored, assume_unique=True).size
            ),
            "vendor_only_uncolored": int(
                np.setdiff1d(vendor_uncolored, candidate_uncolored, assume_unique=True).size
            ),
        },
        "ply": {
            "geometry_bit_exact": bool(
                np.array_equal(candidate[:, geometry_columns], reference[:, geometry_columns])
            ),
            "rgb_absolute_error_sum": int(rgb_delta.sum()),
            "rgb_mae_255": float(rgb_delta.mean()),
            "rgb_exact_point_fraction": float(np.all(rgb_delta == 0, axis=1).mean()),
            "rgb_changed_points": int(np.any(rgb_delta != 0, axis=1).sum()),
            "rgb_max_channel_delta": int(rgb_delta.max(initial=0)),
            "rgb_psnr_db": float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse))),
        },
    }
    if args.probe_index is not None:
        index = args.probe_index
        if index < 0 or index >= point_count:
            raise IndexError(index)
        result["probe"] = {
            "index": index,
            "reference_rgb": reference[index, 12:15].tolist(),
            "candidate_rgb": candidate[index, 12:15].tolist(),
            "exact": bool(np.array_equal(reference[index, 12:15], candidate[index, 12:15])),
        }

    partition = result["partition"]
    assert isinstance(partition, dict)
    guards = {
        "direct_mask": partition["direct_mask_exact_raw_ovs"],
        "colored_indices": partition["vendor_colored_ordered_exact"],
        "uncolored_indices": partition["vendor_uncolored_ordered_exact"],
    }
    result["guards"] = guards

    text = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not all(guards.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
