#!/usr/bin/env python3
"""Reproduce the vendor and clean SceneBrightnessResidual initial cost.

The vendor scene-range capture is a vector of 136 builder objects with stride
0x420.  Its embedded range value begins at +0x14: normalized_weight is float32
at embedded +0, low at +4, high at +5, and median at +6.  This is the object
base used by the Scene functor disassembly; the capture also preserves the
20-byte histogram-builder prefix (distinct/count/ViewId).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


VENDOR_SCENE_STRIDE = 0x420
VENDOR_RANGE_OFFSET = 0x14
BYTE_NORMALIZER_F32 = struct.unpack("<f", struct.pack("<f", 1.0 / 255.0))[0]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def read_vendor_ranges(path: Path) -> list[dict[str, float | int]]:
    raw = path.read_bytes()
    if len(raw) % VENDOR_SCENE_STRIDE:
        raise ValueError(f"{path}: size {len(raw)} is not divisible by 0x420")
    ranges = []
    for offset in range(0, len(raw), VENDOR_SCENE_STRIDE):
        range_offset = offset + VENDOR_RANGE_OFFSET
        weight = struct.unpack_from("<f", raw, range_offset)[0]
        low, high, median = struct.unpack_from("<BBB", raw, range_offset + 4)
        ranges.append(
            {
                "weight": float(weight),
                "low": low,
                "high": high,
                "median": median,
            }
        )
    return ranges


def read_clean_scene_weights_from_ovs(path: Path, view_count: int) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 40:
        raise ValueError(f"{path}: size {len(raw)} is not divisible by 40")

    totals = [0.0] * view_count
    for offset in range(0, len(raw), 40):
        for rank in range(5):
            item = offset + rank * 8
            packed_quality = raw[item + 6] | (raw[item + 7] << 8)
            if packed_quality == 0:
                continue
            capture = (raw[item] << 8) | raw[item + 1]
            view = 4 * capture + raw[item + 2]
            if view >= view_count:
                raise ValueError(f"invalid view {view} at record offset {offset}, rank {rank}")
            intensity = max(raw[item + 3 : item + 6])
            if intensity == 0:
                continue
            quality = f32(f32(float(packed_quality)) / f32(65535.0))
            if quality > f32(0.01):
                totals[view] += float(quality)

    total = math.fsum(totals)
    return [value / total for value in totals]


def scene_metrics(ranges: list[dict[str, float | int]], weights: list[float], joint_count: int) -> dict[str, float | int]:
    selected = [index for index, item in enumerate(ranges) if int(item["high"]) > 203]
    selected_weight = sum(weights[index] for index in selected)
    corrected_high = 0.0
    corrected_low = 0.0
    original_high = 0.0
    for index in selected:
        weight = weights[index]
        high = float(f32(float(ranges[index]["high"]) * BYTE_NORMALIZER_F32))
        low = float(f32(float(ranges[index]["low"]) * BYTE_NORMALIZER_F32))
        # Initial GammaModel is gain=1, exponent=1.
        corrected_high += weight * high
        corrected_low += weight * low
        original_high += weight * high
    high_average = corrected_high / selected_weight
    low_average = corrected_low / selected_weight
    original_high_average = original_high / selected_weight
    residual = high_average - 1.1 * original_high_average
    return {
        "selected_count": len(selected),
        "selected_weight": selected_weight,
        "corrected_high_average": high_average,
        "corrected_low_average_finite_only": low_average,
        "original_high_average": original_high_average,
        "residual": residual,
        "scaled_cost": 0.5 * float(joint_count) * residual * residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-ovs", type=Path, required=True)
    parser.add_argument("--scene-ranges", type=Path, required=True)
    parser.add_argument("--joint-count", type=int, required=True)
    parser.add_argument("--vendor-initial-cost", type=float, required=True)
    parser.add_argument("--clean-initial-cost", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranges = read_vendor_ranges(args.scene_ranges)
    clean_weights = read_clean_scene_weights_from_ovs(args.exposure_ovs, len(ranges))
    vendor_weights = [float(item["weight"]) for item in ranges]
    quantized_clean_weights = [float(f32(value)) for value in clean_weights]

    vendor = scene_metrics(ranges, vendor_weights, args.joint_count)
    clean = scene_metrics(ranges, clean_weights, args.joint_count)
    quantized = scene_metrics(ranges, quantized_clean_weights, args.joint_count)
    initial_gap = args.vendor_initial_cost - args.clean_initial_cost
    scene_gap = float(vendor["scaled_cost"]) - float(clean["scaled_cost"])
    result = {
        "inputs": {
            "exposure_ovs": str(args.exposure_ovs),
            "scene_ranges": str(args.scene_ranges),
            "joint_count": args.joint_count,
            "vendor_initial_cost": args.vendor_initial_cost,
            "clean_initial_cost": args.clean_initial_cost,
        },
        "constants": {
            "selection": "high > 203",
            "byte_normalizer_float32": BYTE_NORMALIZER_F32,
            "scene_scaled_loss": float(args.joint_count),
            "scene_residual_count": 1,
        },
        "vendor_float_weight": vendor,
        "clean_double_weight": clean,
        "clean_weight_quantized_to_float": quantized,
        "weight_comparison": {
            "all_quantized_clean_equal_vendor": quantized_clean_weights == vendor_weights,
            "maximum_abs_vendor_minus_clean": max(
                abs(vendor_weights[i] - clean_weights[i]) for i in range(len(ranges))
            ),
        },
        "cost_gap": {
            "full_initial_vendor_minus_clean": initial_gap,
            "scene_vendor_minus_clean": scene_gap,
            "fraction_explained_by_scene_float_weight": scene_gap / initial_gap,
            "unexplained_remainder": initial_gap - scene_gap,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
