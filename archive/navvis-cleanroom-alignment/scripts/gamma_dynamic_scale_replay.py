#!/usr/bin/env python3
"""Replay nv_colorcloud's Dynamic weight normalization from a packed OVS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct


SELECTED_VIEWS = 5
RECORD_SIZE = SELECTED_VIEWS * 8


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def percentile(histogram: list[int], fraction: float) -> int:
    # Match histogram8uPercentile: float fraction and the lower integer rank.
    total = sum(histogram)
    if total == 0:
        return 0
    target = int(f32(fraction) * f32(float(total - 1)))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative > target:
            return value
    return 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ovs", required=True, type=pathlib.Path)
    parser.add_argument("--view-count", required=True, type=int)
    parser.add_argument("--vendor-caller", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = args.ovs.read_bytes()
    if len(payload) % RECORD_SIZE:
        raise ValueError("OVS size is not divisible by 40")

    histograms = [[0] * 256 for _ in range(args.view_count)]
    raw_weights = [0.0] * args.view_count
    joint_count = 0
    for record_offset in range(0, len(payload), RECORD_SIZE):
        record = payload[record_offset : record_offset + RECORD_SIZE]
        reference = record[3:6]
        accepted = 0
        for rank in range(SELECTED_VIEWS):
            offset = rank * 8
            packed_quality = record[offset + 6] | (record[offset + 7] << 8)
            if packed_quality == 0:
                continue
            capture = (record[offset] << 8) | record[offset + 1]
            view = 4 * capture + record[offset + 2]
            rgb = record[offset + 3 : offset + 6]
            intensity = max(rgb)
            quality = f32(f32(float(packed_quality)) / f32(65535.0))
            if intensity != 0:
                histograms[view][intensity] += 1
                raw_weights[view] += quality
            if intensity < 6 or intensity > 249:
                continue
            if rank > 0 and max(abs(rgb[channel] - reference[channel]) for channel in range(3)) > 50:
                continue
            accepted += 1
        if accepted >= 2:
            joint_count += 1

    candidates = []
    for view, (histogram, raw_weight) in enumerate(zip(histograms, raw_weights)):
        if sum(value > 0 for value in histogram) <= 1 or raw_weight <= 0.0:
            continue
        low = percentile(histogram, 0.02)
        high = percentile(histogram, 0.98)
        if low < high:
            candidates.append((view, low, high, raw_weight))

    # 0x1e61cd..0x1e6495: one scalar accumulator, input order, abs(weight).
    denominator = 0.0
    for _, _, _, raw_weight in candidates:
        denominator += abs(raw_weight)
    first_normalized = [raw_weight / denominator for _, _, _, raw_weight in candidates]

    # The caller performs a second L1 normalization after rejecting invalid
    # percentile ranges.  Even when every view survives, the first normalized
    # vector sums to 1-2 ULP and this second division changes every element.
    selected_denominator = 0.0
    for weight in first_normalized:
        selected_denominator += abs(weight)
    normalized = [weight / selected_denominator for weight in first_normalized]

    # 0x1e66e1..0x1e66ee and 0x1e6780..0x1e67a1.
    count_as_double = float(len(candidates))
    total_scale = float(joint_count) * 1.0e-4
    average_scale = total_scale / count_as_double
    scales = [average_scale * (count_as_double * weight) for weight in normalized]
    simplified_scales = [total_scale * weight for weight in normalized]

    vendor = json.loads(args.vendor_caller.read_text(encoding="utf-8"))
    vendor_weights = [float.fromhex(row["normalized_weight_hex"]) for row in vendor["records"]]
    if len(vendor_weights) != len(normalized):
        raise ValueError("candidate count differs from vendor capture")

    weight_differences = [a != b for a, b in zip(normalized, vendor_weights)]
    vendor_total_scale = float.fromhex(vendor["total_scale_hex"])
    vendor_average_scale = float.fromhex(vendor["average_scale_hex"])
    report = {
        "ovs_sha256": hashlib.sha256(payload).hexdigest(),
        "record_count": len(payload) // RECORD_SIZE,
        "joint_count": joint_count,
        "dynamic_count": len(candidates),
        "denominator": denominator,
        "denominator_hex": denominator.hex(),
        "selected_denominator": selected_denominator,
        "selected_denominator_hex": selected_denominator.hex(),
        "vendor_selected_denominator_hex": vendor["l1_denominator_hex"],
        "selected_denominator_exact": selected_denominator.hex() == vendor["l1_denominator_hex"],
        "normalized_weights_exact": sum(not value for value in weight_differences),
        "normalized_weights_changed": sum(weight_differences),
        "total_scale_hex": total_scale.hex(),
        "vendor_total_scale_hex": vendor["total_scale_hex"],
        "total_scale_exact": total_scale == vendor_total_scale,
        "average_scale_hex": average_scale.hex(),
        "vendor_average_scale_hex": vendor["average_scale_hex"],
        "average_scale_exact": average_scale == vendor_average_scale,
        "grouped_vs_simplified_changed": sum(a != b for a, b in zip(scales, simplified_scales)),
        "grouped_vs_simplified_max_abs": max(abs(a - b) for a, b in zip(scales, simplified_scales)),
        "rows": [
            {
                "index": index,
                "view": candidates[index][0],
                "low": candidates[index][1],
                "high": candidates[index][2],
                "raw_weight_hex": candidates[index][3].hex(),
                "first_normalized_weight_hex": first_normalized[index].hex(),
                "normalized_weight_hex": normalized[index].hex(),
                "vendor_normalized_weight_hex": vendor_weights[index].hex(),
                "grouped_scale_hex": scales[index].hex(),
                "simplified_scale_hex": simplified_scales[index].hex(),
            }
            for index in range(len(candidates))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
