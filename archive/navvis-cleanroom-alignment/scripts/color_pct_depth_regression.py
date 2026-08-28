#!/usr/bin/env python3
"""Pair captured vendor PCT depth maps with clean-room views and compare them.

The vendor renders views concurrently, so debugger completion order is not a
view identifier.  This tool uses a deterministic global assignment based on
depth occupancy and millimetre-quantized common pixels, then writes the raw
vendor maps under the view names accepted by the colorizer's regression hook.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


WIDTH = 684
HEIGHT = 456
PIXELS = WIDTH * HEIGHT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--mapped-output-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=136)
    parser.add_argument("--sample-pixels", type=int, default=16384)
    return parser.parse_args()


def read_depth(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != PIXELS:
        raise ValueError(f"{path}: expected {PIXELS} float32 values, got {values.size}")
    return values


def load_maps(paths: list[Path]) -> np.ndarray:
    return np.stack([read_depth(path) for path in paths])


def millimetre_floor(values: np.ndarray) -> np.ndarray:
    return np.floor(values * np.float32(1000.0)) * np.float32(0.001)


def assignment_costs(official: np.ndarray, clean: np.ndarray, sample: np.ndarray) -> np.ndarray:
    official_sample = official[:, sample]
    clean_sample = clean[:, sample]
    official_valid = official_sample > 0.0
    clean_valid = clean_sample > 0.0
    official_mm = millimetre_floor(official_sample)

    costs = np.empty((official.shape[0], clean.shape[0]), dtype=np.float64)
    for official_index in range(official.shape[0]):
        left_valid = official_valid[official_index]
        common = clean_valid & left_valid[None, :]
        occupancy_disagreement = np.mean(clean_valid != left_valid[None, :], axis=1)
        absolute_error = np.abs(clean_sample - official_mm[official_index][None, :])
        common_count = np.maximum(common.sum(axis=1), 1)
        common_mae = (absolute_error * common).sum(axis=1) / common_count
        # Occupancy uniquely identifies each view.  The value term breaks the
        # rare near-tie between adjacent captures without depending on color.
        costs[official_index] = 16.0 * occupancy_disagreement + common_mae
    return costs


def main() -> None:
    args = parse_args()
    official_paths = sorted(args.official_dir.glob("official_depth_completion_*.f32"))
    clean_paths = [args.clean_dir / f"nv_depth_{view:02d}.f32" for view in range(args.views)]
    if len(official_paths) != args.views:
        raise ValueError(f"expected {args.views} official maps, found {len(official_paths)}")
    missing_clean = [str(path) for path in clean_paths if not path.is_file()]
    if missing_clean:
        raise FileNotFoundError(f"missing clean maps: {missing_clean}")

    official = load_maps(official_paths)
    clean = load_maps(clean_paths)
    generator = np.random.default_rng(0x4E5653)
    sample_count = min(args.sample_pixels, PIXELS)
    sample = np.sort(generator.choice(PIXELS, size=sample_count, replace=False))
    costs = assignment_costs(official, clean, sample)
    official_indices, view_indices = linear_sum_assignment(costs)
    capture_for_view = np.empty(args.views, dtype=np.int64)
    capture_for_view[view_indices] = official_indices

    args.mapped_output_dir.mkdir(parents=True, exist_ok=True)
    per_view: list[dict[str, object]] = []
    aggregate_intersection = 0
    aggregate_union = 0
    aggregate_common = 0
    aggregate_mm_exact = 0
    aggregate_mm_absolute_error = 0.0
    aggregate_raw_absolute_error = 0.0
    aggregate_clean_only = 0
    aggregate_official_only = 0

    for view in range(args.views):
        capture = int(capture_for_view[view])
        vendor = official[capture]
        candidate = clean[view]
        vendor_valid = vendor > 0.0
        candidate_valid = candidate > 0.0
        common = vendor_valid & candidate_valid
        union = vendor_valid | candidate_valid
        vendor_mm = millimetre_floor(vendor)
        mm_error = np.abs(vendor_mm[common] - candidate[common])
        raw_error = np.abs(vendor[common] - candidate[common])
        sorted_costs = np.sort(costs[capture])
        output_path = args.mapped_output_dir / f"nv_depth_{view:02d}.f32"
        shutil.copyfile(official_paths[capture], output_path)

        intersection = int(common.sum())
        union_count = int(union.sum())
        clean_only = int((candidate_valid & ~vendor_valid).sum())
        official_only = int((vendor_valid & ~candidate_valid).sum())
        mm_exact = int(np.count_nonzero(mm_error == 0.0))
        per_view.append(
            {
                "view": view,
                "completion_capture": capture,
                "assignment_cost": float(costs[capture, view]),
                "assignment_second_best_margin": float(sorted_costs[1] - sorted_costs[0]),
                "valid_intersection": intersection,
                "valid_union": union_count,
                "valid_iou": float(intersection / union_count) if union_count else 1.0,
                "clean_only_valid": clean_only,
                "official_only_valid": official_only,
                "common_mm_exact": mm_exact,
                "common_mm_exact_fraction": float(mm_exact / intersection) if intersection else 1.0,
                "common_mm_mae_m": float(mm_error.mean()) if intersection else 0.0,
                "common_mm_max_m": float(mm_error.max()) if intersection else 0.0,
                "common_raw_mae_m": float(raw_error.mean()) if intersection else 0.0,
            }
        )
        aggregate_intersection += intersection
        aggregate_union += union_count
        aggregate_common += intersection
        aggregate_mm_exact += mm_exact
        aggregate_mm_absolute_error += float(mm_error.sum(dtype=np.float64))
        aggregate_raw_absolute_error += float(raw_error.sum(dtype=np.float64))
        aggregate_clean_only += clean_only
        aggregate_official_only += official_only

    result = {
        "shape": [HEIGHT, WIDTH],
        "views": args.views,
        "sample_pixels": sample_count,
        "representation": {
            "official_no_hit": 0.0,
            "clean_no_hit": -1.0,
            "official_positive_depth_quantized_to_mm_for_comparison": True,
        },
        "assignment": {
            "total_cost": float(costs[official_indices, view_indices].sum()),
            "minimum_second_best_margin": float(
                min(item["assignment_second_best_margin"] for item in per_view)
            ),
            "capture_for_view": capture_for_view.tolist(),
        },
        "aggregate": {
            "valid_intersection": aggregate_intersection,
            "valid_union": aggregate_union,
            "valid_iou": aggregate_intersection / aggregate_union,
            "clean_only_valid": aggregate_clean_only,
            "official_only_valid": aggregate_official_only,
            "common_mm_exact": aggregate_mm_exact,
            "common_mm_exact_fraction": aggregate_mm_exact / aggregate_common,
            "common_mm_mae_m": aggregate_mm_absolute_error / aggregate_common,
            "common_raw_mae_m": aggregate_raw_absolute_error / aggregate_common,
        },
        "per_view": per_view,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["assignment"], indent=2))
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
