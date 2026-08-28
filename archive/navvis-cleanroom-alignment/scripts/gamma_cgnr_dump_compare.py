#!/usr/bin/env python3
"""Compare exact-hex Ceres trust-region dumps from vendor and clean runs."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct


JOINT_END = 138_898
DYNAMIC_END = 139_306


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=pathlib.Path, required=True)
    parser.add_argument("--clean", type=pathlib.Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def ordered_bits(value: float) -> int:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def ulp_distance(left: float, right: float) -> int:
    return abs(ordered_bits(left) - ordered_bits(right))


class Metrics:
    def __init__(self) -> None:
        self.count = 0
        self.exact = 0
        self.absolute_sum = 0.0
        self.max_absolute = 0.0
        self.max_ulp = 0
        self.within_1_ulp = 0
        self.within_2_ulp = 0
        self.within_4_ulp = 0
        self.first_differences: list[dict] = []

    def add(self, index: int, vendor: float, clean: float, **context: int) -> None:
        self.count += 1
        absolute = abs(vendor - clean)
        ulp = ulp_distance(vendor, clean)
        self.absolute_sum += absolute
        self.max_absolute = max(self.max_absolute, absolute)
        self.max_ulp = max(self.max_ulp, ulp)
        self.exact += ulp == 0
        self.within_1_ulp += ulp <= 1
        self.within_2_ulp += ulp <= 2
        self.within_4_ulp += ulp <= 4
        if ulp and len(self.first_differences) < 20:
            self.first_differences.append(
                {
                    "index": index,
                    **context,
                    "vendor": vendor.hex(),
                    "clean": clean.hex(),
                    "absolute": absolute,
                    "ulp": ulp,
                }
            )

    def result(self) -> dict:
        return {
            "count": self.count,
            "exact": self.exact,
            "changed": self.count - self.exact,
            "exact_fraction": self.exact / self.count,
            "mae": self.absolute_sum / self.count,
            "max_absolute": self.max_absolute,
            "max_ulp": self.max_ulp,
            "within_1_ulp": self.within_1_ulp,
            "within_2_ulp": self.within_2_ulp,
            "within_4_ulp": self.within_4_ulp,
            "first_differences": self.first_differences,
        }


def parse_hex_value(text: str) -> float:
    return float.fromhex(text)


def load_vector(path: pathlib.Path) -> list[float]:
    return [parse_hex_value(line.strip()) for line in path.read_text().splitlines()]


def compare_vectors(vendor: list[float], clean: list[float]) -> dict:
    if len(vendor) != len(clean):
        raise ValueError(f"vector length mismatch: {len(vendor)} != {len(clean)}")
    metrics = Metrics()
    for index, (vendor_value, clean_value) in enumerate(zip(vendor, clean)):
        metrics.add(index, vendor_value, clean_value)
    return metrics.result()


def row_class(row: int) -> str:
    if row < JOINT_END:
        return "joint"
    if row < DYNAMIC_END:
        return "dynamic"
    return "scene"


def compare_matrix(
    vendor_path: pathlib.Path,
    clean_path: pathlib.Path,
    vendor_residuals: list[float],
    clean_residuals: list[float],
) -> tuple[dict, list[float], list[float]]:
    metrics = Metrics()
    changed_by_class = {"joint": 0, "dynamic": 0, "scene": 0}
    total_by_class = {"joint": 0, "dynamic": 0, "scene": 0}
    vendor_rhs = [0.0] * 272
    clean_rhs = [0.0] * 272
    structure_exact = True

    with vendor_path.open("r", encoding="utf-8") as vendor_stream, clean_path.open(
        "r", encoding="utf-8"
    ) as clean_stream:
        index = 0
        while True:
            vendor_line = vendor_stream.readline()
            clean_line = clean_stream.readline()
            if not vendor_line and not clean_line:
                break
            if not vendor_line or not clean_line:
                raise ValueError("matrix line-count mismatch")
            vendor_row_text, vendor_column_text, vendor_value_text = vendor_line.split()
            clean_row_text, clean_column_text, clean_value_text = clean_line.split()
            vendor_row = int(vendor_row_text)
            vendor_column = int(vendor_column_text)
            clean_row = int(clean_row_text)
            clean_column = int(clean_column_text)
            if (vendor_row, vendor_column) != (clean_row, clean_column):
                structure_exact = False
                raise ValueError(
                    f"matrix structure mismatch at {index}: "
                    f"{(vendor_row, vendor_column)} != {(clean_row, clean_column)}"
                )
            vendor_value = parse_hex_value(vendor_value_text)
            clean_value = parse_hex_value(clean_value_text)
            category = row_class(vendor_row)
            total_by_class[category] += 1
            if vendor_value != clean_value:
                changed_by_class[category] += 1
            metrics.add(
                index,
                vendor_value,
                clean_value,
                row=vendor_row,
                column=vendor_column,
            )
            # This is a deterministic row-major replay of A^T b.  It is useful
            # for attribution, but is deliberately labelled as a replay rather
            # than the runtime CGNR RHS because Eigen may reduce each block in
            # another packet order.
            vendor_rhs[vendor_column] += vendor_value * vendor_residuals[vendor_row]
            clean_rhs[clean_column] += clean_value * clean_residuals[clean_row]
            index += 1

    result = metrics.result()
    result["structure_exact"] = structure_exact
    result["total_by_row_class"] = total_by_class
    result["changed_by_row_class"] = changed_by_class
    return result, vendor_rhs, clean_rhs


def main() -> None:
    args = parse_args()
    prefix = f"ceres_solver_iteration_{args.iteration:03d}"
    vendor_b = load_vector(args.vendor / f"{prefix}_b.txt")
    clean_b = load_vector(args.clean / f"{prefix}_b.txt")
    vendor_d = load_vector(args.vendor / f"{prefix}_D.txt")
    clean_d = load_vector(args.clean / f"{prefix}_D.txt")
    vendor_x = load_vector(args.vendor / f"{prefix}_x.txt")
    clean_x = load_vector(args.clean / f"{prefix}_x.txt")

    matrix, vendor_rhs, clean_rhs = compare_matrix(
        args.vendor / f"{prefix}_A.txt",
        args.clean / f"{prefix}_A.txt",
        vendor_b,
        clean_b,
    )
    residual_metrics = compare_vectors(vendor_b, clean_b)
    changed_residuals_by_class = {"joint": 0, "dynamic": 0, "scene": 0}
    total_residuals_by_class = {"joint": 0, "dynamic": 0, "scene": 0}
    for row, (vendor_value, clean_value) in enumerate(zip(vendor_b, clean_b)):
        category = row_class(row)
        total_residuals_by_class[category] += 1
        changed_residuals_by_class[category] += vendor_value != clean_value
    residual_metrics["total_by_row_class"] = total_residuals_by_class
    residual_metrics["changed_by_row_class"] = changed_residuals_by_class

    result = {
        "vendor": str(args.vendor),
        "clean": str(args.clean),
        "iteration": args.iteration,
        "jacobian_scaled": matrix,
        "residual": residual_metrics,
        "lm_diagonal": compare_vectors(vendor_d, clean_d),
        "linear_step": compare_vectors(vendor_x, clean_x),
        "serial_atb_replay": compare_vectors(vendor_rhs, clean_rhs),
        "serial_atb_vendor_hex": [value.hex() for value in vendor_rhs],
        "serial_atb_clean_hex": [value.hex() for value in clean_rhs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
