#!/usr/bin/env python3
"""Verify the first Joint residual/Jacobian against two product groupings."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_f64(path: Path) -> list[float]:
    payload = path.read_bytes()
    if len(payload) % 8:
        raise ValueError(f"{path} is not f64-aligned")
    return list(struct.unpack(f"<{len(payload) // 8}d", payload))


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def record(index: int, vendor: float, clean: float) -> dict:
    return {
        "index": index,
        "vendor": vendor,
        "clean": clean,
        "vendor_hexfloat": vendor.hex(),
        "clean_hexfloat": clean.hex(),
        "vendor_bits": f"0x{bits(vendor):016x}",
        "clean_bits": f"0x{bits(clean):016x}",
        "bit_exact": bits(vendor) == bits(clean),
        "absolute_delta": abs(clean - vendor),
    }


def parse_grouping(path: Path) -> dict[str, dict[str, list[float]]]:
    result = {
        "left": {"r": [], "J": []},
        "square_first": {"r": [], "J": []},
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        grouping, field, _row, _column, hexfloat, _raw_bits = line.split()
        result[grouping][field].append(float.fromhex(hexfloat))
    for grouping in result.values():
        if len(grouping["r"]) != 2 or len(grouping["J"]) != 8:
            raise ValueError("unexpected grouping probe dimensions")
    return result


def compare(actual: list[float], expected: list[float]) -> dict:
    if len(actual) != len(expected):
        raise ValueError("array dimensions differ")
    matches = [bits(left) == bits(right) for left, right in zip(actual, expected)]
    return {
        "count": len(actual),
        "exact_count": sum(matches),
        "byte_exact": all(matches),
        "mismatch_indices": [index for index, match in enumerate(matches) if not match],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--grouping-results", type=Path, required=True)
    parser.add_argument("--identity-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = {
        "parameters": "first_joint_raw_parameters.bin",
        "residual": "first_joint_raw_residual.bin",
        "jacobian": "first_joint_raw_jacobian.bin",
    }
    vendor = {name: load_f64(args.vendor_dir / filename) for name, filename in names.items()}
    clean = {name: load_f64(args.clean_dir / filename) for name, filename in names.items()}
    grouping = parse_grouping(args.grouping_results)
    identity = json.loads(args.identity_comparison.read_text(encoding="utf-8"))

    result = {
        "verdict": {
            "same_parameters": compare(vendor["parameters"], clean["parameters"])[
                "byte_exact"
            ],
            "vendor_matches_weight_times_squared_difference": compare(
                vendor["residual"], grouping["square_first"]["r"]
            )["byte_exact"]
            and compare(vendor["jacobian"], grouping["square_first"]["J"])[
                "byte_exact"
            ],
            "clean_matches_left_associative_product": compare(
                clean["residual"], grouping["left"]["r"]
            )["byte_exact"]
            and compare(clean["jacobian"], grouping["left"]["J"])["byte_exact"],
            "first_divergence": "JointVarianceResidual raw Jet output before Cauchy correction",
            "root_cause": (
                "vendor evaluates weight * (difference * difference); clean evaluates "
                "(weight * difference) * difference"
            ),
        },
        "capture": {
            "vendor_dir": str(args.vendor_dir),
            "clean_dir": str(args.clean_dir),
            "hashes": {
                side: {
                    filename: sha256(directory / filename)
                    for filename in names.values()
                }
                for side, directory in (
                    ("vendor", args.vendor_dir),
                    ("clean", args.clean_dir),
                )
            },
        },
        "parameters": {
            "comparison": compare(vendor["parameters"], clean["parameters"]),
            "vendor_hexfloat": [value.hex() for value in vendor["parameters"]],
            "clean_hexfloat": [value.hex() for value in clean["parameters"]],
        },
        "raw_residual": {
            "vendor_vs_clean": [
                record(index, vendor_value, clean_value)
                for index, (vendor_value, clean_value) in enumerate(
                    zip(vendor["residual"], clean["residual"])
                )
            ],
            "vendor_vs_square_first": compare(
                vendor["residual"], grouping["square_first"]["r"]
            ),
            "clean_vs_left": compare(clean["residual"], grouping["left"]["r"]),
        },
        "raw_jacobian": {
            "vendor_vs_clean": [
                record(index, vendor_value, clean_value)
                for index, (vendor_value, clean_value) in enumerate(
                    zip(vendor["jacobian"], clean["jacobian"])
                )
            ],
            "vendor_vs_square_first": compare(
                vendor["jacobian"], grouping["square_first"]["J"]
            ),
            "clean_vs_left": compare(clean["jacobian"], grouping["left"]["J"]),
        },
        "identity_propagation": {
            name: {
                key: identity["arrays"][name][key]
                for key in (
                    "count",
                    "changed_count",
                    "first_difference",
                    "maximum_ulp_difference",
                )
            }
            for name in (
                "identity_residual_b.bin",
                "identity_jacobian_scaled_A.bin",
                "identity_jacobian_scaling.bin",
                "identity_cgnr_D.bin",
                "identity_cgnr_rhs_Atb.bin",
                "identity_cgnr_solution_y.bin",
            )
        },
        "inputs": {
            "grouping_results": str(args.grouping_results),
            "grouping_results_sha256": sha256(args.grouping_results),
            "identity_comparison": str(args.identity_comparison),
            "identity_comparison_sha256": sha256(args.identity_comparison),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
