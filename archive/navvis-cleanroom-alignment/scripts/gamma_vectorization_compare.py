#!/usr/bin/env python3
"""Compare first-step Gamma models and Ceres summaries across ISA variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import struct
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=pathlib.Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Variant name and max_iterations=1 output directory",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(path: pathlib.Path) -> list[tuple[int, float, float]]:
    models: list[tuple[int, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        view_text, gain_text, exponent_text = line.split()
        models.append((int(view_text), float(gain_text), float(exponent_text)))
    return models


def ordered_bits(value: float) -> int:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def field_metrics(reference: Iterable[float], candidate: Iterable[float]) -> dict:
    pairs = list(zip(reference, candidate))
    absolute = [abs(left - right) for left, right in pairs]
    ulps = [abs(ordered_bits(left) - ordered_bits(right)) for left, right in pairs]
    return {
        "exact": sum(left == right for left, right in pairs),
        "mae": statistics.fmean(absolute),
        "max_abs": max(absolute),
        "median_ulp": statistics.median(ulps),
        "p95_ulp": percentile(ulps, 0.95),
        "max_ulp": max(ulps),
    }


def iteration_delta(vendor: dict, candidate: dict) -> dict:
    numeric_fields = (
        "cost",
        "cost_change",
        "gradient_max_norm",
        "gradient_norm",
        "step_norm",
        "relative_decrease",
        "trust_region_radius",
        "eta",
    )
    result = {
        "iteration": candidate["iteration"],
        "linear_solver_iterations": candidate["linear_solver_iterations"],
        "linear_solver_iterations_delta": (
            candidate["linear_solver_iterations"] - vendor["linear_solver_iterations"]
        ),
    }
    for field in numeric_fields:
        result[field] = candidate[field]
        result[f"{field}_delta"] = candidate[field] - vendor[field]
    return result


def compare_variant(vendor_dir: pathlib.Path, candidate_dir: pathlib.Path) -> dict:
    vendor_path = vendor_dir / "gamma_models.txt"
    candidate_path = candidate_dir / "gamma_models.txt"
    vendor = load_models(vendor_path)
    candidate = load_models(candidate_path)
    if len(vendor) != len(candidate):
        raise ValueError(f"model count mismatch: {len(vendor)} != {len(candidate)}")
    if [row[0] for row in vendor] != [row[0] for row in candidate]:
        raise ValueError("view order mismatch")

    vendor_iterations = json.loads(
        (vendor_dir / "iteration_summaries.json").read_text(encoding="utf-8")
    )
    candidate_iterations = json.loads(
        (candidate_dir / "iteration_summaries.json").read_text(encoding="utf-8")
    )
    if len(vendor_iterations) != len(candidate_iterations):
        raise ValueError("iteration count mismatch")

    gain_metrics = field_metrics(
        (row[1] for row in vendor), (row[1] for row in candidate)
    )
    exponent_metrics = field_metrics(
        (row[2] for row in vendor), (row[2] for row in candidate)
    )
    exact_views = sum(
        vendor_row[1:] == candidate_row[1:]
        for vendor_row, candidate_row in zip(vendor, candidate)
    )

    return {
        "candidate_dir": str(candidate_dir),
        "gamma_sha256": sha256(candidate_path),
        "byte_exact_gamma": candidate_path.read_bytes() == vendor_path.read_bytes(),
        "views": len(candidate),
        "exact_views": exact_views,
        "gain": gain_metrics,
        "exponent": exponent_metrics,
        "iterations": [
            iteration_delta(vendor_row, candidate_row)
            for vendor_row, candidate_row in zip(vendor_iterations, candidate_iterations)
        ],
    }


def main() -> None:
    args = parse_args()
    variants: dict[str, pathlib.Path] = {}
    for item in args.variant:
        if "=" not in item:
            raise ValueError(f"invalid --variant value: {item}")
        name, directory = item.split("=", 1)
        variants[name] = pathlib.Path(directory)

    result = {
        "vendor_dir": str(args.vendor),
        "vendor_gamma_sha256": sha256(args.vendor / "gamma_models.txt"),
        "variants": {
            name: compare_variant(args.vendor, directory)
            for name, directory in variants.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
