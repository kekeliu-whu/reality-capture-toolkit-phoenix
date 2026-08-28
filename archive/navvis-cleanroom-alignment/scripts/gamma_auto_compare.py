#!/usr/bin/env python3
"""Compare fixed-input automatic Gamma solves and parameter first-use order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


SUMMARY = re.compile(
    r"SUMMARY termination=(?P<termination>-?\d+) "
    r"initial_cost=(?P<initial>[-+0-9.eE]+) final_cost=(?P<final>[-+0-9.eE]+) "
    r"successful_steps=(?P<successful>\d+) unsuccessful_steps=(?P<unsuccessful>\d+)"
)
CLEAN_SUMMARY = re.compile(
    r"Iterations: (?P<iterations>\d+), Initial cost: (?P<initial>[-+0-9.eE]+), "
    r"Final cost: (?P<final>[-+0-9.eE]+), Termination: (?P<termination>\w+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_models(path: Path) -> dict[int, tuple[float, float]]:
    models: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        view_text, gain_text, exponent_text = line.split()
        models[int(view_text)] = (float(gain_text), float(exponent_text))
    return models


def model_metrics(reference: Path, candidate: Path) -> dict[str, object]:
    left = read_models(reference)
    right = read_models(candidate)
    if left.keys() != right.keys():
        raise ValueError("model view coverage differs")
    gain = [abs(left[view][0] - right[view][0]) for view in left]
    exponent = [abs(left[view][1] - right[view][1]) for view in left]
    combined = [max(gain[index], exponent[index]) for index in range(len(gain))]
    worst_index = max(range(len(combined)), key=combined.__getitem__)
    worst_view = list(left)[worst_index]
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
        "byte_exact": reference.read_bytes() == candidate.read_bytes(),
        "view_count": len(left),
        "gain_mae": sum(gain) / len(gain),
        "gain_max": max(gain),
        "exponent_mae": sum(exponent) / len(exponent),
        "exponent_max": max(exponent),
        "parameter_exact_views": sum(
            left[view] == right[view] for view in left
        ),
        "worst_view": worst_view,
        "worst_reference": left[worst_view],
        "worst_candidate": right[worst_view],
    }


def exposure_first_use(path: Path) -> tuple[list[int], int, int]:
    data = path.read_bytes()
    if len(data) % 40:
        raise ValueError("exposure OVS is not 40-byte aligned")
    order: list[int] = []
    seen: set[int] = set()
    joint_blocks = 0
    scalar_residuals = 0
    for offset in range(0, len(data), 40):
        record = data[offset : offset + 40]
        reference = record[3:6]
        observations: list[int] = []
        for rank in range(5):
            start = rank * 8
            quality = struct.unpack_from("<H", record, start + 6)[0]
            if quality == 0:
                continue
            capture = (record[start] << 8) | record[start + 1]
            view = 4 * capture + record[start + 2]
            rgb = record[start + 3 : start + 6]
            intensity = max(rgb)
            if intensity < 6 or intensity > 249:
                continue
            if rank > 0 and max(abs(rgb[channel] - reference[channel]) for channel in range(3)) > 50:
                continue
            observations.append(view)
        if len(observations) < 2:
            continue
        joint_blocks += 1
        scalar_residuals += len(observations)
        for view in observations:
            if view not in seen:
                seen.add(view)
                order.append(view)
    return order, joint_blocks, scalar_residuals


def captured_order(path: Path) -> tuple[list[int], list[int]]:
    rows = []
    first_residuals = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        _, _, first_residual, _, view = line.split("\t")
        rows.append(int(view))
        first_residuals.append(int(first_residual))
    return rows, first_residuals


def parse_summary(directory: Path, vendor: bool) -> dict[str, object]:
    if vendor:
        text = (directory / "capture.txt").read_text(encoding="utf-8")
        match = SUMMARY.search(text)
        if not match:
            raise ValueError(f"vendor summary missing in {directory}")
        return {
            "termination": int(match.group("termination")),
            "initial_cost": float(match.group("initial")),
            "final_cost": float(match.group("final")),
            "successful_steps": int(match.group("successful")),
            "unsuccessful_steps": int(match.group("unsuccessful")),
        }
    text = (directory / "run.stderr").read_text(encoding="utf-8")
    match = CLEAN_SUMMARY.search(text)
    if not match:
        raise ValueError(f"clean summary missing in {directory}")
    return {
        "termination": match.group("termination"),
        "initial_cost_printed": float(match.group("initial")),
        "final_cost_printed": float(match.group("final")),
        "iterations": int(match.group("iterations")),
    }


def repeated_runs(root: Path, vendor: bool) -> dict[str, object]:
    runs = [root / f"run{index}" for index in (1, 2, 3)]
    model_paths = [run / "gamma_models.txt" for run in runs]
    hashes = [sha256(path) for path in model_paths]
    return {
        "model_sha256": hashes,
        "all_byte_exact": len(set(hashes)) == 1,
        "summaries": [parse_summary(run, vendor) for run in runs],
        "run1_vs_run2": model_metrics(model_paths[0], model_paths[1]),
        "run1_vs_run3": model_metrics(model_paths[0], model_paths[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--exposure-ovs", type=Path, required=True)
    parser.add_argument("--vendor-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_order, joint_blocks, scalar_residuals = exposure_first_use(args.exposure_ovs)
    vendor_order, vendor_first_residuals = captured_order(args.vendor_order)
    mismatch = next(
        (index for index, pair in enumerate(zip(expected_order, vendor_order)) if pair[0] != pair[1]),
        None,
    )
    result = {
        "fixed_exposure_ovs": {
            "path": str(args.exposure_ovs),
            "bytes": args.exposure_ovs.stat().st_size,
            "sha256": sha256(args.exposure_ovs),
            "joint_blocks": joint_blocks,
            "scalar_joint_residuals": scalar_residuals,
        },
        "parameter_first_use": {
            "derived_clean_count": len(expected_order),
            "captured_vendor_count": len(vendor_order),
            "exact_order": expected_order == vendor_order,
            "first_mismatch": mismatch,
            "derived_clean_prefix": expected_order[:16],
            "captured_vendor_prefix": vendor_order[:16],
            "vendor_first_residual_prefix": vendor_first_residuals[:16],
        },
        "vendor_t1_repeats": repeated_runs(args.vendor_root, True),
        "clean_t1_repeats": repeated_runs(args.clean_root, False),
        "vendor_t1_vs_clean_t1": model_metrics(
            args.vendor_root / "run1/gamma_models.txt",
            args.clean_root / "run1/gamma_models.txt",
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["parameter_first_use"], indent=2))
    print(json.dumps(result["vendor_t1_vs_clean_t1"], indent=2))


if __name__ == "__main__":
    main()
