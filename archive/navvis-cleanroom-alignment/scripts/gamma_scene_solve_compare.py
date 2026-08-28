#!/usr/bin/env python3
"""Compare the isolated Scene-float solve against vendor and clean baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SUMMARY_PATTERN = re.compile(
    r"SUMMARY termination=(?P<termination>-?\d+) "
    r"initial_cost=(?P<initial>[-+0-9.eE]+) "
    r"final_cost=(?P<final>[-+0-9.eE]+) "
    r"successful_steps=(?P<successful>\d+) unsuccessful_steps=(?P<unsuccessful>\d+)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_models(path: Path) -> dict[int, tuple[float, float]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        view, gain, exponent = line.split()
        result[int(view)] = (float(gain), float(exponent))
    return result


def read_vendor_summary(directory: Path) -> dict[str, float | int]:
    text = (directory / "capture.txt").read_text(encoding="utf-8")
    match = SUMMARY_PATTERN.search(text)
    if not match:
        raise ValueError(f"cannot parse vendor summary in {directory}")
    return {
        "termination": int(match.group("termination")),
        "initial_cost": float(match.group("initial")),
        "final_cost": float(match.group("final")),
        "successful_steps": int(match.group("successful")),
        "unsuccessful_steps": int(match.group("unsuccessful")),
    }


def read_json_summary(directory: Path) -> dict[str, float | int]:
    return json.loads((directory / "clean_solver_summary.json").read_text(encoding="utf-8"))


def read_clean_baseline_summary(directory: Path) -> dict[str, float | int]:
    # The exact clean baseline summary was captured separately because the
    # original three production repeats did not run under GDB.
    text = (directory / "clean_solver_summary.txt").read_text(encoding="utf-8")
    match = SUMMARY_PATTERN.search(text)
    if not match:
        raise ValueError(f"cannot parse clean baseline summary in {directory}")
    return {
        "termination": int(match.group("termination")),
        "initial_cost": float(match.group("initial")),
        "final_cost": float(match.group("final")),
        "successful_steps": int(match.group("successful")),
        "unsuccessful_steps": int(match.group("unsuccessful")),
    }


def model_metrics(reference: dict[int, tuple[float, float]], candidate: dict[int, tuple[float, float]]) -> dict:
    views = sorted(reference)
    if views != sorted(candidate):
        raise ValueError("model view sets differ")
    gain_errors = [abs(candidate[v][0] - reference[v][0]) for v in views]
    exponent_errors = [abs(candidate[v][1] - reference[v][1]) for v in views]
    combined = [gain_errors[i] + exponent_errors[i] for i in range(len(views))]
    worst_index = max(range(len(views)), key=combined.__getitem__)
    worst_view = views[worst_index]
    return {
        "view_count": len(views),
        "exact_view_count": sum(candidate[v] == reference[v] for v in views),
        "gain_mae": sum(gain_errors) / len(gain_errors),
        "gain_max": max(gain_errors),
        "exponent_mae": sum(exponent_errors) / len(exponent_errors),
        "exponent_max": max(exponent_errors),
        "worst_view": worst_view,
        "worst_reference": {
            "gain": reference[worst_view][0],
            "exponent": reference[worst_view][1],
        },
        "worst_candidate": {
            "gain": candidate[worst_view][0],
            "exponent": candidate[worst_view][1],
        },
    }


def repeat_metrics(root: Path, summary_reader) -> dict:
    directories = [root / f"run{index}" for index in (1, 2, 3)]
    hashes = [sha256(directory / "gamma_models.txt") for directory in directories]
    summaries = [summary_reader(directory) for directory in directories]
    return {
        "gamma_sha256": hashes,
        "gamma_byte_deterministic": len(set(hashes)) == 1,
        "summaries": summaries,
        "summary_deterministic": all(summary == summaries[0] for summary in summaries[1:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--patched-root", type=Path, required=True)
    parser.add_argument("--clean-baseline-model", type=Path, required=True)
    parser.add_argument("--clean-baseline-summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vendor_models = read_models(args.vendor_root / "run1" / "gamma_models.txt")
    patched_models = read_models(args.patched_root / "run1" / "gamma_models.txt")
    clean_models = read_models(args.clean_baseline_model)
    vendor_summary = read_vendor_summary(args.vendor_root / "run1")
    patched_summary = read_json_summary(args.patched_root / "run1")
    clean_summary = read_clean_baseline_summary(args.clean_baseline_summary_dir)

    result = {
        "vendor_repeats": repeat_metrics(args.vendor_root, read_vendor_summary),
        "scene_float_repeats": repeat_metrics(args.patched_root, read_json_summary),
        "models": {
            "clean_double_scene_vs_vendor": model_metrics(vendor_models, clean_models),
            "scene_float_vs_vendor": model_metrics(vendor_models, patched_models),
        },
        "costs": {
            "vendor": vendor_summary,
            "clean_double_scene": clean_summary,
            "scene_float": patched_summary,
            "scene_float_initial_minus_vendor": float(patched_summary["initial_cost"])
            - float(vendor_summary["initial_cost"]),
            "scene_float_final_minus_vendor": float(patched_summary["final_cost"])
            - float(vendor_summary["final_cost"]),
            "clean_final_minus_vendor": float(clean_summary["final_cost"])
            - float(vendor_summary["final_cost"]),
        },
    }
    before = result["models"]["clean_double_scene_vs_vendor"]
    after = result["models"]["scene_float_vs_vendor"]
    result["improvement"] = {
        "gain_mae_reduction_fraction": 1.0 - after["gain_mae"] / before["gain_mae"],
        "exponent_mae_reduction_fraction": 1.0
        - after["exponent_mae"] / before["exponent_mae"],
        "final_cost_gap_reduction_fraction": 1.0
        - abs(result["costs"]["scene_float_final_minus_vendor"])
        / abs(result["costs"]["clean_final_minus_vendor"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
