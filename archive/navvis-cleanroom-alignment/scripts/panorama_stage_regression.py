#!/usr/bin/env python3
"""Measure exactness and image quality for frozen panorama stage pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ssim_gray(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    reference = reference.astype(np.float32)
    candidate = candidate.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mean_reference = cv2.GaussianBlur(reference, (11, 11), 1.5)
    mean_candidate = cv2.GaussianBlur(candidate, (11, 11), 1.5)
    variance_reference = (
        cv2.GaussianBlur(reference * reference, (11, 11), 1.5)
        - mean_reference * mean_reference
    )
    variance_candidate = (
        cv2.GaussianBlur(candidate * candidate, (11, 11), 1.5)
        - mean_candidate * mean_candidate
    )
    covariance = (
        cv2.GaussianBlur(reference * candidate, (11, 11), 1.5)
        - mean_reference * mean_candidate
    )
    score = (
        (2.0 * mean_reference * mean_candidate + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_reference * mean_reference + mean_candidate * mean_candidate + c1)
            * (variance_reference + variance_candidate + c2)
        )
    )
    return float(np.mean(score))


def compare(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    reference = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    candidate = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
    if reference is None or candidate is None:
        raise ValueError(f"cannot decode {reference_path} or {candidate_path}")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch for {reference_path} and {candidate_path}: "
            f"{reference.shape} != {candidate.shape}"
        )

    absolute = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    squared = absolute.astype(np.float64) ** 2
    mse = float(np.mean(squared))
    result: dict[str, object] = {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_sha256": sha256(reference_path),
        "candidate_sha256": sha256(candidate_path),
        "shape": list(reference.shape),
        "absolute_error_sum": int(absolute.sum(dtype=np.int64)),
        "mae": float(absolute.mean()),
        "max_absolute_error": int(absolute.max(initial=0)),
        "different_values": int(np.count_nonzero(absolute)),
        "exact": bool(not np.any(absolute)),
        "psnr_db": math.inf if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse),
        "ssim_gray": ssim_gray(reference, candidate),
    }
    if reference.ndim == 2:
        reference_valid = reference != 0
        candidate_valid = candidate != 0
        intersection = int(np.count_nonzero(reference_valid & candidate_valid))
        union = int(np.count_nonzero(reference_valid | candidate_valid))
        result["mask"] = {
            "reference_nonzero": int(np.count_nonzero(reference_valid)),
            "candidate_nonzero": int(np.count_nonzero(candidate_valid)),
            "intersection": intersection,
            "union": union,
            "iou": intersection / union if union else 1.0,
            "false_positive": int(np.count_nonzero(~reference_valid & candidate_valid)),
            "false_negative": int(np.count_nonzero(reference_valid & ~candidate_valid)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair", nargs=3, action="append", metavar=("LABEL", "REFERENCE", "CANDIDATE"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparisons = {
        label: compare(Path(reference), Path(candidate))
        for label, reference, candidate in args.pair
    }
    result = {
        "all_exact": all(item["exact"] for item in comparisons.values()),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
