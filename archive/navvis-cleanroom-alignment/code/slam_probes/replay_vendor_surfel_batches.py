#!/usr/bin/env python3
"""Replay captured binary ray batches through the clean-room surfel kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from navvis_recon.surveyor_frontend import (  # noqa: E402
    SplitSurfelStatistics,
    extract_valid_split_surfels,
    update_split_surfel_statistics,
)


LEVELS = ((0.10, 0.05), (0.30, 0.15), (0.60, 0.0))


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compare(clean: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    clean = np.ascontiguousarray(clean, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    report: dict[str, object] = {
        "clean_count": int(len(clean)),
        "reference_count": int(len(reference)),
        "same_shape": clean.shape == reference.shape,
    }
    if clean.shape == reference.shape:
        same_order = np.linalg.norm(
            clean.astype(np.float64) - reference.astype(np.float64), axis=1
        )
        report["same_order_bit_exact"] = bool(np.array_equal(clean, reference))
        report["same_order_distance_m"] = statistics(same_order)
    if len(clean) and len(reference):
        clean_to_reference = cKDTree(reference).query(clean, workers=-1)[0]
        reference_to_clean = cKDTree(clean).query(reference, workers=-1)[0]
        report["clean_to_reference_m"] = statistics(clean_to_reference)
        report["reference_to_clean_m"] = statistics(reference_to_clean)
    return report


def load_float3(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape((-1, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--batch-limit",
        type=int,
        help="Replay only the first N captured batches for focused probes",
    )
    args = parser.parse_args()

    batch_paths = sorted(args.batches.glob("batch_*.bin"))
    if args.batch_limit is not None:
        if args.batch_limit <= 0:
            raise ValueError("batch limit must be positive")
        batch_paths = batch_paths[: args.batch_limit]
    if not batch_paths:
        raise ValueError("no captured surfel batches were found")

    states: list[SplitSurfelStatistics | None] = [None, None, None]
    runs: list[dict[str, object]] = []
    saved: dict[str, np.ndarray] = {}
    for batch_index, batch_path in enumerate(batch_paths):
        rays = np.fromfile(batch_path, dtype="<f4").reshape((-1, 6))
        origins = rays[:, :3]
        points = rays[:, 3:]
        level_reports = []
        point_levels = []
        normal_levels = []
        for level, (resolution, offset) in enumerate(LEVELS):
            states[level] = update_split_surfel_statistics(
                states[level], points, origins, resolution, offset
            )
            clean_points, clean_normals = extract_valid_split_surfels(
                states[level], resolution
            )
            reference_points = load_float3(
                args.targets / f"call_{batch_index:02d}_target_level_{level}.bin"
            )
            level_reports.append({
                "level": level,
                "resolution_m": resolution,
                "state_count": int(len(states[level].keys)),
                "points": compare(clean_points, reference_points),
            })
            point_levels.append(clean_points)
            normal_levels.append(clean_normals)
            saved[f"batch_{batch_index}_level_{level}_points"] = np.asarray(
                clean_points, dtype=np.float32
            )
            saved[f"batch_{batch_index}_level_{level}_normals"] = np.asarray(
                clean_normals, dtype=np.float32
            )
            saved[f"batch_{batch_index}_level_{level}_keys"] = states[level].keys
            saved[f"batch_{batch_index}_level_{level}_weights"] = states[level].weights
            saved[f"batch_{batch_index}_level_{level}_counts"] = states[level].counts
            saved[f"batch_{batch_index}_level_{level}_means"] = states[level].means
            saved[f"batch_{batch_index}_level_{level}_covariances"] = (
                states[level].covariances
            )
            saved[f"batch_{batch_index}_level_{level}_secondary_weights"] = (
                states[level].secondary_weights
            )
            saved[f"batch_{batch_index}_level_{level}_secondary_counts"] = (
                states[level].secondary_counts
            )
            saved[f"batch_{batch_index}_level_{level}_secondary_means"] = (
                states[level].secondary_means
            )
            saved[
                f"batch_{batch_index}_level_{level}_secondary_covariances"
            ] = states[level].secondary_covariances
            saved[f"batch_{batch_index}_level_{level}_is_split"] = (
                states[level].is_split
            )
            saved[f"batch_{batch_index}_level_{level}_split_normals"] = (
                states[level].split_normals
            )
            saved[f"batch_{batch_index}_level_{level}_viewpoints"] = (
                states[level].viewpoints
            )
            saved[f"batch_{batch_index}_level_{level}_secondary_viewpoints"] = (
                states[level].secondary_viewpoints
            )
            saved[f"batch_{batch_index}_level_{level}_primary_dirty"] = (
                states[level].primary_dirty
            )
            saved[f"batch_{batch_index}_level_{level}_secondary_dirty"] = (
                states[level].secondary_dirty
            )

        clean_points = np.concatenate(point_levels)
        clean_normals = np.concatenate(normal_levels)
        reference_points = load_float3(
            args.targets / f"call_{batch_index:02d}_target_points.bin"
        )
        reference_normals = load_float3(
            args.targets / f"call_{batch_index:02d}_target_normals.bin"
        )
        runs.append({
            "batch": batch_index,
            "ray_count": int(len(rays)),
            "levels": level_reports,
            "flattened_points": compare(clean_points, reference_points),
            "flattened_normals": compare(clean_normals, reference_normals),
        })

    report = {"levels": [list(level) for level in LEVELS], "runs": runs}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.state is not None:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state, **saved)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
