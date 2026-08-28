#!/usr/bin/env python3
"""Compare two same-point-order packed nv_colorcloud OVS captures.

Each point owns five 8-byte observations:
  capture_be16, camera_u8, rgb_u8[3], quality_le16.

The implementation is chunked so full multi-million-point captures can be
compared without constructing an N x 5 x 5 tensor for the entire cloud.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


RECORD_BYTES = 40
SLOTS = 5


def load(path: Path) -> np.memmap:
    size = path.stat().st_size
    if size % RECORD_BYTES:
        raise ValueError(f"{path}: size {size} is not divisible by {RECORD_BYTES}")
    return np.memmap(path, dtype=np.uint8, mode="r").reshape(-1, SLOTS, 8)


def decode(records: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    capture = records[:, :, 0].astype(np.int32) * 256 + records[:, :, 1]
    camera = records[:, :, 2].astype(np.int32)
    quality = (
        records[:, :, 6].astype(np.uint16)
        + records[:, :, 7].astype(np.uint16) * np.uint16(256)
    )
    valid = quality > 0
    view = capture * 4 + camera
    view[~valid] = -1
    return view, records[:, :, 3:6], quality, valid


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--chunk-points", type=int, default=250_000)
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    point_count = reference.shape[0]

    ref_direct_total = 0
    cand_direct_total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_records = 0
    same_best_view = 0
    same_view_set = 0
    matched_observations = 0
    matched_rgb_exact = 0
    matched_rgb_abs = 0
    matched_rgb_max = 0
    matched_quality_exact = 0
    matched_quality_abs = 0
    matched_quality_max = 0
    ref_observations = 0
    cand_observations = 0
    ref_slots = np.zeros(SLOTS + 1, dtype=np.int64)
    cand_slots = np.zeros(SLOTS + 1, dtype=np.int64)
    ref_view_counts: np.ndarray | None = None
    cand_view_counts: np.ndarray | None = None
    matched_view_counts: np.ndarray | None = None

    for begin in range(0, point_count, args.chunk_points):
        end = min(point_count, begin + args.chunk_points)
        ref_raw = np.asarray(reference[begin:end])
        cand_raw = np.asarray(candidate[begin:end])
        ref_view, ref_rgb, ref_quality, ref_valid = decode(ref_raw)
        cand_view, cand_rgb, cand_quality, cand_valid = decode(cand_raw)

        ref_direct = ref_valid.any(axis=1)
        cand_direct = cand_valid.any(axis=1)
        both = ref_direct & cand_direct
        ref_direct_total += int(ref_direct.sum())
        cand_direct_total += int(cand_direct.sum())
        true_positive += int(both.sum())
        false_positive += int((~ref_direct & cand_direct).sum())
        false_negative += int((ref_direct & ~cand_direct).sum())
        exact_records += int(np.all(ref_raw == cand_raw, axis=(1, 2)).sum())
        same_best_view += int((both & (ref_view[:, 0] == cand_view[:, 0])).sum())

        ref_valid_count = ref_valid.sum(axis=1)
        cand_valid_count = cand_valid.sum(axis=1)
        ref_observations += int(ref_valid_count.sum())
        cand_observations += int(cand_valid_count.sum())
        ref_slots += np.bincount(ref_valid_count, minlength=SLOTS + 1)
        cand_slots += np.bincount(cand_valid_count, minlength=SLOTS + 1)

        max_view = max(
            int(ref_view.max(initial=-1)), int(cand_view.max(initial=-1)))
        needed = max_view + 1
        if ref_view_counts is None:
            ref_view_counts = np.zeros(needed, dtype=np.int64)
            cand_view_counts = np.zeros(needed, dtype=np.int64)
            matched_view_counts = np.zeros(needed, dtype=np.int64)
        elif needed > ref_view_counts.size:
            ref_view_counts = np.pad(ref_view_counts, (0, needed - ref_view_counts.size))
            cand_view_counts = np.pad(cand_view_counts, (0, needed - cand_view_counts.size))
            matched_view_counts = np.pad(
                matched_view_counts, (0, needed - matched_view_counts.size))
        ref_view_counts += np.bincount(
            ref_view[ref_valid], minlength=ref_view_counts.size)
        cand_view_counts += np.bincount(
            cand_view[cand_valid], minlength=cand_view_counts.size)

        sorted_ref = np.sort(ref_view, axis=1)
        sorted_cand = np.sort(cand_view, axis=1)
        same_view_set += int((both & np.all(sorted_ref == sorted_cand, axis=1)).sum())

        # A view appears at most once in an OVS record. Match each reference
        # observation to any candidate slot with the same point/view key.
        for ref_slot in range(SLOTS):
            valid_ref_slot = ref_valid[:, ref_slot]
            if not valid_ref_slot.any():
                continue
            matches = (
                valid_ref_slot[:, None]
                & cand_valid
                & (ref_view[:, ref_slot, None] == cand_view)
            )
            has_match = matches.any(axis=1)
            if not has_match.any():
                continue
            cand_slot = matches.argmax(axis=1)
            indices = np.flatnonzero(has_match)
            selected_slots = cand_slot[indices]
            view_ids = ref_view[indices, ref_slot]
            matched_view_counts += np.bincount(
                view_ids, minlength=matched_view_counts.size)
            rgb_delta = np.abs(
                ref_rgb[indices, ref_slot].astype(np.int16)
                - cand_rgb[indices, selected_slots].astype(np.int16)
            )
            quality_delta = np.abs(
                ref_quality[indices, ref_slot].astype(np.int32)
                - cand_quality[indices, selected_slots].astype(np.int32)
            )
            matched_observations += int(indices.size)
            matched_rgb_exact += int(np.all(rgb_delta == 0, axis=1).sum())
            matched_rgb_abs += int(rgb_delta.sum())
            matched_rgb_max = max(matched_rgb_max, int(rgb_delta.max(initial=0)))
            matched_quality_exact += int((quality_delta == 0).sum())
            matched_quality_abs += int(quality_delta.sum())
            matched_quality_max = max(
                matched_quality_max, int(quality_delta.max(initial=0)))

    assert ref_view_counts is not None
    assert cand_view_counts is not None
    assert matched_view_counts is not None
    per_view: dict[str, Any] = {}
    for view in range(ref_view_counts.size):
        if ref_view_counts[view] == 0 and cand_view_counts[view] == 0:
            continue
        per_view[str(view)] = {
            "reference_observations": int(ref_view_counts[view]),
            "candidate_observations": int(cand_view_counts[view]),
            "candidate_minus_reference": int(
                cand_view_counts[view] - ref_view_counts[view]),
            "matched_observations": int(matched_view_counts[view]),
        }

    result = {
        "reference": {
            "path": str(args.reference),
            "sha256": file_hash(args.reference),
            "points": point_count,
            "direct_points": ref_direct_total,
            "observations": ref_observations,
            "valid_slots_per_point": {
                str(i): int(value) for i, value in enumerate(ref_slots)
            },
        },
        "candidate": {
            "path": str(args.candidate),
            "sha256": file_hash(args.candidate),
            "points": point_count,
            "direct_points": cand_direct_total,
            "observations": cand_observations,
            "valid_slots_per_point": {
                str(i): int(value) for i, value in enumerate(cand_slots)
            },
        },
        "comparison": {
            "direct_mask": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": point_count - true_positive - false_positive - false_negative,
                "iou": true_positive / max(1, true_positive + false_positive + false_negative),
            },
            "exact_records": exact_records,
            "exact_record_fraction": exact_records / point_count,
            "same_best_view_points": same_best_view,
            "same_best_view_fraction_of_both_direct": same_best_view / max(1, true_positive),
            "same_view_set_points": same_view_set,
            "same_view_set_fraction_of_both_direct": same_view_set / max(1, true_positive),
            "matched_by_point_and_view": {
                "observations": matched_observations,
                "reference_observation_recall": matched_observations / max(1, ref_observations),
                "rgb_exact": matched_rgb_exact,
                "rgb_channel_mae": matched_rgb_abs / max(1, 3 * matched_observations),
                "rgb_channel_max_delta": matched_rgb_max,
                "quality_exact": matched_quality_exact,
                "quality_u16_mae": matched_quality_abs / max(1, matched_observations),
                "quality_u16_max_delta": matched_quality_max,
            },
            "per_view": per_view,
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
