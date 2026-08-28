#!/usr/bin/env python3
"""Verify the node0 0.04 m HASH_MAP_CENTROID boundary and ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np


def load_capture(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] not in (b"NVHRF24\0", b"NVAVF24\0") or len(data) < 40:
        raise ValueError(f"not a 24-byte range-measurement capture: {path}")
    _, begin, end, _ = struct.unpack_from("<QQQQ", data, 8)
    count = (end - begin) // 24
    records = np.frombuffer(data, dtype="<f4", offset=40).reshape((-1, 6))
    if len(records) != count:
        raise ValueError(f"capture count mismatch: {path}")
    return records.copy()


def voxel_keys(records: np.ndarray, resolution: np.float32) -> np.ndarray:
    # The binary promotes both the input float32 and the float32 configuration
    # value to double before applying floor.  This is bit-equivalent to the
    # expression below for every node0 point.
    return np.floor(
        records[:, 3:6].astype(np.float64) / float(resolution)
    ).astype(np.int32)


def octant(hit: np.ndarray) -> int:
    return (
        int(hit[0] >= np.float32(0.0))
        | (int(hit[1] >= np.float32(0.0)) << 1)
        | (int(hit[2] >= np.float32(0.0)) << 2)
    )


def centroid_filter(records: np.ndarray, resolution: np.float32):
    keys = voxel_keys(records, resolution)
    order: list[tuple[int, int, int]] = []
    members: dict[tuple[int, int, int], list[int]] = {}
    first_index: dict[tuple[int, int, int], int] = {}
    key_octant: dict[tuple[int, int, int], int] = {}

    for index, key_array in enumerate(keys):
        key = tuple(int(value) for value in key_array)
        if key not in members:
            order.append(key)
            members[key] = []
            first_index[key] = index
            key_octant[key] = octant(records[index, 3:6])
        members[key].append(index)

    ordered_keys = [
        key
        for octant_index in range(8)
        for key in order
        if key_octant[key] == octant_index
    ]

    output = np.empty((len(ordered_keys), 6), dtype="<f4")
    for output_index, key in enumerate(ordered_keys):
        indices = members[key]
        mean = records[indices[0]].copy()
        count = np.float32(1.0)
        for input_index in indices[1:]:
            new_count = np.float32(count + np.float32(1.0))
            reciprocal = np.float32(np.float32(1.0) / new_count)
            # Keep every scalar float32 rounding point visible.  Vectorizing
            # this expression can silently contract or promote operations.
            for component in range(6):
                weighted = np.float32(mean[component] * count)
                weighted = np.float32(weighted + records[input_index, component])
                mean[component] = np.float32(weighted * reciprocal)
            count = new_count
        output[output_index] = mean

    return output, ordered_keys, first_index, keys


def digest(records: np.ndarray) -> str:
    return hashlib.sha256(records.tobytes()).hexdigest()


def distance_stats_mm(predicted: np.ndarray, captured: np.ndarray) -> dict:
    distances = np.linalg.norm(
        predicted.astype(np.float64) - captured.astype(np.float64), axis=1
    ) * 1000.0
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.quantile(distances, 0.95)),
        "max": float(np.max(distances)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--cleanroom-npz",
        type=Path,
        help="NPZ containing the clean-room node0 59184x6 array as 'raw'",
    )
    args = parser.parse_args()

    resolution = np.float32(0.04)
    official_input = load_capture(args.capture_dir / "node0_centroid_input.bin")
    official_output = load_capture(args.capture_dir / "node0_centroid_output.bin")
    wrapper_input = load_capture(args.capture_dir / "node0_wrapper_input.bin")
    adaptive_input = load_capture(args.capture_dir / "node0_adaptive_input.bin")

    predicted, ordered_keys, first_index, official_input_keys = centroid_filter(
        official_input, resolution
    )
    official_output_keys = voxel_keys(official_output, resolution)
    predicted_keys = np.asarray(ordered_keys, dtype=np.int32)
    output_octants = np.asarray(
        [octant(hit) for hit in official_output[:, 3:6]], dtype=np.int8
    )
    octant_changes = (np.flatnonzero(output_octants[1:] != output_octants[:-1]) + 1)

    result: dict[str, object] = {
        "resolution_float32": float(resolution),
        "official_input_count": int(len(official_input)),
        "official_output_count": int(len(official_output)),
        "unique_voxel_count": int(len(ordered_keys)),
        "official_formula_records_exact": bool(
            np.array_equal(predicted.view("<u4"), official_output.view("<u4"))
        ),
        "official_formula_key_order_exact": bool(
            np.array_equal(predicted_keys, official_output_keys)
        ),
        "centroid_output_equals_wrapper_input": bool(
            np.array_equal(official_output.view("<u4"), wrapper_input.view("<u4"))
        ),
        "wrapper_input_equals_adaptive_input": bool(
            np.array_equal(wrapper_input.view("<u4"), adaptive_input.view("<u4"))
        ),
        "octant_counts": np.bincount(output_octants, minlength=8).tolist(),
        "octant_change_indices": octant_changes.tolist(),
        "official_input_payload_sha256": digest(official_input),
        "official_output_payload_sha256": digest(official_output),
        "predicted_output_payload_sha256": digest(predicted),
    }

    if args.cleanroom_npz:
        with np.load(args.cleanroom_npz) as archive:
            cleanroom_input = np.asarray(archive["raw"], dtype="<f4")
        cleanroom_predicted, cleanroom_ordered_keys, cleanroom_first, clean_keys = (
            centroid_filter(cleanroom_input, resolution)
        )
        clean_order_array = np.asarray(cleanroom_ordered_keys, dtype=np.int32)
        official_key_tuples = [tuple(int(value) for value in key) for key in official_input_keys]
        clean_key_tuples = [tuple(int(value) for value in key) for key in clean_keys]
        official_first: dict[tuple[int, int, int], int] = {}
        for index, key in enumerate(official_key_tuples):
            official_first.setdefault(key, index)

        result["cleanroom"] = {
            "input_count": int(len(cleanroom_input)),
            "input_voxel_keys_indexwise_exact": int(
                np.count_nonzero(np.all(clean_keys == official_input_keys, axis=1))
            ),
            "input_voxel_key_sequence_exact": bool(
                np.array_equal(clean_keys, official_input_keys)
            ),
            "first_occurrence_indices_exact": int(
                sum(
                    official_first[key] == cleanroom_first[key]
                    for key in official_first.keys() & cleanroom_first.keys()
                )
            ),
            "centroid_key_order_exact": int(
                np.count_nonzero(
                    np.all(clean_order_array == official_output_keys, axis=1)
                )
            ),
            "centroid_key_sequence_exact": bool(
                np.array_equal(clean_order_array, official_output_keys)
            ),
            "hit_xyz_distance_mm": distance_stats_mm(
                cleanroom_predicted[:, 3:6], official_output[:, 3:6]
            ),
            "origin_xyz_distance_mm": distance_stats_mm(
                cleanroom_predicted[:, 0:3], official_output[:, 0:3]
            ),
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    required = (
        result["official_formula_records_exact"],
        result["official_formula_key_order_exact"],
        result["centroid_output_equals_wrapper_input"],
        result["wrapper_input_equals_adaptive_input"],
    )
    if args.cleanroom_npz:
        clean_result = result["cleanroom"]
        required += (
            clean_result["input_voxel_key_sequence_exact"],
            clean_result["centroid_key_sequence_exact"],
        )
    return 0 if all(required) else 1


if __name__ == "__main__":
    sys.exit(main())
