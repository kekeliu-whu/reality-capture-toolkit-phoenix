#!/usr/bin/env python3
"""Verify the recovered Z1 HASH_MAP_FIRST_POINT/node0 selection formula."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np


def load_capture(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:8] != b"NVAVF24\0" or len(data) < 40:
        raise ValueError(f"not an AdaptiveVoxelFilter capture: {path}")
    _, begin, end, _ = struct.unpack_from("<QQQQ", data, 8)
    count = (end - begin) // 24
    records = np.frombuffer(data, dtype=np.dtype("V24"), offset=40)
    if len(records) != count:
        raise ValueError(f"capture count mismatch: {path}")
    return records.copy()


def words(records: np.ndarray) -> np.ndarray:
    return records.view("<u4").reshape((-1, 6))


def stable_first_point_indices(records: np.ndarray, length: float) -> np.ndarray:
    # A RangeMeasurement is two Vector3f values.  The hit position is the
    # second one; the first one is the per-ray origin.
    positions = words(records)[:, 3:6].view("<f4").reshape((-1, 3))
    resolution = np.float32(length)
    keys = np.floor(positions / resolution).astype(np.int64)
    seen: set[tuple[int, int, int]] = set()
    selected: list[int] = []
    for index, key in enumerate(keys):
        value = (int(key[0]), int(key[1]), int(key[2]))
        if value not in seen:
            seen.add(value)
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def adaptive_grid_indices(
    records: np.ndarray,
    minimum_length: float,
    maximum_length: float,
    maximum_points: int,
    maximum_iterations: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Replay the installed float32 bisection and return its dense side."""

    maximum_bound = np.float32(maximum_length)
    minimum_bound = np.float32(minimum_length)
    sparse = stable_first_point_indices(records, maximum_bound)
    dense = stable_first_point_indices(records, minimum_bound)
    calls: list[dict[str, float | int]] = [
        {"resolution_float32": float(maximum_bound), "output_count": len(sparse)},
        {"resolution_float32": float(minimum_bound), "output_count": len(dense)},
    ]
    if len(sparse) > maximum_points:
        return sparse, calls
    for _ in range(maximum_iterations):
        if (len(dense) - maximum_points) / maximum_points <= 0.1:
            break
        middle = np.float32(
            np.float32(0.5) * np.float32(maximum_bound + minimum_bound)
        )
        candidate = stable_first_point_indices(records, middle)
        calls.append(
            {"resolution_float32": float(middle), "output_count": len(candidate)}
        )
        if len(candidate) < maximum_points:
            maximum_bound = middle
            sparse = candidate
        else:
            minimum_bound = middle
            dense = candidate
    return dense, calls


def deterministic_index_indices(count: int, maximum_points: int) -> np.ndarray:
    if count <= maximum_points:
        return np.arange(count, dtype=np.int64)
    # This deliberately starts at input index 1.  It is not
    # floor(k * count / maximum_points).
    return (
        np.floor(
            np.arange(maximum_points, dtype=np.float64)
            * float(count)
            / float(maximum_points)
        ).astype(np.int64)
        + 1
    )


def parse_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def wire_values(data: bytes, field_number: int) -> list[bytes]:
    values: list[bytes] = []
    offset = 0
    while offset < len(data):
        tag, offset = parse_varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            _, offset = parse_varint(data, offset)
        elif wire == 1:
            offset += 8
        elif wire == 2:
            size, offset = parse_varint(data, offset)
            value = data[offset : offset + size]
            offset += size
            if number == field_number:
                values.append(value)
        elif wire == 5:
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
    return values


def frozen_node0_points(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        data = archive.read("trajectory_node_00000000.pb")
    top = wire_values(data, 1)
    if len(top) != 1:
        raise ValueError("unexpected trajectory node cloud wrapper")
    points = []
    for message in wire_values(top[0], 1):
        values: dict[int, float] = {}
        offset = 0
        while offset < len(message):
            tag, offset = parse_varint(message, offset)
            number, wire = tag >> 3, tag & 7
            if wire != 5:
                raise ValueError("unexpected Vector3f encoding")
            values[number] = struct.unpack_from("<f", message, offset)[0]
            offset += 4
        points.append((values[1], values[2], values[3]))
    return np.asarray(points, dtype="<f4")


def digest(records: np.ndarray) -> str:
    return hashlib.sha256(records.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--frozen-node-clouds", type=Path, required=True)
    args = parser.parse_args()

    raw = load_capture(args.capture_dir / "node0_adaptive_input.bin")
    captured_grid = load_capture(args.capture_dir / "node0_index_input.bin")
    captured_output = load_capture(args.capture_dir / "node0_adaptive_output.bin")
    trace_path = args.capture_dir / "trace.json"
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else None

    first, predicted_grid_calls = adaptive_grid_indices(raw, 0.02, 0.4, 5000, 10)
    predicted_grid = raw[first]
    cap = deterministic_index_indices(len(predicted_grid), 5000)
    predicted_output = predicted_grid[cap]
    frozen = frozen_node0_points(args.frozen_node_clouds)
    predicted_points = (
        words(predicted_output)[:, 3:6].view("<f4").reshape((-1, 3))
    )

    result = {
        "adaptive_input_count": int(len(raw)),
        "grid_count": int(len(predicted_grid)),
        "output_count": int(len(predicted_output)),
        "grid_records_exact": bool(np.array_equal(predicted_grid, captured_grid)),
        "output_records_exact": bool(
            np.array_equal(predicted_output, captured_output)
        ),
        "frozen_xyz_float32_exact": bool(
            np.array_equal(predicted_points.view("<u4"), frozen.view("<u4"))
        ),
        "grid_call_trace_exact": bool(
            trace is None
            or [
                (call["resolution_float32"], call["output_count"])
                for call in predicted_grid_calls
            ]
            == [
                (call["resolution_float32"], call["output_count"])
                for call in trace["grid_calls"]
            ]
        ),
        "predicted_grid_calls": predicted_grid_calls,
        "first_grid_raw_indices": first[:16].tolist(),
        "first_index_filter_indices": cap[:16].tolist(),
        "captured_output_sha256": digest(captured_output),
        "predicted_output_sha256": digest(predicted_output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(
        result[key]
        for key in (
            "grid_records_exact",
            "output_records_exact",
            "frozen_xyz_float32_exact",
            "grid_call_trace_exact",
        )
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
