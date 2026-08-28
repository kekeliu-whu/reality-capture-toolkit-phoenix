#!/usr/bin/env python3
"""Verify full-recording CloudBuilder filter decisions against binary oracles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def recover_sensor_interleave(
    sensor_rows: tuple[list[dict[str, str]], list[dict[str, str]]],
    vendor_rows: list[dict[str, str]],
) -> list[int]:
    """Recover the unique merge of the per-sensor scan streams by input count."""
    states: dict[int, None] = {0: None}
    parents: list[dict[int, tuple[int, int]]] = [{}]
    for step, vendor in enumerate(vendor_rows):
        value = vendor["input_count"]
        next_parents: dict[int, tuple[int, int]] = {}
        for horizontal_count in states:
            vertical_count = step - horizontal_count
            if (
                horizontal_count < len(sensor_rows[0])
                and sensor_rows[0][horizontal_count]["input_count"] == value
            ):
                next_parents.setdefault(
                    horizontal_count + 1, (horizontal_count, 0)
                )
            if (
                vertical_count < len(sensor_rows[1])
                and sensor_rows[1][vertical_count]["input_count"] == value
            ):
                next_parents.setdefault(horizontal_count, (horizontal_count, 1))
        if not next_parents:
            raise RuntimeError(
                f"vendor scan {step} with {value} points cannot be interleaved"
            )
        parents.append(next_parents)
        states = dict.fromkeys(next_parents)

    target = len(sensor_rows[0])
    valid = [
        horizontal_count
        for horizontal_count in states
        if horizontal_count == target
        and len(vendor_rows) - horizontal_count == len(sensor_rows[1])
    ]
    if len(valid) != 1:
        raise RuntimeError(f"expected one complete interleave, found {len(valid)}")

    path = [0] * len(vendor_rows)
    horizontal_count = valid[0]
    for step in range(len(vendor_rows), 0, -1):
        previous, sensor = parents[step][horizontal_count]
        path[step - 1] = sensor
        horizontal_count = previous
    return path


def verify(
    clean_path: Path, vendor_fringe_path: Path, vendor_foot_path: Path
) -> dict[str, Any]:
    clean = read_csv(clean_path)
    vendor_fringe = read_csv(vendor_fringe_path)
    vendor_foot = read_csv(vendor_foot_path)

    accepted_by_sensor = tuple(
        [row for row in clean if row["accepted"] == "1" and row["sensor"] == str(sensor)]
        for sensor in (0, 1)
    )
    interleave = recover_sensor_interleave(accepted_by_sensor, vendor_fringe)

    positions = [0, 0]
    fringe_input_mismatches = 0
    fringe_reject_mismatches = 0
    active_scans = 0
    clean_fringe_rejected = 0
    vendor_fringe_rejected = 0
    for sensor, vendor in zip(interleave, vendor_fringe):
        row = accepted_by_sensor[sensor][positions[sensor]]
        positions[sensor] += 1
        fringe_input_mismatches += row["input_count"] != vendor["input_count"]
        if row["insertion_enabled"] != "1":
            continue
        active_scans += 1
        clean_rejected = int(row["fringe_rejected"])
        vendor_rejected = int(vendor["fringe_rejected"])
        clean_fringe_rejected += clean_rejected
        vendor_fringe_rejected += vendor_rejected
        fringe_reject_mismatches += clean_rejected != vendor_rejected

    vertical = accepted_by_sensor[1]
    if len(vertical) != len(vendor_foot):
        raise RuntimeError(
            f"vertical/vendor Foot scan count differs: {len(vertical)}/{len(vendor_foot)}"
        )
    foot_input_mismatches = 0
    foot_reject_mismatches = 0
    active_vertical_scans = 0
    clean_foot_rejected = 0
    vendor_foot_rejected = 0
    for row, vendor in zip(vertical, vendor_foot):
        if row["insertion_enabled"] != "1":
            continue
        active_vertical_scans += 1
        expected_input = int(row["input_count"]) - int(row["fringe_rejected"])
        foot_input_mismatches += expected_input != int(vendor["input_count"])
        clean_rejected = int(row["foot_rejected"])
        vendor_rejected = int(vendor["foot_rejected"])
        clean_foot_rejected += clean_rejected
        vendor_foot_rejected += vendor_rejected
        foot_reject_mismatches += clean_rejected != vendor_rejected

    result = {
        "accepted_scans": len(vendor_fringe),
        "accepted_scans_by_sensor": positions,
        "active_scans": active_scans,
        "fringe_input_mismatches": fringe_input_mismatches,
        "fringe_reject_mismatches": fringe_reject_mismatches,
        "fringe_rejected_clean": clean_fringe_rejected,
        "fringe_rejected_vendor": vendor_fringe_rejected,
        "accepted_vertical_scans": len(vertical),
        "active_vertical_scans": active_vertical_scans,
        "foot_input_mismatches": foot_input_mismatches,
        "foot_reject_mismatches": foot_reject_mismatches,
        "foot_rejected_clean": clean_foot_rejected,
        "foot_rejected_vendor": vendor_foot_rejected,
    }
    result["exact"] = all(
        result[key] == 0
        for key in (
            "fringe_input_mismatches",
            "fringe_reject_mismatches",
            "foot_input_mismatches",
            "foot_reject_mismatches",
        )
    ) and (
        clean_fringe_rejected == vendor_fringe_rejected
        and clean_foot_rejected == vendor_foot_rejected
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--vendor-fringe", required=True, type=Path)
    parser.add_argument("--vendor-foot", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify(args.clean, args.vendor_fringe, args.vendor_foot)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
