#!/usr/bin/env python3
"""Read-only protobuf evidence extractor for NavVis HybridGrid loop constraints.

This intentionally uses a tiny protobuf wire reader instead of generated NavVis
classes.  It never writes to the dataset.
"""

from __future__ import annotations

import argparse
import math
import struct
import zipfile
from pathlib import Path
from typing import Iterator, Optional


DEFAULT_ROOT = Path(
    "/media/cybergeo/12T/CSSJ/datasets_proc_reference_g11_0109/"
    "2026-02-08_07.33.20"
)


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def fields(data: bytes) -> Iterator[tuple[int, int, object]]:
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = read_varint(data, offset)
        elif wire == 1:
            value = struct.unpack_from("<d", data, offset)[0]
            offset += 8
        elif wire == 2:
            size, offset = read_varint(data, offset)
            value = data[offset : offset + size]
            offset += size
        elif wire == 5:
            value = struct.unpack_from("<f", data, offset)[0]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def values(data: bytes, number: int) -> list[object]:
    return [value for field, _, value in fields(data) if field == number]


def first(data: bytes, number: int, default: object = None) -> object:
    found = values(data, number)
    return found[0] if found else default


def packed_count(data: bytes) -> int:
    count = 0
    offset = 0
    while offset < len(data):
        _, offset = read_varint(data, offset)
        count += 1
    return count


def zip_member(path: Path, member: Optional[str] = None) -> bytes:
    with zipfile.ZipFile(path) as archive:
        name = member or archive.namelist()[0]
        return archive.read(name)


def id_index(message: bytes) -> int:
    return int(first(message, 2))


def describe_grid(message: bytes) -> tuple[float, int]:
    resolution = float(first(message, 1, math.nan))
    packed_values = first(message, 6, b"")
    assert isinstance(packed_values, bytes)
    return resolution, packed_count(packed_values)


def describe_result(result: bytes) -> dict[str, object]:
    constraint = first(result, 1)
    assert isinstance(constraint, bytes)
    submap_id = first(constraint, 1)
    node_id = first(constraint, 2)
    assert isinstance(submap_id, bytes) and isinstance(node_id, bytes)
    output: dict[str, object] = {
        "pair": (id_index(submap_id), id_index(node_id)),
        "valid": bool(first(result, 10, 0)),
        "failure": int(first(result, 11, -1)),
        "ray_ratio": first(result, 13),
    }
    fcs = first(result, 8)
    if isinstance(fcs, bytes):
        output["fcs"] = {
            "score": first(fcs, 1),
            "rotational_score": first(fcs, 3),
            "low_resolution_score": first(fcs, 4),
            "score_ok": bool(first(fcs, 8, 0)),
            "low_resolution_score_ok": bool(first(fcs, 9, 0)),
            "rotational_score_ok": bool(first(fcs, 10, 0)),
            "matched": bool(first(fcs, 11, 0)),
        }
    return output


def print_result(label: str, result: dict[str, object]) -> None:
    print(
        f"{label}: pair={result['pair']} valid={result['valid']} "
        f"failure={result['failure']} ray_ratio={result['ray_ratio']}"
    )
    if "fcs" in result:
        fcs = result["fcs"]
        assert isinstance(fcs, dict)
        print(
            "  FCS: "
            + " ".join(f"{key}={value}" for key, value in fcs.items())
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--accepted-pair",
        default="0,620",
        help="submap,node pair selected from the frozen accepted archive",
    )
    parser.add_argument(
        "--diagnostic-zip",
        type=Path,
        help="optional compute_constraints diagnostic constraint_data zip",
    )
    args = parser.parse_args()
    wanted = tuple(int(part) for part in args.accepted_pair.split(","))

    print("HybridGrid constants: min=0.1 max=0.9 value_count=32768")
    print("P(0)=0.1; P(v)=0.1+(v-1)*0.8/32766 for 1<=v<=32767")

    submap_zip = (
        args.root
        / "internal/submaps/submap_clouds/submap_clouds_00000000.zip"
    )
    submap = zip_member(submap_zip, f"submap_{wanted[0]:08d}.pb")
    high = first(submap, 1)
    low = first(submap, 2)
    assert isinstance(high, bytes) and isinstance(low, bytes)
    print("stored high grid: resolution=%g cells=%d" % describe_grid(high))
    print("stored low grid:  resolution=%g cells=%d" % describe_grid(low))

    node_zip = (
        args.root
        / "internal/nodes/trajectory_node_clouds/trajectory_node_clouds_00000000.zip"
    )
    node = zip_member(node_zip, f"trajectory_node_{wanted[1]:08d}.pb")
    high_cloud = first(node, 1)
    low_cloud = first(node, 2)
    assert isinstance(high_cloud, bytes) and isinstance(low_cloud, bytes)
    print(
        "stored node clouds: high_points=%d low_points=%d"
        % (len(values(high_cloud, 1)), len(values(low_cloud, 1)))
    )

    accepted_zip = (
        args.root
        / "internal/constraints_inter_dataset/2026-02-08_07.33.20/"
        "constraints/constraint_data/constraint_data_00000000.zip"
    )
    with zipfile.ZipFile(accepted_zip) as archive:
        accepted = None
        ratios: list[float] = []
        for member in archive.namelist():
            top = archive.read(member)
            result = first(top, 1)
            assert isinstance(result, bytes)
            decoded = describe_result(result)
            if decoded["ray_ratio"] is not None:
                ratios.append(float(decoded["ray_ratio"]))
            if decoded["pair"] == wanted:
                accepted = decoded
        if accepted is None:
            raise SystemExit(f"accepted pair {wanted} not found")
        print_result("frozen accepted", accepted)
        print(
            "accepted archive: count=%d ray_min=%.9f ray_max=%.9f"
            % (len(archive.namelist()), min(ratios), max(ratios))
        )

    if args.diagnostic_zip:
        top = zip_member(args.diagnostic_zip)
        result = first(top, 1)
        assert isinstance(result, bytes)
        print_result("diagnostic", describe_result(result))


if __name__ == "__main__":
    main()
