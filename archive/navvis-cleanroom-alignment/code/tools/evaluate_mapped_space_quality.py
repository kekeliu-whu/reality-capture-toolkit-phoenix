#!/usr/bin/env python3
"""Compare mapped-space quality version-2 outputs by observable semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

import brotli


HEADER = b"#NavVis GmbH binary file format for compressed voxel quality data. version: 2\n"
RECORD = struct.Struct("<QHHB")


def load_output(directory: Path) -> dict:
    binary_path = directory / "quality_voxels.bin"
    sidecar_path = directory / "quality_voxels_sidecar.json"
    binary = binary_path.read_bytes()
    if not binary.startswith(HEADER):
        raise ValueError(f"invalid mapped-space quality header: {binary_path}")
    payload = brotli.decompress(binary[len(HEADER) :])
    if len(payload) % RECORD.size:
        raise ValueError(f"partial 13-byte quality record in {binary_path}")
    records = [record for record in struct.iter_unpack(RECORD.format, payload)]
    by_key = {record[0]: record[1:] for record in records}
    if len(by_key) != len(records):
        raise ValueError(f"duplicate Morton key in {binary_path}")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("quality_grid_format_version") != 2:
        raise ValueError(f"sidecar is not format version 2: {sidecar_path}")
    if int(sidecar.get("num_voxels", -1)) != len(records):
        raise ValueError(f"sidecar voxel count does not match payload: {sidecar_path}")
    if int(sidecar.get("bytes_uncompressed", -1)) != len(payload):
        raise ValueError(f"sidecar uncompressed byte count does not match payload: {sidecar_path}")
    if int(sidecar.get("bytes_compressed", -1)) != len(binary):
        raise ValueError(f"sidecar compressed byte count does not match file: {sidecar_path}")
    return {
        "format": "v2_directory",
        "binary": binary,
        "payload": payload,
        "records": records,
        "by_key": by_key,
        "sidecar": sidecar,
        "sha256": hashlib.sha256(binary).hexdigest(),
    }


def load_raw_records(path: Path) -> dict:
    payload = path.read_bytes()
    if len(payload) % RECORD.size:
        raise ValueError(f"partial 13-byte quality record in {path}")
    records = [record for record in struct.iter_unpack(RECORD.format, payload)]
    by_key = {record[0]: record[1:] for record in records}
    if len(by_key) != len(records):
        raise ValueError(f"duplicate Morton key in {path}")
    return {
        "format": "raw_records",
        "binary": None,
        "payload": payload,
        "records": records,
        "by_key": by_key,
        "sidecar": None,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def compare(reference: dict, candidate: dict) -> dict:
    reference_keys = set(reference["by_key"])
    candidate_keys = set(candidate["by_key"])
    common_keys = reference_keys & candidate_keys
    field_names = ("directional_diversity", "ray_count", "minimum_range")
    field_mismatches = {
        name: sum(
            reference["by_key"][key][index] != candidate["by_key"][key][index]
            for key in common_keys
        )
        for index, name in enumerate(field_names)
    }
    sidecar_applicable = reference["sidecar"] is not None and candidate["sidecar"] is not None
    reference_sidecar = dict(reference["sidecar"] or {})
    candidate_sidecar = dict(candidate["sidecar"] or {})
    if sidecar_applicable:
        # Compressed bytes are order-dependent; compare it separately from the
        # geometry/transform contract represented by the rest of the sidecar.
        reference_sidecar.pop("bytes_compressed", None)
        candidate_sidecar.pop("bytes_compressed", None)
    sidecar_semantics_exact = (
        reference_sidecar == candidate_sidecar if sidecar_applicable else None
    )
    semantic_exact = (
        reference_keys == candidate_keys
        and not any(field_mismatches.values())
        and (sidecar_semantics_exact is not False)
    )
    return {
        "status": "EXACT" if semantic_exact else "MISMATCH",
        "semantic_exact": semantic_exact,
        "reference_format": reference["format"],
        "candidate_format": candidate["format"],
        "reference_voxels": len(reference["records"]),
        "candidate_voxels": len(candidate["records"]),
        "missing_keys": len(reference_keys - candidate_keys),
        "extra_keys": len(candidate_keys - reference_keys),
        "common_keys": len(common_keys),
        "field_mismatches": field_mismatches,
        "record_multiset_exact": Counter(reference["records"]) == Counter(candidate["records"]),
        "record_order_exact": reference["records"] == candidate["records"],
        "uncompressed_payload_byte_exact": reference["payload"] == candidate["payload"],
        "compressed_file_byte_exact": (
            reference["binary"] == candidate["binary"]
            if reference["binary"] is not None and candidate["binary"] is not None
            else None
        ),
        "sidecar_semantics_exact": sidecar_semantics_exact,
        "reference_sha256": reference["sha256"],
        "candidate_sha256": candidate["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--reference", type=Path)
    reference_group.add_argument(
        "--reference-raw",
        type=Path,
        help="Uncompressed packed <QHHB> records from a direct core aggregate",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-byte-exact",
        action="store_true",
        help="Also fail when the partition/container-dependent record order differs",
    )
    args = parser.parse_args()
    try:
        reference = (
            load_raw_records(args.reference_raw)
            if args.reference_raw is not None
            else load_output(args.reference)
        )
        result = compare(reference, load_output(args.candidate))
    except Exception as error:
        print(json.dumps({"status": "ERROR", "error": str(error)}, indent=2))
        return 2
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not result["semantic_exact"]:
        return 1
    if args.require_byte_exact and result["compressed_file_byte_exact"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
