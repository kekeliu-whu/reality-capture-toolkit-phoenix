#!/usr/bin/env python3
"""Locate selected Ceres 2.1 functions in stripped nv_colorcloud.

Only relocation-free entry prefixes are compared.  The selected prefixes end
before the first PC-relative direct call or RIP-relative data access, so an
exact byte match is meaningful across the two PIE link layouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FUNCTIONS = {
    "ResidualBlock::Evaluate": (0xCE570, 24),
    "TrustRegionMinimizer::EvaluateGradientAndJacobian": (0xD2C20, 96),
    "TrustRegionMinimizer::HandleSuccessfulStep": (0xD3DE0, 32),
    "TrustRegionMinimizer::ComputeTrustRegionStep": (0xD6370, 32),
    "LevenbergMarquardtStrategy::ComputeStep": (0x11C550, 32),
    "CgnrSolver::SolveImpl": (0x2C7060, 32),
    "ConjugateGradientsSolver::Solve": (0x2C7EC0, 32),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_offsets(haystack: bytes, needle: bytes) -> list[int]:
    result = []
    start = 0
    while True:
        position = haystack.find(needle, start)
        if position < 0:
            return result
        result.append(position)
        start = position + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean = args.clean.read_bytes()
    vendor = args.vendor.read_bytes()
    matches = {}
    for name, (clean_address, prefix_size) in FUNCTIONS.items():
        prefix = clean[clean_address : clean_address + prefix_size]
        offsets = all_offsets(vendor, prefix)
        matches[name] = {
            "clean_address": clean_address,
            "prefix_size": prefix_size,
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "vendor_file_offsets": offsets,
            # nv_colorcloud's first executable PT_LOAD has file offset and
            # virtual address zero, so the file offset is also the PIE vaddr.
            "vendor_virtual_addresses": offsets,
            "unique": len(offsets) == 1,
        }

    result = {
        "clean": {"path": str(args.clean), "sha256": sha256(args.clean)},
        "vendor": {"path": str(args.vendor), "sha256": sha256(args.vendor)},
        "method": "exact relocation-free entry-prefix match",
        "matches": matches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
