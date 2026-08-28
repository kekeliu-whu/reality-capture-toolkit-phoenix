#!/usr/bin/env python3
"""Fail-closed same-input acceptance for the clean-room Floor estimator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECON_ROOT / "src"))

from navvis_recon.floor_estimator import (  # noqa: E402
    read_trace_csv,
    refined_floor_estimator,
    to_official_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare every artifacts/trace.csv + floors.json pair below a corpus root"
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        raise NotADirectoryError(corpus)
    pairs = sorted(
        (floors_path.with_name("trace.csv"), floors_path)
        for floors_path in corpus.rglob("artifacts/floors.json")
        if floors_path.with_name("trace.csv").is_file()
    )
    if not pairs:
        raise FileNotFoundError(f"no Floor reference pairs below {corpus}")

    started = time.perf_counter()
    cases = []
    exact_count = 0
    trace_sample_count = 0
    reference_floor_count = 0
    reconstructed_floor_count = 0
    for trace_path, floors_path in pairs:
        reference = json.loads(floors_path.read_text())
        samples = read_trace_csv(trace_path)
        reconstructed = to_official_json(refined_floor_estimator(samples))
        exact = reconstructed == reference
        exact_count += int(exact)
        trace_sample_count += len(samples)
        reference_floor_count += len(reference)
        reconstructed_floor_count += len(reconstructed)
        if not exact:
            cases.append(
                {
                    "artifacts": str(floors_path.parent),
                    "exact": False,
                    "trace_samples": len(samples),
                    "reference": reference,
                    "reconstructed": reconstructed,
                }
            )

    report = {
        "schema_version": 1,
        "module": "Floor",
        "classification": "EXACT" if exact_count == len(pairs) else "MISMATCH",
        "comparison": "same serialized trace input; full floors.json object equality",
        "corpus": str(corpus),
        "case_count": len(pairs),
        "exact_case_count": exact_count,
        "mismatch_case_count": len(pairs) - exact_count,
        "trace_sample_count": trace_sample_count,
        "reference_floor_count": reference_floor_count,
        "reconstructed_floor_count": reconstructed_floor_count,
        "elapsed_seconds": time.perf_counter() - started,
        "mismatches": cases,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0 if report["classification"] == "EXACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
