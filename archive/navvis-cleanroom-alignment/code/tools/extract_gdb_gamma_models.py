#!/usr/bin/env python3
"""Convert fixed-build GDB GammaModel lines to clean worker view order."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MODEL = re.compile(
    r"^MODEL\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-views", type=int)
    args = parser.parse_args()

    models: dict[int, tuple[float, float]] = {}
    for line in args.input.read_text(encoding="utf-8").splitlines():
        match = MODEL.match(line)
        if not match:
            continue
        encoded = int(match.group(1))
        capture = encoded >> 16
        camera = (encoded >> 8) & 0xFF
        view = capture * 4 + camera
        models[view] = (float(match.group(2)), float(match.group(3)))
    if not models:
        raise ValueError(f"no MODEL lines found in {args.input}")
    expected = args.expected_views if args.expected_views is not None else max(models) + 1
    missing = sorted(set(range(expected)) - set(models))
    extra = sorted(set(models) - set(range(expected)))
    if missing or extra:
        raise ValueError(f"model coverage mismatch: missing={missing}, extra={extra}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write("# view gain exponent\n")
        for view in range(expected):
            gain, exponent = models[view]
            output.write(f"{view} {gain:.17g} {exponent:.17g}\n")


if __name__ == "__main__":
    main()
