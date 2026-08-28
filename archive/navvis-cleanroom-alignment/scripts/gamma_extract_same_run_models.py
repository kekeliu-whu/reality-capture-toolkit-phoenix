#!/usr/bin/env python3
"""Extract one nv_colorcloud run's captured GammaModels into clean replay format."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MODEL = re.compile(
    r"^MODEL\s+(?P<key>\d+)\s+(?P<gain>[-+0-9.eE]+)\s+(?P<exponent>[-+0-9.eE]+)$"
)


def key_to_view(key: int) -> int:
    capture = key >> 16
    camera = (key >> 8) & 0xFF
    if key & 0xFF or camera >= 4:
        raise ValueError(f"unexpected nv_colorcloud GammaModel key: {key:#x}")
    return capture * 4 + camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-views", type=int, default=136)
    args = parser.parse_args()

    models: dict[int, tuple[str, str, int]] = {}
    for line in args.capture_log.read_text(encoding="utf-8").splitlines():
        match = MODEL.match(line)
        if not match:
            continue
        key = int(match.group("key"))
        view = key_to_view(key)
        if view in models:
            raise ValueError(f"duplicate GammaModel view {view}")
        models[view] = (match.group("gain"), match.group("exponent"), key)

    expected = set(range(args.expected_views))
    actual = set(models)
    if actual != expected:
        raise ValueError(
            f"GammaModel coverage mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# view gain exponent"]
    lines.extend(f"{view} {models[view][0]} {models[view][1]}" for view in sorted(models))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(models)} same-run GammaModels to {args.output}")


if __name__ == "__main__":
    main()
