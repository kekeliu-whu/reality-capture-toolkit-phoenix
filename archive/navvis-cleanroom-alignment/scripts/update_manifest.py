#!/usr/bin/env python3
"""Regenerate the portable bundle's complete SHA-256 manifest."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"


def is_packaged_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not path.is_file() or path == OUTPUT:
        return False
    if relative.parts[0] == "work":
        return False
    if "__pycache__" in relative.parts or ".pytest_cache" in relative.parts:
        return False
    if len(relative.parts) >= 2 and relative.parts[0] == "code":
        if relative.parts[1].startswith("build") or relative.parts[1] in {"bin", "lib"}:
            return False
    return path.suffix not in {".pyc", ".pyo", ".tmp", ".temp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = sorted(path for path in ROOT.rglob("*") if is_packaged_file(path))
    temporary = OUTPUT.with_suffix(".sha256.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for path in files:
            relative = path.relative_to(ROOT)
            stream.write(f"{sha256(path)}  ./{relative}\n")
    os.replace(temporary, OUTPUT)
    print(f"Wrote {len(files)} checksums to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
