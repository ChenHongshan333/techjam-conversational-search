"""Prepare generated read-only assets during a Vercel build."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "catalog.jsonl.gz"
TARGET = ROOT / "data" / "catalog.jsonl"


def main() -> None:
    if TARGET.is_file() and TARGET.stat().st_size > 0:
        print(f"Catalog already prepared: {TARGET}")
        return
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing catalog archive: {SOURCE}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(SOURCE, "rb") as source, TARGET.open("wb") as target:
        shutil.copyfileobj(source, target)
    print(f"Prepared {TARGET} ({TARGET.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
