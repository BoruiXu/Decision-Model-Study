"""Adapter for files that already hold OneJev rows (one JSON object per line)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from harness.jsonl import iter_jsonl


def adapt(path: Path) -> Iterator[dict]:
    yield from iter_jsonl(path)
