"""Adapter for files that already hold OneJev rows (one JSON object per line)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def adapt(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
