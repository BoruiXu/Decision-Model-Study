"""JSON Lines reading and writing shared by every harness module. Imports without torch.

A line ends at "\\n" only. ``str.splitlines`` also splits on U+2028/U+2029 (and other separators), which
``json.dumps(..., ensure_ascii=False)`` leaves raw inside strings, so it must never be used on JSONL.
"""
from __future__ import annotations

import json
from typing import Iterator


def iter_jsonl(path, *, tolerant: bool = False) -> Iterator[dict]:
    """Yield one object per non-blank line; ``tolerant`` skips lines that do not parse (a half-written tail)."""
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                if not tolerant:
                    raise


def read_jsonl(path, *, tolerant: bool = False) -> list[dict]:
    return list(iter_jsonl(path, tolerant=tolerant))


def dumps_line(obj) -> str:
    """One JSONL line, UTF-8 text kept readable; U+2028/U+2029 escaped so line-splitting readers elsewhere
    still see one object per line."""
    text = json.dumps(obj, ensure_ascii=False)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + "\n"
