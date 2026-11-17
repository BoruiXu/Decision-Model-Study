"""Skeleton for a new dataset adapter. Copy this file, rename it, and implement ``adapt``.

An adapter reads one source file and yields OneJev rows, each a dict with:
  id        unique string id
  state     {"messages": [{"role": ..., "content": ...}, ...]}  the model input
  question  {"type": str, "instructions": str, "criteria": {label: description}}
  target    {label: probability}; keys must be in ``criteria`` and sum to 1
  meta      free-form dict (fields referenced by the experiment config live here)
  split     split name; names starting with "test" are never read without --allow-test

Select it in the experiment config with ``data.adapter: <module name>``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


def adapt(path: Path) -> Iterator[dict]:
    raise NotImplementedError("copy adapters/template.py and implement adapt(path)")
