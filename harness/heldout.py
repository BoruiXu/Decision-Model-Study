"""Recognize held-out test data without reading it further: shared by ``ojh inspect`` and ``ojh make-split``.

A row is a test row when any split-like column (a dict key, any case, at any dict depth: split, set,
subset, partition, fold, ...) holds a string starting with "test" (any case). A path is test data when a
directory or file name starts with "test" (any case): every component below ``root`` and the root's own
name, or without a root the file name and its directory (components further up, such as a home directory,
are not judged). Imports without torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

SPLIT_KEYS = {"split", "splits", "set", "subset", "fold", "partition", "part", "dataset_split", "data_split",
              "split_name"}
MAX_DEPTH = 4


def is_test_name(name: str) -> bool:
    return name.strip().lower().startswith("test")


def is_test_value(v) -> bool:
    return isinstance(v, str) and is_test_name(v)


def split_values(row, prefix: str = "", depth: int = 0) -> Iterator[tuple[str, object]]:
    """(dotted path, value) of every split-like key in ``row`` (dicts only, not list items)."""
    if not isinstance(row, dict) or depth > MAX_DEPTH:
        return
    for k, v in row.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from split_values(v, path, depth + 1)
        elif str(k).lower() in SPLIT_KEYS:
            yield path, v


def _get(row, dotted: str):
    cur = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def row_is_test(row, extra_paths=()) -> bool:
    """True when a split-like column, or one of ``extra_paths`` (dotted), holds a test value."""
    if any(is_test_value(v) for _, v in split_values(row)):
        return True
    return any(is_test_value(_get(row, p)) for p in extra_paths)


def is_test_path(path, root=None) -> bool:
    path = Path(path)
    if root is not None:
        root = Path(root)
        try:
            parts = [root.name, *path.relative_to(root).parts]
        except ValueError:
            parts = None
        if parts is not None and path != root:
            return any(is_test_name(p) for p in parts if p)
    return any(is_test_name(p) for p in (path.parent.name, path.name) if p)
