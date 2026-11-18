"""``ojh make-split``: deterministic, group-aware splits. Imports without torch.

``select-report``: for each eval file, whole groups go to ``select`` or ``report`` so the select share is
about ``frac``; the output is the ``data.split_file`` format ``{set: {"assignment": {id: side}, ...}}``
(the extra keys record seed, frac, group field, source sha256 and counts).

``holdout``: whole groups of one train file go to a validation file (about ``frac`` of the rows); writes
``<out>/train.jsonl``, ``<out>/val.jsonl`` and ``<out>/manifest.json``.

Groups are sorted, shuffled with ``random.Random("<seed>:<set>")`` and assigned greedily: a group joins the
first part when that brings its row count closer to the target. Rows in a ``test*`` split, and files whose
name looks like a test split, are refused unless ``allow_test``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from harness.catalog import sha256_file
from harness.config import ConfigError, get_field
from harness.inspect_data import EXTS, load_rows, name_hint
from harness.jsonl import dumps_line
from harness.prepare import TestSplitRefused

SEED = 20261115


def _rows(path, allow_test: bool) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() not in EXTS:
        raise ConfigError(f"{path}: unsupported data file type (use jsonl, json, csv, tsv or parquet)")
    hint = name_hint(path)
    if hint and hint.startswith("test") and not allow_test:
        raise TestSplitRefused(f"{path}: the file name looks like a test split; use --allow-test (needs approval)")
    rows, _ = load_rows(path)
    if rows is None:
        raise ConfigError(f"{path}: holds no rows")
    if not allow_test:
        tests = [str(r.get("id")) for r in rows if isinstance(r, dict) and str(r.get("split", "")).startswith("test")]
        if tests:
            raise TestSplitRefused(f"{path}: {len(tests)} row(s) in a test split ({', '.join(tests[:3])}); "
                                   "use --allow-test (needs approval)")
    return rows


def _group_key(row, group: str):
    v = get_field(row, group)
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _groups(rows, group: str, path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    missing = 0
    for i, r in enumerate(rows):
        try:
            out.setdefault(_group_key(r, group), []).append(i)
        except (KeyError, TypeError):
            missing += 1
    if missing:
        raise ConfigError(f"{path}: {missing} row(s) have no group field {group!r}")
    return out


def _assign(groups: dict[str, list[int]], frac: float, salt: str) -> tuple[list[str], list[str]]:
    """(groups in the first part, groups in the second part)."""
    keys = sorted(groups)
    random.Random(salt).shuffle(keys)
    total = sum(len(v) for v in groups.values())
    target, first_n = frac * total, 0
    first, second = [], []
    for k in keys:
        size = len(groups[k])
        if abs(first_n + size - target) < abs(first_n - target):
            first.append(k)
            first_n += size
        else:
            second.append(k)
    return first, second


def _check_frac(frac: float) -> None:
    if not 0.0 < frac < 1.0:
        raise ConfigError(f"--frac must be between 0 and 1 (got {frac})")


def _values(keys):
    return [json.loads(k) for k in sorted(keys)]


def select_report(files: list[tuple[str, str]], *, group: str, seed: int = SEED, frac: float = 0.5,
                  allow_test: bool = False) -> dict:
    """``files``: (set name, path) pairs. Returns the split-file object."""
    _check_frac(frac)
    out, owner = {}, {}
    for name, path in files:
        if name in out:
            raise ConfigError(f"eval set {name!r} given twice")
        rows = _rows(path, allow_test)
        for r in rows:
            rid = r.get("id") if isinstance(r, dict) else None
            if rid is None:
                raise ConfigError(f"{path}: every row needs an id field (the ids the adapter yields)")
            if rid in owner:
                raise ConfigError(f"id {rid} appears twice ({owner[rid]} and {name})")
            owner[rid] = name
        groups = _groups(rows, group, path)
        sel, rep = _assign(groups, frac, f"{seed}:{name}")
        side = {k: "select" for k in sel} | {k: "report" for k in rep}
        assignment = {r["id"]: side[_group_key(r, group)] for r in rows}
        n_sel = sum(v == "select" for v in assignment.values())
        out[name] = {"assignment": assignment, "seed": seed, "frac": frac, "group": group,
                     "source": str(Path(path).resolve()), "source_sha256": sha256_file(path),
                     "rows": {"select": n_sel, "report": len(assignment) - n_sel},
                     "groups": {"select": _values(sel), "report": _values(rep)}}
    return out


def write_split_file(obj: dict, out) -> Path:
    """Write ``obj``; an existing file is kept when identical and never overwritten when different."""
    out = Path(out)
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    if out.exists() and out.read_text() != text:
        raise ConfigError(f"{out} exists with different content; remove it or choose another --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def holdout(path, out_dir, *, group: str, seed: int = SEED, frac: float = 0.5, allow_test: bool = False) -> dict:
    """Carve ``val.jsonl`` (whole groups, about ``frac`` of the rows) out of one train file."""
    _check_frac(frac)
    out_dir = Path(out_dir)
    if out_dir.suffix.lower() in (".json", ".jsonl") or out_dir.is_file():
        raise ConfigError(f"--mode holdout writes a directory (train.jsonl, val.jsonl, manifest.json); "
                          f"--out {out_dir} looks like a file: pass a directory such as <workspace>/holdout")
    rows = _rows(path, allow_test)
    groups = _groups(rows, group, path)
    val, train = _assign(groups, frac, f"{seed}:holdout")
    val_idx = {i for k in val for i in groups[k]}
    for name in ("train.jsonl", "val.jsonl", "manifest.json"):
        if (out_dir / name).exists():
            raise ConfigError(f"{out_dir / name} exists; choose an empty --out directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = {"train.jsonl": [r for i, r in enumerate(rows) if i not in val_idx],
             "val.jsonl": [r for i, r in enumerate(rows) if i in val_idx]}
    for name, rs in parts.items():
        with open(out_dir / name, "w", encoding="utf-8") as f:
            f.writelines(dumps_line(r) for r in rs)
    manifest = {"mode": "holdout", "source": str(Path(path).resolve()), "source_sha256": sha256_file(path),
                "group": group, "seed": seed, "frac": frac,
                "rows": {"train": len(parts["train.jsonl"]), "val": len(parts["val.jsonl"])},
                "groups": {"train": _values(train), "val": _values(val)},
                "sha256": {name: sha256_file(out_dir / name) for name in parts}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
