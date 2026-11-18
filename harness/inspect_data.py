"""``ojh inspect``: read-only facts about data files, for the agent that decides how to adapt them.

Prints facts only: it makes no decisions and writes nothing. For a file or every data file below a
directory (jsonl, json, csv, tsv; parquet when pyarrow is installed): type, size and row count; per key
path (dotted, ``[]`` for list items) the value types and fill rate; distinct values of low-cardinality
fields; whether numeric fields lie in [0, 1]; average length of strings; a few truncated sample rows;
split-like columns and file names; split-file-shaped JSON; whether the rows already pass the OneJev row
schema (``prepare.validate_rows``).

Test data (``harness.heldout``) is counted only: files or directories named like a test split are not
read beyond a row count, and rows in a test split (any split-like column, any case) are dropped before
fields, values, samples and the schema check; only their split-column counts are shown. Imports without torch.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from harness.heldout import is_test_path, is_test_value, row_is_test, split_values
from harness.prepare import validate_rows

EXTS = {".jsonl": "jsonl", ".json": "json", ".csv": "csv", ".tsv": "tsv", ".parquet": "parquet"}
MAX_DISTINCT = 20  # a field with at most this many distinct scalar values lists them with counts
TEXT_LEN = 20  # strings at least this long on average are reported as text-like
TRUNC = 200  # sample string truncation
MAX_DEPTH = 4
SPLIT_VALUE = re.compile(r"^(train|training|val|valid|validation|dev|test|eval|holdout)([_\-.].*)?$", re.I)
NAME_HINTS = ("train", "training", "val", "valid", "validation", "dev", "test", "eval")


def _type(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def name_hint(path) -> str | None:
    """The split word in a file name (``train``, ``validation``, ``test``, ...), or None."""
    tokens = re.split(r"[^a-z0-9]+", Path(path).stem.lower())
    for t in tokens:
        if t in NAME_HINTS:
            return t
    return None


def _csv_cell(v: str):
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _read_csv(path, delimiter) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _csv_cell(v) if isinstance(v, str) else v for k, v in r.items()}
                for r in csv.DictReader(f, delimiter=delimiter)]


def split_file_shape(obj) -> dict | None:
    """``{set: {"assignment": {id: "select"|"report"}}}`` -> ``{set: {"select": n, "report": n}}``."""
    if not isinstance(obj, dict) or not obj:
        return None
    out = {}
    for name, spec in obj.items():
        a = spec.get("assignment") if isinstance(spec, dict) else None
        if not isinstance(a, dict) or any(v not in ("select", "report") for v in a.values()):
            return None
        vals = list(a.values())
        out[name] = {"select": vals.count("select"), "report": vals.count("report")}
    return out


def load_rows(path) -> tuple[list[dict] | None, dict]:
    """Rows of a data file (None when it holds no rows) and extra facts about its shape."""
    kind = EXTS[Path(path).suffix.lower()]
    extra: dict = {}
    if kind == "jsonl":
        rows, bad = [], 0
        with open(path, encoding="utf-8", newline="\n") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    bad += 1
        if bad:
            extra["unparsable_lines"] = bad
        return rows, extra
    if kind == "json":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        extra["split_file"] = split_file_shape(obj)
        if isinstance(obj, list):
            return obj, extra
        if isinstance(obj, dict):
            lists = [k for k, v in obj.items() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
            if extra["split_file"] is None and len(lists) == 1:
                extra["rows_key"] = lists[0]
                return obj[lists[0]], extra
            extra["top_keys"] = list(obj)[:50]
        return None, extra
    if kind in ("csv", "tsv"):
        return _read_csv(path, "," if kind == "csv" else "\t"), extra
    import pyarrow.parquet as pq  # only reached when pyarrow is importable (see data_files)

    return pq.read_table(path).to_pylist(), extra


def _count_rows(path) -> int | None:
    kind = EXTS[Path(path).suffix.lower()]
    if kind == "jsonl":
        with open(path, encoding="utf-8", newline="\n") as f:
            return sum(1 for line in f if line.strip())
    if kind in ("csv", "tsv"):
        with open(path, newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.reader(f, delimiter="," if kind == "csv" else "\t")) - 1)
    return None


def _walk(value, prefix: str, seen: dict) -> None:
    """Collect every (path, value) of one row; a path seen several times in a row keeps all its values."""
    seen.setdefault(prefix, []).append(value)
    if isinstance(value, dict):
        for k, v in value.items():
            _walk(v, f"{prefix}.{k}" if prefix else str(k), seen)
    elif isinstance(value, list):
        for item in value:
            _walk(item, f"{prefix}[]", seen)


class _Field:
    def __init__(self):
        self.rows = 0  # rows where the path holds at least one non-null value
        self.types: dict[str, int] = {}
        self.distinct: dict = {}
        self.overflow = False
        self.nums: list[float] = []
        self.str_len = self.str_n = 0
        self.list_len = self.list_n = 0

    def add_row(self, values: list) -> None:
        if any(v is not None for v in values):
            self.rows += 1
        for v in values:
            t = _type(v)
            self.types[t] = self.types.get(t, 0) + 1
            if t in ("int", "float") and v == v:  # not NaN
                self.nums.append(float(v))
            if t == "str":
                self.str_len += len(v)
                self.str_n += 1
            if t == "list":
                self.list_len += len(v)
                self.list_n += 1
            if t in ("str", "int", "float", "bool") and not self.overflow:
                key = (t, v)
                self.distinct[key] = self.distinct.get(key, 0) + 1
                if len(self.distinct) > MAX_DISTINCT:
                    self.overflow, self.distinct = True, {}

    def facts(self, n_rows: int) -> dict:
        out: dict = {"types": dict(sorted(self.types.items())), "fill": self.rows / n_rows if n_rows else 0.0}
        n_scalar = sum(self.distinct.values())
        if self.distinct and not self.overflow and len(self.distinct) < n_scalar:
            items = sorted(self.distinct.items(), key=lambda kv: (-kv[1], str(kv[0][1])))
            out["values"] = [{"value": v, "count": c} for (_, v), c in items]
        if self.nums:
            lo, hi = min(self.nums), max(self.nums)
            out.update(min=lo, max=hi, mean=sum(self.nums) / len(self.nums), unit_interval=0.0 <= lo and hi <= 1.0)
        if self.str_n:
            out["avg_len"] = round(self.str_len / self.str_n, 1)
            out["text_like"] = out["avg_len"] >= TEXT_LEN
        if self.list_n:
            out["avg_items"] = round(self.list_len / self.list_n, 2)
        return out


def field_facts(rows: list[dict]) -> dict:
    fields: dict[str, _Field] = {}
    for r in rows:
        seen: dict = {}
        if isinstance(r, dict):
            for k, v in r.items():
                _walk(v, str(k), seen)
        for path, values in seen.items():
            fields.setdefault(path, _Field()).add_row(values)
    return {p: f.facts(len(rows)) for p, f in fields.items()}


def _dict_scalars(row, prefix: str = "", depth: int = 0):
    """(dotted path, value) for string values at dict paths (not inside lists)."""
    if not isinstance(row, dict) or depth > MAX_DEPTH:
        return
    for k, v in row.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _dict_scalars(v, path, depth + 1)
        elif isinstance(v, str):
            yield path, v


def split_columns(rows) -> dict:
    """Split-like columns with their value counts over all rows (counts are the only facts shown for test
    rows): keys named like a split (``harness.heldout.SPLIT_KEYS``, any case), and low-cardinality string
    columns whose values all look like split names."""
    named: dict[str, dict] = {}
    other: dict[str, dict] = {}
    for r in rows:
        for path, v in split_values(r):
            if isinstance(v, (str, int, float, bool)):
                named.setdefault(path, {})
                named[path][v] = named[path].get(v, 0) + 1
        for path, v in _dict_scalars(r):
            if path in named:
                continue
            c = other.setdefault(path, {})
            if c is not None and len(c) <= MAX_DISTINCT:
                c[v] = c.get(v, 0) + 1
    out = dict(named)
    for path, c in other.items():
        if path not in named and c and len(c) <= MAX_DISTINCT and all(SPLIT_VALUE.match(v) for v in c):
            out[path] = c
    return {p: dict(sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))) for p, c in out.items()}


def _truncate(v):
    if isinstance(v, str):
        return v if len(v) <= TRUNC else v[:TRUNC] + "…"
    if isinstance(v, list):
        out = [_truncate(x) for x in v[:5]]
        return out + [f"… ({len(v) - 5} more)"] if len(v) > 5 else out
    if isinstance(v, dict):
        return {k: _truncate(x) for k, x in v.items()}
    return v


def inspect_file(path, rows: int = 3, root=None) -> dict:
    path = Path(path)
    hint = name_hint(path)
    out = {"path": str(path.resolve()), "type": EXTS[path.suffix.lower()], "size_bytes": path.stat().st_size,
           "name_hint": hint, "rows": None, "fields": None, "split_columns": {}, "samples": [],
           "onejev": None, "split_file": None}
    if is_test_path(path, root):
        out["rows"] = _count_rows(path)
        out["note"] = "test-like file or directory name: contents not inspected (rows counted only)"
        return out
    try:
        data, extra = load_rows(path)
    except (ValueError, UnicodeDecodeError, csv.Error) as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out.update(extra)
    if data is None:
        return out
    out["rows"] = len(data)
    dict_rows = [r for r in data if isinstance(r, dict)]
    out["split_columns"] = split_columns(dict_rows)
    test_cols = [p for p, c in out["split_columns"].items() if any(is_test_value(v) for v in c)]
    kept = [r for r in data if not (isinstance(r, dict) and row_is_test(r, test_cols))]
    if len(kept) < len(data):
        out["test_rows"] = len(data) - len(kept)  # counted only: no fields, values or samples from them
    out["fields"] = field_facts(kept)
    kept_dicts = [r for r in kept if isinstance(r, dict)]
    out["samples"] = [_truncate(r) for r in kept_dicts[:max(0, rows)]]
    errors = validate_rows(kept_dicts) if len(kept_dicts) == len(kept) else ["not every row is a JSON object"]
    out["onejev"] = {"passes": bool(kept) and not errors, "errors": len(errors), "first_errors": errors[:5]}
    return out


def _parquet_ok() -> bool:
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        return False
    return True


def data_files(path) -> tuple[list[Path], list[dict]]:
    """Data files at ``path`` (a file, or a directory searched recursively; hidden entries skipped) and the
    files skipped with the reason."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if path.is_file():
        cands = [path]
    else:
        cands = []
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            cands += [Path(root) / f for f in files if not f.startswith(".")]
        cands.sort()
    keep, skipped = [], []
    parquet = None
    for p in cands:
        ext = p.suffix.lower()
        if ext not in EXTS:
            if path.is_file():
                skipped.append({"path": str(p), "reason": f"unsupported extension {ext!r}"})
            continue
        if ext == ".parquet":
            parquet = _parquet_ok() if parquet is None else parquet
            if not parquet:
                skipped.append({"path": str(p), "reason": "parquet needs pyarrow (not installed)"})
                continue
        keep.append(p)
    return keep, skipped


def inspect_path(path, rows: int = 3) -> dict:
    files, skipped = data_files(path)
    root = Path(path) if Path(path).is_dir() else None
    return {"path": str(Path(path).resolve()), "files": [inspect_file(f, rows, root) for f in files],
            "skipped": skipped}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else repr(v)


def render(report: dict) -> str:
    """Human-readable facts (the --json output has everything)."""
    lines = [f"inspect {report['path']}: {len(report['files'])} data file(s)"]
    for s in report["skipped"]:
        lines.append(f"skipped {s['path']}: {s['reason']}")
    for f in report["files"]:
        lines += ["", f"== {f['path']}",
                  f"type {f['type']}, {f['size_bytes']} bytes, rows {f['rows'] if f['rows'] is not None else '-'}"
                  + (f", file name looks like: {f['name_hint']}" if f["name_hint"] else "")]
        for key in ("note", "error", "rows_key", "unparsable_lines", "test_rows"):
            if f.get(key) is not None:
                lines.append(f"{key}: {f[key]}")
        if f.get("top_keys"):
            lines.append("top-level keys: " + ", ".join(map(str, f["top_keys"])))
        if f.get("split_file"):
            lines.append("split file shape: " + "; ".join(f"{k}: select {v['select']}, report {v['report']}"
                                                         for k, v in f["split_file"].items()))
        if f["onejev"] is not None:
            o = f["onejev"]
            lines.append("OneJev rows: " + ("yes" if o["passes"] else f"no ({o['errors']} error(s); first: "
                                            + "; ".join(o["first_errors"][:2]) + ")"))
        for path, vals in f["split_columns"].items():
            lines.append(f"split-like column {path}: " + ", ".join(f"{k}={n}" for k, n in vals.items()))
        if f["fields"]:
            lines.append("fields (path: types, fill[, details]):")
            for path, x in f["fields"].items():
                bits = ["/".join(f"{t}×{n}" for t, n in x["types"].items()), f"fill {x['fill']:.2f}"]
                if "min" in x:
                    bits.append(f"range [{x['min']:.4g}, {x['max']:.4g}]" + (" in [0,1]" if x["unit_interval"] else ""))
                if "avg_len" in x:
                    bits.append(f"avg len {x['avg_len']}" + (" (text-like)" if x["text_like"] else ""))
                if "avg_items" in x:
                    bits.append(f"avg items {x['avg_items']}")
                if "values" in x:
                    bits.append("values " + ", ".join(f"{_fmt(v['value'])}={v['count']}" for v in x["values"]))
                lines.append(f"  {path}: " + "; ".join(bits))
        for i, s in enumerate(f["samples"]):
            lines.append(f"sample {i + 1}: " + json.dumps(s, ensure_ascii=False))
    return "\n".join(lines)
