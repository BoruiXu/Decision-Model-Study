"""Data preparation: validate, subset and merge an experiment's data. Imports without torch."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import ConfigError, DataSpec, ExperimentConfig, SubsetSpec, get_field, is_adapter_path
from harness.jsonl import dumps_line


class ShaMismatch(RuntimeError):
    """The source train file does not match data.train_sha256."""


class IdOverlap(RuntimeError):
    """An id appears in train and an eval set, or in two eval sets."""


class TestSplitRefused(RuntimeError):
    """A row belongs to a test split and --allow-test was not given."""

    __test__ = False  # not a pytest class


def adapter_file(name: str) -> Path:
    """The source file of an adapter: the path itself, or ``adapters/<name>.py`` for a module name."""
    if is_adapter_path(name):
        p = Path(name)
        if not p.is_file():
            raise ConfigError(f"adapter file not found: {name}")
        return p
    spec = importlib.util.find_spec(f"adapters.{name}")
    if spec is None or not spec.origin:
        raise ConfigError(f"unknown adapter {name!r}: no adapters/{name}.py (or give a path to a .py file)")
    return Path(spec.origin)


def _load_adapter(name: str):
    if is_adapter_path(name):
        from harness.plugins import load_plugin

        return load_plugin(str(adapter_file(name)))
    adapter_file(name)
    return importlib.import_module(f"adapters.{name}")


NOUL_CRITERIA = ("true", "false")  # OneJev qev/schema.py NoulCriteria: descriptions of the yes / the no answer


def _labels(qtype, criteria) -> list[str] | None:
    """Labels OneJev renders for a question type (qev/prompt.py render_question). A noul question always
    has the labels yes/no; its optional ``criteria`` is ``{"true": <yes description>, "false": <no
    description>}``."""
    if qtype == "noul":
        return ["yes", "no"]
    if qtype == "choice":
        return list((criteria or {}).keys())
    if qtype == "score":
        return [str(i) for i in range(len(criteria or []))]
    return None


def validate_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set = set()
    for i, r in enumerate(rows):
        rid = r.get("id", f"<row {i}>")
        if rid in seen:
            errors.append(f"duplicate id: {rid}")
        seen.add(rid)
        if not r.get("state"):
            errors.append(f"{rid}: missing state")
        target = r.get("target")
        if not isinstance(target, dict) or not target:
            errors.append(f"{rid}: missing target")
            continue
        q = r.get("question") or {}
        qtype, crit = q.get("type"), q.get("criteria")
        labels = _labels(qtype, crit)
        if labels is None:
            errors.append(f"{rid}: unknown question type {qtype!r}")
            continue
        if qtype == "noul" and crit is not None and (not isinstance(crit, dict) or not set(crit) <= set(NOUL_CRITERIA)):
            got = sorted(crit) if isinstance(crit, dict) else type(crit).__name__
            errors.append(f"{rid}: noul question.criteria keys are true/false (the descriptions of the yes and the "
                          f"no answer; OneJev ignores other keys); got criteria keys {got}. Target keys stay yes/no")
        if not set(target) <= set(labels):
            if qtype == "noul":
                why = "noul target keys are yes/no (question.criteria true/false only describe them)"
            elif qtype == "choice":
                why = f"choice target keys must be the criteria keys {labels}"
            else:
                why = f"score target keys are the level indices {labels}"
            errors.append(f"{rid}: {why}; got target keys {sorted(target)} (labels {labels})")
        if any(not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0 for v in target.values()):
            errors.append(f"{rid}: target values must be in [0, 1]")
        total = sum(v for v in target.values() if isinstance(v, (int, float)))
        if abs(total - 1.0) > 1e-6:
            errors.append(f"{rid}: target does not sum to 1 (sum={total})")
    return errors


def apply_subset(rows: list[dict], spec: SubsetSpec | None) -> list[dict]:
    if spec is None or (not spec.where and spec.n is None):
        return rows
    out = rows
    for dotted, want in spec.where.items():
        def keep(r, dotted=dotted, want=want):
            try:
                return get_field(r, dotted) == want
            except KeyError:
                return False
        out = [r for r in out if keep(r)]
    out = sorted(out, key=lambda r: r["id"])
    if spec.n is not None and spec.n < len(out):
        out = random.Random(spec.seed).sample(out, spec.n)
    return out


@dataclass
class EvalParts:
    select_rows: list[dict] = field(default_factory=list)
    select_index: dict[str, str] = field(default_factory=dict)
    reference: dict[str, dict] = field(default_factory=dict)
    report_ids: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)


def _read(adapter: str, path, args: dict | None = None) -> list[dict]:
    return list(_load_adapter(adapter).adapt(Path(path), **(args or {})))


def train_args(data: DataSpec) -> dict:
    return {**data.adapter_args, **data.train_adapter_args}


def eval_args(data: DataSpec, name: str) -> dict:
    return {**data.adapter_args, **data.eval[name].adapter_args}


def read_train(data: DataSpec) -> list[dict]:
    """Rows of ``data.train`` through the adapter, with data.adapter_args + data.train_adapter_args."""
    return _read(data.adapter, data.train, train_args(data))


def read_eval(data: DataSpec, name: str, path) -> list[dict]:
    """Rows of one of eval set ``name``'s files, with data.adapter_args + that set's adapter_args."""
    return _read(data.adapter, path, eval_args(data, name))


def build_eval_parts(exp: ExperimentConfig, *, allow_test: bool = False) -> EvalParts:
    data = exp.data
    assignments: dict = {}
    if data.split_file:
        assignments = json.loads(Path(data.split_file).read_text())
    parts = EvalParts()
    for name, spec in data.eval.items():
        labels = read_eval(data, name, spec.train_labels)
        if not allow_test:
            for r in labels:
                if str(r.get("split", "")).startswith("test"):
                    raise TestSplitRefused(f"{name}: row {r.get('id')} is in split {r['split']!r}; use --allow-test")
        ref = None
        if spec.reference is not None:
            ref = {r["id"]: r for r in read_eval(data, name, spec.reference)}
        if data.split_file:
            if name not in assignments:
                raise ConfigError(f"data.split_file has no entry for eval set {name!r}")
            assign = assignments[name].get("assignment") or {}
        else:  # no split file: the whole eval set is select
            assign = {r["id"]: "select" for r in labels}
        dropped = parts.dropped.setdefault(name, [])
        unassigned = parts.dropped.setdefault("unassigned", [])
        label_ids = {r["id"] for r in labels}
        if ref is not None:
            dropped += sorted(set(ref) - label_ids)
            dropped += sorted(label_ids - set(ref))
        sel: list[dict] = []
        errs = validate_rows([r for r in labels if assign.get(r["id"]) == "select"
                              and (ref is None or r["id"] in ref)])
        if errs:
            raise ValueError(f"invalid select rows in eval set {name!r}:\n" + "\n".join(errs[:20]))
        reports: list[str] = []
        for r in labels:
            rid = r["id"]
            if ref is not None and rid not in ref:
                continue
            side = assign.get(rid)
            if side is None:
                unassigned.append(rid)
            elif side == "select":
                sel.append(r)
                parts.select_index[rid] = name
            elif side == "report":
                reports.append(rid)
            else:
                raise ValueError(f"{name}: bad assignment {side!r} for {rid}")
            if side in ("select", "report") and ref is not None:
                parts.reference[rid] = ref[rid]
        parts.select_rows += sel
        parts.report_ids[name] = reports
    return parts


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps_line(r))


def prepare(exp: ExperimentConfig, out_dir: Path, *, allow_test: bool = False) -> dict:
    out_dir = Path(out_dir)
    data = exp.data
    src_sha = _sha256(data.train)
    if data.train_sha256 is None:
        warnings.warn(f"data.train_sha256 not set; recording computed sha256 {src_sha}", UserWarning, stacklevel=2)
    elif data.train_sha256 != src_sha:
        raise ShaMismatch(f"train sha256 {src_sha} != expected {data.train_sha256}")

    train_all = read_train(data)
    if not allow_test:
        tests = [str(r.get("id")) for r in train_all if str(r.get("split", "")).startswith("test")]
        if tests:
            raise TestSplitRefused(f"train: {len(tests)} row(s) in a test split ({', '.join(tests[:3])}); "
                                   "use --allow-test")
    errors = validate_rows(train_all)
    if errors:
        raise ValueError("invalid train rows:\n" + "\n".join(errors[:20]))
    train = apply_subset(train_all, exp.subset)

    parts = build_eval_parts(exp, allow_test=allow_test)
    if not parts.select_rows:
        raise ConfigError(f"no select rows in eval sets {sorted(data.eval)}: checkpoint selection needs at least "
                          "one (check data.split_file)" if data.eval else
                          "data.eval lists no eval sets: checkpoint selection needs select rows")
    train_ids = {r["id"] for r in train_all}
    owner: dict[str, str] = {}
    for name, spec in data.eval.items():
        ids = {r["id"] for r in read_eval(data, name, spec.train_labels)}
        both = ids & train_ids
        if both:
            raise IdOverlap(f"{len(both)} id(s) in train and eval set {name}: {sorted(both)[:3]}")
        for i in ids:
            if i in owner and owner[i] != name:
                raise IdOverlap(f"id {i} is in eval sets {owner[i]} and {name}")
            owner[i] = name

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "select.jsonl", parts.select_rows)
    (out_dir / "select_index.json").write_text(json.dumps(parts.select_index, indent=2))
    afile = adapter_file(data.adapter)
    info = {
        "allow_test": bool(allow_test),
        "adapter": {"name": data.adapter, "file": str(afile), "sha256": _sha256(afile)},
        "adapter_args": {"global": data.adapter_args, "train": data.train_adapter_args,
                         "eval": {n: s.adapter_args for n, s in data.eval.items()}},
        "train_sha256_source": src_sha,
        "train_sha256_prepared": _sha256(out_dir / "train.jsonl"),
        "subset": ({"where": exp.subset.where, "n": exp.subset.n, "seed": exp.subset.seed}
                   if exp.subset is not None else None),
        "counts": {"train_source": len(train_all), "train": len(train),
                   "select": len(parts.select_rows),
                   "report": {k: len(v) for k, v in parts.report_ids.items()}},
        "dropped": parts.dropped,
    }
    (out_dir / "prepared.json").write_text(json.dumps(info, indent=2))
    return info
