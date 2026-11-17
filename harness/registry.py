"""Run registry: one JSON line per finalized seed, plus a summary line for multi-seed runs.

Records are keyed by (run_id, seed): ``append`` is an upsert, so re-running finalize replaces a run's
records instead of duplicating them. Writers hold an exclusive ``fcntl`` lock on ``<path>.lock``; readers
hold a shared one. A record is ``complete`` only when every REQUIRED field is present and non-empty;
``conclusion`` starts empty, so a new record stays incomplete (and hidden from ``table``) until ``note``
fills it. Imports without torch.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import stat
import tempfile
from pathlib import Path

REQUIRED = ("exp", "run_id", "seed", "train_label_version", "reference_label_version", "data_sha256", "base",
            "base_revision", "recipe", "mode", "onejev", "gpu_name", "n_gpus", "gpu_hours", "selected_step",
            "checkpoint_sha256", "metrics", "baselines", "gate", "overrides", "conclusion")


def is_complete(record: dict) -> bool:
    for key in REQUIRED:
        value = record.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
    return True


@contextlib.contextmanager
def _locked(path, exclusive: bool = True):
    lock = Path(f"{path}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(fd)  # releases the lock


def _read(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read(path) -> list[dict]:
    with _locked(path, exclusive=False):
        return _read(path)


def _key(record: dict) -> tuple:
    return record.get("run_id"), record.get("seed")


def _rewrite(path: Path, records: list[dict]) -> None:
    """Replace the registry atomically (temp file in the same dir + os.replace), keeping its file mode."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def append(path, record: dict) -> dict:
    """Upsert ``record`` keyed by (run_id, seed) under the registry lock and set ``record["complete"]``.

    A new key is appended as one line. An existing key (a re-run of finalize, the ``seed: "summary"``
    record included) is replaced in place by an atomic rewrite; an empty new ``conclusion`` keeps the
    existing one, so re-finalizing never erases a note.
    """
    path = Path(path)
    with _locked(path):
        records = _read(path)
        hits = [i for i, r in enumerate(records) if _key(r) == _key(record)]
        if hits:
            old = str(records[hits[0]].get("conclusion") or "")
            if not str(record.get("conclusion") or "").strip() and old.strip():
                record["conclusion"] = old
        record["complete"] = is_complete(record)
        if hits:
            records[hits[0]] = record
            _rewrite(path, [r for i, r in enumerate(records) if i not in hits[1:]])
            return record
        data = (json.dumps(record, ensure_ascii=False) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
    return record


def note(path, run_id: str, text: str) -> int:
    """Rewrite the conclusion of every record of ``run_id``; returns how many records changed."""
    path = Path(path)
    with _locked(path):
        records = _read(path)
        n = 0
        for rec in records:
            if rec.get("run_id") == run_id:
                rec["conclusion"] = text
                rec["complete"] = is_complete(rec)
                n += 1
        if n:
            _rewrite(path, records)
    return n


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict) and "mean" in value:
        return f"{_fmt(value['mean'])}±{_fmt(value.get('std'))}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ",".join(_fmt(v) for v in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _get(rec: dict, *keys):
    cur = rec
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


COLUMNS = (("run", ("run_id",)), ("exp", ("exp",)), ("seed", ("seed",)), ("base", ("base",)), ("mode", ("mode",)),
           ("step", ("selected_step",)), ("AUC", ("metrics", "auc")),
           ("组内 AUC", ("metrics", "within_group_auc")), ("TF-IDF AUC", ("baselines", "tfidf", "auc")),
           ("门", ("gate", "passed")), ("GPU h", ("gpu_hours",)), ("状态", ("status",)),
           ("结论", ("conclusion",)))


def table(path, include_incomplete: bool = False) -> str:
    """Markdown table of the registry; incomplete records are hidden unless asked for."""
    rows = [r for r in read(path) if include_incomplete or r.get("complete")]
    lines = ["| " + " | ".join(name for name, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for rec in rows:
        lines.append("| " + " | ".join(_fmt(_get(rec, *keys)) for _, keys in COLUMNS) + " |")
    return "\n".join(lines) + "\n"
