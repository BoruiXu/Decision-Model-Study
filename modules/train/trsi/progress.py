"""progress.json: what this run's trials are doing, for the framework's pages and status lines (the format is the
framework's docs/interfaces/progress.md; the path is run.yaml's framework.progress_file). Written whole (a temporary
file, then os.replace) after `trsi launch`, at every poll of `trsi watch`, before and after `trsi collect` packs each
trial, and by the train command; write() returns its error instead of raising, so a progress file that cannot be
written never stops a watch, a collect or a training.

  summary   the trainings running and their GPUs, how many wait to start, the trial being packed, joined by "; "
  facts     configs (configurations proposed / search.max_configs), GPU hours (spent / the brief's budget), incumbent
            (the incumbent's candidate id and its changes from the start), in this order
  table     one row per trial of this run: trial, changes from the start, seed, state (waiting to start / training
            step k/n · about m min left from the ojh training log / packing (GPU i) / done, not packed / done /
            failed: <reason>), GPUs, time, the primary metric on the valset
  notes     the STOP line `trsi next` prints (cli.stop_reason) and the packing errors of unpacked trials
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from . import archive, harness
from .candidate import Candidate, _same, config_request
from .config import RunConfig

LOG_TAIL = 64 * 1024
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STEP_RE = re.compile(r"^\s*train \[[#.]*\] (\d+)/(\d+) ")
SPEED_RE = re.compile(r"(?<![\w/])s/step\s+([0-9.]+)")   # ojh writes the label first: `s/step 8.3`
ETA_RE = re.compile(r"\beta ([0-9.]+)h")
VALUE_CHARS, CELL_CHARS, NOTE_CHARS = 80, 200, 500


def _float(m) -> float | None:
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def train_step(log: Path) -> tuple[int, int, float | None] | None:
    """(step k, of n, hours left or None) from the last complete step line (`train [###...] 22/32 (68.8%) loss …
    eta 0.1h`) in the last 64 KiB of an ojh training log, without ANSI colour codes; the hours left are (n - k) x the
    line's s/step when it has one (minute resolution), else its eta. None when there is no such line or the log
    cannot be read."""
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            start = max(0, f.tell() - LOG_TAIL)
            f.seek(start)
            text = ANSI_RE.sub("", f.read().decode("utf-8", "replace"))
    except OSError:
        return None
    # progress bars rewrite their line with \r; the text after the last line end may be half written, and the first
    # piece of a tail that starts mid-file is cut
    pieces = re.split(r"[\r\n]", text)[1 if start else 0:-1]
    for line in reversed(pieces):
        m = STEP_RE.match(line)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            speed = _float(SPEED_RE.search(line))
            left = max(n - k, 0) * speed / 3600 if speed is not None else _float(ETA_RE.search(line))
            return k, n, left
    return None


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(s) -> float | None:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _hours(x: float) -> str:
    return "no limit" if math.isinf(x) else f"{x:.1f}".rstrip("0").rstrip(".")


def _duration(seconds: float) -> str:
    m = round(max(seconds, 0) / 60)
    return f"{m} min" if m < 120 else f"{m / 60:.1f} h"


def _count(n: int, word: str) -> str:
    """`n` and `word`, with an "s" added unless n is 1: "1 trial", "3 trials"."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def _gpus(cards: list[str]) -> str:
    """The GPUs of the trials in words, "GPU 2" or "GPUs 0, 4, 6" (one trial's entry may name several: "0, 1")."""
    joined = ", ".join(cards)
    return ("GPU " if len(joined.split(", ")) == 1 else "GPUs ") + joined


def _value(v) -> str:
    if v is None:
        return "default"
    if isinstance(v, float):
        return re.sub(r"e([+-])0+(\d)", r"e\1\2", f"{v:g}")   # 5e-06 -> 5e-6
    return str(v).lower() if isinstance(v, bool) else str(v)


def changes(run: RunConfig, c: Candidate) -> str:
    """A configuration's config_request against the start's, key by key ("base mammoth-1, lr 5e-6"; the seed is
    left out); "start" when they are the same."""
    a = config_request(run.start, 0, run.space.recipe_names)
    b = config_request(c, 0, run.space.recipe_names)
    out = [f"base {b['base']}"] if a["base"] != b["base"] else []
    if a["recipe"] != b["recipe"]:
        out.append("recipe " + (Path(b["recipe"]).stem if b["recipe"] else "default"))
    for k in sorted(set(a["overrides"]) | set(b["overrides"])):
        if not _same(a["overrides"].get(k), b["overrides"].get(k)):
            out.append(f"{k.split('.', 1)[-1]} {_value(b['overrides'].get(k))}")
    if not _same(a["max_steps"], b["max_steps"]):
        out.append(f"max_steps {_value(b['max_steps'])}")
    return ", ".join(out) or "start"


def _state(tid: str, rec: dict, packing: tuple[str, str] | None) -> tuple[str, str]:
    """(the state column, its kind: training, queued, packing, ended, done or failed)."""
    if packing and packing[0] == tid:
        return f"packing (GPU {packing[1]})", "packing"
    st = archive.state_of(rec)
    if st == "running":
        run_dir = Path(rec["run_dir"])
        ojh = harness.state(run_dir, rec.get("session"))
        if ojh == "running":
            step = train_step(run_dir / f"train-s{rec['seed']}.log")
            if step is None:
                return "waiting to start", "queued"
            k, n, eta = step
            left = "" if eta is None else f" · about {_duration(eta * 3600)} left"
            return f"training step {k}/{n}{left}", "training"
        if ojh == "done":
            return "done, not packed", "ended"
        st, rec = "failed", {"result": {"error": harness.read_result(run_dir, rec["seed"]).error}}
    if st == "unpacked":
        return "done, not packed", "ended"
    if st == "done":
        return "done", "done"
    return "failed: " + str((rec.get("result") or {}).get("error") or "reason unknown")[:80], "failed"


def _row(run: RunConfig, tid: str, rec: dict, packing, cards: str | None, start: float | None, end: float | None,
         now: float) -> tuple[list, str]:
    """A trial's table row and its kind; the time runs from its last launch to when ojh wrote its DONE (to now
    while it trains or waits; to its first result after that launch when there is no DONE)."""
    state, kind = _state(tid, rec, packing)
    done = Path(rec["run_dir"]) / "DONE" if rec.get("run_dir") else None
    if done is not None and done.is_file():   # training ended when ojh wrote DONE (collect packs it later)
        end = done.stat().st_mtime
    elif kind in ("training", "queued"):
        end = now
    val = ((rec.get("result") or {}).get("metrics") or {}).get(run.metric.primary)
    ok = isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)
    return [tid, _clip(changes(run, archive._cand_from(rec["candidate"])), CELL_CHARS), rec["seed"], state,
            cards or None, _duration(end - start) if start is not None and end is not None else None,
            float(f"{val:.4g}") if ok else None], kind


def build(run: RunConfig, packing: tuple[str, str] | None = None, now: float | None = None) -> dict:
    """The progress.json document of this run now; `packing` is (the trial being packed, its scoring card)."""
    from .cli import spent, stop_reason   # cli imports this module
    now = time.time() if now is None else now
    m = run.metric.primary
    launch, ended_at = {}, {}
    for e in archive.events(run):
        if e.get("event") == "launch":
            launch[e["trial"]] = e
            ended_at.pop(e["trial"], None)
        elif e.get("event") == "result":
            ended_at.setdefault(e["trial"], _epoch(e.get("t")))
    rows, kinds, notes = [], {}, []
    for tid, rec in archive.trials(run).items():
        le = launch.get(tid)
        cards = ", ".join(str(g) for g in le.get("gpus") or ()) if le else None
        try:
            row, kind = _row(run, tid, rec, packing, cards, _epoch(le.get("t")) if le else None, ended_at.get(tid),
                             now)
        except Exception as e:  # one unreadable trial does not hide the others
            row, kind = [tid, None, rec.get("seed"),
                         _clip(f"cannot read the state: {type(e).__name__}: {e}", CELL_CHARS), cards or None, None,
                         None], "failed"
        rows.append(row)
        kinds.setdefault(kind, []).append(cards)
        err = (rec.get("result") or {}).get("pack_error")
        if err and rec.get("status") == "done" and not (rec.get("result") or {}).get("candidate_dir"):  # unpacked
            notes.append(_clip(f"{tid}: {err}", NOTE_CHARS))
    parts = []
    if kinds.get("training"):
        cs = [c for c in kinds["training"] if c]
        parts.append(f"training {_count(len(kinds['training']), 'trial')}" + (f" ({_gpus(cs)})" if cs else ""))
    if kinds.get("queued"):
        parts.append(f"{_count(len(kinds['queued']), 'trial')} waiting to start")
    if packing:
        parts.append(f"packing 1 trial (GPU {packing[1]})")
    if not parts:
        done = [f"{len(kinds[k])} {w}" for k, w in (("done", "done"), ("ended", "done but not packed"),
                                                    ("failed", "failed")) if kinds.get(k)]
        parts.append("nothing training" + (": " + ", ".join(done) if done else ", no trials yet"))
    proposed = {e["candidate"] for e in archive.events(run, "propose")}
    mc = run.rules.max_configs
    inc = archive.incumbent(run)
    scheme = "not initialized" if inc is None else f"{inc[1]} ({changes(run, inc[0])})"
    try:
        stop = stop_reason(run)
    except Exception as e:  # the notes are not worth losing the rest
        stop = f"STOP check failed: {type(e).__name__}: {e}"
    return {
        "version": 1, "updated_at": _iso(now), "summary": "; ".join(parts),
        "facts": [{"label": "configs", "value": f"{len(proposed)}/{'no limit' if mc is None else mc}"},
                  {"label": "GPU hours", "value": f"{_hours(spent(run))}/{_hours(run.budget_gpu_h)}"},
                  {"label": "incumbent", "value": _clip(scheme, VALUE_CHARS)}],
        "table": {"title": f"Trials of trsi run {run.run_name}",
                  "columns": ["trial", "changes from the start", "seed", "state", "GPU", "time", f"valset {m}"],
                  "rows": rows},
        "notes": ([stop] if stop else []) + notes,
    }


def write(run: RunConfig, packing: tuple[str, str] | None = None) -> str | None:
    """Rewrite run.yaml's framework.progress_file atomically; returns the error in words instead of raising (a stop
    signal, a BaseException in trsi, still goes through)."""
    path = tmp = None
    try:
        path = Path(run.framework.progress_file)
        doc = build(run, packing)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except BaseException:  # a stop between writing and replacing too: leave no temporary file
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        return None
    except Exception as e:
        return f"{path or 'progress.json'}: {type(e).__name__}: {e}"
