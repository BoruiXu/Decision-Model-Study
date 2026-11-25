"""Records. Per run (<out_dir>/<run_name>/): archive.jsonl (append-only events) and incumbent.yaml. Shared by all runs
under the same out_dir: trials/<trial id>.json, one per trained (candidate, seed), so a trial is never trained twice."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .candidate import Candidate, Override
from .config import RunConfig


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def append(run: RunConfig, event: dict) -> None:
    run.home.mkdir(parents=True, exist_ok=True)
    with open(run.home / "archive.jsonl", "a") as f:
        f.write(json.dumps({"t": _now(), **event}, ensure_ascii=False, default=str) + "\n")


def events(run: RunConfig, kind: str | None = None) -> list[dict]:
    p = run.home / "archive.jsonl"
    if not p.is_file():
        return []
    out = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [e for e in out if kind is None or e.get("event") == kind]


# ---- shared trial records -----------------------------------------------------------------------------------

def _trial_path(run: RunConfig, tid: str) -> Path:
    return run.out_dir / "trials" / f"{tid}.json"


def trial_record(run: RunConfig, tid: str) -> dict | None:
    p = _trial_path(run, tid)
    return json.loads(p.read_text()) if p.is_file() else None


def write_trial(run: RunConfig, rec: dict) -> None:
    _write(_trial_path(run, rec["trial"]), json.dumps(rec, indent=2, ensure_ascii=False, default=str) + "\n")


def trials(run: RunConfig) -> dict[str, dict]:
    """Trial id -> shared record, for every trial this run has used (launched, reused or adopted), in first-use order."""
    out: dict[str, dict] = {}
    for e in events(run, "use"):
        rec = trial_record(run, e["trial"])
        if rec is not None and e["trial"] not in out:
            out[e["trial"]] = {**rec, "start": e.get("start", False), "charged": e.get("charged", True)}
    return out


def state_of(rec: dict) -> str:
    """running | failed | done | unpacked (the ojh run finished, its candidate is not packed yet: not comparable)."""
    if rec["status"] == "done" and not (rec.get("result") or {}).get("candidate_dir"):
        return "unpacked"
    return rec["status"]


# ---- incumbent ----------------------------------------------------------------------------------------------

def _cand_from(d: dict) -> Candidate:
    return Candidate(d["base"], Path(d["recipe"]),
                     {k: Override(o["value"], o["reason"]) for k, o in (d.get("overrides") or {}).items()},
                     d.get("note", ""), bool(d.get("default_recipe")))


def incumbent(run: RunConfig) -> tuple[Candidate, str] | None:
    """(candidate, candidate id) of the current scheme, or None before init."""
    p = run.home / "incumbent.yaml"
    if not p.is_file():
        return None
    d = yaml.safe_load(p.read_text())
    return _cand_from(d["candidate"]), d["candidate_id"]


def set_incumbent(run: RunConfig, c: Candidate, cid: str, why: str) -> None:
    _write(run.home / "incumbent.yaml",
           "# the current training scheme; changed only by `trsi next` when a candidate is accepted\n"
           + yaml.safe_dump({"candidate_id": cid, "candidate": c.to_dict(), "since": _now(), "why": why},
                            sort_keys=False, allow_unicode=True))
    append(run, {"event": "incumbent", "candidate": cid, "why": why})
