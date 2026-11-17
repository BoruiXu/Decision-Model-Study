"""Cheap progress report for a run, read from log.jsonl, DONE and the lock."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from harness.jsonl import read_jsonl


def _jsonl(path: Path) -> list[dict]:
    return read_jsonl(path, tolerant=True) if path.is_file() else []  # tolerant: a half-written last line


def status(run_dir) -> dict:
    run_dir = Path(run_dir)
    lock = yaml.safe_load((run_dir / "run.lock.yaml").read_text())
    seeds = lock["seeds"]
    total = lock["plan"]["steps"] * lock["flags"].get("epochs", 1)
    cap = lock["flags"].get("max_steps") or 0
    if cap > 0:
        total = min(total, cap)
    recs, idx = [], 1
    # "latest seed": the last seed whose log exists
    for i in range(len(seeds), 0, -1):
        p = run_dir / "runs" / f"{lock['experiment']}-s{seeds[i - 1]}" / "log.jsonl"
        if p.is_file():
            recs, idx = _jsonl(p), i
            break
    train = [r for r in recs if "step" in r and "sec_per_step" in r and "event" not in r]
    evals = [r for r in recs if r.get("event") == "harness_eval"]
    step = train[-1]["step"] if train else None
    eta = round((total - step) * train[-1]["sec_per_step"]) if train else None
    state = "running"
    done_p = run_dir / "DONE"
    if done_p.is_file():
        if json.loads(done_p.read_text()).get("exit_code"):
            state = "failed"
        else:
            sf = run_dir / "status"
            state = sf.read_text().strip() if sf.is_file() and sf.read_text().strip() else "done"
    gp = run_dir / "gpus.json"
    gpus = json.loads(gp.read_text()) if gp.is_file() else []
    return {"state": state, "seed": f"{idx}/{len(seeds)}", "step": step, "total_steps": total,
            "last_eval": evals[-1] if evals else None, "eta_s": eta, "gpus": gpus}


def render(s: dict) -> str:
    lines = [f"state: {s['state']}  seed: {s['seed']}",
             f"step: {s['step'] if s['step'] is not None else '-'}/{s['total_steps']}"
             + (f"  eta: {s['eta_s']}s" if s["eta_s"] is not None else "")]
    e = s["last_eval"]
    if e:
        lines.append(f"last eval: {e['set']} step {e['step']} n={e['n']} nll={e['nll']}")
    if s["gpus"]:
        lines.append("gpus: " + ", ".join(g["name"] for g in s["gpus"]))
    return "\n".join(lines[:6])
