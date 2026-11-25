"""report.md: the incumbent and its candidates, every proposal's prediction against its result, the budget, the best
metric against compute hours. write_search: the framework's search artifact of this run (one search = one dispatch)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, archive
from .candidate import candidate_id, config_request
from .config import RunConfig
from .harness import TrialResult, exp_sha256


def _f(v, nd=4) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render(run: RunConfig, cost_of, spent: float) -> str:
    m = run.metric.primary
    inc = archive.incumbent(run)
    T = archive.trials(run)
    by_cid: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in T.values():
        by_cid[rec["candidate_id"]][rec["seed"]] = rec
    proposals = {e["candidate"]: e for e in archive.events(run, "propose")}
    verdicts = {e["candidate"]: e for e in archive.events(run, "verdict")}
    probes = defaultdict(list)
    for e in archive.events(run, "probe"):
        probes[e["candidate"]].append(e)

    L = [f"# Training RSI report: {run.run_name}", "",
         f"- environment: `{run.exp}` (sha256 {exp_sha256(run)[:12]}); trsi {__version__}",
         f"- rules: probe seed {run.rules.seeds[0]}, probe margin {run.rules.probe_margin}, seeds "
         f"{list(run.rules.seeds)}, accept margin {run.rules.accept_margin}, guards {run.metric.guards}",
         f"- budget: {spent:.2f} / {run.budget_gpu_h:.1f} compute hours spent (a candidate adopted from an earlier "
         "round is not charged again)", ""]

    L += ["## Incumbent", ""]
    if inc:
        c, cid = inc
        L.append(f"`{cid}` {c.label()}" + (f": {c.note}" if c.note else ""))
        L += ["", "| seed | " + m + " | ece | brier | selected step | candidate | run dir |",
              "|---|---|---|---|---|---|---|"]
        for s, rec in sorted(by_cid.get(cid, {}).items()):
            r = TrialResult.from_dict(rec["result"]) if rec.get("result") else None
            L.append(f"| {s} | {_f(r and r.metrics.get(m))} | {_f(r and r.metrics.get('ece'))} | "
                     f"{_f(r and r.metrics.get('brier'))} | {_f(r and r.selected_step)} | "
                     f"`{(r and r.candidate_dir) or '-'}` | `{rec['run_dir']}` |")
        L.append("")

    L += ["## Candidates", "",
          f"| candidate | scheme | note | predicted {m} / p_accept | probe gain | seeds ({m}) | verdict | hours |",
          "|---|---|---|---|---|---|---|---|"]
    for cid, e in proposals.items():
        pr = e.get("prediction") or {}
        seeds = by_cid.get(cid, {})
        res = ", ".join(f"s{s} {_f((rec.get('result') or {}).get('metrics', {}).get(m))}"
                        if archive.state_of(rec) == "done" else f"s{s} {archive.state_of(rec)}"
                        for s, rec in sorted(seeds.items()))
        pg = probes.get(cid)
        v = verdicts.get(cid)
        cost = sum(cost_of(rec) for rec in seeds.values())
        label = archive._cand_from(e["candidate_dict"]).label()
        L.append(f"| `{cid}` | {label} | {e['candidate_dict'].get('note', '')} | {_f(pr.get('value'))} / "
                 f"{_f(pr.get('p_accept'), 2)} | {_f(pg[-1]['gain']) if pg else '-'} | {res or '-'} | "
                 f"{(v['verdict'] + ': ' + v['detail']) if v else 'open'} | {cost:.2f} |")
    L.append("")

    L += ["## Best scheme against compute hours", "", f"| hours spent | incumbent mean {m} | event |", "|---|---|---|"]
    for e in archive.events(run, "curve"):
        L.append(f"| {e['spent']:.2f} | {_f(e.get('incumbent_mean'))} | {e['why']} |")
    L.append("")

    return "\n".join(L) + "\n"


def write(run: RunConfig, cost_of, spent: float) -> Path:
    p = run.home / "report.md"
    archive._write(p, render(run, cost_of, spent))
    return p


class SearchError(RuntimeError):
    """The search cannot be written (e.g. the start's candidate is not registered)."""


def write_search(run: RunConfig, cost_of, ids: dict, dest: Path) -> Path:
    """<dest>/manifest.json, the framework's search of this run:
    trials        every trial in first-use order: its config_request (with the seed), the registered candidate id
                  (from `ids`, keyed by candidate directory name; an adopted candidate's own id), metric_val (the
                  primary metric of its metrics_val), compute_hours (0 for an adopted candidate: paid in its own
                  round) and status (ok; failed; running; unpacked when its candidate could not be packed;
                  unregistered when its candidate is not in `ids`)
    default       the start: the adopted candidate, else the start's probe-seed trial, else its first ok trial
    best          the incumbent's ok trial with the highest metric_val when the incumbent is not the start and that
                  is not below the default's; otherwise the default
    gain          best - default metric_val"""
    fw = run.framework
    task = fw.task_json()
    m = run.metric.primary
    inc = archive.incumbent(run)
    start_cid = candidate_id(run.start, exp_sha256(run))
    rows = []
    for tid, rec in archive.trials(run).items():
        c = archive._cand_from(rec["candidate"])
        r = rec.get("result") or {}
        reg = rec.get("registered_id") if rec.get("adopted") else ids.get(Path(r.get("candidate_dir") or "").name)
        state = archive.state_of(rec)
        status = ("ok" if reg else "unregistered") if state == "done" else state
        rows.append({"rec": rec, "trial": {
            "candidate_id": reg if status == "ok" else None,
            "config": config_request(c, rec["seed"], run.space.recipe_names),
            "metric_val": (r.get("metrics") or {}).get(m) if status == "ok" else None,
            "compute_hours": 0.0 if rec.get("adopted") else cost_of(rec), "status": status}})
    ok = [x for x in rows if x["trial"]["status"] == "ok"]
    starts = [x for x in ok if x["rec"]["candidate_id"] == start_cid]
    default = (next((x for x in starts if x["rec"].get("adopted")), None)
               or next((x for x in starts if x["rec"]["seed"] == run.rules.seeds[0]), None)
               or (starts[0] if starts else None))
    if default is None:
        raise SearchError("no registered ok trial of the start: the search needs the default's candidate (is it in --ids?)")
    best = default
    if inc and inc[1] != start_cid:
        mine = [x for x in ok if x["rec"]["candidate_id"] == inc[1]]
        top = max(mine, key=lambda x: x["trial"]["metric_val"], default=None)
        if top is not None and top["trial"]["metric_val"] >= default["trial"]["metric_val"]:
            best = top
    trials = [x["trial"] for x in rows]
    used = sum(t["compute_hours"] for t in trials)
    budget = run.budget_gpu_h if math.isfinite(run.budget_gpu_h) and run.budget_gpu_h >= used else used
    unreg = [x["rec"]["trial"] for x in rows if x["trial"]["status"] == "unregistered"]
    manifest = {
        "kind": "search", "id": None, "task_id": task["task_id"], "task_version": task["version"], "round": fw.round,
        "producer": fw.producer, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parents": list(dict.fromkeys([fw.trainset_id, fw.valset_id] + [t["candidate_id"] for t in trials
                                                                         if t["candidate_id"]])),
        "files": {},
        "notes": f"trsi {__version__}, run {run.run_name} ({run.home / 'report.md'})"
                 + (f"; budget {run.budget_gpu_h:.2f} h, used {used:.2f} h" if budget != run.budget_gpu_h else "")
                 + (f"; trials whose candidate is not registered: {', '.join(unreg)}" if unreg else ""),
        "trainset_id": fw.trainset_id, "valset_id": fw.valset_id,
        "strategy": f"trsi rules: probe seed {run.rules.seeds[0]} must beat the incumbent's by more than "
                    f"{run.rules.probe_margin} {m}; then seeds {list(run.rules.seeds)}, accepted when the mean gain "
                    f"is above {run.rules.accept_margin} and the guards {run.metric.guards} hold",
        "budget": {"compute_hours": budget}, "trials": trials,
        "default_candidate": default["trial"]["candidate_id"], "best_candidate": best["trial"]["candidate_id"],
        "gain_vs_default": {"metric": m, "delta": best["trial"]["metric_val"] - default["trial"]["metric_val"]},
    }
    dest.mkdir(parents=True, exist_ok=True)
    archive._write(dest / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return dest
