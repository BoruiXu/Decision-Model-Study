"""``result.json`` and ``RESULT.md``: the one-shot answer, written into the run dir at the end of finalize.

``build_result`` gathers facts already computed by finalize (no new evaluation); ``render_result_md``
turns them into the page the agent reads back to the user: data, setup, training curve and selected
checkpoint, metrics on the select part with 95% bootstrap CIs, locked thresholds, gate verdict with
reasons, warnings, and the next commands (the ones that read report rows or delete files need the
user's approval). Imports without torch.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness import run_files
from harness.evaluator.binary_decision import POOLED
from harness.jsonl import read_jsonl
from harness.select import eval_events

CI_KEYS = ("auc", "within_group_auc")
SMALL_SELECT = 100


def _curve(run_dir: Path, lock: dict, seed) -> list[dict]:
    log = run_dir / "runs" / f"{lock['experiment']}-s{seed}" / "log.jsonl"
    events = read_jsonl(log, tolerant=True) if log.is_file() else []
    return [{"step": e["step"], "nll": e["nll"]} for e in eval_events(events)]


def _methods(entry: dict) -> dict:
    """Pooled-select entry -> {method: metrics}: router renamed model, CIs flattened to ``<key>_ci95``."""
    out = {}
    for name, m in entry.items():
        if not isinstance(m, dict):
            continue
        ci = m.get("ci95") or {}
        skipped = m.get("ci95_skipped") or {}
        row = {}
        for k, v in m.items():
            if k in ("ci95", "ci95_skipped") or isinstance(v, dict):
                continue
            if k == "jaccard_at_own_best_threshold":
                row["jaccard"] = v
                continue
            row[k] = v
            if k in ci:
                row[f"{k}_ci95"] = ci[k]
                if skipped.get(k):
                    row[f"{k}_ci95_skipped"] = skipped[k]
        out["model" if name == "router" else name] = row
    order = ["model"] + [k for k in out if k not in ("model", "all_positive", "oracle")] + \
        [k for k in ("all_positive", "oracle") if k in out]
    return {k: out[k] for k in order if k in out}


def _num(v) -> str:
    return "-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def _gate_reasons(gate: dict | None, sets: dict) -> list[str]:
    reasons = []
    for name, g in (gate or {}).items():
        if not isinstance(g, dict):
            continue
        if "reason" in g:
            reasons.append(f"{name}: {g['reason']}")
            continue
        best = g.get("best") or {}
        router = ((sets or {}).get(name) or {}).get("router") or {}
        keyed = [k for k, v in g.items() if isinstance(v, bool) and k != "passed"]
        for k in keyed:
            b = best.get(k)
            vs = f"best baseline {b[0]} {_num(b[1])}" if b else "no baseline value"
            reasons.append(f"{name}: model {k} {_num(router.get(k))} {'>' if g[k] else '<='} {vs}")
        if not keyed and "passed" in g:
            reasons.append(f"{name}: passed {g['passed']}")
    return reasons


def _dropped_warnings(dropped: dict) -> list[str]:
    out = []
    for key, ids in (dropped.get("eval_parts") or {}).items():
        if ids:
            out.append(f"{len(ids)} eval id(s) dropped ({key}: no split assignment or no reference match); "
                       "see dropped.json")
    for seed, per in (dropped.get("evaluate") or {}).items():
        for key, ids in (per or {}).items():
            if ids:
                out.append(f"seed {seed}: {len(ids)} row(s) dropped from evaluation ({key}: missing label, "
                           "consistency or group); see dropped.json")
    return out


def _next(lock: dict, run_dir: Path, status: str, candidates: dict, report_rows: int) -> list[dict]:
    ojh = f"{lock['repo_root']}/.venv/bin/ojh"
    out = [{"command": f'{ojh} note {run_dir} "<one-line conclusion and its evidence>"', "needs_approval": False,
            "what": "record the conclusion in the registry"}]
    if report_rows:
        out.append({"command": f"{ojh} eval-report {run_dir} --allow-report", "needs_approval": True,
                    "what": f"score the {report_rows} report row(s) with the locked checkpoint and thresholds"})
    out.append({"command": f"{ojh} cleanup {run_dir}", "needs_approval": False,
                "what": "list the deletion candidates (deletes nothing)"})
    cands = candidates.get("candidates") or []
    if cands and status != "failed":
        out.append({"command": f"{ojh} cleanup {run_dir} --confirm", "needs_approval": True,
                    "what": f"delete {len(cands)} unselected checkpoint dir(s), "
                            f"{candidates.get('total_bytes', 0) / 2**30:.2f} GiB"})
    return out


def build_result(run_dir, lock: dict, *, status: str, failures: list[str], comparison: dict, thresholds: dict,
                 timing: dict, versions: dict, manifests: dict, dropped: dict, candidates: dict,
                 bootstrap: dict | None) -> dict:
    run_dir = Path(run_dir)
    data = lock["data"]
    counts = data.get("counts") or {}
    gp = run_dir / "gpus.json"
    cards = json.loads(gp.read_text()) if gp.is_file() else []
    seeds_t = timing.get("seeds") or {}
    seconds = sum(float(s.get("seconds") or 0) for s in seeds_t.values())
    seeds = {}
    select_by_set: dict = {}
    warnings: list[str] = []
    for key, c in comparison.items():
        sets = c.get("sets") or {}
        pooled = sets.get(POOLED) or {}
        if not select_by_set:
            select_by_set = {n: e.get("questions") for n, e in sets.items() if n != POOLED}
        thr = {}
        for name, t in (thresholds.get(key) or {}).items():
            if isinstance(t, dict):
                pooled_t = t.get("pooled") or {k: t[k] for k in ("small_share", "jaccard", "inconsistency")
                                                if k in t}
                thr[name] = {"threshold": t.get("threshold"),
                             "pooled": {"threshold": t.get("threshold"), **pooled_t},
                             "per_set": t.get("per_set") or {}}
            else:
                thr[name] = {"threshold": t, "pooled": {}, "per_set": {}}
        gate = c.get("gate") or {}
        seeds[key] = {"curve": _curve(run_dir, lock, key), "selected_step": c.get("selected_step"),
                      "checkpoint": c.get("checkpoint"),
                      "checkpoint_sha256": (manifests.get(key) or {}).get("checkpoint_sha256"),
                      "rescore": c.get("rescore"),
                      "metrics": {"set": POOLED, "questions": pooled.get("questions"), "methods": _methods(pooled)},
                      "metrics_by_set": {n: {"questions": e.get("questions"), "methods": _methods(e)}
                                         for n, e in sets.items() if n != POOLED},
                      "thresholds": thr,
                      "gate": {"enabled": bool(gate.get("enabled")), "passed": gate.get("passed"),
                               "reasons": _gate_reasons(gate, sets)}}
        unseen = pooled.get("unseen_groups")
        if unseen:
            warnings.append(f"seed {key}: {unseen} select group(s) never appear in train (group prior falls back "
                            "to the overall mean)")
    if counts.get("select") is not None and counts["select"] < SMALL_SELECT:
        warnings.append(f"only {counts['select']} select row(s): confidence intervals are wide")
    if bootstrap and bootstrap.get("note"):
        warnings.append(bootstrap["note"])
    if lock.get("allow_test"):
        warnings.append("test rows were read (--allow-test); the user must have approved this")
    for k, v in (lock.get("overrides") or {}).items():
        warnings.append(f"override {k} = {v.get('value')!r} (reason: {v.get('reason')})")
    if not lock["evaluator"].get("gate"):
        warnings.append("the gate is off (evaluator.gate: false): metrics are reported, not judged")
    if (lock["machine"].get("origin") or {}).get("source") == "inferred":
        warnings.append("no machine config: settings were inferred (see run.lock.yaml machine)")
    warnings += _dropped_warnings(dropped)
    report_rows = sum((counts.get("report") or {}).values())
    return {
        "run_id": lock["run_id"], "experiment": lock["experiment"], "run_dir": str(run_dir), "status": status,
        "data": {"rows": {"train_source": counts.get("train_source"), "train": counts.get("train"),
                          "select": counts.get("select"), "report": counts.get("report") or {},
                          "select_by_set": select_by_set},
                 "label_version": {"train": versions.get("train"), "reference": versions.get("reference")},
                 "sha256": {"source": data.get("source_sha256"), **(data.get("sha256") or {})},
                 "adapter": {"name": data.get("adapter"), "sha256": data.get("adapter_sha256")},
                 "allow_test": bool(lock.get("allow_test"))},
        "base": {"name": lock["base"]["name"], "revision": lock["base"].get("revision")},
        "recipe": {"path": (lock.get("recipe") or {}).get("path"), "sha256": (lock.get("recipe") or {}).get("sha256"),
                   "mode": lock["mode"], "parallel": lock.get("parallel")},
        "gpus": {"n": lock["n_gpus"], "cards": cards, "explicit": lock["machine"]["gpu"].get("explicit")},
        "wall_time": {"seconds": seconds, "gpu_hours": timing.get("gpu_hours"),
                      "per_seed": {k: {x: v.get(x) for x in ("start", "end", "seconds")} for k, v in seeds_t.items()}},
        "seeds": seeds, "bootstrap": bootstrap, "warnings": warnings, "failures": list(failures),
        "next": _next(lock, run_dir, status, candidates, report_rows),
    }


def _ci(v) -> str:
    return f" [{v[0]:.4f}, {v[1]:.4f}]" if isinstance(v, (list, tuple)) and len(v) == 2 else ""


def _metrics_table(methods: dict, reps: int | None = None) -> list[str]:
    cols = list(dict.fromkeys(k for m in methods.values() for k in m
                              if not k.endswith(("_ci95", "_ci95_skipped"))))
    lines = ["| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, m in methods.items():
        cells = [_num(m.get(c)) + _ci(m.get(f"{c}_ci95"))
                 + (f" ({m[f'{c}_ci95_skipped']}/{reps} skipped)" if m.get(f"{c}_ci95_skipped") else "")
                 for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_result_md(r: dict) -> str:
    d, rows = r["data"], r["data"]["rows"]
    per_set = ", ".join(f"{k}: {v}" for k, v in (rows.get("select_by_set") or {}).items())
    report = ", ".join(f"{k}: {v}" for k, v in (rows.get("report") or {}).items()) or "none"
    lines = [f"# Result: {r['experiment']} / {r['run_id']}", "", f"Status: **{r['status']}**", "", "## Data",
             f"- rows: train {rows.get('train')} (source {rows.get('train_source')}); select {rows.get('select')}"
             + (f" ({per_set})" if per_set else "") + f"; report {report}",
             f"- label version: train {d['label_version']['train']}; reference {d['label_version']['reference']}",
             "- sha256: " + "; ".join(f"{k} `{v}`" for k, v in d["sha256"].items()),
             f"- adapter: {d['adapter']['name']} (sha256 `{d['adapter']['sha256']}`)",
             f"- test rows read: {'yes (--allow-test)' if d['allow_test'] else 'no'}", "", "## Setup",
             f"- base: {r['base']['name']} (revision `{r['base']['revision']}`)",
             f"- recipe: {r['recipe']['path']} (mode {r['recipe']['mode']}, {r['recipe']['parallel']}, "
             f"sha256 `{r['recipe']['sha256']}`)"]
    g = r["gpus"]
    names = ", ".join(f"{c.get('name')} ({c.get('uuid')})" for c in g["cards"]) or "-"
    lines.append(f"- GPUs: {g['n']}: {names}" + (f"; requested cards {g['explicit']}" if g.get("explicit") else ""))
    w = r["wall_time"]
    gh = w.get("gpu_hours")
    lines.append(f"- wall time (training): {_hms(w['seconds'])} ({w['seconds']:.0f} s); "
                 + (f"{gh:.2f} GPU-hours" if gh is not None else "GPU-hours unknown"))
    b = r.get("bootstrap")
    ci_note = (f"95% CI by bootstrap ({b['level']}, {b['reps']} reps, seed {b['seed']})" if b
               else "no confidence intervals (bootstrap off)")
    for key, s in r["seeds"].items():
        lines += ["", f"## Seed {key}", "", "### Training curve (select NLL)", "", "| step | NLL |", "|---|---|"]
        for p in s["curve"]:
            mark = " ← selected" if p["step"] == s["selected_step"] else ""
            lines.append(f"| {p['step']} | {_num(p['nll'])}{mark} |")
        if not s["curve"]:
            lines.append("| - | - |")
        rs = s.get("rescore") or {}
        lines += ["", f"- selected step: {s['selected_step']}; checkpoint {s['checkpoint']}; "
                      f"sha256 `{s['checkpoint_sha256']}`"
                  + (f"; rescored NLL {_num(rs.get('rescored_nll'))} (training {_num(rs.get('train_nll'))})"
                     if rs else "")]
        m = s["metrics"]
        lines += ["", f"### Metrics on the select part ({m['set']}, {m['questions']} questions; {ci_note})", ""]
        lines += _metrics_table(m["methods"], b["reps"] if b else None) if m["methods"] else ["- not evaluated"]
        if b and any(k.endswith("_ci95_skipped") for x in m["methods"].values() for k in x):
            lines += ["", "(n/reps skipped): bootstrap replicates where the metric was undefined (one class only); "
                          "the interval uses the rest."]
        for name, ms in (s.get("metrics_by_set") or {}).items():
            lines += ["", f"### Metrics on eval set {name} (select part, {ms['questions']} questions)", ""]
            lines += _metrics_table(ms["methods"], b["reps"] if b else None) if ms["methods"] else ["- not evaluated"]
        lines += ["", "Jaccard: each method at its own best threshold on select. The model's locked thresholds:"]
        lines += ["", "| threshold | value | small share | Jaccard | inconsistency | per set (share / Jaccard) |",
                  "|---|---|---|---|---|---|"]
        for name, t in s["thresholds"].items():
            p = t.get("pooled") or {}
            per = "; ".join(f"{n}: {_num(v.get('small_share'))} / {_num(v.get('jaccard'))}"
                            for n, v in (t.get("per_set") or {}).items())
            lines.append(f"| {name} | {_num(t.get('threshold'))} | {_num(p.get('small_share'))} | "
                         f"{_num(p.get('jaccard'))} | {_num(p.get('inconsistency'))} | {per or '-'} |")
        if not s["thresholds"]:
            lines.append("| - | - | - | - | - | - |")
    lines += ["", "## Gate"]
    for key, s in r["seeds"].items():
        gt = s["gate"]
        verdict = ("off (report only)" if not gt["enabled"]
                   else {True: "PASSED", False: "FAILED", None: "undecided"}.get(gt["passed"], str(gt["passed"])))
        lines.append(f"- seed {key}: **{verdict}** (the model must beat the best available baseline on every "
                     "eval set; a set with no baseline fails)")
        lines += [f"  - {x}" for x in gt["reasons"]]
    if r.get("failures"):
        lines += ["", "## Failures"] + [f"- {x}" for x in r["failures"]]
    lines += ["", "## Warnings"] + ([f"- {x}" for x in r["warnings"]] or ["- none"])
    lines += ["", "## Next"]
    for n in r["next"]:
        lines.append(f"- `{n['command']}`: {n['what']}" + (" (**needs user approval**)" if n["needs_approval"] else ""))
    return "\n".join(lines) + "\n"


def write_result(run_dir, result: dict) -> Path:
    run_dir = Path(run_dir)
    run_files.write_json(run_dir / "result.json", result)
    path = run_dir / "RESULT.md"
    path.write_text(render_result_md(result))
    return path
