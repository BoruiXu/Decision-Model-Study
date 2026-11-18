"""Opt-in evaluation on the report halves with the locked checkpoint and thresholds. Imports without torch.

Nothing is refitted: scores come from the selected checkpoint of one seed and the thresholds are the ones
finalize locked in ``thresholds.json``.
"""
from __future__ import annotations

import contextlib
import json
import statistics
from pathlib import Path

import yaml

from harness import run_files
from harness.config import get_field
from harness.evaluator.labels import label_value, num
from harness.evaluator.metrics import at_threshold, auc, ece, within_group_auc
from harness.finalize import _gpu_env, check_adapter, exp_from_lock, score_kwargs
from harness.prepare import build_eval_parts, read_eval
from harness.score import score_checkpoint

POOLED = "pooled_report"


class ReportRefused(RuntimeError):
    """Report halves are read only with --allow-report."""


def _metrics(ex: list[dict], thresholds: dict, *, group: bool, cons: bool) -> dict:
    scores, rates = [e["pred"] for e in ex], [e["rate"] for e in ex]
    out = {"questions": len(ex), "positive_rate": statistics.fmean(rates), "auc": auc(scores, rates)}
    if group:
        out["within_group_auc"] = within_group_auc([e["group"] for e in ex], scores, rates)
    out["ece"] = ece(scores, rates)
    out["thresholds"] = {}
    for name, t in thresholds.items():
        r = at_threshold(scores, rates, [1 - e["f"] for e in ex] if cons else [0.0] * len(ex), t["threshold"])
        if not cons:
            r.pop("inconsistency")
        out["thresholds"][name] = {"threshold": t["threshold"], **r}
    return out


def run(run_dir, *, allow_report: bool, seed=None, allow_test: bool = False, scorer=score_checkpoint,
        devices: list[str] | None = None) -> dict:
    """Score the report rows with ``seed``'s selected checkpoint and write ``report_eval.json``."""
    if not allow_report:
        raise ReportRefused("report halves are read only with --allow-report")
    run_dir = Path(run_dir).absolute()
    lock = yaml.safe_load((run_dir / "run.lock.yaml").read_text())
    check_adapter(lock)
    seed = lock["seeds"][0] if seed is None else seed
    if seed not in lock["seeds"]:
        raise ValueError(f"seed {seed} is not in the run's seeds {lock['seeds']}")
    key = str(seed)
    comparison = json.loads((run_dir / "comparison.json").read_text()).get(key) or {}
    thresholds = (json.loads((run_dir / "thresholds.json").read_text()).get(key)) or {}
    if not comparison.get("checkpoint"):
        raise ValueError(f"seed {seed} has no selected checkpoint in comparison.json: finalize it first")
    if not thresholds:
        raise ValueError(f"seed {seed} has no locked thresholds in thresholds.json: finalize it first")
    ckpt = (run_dir / comparison["checkpoint"]).resolve()

    exp = exp_from_lock(lock)
    fields = exp.fields
    allow_test = bool(allow_test or lock.get("allow_test"))  # a run prepared with --allow-test keeps it
    parts = build_eval_parts(exp, allow_test=allow_test, check_criteria=False)  # validated at prepare
    rows, owner = [], {}
    for name, spec in exp.data.eval.items():
        wanted = set(parts.report_ids.get(name, []))
        for r in read_eval(exp.data, name, spec.train_labels):
            if r["id"] in wanted:
                rows.append(r)
                owner[r["id"]] = name
    if not rows:
        raise ValueError("no report rows: the split file assigns none to 'report'")

    lora = lock["mode"] == "lora"
    base_dir = Path(lock["machine"]["model_dir"]) / lock["base"]["name"] if lora else None
    ctx = _gpu_env(run_dir, lock, devices) if devices is not None else contextlib.nullcontext()
    with ctx:
        recs = scorer(ckpt, rows, base_dir=base_dir, **score_kwargs(lock))
    pos = fields.label.rsplit(".", 1)[-1]

    has_group, has_f = fields.group is not None, fields.consistency is not None
    examples, dropped = [], {}
    for r in rows:
        rid = r["id"]
        ref = parts.reference.get(rid, r)
        rate = label_value(ref, fields.label)
        f = num(ref, fields.consistency) if has_f else None
        try:
            group = get_field(ref, fields.group) if has_group else None
        except KeyError:
            group = None
        if rate is None or (has_f and f is None) or (has_group and group is None):
            dropped.setdefault(owner[rid], []).append(rid)
            continue
        rec = recs[rid]
        if pos not in rec["labels"]:
            raise ValueError(f"fields.label {fields.label!r}: key {pos!r} is not a scored label {rec['labels']}")
        examples.append({"id": rid, "set": owner[rid], "rate": rate, "f": f, "group": group,
                         "pred": rec["probs"][rec["labels"].index(pos)]})

    sets = {}
    for n in dict.fromkeys(e["set"] for e in examples):
        sets[n] = _metrics([e for e in examples if e["set"] == n], thresholds, group=has_group, cons=has_f)
    if examples:
        sets[POOLED] = _metrics(examples, thresholds, group=has_group, cons=has_f)
    out = {"run_id": lock["run_id"], "seed": seed, "selected_step": comparison.get("selected_step"),
           "checkpoint": comparison["checkpoint"], "allow_test": bool(allow_test),
           "thresholds_locked": {k: t["threshold"] for k, t in thresholds.items()}, "sets": sets,
           "dropped": dropped}
    run_files.write_json(run_dir / "report_eval.json", out)
    return out
