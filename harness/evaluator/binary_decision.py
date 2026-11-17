"""binary_decision evaluator: router vs baselines per eval set, locked thresholds, gate.

Reproduces v1 `regen_20261111/train/gate.py`. Labels (`rate`) and consistency come from the
reference row of each select row (the select row itself when no reference file was given).
"""
from __future__ import annotations

import statistics

from harness.baselines.tfidf import TextBaseline, group_prior
from harness.config import get_field
from harness.evaluator.metrics import (at_threshold, auc, budget_threshold, ece, fit_threshold,
                                       max_share_at_budget, within_group_auc)
from harness.evaluator.text import text_of

POOLED = "pooled_select"


def _num(row, path):
    """Numeric field value, or None when absent or not a number."""
    try:
        v = get_field(row, path)
    except KeyError:
        return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _pct(b) -> int:
    return round(b * 100)


def _summary(examples, scores, budgets, *, group: bool, with_ece: bool = False) -> dict:
    rates = [e["rate"] for e in examples]
    out = {"auc": auc(scores, rates)}
    if group:
        out["within_group_auc"] = within_group_auc([e["group"] for e in examples], scores, rates)
    out["jaccard_at_own_best_threshold"] = fit_threshold(scores, rates)["jaccard"]
    if budgets is not None:
        incons = [1 - e["f"] for e in examples]
        for b in budgets:
            out[f"share_at_budget_{_pct(b)}"] = max_share_at_budget(scores, incons, b)
    if with_ece:
        out["ece"] = ece(scores, rates)
    return out


def _beats(router, others, key) -> bool:
    mine = router.get(key)
    if mine is None:
        return False
    rivals = [o[key] for o in others if o.get(key) is not None]
    return mine > max(rivals) if rivals else True


def evaluate(preds, parts, train_rows, fields, evaluator_cfg) -> dict:
    """Score `preds` (id -> positive-class probability) on the pooled and per-set select rows.

    A select row is its own reference when `parts.reference` lacks its id (eval sets without a
    reference file). The gate is always evaluated per set; `gate["enabled"]` mirrors
    `evaluator_cfg.gate` and `gate["passed"]` is None when the gate is off.
    """
    has_group = fields.group is not None
    has_f = fields.consistency is not None
    budgets = list(evaluator_cfg.budgets) if has_f else None

    dropped: dict[str, list[str]] = {}
    examples: list[dict] = []
    for row in parts.select_rows:
        rid = row["id"]
        ref = parts.reference.get(rid, row)
        rate = _num(ref, fields.label)
        f = _num(ref, fields.consistency) if has_f else None
        group = None
        if has_group:
            try:
                group = get_field(ref, fields.group)
            except KeyError:
                group = None
        if rate is None or (has_f and f is None) or (has_group and group is None):
            dropped.setdefault(parts.select_index[rid], []).append(rid)
            continue
        if rid not in preds:
            raise ValueError(f"no prediction for select row {rid}")
        ex = {"id": rid, "rate": rate, "set": parts.select_index[rid], "pred": float(preds[rid])}
        if has_f:
            ex["f"] = f
        if has_group:
            ex["group"] = group
        if fields.text:
            ex["text"] = text_of(row, fields.text)
        examples.append(ex)

    train = []
    for row in train_rows:
        rate = _num(row, fields.label)
        if rate is None:
            continue
        ex = {"id": row["id"], "rate": rate}
        if has_group:
            try:
                ex["group"] = get_field(row, fields.group)
            except KeyError:
                continue
        if fields.text:
            ex["text"] = text_of(row, fields.text)
        train.append(ex)

    model = TextBaseline(train) if fields.text and train else None
    train_groups = {t["group"] for t in train} if has_group else set()

    names = list(dict.fromkeys(e["set"] for e in examples))
    by_set = {n: [e for e in examples if e["set"] == n] for n in names}
    by_set[POOLED] = examples
    scores_of: dict[str, dict[str, list[float]]] = {}
    sets: dict[str, dict] = {}
    for n, ex in by_set.items():
        if not ex:
            continue
        sc = {"router": [e["pred"] for e in ex]}
        if model is not None:
            sc["tfidf"] = model.predict(ex)
        if has_group and train:
            sc["group_prior"] = group_prior(train, ex)
        sc["all_positive"] = [1.0] * len(ex)
        if has_f:
            sc["oracle"] = [e["f"] for e in ex]
        scores_of[n] = sc
        entry = {"questions": len(ex), "positive_rate": statistics.fmean(e["rate"] for e in ex)}
        if has_group:
            entry["groups"] = len({e["group"] for e in ex})
            entry["unseen_groups"] = len({e["group"] for e in ex} - train_groups)
        for name, s in sc.items():
            entry[name] = _summary(ex, s, budgets, group=has_group, with_ece=name == "router")
        sets[n] = entry

    result: dict = {"sets": sets, "thresholds": {}, "gate": {}, "dropped": dropped,
                    "tfidf_strength": model.strength if model is not None else None}
    if not examples:
        result["gate"] = {"enabled": bool(evaluator_cfg.gate), "passed": False if evaluator_cfg.gate else None}
        return result

    # Thresholds: fitted on pooled select rows with router scores only.
    r = scores_of[POOLED]["router"]
    labels = [e["rate"] for e in examples]
    jac = fit_threshold(r, labels)
    thr = {"best_jaccard": {"threshold": jac["threshold"], "jaccard": jac["jaccard"],
                            "small_share": jac["small_usage"]}}
    if has_f:
        incons = [1 - e["f"] for e in examples]
        for b in budgets:
            thr[f"budget_{_pct(b)}"] = budget_threshold(r, incons, b)
    for t in thr.values():
        t["per_set"] = {}
        for n in names:
            ex = by_set[n]
            t["per_set"][n] = at_threshold(
                scores_of[n]["router"], [e["rate"] for e in ex],
                [1 - e["f"] for e in ex] if has_f else [0.0] * len(ex), t["threshold"])
            if not has_f:
                t["per_set"][n].pop("inconsistency")
    result["thresholds"] = thr

    keys = ["auc"] + (["within_group_auc"] if has_group else [])
    gate = {"enabled": bool(evaluator_cfg.gate)}
    for n in names:
        e = sets[n]
        rivals = [e[k] for k in ("tfidf", "group_prior") if k in e]
        gate[n] = {k: _beats(e["router"], rivals, k) for k in keys}
    gate["passed"] = all(v for n in names for v in gate[n].values()) if evaluator_cfg.gate else None
    result["gate"] = gate
    return result
