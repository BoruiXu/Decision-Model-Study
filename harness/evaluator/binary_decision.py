"""binary_decision evaluator: router vs baselines per eval set, locked thresholds, gate.

Reproduces v1 `regen_20261111/train/gate.py`. Labels (`rate`) and consistency come from the
reference row of each select row (the select row itself when no reference file was given).
"""
from __future__ import annotations

import statistics

from harness.baselines.tfidf import TextBaseline, group_prior
from harness.config import get_field
from harness.evaluator.bootstrap import LEVEL_GROUPED, LEVEL_ITEM, bootstrap_cis
from harness.evaluator.labels import baseline_examples, label_value, num
from harness.evaluator.metrics import (at_threshold, auc, budget_threshold, ece, fit_threshold,
                                       max_share_at_budget, within_group_auc)
from harness.evaluator.text import text_of

POOLED = "pooled_select"
TRAIN_DROPPED = "baseline_train"  # key in ``dropped`` for train rows the baselines could not use


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


NO_RIVAL = "no rival baseline (TF-IDF needs fields.text, the group prior fields.group; both need labeled train rows)"


def _gate(set_names, sets, has_group, evaluator_cfg) -> dict:
    """Per eval set: does the router beat every rival baseline on AUC (and within-group AUC)? A set without
    any rival baseline, or whose select rows were all dropped, records why and fails: a gate with nothing
    to beat cannot pass."""
    keys = ["auc"] + (["within_group_auc"] if has_group else [])
    gate: dict = {"enabled": bool(evaluator_cfg.gate)}
    ok = []
    for n in set_names:
        e = sets.get(n)
        rivals = [e[k] for k in ("tfidf", "group_prior") if k in e] if e is not None else []
        if e is None:
            gate[n] = {"passed": False, "reason": "every select row was dropped (no label, consistency or group)"}
        elif not rivals:
            gate[n] = {"passed": False, "reason": NO_RIVAL}
        else:
            gate[n] = {k: _beats(e["router"], rivals, k) for k in keys}
        ok.append("reason" not in gate[n] and all(gate[n].values()))
    gate["passed"] = (bool(ok) and all(ok)) if evaluator_cfg.gate else None
    return gate


def evaluate(preds, parts, train_rows, fields, evaluator_cfg) -> dict:
    """Score `preds` (id -> positive-class probability) on the pooled and per-set select rows.

    A select row is its own reference when `parts.reference` lacks its id (eval sets without a
    reference file). The gate is always evaluated per eval set that has select rows (see `_gate`);
    `gate["enabled"]` mirrors `evaluator_cfg.gate` and `gate["passed"]` is None when the gate is off.
    """
    has_group = fields.group is not None
    has_f = fields.consistency is not None
    budgets = list(evaluator_cfg.budgets) if has_f else None

    dropped: dict[str, list[str]] = {}
    examples: list[dict] = []
    for row in parts.select_rows:
        rid = row["id"]
        ref = parts.reference.get(rid, row)
        rate = label_value(ref, fields.label)
        f = num(ref, fields.consistency) if has_f else None
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

    train, skipped = baseline_examples(train_rows, fields)
    if skipped:
        dropped[TRAIN_DROPPED] = skipped

    model = TextBaseline(train) if fields.text and train else None
    train_groups = {t["group"] for t in train} if has_group else set()

    reps = int(getattr(evaluator_cfg, "bootstrap_reps", 0) or 0)
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
        if reps > 0:
            cis = bootstrap_cis(ex, sc, budgets=budgets, has_group=has_group, has_f=has_f, reps=reps,
                                seed=evaluator_cfg.bootstrap_seed, salt=n)
            for name, ci in cis.items():
                entry[name]["ci95"] = ci
        sets[n] = entry

    all_sets = list(dict.fromkeys(parts.select_index[r["id"]] for r in parts.select_rows))
    result: dict = {"sets": sets, "thresholds": {}, "gate": _gate(all_sets, sets, has_group, evaluator_cfg),
                    "dropped": dropped, "tfidf_strength": model.strength if model is not None else None,
                    "bootstrap": None}
    if reps > 0:
        result["bootstrap"] = {"reps": reps, "seed": evaluator_cfg.bootstrap_seed, "ci": 0.95,
                               "level": LEVEL_GROUPED if has_group else LEVEL_ITEM}
        if not has_group:
            result["bootstrap"]["note"] = "no group field: items are resampled directly (item level)"
    if not examples:
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
    return result
