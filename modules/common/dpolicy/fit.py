"""The decision policy fitted on a model's valset predictions (docs/interfaces/task.md, policy).

The valset is split into a fit part and a select part (split_rows). Every pair of a calibration and a rule of the
settings (rsi.rules.policy_settings), in order, is fitted on the fit part (the calibration's parameters, then the
rule's threshold on the calibrated fit scores) and applied to the select part. Its objective there is compared with
the identity policy's (calibration none, rule argmax) by the framework's paired cluster bootstrap
(rsi.metrics.paired_delta with the task's metrics.ci, clusters from meta.cluster). The improvement is the delta in
the objective's direction; its lower bound is ci_low (higher is better) or -ci_high (lower is better). A pair is
accepted when that lower bound is above min_lo and the task's primary metric (rsi.metrics, in its direction) on the
select part is not worse with the pair applied than with the identity policy (within PRIMARY_TOL): a policy never
trades the primary metric for its objective (e.g. isotonic's step function can tie scores and lower auprc for an
equal f1). A primary metric rsi does not compute is not checked. Of the accepted pairs the one with the largest
improvement wins (a tie: the earlier pair). Nothing accepted, a part of the split without some label of the task, or
a valset that cannot be split gives the identity policy with the reason. Only valset labels are read.
"""
import hashlib
import json
import math

from rsi import metrics as M
from rsi.datasplit import value_at
from rsi.rows import group_of

from . import calibrate
from .apply import apply_policy
from .rules import INF, choose_threshold

POLICY_FORMAT = 1
PRIMARY_TOL = 1e-12   # how much worse the primary metric may get with a policy (none: rounding only)


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def split_rows(val_rows, split):
    """(fit ids, select ids) of the valset rows, each in row order. `split` is {"by", "key", "select_fraction"}:
    - time: the value at the dotted path `key` of each row (a number); the last max(1, round(n x select_fraction)) of
      the n distinct values go to select. ValueError when there is no key or a row has no number there;
    - group: the value at `key` (for meta.state_id, rsi.rows.group_of, which reads meta.group in old rows; a row
      without one is a group of its own, by its id), the groups ordered by the sha256 of str(group); the last
      ceil(n x select_fraction) of the n groups go to select."""
    by, key, frac = split["by"], split.get("key"), split["select_fraction"]
    if by == "time":
        if not key:
            raise ValueError("a time split needs a key (policy.split.key, or data.split.key of a time split)")
        values = {}
        for r in val_rows:
            found, v = value_at(r, key)
            if not found or not _number(v):
                raise ValueError("row %s has no number at %s" % (r.get("id"), key))
            values[r["id"]] = v
        distinct = sorted(set(values.values()))
        chosen = set(distinct[len(distinct) - max(1, int(round(len(distinct) * frac))):]) if distinct else set()
        member = dict((i, v in chosen) for i, v in values.items())
    elif by == "group":
        groups = {}
        for r in val_rows:
            if key == "meta.state_id":
                g = group_of(r)
            else:
                found, g = value_at(r, key)
                g = g if found else None
            groups[r["id"]] = str(g) if g is not None else "row:%s" % r["id"]
        order = sorted(set(groups.values()), key=lambda g: (hashlib.sha256(g.encode("utf-8")).hexdigest(), g))
        chosen = set(order[len(order) - int(math.ceil(len(order) * frac)):]) if order else set()
        member = dict((i, g in chosen) for i, g in groups.items())
    else:
        raise ValueError("unknown split %r (time or group)" % by)
    ids = [r["id"] for r in val_rows]
    return [i for i in ids if not member[i]], [i for i in ids if member[i]]


def settings_sha256(settings):
    """The sha256 of the canonical JSON of the policy settings."""
    text = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_costs(task):
    """{"cost_fp", "cost_fn"} of a binary task's metrics.costs (true label -> decision -> cost), relative to the right
    decision: cost_fp = costs[negative][positive] - costs[negative][negative], cost_fn likewise; None without costs
    or in a task with more labels."""
    d, costs = task["decision"], task["metrics"].get("costs")
    if d["type"] != "binary" or not costs:
        return None
    pos = d["positive"]
    neg = [label for label in d["labels"] if label != pos][0]
    return {"cost_fp": costs[neg][pos] - costs[neg][neg], "cost_fn": costs[pos][neg] - costs[pos][pos]}


def _threshold_json(thr):
    return None if thr is None or thr == INF else thr


def _policy(task, settings, method, params, rule, threshold, accepted, reason, select):
    d = task["decision"]
    positive = d.get("positive") if d["type"] == "binary" else None
    return {"version": POLICY_FORMAT, "objective": settings["objective"], "accepted": accepted, "reason": reason,
            "labels": list(d["labels"]), "calibration": {"method": method, "params": params},
            "decision": {"rule": rule, "threshold": _threshold_json(threshold), "positive": positive},
            "select": select, "settings_sha256": settings_sha256(settings)}


def identity_policy(task, settings, reason, select=None):
    """The identity policy (calibration none, rule argmax: the model's own most probable label), not accepted, with
    `reason`."""
    binary = task["decision"]["type"] == "binary"
    sel = {"metric": settings["objective"], "delta": None, "ci_low": None, "ci_high": None, "n_fit": 0,
           "n_select": 0}
    sel.update(select or {})
    return _policy(task, settings, "none", {}, "argmax", 0.5 if binary else None, False, reason, sel)


def _pair_name(row):
    return "%s + %s" % (row["calibration"], row["rule"]["rule"])


def _signed(x):
    return "%+.4f" % x


def _primary_value(task, pred_rows, label_rows):
    """(value, None) of the task's primary metric on the rows, (None, why) when rsi cannot compute it there, (None,
    None) when it is not a metric rsi computes (not checked)."""
    primary = task["metrics"]["primary"]
    if primary not in M.KNOWN_METRICS:
        return None, None
    try:
        return M.score(primary, pred_rows, label_rows, task), None
    except ValueError as e:
        return None, "the primary metric %s cannot be computed on the select split (%s)" % (primary, e)


def _primary_check(task, value, identity):
    """The reason the pair is refused when its primary metric `value` is worse than the identity policy's (in the
    metric's direction, beyond PRIMARY_TOL), else None."""
    primary = task["metrics"]["primary"]
    worse = value < identity - PRIMARY_TOL if M.HIGHER_IS_BETTER[primary] else value > identity + PRIMARY_TOL
    if not worse:
        return None
    return ("it worsens the task's primary metric %s on the select split (%.4f against %.4f with the identity policy)"
            % (primary, value, identity))


def fit_policy(task, settings, val_rows, preds):
    """(policy, report) of the prediction rows `preds` (one per valset row) on the valset rows `val_rows` (with
    their targets), for the settings `settings` (rsi.rules.policy_settings). The policy is policy.json's content
    (apply.py); the report holds every pair's numbers: {"split", "n_fit", "n_select", "primary": {"metric",
    "identity"}, "pairs": [{"calibration", "params", "rule", "threshold", "delta", "ci_low", "ci_high",
    "improvement", "low", "primary" (its value), "accepted", and "rejected" (why) when not accepted} or with
    "error"]}. ValueError when the predictions do not cover the valset rows."""
    labels = task["decision"]["labels"]
    positive = task["decision"].get("positive") if task["decision"]["type"] == "binary" else None
    objective = settings["objective"]
    by_id = dict((r["id"], r) for r in preds)
    missing = [r["id"] for r in val_rows if r["id"] not in by_id]
    if missing:
        raise ValueError("the predictions lack %d valset rows, e.g. %s" % (len(missing), missing[:3]))
    report = {"split": dict(settings["split"]), "n_fit": 0, "n_select": 0, "pairs": []}
    try:
        fit_ids, select_ids = split_rows(val_rows, settings["split"])
    except ValueError as e:
        return identity_policy(task, settings, "the valset cannot be split for the policy: %s" % e), report
    counts = {"n_fit": len(fit_ids), "n_select": len(select_ids)}
    report.update(counts)
    rows = dict((r["id"], r) for r in val_rows)
    hard = dict((i, M.hard_label(rows[i]["target"], labels)) for i in rows)
    for part, ids in (("fit", fit_ids), ("select", select_ids)):
        have = set(hard[i] for i in ids)
        for label in labels:
            if label not in have:
                return identity_policy(task, settings, "the %s split lacks label %s: the identity policy (none + "
                                       "argmax)" % (part, label), counts), report
    if positive is not None:
        fit_p = [by_id[i]["probs"][positive] for i in fit_ids]
        fit_y = [1 if hard[i] == positive else 0 for i in fit_ids]
    else:
        fit_p = [[by_id[i]["probs"][label] for label in labels] for i in fit_ids]
        fit_y = [labels.index(hard[i]) for i in fit_ids]
    select_rows = [rows[i] for i in select_ids]
    identity = identity_policy(task, settings, "")
    base = [apply_policy(identity, by_id[i], labels) for i in select_ids]
    base_primary, primary_error = _primary_value(task, base, select_rows)
    report["primary"] = {"metric": task["metrics"]["primary"], "identity": base_primary}
    clusters = M.clusters_of(select_rows)
    ci = M.ci_settings(task)
    sign = 1 if M.HIGHER_IS_BETTER[objective] else -1
    costs = binary_costs(task)
    best = None
    for method in settings["calibrations"]:
        params = calibrate.fit(method, fit_p, fit_y)
        q_fit = calibrate.apply(method, params, fit_p)
        for rule in settings["rules"]:
            thr = choose_threshold(rule, q_fit, fit_y, costs) if positive is not None else None
            pol = _policy(task, settings, method, params, rule["rule"], thr, True, "", None)
            row = {"calibration": method, "params": params, "rule": dict(rule), "threshold": _threshold_json(thr)}
            report["pairs"].append(row)
            cand = [apply_policy(pol, by_id[i], labels) for i in select_ids]
            try:
                d = M.paired_delta(objective, cand, base, select_rows, task, clusters, **ci)
            except ValueError as e:
                row.update(error=str(e), accepted=False)
                continue
            improvement = sign * d["delta"]
            low = d["ci_low"] if sign > 0 else -d["ci_high"]
            value, why = _primary_value(task, cand, select_rows) if base_primary is not None else (None,
                                                                                                    primary_error)
            row.update(d, improvement=improvement, low=low, primary=value)
            if not low > settings["min_lo"]:
                why = "the improvement's lower bound %s is not above min_lo %s" % (_signed(low), settings["min_lo"])
            elif why is None and value is not None:
                why = _primary_check(task, value, base_primary)
            row["accepted"] = why is None
            if why is not None:
                row["rejected"] = why
            if row["accepted"] and (best is None or improvement > best[0]["improvement"]):
                best = (row, pol, d)
    scored = [r for r in report["pairs"] if "error" not in r]
    if best is None:
        if scored:
            top = max(scored, key=lambda r: r["improvement"])   # the first of a tie
            why = ("no calibration and rule was accepted on the select split; the best on %s, %s (%s %s [%s, %s]), "
                   "was not: %s" % (objective, _pair_name(top), objective, _signed(top["delta"]),
                                    _signed(top["ci_low"]), _signed(top["ci_high"]), top["rejected"]))
        else:
            why = "no calibration and rule could be scored on the select split (%s)" % report["pairs"][0]["error"]
        return identity_policy(task, settings, why + ": the identity policy (none + argmax)", counts), report
    row, pol, d = best
    pol["reason"] = ("%s: %s %s [%s, %s] against the identity policy on the select split (%d rows); the improvement's "
                     "lower bound %s is above min_lo %s" % (_pair_name(row), objective, _signed(d["delta"]),
                                                             _signed(d["ci_low"]), _signed(d["ci_high"]),
                                                             len(select_ids), _signed(row["low"]),
                                                             settings["min_lo"]))
    if row["primary"] is not None:
        pol["reason"] += ", and the primary metric %s is not worse (%.4f against %.4f with the identity policy)" % (
            task["metrics"]["primary"], row["primary"], base_primary)
    pol["select"] = dict(metric=objective, delta=d["delta"], ci_low=d["ci_low"], ci_high=d["ci_high"], **counts)
    return pol, report
