"""Metric, baseline and confidence-interval definitions (pure, stdlib only).

Every number the promotion rule and the train gate rely on is computed here:
modules compute the numbers they report with `rsi score`, and registration
recomputes them with the same functions and refuses mismatches. Names that are
not listed here may appear in a task, but rsi neither computes nor verifies them.

Rows are dicts as in the artifact files: label rows (full rows or `labels.jsonl`
rows) carry `id` and `target`, prediction rows `id`, `probs` and `decision`.
Rows are matched by id. Sums use math.fsum, so results do not depend on row
order or on the Python version.
"""
import math
import random

LOG_LOSS_CLIP = 1e-12
# Direction of each known metric; a task's higher_is_better must agree (rsi.rows.check_task).
HIGHER_IS_BETTER = {"accuracy": True, "log_loss": False, "macro_f1": True, "auc": True}
KNOWN_METRICS = ("accuracy", "log_loss", "macro_f1", "auc")
KNOWN_BASELINES = ("majority", "group_prior")
# Optional task["metrics"]["ci"] fields and their defaults; ci_settings applies them.
CI_DEFAULTS = {"resamples": 1000, "seed": 0, "level": 0.95}


def hard_label(target, labels):
    """The label with the largest probability in `target`; ties go to the earlier label in `labels`."""
    best = labels[0]
    for label in labels[1:]:
        if target.get(label, 0) > target.get(best, 0):
            best = label
    return best


def ci_settings(task):
    """{"resamples", "seed", "level"} of task["metrics"]["ci"], with CI_DEFAULTS for absent fields."""
    ci = task["metrics"].get("ci", {})
    return dict((k, ci.get(k, v)) for k, v in CI_DEFAULTS.items())


def _mean(values):
    return math.fsum(values) / len(values)


def _pairs(pred_rows, label_rows, task):
    """[(prediction row, hard label)] in label-row order; the ids must match exactly."""
    labels = task["decision"]["labels"]
    by_id = {}
    for r in pred_rows:
        if r["id"] in by_id:
            raise ValueError("duplicate prediction id %r" % r["id"])
        by_id[r["id"]] = r
    ids = [r["id"] for r in label_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate label row ids")
    missing = [i for i in ids if i not in by_id]
    extra = sorted(set(by_id) - set(ids))
    if missing or extra:
        raise ValueError("predictions must cover exactly the label ids: %d missing (e.g. %s), %d unexpected "
                         "(e.g. %s)" % (len(missing), missing[:3], len(extra), extra[:3]))
    if not ids:
        raise ValueError("no label rows")
    return [(by_id[r["id"]], hard_label(r["target"], labels)) for r in label_rows]


# Per-row values of the metrics that are means over rows.
def _row_accuracy(p, y, task):
    return 1.0 if p["decision"] == y else 0.0


def _row_log_loss(p, y, task):
    prob = min(1.0, max(LOG_LOSS_CLIP, p["probs"][y]))
    return -math.log(prob) + 0.0  # + 0.0 turns -0.0 into 0.0


_ROW_VALUES = {"accuracy": _row_accuracy, "log_loss": _row_log_loss}


def _macro_f1(pairs, task):
    scores = []
    for label in task["decision"]["labels"]:
        tp = sum(1 for p, y in pairs if p["decision"] == label and y == label)
        fp = sum(1 for p, y in pairs if p["decision"] == label and y != label)
        fn = sum(1 for p, y in pairs if p["decision"] != label and y == label)
        scores.append(2.0 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return _mean(scores)


def _auc(pairs, task):
    d = task["decision"]
    if d["type"] != "binary":
        raise ValueError("auc is defined for binary tasks only (decision.type is %s)" % d["type"])
    positive = d["positive"]
    items = sorted((p["probs"][positive], y == positive) for p, y in pairs)
    n_pos = sum(1 for _s, is_pos in items if is_pos)
    n_neg = len(items) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("auc is undefined: the label rows have no %s row"
                         % ("positive" if not n_pos else "negative"))
    # Mann-Whitney U with average ranks for tied scores (a tie counts 0.5); ranks kept doubled
    # so the sum stays an exact integer.
    rank2_sum, i = 0, 0
    while i < len(items):
        j = i
        while j < len(items) and items[j][0] == items[i][0]:
            j += 1
        rank2_sum += (i + 1 + j) * sum(1 for _s, is_pos in items[i:j] if is_pos)
        i = j
    return (rank2_sum - n_pos * (n_pos + 1)) / (2.0 * n_pos * n_neg)


def _on_pairs(metric, pairs, task):
    if metric in _ROW_VALUES:
        fn = _ROW_VALUES[metric]
        return _mean([fn(p, y, task) for p, y in pairs])
    if metric == "macro_f1":
        return _macro_f1(pairs, task)
    if metric == "auc":
        return _auc(pairs, task)
    raise ValueError("unknown metric %r (rsi computes %s)" % (metric, ", ".join(KNOWN_METRICS)))


def accuracy(pred_rows, label_rows, task):
    """Share of rows whose prediction `decision` equals the hard label of `target`."""
    return _on_pairs("accuracy", _pairs(pred_rows, label_rows, task), task)


def log_loss(pred_rows, label_rows, task):
    """Mean of -ln p, p = predicted probability of the hard label, clipped to [1e-12, 1]."""
    return _on_pairs("log_loss", _pairs(pred_rows, label_rows, task), task)


def macro_f1(pred_rows, label_rows, task):
    """Mean F1 over all task labels (decision vs hard label); a label never true nor predicted scores 0."""
    return _on_pairs("macro_f1", _pairs(pred_rows, label_rows, task), task)


def auc(pred_rows, label_rows, task):
    """Binary only: ROC AUC of the predicted probability of decision.positive; ties count 0.5."""
    return _on_pairs("auc", _pairs(pred_rows, label_rows, task), task)


METRICS = {"accuracy": accuracy, "log_loss": log_loss, "macro_f1": macro_f1, "auc": auc}


def score(metric, pred_rows, label_rows, task):
    """The known metric `metric` of the predictions against the label rows (ValueError when unknown)."""
    if metric not in METRICS:
        raise ValueError("unknown metric %r (rsi computes %s)" % (metric, ", ".join(KNOWN_METRICS)))
    return METRICS[metric](pred_rows, label_rows, task)


# ---- baselines: fn(train_rows, scored_rows, task, groups=None) -> prediction rows ----

def _prior(rows, labels):
    if not rows:
        raise ValueError("no training rows")
    return dict((label, math.fsum(r["target"].get(label, 0) for r in rows) / len(rows)) for label in labels)


def _rows_for(scored_rows, probs_of, labels):
    out = []
    for r in scored_rows:
        probs = probs_of(r)
        out.append({"id": r["id"], "probs": dict(probs), "decision": hard_label(probs, labels),
                    "uncertainty": None})
    return out


def _group(row):
    meta = row.get("meta")
    g = meta.get("group") if isinstance(meta, dict) else None
    return g if isinstance(g, str) else None


def _meta_map(rows, key):
    out = {}
    for r in rows:
        meta = r.get("meta")
        value = meta.get(key) if isinstance(meta, dict) else None
        if isinstance(r.get("id"), str) and isinstance(value, str):
            out[r["id"]] = value
    return out


def groups_of(rows):
    """id -> meta.group of the rows that have both (e.g. an evalset inputs file)."""
    return _meta_map(rows, "group")


def clusters_of(rows):
    """id -> meta.cluster of the rows that have both (e.g. an evalset inputs file)."""
    return _meta_map(rows, "cluster")


def majority(train_rows, scored_rows, task, groups=None):
    """Every scored row gets the training prior (mean target); decision = its hard label."""
    labels = task["decision"]["labels"]
    prior = _prior(train_rows, labels)
    return _rows_for(scored_rows, lambda r: prior, labels)


def group_prior(train_rows, scored_rows, task, groups=None):
    """Each scored row gets the prior of its meta.group in the training rows, or the overall prior
    for a group the training rows do not have. The group comes from the scored row's meta.group,
    else from `groups` (id -> group, e.g. groups_of(evalset inputs))."""
    labels = task["decision"]["labels"]
    overall = _prior(train_rows, labels)
    by_group = {}
    for r in train_rows:
        g = _group(r)
        if g is not None:
            by_group.setdefault(g, []).append(r)
    priors = dict((g, _prior(rs, labels)) for g, rs in by_group.items())

    def probs_of(r):
        g = _group(r)
        if g is None:
            g = (groups or {}).get(r["id"])
        if g is None:
            raise ValueError("group_prior needs meta.group for row %r: give rows that carry it, or the "
                             "inputs file (e.g. the evalset inputs.jsonl)" % r["id"])
        return priors.get(g, overall)

    return _rows_for(scored_rows, probs_of, labels)


BASELINES = {"majority": majority, "group_prior": group_prior}


def baseline_predictions(name, train_rows, scored_rows, task, groups=None):
    """Prediction rows of the known baseline `name` for `scored_rows` (ValueError when unknown)."""
    if name not in BASELINES:
        raise ValueError("unknown baseline %r (rsi computes %s)" % (name, ", ".join(KNOWN_BASELINES)))
    return BASELINES[name](train_rows, scored_rows, task, groups)


# ---- paired cluster bootstrap ---------------------------------------------------

def _quantile(sorted_values, q):
    """Linear interpolation between order statistics (position (n - 1) * q)."""
    h = (len(sorted_values) - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def paired_delta(metric, pred_a, pred_b, label_rows, task, clusters, resamples, seed, level):
    """{"delta", "ci_low", "ci_high"} of metric(a) - metric(b) on the same label rows.

    The interval is a paired cluster bootstrap: clusters (id -> cluster; rows
    without one are their own cluster) are ordered by name, each of `resamples`
    draws picks as many clusters as there are, with replacement, by
    random.Random(seed).randrange, and recomputes metric(a) - metric(b) on the
    rows of the drawn clusters. A draw on which the metric is undefined (auc
    without both classes) is skipped. The bounds are the (1 - level) / 2 and
    (1 + level) / 2 quantiles of the draws (linear interpolation).
    """
    pa = _pairs(pred_a, label_rows, task)
    pb = _pairs(pred_b, label_rows, task)
    delta = _on_pairs(metric, pa, task) - _on_pairs(metric, pb, task)
    members = {}
    for i, r in enumerate(label_rows):
        c = (clusters or {}).get(r["id"])
        key = ("c", str(c)) if c is not None else ("r", str(r["id"]))
        members.setdefault(key, []).append(i)
    groups = [members[k] for k in sorted(members)]
    rows_a = rows_b = None
    if metric in _ROW_VALUES:
        fn = _ROW_VALUES[metric]
        rows_a = [fn(p, y, task) for p, y in pa]
        rows_b = [fn(p, y, task) for p, y in pb]
    rng = random.Random(seed)
    k = len(groups)
    draws = []
    for _ in range(resamples):
        idx = []
        for _j in range(k):
            idx.extend(groups[rng.randrange(k)])
        if rows_a is not None:
            draws.append(_mean([rows_a[i] for i in idx]) - _mean([rows_b[i] for i in idx]))
            continue
        try:
            draws.append(_on_pairs(metric, [pa[i] for i in idx], task)
                         - _on_pairs(metric, [pb[i] for i in idx], task))
        except ValueError:
            continue
    if not draws:
        raise ValueError("%s is undefined on every bootstrap draw" % metric)
    draws.sort()
    alpha = (1 - level) / 2
    return {"delta": delta, "ci_low": _quantile(draws, alpha), "ci_high": _quantile(draws, 1 - alpha)}
