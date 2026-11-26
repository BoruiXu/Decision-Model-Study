"""The threshold of a binary decision rule: a row is positive when its calibrated score q >= the threshold, and
float("inf") means never positive.

The candidate thresholds are the distinct scores. A tie in a rule's objective goes to the highest threshold.
- max_f1: the highest F1 (2tp / (2tp + fp + fn), 0 when the denominator is 0);
- cost {cost_fp, cost_fn}: the lowest cost_fp * fp + cost_fn * fn; predicting none (inf) is a candidate too, the
  highest of all. The rule's own costs, else `costs` (from the task's metrics.costs);
- precision_at {target}: of the thresholds with a precision of at least target, the one with the highest recall
  (none reaches it: inf);
- recall_at {target}: of the thresholds with a recall of at least target, the one with the highest precision (no
  positive row: inf);
- top_frac {frac}: the highest threshold at which at least frac of the rows are positive;
- fixed {value}: value;
- argmax: 0.5 (the more probable label: apply.py decides it from the probabilities, a tie by the labels' order).
"""
INF = float("inf")
RULES = ("max_f1", "cost", "precision_at", "recall_at", "top_frac", "fixed", "argmax")


def cuts(p, y):
    """[(threshold, tp, fp)] for every distinct score, from the highest down: the rows with q >= threshold are
    positive."""
    items = sorted(zip(p, y), key=lambda t: -t[0])
    out, tp, fp, i = [], 0, 0, 0
    while i < len(items):
        j = i
        while j < len(items) and items[j][0] == items[i][0]:
            if items[j][1]:
                tp += 1
            else:
                fp += 1
            j += 1
        out.append((items[i][0], tp, fp))
        i = j
    return out


def _f1(tp, fp, fn):
    return 2.0 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def _best(candidates):
    """The threshold of the highest value in [(threshold, value or None)], highest threshold first (the first of a
    tie); inf when no value is given."""
    best_thr, best = INF, None
    for thr, value in candidates:
        if value is not None and (best is None or value > best):
            best_thr, best = thr, value
    return best_thr


def choose_threshold(rule, p, y, costs):
    """The threshold of decision rule `rule` ({"rule", ...its parameters}) on the calibrated scores p and labels y (1
    positive, 0 not); `costs` {"cost_fp", "cost_fn"} or None. ValueError for an unknown rule or a cost rule without
    costs."""
    name = rule["rule"]
    if name not in RULES:
        raise ValueError("unknown decision rule %r (one of %s)" % (name, ", ".join(RULES)))
    if name == "argmax":
        return 0.5
    if name == "fixed":
        return float(rule["value"])
    cs = cuts(p, y)
    n, pos = len(y), sum(1 for t in y if t)
    if not cs:
        return INF
    if name == "top_frac":
        return next((thr for thr, tp, fp in cs if float(tp + fp) / n >= rule["frac"]), cs[-1][0])
    if name == "max_f1":
        return _best([(thr, _f1(tp, fp, pos - tp)) for thr, tp, fp in cs])
    if name == "cost":
        c_fp = rule.get("cost_fp", (costs or {}).get("cost_fp"))
        c_fn = rule.get("cost_fn", (costs or {}).get("cost_fn"))
        if c_fp is None or c_fn is None:
            raise ValueError("the cost rule needs cost_fp and cost_fn, its own or from metrics.costs")
        return _best([(INF, -c_fn * pos)] + [(thr, -(c_fp * fp + c_fn * (pos - tp))) for thr, tp, fp in cs])
    target = rule["target"]
    if name == "precision_at":
        return _best([(thr, float(tp) / pos if pos else 0.0) for thr, tp, fp in cs
                      if float(tp) / (tp + fp) >= target])
    return _best([(thr, float(tp) / (tp + fp)) for thr, tp, fp in cs if pos and float(tp) / pos >= target])
