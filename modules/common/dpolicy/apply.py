"""A decision policy (policy.json) applied to prediction rows. Standard library only: math and json.

This file is self-contained. The train command copies it into every candidate it packs as policy_apply.py, and the
candidate's predict.py loads it by path from there, so a candidate never imports the repository.

policy.json: {"version": 1, "objective", "accepted", "reason", "labels" (the task's labels, in order),
"calibration": {"method", "params"}, "decision": {"rule", "threshold", "positive"}, "select": {...},
"settings_sha256"} (docs/interfaces/models.md). A threshold of null means never positive.

- Calibrations. `none` keeps the probabilities as they are. In a binary task (decision.positive set) the positive
  label's probability p becomes q and the other label's 1 - q: `platt` q = sigmoid(a * logit(p) + b), `temperature`
  q = sigmoid(logit(p) / T), p clipped to [1e-6, 1 - 1e-6] for the logit; `isotonic` q = the linear interpolation of
  the points (x, y), clipped at both ends. With more labels, `temperature` turns the probabilities (in label order,
  each clipped to at least 1e-12) into softmax(log p / T).
- Decision. Rule `argmax` (and every rule of a task with more than two labels): the most probable label, a tie
  going to the earlier label in the labels' order. Any other rule: the positive label when q >= threshold, else the
  other label.
- The row: {"id", "probs", "decision", "uncertainty"}, uncertainty being 1 minus the decision's probability.
"""
import json
import math

EPS = 1e-6         # a binary score is clipped to [EPS, 1 - EPS] before its logit
LOG_CLIP = 1e-12   # a multiclass probability is clipped to at least LOG_CLIP before its log


def load_policy(path):
    """The policy in the JSON file `path`."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def logit(p):
    p = min(1.0 - EPS, max(EPS, float(p)))
    return math.log(p) - math.log1p(-p)


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def interpolate(xs, ys, q):
    """The linear interpolation at q of the points (xs strictly ascending, ys), clipped at both ends."""
    if q <= xs[0]:
        return ys[0]
    if q >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= q:
            lo = mid
        else:
            hi = mid
    return ys[lo] + (ys[hi] - ys[lo]) * (q - xs[lo]) / (xs[hi] - xs[lo])


def calibrated_p(method, params, p):
    """The calibrated probability of the positive label (a binary task) of the score p."""
    if method == "none":
        return float(p)
    if method == "platt":
        return sigmoid(params["a"] * logit(p) + params["b"])
    if method == "temperature":
        return sigmoid(logit(p) / params["T"])
    if method == "isotonic":
        return float(interpolate(params["x"], params["y"], float(p)))
    raise ValueError("unknown calibration %r" % method)


def calibrated_vector(method, params, v):
    """The calibrated probability vector of `v` (the probabilities of every label, in label order)."""
    if method == "none":
        return [float(x) for x in v]
    if method == "temperature":
        z = [math.log(max(LOG_CLIP, float(x))) / params["T"] for x in v]
        top = max(z)
        e = [math.exp(x - top) for x in z]
        total = math.fsum(e)
        return [x / total for x in e]
    raise ValueError("calibration %s is for binary tasks only" % method)


def argmax(probs, labels):
    """The most probable label; a tie goes to the earlier label in `labels`."""
    best = labels[0]
    for label in labels[1:]:
        if probs.get(label, 0) > probs.get(best, 0):
            best = label
    return best


def apply_policy(policy, row, labels):
    """The prediction row `row` with `policy` applied, as a new row {"id", "probs", "decision", "uncertainty"}
    (`labels`: the task's labels, in order). `row` is not changed."""
    method = policy["calibration"]["method"]
    params = policy["calibration"].get("params") or {}
    dec = policy["decision"]
    positive = dec.get("positive")
    probs = dict(row["probs"])
    other = [label for label in labels if label != positive][0] if positive is not None else None
    if method != "none":
        if positive is not None:
            q = calibrated_p(method, params, probs[positive])
            probs = {positive: q, other: 1.0 - q}
        else:
            probs = dict(zip(labels, calibrated_vector(method, params, [probs[label] for label in labels])))
    if positive is None or dec["rule"] == "argmax":
        decision = argmax(probs, labels)
    else:
        thr = dec.get("threshold")
        decision = positive if thr is not None and probs[positive] >= thr else other
    return {"id": row["id"], "probs": probs, "decision": decision, "uncertainty": 1.0 - probs[decision]}
