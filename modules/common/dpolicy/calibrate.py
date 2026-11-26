"""Calibrations of a model's probabilities: fit on labelled rows, apply to any (the formulas are in apply.py).

A binary task's scores p are the positive label's probabilities and y is 1 for a positive row, else 0. A task with
more labels gives each row's probability vector (in label order) and y the index of its label; only `none` and
`temperature` apply to it.

- platt: Newton's method on (a, b) of sigmoid(a * logit(p) + b), minimizing the summed log loss plus L2 1e-6 / 2 *
  (a^2 + b^2), from (1, 0); each step is halved until the loss does not grow; at most 100 steps, stopping when both
  parts of a step are below 1e-10;
- temperature: golden-section search over log T in [-5, 5] (100 iterations) minimizing the mean log loss;
- isotonic: pool adjacent violators on the rows sorted by p (rows of equal p pooled first, then adjacent blocks whose
  means do not increase); params {"x", "y"}: each block's lowest and highest p with its mean y, x strictly
  ascending.
"""
import math

from .apply import LOG_CLIP, calibrated_p, calibrated_vector, logit, sigmoid

METHODS = ("none", "platt", "temperature", "isotonic")
L2 = 1e-6
PLATT_STEPS = 100
PLATT_TOL = 1e-10
LOG_T_RANGE = (-5.0, 5.0)
GOLDEN_ITERATIONS = 100


def _log_sigmoid(t):
    """log(sigmoid(t)), stable for large |t|."""
    return -math.log1p(math.exp(-t)) if t >= 0 else t - math.log1p(math.exp(t))


def _binary_loss(z, y, scale, shift):
    """The summed log loss of sigmoid(scale * z + shift) against y."""
    return -math.fsum(_log_sigmoid((scale * zi + shift) if yi else -(scale * zi + shift)) for zi, yi in zip(z, y))


def _platt(p, y):
    z = [logit(x) for x in p]

    def loss(a, b):
        return _binary_loss(z, y, a, b) + L2 / 2 * (a * a + b * b)

    a, b = 1.0, 0.0
    cur = loss(a, b)
    for _ in range(PLATT_STEPS):
        q = [sigmoid(a * zi + b) for zi in z]
        r = [qi - yi for qi, yi in zip(q, y)]
        w = [qi * (1.0 - qi) for qi in q]
        ga = math.fsum(ri * zi for ri, zi in zip(r, z)) + L2 * a
        gb = math.fsum(r) + L2 * b
        haa = math.fsum(wi * zi * zi for wi, zi in zip(w, z)) + L2
        hab = math.fsum(wi * zi for wi, zi in zip(w, z))
        hbb = math.fsum(w) + L2
        det = haa * hbb - hab * hab
        if not det > 0:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        step = 1.0
        new = loss(a - da, b - db)
        while new > cur and step > 1e-10:
            step /= 2
            new = loss(a - step * da, b - step * db)
        if new > cur:
            break
        a, b, cur = a - step * da, b - step * db, new
        if max(abs(step * da), abs(step * db)) < PLATT_TOL:
            break
    return {"a": a, "b": b}


def _golden(f, lo, hi):
    """The minimizer of `f` on [lo, hi] by golden-section search (GOLDEN_ITERATIONS iterations; a tie goes left)."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(GOLDEN_ITERATIONS):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _temperature(p, y):
    n = float(len(y))
    if p and isinstance(p[0], (list, tuple)):
        logs = [[math.log(max(LOG_CLIP, float(x))) for x in v] for v in p]

        def nll(log_t):
            t = math.exp(log_t)
            out = []
            for lv, k in zip(logs, y):
                z = [x / t for x in lv]
                top = max(z)
                out.append(top + math.log(math.fsum(math.exp(x - top) for x in z)) - z[k])
            return math.fsum(out) / n
    else:
        z = [logit(x) for x in p]

        def nll(log_t):
            return _binary_loss(z, y, 1.0 / math.exp(log_t), 0.0) / n
    return {"T": math.exp(_golden(nll, LOG_T_RANGE[0], LOG_T_RANGE[1]))}


def _isotonic(p, y):
    pooled = []   # [lowest p, highest p, sum of y, rows]
    for x, t in sorted(zip((float(v) for v in p), y)):
        if pooled and pooled[-1][1] == x:
            pooled[-1][2] += t
            pooled[-1][3] += 1
        else:
            pooled.append([x, x, t, 1])
    blocks = []
    for blk in pooled:
        blocks.append(blk)
        # adjacent violators (the earlier block's mean not below the later one's, compared without division) are
        # pooled; pooling equal means changes no value and keeps the points few
        while len(blocks) > 1 and blocks[-2][2] * blocks[-1][3] >= blocks[-1][2] * blocks[-2][3]:
            last = blocks.pop()
            blocks[-1] = [blocks[-1][0], last[1], blocks[-1][2] + last[2], blocks[-1][3] + last[3]]
    xs, ys = [], []
    for lo, hi, total, rows in blocks:
        mean = float(total) / rows
        xs.append(lo)
        ys.append(mean)
        if hi != lo:
            xs.append(hi)
            ys.append(mean)
    return {"x": xs, "y": ys}


def fit(method, p, y):
    """The parameters of calibration `method` fitted on the scores (or probability vectors) p and labels y: {} for
    none, {"a", "b"} for platt, {"T"} for temperature, {"x", "y"} for isotonic. ValueError for an unknown method, a
    binary-only method on probability vectors, or no rows."""
    if method not in METHODS:
        raise ValueError("unknown calibration %r (one of %s)" % (method, ", ".join(METHODS)))
    if method == "none":
        return {}
    if not p or len(p) != len(y):
        raise ValueError("%s needs rows: %d scores, %d labels" % (method, len(p), len(y)))
    if method == "temperature":
        return _temperature(p, y)
    if isinstance(p[0], (list, tuple)):
        raise ValueError("calibration %s is for binary tasks only" % method)
    return _platt(p, y) if method == "platt" else _isotonic(p, y)


def apply(method, params, p):
    """The calibrated scores (binary: floats) or probability vectors of p."""
    return [calibrated_vector(method, params, x) if isinstance(x, (list, tuple)) else calibrated_p(method, params, x)
            for x in p]
