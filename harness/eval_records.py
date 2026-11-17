"""Pure-python helpers for per-row eval dumps (no torch)."""
import math

FLOOR = 1e-12


def _softmax(logits):
    top = max(logits)
    exps = [math.exp(float(x) - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def records(scored, rows):
    """One dict per scored item: row id, slot labels, softmax probabilities, target distribution."""
    return [{"id": rows[sc.idx]["id"], "labels": list(sc.labels), "probs": _softmax(sc.logits),
             "target": [float(t) for t in sc.target]} for sc in scored]


def nll(recs):
    """Mean over rows of -sum_i t_i * log(max(p_i, 1e-12))."""
    total = 0.0
    for r in recs:
        total += -sum(t * math.log(max(p, FLOOR)) for p, t in zip(r["probs"], r["target"]))
    return total / len(recs)


def merge_sorted(parts, eval_rows, set_name):
    """Concatenate per-rank record lists, sort by id, require every eval row scored exactly once."""
    merged = sorted((r for part in parts for r in part), key=lambda r: r["id"])
    ids = [r["id"] for r in merged]
    wanted = {row["id"] for row in eval_rows}
    if len(ids) != len(set(ids)) or set(ids) != wanted:
        missing = len(wanted - set(ids))
        raise RuntimeError(f"eval set {set_name!r}: scored {len(wanted) - missing} of {len(wanted)} rows "
                           f"({missing} missing, {len(ids) - len(set(ids))} duplicated)")
    return merged
