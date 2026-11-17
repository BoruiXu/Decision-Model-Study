"""Dependency-free metrics ported verbatim from v1 (tie rules and tolerances unchanged).

Only `category` is re-keyed to `group` and the functions take explicit arguments.
"""
import math
import statistics


# Provenance: dynamicserve-agreement-router/data_pipeline_v2/sanity_check/common.py `auc`.
def auc(scores, rates):
    """Probability that an agreeing pair's question is scored above a disagreeing pair's question."""
    wins = total = 0.
    for score_a, rate_a in zip(scores, rates):
        for score_b, rate_b in zip(scores, rates):
            weight = rate_a * (1 - rate_b)
            if weight:
                total += weight
                wins += weight * ((score_a > score_b) + .5 * (score_a == score_b))
    return wins / total if total else None


# Provenance: common.py `within_category_auc` (examples/scores -> parallel groups/scores/rates lists).
def within_group_auc(groups, scores, rates):
    """Question-weighted mean AUC inside each group, so a group prior alone scores 0.5."""
    buckets = {}
    for group, score, rate in zip(groups, scores, rates):
        buckets.setdefault(group, []).append((score, rate))
    weighted = count = 0
    for bucket in buckets.values():
        value = auc([score for score, _ in bucket], [rate for _, rate in bucket])
        if value is not None:
            weighted += value * len(bucket)
            count += len(bucket)
    return weighted / count if count else None


# Provenance: evaluation/binary_metrics.py `aggregate` (legacy x16 pair scaling kept).
def aggregate(probabilities, agreements, threshold):
    assert len(probabilities) == len(agreements) and probabilities
    route = [p >= threshold for p in probabilities]
    tp = 16 * sum(y for r, y in zip(route, agreements) if r)
    fp = 16 * sum(1 - y for r, y in zip(route, agreements) if r)
    fn = 16 * sum(y for r, y in zip(route, agreements) if not r)
    union = tp + fp + fn
    return dict(jaccard=tp / union if union else 1., threshold=threshold,
                preservation=1 - fp / (16 * len(route)), small_usage=sum(route) / len(route),
                small_selected=sum(route), questions=len(route), tp_pairs=tp, fp_pairs=fp, fn_pairs=fn)


# Provenance: evaluation/binary_metrics.py `fit_threshold`.
def fit_threshold(probabilities, agreements):
    """Choose a validation threshold; equal scores cannot be split by labels."""
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
    points = [aggregate(probabilities, agreements, t) for t in sorted(set(probabilities) | {0., .5, 1.1})]
    # Fixed tie rule: nearest to .5, then larger threshold.
    return max(points, key=lambda x: (x['jaccard'], -abs(x['threshold'] - .5), x['threshold']))


# Provenance: regen_20261111/train/gate.py `max_share_at_budget`.
def max_share_at_budget(scores, incons, budget):
    """Same rule as stage 4 (stage4_majority75.py): descending score, ties interpolated, longest feasible prefix."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    groups, cur, last = [], [], None
    for i in order:
        if cur and scores[i] != last:
            groups.append(cur)
            cur = []
        cur.append(i)
        last = scores[i]
    if cur:
        groups.append(cur)
    n_r, s_r, best = 0, 0.0, 0.0
    for g in groups:
        sg, ng = sum(incons[i] for i in g), len(g)
        if (s_r + sg) / (n_r + ng) <= budget + 1e-12:
            best = max(best, n_r + ng)
        elif n_r and s_r / n_r <= budget + 1e-12:
            t = (budget * n_r - s_r) / (sg - budget * ng)
            best = max(best, n_r + max(0.0, min(1.0, t)) * ng)
        n_r, s_r = n_r + ng, s_r + sg
    return best / len(scores)


# Provenance: gate.py `budget_threshold`.
def budget_threshold(scores, incons, budget):
    """Deployable threshold: smallest cutoff t (largest small share) with mean(1-f | score >= t) <= budget."""
    best = None
    for t in sorted(set(scores)):
        routed = [i for i, s in enumerate(scores) if s >= t]
        m = statistics.fmean(incons[i] for i in routed)
        if m <= budget + 1e-12:
            best = dict(threshold=t, small_share=len(routed) / len(scores), inconsistency=m)
            break
    return best or dict(threshold=1.1, small_share=0.0, inconsistency=None)


# Provenance: gate.py `at_threshold`.
def at_threshold(scores, labels, incons, t):
    routed = [i for i, s in enumerate(scores) if s >= t]
    tp = sum(labels[i] for i in routed)
    fp = len(routed) - tp
    fn = sum(labels) - tp
    return dict(small_share=len(routed) / len(scores), jaccard=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                inconsistency=statistics.fmean(incons[i] for i in routed) if routed else None)


def ece(probs, rates, bins=10):
    """Expected calibration error: equal-width bins on p, weighted |mean p - mean rate|."""
    buckets = [[] for _ in range(bins)]
    for p, r in zip(probs, rates):
        buckets[min(int(p * bins), bins - 1)].append((p, r))
    n = len(probs)
    return sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(r for _, r in b) / len(b)) for b in buckets if b)
