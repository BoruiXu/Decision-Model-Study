"""Bootstrap confidence intervals for the binary_decision metrics. Needs numpy; imports without torch.

Two-level resampling (design section 10): each replicate draws as many groups as there are, with
replacement, and inside each drawn group as many items as it has, with replacement. A group drawn twice
counts as two clusters (within-group AUC never pairs items across draws). Without a group field the
items are resampled directly (item level). The same draws are used for every method of a set, so the
intervals of the model and its baselines are paired.

The vectorized metrics equal the v1 ports in ``metrics.py`` (``auc``, ``within_group_auc``,
``max_share_at_budget``) up to float summation order; ``tests/test_bootstrap.py`` checks this.
"""
from __future__ import annotations

import zlib

import numpy as np

LEVEL_GROUPED = "group-then-item"
LEVEL_ITEM = "item"
CI = 0.95


def _cluster_auc(clusters, scores, rates):
    """Per cluster: (auc or nan, cluster size), in the weighted pair-counting form of ``metrics.auc``."""
    scores = np.asarray(scores, dtype=float)
    rates = np.asarray(rates, dtype=float)
    _, c = np.unique(clusters, return_inverse=True)
    order = np.lexsort((scores, c))
    c, s, r = c[order], scores[order], rates[order]
    q = 1.0 - r
    n = len(s)
    new_block = np.ones(n, dtype=bool)
    new_block[1:] = (c[1:] != c[:-1]) | (s[1:] != s[:-1])
    starts = np.flatnonzero(new_block)
    block_c = c[starts]
    rb = np.add.reduceat(r, starts)
    qb = np.add.reduceat(q, starts)
    # Q of the blocks below each block inside its cluster: cumulative Q minus the cluster's start offset
    cum = np.cumsum(qb) - qb
    first_of_cluster = np.ones(len(starts), dtype=bool)
    first_of_cluster[1:] = block_c[1:] != block_c[:-1]
    offset = np.maximum.accumulate(np.where(first_of_cluster, np.arange(len(starts)), 0))
    below = cum - cum[offset]
    k = c.max() + 1
    wins = np.bincount(block_c, weights=rb * (below + 0.5 * qb), minlength=k)
    total = np.bincount(block_c, weights=rb, minlength=k) * np.bincount(block_c, weights=qb, minlength=k)
    sizes = np.bincount(c, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(total > 0, wins / np.where(total > 0, total, 1), np.nan)
    return value, sizes


def fast_auc(scores, rates):
    v, _ = _cluster_auc(np.zeros(len(scores), dtype=int), scores, rates)
    return None if np.isnan(v[0]) else float(v[0])


def fast_within_group_auc(groups, scores, rates):
    v, sizes = _cluster_auc(groups, scores, rates)
    ok = ~np.isnan(v)
    if not ok.any():
        return None
    return float((v[ok] * sizes[ok]).sum() / sizes[ok].sum())


def fast_share_at_budget(scores, incons, budget):
    """``metrics.max_share_at_budget``: descending score, ties interpolated, longest feasible prefix."""
    scores = np.asarray(scores, dtype=float)
    incons = np.asarray(incons, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, x = scores[order], incons[order]
    n = len(s)
    new = np.ones(n, dtype=bool)
    new[1:] = s[1:] != s[:-1]
    starts = np.flatnonzero(new)
    ng = np.diff(np.append(starts, n)).astype(float)
    sg = np.add.reduceat(x, starts)
    n_prev = np.cumsum(ng) - ng
    s_prev = np.cumsum(sg) - sg
    whole = (s_prev + sg) / (n_prev + ng) <= budget + 1e-12
    best = np.where(whole, n_prev + ng, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        part_ok = ~whole & (n_prev > 0) & (s_prev / np.where(n_prev > 0, n_prev, 1) <= budget + 1e-12)
        t = (budget * n_prev - s_prev) / (sg - budget * ng)
    partial = np.where(part_ok, n_prev + np.clip(np.nan_to_num(t), 0.0, 1.0) * ng, 0.0)
    return float(max(0.0, best.max(initial=0.0), partial.max(initial=0.0)) / n)


def draw(members: list[np.ndarray] | None, n: int, rng: np.random.Generator):
    """One replicate: (item indices, cluster id per index or None for item level)."""
    if members is None:
        return rng.integers(0, n, n), None
    picks = rng.integers(0, len(members), len(members))
    idx, clusters = [], []
    for j, g in enumerate(picks):
        m = members[g]
        idx.append(m[rng.integers(0, len(m), len(m))])
        clusters.append(np.full(len(m), j))
    return np.concatenate(idx), np.concatenate(clusters)


def _pct(b) -> int:
    return round(b * 100)


def bootstrap_cis(examples: list[dict], scores: dict[str, list[float]], *, budgets, has_group: bool,
                  has_f: bool, reps: int, seed: int, salt: str = "") -> dict:
    """``{method: {"ci95": {metric: [lo, hi] or None}, "ci95_skipped": {metric: n}}}`` for AUC, within-group
    AUC (with groups) and the share at each budget (with consistency), at 95%. A replicate where a metric is
    undefined (one class only) is skipped for that metric and counted. ``salt`` (the set name) gives each
    set its own stream."""
    n = len(examples)
    rng = np.random.default_rng([seed, zlib.crc32(salt.encode())])
    rates = np.array([e["rate"] for e in examples], dtype=float)
    incons = np.array([1 - e["f"] for e in examples], dtype=float) if has_f else None
    members = None
    if has_group:
        _, gid = np.unique(np.array([str(e["group"]) for e in examples]), return_inverse=True)
        members = [np.flatnonzero(gid == k) for k in range(gid.max() + 1)]
    arrays = {m: np.asarray(v, dtype=float) for m, v in scores.items()}
    keys = ["auc"] + (["within_group_auc"] if has_group else []) + \
        ([f"share_at_budget_{_pct(b)}" for b in budgets] if has_f and budgets else [])
    values = {m: {k: [] for k in keys} for m in arrays}
    for _ in range(reps):
        idx, clusters = draw(members, n, rng)
        r = rates[idx]
        for m, sc in arrays.items():
            s = sc[idx]
            out = values[m]
            out["auc"].append(fast_auc(s, r))
            if has_group:
                out["within_group_auc"].append(fast_within_group_auc(clusters, s, r))
            if has_f and budgets:
                for b in budgets:
                    out[f"share_at_budget_{_pct(b)}"].append(fast_share_at_budget(s, incons[idx], b))
    lo_q, hi_q = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    result = {}
    for m, per in values.items():
        cis, skipped = {}, {}
        for k, vals in per.items():
            ok = [v for v in vals if v is not None]
            cis[k] = [float(np.percentile(ok, lo_q)), float(np.percentile(ok, hi_q))] if ok else None
            skipped[k] = len(vals) - len(ok)
        result[m] = {"ci95": cis, "ci95_skipped": skipped}
    return result
