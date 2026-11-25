"""Comparison rules (scheme v1 of the Elliptic training-RSI work):

1. probe: train the candidate's probe seed (the first of rules.seeds) and compare with the incumbent's same seed;
   more than probe_margin higher -> train the other seeds, otherwise reject.
2. decide: with every seed done, accept only if the mean gain over the incumbent (same seeds) exceeds accept_margin
   and each guard metric (lower is better) gets worse by no more than its tolerance.
The margins guard against seed noise: on Elliptic the seed-to-seed std of the incumbent was 0.0175 AUPRC and about 90%
of the difference between seeds came from training itself, so a row bootstrap on one seed would understate it.
"""
from __future__ import annotations

from .config import MetricSpec, Rules
from .harness import TrialResult


def gain(cand: TrialResult, inc: TrialResult, metric: str) -> float:
    return cand.metrics[metric] - inc.metrics[metric]


def after_probe(cand: TrialResult, inc: TrialResult, rules: Rules, metric: MetricSpec) -> tuple[str, float]:
    """("more_seeds" | "reject", gain of the probe seed)."""
    if cand.seed != inc.seed:
        raise ValueError(f"probe compares the same seed (candidate {cand.seed}, incumbent {inc.seed})")
    g = gain(cand, inc, metric.primary)
    return ("more_seeds" if g > rules.probe_margin else "reject"), g


def decide(cand: dict[int, TrialResult], inc: dict[int, TrialResult], rules: Rules,
           metric: MetricSpec) -> tuple[bool, str]:
    """Accept or not, with the numbers behind it. Both sides need every seed in rules.seeds."""
    missing = [s for s in rules.seeds if s not in cand or s not in inc]
    if missing:
        raise ValueError(f"seeds {missing} are missing on one side")
    n = len(rules.seeds)
    m = metric.primary
    gains = [gain(cand[s], inc[s], m) for s in rules.seeds]
    mean_gain = sum(gains) / n
    higher = sum(g > 0 for g in gains)
    parts = [f"mean {m} gain {mean_gain:+.4f} over {n} seeds ({higher}/{n} higher; "
             + ", ".join(f"s{s} {g:+.4f}" for s, g in zip(rules.seeds, gains)) + ")"]
    ok = mean_gain > rules.accept_margin
    parts.append(f"{'>' if ok else '<='} accept margin {rules.accept_margin}")
    for gm, tol in metric.guards.items():
        worse = sum(cand[s].metrics[gm] - inc[s].metrics[gm] for s in rules.seeds) / n
        g_ok = worse <= tol
        ok = ok and g_ok
        parts.append(f"{gm} {worse:+.4f} ({'within' if g_ok else 'beyond'} {tol})")
    return ok, "; ".join(parts)
