"""The one comparison metric trsi computes itself: ECE. Every other number (auprc, brier, log_loss, ...) is the
candidate's metrics_val, computed by `rsi score`.

- ece: 15 equal-width bins on p = P(positive), as the Elliptic team's eval_decisions.py.
- Directions of the metrics a guard may name: guards are lower-is-better (a guard limits how much worse it gets).
- brier, for the record: `rsi score`'s binary Brier, mean of (p_pos - y)^2 with y the hard label (0/1) — the same
  definition the guard tolerance 0.005 was set with. OneJev's qev/calibrate.py sums over both classes, which is twice
  this value on hard targets.
"""
from __future__ import annotations

from typing import Sequence

LOWER_IS_BETTER = ("ece", "brier", "log_loss")
ECE_BINS = 15


def ece(y: Sequence[int], p: Sequence[float], bins: int = ECE_BINS) -> float:
    if len(y) != len(p) or not y:
        raise ValueError(f"y and p must be non-empty and of equal length (got {len(y)} and {len(p)})")
    count = [0] * bins
    sum_y = [0.0] * bins
    sum_p = [0.0] * bins
    for yi, pi in zip(y, p):
        b = min(int(pi * bins), bins - 1)
        count[b] += 1
        sum_y[b] += yi
        sum_p[b] += pi
    n = len(y)
    return sum(abs(sum_y[b] - sum_p[b]) / n for b in range(bins) if count[b])
