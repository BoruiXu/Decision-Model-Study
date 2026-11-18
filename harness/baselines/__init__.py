"""Baselines as plugins: a small registry the evaluators fit and compare against (design section 10).

A baseline is ``Baseline(name, needs, fit, description)``. ``needs`` lists the example keys it uses
("text" from ``fields.text``, "group" from ``fields.group``); it is skipped when the experiment lacks
them. ``fit(train_examples)`` gets ``[{id, rate[, text][, group]}]`` (the labeled train rows) and returns
an object with ``predict(examples) -> list[float]``, or such a function itself: probabilities in [0, 1],
higher for the positive class. Register more with ``register(Baseline(...))``; the gate compares the
model against the best registered baseline available for the run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Baseline:
    name: str
    needs: tuple[str, ...]
    fit: Callable[[list[dict]], Any]
    description: str = ""


REGISTRY: dict[str, Baseline] = {}


def register(b: Baseline) -> Baseline:
    REGISTRY[b.name] = b
    return b


def _has(fields, need: str) -> bool:
    return bool(getattr(fields, need, None))


def available(fields) -> list[Baseline]:
    """Registered baselines whose inputs this experiment provides, in registration order."""
    return [b for b in REGISTRY.values() if all(_has(fields, n) for n in b.needs)]


def predictor(fitted):
    return getattr(fitted, "predict", fitted)


def _tfidf(train):
    from harness.baselines.tfidf import TextBaseline

    return TextBaseline(train)


class _GroupPrior:
    def __init__(self, train):
        self.train = train

    def predict(self, examples):
        from harness.baselines.tfidf import group_prior

        return group_prior(self.train, examples)


def _length(train):
    from harness.baselines.length import LengthBaseline

    return LengthBaseline(train)


register(Baseline("tfidf", ("text",), _tfidf, "TF-IDF words + char n-grams, logistic (v1 cheap_baseline)"))
register(Baseline("group_prior", ("group",), _GroupPrior, "mean train label of the row's group"))
register(Baseline("length", ("text",), _length, "logistic on log character length of the text"))
