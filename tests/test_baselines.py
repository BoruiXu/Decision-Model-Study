import random
import sys
from pathlib import Path

import pytest

from harness import baselines
from harness.baselines import Baseline, available, register
from harness.baselines.length import LengthBaseline
from harness.config import EvaluatorSpec, FieldsSpec
from harness.evaluator.binary_decision import evaluate
from harness.evaluator.metrics import auc
from harness.prepare import EvalParts

FAST = EvaluatorSpec(bootstrap_reps=0)


def test_registry_has_the_builtins():
    assert list(baselines.REGISTRY)[:3] == ["tfidf", "group_prior", "length"]
    names = lambda f: [b.name for b in available(f)]  # noqa: E731
    assert names(FieldsSpec(label="l", text="messages", group="g")) == ["tfidf", "group_prior", "length"]
    assert names(FieldsSpec(label="l", text="messages")) == ["tfidf", "length"]
    assert names(FieldsSpec(label="l", group="g")) == ["group_prior"]
    assert names(FieldsSpec(label="l")) == []


def test_length_baseline_learns_length():
    rng = random.Random(0)
    train = []
    for i in range(200):
        n = rng.randint(1, 60)
        train.append({"id": str(i), "text": "w " * n, "rate": min(1.0, n / 60 + rng.uniform(-.1, .1)) if n > 5 else 0.0})
    model = LengthBaseline(train)
    short, long_ = model.predict([{"text": "w"}, {"text": "w " * 80}])
    assert 0 < short < long_ < 1
    preds = model.predict(train)
    assert auc(preds, [t["rate"] for t in train]) > 0.8


def rows(n, rng, start=0):
    out = []
    for i in range(start, start + n):
        k = rng.randint(1, 40)
        rate = 1.0 if k > 20 else 0.0
        if rng.random() < 0.2:  # length predicts the label, but not perfectly
            rate = 1.0 - rate
        out.append({"id": f"r{i}", "state": {"messages": [{"role": "user", "content": "word " * k}]},
                    "target": {"yes": rate, "no": 1 - rate}, "meta": {"f": 0.5 + rate / 2, "cat": f"g{i % 2}"}})
    return out


def inputs(group=True):
    rng = random.Random(1)
    train, sel = rows(120, rng), rows(60, rng, start=1000)
    parts = EvalParts(select_rows=sel, select_index={r["id"]: "a" for r in sel}, reference={r["id"]: r for r in sel})
    fields = FieldsSpec(label="target.yes", consistency="meta.f", text="messages", group="meta.cat" if group else None)
    noisy = random.Random(5)
    preds = {r["id"]: noisy.random() for r in sel}  # a router no better than chance
    return preds, parts, train, fields


def test_gate_compares_against_the_best_baseline():
    preds, parts, train, fields = inputs()
    out = evaluate(preds, parts, train, fields, FAST)
    e = out["sets"]["a"]
    assert {"tfidf", "group_prior", "length"} <= set(e)
    assert e["length"]["auc"] > 0.7
    g = out["gate"]["a"]
    assert g["auc"] is False and g["best"]["auc"][0] in ("length", "tfidf")
    assert g["best"]["auc"][1] == max(e[k]["auc"] for k in ("tfidf", "group_prior", "length"))
    assert out["gate"]["passed"] is False


def test_router_that_beats_every_baseline_passes():
    preds, parts, train, fields = inputs()
    preds = {r["id"]: r["target"]["yes"] * 0.9 + 0.05 + int(r["id"][1:]) * 1e-6 for r in parts.select_rows}
    out = evaluate(preds, parts, train, fields, FAST)
    assert out["gate"]["a"]["auc"] is True and out["gate"]["passed"] is True


def test_registered_baseline_joins_the_comparison_and_gate():
    preds, parts, train, fields = inputs(group=False)
    preds = {r["id"]: r["target"]["yes"] * 0.9 + 0.05 + int(r["id"][1:]) * 1e-6 for r in parts.select_rows}
    perfect = Baseline("cheat", needs=(), description="knows the label",
                       fit=lambda train_ex: (lambda ex: [e["rate"] * 0.9 + 1e-4 * i for i, e in enumerate(ex)]))
    register(perfect)
    try:
        out = evaluate(preds, parts, train, fields, FAST)
        assert "cheat" in out["sets"]["a"] and out["gate"]["a"]["best"]["auc"][0] == "cheat"
    finally:
        baselines.REGISTRY.pop("cheat")


def test_no_baseline_available_still_fails_the_gate():
    preds, parts, train, fields = inputs(group=False)
    fields.text = None
    out = evaluate(preds, parts, train, fields, FAST)
    assert "reason" in out["gate"]["a"] and out["gate"]["passed"] is False
