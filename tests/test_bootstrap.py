import random

import numpy as np
import pytest

from harness.config import EvaluatorSpec, FieldsSpec
from harness.evaluator.bootstrap import (bootstrap_cis, draw, fast_auc, fast_share_at_budget,
                                         fast_within_group_auc)
from harness.evaluator.metrics import auc, max_share_at_budget, within_group_auc


def data(n, seed, ties=True):
    rng = random.Random(seed)
    scores = [round(rng.random(), 1 if ties else 6) for _ in range(n)]
    rates = [rng.choice([0.0, 1.0, rng.random()]) for _ in range(n)]
    groups = [f"g{rng.randrange(5)}" for _ in range(n)]
    return scores, rates, groups


@pytest.mark.parametrize("seed", range(6))
def test_fast_metrics_match_the_v1_ports(seed):
    s, r, g = data(80, seed)
    assert fast_auc(np.array(s), np.array(r)) == pytest.approx(auc(s, r), abs=1e-12)
    assert fast_within_group_auc(np.array(g), np.array(s), np.array(r)) == pytest.approx(
        within_group_auc(g, s, r), abs=1e-12)
    incons = [1 - x for x in r]
    for b in (0.05, 0.1, 0.3):
        assert fast_share_at_budget(np.array(s), np.array(incons), b) == pytest.approx(
            max_share_at_budget(s, incons, b), abs=1e-12)


def test_fast_auc_undefined_when_one_class():
    assert fast_auc(np.array([0.1, 0.2]), np.array([1.0, 1.0])) is None
    assert fast_within_group_auc(np.array(["a", "b"]), np.array([0.1, 0.2]), np.array([1.0, 0.0])) is None


def test_draw_is_two_level():
    groups = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2)
    members = [np.flatnonzero(groups == x) for x in ("a", "b", "c")]
    rng = np.random.default_rng(0)
    for _ in range(20):
        idx, clusters = draw(members, len(groups), rng)
        for j in np.unique(clusters):
            picked = idx[clusters == j]
            assert len(set(groups[picked])) == 1  # every drawn cluster comes from one group
            assert len(picked) == (groups == groups[picked[0]]).sum()  # with its own size
        assert len(np.unique(clusters)) == 3  # as many draws as groups


def test_draw_item_level():
    idx, clusters = draw(None, 7, np.random.default_rng(1))
    assert len(idx) == 7 and clusters is None


def examples(n=120, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        rate = rng.random()
        out.append({"id": str(i), "rate": rate, "f": 0.5 + rate / 2, "group": f"g{i % 6}"})
    return out


def test_bootstrap_cis_are_deterministic_and_bracket_the_estimate():
    ex = examples()
    scores = {"router": [e["rate"] + random.Random(e["id"]).uniform(-.3, .3) for e in ex],
              "oracle": [e["f"] for e in ex]}
    one = bootstrap_cis(ex, scores, budgets=[0.1], has_group=True, has_f=True, reps=200, seed=3)
    two = bootstrap_cis(ex, scores, budgets=[0.1], has_group=True, has_f=True, reps=200, seed=3)
    assert one == two
    lo, hi = one["router"]["auc"]
    assert lo < auc(scores["router"], [e["rate"] for e in ex]) < hi and hi - lo < 0.3
    assert set(one["router"]) == {"auc", "within_group_auc", "share_at_budget_10"}
    other = bootstrap_cis(ex, scores, budgets=[0.1], has_group=True, has_f=True, reps=200, seed=4)
    assert other != one


def test_bootstrap_without_groups_is_item_level():
    ex = examples()
    out = bootstrap_cis(ex, {"router": [e["rate"] for e in ex]}, budgets=None, has_group=False, has_f=False,
                        reps=50, seed=0)
    assert set(out["router"]) == {"auc"}


def test_evaluate_reports_cis_and_the_level():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_binary_decision import setup as make_inputs  # noqa: E402
    from harness.evaluator.binary_decision import evaluate
    preds, parts, train, fields = make_inputs()
    cfg = EvaluatorSpec(bootstrap_reps=100, bootstrap_seed=1)
    res = evaluate(preds, parts, train, fields, cfg)
    assert res["bootstrap"] == {"reps": 100, "seed": 1, "level": "group-then-item", "ci": 0.95}
    pooled = res["sets"]["pooled_select"]
    assert set(pooled["router"]["ci95"]) >= {"auc", "within_group_auc"}
    flat = FieldsSpec(label=fields.label, consistency=fields.consistency, text=fields.text)
    res = evaluate(preds, parts, train, flat, cfg)
    assert res["bootstrap"]["level"] == "item" and "no group field" in res["bootstrap"]["note"]
    res = evaluate(preds, parts, train, fields, EvaluatorSpec(bootstrap_reps=0))
    assert res["bootstrap"] is None and "ci95" not in res["sets"]["pooled_select"]["router"]
