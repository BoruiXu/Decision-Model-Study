import json
import os
import random
from pathlib import Path

import pytest

from harness.config import (DataSpec, EvalSetSpec, EvaluatorSpec, ExperimentConfig, FieldsSpec)
from harness.evaluator.binary_decision import evaluate
from harness.prepare import EvalParts, build_eval_parts

WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]


def mk(i, rng, rate=None, group="g0", set_name=None):
    rate = rng.random() if rate is None else rate
    row = {"id": f"r{i}", "state": {"messages": [{"role": "user", "content": " ".join(rng.choices(WORDS, k=6))}]},
           "target": {"yes": rate, "no": 1 - rate}, "meta": {"f": min(1.0, 0.5 + rate / 2), "cat": group}}
    return row


def setup(n=40, seed=0, **over):
    rng = random.Random(seed)
    train = [mk(f"t{i}", rng, group=f"g{i % 3}") for i in range(80)]
    sel = [mk(i, rng, group=f"g{i % 4}") for i in range(n)]
    parts = EvalParts(select_rows=sel, select_index={r["id"]: ("a" if i % 2 else "b") for i, r in enumerate(sel)},
                      reference={r["id"]: r for r in sel})
    fields = FieldsSpec(label="target.yes", group="meta.cat", consistency="meta.f", text="messages")
    for k, v in over.items():
        setattr(fields, k, v)
    preds = {r["id"]: r["target"]["yes"] for r in sel}
    return preds, parts, train, fields


def test_full_shape():
    preds, parts, train, fields = setup()
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    assert set(out["sets"]) == {"a", "b", "pooled_select"}
    e = out["sets"]["pooled_select"]
    assert {"router", "tfidf", "group_prior", "all_positive", "oracle"} <= set(e)
    assert e["router"]["auc"] > 0.8 and "ece" in e["router"] and "ece" not in e["tfidf"]
    assert {"share_at_budget_10", "share_at_budget_15"} <= set(e["router"])
    assert e["unseen_groups"] == 1 and e["groups"] == 4
    assert set(out["thresholds"]) == {"best_jaccard", "budget_10", "budget_15"}
    assert set(out["thresholds"]["budget_10"]["per_set"]) == {"a", "b"}
    assert out["gate"]["passed"] is True and out["tfidf_strength"] == 10.0


def test_no_consistency_no_budgets():
    preds, parts, train, fields = setup(consistency=None)
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    e = out["sets"]["pooled_select"]
    assert "oracle" not in e and not any(k.startswith("share_at_budget") for k in e["router"])
    assert set(out["thresholds"]) == {"best_jaccard"}


def test_no_group_no_prior():
    preds, parts, train, fields = setup(group=None)
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    e = out["sets"]["pooled_select"]
    assert "group_prior" not in e and "within_group_auc" not in e["router"] and "groups" not in e
    assert set(out["gate"]["a"]) == {"auc"}


def test_gate_fails_when_router_worse():
    preds, parts, train, fields = setup()
    preds = {k: 1 - v for k, v in preds.items()}
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    assert out["gate"]["passed"] is False and out["gate"]["a"]["auc"] is False


def test_gate_off_reports_but_does_not_decide():
    preds, parts, train, fields = setup()
    preds = {k: 1 - v for k, v in preds.items()}
    out = evaluate(preds, parts, train, fields, EvaluatorSpec(gate=False))
    assert out["gate"]["enabled"] is False and out["gate"]["passed"] is None
    assert out["gate"]["a"]["auc"] is False
    on = evaluate(preds, parts, train, fields, EvaluatorSpec())
    assert on["gate"]["enabled"] is True and on["gate"]["passed"] is False


def test_unlabeled_rows_dropped_and_counted():
    preds, parts, train, fields = setup()
    del parts.reference["r3"]["target"]  # no target at all (a missing key alone is a 0, see below)
    del parts.reference["r4"]["meta"]["f"]
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    assert sorted(i for v in out["dropped"].values() for i in v) == ["r3", "r4"]
    assert out["sets"]["pooled_select"]["questions"] == 38


V1 = os.environ.get("HARNESS_V1_DIR")
KEYS = {"within_category_auc": "within_group_auc", "category_prior": "group_prior", "oracle_f": "oracle",
        "small_share_at_incons_10pct": "share_at_budget_10", "small_share_at_incons_15pct": "share_at_budget_15",
        "categories": "groups", "unseen_categories": "unseen_groups"}


@pytest.mark.skipif(not V1, reason="HARNESS_V1_DIR not set")
def test_matches_v1_gate():
    v1 = Path(V1)
    b = v1 / "full/export_majority75/binary"
    data = DataSpec(train=str(b / "train.jsonl"),
                    eval={"val_id": EvalSetSpec(str(b / "val_id.jsonl")),
                          "val_catood": EvalSetSpec(str(b / "val_catood.jsonl"))},
                    split_file=str(v1 / "train/splits.json"))
    fields = FieldsSpec(label="target.yes", group="meta.category", consistency="meta.f", text="messages")
    exp = ExperimentConfig(name="v1", base="b", recipe="r", data=data, fields=fields)
    parts = build_eval_parts(exp)
    train = [json.loads(l) for l in open(b / "train.jsonl")]
    preds = {}
    for n in ("val_id", "val_catood"):
        preds.update({r["id"]: r["p_agree"] for r in map(json.loads, open(v1 / f"train/predictions_select_elephant_{n}.jsonl"))})
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    want = json.loads((v1 / "train/gate_elephant.json").read_text())
    assert out["tfidf_strength"] == want["tfidf_strength"]

    def check(got, exp_, path):
        for k, v in exp_.items():
            gk = KEYS.get(k, k)
            if isinstance(v, dict):
                check(got[gk], v, f"{path}.{k}")
            else:
                assert got[gk] == pytest.approx(v, abs=1e-3), f"{path}.{k}"
    check(out["sets"], want["sets"], "sets")
    for n, d in want["pass"].items():
        assert {KEYS.get(k, k): x for k, x in d.items()} == out["gate"][n]
    assert out["gate"]["passed"] == want["passed"]
    t = out["thresholds"]
    assert t["budget_10"]["threshold"] == pytest.approx(0.75491, abs=1e-5)
    assert t["budget_15"]["threshold"] == pytest.approx(0.43782, abs=1e-5)
    assert t["best_jaccard"]["threshold"] == pytest.approx(0.33459, abs=1e-5)


def test_hard_negative_missing_positive_key_reads_zero():
    import statistics
    preds, parts, train, fields = setup()
    parts.reference["r3"]["target"] = {"no": 1.0}  # prepare accepts a subset of labels; OneJev fills 0
    train[0]["target"] = {"no": 1.0}
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    assert out["dropped"] == {}
    pooled = out["sets"]["pooled_select"]
    assert pooled["questions"] == 40
    rates = [0.0 if r["id"] == "r3" else r["target"]["yes"] for r in parts.select_rows]
    assert pooled["positive_rate"] == pytest.approx(statistics.fmean(rates))


def test_label_outside_target_missing_is_unlabeled():
    preds, parts, train, fields = setup(label="meta.rate")
    for r in parts.select_rows + train:
        r["meta"]["rate"] = r["target"]["yes"]
    del parts.reference["r3"]["meta"]["rate"]
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    assert out["dropped"] == {parts.select_index["r3"]: ["r3"]}
    assert out["sets"]["pooled_select"]["questions"] == 39


def test_skipped_train_rows_counted():
    preds, parts, train, fields = setup()
    del train[0]["meta"]["cat"]  # no group
    del train[2]["target"]  # no target dict: unlabeled
    train[1]["target"] = {"no": 1.0}  # a hard negative is kept
    out = evaluate(preds, parts, train, fields, EvaluatorSpec())
    assert out["dropped"] == {"baseline_train": ["rt0", "rt2"]}


def test_plugin_default_baselines_share_label_rules():
    import types
    from harness.plugins import evaluate_plugin
    preds, parts, train, fields = setup()
    recs = {rid: {"id": rid, "labels": ["yes", "no"], "probs": [p, 1 - p], "target": [p, 1 - p]}
            for rid, p in preds.items()}
    parts.select_rows[3]["target"] = {"no": 1.0}
    del train[0]["meta"]["cat"]
    train[1]["target"] = {"no": 1.0}
    out = evaluate_plugin(types.ModuleType("empty"), recs, parts, train, fields, EvaluatorSpec())
    assert out["dropped"] == {"baseline_train": ["rt0"]}
    assert {"tfidf", "group_prior"} <= set(out["sets"]["pooled_select"])
