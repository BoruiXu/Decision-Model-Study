import math
from types import SimpleNamespace

import pytest

from harness.eval_records import nll, records


def sc(idx, labels, logits, target):
    return SimpleNamespace(idx=idx, labels=labels, logits=logits, target=target)


ROWS = [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def test_records_softmax():
    out = records([sc(1, ["no", "yes"], [0.0, math.log(3)], [0.0, 1.0])], ROWS)
    assert out[0]["id"] == "b"
    assert out[0]["labels"] == ["no", "yes"]
    assert out[0]["probs"] == pytest.approx([0.25, 0.75])
    assert out[0]["target"] == [0.0, 1.0]


def test_records_softmax_stable_for_large_logits():
    out = records([sc(0, ["no", "yes"], [1000.0, 1001.0], [1.0, 0.0])], ROWS)
    assert sum(out[0]["probs"]) == pytest.approx(1.0)


def test_nll_matches_v1_binary():
    items = [(0.8, 1.0), (0.3, 0.0), (0.6, 0.25)]
    recs = [{"probs": [1 - p, p], "target": [1 - y, y]} for p, y in items]
    v1 = -sum(y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12)) for p, y in items) / len(items)
    assert nll(recs) == pytest.approx(v1)


def test_nll_clamps_zero_probability():
    assert nll([{"probs": [0.0, 1.0], "target": [1.0, 0.0]}]) == pytest.approx(-math.log(1e-12))


def test_records_preserve_order_and_ids():
    scored = [sc(2, ["x", "y"], [0, 0], [1, 0]), sc(0, ["x", "y"], [0, 0], [0, 1])]
    assert [r["id"] for r in records(scored, ROWS)] == ["c", "a"]


def test_records_sorted_and_complete():
    from harness.eval_records import merge_sorted

    parts = [[{"id": "c"}, {"id": "a"}], [{"id": "b"}]]
    assert [r["id"] for r in merge_sorted(parts, ROWS, "select")] == ["a", "b", "c"]
    with pytest.raises(RuntimeError, match=r"select.*2 of 3.*1 missing"):
        merge_sorted([[{"id": "a"}, {"id": "b"}]], ROWS, "select")
    with pytest.raises(RuntimeError, match="select"):
        merge_sorted([[{"id": "a"}, {"id": "a"}, {"id": "b"}]], ROWS, "select")
