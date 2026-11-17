import json
import os
from pathlib import Path

import pytest

from harness.baselines.tfidf import TextBaseline, group_prior
from harness.evaluator.text import text_of


def _ex(i, text, rate, group="g"):
    return dict(id=str(i), text=text, rate=rate, group=group)


def test_tfidf_small_train_uses_10():
    train = [_ex(i, f"word{i % 3} common text {i}", (i % 2) * 1.0) for i in range(20)]
    model = TextBaseline(train)
    assert model.strength == 10.0
    preds = model.predict(train[:3])
    assert len(preds) == 3 and all(0 <= p <= 1 for p in preds)


def test_group_prior():
    train = [_ex(0, "a", 1., "x"), _ex(1, "b", 0., "x"), _ex(2, "c", 1., "y")]
    rows = [_ex(3, "d", 0., "x"), _ex(4, "e", 0., "z")]
    assert group_prior(train, rows) == [.5, pytest.approx(2 / 3)]


V1 = os.environ.get("HARNESS_V1_DIR")


def _examples(path, ids=None):
    out = []
    for line in Path(path).read_text().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if ids is not None and row["id"] not in ids:
            continue
        out.append(dict(id=row["id"], text=text_of(row, "messages"), rate=row["target"]["yes"],
                        group=row["meta"]["category"]))
    return out


@pytest.mark.skipif(not V1, reason="HARNESS_V1_DIR not set")
def test_tfidf_matches_v1():
    root = Path(V1)
    binary = root / "full/export_majority75/binary"
    model = TextBaseline(_examples(binary / "train.jsonl"))
    assert model.strength == 1.0
    for name in ("val_id", "val_catood"):
        lines = (root / f"full/check_majority75_preds_{name}.jsonl").read_text().split("\n")
        old = {r["id"]: r["tfidf"] for r in map(json.loads, filter(None, lines))}
        ex = _examples(binary / f"{name}.jsonl", set(old))
        assert ex
        got = model.predict(ex)
        assert max(abs(old[e["id"]] - p) for e, p in zip(ex, got)) < 1e-6
