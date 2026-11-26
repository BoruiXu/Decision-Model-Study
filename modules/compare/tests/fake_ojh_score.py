"""A stand-in for onejev-harness's harness.score.score_checkpoint: model.predict (trsi.pack.PREDICT_PY) loads it from
the test_scorer of its predict.json, which compare.gpu.score_base writes from COMPARE_TEST_SCORER (tests only).

The model directory must be there (score_base links the base checkpoint as model/). Each row's probability of yes
is a logistic function of its state's `score`, a little higher with few-shot examples in the instructions, so the
zero-shot and few-shot baselines differ."""
import math
import os


def score_checkpoint(model_dir, rows, *, base_dir=None, device="cuda:0", **kw):
    if not os.path.isdir(model_dir):
        raise RuntimeError("fake scorer: no model directory %s" % model_dir)
    out = {}
    for r in rows:
        shots = r["question"]["instructions"].count("Answer: ")
        p = 1 / (1 + math.exp(-(2.0 + 0.1 * shots) * (float(r["state"]["score"]) - 0.7)))
        out[r["id"]] = {"id": r["id"], "labels": ["yes", "no"], "probs": [p, 1 - p],
                        "target": [float(r["target"].get(lab, 0)) for lab in ("yes", "no")]}
    return out
