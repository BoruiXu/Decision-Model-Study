"""A stand-in for harness.score.score_checkpoint (model.predict loads it from TRSI_TEST_SCORER).

Reads <model_dir>/fake_model.json (written by fake_ojh) and scores each row from the label written in its text
("label=yes"), with noise from the scheme key, seed and row id: higher quality separates the classes better.
It raises when FAKE_SCORER_FAIL (comma-separated) names "<scheme key>#<seed>" of the model (a broken scoring
environment, fixed by unsetting it), and prints CUDA's out-of-memory error and exits 1 when CUDA_VISIBLE_DEVICES is
FAKE_SCORER_OOM_CARD. Rows must carry a target (model.predict fills a uniform one for input rows).
Every call appends {"model_dir", "base_dir", "device", "cuda", "kw"} to FAKE_SCORER_CALLS when that is set.
FAKE_SCORER_SHIFT (a number) is subtracted from every row's logit: the ranking stays, the probabilities drop (positives
scored below one half).
"""
import json
import math
import os
import random
import sys
from pathlib import Path


def score_checkpoint(model_dir, rows, *, base_dir=None, device="cuda:0", **kw):
    m = json.loads((Path(model_dir) / "fake_model.json").read_text())
    if os.environ.get("FAKE_SCORER_CALLS"):
        with open(os.environ["FAKE_SCORER_CALLS"], "a") as f:
            f.write(json.dumps({"model_dir": str(model_dir), "base_dir": base_dir, "device": device,
                                "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "kw": kw}) + "\n")
    if f"{m['key']}#{m['seed']}" in os.environ.get("FAKE_SCORER_FAIL", "").split(","):
        raise RuntimeError("fake scorer: the checkpoint does not load")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("FAKE_SCORER_OOM_CARD", "-"):
        print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", file=sys.stderr)
        sys.exit(1)
    out = {}
    for r in rows:
        target = r["target"]
        labels = list(target)
        text = " ".join(x["content"] for x in r["state"]["messages"])
        y = 1 if "label=yes" in text else 0
        z = float(m["quality"]) * (2 * y - 1) + random.Random(f"{m['key']}|{m['seed']}|{r['id']}").gauss(0, 1)
        z -= float(os.environ.get("FAKE_SCORER_SHIFT") or 0)
        p = 1 / (1 + math.exp(-z))
        out[r["id"]] = {"id": r["id"], "labels": ["yes", "no"], "probs": [p, 1 - p],
                        "target": [float(target.get(lab, 0)) for lab in labels]}
    return out
