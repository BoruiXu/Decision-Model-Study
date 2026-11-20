"""Control for the transaction-history state: XGB on exactly what Elephant sees in data/onejev/addr_txhist_noul,
flattened to a table (summary fields; the newest --k transactions in listed order, padded with NaN; min / max / mean
of each transaction field over the listed transactions). Same XGB settings as scripts/baseline_addr.py.
Shows whether the information in the new state still separates the classes, independently of the LLM.
Usage: python scripts/xgb_txhist.py [--data data/onejev/addr_txhist_noul] [--out results/addr_txhist]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/onejev/addr_txhist_noul")
ap.add_argument("--out", default="results/addr_txhist")
ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ROLES = ["sent", "received", "both"]


def flat(r):
    s, txs = r["state"]["summary"], r["state"]["transactions"]
    f = dict(s)
    fields = [k for k in txs[0] if k != "role"]
    for i in range(a.k):
        t = txs[i] if i < len(txs) else None
        for k in fields:
            f[f"tx{i}_{k}"] = t[k] if t else np.nan
        for ro in ROLES:
            f[f"tx{i}_role_{ro}"] = (t["role"] == ro) if t else np.nan
    for k in fields:
        v = np.array([t[k] for t in txs], dtype=float)
        f[f"min_{k}"], f[f"max_{k}"], f[f"mean_{k}"] = v.min(), v.max(), v.mean()
    return f


def load(split):
    rows = [json.loads(l) for l in open(Path(a.data) / f"{split}.jsonl")]
    X = pd.DataFrame([flat(r) for r in rows]).astype(float)
    meta = pd.DataFrame([{"id": r["id"], **r["meta"]} for r in rows])
    return X, meta


Xtr, mtr = load("train")
m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                  n_jobs=16, tree_method="hist", random_state=0).fit(Xtr, mtr.y)
for split in ["val", "test"]:
    X, meta = load(split)
    meta["p"] = m.predict_proba(X[Xtr.columns])[:, 1]
    meta.drop(columns="state_id").to_parquet(out / f"XGB_txhist_{split}.parquet", index=False)
    print(f"XGB_txhist {split}: AUPRC {average_precision_score(meta.y, meta.p):.4f}  "
          f"AUROC {roc_auc_score(meta.y, meta.p):.4f}  (n {len(meta):,}, {Xtr.shape[1]} features)", flush=True)
