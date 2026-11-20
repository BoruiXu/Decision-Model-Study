"""Rolling (refit every step) XGBoost on the transaction-history state (data/onejev/addr_txhist_noul), the
txhist counterpart of scripts/rolling_xgb_addr.py: same protocol (for each test step t = 35-49, refit on all
labeled rows with step <= t - delay, score step t), same XGB settings, and the same flattened features as
scripts/xgb_txhist.py (whose flat() it reuses). Writes results/addr_txhist/XGB_txhist_roll_d{delay}_test.parquet;
the val file is the static model's val predictions (as for XGB_roll_d1), so eval_decisions.py picks the threshold on val.
Usage: python scripts/rolling_xgb_txhist.py [--data data/onejev/addr_txhist_noul] [--out results/addr_txhist] [--delay 1]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/onejev/addr_txhist_noul")
ap.add_argument("--out", default="results/addr_txhist")
ap.add_argument("--delay", type=int, default=1)
ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ROLES = ["sent", "received", "both"]


def flat(r):  # identical to scripts/xgb_txhist.py
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


rows = [json.loads(l) for s in ["train", "val", "test"] for l in open(Path(a.data) / f"{s}.jsonl")]
X = pd.DataFrame([flat(r) for r in rows]).astype(float)
meta = pd.DataFrame([{"id": r["id"], **r["meta"]} for r in rows])

t0, parts = time.time(), []
for t in range(35, 50):
    tr, te = (meta.step <= t - a.delay).values, (meta.step == t).values
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                      n_jobs=16, tree_method="hist", random_state=0).fit(X[tr], meta.y[tr])
    parts.append(meta.loc[te, ["id", "y", "post43", "step", "seen_in_train"]].assign(p=m.predict_proba(X[te])[:, 1]))
    print(f"step {t}: train rows {tr.sum():,}  {time.time() - t0:.0f}s", flush=True)
name = f"XGB_txhist_roll_d{a.delay}"
res = pd.concat(parts)
res.to_parquet(out / f"{name}_test.parquet", index=False)
shutil.copy(out / "XGB_txhist_val.parquet", out / f"{name}_val.parquet")
for sub, mk in [("all", slice(None)), ("pre43", ~res.post43), ("post43", res.post43)]:
    d = res[mk]
    print(f"{name} test {sub}: AUPRC {average_precision_score(d.y, d.p):.4f}  AUROC {roc_auc_score(d.y, d.p):.4f}")
print(f"total {time.time() - t0:.0f}s (refits only)")
