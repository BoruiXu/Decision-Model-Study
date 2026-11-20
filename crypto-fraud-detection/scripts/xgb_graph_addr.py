"""XGB-Graph (GADBench, Tang et al. NeurIPS 2023, github.com/squareRoot3/GADBench, models/detector.py XGBGraphDetector)
on the address-level SQO dataset: XGBoost on [x, mean-neighbour(x), mean-neighbour^2(x)].

GADBench builds the extra features with a parameter-free GIN (GIN_noparam: init_eps = -1, aggregator 'mean',
2 layers), i.e. h_k = mean of the neighbours' h_{k-1}, and concatenates [h_0, h_1, h_2]. Here the graph is the same
per-step snapshot used by gnn_addr.py (graph_snapshots.pt: nodes = all addresses of that step, unknown labels
included as features only; edges = input -> output address of that step's transactions, undirected, no self loops),
so aggregation never crosses time steps and never sees labels. Node features = the snapshot's signed-log1p wallet
features. XGB hyper-parameters = baseline_addr.py's XGB (no class reweighting, like our other baselines;
GADBench reweights).
  static       : fit on steps 1-29 -> results/addr/XGBG_{val,test}.parquet
  --roll refit : before test step t, refit on steps <= t-delay (as rolling_xgb_addr.py) -> XGBG_roll_d1_test
Usage: python scripts/xgb_graph_addr.py [--roll refit]
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--roll", choices=["none", "refit"], default="none")
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--delay", type=int, default=1)
ap.add_argument("--data", default="data/processed/elliptic_pp_addr_sqo")
ap.add_argument("--results", default="results/addr")
ap.add_argument("--cache", default="data/processed/elliptic_pp_addr_sqo/graph_snapshots.pt")
ap.add_argument("--n_jobs", type=int, default=16)
a = ap.parse_args()
D, RES = Path(a.data), Path(a.results)
name = "XGBG" if a.roll == "none" else f"XGBG_roll_d{a.delay}"
if (RES / f"{name}_test.parquet").exists():
    raise SystemExit(f"{RES / f'{name}_test.parquet'} exists, refusing to overwrite")

G = torch.load(a.cache, weights_only=False)   # built by gnn_addr.py
rows = []
for t in sorted(G):
    x = G[t]["x"].numpy().astype(np.float32)
    n = len(x)
    ei = G[t]["ei"].numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1], np.float32), (ei[1], ei[0])), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    M = sp.diags(np.where(deg > 0, 1 / np.maximum(deg, 1), 0).astype(np.float32)) @ A   # row-normalized: mean
    hs, h = [x], x
    for _ in range(a.layers):
        h = M @ h
        hs.append(h)
    lab = G[t]["y"].numpy() >= 0
    f = np.concatenate(hs, 1)[lab]
    rows.append(pd.DataFrame(f).assign(id=G[t]["ids"][lab], step=t))
F = pd.concat(rows, ignore_index=True)
feat_cols = [c for c in F.columns if c not in ("id", "step")]
meta = pd.concat([pd.read_json(D / f"{s}.jsonl", lines=True)[["id", "y", "post43", "seen_in_train"]].assign(split=s)
                  for s in ["train", "val", "test"]])
F = meta.merge(F, on="id", how="left")
assert F[feat_cols[0]].notna().all(), "some samples missing from the snapshots"
print(f"samples {len(F):,}, features {len(feat_cols)}", flush=True)


def xgb():
    return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                         n_jobs=a.n_jobs, tree_method="hist", random_state=0)


def report(tag, df):
    print(f"{tag}: AUPRC {average_precision_score(df.y, df.p):.4f} AUROC {roc_auc_score(df.y, df.p):.4f}", flush=True)
    if "post43" in df and df.post43.any():
        for k, mk in [("pre43", ~df.post43), ("post43", df.post43)]:
            print(f"  {k}: AUPRC {average_precision_score(df.y[mk], df.p[mk]):.4f}", flush=True)


keep = ["id", "y", "post43", "seen_in_train"]
t0 = time.time()
if a.roll == "none":
    tr = F.split == "train"
    m = xgb().fit(F.loc[tr, feat_cols].values, F.y[tr].values)
    for s in ["val", "test"]:
        d = F[F.split == s]
        out = d[keep].assign(p=m.predict_proba(d[feat_cols].values)[:, 1])
        out.to_parquet(RES / f"{name}_{s}.parquet", index=False)
        report(f"{name} {s}", out)
else:
    parts = []
    for t in range(35, 50):
        tr, te = (F.step <= t - a.delay).values, (F.step == t).values
        m = xgb().fit(F.loc[tr, feat_cols].values, F.y[tr].values)
        parts.append(F.loc[te, keep].assign(p=m.predict_proba(F.loc[te, feat_cols].values)[:, 1]))
        print(f"step {t}: train rows {tr.sum():,} {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(parts)
    out.to_parquet(RES / f"{name}_test.parquet", index=False)
    shutil.copy(RES / "XGBG_val.parquet", RES / f"{name}_val.parquet")
    report(f"{name} test", out)
