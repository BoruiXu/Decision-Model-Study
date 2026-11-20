"""Rolling recalibration of a frozen (untrained) model: the "rolling" setting for zero-shot Elephant-1.

The model weights never change; only a 2-parameter Platt map p' = sigmoid(a * z + b) on the model's logit
z = logit(p) and the F1 decision threshold are refit before each test step t from the labels already available
(steps t-delay-window+1 .. t-delay, labels of every split; the training steps count too, since nothing was trained on
them). The slope a is unconstrained: with a > 0 the ranking inside a step is unchanged (per-step AUROC/AUPRC equal
the raw ones; pooled AUPRC, Brier, ECE and thresholded P/R/F1 change); a fitted a < 0 means the recent labels say the
model ranks backwards, and the map flips it (seen with zero-shot Kev: pooled test AUPRC 0.040 -> 0.096). Report a
per step. CPU only.
Inputs: <results>/<name>_{val,test}.parquet (elephant/scripts/onejev_score.py), the SQO jsonl for the labels of steps 1-29
(needs a scored train file <name>_train.parquet if --window reaches into steps 1-29; otherwise val/test suffice).
Output: <results>/<name>_recal_roll_d<delay>_test.parquet with p (recalibrated), p_raw, thr (per-step F1 threshold
chosen on the fitting window); <name>_recal_roll_d<delay>_val.parquet = the val file with p unchanged (for
eval_decisions.py, whose val-chosen threshold then only serves as a reference).
Usage: python elephant/scripts/rolling_recalib.py --results results/elephant --name EL0_json_choice --window 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--delay", type=int, default=1)
ap.add_argument("--window", type=int, default=5, help="number of most recent labeled steps used for the fit")
a = ap.parse_args()
R = Path(a.results)
parts = [pd.read_parquet(R / f"{a.name}_{s}.parquet") for s in ["train", "val", "test"]
         if (R / f"{a.name}_{s}.parquet").exists()]
allp = pd.concat(parts, ignore_index=True)
if "step" not in allp:
    allp["step"] = allp.id.str.split("_").str[1].astype(int)
eps = 1e-12
allp["z"] = np.log(allp.p.clip(eps, 1 - eps)) - np.log1p(-allp.p.clip(eps, 1 - eps))
test = pd.read_parquet(R / f"{a.name}_test.parquet")
if "step" not in test:
    test["step"] = test.id.str.split("_").str[1].astype(int)

out = []
for t in sorted(test.step.unique()):
    hi = t - a.delay
    fit = allp[(allp.step <= hi) & (allp.step > hi - a.window)]
    assert len(fit) and fit.y.nunique() == 2, f"step {t}: fitting window has {len(fit)} rows / one class"
    lr = LogisticRegression(C=1e6).fit(fit[["z"]], fit.y)
    pf = lr.predict_proba(fit[["z"]])[:, 1]
    prec, rec, thr = precision_recall_curve(fit.y, pf)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    th = float(thr[np.argmax(f1[:-1])])
    g = test[test.step == t].copy()
    zt = np.log(g.p.clip(eps, 1 - eps)) - np.log1p(-g.p.clip(eps, 1 - eps))
    g["p_raw"] = g.p
    g["p"] = lr.predict_proba(pd.DataFrame({"z": zt}))[:, 1]
    g["thr"] = th
    g["platt_a"], g["platt_b"] = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    out.append(g)
    print(f"step {t}: fit steps {max(1, hi - a.window + 1)}-{hi} ({len(fit):,} rows), a {lr.coef_[0, 0]:.3f} "
          f"b {lr.intercept_[0]:.3f}, thr {th:.3f}", flush=True)
res = pd.concat(out)
res.to_parquet(R / f"{a.name}_recal_roll_d{a.delay}_test.parquet", index=False)
pd.read_parquet(R / f"{a.name}_val.parquet").to_parquet(R / f"{a.name}_recal_roll_d{a.delay}_val.parquet", index=False)
f1s = []
for t, g in res.groupby("step"):
    pred = g.p >= g.thr
    tp = (pred & (g.y == 1)).sum()
    f1s.append((t, 2 * tp / max(pred.sum() + g.y.sum(), 1)))
pred = res.p >= res.thr
tp = (pred & (res.y == 1)).sum()
print(f"pooled F1 at per-step thresholds: {2 * tp / max(pred.sum() + res.y.sum(), 1):.4f}", flush=True)
