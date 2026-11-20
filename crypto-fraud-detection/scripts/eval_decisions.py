"""Evaluate address-level decision scores (baselines, E0, E2, rolling variants) on one protocol.

Each method writes results/addr/<method>_{val,test}.parquet with columns
id, y, p (P(illicit)), post43, seen_in_train and optionally latency_ms.
The F1 threshold is chosen on val; metrics are reported on test for these subsets:
  all, pre43, post43,
  new_addr   - address not in the method's own training data (definition differs static vs rolling),
  first_seen - the address's first labeled appearance (same rows for every method).
95% intervals: bootstrap over test rows, stratified by time step. Paired differences use the same
resamples for both methods (reference: XGB for static methods, XGB_roll_d<k> for <method>_roll_d<k>).
Usage: python scripts/eval_decisions.py [results_dir] [data_dir] [n_boot]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/addr")
DATA = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/elliptic_pp_addr_sqo")
N_BOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 500
SUBSETS = ["all", "pre43", "post43", "new_addr", "first_seen"]


# Equal-width-bin ECE.
def ece(y, p, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    return th[np.argmax(2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12))]


def metrics(y, p, thr):
    yhat = p >= thr
    out = dict(n=len(y), n_ill=int(y.sum()), P=precision_score(y, yhat, zero_division=0),
               R=recall_score(y, yhat, zero_division=0), F1=f1_score(y, yhat, zero_division=0))
    if 0 < y.sum() < len(y):
        out.update(AUPRC=average_precision_score(y, p), AUROC=roc_auc_score(y, p))
    out.update(Brier=brier_score_loss(y, p), ECE=ece(y, p))
    return out


def first_seen_ids():
    """ids of test rows that are the address's first labeled appearance (any split)."""
    ids = pd.concat([pd.read_json(DATA / f"{s}.jsonl", lines=True)[["id", "address", "step"]]
                     for s in ["train", "val", "test"]])
    first = ids.groupby("address").step.transform("min")
    return set(ids.id[(ids.step == first) & (ids.step >= 35)])


def masks(te, fs):
    return {"all": np.ones(len(te), bool), "pre43": ~te.post43.values, "post43": te.post43.values,
            "new_addr": ~te.seen_in_train.values, "first_seen": te.id.isin(fs).values}


def boot_index(steps, rng):
    """One bootstrap resample of row positions, stratified by time step."""
    return np.concatenate([rng.choice(ix, len(ix)) for ix in steps.values()])


def main():
    fs = first_seen_ids()
    methods = sorted({f.stem.rsplit("_", 1)[0] for f in RES.glob("*_test.parquet")})
    te_all = {m: pd.read_parquet(RES / f"{m}_test.parquet") for m in methods}
    ids = sorted(set.intersection(*(set(t.id) for t in te_all.values())))
    rows, preds = [], {}
    for m in methods:
        va = pd.read_parquet(RES / f"{m}_val.parquet")
        te = te_all[m].set_index("id").loc[ids].reset_index()
        te["step"] = te.id.str.split("_").str[1].astype(int)
        thr = best_threshold(va.y.values, va.p.values)
        preds[m] = te
        for subset, mk in masks(te, fs).items():
            rows.append(dict(method=m, subset=subset, thr=thr, **metrics(te.y.values[mk], te.p.values[mk], thr),
                             latency_ms=te.latency_ms.mean() if "latency_ms" in te else np.nan))
    out = pd.DataFrame(rows)

    # Bootstrap intervals and paired differences vs. the matching XGB reference.
    ref_of = {m: ("XGB" + m[m.index("_roll_d"):] if "_roll_d" in m else "XGB") for m in methods}
    rng = np.random.default_rng(0)
    base = preds[methods[0]]
    ci = []
    for subset in SUBSETS:
        mk = masks(base, fs)[subset] if subset != "new_addr" else None
        boot = {m: {"AUPRC": [], "AUROC": []} for m in methods}
        diff = {m: [] for m in methods}
        sel = {m: (masks(preds[m], fs)[subset] if subset == "new_addr" else mk) for m in methods}
        for _ in range(N_BOOT):
            # same row resample for every method; subset masks applied afterwards
            idx = boot_index(base.groupby("step").indices, rng)
            for m in methods:
                i = idx[sel[m][idx]]
                y, p = preds[m].y.values[i], preds[m].p.values[i]
                if 0 < y.sum() < len(y):
                    boot[m]["AUPRC"].append(average_precision_score(y, p))
                    boot[m]["AUROC"].append(roc_auc_score(y, p))
            for m in methods:
                r = ref_of[m]
                if r in boot and r != m and boot[m]["AUPRC"] and boot[r]["AUPRC"]:
                    diff[m].append(boot[m]["AUPRC"][-1] - boot[r]["AUPRC"][-1])
        for m in methods:
            q = lambda v: np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
            (pl, ph), (rl, rh), (dl, dh) = q(boot[m]["AUPRC"]), q(boot[m]["AUROC"]), q(diff[m])
            ci.append(dict(method=m, subset=subset, AUPRC_lo=pl, AUPRC_hi=ph, AUROC_lo=rl, AUROC_hi=rh,
                           ref=ref_of[m] if ref_of[m] != m else "", dAUPRC_lo=dl, dAUPRC_hi=dh))
    out = out.merge(pd.DataFrame(ci), on=["method", "subset"])
    out.to_csv(RES / "summary.csv", index=False)

    f = lambda x: f"{x:.3f}"
    print(f"common test rows: {len(ids):,}; bootstrap resamples: {N_BOOT}")
    for subset in SUBSETS:
        s = out[out.subset == subset].copy()
        s["AUPRC [95% CI]"] = s.apply(lambda r: f"{r.AUPRC:.3f} [{r.AUPRC_lo:.3f}, {r.AUPRC_hi:.3f}]", axis=1)
        s["AUROC [95% CI]"] = s.apply(lambda r: f"{r.AUROC:.3f} [{r.AUROC_lo:.3f}, {r.AUROC_hi:.3f}]", axis=1)
        s["dAUPRC vs ref [95% CI]"] = s.apply(
            lambda r: f"{r.ref}: [{r.dAUPRC_lo:+.3f}, {r.dAUPRC_hi:+.3f}]" if r.ref else "", axis=1)
        print(f"\n== test [{subset}]  n={int(s.n.iloc[0]):,} illicit={int(s.n_ill.iloc[0]):,}")
        print(s[["method", "F1", "AUPRC [95% CI]", "AUROC [95% CI]", "ECE", "dAUPRC vs ref [95% CI]",
                 "latency_ms"]].to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
