"""Rolling (incrementally retrained) XGBoost on the address-level SQO dataset.

For each test step t (35-49), refit on all labeled rows with step <= t - delay and score step t.
delay models label latency (1 step ~ 2 weeks). Writes results/addr/XGB_roll_d{delay}_test.parquet;
the val file is the static model's val predictions so eval_decisions.py picks its threshold on val.
Usage: python scripts/rolling_xgb_addr.py [data_dir] [results_dir]
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/elliptic_pp_addr_sqo")
RES = Path(sys.argv[2] if len(sys.argv) > 2 else "results/addr")
N_JOBS = 16  # shared machine

df = pd.concat([pd.read_json(DATA / f"{s}.jsonl", lines=True) for s in ["train", "val", "test"]])
X = pd.DataFrame(df.state.tolist(), index=df.index)
df = df.reset_index(drop=True)
X = X.reset_index(drop=True)


def xgb():
    return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.5, n_jobs=N_JOBS, tree_method="hist", random_state=0)


for delay in [1, 3]:
    parts = []
    for t in range(35, 50):
        tr, te = df.step <= t - delay, df.step == t
        m = xgb().fit(X[tr].values, df.y[tr].values)
        parts.append(df.loc[te, ["id", "y", "post43", "step"]].assign(
            seen_in_train=df.loc[te, "address"].isin(set(df.loc[tr, "address"])).values,
            p=m.predict_proba(X[te].values)[:, 1]))
        print(f"delay {delay} step {t}: train rows {tr.sum():,}", flush=True)
    name = f"XGB_roll_d{delay}"
    pd.concat(parts).to_parquet(RES / f"{name}_test.parquet", index=False)
    shutil.copy(RES / "XGB_val.parquet", RES / f"{name}_val.parquet")
