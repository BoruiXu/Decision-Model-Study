"""Reference tabular baselines (LR, RF, XGB) on the address-level SQO dataset.

Same samples, split and features as the LLM experiments: fit on train (steps 1-29),
score val and test, write results/addr/<model>_{val,test}.parquet for eval_decisions.py.
Usage: python scripts/baseline_addr.py [data_dir] [results_dir]
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/elliptic_pp_addr_sqo")
RES = Path(sys.argv[2] if len(sys.argv) > 2 else "results/addr")
RES.mkdir(parents=True, exist_ok=True)
N_JOBS = 16  # shared machine


def load(split):
    df = pd.read_json(DATA / f"{split}.jsonl", lines=True)
    X = pd.DataFrame(df.state.tolist())
    return df[["id", "y", "post43", "seen_in_train"]], X


meta = {s: load(s) for s in ["train", "val", "test"]}
models = {
    "LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=2000)),
    "RF": RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=2,
                                 n_jobs=N_JOBS, random_state=0),
    "XGB": XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.5, n_jobs=N_JOBS, tree_method="hist", random_state=0),
}
(m_tr, X_tr) = meta["train"]
for name, m in models.items():
    m.fit(X_tr.values, m_tr.y.values)
    for s in ["val", "test"]:
        df, X = meta[s]
        df.assign(p=m.predict_proba(X.values)[:, 1]).to_parquet(RES / f"{name}_{s}.parquet", index=False)
    print("done", name, flush=True)
