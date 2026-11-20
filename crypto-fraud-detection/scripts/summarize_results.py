"""Collect test-set metrics for every prediction file under results/.

Usage: python scripts/summarize_results.py results/summary_all.csv
Each *_test.parquet must have columns y, p, post43. Writes one row per file:
AUPRC / AUROC on all test rows, pre-43 and post-43 (steps 35-42 / 43-49).
Refuses to overwrite an existing output file.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

out = Path(sys.argv[1])
assert not out.exists(), f"{out} exists, refusing to overwrite"

rows = []
for f in sorted(Path("results").rglob("*_test.parquet")):
    d = pd.read_parquet(f, columns=None)
    if not {"y", "p", "post43"} <= set(d.columns) or d["y"].nunique() < 2:
        continue
    r = {"group": str(f.parent.relative_to("results")), "method": f.name[: -len("_test.parquet")], "n": len(d)}
    for name, m in [("all", slice(None)), ("pre43", ~d["post43"].astype(bool)), ("post43", d["post43"].astype(bool))]:
        s = d.loc[m]
        if s["y"].nunique() < 2:
            continue
        r[f"AUPRC_{name}"] = average_precision_score(s["y"], s["p"])
        r[f"AUROC_{name}"] = roc_auc_score(s["y"], s["p"])
    rows.append(r)

df = pd.DataFrame(rows)
os.makedirs(out.parent, exist_ok=True)
df.to_csv(out, index=False, float_format="%.4f")
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
