"""Build the Elliptic++ transaction-level State-Question-Options-Label dataset, the target task of the
address -> transaction transfer experiments.

One sample per labeled transaction of txs_features.csv (unknown dropped, never used as licit); same split
(train 1-29, val 30-34, test 35-49), value formatting, options (A. Illicit / B. Licit) and record fields as the address
dataset (scripts/build_addr_sqo.py), so the address-trained E1 / E2 read it with their unchanged renderers; only the
state header ("Bitcoin transaction statistics:", field `header`) and the question wording change. txId and time step
stay metadata. Two input variants with the same rows and ids:
  <out>/        all non-time features: 93 anonymized local + 72 anonymized 1-hop aggregate + 17 named Elliptic++ ones = 182,
                minus the aggregate features that are time-step proxies (see CHECKS.md section 3);
                the "183 features" of the Elliptic++ paper count the time step, which stays out of the prompt)
  <out>_named/  only the 17 named features (semantic control, ~10x shorter)
Also writes <out>/budgets/{frac}_s{seed}.txt: nested label-budget samples of the train rows (stratified by step x
label; for a seed, every smaller budget is a subset of every larger one), shared by every method and both variants.
Usage: python scripts/build_tx_sqo.py [raw_dir] [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/elliptic_pp")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/elliptic_pp_tx_sqo")
OUT_NAMED = OUT.parent / (OUT.name + "_named")
for o in [OUT, OUT_NAMED, OUT / "budgets"]:
    o.mkdir(parents=True, exist_ok=True)
TS = "Time step"
HEADER = "Bitcoin transaction statistics:"
QUESTION = "Is this Bitcoin transaction involved in illicit activity?"
OPTIONS = {"A": "Illicit", "B": "Licit"}
SHUTDOWN_STEP = 43
BUDGETS = [0.01, 0.05, 0.10]
SEEDS = range(5)


def split_of(step):   # = scripts/build_addr_sqo.py
    return np.select([step <= 29, step <= 34], ["train", "val"], "test")


def fmt(v):           # = scripts/build_addr_sqo.py
    if pd.isna(v):
        return "NA"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.6g}"


report = []
def log(s=""):
    print(s)
    report.append(s)


# ---------------------------------------------------------------- load
f = pd.read_csv(RAW / "txs_features.csv")
cls = pd.read_csv(RAW / "txs_classes.csv")
feats = [c for c in f.columns if c not in ("txId", TS)]
named = [c for c in feats if not c.startswith(("Local_feature_", "Aggregate_feature_"))]
assert len(feats) == 182 and len(named) == 17, (len(feats), len(named))
assert not f.txId.duplicated().any(), "duplicate txId in txs_features.csv"
w = f.merge(cls, on="txId", how="left", validate="one_to_one")
assert w["class"].notna().all()
w["step"] = w[TS]
w["split"] = split_of(w.step)

log("# Elliptic++ transaction-level SQO dataset — checks")
log(f"\nfeatures: {len(feats)} (anonymized local {sum(c.startswith('Local_') for c in feats)}, anonymized aggregate "
    f"{sum(c.startswith('Aggregate_') for c in feats)}, named {len(named)}); transactions: {len(w):,}; "
    f"duplicate txId: 0")
log(f"named features: {', '.join(named)}")

log("\n## 1. Temporal split")
steps = w.groupby("split").step.agg(["min", "max"]).reindex(["train", "val", "test"])
log(steps.to_string())
assert steps.loc["train", "max"] < steps.loc["val", "min"] <= steps.loc["val", "max"] < steps.loc["test", "min"]

log("\n## 2. Labels (1 illicit, 2 licit, 3 unknown -> dropped, never licit)")
log(pd.crosstab(w.split, w["class"]).reindex(["train", "val", "test"]).to_string())

d = w[w["class"] != 3].copy()
d["y"] = (d["class"] == 1).astype(int)
d["label"] = np.where(d.y == 1, "A", "B")
d["post43"] = d.step >= SHUTDOWN_STEP
d["seen_in_train"] = False   # every transaction occurs once (column kept for eval_decisions.py)
d["id"] = "tx_" + d.step.astype(str) + "_" + d.txId.astype(str)
d = d.sort_values(["step", "txId"]).reset_index(drop=True)

log("\n## 3. Features")
na = d[feats].isna().mean()
log(f"columns with missing values: {int((na > 0).sum())} (max share {na.max():.2%}): "
    f"{', '.join(na[na > 0].index[:20])}{' ...' if (na > 0).sum() > 20 else ''}")
corr = f[feats].corrwith(f[TS], method="spearman").abs().sort_values(ascending=False)
within = f.groupby(TS)[feats].std().mean() / f[feats].std()
log("largest |Spearman| with the time step (all txs), with within-step std / total std: "
    + ", ".join(f"{k} {v:.3f} ({within[k]:.2f})" for k, v in corr.head(6).items()))
# The original Elliptic local feature 1 was the time step; Elliptic++ moved it out of the local features, but some
# aggregate features (neighbour time-step statistics) still track it almost exactly. The address prompts never show the
# time step, so these proxies are dropped too (rule: |Spearman| >= 0.99 and within-step std < 0.2 of the total).
time_proxy = [c for c in feats if corr[c] >= 0.99 and within[c] < 0.2]
log(f"dropped as time-step proxies: {', '.join(time_proxy)}")
feats = [c for c in feats if c not in time_proxy]
log(f"features used: {len(feats)} (all variant), {len(named)} (named variant)")
loc = [c for c in feats if c.startswith("Local_")]
log(f"anonymized features already standardized by the dataset authors: mean |mean| "
    f"{f[loc].mean().abs().mean():.3f}, mean std {f[loc].std().mean():.3f} over all {len(f):,} txs")
log("aggregate features: 1-hop neighbourhood aggregates within the same time step (Weber et al. 2019; the per-step "
    "transaction graphs have no cross-step edges), so they carry no information from later steps")
same = d.duplicated(feats, keep=False)
conf = d[same].groupby(feats, dropna=False).y.nunique()
log(f"labeled rows sharing an identical feature vector: {same.sum():,}; groups with conflicting labels: "
    f"{int((conf > 1).sum())}")

# ---------------------------------------------------------------- address <-> transaction label linkage
log("\n## 4. Link to the address task (interpretation of address -> transaction transfer)")
ins = pd.read_csv(RAW / "AddrTx_edgelist.csv").rename(columns={"input_address": "address"})
outs = pd.read_csv(RAW / "TxAddr_edgelist.csv").rename(columns={"output_address": "address"})
inc = pd.concat([ins, outs]).drop_duplicates()
txc = cls.set_index("txId")["class"]
txs = f.set_index("txId")[TS]
inc["tx_class"] = inc.txId.map(txc)
inc["tx_step"] = inc.txId.map(txs)
wl = pd.read_csv(RAW / "wallets_classes.csv").set_index("address")["class"]
log("wallet labels are lifetime labels derived from the tx labels: address illicit <=> touches an illicit tx "
    "(asserted in elliptic_pp_addr_sqo/CHECKS.md)")
first_ill = inc[inc.tx_class == 1].groupby("address").tx_step.min()
wf = pd.read_csv(RAW / "wallets_features.csv", usecols=["address", TS]).drop_duplicates()
tr_rows = wf[wf[TS] <= 29].assign(cls=lambda x: x.address.map(wl))
tr_ill = tr_rows[tr_rows.cls == 1]
fut = (tr_ill.address.map(first_ill) >= 30).mean()
log(f"address train rows (steps 1-29) labeled illicit: {len(tr_ill):,}; share whose address touches no illicit tx "
    f"before step 30 (label justified only by val/test-period transactions): {fut:.1%}")
tr_addr = set(tr_rows.loc[tr_rows.cls.isin([1, 2]), "address"])
tr_ill_addr = set(tr_ill.address)
te = d[d.split == "test"][["txId", "y"]]
ti = inc[inc.txId.isin(te.txId)]
link = ti.groupby("txId").address.agg(lambda a: bool(tr_addr.intersection(a)))
link_ill = ti.groupby("txId").address.agg(lambda a: bool(tr_ill_addr.intersection(a)))
te = te.assign(link=te.txId.map(link).fillna(False).astype(bool), link_ill=te.txId.map(link_ill).fillna(False).astype(bool))
log(f"test transactions touching an address that has a labeled address-train row: {te.link.mean():.1%} "
    f"(illicit rate {te[te.link].y.mean():.1%} vs {te[~te.link].y.mean():.1%} without)")
log(f"test transactions touching an address labeled illicit in the address-train rows: {te.link_ill.mean():.2%} "
    f"(illicit rate among them {te[te.link_ill].y.mean():.1%}; share of all test illicit txs: "
    f"{te[te.y == 1].link_ill.mean():.1%})")
log("-> the address-trained checkpoints never saw transaction features or the test steps, but their train labels "
    "are functions of transaction labels (including later ones): '0% transaction labels' means no transaction-level "
    "supervision in the transaction input format, not independence from transaction labels")
d = d.merge(te[["txId", "link", "link_ill"]].rename(columns={"link": "addr_link", "link_ill": "addr_link_illicit"}),
            on="txId", how="left")
d[["addr_link", "addr_link_illicit"]] = d[["addr_link", "addr_link_illicit"]].fillna(False).astype(bool)


# ---------------------------------------------------------------- SQO records
def records(df, cols):
    for row in df.to_dict("records"):
        yield {
            "id": row["id"], "txId": int(row["txId"]), "step": int(row["step"]), "split": row["split"],
            "y": int(row["y"]), "label": row["label"],
            "state": {c: (None if pd.isna(row[c]) else float(row[c])) for c in cols},
            "header": HEADER,
            "state_text": "\n".join(f"- {c}: {fmt(row[c])}" for c in cols),
            "question": QUESTION, "options": OPTIONS,
            "seen_in_train": False, "post43": bool(row["post43"]),
            "addr_link": bool(row["addr_link"]), "addr_link_illicit": bool(row["addr_link_illicit"]),
        }


for out, cols in [(OUT, feats), (OUT_NAMED, named)]:
    for s in ["train", "val", "test"]:
        with open(out / f"{s}.jsonl", "w") as fh:
            for r in records(d[d.split == s], cols):
                assert str(r["txId"]) not in r["state_text"]
                fh.write(json.dumps(r) + "\n")
log("\n## 5. Prompt hygiene: txId and time step absent from every state_text: yes")
log("counts: " + ", ".join(f"{s} {int((d.split == s).sum()):,} (illicit {int(d[d.split == s].y.sum()):,})"
                           for s in ["train", "val", "test"]))

# ---------------------------------------------------------------- label budgets (nested, stratified)
tr = d[d.split == "train"].reset_index(drop=True)
log("\n## 6. Label budgets of the train rows (stratified by step x label, nested within a seed)")
rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    u = rng.random(len(tr))
    g = tr.groupby(["step", "y"])
    rank = pd.Series(u).groupby([tr.step, tr.y]).rank(method="first").values - 1   # 0-based rank within stratum
    n_g = g.y.transform("size").values
    v = g.ngroup().map(dict(enumerate(rng.random(g.ngroups)))).values             # one offset per stratum
    for b in BUDGETS:
        k = np.floor(b * n_g + v)                       # stochastic rounding of the per-stratum count, monotone in b
        ids = tr.id[rank < k]
        (OUT / "budgets" / f"{b:.2f}_s{seed}.txt").write_text("\n".join(ids) + "\n")
        rows.append({"budget": b, "seed": seed, "rows": len(ids), "illicit": int(tr.y[rank < k].sum()),
                     "steps_covered": tr.step[rank < k].nunique()})
    prev = None
    for b in BUDGETS:
        cur = set((OUT / "budgets" / f"{b:.2f}_s{seed}.txt").read_text().split())
        assert prev is None or prev <= cur, "budgets not nested"
        prev = cur
log(pd.DataFrame(rows).groupby("budget")[["rows", "illicit", "steps_covered"]].agg(["mean", "min", "max"]).to_string())
log(f"100% = all {len(tr):,} train rows (illicit {int(tr.y.sum()):,})")

# ---------------------------------------------------------------- token lengths
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    log("\n## 7. Token length (Qwen3-8B tokenizer, header + state + question, 1,000 samples)")
    for name, out in [("all", OUT), ("named 17", OUT_NAMED)]:
        smp = pd.read_json(out / "train.jsonl", lines=True).sample(1000, random_state=0)
        lens = [len(tok(HEADER + "\n" + r.state_text + "\n" + QUESTION).input_ids) for r in smp.itertuples()]
        log(f"{name}: " + pd.Series(lens).describe(percentiles=[.5, .99]).round(0).to_string().replace("\n", "; "))
except (ImportError, OSError):
    log("\n## 7. Token length: skipped (tokenizer not available)")

for out, cols in [(OUT, feats), (OUT_NAMED, named)]:
    json.dump({"features": cols, "header": HEADER, "question": QUESTION, "options": OPTIONS,
               "split_steps": {"train": [1, 29], "val": [30, 34], "test": [35, 49]},
               "counts": {s: int((d.split == s).sum()) for s in ["train", "val", "test"]},
               "budgets": f"{OUT}/budgets", "label_source": "txs_classes.csv (1 illicit -> A, 2 licit -> B, 3 dropped)"},
              open(out / "manifest.json", "w"), indent=1)
    (out / "CHECKS.md").write_text("\n".join(report) + "\n")
print(f"\nwrote {OUT} and {OUT_NAMED}")
