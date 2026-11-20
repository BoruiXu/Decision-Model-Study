"""Build the Elliptic++ address-level State-Question-Options-Label dataset.

One sample per labeled (address, time step) row of wallets_features.csv; input = the original
Elliptic++ wallet features (whole-lifetime statistics, leakage quantified in CHECKS.md);
label = original wallets_classes (unknown dropped). Address string and time step are kept as
metadata only and never enter the prompt text.

Usage: python scripts/build_addr_sqo.py [raw_dir] [out_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/elliptic_pp")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed/elliptic_pp_addr_sqo")
OUT.mkdir(parents=True, exist_ok=True)
TS = "Time step"
QUESTION = "Is this Bitcoin address involved in illicit activity?"
OPTIONS = {"A": "Illicit", "B": "Licit"}
SHUTDOWN_STEP = 43


# Temporal split: steps 1-29 train, 30-34 val, 35-49 test.
def split_of(step):
    return np.select([step <= 29, step <= 34], ["train", "val"], "test")


def fmt(v):
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
w = pd.read_csv(RAW / "wallets_features.csv")
cls = pd.read_csv(RAW / "wallets_classes.csv")
feats = [c for c in w.columns if c not in ("address", TS)]
names = {c: c.replace(" ", "_") for c in feats}
n_dup = w.duplicated(["address", TS]).sum()
# The raw file repeats some (address, step) rows verbatim (up to 568x); keep one copy.
assert n_dup == w.duplicated().sum(), "non-identical duplicate (address, step) rows"
w = w.drop_duplicates(["address", TS])
w = w.merge(cls, on="address", how="left", validate="many_to_one")
assert w["class"].notna().all()
w["step"] = w[TS]
w["split"] = split_of(w.step)

log("# Elliptic++ address-level SQO dataset — checks")
log(f"\nfeatures: {len(feats)}; wallet rows: {len(w):,}; exact-duplicate (address, step) rows dropped: {n_dup:,}")

# ---------------------------------------------------------------- leakage audit (all rows)
log("\n## 1. Temporal split")
steps = w.groupby("split").step.agg(["min", "max"]).reindex(["train", "val", "test"])
log(steps.to_string())
assert steps.loc["train", "max"] < steps.loc["val", "min"] <= steps.loc["val", "max"] < steps.loc["test", "min"]

log("\n## 2. Labels")
assert (w.groupby("address")["class"].nunique() == 1).all(), "label not constant per address"
log("label constant per address: yes")
inc = pd.concat([pd.read_csv(RAW / "AddrTx_edgelist.csv").rename(columns={"input_address": "address"}),
                 pd.read_csv(RAW / "TxAddr_edgelist.csv").rename(columns={"output_address": "address"})])
txc = pd.read_csv(RAW / "txs_classes.csv").set_index("txId")["class"]
touch_ill = (inc.txId.map(txc) == 1).groupby(inc.address).any()
lab = cls.set_index("address")["class"]
ct = pd.crosstab(lab, touch_ill.reindex(lab.index).fillna(False), rownames=["class"], colnames=["touches illicit tx"])
log(ct.to_string())
assert ((lab == 1) == touch_ill.reindex(lab.index).fillna(False)).all(), "illicit <=> touches illicit tx violated"
log("rule 'illicit <=> touches an illicit tx' holds for every address")

log("\n## 3. Future information in the wallet features")
multi = w[w.groupby("address").step.transform("size") > 1]
const = (multi.groupby("address")[feats].nunique() == 1).all(axis=1).mean()
log(f"multi-step addresses: {multi.address.nunique():,} ({len(multi):,} rows)")
log(f"  share whose {len(feats)} features are identical in every row: {const:.1%}")
blk = w[w.num_timesteps_appeared_in == 1].groupby("step").first_block_appeared_in.max()
fut = (w.last_block_appeared_in > w.step.map(blk)).mean()
log(f"rows with last_block_appeared_in beyond the current step's block range: {fut:.1%}")
log(f"rows with num_timesteps_appeared_in > 1: {(w.num_timesteps_appeared_in > 1).mean():.1%}")

# ---------------------------------------------------------------- labeled samples
d = w[w["class"] != 3].copy()
assert len(d) and (d["class"].isin([1, 2])).all()
d["y"] = (d["class"] == 1).astype(int)
d["label"] = np.where(d.y == 1, "A", "B")
train_addrs = set(w.loc[w.split == "train", "address"])
d["seen_in_train"] = d.address.isin(train_addrs) & (d.split != "train")
d["post43"] = d.step >= SHUTDOWN_STEP
d = d.sort_values(["step", "address"]).reset_index(drop=True)
d["id"] = "addr_" + d.step.astype(str) + "_" + d.address

log("\n## 4. Entity overlap")
for s in ["val", "test"]:
    x = d[d.split == s]
    log(f"{s}: rows whose address also appears in train steps: {x.seen_in_train.mean():.1%} "
        f"({x.seen_in_train.sum():,} of {len(x):,}); illicit share among them: "
        f"{x[x.seen_in_train].y.mean():.1%}")
log("-> also report metrics on the seen_in_train == False subset")

log("\n## 5. Class distribution (unknown excluded)")
dist = d.pivot_table(index="split", columns="y", values="step", aggfunc="size").reindex(["train", "val", "test"])
dist.columns = ["licit", "illicit"]
dist["illicit_%"] = (100 * dist.illicit / dist.sum(axis=1)).round(2)
log(dist.to_string())
log(f"test pre-43 illicit: {d[(d.split == 'test') & ~d.post43].y.sum():,}; "
    f"post-43 illicit: {d[(d.split == 'test') & d.post43].y.sum():,}")


# ---------------------------------------------------------------- SQO records
def state_text(row):
    return "\n".join(f"- {names[c]}: {fmt(row[c])}" for c in feats)


def records(df):
    for row in df.to_dict("records"):
        yield {
            "id": row["id"], "address": row["address"], "step": int(row["step"]), "split": row["split"],
            "y": int(row["y"]), "label": row["label"],
            "state": {names[c]: (None if pd.isna(row[c]) else float(row[c])) for c in feats},
            "state_text": state_text(row), "question": QUESTION, "options": OPTIONS,
            "seen_in_train": bool(row["seen_in_train"]), "post43": bool(row["post43"]),
        }


n_leak = 0
for s in ["train", "val", "test"]:
    with open(OUT / f"{s}.jsonl", "w") as f:
        for r in records(d[d.split == s]):
            n_leak += r["address"] in r["state_text"] or f"step: {r['step']}" in r["state_text"]
            f.write(json.dumps(r) + "\n")
assert n_leak == 0
log("\n## 6. Prompt hygiene")
log("address string and time step absent from every state_text: yes")
log(f"time-step-like feature names in state: {[n for n in names.values() if 'step' in n.lower()]}")

# ---------------------------------------------------------------- token lengths
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    sample = d.sample(2000, random_state=0)
    lens = [len(tok(state_text(r) + "\n" + QUESTION).input_ids) for r in sample.to_dict("records")]
    log("\n## 7. Token length (Qwen3-8B tokenizer, state + question, 2,000 samples)")
    log(pd.Series(lens).describe(percentiles=[.5, .9, .99]).round(1).to_string())
except (ImportError, OSError):
    log("\n## 7. Token length: skipped (Qwen3-8B tokenizer not available locally)")

json.dump({"features": [names[c] for c in feats], "question": QUESTION, "options": OPTIONS,
           "split_steps": {"train": [1, 29], "val": [30, 34], "test": [35, 49]},
           "counts": {s: int((d.split == s).sum()) for s in ["train", "val", "test"]},
           "label_source": "wallets_classes.csv (1 illicit -> A, 2 licit -> B, 3 unknown dropped)"},
          open(OUT / "manifest.json", "w"), indent=1)
(OUT / "CHECKS.md").write_text("\n".join(report) + "\n")
print(f"\nwrote {OUT}")
