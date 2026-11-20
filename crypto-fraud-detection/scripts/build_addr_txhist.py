"""Elliptic++ address samples with a transaction-history state instead of the 56 wallet statistics, written
directly as OneJev rows (the json + noul form that won on val for the statistics state).

Same samples, ids, splits, labels and meta as data/processed/elliptic_pp_addr_sqo (read from there); only the
state changes. For sample (address a, step t) the state lists the transactions of a whose time step is <= t (no
future information, unlike the lifetime wallet statistics), newest first, at most --max_txs (summary keeps the
full counts). Per transaction: how many steps ago, the address's role (sent = input address, received = output
address, both) and the transaction's own named Elliptic++ features (total BTC, fee, numbers of input / output
addresses, smallest / largest output). Elliptic++ gives no per-address amount within a transaction.
Never in the state: transaction labels (the address label is derived from them), txIds, address strings,
absolute time steps (asserted).
Usage: python scripts/build_addr_txhist.py [--out data/onejev/addr_txhist_noul] [--max_txs 32]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw/elliptic_pp")
ap.add_argument("--sqo", default="data/processed/elliptic_pp_addr_sqo")
ap.add_argument("--out", default="data/onejev/addr_txhist_noul")
ap.add_argument("--max_txs", type=int, default=32)
a = ap.parse_args()
RAW, OUT = Path(a.raw), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
HEADER = "Bitcoin address transaction history"
TXF = {"total_BTC": "tx_total_btc", "fees": "tx_fee_btc", "num_input_addresses": "tx_num_input_addresses",
       "num_output_addresses": "tx_num_output_addresses", "out_BTC_min": "tx_smallest_output_btc",
       "out_BTC_max": "tx_largest_output_btc"}

report = []
def log(s=""):
    print(s, flush=True)
    report.append(s)


def num(v):
    v = float(v)
    return int(v) if v.is_integer() else float(f"{v:.6g}")


tx = pd.read_csv(RAW / "txs_features.csv", usecols=["txId", "Time step", *TXF]).set_index("txId")
inc = pd.concat([pd.read_csv(RAW / "AddrTx_edgelist.csv").rename(columns={"input_address": "address"}).assign(s=1),
                 pd.read_csv(RAW / "TxAddr_edgelist.csv").rename(columns={"output_address": "address"}).assign(r=1)])
assert inc.txId.isin(tx.index).all()
inc = inc.fillna(0).groupby(["address", "txId"], as_index=False)[["s", "r"]].max()
inc["role"] = np.select([(inc.s == 1) & (inc.r == 1), inc.s == 1], ["both", "sent"], "received")
inc = inc.join(tx, on="txId").rename(columns={"Time step": "tstep"}).sort_values(["address", "tstep", "txId"])
events = {addr: g for addr, g in inc.groupby("address", sort=False)}
log("# Elliptic++ address transaction-history state (txhist) — checks")
log(f"\naddress-transaction incidences: {len(inc):,} (role counts {inc.role.value_counts().to_dict()})")


def state_of(addr, t):
    g = events[addr]
    g = g[g.tstep <= t]
    assert len(g), (addr, t)
    shown = g.iloc[::-1].iloc[:a.max_txs]                     # newest first
    txs = [{"steps_ago": int(t - r.tstep), "role": r.role, **{TXF[k]: num(getattr(r, k)) for k in TXF}}
           for r in shown.itertuples()]
    summ = {"txs_sent": int((g.role != "received").sum()), "txs_received": int((g.role != "sent").sum()),
            "first_seen_steps_ago": int(t - g.tstep.min()), "active_steps": int(g.tstep.nunique()),
            "txs_listed": len(txs)}
    return {"entity": HEADER, "summary": summ, "transactions": txs}, len(g)


n_hist, n_trunc = [], 0
for split in ["train", "val", "test"]:
    n = n_ill = 0
    with open(Path(a.sqo) / f"{split}.jsonl") as fin, open(OUT / f"{split}.jsonl", "w") as fout:
        for line in fin:
            r = json.loads(line)
            state, k = state_of(r["address"], r["step"])
            text = json.dumps(state)
            assert r["address"] not in text and '"step"' not in text and "tstep" not in text, r["id"]
            n_hist.append(k)
            n_trunc += k > a.max_txs
            y = int(r["y"])
            fout.write(json.dumps({"id": r["id"], "split": split, "state": state,
                                   "question": {"type": "noul", "instructions": r["question"]},
                                   "target": {"yes": float(y), "no": float(1 - y)},
                                   "meta": {"state_id": r["address"], "step": int(r["step"]), "y": y,
                                            "post43": bool(r["post43"]),
                                            "seen_in_train": bool(r["seen_in_train"])}}) + "\n")
            n += 1
            n_ill += y
    log(f"{split}: {n:,} rows, {n_ill:,} illicit")
lines = open(OUT / "val.jsonl").read().splitlines()
pick = sorted(random.Random(0).sample(range(len(lines)), 4000))
(OUT / "val_sub4000.jsonl").write_text("\n".join(lines[i] for i in pick) + "\n")
h = pd.Series(n_hist)
log(f"\ntransactions up to the sample's step, per sample: {h.describe(percentiles=[.5, .9, .99]).round(1).to_dict()}")
log(f"samples truncated to the newest {a.max_txs}: {n_trunc:,} ({n_trunc / len(h):.2%})")
log("never in the state (asserted): address strings, txIds, absolute time steps, transaction labels")
(OUT / "CHECKS.md").write_text("\n".join(report) + "\n")
(OUT / "export_args.json").write_text(json.dumps(vars(a), indent=2))
