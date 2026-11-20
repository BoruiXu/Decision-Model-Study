"""GNN baselines (GraphSAGE / GAT) on the address-level SQO dataset: static, rolling refit, rolling incremental.

Graph: one snapshot per time step, inductive (Weber et al. 2019 style). Nodes = every (address, step) row of
wallets_features.csv for that step, unknown-label addresses included (features only, never labels). Edges =
input-address -> output-address of every transaction in that step (AddrTx x TxAddr joined on txId, tx step from
txs_features.csv), made undirected, self-loops dropped. Node features = the same 56 address features as the LLM
states, signed log1p then standardized on train rows. Loss = plain cross-entropy on labeled nodes (no reweighting,
like the LLMs), Adam, full-batch per step graph.
  static        : train on steps 1-29, pick the epoch by val (steps 30-34) AUPRC, score val and test
                  -> <MODEL>_{val,test}.parquet
  --roll refit  : for each test step t, retrain from scratch on steps <= t-delay for the static best epoch count
                  (same protocol as XGB_roll_d1) -> <MODEL>_roll_d1_test.parquet
  --roll inc    : start from the static model; before step t, continue training on the newly labeled steps plus a
                  replay of 0.5 older labeled nodes per new one, lr / 5 (same protocol as the rolling LLMs)
                  -> <MODEL>_inc_roll_d1_test.parquet
Rolling val files are copies of the static val file. Usage:
  CUDA_VISIBLE_DEVICES=3 python baselines/gnn_addr.py --model SAGE [--roll refit|inc]
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import to_undirected, remove_self_loops

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["SAGE", "GAT"], default="SAGE")
ap.add_argument("--roll", choices=["none", "refit", "inc"], default="none")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--inc_epochs", type=int, default=20, help="passes over new + replay graphs per incremental update")
ap.add_argument("--delay", type=int, default=1)
ap.add_argument("--replay", type=float, default=0.5)
ap.add_argument("--raw", default="data/raw/elliptic_pp")
ap.add_argument("--data", default="data/processed/elliptic_pp_addr_sqo")
ap.add_argument("--results", default="results/addr")
ap.add_argument("--cache", default="data/processed/elliptic_pp_addr_sqo/graph_snapshots.pt")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
RAW, D, RES = Path(a.raw), Path(a.data), Path(a.results)
dev = "cuda"
TS = "Time step"


# ---------------------------------------------------------------- per-step snapshot graphs (cached)
def build_graphs():
    w = pd.read_csv(RAW / "wallets_features.csv").drop_duplicates(["address", TS])
    feats = [c for c in w.columns if c not in ("address", TS)]
    tx_step = pd.read_csv(RAW / "txs_features.csv", usecols=["txId", TS]).set_index("txId")[TS]
    ins = pd.read_csv(RAW / "AddrTx_edgelist.csv")
    outs = pd.read_csv(RAW / "TxAddr_edgelist.csv")
    e = ins.merge(outs, on="txId")
    e["step"] = e.txId.map(tx_step)
    e = e.dropna(subset=["step"]).astype({"step": int}).drop_duplicates(["input_address", "output_address", "step"])
    lab = {}
    for s in ["train", "val", "test"]:
        lab.update(pd.read_json(D / f"{s}.jsonl", lines=True).set_index("id").y.to_dict())
    x = w[feats].values.astype(np.float64)
    x = np.sign(x) * np.log1p(np.abs(x))
    graphs = {}
    for t, g in w.assign(x_row=np.arange(len(w))).groupby(TS):
        idx = {ad: i for i, ad in enumerate(g.address)}
        et = e[e.step == t]
        src = et.input_address.map(idx); dst = et.output_address.map(idx)
        ok = src.notna() & dst.notna()
        ei = torch.tensor(np.stack([src[ok].values, dst[ok].values]).astype(np.int64))
        ei, _ = remove_self_loops(to_undirected(ei, num_nodes=len(g)))
        ids = ("addr_" + str(t) + "_" + g.address).values
        y = np.array([lab.get(i, -1) for i in ids])
        graphs[int(t)] = dict(x=torch.tensor(x[g.x_row.values], dtype=torch.float32), ei=ei, ids=ids,
                              y=torch.tensor(y))
        print(f"step {t}: nodes {len(g):,} labeled {(y >= 0).sum():,} edges {ei.shape[1]:,}", flush=True)
    return graphs


cache = Path(a.cache)
if cache.exists():
    G = torch.load(cache, weights_only=False)
else:
    G = build_graphs()
    torch.save(G, cache)
xs = torch.cat([G[t]["x"] for t in range(1, 30)])
mu, sd = xs.mean(0), xs.std(0).clamp_min(1e-6)
for t in G:
    G[t]["xn"] = ((G[t]["x"] - mu) / sd).to(dev)
    G[t]["ei_d"] = G[t]["ei"].to(dev)
    G[t]["y_d"] = G[t]["y"].to(dev)


# ---------------------------------------------------------------- model
class Net(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        h = a.hidden
        if a.model == "SAGE":
            self.c1, self.c2 = SAGEConv(d_in, h), SAGEConv(h, h)
        else:
            self.c1, self.c2 = GATConv(d_in, h // 4, heads=4), GATConv(h, h // 4, heads=4)
        self.out = torch.nn.Linear(h + d_in, 2)   # skip connection from raw features (Skip-GCN style)

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.c1(x, ei)), a.dropout, self.training)
        h = F.dropout(F.relu(self.c2(h, ei)), a.dropout, self.training)
        return self.out(torch.cat([h, x], 1))


def new_model():
    torch.manual_seed(a.seed)
    return Net(G[1]["xn"].shape[1]).to(dev)


def epoch(model, opt, masks):
    """One pass: one optimizer step per step graph, loss on the masked labeled nodes. masks: {t: bool tensor}."""
    model.train()
    for t in np.random.permutation(list(masks)):
        m = masks[t]
        z = model(G[t]["xn"], G[t]["ei_d"])
        loss = F.cross_entropy(z[m], G[t]["y_d"][m])
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def predict(model, steps):
    model.eval()
    out = []
    for t in steps:
        p = torch.softmax(model(G[t]["xn"], G[t]["ei_d"]), 1)[:, 1].cpu().numpy()
        lab = G[t]["y"].numpy() >= 0
        out.append(pd.DataFrame({"id": G[t]["ids"][lab], "p": p[lab], "step": t}))
    return pd.concat(out, ignore_index=True)


def labeled(steps):
    return {t: G[t]["y_d"] >= 0 for t in steps}


def ap_of(df):
    return average_precision_score(df.y, df.p)


meta = {s: pd.read_json(D / f"{s}.jsonl", lines=True)[["id", "y", "post43", "seen_in_train"]]
        for s in ["val", "test"]}
np.random.seed(a.seed)
TR, VA, TE = range(1, 30), range(30, 35), range(35, 50)

# ---------------------------------------------------------------- static (also gives the epoch count for refit)
model = new_model()
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=5e-4)
best, best_ep, best_state, t0 = -1, 0, None, time.time()
for ep in range(1, a.epochs + 1):
    epoch(model, opt, labeled(TR))
    if ep % a.eval_every == 0:
        v = meta["val"].merge(predict(model, VA), on="id")
        r = ap_of(v)
        if r > best:
            best, best_ep, best_state = r, ep, {k: x.clone() for k, x in model.state_dict().items()}
        print(f"epoch {ep} val AUPRC {r:.4f} (best {best:.4f} @ {best_ep}) {time.time() - t0:.0f}s", flush=True)
model.load_state_dict(best_state)
if a.roll == "none":
    for s, steps in [("val", VA), ("test", TE)]:
        df = meta[s].merge(predict(model, steps).drop(columns="step"), on="id", how="left")
        assert df.p.notna().all() and len(df) == len(meta[s])
        df.to_parquet(RES / f"{a.model}_{s}.parquet", index=False)
        print(f"{a.model} {s}: AUPRC {ap_of(df):.4f} AUROC {roc_auc_score(df.y, df.p):.4f}", flush=True)
    raise SystemExit

# ---------------------------------------------------------------- rolling
rng = np.random.default_rng(a.seed)
parts, prev_avail = [], 29
name = f"{a.model}_roll_d{a.delay}" if a.roll == "refit" else f"{a.model}_inc_roll_d{a.delay}"
for t in TE:
    avail = t - a.delay
    if a.roll == "refit":
        model = new_model()
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=5e-4)
        for _ in range(best_ep):
            epoch(model, opt, labeled(range(1, avail + 1)))
    else:
        new = labeled(range(prev_avail + 1, avail + 1))
        n_new = sum(int(m.sum()) for m in new.values())
        old = [(s, i) for s in range(1, prev_avail + 1) for i in torch.nonzero(G[s]["y_d"] >= 0).flatten().tolist()]
        pick = rng.choice(len(old), min(len(old), int(a.replay * n_new)), replace=False)
        masks = dict(new)
        for k in pick:
            s, i = old[k]
            if s not in masks:
                masks[s] = torch.zeros_like(G[s]["y_d"], dtype=torch.bool)
            masks[s][i] = True
        opt = torch.optim.Adam(model.parameters(), lr=a.lr / 5, weight_decay=5e-4)
        for _ in range(a.inc_epochs):
            epoch(model, opt, masks)
    parts.append(predict(model, [t]))
    print(f"step {t}: trained through step {avail}", flush=True)
    prev_avail = avail

te = meta["test"].merge(pd.concat(parts), on="id", how="left")
assert te.p.notna().all()
te.to_parquet(RES / f"{name}_test.parquet", index=False)
shutil.copy(RES / f"{a.model}_val.parquet", RES / f"{name}_val.parquet")
print(f"{name} test: AUPRC {ap_of(te):.4f} AUROC {roc_auc_score(te.y, te.p):.4f}")
