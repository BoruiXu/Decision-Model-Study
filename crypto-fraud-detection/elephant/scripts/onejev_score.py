"""Score OneJev rows (elephant/scripts/export_onejev_data.py) with an Elephant-1 / OneJev checkpoint and write our parquet format.

Uses OneJev's own code path (the one train.eval_mm uses): common.load_bundle (text-only Qwen3.5 LM), the readout of
the checkpoint (readout.json / config.json modeling_overrides -> block-bidirectional attention + slot_lm for
Elephant-1), the frozen production rendering (no option permutation) and common.collect_slot_logits under bf16
autocast. Unlike `train.eval_mm --dump-preds` (probabilities rounded to 5 decimals) it keeps the raw slot logits, so
AUPRC has no artificial ties. P(illicit) = softmax over the two slots, at the slot labelled "Illicit" (choice) or
"yes" (noul).
Output: <out>/<name>_<split>.parquet with id, y, p, post43, seen_in_train, step, logit_pos, logit_neg, latency_ms
(latency_ms = wall time of the whole scoring loop / rows, a throughput figure; per-request latency is measured
separately). Also prints acc / NLL / ECE (10 bins, OneJev's definition) to compare with train.eval_mm.
Run inside the OneJev environment:
  python elephant/scripts/onejev_score.py --onejev $ONEJEV_ROOT --model $ELEPHANT_MODEL \
      --rows data/onejev/addr_json_choice/val.jsonl --name EL0_json_choice --split val --out results/elephant
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--onejev", required=True, help="OneJev source tree (commit 81ce62f + elephant-1 serve.patch)")
ap.add_argument("--model", required=True, help="model dir: elephant-1, a full fine-tune `final`, or a LoRA `merged`")
ap.add_argument("--adapter", default=None, help="a train.sft LoRA dir (adapter / adapter-stepN): merged into --model "
                "in memory; its readout.json gives the attention")
ap.add_argument("--rows", required=True, help="OneJev jsonl from export_onejev_data.py")
ap.add_argument("--name", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--out", default="results/elephant")
ap.add_argument("--attention", choices=["auto", "noncausal", "causal"], default="auto",
                help="auto = what the checkpoint declares; Elephant-1 and its fine-tunes must resolve to noncausal")
ap.add_argument("--max_len", type=int, default=16384)
ap.add_argument("--batch_tokens", type=int, default=16384)
ap.add_argument("--max_rows", type=int, default=64)
ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke tests)")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.onejev).resolve()))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from qev.readout import HEAD_FILE, Readout, ReadoutConfig  # noqa: E402
from train import common  # noqa: E402

rows = [json.loads(l) for l in open(a.rows) if l.strip()]
if a.limit:
    rows = rows[: a.limit]
bundle = common.load_bundle(a.model, device=a.device, dtype="bfloat16", multimodal=False)
rcfg = ReadoutConfig.load(a.adapter or a.model)
if a.adapter:
    from peft import PeftModel
    bundle.model = PeftModel.from_pretrained(bundle.model, a.adapter).merge_and_unload().eval()
if a.attention != "auto":
    rcfg = dataclasses.replace(rcfg, attention=a.attention)
bundle.readout = Readout.attach(bundle.model, rcfg, slot_ids=bundle.slot_ids,
                                head_path=Path(a.adapter or a.model) / HEAD_FILE if rcfg.head else None)
print(f"model {a.model} adapter {a.adapter}: attention {rcfg.attention}, readout {rcfg.readout}", flush=True)
assert rcfg.readout == "slot_lm", "this script expects the slot_lm readout"

torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.autocast("cuda", dtype=torch.bfloat16):
    scored = common.collect_slot_logits(bundle, rows, max_len=a.max_len, batch_tokens=a.batch_tokens,
                                        max_rows=a.max_rows, device=a.device, quiet=True)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
if len(scored) != len(rows):
    raise SystemExit(f"{len(rows) - len(scored)} rows exceed max_len {a.max_len} and were skipped; refusing to "
                     "write a partial result")

recs, bins, nll = [], [[] for _ in range(10)], 0.0
for sc in scored:
    r = rows[sc.idx]
    pos = "Illicit" if "Illicit" in sc.labels else "yes"
    i_pos = sc.labels.index(pos)
    i_neg = 1 - i_pos
    assert sc.k == 2, f"{r['id']}: {sc.k} slots"
    z_pos, z_neg = sc.logits[i_pos], sc.logits[i_neg]
    p = 1.0 / (1.0 + math.exp(z_neg - z_pos))
    m = r["meta"]
    recs.append({"id": r["id"], "y": m["y"], "p": p, "post43": m["post43"], "seen_in_train": m["seen_in_train"],
                 "step": m["step"], "logit_pos": z_pos, "logit_neg": z_neg, "latency_ms": 1000 * elapsed / len(rows)})
    conf, ok = max(p, 1 - p), int((p >= 0.5) == bool(m["y"]))
    bins[min(9, int(conf * 10))].append((conf, ok))
    nll -= math.log(max(p if m["y"] else 1 - p, 1e-12))
order = {r["id"]: i for i, r in enumerate(rows)}
df = pd.DataFrame(sorted(recs, key=lambda x: order[x["id"]]))     # source row order (scoring is length-bucketed)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
path = out / f"{a.name}_{a.split}.parquet"
df.to_parquet(path, index=False)
n = len(df)
ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b)
print(f"{a.name} {a.split}: n {n:,}, illicit {int(df.y.sum()):,}, mean p {df.p.mean():.4f}, "
      f"acc {(df.y == (df.p >= 0.5)).mean():.4f}, nll {nll / n:.4f}, ece {ece:.4f}, "
      f"{elapsed:.0f} s ({1000 * elapsed / n:.1f} ms/row), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
      f"-> {path}", flush=True)
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"  AUPRC {average_precision_score(df.y, df.p):.4f}  AUROC {roc_auc_score(df.y, df.p):.4f}", flush=True)
except ImportError:
    pass
