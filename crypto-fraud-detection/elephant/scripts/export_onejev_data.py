"""Export an SQO dataset (scripts/build_addr_sqo.py / build_tx_sqo.py) as OneJev unified rows for Elephant-1.

One row per labeled sample, same ids / split / order as the source jsonl:
  {"id", "split", "state", "question", "target", "meta": {"state_id", "step", "y", "post43", "seen_in_train"}}
--state json : the feature dict, wrapped as {"entity": <header>, "features": {...}}; OneJev renders it as indent-2 JSON
               inside <state>...</state> (Elephant's native form). Numbers: integers as int, others rounded to 6
               significant digits (the same values as state_text), missing as null.
--state text : the exact text the Qwen/Kev experiments used: header + "\n" + state_text ("- name: value" lines).
--question choice : {"type": "choice", "criteria": {"Illicit": null, "Licit": null}}, target {"Illicit": y, "Licit": 1-y}
--question noul   : {"type": "noul"}, target {"yes": y, "no": 1-y}   (P(illicit) = P(yes))
--ids FILE        : keep only the train rows whose id is listed (label budgets, e.g. budgets/0.05_s0.txt)
--swap            : list the options as Licit, Illicit (position-bias check; choice only)
Also writes val_sub4000.jsonl: 4,000 val rows drawn uniformly (seed 0) for in-training evaluation -- train.sft
--eval-limit takes the FIRST rows of a file, which in our step-sorted val would all be step 30.
The address string, txId and time step never enter the state (asserted). Writes <out>/{train,val,test}.jsonl.
Usage: python elephant/scripts/export_onejev_data.py --data data/processed/elliptic_pp_addr_sqo \
           --out data/onejev/addr_json_choice --state json --question choice
"""
import argparse
import json
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="SQO dataset dir with {train,val,test}.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--state", choices=["json", "text"], default="json")
ap.add_argument("--question", choices=["choice", "noul"], default="choice")
ap.add_argument("--ids", default=None, help="optional id list restricting the train split")
ap.add_argument("--swap", action="store_true")
ap.add_argument("--splits", default="train,val,test")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
keep = set(Path(a.ids).read_text().split()) if a.ids else None


def num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if float(v).is_integer():
        return int(v)
    return float(f"{v:.6g}")


def row(r):
    header = r.get("header") or "Bitcoin address statistics:"
    if a.state == "json":
        state = {"entity": header.rstrip(":"), "features": {k: num(v) for k, v in r["state"].items()}}
        text = json.dumps(state)
    else:
        state = header + "\n" + r["state_text"]
        text = state
    assert f"step: {r['step']}" not in text and (r.get("address") or "@@") not in text and str(r.get("txId", "@@")) not in text.split(), r["id"]
    y = int(r["y"])
    if a.question == "choice":
        assert r["options"] == {"A": "Illicit", "B": "Licit"}
        crit = {"Licit": None, "Illicit": None} if a.swap else {"Illicit": None, "Licit": None}
        q = {"type": "choice", "instructions": r["question"], "criteria": crit}
        target = {"Illicit": float(y), "Licit": float(1 - y)}
    else:
        q = {"type": "noul", "instructions": r["question"]}
        target = {"yes": float(y), "no": float(1 - y)}
    return {"id": r["id"], "split": r["split"], "state": state, "question": q, "target": target,
            "meta": {"state_id": r.get("address") or r["id"], "step": int(r["step"]), "y": y,
                     "post43": bool(r["post43"]), "seen_in_train": bool(r["seen_in_train"])}}


for split in a.splits.split(","):
    n = n_ill = 0
    with open(Path(a.data) / f"{split}.jsonl") as fin, open(out / f"{split}.jsonl", "w") as fout:
        for line in fin:
            r = json.loads(line)
            if split == "train" and keep is not None and r["id"] not in keep:
                continue
            fout.write(json.dumps(row(r), ensure_ascii=False) + "\n")
            n += 1
            n_ill += int(r["y"])
    if split == "train" and keep is not None:
        assert n == len(keep), f"{len(keep) - n} listed ids not found in train"
    print(f"{split}: {n:,} rows, {n_ill:,} illicit -> {out / (split + '.jsonl')}", flush=True)
if (out / "val.jsonl").exists():
    import random
    lines = open(out / "val.jsonl").read().splitlines()
    pick = sorted(random.Random(0).sample(range(len(lines)), min(4000, len(lines))))
    (out / "val_sub4000.jsonl").write_text("\n".join(lines[i] for i in pick) + "\n")
    print(f"val_sub4000: {len(pick):,} rows (uniform, seed 0)", flush=True)
(out / "export_args.json").write_text(json.dumps(vars(a), indent=2))
