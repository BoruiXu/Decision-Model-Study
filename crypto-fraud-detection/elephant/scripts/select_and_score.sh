#!/bin/bash
# Checkpoint selection + final scoring for a train.sft run (called by run_elephant_ft.sh).
# LoRA runs: adapter-step* and the final adapter; full fine-tunes: ckpt-step* and final.
# 1. score every saved checkpoint on val_sub4000 (resumable: existing per-checkpoint files are reused);
# 2. pick the highest AUPRC (rounded to 4 decimals), ties -> earlier step;
# 3. score the chosen checkpoint on the full val and test (refuses to overwrite).
# Run from crypto-fraud-detection/. Needs: ONEJEV_PYTHON, ELEPHANT_MODEL, ONEJEV_ROOT.
# Usage: elephant/scripts/select_and_score.sh <run_name> <lora|full> <onejev_data_dir>
set -euo pipefail
PY=${ONEJEV_PYTHON:?} EL=${ELEPHANT_MODEL:?} OJ=${ONEJEV_ROOT:?}
NAME=$1 KIND=$2 D=$3
RUN=ckpt/onejev/$NAME
RES=results/elephant
SEL=$RES/ckpt_select
mkdir -p "$SEL"

if [ "$KIND" = lora ]; then FINAL=adapter; PATTERN="adapter-step*"; else FINAL=final; PATTERN="ckpt-step*"; fi
CKPTS=()
while IFS= read -r c; do CKPTS+=("$c"); done < <(cd "$RUN" && ls -d $PATTERN "$FINAL" 2>/dev/null)
[ ${#CKPTS[@]} -gt 0 ] || { echo "FATAL: no checkpoints in $RUN"; exit 1; }
echo "checkpoints: ${CKPTS[*]}"

score() {  # score <ckpt_tag> <rows> <name> <split> <outdir>
  if [ "$KIND" = lora ]; then   # the adapter is merged into the base model in memory
    $PY elephant/scripts/onejev_score.py --onejev "$OJ" --model "$EL" --adapter "$RUN/$1" --rows "$2" --name "$3" --split "$4" --out "$5"
  else
    $PY elephant/scripts/onejev_score.py --onejev "$OJ" --model "$RUN/$1" --rows "$2" --name "$3" --split "$4" --out "$5"
  fi
}

for C in "${CKPTS[@]}"; do
  f=$SEL/${NAME}_${C}_valsub.parquet
  if [ -e "$f" ]; then echo "reuse $f"; continue; fi
  score "$C" "$D/val_sub4000.jsonl" "${NAME}_${C}" valsub "$SEL"
done

BEST=$($PY - "$NAME" "$RUN" "$SEL" "$FINAL" "${CKPTS[@]}" <<'P'
import json, re, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
name, run, sel, final, *ckpts = sys.argv[1:]
done = [json.loads(l) for l in open(Path(run) / "log.jsonl") if '"event": "done"' in l]
final_step = done[-1]["steps"] if done else 10**9
rows = []
for c in ckpts:
    df = pd.read_parquet(Path(sel) / f"{name}_{c}_valsub.parquet")
    step = final_step if c == final else int(re.search(r"step(\d+)", c).group(1))
    rows.append({"ckpt": c, "step": step, "auprc": average_precision_score(df.y, df.p),
                 "auroc": roc_auc_score(df.y, df.p), "n": len(df)})
t = pd.DataFrame(rows).sort_values("step")
t["auprc_r"] = t.auprc.round(4)
best = t.sort_values(["auprc_r", "step"], ascending=[False, True]).iloc[0]
print(t.drop(columns="auprc_r").to_string(index=False), file=sys.stderr)
print(f"selected {best.ckpt} (step {best.step}, val_sub4000 AUPRC {best.auprc:.4f})", file=sys.stderr)
out = Path(sel) / f"{name}_selection.json"
if not out.exists():
    out.write_text(json.dumps({"selected": best.ckpt, "step": int(best.step), "rule": "max AUPRC (4 dp), ties -> earlier",
                               "table": t.drop(columns="auprc_r").to_dict("records")}, indent=1))
print(best.ckpt)
P
)
echo "BEST=$BEST"

for f in "$RES/${NAME}_val.parquet" "$RES/${NAME}_test.parquet"; do
  [ ! -e "$f" ] || { echo "FATAL: $f already exists (use a new name)"; exit 1; }
done
for S in val test; do
  score "$BEST" "$D/$S.jsonl" "$NAME" "$S" "$RES"
done
echo "SELECT_AND_SCORE_DONE $NAME -> $BEST"
