#!/bin/bash
# Elephant-1 static fine-tuning on all 169,438 train rows with OneJev's train.sft (seed 0, 1 epoch), then checkpoint
# selection on val_sub4000 and scoring of the chosen checkpoint on the full val and test (select_and_score.sh).
#   <state>  json   : wallet statistics (data/onejev/addr_json_noul, export_onejev_data.py --state json --question noul)
#            txhist : transaction history (data/onejev/addr_txhist_noul, ../scripts/build_addr_txhist.py)
#   <mode>   full   : full fine-tune, official train/configs/onejev_9b_full.yaml (FSDP), lr 5e-6, >= 2 GPUs
#            lora   : LoRA (train.sft defaults r 32 / alpha 64), lr 1e-4, --accum-rows 64, 1 GPU; also writes merged/
# Method name: EL_<mode>_<state>_all_s0. Run from crypto-fraud-detection/ after the OneJev export (elephant/README.md).
# Needs: ONEJEV_PYTHON (python of the onejev env), ELEPHANT_MODEL, ONEJEV_ROOT.
# Usage: elephant/scripts/run_elephant_ft.sh <json|txhist> <full|lora> <gpus, e.g. 0 or 0,1>
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <json|txhist> <full|lora> <gpus>"; exit 1; }
STATE=$1 MODE=$2
case $STATE in json|txhist) ;; *) echo "state must be json or txhist"; exit 1 ;; esac
case $MODE in full|lora) ;; *) echo "mode must be full or lora"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY=${ONEJEV_PYTHON:?} EL=${ELEPHANT_MODEL:?} OJ=${ONEJEV_ROOT:?}
W=$PWD                       # train.sft runs inside $ONEJEV_ROOT, so pass absolute paths
RUNS=$W/ckpt/onejev
NGPU=$(echo "$3" | tr ',' '\n' | wc -l | tr -d ' ')
NAME=EL_${MODE}_${STATE}_all_s0
D=$W/data/onejev/addr_${STATE}_noul
[ ! -e "$RUNS/$NAME" ] || { echo "FATAL: $RUNS/$NAME already exists (use a new name)"; exit 1; }
for f in train val test val_sub4000; do [ -s "$D/$f.jsonl" ] || { echo "FATAL: missing $D/$f.jsonl"; exit 1; }; done
# Both input forms must hold the same samples in the same order (same ids as addr_json_noul).
R=$W/data/onejev/addr_json_noul
for f in train val_sub4000; do
  cmp -s <(cut -d'"' -f4 "$D/$f.jsonl") <(cut -d'"' -f4 "$R/$f.jsonl") \
    || { echo "FATAL: ids of $D/$f.jsonl differ from $R/$f.jsonl"; exit 1; }
done

COMMON="--train $D/train.jsonl --name $NAME --runs-dir $RUNS --seed 0 --attention noncausal --readout slot_lm \
  --eval val=$D/val_sub4000.jsonl --eval-every 250 --eval-at-start --save-every 400"
if [ "$MODE" = full ]; then
  [ "$NGPU" -ge 2 ] || { echo "FATAL: the full fine-tune uses FSDP on >= 2 GPUs"; exit 1; }
  # Launch through the env's python: a system `torchrun` may start another interpreter without the onejev packages.
  CMD="$PY -m torch.distributed.run --standalone --nproc-per-node $NGPU -m train.sft \
    --config train/configs/onejev_9b_full.yaml --model $EL --media-root $D --lr 5e-6 $COMMON"
else
  [ "$NGPU" -eq 1 ] || { echo "FATAL: LoRA runs use one GPU"; exit 1; }
  CMD="$PY -m train.sft --model $EL --epochs 1 --lr 1e-4 --max-len 16384 --accum-rows 64 --merge $COMMON"
fi
mkdir -p "$RUNS" logs
echo "$CMD"
(cd "$OJ" && $CMD) 2>&1 | tee "logs/train_$NAME.log"

elephant/scripts/select_and_score.sh "$NAME" "$MODE" "$D" 2>&1 | tee "logs/select_$NAME.log"
