#!/bin/bash
# Elephant-1 rolling evaluation on the address test steps 35-49 (label delay 1, replay 0.5, seed 0), starting from the
# static model EL_<mode>_<state>_all_s0 of run_elephant_ft.sh. Each step: one train.sft epoch on the new labels plus
# replay (onejev_rolling.py), then scoring of the step.
#   full : init = the checkpoint select_and_score.sh chose; full fine-tune (onejev_9b_full.yaml, FSDP) at lr 1e-6, >= 2 GPUs
#   lora : init = the static run's merged/ model (train.sft --merge merges the final adapter); a fresh LoRA adapter
#          (r 32 / alpha 64, --accum-rows 64) at lr 1e-5 every step, merged before scoring; 1 GPU
# Method name: EL_<mode>_<state>_all_roll_d1. Run from crypto-fraud-detection/. Needs: ONEJEV_PYTHON, ONEJEV_ROOT.
# Usage: elephant/scripts/run_elephant_rolling.sh <json|txhist> <full|lora> <gpus, e.g. 0 or 0,1>
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <json|txhist> <full|lora> <gpus>"; exit 1; }
STATE=$1 MODE=$2
case $STATE in json|txhist) ;; *) echo "state must be json or txhist"; exit 1 ;; esac
case $MODE in full|lora) ;; *) echo "mode must be full or lora"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
# onejev_rolling.py starts torch.distributed.run without --standalone (default port 29500): pick another port.
export PET_MASTER_PORT=${PET_MASTER_PORT:-$((20000 + RANDOM % 20000))}
PY=${ONEJEV_PYTHON:?} OJ=${ONEJEV_ROOT:?}
NGPU=$(echo "$3" | tr ',' '\n' | wc -l | tr -d ' ')
INIT_RUN=EL_${MODE}_${STATE}_all_s0
NAME=EL_${MODE}_${STATE}_all
D=$PWD/data/onejev/addr_${STATE}_noul
RES=results/elephant
WORK=$PWD/ckpt/${NAME}_roll_d1
[ ! -e "$RES/${NAME}_roll_d1_test.parquet" ] || { echo "FATAL: $RES/${NAME}_roll_d1_test.parquet exists"; exit 1; }
[ ! -e "$WORK/state.json" ] || echo "note: $WORK/state.json exists, resuming after the last finished step"

if [ "$MODE" = full ]; then
  [ "$NGPU" -ge 2 ] || { echo "FATAL: the full fine-tune uses FSDP on >= 2 GPUs"; exit 1; }
  SELECTED=$($PY -c "import json; print(json.load(open('$RES/ckpt_select/${INIT_RUN}_selection.json'))['selected'])")
  INIT=$PWD/ckpt/onejev/$INIT_RUN/$SELECTED
  LR=1e-6 SFT_ARGS=""
else
  [ "$NGPU" -eq 1 ] || { echo "FATAL: LoRA runs use one GPU"; exit 1; }
  INIT=$PWD/ckpt/onejev/$INIT_RUN/merged
  # train.sft does not write readout.json into merged/; without it the model would load with causal attention.
  [ -e "$INIT/readout.json" ] || cp "$PWD/ckpt/onejev/$INIT_RUN/adapter/readout.json" "$INIT/readout.json"
  LR=1e-5 SFT_ARGS="--accum-rows 64"
fi
[ -s "$INIT/model.safetensors" ] || { echo "FATAL: $INIT/model.safetensors missing"; exit 1; }
echo "init $INIT"   # outside the rolling work dir, so onejev_rolling.py never deletes it
mkdir -p logs

$PY elephant/scripts/onejev_rolling.py --onejev "$OJ" --init "$INIT" --init_trained_through 29 --data "$D" \
  --name "$NAME" --mode "$MODE" --nproc "$NGPU" --lr $LR --replay 0.5 --delay 1 --first 35 --last 49 --seed 0 \
  --ckpt "$WORK" --results "$PWD/$RES" --python "$PY" --sft_args="$SFT_ARGS" 2>&1 | tee "logs/roll_${NAME}_d1.log"

# eval_decisions.py picks the F1 threshold on <method>_val.parquet: use the val scores of the init model.
VAL=$RES/${NAME}_roll_d1_val.parquet
if [ ! -e "$VAL" ]; then
  if [ "$MODE" = full ]; then
    cp "$RES/${INIT_RUN}_val.parquet" "$VAL"            # select_and_score.sh scored val with this same checkpoint
  else
    $PY elephant/scripts/onejev_score.py --onejev "$OJ" --model "$INIT" --rows "$D/val.jsonl" \
      --name "${NAME}_roll_d1" --split val --out "$RES"
  fi
fi
echo "ROLLING_DONE ${NAME}_roll_d1"
