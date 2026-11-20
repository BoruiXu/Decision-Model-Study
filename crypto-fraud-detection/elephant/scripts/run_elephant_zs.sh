#!/bin/bash
# Elephant-1 zero-shot on one GPU: address val for the 4 input forms + the option swap,
# pick the form by val AUPRC, score address test with it, then the transaction task (full and named, val and test).
# Run from crypto-fraud-detection/ after exporting the OneJev rows (see elephant/README.md).
# Needs: ONEJEV_PYTHON (python of the onejev env), JEV_PYTHON (python of the jev env), ELEPHANT_MODEL, ONEJEV_ROOT.
# Usage: elephant/scripts/run_elephant_zs.sh <gpu>
export CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY=${ONEJEV_PYTHON:?}
EL=${ELEPHANT_MODEL:?}
OJ=${ONEJEV_ROOT:?}
R=results/elephant
mkdir -p $R logs
score() {  # <data dir> <split> <name>
  nice -n 10 $PY elephant/scripts/onejev_score.py --onejev $OJ --model $EL --rows data/onejev/$1/$2.jsonl --name $3 --split $2 \
      --out $R > logs/el0_$3_$2.log 2>&1
  grep -E "^EL0|AUPRC|Error|error" logs/el0_$3_$2.log | tail -3
}
for D in addr_json_choice addr_json_noul addr_text_choice addr_text_noul addr_json_choice_swap; do
  score $D val EL0_$D
done
BEST=$(${JEV_PYTHON:?} -c "
import pandas as pd
from sklearn.metrics import average_precision_score as ap
s = {d: ap(*pd.read_parquet(f'$R/EL0_{d}_val.parquet')[['y', 'p']].T.values)
     for d in ['addr_json_choice', 'addr_json_noul', 'addr_text_choice', 'addr_text_noul']}
for d, v in sorted(s.items(), key=lambda x: -x[1]): print(f'{d} val AUPRC {v:.4f}', file=__import__('sys').stderr)
print(max(s, key=s.get))")
echo "best form on val: $BEST"
echo $BEST > $R/best_form.txt
score $BEST test EL0_$BEST
for D in tx_json_choice tx_named_json_choice; do for S in val test; do score $D $S EL0_$D; done; done
echo done
