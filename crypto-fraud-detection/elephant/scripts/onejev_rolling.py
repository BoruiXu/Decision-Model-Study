"""Rolling (online) updates of an Elephant-1 / OneJev checkpoint with OneJev's own trainer (train.sft), one test
step at a time -- the protocol of scripts/kev_rolling.py and e2_rolling.py, with OneJev doing the training.

Before scoring test step t (first..last, default 35..49) with label delay d (default 1):
  new  = labeled rows with  prev_avail < step <= t - d   (t = 35: steps 30-34 if the init was trained on 1-29;
                                                          steps 1-34 with --init_trained_through 0, i.e. raw Elephant)
  old  = labeled rows with  step <= prev_avail
  rows = new + a uniform sample without replacement of int(replay * len(new)) old rows, shuffled
  train.sft one epoch on rows from the current model, with OneJev's own recipe and nothing patched: LoRA = the
  train.sft defaults (r 32, alpha 64, a fresh adapter every step) + --merge; full = the official
  train/configs/onejev_9b_full.yaml (FSDP), as in Baseten's elephant-1-sft example. Both add only what that example
  adds: --attention noncausal --readout slot_lm --save-every 0. The merged LoRA model gets the adapter's readout.json
  (train.sft does not write it into merged/), so the next step and the scorer load the same attention; full
  fine-tunes continue from <run>/final.
  score step t with elephant/scripts/onejev_score.py -> <ckpt>/steps/step_<t>.parquet; the previous model dir is deleted.
At the end: <results>/<name>_roll_d<d>_test.parquet. A rerun resumes after the last finished step (the replay sample
of a step depends only on --seed and t, so a resumed run draws the same rows as an uninterrupted one).
Note: train.sft applies its own schedule inside each update (warm-up = warmup_frac of the update's steps, cosine
decay to 0), not the constant schedule of the Qwen/Kev rolling scripts.
Usage (LoRA, one GPU):
  python elephant/scripts/onejev_rolling.py --onejev $ONEJEV_ROOT --init $ELEPHANT_MODEL --init_trained_through 0 \
      --data data/onejev/addr_json_choice --name EL_lora --mode lora --lr 1e-5
Full fine-tune on 4 GPUs: --mode full --nproc 4 --lr 1e-6 (adds --full-finetune --fsdp).
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--onejev", required=True)
ap.add_argument("--init", required=True, help="starting model dir (elephant-1, or a static fine-tune)")
ap.add_argument("--init_trained_through", type=int, default=29,
                help="last step the init model was trained on (0 for the raw, untrained Elephant-1)")
ap.add_argument("--data", required=True, help="dir with OneJev {train,val,test}.jsonl (export_onejev_data.py)")
ap.add_argument("--name", required=True)
ap.add_argument("--mode", choices=["lora", "full"], default="lora")
ap.add_argument("--nproc", type=int, default=1, help="GPUs for torchrun (--mode full uses FSDP: give >= 4)")
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--replay", type=float, default=0.5)
ap.add_argument("--delay", type=int, default=1)
ap.add_argument("--first", type=int, default=35)
ap.add_argument("--last", type=int, default=49)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ckpt", default=None, help="work dir (default ckpt/<name>_roll_d<delay>)")
ap.add_argument("--results", default="results/elephant")
ap.add_argument("--sft_args", default="", help="extra train.sft flags, e.g. '--lora-r 16 --batch-tokens 16384'")
ap.add_argument("--python", default=sys.executable)
a = ap.parse_args()
if a.mode == "full" and a.nproc < 2:
    raise SystemExit("--mode full uses the official FSDP config: run it with --nproc 4 (or more)")

here = Path(__file__).resolve().parent
work = Path(a.ckpt or f"ckpt/{a.name.lower()}_roll_d{a.delay}").resolve()
(work / "steps").mkdir(parents=True, exist_ok=True)
(work / "args.json").write_text(json.dumps(vars(a), indent=2))
rows = []
for split in ["train", "val", "test"]:
    rows += [json.loads(l) for l in open(Path(a.data) / f"{split}.jsonl") if l.strip()]
by_step = {}
for r in rows:
    by_step.setdefault(r["meta"]["step"], []).append(r)
state_file = work / "state.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {"model": str(Path(a.init).resolve()),
                                                                          "prev_avail": a.init_trained_through,
                                                                          "done": []}


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=a.onejev)
    if rc:
        raise SystemExit(f"command failed ({rc}), see {log}")


for t in range(a.first, a.last + 1):
    if t in state["done"]:
        continue
    avail = t - a.delay
    new = [r for s in range(state["prev_avail"] + 1, avail + 1) for r in by_step.get(s, [])]
    old = [r for s in range(1, state["prev_avail"] + 1) for r in by_step.get(s, [])]
    model = state["model"]
    if new:
        rng = random.Random(a.seed * 1000003 + t)
        rep = rng.sample(old, min(len(old), int(a.replay * len(new))))
        train = new + rep
        rng.shuffle(train)
        tfile = work / f"train_step{t}.jsonl"
        with open(tfile, "w") as f:
            for r in train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        launcher = ([a.python, "-m", "torch.distributed.run", "--nproc-per-node", str(a.nproc)] if a.nproc > 1
                    else [a.python])
        cmd = launcher + ["-m", "train.sft", "--model", model, "--train", str(tfile), "--name", f"step{t}",
                          "--runs-dir", str(work), "--epochs", "1", "--lr", str(a.lr), "--seed", str(a.seed + t),
                          "--attention", "noncausal", "--readout", "slot_lm", "--save-every", "0"]
        if a.mode == "full":   # official 9B recipe: full fine-tune + FSDP, max-len 16384, augment, pad_options 0.5
            i = cmd.index("train.sft") + 1
            cmd[i:i] = ["--config", "train/configs/onejev_9b_full.yaml",
                                                            "--media-root", str(work)]
        else:                  # train.sft defaults (LoRA r 32 / alpha 64, max-len 4096 -> raised for tx rows)
            cmd += ["--max-len", "16384", "--merge"]
        cmd += a.sft_args.split()
        print(f"step {t}: new {len(new):,} (steps {state['prev_avail'] + 1}-{avail}), replay {len(rep):,}", flush=True)
        run(cmd, work / f"train_step{t}.log")
        run_dir = work / f"step{t}"
        if a.mode == "lora":
            model_dir = run_dir / "merged"
            shutil.copy(run_dir / "adapter" / "readout.json", model_dir / "readout.json")
        else:
            model_dir = run_dir / "final"
        tfile.unlink()
    else:
        model_dir = Path(model)
        print(f"step {t}: no new labels, scoring with the current model", flush=True)
    sfile = work / f"score_step{t}.jsonl"
    with open(sfile, "w") as f:
        for r in by_step[t]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    run([a.python, str(here / "onejev_score.py"), "--onejev", a.onejev, "--model", str(model_dir),
         "--rows", str(sfile), "--name", f"step_{t}", "--split", "test", "--out", str(work / "steps")],
        work / f"score_step{t}.log")
    sfile.unlink()
    prev_model = Path(state["model"])
    if new and prev_model != Path(a.init).resolve() and work in prev_model.parents:
        shutil.rmtree(prev_model.parent if prev_model.name in ("merged", "final") else prev_model, ignore_errors=True)
    if new and a.mode == "lora":
        shutil.rmtree(run_dir / "adapter", ignore_errors=True)       # the merged model is what the next step loads
    state.update(model=str(model_dir), prev_avail=avail, done=state["done"] + [t])
    state_file.write_text(json.dumps(state, indent=2))

import pandas as pd  # noqa: E402

out = Path(a.results)
out.mkdir(parents=True, exist_ok=True)
res = pd.concat([pd.read_parquet(work / "steps" / f"step_{t}_test.parquet") for t in range(a.first, a.last + 1)])
path = out / f"{a.name}_roll_d{a.delay}_test.parquet"
res.to_parquet(path, index=False)
print(f"wrote {path} ({len(res):,} rows)", flush=True)
