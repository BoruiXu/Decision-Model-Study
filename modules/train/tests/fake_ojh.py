"""A stand-in for `ojh` in tests: `run` writes a finished harness-style run directory at once (run.lock.yaml, gpus.json,
result.json, DONE and the selected checkpoint, plus the merged model when the experiment asks for a LoRA merge, an
unselected checkpoint and deletion_candidates.json), `check` prints ok, `cleanup <run_dir> --confirm` deletes the
deletion candidates. The checkpoint holds fake_model.json, which tests/fake_scorer.py reads. Every run's
train-s<seed>.log has one step line (step 5 of 32, 0.2 h left).

Environment:
  FAKE_OJH_OUT      where run directories go
  FAKE_OJH_CALLS    a file that gets one line per `run` call
  FAKE_OJH_QUALITY  JSON {scheme key: strength or "fail"}; the key is "<base>" plus "|<override>=<value>" for each
                    override, sorted; unknown keys have strength 1.0. Higher strength separates the classes better.
                    "fail": training fails (DONE exit code 1).
  FAKE_OJH_PENDING  "1": leave the run unfinished (no DONE)
  FAKE_OJH_SLEEP    seconds `run` sleeps after printing its JSON (it touches $FAKE_OJH_OUT/sleeping first)
  FAKE_OJH_CLEANUP_SKIP  "1": `cleanup` skips every candidate (and says so on stderr)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("exp")
    ap.add_argument("--base")
    ap.add_argument("--recipe")
    ap.add_argument("--gpus")
    ap.add_argument("--machine")
    ap.add_argument("--allow-shared", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "cleanup":   # the positional is the run directory
        p = Path(a.exp) / "deletion_candidates.json"
        for c in (json.loads(p.read_text())["candidates"] if p.is_file() and a.confirm else []):
            if os.environ.get("FAKE_OJH_CLEANUP_SKIP") == "1":
                print(f"skipped {c['path']}: not strictly below the run's runs/ directory", file=sys.stderr)
                continue
            shutil.rmtree(c["path"])
            print(f"deleted {c['path']}")
        return 0
    exp = yaml.safe_load(Path(a.exp).read_text())
    if a.cmd == "check":
        print(json.dumps({"ok": True, "cards_checked": False, "base": a.base}))
        return 0
    with open(os.environ["FAKE_OJH_CALLS"], "a") as f:
        f.write(f"{exp['name']} --gpus {a.gpus} --machine {a.machine}\n")
    seed = exp["seeds"][0]
    ovs = {k: v["value"] for k, v in (exp.get("overrides") or {}).items()}
    key = "|".join([a.base] + [f"{k}={ovs[k]}" for k in sorted(ovs)])
    quality = json.loads(os.environ.get("FAKE_OJH_QUALITY", "{}")).get(key, 1.0)
    root = Path(os.environ["FAKE_OJH_OUT"]) / exp["name"]
    n = len(list(root.glob("r*"))) if root.is_dir() else 0
    rd = root / f"r{n}"
    rd.mkdir(parents=True)
    recipe = Path(a.recipe)
    rec = yaml.safe_load(recipe.read_text())
    mode = rec["mode"]
    flags = dict(rec.get("onejev") or {}, **({"full_finetune": True} if mode == "full" else {}))
    flags.update({k.split(".", 1)[1]: v for k, v in ovs.items() if k.startswith("onejev.")})
    (rd / "run.lock.yaml").write_text(yaml.safe_dump({
        "experiment": exp["name"], "run_dir": str(rd), "mode": mode, "seeds": [seed], "flags": flags,
        "base": {"name": a.base, "multimodal": True}, "merge": bool(exp.get("merge")),
        "machine": {"model_dir": "/models", "env": {"CUDA_HOME": "/usr/local/cuda"}},
        "recipe": {"path": str(recipe), "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()},
        "overrides": exp.get("overrides") or {}, "onejev": {"revision": "81ce62f"},
        "data": {"prepared_dir": str(rd / "prepared"), "source": exp["data"]["train"],
                 "train": str(rd / "prepared" / "train.jsonl"), "sha256": {"train.jsonl": "cd" * 32},
                 "eval_sha256": {k: {"train_labels": "ef" * 32, "reference": None} for k in exp["data"]["eval"]}},
        "eval_sets": {k: {"train_labels": v["train_labels"], "reference": v.get("reference"), "adapter_args": {}}
                      for k, v in exp["data"]["eval"].items()},
        "split_file": exp["data"].get("split_file")}))
    gpus = [int(g) for g in a.gpus.split(",")]
    cards = [{"uuid": f"GPU-{g}", "name": "NVIDIA A100"} for g in gpus]
    (rd / "gpus.json").write_text(json.dumps(cards))
    session = f"ojh-test-{exp['name']}-{rd.name}"
    out = {"run_dir": str(rd), "session": session, "log": str(rd / "runner.log"),
           "done": str(rd / "DONE"), "result": str(rd / "RESULT.md"), "base": a.base, "recipe": str(recipe),
           "gpus": gpus}
    (rd / f"train-s{seed}.log").write_text("prepare ok\ntrain [##....] 5/32 (15.6%) loss 0.1 eta 0.2h\n")
    print("prepare: ok", file=sys.stderr)
    print(json.dumps(out, indent=2), flush=True)
    if os.environ.get("FAKE_OJH_SLEEP"):
        (Path(os.environ["FAKE_OJH_OUT"]) / "sleeping").touch()
        time.sleep(float(os.environ["FAKE_OJH_SLEEP"]))
    if os.environ.get("FAKE_OJH_PENDING") == "1":
        return 0
    finish(rd, exp, seed, mode, key, quality, len(gpus), cards)
    return 0


def finish(rd: Path, exp: dict, seed: int, mode: str, key: str, quality, n_gpus: int, cards: list) -> None:
    """Write what the harness writes at the end of a run (tests call it to finish a pending run)."""
    if quality == "fail":
        (rd / "runs" / f"{exp['name']}-s{seed}" / "ckpt-step400").mkdir(parents=True)   # a save point before it failed
        (rd / "timing.json").write_text(json.dumps({"seeds": {str(seed): {"seconds": 3600}}}))
        (rd / "DONE").write_text(json.dumps({"exit_code": 1, "stage": "train", "error": "CUDA out of memory"}))
        return
    seed_dir = Path("runs") / f"{exp['name']}-s{seed}"
    ckpt = seed_dir / ("ckpt-step800" if mode == "full" else "adapter-step800")
    model = {"quality": quality, "key": key, "seed": seed}
    (rd / ckpt).mkdir(parents=True)
    (rd / ckpt / "fake_model.json").write_text(json.dumps({**model, "adapter_only": mode != "full"}))
    gone = [rd / seed_dir / ("ckpt-step400" if mode == "full" else "adapter-step400")]   # unselected
    gone[0].mkdir()
    if exp.get("merge"):
        (rd / seed_dir / "merged-step800").mkdir()
        (rd / seed_dir / "merged-step800" / "fake_model.json").write_text(json.dumps(model))
        gone.append(rd / seed_dir / "merged-step800")
    (rd / "deletion_candidates.json").write_text(json.dumps({"candidates": [{"path": str(d), "bytes": 1} for d in gone]}))
    (rd / "result.json").write_text(json.dumps({
        "wall_time": {"seconds": 7200, "gpu_hours": 2.0 * n_gpus},
        "gpus": {"n": n_gpus, "cards": cards},
        "seeds": {str(seed): {"selected_step": 800, "checkpoint": str(ckpt), "checkpoint_sha256": "ab" * 32,
                              "curve": [{"step": s, "nll": 0.1} for s in (400, 800, 1200)]}}}))
    (rd / "DONE").write_text(json.dumps({"exit_code": 0, "stage": "finalize"}))


def finish_pending(rd, quality=1.0) -> None:
    """Finish a run left unfinished by FAKE_OJH_PENDING: a success of strength `quality`, or "fail"."""
    rd = Path(rd)
    lock = yaml.safe_load((rd / "run.lock.yaml").read_text())
    cards = json.loads((rd / "gpus.json").read_text())
    finish(rd, {"name": lock["experiment"], "merge": lock["merge"]}, lock["seeds"][0], lock["mode"],
           lock["base"]["name"], quality, len(cards), cards)


if __name__ == "__main__":
    sys.exit(main())
