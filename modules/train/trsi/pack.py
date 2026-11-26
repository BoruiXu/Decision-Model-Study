"""A finished ojh trial as a framework candidate, and the numbers trsi compares it by.

build() writes <outdir>/cand-<trial>/:
  model/                 the selected checkpoint (LoRA: the merged model ojh wrote next to it), hard-linked (copied
                         when the files are on another file system), so `ojh cleanup` of the run leaves it whole
  predict, predict.py    model.predict: `predict --inputs <inputs.jsonl> --out <predictions.jsonl>` runs predict.py in
  predict.json           the ojh .venv python written at packing time; predict.py scores model/ with
                         harness.score.score_checkpoint and the settings in predict.json (those of ojh's own rescoring)
  predictions_val.jsonl  predict on the valset's inputs.jsonl
  policy.json            with the task's policy section only: the decision policy (modules/common/dpolicy) fitted on
  policy_report.json     the raw valset predictions (predictions_val_raw.jsonl), every pair's numbers, and dpolicy's
  policy_apply.py        self-contained apply.py, through which predict.py applies it; predictions_val.jsonl then
  predictions_val_raw.jsonl  holds the rows with the policy applied, exactly what predict writes
  manifest.json          metrics_val, baselines_val and train_gate from `rsi score`; config (run.lock.yaml without its
                         paths to labelled data, see config_of), config_request, cost (GPU hours + scoring time),
                         code_version (modules/train/train)
and checks it with `rsi validate candidate`. model.predict scores on the card collect picks (CUDA_VISIBLE_DEVICES, by
uuid; scoring_needs says what the card needs). A trial that cannot be packed leaves no directory (PackError; PackOOM
when the card ran out of memory); its checkpoint stays in the ojh run, and the next collect packs it again.

The comparison numbers of a candidate (metrics()) are its metrics_val plus the ECE of its predictions_val on the valset
rows; adopt() reads the same from a registered candidate. Only the trainset and the valset are read.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .candidate import Candidate, config_request, file_sha256
from .config import RunConfig, rows_file
from .harness import TrialResult, scoring_lock_settings
from .metrics import ece

REPO = Path(__file__).resolve().parents[3]    # the framework repository (rsi score, rsi validate)

PREDICT_SH = """#!/bin/sh
# model.predict of a trsi candidate: predict --inputs <inputs.jsonl> --out <predictions.jsonl>
exec {python} "$(dirname "$0")/predict.py" "$@"
"""

PREDICT_PY = r'''"""model.predict of a trsi candidate (run through ./predict, in the ojh .venv python written at packing time).

predict --inputs <inputs.jsonl> --out <predictions.jsonl>
Reads each input row's id, state and question (a row without target gets a uniform placeholder target, which
OneJev's scoring path expects and does not use for the probabilities); scores the checkpoint in model/ beside this
file with harness.score.score_checkpoint, using the settings in predict.json (and the machine environment ojh trained
with); writes {"id", "probs", "decision", "uncertainty"}: decision is the most probable label, uncertainty 1 - its
probability. When policy.json is beside this file, each row goes through its decision policy (policy_apply.py, loaded
by path: calibrated probabilities, the policy's decision). Card: CUDA_VISIBLE_DEVICES, else the idle card (under 1 GiB
in use) nvidia-smi shows with the least memory in use, by uuid (a card without memory figures, [N/A], is not idle);
with nvidia-smi and no idle card it stops.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))


def scorer(cfg):
    path = cfg.get("test_scorer")  # tests only: a stand-in for harness.score, written at packing time
    if path:
        spec = importlib.util.spec_from_file_location("trsi_test_scorer", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.score_checkpoint
    from harness.score import score_checkpoint
    return score_checkpoint


def idle_card():
    """The uuid of the idle card with the least memory in use; None without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    cards = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            used = float(parts[2]) if len(parts) == 3 and parts[0].isdigit() else None
        except ValueError:  # [N/A], e.g. MIG
            used = None
        if used is not None and used < 1024:
            cards.append((used, parts[1]))
    if not cards:
        raise SystemExit("predict: no idle card (under 1 GiB in use); set CUDA_VISIBLE_DEVICES")
    return min(cards)[1]


def with_policy(rows):
    """The rows with the decision policy in policy.json beside this file applied (policy_apply.py), when it is there."""
    path = os.path.join(HERE, "policy.json")
    if not os.path.isfile(path):
        return rows
    sys.dont_write_bytecode = True   # no __pycache__ in the candidate directory
    spec = importlib.util.spec_from_file_location("trsi_policy_apply", os.path.join(HERE, "policy_apply.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    policy = mod.load_policy(path)
    return [mod.apply_policy(policy, r, policy["labels"]) for r in rows]


def labels_of(question):
    kind, crit = question.get("type"), question.get("criteria")
    if kind == "noul":
        return ["yes", "no"]
    if kind == "score":
        return [str(i) for i in range(len(crit))]
    return list(crit)


def main():
    ap = argparse.ArgumentParser(prog="predict")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(os.path.join(HERE, "predict.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    os.environ.update((k, str(v)) for k, v in cfg["env"].items() if k != "CUDA_VISIBLE_DEVICES")  # as ojh's finalize
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        card = idle_card()
        if card is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = card
    with open(a.inputs, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    feed = []
    for r in rows:
        x = dict(r)
        if "target" not in x:
            labels = labels_of(r["question"])
            x["target"] = dict((lab, 1.0 / len(labels)) for lab in labels)
        feed.append(x)
    recs = scorer(cfg)(os.path.join(HERE, "model"), feed, base_dir=None, device="cuda:0", **cfg["kwargs"])
    out = []
    for r in rows:
        rec = recs[r["id"]]
        probs = dict(zip(rec["labels"], (float(p) for p in rec["probs"])))
        total = sum(probs.values())
        probs = dict((k, v / total) for k, v in probs.items())
        dec = max(probs, key=probs.get)
        out.append({"id": r["id"], "probs": probs, "decision": dec, "uncertainty": 1.0 - probs[dec]})
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in with_policy(out):
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, a.out)


main()
'''


class PackError(RuntimeError):
    pass


class PackOOM(PackError):
    """model.predict ran out of memory on the card `device` (uuid): another card may do."""

    def __init__(self, msg: str, device: str):
        super().__init__(msg)
        self.device = device


OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED")


def _jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_kwargs(lock: dict) -> dict:
    """The forward-pass settings ojh's finalize rescores the checkpoint with (harness.finalize.score_kwargs)."""
    f = lock.get("flags") or {}
    full = bool(f.get("full_finetune"))
    return {"multimodal": bool((lock.get("base") or {}).get("multimodal")),
            "dtype": "float32" if full and not f.get("fsdp") else (f.get("dtype") or "bfloat16"),
            "autocast": bool(f.get("lora_autocast") or (full and not f.get("no_autocast"))),
            "disable_fla_kernels": bool(f.get("disable_fla_kernels")),
            "chat_template": f.get("chat_template"), "media_root": f.get("media_root")}


def rsi(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "rsi", *map(str, args)], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": str(REPO)})


def _model_dir(run_dir: Path, lock: dict, res: TrialResult) -> Path:
    if not res.checkpoint:
        raise PackError(f"{run_dir}: result.json names no selected checkpoint")
    ckpt = run_dir / res.checkpoint
    if lock.get("mode") == "lora":
        merged = ckpt.parent / f"merged-step{res.selected_step}"
        if not merged.is_dir():
            raise PackError(f"{merged}: a LoRA trial needs the merged model (merge: true in its exp.yaml)")
        return merged
    if not ckpt.is_dir():
        raise PackError(f"{ckpt}: the selected checkpoint is missing")
    return ckpt


def _dir_mib(d: Path) -> float:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.stat(os.path.join(root, f)).st_size
    return total / 2 ** 20


def scoring_needs(run: RunConfig, res: TrialResult) -> tuple[int, int, Path | None]:
    """What scoring finished trial `res` asks of a card (harness.pick_scoring_card): need_mib, 1.5 x the size of the
    model it packs; min_free_mib and lock_dir, from its run.lock.yaml's machine.gpu, else the machine config
    run.ojh.machine (lock_dir relative to that file), else 0 and None."""
    run_dir = Path(res.run_dir)
    try:
        lock = yaml.safe_load((run_dir / "run.lock.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise PackError(f"{run_dir / 'run.lock.yaml'}: {e}") from None
    need = math.ceil(1.5 * _dir_mib(_model_dir(run_dir, lock, res)))
    return (need,) + scoring_lock_settings(((lock.get("machine") or {}).get("gpu")) or {}, run.ojh.machine)


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:   # another file system (or links not allowed)
        shutil.copy2(src, dst)


def config_of(lock: dict) -> dict:
    """The manifest's config: the run's lock, without its paths to labelled data (deliver reads the config): data's
    train, select, select_index, prepared_dir and source, every eval set's train_labels and reference, and split_file
    are emptied (null); their sha256 stay."""
    cfg = json.loads(json.dumps(lock, default=str))
    data = cfg.get("data")
    for d, keys in ((data, ("train", "select", "select_index", "prepared_dir", "source")),
                    *((s, ("train_labels", "reference")) for s in (cfg.get("eval_sets") or {}).values())):
        if isinstance(d, dict):
            d.update((k, None) for k in keys if k in d)
    if "split_file" in cfg:
        cfg["split_file"] = None
    return cfg


def _hard(target: dict, labels: list) -> str:
    return max(labels, key=lambda lab: float(target.get(lab, 0)))


def metrics(run: RunConfig, manifest: dict, cand: Path) -> dict:
    """metrics_val plus the ECE of predictions_val on the valset rows (positive label: the task's)."""
    task = run.framework.task_json()
    labels, pos = task["decision"]["labels"], task["decision"]["positive"]
    rows = {r["id"]: r for r in _jsonl(rows_file(run.framework.valset))}
    preds = _jsonl(cand / manifest["files"]["predictions_val"]["path"])
    y = [int(_hard(rows[p["id"]]["target"], labels) == pos) for p in preds]
    return {**manifest["metrics_val"], "ece": ece(y, [float(p["probs"][pos]) for p in preds])}


def _policy_tools():
    """dpolicy's apply module and fit_policy, and rsi.rules.policy_settings: <REPO>/modules/common and <REPO> go on
    sys.path first (trsi collect runs with only modules/train on it; modules/common ahead of anything installed)."""
    for p in (str(REPO), str(REPO / "modules" / "common")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    from dpolicy import apply as policy_apply
    from dpolicy.fit import fit_policy
    from rsi.rules import policy_settings
    return policy_apply, fit_policy, policy_settings


def fit_policy_into(fw, task: dict, dest: Path, pv: Path) -> dict | None:
    """With the task's policy section (rsi.rules.policy_settings), fit the decision policy on model.predict's valset
    predictions `pv` (kept as predictions_val_raw.jsonl), write policy.json, its fit report (every pair's numbers) as
    policy_report.json and dpolicy's apply.py as policy_apply.py, and write the rows with the policy applied as `pv`,
    line by line as predict.py writes them. Returns the policy; None (nothing changed) without a policy section."""
    if task.get("policy") is None:
        return None
    policy_apply, fit_policy, policy_settings = _policy_tools()
    settings = policy_settings(task)
    raw = dest / "predictions_val_raw.jsonl"
    os.replace(pv, raw)
    preds = _jsonl(raw)
    policy, report = fit_policy(task, settings, _jsonl(rows_file(fw.valset)), preds)
    (dest / "policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    (dest / "policy_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(policy_apply.__file__, dest / "policy_apply.py")
    with open(pv, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(policy_apply.apply_policy(policy, r, policy["labels"]), sort_keys=True) + "\n")
    return policy


def _gate(task: dict, sc: dict) -> dict:
    primary, higher = task["metrics"]["primary"], task["metrics"]["higher_is_better"]
    if primary not in sc["metrics"]:
        raise PackError(f"rsi score did not compute the primary metric {primary}")
    mine = sc["metrics"][primary]
    base = {b: v[primary] for b, v in sorted(sc["baselines"].items()) if primary in v}
    behind = [b for b, v in base.items() if not (mine > v if higher else mine < v)]
    shown = ", ".join(f"{b} {base[b]:.4f}" for b in (behind or base))
    return {"passed": not behind,
            "detail": f"val {primary} {mine:.4f} " + ("does not beat " if behind else "beats ") + (shown or "no baseline")}


def build(run: RunConfig, rec: dict, res: TrialResult, c: Candidate, start: bool, dest: Path, device: str) -> dict:
    """Pack trial `rec` (its ojh run finished: `res`; `start`: a trial of the start) into `dest`, scoring on
    CUDA_VISIBLE_DEVICES=`device`; returns {"metrics", "compute_hours", "dir"}."""
    if dest.exists():
        shutil.rmtree(dest)
    try:
        return _build(run, rec, res, c, start, dest, device)
    except PackOOM:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise PackError(str(e)) from None


def _build(run: RunConfig, rec: dict, res: TrialResult, c: Candidate, start: bool, dest: Path, device: str) -> dict:
    fw = run.framework
    task = fw.task_json()
    run_dir = Path(res.run_dir)
    lock = yaml.safe_load((run_dir / "run.lock.yaml").read_text())
    src = _model_dir(run_dir, lock, res)
    dest.mkdir(parents=True)
    shutil.copytree(src, dest / "model", copy_function=_link_or_copy)
    env = {k: str(v) for k, v in ((lock.get("machine") or {}).get("env") or {}).items()}
    cfg = {"kwargs": score_kwargs(lock), "env": env, "trsi": __version__}
    if os.environ.get("TRSI_TEST_SCORER"):   # tests only; model.predict reads nothing else from the environment
        cfg["test_scorer"] = os.environ["TRSI_TEST_SCORER"]
    (dest / "predict.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dest / "predict.py").write_text(PREDICT_PY)
    entry = dest / "predict"
    entry.write_text(PREDICT_SH.format(python=shlex.quote(fw.python)))
    entry.chmod(entry.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    vman = json.loads((fw.valset / "manifest.json").read_text())
    inputs = fw.valset / vman["files"]["inputs"]["path"]
    pv = dest / "predictions_val.jsonl"
    t0 = time.time()
    p = subprocess.run([str(entry), "--inputs", str(inputs), "--out", str(pv)], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, env={**os.environ, "CUDA_VISIBLE_DEVICES": device})
    score_s = time.time() - t0
    if p.returncode != 0:
        said = (p.stderr or p.stdout).strip()
        if OOM_RE.search(p.stderr + p.stdout):   # the whole output: the last lines may be another error after it
            raise PackOOM(f"model.predict ran out of memory on {device} (exit {p.returncode}): {said[-1500:]}", device)
        raise PackError(f"model.predict failed (exit {p.returncode}): {said[-1500:]}")
    policy = fit_policy_into(fw, task, dest, pv)
    files = {"predictions_val": {"path": pv.name, "rows": len(_jsonl(pv))}}
    if policy is not None:
        files["predictions_val_raw"] = {"path": "predictions_val_raw.jsonl", "rows": files["predictions_val"]["rows"]}
        files["policy"] = {"path": "policy.json"}
        files["policy_report"] = {"path": "policy_report.json"}
    sc = rsi("score", "--task", fw.task, "--labels", rows_file(fw.valset), "--predictions", pv,
             "--train", rows_file(fw.trainset))
    if sc.returncode != 0:
        raise PackError(f"rsi score failed: {(sc.stderr or sc.stdout).strip()[-1500:]}")
    sc = json.loads(sc.stdout)
    seed_info = (json.loads((run_dir / "result.json").read_text()).get("seeds") or {}).get(str(rec["seed"])) or {}
    sha = None if lock.get("mode") == "lora" else seed_info.get("checkpoint_sha256")   # LoRA: the merged model
    hours = float(res.gpu_hours) + score_s / 3600
    manifest = {
        "kind": "candidate", "id": None, "task_id": task["task_id"], "task_version": task["version"],
        "round": fw.round, "producer": fw.producer,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parents": [x for x in (fw.trainset_id, fw.valset_id) if x != "unregistered"],
        "files": files,
        "notes": f"trsi {__version__}: trial {rec['trial']} ({c.label()}, seed {rec['seed']}), ojh run {run_dir}, "
                 f"selected step {res.selected_step}" + ("" if policy is None else
                                                         f"; decision policy {policy['calibration']['method']} + "
                                                         f"{policy['decision']['rule']} "
                                                         f"({'accepted' if policy['accepted'] else 'the identity'})"),
        "trainset_id": fw.trainset_id, "valset_id": fw.valset_id,
        "config": config_of(lock), "config_source": fw.start_source if start else "tune",
        "config_request": config_request(c, rec["seed"], run.space.recipe_names),
        "code_version": {"entry": str(fw.train_command), "sha256": file_sha256(fw.train_command)},
        "model": {"location": "model", "sha256": sha if isinstance(sha, str) else None,
                  "load": f"{lock.get('mode')} checkpoint of {(lock.get('base') or {}).get('name')} in model/, loaded "
                          "by harness.score (ojh) with the settings in predict.json", "predict": "predict"},
        "metrics_val": sc["metrics"], "baselines_val": sc.get("baselines") or {}, "train_gate": _gate(task, sc),
        "cost": {"compute_hours": hours},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    v = rsi("validate", "candidate", dest, "--task", fw.task)
    if v.returncode != 0:
        raise PackError(f"rsi validate candidate: {(v.stderr or v.stdout).strip()[-1500:]}")
    return {"metrics": metrics(run, manifest, dest), "compute_hours": hours, "dir": str(dest)}
