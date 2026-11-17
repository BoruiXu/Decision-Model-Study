"""Resolve an experiment into ``run.lock.yaml``: the only thing the runner executes. Imports without torch."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml

from .catalog import ROOT, load_base, sha256_file
from .config import ConfigError, ExperimentConfig, MachineConfig
from .gpu import Gpu, _nvidia_smi, parse_nvidia_smi
from .onejev import load_lock

# Initial memory-estimate constants (design section 8); recalibrate after the first real runs.
FULL_BYTES_PER_PARAM = 16
LORA_BYTES_PER_PARAM = 2
LORA_OVERHEAD = 1.15
FSDP_MIN_PARAMS_B = 9

_RECIPE_TOP = {"mode", "gpus", "fsdp", "onejev", "lora"}
_OVERRIDE_TOP = {"mode", "gpus", "fsdp"}


class Infeasible(RuntimeError):
    """The recipe cannot run on the available cards; ``alternatives`` lists concrete options."""

    def __init__(self, message: str, alternatives: list[str] | None = None):
        self.alternatives = list(alternatives or [])
        super().__init__(message + ("\nalternatives:\n- " + "\n- ".join(self.alternatives)
                                    if self.alternatives else ""))


def estimate_gib(params_b: float, mode: str, n_gpus: int, fsdp: bool) -> float:
    """Rough per-card memory need in GiB."""
    if mode == "full":
        total = params_b * FULL_BYTES_PER_PARAM
        return total / n_gpus if fsdp else total
    return params_b * LORA_BYTES_PER_PARAM * LORA_OVERHEAD


def parallel_for(mode: str, params_b: float, n_gpus: int, requested_fsdp: bool | None) -> str:
    if mode == "lora":
        if requested_fsdp:
            raise Infeasible("OneJev's --fsdp shards a full fine-tune; LoRA cannot use FSDP",
                             ["use mode: full with fsdp", "use LoRA with DDP (fsdp: false)"])
        return "single" if n_gpus == 1 else "ddp"
    if n_gpus == 1:
        return "single"
    fsdp = requested_fsdp if requested_fsdp is not None else params_b >= FSDP_MIN_PARAMS_B
    return "fsdp" if fsdp else "ddp"


def parse_plan_only(stdout: str) -> int:
    if "PLAN_OK" not in stdout:
        raise ValueError(f"plan-only did not report PLAN_OK:\n{stdout[-500:]}")
    m = re.search(r"(\d+) steps for (\d+) ranks", stdout)
    if not m:
        raise ValueError(f"no '<n> steps for <m> ranks' line in plan-only output:\n{stdout[-500:]}")
    return int(m.group(1))


def _plan_ranks(stdout: str) -> int:
    return int(re.search(r"(\d+) steps for (\d+) ranks", stdout).group(2))


def eval_interval(steps: int) -> int:
    return max(1, math.ceil(steps / 8))


def _kebab(name: str) -> str:
    return "--" + name.replace("_", "-")


def _flag_tokens(flags: dict) -> list[str]:
    out: list[str] = []
    for k, v in flags.items():
        if v is False or v is None:
            continue
        out.append(_kebab(k))
        if v is True:
            continue
        out.extend(str(x) for x in v) if isinstance(v, (list, tuple)) else out.append(str(v))
    return out


def _tail(lock: dict, seed: int) -> list[str]:
    d, name = lock["data"], f"{lock['experiment']}-s{seed}"
    return ["--eval", f"select={d['select']}", "--seed", str(seed), "--name", name,
            "--runs-dir", str(Path(lock["run_dir"]) / "runs"), "--train", d["train"],
            "--model", str(Path(lock["machine"]["model_dir"]) / lock["base"]["name"])]


def build_command(lock: dict, seed: int) -> list[str]:
    """The training command for one seed; needs nothing but the lock."""
    py, n = lock["machine"]["python"], lock["n_gpus"]
    head = [py, "-m"]
    if n > 1:
        head += ["torch.distributed.run", "--standalone", "--nproc_per_node", str(n), "-m"]
    return head + ["harness.train_entry"] + _flag_tokens(lock["flags"]) + _tail(lock, seed)


def _default_plan_runner(lock: dict) -> str:
    m = lock["machine"]
    cmd = [m["python"], "-m", "train.sft"] + _flag_tokens(lock["flags"]) + _tail(lock, lock["seeds"][0]) \
        + ["--plan-only"]
    env = {**os.environ, **m["env"], "PLAN_WORLD": str(lock["n_gpus"]),
           "PYTHONPATH": str(Path(lock["repo_root"]) / "vendor" / "OneJev")}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"plan-only failed ({proc.returncode}):\n{(proc.stdout + proc.stderr)[-1500:]}")
    return proc.stdout


def _fingerprint(machine: MachineConfig) -> dict:
    freeze = subprocess.run([machine.python, "-m", "pip", "freeze"], check=True,
                            capture_output=True, text=True).stdout
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True).stdout.split()
    return {"pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
            "driver_version": drv[0] if drv else None}


def _load_recipe(path: str) -> dict:
    with open(path) as f:
        r = yaml.safe_load(f)
    if not isinstance(r, dict):
        raise ConfigError(f"{path}: recipe must be a mapping")
    unknown = set(r) - _RECIPE_TOP
    if unknown:
        raise ConfigError(f"{path}: unknown recipe key(s): {sorted(unknown)}")
    return r


def _apply_overrides(recipe: dict, overrides: dict) -> dict:
    recorded = {}
    for key, spec in (overrides or {}).items():
        if not isinstance(spec, dict) or "value" not in spec:
            raise ConfigError(f"override {key}: expected {{value: ..., reason: ...}}")
        if not str(spec.get("reason") or "").strip():
            raise ConfigError(f"override {key}: a non-empty reason is required")
        head, _, rest = key.partition(".")
        if head in ("onejev", "lora") and rest:
            recipe.setdefault(head, {})[rest] = spec["value"]
        elif key in _OVERRIDE_TOP:
            recipe[key] = spec["value"]
        else:
            raise ConfigError(f"override {key}: key must be mode, gpus, fsdp, onejev.<flag> or lora.<field>")
        recorded[key] = {"value": spec["value"], "reason": spec["reason"]}
    return recorded


def _candidate_cards(machine: MachineConfig, n: int, query) -> list[Gpu] | None:
    """Free-memory view of the cards the run would get; None when unknown."""
    if machine.gpu_source == "preassigned":
        ids = [p.strip() for p in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if p.strip()][:n]
        try:
            known = parse_nvidia_smi(query())
        except Exception:
            return None
        cards = [next((g for g in known if g.uuid == i or str(g.index) == i), None) for i in ids]
        return cards if cards and all(cards) else None
    cards = [g for g in parse_nvidia_smi(query())
             if (machine.gpu_include is None or g.index in machine.gpu_include)
             and g.index not in machine.gpu_exclude]
    cards.sort(key=lambda g: g.free_mib, reverse=True)
    return cards


def _check_feasible(machine, base, mode, n, parallel, query) -> float:
    need = estimate_gib(base["params_b"], mode, n, parallel == "fsdp")
    cards = _candidate_cards(machine, n, query)
    if cards is None:
        return need
    chosen = cards[:n]
    if len(chosen) < n:
        raise Infeasible(f"need {n} GPUs but only {len(chosen)} candidate card(s) after filtering",
                         [f"wait or free cards (or relax gpu_include/gpu_exclude)",
                          f"use fewer GPUs (gpus: {len(chosen)})" if chosen else "use another machine"])
    free = min(g.free_mib for g in chosen) / 1024
    if need <= free:
        return need
    p = base["params_b"]
    alts = []
    if mode == "full":
        if parallel == "fsdp":
            total = p * FULL_BYTES_PER_PARAM
            alts.append(f"more GPUs: full FSDP needs >= {math.ceil(total / free)} cards of {free:.0f} GiB")
        lora = estimate_gib(p, "lora", n, False)
        alts.append(f"lora on {n} GPUs needs ~{lora:.0f} GiB/card"
                    + ("" if lora <= free else " (still above the free memory)"))
    else:
        alts.append(f"cards with >= {need:.0f} GiB free (now {free:.0f} GiB on the best {n})")
    alts.append("smaller base")
    raise Infeasible(f"{mode} of {base['name']} on {n} GPU(s) ({parallel}) needs ~{need:.0f} GiB/card; "
                     f"the best {n} candidate cards have {free:.0f} GiB free", alts)


def _sha(path: Path) -> str:
    return sha256_file(path)


def resolve(exp: ExperimentConfig, machine: MachineConfig, *, query=_nvidia_smi, plan_runner=None,
            fingerprint=None, now: dt.datetime | None = None) -> Path:
    recipe = copy.deepcopy(_load_recipe(exp.recipe))
    if exp.gpus is not None:
        recipe["gpus"] = exp.gpus
    recorded = _apply_overrides(recipe, exp.overrides)
    mode = recipe.get("mode")
    if mode not in ("full", "lora"):
        raise ConfigError(f"recipe {exp.recipe}: 'mode' is required and must be full or lora (got {mode!r})")
    n = recipe.get("gpus")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ConfigError("gpus must be a positive integer (recipe or experiment)")

    base = load_base(exp.base)
    parallel = parallel_for(mode, base["params_b"], n, recipe.get("fsdp"))
    need = _check_feasible(machine, base, mode, n, parallel, query)

    prepared = Path(machine.out_root) / exp.name / "prepared"
    info_path = prepared / "prepared.json"
    if not info_path.is_file():
        raise ConfigError(f"no prepared data at {prepared}; run `ojh prepare` first")
    info = json.loads(info_path.read_text())
    data = {"prepared_dir": str(prepared), "source": exp.data.train,
            "sha256": {f: _sha(prepared / f) for f in ("train.jsonl", "select.jsonl", "select_index.json")},
            "source_sha256": info["train_sha256_source"], "subset": info.get("subset"),
            "counts": info.get("counts")}
    for f in ("train", "select", "select_index"):
        data[f] = str(prepared / (f + (".json" if f == "select_index" else ".jsonl")))

    flags = dict(recipe.get("onejev") or {})
    if mode == "full":
        flags["full_finetune"] = True
    if parallel == "fsdp":
        flags["fsdp"] = True
    lora = dict(recipe.get("lora") or {}) if mode == "lora" else None
    for k, v in (lora or {}).items():
        flags[f"lora_{k}"] = v
    flags["attention"], flags["readout"] = base["attention"], base["readout"]
    if base.get("multimodal"):
        flags["multimodal"] = True
    flags["eval_at_start"] = True

    out_root = Path(machine.out_root)
    lock = {
        "experiment": exp.name, "repo_root": str(ROOT),
        "run_dir": str(out_root / exp.name / "plan"),
        "machine": {"python": machine.python, "out_root": machine.out_root, "model_dir": machine.model_dir,
                    "registry": machine.registry_path, "env": dict(machine.env),
                    "gpu": {"include": machine.gpu_include, "exclude": machine.gpu_exclude,
                            "min_free_mib": machine.min_free_mib, "lock_dir": machine.lock_dir,
                            "order": machine.order, "wait_timeout_s": machine.wait_timeout_s,
                            "source": machine.gpu_source},
                    "forbidden_write_roots": list(machine.forbidden_write_roots)},
        "mode": mode, "lora": lora, "n_gpus": n, "parallel": parallel, "seeds": list(exp.seeds),
        "estimate_gib": round(need, 1), "flags": flags, "data": data,
        "eval_sets": {k: asdict(v) for k, v in exp.data.eval.items()},
        "split_file": exp.data.split_file, "fields": asdict(exp.fields),
        "evaluator": asdict(exp.evaluator), "merge": exp.merge,
        "base": {"name": exp.base, **{k: v for k, v in base.items() if k != "name"}},
        "onejev": load_lock(), "overrides": recorded,
    }

    out = (plan_runner or _default_plan_runner)(lock)
    steps = parse_plan_only(out)
    ranks = _plan_ranks(out)
    if ranks != n:
        raise RuntimeError(f"plan-only reported {ranks} ranks, expected {n}")
    interval = eval_interval(steps)
    flags["eval_every"] = flags["save_every"] = interval
    lock["plan"] = {"steps": steps, "ranks": ranks, "interval": interval}
    lock["fingerprint"] = (fingerprint or (lambda: _fingerprint(machine)))()

    body = json.dumps(lock, sort_keys=True, default=str)
    stamp = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{hashlib.sha256(body.encode()).hexdigest()[:8]}"
    run_dir = out_root / exp.name / run_id
    lock["run_id"], lock["run_dir"] = run_id, str(run_dir)
    run_dir.mkdir(parents=True)
    path = run_dir / "run.lock.yaml"
    path.write_text(yaml.safe_dump(lock, sort_keys=False))
    return path
