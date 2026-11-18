"""Resolve an experiment into ``run.lock.yaml``: the only thing the runner executes. Imports without torch."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import yaml

from .catalog import ROOT, load_base, sha256_file
from .config import ConfigError, ExperimentConfig, MachineConfig, check_writable
from .gpu import Gpu, GpuUnavailable, _nvidia_smi, _nvidia_smi_apps, check_explicit, parse_nvidia_smi
from .onejev import load_lock

# Initial memory-estimate constants (design section 8); recalibrate after the first real runs.
FULL_BYTES_PER_PARAM = 16
LORA_BYTES_PER_PARAM = 2
LORA_OVERHEAD = 1.15
FSDP_MIN_PARAMS_B = 9

_RECIPE_TOP = {"mode", "gpus", "fsdp", "onejev", "lora"}
_OVERRIDE_TOP = {"mode", "gpus", "fsdp"}
# flags the harness derives itself; they may not appear under ``onejev`` (recipe or override)
_RESERVED_FLAGS = {"full_finetune": "mode", "fsdp": "fsdp", "attention": "the base catalog",
                   "readout": "the base catalog", "multimodal": "the base catalog",
                   "eval_every": "the plan-only interval", "save_every": "the plan-only interval",
                   "eval_at_start": "the harness", "seed": "seeds", "name": "the experiment name",
                   "train": "the prepared data", "model": "the base", "eval": "the prepared data",
                   "runs_dir": "the run directory", "config": "the lock (a OneJev --config file could set any flag)",
                   "plan_only": "resolve", "device": "the leased cards", "merge": "the experiment's merge key",
                   "merged_dir": "the experiment's merge key", "worst_case": "the harness",
                   "dump_first_grad": "the harness", "eval_limit": "the prepared data",
                   "dist_backend": "the harness"}


def _check_reserved(flags: dict, where: str) -> None:
    for k in flags:
        if k in _RESERVED_FLAGS:
            raise ConfigError(f"{where}: onejev.{k} is set by the harness (from {_RESERVED_FLAGS[k]}); "
                              "use the proper key instead")


DATA_FILES = ("train.jsonl", "select.jsonl", "select_index.json")


class DataChanged(RuntimeError):
    """The prepared files no longer match the sha256 recorded in the lock."""


def data_changes(lock: dict) -> list[str]:
    """Prepared files whose sha256 differs from the lock; a re-run of prepare overwrites them in place."""
    d = lock.get("data") or {}
    want, root = d.get("sha256") or {}, d.get("prepared_dir")
    if not root or not want:
        return ["the lock records no data.prepared_dir / data.sha256"]
    out = []
    for name in DATA_FILES:
        p = Path(root) / name
        if name not in want:
            out.append(f"{name}: no sha256 in the lock")
        elif not p.is_file():
            out.append(f"{p}: missing")
        else:
            got = sha256_file(p)
            if got != want[name]:
                out.append(f"{p}: sha256 {got[:12]}… differs from the lock's {want[name][:12]}…")
    return out


def check_data(lock: dict) -> None:
    changes = data_changes(lock)
    if changes:
        raise DataChanged("prepared data changed after resolve (re-run `ojh resolve` for a new lock): "
                          + "; ".join(changes))


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
        out.append(",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v))
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
    # cwd: the repo, so a train/ package in the caller's directory cannot shadow OneJev's
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=lock["repo_root"])
    if proc.returncode:
        raise RuntimeError(f"plan-only failed ({proc.returncode}):\n{(proc.stdout + proc.stderr)[-1500:]}")
    return proc.stdout


# Lists the training python's distributions without pip (a venv made by `uv venv` has none).
_PACKAGES_SCRIPT = ("import importlib.metadata as m\n"
                    "names = {f\"{(d.metadata['Name'] or '').lower()}=={d.version}\" for d in m.distributions()}\n"
                    "print('\\n'.join(sorted(n for n in names if not n.startswith('=='))))")


def installed_packages(python: str) -> list[str]:
    """Sorted ``name==version`` of every distribution importable by ``python`` (importlib.metadata)."""
    out = subprocess.run([python, "-c", _PACKAGES_SCRIPT], check=True, capture_output=True, text=True).stdout
    return sorted(line for line in out.splitlines() if line.strip())


def _driver_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0] if out else None


SETUP_STAMP = ".ojh-setup.json"  # written into the venv by setup.sh: uv version, python, torch index, ...


def setup_stamp(python: str) -> dict | None:
    """setup.sh's record of how the venv of ``python`` (``<venv>/bin/python``, symlinks not resolved) was
    built, or None when there is none (a venv not made by setup.sh, or a bare interpreter)."""
    p = Path(python).parent.parent / SETUP_STAMP
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def _fingerprint(machine: MachineConfig) -> dict:
    """Environment fingerprint for the lock: the training python's packages (importlib.metadata, so it works
    in a venv without pip), the NVIDIA driver version (None without nvidia-smi) and setup.sh's stamp (uv
    version, python, torch index)."""
    pkgs = installed_packages(machine.python)
    return {"packages_sha256": hashlib.sha256("\n".join(pkgs).encode()).hexdigest(), "packages": len(pkgs),
            "packages_source": "importlib.metadata", "driver_version": _driver_version(),
            "setup": setup_stamp(machine.python)}


def _load_recipe(path: str) -> dict:
    with open(path) as f:
        r = yaml.safe_load(f)
    if not isinstance(r, dict):
        raise ConfigError(f"{path}: recipe must be a mapping")
    unknown = set(r) - _RECIPE_TOP
    if unknown:
        raise ConfigError(f"{path}: unknown recipe key(s): {sorted(unknown)}")
    _check_reserved(r.get("onejev") or {}, path)
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
            if head == "onejev":
                _check_reserved({rest: None}, f"override {key}")
            recipe.setdefault(head, {})[rest] = spec["value"]
        elif key in _OVERRIDE_TOP:
            recipe[key] = spec["value"]
        else:
            raise ConfigError(f"override {key}: key must be mode, gpus, fsdp, onejev.<flag> or lora.<field>")
        recorded[key] = {"value": spec["value"], "reason": spec["reason"]}
    return recorded


def _candidate_cards(machine: MachineConfig, n: int, query) -> tuple[list[Gpu] | None, str | None]:
    """Free-memory view of the cards the run would get, or (None, why it could not be determined)."""
    if machine.gpu_source == "preassigned":
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ids = [p.strip() for p in raw.split(",") if p.strip()]
        if len(ids) < n:
            raise Infeasible(f"gpu_source is 'preassigned' but CUDA_VISIBLE_DEVICES={raw!r} lists "
                             f"{len(ids)} device(s); need {n}",
                             ["set CUDA_VISIBLE_DEVICES to the assigned cards", f"use fewer GPUs (gpus: {len(ids)})"])
        try:
            known = parse_nvidia_smi(query())
        except Exception as e:
            return None, f"nvidia-smi query failed: {e}"
        cards = [next((g for g in known if g.uuid == i or str(g.index) == i), None) for i in ids[:n]]
        missing = [i for i, c in zip(ids, cards) if c is None]
        if missing:
            return None, f"CUDA_VISIBLE_DEVICES id(s) not found in nvidia-smi: {missing}"
        return cards, None
    cards = [g for g in parse_nvidia_smi(query())
             if (machine.gpu_include is None or g.index in machine.gpu_include)
             and g.index not in machine.gpu_exclude]
    cards.sort(key=lambda g: g.free_mib, reverse=True)
    return cards, None


GENERIC_LORA = ("examples/recipes/joey_lora_1gpu.yaml", "examples/recipes/mammoth_lora_ddp.yaml")  # 1 card, more
NO_SWITCH = "ojh never switches base, recipe or training mode by itself: ask the user which option to take"


def _lora_recipe(base: dict, n: int) -> str:
    """The base's LoRA default recipe for ``n`` cards (else any LoRA default it has), else a generic one."""
    from .catalog import parse_gpu_range
    lora = []
    for key, path in (base.get("default_recipes") or {}).items():
        r = yaml.safe_load((ROOT / path).read_text()) or {}
        if r.get("mode") == "lora":
            lo, hi = parse_gpu_range(key)
            lora.append((not (lo <= n and (hi is None or n <= hi)), path, key))
    if lora:
        _, path, key = sorted(lora)[0]
        return f"{path} (the base's default for {key} GPU(s))"
    return f"{GENERIC_LORA[0] if n == 1 else GENERIC_LORA[1]} (generic; its hyperparameters are not measured for " \
           f"{base['name']})"


def _memory_alternatives(base: dict, mode: str, n: int, parallel: str, need: float, free: float) -> list[str]:
    """Concrete options when ``need`` GiB per card does not fit in ``free`` GiB: more cards (full FSDP),
    LoRA (the base's LoRA recipe or a generic one), smaller bases that fit. Never applied automatically."""
    from .catalog import CATALOG_DIR
    p, alts = base["params_b"], []
    if mode == "full":
        total = p * FULL_BYTES_PER_PARAM
        k = math.ceil(total / free) if free > 0 else None
        if k is not None and k > n:
            alts.append(f"more GPUs: full FSDP over {k} cards needs ~{total / k:.0f} GiB/card "
                        f"(cards with >= {total / k:.0f} GiB free each)")
        lora = estimate_gib(p, "lora", n, False)
        alts.append(f"LoRA (a different training mode): {_lora_recipe(base, n)} needs ~{lora:.0f} GiB/card on "
                    f"{n} card(s)" + ("" if lora <= free else f" (still above the {free:.0f} GiB free)")
                    + "; only with the user's consent, via --recipe")
    else:
        alts.append(f"cards with >= {need:.0f} GiB free each (the best requested card has {free:.0f} GiB)")
    smaller = []
    for path in sorted(CATALOG_DIR.glob("*.yaml")):
        b = yaml.safe_load(path.read_text())
        if b["params_b"] >= p:
            continue
        par = parallel_for(mode, b["params_b"], n, None) if mode == "full" else "ddp"
        est = estimate_gib(b["params_b"], mode, n, par == "fsdp")
        if est <= free:
            smaller.append((b["params_b"], f"{b['name']} (~{est:.0f} GiB/card)"))
    alts.append("smaller base: " + (", ".join(x for _, x in sorted(smaller, reverse=True))
                                    if smaller else f"none of the catalog bases fits {free:.0f} GiB per card"))
    alts.append(NO_SWITCH)
    return alts


def _explicit_feasible(machine, explicit, need, query, apps_query, allow_shared=False, ctx=None) -> dict:
    need_mib = math.ceil(need * 1024)
    try:
        cards = check_explicit(machine, explicit, need_mib, query=query, apps_query=apps_query,
                               allow_shared=allow_shared)
    except GpuUnavailable as e:
        floor = max(need_mib, machine.min_free_mib)
        good = [c for c in e.cards if c["free_mib"] >= floor and c["index"] not in explicit
                and (allow_shared or c["processes"] == [])
                and c["index"] not in machine.gpu_exclude
                and (machine.gpu_include is None or c["index"] in machine.gpu_include)]
        alts = [("other cards with enough free memory and no other process now: "
                 + ", ".join(f"GPU {c['index']} ({c['name']}, {c['free_mib']} MiB free)" for c in good))
                if good else f"no other card has {floor} MiB free and no other process now"]
        by_index = {c["index"]: c for c in e.cards}
        if any(r.startswith("shared with") for r in e.reasons.values()):
            alts.append("ask the user whether the run may share the card(s) with the processes listed; only if "
                        "they agree, run again with --allow-shared (or wait until those processes end)")
        short = [by_index[i] for i, r in e.reasons.items() if "MiB needed" in r and i in by_index]
        for c in short:  # waiting helps only when the card would hold the run without the other processes
            if c["processes"] and c["free_mib"] + sum(x["used_mib"] or 0 for x in c["processes"]) >= floor:
                alts.append(f"GPU {c['index']} would have enough memory once the listed processes end "
                            "(they are not ours: wait, or ask their owner)")
        if short and ctx is not None:
            free = min(c["free_mib"] for c in (by_index[i] for i in explicit if i in by_index)) / 1024
            alts += _memory_alternatives(*ctx, need, free)
        else:
            alts.append(NO_SWITCH)
        raise Infeasible(str(e), alts) from None
    return {"checked": True, "need_gib_per_card": round(need, 1), "need_mib_per_card": cards[0]["need_mib"],
            "min_free_gib": round(min(c["free_mib"] for c in cards) / 1024, 1), "cards": cards}


def _check_feasible(machine, base, mode, n, parallel, query, explicit=None,
                    apps_query=_nvidia_smi_apps, allow_shared=False) -> tuple[float, dict]:
    need = estimate_gib(base["params_b"], mode, n, parallel == "fsdp")
    if explicit is not None:
        return need, _explicit_feasible(machine, explicit, need, query, apps_query, allow_shared,
                                        (base, mode, n, parallel))
    cards, why = _candidate_cards(machine, n, query)
    if cards is None:
        return need, {"checked": False, "reason": why}
    chosen = cards[:n]
    if len(chosen) < n:
        raise Infeasible(f"need {n} GPUs but only {len(chosen)} candidate card(s) after filtering",
                         [f"wait or free cards (or relax gpu_include/gpu_exclude)",
                          f"use fewer GPUs (gpus: {len(chosen)})" if chosen else "use another machine"])
    free = min(g.free_mib for g in chosen) / 1024
    if need <= free:
        return need, {"checked": True, "need_gib_per_card": round(need, 1), "min_free_gib": round(free, 1),
                      "cards": [g.uuid for g in chosen]}
    alts = _memory_alternatives(base, mode, n, parallel, need, free)
    raise Infeasible(f"{mode} of {base['name']} on {n} GPU(s) ({parallel}) needs ~{need:.0f} GiB/card; "
                     f"the best {n} candidate cards have {free:.0f} GiB free", alts)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _source_sha(path, what: str) -> str:
    if not Path(path).is_file():
        raise ConfigError(f"{what} not found: {path}")
    return sha256_file(path)


def _explicit_cards(exp: ExperimentConfig, machine: MachineConfig) -> list[int] | None:
    """Card indices when ``gpus`` lists them (``ojh run --gpus 0,1``); None for a plain count."""
    if isinstance(exp.gpus, list):
        if not exp.gpus or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in exp.gpus):
            raise ConfigError(f"gpus must be a count or a non-empty list of card indices (got {exp.gpus!r})")
        return list(exp.gpus)
    if machine.gpu_source == "explicit":
        raise ConfigError("gpu_source is 'explicit': list the cards with --gpus i,j,... (or gpus: [i, j] in "
                          "the experiment)")
    return None


def _check_prepared(exp: ExperimentConfig, info: dict, prepared: Path, source: str | None) -> None:
    """The prepared data must come from this experiment file and this train file, unchanged since."""
    owner = (info.get("experiment") or {}).get("path")
    if source and owner and os.path.abspath(source) != owner:
        raise ConfigError(f"{prepared} was prepared from {owner}, not from {source} (both use the experiment name "
                          f"{exp.name!r}); run `ojh prepare {source} --force` to prepare this experiment's data "
                          "there, or give this experiment another name")
    then = info.get("train_source")
    if then and os.path.abspath(exp.data.train) != os.path.abspath(then):
        raise ConfigError(f"{prepared} was prepared from the train file {then}, but data.train is {exp.data.train}; "
                          "run `ojh prepare` (with --force if another experiment owns the name) or use another name")
    if not Path(exp.data.train).is_file():
        raise ConfigError(f"data.train not found: {exp.data.train}")
    sha = sha256_file(exp.data.train)
    if sha != info.get("train_sha256_source"):
        raise ConfigError(f"the train file {exp.data.train} changed since prepare (sha256 {sha[:12]}…, prepared from "
                          f"{str(info.get('train_sha256_source'))[:12]}…): re-run `ojh prepare`")


def _build(exp: ExperimentConfig, machine: MachineConfig, *, query, plan_runner, fingerprint, machine_origin,
           apps_query, check_gpus: bool, allow_shared: bool = False, source: str | None = None) -> tuple[dict, Path]:
    """The lock and the adapter file. Order: config, prepared data and adapter checks, plan-only, fingerprint,
    and the cards last, so busy cards never hide a data or plan problem (``check_gpus=False`` skips them)."""
    if not exp.base:
        from .catalog import CATALOG_DIR
        raise ConfigError("no base: set base in the experiment or pass --base; catalog bases: "
                          + ", ".join(sorted(p.stem for p in CATALOG_DIR.glob("*.yaml"))))
    if not exp.recipe:
        raise ConfigError("no recipe: set recipe in the experiment or pass --recipe")
    out_root = Path(machine.out_root)
    check_writable(machine, out_root / exp.name, machine.registry_path, machine.lock_dir)
    recipe = copy.deepcopy(_load_recipe(exp.recipe))
    explicit = _explicit_cards(exp, machine)
    if explicit is not None:
        recipe["gpus"] = len(explicit)
    elif exp.gpus is not None:
        recipe["gpus"] = exp.gpus
    recorded = _apply_overrides(recipe, exp.overrides)
    mode = recipe.get("mode")
    if mode == "full" and any(k.startswith("lora.") for k in recorded):
        raise ConfigError("lora.* override on a full recipe would have no effect; set mode: lora first")
    if mode not in ("full", "lora"):
        raise ConfigError(f"recipe {exp.recipe}: 'mode' is required and must be full or lora (got {mode!r})")
    if exp.merge and mode != "lora":
        raise ConfigError("merge: true merges a LoRA adapter into its base; it has no effect with mode: full")
    n = recipe.get("gpus")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ConfigError("gpus must be a positive integer (recipe or experiment)")

    base = load_base(exp.base)
    parallel = parallel_for(mode, base["params_b"], n, recipe.get("fsdp"))
    if explicit is not None and n != len(explicit):
        raise ConfigError(f"{len(explicit)} card(s) listed but the run needs {n} (an override sets gpus)")
    need = estimate_gib(base["params_b"], mode, n, parallel == "fsdp")

    prepared = Path(machine.out_root) / exp.name / "prepared"
    info_path = prepared / "prepared.json"
    if not info_path.is_file():
        raise ConfigError(f"no prepared data at {prepared}; run `ojh prepare` first")
    info = json.loads(info_path.read_text())
    _check_prepared(exp, info, prepared, source)
    from .prepare import adapter_file
    afile = adapter_file(exp.data.adapter)
    asha = sha256_file(afile)
    prepared_sha = (info.get("adapter") or {}).get("sha256")
    if prepared_sha is not None and prepared_sha != asha:
        raise ConfigError(f"adapter changed after prepare ({afile}: sha256 {asha[:12]}… vs {prepared_sha[:12]}… "
                          "when prepared); re-run `ojh prepare`")
    args_now = {"global": exp.data.adapter_args, "train": exp.data.train_adapter_args,
                "eval": {n: s.adapter_args for n, s in exp.data.eval.items()}}
    args_then = info.get("adapter_args")
    if args_then is not None and json.loads(json.dumps(args_then)) != json.loads(json.dumps(args_now)):
        raise ConfigError(f"adapter_args changed after prepare (prepared with {args_then}, the experiment now "
                          f"says {args_now}); re-run `ojh prepare` so the prepared data matches")
    data = {"prepared_dir": str(prepared), "source": exp.data.train, "adapter": exp.data.adapter,
            "adapter_file": str(afile), "adapter_sha256": asha,
            "adapter_args": dict(exp.data.adapter_args), "train_adapter_args": dict(exp.data.train_adapter_args),
            "sha256": {f: _sha(prepared / f) for f in DATA_FILES},
            "source_sha256": info["train_sha256_source"], "subset": info.get("subset"),
            "counts": info.get("counts"),
            # the label and split files finalize and eval-report read again (design section 8)
            "eval_sha256": {name: {"train_labels": _source_sha(s.train_labels, f"data.eval.{name}.train_labels"),
                                   "reference": None if s.reference is None
                                   else _source_sha(s.reference, f"data.eval.{name}.reference")}
                            for name, s in exp.data.eval.items()},
            "split_file_sha256": None if exp.data.split_file is None
            else _source_sha(exp.data.split_file, "data.split_file")}
    for f in ("train", "select", "select_index"):
        data[f] = str(prepared / (f + (".json" if f == "select_index" else ".jsonl")))

    flags = dict(recipe.get("onejev") or {})
    if mode == "full":
        flags["full_finetune"] = True
    if parallel == "fsdp":
        flags["fsdp"] = True
    lora = dict(recipe.get("lora") or {}) if mode == "lora" else None
    for k, v in (lora or {}).items():
        flags[k if k == "target_modules" else f"lora_{k}"] = v  # OneJev: --target-modules, --lora-r, ...
    flags["attention"], flags["readout"] = base["attention"], base["readout"]
    if base.get("multimodal"):
        flags["multimodal"] = True
    flags["eval_at_start"] = True

    lock = {
        "experiment": exp.name, "repo_root": str(ROOT),
        "run_dir": str(out_root / exp.name / "plan"),
        "machine": {"python": machine.python, "out_root": machine.out_root, "model_dir": machine.model_dir,
                    "registry": machine.registry_path, "env": dict(machine.env),
                    "gpu": {"include": machine.gpu_include, "exclude": machine.gpu_exclude,
                            "min_free_mib": machine.min_free_mib, "lock_dir": machine.lock_dir,
                            "order": machine.order, "wait_timeout_s": machine.wait_timeout_s,
                            "source": "explicit" if explicit is not None else machine.gpu_source,
                            "explicit": explicit,
                            # explicit cards may carry other processes only with ojh run --allow-shared
                            "allow_shared": bool(allow_shared)},
                    "forbidden_write_roots": list(machine.forbidden_write_roots),
                    # where these settings came from (--machine, $OJH_MACHINE, user config or inferred)
                    "origin": machine_origin or {"source": "given", "path": None}},
        "mode": mode, "lora": lora, "n_gpus": n, "parallel": parallel, "seeds": list(exp.seeds),
        "estimate_gib": round(need, 1), "feasibility": None, "flags": flags, "data": data,
        "eval_sets": {k: asdict(v) for k, v in exp.data.eval.items()},
        "split_file": exp.data.split_file, "fields": asdict(exp.fields),
        "evaluator": asdict(exp.evaluator), "merge": exp.merge,
        "allow_test": bool(info.get("allow_test", False)),  # prepare --allow-test, carried to finalize
        "base": {"name": exp.base, **{k: v for k, v in base.items() if k != "name"}},
        "onejev": load_lock(), "overrides": recorded,
        "recipe": {"path": exp.recipe, "sha256": sha256_file(exp.recipe)},
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
    if check_gpus:
        _, lock["feasibility"] = _check_feasible(machine, base, mode, n, parallel, query, explicit, apps_query,
                                                 allow_shared)
    else:
        lock["feasibility"] = {"checked": False, "reason": "cards not checked (ojh check)"}
    return lock, afile


def check(exp: ExperimentConfig, machine: MachineConfig, *, query=_nvidia_smi, plan_runner=None,
          fingerprint=None, machine_origin: dict | None = None, apps_query=_nvidia_smi_apps,
          source: str | None = None) -> dict:
    """Everything resolve validates except the cards (config, prepared data, adapter, plan-only,
    fingerprint); writes nothing and returns the would-be lock."""
    lock, _ = _build(exp, machine, query=query, plan_runner=plan_runner, fingerprint=fingerprint,
                     machine_origin=machine_origin, apps_query=apps_query, check_gpus=False, source=source)
    return lock


def resolve(exp: ExperimentConfig, machine: MachineConfig, *, query=_nvidia_smi, plan_runner=None,
            fingerprint=None, now: dt.datetime | None = None, machine_origin: dict | None = None,
            apps_query=_nvidia_smi_apps, allow_shared: bool = False, source: str | None = None) -> Path:
    """``source``: the experiment file; the prepared data must have been prepared from it."""
    lock, afile = _build(exp, machine, query=query, plan_runner=plan_runner, fingerprint=fingerprint,
                         machine_origin=machine_origin, apps_query=apps_query, check_gpus=True,
                         allow_shared=allow_shared, source=source)
    out_root, data = Path(machine.out_root), lock["data"]
    body = json.dumps(lock, sort_keys=True, default=str)
    stamp = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{hashlib.sha256(body.encode()).hexdigest()[:8]}"
    run_dir = out_root / exp.name / run_id
    lock["run_id"], lock["run_dir"] = run_id, str(run_dir)
    check_writable(machine, run_dir)
    run_dir.mkdir(parents=True)
    # the adapter as it was at resolve time, next to the lock (the lock also records its sha256)
    kept = run_dir / "adapter" / afile.name
    kept.parent.mkdir()
    shutil.copy2(afile, kept)
    data["adapter_copy"] = str(kept)
    path = run_dir / "run.lock.yaml"
    path.write_text(yaml.safe_dump(lock, sort_keys=False))
    return path
