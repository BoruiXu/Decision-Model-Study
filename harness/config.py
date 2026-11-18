"""Machine and experiment config loading. Imports without torch.

Every path in a config file has ``~`` and ``$VAR`` expanded and is made absolute when loaded: relative paths
in a machine file are relative to that file's directory, relative paths in an experiment file to the
experiment file's directory (a recipe that is not there but exists below the repo root, such as
``examples/recipes/...``, resolves to the repo root). Later stages never depend on the caller's cwd.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness.catalog import ROOT


class ConfigError(ValueError):
    """A config file is missing a key or has a malformed value."""


def get_field(row: dict, dotted: str) -> Any:
    """Look up a dotted path such as ``meta.f``; raise KeyError if any step is absent."""
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


@dataclass
class MachineConfig:
    python: str
    model_dir: str
    out_root: str
    registry: str | None = None
    gpu_include: list[int] | None = None
    gpu_exclude: list[int] = field(default_factory=list)
    min_free_mib: int = 0
    lock_dir: str | None = None
    order: str = "free_desc"
    wait_timeout_s: int = 3600
    gpu_source: str = "scan"
    forbidden_write_roots: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def registry_path(self) -> str:
        return self.registry or os.path.join(self.out_root, "registry.jsonl")


@dataclass
class EvalSetSpec:
    train_labels: str
    reference: str | None = None
    adapter_args: dict[str, Any] = field(default_factory=dict)  # merged over data.adapter_args for this set


def is_adapter_path(name: str) -> bool:
    """``data.adapter`` names a .py file (resolved relative to the experiment file) rather than a module."""
    return name.endswith(".py") or "/" in name or os.sep in name


@dataclass
class DataSpec:
    train: str
    adapter: str = "onejev_rows"
    train_sha256: str | None = None
    eval: dict[str, EvalSetSpec] = field(default_factory=dict)
    split_file: str | None = None
    adapter_args: dict[str, Any] = field(default_factory=dict)  # keyword arguments: adapt(path, **args)
    train_adapter_args: dict[str, Any] = field(default_factory=dict)  # merged over adapter_args for train


@dataclass
class SubsetSpec:
    where: dict[str, Any] = field(default_factory=dict)
    n: int | None = None
    seed: int = 0


@dataclass
class FieldsSpec:
    label: str
    group: str | None = None
    consistency: str | None = None
    text: str | None = None


@dataclass
class EvaluatorSpec:
    type: str = "binary_decision"
    budgets: list[float] = field(default_factory=lambda: [0.10, 0.15])
    gate: bool = True
    path: str | None = None
    bootstrap_reps: int = 1000  # 95% CIs by two-level bootstrap (groups, then items); 0 turns them off
    bootstrap_seed: int = 20261115


@dataclass
class ExperimentConfig:
    name: str
    base: str | None  # may be omitted in the file: ``ojh run --base`` supplies it
    recipe: str | None  # may be omitted: ``ojh run --recipe`` or the base's default_recipes
    data: DataSpec
    fields: FieldsSpec
    subset: SubsetSpec | None = None
    evaluator: EvaluatorSpec = field(default_factory=EvaluatorSpec)
    seeds: list[int] = field(default_factory=lambda: [42])
    gpus: int | list[int] | None = None  # a count, or card indices (gpu source explicit)
    merge: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)


def _under(path, root) -> bool:
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def check_writable(machine: MachineConfig, *paths) -> None:
    """Raise ConfigError when a path the harness is about to write is at or below a ``forbidden_write_roots``
    entry (symlinks resolved). ``None`` entries are skipped."""
    for p in paths:
        if p is None:
            continue
        for root in machine.forbidden_write_roots:
            if _under(p, root):
                raise ConfigError(f"refusing to write {p}: it is under forbidden_write_roots entry {root}")


def _load_yaml(path) -> dict:
    with open(path) as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ConfigError(f"{path}: expected a mapping at top level")
    return obj


def _require(d: dict, key: str, prefix: str = "") -> Any:
    if not isinstance(d, dict) or d.get(key) is None:
        raise ConfigError(f"missing required key: {prefix}{key}")
    return d[key]


def _section(d: dict, key: str) -> dict:
    v = _require(d, key)
    if not isinstance(v, dict):
        raise ConfigError(f"key {key} must be a mapping")
    return v


def _expand(p) -> str:
    return os.path.expanduser(os.path.expandvars(str(p)))


def _abs(p, base: Path) -> str:
    """``p`` with ``~``/``$VAR`` expanded, made absolute against ``base`` when relative."""
    return os.path.normpath(os.path.join(base, _expand(p)))


def _config_dir(path) -> Path:
    return Path(os.path.abspath(path)).parent


def _env(raw) -> dict[str, str]:
    """Process environment values must be strings: scalars are converted, anything nested is refused."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("env must be a mapping of NAME: value")
    out = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, (int, float)):
            v = str(v)
        elif not isinstance(v, str):
            raise ConfigError(f"env.{k} must be a string, number or boolean (got {type(v).__name__})")
        out[str(k)] = v
    return out


def load_machine(path) -> MachineConfig:
    d = _load_yaml(path)
    for k in ("python", "model_dir", "out_root"):
        _require(d, k)
    unknown = set(d) - set(MachineConfig.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"unknown machine key(s): {sorted(unknown)}")
    base = _config_dir(path)
    py = _expand(d["python"])
    d["python"] = _abs(py, base) if os.sep in py else py  # a bare name such as python3 is looked up on PATH
    for k in ("model_dir", "out_root", "registry", "lock_dir"):
        if d.get(k) is not None:
            d[k] = _abs(d[k], base)
    roots = d.get("forbidden_write_roots") or []
    if not isinstance(roots, list):
        raise ConfigError("forbidden_write_roots must be a list of paths")
    d["forbidden_write_roots"] = [_abs(r, base) for r in roots]
    d["env"] = _env(d.get("env"))
    return MachineConfig(**d)


USER_MACHINE = "~/.config/onejev-harness/machine.yaml"


def inferred_machine() -> MachineConfig:
    """The machine used when no machine file is found: the repo's .venv python, ``$OJH_OUT_ROOT`` or
    ``./ojh_runs`` (made absolute), ``$OJH_MODEL_DIR`` or ``~/.cache/onejev-harness/models``, the registry
    under out_root, no lock_dir, and GPUs given explicitly (``ojh run --gpus``)."""
    cwd = Path.cwd()
    out_root = _abs(os.environ.get("OJH_OUT_ROOT") or "ojh_runs", cwd)
    model_dir = _abs(os.environ.get("OJH_MODEL_DIR") or "~/.cache/onejev-harness/models", cwd)
    return MachineConfig(python=str(ROOT / ".venv" / "bin" / "python"), model_dir=model_dir, out_root=out_root,
                         gpu_source="explicit")


def find_machine(path=None) -> tuple[MachineConfig, dict]:
    """The machine config and where it came from. Order: ``path`` (``--machine``), ``$OJH_MACHINE``,
    ``~/.config/onejev-harness/machine.yaml``, then ``inferred_machine()``."""
    if path:
        return load_machine(path), {"source": "--machine", "path": os.path.abspath(path)}
    env = os.environ.get("OJH_MACHINE")
    if env:
        p = _expand(env)
        if not os.path.isfile(p):
            raise ConfigError(f"$OJH_MACHINE points to {p}, which is not a file")
        return load_machine(p), {"source": "$OJH_MACHINE", "path": os.path.abspath(p)}
    user = _expand(USER_MACHINE)
    if os.path.isfile(user):
        return load_machine(user), {"source": "user config", "path": os.path.abspath(user)}
    return inferred_machine(), {"source": "inferred", "path": None}


def _check_keys(d: Any, cls, prefix: str = "") -> dict:
    """Require a mapping whose keys are all fields of ``cls``."""
    where = prefix.rstrip(".") or "top level"
    if not isinstance(d, dict):
        raise ConfigError(f"{where} must be a mapping")
    for k in d:
        if k not in cls.__dataclass_fields__:
            raise ConfigError(f"unknown key: {prefix}{k}")
    return d


def _args(raw, where: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping of keyword arguments for adapt(path, **args)")
    return dict(raw)


def _recipe_path(p, base: Path) -> str:
    local = _abs(p, base)
    if not os.path.exists(local) and not os.path.isabs(_expand(p)):
        in_repo = _abs(p, ROOT)
        if os.path.exists(in_repo):
            return in_repo
    return local


def load_experiment(path) -> ExperimentConfig:
    d = _check_keys(_load_yaml(path), ExperimentConfig)
    base = _config_dir(path)
    _require(d, "name")
    data = _check_keys(_section(d, "data"), DataSpec, "data.")
    fields_ = _check_keys(_section(d, "fields"), FieldsSpec, "fields.")
    train = _require(data, "train", "data.")
    label = _require(fields_, "label", "fields.")
    eval_sets = {}
    raw_eval = data.get("eval") or {}
    if not isinstance(raw_eval, dict):
        raise ConfigError("data.eval must be a mapping")
    for name, spec in raw_eval.items():
        prefix = f"data.eval.{name}."
        _check_keys(spec, EvalSetSpec, prefix)
        ref = spec.get("reference")
        eval_sets[name] = EvalSetSpec(
            train_labels=_abs(_require(spec, "train_labels", prefix), base),
            reference=_abs(ref, base) if ref is not None else None,
            adapter_args=_args(spec.get("adapter_args"), f"{prefix}adapter_args"),
        )
    data_kw = {k: data[k] for k in ("adapter", "train_sha256", "split_file") if k in data}
    if data_kw.get("split_file") is not None:
        data_kw["split_file"] = _abs(data_kw["split_file"], base)
    if data_kw.get("adapter") is not None:
        a = str(data_kw["adapter"])
        data_kw["adapter"] = _abs(a, base) if is_adapter_path(a) else a
    for k in ("adapter_args", "train_adapter_args"):
        data_kw[k] = _args(data.get(k), f"data.{k}")
    sub = d.get("subset")
    subset = SubsetSpec(**_check_keys(sub, SubsetSpec, "subset.")) if sub is not None else None
    ev = d.get("evaluator")
    evaluator = (EvaluatorSpec(**_check_keys(ev, EvaluatorSpec, "evaluator."))
                 if ev is not None else EvaluatorSpec())
    if evaluator.path is not None:
        evaluator.path = _abs(evaluator.path, base)
    kw = {k: d[k] for k in ("seeds", "gpus", "merge", "overrides") if k in d}
    return ExperimentConfig(
        name=d["name"], base=d.get("base"),
        recipe=_recipe_path(d["recipe"], base) if d.get("recipe") is not None else None,
        data=DataSpec(train=_abs(train, base), eval=eval_sets, **data_kw),
        fields=FieldsSpec(label=label, **{k: fields_.get(k) for k in ("group", "consistency", "text")}),
        subset=subset, evaluator=evaluator, **kw,
    )
