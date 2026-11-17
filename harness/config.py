"""Machine and experiment config loading. Imports without torch."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


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


@dataclass
class DataSpec:
    train: str
    adapter: str = "onejev_rows"
    train_sha256: str | None = None
    eval: dict[str, EvalSetSpec] = field(default_factory=dict)
    split_file: str | None = None


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


@dataclass
class ExperimentConfig:
    name: str
    base: str
    recipe: str
    data: DataSpec
    fields: FieldsSpec
    subset: SubsetSpec | None = None
    evaluator: EvaluatorSpec = field(default_factory=EvaluatorSpec)
    seeds: list[int] = field(default_factory=lambda: [42])
    gpus: int | None = None
    merge: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)


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


def load_machine(path) -> MachineConfig:
    d = _load_yaml(path)
    for k in ("python", "model_dir", "out_root"):
        _require(d, k)
    unknown = set(d) - set(MachineConfig.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"unknown machine key(s): {sorted(unknown)}")
    return MachineConfig(**d)


def _check_keys(d: Any, cls, prefix: str = "") -> dict:
    """Require a mapping whose keys are all fields of ``cls``."""
    where = prefix.rstrip(".") or "top level"
    if not isinstance(d, dict):
        raise ConfigError(f"{where} must be a mapping")
    for k in d:
        if k not in cls.__dataclass_fields__:
            raise ConfigError(f"unknown key: {prefix}{k}")
    return d


def load_experiment(path) -> ExperimentConfig:
    d = _check_keys(_load_yaml(path), ExperimentConfig)
    for k in ("name", "base", "recipe"):
        _require(d, k)
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
        eval_sets[name] = EvalSetSpec(
            train_labels=_require(spec, "train_labels", prefix),
            reference=spec.get("reference"),
        )
    data_kw = {k: data[k] for k in ("adapter", "train_sha256", "split_file") if k in data}
    sub = d.get("subset")
    subset = SubsetSpec(**_check_keys(sub, SubsetSpec, "subset.")) if sub is not None else None
    ev = d.get("evaluator")
    evaluator = (EvaluatorSpec(**_check_keys(ev, EvaluatorSpec, "evaluator."))
                 if ev is not None else EvaluatorSpec())
    kw = {k: d[k] for k in ("seeds", "gpus", "merge", "overrides") if k in d}
    return ExperimentConfig(
        name=d["name"], base=d["base"], recipe=d["recipe"],
        data=DataSpec(train=train, eval=eval_sets, **data_kw),
        fields=FieldsSpec(label=label, **{k: fields_.get(k) for k in ("group", "consistency", "text")}),
        subset=subset, evaluator=evaluator, **kw,
    )
