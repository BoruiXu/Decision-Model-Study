"""Run configuration (run.yaml): the environment (a harness exp.yaml), the start, the search space, GPUs, the
comparison rules, the budget, and the framework's side (task, sets, output directory). `trsi init --brief` writes
run.yaml and exp.yaml from a train brief and its task (write_from_brief); the train command writes them for one
configuration (write_run). Nobody writes run.yaml by hand.

From the brief and the task:
  run_name             the dispatch id (brief dispatch_id; else r<round>-<output directory name>)
  out_dir              <brief outdir>/trsi
  ojh                  resources.onejev: ojh_bin, env_file, machine (or $RSI_ONEJEV_MACHINE); root, the ojh repository,
                       is found from ojh_bin (or python): <root>/.venv/bin/ojh
  start                the config_request of the brief's best_candidate (the current best version), else
                       resources.onejev.default_config
  space                search_space: base, recipe, overrides.onejev.<flag>, overrides.lora.<field>, max_steps
                       (onejev.max_steps); each a list of allowed values or {min, max}, optionally {values: [...]}
                       and dead: {value: why}
  gpus.ids             resources.gpu_ids (null: every card); the agent picks free ones at each launch
  metric               primary: metrics.primary (higher is better); guards: search.guards (default ece 0.02, brier
                       0.005; each ece or one of the task's metrics)
  rules                search.seeds (else search_space.seed, else 0, 1, 2), probe_margin, accept_margin, max_configs
  budget               gpu_hours: the brief's budget_left.compute_hours (null: no limit); estimate:
                       resources.onejev.estimate ({base: hours of one seed} or one number for every base)
  framework            what packing a candidate needs: task, trainset(_id), valset(_id), outdir, round, producer,
                       python (resources.onejev.python with ~ and $VAR expanded: model.predict's interpreter; a
                       variable that is not set is refused), train_command (modules/train/train),
                       start_source (the config_source of the start's candidates), progress_file (the brief's
                       progress_file, else <outdir>/progress.json: trsi.progress writes it)
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .candidate import (MAX_STEPS, MODULE_DIR, Candidate, CandidateError, parse_candidate, request_mapping,
                        resolve_path, resolve_recipe)
from .metrics import LOWER_IS_BETTER

OVERRIDE_PREFIXES = ("onejev.", "lora.")
# Flags the harness derives itself and refuses as overrides (ojh's harness/resolve.py _RESERVED_FLAGS); the training
# mode (mode, fsdp) changes only through a recipe, and the GPU count through the recipe's gpus.
DERIVED = {"full_finetune", "fsdp", "attention", "readout", "multimodal", "eval_every", "save_every", "eval_at_start",
           "seed", "name", "train", "model", "eval", "runs_dir", "config", "plan_only", "device", "merge",
           "merged_dir", "worst_case", "dump_first_grad", "eval_limit", "dist_backend"}
DEFAULT_GUARDS = {"ece": 0.02, "brier": 0.005}
TRAIN_COMMAND = MODULE_DIR / "train"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OjhSettings:
    bin: str                       # the ojh executable, e.g. $OJH_REPO/.venv/bin/ojh
    env_file: Path | None = None   # sourced in the same shell command before ojh (harness README "环境配置")
    machine: Path | None = None    # harness machine.yaml
    root: Path | None = None       # the ojh repository: catalog default recipes, relative recipe paths


@dataclass(frozen=True)
class OverrideSpace:
    values: tuple = ()
    range: tuple[float, float] | None = None
    dead: dict = field(default_factory=dict)   # value -> reason it is a dead end


@dataclass(frozen=True)
class Space:
    bases: tuple[str, ...]
    recipes: tuple[Path, ...]                  # named recipe files
    overrides: dict[str, OverrideSpace]
    default_recipe: bool = False               # recipe null (each base's catalog default) is allowed
    recipe_names: dict = field(default_factory=dict)   # recipe file -> its name in the task


@dataclass(frozen=True)
class GpuPolicy:
    ids: tuple[int, ...] | None                # None: every card
    allow_shared: bool = False


@dataclass(frozen=True)
class MetricSpec:
    primary: str = "auprc"
    guards: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GUARDS))


@dataclass(frozen=True)
class Rules:
    seeds: tuple[int, ...] = (0, 1, 2)   # the first one is the probe seed
    probe_margin: float = 0.035
    accept_margin: float = 0.035
    max_configs: int | None = None       # proposals one search may try (the start not counted)


@dataclass(frozen=True)
class Framework:
    task: Path
    trainset: Path
    trainset_id: str
    valset: Path
    valset_id: str
    outdir: Path                          # candidates go to <outdir>/cand-<trial>/
    round: int
    producer: str
    python: str                           # interpreter of model.predict (the ojh .venv python)
    train_command: Path                   # code_version.entry
    start_source: str = "default"         # config_source of the start's candidates (proposals: "tune")
    progress_file: Path | None = None     # progress.json the framework shows (docs/interfaces/progress.md)

    def task_json(self) -> dict:
        return json.loads(self.task.read_text())


@dataclass(frozen=True)
class RunConfig:
    path: Path
    run_name: str
    out_dir: Path
    exp: Path
    ojh: OjhSettings
    start: Candidate
    space: Space
    gpus: GpuPolicy
    metric: MetricSpec
    rules: Rules
    budget_gpu_h: float                   # inf: no limit
    estimate_gpu_h: dict[str, float]      # base -> expected compute hours of one seed, before any result exists
    framework: Framework

    @property
    def home(self) -> Path:
        """Where this run keeps its records: <out_dir>/<run_name>/."""
        return self.out_dir / self.run_name


def _need(d: dict, key: str, where: str):
    if key not in d or d[key] in (None, ""):
        raise ConfigError(f"{where}: missing '{key}'")
    return d[key]


def _float(v, where: str, lo: float | None = None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where} must be a number (got {v!r})")
    if lo is not None and v < lo:
        raise ConfigError(f"{where} must be >= {lo} (got {v})")
    return float(v)


def _space(d: dict, root: Path | None) -> Space:
    bases = tuple(str(b) for b in _need(d, "bases", "space"))
    recipes, names, default = [], {}, False
    for r in _need(d, "recipes", "space"):
        if r is None:
            default = True
            continue
        try:
            p = resolve_recipe(r, "", root)
        except CandidateError as e:
            raise ConfigError(f"space.recipes: {e}") from None
        recipes.append(p)
        names.setdefault(p, str(r))
    ovs = {}
    for k, s in (d.get("overrides") or {}).items():
        k = str(k)
        if not k.startswith(OVERRIDE_PREFIXES):
            raise ConfigError(f"space.overrides.{k}: only onejev.<flag> and lora.<field> can be searched; the training "
                              "mode and GPU count change through recipes")
        if k.split(".", 1)[1] in DERIVED:
            raise ConfigError(f"space.overrides.{k}: the harness derives this flag itself and refuses it as an override")
        s = s or {}
        values = tuple(s.get("values") or ())
        rng = s.get("range")
        if rng is not None:
            if not (isinstance(rng, list) and len(rng) == 2):
                raise ConfigError(f"space.overrides.{k}.range must be [low, high]")
            rng = (_float(rng[0], f"space.overrides.{k}.range"), _float(rng[1], f"space.overrides.{k}.range"))
        if not values and rng is None:
            raise ConfigError(f"space.overrides.{k}: give values or range")
        # keys may arrive as strings (e.g. JSON); read them as YAML scalars so "3" matches the value 3
        dead = {(yaml.safe_load(v) if isinstance(v, str) else v): str(why) for v, why in (s.get("dead") or {}).items()}
        ovs[k] = OverrideSpace(values, rng, dead)
    return Space(bases, tuple(recipes), ovs, default, names)


def _framework(d: dict) -> Framework:
    f = _need(d, "framework", "run")
    try:
        return Framework(Path(f["task"]), Path(f["trainset"]), str(f["trainset_id"]), Path(f["valset"]),
                         str(f["valset_id"]), Path(f["outdir"]), int(f["round"]), str(f["producer"]),
                         str(f["python"]), Path(f["train_command"]), str(f.get("start_source") or "default"),
                         Path(f.get("progress_file") or Path(f["outdir"]) / "progress.json"))
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"framework: {e!r} (run.yaml is written by `trsi init --brief`)") from None


def load_run(path: Path) -> RunConfig:
    path = Path(path).resolve()
    d = yaml.safe_load(path.read_text()) or {}
    base_dir = path.parent
    name = str(_need(d, "run_name", "run"))
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ConfigError(f"run_name {name!r}: use letters, digits, '.', '_' and '-' only")
    try:
        out_dir = resolve_path(base_dir, _need(d, "out_dir", "run"))
        exp = resolve_path(base_dir, _need(d, "exp", "run"))
        o = _need(d, "ojh", "run")
        ojh = OjhSettings(os.path.expandvars(os.path.expanduser(str(_need(o, "bin", "ojh")))),  # unset vars: see harness
                          resolve_path(base_dir, o["env_file"]) if o.get("env_file") else None,
                          resolve_path(base_dir, o["machine"]) if o.get("machine") else None,
                          resolve_path(base_dir, o["root"]) if o.get("root") else None)
        start = parse_candidate(_need(d, "start", "run"), "start", ojh.root)
        space = _space(_need(d, "space", "run"), ojh.root)
    except CandidateError as e:
        raise ConfigError(str(e)) from None
    if not start.default_recipe and start.recipe not in space.recipe_names:
        space.recipe_names[start.recipe] = str(d["start"]["recipe"])
    if not exp.is_file():
        raise ConfigError(f"exp {exp} does not exist")
    g = _need(d, "gpus", "run")
    ids = None if g.get("ids") is None else tuple(int(i) for i in g["ids"])
    gpus = GpuPolicy(ids, bool(g.get("allow_shared", False)))
    m = d.get("metric") or {}
    primary = str(m.get("primary", "auprc"))
    guards = {str(k): _float(v, f"metric.guards.{k}", 0)
              for k, v in (m["guards"] if "guards" in m else DEFAULT_GUARDS).items()}
    for k in guards:
        if k not in LOWER_IS_BETTER:
            raise ConfigError(f"metric.guards.{k}: guards are lower-is-better metrics ({', '.join(LOWER_IS_BETTER)})")
    r = d.get("rules") or {}
    seeds = tuple(int(s) for s in r.get("seeds", (0, 1, 2)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ConfigError("rules.seeds must be distinct and non-empty")
    mc = r.get("max_configs")
    rules = Rules(seeds, _float(r.get("probe_margin", 0.035), "rules.probe_margin", 0),
                  _float(r.get("accept_margin", 0.035), "rules.accept_margin", 0), None if mc is None else int(mc))
    b = _need(d, "budget", "run")
    gh = b.get("gpu_hours")
    return RunConfig(
        path=path, run_name=name, out_dir=out_dir, exp=exp, ojh=ojh, start=start, space=space, gpus=gpus,
        metric=MetricSpec(primary, guards), rules=rules,
        budget_gpu_h=math.inf if gh is None else _float(gh, "budget.gpu_hours", 0),
        estimate_gpu_h={str(k): _float(v, f"budget.estimate.{k}", 0) for k, v in (b.get("estimate") or {}).items()},
        framework=_framework(d),
    )


# ---- writing run.yaml and exp.yaml ------------------------------------------------------------------------------

def _read_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read {p}: {e}") from None


def rows_file(set_dir: Path) -> Path:
    """The rows.jsonl of a trainset or valset directory (its manifest's files.rows)."""
    m = _read_json(Path(set_dir) / "manifest.json")
    try:
        return Path(set_dir) / m["files"]["rows"]["path"]
    except (KeyError, TypeError):
        raise ConfigError(f"{set_dir}/manifest.json: no files.rows") from None


def ojh_root_of(onejev: dict) -> Path | None:
    """The ojh repository: <root>/.venv/bin/ojh (or .../python) with <root>/catalog."""
    for k in ("ojh_bin", "python"):
        v = onejev.get(k)
        if v:
            p = Path(os.path.expandvars(os.path.expanduser(str(v))))
            if len(p.parents) > 2 and (p.parents[2] / "catalog").is_dir():
                return p.parents[2]
    return None


def machine_of(onejev: dict) -> str | None:
    return onejev.get("machine") or os.environ.get("RSI_ONEJEV_MACHINE") or None


def _values(v, where: str) -> list:
    if isinstance(v, list):
        return list(v)
    if isinstance(v, dict) and isinstance(v.get("values"), list):
        return list(v["values"])
    raise ConfigError(f"search_space.{where}: expected a list of allowed values")


def _spec(v, where: str) -> dict:
    if isinstance(v, list):
        return {"values": list(v)}
    if not isinstance(v, dict):
        raise ConfigError(f"search_space.{where}: expected a list of values or {{min, max}}")
    out = {}
    if v.get("values") is not None:
        out["values"] = _values(v, where)
    if "min" in v or "max" in v:
        out["range"] = [v.get("min"), v.get("max")]
    if not out:
        raise ConfigError(f"search_space.{where}: give a list of values or {{min, max}}")
    if v.get("dead"):
        out["dead"] = {str(k): str(why) for k, why in v["dead"].items()}
    return out


def space_from(search_space: dict, start: dict) -> tuple[dict, list | None]:
    """trsi's space from the task's search_space, and the allowed seeds (None when not given)."""
    space = {"bases": [start["base"]], "recipes": [start["recipe"]], "overrides": {}}
    seeds = None
    for key, v in (search_space or {}).items():
        if key == "base":
            space["bases"] = _values(v, key)
        elif key == "recipe":
            space["recipes"] = _values(v, key)
        elif key == "seed":
            seeds = [int(s) for s in _values(v, key)]
        elif key == "max_steps":
            space["overrides"][MAX_STEPS] = _spec(v, key)
        elif key.startswith("overrides.") and key[len("overrides."):].startswith(OVERRIDE_PREFIXES):
            space["overrides"][key[len("overrides."):]] = _spec(v, key)
        else:
            raise ConfigError(f"search_space.{key}: unknown key (base, recipe, overrides.onejev.<flag>, "
                              "overrides.lora.<field>, seed, max_steps)")
    return space, seeds


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_run(home: Path, *, task_path: Path, trainset: Path, trainset_id: str, valset: Path, valset_id: str,
              outdir: Path, round_: int, producer: str, run_name: str, start_req: dict, start_source: str,
              start_note: str, budget, search: bool, progress_file: Path | None = None) -> Path:
    """Write <home>/exp.yaml and <home>/run.yaml (see the module docstring); `search` False: one configuration
    (the train command): the space is the start, the seed its own, no guards. progress_file defaults to
    <outdir>/progress.json. Returns the run.yaml path."""
    task = _read_json(task_path)
    d = task.get("decision") or {}
    if d.get("type") != "binary" or not d.get("positive"):
        raise ConfigError("trsi trains binary decisions (decision.type binary with a positive label)")
    m = task.get("metrics") or {}
    if not m.get("higher_is_better", True):
        raise ConfigError(f"metrics.primary {m.get('primary')} is lower-is-better; trsi's rules compare a "
                          "higher-is-better primary metric")
    onejev = (task.get("resources") or {}).get("onejev") or {}
    for k in ("python", "ojh_bin"):
        if not onejev.get(k):
            raise ConfigError(f"the task's resources.onejev.{k} is missing")
    python = os.path.expandvars(os.path.expanduser(str(onejev["python"])))
    if "$" in python:
        raise ConfigError(f"the task's resources.onejev.python {onejev['python']!r} names a variable that is not set "
                          f"here ({python})")
    start = request_mapping(start_req, "start", start_note)
    if not start["base"]:
        raise ConfigError("the start's config_request has no base")
    s = task.get("search") or {}
    if search:
        space, allowed = space_from(task.get("search_space") or {}, start)
        seeds = [int(x) for x in (s.get("seeds") or allowed or (0, 1, 2))]
        if allowed is not None and not set(seeds) <= set(allowed):
            raise ConfigError(f"search.seeds {seeds} are not all in search_space.seed {allowed}")
        guards = s["guards"] if s.get("guards") is not None else dict(DEFAULT_GUARDS)
        known = {"ece", m.get("primary")} | set(m.get("others") or ())
        for g in guards:
            if g not in known:
                raise ConfigError(f"search.guards.{g}: not ece and not one of the task's metrics (primary, others), "
                                  "so no candidate reports it; add it to metrics.others or change search.guards")
    else:
        space, seeds, guards = {"bases": [start["base"]], "recipes": [start["recipe"]], "overrides": {}}, \
            [int(start_req.get("seed") or 0)], {}
    est = onejev.get("estimate")
    bases = list(dict.fromkeys(space["bases"] + [start["base"]]))
    estimate = {b: est for b in bases} if isinstance(est, (int, float)) else dict(est or {})
    root = ojh_root_of(onejev)
    ojh = {"bin": onejev["ojh_bin"]}
    if onejev.get("env_file"):
        ojh["env_file"] = onejev["env_file"]
    if machine_of(onejev):
        ojh["machine"] = machine_of(onejev)
    if root is not None:
        ojh["root"] = str(root)
    rules = {"seeds": seeds, "probe_margin": s.get("probe_margin", 0.035), "accept_margin": s.get("accept_margin", 0.035)}
    if search and s.get("max_configs") is not None:
        rules["max_configs"] = int(s["max_configs"])
    run = {
        "run_name": run_name, "out_dir": str(home), "exp": "exp.yaml", "ojh": ojh, "start": start,
        "space": space, "gpus": {"ids": (task.get("resources") or {}).get("gpu_ids")},
        "metric": {"primary": m.get("primary"), "guards": guards}, "rules": rules,
        "budget": {"gpu_hours": budget, "estimate": estimate},
        "framework": {"task": str(task_path), "trainset": str(trainset), "trainset_id": trainset_id,
                      "valset": str(valset), "valset_id": valset_id, "outdir": str(outdir), "round": int(round_),
                      "producer": producer, "python": python, "train_command": str(TRAIN_COMMAND),
                      **({} if start_source == "default" else {"start_source": start_source}),
                      "progress_file": str(progress_file or Path(outdir) / "progress.json")},
    }
    exp = {"name": re.sub(r"[^A-Za-z0-9_-]", "_", f"{task.get('task_id', 'task')}-{run_name}"),
           "data": {"adapter": "onejev_rows", "train": str(rows_file(trainset)),
                    "eval": {"val": {"train_labels": str(rows_file(valset))}}},
           "fields": {"label": f"target.{d['positive']}", "group": "meta.state_id"},
           # the framework's train gate (rsi score against the task's baselines) decides; ojh's own gate is off
           "evaluator": {"type": "binary_decision", "gate": False},
           "seeds": [0], "overrides": {}}
    _write(home / "exp.yaml", "# written by trsi from the task and the brief; trials copy it\n"
           + yaml.safe_dump(exp, sort_keys=False))
    run_path = home / "run.yaml"
    tmp = run_path.with_name("run.yaml.tmp")
    tmp.write_text("# written by trsi from the task and the brief (do not edit)\n"
                   + yaml.safe_dump(run, sort_keys=False, allow_unicode=True))
    try:
        load_run(tmp)  # refuse a configuration trsi cannot read: run.yaml only ever holds one it can
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, run_path)
    return run_path


def write_from_brief(brief_path: Path) -> Path:
    """run.yaml and exp.yaml in <brief outdir>/trsi/ from a train brief; an existing run.yaml is kept once its run
    is initialised (incumbent.yaml written), so a return goes on with the same run; else it is written again."""
    b = _read_json(brief_path)
    for k in ("task_json", "outdir", "inputs", "round"):
        if k not in b:
            raise ConfigError(f"{brief_path}: no '{k}' (is it a brief.json?)")
    inputs, ids, outdir = b["inputs"], b.get("ids") or {}, Path(b["outdir"])
    for k in ("trainset", "valset"):
        if not inputs.get(k) or not ids.get(k):
            raise ConfigError(f"{brief_path}: the brief gives no {k} (inputs.{k} and ids.{k})")
    run_name = b.get("dispatch_id") or re.sub(r"[^A-Za-z0-9._-]", "_", f"r{b['round']}-{outdir.name}")
    home = outdir / "trsi"
    run_path = home / "run.yaml"
    if run_path.is_file() and (home / run_name / "incumbent.yaml").is_file():
        return run_path
    task = _read_json(Path(b["task_json"]))
    if inputs.get("best_candidate"):
        cm = _read_json(Path(inputs["best_candidate"]) / "manifest.json")
        if not isinstance(cm.get("config_request"), dict):
            raise ConfigError(f"best_candidate {inputs['best_candidate']}: its manifest has no config_request")
        req, source = cm["config_request"], "tune"
        note = f"the current best version's config ({cm.get('id') or inputs['best_candidate']})"
    else:
        req = ((task.get("resources") or {}).get("onejev") or {}).get("default_config")
        if not isinstance(req, dict):
            raise ConfigError("the task has no resources.onejev.default_config")
        source, note = "default", "the task's default config"
    return write_run(home, task_path=Path(b["task_json"]), trainset=Path(inputs["trainset"]),
                     trainset_id=ids["trainset"], valset=Path(inputs["valset"]), valset_id=ids["valset"],
                     outdir=outdir, round_=b["round"], producer="train", run_name=run_name, start_req=req,
                     start_source=source, start_note=note,
                     budget=(b.get("budget_left") or {}).get("compute_hours"), search=True,
                     progress_file=Path(b["progress_file"]) if b.get("progress_file") else None)
