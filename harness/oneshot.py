"""``ojh run``: prepare -> resolve -> launch in one call; returns at once with a JSON-able summary.

The user supplies the base and the cards (``--base``, ``--gpus``, or ``base``/``gpus`` in the experiment
file); nothing is chosen automatically. The recipe is ``--recipe``, else the experiment's ``recipe``, else
the base's ``default_recipes`` entry for the card count (an error listing the options when there is none).
An infeasible setup raises ``Infeasible`` with its alternatives and launches nothing. Imports without torch.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from harness import launch, prepare, resolve
from harness.catalog import CATALOG_DIR, default_recipe, load_base
from harness.config import ConfigError, _recipe_path, check_writable, find_machine, load_experiment


def catalog_bases() -> list[str]:
    return sorted(p.stem for p in CATALOG_DIR.glob("*.yaml"))


def parse_gpus(text: str) -> list[int]:
    """``"0,1,3"`` -> ``[0, 1, 3]``."""
    try:
        out = [int(p) for p in str(text).split(",") if p.strip() != ""]
    except ValueError:
        raise ConfigError(f"--gpus takes card indices such as 0,1 (got {text!r})") from None
    if not out or any(i < 0 for i in out):
        raise ConfigError(f"--gpus takes card indices such as 0,1 (got {text!r})")
    return out


def _n_gpus(gpus) -> int:
    return len(gpus) if isinstance(gpus, list) else int(gpus)


def _prepared(exp_path, base, gpus, recipe, machine, allow_test, force=False):
    """Machine, origin, experiment (base, gpus and recipe filled in) and recipe source, after prepare."""
    m, origin = find_machine(machine)
    exp = load_experiment(exp_path)
    if base:
        exp.base = base
    if not exp.base:
        raise ConfigError("no base: pass --base <name> (or set base in the experiment); catalog bases: "
                          + ", ".join(catalog_bases()))
    load_base(exp.base)  # unknown base: CatalogError listing the known ones
    if gpus:
        exp.gpus = list(gpus)
    if exp.gpus is None:
        raise ConfigError("no GPUs: pass --gpus i,j,... (or set gpus in the experiment)")
    if recipe:
        exp.recipe, source = _recipe_path(recipe, Path.cwd()), "--recipe"
    elif exp.recipe:
        source = "experiment"
    else:
        exp.recipe, source = default_recipe(exp.base, _n_gpus(exp.gpus)), "catalog default"
    if not Path(exp.recipe).is_file():
        raise ConfigError(f"recipe not found: {exp.recipe}")

    out = Path(m.out_root) / exp.name / "prepared"
    check_writable(m, out)
    prepare.prepare(exp, out, allow_test=allow_test, source=os.path.abspath(exp_path), force=force)
    return m, origin, exp, source


def _probes(**kw) -> dict:
    return {k: v for k, v in kw.items() if v is not None}


def check(exp_path, *, base: str | None = None, gpus: list[int] | None = None, recipe: str | None = None,
          machine: str | None = None, allow_test: bool = False, force: bool = False, query=None, apps_query=None,
          plan_runner=None, fingerprint=None) -> dict:
    """``ojh check``: what ``ojh run`` validates except the cards (prepare, config, adapter, plan-only,
    fingerprint). Writes the prepared data like run, but no lock and no run dir; launches nothing."""
    m, origin, exp, source = _prepared(exp_path, base, gpus, recipe, machine, allow_test, force)
    lock = resolve.check(exp, m, machine_origin=origin, source=os.path.abspath(exp_path), **_probes(
        query=query, apps_query=apps_query, plan_runner=plan_runner, fingerprint=fingerprint))
    return {"ok": True, "cards_checked": False, "base": exp.base,
            "recipe": {"path": exp.recipe, "source": source, "mode": lock["mode"]},
            "n_gpus": lock["n_gpus"], "parallel": lock["parallel"], "gpus": exp.gpus,
            "estimate_gib_per_card": lock["estimate_gib"], "plan": lock["plan"],
            "fingerprint": lock["fingerprint"],
            "data_sha256": {"source": lock["data"]["source_sha256"], **lock["data"]["sha256"]},
            "counts": lock["data"].get("counts"), "machine": origin,
            "next": "ojh run with the same arguments validates the cards and launches"}


def run(exp_path, *, base: str | None = None, gpus: list[int] | None = None, recipe: str | None = None,
        machine: str | None = None, allow_test: bool = False, allow_shared: bool = False, force: bool = False,
        query=None, apps_query=None, plan_runner=None, fingerprint=None, tmux=None) -> dict:
    """``allow_shared``: the requested cards may carry other users' processes (only after asking the user);
    recorded in the lock."""
    m, origin, exp, source = _prepared(exp_path, base, gpus, recipe, machine, allow_test, force)
    lock_path = resolve.resolve(exp, m, machine_origin=origin, allow_shared=allow_shared,
                                source=os.path.abspath(exp_path), **_probes(
        query=query, apps_query=apps_query, plan_runner=plan_runner, fingerprint=fingerprint))
    started = launch.launch(lock_path, tmux=tmux or launch._tmux)

    lock = yaml.safe_load(Path(lock_path).read_text())
    feas = lock.get("feasibility") or {}
    run_dir = Path(lock["run_dir"])
    return {"run_dir": str(run_dir), "session": started["session"], "log": started["log"],
            "done": started["done"], "base": exp.base,
            "recipe": {"path": exp.recipe, "source": source, "mode": lock["mode"]},
            "gpus": feas.get("cards") or lock["machine"]["gpu"].get("explicit") or lock["n_gpus"],
            "allow_shared": bool(lock["machine"]["gpu"].get("allow_shared")),
            "data_sha256": {"source": lock["data"]["source_sha256"], **lock["data"]["sha256"]},
            "machine": origin, "result": str(run_dir / "RESULT.md"),
            "next": f"ojh status {run_dir}"}
