"""A candidate training scheme: base model + harness recipe + overrides (each with a reason) + a note.

Its id is computed from content (base, recipe file contents, override values, environment exp.yaml contents), so the
same scheme is never trained twice; reasons and notes do not change the id.

Recipes are named as in the framework's config_request: null is the catalog default of the base for one card
(<ojh repo>/catalog/bases/<base>.yaml, default_recipes); a relative path is looked up under modules/train, then under
the ojh repository; an absolute path is used as is.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .config import RunConfig, Space

MODULE_DIR = Path(__file__).resolve().parent.parent   # modules/train
MAX_STEPS = "onejev.max_steps"                         # config_request.max_steps is this override


class CandidateError(ValueError):
    pass


@dataclass(frozen=True)
class Override:
    value: object
    reason: str


@dataclass(frozen=True)
class Candidate:
    base: str
    recipe: Path
    overrides: dict[str, Override] = field(default_factory=dict)
    note: str = ""
    default_recipe: bool = False   # recipe null: the base's catalog default (recipe is that file)

    def values(self) -> dict:
        return {k: o.value for k, o in sorted(self.overrides.items())}

    def label(self) -> str:
        """Short human-readable description, e.g. 'mammoth-1 lora_1g lr=5e-05'."""
        ov = " ".join(f"{k.split('.', 1)[-1]}={v}" for k, v in self.values().items())
        return " ".join(x for x in (self.base, self.recipe.stem, ov) if x)

    def to_dict(self) -> dict:
        return {"base": self.base, "recipe": str(self.recipe), "recipe_sha256": file_sha256(self.recipe),
                "default_recipe": self.default_recipe,
                "overrides": {k: {"value": o.value, "reason": o.reason} for k, o in sorted(self.overrides.items())},
                "note": self.note}


@dataclass(frozen=True)
class Prediction:
    metric: str        # the run's primary metric, e.g. auprc
    value: float       # expected value of the primary metric for the probe seed
    p_accept: float    # probability that the candidate ends up accepted


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(base_dir: Path, value) -> Path:
    s = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in s:
        raise CandidateError(f"unset environment variable in path {value!r}")
    p = Path(s)
    return p if p.is_absolute() else (base_dir / p).resolve()


def _fits(key, n: int) -> bool:
    s = str(key).strip()
    try:
        if s.endswith("+"):
            return n >= int(s[:-1])
        if "-" in s:
            lo, hi = s.split("-", 1)
            return int(lo) <= n <= int(hi)
        return int(s) == n
    except ValueError:
        return False


def resolve_recipe(value, base: str, ojh_root: Path | None) -> Path:
    """The recipe file a config_request's recipe names (see the module docstring)."""
    if value is None:
        if ojh_root is None:
            raise CandidateError("recipe null (the catalog default) needs the ojh repository (run.yaml ojh.root)")
        cat = Path(ojh_root) / "catalog" / "bases" / f"{base}.yaml"
        if not cat.is_file():
            raise CandidateError(f"base {base}: no catalog entry {cat}")
        recipes = (yaml.safe_load(cat.read_text()) or {}).get("default_recipes") or {}
        rel = next((r for k, r in recipes.items() if _fits(k, 1)), None)
        if rel is None:
            raise CandidateError(f"base {base} has no default recipe for one card (default_recipes: {recipes}); "
                                 "name a recipe")
        return Path(ojh_root) / rel
    s = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in s:
        raise CandidateError(f"unset environment variable in recipe {value!r}")
    p = Path(s)
    tries = [p] if p.is_absolute() else [MODULE_DIR / p] + ([Path(ojh_root) / p] if ojh_root else [])
    for t in tries:
        if t.is_file():
            return t.resolve()
    raise CandidateError(f"recipe {value!r} not found (looked at {', '.join(str(t) for t in tries)})")


def parse_candidate(d: dict, where: str, ojh_root: Path | None) -> Candidate:
    """Candidate from a mapping {base, recipe (null: the catalog default), overrides: {key: {value, reason}}, note}."""
    if not isinstance(d, dict):
        raise CandidateError(f"{where}: expected a mapping")
    if not d.get("base"):
        raise CandidateError(f"{where}: missing 'base'")
    ovs = {}
    for k, o in (d.get("overrides") or {}).items():
        if not isinstance(o, dict) or "value" not in o:
            raise CandidateError(f"{where}: override {k} must be {{value: ..., reason: ...}}")
        if not str(o.get("reason") or "").strip():
            raise CandidateError(f"{where}: override {k} needs a reason")
        ovs[str(k)] = Override(o["value"], str(o["reason"]).strip())
    try:
        recipe = resolve_recipe(d.get("recipe"), str(d["base"]), ojh_root)
    except CandidateError as e:
        raise CandidateError(f"{where}: {e}") from None
    return Candidate(str(d["base"]), recipe, ovs, str(d.get("note") or "").strip(), d.get("recipe") is None)


def request_mapping(req: dict, reason: str, note: str) -> dict:
    """A config_request {base, recipe, overrides, seed, max_steps} as a candidate mapping (the seed is the trial's)."""
    ovs = dict(req.get("overrides") or {})
    if req.get("max_steps") is not None:
        ovs[MAX_STEPS] = req["max_steps"]
    return {"base": req.get("base"), "recipe": req.get("recipe"),
            "overrides": {k: {"value": v, "reason": reason} for k, v in sorted(ovs.items())}, "note": note}


def config_request(c: Candidate, seed: int, recipe_names: dict) -> dict:
    """The framework's config_request of one trial: the recipe as the task names it (null for the catalog
    default), the overrides without reasons, max_steps from onejev.max_steps."""
    ovs = c.values()
    steps = ovs.pop(MAX_STEPS, None)
    recipe = None if c.default_recipe else recipe_names.get(c.recipe, str(c.recipe))
    return {"base": c.base, "recipe": recipe, "overrides": ovs, "seed": seed, "max_steps": steps}


def load_proposal(path: Path, run: "RunConfig") -> tuple[Candidate, Prediction]:
    """A proposal YAML: the candidate fields plus prediction: {<primary metric>: x, p_accept: q}."""
    path = Path(path)
    d = yaml.safe_load(path.read_text()) or {}
    c = parse_candidate(d, str(path), run.ojh.root)
    if not c.note:
        raise CandidateError(f"{path}: write a note (which evidence it answers and why it may help)")
    pr = d.get("prediction")
    m = run.metric.primary
    if not isinstance(pr, dict) or not isinstance(pr.get(m), (int, float)) or not isinstance(pr.get("p_accept"), (int, float)):
        raise CandidateError(f"{path}: prediction must be {{{m}: <expected value for seed {run.rules.seeds[0]}>, "
                             "p_accept: <0..1>}}")
    if not 0 <= pr["p_accept"] <= 1:
        raise CandidateError(f"{path}: prediction.p_accept must be in [0, 1]")
    return c, Prediction(m, float(pr[m]), float(pr["p_accept"]))


def _norm(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    return float(v)


def candidate_id(c: Candidate, exp_sha256: str) -> str:
    key = {"base": c.base, "recipe": file_sha256(c.recipe), "overrides": {k: _norm(v) for k, v in c.values().items()},
           "exp": exp_sha256}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def trial_id(c: Candidate, seed: int, exp_sha256: str) -> str:
    return f"{candidate_id(c, exp_sha256)}-s{seed}"


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def outside_space(c: Candidate, space: "Space") -> list[str]:
    """Why the candidate is not in the task's search_space; empty when it is."""
    out = []
    if c.base not in space.bases:
        out.append(f"base {c.base} is not in the search_space (allowed: {', '.join(space.bases)})")
    allowed = ", ".join(["null (catalog default)"] * space.default_recipe + [str(r) for r in space.recipes])
    if (not space.default_recipe) if c.default_recipe else (c.recipe not in space.recipes):
        name = "null (catalog default)" if c.default_recipe else str(c.recipe)
        out.append(f"recipe {name} is not in the search_space (allowed: {allowed})")
    for k, o in sorted(c.overrides.items()):
        s = space.overrides.get(k)
        if s is None:
            out.append(f"override {k} is not in the search_space")
            continue
        in_values = any(_same(o.value, v) for v in s.values)
        in_range = (s.range is not None and isinstance(o.value, (int, float)) and not isinstance(o.value, bool)
                    and s.range[0] <= o.value <= s.range[1])
        if not (in_values or in_range):
            allowed = ", ".join(repr(v) for v in s.values) + (f"; range {list(s.range)}" if s.range else "")
            out.append(f"override {k} = {o.value!r} is not in the search_space (allowed: {allowed})")
    return out


def dead_ends(c: Candidate, space: "Space") -> list[str]:
    """Allowed values that are marked as dead ends, with their reasons."""
    out = []
    for k, o in sorted(c.overrides.items()):
        s = space.overrides.get(k)
        for v, why in (s.dead.items() if s else ()):
            if _same(o.value, v):
                out.append(f"{k} = {o.value!r} is a dead end: {why}")
    return out
