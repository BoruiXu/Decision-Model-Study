"""Recipes and the generation methods. A recipe is one method, a number of rows and, for the targeted methods, the
weakness types they aim at; one recipe makes one arm, <out>/arms/<sha12>/.

The recipe JSON: {"name", "method", "n", "target": {"kinds", "types", "alloc", "skip_hard_share"}, "params", "seed",
"max_usd"}; an absent key takes its default (DEFAULT), an unknown key is an error, so a typo never silently does
nothing. sha12 is the first 12 hex digits of the sha256 of the recipe with its defaults, without `name`: two recipes
with the same content are one arm.

The methods (all keep their seed's label; seeds, anchors and partners are always real rows: dgen.inputs, never prompt
variants or rows generated in an earlier round):
- hard: the real failure rows themselves, unchanged, without replacement; n is spread over the targeted weakness
  types (target.alloc: size, in proportion to their sizes, or uniform); a type with fewer failures than its share
  gives them all (stats.short). Standard library only.
- smote (SMOTE, Chawla et al. 2002): on the real trainset rows labelled params.label (default: a binary task's
  positive label), an anchor and a partner among its k = 5 nearest same-label rows in the z-space that has the same
  edit-view keys; the new row interpolates their edit-view numeric leaves with lambda ~ U[0, 1) and keeps the
  anchor's other leaves. An anchor without such a partner is counted in skipped_structure. Not targeted.
- borderline (Borderline-SMOTE-1, Han et al. 2005): smote with the anchors weighted by danger (m = 10 nearest
  trainset rows: half or more but not all of another label; when fewer than 10 anchors are in danger, the weight is
  the share of the other label among the m, plus 1e-9).
- llm_imitate (M2, LLM2LLM-style), llm_pattern (M3, PaDA-Agent-style): rows an LLM generator writes from the targeted
  real failures (one failure per call; or one analysis per weakness type, then its pattern and strategy), kept only
  when an independent reviewer scores their realism at least the threshold calibrated on 60 real pool rows (stratified
  by label, drawn with the recipe's seed); dgen.llm_gen. The generator and the reviewer are llm.claude.ClaudeCall
  objects in <out>/llm/<sha12>/{gen,review}/, with params.model and params.effort (default: the task's agent model
  and effort), the recipe's max_usd as their shared cap and the brief's budget_left.api_usd as the dispatch's account
  (the spend file's api_usd lines); a reached cap stops the recipe short (stats.stopped "api budget"), never with an
  error. Standard library only.
Every built row is applied through the adapter, repaired and checked (rows.edited_state); a violation drops it
(stats.dropped_by_rule, by message), and smote and borderline draw again (at most 50 n draws, then stats.short).
"""
import copy
import hashlib
import json
import os
import random

from rsi import rows as R

from . import DgenError, need_ml
from . import rows as RW

METHODS = ("hard", "smote", "borderline", "llm_imitate", "llm_pattern")
TARGETED = ("hard", "llm_imitate", "llm_pattern")
LLM = ("llm_imitate", "llm_pattern")
PARAMS = {"hard": (), "smote": ("label",), "borderline": ("label",), "llm_imitate": ("model", "effort"),
          "llm_pattern": ("model", "effort")}
DEFAULT = {"name": None, "method": None, "n": None,
           "target": {"kinds": [], "types": [], "alloc": "size", "skip_hard_share": None},
           "params": {}, "seed": 0, "max_usd": None}
K_NEIGHBOURS, M_DANGER, MAX_DRAWS = 5, 10, 50


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def normalize(raw, task):
    """The recipe with its defaults; DgenError naming every problem."""
    if not isinstance(raw, dict):
        raise DgenError("a recipe is a JSON object")
    errors = []
    unknown = sorted(set(raw) - set(DEFAULT))
    if unknown:
        errors.append("unknown keys %s (a recipe has %s)" % (", ".join(unknown), ", ".join(DEFAULT)))
    out = copy.deepcopy(DEFAULT)
    for k, v in raw.items():
        if k == "target":
            if not isinstance(v, dict):
                errors.append("target: an object")
                continue
            bad = sorted(set(v) - set(DEFAULT["target"]))
            if bad:
                errors.append("unknown keys %s under target (it has %s)" % (", ".join(bad),
                                                                            ", ".join(DEFAULT["target"])))
            out["target"].update(copy.deepcopy(v))
        elif k in DEFAULT:
            out[k] = copy.deepcopy(v)
    if not isinstance(out["name"], str) or not out["name"]:
        errors.append("name: a non-empty string")
    m = out["method"]
    if m not in METHODS:
        errors.append("method: one of %s (got %r)" % (", ".join(METHODS), m))
    if not isinstance(out["n"], int) or isinstance(out["n"], bool) or out["n"] < 1:
        errors.append("n: an integer >= 1")
    if not isinstance(out["seed"], int) or isinstance(out["seed"], bool) or out["seed"] < 0:
        errors.append("seed: an integer >= 0")
    if out["max_usd"] is not None and not (_number(out["max_usd"]) and out["max_usd"] >= 0):
        errors.append("max_usd: null or a number >= 0")
    t = out["target"]
    labels = task["decision"]["labels"]
    kinds = ["%s→%s" % (a, b) for a in labels for b in labels if a != b]
    if not isinstance(t["kinds"], list) or [k for k in t["kinds"] if k not in kinds]:
        errors.append("target.kinds: a list of error kinds \"<true>→<decision>\" of the task's labels")
    if not isinstance(t["types"], list) or [x for x in t["types"] if not isinstance(x, str)]:
        errors.append("target.types: a list of weakness type keys (weakness.json)")
    if t["alloc"] not in ("size", "uniform"):
        errors.append("target.alloc: size or uniform")
    if t["skip_hard_share"] is not None and not (_number(t["skip_hard_share"]) and 0 <= t["skip_hard_share"] <= 1):
        errors.append("target.skip_hard_share: null or a share in [0, 1]")
    params = out["params"] if isinstance(out["params"], dict) else {}
    if not isinstance(out["params"], dict):
        errors.append("params: an object")
    elif m in PARAMS:
        bad = sorted(set(out["params"]) - set(PARAMS[m]))
        if bad:
            errors.append("params: unknown keys %s for %s (it takes %s)" % (", ".join(bad), m,
                                                                           ", ".join(PARAMS[m]) or "none"))
    if m in ("smote", "borderline") and t != DEFAULT["target"]:
        errors.append("target: %s is not targeted (it draws from the trainset rows of params.label): leave target "
                      "out" % m)
    if m in ("smote", "borderline") and "label" in params and params["label"] not in labels:
        errors.append("params.label: one of the task's labels %s" % ", ".join(labels))
    if m in LLM:
        for k in ("model", "effort"):
            if k in params and not (isinstance(params[k], str) and params[k].strip()):
                errors.append("params.%s: a non-empty string (the claude -p --%s value)" % (k, k))
    if errors:
        raise DgenError("recipe: %s" % "; ".join(errors))
    return out


def sha12(recipe):
    core = dict((k, v) for k, v in recipe.items() if k != "name")
    return hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


class Context(object):
    """What a method needs: inputs (dgen.inputs.Inputs), the diagnosis (failures, weakness, learnability; None for
    the untargeted methods) and the task's adapter."""

    def __init__(self, inputs, diagnosis, adapter, workdir=None):
        self.inputs, self.diagnosis, self.adapter = inputs, diagnosis, adapter
        self.workdir = workdir  # where the LLM methods' calls keep their directories and logs
        self.task = inputs.task


def generate(ctx, recipe, h):
    """(rows, provenance, stats) of the recipe (sha12 h)."""
    m = recipe["method"]
    stats = {"method": m, "requested": recipe["n"], "dropped_by_rule": {}}
    if m in LLM:
        rows, prov = _llm(ctx, recipe, h, stats)
    elif m == "hard":
        rows, prov = _hard(ctx, recipe, stats)
    else:
        rows, prov = _smote(ctx, recipe, h, stats, borderline=m == "borderline")
    stats["n"] = len(rows)
    return rows, prov, stats


# ---- hard ------------------------------------------------------------------------------------------------------------

def _targets(ctx, recipe, stats):
    """{type key: [failure records]} of the failures the recipe targets."""
    if ctx.diagnosis is None:
        raise DgenError("method %s needs a diagnosis: give --diagnosis <the directory dgen diagnose wrote>"
                        % recipe["method"])
    t = recipe["target"]
    W, learn = ctx.diagnosis["weakness"], ctx.diagnosis["learnability"]
    known = [x["key"] for x in W["types"]]
    unknown = [x for x in t["types"] if x not in known]
    if unknown:
        raise DgenError("target.types: %s not in the diagnosis's weakness types (%s)"
                        % (", ".join(unknown), ", ".join(known) or "none"))
    allowed = [x["key"] for x in W["types"] if (not t["kinds"] or x["kind"] in t["kinds"])
               and (not t["types"] or x["key"] in t["types"])]
    skipped = []
    if t["skip_hard_share"] is not None:
        by_type = (learn or {}).get("by_type") or {}
        skipped = [k for k in allowed if (by_type.get(k, {}).get("hard") or 0) > t["skip_hard_share"]]
        allowed = [k for k in allowed if k not in skipped]
    stats["skipped_types_hard_share"] = skipped
    pools = dict((k, []) for k in allowed)
    for f in ctx.diagnosis["failures"]:
        if f.get("type") in pools:
            pools[f["type"]].append(f)
    pools = dict((k, v) for k, v in pools.items() if v)
    if not pools:
        raise DgenError("no failure of the diagnosis matches the recipe's target (kinds %s, types %s%s)"
                        % (t["kinds"] or "all", t["types"] or "all",
                           ", %d types skipped by skip_hard_share" % len(skipped) if skipped else ""))
    return pools


def _seeds(ctx, pools):
    """{type key: [(pool row, failure record)]} of the targeted failures."""
    pool = dict((r["id"], r) for r in ctx.inputs.pool_rows)
    out = {}
    for key, fails in pools.items():
        for f in fails:
            if f["id"] not in pool:
                raise DgenError("failure %s is not a row of the brief's pool (a diagnosis of another pool?)" % f["id"])
        out[key] = [(pool[f["id"]], f) for f in fails]
    return out


def _hard(ctx, recipe, stats):
    pools = _targets(ctx, recipe, stats)
    quota = RW.allocate(recipe["n"], dict((k, len(v)) for k, v in pools.items()), recipe["target"]["alloc"])
    stats["quota"] = quota
    seeds = _seeds(ctx, pools)
    rng = random.Random(recipe["seed"])
    rows, prov, short = [], [], {}
    for key in sorted(pools):
        fails = seeds[key]
        for i in sorted(rng.sample(range(len(fails)), min(quota[key], len(fails)))):
            row, f = fails[i]
            rows.append(copy.deepcopy(row))
            prov.append({"id": f["id"], "method": "hard", "seed_id": f["id"], "type": key, "kind": f["kind"],
                         "real": True})
        if quota[key] > len(fails):
            short[key] = quota[key] - len(fails)
    stats["short"] = sum(short.values())
    stats["short_by_type"] = short
    return rows, prov


# ---- smote and borderline --------------------------------------------------------------------------------------------

def _label(ctx, recipe):
    label = recipe["params"].get("label")
    if label is None:
        d = ctx.task["decision"]
        if d["type"] != "binary":
            raise DgenError("%s: params.label is needed (the task is not binary, so there is no positive label to "
                            "default to)" % recipe["method"])
        label = d["positive"]
    return label


def _smote(ctx, recipe, h, stats, borderline):
    need_ml("dgen generate (method %s)" % recipe["method"])
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from .space import Space, featurizer

    inputs, task, adapter = ctx.inputs, ctx.task, ctx.adapter
    label = _label(ctx, recipe)
    train = inputs.trainset_rows  # real rows only
    real = inputs.real()
    space = Space(task).fit(real, [inputs.label(r) for r in real])
    y = np.array([inputs.label(r) for r in train], dtype=object)
    pos = np.where(y == label)[0]
    stats["label"], stats["anchors"] = label, int(len(pos))
    if len(pos) < 2:
        raise DgenError("%s: the trainset has %d real rows labelled %s; it needs at least 2"
                        % (recipe["method"], len(pos), label))
    Z = space.z(train)
    k = min(K_NEIGHBOURS, len(pos) - 1)
    near = NearestNeighbors(n_neighbors=k + 1).fit(Z[pos]).kneighbors(Z[pos], return_distance=False)
    fz = featurizer(task)
    views = [fz.edit_view(train[i]["state"]) for i in pos]
    keys = [frozenset(v) for v in views]
    partners = []
    for a in range(len(pos)):
        nb = [int(b) for b in near[a] if b != a][:k]
        partners.append([b for b in nb if keys[b] == keys[a]])
    w = np.ones(len(pos))
    if borderline:
        m = min(M_DANGER, len(train) - 1)
        nb_all = NearestNeighbors(n_neighbors=m + 1).fit(Z).kneighbors(Z[pos], return_distance=False)
        others = np.array([sum(1 for b in [b for b in row if b != i][:m] if y[b] != label)
                           for row, i in zip(nb_all, pos)], dtype=float)
        danger = (others >= m / 2.0) & (others < m)
        stats["n_danger"] = int(danger.sum())
        w = danger.astype(float) if danger.sum() >= 10 else others / float(max(m, 1)) + 1e-9
    has = np.array([bool(p) for p in partners])
    stats["skipped_structure"] = int((~has).sum())
    w = w * has
    if w.sum() <= 0:
        raise DgenError("%s: no anchor labelled %s can be drawn: %d of %d have no partner among their %d nearest "
                        "same-label rows with the same edit-view keys (skipped_structure)%s"
                        % (recipe["method"], label, int((~has).sum()), len(pos), k,
                           "; borderline draws only the %d anchors in danger, and none of them has one"
                           % stats["n_danger"] if borderline and stats["n_danger"] >= 10 else ""))
    w = w / w.sum()
    rng = np.random.default_rng(recipe["seed"])
    rows, prov, drops, draws = [], [], stats["dropped_by_rule"], 0
    while len(rows) < recipe["n"] and draws < MAX_DRAWS * recipe["n"]:
        draws += 1
        a = int(rng.choice(len(pos), p=w))
        b = int(partners[a][int(rng.integers(len(partners[a])))])
        lam = float(rng.random())
        anchor, partner = train[pos[a]], train[pos[b]]
        changes = RW.interpolate(views[a], views[b], lam)
        try:
            state, bad = RW.edited_state(adapter, anchor["state"], changes)
        except (KeyError, ValueError) as e:  # a path the adapter's apply refuses
            state, bad = None, ["apply: %s" % e]
        if bad:
            for msg in bad:
                drops[msg] = drops.get(msg, 0) + 1
            continue
        row = RW.new_row(task, anchor, state, recipe["method"], h, len(rows), partner["id"])
        bad = R.check_full_rows([row], task)
        if bad:
            for msg in bad:
                msg = "row: " + msg.split(": ", 1)[-1]
                drops[msg] = drops.get(msg, 0) + 1
            continue
        rows.append(row)
        prov.append({"id": rows[-1]["id"], "method": recipe["method"], "seed_id": anchor["id"],
                     "partner_id": partner["id"], "lam": round(lam, 6), "type": None})
    stats["draws"] = draws
    stats["short"] = recipe["n"] - len(rows)
    return rows, prov


# ---- llm_imitate and llm_pattern -------------------------------------------------------------------------------------

def _llm(ctx, recipe, h, stats):
    from llm.claude import ClaudeCall, LlmBudgetExceeded
    from . import llm_gen as LG
    inputs, task, m = ctx.inputs, ctx.task, recipe["method"]
    if ctx.workdir is None:
        raise DgenError("method %s needs a work directory for its LLM calls" % m)
    seeds = _seeds(ctx, _targets(ctx, recipe, stats))
    params = recipe["params"]
    api_left = (inputs.brief.get("budget_left") or {}).get("api_usd")
    calls = dict((role, ClaudeCall(task, os.path.join(ctx.workdir, role), inputs.spend_file, recipe["max_usd"],
                                   model=params.get("model"), effort=params.get("effort"), api_left=api_left))
                 for role in ("gen", "review"))
    gen, rev = calls["gen"], calls["review"]
    gen.shared, rev.shared = rev, gen  # the recipe's cap counts both
    stats.update(llm_dir=ctx.workdir, max_usd=recipe["max_usd"], api_left=api_left, model=params.get("model"),
                 effort=params.get("effort"), reviewer_threshold=None)
    real = RW.stratified_sample(inputs.pool_rows, inputs.label, LG.CALIBRATION_ROWS, recipe["seed"])
    stats["calibration_rows"] = len(real)
    rows, prov = [], []
    try:
        t = LG.calibrate_reviewer(rev, task, real)
    except LlmBudgetExceeded as e:
        stats.update(short=recipe["n"], stopped="api budget", stop_detail=str(e))
    except LG.Uncalibrated as e:
        stats.update(short=recipe["n"], stopped="reviewer not calibrated", stop_detail=str(e))
    else:
        stats["reviewer_threshold"] = t
        system = LG.system_prompt(task, ctx.adapter, inputs.real(), recipe["seed"])
        kw = dict(n=recipe["n"], h=h, system=system, threshold=t, seed=recipe["seed"],
                  alloc=recipe["target"]["alloc"])
        if m == "llm_imitate":
            rows, st = LG.imitate(gen, rev, task, ctx.adapter, seeds, **kw)
        else:
            pool = dict((r["id"], r) for r in inputs.pool_rows)
            ok = [(pool[o["id"]], o["true"]) for o in ctx.diagnosis.get("ok_sample") or [] if o["id"] in pool]
            rows, st = LG.pattern(gen, rev, task, ctx.adapter, seeds, ok_rows=ok, **kw)
        prov = st.pop("provenance")
        stats.update(st)
    stats["calls"] = {"gen": gen.calls, "review": rev.calls}
    stats["usd"] = {"gen": round(gen.spent, 6), "review": round(rev.spent, 6)}
    return rows, prov
