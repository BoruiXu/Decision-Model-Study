"""Fixed rules shared by artifact checks and the orchestrator.

decide_promotion and check_stop are pure: they take dicts and return
(conclusion, reason). The workspace and the CLI call them; the orchestrating
agent never makes these decisions itself.
"""
import copy
import math
import os

from rsi.timing import plural

MODULE_OF = {
    "data_coverage": "data",
    "label_noise": "data",
    "label_definition": "data",
    "model_capacity": "train",
    "calibration": "train",
    "config": "train",
    "runtime": "deliver",
}

APPROVAL_ITEMS = ("task_change", "over_budget", "use_testset", "refresh_sets", "plan", "delete")

# Tolerance for comparisons on values computed by subtraction or summation
# (guard regressions, compute hours used), so that e.g. 0.52 - 0.5 counts as 0.02.
EPS = 1e-9

# The spend ledger (docs/interfaces/spend.md): (the used key in STATE budget_used, the budget key in task["budget"], the
# label). The two dollar accounts are never added together: api_usd is what the LLM calls that generate or review data
# cost (spend-file lines), agent_usd what the module and decision agents' sessions cost (the engine counts it).
RESOURCES = (("compute_hours", "compute_hours", "GPU hours"), ("api_usd", "api_cost", "data LLM $"),
             ("agent_usd", "agent_cost", "agent $"), ("annotation_items", "annotation_items", "annotation"))
USD_KEYS = ("api_usd", "agent_usd")


def check_attribution(attribution, module):
    """Return error strings when the attribution is unknown or does not belong to `module`."""
    if attribution not in MODULE_OF:
        return ["attribution '%s' is unknown (expected one of %s)"
                % (attribution, ", ".join(sorted(MODULE_OF)))]
    if MODULE_OF[attribution] != module:
        return ["attribution '%s' belongs to module '%s', not '%s'"
                % (attribution, MODULE_OF[attribution], module)]
    return []


def is_number(x):
    """A finite int or float (bool is not a number here)."""
    if isinstance(x, bool):
        return False
    return isinstance(x, int) or (isinstance(x, float) and math.isfinite(x))


def _g(x):
    return "%.6g" % x


def decide_promotion(task, evaluation, has_best, candidate_gate_passed):
    """Apply task["promotion_rule"] to an evaluation; return ("promote" or "keep", reason).

    Without a best version the candidate is promoted when it passed its train
    gate. With one, the improvement s * delta on the primary metric (s = 1 when
    higher is better, -1 otherwise) must be above min_delta, its interval lower
    bound (ci_low, or -ci_high when lower is better) at least ci_lower_min when
    that is set, and no guard may regress by more than its max_regression.
    Any value the rule needs that is missing or not a finite number means keep.
    """
    if not has_best:
        if candidate_gate_passed is True:
            return "promote", "no best version yet and the candidate passed its train gate"
        return "keep", "no best version yet and the candidate did not pass its train gate"

    primary = task["metrics"]["primary"]
    s = 1 if task["metrics"]["higher_is_better"] else -1
    rule = task["promotion_rule"]
    missing = []

    def value(obj, key, label):
        v = obj.get(key) if isinstance(obj, dict) else None
        if not is_number(v):
            missing.append(label)
            return None
        return v

    deltas = evaluation.get("deltas")
    d = deltas.get(primary) if isinstance(deltas, dict) else None
    delta = bound = None
    ci_min = rule["ci_lower_min"]
    if not isinstance(d, dict):
        missing.append("deltas.%s" % primary)
    else:
        delta = value(d, "delta", "deltas.%s.delta" % primary)
        if ci_min is not None:
            key = "ci_low" if s == 1 else "ci_high"
            bound = value(d, key, "deltas.%s.%s" % (primary, key))
    metrics = evaluation.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    guards = []
    for g in rule["guards"]:
        m = g["metric"]
        guards.append((g, value(metrics.get("candidate"), m, "metrics.candidate.%s" % m),
                       value(metrics.get("best"), m, "metrics.best.%s" % m)))
    if missing:
        return "keep", "missing %s: the promotion rule cannot be applied" % ", ".join(missing)

    better = "higher" if s == 1 else "lower"
    passed, failed = [], []
    imp = s * delta
    if imp > rule["min_delta"] + EPS:
        passed.append("improvement %s on %s (%s is better) > min_delta %s"
                      % (_g(imp), primary, better, _g(rule["min_delta"])))
    else:
        failed.append("improvement %s on %s (%s is better) is not above min_delta %s"
                      % (_g(imp), primary, better, _g(rule["min_delta"])))
    if ci_min is not None:
        lo = bound if s == 1 else -bound
        if lo >= ci_min - EPS:
            passed.append("improvement interval lower bound %s >= ci_lower_min %s" % (_g(lo), _g(ci_min)))
        else:
            failed.append("improvement interval lower bound %s is below ci_lower_min %s" % (_g(lo), _g(ci_min)))
    for g, cand, best in guards:
        reg = best - cand if g["higher_is_better"] else cand - best
        text = "guard %s regression %s (best %s, candidate %s)" % (g["metric"], _g(reg), _g(best), _g(cand))
        if reg <= g["max_regression"] + EPS:
            passed.append("%s within max_regression %s" % (text, _g(g["max_regression"])))
        else:
            failed.append("%s exceeds max_regression %s" % (text, _g(g["max_regression"])))
    if failed:
        return "keep", "; ".join(failed)
    return "promote", "; ".join(passed)


# Optional task["stop"] fields and their defaults; stop_settings applies them.
STOP_DEFAULTS = {"patience": 3, "max_attempts_per_round": 3}


def stop_settings(task):
    """The effective stop settings of a task: task["stop"] with STOP_DEFAULTS for absent fields.

    Returns {"target", "max_rounds", "patience", "max_attempts_per_round",
    "defaulted"}; "defaulted" lists (sorted) the fields that took their
    default. The task is not changed. Every reader of these settings uses
    this function, so the defaults live in one place.
    """
    stop = task["stop"]
    out = {"target": stop["target"], "max_rounds": stop["max_rounds"]}
    for key, default in STOP_DEFAULTS.items():
        out[key] = stop.get(key, default)
    out["defaulted"] = sorted(k for k in STOP_DEFAULTS if k not in stop)
    return out


# Optional task["flow"] fields and their defaults; flow_settings applies them.
FLOW_DEFAULTS = {"max_returns": 2, "max_continues": 10, "checkin_minutes": None, "max_checkins": 48,
                 "page_refresh_seconds": 60, "status_minutes": 5}


def flow_settings(task):
    """The effective flow settings of a task: task["flow"] with FLOW_DEFAULTS for absent fields.

    Returns {"max_returns", "max_continues", "checkin_minutes", "max_checkins", "page_refresh_seconds",
    "status_minutes", "defaulted"}. max_returns: how many times a module may be sent back in a round
    (rsi.flow.returns_used: its dispatches' returns plus its dispatches beyond the first) before a decision defaults to
    asking the user. max_continues: how many times the engine continues a dispatch's session by itself after a long
    job's marker says ok (the dispatch's "continued"); at the cap the user is asked. checkin_minutes: while a long
    job's marker is not there, its session is checked in on every this many minutes (rsi.flow: resume mode checkin;
    null or absent: no check-ins); max_checkins: how many check-ins a dispatch gets before the user is asked.
    page_refresh_seconds: while `rsi run` waits (for agents, or for a long job's marker with check-ins on) it takes a
    GPU snapshot and rewrites the pages this often; status_minutes: and prints a status line to stderr when it has
    printed none for this long (rsi.engine.Engine.tick). "defaulted" lists (sorted) the fields that took their
    default; the task is not changed. flow.tune, from before train and tune were merged, is still a valid field and
    has no effect.
    """
    flow = task.get("flow") or {}
    out = dict((key, flow.get(key, default)) for key, default in FLOW_DEFAULTS.items())
    out["defaulted"] = sorted(k for k in FLOW_DEFAULTS if k not in flow)
    return out


# The repository root: a relative features.adapter is taken from it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def features_settings(task):
    """The effective features settings of a task: {"adapter", "constraints"} from task["features"], the adapter
    None (the generic views) and the constraints [] when absent. The adapter is as written; adapter_path resolves
    it."""
    f = task.get("features") or {}
    return {"adapter": f.get("adapter"), "constraints": list(f.get("constraints", []))}


def adapter_path(task):
    """The absolute path of the task's features adapter (a relative path is taken from the repository root), or
    None when the task has none."""
    p = features_settings(task)["adapter"]
    if p is None:
        return None
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(REPO_ROOT, p))


def verbatim_paths(task):
    """The task's input.verbatim, [] when absent: the state paths ("[]" for list items, e.g. "messages[].content")
    whose text is a row's own request, never edited by data generation or prompt variation."""
    return list((task.get("input") or {}).get("verbatim", []))


# The baselines of ③ Compare that run on GPUs (modules/compare: one long job; their milestone scoring only predicts
# and is not held by the compute budget, which the milestone question says).
GPU_BASELINES = ("zero_shot", "few_shot", "team_recipe")

# Optional task["compare"] fields and their defaults; compare_settings applies them. A dict default is merged one
# level deep with the task's (the task's few_shot.k replaces only k; its grids.<name> adds or replaces one grid).
COMPARE_DEFAULTS = {"seed": 0, "python": None, "max_train_rows": None, "grids": {}, "zero_shot": {"base": None},
                    "few_shot": {"base": None, "k": 4, "max_example_chars": 2000},
                    "gpu_hours": {"zero_shot": 0.1, "few_shot": 0.1, "predict": 0.1}}


def compare_settings(task):
    """The effective settings of ③ Compare: task["compare"] with COMPARE_DEFAULTS for absent fields, plus
    "team_recipe" (the task's config, None when absent); None when the task has no compare section (③ is off).
    The task is not changed."""
    c = task.get("compare")
    if c is None:
        return None
    out = copy.deepcopy(COMPARE_DEFAULTS)
    for key, value in c.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key].update(copy.deepcopy(value))
        else:
            out[key] = copy.deepcopy(value)
    out.setdefault("team_recipe", None)
    return out


# The decision policy (docs/interfaces/task.md, policy; modules/common/dpolicy): its calibrations, the calibrations and
# decision rules that need a binary task, and the default select share of the valset.
POLICY_CALIBRATIONS = ("none", "platt", "temperature", "isotonic")
BINARY_CALIBRATIONS = ("platt", "isotonic")
THRESHOLD_RULES = ("max_f1", "cost", "precision_at", "recall_at", "top_frac", "fixed")
POLICY_SELECT_FRACTION = 0.4


def policy_settings(task):
    """The effective settings of the task's decision policy, None when the task has no policy section (no policy is
    fitted): {"objective", "calibrations", "rules", "split", "min_lo"} with the defaults for absent fields:
    - objective: f1 for a binary task, log_loss for choice and score;
    - calibrations: binary none, platt, temperature, isotonic; else none, temperature;
    - rules: [max_f1] for the objective f1, [cost] for expected_cost, else [argmax] (a non-binary task: argmax);
    - split: {"by", "key", "select_fraction"}: by time on data.split.key when data.split.by is time, else by group on
      meta.state_id; select_fraction 0.4. A split the task writes in part keeps its fields; a key it leaves out is
      data.split.key for a time split (None when the data split is not by time) and meta.state_id for a group split;
    - min_lo: 0.
    The task is not changed."""
    p = task.get("policy")
    if p is None:
        return None
    binary = task["decision"]["type"] == "binary"
    objective = p.get("objective") or ("f1" if binary else "log_loss")
    if "calibrations" in p:
        calibrations = list(p["calibrations"])
    else:
        calibrations = list(POLICY_CALIBRATIONS) if binary else ["none", "temperature"]
    if "rules" in p:
        rules = copy.deepcopy(p["rules"])
    else:
        rule = {"f1": "max_f1", "expected_cost": "cost"}.get(objective, "argmax") if binary else "argmax"
        rules = [{"rule": rule}]
    data_split = (task.get("data") or {}).get("split") or {}
    given = p.get("split") or {}
    by = given.get("by") or ("time" if data_split.get("by") == "time" else "group")
    if "key" in given:
        key = given["key"]
    elif by == "time":
        key = data_split.get("key") if data_split.get("by") == "time" else None
    else:
        key = "meta.state_id"
    split = {"by": by, "key": key, "select_fraction": given.get("select_fraction", POLICY_SELECT_FRACTION)}
    return {"objective": objective, "calibrations": calibrations, "rules": rules, "split": split,
            "min_lo": p.get("min_lo", 0)}


# Default of the task's optional delivery.reproduction_tolerance (R43).
REPRODUCTION_TOL_DEFAULT = 1e-6


def reproduction_tolerance(task):
    """How far a delivery's predictions_val probabilities may be from its candidate's:
    task["delivery"]["reproduction_tolerance"], or REPRODUCTION_TOL_DEFAULT when absent."""
    return task.get("delivery", {}).get("reproduction_tolerance", REPRODUCTION_TOL_DEFAULT)


def max_decision_mismatches(task, rows):
    """How many of a delivery's `rows` predictions_val rows may have another decision than the candidate's
    (each still within the reproduction tolerance): floor(task["delivery"]["max_decision_mismatch_rate"]
    x rows), the rate defaulting to 0. rsi cannot know the model's decision rule, so near-boundary flips
    are allowed as a capped count."""
    rate = task.get("delivery", {}).get("max_decision_mismatch_rate", 0)
    return int(math.floor(rate * rows + EPS))


def completed_rounds(state):
    """Rounds whose decision is set (promote or keep)."""
    return sum(1 for r in state["rounds"].values() if r.get("decision") is not None)


def ledger(task, state):
    """[(used key, label, used, budget)] for each resource of RESOURCES, in order: STATE budget_used's amount (a
    missing key, as in a workspace made before the ledger, is 0) and task["budget"]'s (None: no budget)."""
    used, budget = state.get("budget_used") or {}, task.get("budget") or {}
    out = []
    for ukey, bkey, label in RESOURCES:
        u, b = used.get(ukey), budget.get(bkey)
        out.append((ukey, label, u if is_number(u) else 0, b if is_number(b) else None))
    return out


def reached(key, used, budget):
    """True when the use `used` of resource `key` (a used key of RESOURCES) has reached its budget `budget` (None: no
    budget). A budget of 0 for anything but GPU hours (e.g. annotation_items 0: no annotation) is reached only once
    something is used; GPU hours keep their rule (a budget of 0 is reached at once)."""
    if budget is None or used < budget - EPS:
        return False
    return key == "compute_hours" or budget > EPS or used > EPS


def exhausted(task, state):
    """[(label, used, budget)] for every resource with a budget that its use has reached (reached)."""
    return [(label, u, b) for k, label, u, b in ledger(task, state) if reached(k, u, b)]


def amount_text(key, x):
    """An amount of resource `key` (a used key of RESOURCES; None: a budget, as written) for people: dollars used
    with two decimals, the rest as %.6g; "–" for None (no budget)."""
    if x is None:
        return "–"
    return "%.2f" % x if key in USD_KEYS else _g(x)


def check_stop(task, state, best_primary):
    """Return (stop, reason) from task["stop"], task["budget"] and the run state.

    Stops when the best version's primary metric reaches stop.target (at or
    above it when higher is better, at or below otherwise), the completed
    rounds reach max_rounds, no_promotion_streak reaches patience (default 3,
    see stop_settings), or the use of any resource with a budget reaches it
    (RESOURCES: budget.compute_hours, api_cost, agent_cost, annotation_items).
    A run already stopped stays stopped.
    """
    stopped = state.get("stopped")
    if stopped:
        return True, "the run stopped at %s: %s" % (stopped.get("at"), stopped.get("reason"))
    stop = stop_settings(task)
    primary = task["metrics"]["primary"]
    higher = task["metrics"]["higher_is_better"]
    reasons, status = [], []

    target = stop["target"]
    if target is not None:
        if best_primary is None:
            status.append("no best %s yet (target %s)" % (primary, _g(target)))
        elif (best_primary >= target) if higher else (best_primary <= target):
            reasons.append("best version %s %s reached target %s" % (primary, _g(best_primary), _g(target)))
        else:
            status.append("best %s %s (target %s)" % (primary, _g(best_primary), _g(target)))

    done = completed_rounds(state)
    if done >= stop["max_rounds"]:
        reasons.append("%s reached max_rounds %d" % (plural(done, "completed round"), stop["max_rounds"]))
    else:
        status.append("%d of max_rounds %d completed" % (done, stop["max_rounds"]))

    streak = state["no_promotion_streak"]
    if streak >= stop["patience"]:
        reasons.append("%s in a row without promotion reached patience %d" % (plural(streak, "round"),
                                                                                stop["patience"]))
    else:
        status.append("no-promotion streak %d of patience %d" % (streak, stop["patience"]))

    for key, label, used, budget in ledger(task, state):
        if reached(key, used, budget):
            reasons.append("%s used %s reached budget %s" % (label, _g(used), _g(budget)))
        elif budget is not None:
            status.append("%s %s of %s" % (label, _g(used), _g(budget)))

    if reasons:
        return True, "; ".join(reasons)
    return False, "no stop condition met (%s)" % ", ".join(status)
