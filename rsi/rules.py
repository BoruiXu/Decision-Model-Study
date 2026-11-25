"""Fixed rules shared by artifact checks and the orchestrator.

decide_promotion and check_stop are pure: they take dicts and return
(conclusion, reason). The workspace and the CLI call them; the orchestrating
agent never makes these decisions itself.
"""
import math

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


def check_stop(task, state, best_primary):
    """Return (stop, reason) from task["stop"], task["budget"] and the run state.

    Stops when the best version's primary metric reaches stop.target (at or
    above it when higher is better, at or below otherwise), the completed
    rounds reach max_rounds, no_promotion_streak reaches patience (default 3,
    see stop_settings), or the compute hours used reach budget.compute_hours.
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
        reasons.append("%d completed rounds reached max_rounds %d" % (done, stop["max_rounds"]))
    else:
        status.append("%d of max_rounds %d completed" % (done, stop["max_rounds"]))

    streak = state["no_promotion_streak"]
    if streak >= stop["patience"]:
        reasons.append("%d rounds in a row without promotion reached patience %d" % (streak, stop["patience"]))
    else:
        status.append("no-promotion streak %d of patience %d" % (streak, stop["patience"]))

    budget = task["budget"]["compute_hours"]
    used = state["budget_used"]["compute_hours"]
    if budget is not None and used >= budget - EPS:
        reasons.append("compute hours used %s reached budget %s" % (_g(used), _g(budget)))
    elif budget is not None:
        status.append("compute hours %s of %s" % (_g(used), _g(budget)))

    if reasons:
        return True, "; ".join(reasons)
    return False, "no stop condition met (%s)" % ", ".join(status)
