"""Milestone scoring (purpose milestone_compare, after the user's yes to the milestone release): each baseline of the
best version's comparison scored on the testset inputs.

For each baseline <name> (id <bid>) of the brief's comparison it writes bl-<name>-test/: a baseline with reused_from
<bid>, its predictions_val and predictions_eval copied, its spec, config and features kept, testset_id the brief's
testset and predictions_test on the testset inputs.
- A CPU baseline is refitted inline on the rows of its trainset with its recorded grid point and seed, and its
  recorded decision policy (policy.json, when it has one) applied; it must reproduce its registered predictions_val
  exactly (probabilities and decisions), else it is skipped ("refit does not reproduce <bid>").
- team_recipe runs its stored model.predict (which applies its own policy.json); zero_shot and few_shot rescore with
  the base model, their recorded policy applied to the new scores; both in the GPU job (compare.gpu), whose hours go
  to the spend file. Milestone scoring only predicts, and the user approved it, so it
  is not held by budget_left.compute_hours.
Every baseline is handed back in one hand-back, after the GPU job when there is one: the milestone's compare step
counts as done once a baseline with predictions_test is registered. Compare gets no testset labels: the judge
computes the testset numbers.
"""
import collections
import os

from rsi.rules import GPU_BASELINES

from . import cpu
from . import gpu
from . import preds
from . import write as W
from .inputs import read_json, read_jsonl, set_file
from .reuse import Plan


def _kept(outdir, sub, bid, inputs):
    """True when an earlier launch left a valid <sub> made from `bid` on this testset."""
    path = os.path.join(outdir, sub, "manifest.json")
    try:
        m = read_json(path)
    except (OSError, ValueError):
        return False
    return (m.get("reused_from") == bid and m.get("testset_id") == inputs.testset_id
            and not W.check(outdir, sub, "baseline", inputs.task))


def refit(inputs, bid):
    """(P_val, P_test) of CPU baseline `bid` refitted on the rows of its trainset (its spec's max_train_rows and seed,
    its config's grid point) and predicting its valset rows and the testset inputs, its recorded decision policy
    applied (preds.Decided rows). ValueError when the brief lacks those rows or the refit does not reproduce its
    registered predictions_val exactly (probabilities and decisions)."""
    b = inputs.baselines[bid]
    m, rows = b["manifest"], inputs.baseline_rows[bid]
    if not rows["train"] or not rows["val"]:
        raise ValueError("the brief gives no rows of %s or %s" % (m.get("trainset_id"), m["valset_id"]))
    spec, point = m["spec"], m["config"]["grid_point"]
    table = cpu.feature_table(inputs.task, rows["train"], rows["val"], inputs.testset_inputs,
                              spec.get("max_train_rows"), spec["seed"])
    grid = collections.OrderedDict((k, [v]) for k, v in point.items())
    P_val, P_test, _ = cpu.fit_cpu(m["name"], table.X_tr, table.y, table.X_val, table.X_other, grid, spec["seed"],
                                   inputs.task, rows["val"])
    policy = preds.recorded_policy(b["dir"])
    if policy is not None:
        labels = inputs.task["decision"]["labels"]
        P_val, P_test = preds.applied(policy, P_val, labels), preds.applied(policy, P_test, labels)
    registered = dict((r["id"], (r["probs"], r["decision"])) for r in read_jsonl(set_file(b["dir"], "predictions_val")))
    mine = preds.rows([r["id"] for r in rows["val"]], P_val, inputs.task["decision"]["labels"])
    if len(registered) != len(mine) or any(registered.get(r["id"]) != (r["probs"], r["decision"]) for r in mine):
        raise ValueError("refit does not reproduce %s" % bid)
    return P_val, P_test


def _cpu(outdir, inputs, name, bid, sub):
    """bl-<name>-test of CPU baseline `bid`, or the reason it is skipped."""
    try:
        _, P_test = refit(inputs, bid)
    except ValueError as e:
        return None, str(e)
    b = inputs.baselines[bid]
    m = b["manifest"]
    p = Plan(name, "cpu", m["spec"], m["spec_sha256"], "test", bid)
    sub = W.baseline_dir(outdir, name, inputs.of_baseline(bid), p, m["config"], set_file(b["dir"], "predictions_val"),
                         set_file(b["dir"], "predictions_eval"), m["features"], 0.0, reused_from=bid,
                         testset_id=inputs.testset_id, P_test=P_test, sub=sub,
                         notes="%s: %s refitted on %s (its grid point and seed), its predictions_val reproduced"
                               % (name, bid, m.get("trainset_id")), policy=preds.recorded_policy(b["dir"]))
    errors = W.check(outdir, sub, "baseline", inputs.task)
    if errors:
        raise RuntimeError("%s is not a valid baseline: %s" % (sub, "; ".join(errors)))
    return sub, None


def _score(inputs, outdir, start):
    """One pass over the comparison's baselines: (status, artifacts, notes, marker)."""
    reasons = collections.OrderedDict()
    new, todo = [], []
    for name, bid in inputs.comparison["baselines"].items():
        sub = "bl-%s-test" % name
        if bid not in inputs.baselines:
            reasons[name] = "the brief does not give %s" % bid
            continue
        if _kept(outdir, sub, bid, inputs):
            new.append(sub)
            continue
        m = inputs.baselines[bid]["manifest"]
        if inputs.baseline_rows[bid]["val"] is None:
            reasons[name] = "the brief gives no rows of %s, the valset of %s" % (m["valset_id"], bid)
            continue
        if name in cpu.CPU_BASELINES:
            sub, why = _cpu(outdir, inputs, name, bid, sub)
            if why:
                reasons[name] = why
            else:
                new.append(sub)
            continue
        if name not in GPU_BASELINES:
            reasons[name] = "compare cannot score the baseline %s on the testset (not one it knows)" % name
            continue
        p = Plan(name, "gpu", m["spec"], m["spec_sha256"], "test", bid)
        item = gpu.item_of(p, inputs, sub)
        res = gpu.finished(outdir, item, retry=start)
        if res is None:
            if start:
                todo.append(item)
            else:
                reasons[name] = gpu.no_result(outdir, item)
        elif res["status"] != "ok":
            reasons[name] = "%s failed: %s" % (name, res["reason"])
        else:
            new.append(gpu.write_baseline(outdir, inputs, item, res, p))
    if todo:
        hb = gpu.long_job(inputs, outdir, todo, reasons, ["ready on the testset: %s" % (", ".join(new) or "none")])
        return hb if hb is not None else _score(inputs, outdir, False)
    W.skipped(outdir, dict(reasons))
    skipped = "; ".join("%s (%s)" % kv for kv in reasons.items()) or "none"
    if not new:
        return "failed", [], ("no baseline of %s could be scored on the testset inputs; skipped: %s"
                              % (inputs.comparison.get("id"), skipped)), None
    return "ok", new, ("the baselines of %s scored on the testset inputs of %s: %s (each with reused_from, its "
                       "valset and evalset predictions copied); skipped: %s"
                       % (inputs.comparison.get("id"), inputs.testset_id, ", ".join(new), skipped)), None


def run_milestone(inputs, outdir, mode):
    """One launch of a milestone_compare dispatch: (status, artifacts, notes, marker). first, resume and fresh score
    every baseline (the GPU ones through the long job); continue collects the job's results; checkin as for
    compare; followup and answer have nothing to do here."""
    if inputs.comparison is None or not inputs.comparison.get("baselines"):
        return "failed", [], "the milestone brief gives no comparison with a baseline to score", None
    if not inputs.testset_inputs or inputs.testset_id is None:
        return "failed", [], "the milestone brief gives no testset inputs", None
    if mode in ("first", "resume", "fresh"):
        return _score(inputs, outdir, True)
    if mode == "continue":
        return _score(inputs, outdir, False)
    if mode == "checkin":
        return gpu.check_in(outdir, lambda: _score(inputs, outdir, False))
    return "failed", [], "mode %s: the milestone's compare step has no second part and asks nothing" % mode, None
