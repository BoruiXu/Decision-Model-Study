"""The command line (modules/compare/SKILL.md is the contract).

    python -m compare run --brief <brief.json> --outdir <dir> --mode <mode> [--feedback <file>]
    python -m compare gpujob --brief <brief.json> --outdir <dir> --marker <marker>   (started by compare itself)

Purpose compare: modes first, resume and fresh plan every baseline of compare.baselines (rsi reuse rules:
compare.reuse), train or re-predict the CPU ones inline and start the GPU ones as one long job (compare.gpu); with a
GPU job they hand back long_job with its marker, else the new baseline directories (status ok); when there is none
(every baseline reused or skipped), they hand back the comparison (compare.report). continue collects the job's
results and hands back every new baseline (or, with none, the comparison); checkin hands back long_job while the job
runs. followup hands back the comparison: the baselines registered-<n>.json (beside the --feedback file) names, and the
reused ones with their ids from the brief.
A name the module does not know, or a GPU baseline that cannot run (the compute budget, no config), is not trained:
skipped.json says why. Purpose milestone_compare: compare.milestone. Every run writes handback.json and exits 0; an
error is a `failed` hand-back whose notes say what went wrong (the traceback goes to stderr, the dispatch's log).
"""
import argparse
import collections
import json
import os
import re
import sys
import traceback

from . import COMPARE_VERSION
from . import cpu
from . import gpu
from . import milestone
from . import preds
from . import write as W
from .inputs import Inputs, read_json
from .report import write_comparison
from .reuse import plan, unknown

MODES = ("first", "resume", "fresh", "continue", "followup", "checkin", "answer")


def _parser():
    p = argparse.ArgumentParser(prog="python -m compare", description="The compare module (③ Compare).")
    sub = p.add_subparsers(dest="command")
    r = sub.add_parser("run", help="one launch of a compare dispatch")
    r.add_argument("--brief", required=True, help="the dispatch's brief.json")
    r.add_argument("--outdir", required=True, help="the dispatch's output directory")
    r.add_argument("--mode", required=True, choices=MODES, help="the launch mode")
    r.add_argument("--feedback", help="the feedback or history file of this launch")
    g = sub.add_parser("gpujob", help="the GPU baselines' long job (started by compare run)")
    g.add_argument("--brief", required=True)
    g.add_argument("--outdir", required=True)
    g.add_argument("--marker", required=True)
    return p


def feature_table(inputs):
    """The CPU baselines' feature table (cpu.feature_table) of the trainset, the valset and the evalset inputs."""
    st = inputs.settings
    return cpu.feature_table(inputs.task, inputs.train_rows, inputs.val_rows, inputs.eval_inputs,
                             st["max_train_rows"], st["seed"])


def _kept(outdir, p, inputs):
    """The subdirectory bl-<name> an earlier launch of this dispatch left for plan `p`, when it matches the plan
    (spec_sha256, sets, reused_from) and is valid; else None."""
    sub = "bl-%s" % p.name
    path = os.path.join(outdir, sub, "manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        m = read_json(path)
    except ValueError:
        return None
    ids = inputs.ids
    want = {"spec_sha256": p.spec_sha256, "trainset_id": ids["trainset"], "valset_id": ids["valset"],
            "evalset_id": ids["evalset"], "reused_from": p.source_id if p.action == "repredict" else None}
    if any(m.get(k) != v for k, v in want.items()):
        return None
    return sub if not W.check(outdir, sub, "baseline", inputs.task) else None


def _note_of(outdir, sub):
    return read_json(os.path.join(outdir, sub, "manifest.json")).get("notes", sub)


def _skipped_text(reasons):
    return "; ".join("%s (%s)" % (n, r) for n, r in reasons.items()) or "none"


def comparison(inputs, outdir, baseline_ids, reasons):
    """The second part: the comparison of the brief's candidate with the registered baselines `baseline_ids` (name ->
    id, in the order of compare.baselines), the skip reasons `reasons` copied (compare.report.write_comparison)."""
    order = inputs.settings["baselines"]
    ids = collections.OrderedDict(sorted(baseline_ids.items(), key=lambda kv: (
        order.index(kv[0]) if kv[0] in order else len(order), kv[0])))
    sub = write_comparison(inputs, outdir, ids, reasons)
    m = read_json(os.path.join(outdir, sub, "manifest.json"))
    return "ok", [sub], ("%s. Compared: %s; skipped: %s." % (
        m["notes"], ", ".join("%s (%s)" % kv for kv in m["baselines"].items()) or "none",
        _skipped_text(m["skipped"]))), None


def registered(outdir, feedback):
    """name -> id of the baselines registered-<n>.json beside the feedback file `feedback` (feedback-<n>.md) lists
    ([{"dir", "kind", "id"}], written by the engine for a followup), each name from its subdirectory's manifest; {}
    without that file."""
    m = re.match(r"^feedback-(\d+)\.md$", os.path.basename(feedback or ""))
    path = os.path.join(os.path.dirname(os.path.abspath(feedback)), "registered-%s.json" % m.group(1)) if m else ""
    if not os.path.isfile(path):
        return {}
    out = {}
    for x in read_json(path):
        mpath = os.path.join(outdir, x.get("dir") or "", "manifest.json")
        if x.get("kind") == "baseline" and x.get("id") and os.path.isfile(mpath):
            out[read_json(mpath)["name"]] = x["id"]
    return out


def followup(inputs, outdir, feedback):
    """Mode followup: the comparison of the baselines the first part registered (registered) and those the plan reuses
    (their ids from the brief; once the brief is rebuilt after the registration, those registered too), with the skip
    reasons of the first part (skipped.json)."""
    ids = dict((p.name, p.source_id) for p in plan(inputs) if p.action == "reuse")
    ids.update(registered(outdir, feedback))
    path = os.path.join(outdir, "skipped.json")
    reasons = read_json(path) if os.path.isfile(path) else {}
    return comparison(inputs, outdir, ids, reasons)


def _gpu_plan(outdir, inputs, p, left, new, notes, reasons, todo, start):
    """GPU plan `p`: its result from a job of this dispatch becomes its baseline directory (or its skip reason);
    without one (start: or with a failed one) it joins `todo` when it can run (gpu.why_not). Returns the GPU hours
    left."""
    item = gpu.item_of(p, inputs)
    res = gpu.finished(outdir, item, retry=start)
    if res is None:
        # a team recipe an earlier job trained (its prediction failed) needs only a prediction now
        trained = p.name == "team_recipe" and p.action == "train" and gpu.trained_team(outdir, item) is not None
        why, left = gpu.why_not(p, inputs, left, predict_only=trained)
        if why:
            reasons[p.name] = why
        else:
            todo.append(item)
        return left
    if res["status"] != "ok":
        reasons[p.name] = "%s failed: %s" % (p.name, res["reason"])
        return left
    sub = gpu.write_baseline(outdir, inputs, item, res, p)
    new.append(sub)
    notes.append(_note_of(outdir, sub))
    return left


def baselines(inputs, outdir, start=True):
    """Modes first, resume and fresh (start: a GPU baseline without a result starts the GPU job) and continue
    (start False: the job has ended): train or re-predict every planned baseline that is not reused, and hand back
    the new baseline directories, long_job while the GPU job runs, or with none the comparison."""
    task, settings = inputs.task, inputs.settings
    reasons = collections.OrderedDict((n, "unknown baseline %s" % n) for n in unknown(settings))
    plans = plan(inputs)
    new, notes, todo = [], [], []
    left = (inputs.budget_left or {}).get("compute_hours")
    table, table_error = None, None
    for p in plans:
        if p.action == "reuse":
            continue
        sub = _kept(outdir, p, inputs)
        if sub is not None:
            new.append(sub)
            notes.append(_note_of(outdir, sub) + " (from an earlier launch)")
            continue
        if p.kind == "gpu":
            left = _gpu_plan(outdir, inputs, p, left, new, notes, reasons, todo, start)
            continue
        if table is None and table_error is None:
            try:
                table = feature_table(inputs)
            except ValueError as e:
                table_error = "no features: %s" % e
        if table_error is not None:
            reasons[p.name] = table_error
            continue
        try:
            P_val, P_eval, config = cpu.fit_cpu(p.name, table.X_tr, table.y, table.X_val, table.X_other,
                                                cpu.grid_of(p.name, settings), settings["seed"], task,
                                                inputs.val_rows)
        except ValueError as e:
            reasons[p.name] = "%s could not be fitted: %s" % (p.name, e)
            continue
        P_val, P_eval, policy = preds.with_policy(task, inputs.val_rows, P_val, P_eval)
        source = p.source_id if p.action == "repredict" else None
        sub = W.baseline_dir(outdir, p.name, inputs, p, config, P_val, P_eval, table.signature, 0.0,
                             reused_from=source, policy=policy)
        errors = W.check(outdir, sub, "baseline", task)
        if errors:
            raise RuntimeError("%s is not a valid baseline: %s" % (sub, "; ".join(errors)))
        new.append(sub)
        notes.append(_note_of(outdir, sub) + (" (re-predicted from %s)" % source if source else ""))
    if todo and start:
        hb = gpu.long_job(inputs, outdir, todo, reasons, notes)
        # None: the job ended before the hand-back: collect it now (start False never starts another)
        return hb if hb is not None else baselines(inputs, outdir, start=False)
    for item in todo:   # the job has ended without a result for it
        reasons[item["name"]] = gpu.no_result(outdir, item)
    W.skipped(outdir, dict(reasons))
    reused = [(p.name, p.source_id) for p in plans if p.action == "reuse"]
    if not new:
        return comparison(inputs, outdir, dict(reused), reasons)
    return "ok", new, ("new: %s. reused: %s. skipped: %s. The comparison follows once these are registered."
                       % ("; ".join(notes), ", ".join("%s (%s)" % r for r in reused) or "none",
                          _skipped_text(reasons))), None


def run(brief, outdir, mode, feedback=None):
    """(status, artifacts, notes, marker) of one launch (`feedback`: the --feedback file)."""
    inputs = Inputs.load(brief)
    if inputs.settings is None:
        return "failed", [], "the task has no compare section, so ③ Compare is off", None
    if inputs.purpose == "milestone_compare":
        return milestone.run_milestone(inputs, outdir, mode)
    if inputs.purpose != "compare":
        return "failed", [], "purpose %s is not supported by compare %s" % (inputs.purpose, COMPARE_VERSION), None
    if mode in ("first", "resume", "fresh"):
        return baselines(inputs, outdir)
    if mode == "continue":
        return baselines(inputs, outdir, start=False)
    if mode == "checkin":
        return gpu.check_in(outdir, lambda: baselines(inputs, outdir, start=False))
    if mode == "followup":
        return followup(inputs, outdir, feedback)
    return "failed", [], "mode %s: compare asks no questions" % mode, None


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "gpujob":
        return gpu.run_job(args.brief, os.path.abspath(args.outdir), os.path.abspath(args.marker))
    if args.command != "run":
        print("compare: give a command: run or gpujob", file=sys.stderr)
        return 2
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    try:
        status, artifacts, notes, marker = run(args.brief, outdir, args.mode, args.feedback)
    except Exception as e:  # noqa: BLE001  (any error is a failed hand-back, never a crash without one)
        traceback.print_exc()
        status, artifacts, notes, marker = "failed", [], "compare failed: %s: %s" % (type(e).__name__, e), None
    W.handback(outdir, status, artifacts, marker=marker, notes=notes)
    print(json.dumps({"status": status, "artifacts": artifacts, "marker": marker}))
    return 0
