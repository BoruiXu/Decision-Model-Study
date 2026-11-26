"""The comparison, the second part of a compare dispatch: the brief's candidate against each registered baseline on the
valset (report only, never a gate).

Every number is computed with rsi.metrics from the registered prediction files, exactly as registration recomputes it
(docs/interfaces/compare.md): metrics_val (the candidate's predictions_val, which carry its decision policy, and each
baseline's), deltas_val (rsi.metrics.baseline_delta: the paired cluster bootstrap of the primary metric, candidate
minus baseline, clusters from the valset's inputs.jsonl, the task's interval settings) and strongest (rsi.workspace.strongest_of: the best primary
metric, a tie to the alphabetically first name). The table of report.md is rsi.report.comparison_table, the one
REPORT.md shows.
"""
import collections
import os
import shutil

from rsi import metrics as M
from rsi.artifacts import CANDIDATE_KEY
from rsi.report import comparison_table, skipped_lines, strongest_line
from rsi.workspace import strongest_of

from . import preds
from . import write as W
from .inputs import read_json, read_jsonl, set_file

SUB = "comparison"


def baseline_dir(inputs, outdir, name, bid):
    """The directory of registered baseline `bid` named `name`: the brief's copy, else this dispatch's bl-<name>/
    (registration copies it unchanged; a brief built before the registration does not list it yet)."""
    if bid in inputs.baselines:
        return inputs.baselines[bid]["dir"]
    d = os.path.join(outdir, "bl-%s" % name)
    if not os.path.isfile(os.path.join(d, "manifest.json")):
        raise ValueError("the brief gives no %s and the output directory has no bl-%s/" % (bid, name))
    return d


def _off_sets(bm, cm, evalset_id):
    """Why baseline manifest `bm` cannot be compared with candidate manifest `cm` on `evalset_id` (registration would
    refuse it), or None."""
    if (bm["valset_id"], bm["evalset_id"]) == (cm["valset_id"], evalset_id) and \
            bm["trainset_id"] in (None, cm["trainset_id"]):
        return None
    return ("on trainset %s, valset %s and evalset %s; the candidate is on trainset %s and valset %s, the evalset is %s"
            % (bm["trainset_id"], bm["valset_id"], bm["evalset_id"], cm["trainset_id"], cm["valset_id"], evalset_id))


def write_comparison(inputs, outdir, baseline_ids, skipped):
    """Write <outdir>/comparison/: the brief's candidate against each registered baseline of `baseline_ids` (name -> id,
    in order) on the valset, `skipped` (name -> why) copied with the baselines that cannot be compared added (another
    set than the candidate's), and report.md (strongest_line, the table, the skipped lines). Checked with `rsi
    validate comparison` (RuntimeError when it fails); returns "comparison". ValueError when the brief gives no
    candidate, the candidate is on another valset than the brief's, or the primary metric is not one rsi computes."""
    task, ids = inputs.task, inputs.ids
    if not inputs.candidate_dir:
        raise ValueError("the brief gives no candidate to compare")
    cm = read_json(os.path.join(inputs.candidate_dir, "manifest.json"))
    cid = cm.get("id") or os.path.basename(os.path.normpath(inputs.candidate_dir))
    if cm["valset_id"] != ids["valset"]:
        raise ValueError("candidate %s is on valset %s, but the brief's valset is %s" % (cid, cm["valset_id"],
                                                                                         ids["valset"]))
    primary, higher = task["metrics"]["primary"], task["metrics"]["higher_is_better"]
    preds.metric_names(task)  # ValueError for a primary metric rsi does not compute
    labels, clusters = inputs.val_rows, M.clusters_of(inputs.val_inputs)
    cand = read_jsonl(set_file(inputs.candidate_dir, "predictions_val"))
    metrics = collections.OrderedDict([(CANDIDATE_KEY, preds.metric_values(task, cand, labels))])
    skipped = collections.OrderedDict(skipped)
    listed, deltas = collections.OrderedDict(), {}
    for name, bid in baseline_ids.items():
        d = baseline_dir(inputs, outdir, name, bid)
        why = _off_sets(read_json(os.path.join(d, "manifest.json")), cm, ids["evalset"])
        if why:
            skipped[name] = "%s (%s) is %s" % (name, bid, why)
            continue
        rows = read_jsonl(set_file(d, "predictions_val"))
        metrics[name] = preds.metric_values(task, rows, labels)
        listed[name] = bid
        skipped.pop(name, None)
        try:
            deltas[name] = M.baseline_delta(task, cand, rows, labels, clusters)
        except ValueError:  # undefined on every bootstrap draw: no delta for this baseline (the table says n/a)
            pass
    m = {"kind": "comparison", "id": None, "task_id": task["task_id"], "task_version": task["version"],
         "round": inputs.round, "producer": "compare", "created_at": W.now_iso(),
         "parents": [cid, cm["trainset_id"], cm["valset_id"], ids["evalset"]] + list(listed.values()),
         "files": {"report": {"path": "report.md"}}, "candidate_id": cid, "trainset_id": cm["trainset_id"],
         "valset_id": cm["valset_id"], "evalset_id": ids["evalset"], "baselines": listed, "metrics_val": metrics,
         "deltas_val": deltas, "skipped": skipped,
         "strongest": strongest_of(dict((n, metrics[n][primary]) for n in listed), higher)}
    m["notes"] = strongest_line(m, primary)
    final = os.path.join(outdir, SUB)
    tmp = "%s.tmp-%d" % (final, os.getpid())
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    lines = ["# Comparison: %s against the baselines on %s" % (cid, cm["valset_id"]), "",
             "③ Compare is a report only, never a gate. Every number is rsi.metrics on the valset rows (`rsi score`); "
             "Δ is the candidate minus the model, with its paired cluster bootstrap interval (clusters from the valset "
             "inputs).", "", m["notes"], ""] + comparison_table(m, primary) + [""] + skipped_lines(m)
    with open(os.path.join(tmp, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")
    W.write_json(os.path.join(tmp, "manifest.json"), m)
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    errors = W.check(outdir, SUB, "comparison", task)
    if errors:
        raise RuntimeError("%s is not a valid comparison: %s" % (SUB, "; ".join(errors)))
    return SUB
