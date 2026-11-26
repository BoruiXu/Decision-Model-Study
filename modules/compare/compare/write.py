"""What the compare module writes: baseline directories (bl-<name>/, bl-<name>-test/ at the milestone), skipped.json
and handback.json in its output directory, and spend lines in the brief's spend file. Every file is written whole and
then moved into place, so a killed run leaves no half-written one."""
import json
import os
import shutil
import time

from rsi.artifacts import validate_artifact

from . import preds
from .inputs import read_json, read_jsonl


def write_json(path, obj):
    tmp = "%s.tmp-%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _notes(name, config, metrics, primary):
    point = ", ".join("%s=%s" % (k, json.dumps(v)) for k, v in sorted((config.get("grid_point") or {}).items()))
    lib = " (%s)" % config["library"] if "library" in config else ""
    return "%s%s: %s (valset %s %.4f, %s trainset rows)" % (name, lib, point or "no grid point", primary,
                                                            metrics[primary], config.get("train_rows", "no"))


def _pred_rows(P, ids, labels, logical):
    """The prediction rows of logical file `logical`: `P` is a list of label -> probability rows in the order of
    `ids` (written as preds.rows), or the path of a prediction rows file copied as it is (its ids must be `ids` when
    those are known)."""
    if not isinstance(P, str):
        return preds.rows(ids, P, labels), None
    rows = read_jsonl(P)
    if ids is not None and sorted(r["id"] for r in rows) != sorted(ids):
        raise ValueError("%s: %s does not cover the expected ids" % (logical, P))
    return rows, P


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:   # another file system, or links not allowed
        shutil.copy2(src, dst)


def link_model(src, dst):
    """Put the stored model of directory `src` (a packed candidate, or a baseline with model.predict) into `dst`: every
    entry of `src` but its manifest.json, the files its manifest lists and prediction rows (predictions_*.jsonl),
    hard-linked (copied across file systems), so a model of many GB costs no space."""
    m = read_json(os.path.join(src, "manifest.json"))
    skip = set(["manifest.json"]) | set(f["path"] for f in (m.get("files") or {}).values())
    for name in sorted(os.listdir(src)):
        if name in skip or (name.startswith("predictions_") and name.endswith(".jsonl")):
            continue
        path = os.path.join(src, name)
        if os.path.isdir(path):
            shutil.copytree(path, os.path.join(dst, name), copy_function=_link_or_copy)
        else:
            _link_or_copy(path, os.path.join(dst, name))


def baseline_dir(outdir, name, inputs, plan, config, P_val, P_eval, features_sig, cost_hours, model=None,
                 reused_from=None, testset_id=None, P_test=None, model_from=None, sub=None, notes=None, policy=None):
    """Write <outdir>/<sub> (default bl-<name>): predictions_val.jsonl (every row of inputs.val_rows),
    predictions_eval.jsonl (every input of inputs.eval_inputs) and, with testset_id, predictions_test.jsonl (every
    input of inputs.testset_inputs); each P_* is a list of label -> probability rows (preds.Decided rows keep their
    policy's decision) or the path of a prediction file to copy (_pred_rows). With `policy` (its decision policy, the
    P_* already applied) it writes policy.json and lists it as the file `policy`, and with the policy's fit report
    (preds.Policy.report) policy_report.json as the file `policy_report`. The manifest names inputs.ids' sets
    (no trainset for zero_shot), metrics_val by rsi.metrics on the valset rows, `model` ({"location", "predict"}) when
    given, its files linked from `model_from`, and `notes` (default: the grid point and the primary metric). The
    directory is built beside and moved into place; returns its name."""
    task, ids = inputs.task, inputs.ids
    labels, primary = task["decision"]["labels"], task["metrics"]["primary"]
    sub = sub or "bl-%s" % name
    final = os.path.join(outdir, sub)
    tmp = "%s.tmp-%d" % (final, os.getpid())
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    sets = [("predictions_val", P_val, [r["id"] for r in inputs.val_rows]),
            ("predictions_eval", P_eval, None if inputs.eval_inputs is None else [r["id"] for r in inputs.eval_inputs])]
    if testset_id is not None:
        sets.append(("predictions_test", P_test, [r["id"] for r in inputs.testset_inputs]))
    files, rows_val = {}, None
    for logical, P, want in sets:
        rows, src = _pred_rows(P, want, labels, logical)
        path = os.path.join(tmp, logical + ".jsonl")
        if src is not None:
            shutil.copyfile(src, path)
        else:
            write_jsonl(path, rows)
        files[logical] = {"path": logical + ".jsonl", "rows": len(rows)}
        rows_val = rows if logical == "predictions_val" else rows_val
    if model_from is not None:
        link_model(model_from, tmp)
    if policy is not None:   # after the model's files: a stored model.predict applies this file
        write_json(os.path.join(tmp, "policy.json"), dict(policy))
        files["policy"] = {"path": "policy.json"}
        if getattr(policy, "report", None) is not None:
            write_json(os.path.join(tmp, "policy_report.json"), policy.report)
            files["policy_report"] = {"path": "policy_report.json"}
    metrics = preds.metric_values(task, rows_val, inputs.val_rows)
    trainset_id = None if name == "zero_shot" else ids["trainset"]
    parents = [x for x in (trainset_id, ids["valset"], ids["evalset"], reused_from, testset_id) if x]
    m = {"kind": "baseline", "id": None, "task_id": task["task_id"], "task_version": task["version"],
         "round": inputs.round, "producer": "compare", "created_at": now_iso(), "parents": parents, "files": files,
         "notes": ("%s (valset %s %.4f)" % (notes, primary, metrics[primary]) if notes
                   else _notes(name, config, metrics, primary)),
         "name": name, "trainset_id": trainset_id, "valset_id": ids["valset"], "evalset_id": ids["evalset"],
         "testset_id": testset_id, "spec": plan.spec, "spec_sha256": plan.spec_sha256, "config": config,
         "features": features_sig, "metrics_val": metrics, "cost": {"compute_hours": float(cost_hours)},
         "reused_from": reused_from}
    if model is not None:
        m["model"] = {"location": model["location"], "predict": model["predict"]}
    write_json(os.path.join(tmp, "manifest.json"), m)
    shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    return sub


def spend(spend_file, source, compute_hours, note=None):
    """Append one line {"at", "source", "compute_hours"[, "note"]} to the spend file (docs/interfaces/spend.md): one
    write of the whole line, in append mode. Nothing without a spend file."""
    if not spend_file:
        return
    line = {"at": now_iso(), "source": source, "compute_hours": max(0.0, float(compute_hours))}
    if note:
        line["note"] = note
    os.makedirs(os.path.dirname(os.path.abspath(spend_file)), exist_ok=True)
    with open(spend_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def check(outdir, sub, kind, task):
    """The validation errors of <outdir>/<sub> as `rsi validate <kind>` reports them ([] when valid)."""
    return validate_artifact(kind, os.path.join(outdir, sub), task)


def skipped(outdir, reasons):
    """<outdir>/skipped.json: baseline name -> why it was not compared (read by the comparison)."""
    write_json(os.path.join(outdir, "skipped.json"), reasons)


def handback(outdir, status, artifacts, marker=None, notes=""):
    """<outdir>/handback.json in the format every module hands back; compare never asks questions."""
    write_json(os.path.join(outdir, "handback.json"), {"status": status, "artifacts": list(artifacts),
                                                       "marker": marker, "notes": notes, "questions": []})
