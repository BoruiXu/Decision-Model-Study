"""Shared test fixtures: fictional tasks and decision rows."""
import json
import os

from rsi import metrics as M
from rsi.rules import decide_promotion


def binary_task():
    return {
        "kind": "task",
        "task_id": "spam-check",
        "version": 1,
        "decision": {
            "type": "binary",
            "instructions": "Decide whether a message is spam.",
            "labels": ["yes", "no"],
            "label_meanings": {"yes": "the message is spam", "no": "the message is not spam"},
            "positive": "yes",
        },
        "input": {"format": "messages", "description": "A single user message."},
        "data": {"sources": [], "label_sources": ["rule", "human"]},
        "metrics": {
            "primary": "auc",
            "higher_is_better": True,
            "others": ["log_loss"],
            "baselines": ["majority"],
        },
        "promotion_rule": {
            "min_delta": 0.01,
            "ci_lower_min": None,
            "guards": [{"metric": "log_loss", "higher_is_better": False, "max_regression": 0.02}],
        },
        "stop": {"target": None, "max_rounds": 5, "patience": 2},
        "budget": {"compute_hours": None, "annotation_items": None, "api_cost": None},
        "resources": {},
        "eval_use_limit": 3,
        "approvals": ["task_change", "over_budget", "use_testset"],
    }


def choice_task():
    t = binary_task()
    t["task_id"] = "txn-review"
    t["decision"] = {
        "type": "choice",
        "instructions": "Pick the handling for a transaction.",
        "labels": ["allow", "block", "review"],
        "label_meanings": {
            "allow": "let it through",
            "block": "reject it",
            "review": "send to a human",
        },
    }
    t["metrics"]["primary"] = "accuracy"
    t["metrics"]["others"] = ["log_loss"]
    return t


def full_row(i, task, **over):
    labels = task["decision"]["labels"]
    if task["input"]["format"] == "fields":
        inp = {"fields": {f: "v%d" % i for f in task["input"].get("fields", [])}}
    else:
        inp = {"messages": [{"role": "user", "content": "message %d" % i}]}
    target = {l: 0.0 for l in labels}
    target[labels[i % len(labels)]] = 1.0
    row = {
        "id": "q%d" % i,
        "input": inp,
        "target": target,
        "meta": {
            "source": "synthetic",
            "group": "g%d" % (i % 2),
            "cluster": "c%d" % i,
            "label_source": "rule",
            "n_obs": 1,
        },
    }
    row.update(over)
    return row


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def input_row(i, task):
    r = full_row(i, task)
    return {"id": r["id"], "input": r["input"],
            "meta": {"group": r["meta"]["group"], "cluster": r["meta"]["cluster"]}}


def label_row(i, task):
    r = full_row(i, task)
    return {"id": r["id"], "target": r["target"],
            "meta": {"source": "synthetic", "label_source": "rule", "n_obs": 1}}


DEFAULT_IDS = range(4)


class Opts(object):
    """What make_artifact's builders read besides the task.

    ids: rows written (q<i>); wrong: ids whose predictions pick the wrong label;
    train_ids: the trainset rows the baselines come from; best_wrong: wrong ids of
    the best version's predictions (evaluation); test_ids: testset rows (milestone
    evaluation); val_ids, val_wrong: a delivery's predictions_val, which reproduce
    the candidate's (a candidate built with ids=val_ids, wrong=val_wrong); fields:
    the manifest overrides given to make_artifact.
    """

    def __init__(self, ids=None, wrong=(), train_ids=None, best_wrong=(), test_ids=None, fields=None,
                 val_ids=None, val_wrong=()):
        self.given_ids = ids
        self.ids = list(DEFAULT_IDS if ids is None else ids)
        self.wrong = set(wrong)
        self.train_ids = list(DEFAULT_IDS if train_ids is None else train_ids)
        self.best_wrong = set(best_wrong)
        self.test_ids = list(DEFAULT_IDS if test_ids is None else test_ids)
        self.val_ids = list(DEFAULT_IDS if val_ids is None else val_ids)
        self.val_wrong = set(val_wrong)
        self.fields = fields or {}


def _full_files(d, task, o):
    write_jsonl(os.path.join(d, "rows.jsonl"), [full_row(i, task) for i in o.ids])
    return {"rows": {"path": "rows.jsonl", "rows": len(o.ids)}}


def _trainset_files(d, task, o):
    files = _full_files(d, task, o)
    with open(os.path.join(d, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump({"passed": True}, f)
    files["quality_report"] = {"path": "quality_report.json"}
    return files


def _valset_files(d, task, o):
    files = _full_files(d, task, o)
    write_jsonl(os.path.join(d, "inputs.jsonl"), [input_row(i, task) for i in o.ids])
    files["inputs"] = {"path": "inputs.jsonl", "rows": len(o.ids)}
    return files


def _split_files(d, task, o):
    write_jsonl(os.path.join(d, "inputs.jsonl"), [input_row(i, task) for i in o.ids])
    write_jsonl(os.path.join(d, "labels.jsonl"), [label_row(i, task) for i in o.ids])
    return {"inputs": {"path": "inputs.jsonl", "rows": len(o.ids)},
            "labels": {"path": "labels.jsonl", "rows": len(o.ids)}}


def prediction_row(i, task, wrong=False):
    """Prediction for q<i>: probability 0.9 on its decision, the rest shared by the other labels.

    The decision is the label of full_row(i) (right), or the next label when `wrong`.
    """
    labels = task["decision"]["labels"]
    k = len(labels)
    decision = labels[(i + 1) % k] if wrong else labels[i % k]
    probs = {l: 0.1 / (k - 1) for l in labels}
    probs[decision] = 0.9
    return {"id": "q%d" % i, "probs": probs, "decision": decision, "uncertainty": 0.1}


def predictions(task, ids, wrong=()):
    return [prediction_row(i, task, i in wrong) for i in ids]


def feedback_row(i, task, **over):
    labels = task["decision"]["labels"]
    r = prediction_row(i, task)
    row = {
        "id": "call%d" % i,
        "input": full_row(i, task)["input"],
        "output": {"probs": r["probs"], "decision": r["decision"]},
        "model": "candidate-0001",
        "latency_ms": 12.5,
        "timestamp": "2026-11-21T00:00:00Z",
        "outcome": labels[0] if i % 2 else None,
    }
    row.update(over)
    return row


def _pred_file(d, task, name, ids=None, wrong=()):
    ids = list(DEFAULT_IDS if ids is None else ids)
    write_jsonl(os.path.join(d, name + ".jsonl"), predictions(task, ids, wrong))
    return {"path": name + ".jsonl", "rows": len(ids)}


def _write_executable(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o755)


# The prediction entry of a fixture candidate (model.predict): rsi never runs it, so it is a stub.
PREDICT_STUB = "#!/bin/sh\necho 'fixture candidate: no model behind this entry' >&2\nexit 1\n"
# code_version of fixture candidates: a name, not a local file, so registration does not hash it.
CODE_VERSION = {"entry": "fixture-trainer", "sha256": "1" * 64}


def _candidate_files(d, task, o):
    _write_executable(os.path.join(d, "predict"), PREDICT_STUB)
    return {"predictions_val": _pred_file(d, task, "predictions_val", o.ids, o.wrong)}


def _delivery_files(d, task, o):
    """predictions_eval on o.ids, predictions_val on o.val_ids, and the package directory package/."""
    os.makedirs(os.path.join(d, "package"), exist_ok=True)
    _write_executable(os.path.join(d, "package", "predict"), PREDICT_STUB)
    return {"predictions_eval": _pred_file(d, task, "predictions_eval", o.ids, o.wrong),
            "predictions_val": _pred_file(d, task, "predictions_val", o.val_ids, o.val_wrong)}


def _feedback_files(d, task, o):
    ids = list(range(3)) if o.given_ids is None else o.ids
    write_jsonl(os.path.join(d, "trace.jsonl"), [feedback_row(i, task) for i in ids])
    return {"trace": {"path": "trace.jsonl", "rows": len(ids)}}


def _no_files_report(d, task, o):
    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write("# report\n")
    return {"report": {"path": "report.md"}}


def _no_files(d, task, o):
    return {}


def _failure():
    return {"issue": "misses short messages", "evidence": "slice len<5 auc 0.55",
            "attribution": "data_coverage", "module": "data"}


def _plan_issue():
    f = _failure()
    f.update({"change": "add short messages",
              "expected": {"metric": "auc", "direction": "up", "size": 0.02}})
    return f


# ---- reported numbers, computed with rsi.metrics as the modules would with `rsi score` ----

def known_metrics(task):
    """The primary and other metrics of the task that rsi computes, primary first."""
    m = task["metrics"]
    out = []
    for name in [m["primary"]] + list(m["others"]):
        if name in M.KNOWN_METRICS and name not in out:
            out.append(name)
    return out


def scores(task, preds, labels):
    """metric -> value for every known metric of the task (a metric undefined on the rows is left out)."""
    out = {}
    for name in known_metrics(task):
        try:
            out[name] = M.score(name, preds, labels, task)
        except ValueError:
            pass
    return out


def baseline_scores(task, train_ids, labels, groups=None):
    train = [full_row(i, task) for i in train_ids]
    return dict((b, scores(task, M.baseline_predictions(b, train, labels, task, groups), labels))
                for b in task["metrics"]["baselines"] if b in M.KNOWN_BASELINES)


def _gate(task, metrics_val, baselines_val):
    """train_gate from the numbers: the primary metric beats every computed baseline (strictly)."""
    primary, higher = task["metrics"]["primary"], task["metrics"]["higher_is_better"]
    value = metrics_val[primary]
    beaten, failed = [], []
    for b, values in baselines_val.items():
        if primary not in values:
            continue
        ok = value > values[primary] if higher else value < values[primary]
        (beaten if ok else failed).append("%s %.6g" % (b, values[primary]))
    if failed:
        return {"passed": False, "detail": "%s %.6g does not beat %s" % (primary, value, ", ".join(failed))}
    return {"passed": True, "detail": "%s %.6g beats %s" % (primary, value, ", ".join(beaten) or "no baseline")}


def _candidate_extra(task, o):
    labels = [full_row(i, task) for i in o.ids]
    metrics_val = scores(task, predictions(task, o.ids, o.wrong), labels)
    metrics_val.setdefault(task["metrics"]["primary"], 0.8)  # a primary rsi does not compute
    baselines_val = baseline_scores(task, o.train_ids, labels)
    return {
        "producer": "train",
        "trainset_id": "trainset-0001",
        "valset_id": "valset-0001",
        "config": {"lr": 0.001},
        "config_source": "default",
        "code_version": dict(CODE_VERSION),
        "model": {"location": "/models/m1", "sha256": None, "load": "load_model(path)", "predict": "predict"},
        "metrics_val": metrics_val,
        "baselines_val": baselines_val,
        "train_gate": _gate(task, metrics_val, baselines_val),
        "cost": {"compute_hours": 1.5},
    }


def _evaluation_extra(task, o):
    """An evaluation whose numbers are computed from the rows: the delivery's predictions on
    o.ids (wrong o.wrong) against the evalset labels, the best version's (wrong o.best_wrong)
    when best_before is set, baselines from o.train_ids, metrics_test on o.test_ids."""
    primary = task["metrics"]["primary"]
    best_before = o.fields.get("best_before", {"version": "v1", "evaluation_id": "evaluation-0001"})
    uses_testset = o.fields.get("uses_testset", False)
    labels = [label_row(i, task) for i in o.ids]
    inputs = [input_row(i, task) for i in o.ids]
    preds = predictions(task, o.ids, o.wrong)
    metrics = {"candidate": scores(task, preds, labels), "best": None,
               "baselines": baseline_scores(task, o.train_ids, labels, M.groups_of(inputs))}
    metrics["candidate"].setdefault(primary, 0.85)  # a primary rsi does not compute
    deltas = {}
    if best_before is not None:
        best = predictions(task, o.ids, o.best_wrong)
        metrics["best"] = scores(task, best, labels)
        metrics["best"].setdefault(primary, 0.8)
        if primary in M.KNOWN_METRICS:
            deltas[primary] = M.paired_delta(primary, preds, best, labels, task, M.clusters_of(inputs),
                                             **M.ci_settings(task))
        else:
            deltas[primary] = {"delta": 0.05, "ci_low": 0.01, "ci_high": 0.09}
    out = {
        "producer": "judge",
        "delivery_id": "delivery-0001",
        "candidate_id": "candidate-0002",
        "evalset_id": "evalset-0001",
        "best_before": best_before,
        "metrics": metrics,
        "deltas": deltas,
        "failures": [_failure()],
        "uses_testset": uses_testset,
    }
    if uses_testset:
        test_labels = [label_row(i, task) for i in o.test_ids]
        out["metrics_test"] = scores(task, predictions(task, o.test_ids, o.wrong), test_labels)
    return out


def _finish_evaluation(task, manifest):
    """Unless given, the recommendation is what the promotion rule gives on the final numbers
    (keep for a milestone evaluation, which decides nothing)."""
    if "recommendation" in manifest:
        return
    if manifest.get("uses_testset"):
        manifest["recommendation"] = {"decision": "keep", "reason": "milestone release"}
        return
    decision, reason = decide_promotion(task, manifest, manifest["best_before"] is not None, True)
    manifest["recommendation"] = {"decision": decision, "reason": reason}


# kind -> (function writing files and returning the manifest "files", function giving the
# kind's manifest fields); both take (…, task, Opts).
ARTIFACT_BUILDERS = {
    "trainset": (_trainset_files, lambda task, o: {
        "label_method": "rule",
        "parent_trainset": None,
        "changes": "initial version",
        "quality": {"passed": True,
                    "checks": [{"name": "dup_rate", "value": 0.0, "threshold": 0.05, "passed": True}]},
    }),
    "valset": (_valset_files, lambda task, o: {}),
    "evalset": (_split_files, lambda task, o: {}),
    "testset": (_split_files, lambda task, o: {}),
    "candidate": (_candidate_files, _candidate_extra),
    "search": (_no_files, lambda task, o: {
        "producer": "tune",
        "trainset_id": "trainset-0001",
        "valset_id": "valset-0001",
        "strategy": "random",
        "budget": {"compute_hours": 10},
        "trials": [
            {"candidate_id": "candidate-0001", "config": {"lr": 0.001}, "metric_val": 0.8,
             "compute_hours": 2, "status": "ok"},
            {"candidate_id": "candidate-0002", "config": {"lr": 0.01}, "metric_val": 0.85,
             "compute_hours": 3, "status": "ok"},
        ],
        "default_candidate": "candidate-0001",
        "best_candidate": "candidate-0002",
        "gain_vs_default": {"metric": task["metrics"]["primary"], "delta": 0.05},
    }),
    "delivery": (_delivery_files, lambda task, o: {
        "producer": "deliver",
        "candidate_id": "candidate-0002",
        "evalset_id": "evalset-0001",
        "valset_id": "valset-0001",
        "package": {"location": "package", "entry": "package/predict --inputs <inputs.jsonl> --out <out.jsonl>",
                    "environment": "python3"},
        "runtime": {"latency_ms_p50": 5, "latency_ms_p95": 9, "cost_per_1k": None,
                    "consistent_with_offline": True},
    }),
    "feedback": (_feedback_files, lambda task, o: {
        "producer": "deliver",
        "delivery_id": "delivery-0001",
    }),
    "evaluation": (_no_files_report, _evaluation_extra),
    "plan": (_no_files, lambda task, o: {
        "producer": "judge",
        "evaluation_id": "evaluation-0001",
        "issues": [_plan_issue()],
        "entry_module": "data",
    }),
    "record": (_no_files, lambda task, o: {
        "producer": "data",
        "module": "data",
        "owner": "alice",
        "failure": "short messages misclassified",
        "hypothesis": "coverage gap on short messages",
        "action": "added 200 short messages",
        "result": {"metric": "auc", "before": 0.8, "after": 0.85, "conclusion": "success"},
        "knowledge": "short slices need explicit coverage",
        "automation": "manual",
        "artifacts": ["trainset-0002"],
    }),
}


def make_artifact(kind, dir, task, ids=None, wrong=(), train_ids=None, best_wrong=(), test_ids=None,
                  val_ids=None, val_wrong=(), **fields):
    """Write a minimal valid artifact directory and return its path.

    `ids` (ints) picks the decision rows q<i> written to row files (rows,
    inputs/labels, predictions, trace); the default is q0..q3 (q0..q2 for a
    feedback trace). Predictions of the ids in `wrong` pick the wrong label.
    Candidates and evaluations report the numbers rsi.metrics computes from
    these rows (see Opts for train_ids, best_wrong, test_ids, val_ids and
    val_wrong), and the train gate and recommendation those numbers give.
    Other keyword arguments override manifest fields.
    """
    os.makedirs(dir, exist_ok=True)
    build, extra = ARTIFACT_BUILDERS[kind]
    o = Opts(ids, wrong, train_ids, best_wrong, test_ids, fields, val_ids, val_wrong)
    manifest = {
        "kind": kind,
        "id": None,
        "task_id": task["task_id"],
        "task_version": task["version"],
        "round": 1,
        "producer": "data",
        "created_at": "2026-11-21T00:00:00Z",
        "parents": [],
        "files": build(dir, task, o),
        "notes": "",
    }
    manifest.update(extra(task, o))
    manifest.update(fields)
    if kind == "evaluation":
        _finish_evaluation(task, manifest)
    with open(os.path.join(dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return dir


def make_task_dir(dir, task):
    """Write a task artifact directory (task.json and TASK.md) and return its path."""
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "task.json"), "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dir, "TASK.md"), "w", encoding="utf-8") as f:
        f.write("# 任务说明：%s\n\nversion %d\n" % (task["task_id"], task["version"]))
    return dir


def update_manifest(dir, **fields):
    """Override manifest fields of an artifact directory in place; return the path."""
    p = os.path.join(dir, "manifest.json")
    with open(p, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.update(fields)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return dir


def add_predictions(dir, task, logical, ids, wrong=(), **fields):
    """Write <logical>.jsonl with predictions for q<i>, i in ids, and list it in the manifest."""
    p = os.path.join(dir, "manifest.json")
    with open(p, encoding="utf-8") as f:
        files = json.load(f)["files"]
    files[logical] = _pred_file(dir, task, logical, ids, wrong)
    return update_manifest(dir, files=files, **fields)
