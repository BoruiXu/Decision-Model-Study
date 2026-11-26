"""Shared test fixtures: fictional tasks and decision rows."""
import json
import os

from rsi import metrics as M
from rsi.datasplit import filled, source_paths
from rsi.rows import input_row_of, question_of
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


# A complete split rule (data.split, docs/interfaces/task.md): by time on meta.extra.step, the other keys left to their
# defaults (rsi.datasplit.SPLIT_DEFAULTS).
TIME = {"by": "time", "key": "meta.extra.step",
        "ranges": {"trainset": [1, 6], "valset": [7, 8], "evalset": [9, 10], "testset": [11, 12]}}
TIME_SIZES = {"trainset": 6, "valset": 3, "evalset": 3, "testset": 4}


def time_task():
    """binary_task with raw data at /raw/rows, a split rule that only writes the sizes (the data module fills in the
    rest) and the quality checks dset computes itself."""
    t = binary_task()
    t["data"]["sources"] = [{"path": "/raw/rows"}]
    t["data"]["split"] = {"sizes": dict(TIME_SIZES)}
    t["quality_checks"] = {"min_rows": 6, "min_positive_rows": 1}
    return t


def filled_time_task(**rule):
    """time_task with its rule filled in: TIME with the sizes and `rule`, defaults written out."""
    t = time_task()
    t["data"]["split"] = filled(dict(TIME, sizes=dict(TIME_SIZES), **rule))
    return t


def corpus_rows(task, steps=12, per_step=4):
    """All labelled rows of a corpus for TIME: per time step 1..steps, rows r<step>_<j> (j < per_step) with
    meta.extra.step = step and meta.state_id a<(step * per_step + j) % 16> (an address recurs every four steps, so the
    training steps cover all 16 and steps 7-8 share their 8 addresses with steps 11-12); j < 2 is the positive label
    (half of the rows), the rest the other one; split "all"; every input different."""
    labels = task["decision"]["labels"]
    pos = task["decision"].get("positive", labels[0])
    neg = [l for l in labels if l != pos][0]
    out = []
    for step in range(1, steps + 1):
        for j in range(per_step):
            rid = "r%d_%d" % (step, j)
            row = full_row(0, task)
            row["id"] = rid
            if task["input"]["format"] == "fields":
                row["state"] = {"fields": dict((f, "%s %s" % (f, rid)) for f in task["input"].get("fields", []))}
            else:
                row["state"] = {"messages": [{"role": "user", "content": "row %s" % rid}]}
            row["target"] = dict((l, 1.0 if l == (pos if j < 2 else neg) else 0.0) for l in labels)
            row["meta"].update(state_id="a%d" % ((step * per_step + j) % 16), cluster="k%s" % rid,
                               extra={"step": step})
            row["split"] = "all"
            out.append(row)
    return out


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
    """The OneJev row q<i>: hard label labels[i % k], group (meta.state_id) g<i % 2>, cluster c<i>."""
    labels = task["decision"]["labels"]
    if task["input"]["format"] == "fields":
        state = {"fields": {f: "v%d" % i for f in task["input"].get("fields", [])}}
    else:
        state = {"messages": [{"role": "user", "content": "message %d" % i}]}
    target = {l: 0.0 for l in labels}
    target[labels[i % len(labels)]] = 1.0
    row = {
        "id": "q%d" % i,
        "state": state,
        "question": question_of(task),
        "target": target,
        "meta": {
            "source": "synthetic",
            "state_id": "g%d" % (i % 2),
            "cluster": "c%d" % i,
            "label_source": "rule",
            "n_obs": 1,
        },
        "split": "train",
    }
    row.update(over)
    return row


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def input_row(i, task):
    """The inputs.jsonl row of q<i>: its full row without target, meta reduced to rsi.rows.INPUT_META."""
    return input_row_of(full_row(i, task))


def label_row(i, task):
    """The labels.jsonl row of q<i>: id, target and meta.label_source."""
    return label_row_of(full_row(i, task))


def label_row_of(r):
    """The labels.jsonl row of full row `r`: id, target and meta.label_source."""
    return {"id": r["id"], "target": r["target"], "meta": {"label_source": r["meta"]["label_source"]}}


DEFAULT_IDS = range(4)


class Opts(object):
    """What make_artifact's builders read besides the task.

    ids: rows written (q<i>); rows: full rows written instead (a data kind); wrong: ids whose predictions pick
    the wrong label;
    train_ids: the trainset rows the baselines come from; best_wrong: wrong ids of
    the best version's predictions (evaluation); test_ids: testset rows (milestone
    evaluation); val_ids, val_wrong: a delivery's predictions_val, which reproduce
    the candidate's (a candidate built with ids=val_ids, wrong=val_wrong); eval_ids: a
    baseline's predictions_eval rows; candidate, baselines: what a comparison compares
    (see _comparison_extra), and the baselines an evaluation scores (baseline id -> (name, wrong ids):
    baselines_eval, or baselines_test at a milestone); fields: the manifest overrides given to make_artifact.
    """

    def __init__(self, ids=None, wrong=(), train_ids=None, best_wrong=(), test_ids=None, fields=None,
                 val_ids=None, val_wrong=(), rows=None, eval_ids=None, candidate=None, baselines=None):
        self.given_ids = ids
        self.rows = rows
        self.ids = list(DEFAULT_IDS if ids is None else ids)
        self.wrong = set(wrong)
        self.train_ids = list(DEFAULT_IDS if train_ids is None else train_ids)
        self.best_wrong = set(best_wrong)
        self.test_ids = list(DEFAULT_IDS if test_ids is None else test_ids)
        self.val_ids = list(DEFAULT_IDS if val_ids is None else val_ids)
        self.val_wrong = set(val_wrong)
        self.eval_ids = list(DEFAULT_IDS if eval_ids is None else eval_ids)
        self.candidate = "candidate-0001" if candidate is None else candidate
        self.baselines = dict(baselines or {})
        self.fields = fields or {}


def _rows(task, o):
    """The full rows a data kind writes: the given rows, else q<i> for i in o.ids."""
    return list(o.rows) if o.rows is not None else [full_row(i, task) for i in o.ids]


def _full_files(d, task, o):
    rows = _rows(task, o)
    write_jsonl(os.path.join(d, "rows.jsonl"), rows)
    return {"rows": {"path": "rows.jsonl", "rows": len(rows)}}


# The preprocessing script of a fixture corpus: rsi never runs it.
PREPROCESS_STUB = "# fixture corpus: the rows were written by the test, not by this script\n"


def _corpus_files(d, task, o):
    files = _full_files(d, task, o)
    with open(os.path.join(d, "preprocess.py"), "w", encoding="utf-8") as f:
        f.write(PREPROCESS_STUB)
    files["script"] = {"path": "preprocess.py"}
    return files


def _trainset_files(d, task, o):
    files = _full_files(d, task, o)
    with open(os.path.join(d, "quality_report.json"), "w", encoding="utf-8") as f:
        json.dump({"passed": True}, f)
    files["quality_report"] = {"path": "quality_report.json"}
    return files


def _valset_files(d, task, o):
    files = _full_files(d, task, o)
    rows = _rows(task, o)
    write_jsonl(os.path.join(d, "inputs.jsonl"), [input_row_of(r) for r in rows])
    files["inputs"] = {"path": "inputs.jsonl", "rows": len(rows)}
    return files


def _split_files(d, task, o):
    rows = _rows(task, o)
    write_jsonl(os.path.join(d, "inputs.jsonl"), [input_row_of(r) for r in rows])
    write_jsonl(os.path.join(d, "labels.jsonl"), [label_row_of(r) for r in rows])
    return {"inputs": {"path": "inputs.jsonl", "rows": len(rows)},
            "labels": {"path": "labels.jsonl", "rows": len(rows)}}


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
        "state": full_row(i, task)["state"],
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
# config_request of fixture candidates: nothing asked for beyond the train command's defaults (in any search space).
CONFIG_REQUEST = {"base": None, "recipe": None, "overrides": {}, "seed": None, "max_steps": None}


def _candidate_files(d, task, o):
    _write_executable(os.path.join(d, "predict"), PREDICT_STUB)
    return {"predictions_val": _pred_file(d, task, "predictions_val", o.ids, o.wrong)}


def _delivery_files(d, task, o):
    """predictions_eval on o.ids, predictions_val on o.val_ids, and the package directory package/."""
    os.makedirs(os.path.join(d, "package"), exist_ok=True)
    _write_executable(os.path.join(d, "package", "predict"), PREDICT_STUB)
    return {"predictions_eval": _pred_file(d, task, "predictions_eval", o.ids, o.wrong),
            "predictions_val": _pred_file(d, task, "predictions_val", o.val_ids, o.val_wrong)}


def _baseline_files(d, task, o):
    """predictions_val on o.ids and predictions_eval on o.eval_ids (o.wrong wrong in both)."""
    return {"predictions_val": _pred_file(d, task, "predictions_val", o.ids, o.wrong),
            "predictions_eval": _pred_file(d, task, "predictions_eval", o.eval_ids, o.wrong)}


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
        "config_request": dict(CONFIG_REQUEST),
        "code_version": dict(CODE_VERSION),
        "model": {"location": "/models/m1", "sha256": None, "load": "load_model(path)", "predict": "predict"},
        "metrics_val": metrics_val,
        "baselines_val": baselines_val,
        "train_gate": _gate(task, metrics_val, baselines_val),
        "cost": {"compute_hours": 1.5},
    }


def _baseline_extra(task, o):
    """A CPU baseline lr on trainset-0001, valset-0001 and evalset-0001; metrics_val computed from its predictions_val."""
    labels = [full_row(i, task) for i in o.ids]
    metrics_val = scores(task, predictions(task, o.ids, o.wrong), labels)
    metrics_val.setdefault(task["metrics"]["primary"], 0.8)  # a primary rsi does not compute
    return {
        "producer": "compare",
        "name": "lr",
        "trainset_id": "trainset-0001",
        "valset_id": "valset-0001",
        "evalset_id": "evalset-0001",
        "testset_id": None,
        "spec": {"grid": {"C": [0.1, 1.0]}},
        "spec_sha256": "2" * 64,
        "config": {"C": 1.0},
        "features": None,
        "metrics_val": metrics_val,
        "cost": {"compute_hours": 0},
        "reused_from": None,
    }


def _comparison_extra(task, o):
    """A comparison on the valset rows o.ids of candidate o.candidate (the predictions of a candidate built with the same
    ids and wrong) against o.baselines: name -> baseline id, or (id, the ids its predictions_val get wrong). metrics_val,
    deltas_val and strongest are computed with rsi.metrics as registration recomputes them."""
    primary, higher = task["metrics"]["primary"], task["metrics"]["higher_is_better"]
    labels = [full_row(i, task) for i in o.ids]
    clusters = M.clusters_of([input_row(i, task) for i in o.ids])
    cand = predictions(task, o.ids, o.wrong)
    metrics_val = {"candidate": scores(task, cand, labels)}
    metrics_val["candidate"].setdefault(primary, 0.85)  # a primary rsi does not compute
    ids, deltas, strongest = {}, {}, None
    for name in sorted(o.baselines):
        b = o.baselines[name]
        ids[name], wrong = (b, ()) if isinstance(b, str) else b
        preds = predictions(task, o.ids, wrong)
        metrics_val[name] = scores(task, preds, labels)
        metrics_val[name].setdefault(primary, 0.8)
        if primary in M.KNOWN_METRICS:
            deltas[name] = dict(M.paired_delta(primary, cand, preds, labels, task, clusters, **M.ci_settings(task)),
                                metric=primary)
        v = metrics_val[name][primary]
        if strongest is None or (v > metrics_val[strongest][primary] if higher else v < metrics_val[strongest][primary]):
            strongest = name
    return {
        "producer": "compare",
        "candidate_id": o.candidate,
        "trainset_id": "trainset-0001",
        "valset_id": "valset-0001",
        "evalset_id": "evalset-0001",
        "baselines": ids,
        "metrics_val": metrics_val,
        "deltas_val": deltas,
        "strongest": strongest,
        "skipped": {},
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
    if o.baselines:  # baseline id -> (name, the ids its predictions get wrong): on the testset at a milestone
        field, ids = ("baselines_test", o.test_ids) if uses_testset else ("baselines_eval", o.ids)
        out[field] = baselines_scored(task, ids, o.wrong, o.baselines)
    return out


def baselines_scored(task, ids, wrong, baselines):
    """An evaluation's baselines_eval (or baselines_test) as the judge computes them with `rsi score` on the rows q<i>, i
    in ids: for each baseline id -> (name, its wrong ids), its metrics and the paired delta of the primary metric, the
    delivery's predictions (wrong `wrong`) minus the baseline's, clusters from the rows' inputs."""
    primary = task["metrics"]["primary"]
    labels = [label_row(i, task) for i in ids]
    clusters = M.clusters_of([input_row(i, task) for i in ids])
    preds = predictions(task, ids, wrong)
    out = {}
    for bid, (name, bwrong) in sorted(baselines.items()):
        base = predictions(task, ids, bwrong)
        delta = M.paired_delta(primary, preds, base, labels, task, clusters, **M.ci_settings(task))
        out[bid] = {"name": name, "metrics": scores(task, base, labels), "delta": dict(delta, metric=primary)}
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
    "corpus": (_corpus_files, lambda task, o: {
        "command": "python3 preprocess.py",
        "sources": source_paths(task),
        "label_method": "given",
    }),
    "pool": (_full_files, lambda task, o: {}),
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
    "baseline": (_baseline_files, _baseline_extra),
    "comparison": (_no_files_report, _comparison_extra),
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
                  val_ids=None, val_wrong=(), rows=None, eval_ids=None, candidate=None, baselines=None, **fields):
    """Write a minimal valid artifact directory and return its path.

    `ids` (ints) picks the decision rows q<i> written to row files (rows,
    inputs/labels, predictions, trace); the default is q0..q3 (q0..q2 for a
    feedback trace). A data kind (trainset, valset, evalset, testset, corpus,
    pool) writes the full rows `rows` instead when given (an evalset or testset
    their input rows and label rows). A corpus also has the script
    preprocess.py. Predictions of the ids in `wrong` pick the wrong label.
    Candidates, baselines, comparisons and evaluations report the numbers
    rsi.metrics computes from these rows (see Opts for train_ids, best_wrong,
    test_ids, val_ids, val_wrong, eval_ids, candidate and baselines), and the
    train gate, strongest baseline and recommendation those numbers give.
    Other keyword arguments override manifest fields.
    """
    os.makedirs(dir, exist_ok=True)
    build, extra = ARTIFACT_BUILDERS[kind]
    o = Opts(ids, wrong, train_ids, best_wrong, test_ids, fields, val_ids, val_wrong, rows, eval_ids, candidate,
             baselines)
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


def make_task_dir(dir, task, md=None):
    """Write a task artifact directory (task.json and TASK.md, `md` or a line naming the version) and return its
    path."""
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "task.json"), "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dir, "TASK.md"), "w", encoding="utf-8") as f:
        f.write("# Task: %s\n\nversion %d\n" % (task["task_id"], task["version"]) if md is None else md)
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


def labels_path(ws, kind):
    """The absolute path of the labels file of the current `kind` set of workspace `ws`, from its manifest's files."""
    aid = ws.current(kind)
    return os.path.join(os.path.abspath(ws.path(aid)), ws.manifest(aid)["files"]["labels"]["path"])


def FakeEngine(ws):
    """A real engine on workspace `ws` whose runner launches nothing (ManualRunner), without the pages."""
    from rsi.engine import Engine
    from rsi.runners import ManualRunner
    return Engine(ws.root, ManualRunner(), pages=False)
