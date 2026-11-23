"""Task definition and decision-row checks (task-agnostic, stdlib only)."""
import hashlib
import json

from .metrics import HIGHER_IS_BETTER
from .schema import load_schema, validate

TOL = 1e-4
LABEL_SOURCES = ("rule", "model_judge", "human", "outcome")


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise ValueError("%s: line %d: invalid JSON (%s)" % (path, n, e))
            if not isinstance(obj, dict):
                raise ValueError("%s: line %d: expected a JSON object" % (path, n))
            out.append(obj)
    return out


def check_task(task):
    errors = validate(task, load_schema("task"))
    if errors:
        return errors
    d = task["decision"]
    labels = d["labels"]
    if len(set(labels)) != len(labels):
        errors.append("$.decision.labels: duplicate labels")
    if set(d["label_meanings"]) != set(labels):
        errors.append("$.decision.label_meanings: keys must equal labels")
    if d["type"] == "score" and labels != [str(i) for i in range(len(labels))]:
        errors.append('$.decision.labels: score labels must be "0" to "k-1" in order')
    if d["type"] == "binary":
        if len(labels) != 2:
            errors.append("$.decision.labels: binary needs exactly 2 labels")
        if "positive" not in d:
            errors.append("$.decision.positive: required for binary")
        elif d["positive"] not in labels:
            errors.append("$.decision.positive: must be one of labels")
    elif "positive" in d:
        errors.append("$.decision.positive: only allowed for binary")
    inp = task["input"]
    if inp["format"] == "fields" and not inp.get("fields"):
        errors.append("$.input.fields: required and non-empty when format is fields")
    if inp["format"] == "messages" and "fields" in inp:
        errors.append("$.input.fields: only allowed when format is fields")
    m = task["metrics"]
    known = set([m["primary"]] + list(m["others"]))
    for i, g in enumerate(task["promotion_rule"]["guards"]):
        if g["metric"] not in known:
            errors.append("$.promotion_rule.guards[%d].metric: %r is not primary or in others"
                          % (i, g["metric"]))
    # auc is defined for binary decisions only (rsi.metrics.auc scores the positive label).
    if d["type"] != "binary":
        named = [("$.metrics.primary", m["primary"])]
        named += [("$.metrics.others[%d]" % i, n) for i, n in enumerate(m["others"])]
        named += [("$.promotion_rule.guards[%d].metric" % i, g["metric"])
                  for i, g in enumerate(task["promotion_rule"]["guards"])]
        for where, name in named:
            if name == "auc":
                errors.append("%s: auc is defined for binary tasks only (this task's decision type is %s)"
                              % (where, d["type"]))
    tol = task.get("delivery", {}).get("reproduction_tolerance")
    if tol is not None and not tol > 0:
        errors.append("$.delivery.reproduction_tolerance: must be greater than 0 (got %s)" % tol)
    rate = task.get("delivery", {}).get("max_decision_mismatch_rate")
    if rate is not None and not 0 <= rate < 1:
        errors.append("$.delivery.max_decision_mismatch_rate: must be at least 0 and below 1 (got %s)" % rate)
    level = m.get("ci", {}).get("level")
    if level is not None and not 0 < level < 1:
        errors.append("$.metrics.ci.level: must be strictly between 0 and 1 (got %s)" % level)
    # A metric rsi computes has a fixed direction (rsi.metrics.HIGHER_IS_BETTER).
    directions = [("$.metrics.higher_is_better", m["primary"], m["higher_is_better"])]
    directions += [("$.promotion_rule.guards[%d].higher_is_better" % i, g["metric"], g["higher_is_better"])
                   for i, g in enumerate(task["promotion_rule"]["guards"])]
    for where, name, higher in directions:
        if name in HIGHER_IS_BETTER and higher != HIGHER_IS_BETTER[name]:
            errors.append("%s: %s is %s-is-better as rsi computes it, so this must be %s"
                          % (where, name, "higher" if HIGHER_IS_BETTER[name] else "lower",
                             "true" if HIGHER_IS_BETTER[name] else "false"))
    return errors


def _prob_map(obj, labels, path):
    if not isinstance(obj, dict):
        return ["%s: expected object" % path]
    errors = []
    for k in obj:
        if k not in labels:
            errors.append("%s: unknown label %r" % (path, k))
    for l in labels:
        if l not in obj:
            errors.append("%s: missing label %r" % (path, l))
    if errors:
        return errors
    for k, v in obj.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            errors.append("%s.%s: probability must be a number in [0, 1]" % (path, k))
    if not errors and abs(sum(obj.values()) - 1) > TOL:
        errors.append("%s: probabilities must sum to 1 (got %s)" % (path, sum(obj.values())))
    return errors


def check_target(target, task):
    return _prob_map(target, task["decision"]["labels"], "target")


def check_input(inp, task):
    if not isinstance(inp, dict):
        return ["input: expected object"]
    fmt = task["input"]["format"]
    if fmt == "messages":
        msgs = inp.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return ["input.messages: must be a non-empty list"]
        errors = []
        for i, m in enumerate(msgs):
            if (not isinstance(m, dict) or not isinstance(m.get("role"), str)
                    or not isinstance(m.get("content"), str)):
                errors.append("input.messages[%d]: needs string role and content" % i)
        return errors
    f = inp.get("fields")
    if not isinstance(f, dict):
        return ["input.fields: expected object"]
    missing = [k for k in task["input"].get("fields", []) if k not in f]
    return ["input.fields: missing %s" % ", ".join(missing)] if missing else []


def _row_prefix(errors, row, i):
    rid = row.get("id") if isinstance(row, dict) else None
    return ["row %d (id=%s): %s" % (i, rid, e) for e in errors]


def _meta_errors(row, str_keys, label_source=False, n_obs=False):
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return ["meta: expected object"]
    errors = []
    for k in str_keys:
        if not isinstance(meta.get(k), str):
            errors.append("meta.%s: expected string" % k)
    if label_source and meta.get("label_source") not in LABEL_SOURCES:
        errors.append("meta.label_source: must be one of %s" % ", ".join(LABEL_SOURCES))
    if n_obs:
        v = meta.get("n_obs")
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            errors.append("meta.n_obs: expected integer >= 1")
    if "extra" in meta and not isinstance(meta["extra"], dict):
        errors.append("meta.extra: expected object")
    return errors


def _check_rows(rows, check_one):
    errors = []
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append("row %d: expected object" % i)
            continue
        errs = []
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            errs.append("id: expected non-empty string")
        elif rid in seen:
            errs.append("duplicate id")
        else:
            seen.add(rid)
        errs += check_one(row)
        errors += _row_prefix(errs, row, i)
    return errors


def check_full_rows(rows, task):
    def one(r):
        return (check_input(r.get("input"), task) + check_target(r.get("target"), task)
                + _meta_errors(r, ["source", "group", "cluster"], True, True))
    return _check_rows(rows, one)


def check_input_rows(rows, task):
    """Input rows (an `inputs.jsonl`): id, input, meta.group, meta.cluster, and no labels."""
    def one(r):
        errs = check_input(r.get("input"), task) + _meta_errors(r, ["group", "cluster"])
        if "target" in r:
            errs.append("target: input rows carry no labels (inputs.jsonl is what deliver gets)")
        return errs
    return _check_rows(rows, one)


def check_label_rows(rows, task):
    def one(r):
        return check_target(r.get("target"), task) + _meta_errors(r, ["source"], True, True)
    return _check_rows(rows, one)


def check_target_rows(rows, task):
    """Rows with a unique id and a valid target: full rows or labels.jsonl rows (as `rsi score` reads them)."""
    return _check_rows(rows, lambda r: check_target(r.get("target"), task))


def check_predictions(rows, task, expected_ids=None):
    """expected_ids None: check row validity only, not id coverage."""
    labels = task["decision"]["labels"]

    def one(r):
        errs = _prob_map(r.get("probs"), labels, "probs")
        if r.get("decision") not in labels:
            errs.append("decision: must be one of labels")
        u = r.get("uncertainty", "absent")
        if u == "absent" or (u is not None and (isinstance(u, bool) or not isinstance(u, (int, float)))):
            errs.append("uncertainty: expected number or null")
        return errs

    errors = _check_rows(rows, one)
    if expected_ids is None:
        return errors
    ids = set(r["id"] for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str))
    miss = sorted(set(expected_ids) - ids)
    extra = sorted(ids - set(expected_ids))
    if miss:
        errors.append("missing predictions for %d ids, e.g. %s" % (len(miss), miss[:3]))
    if extra:
        errors.append("unexpected predictions for %d ids, e.g. %s" % (len(extra), extra[:3]))
    return errors


def check_feedback_rows(rows, task):
    labels = task["decision"]["labels"]

    def num(v):
        return not isinstance(v, bool) and isinstance(v, (int, float))

    def one(r):
        errs = check_input(r.get("input"), task)
        out = r.get("output")
        if not isinstance(out, dict):
            errs.append("output: expected object")
        else:
            errs += _prob_map(out.get("probs"), labels, "output.probs")
            if out.get("decision") not in labels:
                errs.append("output.decision: must be one of labels")
        if not isinstance(r.get("model"), str) or not r.get("model"):
            errs.append("model: expected non-empty string")
        if not num(r.get("latency_ms")):
            errs.append("latency_ms: expected number")
        if not isinstance(r.get("timestamp"), str):
            errs.append("timestamp: expected string")
        if "outcome" not in r or (r["outcome"] is not None and r["outcome"] not in labels):
            errs.append("outcome: must be a label or null")
        return errs
    return _check_rows(rows, one)


def _norm(v):
    if isinstance(v, str):
        return " ".join(v.split()).lower()
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return dict((k, _norm(x)) for k, x in v.items())
    return v


def input_fingerprint(inp):
    s = json.dumps(_norm(inp), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def overlap(rows_a, rows_b):
    def ids(rs):
        return set(r["id"] for r in rs if "id" in r)

    def fps(rs):
        return set(input_fingerprint(r["input"]) for r in rs if "input" in r)

    def clusters(rs):
        return set(r["meta"]["cluster"] for r in rs
                   if isinstance(r.get("meta"), dict) and "cluster" in r["meta"])

    return {
        "ids": len(ids(rows_a) & ids(rows_b)),
        "inputs": len(fps(rows_a) & fps(rows_b)),
        "clusters": len(clusters(rows_a) & clusters(rows_b)),
    }
