"""Task definition and decision-row checks (task-agnostic, stdlib only).

Dataset rows are OneJev rows (docs/interfaces/README.md, Decision rows): `id`, `state`, `question`, `target`,
`meta` (with `state_id`) and `split`. Rows of workspaces from before that format carry `input` and `meta.group`
instead of `state` and `meta.state_id`: state_of and group_of read both, so old rows are still read.
"""
import hashlib
import json
import math
import os

from .schema import load_schema, validate

TOL = 1e-4  # predictions, and targets as `rsi score` reads them: probabilities sum to 1 within TOL
# targets at registration: OneJev's rule (onejev-harness harness/prepare.py validate_rows): sum to 1 within TARGET_TOL.
TARGET_TOL = 1e-6
LABEL_SOURCES = ("rule", "model_judge", "human", "outcome")
NOUL_LABELS = ("yes", "no")  # the labels of a OneJev noul question; criteria "true"/"false" describe them
# The only meta keys of an input row (inputs.jsonl, what deliver and model.predict get): any other meta key,
# e.g. a OneJev export's agreement rate f, or label_source (how a label was made can follow the label), may
# carry the label (ruling R8). label_source lives in the full rows and the label rows.
INPUT_META = ("state_id", "cluster", "source", "n_obs")
# The only top-level keys of an input row (`input` only in a row of an old workspace, which has no `state`): a full
# row may carry more keys, and any of them may carry the label. A label row has exactly LABEL_KEYS.
INPUT_KEYS = ("id", "state", "input", "question", "meta", "split")
STATE_INPUT_KEYS = tuple(k for k in INPUT_KEYS if k != "input")  # those of an input row with `state` (every new row)
LABEL_KEYS = ("id", "target", "meta")


def _input_keys(row):
    """The top-level keys an input row may have: INPUT_KEYS, the old `input` only in a row without `state`."""
    return INPUT_KEYS if "state" not in row else STATE_INPUT_KEYS


def input_row_of(row):
    """The inputs.jsonl row of a full row: its INPUT_KEYS (so no target, nor any other key), its meta reduced to
    INPUT_META."""
    out = dict((k, v) for k, v in row.items() if k in _input_keys(row))
    if isinstance(row.get("meta"), dict):
        out["meta"] = dict((k, v) for k, v in row["meta"].items() if k in INPUT_META)
    return out


def state_of(row):
    """The state (model input) of a row: its `state`, or its `input` in a row of an old workspace."""
    return row["state"] if "state" in row else row.get("input")


GENERATED_PREFIX = "dgen:"  # the meta.source prefix of a row the data module generated (modules/data/dgen)


def _extra(row):
    meta = row.get("meta") if isinstance(row, dict) else None
    extra = meta.get("extra") if isinstance(meta, dict) else None
    return extra if isinstance(extra, dict) else None


def is_variant(row):
    """Whether a row is a prompt variant (meta.extra.variant; rsi.variants): a rewrite of its seed's prompt, held only
    by a trainset."""
    extra = _extra(row)
    return extra is not None and "variant" in extra


def is_generated(row):
    """Whether the data module generated the row (meta.source starting with "dgen:")."""
    meta = row.get("meta") if isinstance(row, dict) else None
    source = meta.get("source") if isinstance(meta, dict) else None
    return isinstance(source, str) and source.startswith(GENERATED_PREFIX)


def group_of(row):
    """The group of a row: its meta.state_id, or meta.group in a row of an old workspace (None when the row
    has neither as a string)."""
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return None
    for key in ("state_id", "group"):
        if isinstance(meta.get(key), str):
            return meta[key]
    return None


def question_of(task):
    """The OneJev question a task's decision asks; every dataset row carries exactly this question.

    A binary decision on the labels yes and no is a noul question (criteria "true" / "false" are the meanings
    of yes / no); a score decision is a score question (criteria: the meanings of "0" to "k-1", in order); any
    other decision (choice, or binary on other labels) is a choice question (criteria: label -> meaning).
    """
    d = task["decision"]
    meanings = d["label_meanings"]
    if d["type"] == "binary" and set(d["labels"]) == set(NOUL_LABELS):
        qtype, criteria = "noul", {"true": meanings["yes"], "false": meanings["no"]}
    elif d["type"] == "score":
        qtype, criteria = "score", [meanings[label] for label in d["labels"]]
    else:
        qtype, criteria = "choice", dict((label, meanings[label]) for label in d["labels"])
    return {"type": qtype, "instructions": d["instructions"], "criteria": criteria}


def iter_jsonl(path):
    """The JSON objects of a JSONL file one at a time (blank lines skipped), so a large file is never held whole;
    ValueError names the line of invalid JSON or of a value that is not an object."""
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
            yield obj


def read_jsonl(path):
    return list(iter_jsonl(path))


def check_task(task):
    from .metrics import BINARY_ONLY, HIGHER_IS_BETTER  # rsi.metrics imports this module (group_of)
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
    if inp["format"] != "fields" and "fields" in inp:
        errors.append("$.input.fields: only allowed when format is fields")
    m = task["metrics"]
    known = set([m["primary"]] + list(m["others"]))
    for i, g in enumerate(task["promotion_rule"]["guards"]):
        if g["metric"] not in known:
            errors.append("$.promotion_rule.guards[%d].metric: %r is not primary or in others"
                          % (i, g["metric"]))
    # auc, auprc and brier are defined for binary decisions only (they score the positive label).
    if d["type"] != "binary":
        named = [("$.metrics.primary", m["primary"])]
        named += [("$.metrics.others[%d]" % i, n) for i, n in enumerate(m["others"])]
        named += [("$.promotion_rule.guards[%d].metric" % i, g["metric"])
                  for i, g in enumerate(task["promotion_rule"]["guards"])]
        if (task.get("policy") or {}).get("objective") is not None:
            named.append(("$.policy.objective", task["policy"]["objective"]))
        for where, name in named:
            if name in BINARY_ONLY:
                errors.append("%s: %s is defined for binary tasks only (this task's decision type is %s)"
                              % (where, name, d["type"]))
    errors += _cost_errors(task) + _policy_errors(task)
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
    # The train module searches only the search space the user approved (rsi.space).
    if "search_space" in task:
        from .space import check_space  # rsi.space imports rsi.rules only; kept local like the metrics import
        errors += check_space(task["search_space"])
    seeds = task.get("search", {}).get("seeds")
    if seeds is not None and len(set(seeds)) != len(seeds):
        errors.append("$.search.seeds: the seeds must be distinct (got %s)" % seeds)
    # ③ Compare trains each named baseline once: a name listed twice would be two baselines of one name.
    names = task.get("compare", {}).get("baselines", [])
    if len(set(names)) != len(names):
        errors.append("$.compare.baselines: each baseline is named once (got %s)" % ", ".join(names))
    # data.sources paths, data.rows and the split rule data.split (docs/interfaces/task.md, data).
    from .datasplit import check_data  # kept local like the space import
    errors += check_data(task)
    # features.adapter: a file that exists (docs/interfaces/task.md, features).
    from .rules import adapter_path  # kept local like the space import
    path = adapter_path(task)
    if path is not None and not os.path.isfile(path):
        errors.append("$.features.adapter: %s is not a file (a relative path is taken from the repository root)"
                      % path)
    return errors


def _cost_errors(task):
    """metrics.costs: true label -> decision -> cost, both levels keyed by exactly the task's labels; expected_cost,
    wherever the task names it, needs it."""
    m, labels = task["metrics"], task["decision"]["labels"]
    costs = m.get("costs")
    errors = []
    if costs is not None and (set(costs) != set(labels) or any(set(row) != set(labels) for row in costs.values())):
        errors.append("$.metrics.costs: the keys must be the task's labels at both levels (true label -> decision -> "
                      "cost; labels %s)" % ", ".join(labels))
    if not costs:
        named = [("$.metrics.primary", m["primary"])]
        named += [("$.metrics.others[%d]" % i, n) for i, n in enumerate(m["others"])]
        named += [("$.promotion_rule.guards[%d].metric" % i, g["metric"])
                  for i, g in enumerate(task["promotion_rule"]["guards"])]
        named.append(("$.policy.objective", (task.get("policy") or {}).get("objective")))
        errors += ["%s: expected_cost needs metrics.costs (true label -> decision -> cost)" % where
                   for where, name in named if name == "expected_cost"]
    return errors


def _policy_errors(task):
    """The task's policy section (docs/interfaces/task.md, policy): an objective rsi computes, calibrations and
    threshold rules a non-binary task can use, each rule's parameters, and its split."""
    from .metrics import KNOWN_METRICS  # rsi.metrics imports this module (group_of)
    from .rules import BINARY_CALIBRATIONS, THRESHOLD_RULES
    p = task.get("policy")
    if p is None:
        return []
    errors = []
    kind = task["decision"]["type"]
    objective = p.get("objective")
    if objective is not None and objective not in KNOWN_METRICS:
        errors.append("$.policy.objective: %r is not a metric rsi computes (%s)"
                      % (objective, ", ".join(KNOWN_METRICS)))
    if kind != "binary":
        errors += ["$.policy.calibrations[%d]: %s is for binary tasks only (this task's decision type is %s; use none "
                   "or temperature)" % (i, c, kind) for i, c in enumerate(p.get("calibrations", []))
                   if c in BINARY_CALIBRATIONS]
    needs = {"precision_at": "target", "recall_at": "target", "top_frac": "frac", "fixed": "value"}
    for i, r in enumerate(p.get("rules", [])):
        where, name = "$.policy.rules[%d]" % i, r["rule"]
        if name in THRESHOLD_RULES and kind != "binary":
            errors.append("%s: %s is a threshold rule, for binary tasks only (this task's decision type is %s; use "
                          "argmax)" % (where, name, kind))
        if name in needs and needs[name] not in r:
            errors.append("%s: %s needs %s" % (where, name, needs[name]))
        if name == "cost" and not ("cost_fp" in r and "cost_fn" in r) and not task["metrics"].get("costs"):
            errors.append("%s: cost needs cost_fp and cost_fn, or metrics.costs" % where)
    split = p.get("split") or {}
    f = split.get("select_fraction")
    if f is not None and not 0 < f < 1:
        errors.append("$.policy.split.select_fraction: must be strictly between 0 and 1 (got %s)" % f)
    data_by = ((task.get("data") or {}).get("split") or {}).get("by")
    if split.get("by") == "time" and "key" not in split and data_by != "time":
        errors.append("$.policy.split.key: a time split needs a key (the task's data.split is not by time)")
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


def check_target(target, task, tol=TARGET_TOL):
    """OneJev's rule for a target: a non-empty object whose keys are labels of the row's question (the task's
    labels; a label left out has probability 0), each value a number in [0, 1], summing to 1 within `tol`
    (TARGET_TOL, OneJev's; `rsi score` reads targets within TOL)."""
    labels = task["decision"]["labels"]
    if not isinstance(target, dict) or not target:
        return ["target: expected a non-empty object of label -> probability"]
    errors = ["target: %r is not a label of the question (labels: %s)" % (k, ", ".join(labels))
              for k in target if k not in labels]
    if errors:
        return errors
    for k, v in target.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            errors.append("target.%s: probability must be a number in [0, 1]" % k)
    total = math.fsum(target.values()) if not errors else None
    if total is not None and abs(total - 1) > tol:
        errors.append("target: probabilities must sum to 1 within %g (got %r)" % (tol, total))
    return errors


def check_question(question, task, variant=None):
    """A row's question must be question_of(task), or with `variant` (a prompt-variant spec; a variant row of a
    trainset) the spec's question (rsi.variants.question): errors name the parts that differ."""
    if variant is None:
        want = question_of(task)
        instructions, meanings = "the task's decision.instructions", "the task's label_meanings"
    else:
        from .variants import question as variant_question  # rsi.variants imports this module
        want = variant_question(variant, task)
        instructions = "the instructions of prompt variant %s (the task's when it has none)" % variant.get("id")
        meanings = "the criteria of prompt variant %s (the task's label_meanings when it has none)" % variant.get("id")
    if question == want:
        return []
    if not isinstance(question, dict):
        return ["question: expected the task's question, an object {type, instructions, criteria} "
                "(docs/interfaces/README.md, Decision rows)"]
    errors = []
    if question.get("type") != want["type"]:
        errors.append("question.type: expected %r for this task's decision (got %r)" % (want["type"],
                                                                                       question.get("type")))
    if question.get("instructions") != want["instructions"]:
        errors.append("question.instructions: must equal %s" % instructions)
    if question.get("criteria") != want["criteria"]:
        errors.append("question.criteria: expected %s (%s)" % (json.dumps(want["criteria"], ensure_ascii=False),
                                                               meanings))
    extra = sorted(set(question) - set(want))
    if extra:
        errors.append("question: unexpected key(s) %s" % ", ".join(extra))
    return errors


def check_state(state, task, name="state"):
    """A row's state in the task's input format: {"messages": [{"role", "content"}, ...]} (non-empty) for
    format messages; {"fields": {...}} covering the task's input.fields for format fields; any non-empty
    object for format object (handed to the model as it is, as OneJev takes any state object)."""
    fmt = task["input"]["format"]
    if fmt == "object":
        return [] if isinstance(state, dict) and state else ["%s: expected a non-empty object" % name]
    if not isinstance(state, dict):
        return ["%s: expected object" % name]
    if fmt == "messages":
        msgs = state.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return ["%s.messages: must be a non-empty list" % name]
        errors = []
        for i, m in enumerate(msgs):
            if (not isinstance(m, dict) or not isinstance(m.get("role"), str)
                    or not isinstance(m.get("content"), str)):
                errors.append("%s.messages[%d]: needs string role and content" % (name, i))
        return errors
    f = state.get("fields")
    if not isinstance(f, dict):
        return ["%s.fields: expected object" % name]
    missing = [k for k in task["input"].get("fields", []) if k not in f]
    return ["%s.fields: missing %s" % (name, ", ".join(missing))] if missing else []


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


def _onejev_errors(r, task, full, variants=None):
    """A OneJev row apart from its target: state, question, meta and split. A full row's meta also has
    label_source and may have more keys, as the row may; an input row has only the INPUT_KEYS (the old `input`
    only without `state`) and its meta exactly the INPUT_META keys. A prompt-variant row (is_variant) is refused unless
    `variants` is given (a trainset's rows: its question is then the spec's, which rsi.variants.check_variant_rows
    checks)."""
    if "state" not in r and "input" in r:
        errs = ["state: missing (rows are OneJev rows now: write the input as `state`)"]
    else:
        errs = check_state(r.get("state"), task)
    if not is_variant(r):
        errs += check_question(r.get("question"), task)
    elif variants is None:
        errs += ["meta.extra.variant: a prompt-variant row belongs only in a trainset, whose manifest lists its spec "
                 "in prompt_variants (docs/interfaces/datasets.md, Prompt-variant rows)"]
        errs += check_question(r.get("question"), task)
    errs += _meta_errors(r, ["source", "state_id", "cluster"], full, True)
    if not full and isinstance(r.get("meta"), dict):
        errs += ["meta.%s: not allowed in an input row, whose meta holds only %s since another meta field "
                 "may carry the label (drop it when writing inputs.jsonl)" % (k, ", ".join(INPUT_META))
                 for k in sorted(r["meta"]) if k not in INPUT_META]
    if not isinstance(r.get("split"), str) or not r["split"]:
        errs.append("split: expected non-empty string")
    if not full:
        errs += ["%s: not allowed in an input row (inputs.jsonl is what deliver gets), whose top-level keys are only "
                 "%s since any other key may carry the label (input_row_of drops it)" % (k, ", ".join(STATE_INPUT_KEYS))
                 for k in sorted(r) if k not in _input_keys(r)]
    return errs


def check_full_rows(rows, task, variants=None):
    """Full rows (a trainset or valset `rows.jsonl`): OneJev rows with a target. A row with meta.extra.variant is
    refused unless `variants` (a trainset's prompt_variants, [] when it lists none) is given; its question is then left
    to rsi.variants.check_variant_rows, which checks it against its spec and its seed."""
    return _check_rows(rows, lambda r: _onejev_errors(r, task, True, variants) + check_target(r.get("target"), task))


def check_input_rows(rows, task):
    """Input rows (an `inputs.jsonl`): full rows with only the INPUT_KEYS (so no target), meta only INPUT_META
    (input_row_of)."""
    return _check_rows(rows, lambda r: _onejev_errors(r, task, False))


def check_label_rows(rows, task):
    """Label rows (an evalset or testset `labels.jsonl`): only the LABEL_KEYS id, target and meta
    {"label_source"}, its only key."""
    def one(r):
        errs = ["%s: not allowed in a label row, whose top-level keys are only %s" % (k, ", ".join(LABEL_KEYS))
                for k in sorted(r) if k not in LABEL_KEYS]
        errs += check_target(r.get("target"), task)
        meta = r.get("meta")
        if not isinstance(meta, dict):
            return errs + ['meta: expected object {"label_source": ...}']
        if meta.get("label_source") not in LABEL_SOURCES:
            errs.append("meta.label_source: must be one of %s" % ", ".join(LABEL_SOURCES))
        return errs + ["meta.%s: not allowed in a label row, whose meta holds only label_source" % k
                       for k in sorted(meta) if k != "label_source"]
    return _check_rows(rows, one)


def check_target_rows(rows, task):
    """Rows with a unique id and a valid target: full rows or labels.jsonl rows, as `rsi score` reads them: the
    target sums to 1 within TOL, like a prediction, so a soft label written rounded (0.33333) is scored."""
    return _check_rows(rows, lambda r: check_target(r.get("target"), task, TOL))


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
        key = "input" if "state" not in r and "input" in r else "state"  # `input` in a trace of an old workspace
        errs = check_state(r.get(key), task, key)  # a call's input: the state of an input row
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
    """Fingerprint of a row's input (its state, state_of): case and whitespace do not count."""
    s = json.dumps(_norm(inp), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _coarse(v):
    if isinstance(v, dict):
        return dict((k, _coarse(x)) for k, x in v.items())
    if isinstance(v, list):
        return [_coarse(x) for x in v]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        try:
            return float("%.4g" % v) + 0.0  # + 0.0: -0.0 is 0.0
        except OverflowError:  # an int too large for a float
            return v
    return v


def coarse_fingerprint(state):
    """input_fingerprint of the state with every number rounded to 4 significant digits (and written as a float, so 5
    and 5.0 are one): the held-out check of generated rows (Workspace._generated_heldout_matches)."""
    return input_fingerprint(_coarse(state))


def overlap(rows_a, rows_b):
    """{"ids", "inputs", "clusters"} shared by two row lists (inputs by input_fingerprint of state_of)."""
    def ids(rs):
        return set(r["id"] for r in rs if "id" in r)

    def fps(rs):
        return set(input_fingerprint(state_of(r)) for r in rs if state_of(r) is not None)

    def clusters(rs):
        return set(r["meta"]["cluster"] for r in rs
                   if isinstance(r.get("meta"), dict) and "cluster" in r["meta"])

    return {
        "ids": len(ids(rows_a) & ids(rows_b)),
        "inputs": len(fps(rows_a) & fps(rows_b)),
        "clusters": len(clusters(rows_a) & clusters(rows_b)),
    }


def shared_state_ids(rows_a, rows_b):
    """The number of state ids (group_of: meta.state_id, meta.group in a row of an old workspace) two row lists share;
    a row without one counts for none."""
    def state_ids(rs):
        return set(g for g in (group_of(r) for r in rs if isinstance(r, dict)) if g is not None)
    return len(state_ids(rows_a) & state_ids(rows_b))
