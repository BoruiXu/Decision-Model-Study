"""The candidate's errors on the pool, their weakness types, their CLLM learnability and rule-based findings.

An error is a scored pool row whose hard label differs from the candidate's own `decision` (what its model.predict
says, its decision policy applied); its kind is "<true>→<decision>" (in a binary task FN is positive→negative and FP
negative→positive). No threshold is set from valset labels, and nothing of the valset, the evalset or the testset is
read. Confident: probs[decision] >= 0.9; near the boundary: probs[decision] - probs[true] < 0.1.

The findings are prompts for the agent, not conclusions; each has issue, evidence and try (dgen 0.4's table).
"""
import os

import numpy as np

from rsi import rows as R

from . import DGEN_VERSION, DgenError
from . import cllm
from . import weakness
from .inputs import SCORES_META, read_json, read_jsonl, sha256_file
from .space import Space

CONFIDENT, NEAR, N_OK = 0.9, 0.1, 2000
RECIPES = "hard, smote, borderline, llm_imitate, llm_pattern"


def kind_of(true, decision):
    return "%s→%s" % (true, decision)


def binary_names(task):
    """{kind: "FN" | "FP"} of a binary task (FN: the positive label decided negative), {} otherwise."""
    d = task["decision"]
    if d["type"] != "binary":
        return {}
    pos = d["positive"]
    neg = [label for label in d["labels"] if label != pos][0]
    return {kind_of(pos, neg): "FN", kind_of(neg, pos): "FP"}


def scores_source(inputs, scores_path):
    """The score_pool.json score-pool wrote beside `scores_path`, checked: the scores are the brief's best candidate's
    decisions (its id) and the file is the one score-pool wrote (its sha256)."""
    meta_path = os.path.join(os.path.dirname(os.path.abspath(scores_path)), SCORES_META)
    if not os.path.isfile(meta_path):
        raise DgenError("--scores %s: no %s beside it, so these are not scores dgen score-pool wrote; score the pool "
                        "with dgen score-pool" % (scores_path, SCORES_META))
    meta = read_json(meta_path, "--scores")
    want = inputs.candidate_manifest().get("id")
    if meta.get("candidate") != want:
        raise DgenError("--scores %s: scored by %s, but the brief's best candidate is %s: score the pool again with "
                        "dgen score-pool" % (scores_path, meta.get("candidate"), want))
    if meta.get("predictions_sha256") != sha256_file(scores_path):
        raise DgenError("--scores %s: not the file score-pool wrote (its sha256 differs from %s)"
                        % (scores_path, meta_path))
    return meta


def scored_rows(inputs, scores_path):
    """[(pool row, prediction row)] of the scored pool rows, in the order of the predictions."""
    preds = read_jsonl(scores_path, "--scores")
    errors = R.check_predictions(preds, inputs.task)
    if errors:
        raise DgenError("--scores %s: %d problems: %s" % (scores_path, len(errors), "; ".join(errors[:3])))
    pool = dict((r["id"], r) for r in inputs.pool_rows)
    missing = [p["id"] for p in preds if p["id"] not in pool]
    if missing:
        raise DgenError("--scores %s: %d rows are not pool rows, e.g. %s (score the brief's pool with dgen "
                        "score-pool)" % (scores_path, len(missing), missing[0]))
    if not preds:
        raise DgenError("--scores %s: no rows" % scores_path)
    return [(pool[p["id"]], p) for p in preds]


def failures(inputs, scored, seed=0):
    """(errors [(row, diag)], ok sample [(row, diag)], summary)."""
    errors, ok = [], {}
    for row, p in scored:
        true, dec = inputs.label(row), p["decision"]
        probs = p["probs"]
        d = {"id": row["id"], "true": true, "decision": dec, "kind": kind_of(true, dec),
             "p_decision": float(probs.get(dec, 0.0)), "p_true": float(probs.get(true, 0.0)), "probs": probs}
        if true == dec:
            ok.setdefault(true, []).append((row, d))
            continue
        d["confident"] = d["p_decision"] >= CONFIDENT
        d["near_boundary"] = d["p_decision"] - d["p_true"] < NEAR
        errors.append((row, d))
    rng = np.random.default_rng(seed)
    sample = []
    for label in inputs.labels:
        rows = ok.get(label, [])
        for i in sorted(rng.choice(len(rows), min(N_OK, len(rows)), replace=False)) if rows else []:
            sample.append(rows[i])
    names = binary_names(inputs.task)
    by_kind = {}
    for _, d in errors:
        e = by_kind.setdefault(d["kind"], {"n": 0, "confident": 0, "near_boundary": 0})
        e["n"] += 1
        e["confident"] += int(d["confident"])
        e["near_boundary"] += int(d["near_boundary"])
    for kind, e in by_kind.items():
        if kind in names:
            e["binary"] = names[kind]
    labels = [inputs.label(r) for r, _ in scored]
    summ = {"rows": len(scored), "label_counts": dict((l, labels.count(l)) for l in inputs.labels),
            "correct": len(scored) - len(errors), "error_count": len(errors),
            "errors": dict(sorted(by_kind.items())),
            "confident_errors": sum(e["confident"] for e in by_kind.values()),
            "errors_near_boundary": sum(e["near_boundary"] for e in by_kind.values())}
    return errors, sample, summ


def learnability(inputs, space, errors, types, th=cllm.TH):
    """learnability.json: the CLLM groups of the real failures, by kind and by weakness type (the reference fitted on
    the trainset and the pool: `space` is fitted on inputs.real(), so its X holds those rows in that order)."""
    real = inputs.real()  # the rows the space was fitted on, in the same order: its X
    ref = cllm.Reference(space.z(space.X), [inputs.label(r) for r in real], [r["id"] for r in real], inputs.labels)
    rows = [r for r, _ in errors]
    y = [d["true"] for _, d in errors]
    conf, al = ref.score(space.z(rows), y, [cllm.fold_of(r["id"]) for r in rows])
    g = cllm.groups(conf, al, th)
    kinds = np.array([d["kind"] for _, d in errors], dtype=object)
    of = dict((i, t["key"]) for t in types for i in t["member_ids"])
    tt = np.array([of.get(r["id"], "other") for r in rows], dtype=object)
    return {"reference": cllm.library(), "threshold": th, "folds": cllm.K,
            "by_kind": dict((k, cllm.shares(g, kinds == k)) for k in sorted(set(kinds))),
            "by_type": dict((k, cllm.shares(g, tt == k)) for k in sorted(set(tt))),
            "rows": dict((r["id"], {"conf": round(float(c), 4), "aleatoric": round(float(a), 4), "group": str(x)})
                         for r, c, a, x in zip(rows, conf, al, g))}


def _pct(x):
    return "%.0f%%" % (100 * x)


def findings(summ, W, learn):
    """dgen 0.4's findings table, by kind instead of FN/FP."""
    F = []
    err = summ["error_count"]
    if err < 100:
        F.append({"issue": "few_failures", "evidence": "%d errors in %d scored pool rows" % (err, summ["rows"]),
                  "try": "results will be noisy: prefer `hard` (the real failures) and a small n; or score more "
                         "of the pool (a larger score-pool --max-rows, or none)"})
    conf = summ["confident_errors"]
    if err and conf > 0.3 * err:
        F.append({"issue": "confident_errors", "evidence": "%d errors with probs[decision] >= %.1f, %s of all errors"
                                                          % (conf, CONFIDENT, _pct(conf / float(err))),
                  "try": "the score is wrong, not the boundary: targeted generation for these types (`hard`, "
                         "`llm_imitate`); check their learnability (CLLM) before generating"})
    near = summ["errors_near_boundary"]
    if err and near > 0.5 * err:
        F.append({"issue": "errors_near_threshold", "evidence": "%s of errors have probs[decision] - probs[true] < %.1f"
                                                               % (_pct(near / float(err)), NEAR),
                  "try": "rows near the decision boundary: `borderline`; also a decision-policy question (the "
                         "task's policy section), for the train module"})
    counts = dict((k, e["n"]) for k, e in summ["errors"].items())
    for kind, n in sorted(counts.items()):
        others = max([m for k, m in counts.items() if k != kind] or [0])
        if n > 3 * max(others, 1):
            name = summ["errors"][kind].get("binary")
            F.append({"issue": "%s_dominates" % kind,
                      "evidence": "%s%s %d vs at most %d of any other kind" % (kind, " (%s)" % name if name else "",
                                                                             n, others),
                      "try": "target.kinds: [\"%s\"] (rows labelled %s); watch the other kinds in the next "
                             "evaluation" % (kind, kind.split("→")[0])})
    hard_types = set()
    for key, d in sorted(((learn or {}).get("by_type") or {}).items()):
        if key != "other" and d["hard"] is not None and d["hard"] > 0.3 and d["n"] >= 20:
            hard_types.add(key)
            F.append({"issue": "unlearnable_type",
                      "evidence": "%s: %s of its %d real failures conflict with the labels of similar real rows "
                                  "(CLLM hard)" % (key, _pct(d["hard"]), d["n"]),
                      "try": "do not target %s with synthetic rows (they will be filtered or hurt): "
                             "target.skip_hard_share 0.3, or `hard` only; report possible label noise" % key})
    for t in W["types"]:
        if t["size"] < 30:
            F.append({"issue": "small_type", "evidence": "%s has %d failures" % (t["key"], t["size"]),
                      "try": "few seeds: generated rows will be near copies; leave it to `hard`, or a small "
                             "`llm_imitate` batch"})
        if t["signature"] and abs(t["signature"][0]["diff_sd"]) > 1.5 and t["key"] not in hard_types:
            s = t["signature"][0]
            F.append({"issue": "distinct_pattern",
                      "evidence": "%s: %s median %.4g vs %.4g in correct rows (%+.1f sd)"
                                  % (t["key"], s["feature"], s["median_type"], s["median_correct"], s["diff_sd"]),
                      "try": "a pattern the trainset lacks: `llm_pattern` rows of type %s, or `hard` with "
                             "target.types [\"%s\"]" % (t["key"], t["key"])})
    return F


def _table(head, rows):
    def cell(v):
        return ("%.4g" % v) if isinstance(v, float) else str(v).replace("|", "\\|")
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
           ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]


def markdown(task, candidate_id, summ, W, learn, F, errors):
    names = binary_names(task)
    L = ["# Diagnosis: %s on the pool" % candidate_id, "",
         "Errors are the scored pool rows whose hard label differs from the candidate's decision (no threshold from "
         "valset labels). dgen %s." % DGEN_VERSION, "", "## Overall", ""]
    L += _table(["rows", "correct", "errors", "confident", "near the boundary"],
                [[summ["rows"], summ["correct"], summ["error_count"], summ["confident_errors"],
                  summ["errors_near_boundary"]]])
    L += ["", "## Errors by kind", ""]
    L += _table(["kind", "binary", "n", "confident", "near the boundary"],
                [[k, names.get(k, "-"), e["n"], e["confident"], e["near_boundary"]]
                 for k, e in summ["errors"].items()]) if summ["errors"] else ["- no errors"]
    L += ["", "## Findings", ""]
    L += ["- **%s**: %s. Try: %s" % (f["issue"], f["evidence"], f["try"]) for f in F] or ["- none triggered"]
    L += ["", "## Weakness types", "", "Signature: the features whose median differs most from the correctly decided "
          "rows of the same true label (difference in sd; medians in the model view's units). hard: the CLLM hard "
          "share of the type's real failures.", ""]
    by_type = (learn or {}).get("by_type", {})
    rows = []
    for t in W["types"]:
        sig = "; ".join("%s %.4g vs %.4g (%+.1f)" % (s["feature"], s["median_type"], s["median_correct"],
                                                     s["diff_sd"]) for s in t["signature"][:3])
        h = by_type.get(t["key"], {}).get("hard")
        rows.append([t["key"], t["size"], t["share_of_kind"], t["mean_p_decision"], "-" if h is None else _pct(h),
                     sig])
    L += _table(["type", "size", "share of kind", "mean p(decision)", "hard", "signature"], rows) if rows \
        else ["- no errors"]
    if learn and learn["by_kind"]:
        L += ["", "## Learnability by kind (CLLM groups of the real failures; reference: %s)" % learn["reference"], ""]
        L += _table(["kind", "n", "easy", "ambiguous", "hard"],
                    [[k, v["n"], v["easy"], v["ambiguous"], v["hard"]] for k, v in learn["by_kind"].items()])
    for kind in summ["errors"]:
        top = sorted([d for _, d in errors if d["kind"] == kind], key=lambda d: -d["p_decision"])[:10]
        L += ["", "## Most confident errors %s%s" % (kind, " (%s)" % names[kind] if kind in names else ""), ""]
        L += _table(["id", "p(decision)", "p(true)"], [[d["id"], d["p_decision"], d["p_true"]] for d in top])
    return "\n".join(L) + "\n"


def run(inputs, scores_path, seed=0):
    """Everything diagnose writes, as {file name: object or text}."""
    source = scores_source(inputs, scores_path)
    scored = scored_rows(inputs, scores_path)
    errors, ok, summ = failures(inputs, scored, seed)
    real = inputs.real()
    space = Space(inputs.task).fit(real, [inputs.label(r) for r in real])
    W = weakness.build(space, errors, [r for r, _ in ok], inputs.label, seed) if errors else \
        {"columns": list(space.names), "scaler": space.to_json(), "types": []}
    learn = learnability(inputs, space, errors, W["types"]) if errors else None
    F = findings(summ, W, learn)
    type_of = dict((i, t["key"]) for t in W["types"] for i in t["member_ids"])
    hard_of = dict((t["key"], (learn or {}).get("by_type", {}).get(t["key"], {}).get("hard")) for t in W["types"])
    cand = inputs.candidate_manifest()
    diag = dict(summ, dgen_version=DGEN_VERSION, candidate=cand.get("id"), seed=seed, scores=source,
                types=[dict(((k, v) for k, v in t.items() if k not in ("member_ids", "centroid")),
                            cllm_hard=hard_of[t["key"]]) for t in W["types"]],
                findings=F)
    keys = ("id", "kind", "true", "decision", "p_decision", "p_true", "confident", "near_boundary", "probs")
    fail_lines = [dict(((k, d[k]) for k in keys), type=type_of.get(d["id"]),
                       cllm=(learn or {}).get("rows", {}).get(d["id"])) for _, d in errors]
    ok_lines = [dict((k, d[k]) for k in ("id", "true", "decision", "p_decision", "probs")) for _, d in ok]
    if learn:
        learn = dict((k, v) for k, v in learn.items() if k != "rows")
    return {"failures.jsonl": fail_lines, "ok_sample.jsonl": ok_lines, "weakness.json": W,
            "learnability.json": learn or {"reference": None, "by_kind": {}, "by_type": {}},
            "diagnosis.md": markdown(inputs.task, cand.get("id") or "the candidate", summ, W, learn, F, errors),
            "diagnosis.json": diag}
