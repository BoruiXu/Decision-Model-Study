"""Mock rsi-judge (see mocklib for the call, the task and the knobs).

purpose judge            the evaluation of the round's delivery: every number from `rsi score` (candidate,
                         baselines from the candidate's trainset rows, the best version's side and the delta with
                         its interval: the best delivery on the current evalset after a refresh), the recommendation
                         by the task's promotion rule, failures by group (meta.state_id), a report; handed back
                         alone. Continued in the same session once the round is decided (the feedback names the
                         evaluation and the decision): the plan and a record. A user's revise answer: a new plan in a
                         new subdirectory.
purpose plan             the plan of a round without a judge session to continue (a closed round: no evaluation)
purpose milestone_judge  the milestone evaluation of the best version: uses_testset, keep, metrics_test from the
                         testset labels (the evalset numbers compare the delivery with itself)

With ③ Compare the evaluation also reports each baseline the brief gives (mocklib.baselines_scored): baselines_eval on
the evalset, and at the milestone baselines_test on the testset.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mocklib as M  # noqa: E402

MODULES = ("data", "train", "deliver")
EPS = 1e-9


def promotion(task, ev, has_best, gate):
    """The promotion rule of task.md on the evaluation's numbers: (decision, reason)."""
    if not has_best:
        if gate:
            return "promote", "no best version yet and the candidate passed its train gate"
        return "keep", "no best version yet and the candidate did not pass its train gate"
    rule, primary = task["promotion_rule"], task["metrics"]["primary"]
    s = 1 if task["metrics"]["higher_is_better"] else -1
    d = ev["deltas"][primary]
    ok, bad = [], []
    imp = s * d["delta"]
    (ok if imp > rule["min_delta"] + EPS else bad).append("improvement %.6g vs min_delta %s" % (imp, rule["min_delta"]))
    if rule["ci_lower_min"] is not None:
        lo = d["ci_low"] if s == 1 else -d["ci_high"]
        (ok if lo >= rule["ci_lower_min"] - EPS else bad).append(
            "interval lower bound %.6g vs ci_lower_min %s" % (lo, rule["ci_lower_min"]))
    for g in rule["guards"]:
        cand, best = ev["metrics"]["candidate"][g["metric"]], ev["metrics"]["best"][g["metric"]]
        reg = best - cand if g["higher_is_better"] else cand - best
        (ok if reg <= g["max_regression"] + EPS else bad).append(
            "guard %s regression %.6g vs max_regression %s" % (g["metric"], reg, g["max_regression"]))
    if bad:
        return "keep", "promotion rule: " + "; ".join(bad)
    return "promote", "promotion rule: " + "; ".join(ok)


def predictions(delivery_dir, logical):
    m = M.read_json(os.path.join(delivery_dir, "manifest.json"))
    return os.path.join(delivery_dir, m["files"][logical]["path"])


def slices(c, preds_path):
    labels = dict((r["id"], M.hard(r["target"])) for r in M.read_jsonl(c.inputs["evalset_labels"]))
    preds = dict((r["id"], r["decision"]) for r in M.read_jsonl(preds_path))
    groups, fn, fp = {}, 0, 0
    for r in M.read_jsonl(c.inputs["evalset_inputs"]):
        g = groups.setdefault(r["meta"]["state_id"], [0, 0])
        right = preds[r["id"]] == labels[r["id"]]
        g[0] += right
        g[1] += 1
        if not right:
            fn += labels[r["id"]] == "yes"
            fp += labels[r["id"]] == "no"
    return groups, fn, fp


def entry_of(c):
    return c.knob("entry") or "train"


def write_evaluation(c, milestone):
    task, primary = c.task, c.task["metrics"]["primary"]
    dlv, cand = c.inputs["delivery"], c.inputs["candidate"]
    dlv_id, cand_id = M.art_id(dlv), M.art_id(cand)
    best = c.brief["best"]
    labels, clusters = c.inputs["evalset_labels"], c.inputs["evalset_inputs"]
    pe = predictions(dlv, "predictions_eval")
    parents = [dlv_id, cand_id, c.brief["ids"]["evalset"]]
    compare = None
    if milestone:
        compare = pe  # the candidate is the best version: the delivery against itself
    elif best is not None:
        side = c.inputs.get("best_delivery_current_evalset") or c.inputs["best_delivery"]
        compare = predictions(side, "predictions_eval")
        if "best_delivery_current_evalset" in c.inputs:
            parents.append(M.art_id(side))
    sc = M.score(c.brief, labels, pe, train=c.inputs["trainset_rows"], clusters=clusters, compare=compare)
    ev = {"metrics": {"candidate": sc["metrics"], "baselines": sc["baselines"], "best": None}, "deltas": {}}
    if compare:
        ev["metrics"]["best"] = M.score(c.brief, labels, compare, train=c.inputs["trainset_rows"],
                                        clusters=clusters)["metrics"]
        ev["deltas"][primary] = sc["delta"]
    if milestone:
        decision, reason = "keep", ("milestone evaluation: checks the current best version %s on the sealed testset; "
                                    "no promotion decision" % best["version"])
    else:
        gate = M.read_json(os.path.join(cand, "manifest.json"))["train_gate"]["passed"]
        decision, reason = promotion(task, ev, best is not None, gate)
    groups, fn, fp = slices(c, pe)
    entry = entry_of(c)
    failures = []
    if fn or fp:
        failures.append({"issue": "on the evalset, spam is missed or normal mail flagged",
                         "evidence": "evalset errors: %d missed, %d wrongly flagged; accuracy per group %s" % (
                             fn, fp, ", ".join(
                             "%s %.3f" % (g, v[0] / float(v[1])) for g, v in sorted(groups.items()))),
                         "attribution": M.ATTRIBUTION[entry], "module": entry})
    d = c.new_dir("milestone-evaluation" if milestone else "evaluation")
    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Evaluation %s (candidate %s)\n\n| Group | Accuracy | Rows |\n|---|---|---|\n" % (dlv_id, cand_id))
        for g, v in sorted(groups.items()):
            f.write("| %s | %.4f | %d |\n" % (g, v[0] / float(v[1]), v[1]))
        f.write("\n%s %s, baselines %s, delta %s\n" % (primary, sc["metrics"][primary], sc["baselines"],
                                                     sc.get("delta")))
    m = M.base_manifest(c.brief, "evaluation", "judge", parents)
    m["files"] = {"report": {"path": "report.md"}}
    m.update({"delivery_id": dlv_id, "candidate_id": cand_id, "evalset_id": c.brief["ids"]["evalset"],
              "best_before": {"version": best["version"], "evaluation_id": best["evaluation_id"]} if best else None,
              "metrics": ev["metrics"], "deltas": ev["deltas"],
              "recommendation": {"decision": decision, "reason": reason}, "failures": failures,
              "uses_testset": milestone})
    if milestone:
        test = M.score(c.brief, c.inputs["testset_labels"], predictions(dlv, "predictions_test"))
        m["metrics_test"] = test["metrics"]
    scored = M.baselines_scored(c, milestone)
    if scored:
        m["baselines_test" if milestone else "baselines_eval"] = scored
    M.write_json(os.path.join(d, "manifest.json"), m)
    c.check("evaluation", d)
    return d, m


def evaluate(c):
    d, m = write_evaluation(c, False)
    primary = c.task["metrics"]["primary"]
    delta = m["deltas"].get(primary)
    c.save_state({"evaluation": os.path.basename(d)})
    c.handback([d], notes="evalset %s %.4f%s; recommends %s (%s)" % (
        primary, m["metrics"]["candidate"][primary],
        ", against the best %+.4f [%.4f, %.4f]" % (delta["delta"], delta["ci_low"], delta["ci_high"]) if delta else "",
        m["recommendation"]["decision"], m["recommendation"]["reason"]))


def milestone(c):
    d, m = write_evaluation(c, True)
    primary = c.task["metrics"]["primary"]
    r = M.write_record(c, "record", "a release must be checked on the sealed testset first",
                       "the sealed testset never took part in any choice",
                       "scored %s on the testset" % m["delivery_id"], "test %s" % primary, None,
                       m["metrics_test"][primary],
                       "success", "testset %s %.4f, evalset %.4f" % (primary, m["metrics_test"][primary],
                                                                   m["metrics"]["candidate"][primary]),
                       artifacts=[m["delivery_id"]])
    c.handback([d, r], notes="milestone evaluation: testset %s %.4f" % (primary, m["metrics_test"][primary]))


def plan(c, revise):
    st = c.state()
    reason = c.reason()
    entry = st.get("entry") or entry_of(c)
    if revise:  # the user's words name the module to enter at
        words = re.search(r"The user asked for changes to plan \S+? \([^)]*\): (.*)", c.feedback)
        said = words.group(1) if words else reason
        named = [m for m in MODULES if re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % m, said)]
        entry = named[0] if named else entry
    head = c.feedback.split("\n## ")[0]  # why the session got the feedback (the sections after it are details)
    ev_id = None
    m = re.search(r"evaluation-\d{4}", head)
    if m and "closed, no evaluation" not in head:
        ev_id = m.group(0)
    elif c.purpose == "plan":
        evs = [k for k in c.inputs if k.startswith("evaluation-")]
        ev_id = evs[-1] if evs else None
    primary = c.task["metrics"]["primary"]
    if ev_id and st.get("evaluation"):
        em = M.read_json(os.path.join(c.outdir, st["evaluation"], "manifest.json"))
        evidence = "%s: evalset %s %.4f; %s" % (ev_id, primary, em["metrics"]["candidate"][primary],
                                             "; ".join(f["evidence"] for f in em["failures"]) or "no failures")
    elif ev_id:
        evidence = "%s (%s)" % (ev_id, reason)
    else:
        evidence = "round closed without evaluation: no candidate passed the train gate (%s)" % (
            reason or "see this round's candidates")
    issue = {"issue": "improve from %s next round" % entry, "evidence": evidence, "attribution": M.ATTRIBUTION[entry],
             "module": entry, "change": {"data": "widen the training data's coverage",
                                         "train": "tune capacity and learning rate in the config space",
                                         "deliver": "make the delivery package run better"}[entry],
             "expected": {"metric": "eval %s" % primary, "direction": "up", "size": 0.02}}
    d = c.new_dir("plan")
    pm = M.base_manifest(c.brief, "plan", "judge", [ev_id] if ev_id else [])
    pm.update({"evaluation_id": ev_id, "issues": [issue], "entry_module": entry})
    M.write_json(os.path.join(d, "manifest.json"), pm)
    c.check("plan", d)
    st["entry"] = entry
    c.save_state(st)
    r = M.write_record(c, "record", evidence[:200], "the plan targets the largest attributed source of errors",
                       "plan: enter at %s next round%s" % (entry, " (as the user asked to change it)" if revise
                                                          else ""),
                       "eval %s (expected)" % primary, None, None, "inconclusive",
                       "next round shows whether the plan hit",
                       artifacts=[ev_id] if ev_id else [])
    c.handback([d, r], notes="plan: enter at %s next round" % entry)


def main():
    c = M.Call("judge")
    if c.rehand_if_answered():
        return
    if c.purpose == "milestone_judge":
        return milestone(c)
    revise = "The user asked for changes to plan" in c.feedback
    if c.purpose == "plan" or revise or "write the next round's plan" in c.reason():
        return plan(c, revise)
    evaluate(c)


if __name__ == "__main__":
    main()
