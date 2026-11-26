"""Mock rsi-compare, the compare module (see mocklib for the call, the task and the knobs).

Under `rsi run --runner mock` this script stands in for compare's fixed command (modules/compare/SKILL.md), and works
the way that contract says, in two parts in one session. First part: one baseline, `prior`, in bl-prior/: the
trainset's class prior (each label's mean target over the trainset rows) as the prediction of every valset row and
every evalset input; its metrics_val come from `rsi score`. It is trained again in every dispatch (no reuse). Second
part, after the engine registered it (its id read from registered-<n>.json beside the feedback-<n>.md this launch
gets, as the contract says): the comparison of the round's candidate with it
on the valset in comparison/, every number from `rsi score` (the delta with --compare and the valset inputs as
clusters), handed back with a record. With the long_job knob the first call hands back long_job first (marker
baselines.marker); a check-in hands back long_job again; the session continued after the marker says ok trains the
baseline. The milestone purpose (milestone_compare): each baseline of the brief's comparison scored on the testset
inputs (bl-<name>-test/: reused_from its id, its valset and evalset predictions copied, the prior it recorded as the
prediction of every testset input), every one in one hand-back.
"""
import hashlib
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mocklib as M  # noqa: E402

NAME = "prior"
SPEC = {"model": "class prior", "fitted_on": "trainset targets"}
MARKER = "baselines.marker"


def files_of(art_dir, logical):
    """The path of a logical file of an artifact directory (its manifest's `files` entry)."""
    return os.path.join(art_dir, M.read_json(os.path.join(art_dir, "manifest.json"))["files"][logical]["path"])


def prior(c):
    """label -> the mean target of the trainset rows (the class prior), and the decision it gives."""
    labels = c.task["decision"]["labels"]
    rows = M.rows_of(c.inputs["trainset"])
    probs = dict((x, sum(r["target"].get(x, 0.0) for r in rows) / len(rows)) for x in labels)
    decision = max(labels, key=lambda x: (probs[x], -labels.index(x)))
    return probs, decision


def predictions(ids, probs, decision):
    return [{"id": i, "probs": dict(probs), "decision": decision, "uncertainty": round(1 - probs[decision], 6)}
            for i in ids]


def baseline(c, st):
    """The first part: bl-prior/, handed back alone (the comparison follows once it is registered)."""
    ids = c.brief["ids"]
    probs, decision = prior(c)
    d = c.new_dir("bl-prior")
    val = [r["id"] for r in M.rows_of(c.inputs["valset"])]
    ev = [r["id"] for r in M.read_jsonl(c.inputs["evalset_inputs"])]
    files = {}
    for logical, row_ids in (("predictions_val", val), ("predictions_eval", ev)):
        n = M.write_jsonl(os.path.join(d, logical + ".jsonl"), predictions(row_ids, probs, decision))
        files[logical] = {"path": logical + ".jsonl", "rows": n}
    metrics = M.score(c.brief, files_of(c.inputs["valset"], "rows"), os.path.join(d, "predictions_val.jsonl"))
    m = M.base_manifest(c.brief, "baseline", "compare", [ids["trainset"], ids["valset"], ids["evalset"]],
                        notes="launch %d (session %s)" % (c.n, c.session))
    m["files"] = files
    m.update({"name": NAME, "trainset_id": ids["trainset"], "valset_id": ids["valset"], "evalset_id": ids["evalset"],
              "testset_id": None, "spec": SPEC,
              "spec_sha256": hashlib.sha256(json.dumps([NAME, SPEC], sort_keys=True).encode("utf-8")).hexdigest(),
              "config": {"probs": probs}, "features": None, "metrics_val": metrics["metrics"],
              "cost": {"compute_hours": 0.0}, "reused_from": None})
    M.write_json(os.path.join(d, "manifest.json"), m)
    c.check("baseline", d)
    st["baseline"] = os.path.basename(d)
    st.pop("long_job", None)
    c.save_state(st)
    primary = c.task["metrics"]["primary"]
    c.handback([d], notes="baseline %s (the trainset's class prior %s): valset %s %s; the comparison follows once it "
                          "is registered" % (NAME, json.dumps(probs, sort_keys=True), primary,
                                             metrics["metrics"].get(primary)))


def registered_baseline(c, st):
    """The id the engine registered bl-prior under: registered-<n>.json beside this launch's feedback-<n>.md (a
    followup's or a return's: [{"dir", "kind", "id"}]); None when it lists none."""
    m = re.match(r"^feedback-(\d+)\.md$", c.feedback_file or "")
    path = os.path.join(c.outdir, "registered-%s.json" % m.group(1)) if m else ""
    if not (st.get("baseline") and os.path.isfile(path)):
        return None
    ids = [x["id"] for x in M.read_json(path) if x.get("kind") == "baseline" and x.get("dir") == st["baseline"]]
    return ids[-1] if ids else None


def comparison(c, st, bid):
    """The second part: comparison/ of the candidate with the registered baseline `bid`, and a record."""
    primary = c.task["metrics"]["primary"]
    cand = c.inputs["candidate"]
    cid, cm = M.art_id(cand), M.read_json(os.path.join(cand, "manifest.json"))
    labels, clusters = files_of(c.inputs["valset"], "rows"), files_of(c.inputs["valset"], "inputs")
    cand_pv = os.path.join(cand, cm["files"]["predictions_val"]["path"])
    base_pv = os.path.join(c.outdir, st["baseline"], "predictions_val.jsonl")
    s = M.score(c.brief, labels, cand_pv, clusters=clusters, compare=base_pv)
    base = M.score(c.brief, labels, base_pv)["metrics"]
    delta = dict((k, s["delta"][k]) for k in ("delta", "ci_low", "ci_high"))
    d = c.new_dir("comparison")
    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write("| Model | %s |\n|---|---|\n| candidate %s | %s |\n| %s | %s |\n\nCandidate minus %s: %s [%s, %s]\n"
                % (primary, cid, s["metrics"].get(primary), NAME, base.get(primary), NAME, delta["delta"],
                   delta["ci_low"], delta["ci_high"]))
    m = M.base_manifest(c.brief, "comparison", "compare", [cid, bid, cm["trainset_id"], cm["valset_id"],
                                                            c.brief["ids"]["evalset"]])
    m["files"] = {"report": {"path": "report.md"}}
    m.update({"candidate_id": cid, "trainset_id": cm["trainset_id"], "valset_id": cm["valset_id"],
              "evalset_id": c.brief["ids"]["evalset"], "baselines": {NAME: bid},
              "metrics_val": {"candidate": s["metrics"], NAME: base}, "deltas_val": {NAME: dict(delta, metric=primary)},
              "strongest": NAME, "skipped": {}})
    M.write_json(os.path.join(d, "manifest.json"), m)
    c.check("comparison", d)
    r = M.write_record(c, "record", "the round's candidate has not been compared with a simple baseline",
                       "the candidate beats the trainset's class prior on the valset",
                       "fitted the class prior on %s and compared %s with it" % (cm["trainset_id"], cid),
                       "val %s" % primary, base.get(primary), s["metrics"].get(primary),
                       "success" if delta["delta"] > 0 else "failure",
                       "%s minus the class prior: %+.4f [%+.4f, %+.4f]" % (primary, delta["delta"], delta["ci_low"],
                                                                           delta["ci_high"]),
                       artifacts=[cid, bid])
    c.handback([d, r], notes="comparison of %s with %s (%s): %s %+.4f [%+.4f, %+.4f] on the valset" % (
        cid, NAME, bid, primary, delta["delta"], delta["ci_low"], delta["ci_high"]))


def milestone(c):
    """Purpose milestone_compare: bl-<name>-test/ for each baseline of the brief's comparison, handed back together."""
    comp = M.read_json(os.path.join(c.inputs["comparison"], "manifest.json"))
    labels = c.task["decision"]["labels"]
    test_ids = [r["id"] for r in M.read_jsonl(c.inputs["testset_inputs"])]
    dirs = []
    for name, bid in sorted(comp["baselines"].items()):
        src = c.inputs[bid]
        sm = M.read_json(os.path.join(src, "manifest.json"))
        d = c.new_dir("bl-%s-test" % name)
        files = {}
        for logical in ("predictions_val", "predictions_eval"):
            shutil.copyfile(files_of(src, logical), os.path.join(d, logical + ".jsonl"))
            files[logical] = dict(sm["files"][logical], path=logical + ".jsonl")
        probs = sm["config"]["probs"]
        decision = max(labels, key=lambda x: (probs[x], -labels.index(x)))
        n = M.write_jsonl(os.path.join(d, "predictions_test.jsonl"), predictions(test_ids, probs, decision))
        files["predictions_test"] = {"path": "predictions_test.jsonl", "rows": n}
        tid = c.brief["ids"]["testset"]
        m = M.base_manifest(c.brief, "baseline", "compare", [bid, tid], notes="%s scored on %s" % (bid, tid))
        m["files"] = files
        m.update(dict((k, sm[k]) for k in ("name", "trainset_id", "valset_id", "evalset_id", "spec", "spec_sha256",
                                           "config", "features", "metrics_val")))
        m.update({"testset_id": tid, "cost": {"compute_hours": 0.0}, "reused_from": bid})
        M.write_json(os.path.join(d, "manifest.json"), m)
        c.check("baseline", d)
        dirs.append(d)
    c.handback(dirs, notes="the baselines of %s scored on the testset inputs: %s" % (
        comp["id"], ", ".join(os.path.basename(d) for d in dirs)))


def main():
    c = M.Call("compare")
    if c.rehand_if_answered():
        return
    if c.purpose == "milestone_compare":
        return milestone(c)
    if c.purpose != "compare":
        c.fail("the mock compare does not simulate purpose %s" % c.purpose)
    st = c.state()
    job = st.get("long_job")
    if job and c.feedback.startswith("# Check-in"):
        return c.handback([], status="long_job", marker=job["marker"],
                          notes="check-in: background job pid %d %s" % (job["pid"], "running" if M.pid_alive(job["pid"])
                                                                       else "ended"))
    if job and "Long job finished" in c.feedback:
        first = (M.read_text(job["marker"]).splitlines() or [""])[0].strip()
        if first != "ok":
            c.fail("the completion marker %s is not ok: %s" % (job["marker"], first or "(no marker)"))
        return baseline(c, st)
    bid = registered_baseline(c, st)
    if bid:
        return comparison(c, st, bid)
    knob = c.knob("long_job") if c.first else None
    if knob:
        marker = os.path.join(c.outdir, MARKER)
        pid = M.start_background([], marker, os.path.join(c.outdir, "baselines.log"), release=knob.get("release"),
                                 wait_s=knob.get("seconds", 1))
        st["long_job"] = {"marker": marker, "pid": pid}
        c.save_state(st)
        return c.handback([], status="long_job", marker=marker,
                          notes="the baseline is trained in the background (pid %d); the marker %s follows" % (pid,
                                                                                                              marker))
    baseline(c, st)


if __name__ == "__main__":
    main()
