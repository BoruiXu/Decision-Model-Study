"""Mock rsi-data (see mocklib for the call, the task and the knobs).

purpose sets      round 1: testset, evalset, valset (60 clusters of 2 rows each) and a trainset (100 clusters),
                  plus a record
purpose trainset  a new trainset: the current one plus 30 new clusters (a new namespace per round), plus a record
                  (no training: the round's train and judge dispatches test the new trainset)
purpose refresh   a new evalset (a namespace no earlier set used), plus a record

Sent back because the trainset failed its quality gate (sets or trainset), it hands back the trainset without its
near duplicates; it reports any other return as failed. With the questions knob (sets or trainset) the first
call's trainset also holds UNKNOWN_CLUSTERS clusters of rows of unknown source (meta.source unknown-export); given
the user's answer (the engine registers nothing before it), it drops them when the answer says "drop" and hands back
the new trainset with a record, else it hands back the same subdirectories.

The quality gate computes the task's quality_checks (min_rows, max_near_dup_rate, min_label_agreement).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mocklib as M  # noqa: E402

SPLIT_CLUSTERS, TRAIN_CLUSTERS, ADDED_CLUSTERS = 60, 100, 30
UNKNOWN_CLUSTERS, UNKNOWN_SOURCE = 6, "unknown-export"


def unknown_rows(c):
    """The rows of unknown source the first call adds when it has questions to ask (none otherwise)."""
    if not (c.first and c.knob("questions")):
        return []
    rows = M.gen_rows(M.question_of(c.task), "train", 90 + c.round, "unk%d" % c.round, UNKNOWN_CLUSTERS)
    for r in rows:
        r["meta"]["source"] = UNKNOWN_SOURCE
    return rows


def quality(task, rows):
    seen, dups = set(), 0
    for r in rows:
        fp = M.fingerprint(r["state"])
        dups += fp in seen
        seen.add(fp)
    by_cluster = {}
    for r in rows:
        by_cluster.setdefault(r["meta"]["cluster"], []).append(M.hard(r["target"]))
    agree = 0
    for labs in by_cluster.values():
        agree += max(labs.count(x) for x in set(labs))
    values = {"min_rows": (len(rows), lambda v, t: v >= t),
              "max_near_dup_rate": (round(dups / float(len(rows)), 6), lambda v, t: v <= t),
              "min_label_agreement": (round(agree / float(len(rows)), 6), lambda v, t: v >= t)}
    checks = []
    for name, thr in sorted((task.get("quality_checks") or {}).items()):
        if name in values:
            v, ok = values[name]
            checks.append({"name": name, "value": v, "threshold": thr, "passed": bool(ok(v, thr))})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def write_trainset(c, rows, parent, changes):
    d = c.new_dir("trainset")
    q = quality(c.task, rows)
    n = M.write_jsonl(os.path.join(d, "rows.jsonl"), rows)
    M.write_json(os.path.join(d, "quality_report.json"),
                 {"checks": q["checks"], "rows": n, "method": "near duplicates by input fingerprint; label "
                                                               "agreement within a cluster"})
    m = M.base_manifest(c.brief, "trainset", "data", [parent] if parent else [])
    m["files"] = {"rows": {"path": "rows.jsonl", "rows": n}, "quality_report": {"path": "quality_report.json"}}
    m.update({"label_method": "rule: template words (synthetic)", "parent_trainset": parent, "changes": changes,
              "quality": q})
    M.write_json(os.path.join(d, "manifest.json"), m)
    c.check("trainset", d)
    return d, q


def write_split(c, kind, rows):
    d = c.new_dir(kind)
    m = M.base_manifest(c.brief, kind, "data")
    ni = M.write_jsonl(os.path.join(d, "inputs.jsonl"), [M.input_row(r) for r in rows])
    if kind == "valset":
        nr = M.write_jsonl(os.path.join(d, "rows.jsonl"), rows)
        m["files"] = {"rows": {"path": "rows.jsonl", "rows": nr}, "inputs": {"path": "inputs.jsonl", "rows": ni}}
    else:
        nl = M.write_jsonl(os.path.join(d, "labels.jsonl"), [M.label_row(r) for r in rows])
        m["files"] = {"inputs": {"path": "inputs.jsonl", "rows": ni}, "labels": {"path": "labels.jsonl", "rows": nl}}
    M.write_json(os.path.join(d, "manifest.json"), m)
    c.check(kind, d)
    return d


def gate_text(q):
    return "; ".join("%s = %s (threshold %s, %s)" % (x["name"], x["value"], x["threshold"],
                                                     "passed" if x["passed"] else "failed") for x in q["checks"])


def dedup(rows):
    seen, kept = set(), []
    for r in rows:
        fp = M.fingerprint(r["state"])
        if fp not in seen:
            seen.add(fp)
            kept.append(r)
    return kept


def fix_quality(c, st):
    """A return for a failed quality gate: the last trainset without its near duplicates."""
    rows = M.rows_of(os.path.join(c.outdir, st["trainset"]))
    kept = dedup(rows)
    d, q = write_trainset(c, kept, st.get("parent"), "%s; dropped %d near-duplicate rows by the return's feedback "
                          "(case and whitespace variants)" % (st["changes"], len(rows) - len(kept)))
    before = st.get("quality")
    st.update(trainset=os.path.basename(d), quality=q)
    c.save_state(st)
    r = M.write_record(c, "record", "the trainset failed the quality gate (near-duplicate rate over the limit)",
                       "without the near duplicates the near-duplicate rate falls below the threshold",
                       "dropped %d near-duplicate rows" % (len(rows) - len(kept)), "trainset near_dup_rate",
                       _check(before, "max_near_dup_rate"), _check(q, "max_near_dup_rate"),
                       "success" if q["passed"] else "failure",
                       "merged exports must be deduplicated by input fingerprint first")
    c.handback([d, r], notes="quality gate: %s" % gate_text(q))


def _check(q, name):
    for x in (q or {}).get("checks", []):
        if x["name"] == name:
            return x["value"]
    return None


def apply_answer(c, st):
    """The user answered the questions of the last hand-back (none of it registered): an answer saying "drop" drops
    the rows of unknown source from that trainset (a new trainset and a record replace the old ones); any other
    answer hands back the same subdirectories."""
    question, answer = c.answers()[-1]
    names = c.previous_artifacts()
    old = st.get("trainset")
    if "drop" not in answer or not old or old not in names:
        return c.handback([os.path.join(c.outdir, n) for n in names],
                          notes="the user's answer: %s; the artifacts handed back are unchanged" % answer)
    rows = M.rows_of(os.path.join(c.outdir, old))
    kept = [r for r in rows if r["meta"].get("source") != UNKNOWN_SOURCE]
    changes = "%s; dropped %d rows of unknown source by the user's answer" % (st["changes"], len(rows) - len(kept))
    d, q = write_trainset(c, kept, st.get("parent"), changes)
    r = M.write_record(c, "record", "the new batch of data has samples of unknown source",
                       "without the samples of unknown source the trainset keeps only data of known source",
                       "dropped %d rows by the user's answer (%s)" % (len(rows) - len(kept), answer), "trainset rows",
                       len(rows), len(kept), "success", "ask the user before using data of unknown source",
                       artifacts=[st["parent"]] if st.get("parent") else [])
    st.update(trainset=os.path.basename(d), changes=changes, quality=q)
    c.save_state(st)
    keep = [os.path.join(c.outdir, n) for n in names if n != old and not n.startswith("record")]
    c.handback(keep + [d, r], notes="dropped %d rows of unknown source by the user's answer; quality gate: %s" % (
        len(rows) - len(kept), gate_text(q)))


def returned(c, st):
    """A return to this session: True when it was handled (the user's answer to its questions: apply_answer; a
    failed quality gate: the trainset deduplicated; anything else: reported as failed, the sets already handed back
    are not made again)."""
    if not c.handed_back_before():
        return False  # e.g. a session that died before handing back: do the whole job
    if c.answers():
        apply_answer(c, st)
    elif st.get("trainset") and "quality gate" in c.feedback:
        fix_quality(c, st)
    else:
        c.fail("rsi-data does not know what to change for this return (it handles only a failed quality gate): %s"
               % (c.reason() or "the feedback gives no reason"))
    return True


def sets(c):
    st = c.state()
    if returned(c, st):
        return
    q = M.question_of(c.task)
    rows = {"testset": M.gen_rows(q, "test", 11, "te", SPLIT_CLUSTERS),
            "evalset": M.gen_rows(q, "eval", 12, "ev", SPLIT_CLUSTERS),
            "valset": M.gen_rows(q, "val", 13, "va", SPLIT_CLUSTERS)}
    train = M.gen_rows(q, "train", 14, "tr1", TRAIN_CLUSTERS)
    rate = c.knob("near_dups") if c.first else None
    changes = "first version: %d clusters" % TRAIN_CLUSTERS
    unknown = unknown_rows(c)
    if unknown:
        train += unknown
        changes += ", with %d rows of unknown source" % len(unknown)
    if rate:
        train += M.near_duplicates(train, rate, "dp1")
        changes += ", merged with a second export (with near duplicates)"
    for kind in ("testset", "evalset", "valset"):
        ov = M.overlap(train, rows[kind])
        if any(ov):
            c.fail("trainset overlaps %s: %s" % (kind, ov))
    dirs = [write_split(c, kind, rows[kind]) for kind in ("testset", "evalset", "valset")]
    d, q = write_trainset(c, train, None, changes)
    st.update(trainset=os.path.basename(d), parent=None, changes=changes, quality=q)
    c.save_state(st)
    r = M.write_record(c, "record", "no data yet", "labelling by template rules is cheap and consistent",
                       "generated testset/evalset/valset/trainset",
                       "trainset label agreement", None, _check(q, "min_label_agreement"),
                       "success" if q["passed"] else "failure", "first version of the data")
    c.handback(dirs + [d, r], notes="quality gate: %s" % gate_text(q))


def trainset(c):
    st = c.state()
    if returned(c, st):
        return
    prev_dir, val_dir = c.inputs["trainset"], c.inputs["valset"]
    prev_id, val_id = M.art_id(prev_dir), M.art_id(val_dir)
    prev_rows = M.rows_of(prev_dir)
    added = M.gen_rows(M.question_of(c.task), "train", 20 + c.round, "tr%d" % c.round, ADDED_CLUSTERS)
    rows = prev_rows + added
    changes = "added %d new clusters (%d rows) to %s" % (ADDED_CLUSTERS, len(added), prev_id)
    unknown = unknown_rows(c)
    if unknown:
        rows += unknown
        changes += ", with %d rows of unknown source (%d clusters)" % (len(unknown), UNKNOWN_CLUSTERS)
    rate = c.knob("near_dups") if c.first else None
    if rate:
        rows += M.near_duplicates(rows, rate, "dp%d" % c.round)
        changes += ", merged with a second export (with near duplicates)"
    val_rows = M.rows_of(val_dir)
    if any(M.overlap(rows, val_rows)):
        c.fail("the new trainset overlaps the valset: %s" % (M.overlap(rows, val_rows),))
    d, q = write_trainset(c, rows, prev_id, changes)
    st.update(trainset=os.path.basename(d), parent=prev_id, changes=changes, quality=q)
    c.save_state(st)
    primary = c.task["metrics"]["primary"]
    # The data module does not train: the round's train and judge dispatches test the new trainset.
    r = M.write_record(c, "record", "the plan attributes the issue to data coverage",
                       "more samples let the model learn more of the spam words",
                       changes, "eval %s" % primary, None, None, "inconclusive",
                       "tested by this round's ② Train & tune and ⑤ Judge & decide", artifacts=[prev_id, val_id])
    c.handback([d, r], notes="quality gate: %s; this round's ② Train & tune and ⑤ Judge & decide test the new "
                             "trainset" % gate_text(q))


def refresh(c):
    ns = 40 + c.round
    rows = M.gen_rows(M.question_of(c.task), "eval", ns, "ev%d" % c.round, SPLIT_CLUSTERS)
    seen = []
    for name in ("trainset", "valset"):
        path = c.inputs.get(name)
        if path:
            seen += M.rows_of(path)
    ov = M.overlap(rows, seen)
    if any(ov):
        c.fail("the new evalset overlaps the trainset or the valset: %s" % (ov,))
    d = write_split(c, "evalset", rows)
    r = M.write_record(c, "record", "the evalset reached its use limit",
                       "new samples from the same generator keep the evalset comparable",
                       "generated a new evalset (%d rows, new namespace ev%d)" % (len(rows), c.round), "eval rows",
                       None, len(rows), "success", "a refreshed evalset needs new id and cluster namespaces")
    c.handback([d, r], notes="new evalset of %d rows; overlap (ids, inputs, clusters) with the trainset and valset "
                             "given: %s" % (len(rows), ov))


def main():
    c = M.Call("data")
    if c.purpose == "sets":
        sets(c)
    elif c.purpose == "trainset":
        trainset(c)
    elif c.purpose == "refresh":
        refresh(c)
    else:
        c.fail("rsi-data does not know the purpose %r" % c.purpose)


if __name__ == "__main__":
    main()
