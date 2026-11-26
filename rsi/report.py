"""The run report of `rsi report`: fixed-code Markdown.

Everything comes from STATE.json, the registry and the registered manifests (and,
for the total time, the first `rsi run` in work/run.jsonl); building the report
writes nothing. When no version was ever promoted the report opens with a
"No releasable version was produced" summary listing each round's outcome.
Before a stop it is a progress report. With ③ Compare on, each round gets a
`### ③ Compare (round n)` subsection (the candidate against its strongest
baseline, compare_line, and the comparison's table), and the milestone section
lists the baselines on the sealed testset; both are report only. The
orchestrating agent adds the interpretation (what was learned, what to try
next) on top of it.
"""
import datetime
import os

from rsi import timing
from rsi.rules import amount_text, compare_settings, is_number, ledger, stop_settings
from rsi.timing import plural
from rsi.workspace import RsiError, latest_comparison

# The sections under Experiment records: (heading, the record modules it holds), in order. The stages of the main
# flow, an old workspace's tune records under ② (as rsi.dashboard.stage_of), then the orchestrator's.
RECORD_SECTIONS = (("① Data", ("data",)), ("② Train & tune", ("train", "tune")), ("③ Compare", ("compare",)),
                   ("④ Deliver", ("deliver",)), ("⑤ Judge & decide", ("judge",)), ("Orchestrator", ("orchestrator",)))
# What became of a long job still running when the run stopped or its dispatch was closed (a dispatch record's
# long_job_stop.result, and each of its others', written by rsi.engine): terminated, gone (its process had ended),
# or one that may still be running (LONG_JOB_LEFT; error: terminating it raised). A close whose termination has not
# run yet says pending (rsi.decisions.close_dispatch).
LONG_JOB_LEFT = ("unconfirmed", "no_pid_file", "bad_pid_file", "error")


def _pid_files(st):
    """The pid files a long_job_stop dealt with: its marker's (the record itself), then the orphans (others)."""
    return [st] + [x for x in st.get("others") or [] if isinstance(x, dict)]


def _job_name(d, pids):
    """"d-0002 (rsi-train, pid 9627)", "d-0002 (rsi-train, pids 9627, 9630)" or "d-0002 (rsi-train)"."""
    pids = [str(p) for p in pids if p]
    return "%s (rsi-%s%s)" % (d.get("id"), d.get("module"), "" if not pids else ", pid %s" % pids[0]
                              if len(pids) == 1 else ", pids %s" % ", ".join(pids))


def long_job_lines(ws):
    """The long jobs a stop or a closed dispatch dealt with, from the dispatch records' long_job_stop (each pid file):
    (terminated by the stop, terminated when a decision or `rsi close-round` closed the dispatch (context close),
    left), each a list of (text, number of long jobs): one "d-0002 (rsi-train, pid 9627)" per dispatch for the
    terminated ones (its pids together), one per pid file for a left one (it may still be running), which also says
    why."""
    stopped, closed, left = [], [], []
    for d in ws.dispatches():
        st = d.get("long_job_stop")
        if not st:
            continue
        ended = [x for x in _pid_files(st) if x.get("result") == "terminated"]
        if ended:
            (closed if st.get("context") == "close" else stopped).append(
                (_job_name(d, [x.get("pid") for x in ended]), len(ended)))
        for x in _pid_files(st):
            if x.get("result") in LONG_JOB_LEFT:
                left.append(("%s: %s" % (_job_name(d, [x.get("pid")]), _clause(x.get("why"))), 1))
    return stopped, closed, left


def long_job_text(ws):
    """long_job_lines in sentences, separated by one space ("" for none): the stop's terminations apart from an
    earlier close's."""
    stopped, closed, left = long_job_lines(ws)
    out = []
    if stopped:
        out.append("The stop terminated %s %s." % (_jobs(stopped), _texts(stopped, ", ")))
    if closed:
        out.append("Closing the %s terminated %s %s." % ("dispatch" if len(closed) == 1 else "dispatches",
                                                          _jobs(closed), _texts(closed, ", ")))
    if left:
        one = len(left) == 1
        out.append("%s not terminated (%s may still be running; end %s by hand): %s." % (
            _jobs(left).capitalize(), "it" if one else "they", "it" if one else "them", _texts(left, "; ")))
    return " ".join(out)


def _jobs(items):
    """"long job" or "long jobs" for the long_job_lines entries `items`."""
    return "long job" if sum(n for _t, n in items) == 1 else "long jobs"


def _texts(items, sep):
    return sep.join(t for t, _n in items)


def _num(x):
    return "%.6g" % x if is_number(x) else "n/a"


def _clause(text):
    """`text` (any value) as a clause that goes on, in brackets or before more words: its closing full stop
    dropped, so a reason ending in "." never makes ".." or "(reason.)."."""
    return str(text).rstrip(".")


def best_scores(ws):
    """Where the best version's primary metric was measured (the pages and the report say it): None without a best
    version, else {"version", "value", "evaluation_id", "evalset_id" (of that evaluation), "current_evalset",
    "refreshed", "current"}. refreshed: the evalset was refreshed since that evaluation (its evalset is no longer
    the current one); then current is the best version's score on the current evalset, {"value", "evaluation_id",
    "delivery_id"}, from the latest evaluation there that compared against it (best_before is the best version:
    metrics.best, computed on its re-delivery, a parent of that evaluation), or None while there is none yet."""
    best = ws.state["best"]
    if best is None:
        return None
    v = ws.version(best)
    primary = ws.task()["metrics"]["primary"]
    own = ws.manifest(v["evaluation_id"])["evalset_id"]
    cur = ws.current("evalset")
    out = {"version": best, "value": ws.best_primary(), "evaluation_id": v["evaluation_id"], "evalset_id": own,
           "current_evalset": cur, "refreshed": cur is not None and cur != own, "current": None}
    if not out["refreshed"]:
        return out
    for e in reversed(ws.entries("evaluation")):
        m = ws.manifest(e["id"])
        if m["evalset_id"] != cur or m.get("uses_testset") or (m.get("best_before") or {}).get("version") != best:
            continue
        x = (m["metrics"].get("best") or {}).get(primary)
        dl = [p for p in m.get("parents") or [] if ws.is_kind(p, "delivery")
              and ws.manifest(p)["candidate_id"] == v["candidate_id"] and ws.manifest(p)["evalset_id"] == cur]
        out["current"] = {"value": x if is_number(x) else None, "evaluation_id": e["id"],
                          "delivery_id": dl[0] if dl else ws.best_delivery_on_current_evalset()}
        break
    return out


def _cell(text):
    """A Markdown table cell: one line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(header, rows):
    lines = ["| %s |" % " | ".join(header), "|" + "---|" * len(header)]
    return lines + ["| %s |" % " | ".join(_cell(c) for c in row) for row in rows]


def _rounds(ws):
    """(round number, STATE entry) in round order."""
    return sorted((int(k), r) for k, r in ws.state["rounds"].items())


def _outcome(ws, n, r):
    """(decision, reason, evaluation id) of round `n` (STATE entry `r`).

    An undecided round gives "undecided", followed by its registered evaluations (for
    example one left undecided by a user stop).
    """
    d = r["decision"]
    if d is None:
        pending = ws.evaluations(n)
        return ("undecided (%s)" % ", ".join(pending) if pending else "undecided"), "", ""
    decision = d.get("decision") + (" " + d["version"] if d.get("version") else "")
    return decision, d.get("reason") or "", d.get("evaluation_id") or "none"


def _metrics(metrics, primary):
    """`name value` pairs, primary metric first."""
    keys = [primary] if primary in metrics else []
    keys += [k for k in metrics if k != primary]
    return ", ".join("%s %s" % (k, _num(metrics[k])) for k in keys)


def _summary(ws):
    if ws.state["versions"]:
        return []
    lines = ["**No releasable version was produced.** No candidate was promoted to a version. Each round's outcome:",
             ""]
    for n, r in _rounds(ws):
        decision, reason, _eval = _outcome(ws, n, r)
        lines.append("- Round %d: %s%s" % (n, decision, "; " + reason if reason else ""))
    return lines + [""]


def _task(ws, task):
    s = ws.state
    d, m = task["decision"], task["metrics"]
    st = stop_settings(task)
    settings = ["max_rounds %d" % st["max_rounds"]]
    for key in ("patience", "max_attempts_per_round"):
        settings.append("%s %d%s" % (key, st[key], " (default)" if key in st["defaulted"] else ""))
    return [
        "## Task", "",
        "- Task: %s v%d (%s)" % (s["task_id"], s["task_version"], ws.current("task")),
        "- Decision type: %s (labels: %s)" % (d["type"], ", ".join(d["labels"])),
        "- Primary metric: %s (%s)" % (m["primary"], "higher is better" if m["higher_is_better"]
                                       else "lower is better"),
        "- Target: %s" % ("not set" if st["target"] is None else "%s %s" % (m["primary"], _num(st["target"]))),
        "- Stop settings: %s" % ", ".join(settings),
        "",
    ]


def _time_line(ws, now):
    """The run's total time as a report line, or None when its start is unknown: from the first `rsi run`
    (timing.run_window; the workspace may have been made hours before), else from round 1's opening (`rsi init`)."""
    start, end, approx = timing.run_window(ws)
    if start is None:
        return None
    made = timing.made_at(ws)[0] or start
    since = "" if timing.format_full(made) == timing.format_full(start) \
        else "; workspace created %s" % timing.format_full(made)
    if timing.stopped_unknown(ws):
        return "- Total time: unknown (started %s%s)" % (timing.format_full(start), since)
    dur = timing.approx_duration(timing.seconds_between(start, end, now), approx)
    if end is None:
        return "- Elapsed: %s (started %s, up to this report%s)" % (dur, timing.format_full(start), since)
    return "- Total time: %s (started %s, stopped %s%s)" % (dur, timing.format_full(start), timing.format_full(end),
                                                            since)


def _stop(ws, now):
    st = ws.state["stopped"]
    if st:
        when = ([] if st.get("round") is None else ["round %d" % st["round"]]) + [timing.format_full(st["at"]) if timing._good(st.get("at")) else str(st.get("at"))]
        line = "- Stopped (%s): %s" % (", ".join(when), _clause(st.get("reason")))
        if st.get("by"):
            line += "; by %s" % st["by"]
    else:
        stop, why = ws.stop_check()
        line = "- Not stopped. Stop check: %s: %s" % ("stop" if stop else "continue", why)
    extra = _time_line(ws, now)
    return ["## Stop status", "", line] + ([extra] if extra else []) + _long_jobs(ws) + [""]


def _long_jobs(ws):
    """Lines on the long jobs still running when the run stopped (long_job_stop context stop, or none: an older
    record) or when a decision or `rsi close-round` closed their dispatch (context close), each under its own words:
    terminated, or left running (with why, the marker and the pid, for the user to end it), or not terminated yet
    (pending: a close cut off after its save). Of the orphan pid files (others) those whose process had ended are
    left out."""
    lines = []
    for d in ws.dispatches():
        st = d.get("long_job_stop")
        if not st:
            continue
        head = "- %s, long job %s (rsi-%s): " % ("When its dispatch was closed" if st.get("context") == "close"
                                                 else "At the stop", d.get("id"), d.get("module"))
        if st.get("result") == "pending":
            lines.append(head + "not terminated yet (cut off after the dispatch was closed); the next rsi run, dispatch "
                         "close or stop of the run terminates it; completion marker %s" % d.get("marker"))
            continue
        for i, x in enumerate(_pid_files(st)):
            marker = d.get("marker") if i == 0 else (x.get("pid_file") or "")[:-len(".pid")]
            if x.get("result") == "terminated":
                lines.append(head + "terminated (pid %s)" % x.get("pid"))
            elif x.get("result") == "gone":
                if i == 0:
                    lines.append(head + "its process was already gone (pid %s)" % x.get("pid"))
            else:
                lines.append(head + "not terminated, may still be running (%s); completion marker %s; end it by "
                             "hand%s" % (_clause(x.get("why")), marker,
                                         " (pid %s)" % x["pid"] if x.get("pid") else ""))
    return lines


def _rounds_table(ws, now):
    rows = []
    wins = timing.round_windows(ws)  # round 1 from the first `rsi run`, as the total time
    for n, r in _rounds(ws):
        decision, reason, evaluation = _outcome(ws, n, r)
        rows.append((n, r["entry"], decision, reason, evaluation, timing.round_text(wins[n], now, r["decision"] is not None) or "none"))
    header = ("Round", "Entry module", "Decision", "Reason", "Evaluation", "Elapsed")
    return ["## Rounds", ""] + _table(header, rows) + [""] + _compare_sections(ws)


# ---- ③ Compare: candidate vs the baselines (report only) ------------------------------------------------------

def delta_text(d):
    """A delta with its interval, e.g. "+0.031 [+0.012, +0.050]"."""
    return "%+.3f [%+.3f, %+.3f]" % (d["delta"], d["ci_low"], d["ci_high"])


def comparison_table(m, primary):
    """The table of the comparison manifest `m` (its report.md and REPORT.md show the same): the candidate, then each
    baseline as `name (id)` in the order of `baselines`; the valset's primary metric, each other metric, and the
    primary metric's delta, candidate minus model, with its interval."""
    others = []
    for values in m["metrics_val"].values():
        others += [n for n in values if n != primary and n not in others]
    head = ["Model", primary] + others + ["Δ%s (candidate − model) [CI]" % primary]

    def row(label, values, delta):
        return [label] + [_num(values.get(n)) for n in [primary] + others] + [delta]

    rows = [row("candidate (%s)" % m["candidate_id"], m["metrics_val"]["candidate"], "")]
    for name, bid in m["baselines"].items():
        d = m["deltas_val"].get(name)
        rows.append(row("%s (%s)" % (name, bid), m["metrics_val"].get(name, {}), delta_text(d) if d else "n/a"))
    return _table(head, rows)


def skipped_lines(m):
    """One line per baseline the comparison `m` skipped, with why."""
    return ["- Skipped %s: %s" % (name, _clause(why)) for name, why in m["skipped"].items()]


def strongest_line(m, primary, eval_delta=None):
    """"Candidate vs strongest baseline (<name>): Δ<primary> <delta> [<lo>, <hi>] on valset", with "; <delta> [<lo>,
    <hi>] on evalset" when the judge reported the strongest baseline there (`eval_delta`)."""
    s = m["strongest"]
    if s is None:
        return "Candidate vs strongest baseline: none compared (%s)" % (
            "skipped: %s" % ", ".join(m["skipped"]) if m["skipped"] else "no baseline")
    d = m["deltas_val"].get(s)
    line = "Candidate vs strongest baseline (%s): Δ%s %s on valset" % (s, primary, delta_text(d) if d else "n/a")
    return line + ("; %s on evalset" % delta_text(eval_delta) if eval_delta else "")


def round_comparison(ws, rnd):
    """The comparison round `rnd` reports (REPORT.md, `rsi status`, the ③ node), selected as the judge's brief and
    registration select it (rsi.workspace.latest_comparison): of the candidate of the round's latest evaluation (no
    milestone) on that evaluation's evalset; before that evaluation, in the current round, of the chosen candidate
    (rsi.flow.chosen_candidate, the one ④ delivers) on the current evalset; None without one."""
    from rsi import flow as F  # rsi.flow imports this module
    ids = ws.state["rounds"][str(rnd)]["artifacts"]
    for a in reversed(ids):
        if ws.entry(a)["kind"] == "evaluation" and not ws.manifest(a)["uses_testset"]:
            e = ws.manifest(a)
            return latest_comparison(ws, ids, e["candidate_id"], e["evalset_id"])
    if rnd != ws.state["round"]:
        return None
    return latest_comparison(ws, ids, F.chosen_candidate(ws), ws.current("evalset"))


def strongest_on_evalset(ws, rnd, m):
    """The delta of comparison `m`'s strongest baseline on the evalset, from the latest evaluation of round `rnd` (no
    milestone) of its candidate that reports it in baselines_eval; None without one."""
    bid = m["baselines"].get(m["strongest"])
    for a in reversed(ws.state["rounds"][str(rnd)]["artifacts"]):
        if ws.entry(a)["kind"] != "evaluation":
            continue
        e = ws.manifest(a)
        if not e["uses_testset"] and e["candidate_id"] == m["candidate_id"] and bid in (e.get("baselines_eval") or {}):
            return e["baselines_eval"][bid]["delta"]
    return None


def compare_line(ws, rnd):
    """Round `rnd`'s candidate against its strongest baseline (strongest_line), on the valset and, once the judge
    reported it, the evalset; None when the round has no comparison."""
    comp = round_comparison(ws, rnd)
    if comp is None:
        return None
    m = ws.manifest(comp)
    return strongest_line(m, ws.task()["metrics"]["primary"], strongest_on_evalset(ws, rnd, m))


def _compare_sections(ws):
    """A `### ③ Compare (round n)` subsection per round while the task has a compare section (or a round holds a
    comparison): compare_line, the comparison's table and the baselines it skipped. Nothing for an old workspace."""
    on = compare_settings(ws.task()) is not None
    primary = ws.task()["metrics"]["primary"]
    lines = []
    for n, _r in _rounds(ws):
        comp = round_comparison(ws, n)
        if comp is None and not on:
            continue
        lines += ["### ③ Compare (round %d)" % n, ""]
        if comp is None:
            lines += ["No comparison in this round.", ""]
            continue
        m = ws.manifest(comp)
        lines += [compare_line(ws, n), "", "%s of %s on %s (report only, never a gate):" % (
            comp, m["candidate_id"], m["valset_id"]), ""] + comparison_table(m, primary) + [""]
        skipped = skipped_lines(m)
        lines += skipped + ([""] if skipped else [])
    return lines


def _versions(ws, primary):
    s = ws.state
    lines = ["## Versions", ""]
    if not s["versions"]:
        return lines + ["No versions.", "", "Current best: none", ""]
    lines += _table(("Version", "Candidate", "Delivery", "Evaluation", "Round"),
                    [(v["version"], v["candidate_id"], v["delivery_id"], v["evaluation_id"], v["round"])
                     for v in s["versions"]])
    v = ws.version(s["best"])
    sc = best_scores(ws)
    lines += ["", "Current best: %s, %s %s (%s), on evalset %s" % (s["best"], primary, _num(sc["value"]),
                                                                  v["evaluation_id"], sc["evalset_id"])]
    if sc["refreshed"]:  # the best version's score is on an evalset that is no longer the current one
        cur = sc["current"]
        if cur is None:
            lines.append("- No score on the current evalset %s yet (it waits for the evaluation after its "
                         "re-delivery)" % sc["current_evalset"])
        else:
            lines.append("- On the current evalset %s: %s %s (%s, computed with the re-delivered %s)" % (
                sc["current_evalset"], primary, _num(cur["value"]), cur["evaluation_id"], cur["delivery_id"]))
        lines.append("- The stop target is still judged by the score on %s" % sc["evalset_id"])
    lines += [""]
    package = ws.manifest(v["delivery_id"])["package"]
    report = ws.manifest(v["evaluation_id"])["files"]["report"]["path"]
    lines += [
        "- Model package: %s" % ws.package_location(v["delivery_id"]),
        "- Entry point: %s" % package["entry"],
        "- Environment: %s" % package["environment"],
        "- Delivery directory: %s" % ws.path(v["delivery_id"]),
        "- Evaluation report: %s" % os.path.join(ws.path(v["evaluation_id"]), report),
        "",
    ]
    return lines


def _milestones(ws, primary):
    """Milestone evaluations with their testset metrics, then the files each one hands over."""
    rows, files = [], []
    for e in ws.entries("evaluation"):
        m = ws.manifest(e["id"])
        if not m["uses_testset"]:
            continue
        rows.append((e["id"], m["candidate_id"], e["round"], _metrics(m["metrics_test"], primary)))
        files.append("- %s: evaluation report %s; delivery %s (%s), model package %s"
                     % (e["id"], os.path.join(ws.path(e["id"]), m["files"]["report"]["path"]),
                        m["delivery_id"], ws.path(m["delivery_id"]), ws.package_location(m["delivery_id"])))
        for bid, b in sorted((m.get("baselines_test") or {}).items()):  # ③'s baselines scored on the testset
            files.append("- On the sealed testset: candidate vs %s Δ%s %s (%s, %s; report only)"
                         % (b["name"], b["delta"]["metric"], delta_text(b["delta"]), e["id"], bid))
    lines = ["## Milestone evaluations", ""]
    if not rows:
        return lines + ["None.", ""]
    return lines + _table(("Evaluation", "Candidate", "Round", "Test metrics"), rows) + [""] + files + [""]


def _records(ws):
    """The records by section (RECORD_SECTIONS; a module of none gets its own section, after them), each in
    registry order."""
    heading = dict((m, title) for title, modules in RECORD_SECTIONS for m in modules)
    groups = {}
    for e in ws.entries("record"):
        m = ws.manifest(e["id"])
        title = heading.get(m["module"], m["module"])
        groups.setdefault(title, []).append((e["id"], e["round"], m["result"]["conclusion"]))
    lines = ["## Experiment records", ""]
    if not groups:
        return lines + ["None.", ""]
    titles = [t for t, _m in RECORD_SECTIONS]
    for title in [t for t in titles if t in groups] + sorted(t for t in groups if t not in titles):
        lines += ["### %s" % title, ""] + _table(("Record", "Round", "Conclusion"), groups[title]) + [""]
    return lines


# What each resource of the spend ledger counts (rsi.rules.RESOURCES; docs/interfaces/spend.md), for the report.
LEDGER_NOTES = {
    "compute_hours": "budget.compute_hours: the cost.compute_hours of registered and refused candidates, and the "
                     "compute_hours lines of the spend files",
    "api_usd": "budget.api_cost: the LLM calls that generate or review data, from the spend files",
    "agent_usd": "budget.agent_cost: the module and decision agents' sessions, as their runner reports them",
    "annotation_items": "budget.annotation_items: the annotation_items lines of the spend files",
}


def _budget(ws, task):
    s = ws.state
    limit = ws.attempt_limit()
    st = stop_settings(task)
    lines = ["## Attempts and budget", "",
             "At most %s per round (max_attempts_per_round%s; each search counts once, the trial candidates it "
             "compares do not)." % (plural(limit, "attempt"),
                                    ", default" if "max_attempts_per_round" in st["defaulted"] else ""), ""]
    rows = []
    for n, _r in _rounds(ws):
        done = ws.attempts(n)
        rows.append((n, "%d of %d" % (len(done), limit), ", ".join(done)))
    lines += _table(("Round", "Attempts", "Artifacts"), rows) + [""]
    for key, label, used, budget in ledger(task, s):  # the spend ledger, the two dollar accounts apart
        lines.append("- %s: %s used, %s (%s)" % (label[:1].upper() + label[1:], amount_text(key, used),
                                                 "no budget set" if budget is None else "budget %s" % _num(budget),
                                                 LEDGER_NOTES[key]))
    problems = ws.spend_problems()
    if problems:
        lines.append("- Spend-file problems: %s (each logged as spend_invalid in work/run.jsonl; the amounts on them "
                     "that are numbers >= 0 are counted)" % plural(problems, "line"))
    lines += ["- Evalset uses: %d / %d (%s)" % (s["eval_use_count"], task["eval_use_limit"],
                                                ws.current("evalset") or "none"), ""]
    return lines


def report_markdown(ws, now=None):
    """The run report of workspace `ws` as Markdown text. Reads only. `now` (a datetime) is the
    report time the running durations are measured to (default: the current time)."""
    now = datetime.datetime.now() if now is None else now
    task = ws.task()
    primary = task["metrics"]["primary"]
    lines = ["# RSI run report: %s" % ws.state["task_id"], ""]
    lines += _summary(ws)
    if not ws.state["stopped"]:
        lines += ["The run has not stopped yet; this is a progress report.", ""]
    lines += _task(ws, task)
    lines += _stop(ws, now)
    lines += _rounds_table(ws, now)
    lines += _versions(ws, primary)
    lines += _milestones(ws, primary)
    lines += _records(ws)
    lines += _budget(ws, task)
    return "\n".join(lines).rstrip("\n") + "\n"


def _same(a, b):
    return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)


def _protected(ws, path):
    """True when writing `path` would write STATE.json, registry.jsonl or anything under artifacts/.

    Paths are compared with os.path.samefile, not as strings, so another
    spelling of the same file is caught too: a case variant on a
    case-insensitive filesystem, a hard link, a symbolic link. The path as
    given and its resolved form (which follows a dangling link) are both
    checked, each with all its ancestors against artifacts/.
    """
    own = [os.path.join(ws.root, name) for name in ("STATE.json", "registry.jsonl")]
    arts = os.path.join(ws.root, "artifacts")
    for p in (os.path.abspath(path), os.path.realpath(path)):
        if any(_same(p, o) for o in own):
            return True
        while True:
            if _same(p, arts):
                return True
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
    return False


def write_report(ws, out):
    """Write the report to `out`; return its absolute path.

    Refuses the workspace's own files (STATE.json, registry.jsonl) and any
    path under artifacts/, which only registration writes, under any spelling
    (see _protected).
    """
    path = os.path.abspath(out)
    if _protected(ws, path):
        raise RsiError("--out %s: the report must not replace STATE.json or registry.jsonl or go under "
                       "artifacts/; write it elsewhere, e.g. <ws>/REPORT.md" % out)
    if os.path.isdir(path):
        raise RsiError("--out %s is a directory; give a file path, e.g. <ws>/REPORT.md" % out)
    text = report_markdown(ws)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        raise RsiError("--out %s: cannot write the report: %s" % (out, e))
    return path
