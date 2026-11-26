"""The main flow of a run as one pure function: next_action(ws) -> the next thing the run does.

next_action reads the workspace and writes nothing: STATE.json (with the engine's records dispatches,
decisions and waiting), the registry, registered manifests (task.json for the task), and whether
<ws>/REPORT.md and a long job's completion marker exist (the marker written by that job: marker_done); and the
clock, for a long job's check-in. The engine carries the action out and asks again; the progress pages
(rsi.dashboard) show the same action as what the run waits for, so the two cannot disagree.

An action is a dict. "type" is one of ACTION_TYPES; its other fields:

  dispatch          module, purpose, reason; a deliver dispatch also candidate_id (the candidate to deliver,
                    None when only the best version's re-delivery is owed)
  resume            dispatch_id, feedback (what to send to that dispatch's session), mode: "resume" (a return,
                    counted against flow.max_returns), "followup" (the second part of a two-part hand-back: the
                    train module's search, the judge's plan; counted apart, no return), "answer" (the user
                    answered the questions of the dispatch's hand-back, which waits unregistered: counted apart, no
                    return) or "checkin" (a long job's marker is not there and its check-in is due: the session
                    looks at the job and hands back again; counted apart, no return)
  wait_agent        dispatch_ids (the agents running, and records pending: not launched yet)
  wait_long_job     dispatch_id, marker, checkin_at (when the next check-in of a long job is due, UTC ISO rounded up
                    to the second; None without check-ins: the engine then stops and waits for `rsi run` again)
  process_handback  dispatch_id
  decide            point, options (decision_options: a compare dispatch's also offer skip), default (the code's
                    choice: rsi.decisions.default_choice), dispatch_id (the dispatch concerned, or None), reason
  ask               kind ("approval" | "question"), what, ref, question, options (ASK_OPTIONS for an approval,
                    MILESTONE_OPTIONS for the milestone; a question the engine asks for a decision offers the
                    exits of rsi.decisions)
  promote           evaluation_id
  close_round       reason
  next              reason
  report, done

Every action also carries "step" (one of STEPS): where in the main flow the run is, for the pages
and the run log. An ask copied from STATE waiting carries the step of the round underneath it.
The engine stores a dispatch's purpose (PURPOSES) in its record; a dispatch made for a decision's
choice takes CHOICE_PURPOSE of that choice. A record without a purpose (written before purposes)
counts as its module's usual one (dispatch_purpose).

Order (the first that holds):

  1. stopped: report until REPORT.md exists, then done;
  2. STATE waiting set: that ask again (the engine stops there until the answer is recorded), unless
     it is stale (stale_waiting: an approval STATE already holds, or one asked for a held hand-back whose
     dispatch was closed; a decision's question the flow, leaving waiting aside, no longer gives as the same
     decide, or whose dispatch was closed; a milestone question no longer open; a check-in question the flow no
     longer asks: its dispatch closed, its marker there, the user's answer recorded) and the flow goes on (the
     engine holds a hand-back whose candidates would go over the compute budget and asks over_budget, waiting
     dispatch_id: budget_held; approved, the same hand-back is processed again);
  3. dispatches (any round): a hand-back whose questions the user has answered goes back to its session
     (resume, mode answer: it was held unregistered while the question waited); a hand-back to process (status
     handed_back, or long_job whose marker is there: marker_done); a failed dispatch that is the latest of its module in the current round, and in a decided
     round only one of the steps after the decision (DECIDED_PURPOSES: the judge's plan, the milestone) (decide
     handback_refused when its latest launch handed back artifacts that registration refused, else
     handback_failed: the engine resolves it by changing the dispatch's status); agents running or records
     pending (wait_agent, all of them: the engine launches a pending one); a long job whose check-in is due
     (resume, mode checkin; at the cap ask the user: CHECKIN_OPTIONS), else the long job (wait_long_job);
  4. the current round, undecided: its evaluation (promote); the attempts (searches) used up with no candidate
     through the train gate (close_round); else the first main-flow step not done, from the round's
     entry: ① data (round 1: the four sets; any round entering at data: a trainset through the quality
     gate), ② train (the train module: its search's trial candidates registered without their search,
     while attempts remain: the search; else a candidate through the train gate), ③ compare (only when the task
     has a compare section: the baselines, then the comparison of the chosen candidate on the current evalset, in
     one session; report only: done once the comparison is registered or a decision chose skip), ④ deliver (a
     delivery of the round's candidate on the current evalset, plus the best version's re-delivery after a refresh),
     ⑤ judge; before ③, ④ and ⑤ the evalset's use limit (ask refresh_sets, then data for the new evalset), before
     ① a trainset (a round after the first), ② and a new ③ dispatch the budget: any resource of the spend ledger
     whose budget its use has reached (rsi.rules.reached) asks over_budget when the task requires it, naming each one;
  5. the current round, decided: when the stop check says stop, the milestone release (ask once when
     there is a best version, then deliver, compare when it is on and the best version has a comparison (its
     baselines scored on the testset inputs), and judge on the testset if the user said yes) and next
     (which stops); else the judge's plan, its approval when the task requires it (the user's revise
     sends the plan back to the judge's session, or to a new plan dispatch, once per answer), next.

The user's answers are STATE decisions too: flow reads the milestone answer and a plan's revise; an answer
at a decision point is carried out by the engine (rsi.decisions), approvals are STATE approvals.

A step is dispatched when its module has no dispatch for it (by purpose) in the round yet. When it
was dispatched and the step is still not done, the step returns to that dispatch: a dispatch that
produced nothing (no registered artifact from its output directory) is a failed hand-back
(handback_failed) before any gate is looked at; else the decision points train_gate_failed (about the
round's latest train dispatch, or without one the dispatch of the latest candidate) and
quality_gate_failed for a gate that failed, handback_failed otherwise; a two-part hand-back (the search of the
train module after its trial candidates are registered, compare's comparison after its baselines are registered, the
judge's plan after the round is decided) goes on in the same session as a followup, no return (second_part_due: once
after the launch that registered the first part; asked again, it is a return). Returns are capped per module and
round by the task's flow.max_returns (rsi.rules.flow_settings, counted by returns_used): at the cap a decision
defaults to asking the user, and for a compare dispatch to skip (③ is report only, never a block).
The default at a decision point comes from rsi.decisions.default_choice (the one place that knows the
decision table; imported inside _decide, since rsi.decisions builds on this module).
What a dispatch produced is read from the registry (artifacts whose source is in its output directory),
never from times, which have whole seconds only.
"""
import datetime
import os

from rsi import workspace as W
from rsi.report import _num
from rsi.rules import EPS, GPU_BASELINES, compare_settings, flow_settings, is_number
from rsi.timing import plural
from rsi.workspace import DATASETS, DISPATCH_STATUSES, RsiError, parse_utc  # noqa: F401  (DISPATCH_STATUSES: re-exported)

# How far a time may be off, in whole seconds: a long job's completion marker's or pid file's modification time and a
# process's start time against the launch of the dispatch as the host's clock recorded it (whole seconds). On Linux `ps` reckons a start
# time from the boot time (/proc/stat btime, whole seconds) and the uptime, which drift apart from the wall clock: it
# can put a process a second before it started.
LONG_JOB_SLACK_S = 2
# The stamp the engine writes into a dispatch's output directory at each launch (n: the record's launches), before
# its agent starts: on the filesystem of the output directory, so its modification time comes from the same clock as
# a long job's marker's (an NFS server's, which may run apart from this host's). marker_done compares against it.
LAUNCH_STAMP = ".launch-%d"

# The modules: data, train (the train module: one dispatch runs one search in the task's
# search_space), compare (③, report only, on when the task has a compare section), deliver, judge. Old workspaces may
# hold dispatches and artifacts of "tune" (merged into train): they are read as they are and never dispatched again.
MODULES = ("data", "train", "compare", "deliver", "judge")
# Code modules: their dispatches run a fixed command, not an agent (the engine's code runner).
CODE_MODULES = ("compare",)


def stage_of(name):
    """The stage (a MODULES name) of a dispatch's module, an artifact's producer or a record's module: an old
    workspace's "tune" (merged into the train module) is train's, a module is its own, anything else
    (orchestrator, an unknown name) has none (None). The briefs and the pages (rsi.dashboard re-exports it) map an
    old tune to train only through it."""
    if name == "tune":
        return "train"
    return name if name in MODULES else None


# Main-flow position of each module and of a round's entry module; "tune", the entry of an old plan, opens the
# round at train.
STEP_INDEX = {"data": 0, "train": 1, "tune": 1, "compare": 2, "deliver": 3, "judge": 4}
REPORT_FILE = "REPORT.md"

ACTION_TYPES = ("dispatch", "resume", "wait_agent", "wait_long_job", "process_handback", "decide",
                "ask", "promote", "close_round", "next", "report", "done")
# data, refresh (data makes a new evalset), train (② a search and its search artifact), compare (③ the baselines and
# the comparison), deliver, judge, promote, close (close_round), write_plan (the judge writes the plan), plan (its
# approval), decided (next, milestone question), milestone (the milestone release under way), running, handed, stopped.
STEPS = ("data", "refresh", "train", "compare", "deliver", "judge", "promote", "close", "write_plan", "plan",
         "decided", "milestone", "running", "handed", "stopped")
# What a dispatch is for: sets (round 1's four sets), trainset (a new trainset), refresh (a new evalset),
# train (a search), compare (the baselines, then the comparison of the round's candidate), deliver (the round's
# delivery and the best version's re-delivery), judge (the evaluation), plan (the plan of a round whose judge has no
# session to resume), milestone_deliver, milestone_compare (the baselines of the best version's comparison scored on
# the testset inputs), milestone_judge. An old workspace's tune dispatches keep their purpose "tune"
# (dispatch_purpose).
PURPOSES = ("sets", "trainset", "refresh", "train", "compare", "deliver", "judge", "plan", "milestone_deliver",
            "milestone_compare", "milestone_judge")

TRAIN_GATE_FAILED = "train_gate_failed"
QUALITY_GATE_FAILED = "quality_gate_failed"
HANDBACK_REFUSED = "handback_refused"
HANDBACK_FAILED = "handback_failed"
DECISION_OPTIONS = {TRAIN_GATE_FAILED: ("train", "data", "ask"), QUALITY_GATE_FAILED: ("resume_data", "ask"),
                    HANDBACK_REFUSED: ("resume", "ask"), HANDBACK_FAILED: ("resume", "ask")}
CHOICE_PURPOSE = {"train": "train", "data": "trainset"}  # train_gate_failed choices dispatched anew
# The choice that closes a compare dispatch that failed or was refused and goes on without its step (③ Compare is
# report only, never a gate): decision_options offers it for a compare dispatch, before ask.
SKIP = "skip"
# The options of each approval the flow asks for: every answer changes the state (approve records the approval;
# revise sends the plan back to the judge with the user's words; stop stops the run, as `rsi stop`).
# refresh_sets may also close the round instead (no new evalset, the judge writes the plan), as the manual flow could.
ASK_OPTIONS = {"plan": ("approve", "revise"), "refresh_sets": ("approve", "close_round", "stop"),
               "over_budget": ("approve", "stop")}
# Purposes of the dispatches that still work once the round is decided: the judge's plan (also the second part
# of its judge dispatch) and the milestone release. A failed dispatch of any other purpose is moot then.
DECIDED_PURPOSES = ("judge", "plan", "milestone_deliver", "milestone_compare", "milestone_judge")
MILESTONE = "milestone"  # the decision point (STATE decisions) holding the user's answer, and the ask's what
MILESTONE_OPTIONS = ("yes", "no")
# A long job's check-ins at flow.max_checkins: the question's what (ref: the dispatch) and the decision point holding
# the user's answer: continue (flow.max_checkins more check-ins), wait (no more check-ins: wait for the marker),
# stop (stop the run, as `rsi stop`: the long job is terminated).
CHECKIN = "checkin"
CHECKIN_OPTIONS = ("continue", "wait", "stop")


def _act(type_, step, **fields):
    out = {"type": type_}
    out.update(fields)
    out["step"] = step
    return out


def _round(ws):
    s = ws.state
    return s["rounds"][str(s["round"])]


def _arts(ws, rnd=None):
    return [ws.entry(a) for a in (_round(ws) if rnd is None else rnd)["artifacts"]]


def _inside(child, parent):
    return os.path.commonpath([child, parent]) == parent


# ---- facts about the round, shared with the pages ----------------------------------------------

def plan_approved(ws, plan):
    """True when STATE records a plan approval with ref `plan` (the rule of `rsi status`)."""
    return any(a.get("what") == "plan" and a.get("ref") == plan for a in ws.state["approvals"])


def decision_text(d):
    """promote V1 (evaluation-0001), keep (evaluation-0002), or keep (closed, no evaluation)."""
    if d.get("evaluation_id") is None:
        return "%s (closed, no evaluation)" % d.get("decision")
    return "%s%s (%s)" % (d.get("decision"), " " + d["version"] if d.get("version") else "", d["evaluation_id"])


def data_missing(ws, arts):
    """Why step ① is not complete in the current round, or None: in round 1 a current
    testset, evalset, valset or trainset is missing; in any round no trainset of this round
    passed its quality gate (one that failed is evidence only, never current)."""
    if ws.state["round"] == 1:
        missing = [k for k in reversed(DATASETS) if ws.current(k) is None]
        if missing:
            return "round 1 still lacks %s" % ", ".join(missing)
    if not any(e["kind"] == "trainset" and _quality_passed(ws, e) for e in arts):
        return "no trainset of this round has passed its quality gate yet"
    return None


def _quality_passed(ws, e):
    return ws.manifest(e["id"])["quality"]["passed"] is True


def attempts_exhausted(ws, rnd):
    """True when the round's attempts are used up with no candidate through the train gate, the
    round is undecided, has no evaluation and the run is not stopped: only close_round is left."""
    return (len(ws.attempts()) >= ws.attempt_limit() and not ws.passed_candidates() and rnd["decision"] is None
            and not ws.evaluations() and not ws.state["stopped"])


def own_delivery(ws, arts):
    """True when this round has the delivery step ④ asks for: a delivery on the current evalset of one
    of the round's own candidates (the best version's re-delivery after a refresh, whose candidate is from
    an earlier round, does not count); in a round with no candidate of its own (entry deliver) any delivery
    on the current evalset; a milestone delivery (predictions_test) always."""
    cands = set(e["id"] for e in arts if e["kind"] == "candidate")
    cur = ws.current("evalset")
    for e in arts:
        if e["kind"] == "delivery":
            m = ws.manifest(e["id"])
            if "predictions_test" in m["files"]:
                return True
            if m["evalset_id"] == cur and (not cands or m["candidate_id"] in cands):
                return True
    return False


def redeliver_pending(ws):
    """After a refreshed evalset the best version has no delivery on the new evalset yet (the rule of
    `rsi status`): ④ also re-delivers it, the evaluation compares the candidate with it there."""
    return ws.state["best"] is not None and ws.best_delivery_on_current_evalset() is None


def delivered(ws, arts):
    """True when step ④ is done in an undecided round: the round's own delivery is registered and
    no re-delivery of the best version is owed."""
    return own_delivery(ws, arts) and not redeliver_pending(ws)


def milestone_deliveries(ws):
    """This round's registered deliveries carrying predictions_test (milestone deliveries)."""
    return [a for a in _round(ws)["artifacts"]
            if ws.entry(a)["kind"] == "delivery" and "predictions_test" in ws.manifest(a)["files"]]


def milestone_evaluations(ws):
    return [a for a in _round(ws)["artifacts"]
            if ws.entry(a)["kind"] == "evaluation" and ws.manifest(a)["uses_testset"] is True]


def compare_enabled(task):
    """True when ③ Compare is on: the task has a compare section (rsi.rules.compare_settings). Without one (an old
    workspace, or a task that wants none) the flow goes from ② to ④ and the pages show ③ as off."""
    return compare_settings(task) is not None


def is_compare(d):
    """True for a dispatch record of the compare module (None is none)."""
    return d is not None and d.get("module") == "compare"


def comparison_of(ws, arts, candidate_id):
    """The latest comparison among the registry entries `arts` of candidate `candidate_id` on the current evalset, or
    None (rsi.workspace.latest_comparison, the selection the judge and registration use)."""
    return W.latest_comparison(ws, [e["id"] for e in arts], candidate_id, ws.current("evalset"))


def compare_skipped(ws, purpose="compare"):
    """True when a decision of the current round chose skip (SKIP) on a compare dispatch of `purpose` (compare, or
    milestone_compare): the step is left out, as ③ is report only."""
    for x in ws.decisions(rnd=ws.state["round"]):
        if x.get("choice") != SKIP or not x.get("dispatch_id"):
            continue
        try:
            d = ws.dispatch(x["dispatch_id"])
        except RsiError:
            continue
        if is_compare(d) and dispatch_purpose(d) == purpose:
            return True
    return False


def best_comparison(ws):
    """The latest registered comparison (of any round) of the best version's candidate, or None (no best version, or
    its candidate was never compared)."""
    best = ws.state["best"]
    v = ws.version(best) if best is not None else None
    if v is None:
        return None
    out = None
    for e in ws.entries("comparison"):
        if ws.manifest(e["id"])["candidate_id"] == v["candidate_id"]:
            out = e["id"]
    return out


def milestone_baselines(ws):
    """This round's registered baselines carrying predictions_test (the milestone's: the baselines of the best
    version's comparison scored on the testset inputs; rsi.workspace.milestone_baselines)."""
    return W.milestone_baselines(ws)


def compare_done(ws, arts):
    """True when step ③ needs nothing more in an undecided round: the task has no compare section, the round's entry
    is after ③, the round holds a comparison of the chosen candidate on the current evalset (comparison_of), or a
    decision of this round skipped its compare dispatch (compare_skipped)."""
    if not compare_enabled(ws.task()) or STEP_INDEX.get(_round(ws)["entry"], 0) > STEP_INDEX["compare"]:
        return True
    return comparison_of(ws, arts, chosen_candidate(ws)) is not None or compare_skipped(ws)


def milestone_comparison(ws):
    """The comparison the milestone's compare step scores on the testset: best_comparison(ws) when the task has a
    compare section and that comparison lists at least one baseline (one whose baselines were all skipped has nothing
    to score, and a dispatch for it could never register a baseline); else None."""
    if not compare_enabled(ws.task()):
        return None
    comp = best_comparison(ws)
    return comp if comp is not None and ws.manifest(comp)["baselines"] else None


def milestone_compare_due(ws):
    """True when the milestone release still owes its compare step (between the milestone delivery and the milestone
    judge): there is a milestone_comparison (a compare section, and a comparison of the best version's candidate that
    lists at least one baseline), no baseline of this round carries predictions_test yet (compare registers all of
    them in one hand-back) and no decision skipped the milestone compare dispatch."""
    return (milestone_comparison(ws) is not None and not milestone_baselines(ws)
            and not compare_skipped(ws, "milestone_compare"))


def milestone_state(ws):
    """None unless the milestone release is on the table (not stopped, the round decided, a best
    version, the stop check says stop); then "yes" when the user agreed (the latest milestone decision of
    the round says yes or, from the manual flow, a use_testset approval of this round made at or after the
    round's decision), "no" when the latest milestone decision says anything else, else "ask"."""
    s = ws.state
    decision = _round(ws)["decision"]
    if s["stopped"] or decision is None or s["best"] is None or not ws.stop_check()[0]:
        return None
    answers = ws.decisions(rnd=s["round"], point=MILESTONE)
    if answers:
        return "yes" if answers[-1].get("choice") == "yes" else "no"
    at = parse_utc(decision.get("at"))
    for a in s["approvals"]:
        if a.get("what") == "use_testset" and a.get("round") == s["round"]:
            when = parse_utc(a.get("at"))
            if at is None or (when is not None and when >= at):
                return "yes"
    return "ask"


def chosen_candidate(ws):
    """The candidate steps ③ and ④ compare and deliver: of this round's candidates through the train gate, the best
    on the valset by the task's primary metric (its direction; ties and values that are no number go to the one
    registered first); in a round with none of its own (entry deliver) the best version's candidate; else
    None. The evalset never takes part in this choice."""
    passed = ws.passed_candidates()
    if not passed:
        best = ws.state["best"]
        v = ws.version(best) if best is not None else None
        return v["candidate_id"] if v else None
    metrics = ws.task()["metrics"]
    chosen, value = None, None
    for c in passed:
        x = ws.manifest(c)["metrics_val"].get(metrics["primary"])
        if not is_number(x):
            continue
        if value is None or (x > value + EPS if metrics["higher_is_better"] else x < value - EPS):
            chosen, value = c, x
    return chosen or passed[0]


def marker_path(ws, d):
    """The completion marker of a long-job dispatch: absolute, or relative to the workspace; None if unset."""
    m = d.get("marker")
    if not m:
        return None
    return m if os.path.isabs(m) else os.path.join(ws.root, m)


def launch_stamp(ws, d, n):
    """The stamp of launch `n` of dispatch `d` (LAUNCH_STAMP in its output directory, absolute); None without an
    output directory or a launch number."""
    if not d.get("outdir") or not n:
        return None
    return os.path.join(ws.root, d["outdir"], LAUNCH_STAMP % int(n))


def marker_done(ws, d):
    """True when long-job dispatch `d`'s completion marker is there and this long job wrote it: modified no earlier
    than the launch that started the job. That launch's time is the modification time of its stamp (launch_stamp of
    the record's long_job_launch), on the same filesystem and clock as the marker: compared with no slack (>= also
    holds on a filesystem with whole-second times for a marker written in the stamp's second). Only without the stamp
    (an old record or workspace) the host's clock: the record's long_job_since, else launched_at, LONG_JOB_SLACK_S of
    slack. An older file at that path is an earlier job's, left by a module that reused the marker path."""
    p = marker_path(ws, d)
    if p is None:
        return False
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return False
    try:
        return mtime >= os.path.getmtime(launch_stamp(ws, d, d.get("long_job_launch")) or "")
    except OSError:
        since = parse_utc(d.get("long_job_since")) or parse_utc(d.get("launched_at"))
        return since is None or mtime >= since.timestamp() - LONG_JOB_SLACK_S


def products(ws, d):
    """The registered artifacts that came from dispatch `d`: the registry's entries of its round whose source
    lies in its output directory (sources are recorded as real paths), in registry order; read from the registry,
    so an entry whose registration was cut off before STATE was saved counts too (the engine repairs STATE)."""
    if not d.get("outdir"):
        return []
    out = os.path.realpath(os.path.join(ws.root, d["outdir"]))
    return [e for e in ws.entries() if e.get("round") == d.get("round") and e["kind"] != "task"
            and _inside(e["source"], out)]


def dispatch_purpose(d):
    """A dispatch record's purpose; a record without one counts as its module's usual purpose (data: sets
    in round 1, trainset later; the other modules: their name)."""
    if d.get("purpose"):
        return d["purpose"]
    if d.get("module") == "data":
        return "sets" if d.get("round") == 1 else "trainset"
    return d.get("module")


def returns_used(ws, module, rnd=None):
    """How often `module` was sent back in round `rnd` (default: the current round): the returns of its
    dispatches there plus its dispatches beyond the first (a fresh dispatch cannot reset the count)."""
    ds = ws.dispatches(rnd=ws.state["round"] if rnd is None else rnd, module=module)
    return sum(int(d.get("returns") or 0) for d in ds) + max(0, len(ds) - 1)


def may_return(ws, module):
    """True while `module` may be sent back once more this round (returns_used below flow.max_returns)."""
    return returns_used(ws, module) < flow_settings(ws.task())["max_returns"]


def checkin_answers(ws, d):
    """The user's answers (STATE decisions at point CHECKIN) to the check-in question about dispatch `d`."""
    return [x for x in ws.decisions(point=CHECKIN) if x.get("dispatch_id") == d["id"]]


def checkin_limit(ws, d):
    """How many check-ins dispatch `d` may have: flow.max_checkins, plus as many again (at least one) for each
    continue the user answered; None when the user answered wait (no more check-ins)."""
    answers = checkin_answers(ws, d)
    if answers and answers[-1].get("choice") == "wait":
        return None
    cap = flow_settings(ws.task())["max_checkins"]
    return cap + max(1, cap) * sum(1 for x in answers if x.get("choice") == "continue")


def checkin_due(ws, d):
    """When the next check-in of long-job dispatch `d` is due (an aware UTC datetime): flow.checkin_minutes after its
    latest hand-back (ended_at; launched_at, then started_at, of a record without it); None without check-ins (no
    flow.checkin_minutes, or the user answered wait)."""
    minutes = flow_settings(ws.task())["checkin_minutes"]
    if not minutes or checkin_limit(ws, d) is None:
        return None
    since = parse_utc(d.get("ended_at")) or parse_utc(d.get("launched_at")) or parse_utc(d.get("started_at"))
    if since is None:
        return datetime.datetime.now(datetime.timezone.utc)
    return since + datetime.timedelta(minutes=minutes)


def _iso_up(dt):
    """A UTC datetime as now_iso writes it, rounded up to the second (never before `dt`)."""
    whole = dt.replace(microsecond=0)
    if whole < dt:
        whole += datetime.timedelta(seconds=1)
    return whole.strftime("%Y-%m-%dT%H:%M:%SZ")


def _checkin(ws, d):
    """For long-job dispatch `d` whose marker is not there and whose check-in is due: the check-in (resume, mode
    checkin), or at the cap the question to the user; None when no check-in is due."""
    due = checkin_due(ws, d)
    if due is None or datetime.datetime.now(datetime.timezone.utc) < due:
        return None
    done, cap = int(d.get("checkins") or 0), checkin_limit(ws, d)
    marker = marker_path(ws, d)
    if done < cap:
        return _act("resume", "running", dispatch_id=d["id"], mode="checkin",
                    feedback="the long job is still running (completion marker %s is not there yet); this is check-in "
                             "%d (one every %s)" % (marker, done + 1,
                                                    plural(flow_settings(ws.task())["checkin_minutes"], "minute")))
    return _ask("running", "question", CHECKIN, d["id"],
                "The long job of rsi-%s (%s) has not finished (completion marker %s is not there yet) after %s, "
                "reaching the cap flow.max_checkins = %d. continue: %s; wait: no more check-ins, only wait for the "
                "completion marker; stop: stop the run (as rsi stop, which terminates the long job)"
                % (d.get("module"), d["id"], marker, plural(done, "check-in"), flow_settings(ws.task())["max_checkins"],
                   plural(max(1, flow_settings(ws.task())["max_checkins"]), "more check-in")), CHECKIN_OPTIONS)


# ---- next_action -------------------------------------------------------------------------------

def next_action(ws):
    """The next action of the run (see the module docstring). Reads only; writes nothing."""
    s = ws.state
    if s["stopped"]:
        return _act("done" if os.path.isfile(os.path.join(ws.root, REPORT_FILE)) else "report", "stopped")
    w = ws.waiting()
    if w and not stale_waiting(ws, w):
        return _act("ask", _round_action(ws)["step"], kind=w.get("kind"), what=w.get("what"), ref=w.get("ref"),
                    question=w.get("question"), options=list(w.get("options") or []))
    return _agents(ws) or _round_action(ws)


def stale_waiting(ws, w):
    """True for a waiting question nothing is waiting for any more: an approval STATE already holds (an open
    approval of its what and ref, or one of this round made since the question was asked: a single-use approval
    is consumed once used), or an approval asked for a held hand-back (its dispatch_id: over_budget) once that
    dispatch is closed (or unknown); a decision's question that is moot (decision_moot); a milestone question once
    the milestone is answered or no longer on the table; a module's hand-back question once its dispatch is closed
    (or unknown)."""
    if w.get("kind") == "question" and w.get("what") in DECISION_OPTIONS:
        return decision_moot(ws, w.get("what"), w.get("ref"))
    if w.get("kind") == "question" and w.get("what") == "handback" and w.get("ref"):
        try:
            return bool(ws.dispatch(w["ref"]).get("closed"))
        except RsiError:
            return True
    if w.get("kind") == "question" and w.get("what") == MILESTONE:
        return milestone_state(ws) != "ask"
    if w.get("kind") == "question" and w.get("what") == CHECKIN:
        return checkin_moot(ws, w.get("ref"))
    if w.get("kind") != "approval" or not w.get("what"):
        return False
    if w.get("dispatch_id"):  # a hand-back held for this approval (over_budget): moot once its dispatch is closed
        try:
            if ws.dispatch(w["dispatch_id"]).get("closed"):
                return True
        except RsiError:
            return True
    what, ref = w["what"], w.get("ref")
    if ws.has_approval(what, ref):
        return True
    since = parse_utc(w.get("since"))
    for a in ws.state["approvals"]:
        if a.get("what") == what and a.get("round") == ws.state["round"] and (ref is None or a.get("ref") == ref):
            at = parse_utc(a.get("at"))
            if since is not None and at is not None and at >= since:
                return True
    return False


def open_failure(ws, d):
    """True for a failed dispatch the flow still decides on: of the current round, of a module the flow dispatches
    (not an old workspace's tune), the latest of its module, and, once the round is decided, one of
    DECIDED_PURPOSES."""
    return (d.get("status") == "failed" and d.get("round") == ws.state["round"] and d.get("module") in MODULES
            and _latest(ws, d.get("module")) is d
            and (_round(ws)["decision"] is None or dispatch_purpose(d) in DECIDED_PURPOSES))


def checkin_moot(ws, did):
    """True when the check-in question about dispatch `did` no longer applies: the dispatch is unknown, closed or no
    long job waiting for its marker, or the flow no longer asks it (the user answered, the cap was raised)."""
    try:
        d = ws.dispatch(did)
    except RsiError:
        return True
    if d.get("closed") or d.get("status") != "long_job" or marker_done(ws, d):
        return True
    a = _checkin(ws, d)
    return not (a is not None and a["type"] == "ask")


def decision_moot(ws, point, did):
    """True when a decision's question about dispatch `did` no longer applies: the dispatch is unknown, of
    another round or closed; or the flow, leaving STATE waiting aside, no longer gives the decide with this point
    and dispatch (it was resumed, replaced, its step got done, the round was decided...). A question at the cap on
    a dispatch whose step is incomplete (status done) stays open: the flow still gives that decide."""
    if did:
        try:
            d = ws.dispatch(did)
        except RsiError:
            return True
        if d.get("round") != ws.state["round"] or d.get("closed"):
            return True
    a = _agents(ws) or _round_action(ws)
    return not (a["type"] == "decide" and a.get("point") == point and a.get("dispatch_id") == did)


def _latest(ws, module, purposes=None):
    """The latest dispatch of `module` in the current round (with one of `purposes` when given), or None."""
    ds = [d for d in ws.dispatches(rnd=ws.state["round"], module=module)
          if purposes is None or dispatch_purpose(d) in purposes]
    return ds[-1] if ds else None


def decision_options(point, d):
    """The options of decision `point` about dispatch `d`: DECISION_OPTIONS[point], with SKIP inserted before ask when d
    is a compare dispatch (③ is report only: its failure never blocks the flow)."""
    opts = list(DECISION_OPTIONS[point])
    if is_compare(d) and SKIP not in opts:
        opts.insert(opts.index("ask"), SKIP)
    return opts


def _decide(ws, point, step, d, reason):
    from rsi.decisions import default_choice  # rsi.decisions imports this module
    did = d["id"] if d else None
    default, _why = default_choice(ws, point, {"dispatch_id": did})
    return _act("decide", step, point=point, options=decision_options(point, d), default=default,
                dispatch_id=did, reason=reason)


def _handback_failed(ws, d, step, reason):
    return _decide(ws, HANDBACK_FAILED, step, d, reason)


def refused_now(d):
    """True when the latest launch of dispatch `d` handed back artifacts that registration refused (its
    record's refused entries name the launch)."""
    launch = int(d.get("launches") or 1)
    return any(int(x.get("launch") or launch) == launch for x in d.get("refused") or [])


def refused_decision(ws, d):
    """decide handback_refused for dispatch `d`: the registration refused what its latest launch handed back. The
    reason gives the first line of each refusal once; a line several refusals share (the data artifacts refused
    unregistered after the first refusal of the hand-back, rsi.engine.DATA_CHAIN) is followed by their kinds."""
    m = d.get("module")
    launch = int(d.get("launches") or 1)
    lines = {}  # first line -> the kinds refused with it, in order
    for x in d.get("refused") or []:
        if int(x.get("launch") or launch) == launch:
            e = x.get("error") or ""
            lines.setdefault(e.splitlines()[0] if e else "", []).append(x.get("kind") or x.get("dir") or "?")
    text = "; ".join(line if len(kinds) == 1 else "%s (not registered: %s)" % (line, ", ".join(kinds))
                     for line, kinds in lines.items())
    return _decide(ws, HANDBACK_REFUSED, m if m in STEP_INDEX else "running", d,
                   "registration refused what rsi-%s's dispatch %s handed back: %s" % (m, d["id"], text))


def _ask(step, kind, what, ref, question, options):
    return _act("ask", step, kind=kind, what=what, ref=ref, question=question, options=list(options))


def _dispatch(step, module, purpose, reason, **fields):
    return _act("dispatch", step, module=module, purpose=purpose, reason=reason, **fields)


def handback_answers(ws, d):
    """The user's answers (STATE decisions at point "handback") to the questions dispatch `d` asked with its latest
    launch's hand-back."""
    launch = int(d.get("launches") or 1)
    return [x for x in ws.decisions(point="handback")
            if x.get("dispatch_id") == d["id"] and int(x.get("launches") or 0) == launch]


def question_answered(ws, d):
    """True when dispatch `d` holds a hand-back whose questions the engine asked (the record's `asked`, for its
    latest launch) and the user has answered them: the hand-back goes back to the session with the answers."""
    asked = d.get("asked") or {}
    return int(asked.get("launch") or 0) == int(d.get("launches") or 1) and bool(handback_answers(ws, d))


def budget_held(ws, d):
    """True while dispatch `d` holds a hand-back the engine keeps unregistered for an over_budget approval: the
    record's held_budget names its latest launch, the hand-back is still to be processed (handed_back, or a long
    job's), the current round has no over_budget approval yet and the run has not stopped (a stop ends the hold:
    Workspace.stop counts the held candidates' compute). Once approved, the next process_handback registers it;
    held_budget stays in the record."""
    held = d.get("held_budget") or {}
    return (int(held.get("launch") or 0) == int(d.get("launches") or 1)
            and d.get("status") in ("handed_back", "long_job") and not ws.has_approval("over_budget")
            and not ws.state["stopped"])


def _agents(ws):
    ds = ws.dispatches()
    for d in ds:
        status = d.get("status")
        if status == "handed_back" and question_answered(ws, d):
            return _act("resume", "handed", dispatch_id=d["id"], mode="answer",
                        feedback="You asked questions in your last hand-back and the user has answered them (see "
                                 "below). Nothing of that hand-back is registered yet: change or confirm it by the "
                                 "answers, then hand back again every artifact to register.")
        if status == "handed_back" or (status == "long_job" and marker_done(ws, d)):
            return _act("process_handback", "handed", dispatch_id=d["id"])
    for d in ds:
        if open_failure(ws, d):
            if refused_now(d):
                return refused_decision(ws, d)
            m = d.get("module")
            return _handback_failed(ws, d, m if m in STEP_INDEX else "running",
                                    "rsi-%s's dispatch %s failed with no valid hand-back" % (m, d["id"]))
    running = [d["id"] for d in ds if d.get("status") in ("pending", "running")]
    if running:
        return _act("wait_agent", "running", dispatch_ids=running)
    jobs = [d for d in ds if d.get("status") == "long_job"]
    for d in jobs:
        a = _checkin(ws, d)
        if a is not None:
            return a
    if jobs:
        dues = [x for x in (checkin_due(ws, d) for d in jobs) if x is not None]
        return _act("wait_long_job", "running", dispatch_id=jobs[0]["id"], marker=jobs[0].get("marker"),
                    checkin_at=_iso_up(min(dues)) if dues else None)
    return None


def _round_action(ws):
    """The action the current round needs, leaving aside stopped, waiting and the dispatches' states."""
    rnd = _round(ws)
    if rnd["decision"] is None:
        evaluations = ws.evaluations()
        if evaluations:
            return _act("promote", "promote", evaluation_id=evaluations[-1])
        if attempts_exhausted(ws, rnd):
            return _act("close_round", "close", reason="round %d used up its %s with no candidate through the "
                        "train gate" % (ws.state["round"], plural(ws.attempt_limit(), "attempt")))
        return _undecided(ws, rnd)
    return _decided(ws, rnd)


def _attempts_text(ws):
    return "attempts used %d / %d" % (len(ws.attempts()), ws.attempt_limit())


def _undecided(ws, rnd):
    arts = _arts(ws, rnd)
    start = STEP_INDEX.get(rnd["entry"], 0)
    if start == 0:
        why = data_missing(ws, arts)
        if why:
            return _data_step(ws, arts, why)
    if start <= 1:
        trials = search_owed(ws, arts)
        if trials:
            return _search_step(ws, trials)
        if not ws.passed_candidates():
            return _train_step(ws, rnd, arts)
    if start <= STEP_INDEX["compare"] and not compare_done(ws, arts):
        return _refresh(ws, "compare") or _compare_step(ws, arts)
    if not delivered(ws, arts):
        return _refresh(ws, "deliver") or _deliver_step(ws, arts)
    return _refresh(ws, "judge") or _judge_step(ws)


def _quality_decision(ws, d, reason):
    """quality_gate_failed on the data dispatch `d` (None: no dispatch record, a workspace run by hand)."""
    return _decide(ws, QUALITY_GATE_FAILED, "data", d, reason)


def _data_step(ws, arts, why):
    if ws.state["round"] != 1:  # a trainset: the budget first
        ask = _over_budget(ws, "data")
        if ask:
            return ask
    d = _latest(ws, "data", ("sets", "trainset"))
    if d is not None and not products(ws, d):
        return _handback_failed(ws, d, "data", "rsi-data's dispatch %s handed back but registered no artifact (%s)"
                                % (d["id"], why))
    failed = [e["id"] for e in arts if e["kind"] == "trainset" and not _quality_passed(ws, e)]
    if failed:  # data_missing says no trainset of this round passed: the latest one failed its quality gate
        return _quality_decision(ws, d, "trainset %s failed its quality gate (%s)" % (failed[-1], why))
    if d is not None:
        return _handback_failed(ws, d, "data", "rsi-data handed back and the step is still not complete: %s" % why)
    return _dispatch("data", "data", "sets" if ws.state["round"] == 1 else "trainset", why)


# What step `step` would spend that the over_budget question asks about.
OVER_BUDGET_WHAT = {"data": "a new trainset", "train": "more training", "compare": "the comparison"}


def _over_budget(ws, step):
    """Before ① a trainset, ② and a new ③ dispatch (`step`): ask over_budget when a resource's use has reached its
    budget (Workspace.budget_exhausted), the task lists over_budget among its approvals and the round has none; None
    otherwise. The question names every exhausted resource."""
    out = ws.budget_exhausted()
    if not out or "over_budget" not in ws.task()["approvals"] or ws.has_approval("over_budget"):
        return None
    # Approved, the round goes on to its decision; then the stop check stops the run (a budget is reached).
    return _ask(step, "approval", "over_budget", None,
                "over_budget: %s; %s needs your approval to go over budget. approve: approve going over budget (the "
                "round goes on until it is decided, then the run stops because the budget is reached); stop: stop the "
                "run (as rsi stop)"
                % ("; ".join("%s used %s, which reaches the budget of %s" % (label, _num(used), _num(budget))
                             for label, used, budget in out), OVER_BUDGET_WHAT[step]),
                ASK_OPTIONS["over_budget"])


def producer_of(ws, aid):
    """The dispatch of the current round whose output directory `aid` was registered from, or None."""
    for d in ws.dispatches(rnd=ws.state["round"]):
        if any(e["id"] == aid for e in products(ws, d)):
            return d
    return None


def _train_step(ws, rnd, arts):
    ask = _over_budget(ws, "train")
    if ask:
        return ask
    why = "no candidate of this round has passed the train gate yet; %s" % _attempts_text(ws)
    ds = ws.dispatches(rnd=ws.state["round"])
    tt = [d for d in ds if d.get("module") == "train"]
    # The dispatches of this step: train's, and a data dispatch for a new trainset chosen after them.
    mine = tt + [d for d in ds if tt and d.get("module") == "data" and dispatch_purpose(d) in ("sets", "trainset")
                 and ds.index(d) > ds.index(tt[0])]
    last = max(mine, key=ds.index) if mine else None
    if last is not None:
        made = products(ws, last)
        if not made:
            return _handback_failed(ws, last, "data" if last.get("module") == "data" else "train",
                                    "rsi-%s's dispatch %s handed back but registered no artifact (%s)"
                                    % (last.get("module"), last["id"], why))
        trainsets = [e for e in made if e["kind"] == "trainset"]
        if last.get("module") == "data" and trainsets and not _quality_passed(ws, trainsets[-1]):
            return _quality_decision(ws, last, "new trainset %s failed its quality gate" % trainsets[-1]["id"])
    cands = [e["id"] for e in arts if e["kind"] == "candidate"]
    if cands:  # about the round's latest train dispatch; without one, the dispatch of the latest candidate
        return _decide(ws, TRAIN_GATE_FAILED, "train", _latest(ws, "train") or producer_of(ws, cands[-1]), why)
    if tt:
        return _handback_failed(ws, tt[-1], "train", "rsi-%s handed back and the round still has no candidate (%s)"
                                % (tt[-1].get("module"), why))
    return _dispatch("train", "train", "train", why)


def second_part_due(d, kind):
    """True when the second part of dispatch `d`'s two-part hand-back follows as a continuation of its session (a
    followup, no return): its latest launch registered the first part, an artifact of `kind` (the trial candidates
    of train's search: candidate; the judge's evaluation). The engine notes each registered artifact's launch in the
    record; a record without those notes (made before them, or by hand) counts its latest launch as the first part's unless that
    launch was itself a followup."""
    launch = int(d.get("launches") or 1)
    regs = d.get("registered")
    if regs is not None and (not regs or any("launch" in x for x in regs)):
        return any(x.get("kind") == kind and int(x.get("launch") or 0) == launch for x in regs)
    return d.get("launch_mode") != "followup"


def search_owed(ws, arts):
    """The trial candidates whose search is still owed: those the round's train dispatches (purpose train)
    registered after the round's latest search, while attempts remain (a search beyond them would be refused);
    [] otherwise. Candidates registered by hand (no dispatch) owe nothing."""
    if len(ws.attempts()) >= ws.attempt_limit():
        return []
    ds = [d for d in ws.dispatches(rnd=ws.state["round"], module="train") if dispatch_purpose(d) == "train"]
    mine = set(e["id"] for d in ds for e in products(ws, d))
    last = max([i for i, e in enumerate(arts) if e["kind"] == "search"] + [-1])
    return [e["id"] for e in arts[last + 1:] if e["kind"] == "candidate" and e["id"] in mine]


def _search_step(ws, trials):
    """The second part of train's search: the search over the registered trials, in the same session (a followup
    once after the launch that registered them; asked again, a return)."""
    d = _latest(ws, "train", ("train",))
    feedback = "the trial candidates are registered as %s: write the search with these ids and hand it back" \
        % ", ".join(trials)
    if second_part_due(d, "candidate"):
        return _act("resume", "train", dispatch_id=d["id"], mode="followup", feedback=feedback)
    if may_return(ws, "train"):
        return _act("resume", "train", dispatch_id=d["id"], mode="resume", feedback=feedback)
    return _handback_failed(ws, d, "train", "rsi-train has not handed back the round's search yet (trial candidates "
                            "%s registered)" % ", ".join(trials))


def _refresh(ws, step):
    """Before ③, ④ and ⑤: with the evalset's uses at the limit, ask refresh_sets, then data for the new evalset."""
    s, task = ws.state, ws.task()
    if s["eval_use_count"] < task["eval_use_limit"]:
        return None
    if not ws.has_approval("refresh_sets"):
        stop, why = ws.stop_check_if_closed()
        return _ask(step, "approval", "refresh_sets", None,
                    "refresh_sets: the evalset has been used %s, reaching its limit of %d; nothing more is evaluated "
                    "before a new evalset (it needs your refresh_sets approval). approve: approve the refresh "
                    "(rsi-data is dispatched next to produce a new evalset, then %s follow); "
                    "close_round: no refresh, close the round (no evaluation, recorded as keep), %s; stop: stop the "
                    "run (as rsi stop)"
                    % (plural(s["eval_use_count"], "time"), task["eval_use_limit"],
                       "the comparison, delivery and evaluation" if step == "compare" else "delivery and evaluation",
                       "then the run stops (%s)" % why if stop else "then rsi-judge writes the plan"),
                    ASK_OPTIONS["refresh_sets"])
    d = _latest(ws, "data", ("refresh",))
    if d is not None and not any(e["kind"] == "evalset" for e in products(ws, d)):
        return _handback_failed(ws, d, "refresh", "rsi-data's refresh dispatch %s handed back and there is still no "
                                "new evalset" % d["id"])
    return _dispatch("refresh", "data", "refresh", "the evalset has been used %s, reaching its limit of %d, and the "
                     "refresh is approved: produce a new evalset"
                     % (plural(s["eval_use_count"], "time"), task["eval_use_limit"]))


def _compare_step(ws, arts):
    """③: dispatch compare for the round's chosen candidate; its hand-back comes in two parts, the baselines and then
    the comparison, in the same session (second_part_due: a followup once after the launch that registered the
    baselines; asked again, a return); anything else is a failed hand-back (decided like any other: skip at the
    cap)."""
    d = _latest(ws, "compare", ("compare",))
    if d is None:  # a new dispatch: the budget first (its second part and its decisions are not held up)
        ask = _over_budget(ws, "compare")
        if ask:
            return ask
        cand = chosen_candidate(ws)
        return _dispatch("compare", "compare", "compare", "compare the round's candidate %s with the baselines on the "
                         "valset (report only, never a gate)" % cand, candidate_id=cand)
    made = products(ws, d)
    bases = [e["id"] for e in made if e["kind"] == "baseline"]
    if bases and not any(e["kind"] == "comparison" for e in made):
        feedback = "Baselines registered as %s: write the comparison and hand it back." % ", ".join(bases)
        if second_part_due(d, "baseline"):
            return _act("resume", "compare", dispatch_id=d["id"], mode="followup", feedback=feedback)
        if may_return(ws, "compare"):
            return _act("resume", "compare", dispatch_id=d["id"], mode="resume", feedback=feedback)
        return _handback_failed(ws, d, "compare", "rsi-compare handed back without a comparison (baselines %s "
                                "registered)" % ", ".join(bases))
    return _handback_failed(ws, d, "compare", "rsi-compare handed back without a comparison")


def last_evalset_use(ws):
    """The note that the next evaluation is the current evalset's last use (uses one below the task's
    eval_use_limit), or "" (the manual flow told the user so before ④)."""
    used, limit = ws.state["eval_use_count"], ws.task()["eval_use_limit"]
    if limit >= 1 and used == limit - 1:
        return "; this is the current evalset's last evaluation (uses %d / %d): after it the evalset must be " \
            "refreshed before anything more is evaluated" % (used, limit)
    return ""


def _deliver_step(ws, arts):
    own = own_delivery(ws, arts)
    owed = [] if own else ["deliver the round's candidate for evaluation"]
    if redeliver_pending(ws):
        owed.append("re-deliver the current best version %s on the new evalset" % ws.state["best"])
    d = _latest(ws, "deliver", ("deliver",))
    cur = ws.current("evalset")
    spent = d is not None and any(e["kind"] == "delivery" and ws.manifest(e["id"])["evalset_id"] != cur
                                  for e in products(ws, d))
    if d is not None and not spent:
        return _handback_failed(ws, d, "deliver", "rsi-deliver handed back but still has to %s" % " and ".join(owed))
    return _dispatch("deliver", "deliver", "deliver", "%s%s" % (" and ".join(owed), last_evalset_use(ws)),
                     candidate_id=None if own else chosen_candidate(ws))


def _judge_step(ws):
    d = _latest(ws, "judge", ("judge",))
    if d is not None:
        return _handback_failed(ws, d, "judge", "rsi-judge handed back and the round still has no registered "
                                "evaluation")
    return _dispatch("judge", "judge", "judge", "the round's delivery is registered: evaluate it on the evalset%s"
                     % last_evalset_use(ws))


def _decided(ws, rnd):
    stop, why = ws.stop_check()
    if stop:
        ms = milestone_state(ws)
        if ms == "ask":
            comp = milestone_comparison(ws)
            if comp is None:
                uses = ("twice (deliver runs the predictions, judge scores them, each with its own use_testset "
                        "approval)")
            else:  # milestone_compare_due: rsi-compare scores the comparison's baselines in between (or skips some)
                names = ws.manifest(comp)["baselines"]
                n = len(names)
                gpu = (", and its GPU baselines' predictions may use GPU hours beyond the compute budget"
                       if any(x in GPU_BASELINES for x in names) else "")
                uses = ("up to %d times, each with its own use_testset approval quoting this answer (deliver runs the "
                        "predictions; rsi-compare scores %s of the best version's comparison %s on the testset inputs, "
                        "report only, one approval per baseline it registers%s; judge scores them)"
                        % (n + 2, plural(n, "baseline"), comp, gpu))
            return _ask("decided", "question", MILESTONE, None,
                        "Milestone release: verify the current best version %s on the sealed testset? yes uses the "
                        "testset %s; no ends the run (rsi next stops)" % (ws.state["best"], uses), MILESTONE_OPTIONS)
        if ms == "yes" and not milestone_evaluations(ws):
            return _milestone_step(ws)
        return _act("next", "decided", reason=why)
    plan = rnd["plan_id"]
    if plan is None:
        return _plan_step(ws, rnd)
    if "plan" in ws.task()["approvals"] and not ws.has_approval("plan", plan):
        return _plan_revision(ws, rnd, plan) or _ask(
            "plan", "approval", "plan", plan, "Approve plan %s: approve records a plan approval and moves on to the "
            "next round; revise sends your words back to rsi-judge to revise the plan (counts as a return)" % plan,
            ASK_OPTIONS["plan"])
    return _act("next", "decided", reason="plan %s %s: moving on to round %d (starting from %s)"
                % (plan, "approved" if plan_approved(ws, plan) else "registered", ws.state["round"] + 1,
                   ws.manifest(plan)["entry_module"]))


def _plan_revision(ws, rnd, plan):
    """The user's latest answer on `plan` this round said revise and it is not carried out yet: back to the
    judge session that wrote the plan (its launches still as at the answer), or, for a plan without such a
    dispatch, a new plan dispatch (no dispatch made since the answer). None otherwise (the plan is asked again)."""
    answers = [d for d in ws.decisions(rnd=ws.state["round"], point="plan") if d.get("ref") == plan]
    if not answers or answers[-1].get("choice") != "revise":
        return None
    a = answers[-1]
    feedback = ("The user asked for changes to plan %s (%s): %s\nThe round was decided as %s. Rewrite the next "
                "round's plan as the user says (plan, in a new subdirectory) and hand it back."
                % (plan, a.get("by"), a.get("quote") or a.get("reason") or "", decision_text(rnd["decision"])))
    did = a.get("dispatch_id")
    if did:
        d = ws.dispatch(did)
        if int(d.get("launches") or 1) == int(a.get("launches") or 1) and d.get("status") in ("done", "failed"):
            return _act("resume", "write_plan", dispatch_id=did, mode="resume", feedback=feedback)
        return None
    if len(ws.dispatches()) == int(a.get("dispatches") or 0):
        return _dispatch("write_plan", "judge", "plan", feedback)
    return None


def _milestone_step(ws):
    best = ws.state["best"]
    if not milestone_deliveries(ws):
        d = _latest(ws, "deliver", ("milestone_deliver",))
        if d is not None:
            return _handback_failed(ws, d, "milestone", "rsi-deliver has not handed back the milestone delivery "
                                    "(testset predictions) yet")
        return _dispatch("milestone", "deliver", "milestone_deliver", "milestone release: run the predictions of the "
                         "current best version %s on the testset inputs" % best,
                         candidate_id=ws.version(best)["candidate_id"])
    if milestone_compare_due(ws):
        d = _latest(ws, "compare", ("milestone_compare",))
        if d is not None:
            return _handback_failed(ws, d, "milestone", "rsi-compare has not handed back the baselines scored on the "
                                    "testset inputs yet")
        return _dispatch("milestone", "compare", "milestone_compare", "milestone release: score the baselines of %s "
                         "(the comparison of the best version %s) on the testset inputs (report only)"
                         % (milestone_comparison(ws), best), candidate_id=ws.version(best)["candidate_id"])
    d = _latest(ws, "judge", ("milestone_judge",))
    if d is not None:
        return _handback_failed(ws, d, "milestone", "rsi-judge has not handed back the milestone evaluation yet")
    return _dispatch("milestone", "judge", "milestone_judge", "milestone release: score the milestone delivery with "
                     "the testset labels")


def _plan_step(ws, rnd):
    text = decision_text(rnd["decision"])
    d = _latest(ws, "judge", ("judge", "plan"))
    if d is None:
        return _dispatch("write_plan", "judge", "plan", "round decided as %s: write the next round's plan" % text)
    if dispatch_purpose(d) == "plan" or not products(ws, d):  # a plan dispatch that brought no plan
        return _handback_failed(ws, d, "write_plan", "rsi-judge has not handed back the round's plan yet")
    feedback = "round decided as %s (%s): write the next round's plan (plan) and a record, then hand them back" % (
        text, rnd["decision"].get("reason"))
    if second_part_due(d, "evaluation"):
        return _act("resume", "write_plan", dispatch_id=d["id"], mode="followup", feedback=feedback)
    if may_return(ws, "judge"):
        return _act("resume", "write_plan", dispatch_id=d["id"], mode="resume", feedback=feedback)
    return _handback_failed(ws, d, "write_plan", "rsi-judge has not handed back the round's plan yet")
