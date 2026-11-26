"""Decision points: where the run makes a choice, how the choice is made, and how it is carried out.

DECISION_POINTS (the decision table of docs/plan.md; options from rsi.flow.DECISION_OPTIONS, and for a compare
dispatch also skip: rsi.flow.decision_options):

  train_gate_failed    train, data, ask         no candidate of the round passed the train gate, attempts left
  quality_gate_failed  resume_data, ask         the round's (new) trainset failed its quality gate
  handback_refused     resume, (skip,) ask      registration refused what a module's latest launch handed back
  handback_failed      resume, (skip,) ask      a module's hand-back did not complete its step

default_choice(ws, point, context) is the code's choice and why (rsi.flow puts it in the decide action):
train_gate_failed: back to train (its session of the round: a new search, the next attempt); quality_gate_failed:
back to the data dispatch (resume_data); handback_refused and handback_failed: back to the same session (resume). Every
return is capped per module and round by the task's flow.max_returns (rsi.flow.returns_used); a choice the cap
does not allow (allowed()) becomes ask, as does quality_gate_failed with no dispatch record to send back to,
a long job stopped at flow.max_continues (the dispatch's "capped"), and train_gate_failed when train's latest launch
registered only experiment records (records_only: trsi said STOP). For a compare dispatch it becomes skip instead,
capped or not: ③ Compare is report only and never stops the run to ask (skip_compare closes the dispatch and the flow
goes on without it, to ④, or at the milestone to its judge). The candidate reports carry no structured
quality flag, so the table's "data when the failures are quality related" is left to an agent or the user.

choose(engine, ws, action): the code's default, or with the task's decisions.<point> "agent" a decision agent's
choice: the runner's decide() gets a prompt (the situation, the options the caps allow, the code's default, the
facts) and a JSON schema {choice, reason}; the code checks the answer and falls back to its default when the
runner has no decision agent or the answer is not one of the allowed options with a reason. With only "ask"
allowed no agent is called. The agent's session cost (total_cost_usd in its log) goes to the agent account of the
spend ledger once per decision key, and on the decision_agent (or decision_agent_failed) event as cost_usd.

apply(engine, ws, action) is the engine's decide: it records the decision once per decision_key (a rerun after an
interruption, and the user's answer to the question, reuse the latest decision with that key) and carries it
out (carry_out), first reading STATE again: when the run was stopped meanwhile (rsi stop from another process,
e.g. while a decision agent thought) nothing is launched, the failed dispatch is closed and the engine goes on to
the report. resume / resume_data send the dispatch back to its session with the reason; skip closes a compare
dispatch as skipped (skip_compare), which rsi.flow.compare_skipped reads from the decision; train and data go
to that module (rsi.engine.Engine.send_to, purpose flow.CHOICE_PURPOSE); new_session closes the dispatch and
makes a fresh one of its module for the step's current part (fresh_purpose: the judge's plan once the round is
decided, else the dispatch's purpose); close_round and stop are what `rsi close-round` and `rsi stop` do (each in
one transaction with closing the failed dispatch: close_dispatch only collects the long jobs to end, and
terminate_after_commit ends them once the transaction is saved; close_round also closes the round's other failed
dispatches whose long job may still run, as `rsi close-round` does: close_round); ask
sets STATE waiting to a question whose options are the exits (exit_options): resume, skip (a compare dispatch),
new_session, close_round (while the round can be closed), stop, and for train_gate_failed train and data; each said
in words of what follows (close_round: the judge's plan, or the stop when closing the round meets a stop condition:
Workspace.stop_check_if_closed). Every exit changes the state, so the flow never meets the same question again
unanswered; a question that no longer applies (its dispatch closed or resumed, the round decided) is stale
(rsi.flow.stale_waiting) and dropped.

answer(ws, option, by, quote) is `rsi answer`: the option must be one of STATE waiting's options; it records the
answer as a decision (by the user, with the quote) and clears waiting, in one transaction. What needs no agent
happens at once: an approval (approve), stop, close_round, skip (the long jobs of the dispatches they close end
after the save: terminate_after_commit). What needs an agent is carried out by the next
`rsi run`: at a decision point the flow meets the same decide, whose key the answer carries; a plan's revise is
read by rsi.flow. With ref (`--ref`, the waiting's id) the answer is refused when another question waits now. A
milestone yes (or, from the manual flow, a use_testset approval given after the round's decision) is the consent
the engine quotes in the two use_testset approvals, each written right before the registration that uses it
(milestone_consent). An answer to a module's questions goes back to the asking dispatch's session with its next
launch (the hand-back waited unregistered: rsi.flow gives a resume with mode answer; question_answers) or, when
that dispatch was closed first, into the brief of the next dispatch of the same module and purpose in any round
(pending_answers), once: it records where it went (delivered_to). The judge's plan brief lists all the round's
answers (round_answers). The answer to a long job's check-in question (point checkin, rsi.flow.CHECKIN_OPTIONS) records
the dispatch: continue and wait are read by rsi.flow, stop stops the run. approve(ws, ...) is
`rsi approve`, which also answers a matching approval question. Both write their run.jsonl line inside the
transaction that saves the answer. An answer to an over_budget question asked for a held hand-back
(waiting dispatch_id) records that dispatch.
"""
import os

from rsi import flow as F
from rsi.report import _clause
from rsi.rules import flow_settings, is_number
from rsi.timing import plural
from rsi.workspace import RsiError, now_iso, parse_utc

POINT_TEXT = {F.TRAIN_GATE_FAILED: "the train gate failed",
              F.QUALITY_GATE_FAILED: "the trainset failed its quality gate",
              F.HANDBACK_REFUSED: "registration refused what was handed back",
              F.HANDBACK_FAILED: "the module's hand-back did not complete the step"}
DECISION_POINTS = dict((p, {"options": F.DECISION_OPTIONS[p], "text": POINT_TEXT[p]}) for p in F.DECISION_OPTIONS)
MODES = ("code", "agent")  # task decisions.<point>; "code" when absent
EXITS = ("resume", "new_session", "close_round", "stop")
RESUME_CHOICES = ("resume", "resume_data")
GATE_LINES = 5  # failing candidates of the round named in a train_gate_failed feedback (the latest ones)


# ---- the code's default and the caps ---------------------------------------------------------------

def _dispatch(ws, did):
    return ws.dispatch(did) if did else None


def _may(ws, module):
    """A module may get the round's work once more: not dispatched in the round yet, or under its return cap."""
    return not ws.dispatches(rnd=ws.state["round"], module=module) or F.may_return(ws, module)


def allowed(ws, point, did):
    """The options the caps allow at `point` for dispatch `did` (ask always)."""
    d = _dispatch(ws, did)
    if point == F.TRAIN_GATE_FAILED:
        out = [m for m in ("train", "data") if _may(ws, m)]
    elif point == F.QUALITY_GATE_FAILED:
        out = ["resume_data"] if d is not None and F.may_return(ws, "data") else []
    else:
        out = ["resume"] if d is not None and not d.get("capped") and F.may_return(ws, d.get("module")) else []
        if F.is_compare(d):
            out.append(F.SKIP)
    return out + ["ask"]


def after_skip(d):
    """Where the flow goes once compare dispatch `d` is skipped: ④ Deliver, or at the milestone its judge."""
    return "the milestone judge" if F.dispatch_purpose(d) == "milestone_compare" else "④ Deliver"


def _returns(ws, module):
    return "rsi-%s was sent back %s this round (cap flow.max_returns = %d)" % (
        module, plural(F.returns_used(ws, module), "time"), flow_settings(ws.task())["max_returns"])


def records_only(d):
    """The launch number when dispatch `d`'s latest launch registered experiment records (`record`) and nothing
    else (e.g. trsi said STOP: its search is used up), else None."""
    n = int(d.get("launches") or 1)
    kinds = [x.get("kind") for x in d.get("registered") or [] if int(x.get("launch") or 0) == n]
    return n if kinds and all(k == "record" for k in kinds) else None


def default_choice(ws, point, context):
    """(choice, reason): the code's choice at decision `point`; context {"dispatch_id": the dispatch concerned
    or None}. Reads the workspace only."""
    did = (context or {}).get("dispatch_id")
    d = _dispatch(ws, did)
    ok = allowed(ws, point, did)
    if point == F.TRAIN_GATE_FAILED:
        n = records_only(d) if d is not None and d.get("module") == "train" else None
        if n is not None:  # another return would only hand back a record again, until the cap
            return "ask", ("rsi-train (%s) registered only experiment records in its latest launch (launch %d), no new "
                           "trial candidates or search (e.g. trsi said STOP: the configs of this search are used up): "
                           "another return would only do the same, so ask the user" % (d["id"], n))
        if "train" in ok:
            return "train", "back to rsi-train for another search in the search space; %s" % _returns(ws, "train")
        return "ask", "%s, at the cap: ask the user" % _returns(ws, "train")
    if point == F.QUALITY_GATE_FAILED:
        if d is None:
            return "ask", "no rsi-data dispatch record to send back to (a workspace run by hand): ask the user"
        if "resume_data" in ok:
            return "resume_data", "%s: send it back to the same rsi-data session to redo the trainset" % _returns(
                ws, "data")
        return "ask", "%s, at the cap: ask the user" % _returns(ws, "data")
    if d is None:
        return "ask", "no dispatch record to send back to: ask the user"
    if d.get("capped"):
        continued = "the long job's session was continued %s, reaching the cap flow.max_continues = %d" % (
            plural(int(d.get("continued") or 0), "time"), flow_settings(ws.task())["max_continues"])
        if F.SKIP in ok:  # ③ never blocks the run: no question for the user
            return F.SKIP, "%s: ③ Compare is report only, so skip it and go on to %s" % (continued, after_skip(d))
        return "ask", "%s: whether to continue it again is up to the user" % continued
    if "resume" in ok:
        return "resume", "%s: %s" % (_returns(ws, d.get("module")), "send it back to the same session with the "
                                     "registration's error text to redo it" if point == F.HANDBACK_REFUSED
                                     else "send it back to the same session to redo it")
    if F.SKIP in ok:
        return F.SKIP, "%s, at the cap: ③ Compare is report only, so skip it and go on to %s" % (
            _returns(ws, d.get("module")), after_skip(d))
    return "ask", "%s, at the cap: ask the user" % _returns(ws, d.get("module"))


# ---- code or agent ---------------------------------------------------------------------------------

def mode(ws, point):
    """How the task decides at `point`: "code" (default) or "agent"."""
    return (ws.task().get("decisions") or {}).get(point, "code")


def choice_schema(options):
    """The JSON schema a decision agent's answer follows."""
    return {"type": "object", "properties": {"choice": {"type": "string", "enum": list(options)},
                                             "reason": {"type": "string", "minLength": 1}},
            "required": ["choice", "reason"], "additionalProperties": False}


def check_choice(out, options):
    """Why a decision agent's answer cannot be used ([] when it can)."""
    if not isinstance(out, dict):
        return ["the agent's output is not a JSON object"]
    errors = []
    if out.get("choice") not in options:
        errors.append("choice %r is not one of the options (%s)" % (out.get("choice"), ", ".join(options)))
    if not isinstance(out.get("reason"), str) or not out["reason"].strip():
        errors.append("reason is empty")
    return errors


def choose(engine, ws, action):
    """(choice, by, reason) at the decide `action`: the code's default, or a decision agent's checked choice when
    the task says decisions.<point>: "agent" (the code's default when the agent gives no usable answer)."""
    point, did = action["point"], action.get("dispatch_id")
    default, why = default_choice(ws, point, {"dispatch_id": did})
    situation = action.get("reason") or ""
    code_reason = "%s; %s" % (situation, why) if situation else why
    if mode(ws, point) != "agent":
        return default, "code", code_reason
    ok = allowed(ws, point, did)
    if ok == ["ask"]:
        return "ask", "code", code_reason  # nothing for an agent to choose
    choice, reason, error, usd = agent_choice(engine, ws, action, ok, default, why)
    if error:
        engine.log("decision_agent_failed", point=point, dispatch_id=did, error=error, cost_usd=usd)
        return default, "code", "%s (the agent gave no usable choice: %s; the code's default is used)" % (
            code_reason, error)
    return choice, "agent", reason


def _choice_text(ws, option, d):
    if option == F.SKIP:
        return "skip ③ Compare this time: close %s and go on to %s without it (report only)" % (d["id"], after_skip(d))
    if option in F.CHOICE_PURPOSE:
        return "hand it to rsi-%s (back to its session of this round with purpose %s if there is one, else a new " \
            "dispatch)" % (option, F.CHOICE_PURPOSE[option])
    if option in RESUME_CHOICES and d is not None:
        return "send it back to the same rsi-%s session (%s) to redo it" % (d.get("module"), d["id"])
    return "ask the user"


def _facts(engine, ws, action, d):
    lines = []
    if d is not None:
        lines.append("- Dispatch %s: rsi-%s, purpose %s, status %s; %s" % (
            d["id"], d.get("module"), F.dispatch_purpose(d), d.get("status"), _returns(ws, d.get("module"))))
        lines += ["- Error: %s" % e for e in d.get("errors") or []]
    ids = [e["id"] for e in F.products(ws, d)] if d is not None else []
    ids += [a for a in failing(ws, action["point"]) if a not in ids]
    lines += engine._gates(ws, ids)
    if d is not None:
        notes = engine._last_notes(engine.outdir(d), d)
        if notes:
            lines.append("- Notes of the last hand-back: %s" % notes)
    return lines


def decision_prompt(engine, ws, action, options, default, why):
    """What a decision agent is told."""
    point, d = action["point"], _dispatch(ws, action.get("dispatch_id"))
    lines = ["# RSI decision point: %s (%s)" % (point, POINT_TEXT[point]), "",
             "You are the decision agent of an RSI run: choose one of the options below and give your reason. The "
             "workspace `%s` is read-only: do not change any file." % ws.root, "", "## Situation", "",
             action.get("reason") or "", "", "## Options", ""]
    lines += ["- `%s`: %s" % (o, _choice_text(ws, o, d)) for o in options]
    lines += ["", "The code's default choice: `%s` (%s)." % (default, why), ""]
    facts = _facts(engine, ws, action, d)
    if facts:
        lines += ["## Relevant facts", ""] + facts + [""]
    lines += ["## Output", "", 'Output only one JSON object: {"choice": <one of the options above>, "reason": <your '
              'reason>}.']
    return "\n".join(lines) + "\n"


def session_cost(r):
    """What a decision agent's session cost: total_cost_usd of the result object in its log (r["log"]), read with
    the claude runner's parser (rsi.runners.claude.parse_output); None when there is none."""
    from rsi.runners.claude import parse_output  # rsi.runners imports rsi.briefs, which imports rsi.flow
    usd = parse_output(r.get("log"))[1].get("total_cost_usd") if r.get("log") else None
    return usd if is_number(usd) and usd >= 0 else None


def agent_choice(engine, ws, action, options, default, why):
    """(choice, reason, None, cost) from the runner's decision agent, or (None, None, why it gave none, cost). Its
    session's cost (session_cost; None when the log reports none) goes to the agent account once per decision key
    (Workspace.add_decision_cost), used or not. The prompt and the agent's files go to
    <ws>/work/decisions/<n>-<point>/."""
    point = action["point"]
    key = decision_key(ws, action)
    workdir = os.path.join(ws.root, "work", "decisions", "%04d-%s" % (len(ws.decisions()) + 1, point))
    os.makedirs(workdir, exist_ok=True)
    prompt = decision_prompt(engine, ws, action, options, default, why)
    with open(os.path.join(workdir, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(prompt)
    request = {"point": point, "options": list(options), "default": default, "prompt": prompt,
               "schema": choice_schema(options), "dir": workdir}
    try:
        try:
            r = engine.runner.decide(ws, request, cancel=engine.stopped)
        except TypeError as e:
            if "cancel" not in str(e):
                raise
            r = engine.runner.decide(ws, request)  # a runner whose decide takes no cancel
    except Exception as e:  # an agent never stops the run: the code's default takes over
        r = {"output": None, "error": "%s: %s" % (type(e).__name__, e)}
    if r is None:
        return None, None, "runner %s has no decision agent" % engine.runner.name, None
    usd = session_cost(r)
    if usd is not None:
        ws.add_decision_cost(key, usd)
    errors = check_choice(r.get("output"), options)
    if errors:
        errors += [x for x in (r.get("error"), "output in %s" % r["log"] if r.get("log") else None) if x]
        return None, None, "; ".join(errors), usd
    out = r["output"]
    engine.log("decision_agent", point=point, choice=out["choice"], reason=out["reason"], log=r.get("log"),
               cost_usd=usd)
    return out["choice"], out["reason"].strip(), None, usd


# ---- recording and carrying out ----------------------------------------------------------------------

def decision_key(ws, action):
    """What identifies one decision: the round, the point, the dispatch and its launch, how many dispatches and
    artifacts the round has. A rerun after an interruption meets the same key and reuses the recorded decision,
    and so does the next run after the user answered the question it asked; any progress since (a launch, a
    dispatch, an artifact) makes a new one."""
    did = action.get("dispatch_id")
    launches = ws.dispatch(did).get("launches") if did else None
    rnd = ws.state["round"]
    return "%d|%s|%s|%s|%d|%d" % (rnd, action["point"], did, launches, len(ws.dispatches()),
                                  len(ws.state["rounds"][str(rnd)]["artifacts"]))


def apply(engine, ws, action):
    """The engine's decide: the latest decision recorded with this action's decision_key (an interrupted run's,
    or the user's answer to the question that decision asked), else a new one (choose); then carry_out.
    Returns (result, (status, reason) to stop, or None)."""
    key = decision_key(ws, action)
    same = [d for d in ws.decisions() if d.get("key") == key]
    if same:
        record = same[-1]
        engine.log("decision_reused", key=key, choice=record.get("choice"), by=record.get("by"))
    else:
        choice, by, reason = choose(engine, ws, action)
        record = ws.add_decision({"point": action["point"], "options": list(action.get("options") or []),
                                  "choice": choice, "by": by, "reason": reason,
                                  "dispatch_id": action.get("dispatch_id"), "key": key})
    engine.log("decision", point=record["point"], choice=record["choice"], by=record.get("by"),
               reason=record.get("reason"), quote=record.get("quote"), dispatch_id=record.get("dispatch_id"),
               reused=bool(same))
    out, stop = carry_out(engine, ws, action, record)
    out["decision"] = record
    return out, stop


def failing(ws, point):
    """The round's candidates that failed the train gate (the latest GATE_LINES) for a train_gate_failed
    feedback; () at other points."""
    if point != F.TRAIN_GATE_FAILED:
        return ()
    ids = [a for a in ws.state["rounds"][str(ws.state["round"])]["artifacts"]
           if ws.entry(a)["kind"] == "candidate" and ws.manifest(a)["train_gate"]["passed"] is not True]
    return ids[-GATE_LINES:]


def _why(action, record):
    """What the receiving session is told about the decision."""
    situation = action.get("reason") or ""
    lead = "%s. " % _clause(situation) if situation else ""
    if record.get("quote"):
        return "%sThe user (%s) chose %s, in their words: %s" % (lead, record.get("by"), record["choice"],
                                                                 record["quote"])
    if record.get("by") == "agent":
        return "%sThe decision agent chose %s: %s" % (lead, record["choice"], record.get("reason"))
    return record.get("reason") or situation


def close_dispatch(ws, d, why, context):
    """A failed dispatch that will not be resumed (stop, new_session, close_round): status done, with why (its errors
    stay; capped goes). When a long job of it may still run (rsi.engine.long_job_open: its marker is not there, e.g. a
    check-in failed, or a long-job dispatch's return) the same save records long_job_stop {"result": "pending",
    "context": `context`, "at"} and [its id] is returned, else []: the caller ends the job with
    terminate_after_commit once the transaction this ran in is saved (a transaction that fails leaves the job
    running; a termination cut off after the save is finished by the next one, or at the stop), so the done reason
    and REPORT.md list it (R16)."""
    if d is None:
        return []
    d = ws.dispatch(d["id"])  # as STATE is now (inside a transaction: as reloaded under its lock)
    if d.get("status") != "failed":
        return []
    from rsi import engine as E  # rsi.engine imports this module
    ws.count_spend(d)  # what its spend file gained since it failed (e.g. its long job)
    extra = {}
    if E.long_job_open(ws, d):
        extra["long_job_stop"] = {"result": "pending", "context": context, "at": now_iso()}
    ws.update_dispatch(d["id"], status="done", closed=why, capped=None, ended_at=d.get("ended_at") or now_iso(),
                       **extra)
    return [d["id"]] if extra else []


def close_failed_jobs(ws, why):
    """close_dispatch (context close) for every failed dispatch of the current round whose long job may still run
    (before the round is closed, F16); returns their ids for terminate_after_commit."""
    from rsi import engine as E  # rsi.engine imports this module
    out = []
    for d in ws.dispatches(rnd=ws.state["round"]):
        if d.get("status") == "failed" and E.long_job_open(ws, d):
            out += close_dispatch(ws, d, why, "close")
    return out


def terminate_after_commit(ws, dids, context):
    """After the transaction that closed dispatches `dids` (close_dispatch) is saved, outside any transaction: end each
    one's long job that may still run, and any other long_job_stop still "pending" (a termination cut off after its
    save), whatever the dispatch's status (rsi.engine.end_long_job: terminate_long_job, its marker's pid file and
    the orphan pid files, each confirmed by long_job_identity; long_job_stop and the note in `closed` in a save of
    its own; an error is recorded, and the others still go on). `context`: "stop" (a stop answered or chosen) or
    "close" (a new_session or close_round); a pending one keeps its own."""
    from rsi import engine as E  # rsi.engine imports this module
    todo = list(dids) + [d["id"] for d in E.pending_long_job_stops(ws) if d["id"] not in dids]
    for did in todo:
        d = ws.dispatch(did)
        E.end_long_job(ws, d, (d.get("long_job_stop") or {}).get("context") or context)


def close_round(ws, reason):
    """`rsi close-round <ws> --reason` and the engine's close_round: in one transaction the round's failed dispatches
    whose long job may still run are closed (close_failed_jobs) and the round is closed (Workspace.close_round, its
    refusals included: then nothing is saved and no job ended); then those jobs are ended (terminate_after_commit,
    context close). Returns Workspace.close_round's ("keep", None, reason)."""
    with ws.transaction():
        dids = close_failed_jobs(ws, "round closed (%s): this dispatch is not continued" % reason)
        out = ws.close_round(reason)
    terminate_after_commit(ws, dids, "close")
    return out


def fresh_purpose(ws, step, d):
    """The purpose of a fresh dispatch taking over from dispatch `d` at `step` (the decide action's step): the
    step's current part: the judge's plan when the round is decided (the second part of a judge dispatch); the
    trainset when round 1's testset, evalset and valset are in (what is left of the four sets); else d's purpose
    (train's search is the second part of its own purpose, whose brief lists the registered trials). Without d
    (quality_gate_failed of a workspace run by hand) data for what the round still needs."""
    purpose = F.dispatch_purpose(d) if d is not None else "sets"
    decided = ws.state["rounds"][str(ws.state["round"])]["decision"] is not None
    if d is not None and d.get("module") == "judge" and purpose == "judge" and (step == "write_plan" or decided):
        return "plan"
    if purpose == "sets" and (ws.state["round"] != 1 or all(ws.current(k) for k in ("testset", "evalset", "valset"))):
        return "trainset"
    return purpose


def _words(record):
    return record.get("quote") or record.get("reason") or record.get("choice")


def close_round_now(ws, d, record):
    """close_round: what `rsi close-round` does, quoting the user, in one transaction: the failed dispatch `d` and
    the round's other failed dispatches whose long job may still run are closed, then the round. Returns (result,
    the ids whose long jobs terminate_after_commit ends, context close, once the transaction is saved)."""
    why = "%s chose close_round" % record.get("by")
    with ws.transaction():
        dids = close_dispatch(ws, d, why, "close")
        dids += close_failed_jobs(ws, why)
        if ws.state["rounds"][str(ws.state["round"])]["decision"] is None:
            ws.close_round("%s chose to close the round: %s" % (record.get("by"), _words(record)))
    return {"closed_round": ws.state["round"]}, dids


SKIPPED = "skipped: ③ Compare is report only"


def _close_skipped(ws, d):
    """Inside a transaction: close compare dispatch `d` as skipped (status done, closed SKIPPED; a failed one through
    close_dispatch, which also notes a long job that may still run). Returns the ids for terminate_after_commit."""
    dids = close_dispatch(ws, d, SKIPPED, "close")
    cur = ws.dispatch(d["id"])
    if not cur.get("closed"):  # a done dispatch whose step is incomplete (the question at the cap)
        ws.update_dispatch(d["id"], status="done", closed=SKIPPED, capped=None,
                           ended_at=cur.get("ended_at") or now_iso())
    return dids


def skip_compare(ws, d, record):
    """skip: close compare dispatch `d` with status done and closed SKIPPED, in one transaction, then end a long job
    of it that may still run (terminate_after_commit, context close). The flow goes on without ③ (or without the
    milestone's compare): rsi.flow.compare_skipped reads the recorded decision. Returns the result."""
    with ws.transaction():
        dids = _close_skipped(ws, d)
    terminate_after_commit(ws, dids, "close")
    return {"skipped": d["id"], "by": record.get("by")}


def stop_now(ws, d, record):
    """stop: what `rsi stop` does, with the user's words, in one transaction with closing the failed dispatch.
    Returns (result, the ids for terminate_after_commit, context stop)."""
    with ws.transaction():
        dids = close_dispatch(ws, d, "%s chose stop" % record.get("by"), "stop")
        if not ws.state["stopped"]:
            ws.stop(record.get("by") or "user", _words(record))
    return {"stopped": ws.state["stopped"]["reason"]}, dids


def new_session(engine, ws, action, d, why, by=None):
    """new_session: close dispatch `d` and make a fresh dispatch of its module for the step's current part
    (fresh_purpose; quality_gate_failed without a record: data for the round's sets or trainset), told why and
    what went wrong in the old one."""
    fresh = {"module": d.get("module") if d is not None else "data",
             "purpose": fresh_purpose(ws, action.get("step"), d)}
    if d is not None:
        if "candidate_id" in d:
            fresh["candidate_id"] = d["candidate_id"]
        if d.get("errors"):
            why += "\n\nWhat went wrong in the previous session (%s):\n%s" % (
                d["id"], "\n".join("- %s" % e for e in d["errors"]))
        terminate_after_commit(ws, close_dispatch(ws, d, "%s chose new_session: %s is not continued, a new session "
                                                  "takes over" % (by or "user", d["id"]), "close"), "close")
    fresh["reason"] = why
    out = engine.dispatch(ws, [fresh], why=(why, failing(ws, action["point"])))
    if d is not None:
        out["closed"] = d["id"]
    return out


def carry_out(engine, ws, action, record):
    """Carry out the recorded choice at the decide `action` (see the module docstring)."""
    point, did = action["point"], action.get("dispatch_id")
    d = _dispatch(ws, did)
    ws = engine.halt_if_stopped(d)  # STATE as it is now: a stop recorded meanwhile launches nothing
    choice, d = record["choice"], _dispatch(ws, did)
    why = _why(action, record)
    if choice in RESUME_CHOICES and d is not None:
        return dict(engine.resume(ws, d, why)), None
    if choice == F.SKIP and d is not None:
        return skip_compare(ws, d, record), None
    if choice in F.CHOICE_PURPOSE:
        return dict(engine.send_to(ws, choice, F.CHOICE_PURPOSE[choice], did, why, failing(ws, point))), None
    if choice == "new_session":
        return new_session(engine, ws, action, d, why, record.get("by")), None
    if choice in ("close_round", "stop"):
        out, dids = (close_round_now if choice == "close_round" else stop_now)(ws, d, record)
        terminate_after_commit(ws, dids, "close" if choice == "close_round" else "stop")
        return out, None
    w = ws.waiting()
    if not (w and (w.get("what"), w.get("ref")) == (point, did)):
        w = ws.set_waiting(question(ws, action, record))
    return {"waiting": w}, ("waiting_user", ask_reason(w))


# ---- questions and answers ---------------------------------------------------------------------------

def closable(ws):
    """True while the current round can be closed (`rsi close-round`): undecided, no evaluation, not stopped."""
    return (not ws.state["stopped"] and ws.state["rounds"][str(ws.state["round"])]["decision"] is None
            and not ws.evaluations())


def exit_options(ws, point, d):
    """The options of the question a decision asks the user, each an action that changes the state."""
    opts = ["train", "data"] if point == F.TRAIN_GATE_FAILED else (["resume"] if d is not None else [])
    if F.is_compare(d):
        opts.append(F.SKIP)
    if d is not None or point == F.QUALITY_GATE_FAILED:
        opts.append("new_session")
    if closable(ws):
        opts.append("close_round")
    return opts + ["stop"]


def _exit_text(ws, option, d, step=None):
    if option == "resume":
        return "resume: send your words back to the same rsi-%s session (%s) to redo it" % (d.get("module"), d["id"])
    if option == F.SKIP:
        return "skip: close %s and go on to %s without it (③ Compare is report only, never a gate)" % (
            d["id"], after_skip(d))
    if option == "new_session":
        return "new_session: %sdispatch a new rsi-%s session (purpose %s) with your words" % (
            "close %s and " % d["id"] if d is not None and d.get("status") == "failed" else "",
            d.get("module") if d is not None else "data", fresh_purpose(ws, step, d))
    if option in F.CHOICE_PURPOSE:
        return "%s: hand your words to rsi-%s (back to its session of this round with purpose %s if there is one, " \
            "else a new dispatch)" % (option, option, F.CHOICE_PURPOSE[option])
    if option == "close_round":  # the plan follows, unless closing the round meets a stop condition
        stop, why = ws.stop_check_if_closed()
        if stop:
            return "close_round: close round %d (no evaluation, recorded as keep), then the run stops (%s)" % (
                ws.state["round"], why)
        return "close_round: close round %d (no evaluation, recorded as keep), then rsi-judge writes the plan" \
            % ws.state["round"]
    return "stop: stop the run (as rsi stop, with your words recorded)"


def question(ws, action, record):
    """The STATE waiting a decision that asks the user sets: the situation, why it asks, what went wrong, and
    the exits, each said in words."""
    point, did = action["point"], action.get("dispatch_id")
    d = _dispatch(ws, did)
    opts = exit_options(ws, point, d)
    why = ("decision agent: %s" % record.get("reason")) if record.get("by") == "agent" else \
        default_choice(ws, point, {"dispatch_id": did})[1]
    errors = [_clause(e.splitlines()[0]) for e in (d.get("errors") or [])[:3] if e] if d is not None else []
    text = "%s: %s%s. %s. Choose one: %s" % (
        _upper(POINT_TEXT[point]), _clause(action.get("reason") or ""), " (%s)" % "; ".join(errors) if errors else "",
        _upper(_clause(why)), "; ".join(_exit_text(ws, o, d, action.get("step")) for o in opts))
    return {"kind": "question", "what": point, "ref": did, "question": text, "options": opts}


def _upper(text):
    """`text` with its first letter capitalised (it starts a sentence), unless it starts with a module's name
    (rsi-data, ...), which stays as it is."""
    return text if text.startswith("rsi-") else text[:1].upper() + text[1:]


def ask_reason(w):
    text = "Waiting for the user to answer%s: %s (options: %s). Run rsi run again after the answer." % (
        " question %s" % w["id"] if w.get("id") else "", _clause(w.get("question")),
        " / ".join(w.get("options") or []))
    return " ".join(text.split())


def milestone_consent(ws):
    """The user's yes to the milestone release this round: the latest milestone decision when it says yes; with
    no milestone decision (the manual flow) the first use_testset approval of the round given at or after its
    decision (as rsi.flow.milestone_state reads it), as {"by", "quote", "at"}; else None."""
    answers = ws.decisions(rnd=ws.state["round"], point=F.MILESTONE)
    if answers:
        return answers[-1] if answers[-1].get("choice") == "yes" else None
    decision = ws.state["rounds"][str(ws.state["round"])]["decision"]
    at = parse_utc((decision or {}).get("at"))
    for a in ws.state["approvals"]:
        if a.get("what") == "use_testset" and a.get("round") == ws.state["round"] and decision is not None:
            when = parse_utc(a.get("at"))
            if at is None or (when is not None and when >= at):
                return {"by": a.get("by"), "quote": a.get("quote"), "at": a.get("at")}
    return None


def question_answers(ws, d):
    """The user's answers to questions dispatch `d` handed back that go into its next launch's feedback: those no
    launch or brief received yet, and those to the questions of its latest launch's hand-back even when already
    marked delivered (a run cut off after marking them and before recording the new launch sends them again)."""
    launch = int(d.get("launches") or 1)
    return [x for x in ws.decisions(point="handback") if x.get("dispatch_id") == d["id"]
            and (not x.get("delivered_to") or int(x.get("launches") or 0) == launch)]


def pending_answers(ws, d):
    """The user's answers to questions other dispatches of d's module and purpose handed back, in any round, that
    no launch or brief received yet (e.g. the asking dispatch was closed before its answer went back): they go
    into d's brief."""
    out = []
    for x in ws.decisions(point="handback"):
        did = x.get("dispatch_id")
        if not did or did == d["id"] or x.get("delivered_to"):
            continue
        try:
            other = ws.dispatch(did)
        except RsiError:
            continue
        if other.get("module") == d.get("module") and F.dispatch_purpose(other) == F.dispatch_purpose(d):
            out.append(x)
    return out


def round_answers(ws, rnd):
    """Every answer to a module's questions in round `rnd` (the judge's plan brief lists them all)."""
    return [x for x in ws.decisions(rnd=rnd, point="handback") if x.get("dispatch_id")]


def mark_delivered(ws, answers, to):
    """Record that `answers` were given to `to` (a dispatch's launch or brief): they are not given again."""
    qids = [x["question_id"] for x in answers if x.get("question_id")]
    if qids:
        ws.mark_delivered(qids, to)


def _log(ws, event, **fields):
    from rsi import engine  # rsi.engine imports this module
    engine.log_event(ws.root, event, **fields)


def answer(ws, option, by, quote, ref=None):
    """`rsi answer`: record the user's answer to STATE waiting (see the module docstring) and return the
    decision record. Refused, with nothing changed, when nothing waits, `ref` is given and is not the id of the
    question waiting now, the question is stale (rsi.flow.stale_waiting), or `option` is not one of its
    options."""
    errors = []
    if not by or not by.strip():
        errors.append("--by: the name of the person who answers is required")
    if not quote or not quote.strip():
        errors.append("--quote: the user's own words are required")
    if errors:
        raise RsiError(errors)
    logged, ends = [], []
    try:
        record = _answer(ws, option, by, quote, ref, logged, ends)
    except BaseException as e:
        if logged:  # the line was written, the save did not happen
            _log(ws, "answer_failed", ref=ref, choice=option, error="%s: %s" % (type(e).__name__, e))
        raise
    for dids, context in ends:  # the answer is saved: now the long jobs of the dispatches it closed end
        terminate_after_commit(ws, dids, context)
    return record


def _answer(ws, option, by, quote, qid, logged, ends):
    """The transaction of answer; `ends` gets (dispatch ids, context) for terminate_after_commit after the save."""
    with ws.transaction():
        w = ws.waiting()
        if not w:
            raise RsiError("nothing is waiting for an answer in %s (STATE waiting is empty); `rsi run` says what the "
                           "run needs next" % ws.root)
        if qid and qid != w.get("id"):
            raise RsiError("--ref %s is not the question waiting now (%s, on %s): it was answered or replaced; read "
                           "the question again (rsi run, the progress page) and answer that one"
                           % (qid, w.get("id") or "no id", w.get("what") or "a question"))
        if F.stale_waiting(ws, w):
            raise RsiError("the question %s on %s no longer applies (already answered, or what it asks about has "
                           "changed): run `rsi run` to go on" % (w.get("id") or "", w.get("what")))
        options = list(w.get("options") or [])
        if option not in options:
            raise RsiError("%r is not an option of the question on %s: answer one of %s"
                           % (option, w.get("what") or "it", ", ".join(options)))
        what, ref, kind = w.get("what"), w.get("ref"), w.get("kind")
        rec = {"point": what or "question", "options": options, "choice": option, "by": by, "quote": quote,
               "reason": "the user's answer", "kind": kind, "ref": ref, "question": w.get("question"),
               "question_id": w.get("id")}
        effect = "rsi run goes on"
        if kind == "approval":
            if w.get("dispatch_id"):  # a hand-back held for this approval (over_budget)
                rec["dispatch_id"] = w["dispatch_id"]
            if option == "approve":
                ws.approve(what, by, quote, ref)
                effect = "approved %s%s" % (what, " " + ref if ref else "")
                if w.get("dispatch_id"):
                    effect += "; rsi run registers what %s handed back" % w["dispatch_id"]
            elif option == "stop":
                ws.stop(by, quote)
                effect = "the run is stopped"
            elif option == "close_round":
                ends.append((close_round_now(ws, None, rec)[1], "close"))
                effect = "round %d is closed" % ws.state["round"]
            elif option == "revise":
                d = F.producer_of(ws, ref) if ref else None
                rec.update(dispatch_id=d["id"] if d else None, launches=int(d.get("launches") or 1) if d else None,
                           dispatches=len(ws.dispatches()))
                effect = "rsi run sends %s back to rsi-judge" % ref
        elif what in DECISION_POINTS:
            d = _dispatch(ws, ref)
            rec.update(dispatch_id=ref, key=decision_key(ws, {"point": what, "dispatch_id": ref}))
            if option == "close_round":
                ends.append((close_round_now(ws, d, rec)[1], "close"))
                effect = "round %d is closed" % ws.state["round"]
            elif option == "stop":
                ends.append((stop_now(ws, d, rec)[1], "stop"))
                effect = "the run is stopped"
            elif option == F.SKIP and d is not None:
                ends.append((_close_skipped(ws, d), "close"))
                effect = "%s is closed as skipped; rsi run goes on without it" % ref
            else:
                effect = "rsi run carries out %s" % option
        elif what == F.CHECKIN and ref:
            rec.update(dispatch_id=ref, checkins=int(ws.dispatch(ref).get("checkins") or 0))
            if option == "stop":
                ends.append((stop_now(ws, None, rec)[1], "stop"))
                effect = "the run is stopped"
            else:
                effect = {"continue": "rsi run checks in on %s again" % ref,
                          "wait": "rsi run waits for the marker of %s without check-ins" % ref}.get(option, effect)
        elif what == "handback" and ref:
            rec.update(dispatch_id=ref, launches=int(ws.dispatch(ref).get("launches") or 1))
            effect = ("rsi run sends the answer back to rsi-%s's session (%s); what it handed back is registered "
                      "once it hands back again" % (ws.dispatch(ref).get("module"), ref))
        record = ws.add_decision(rec)
        ws.clear_waiting()
        _log(ws, "answer", point=record["point"], choice=option, by=by, quote=quote, ref=ref, question_id=w.get("id"),
             dispatch_id=record.get("dispatch_id"), effect=effect)
        logged.append(1)  # written under the lock, right before the save
    return dict(record, effect=effect)


def approve(ws, what, by, quote, ref=None):
    """`rsi approve`: Workspace.approve, which also clears a STATE waiting that asks for this approval; that
    answer is recorded as a decision too. Returns the approval."""
    logged = []
    try:
        with ws.transaction():
            w = ws.waiting()
            a = ws.approve(what, by, quote, ref)
            answered = w is not None and ws.waiting() is None
            if answered:
                rec = {"point": what, "choice": "approve", "by": by, "quote": quote, "reason": "rsi approve",
                       "kind": "approval", "ref": ref, "question": w.get("question"), "question_id": w.get("id")}
                if w.get("dispatch_id"):  # a hand-back held for this approval (over_budget)
                    rec["dispatch_id"] = w["dispatch_id"]
                ws.add_decision(rec)
            _log(ws, "approve", what=what, ref=ref, by=by, quote=quote, answered=answered)
            logged.append(1)  # written under the lock, right before the save
    except BaseException as e:
        if logged:
            _log(ws, "approve_failed", what=what, ref=ref, error="%s: %s" % (type(e).__name__, e))
        raise
    return a
