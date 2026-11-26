"""The progress pages: where the run is, what it waits for and what happened so far.

PAGES are the overview <ws>/dashboard.html and three secondary pages under <ws>/dashboard/
(rounds, events, run report); rsi.dashboard_pages renders them from the derivations here
(module_progress, waiting_for, now_lead, pending_questions, flow_states; now_lead is also the status line's text
when no dispatch is open, rsi.live.now_line). What the run waits for, the question
the run asks and the next step of the workflow diagram all come from rsi.flow.next_action, the
function the engine follows, so a page never says something else than the engine does. Rendering
reads only STATE.json, registry.jsonl, registered manifests (task.json for a task), <ws>/REPORT.md,
the `## Questions for the user` section of <ws>/work/notes.md, the engine's log <ws>/work/run.jsonl (events page),
the pid in <ws>/work/run.lock (whether an `rsi run` works on the workspace now: run_active), the directory listing
of the current round's work/round-<n>/ and the manifest.json of each unregistered hand-back there (for its kind),
and for the overview's live panels the progress.json in each open dispatch's output directory (rsi.progress), the
directory listing of every dispatch's output directory with the handback.json and handback-<n>.json in it
(rsi.live.handback_notes) and <ws>/work/gpus.json (rsi.gpus.read_snapshot); it opens no data row file, runs no command
(nvidia-smi only runs in the engine's refresh) and writes nothing. write_dashboard writes each page like
Workspace.save_state (temp file in the page's directory, then os.replace), so a browser never reads half a page. The
pages reload themselves every REFRESH_SECONDS seconds.

Module activity (module_progress) has two sources. Once the engine has recorded a dispatch (STATE
dispatches) the records are the truth: a module runs while one of its dispatches in the round is
running (or is a long job whose marker is not there yet), is pending while its dispatch record is written
but its agent not launched yet, and waits to be registered while one is handed back; directories
without a record are not looked at. A workspace without any dispatch
record is run by hand (the manual flow): the orchestrator gives each module the output directory
<ws>/work/round-<n>/<module>/, with a suffix when the module is dispatched again in the round
(train-2; an old workspace's tune/ is ②'s, see stage_of). Inside it each artifact is a subdirectory holding a
manifest.json. A directory from which nothing was handed back yet (no artifact subdirectory, no registered source
inside it) means the module is running; an artifact subdirectory that is no registered artifact's source is handed back
but not registered; otherwise the module is done (an evalset or testset leaves an empty directory
behind once registration has moved it away). In the manual flow a module running or waiting to be
registered comes first in what the run waits for, before next_action's step (next_action reads no
directory).

The stages are ① Data → ② Train & tune → ③ Compare → ④ Deliver → ⑤ Judge & decide (MODULES, STEP_NAMES). ③ is on
only when the task has a compare section (rsi.flow.compare_enabled); otherwise the pages draw it as ③ Compare (off)
(stage_title) and the flow goes from ② to ④. An old workspace's tune
dispatches, artifacts and records (merged into the train module) are read as they are and shown under ②: stage_of
(rsi.flow's, re-exported here) is the one place the pages map a dispatch's module or an artifact's producer to its stage
(rsi.report.RECORD_SECTIONS groups the report's records the same way; a test ties the two together). A dispatch keeps
its own module name where it is shown (rsi-tune in its line).
"""
import datetime
import itertools
import json
import os
import tempfile

from rsi import flow as F
from rsi.flow import next_action, stage_of
from rsi.report import _clause, _num  # noqa: F401  (the pages and their tests format numbers with D._num)
from rsi.report import long_job_text
from rsi.runners.base import pid_alive
from rsi.timing import plural
from rsi.workspace import DATASETS, RsiError, parse_utc as _utc

# The pages, relative to the workspace root (forward slashes): the overview first.
PAGES = ("dashboard.html", "dashboard/rounds.html", "dashboard/events.html", "dashboard/report.html")
DASHBOARD_FILE = PAGES[0]
PAGES_DIR = "dashboard"
REFRESH_SECONDS = 5
REPORT_FILE = F.REPORT_FILE
NOTES_FILE = os.path.join("work", "notes.md")
RUN_LOG = os.path.join("work", "run.jsonl")
RUN_LOCK = os.path.join("work", "run.lock")  # rsi.engine.RUN_LOCK: held by `rsi run` for its whole life
RUN_LOG_TAIL = 2 * 1024 * 1024
NOTES_SECTION = "## Questions for the user"

# The pages' nodes: the stages of the main flow, which are the flow's modules.
MODULES = F.MODULES
STEP_INDEX = dict(F.STEP_INDEX)
STEP_NAMES = {"data": "① Data", "train": "② Train & tune", "compare": "③ Compare", "deliver": "④ Deliver",
              "judge": "⑤ Judge & decide"}
OFF = " (off)"  # after ③'s name while the task has no compare section (stage_title)
# Where the run is for each main-flow step key.
STEP_WHERE = dict(STEP_NAMES, refresh=STEP_NAMES["data"])
AFTER_STEPS = max(STEP_INDEX.values()) + 1  # the position of a round past ⑤ (waiting for the top-level check)
BUSY = ("running", "pending", "unregistered")  # module_progress states of a module whose output is not in yet


def _stages(names):
    """The stages of `names`, in order, each once; a name without a stage is skipped."""
    out = []
    for n in names:
        s = stage_of(n)
        if s is not None and s not in out:
            out.append(s)
    return out


def _skill(module):
    return "rsi-" + module


def stage_title(ws, stage):
    """A stage's name on the pages: STEP_NAMES, with OFF after ③ while the task has no compare section."""
    off = stage == "compare" and not F.compare_enabled(ws.task())
    return STEP_NAMES[stage] + (OFF if off else "")


# ---- module activity -----------------------------------------------------------------------------

def _module_of(name):
    """The stage whose output directory `name` is: a module name, or a module name followed by a character
    that is not a letter (train-2, train_2, train2; not trainer; an old workspace's tune/ is ②'s); else None."""
    return stage_of("".join(itertools.takewhile(str.isalpha, name)))


def _stage_dispatches(ws, stage, rnd=None):
    """The dispatch records whose module is of `stage` (of round `rnd` when given), in order."""
    return [d for d in ws.dispatches(rnd=rnd) if stage_of(d.get("module")) == stage]


def _subdirs(path):
    """Names of the visible subdirectories of `path`, sorted. Hidden ones (for example a hand-over
    leftover .<name>.rsi-handed-over-<hex>) are no module output."""
    return sorted(n for n in os.listdir(path) if not n.startswith(".") and os.path.isdir(os.path.join(path, n)))


def _handed_dirs(outdir):
    """The subdirectories of a dispatch's output directory its hand-back names: the `artifacts` of its handback.json
    that are directories there (a task fill task/ too), sorted; without a readable list, the subdirectories holding a
    manifest.json or a task.json. Drafts the module keeps there (e.g. rsi-data's redo/ of a resumed sets) are not named."""
    try:
        with open(os.path.join(outdir, "handback.json"), encoding="utf-8") as f:
            names = json.load(f)["artifacts"]
        if not (isinstance(names, list) and all(isinstance(n, str) and n and os.sep not in n for n in names)):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        names = [n for n in _subdirs(outdir)
                 if any(os.path.isfile(os.path.join(outdir, n, x)) for x in ("manifest.json", "task.json"))]
    return sorted(n for n in set(names) if not n.startswith(".") and os.path.isdir(os.path.join(outdir, n)))


def _inside(child, parent):
    return os.path.commonpath([child, parent]) == parent


def _iso(mtime):
    """A file time as local ISO 8601 with its UTC offset, to the second."""
    return datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def module_progress(ws):
    """Each stage's activity in the current round: stage (MODULES) -> {"state", "since", "pending", "superseded"}.

    state is "running", "pending" (a dispatch record written, its agent not launched yet; only with
    dispatch records), "unregistered" (handed back, waiting to be registered), "done" (it has a
    dispatch or an output directory and nothing outstanding) or "idle"; since is when the earliest
    running dispatch or directory started (local ISO) while running, else None; pending lists the
    names of the unregistered artifact subdirectories, sorted. With dispatch records (see the module
    docstring) they come from the records (_dispatch_progress), else from the directories
    (_dir_progress), which alone fill superseded."""
    if ws.dispatches():
        return _dispatch_progress(ws)
    return _dir_progress(ws)


def _started(ws, d):
    """When dispatch `d` started, as local ISO: its started_at, else its output directory's modification
    time, else now."""
    iso = _utc_iso(d.get("started_at"))
    if iso is not None:
        return iso
    try:
        return _iso(os.path.getmtime(os.path.join(ws.root, d.get("outdir") or "")))
    except OSError:
        return _iso(datetime.datetime.now().timestamp())


def _dispatch_progress(ws):
    """module_progress from the dispatch records: running while a dispatch of the module (of any round:
    the run waits for it all the same) runs or waits for its long job's marker, else pending while one is
    not launched yet, unregistered while one is handed back (or its marker is there; pending: the subdirectories of
    its output directory its hand-back names, _handed_dirs, that are no registered source), done once the module has a
    dispatch in the current round, else idle. A failed dispatch is no running module. A stage holds the dispatches
    of its modules (stage_of)."""
    sources = set(os.path.realpath(e["source"]) for e in ws.entries())
    out = {}
    for m in MODULES:
        ds = _stage_dispatches(ws, m)
        running, queued, handed, pending = [], [], [], []
        for d in ds:
            status = d.get("status")
            if status == "running" or (status == "long_job" and not F.marker_done(ws, d)):
                running.append(d)
            elif status == "pending":
                queued.append(d)
            elif status == "handed_back" or status == "long_job":
                handed.append(d)
                base = os.path.join(ws.root, d.get("outdir") or "")
                if d.get("outdir") and os.path.isdir(base):
                    pending += [n for n in _handed_dirs(base) if os.path.realpath(os.path.join(base, n)) not in sources]
        if running:
            state = "running"
            since = min((_started(ws, d) for d in running), key=datetime.datetime.fromisoformat)
        elif queued:
            state, since = "pending", None
        else:
            here = _stage_dispatches(ws, m, ws.state["round"])
            state, since = ("unregistered" if handed else "done" if here else "idle"), None
        out[m] = {"state": state, "since": since, "pending": sorted(pending), "superseded": {}}
    return out


def running_agents(ws):
    """The agents the engine has running: [{"id", "module", "purpose", "runner", "session", "started",
    "launches", "returns"}] in dispatch order, for the dispatches with status running (started: local ISO).
    Empty without dispatch records."""
    return _agent_rows(ws, lambda d: d.get("status") == "running")


def waiting_long_jobs(ws):
    """Like running_agents, for the long_job dispatches whose completion marker is not there yet (the agent
    has handed back and exited: the run waits for the marker, no agent runs)."""
    return _agent_rows(ws, lambda d: d.get("status") == "long_job" and not F.marker_done(ws, d))


def pending_agents(ws):
    """Like running_agents, for the dispatch records whose agent is not launched yet (status pending)."""
    return _agent_rows(ws, lambda d: d.get("status") == "pending")


def run_active(ws):
    """True while an `rsi run` works on the workspace: <ws>/work/run.lock (rsi.engine holds it for the run's whole
    life and removes it at the end) names a live process. A run that was killed leaves the file behind with its
    dead pid: no run. Only the top bar's "● running" depends on it."""
    try:
        with open(os.path.join(ws.root, RUN_LOCK), encoding="utf-8", errors="replace") as f:
            pid = int(f.read().strip() or 0)
    except (OSError, ValueError):
        return False
    return pid > 1 and pid_alive(pid)


def idle_label(ws):
    """The top bar's word while no `rsi run` works and nothing is asked: what the run waits for (a long job's
    marker, a manual hand-back) or, for anything else, the next `rsi run`."""
    a = next_action(ws)
    if a["type"] == "wait_long_job":
        return "waiting for the long job"
    ds = _action_dispatches(ws, a) if a["type"] == "wait_agent" else []
    if ds and all(d.get("status") == "running" and d.get("runner") == "manual" for d in ds):
        return "waiting for a manual hand-back"
    return "waiting for rsi run"


def stop_leftovers(ws):
    """After a stop, before the report step of `rsi run` has settled the records: (agents still running, long jobs
    it still has to end) as dispatch records. A long job is one of an open record that may still run
    (rsi.engine.long_job_open: a long job, a check-in working or failed while its job runs on) or one whose close
    has not terminated it yet (long_job_stop pending). Records never launched (pending) are only closed."""
    from rsi.engine import long_job_open  # rsi.engine imports this module (through rsi.live)
    agents = [d for d in ws.dispatches() if d.get("status") == "running"]
    jobs = [d for d in ws.dispatches() if (d.get("long_job_stop") or {}).get("result") == "pending"
            or (d.get("status") != "done" and long_job_open(ws, d))]
    return agents, jobs


def _leftover_words(agents, jobs):
    """"1 agent and 2 long jobs", "" for none."""
    return " and ".join(([plural(len(agents), "agent")] if agents else [])
                        + ([plural(len(jobs), "long job")] if jobs else []))


def stopped_texts(ws, a):
    """(lead, detail, sentence) once the run has stopped: the lead and the plain detail of the now card, and what
    the run waits for as one sentence. While agents or long jobs still run: how many, and that `rsi run` ends them
    and writes the report (the one that holds run.lock is doing it already); without REPORT.md: that `rsi run`
    writes it; once done: the long jobs the stop terminated, those an earlier close terminated, and those neither could
    (rsi.report.long_job_text)."""
    reason = _clause((ws.state["stopped"] or {}).get("reason"))
    active = run_active(ws)
    agents, jobs = stop_leftovers(ws)
    words = _leftover_words(agents, jobs)
    if words:
        lead = "Stopped, %s still running" % words
        them = "it" if len(agents) + len(jobs) == 1 else "them"
        todo = ("rsi run is terminating %s and writing the report; wait for it to exit." if active
                else "Run rsi run to terminate %s and write the report.") % them
        detail = "%s Stop reason: %s." % (todo, reason)
        return lead, detail, "%s: %s" % (lead, detail)
    lead = "Stopped: %s" % reason
    if a["type"] != "done":
        todo = "rsi run is writing the run report." if active else "Run rsi run to write the run report."
        return lead, todo, "%s. %s" % (lead, todo)
    jobs_text = long_job_text(ws)
    return lead, jobs_text, "%s. The run report is in REPORT.md.%s" % (lead, " " + jobs_text if jobs_text else "")


def _agent_rows(ws, keep):
    return [{"id": d.get("id"), "module": d.get("module"), "purpose": d.get("purpose"),
             "runner": d.get("runner"), "session": d.get("session"), "started": _started(ws, d),
             "launches": d.get("launches"), "returns": int(d.get("returns") or 0)}
            for d in ws.dispatches() if keep(d)]


def run_events(ws):
    """The lines of the last RUN_LOG_TAIL bytes of <ws>/work/run.jsonl that parse as objects, in order ([] when
    the file is missing or unreadable): {"at", "event", fields...}. Only the tail is read, so rewriting the
    pages stays cheap on a long run."""
    try:
        with open(os.path.join(ws.root, RUN_LOG), "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - RUN_LOG_TAIL))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > RUN_LOG_TAIL:
        lines = lines[1:]  # the first line is cut
    out = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("event"):
            out.append(rec)
    return out


def _dir_progress(ws):
    """module_progress of the manual flow, from the directories under work/round-<n>/.

    state: "running" when one of a module's directories has handed nothing back yet (since: the earliest
    such directory's modification time, ISO); else "unregistered" when an artifact subdirectory is no
    registered artifact's source (paths compared with os.path.realpath) and not superseded; else "done"
    when it has a directory; else "idle". pending lists those subdirectory names, sorted. superseded maps
    the name of an unregistered subdirectory (a refused hand-back) to the id of the artifact of the same
    kind (its manifest's kind) this module registered in the round afterwards: that registration replaced
    it, so it no longer waits to be registered. "Afterwards" is read from the directory's mtime: the
    entry's registered_at (whole seconds) is not before the mtime's second. A refused directory is
    never touched again, so its mtime is when the module handed it back; an unreadable manifest or
    an artifact registered earlier never supersedes.
    """
    base = os.path.join(ws.root, "work", "round-%d" % ws.state["round"])
    sources = set(os.path.realpath(e["source"]) for e in ws.entries())
    dirs = dict((m, []) for m in MODULES)
    if os.path.isdir(base):
        for name in _subdirs(base):
            m = _module_of(name)
            if m is not None:
                dirs[m].append(os.path.join(base, name))
    rnd = _round(ws)
    out = {}
    for m in MODULES:
        waiting, pending, superseded = [], [], {}
        for d in dirs[m]:
            real = os.path.realpath(d)
            arts = [n for n in _subdirs(d) if os.path.isfile(os.path.join(d, n, "manifest.json"))]
            for n in arts:
                if os.path.realpath(os.path.join(d, n)) in sources:
                    continue
                later = _superseding(ws, rnd, m, os.path.join(d, n), real)
                if later is None:
                    pending.append(n)
                else:
                    superseded[n] = later
            if not arts and not any(_inside(s, real) for s in sources):
                waiting.append(os.path.getmtime(d))
        if waiting:
            state, since = "running", _iso(min(waiting))
        else:
            state, since = ("unregistered" if pending else "done" if dirs[m] else "idle"), None
        out[m] = {"state": state, "since": since, "pending": sorted(pending), "superseded": superseded}
    return out


def _superseding(ws, rnd, module, path, outdir):
    """The id of the first artifact of the manifest's kind that `module` registered in the round after
    the directory `path` was written (see _dir_progress), or None. Only an artifact registered from
    outside `outdir` (the real path of the module output directory holding `path`) counts: it came from a
    re-dispatch. A sibling registered from the same directory is part of the same hand-back."""
    try:
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            kind = json.load(f).get("kind")
        written = int(os.path.getmtime(path))
    except (OSError, ValueError, AttributeError):
        return None
    for a in rnd["artifacts"]:
        e = ws.entry(a)
        at = _utc(e.get("registered_at"))
        if stage_of(e["producer"]) == module and e["kind"] == kind and at is not None and at.timestamp() >= written \
                and os.path.dirname(os.path.realpath(e["source"])) != outdir:
            return a
    return None


# ---- what the run waits for -------------------------------------------------------------------

def _round(ws):
    s = ws.state
    return s["rounds"][str(s["round"])]


def _show(dt):
    """A datetime as local time; a naive one is taken as local already."""
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _local(ts):
    """A UTC time from STATE or the registry (2026-11-22T01:02:03Z) as local time."""
    dt = _utc(ts)
    if dt is None:
        return "none" if not ts else str(ts)
    return _show(dt)


def _since(iso):
    return _show(datetime.datetime.fromisoformat(iso))


def _close_round_text(ws):
    return ("Round %d has used up its %s and no candidate passed the train gate: rsi run closes the round (as "
            "rsi close-round does), then dispatches rsi-judge to write the plan from the round's candidates and "
            "records." % (ws.state["round"], plural(ws.attempt_limit(), "attempt")))


PASSED_SHOWN = 3  # candidate ids named in the deliver step's sentence (the rounds page lists them all)


def _passed_text(ws):
    """The round's candidates through the train gate, for the deliver step: train-produced first,
    the count and at most PASSED_SHOWN ids; "" when there is none."""
    passed = sorted(ws.passed_candidates(), key=lambda a: ws.entry(a)["producer"] != "train")
    if not passed:
        return ""
    if len(passed) == 1:
        return " (the round's candidate that passed the train gate: %s)" % passed[0]
    return " (%d candidates of the round passed the train gate: %s%s; the best by the valset primary metric goes to " \
           "evaluation)" % (len(passed), ", ".join(passed[:PASSED_SHOWN]), " and others" if len(passed) > PASSED_SHOWN
                            else "")


def _redeliver_sentence(ws, own):
    """The ④ sentence when the best version must be re-delivered on the refreshed evalset first."""
    return ("%s evalset was refreshed: first re-deliver the current best version %s on the new evalset (it has no "
            "delivery there yet)." % ("The delivery of the round's candidate is registered; the" if own else "The",
                                       ws.state["best"]))


def _key(a):
    """The page's name for the case an action stands for: stopped, running, handed, plan, question (a
    question from STATE waiting that is no approval the flow knows), promote, close, decided (next, the
    milestone question, the judge writing the plan), milestone (the release under way), or the step key
    data, refresh, train, compare, deliver, judge."""
    t, step = a["type"], a.get("step")
    if t in ("report", "done"):
        return "stopped"
    if t == "process_handback":
        return "handed"
    if t in ("wait_agent", "wait_long_job"):
        return "running"
    if t == "promote":
        return "promote"
    if t == "close_round":
        return "close"
    if t == "next":
        return "decided"
    if t == "ask":
        what = a.get("what")
        if what == "plan":
            return "plan"
        if what == F.MILESTONE:
            return "decided"
        if what in ("over_budget", "refresh_sets") and step in NEXT_STEPS:
            return step
        return "question"
    return _step_key(step)


def _step_key(step):
    return "decided" if step == "write_plan" else step


def _decide_tail(a):
    return "%s (decision point %s: options %s; code default %s)." % (
        _clause(a.get("reason")), a.get("point"), ", ".join(a.get("options") or []), a.get("default"))

def _dispatch_or_none(ws, did):
    try:
        return ws.dispatch(did)
    except RsiError:  # an id the records do not hold: shown without its module
        return None


def _action_dispatches(ws, a):
    """The dispatch records an action names (dispatch_ids or dispatch_id), looked up by id in any round."""
    ids = a.get("dispatch_ids") or ([a["dispatch_id"]] if a.get("dispatch_id") else [])
    return [d for d in (_dispatch_or_none(ws, i) for i in ids) if d]


def action_modules(ws, a):
    """The modules of the dispatches an action names, in order, each once."""
    mods = []
    for d in _action_dispatches(ws, a):
        if d.get("module") not in mods:
            mods.append(d.get("module"))
    return mods


def _skills_of(ds):
    mods = []
    for d in ds:
        if d.get("module") not in mods:
            mods.append(d.get("module"))
    return ", ".join(_skill(str(m)) for m in mods)


def answer_resume(a):
    """True for the action that sends the user's answers back to the session that asked (resume, mode answer)."""
    return a["type"] == "resume" and a.get("mode") == "answer"


def answer_going_back(ws, a):
    """True when the action sends the user's answers back to the asking session: the answer resume, or, for a long
    job's hand-back that asked (rsi.flow gives process_handback for it), its processing, which continues the session
    with the answers and the job's results."""
    if answer_resume(a):
        return True
    ds = _action_dispatches(ws, a) if a["type"] == "process_handback" else []
    return bool(ds) and F.question_answered(ws, ds[0])


def held_handbacks(ws, module):
    """The dispatches of stage `module` whose latest hand-back waits unregistered for the user's answer to its
    questions (the record's asked names its latest launch; a long job's hand-back too)."""
    return [d for d in _stage_dispatches(ws, module) if d.get("status") in ("handed_back", "long_job")
            and int((d.get("asked") or {}).get("launch") or 0) == int(d.get("launches") or 1)]


def _agents_text(ws, a):
    """(where, what) for agents running or not launched yet, a long job, a hand-back to process and the user's
    answer going back to the asking session, from the dispatch records."""
    ids = a.get("dispatch_ids") or [a.get("dispatch_id")]
    ds = _action_dispatches(ws, a)
    mods = action_modules(ws, a)
    where = " / ".join([STEP_NAMES[s] for s in _stages(mods)] + [str(m) for m in mods if stage_of(m) is None]) \
        or "module"
    skills = ", ".join(_skill(str(m)) for m in mods) or ", ".join(str(i) for i in ids)
    if answer_going_back(ws, a):
        if ds and ds[0].get("status") == "long_job":
            return where, ("The long job has ended; sending the user's answer back to the same session of %s (%s): "
                           "it writes or fixes its artifacts by the answer and hands back again; only then are they "
                           "registered." % (skills, ids[0]))
        return where, ("Sending the user's answer back to the same session of %s (%s): it changes or confirms its "
                       "hand-back by the answer and hands back again; only then is it registered." % (skills, ids[0]))
    if a["type"] == "process_handback":
        return where, "%s has handed back (%s): waiting for the handed-back artifacts to be checked and registered." % (
            skills, ids[0])
    if a["type"] == "wait_long_job":
        return where, "Waiting for the long job of %s to finish (completion marker %s has not appeared yet)." % (
            skills, a.get("marker") or "not given")
    run = [d for d in ds if d.get("status") != "pending"]
    queued = [d for d in ds if d.get("status") == "pending"]
    parts = []
    if run or not queued:
        first = min((datetime.datetime.fromisoformat(_started(ws, d)) for d in run), default=None)
        parts.append("Waiting for %s to hand back (running since %s)" % (_skills_of(run) or skills,
                                                                          _show(first) if first else "unknown"))
    if queued:
        parts.append("%s not launched yet (rsi run launches %s)" % (_skills_of(queued),
                                                                    "it" if len(queued) == 1 else "them"))
    return where, "; ".join(parts) + "."


def _step_text(ws, a, key):
    """The sentence of a main-flow step (data, refresh, train, compare, deliver, judge)."""
    where, t = STEP_WHERE[key], a["type"]
    if t == "ask" and a.get("what") == "refresh_sets":  # the new evalset (① data) comes before ③ / ④ / ⑤
        return ("Next %s: The evalset has reached its use limit. Answer refresh_sets first: once it is approved, "
                "rsi-data is dispatched to refresh the evalset (it produces a new evalset), and only then come %s; the "
                "round can also be closed or the run stopped." % (STEP_WHERE["refresh"], "the comparison, delivery "
                                                                  "and evaluation" if a.get("step") == "compare"
                                                                  else "delivery and evaluation"))
    if t == "ask" and a.get("what") == "over_budget":  # before ① a trainset, ② or ③
        return "Next %s: The user must approve going over budget (over_budget) before %s is dispatched." % (
            where, F.OVER_BUDGET_WHAT.get(key, "more work"))
    if t == "decide":
        if a.get("point") == F.TRAIN_GATE_FAILED:
            body = ("The train gate failed: go back to rsi-train for another search, or dispatch rsi-data for a new "
                    "trainset (decision point train_gate_failed, code default %s; %s)." % (a.get("default"),
                                                                                         _clause(a.get("reason"))))
        elif a.get("point") == F.QUALITY_GATE_FAILED:
            body = "%s: return it to rsi-data to redo, or ask the user (decision point quality_gate_failed, code " \
                   "default %s)." % (_clause(a.get("reason")), a.get("default"))
        else:
            body = _decide_tail(a)
        return "Next %s: %s" % (where, body)
    if key == "data":
        return "Next %s: Dispatch rsi-data (%s)." % (where, _clause(a.get("reason")))
    if key == "refresh":
        return "Next %s: Dispatch rsi-data to produce a new evalset (%s)." % (where, _clause(a.get("reason")))
    if key == "train":
        if t == "resume":  # the second part of train's search
            return "Next %s: Waiting for rsi-train to write the search (%s); delivery comes after it." % (
                where, _clause(a.get("feedback")))
        return "Next %s: Dispatch %s (%s)." % (where, _skill(a.get("module", "train")),
                                               _clause(a.get("reason")))
    if key == "compare":
        if t == "resume":  # the second part of compare's hand-back
            return "Next %s: Waiting for rsi-compare to write the comparison (%s); delivery comes after it." % (
                where, _clause(a.get("feedback")))
        return "Next %s: Dispatch rsi-compare (%s)." % (where, _clause(a.get("reason")))
    if key == "deliver":
        arts = [ws.entry(x) for x in _round(ws)["artifacts"]]
        own = F.own_delivery(ws, arts)
        head = "Dispatch rsi-deliver:" if own else "Dispatch rsi-deliver to package the candidate for evaluation%s." \
            % _passed_text(ws)
        parts = [head] + ([_redeliver_sentence(ws, own)] if F.redeliver_pending(ws) else [])
        return "Next %s: %s%s" % (where, " ".join(parts), _last_use(ws))
    return "Next %s: Dispatch rsi-judge to evaluate the round's delivery on the evalset.%s" % (where, _last_use(ws))


def _last_use(ws):
    """F.last_evalset_use's note as a sentence in brackets after the step's sentence ("" without one)."""
    note = F.last_evalset_use(ws).lstrip("; ")
    return " (%s%s.)" % (note[:1].upper(), note[1:]) if note else ""


def _decided_text(ws, a):
    """(where, what) for a decided round: the judge writing the plan, the stop condition met, or next."""
    rnd = _round(ws)
    decision = F.decision_text(rnd["decision"]) if rnd["decision"] else "(undecided)"
    if a.get("step") == "write_plan":
        if a["type"] == "decide":
            return STEP_NAMES["judge"], "Round decided as %s: %s" % (decision, _decide_tail(a))
        return STEP_NAMES["judge"], ("Round decided as %s: waiting for rsi-judge to write the round's plan; rsi run "
                                     "registers it once it is handed back." % decision)
    stop, why = ws.stop_check()
    if stop:
        return ("Stop check", "Round decided and a stop condition is met (%s): no plan is needed; rsi run ends the run "
                "next (as rsi next does)%s." % (_clause(why), "" if F.milestone_evaluations(ws) else
                                                 " (with a best version it first asks the user whether to make a "
                                                 "milestone release)"))
    plan = rnd["plan_id"]
    if plan is None:
        return "Next round", ("rsi run moves on to the next round (as rsi next does; %s)."
                              % _clause(a.get("reason")))
    return "Next round", "Plan %s %s: rsi run moves on to round %d (starting from %s; as rsi next does)." % (
        plan, "approved" if F.plan_approved(ws, plan) else "registered", ws.state["round"] + 1,
        ws.manifest(plan)["entry_module"])


def _say(ws, a, key):
    """(where, what) for an action and its key."""
    if key in ("running", "handed"):
        return _agents_text(ws, a)
    if key == "plan":
        plan = a.get("ref") or _round(ws)["plan_id"]
        return "Plan approval", ("Waiting for the user to approve plan %s: explain the plan to the user and record "
                                 "the user's answer with the rsi answer command below (approve approves it; revise has "
                                 "rsi-judge revise the plan by the user's words)." % plan)
    if key == "close":
        return "Close round", _close_round_text(ws)
    if key == "promote":
        return "Top-level check", ("The round's evaluation is registered: rsi run decides the round next (promote "
                                   "or keep, as rsi promote does; evaluation %s)." % a.get("evaluation_id"))
    if key == "question":
        return "Waiting for the user", "Waiting for the user: %s" % a.get("question")
    if key == "milestone":
        if a["type"] == "decide":
            return "Milestone release", "Milestone release in progress: %s" % _decide_tail(a)
        if a.get("module") == "deliver":
            return ("Milestone release", "Milestone release in progress: the user agreed to verify the best version "
                    "%s on the testset; waiting for rsi-deliver to run the testset predictions and hand them back for "
                    "registration (do not end the run now)." % ws.state["best"])
        if a.get("module") == "compare":
            return ("Milestone release", "Milestone release in progress: the milestone delivery is registered; waiting "
                    "for rsi-compare to score the baselines of the best version's comparison on the testset inputs "
                    "(report only; do not end the run now).")
        return "Milestone release", ("Milestone release in progress: the milestone delivery is registered; waiting "
                                     "for rsi-judge to score it on the testset and hand it back for registration (do "
                                     "not end the run now).")
    if key == "decided":
        return _decided_text(ws, a)
    return STEP_WHERE[key], _step_text(ws, a, key)


def _manual_activity(ws, progress):
    """The manual flow's running or handed-back modules (from their output directories), or None."""
    running = [m for m in MODULES if progress[m]["state"] == "running"]
    if running:
        first = min(datetime.datetime.fromisoformat(progress[m]["since"]) for m in running)
        return ("running", " / ".join(STEP_NAMES[m] for m in running), "Waiting for %s to hand back (no artifact in "
                "the output directory since %s)." % (", ".join(_skill(m) for m in running), _show(first)))
    handed = [m for m in MODULES if progress[m]["state"] == "unregistered"]
    if handed:
        return ("handed", " / ".join(STEP_NAMES[m] for m in handed), "%s: waiting for validation and spot checks, then "
                "registration (rsi register)." % "; ".join("%s handed back %s" % (_skill(m), ", ".join(
                    progress[m]["pending"])) for m in handed))
    return None


def _view(ws, progress):
    """(key, where, what, action): action is next_action(ws), where and what say it in words, key names
    the case (see _key). In the manual flow (no dispatch record) a module running or waiting to be
    registered in its output directory comes first (key running or handed), as the orchestrator
    registers hand-backs before it goes on."""
    a = next_action(ws)
    key = _key(a)
    if key == "stopped":
        return key, "Stopped", stopped_texts(ws, a)[2], a
    if not ws.dispatches():
        manual = _manual_activity(ws, progress)
        if manual:
            return manual + (a,)
    where, what = _say(ws, a, key)
    return key, where, what, a


def _status(ws, progress):
    """(key, where the run is, what it waits for), from _view."""
    return _view(ws, progress)[:3]


def _wait(ws, progress):
    """_status with (where, what) replaced by a wait for the user while the action asks the user (the
    plan wait already says so)."""
    key, where, what, a = _view(ws, progress)
    asked = _state_questions(ws, key, a)
    if asked and key != "plan":
        where = "Waiting for the user"
        what = "Waiting for the user: %s No module is dispatched before the answer." % " ".join(
            _sentence(q["text"]) for q in asked)
    return key, where, what


# What ends a sentence, full-width forms included.
SENTENCE_ENDS = (".", "?", "!", "\u3002", "\uff1f", "\uff01")


def _sentence(text):
    """`text` as a sentence: trimmed, with a full stop unless it already ends in a sentence end (SENTENCE_ENDS)."""
    text = text.strip()
    return text if text.endswith(SENTENCE_ENDS) else text + "."


def waiting_for(ws, progress):
    """What the run waits for now, in one sentence (next_action's action, see _view)."""
    return _wait(ws, progress)[2]


# ---- pauses waiting for the user --------------------------------------------------------------

def _utc_iso(ts):
    dt = _utc(ts)
    return None if dt is None else _iso(dt.timestamp())


def _any_iso(ts):
    """A time from STATE as local ISO: UTC as now_iso writes it, or any ISO 8601 time (naive: local)."""
    iso = _utc_iso(ts)
    if iso is None and isinstance(ts, str):
        try:
            iso = _iso(datetime.datetime.fromisoformat(ts).timestamp())
        except ValueError:
            return None
    return iso


def _milestone_open(ws):
    """True while the milestone question is open: the release is on the table and not answered yet."""
    return F.milestone_state(ws) == "ask"


def _milestone_running(ws):
    """True while a milestone release is under way: the user said yes, its evaluation is not registered yet."""
    return F.milestone_state(ws) == "yes" and not F.milestone_evaluations(ws)


def _notes_questions(ws):
    """The lines of the `## Questions for the user` section of <ws>/work/notes.md (a leading "- " or "* " dropped)."""
    path = os.path.join(ws.root, NOTES_FILE)
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        since = _iso(os.path.getmtime(path))
    except (OSError, ValueError):  # missing or unreadable notes: no questions, the page still renders
        return []
    out, inside = [], False
    for line in lines:
        if inside and line.startswith("## "):
            break
        if line == NOTES_SECTION:
            inside = True
        elif inside and line.strip():
            text = line.strip()
            if text[:2] in ("- ", "* "):
                text = text[2:].strip()
            if text:
                out.append({"source": "notes", "text": text, "since": since})
    return out


def _ask_since(ws, a):
    """Since when the user is asked: STATE waiting's since; for a question the flow derives, the plan's
    registration (plan), the round's decision (milestone), the evaluation that used up the evalset
    (refresh_sets); else None."""
    w = ws.waiting()
    if w:
        return _any_iso(w.get("since"))
    what = a.get("what")
    if what == "plan" and ws.is_kind(a.get("ref"), "plan"):
        return _utc_iso(ws.entry(a["ref"]).get("registered_at"))
    if what == F.MILESTONE and _round(ws)["decision"]:
        return _utc_iso(_round(ws)["decision"].get("at"))
    if what == "refresh_sets":
        used = [e for e in ws.entries("evaluation") if not ws.manifest(e["id"])["uses_testset"]]
        return _utc_iso(used[-1].get("registered_at")) if used else None
    return None


def _state_questions(ws, key, a):
    """The question next_action asks: [{"source": "state", "text", "since"}] for an ask action, [] for any
    other action and while the manual flow shows a module running or waiting to be registered."""
    if a["type"] != "ask" or key in ("running", "handed"):
        return []
    return [{"source": "state", "text": a.get("question") or "", "since": _ask_since(ws, a),
             "options": list(a.get("options") or []), "what": a.get("what"), "ref": a.get("ref")}]


def pending_questions(ws, progress):
    """The pauses where the run waits for the user: [{"source": "state" | "notes", "text", "since"}], the
    question of next_action first (plan approval, milestone release, over_budget, refresh_sets, or the
    question in STATE waiting), then the notes' lines in order. since: ISO time or None."""
    key, _where, _what, a = _view(ws, progress)
    return _state_questions(ws, key, a) + _notes_questions(ws)


# ---- the workflow diagram ---------------------------------------------------------------------

def _step_artifacts(ws, module, arts):
    """This round's registered artifacts produced by stage `module` (stage_of their producer; records and task
    versions aside: data's task fill shows under Task intake)."""
    return [e for e in arts if stage_of(e["producer"]) == module and e["kind"] not in ("record", "task")]


def _registered_status(ws, module, arts):
    """A step's status from this round's registered artifacts: "done", "fail" (its only output
    failed its gate), "partial" (registration under way: some data sets, ②'s trial candidates whose
    search is still owed, or ③'s baselines without their comparison), "skipped" (③ skipped by a decision),
    "closed" (⑤ in a round closed by `rsi close-round`) or "idle". ② is done once one of its candidates (an old
    workspace's tune trials too) passed the train gate; ③ once the round holds a comparison or the milestone's
    baselines."""
    mine = _step_artifacts(ws, module, arts)
    kinds = [e["kind"] for e in mine]
    if module == "data":
        if not any(k in DATASETS for k in kinds):
            return "idle"
        if STEP_INDEX.get(_round(ws)["entry"], 0) > 0 and "trainset" not in kinds:
            return "idle"  # a refreshed evalset or valset alone is no data step
        if F.data_missing(ws, arts) is None:
            return "done"
        trainsets = [e for e in mine if e["kind"] == "trainset"]
        if trainsets and ws.current("trainset") not in [e["id"] for e in trainsets]:
            return "fail"
        return "partial"
    if module == "train":
        cands = [e["id"] for e in mine if e["kind"] == "candidate"]
        if not cands:
            return "idle"
        passed = ws.passed_candidates()
        if any(c in passed for c in cands):
            return "done"
        # The search over trials registered without it is still owed (rsi.flow.search_owed): under way. Trials
        # that owe none (registered by hand, attempts used up) are a failed gate, as the flow's train_gate_failed.
        return "partial" if F.search_owed(ws, arts) else "fail"
    if module == "compare":
        if any(e["kind"] == "comparison" for e in mine) or F.milestone_baselines(ws):
            return "done"
        if F.compare_skipped(ws):
            return "skipped"
        return "partial" if mine else "idle"
    if module == "deliver":
        done = F.own_delivery(ws, arts) and not (_round(ws)["decision"] is None and F.redeliver_pending(ws))
        return "done" if done else "idle"
    if ws.evaluations():
        return "done"
    if ws.closed():
        return "closed"
    return "idle"


# The steps each next-step key sends the run to.
NEXT_STEPS = {"data": ("data",), "refresh": ("data",), "train": ("train",), "compare": ("compare",),
              "deliver": ("deliver",), "judge": ("judge",)}


def _node(state, text=""):
    return {"state": state, "text": text}


def _hm(dt):
    return dt.astimezone().strftime("%H:%M")


def _done_text(ws, module, mine, rnd):
    """The short status phrase of a step done in this round."""
    if module == "data":
        sets = sum(1 for e in mine if e["kind"] in DATASETS and ws.current(e["kind"]) == e["id"])
        return "✓ " + plural(sets, "set")
    if module == "train":
        return "✓ gate passed · " + plural(sum(1 for e in mine if e["kind"] == "candidate"), "trial")
    if module == "compare":
        tested = F.milestone_baselines(ws)
        parts = [plural(sum(1 for e in mine if e["kind"] == "baseline" and e["id"] not in tested), "baseline")]
        if tested:
            parts.append("milestone")
        return "✓ " + " · ".join(parts)
    if module == "deliver":
        ids = [e["id"] for e in mine if e["kind"] == "delivery"]
        return "✓ " + ids[0] if len(ids) == 1 else "✓ " + plural(len(ids), "delivery package")
    parts = []
    d = rnd["decision"]
    if ws.evaluations():
        if d is None:
            parts.append("evaluated")
        else:
            parts.append("promote %s" % d.get("version") if d.get("decision") == "promote" else "keep")
    if F.milestone_evaluations(ws):
        parts.append("milestone")  # the round that releases a milestone: "plan" is left out, the node is full
    elif rnd["plan_id"] is not None:
        parts.append("plan")
    return "✓ " + " · ".join(parts)


def flow_states(ws, progress):
    """The workflow diagram's nodes: key -> {"state", "text"} for task, the five stages (data, train, compare,
    deliver, judge), check (Top-level check) and stop (Stop & deliver). ③ is skip ("off") while the task has no compare
    section.

    state is done, current, wait (waits for the user), fail (the step's only output this round
    failed its gate), skip (not passed through this round) or todo; text is a short status phrase.
    A step is current while one of its modules runs or waits to be registered, or when next_action's
    step is it; done, fail and skip follow the round's registered artifacts; a step
    before the round's entry, or behind the step the round has reached with no output of its own,
    is skipped (with the attempts used up, ⑤ is still to come: the judge writes the plan of the
    closed round). The step's node waits instead while next_action asks for an approval that step
    needs (over_budget, refresh_sets). check waits while the user is asked to approve the plan or the
    milestone release, is current while the round waits for promote, close_round or next and done
    once the round is decided. While the notes hold questions for the user, or STATE waiting holds a
    question of another kind, every current node waits instead; a module's own question (asked with its
    hand-back) pauses that module's stage node only. A step left partial (② trials whose search is owed) is
    current only while the round is undecided. Once stopped, stop is current and the other nodes keep their
    state from the round's artifacts (check done); a module whose agent or long job still runs after the stop
    says so."""
    s = ws.state
    rnd = _round(ws)
    arts = [ws.entry(a) for a in rnd["artifacts"]]
    stopped = s["stopped"]
    key, _where, _what, action = _view(ws, progress)
    step = _step_key(action.get("step")) if key == "question" else key  # a free question pauses the step
    entry = STEP_INDEX.get(rnd["entry"], 0)
    if stopped:
        reached = [STEP_INDEX[m] for m in MODULES if _step_artifacts(ws, m, arts)]
        position, nxt, asked, writing_plan = max(reached + [-1]), (), False, False
    else:
        busy = [STEP_INDEX[m] for m in MODULES if progress[m]["state"] in BUSY]
        if busy:
            position = max(busy)
        elif step == "refresh":  # the new evalset comes before ③ (or ④, ⑤)
            position = STEP_INDEX["deliver" if F.compare_done(ws, arts) else "compare"]
        elif step in NEXT_STEPS:
            position = STEP_INDEX[step]
        elif step == "close":
            position = STEP_INDEX["judge"]  # ③ and ④ are not passed through; the judge still writes the plan
        else:
            position = AFTER_STEPS
        nxt = NEXT_STEPS.get(step, ())
        asked = bool(nxt) and key != "question" and bool(_state_questions(ws, key, action))
        writing_plan = action.get("step") == "write_plan"
    out = {"task": _node("done", "✓ done")}
    records = bool(ws.dispatches())
    answering = _stages(action_modules(ws, action)) if answer_going_back(ws, action) else []
    compare_on = F.compare_enabled(ws.task())
    for m in MODULES:
        p = progress[m]
        mine = _step_artifacts(ws, m, arts)
        status = _registered_status(ws, m, arts)
        if m == "judge" and rnd["plan_id"] is not None:
            status = "done"  # the judge's plan is in (also for a round closed without evaluation)
        if m == "compare" and not compare_on and p["state"] == "idle":
            out[m] = _node("skip", "no compare section")
        elif p["state"] == "running" and not stopped:
            out[m] = _node("current", "● running since %s" % _hm(datetime.datetime.fromisoformat(p["since"])))
        elif p["state"] == "running" and records:  # the stop has not reached its agent or long job yet
            out[m] = _node("current", "● still running after stop")
        elif p["state"] == "pending" and not stopped:
            out[m] = _node("current", "● not launched yet")
        elif p["state"] == "unregistered" and not stopped:
            out[m] = _node("current", "● sending answer back" if m in answering else "● waiting to register")
        elif m == "judge" and writing_plan:
            out[m] = _node("current", "● writing plan")
        elif m == "data" and step == "refresh":
            out[m] = _node("current", "● refreshing evalset")  # also when ① is done in this round
        elif status == "done":
            out[m] = _node("done", _done_text(ws, m, mine, rnd))
        elif status == "fail":
            out[m] = _node("fail", "gate failed · redispatch" if m in nxt else "gate failed")
        elif status == "skipped":  # ③ is report only: a decision chose skip
            out[m] = _node("skip", "skipped (report only)")
        elif p["state"] == "pending":
            out[m] = _node("skip", "not launched at stop")
        elif p["state"] == "unregistered" and records:  # handed back, not registered, and the run has stopped
            held = held_handbacks(ws, m)
            out[m] = _node("skip", ("answer not sent at stop" if all(F.question_answered(ws, d) for d in held)
                                    else "unanswered at stop") if held else "not registered")
        elif p["state"] in BUSY:
            out[m] = _node("skip", "not handed back at stop")
        elif status == "closed":
            out[m] = (_node("done", "✓ round closed · milestone") if F.milestone_evaluations(ws)
                      else _node("skip", "round closed"))
        elif status == "partial":  # e.g. ② trials whose search is owed: under way until the round is decided
            out[m] = (_node("todo", "unfinished at stop") if stopped else _node("todo", "decided, unfinished")
                      if rnd["decision"] is not None else _node("current", "● in progress"))
        elif m in nxt:
            late = m == "deliver" and F.own_delivery(ws, arts) and F.redeliver_pending(ws)
            out[m] = (_node("wait", "⏸ waiting for approval") if asked
                      else _node("current", "● re-deliver best version" if late else "● next"))
        elif STEP_INDEX[m] < entry or STEP_INDEX[m] < position:
            out[m] = _node("skip", "skipped this round")
        else:
            out[m] = _node("todo")
    if stopped:
        out["check"] = _node("done", "✓ checked")
    elif step == "plan":
        out["check"] = _node("wait", "⏸ plan needs approval")
    elif _milestone_open(ws):
        out["check"] = _node("wait", "⏸ milestone question")
    elif step == "promote":
        out["check"] = _node("current", "● waiting for rsi promote")
    elif step == "close":
        out["check"] = _node("current", "● waiting for round close")
    elif step == "milestone" or _milestone_running(ws):
        out["check"] = _node("current", "● milestone release")
    elif step == "decided" and not writing_plan:
        out["check"] = _node("current", "● waiting for rsi next")
    elif rnd["decision"] is not None:
        out["check"] = _node("done", "✓ decided")
    else:
        out["check"] = _node("todo")
    if stopped:
        at = _utc(stopped.get("at"))
        out["stop"] = _node("current", "■ stopped %s" % _hm(at) if at else "■ stopped")
    else:
        out["stop"] = _node("todo")
        notes = bool(_notes_questions(ws))
        asker = None if notes else _asking_module(ws, key, action)
        if key == "question" or notes:  # the run waits for the user before going on
            for k, node in out.items():
                if node["state"] == "current" and (asker is None or k == asker):
                    running = k in progress and progress[k]["state"] == "running"
                    out[k] = _node("wait", "⏸ running · awaiting user" if running else "⏸ waiting for the user")
            if asker is not None and out[asker]["state"] != "wait":
                out[asker] = _node("wait", "⏸ waiting for the user")
    return out


# ---- the overview's lead ----------------------------------------------------------------------

# Short phrases for the orchestrator's own commands (_status keys); a next step leads with its name.
LEADS = {"promote": "Top-level check: waiting for rsi promote to decide the round",
         "close": "Waiting for the round to be closed (rsi close-round)",
         "milestone": "Milestone release in progress: waiting for the milestone delivery and evaluation to be "
                      "registered"}
# The same with the engine's records: rsi run carries these steps out itself.
ENGINE_LEADS = dict(LEADS, promote="Top-level check: rsi run decides the round (promote or keep)",
                    close="rsi run closes the round (as rsi close-round does)")


def now_lead(ws, progress=None, questions=None, view=None):
    """What the run waits for in a few words: the title of the overview's "What the run waits for" (its detail is
    waiting_for's sentence) and the status line's text when no dispatch is open (rsi.live.now_line). `progress`
    (module_progress), `questions` (pending_questions) and `view` (_view's (key, where, what, action)) are computed
    when not given."""
    progress = module_progress(ws) if progress is None else progress
    questions = pending_questions(ws, progress) if questions is None else questions
    key, where, _what, action = _view(ws, progress) if view is None else view
    s = ws.state
    rnd = _round(ws)
    if s["stopped"]:
        return stopped_texts(ws, action)[0]
    if key == "plan":
        return "Waiting for the user to approve plan %s" % rnd["plan_id"]
    if questions:
        return ("Waiting for the user to answer the question above in Claude" if len(questions) == 1
                else "Waiting for the user to answer the %d questions above in Claude" % len(questions))
    if key in ("running", "handed"):
        state = "running" if key == "running" else "unregistered"
        if ws.dispatches():  # the engine's records: the dispatches the action names, of any round
            mods = [m for m in action_modules(ws, action) if stage_of(m)]
        else:
            mods = [m for m in MODULES if progress[m]["state"] == state]
        where = " / ".join(STEP_NAMES[m] for m in _stages(mods))
        if answer_going_back(ws, action):
            return "%s: sending the user's answer back to %s" % (where, ", ".join(_skill(m) for m in mods))
        if key == "running" and ws.dispatches():  # an agent not launched yet is no running one
            ds = _action_dispatches(ws, action)
            run = [m for m in mods if any(d.get("module") == m and d.get("status") != "pending" for d in ds)]
            queued = [m for m in mods if m not in run]
            parts = (["%s running" % ", ".join(_skill(m) for m in run)] if run else []) + (
                ["%s not launched yet" % ", ".join(_skill(m) for m in queued)] if queued else [])
            return "%s: %s" % (where, "; ".join(parts))
        return "%s: %s %s" % (where, ", ".join(_skill(m) for m in mods),
                              "running" if key == "running" else "handed back, waiting to be registered")
    if key in NEXT_STEPS:
        return "Next: %s" % where
    records = bool(ws.dispatches())  # the engine's run: rsi run does promote, close_round and next itself
    if key in LEADS:
        return (ENGINE_LEADS if records else LEADS)[key]
    if ws.stop_check()[0]:
        return ("A stop condition is met: rsi run ends the run" if records
                else "A stop condition is met: waiting for rsi next to end the run")
    if rnd["plan_id"] is None:
        return "%s: waiting for rsi-judge to write the plan" % STEP_NAMES["judge"]
    return ("rsi run moves on to round %d" if records
            else "Waiting to move on to round %d (rsi next)") % (s["round"] + 1)


def _asking_module(ws, key, action):
    """The stage of the module whose hand-back asked the question the run waits for (STATE waiting of what
    "handback": its ref is the dispatch), or None for any other question."""
    if key != "question" or action.get("what") != "handback" or not action.get("ref"):
        return None
    d = _dispatch_or_none(ws, action["ref"])
    return stage_of(d.get("module")) if d else None


# ---- writing the pages -----------------------------------------------------------------------

def _write_atomic(path, text):
    """Write `text` to `path` through a temp file in the same directory and os.replace, mode 0644.
    On failure the temp file is removed and the error raised."""
    fd, tmp = tempfile.mkstemp(prefix=".dashboard.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)  # mkstemp makes it 0600; the pages are no secret
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_dashboard(ws):
    """Write the four pages (PAGES, under <ws>/dashboard/ for the secondary ones), each atomically,
    and return the overview's absolute path. The overview is written last, so the pages it links
    to exist when it appears. Any failure is raised."""
    from rsi.dashboard_pages import render_pages
    pages = render_pages(ws)
    os.makedirs(os.path.join(ws.root, PAGES_DIR), exist_ok=True)
    for rel in reversed(PAGES):
        _write_atomic(os.path.join(ws.root, *rel.split("/")), pages[rel])
    return os.path.join(ws.root, DASHBOARD_FILE)
