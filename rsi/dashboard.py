"""The progress pages: where the run is, what it waits for and what happened so far.

PAGES are the overview <ws>/dashboard.html and three secondary pages under <ws>/dashboard/
(rounds, events, run report); rsi.dashboard_pages renders them from the derivations here
(module_progress, waiting_for, now_lead, pending_questions, flow_states; now_lead is also the status line's text
when no dispatch is open, rsi.live.now_line). What the run waits for, the question
the run asks and the next step of the workflow diagram all come from rsi.flow.next_action, the
function the engine follows, so a page never says something else than the engine does. Rendering
reads only STATE.json, registry.jsonl, registered manifests (task.json for a task), <ws>/REPORT.md,
the `## 待用户回答` section of <ws>/work/notes.md, the engine's log <ws>/work/run.jsonl (events page), the pid in
<ws>/work/run.lock (whether an `rsi run` works on the workspace now: run_active), the directory listing of the current
round's
work/round-<n>/ and the manifest.json of each unregistered hand-back there (for its kind), and for the overview's live
panels the progress.json in each open dispatch's output directory (rsi.progress), the directory listing of every
dispatch's output directory with the handback.json and handback-<n>.json in it (rsi.live.handback_notes) and
<ws>/work/gpus.json (rsi.gpus.read_snapshot); it opens no data row file, runs no command (nvidia-smi only runs in the
engine's refresh) and writes nothing. write_dashboard writes each page like Workspace.save_state
(temp file in the page's directory, then os.replace), so a browser never reads half a page. The
pages reload themselves every REFRESH_SECONDS seconds.

Module activity (module_progress) has two sources. Once the engine has recorded a dispatch (STATE
dispatches) the records are the truth: a module runs while one of its dispatches in the round is
running (or is a long job whose marker is not there yet), is pending while its dispatch record is written
but its agent not launched yet (待启动), and waits to be registered while one is handed back; directories
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

The stages are ① 数据 → ② 训练调优 → ③ 交付 → ④ 评测与决策 (MODULES, STEP_NAMES). An old workspace's tune
dispatches, artifacts and records (merged into the training-tuning module train) are read as they are and shown
under ②: stage_of (rsi.flow's, re-exported here) is the one place the pages map a dispatch's module or an
artifact's producer to its stage (rsi.report.RECORD_SECTIONS groups the report's records the same way; a test ties
the two together). A dispatch keeps its own module name where it is shown (rsi-tune in its line).
"""
import datetime
import itertools
import json
import os
import tempfile

from rsi import flow as F
from rsi.flow import next_action, stage_of
from rsi.report import _num  # noqa: F401  (the pages and their tests format numbers with D._num)
from rsi.report import long_job_text
from rsi.runners.base import pid_alive
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
NOTES_SECTION = "## 待用户回答"

# The pages' nodes: the stages of the main flow, which are the flow's modules.
MODULES = F.MODULES
STEP_INDEX = dict(F.STEP_INDEX)
STEP_NAMES = {"data": "① 数据", "train": "② 训练调优", "deliver": "③ 交付", "judge": "④ 评测与决策"}
# Where the run is for each main-flow step key.
STEP_WHERE = dict(STEP_NAMES, refresh=STEP_NAMES["data"])
AFTER_STEPS = max(STEP_INDEX.values()) + 1  # the position of a round past ④ (waiting for 顶层核对)
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

    state is "running", "pending" (a dispatch record written, its agent not launched yet: 待启动; only with
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
    """Like running_agents, for the dispatch records whose agent is not launched yet (status pending: 待启动)."""
    return _agent_rows(ws, lambda d: d.get("status") == "pending")


def run_active(ws):
    """True while an `rsi run` works on the workspace: <ws>/work/run.lock (rsi.engine holds it for the run's whole
    life and removes it at the end) names a live process. A run that was killed leaves the file behind with its
    dead pid: no run. Only the top bar's "● 运行中" depends on it."""
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
        return "等长任务完成"
    ds = _action_dispatches(ws, a) if a["type"] == "wait_agent" else []
    if ds and all(d.get("status") == "running" and d.get("runner") == "manual" for d in ds):
        return "等手动交回"
    return "等待 rsi run"


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
    """"1 个 agent 和 2 个长任务", "" for none."""
    return " 和 ".join((["%d 个 agent" % len(agents)] if agents else []) + (["%d 个长任务" % len(jobs)] if jobs else []))


def stopped_texts(ws, a):
    """(lead, detail, sentence) once the run has stopped: the lead and the plain detail of the now card, and what
    the run waits for as one sentence. While agents or long jobs still run: how many, and that `rsi run` ends them
    and writes the report (the one that holds run.lock is doing it already); without REPORT.md: that `rsi run`
    writes it; once done: the long jobs the stop terminated, those an earlier close terminated, and those neither could
    (rsi.report.long_job_text)."""
    reason = (ws.state["stopped"] or {}).get("reason")
    active = run_active(ws)
    agents, jobs = stop_leftovers(ws)
    words = _leftover_words(agents, jobs)
    if words:
        lead = "已停止，还有 %s%s在运行" % (words, " " if words[-1].isascii() else "")
        them = "它" if len(agents) + len(jobs) == 1 else "它们"
        todo = ("rsi run 正在终止%s并写报告，等它退出。" if active else "运行 rsi run 终止%s并写报告。") % them
        detail = "%s停止原因：%s。" % (todo, reason)
        return lead, detail, "%s：%s" % (lead, detail)
    lead = "已停止：%s" % reason
    if a["type"] != "done":
        todo = "rsi run 正在写运行报告。" if active else "运行 rsi run 写运行报告。"
        return lead, todo, "%s。%s" % (lead, todo)
    jobs_text = long_job_text(ws)
    return lead, jobs_text, "%s。运行报告见 REPORT.md。%s" % (lead, jobs_text)


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
        return "无" if not ts else str(ts)
    return _show(dt)


def _since(iso):
    return _show(datetime.datetime.fromisoformat(iso))


def _close_round_text(ws):
    return ("第 %d 轮的 %d 次尝试已用完，没有候选通过训练门：rsi run 关闭本轮（rsi close-round 的效果），"
            "再派 rsi-judge 依据本轮的候选和记录写计划。" % (ws.state["round"], ws.attempt_limit()))


PASSED_SHOWN = 3  # candidate ids named in the deliver step's sentence (the rounds page lists them all)


def _passed_text(ws):
    """The round's candidates through the train gate, for the deliver step: train-produced first,
    the count and at most PASSED_SHOWN ids; "" when there is none."""
    passed = sorted(ws.passed_candidates(), key=lambda a: ws.entry(a)["producer"] != "train")
    if not passed:
        return ""
    if len(passed) == 1:
        return "（本轮通过训练门的候选：%s）" % passed[0]
    return "（本轮通过训练门的候选 %d 个：%s%s；按验证集主指标选最好的送评）" % (
        len(passed), "、".join(passed[:PASSED_SHOWN]), " 等" if len(passed) > PASSED_SHOWN else "")


def _redeliver_sentence(ws, own):
    """The ③ sentence when the best version must be re-delivered on the refreshed evalset first."""
    return ("%s评测集已刷新，先把当前最佳版本 %s 在新评测集上重新交付（它在新评测集上还没有 delivery）。"
            % ("本轮候选的交付已登记，" if own else "", ws.state["best"]))


def _key(a):
    """The page's name for the case an action stands for: stopped, running, handed, plan, question (a
    question from STATE waiting that is no approval the flow knows), promote, close, decided (next, the
    milestone question, the judge writing the plan), milestone (the release under way), or the step key
    data, refresh, train, deliver, judge."""
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
    return "%s（决策点 %s：可选 %s，代码默认 %s）。" % (a.get("reason"), a.get("point"), "、".join(a.get("options") or []),
                                                a.get("default"))


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
    return "、".join(_skill(str(m)) for m in mods)


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
        or "模块"
    skills = "、".join(_skill(str(m)) for m in mods) or "、".join(str(i) for i in ids)
    if answer_going_back(ws, a):
        if ds and ds[0].get("status") == "long_job":
            return where, ("长任务已结束，把你的回答送回 %s 的同一个会话（%s）：它按回答写好或改好产物后重新交回，那时才登记。"
                           % (skills, ids[0]))
        return where, "把你的回答送回 %s 的同一个会话（%s）：它按回答修改或确认后重新交回，那时才登记。" % (skills, ids[0])
    if a["type"] == "process_handback":
        return where, "%s 已交回（%s）：等校验、登记交回的产物。" % (skills, ids[0])
    if a["type"] == "wait_long_job":
        return where, "等 %s 的长任务完成（完成标记 %s 还没有出现）。" % (skills, a.get("marker") or "未给出")
    run = [d for d in ds if d.get("status") != "pending"]
    queued = [d for d in ds if d.get("status") == "pending"]
    parts = []
    if run or not queued:
        first = min((datetime.datetime.fromisoformat(_started(ws, d)) for d in run), default=None)
        parts.append("等 %s 交回（自 %s 起运行）" % (_skills_of(run) or skills, _show(first) if first else "未知"))
    if queued:
        parts.append("%s 待启动（rsi run 启动它）" % _skills_of(queued))
    return where, "；".join(parts) + "。"


def _step_text(ws, a, key):
    """The sentence of a main-flow step (data, refresh, train, deliver, judge)."""
    where, t = STEP_WHERE[key], a["type"]
    if t == "ask" and a.get("what") == "refresh_sets":  # the new evalset (① data) comes before ③ / ④
        return ("下一步 %s：评测集已用到上限，先回答 refresh_sets——批准后派 rsi-data 刷新评测集（产出新的评测集），然后才"
                "交付和评测；也可以关闭本轮或停止。" % STEP_WHERE["refresh"])
    if t == "decide":
        if a.get("point") == F.TRAIN_GATE_FAILED:
            body = ("训练门没过，回到 rsi-train 再搜一次，或派 rsi-data 出新训练集（决策点 train_gate_failed，"
                    "代码默认 %s；%s）。" % (a.get("default"), a.get("reason")))
        elif a.get("point") == F.QUALITY_GATE_FAILED:
            body = "%s：交回 rsi-data 重做，或问你（决策点 quality_gate_failed，代码默认 %s）。" % (
                a.get("reason"), a.get("default"))
        else:
            body = _decide_tail(a)
        return "下一步 %s：%s" % (where, body)
    if key == "data":
        return "下一步 %s：派 rsi-data（%s）。" % (where, a.get("reason"))
    if key == "refresh":
        return "下一步 %s：派 rsi-data 产出新的评测集（%s）。" % (where, a.get("reason"))
    if key == "train":
        if t == "resume":  # the second part of train's search
            return "下一步 %s：等 rsi-train 写 search（%s），写好后再交付。" % (where, a.get("feedback"))
        if t == "ask":
            return "下一步 %s：继续派训练之前要你批准超预算（over_budget）。" % where
        return "下一步 %s：派 %s（%s）。" % (where, _skill(a.get("module", "train")), a.get("reason"))
    if key == "deliver":
        arts = [ws.entry(x) for x in _round(ws)["artifacts"]]
        own = F.own_delivery(ws, arts)
        head = "派 rsi-deliver：" if own else "派 rsi-deliver 打包送评的候选%s。" % _passed_text(ws)
        return "下一步 %s：%s%s%s" % (where, head, _redeliver_sentence(ws, own) if F.redeliver_pending(ws) else "",
                                   _last_use(ws))
    return "下一步 %s：派 rsi-judge 在评测集上评测本轮的交付。%s" % (where, _last_use(ws))


def _last_use(ws):
    note = F.last_evalset_use(ws)
    return "（%s。）" % note.lstrip("；") if note else ""


def _decided_text(ws, a):
    """(where, what) for a decided round: the judge writing the plan, the stop condition met, or next."""
    rnd = _round(ws)
    decision = F.decision_text(rnd["decision"]) if rnd["decision"] else "（未判定）"
    if a.get("step") == "write_plan":
        if a["type"] == "decide":
            return STEP_NAMES["judge"], "本轮已判定 %s：%s" % (decision, _decide_tail(a))
        return STEP_NAMES["judge"], "本轮已判定 %s：等 rsi-judge 写本轮的计划，交回后由 rsi run 登记。" % decision
    stop, why = ws.stop_check()
    if stop:
        return ("停止检查", "本轮已判定，停止条件已满足（%s）：不再需要计划，rsi run 接着结束运行（rsi next 的效果）%s。"
                % (why, "" if F.milestone_evaluations(ws) else "（有最佳版本时先问用户是否做里程碑发布）"))
    plan = rnd["plan_id"]
    if plan is None:
        return "推进轮次", "rsi run 接着推进轮次（rsi next 的效果；%s）。" % a.get("reason")
    return "推进轮次", "计划 %s %s：rsi run 接着进入第 %d 轮（从 %s 开始；rsi next 的效果）。" % (
        plan, "已批准" if F.plan_approved(ws, plan) else "已登记", ws.state["round"] + 1, ws.manifest(plan)["entry_module"])


def _say(ws, a, key):
    """(where, what) for an action and its key."""
    if key in ("running", "handed"):
        return _agents_text(ws, a)
    if key == "plan":
        plan = a.get("ref") or _round(ws)["plan_id"]
        return "审批计划", ("等用户批准计划 %s：把计划讲给用户，用下面的 rsi answer 命令记下用户的回答（approve 批准，"
                            "revise 让 rsi-judge 按用户的话改计划）。" % plan)
    if key == "close":
        return "关闭本轮", _close_round_text(ws)
    if key == "promote":
        return "顶层核对", "本轮评测已登记：rsi run 接着判定本轮（升级或保留，rsi promote 的效果；评测 %s）。" % a.get(
            "evaluation_id")
    if key == "question":
        return "等用户回答", "等用户回答：%s" % a.get("question")
    if key == "milestone":
        if a["type"] == "decide":
            return "里程碑发布", "里程碑发布进行中：%s" % _decide_tail(a)
        if a.get("module") == "deliver":
            return ("里程碑发布", "里程碑发布进行中：用户已同意用 test 验证最佳版本 %s，等 rsi-deliver 跑 test 预测并交回登记"
                    "（这时不要结束运行）。" % ws.state["best"])
        return "里程碑发布", "里程碑发布进行中：里程碑 delivery 已登记，等 rsi-judge 用 test 打分并交回登记（这时不要结束运行）。"
    if key == "decided":
        return _decided_text(ws, a)
    return STEP_WHERE[key], _step_text(ws, a, key)


def _manual_activity(ws, progress):
    """The manual flow's running or handed-back modules (from their output directories), or None."""
    running = [m for m in MODULES if progress[m]["state"] == "running"]
    if running:
        first = min(datetime.datetime.fromisoformat(progress[m]["since"]) for m in running)
        return ("running", " / ".join(STEP_NAMES[m] for m in running), "等 %s 交回（输出目录自 %s 起还没有交回产物）。"
                % ("、".join(_skill(m) for m in running), _show(first)))
    handed = [m for m in MODULES if progress[m]["state"] == "unregistered"]
    if handed:
        return ("handed", " / ".join(STEP_NAMES[m] for m in handed), "%s：等校验、抽查后登记（rsi register）。"
                % "；".join("%s 交回了 %s" % (_skill(m), "、".join(progress[m]["pending"])) for m in handed))
    return None


def _view(ws, progress):
    """(key, where, what, action): action is next_action(ws), where and what say it in Chinese, key names
    the case (see _key). In the manual flow (no dispatch record) a module running or waiting to be
    registered in its output directory comes first (key running or handed), as the orchestrator
    registers hand-backs before it goes on."""
    a = next_action(ws)
    key = _key(a)
    if key == "stopped":
        return key, "已停止", stopped_texts(ws, a)[2], a
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
        where = "等用户回答"
        what = "等用户回答：%s回答之前不派模块。" % "".join(q["text"].rstrip("。？?") + "。" for q in asked)
    return key, where, what


def waiting_for(ws, progress):
    """What the run waits for now, one Chinese sentence (next_action's action, see _view)."""
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
    """The lines of the `## 待用户回答` section of <ws>/work/notes.md (a leading "- " or "* " dropped)."""
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
    versions aside: data's task fill shows under 任务接入)."""
    return [e for e in arts if stage_of(e["producer"]) == module and e["kind"] not in ("record", "task")]


def _registered_status(ws, module, arts):
    """A step's status from this round's registered artifacts: "done", "fail" (its only output
    failed its gate), "partial" (registration under way: some data sets, or ②'s trial candidates whose
    search is still owed), "closed" (④ in a round closed by `rsi close-round`) or "idle". ② is done once
    one of its candidates (an old workspace's tune trials too) passed the train gate."""
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
    if module == "deliver":
        done = F.own_delivery(ws, arts) and not (_round(ws)["decision"] is None and F.redeliver_pending(ws))
        return "done" if done else "idle"
    if ws.evaluations():
        return "done"
    if ws.closed():
        return "closed"
    return "idle"


# The steps each next-step key sends the run to.
NEXT_STEPS = {"data": ("data",), "refresh": ("data",), "train": ("train",), "deliver": ("deliver",),
              "judge": ("judge",)}


def _node(state, text=""):
    return {"state": state, "text": text}


def _hm(dt):
    return dt.astimezone().strftime("%H:%M")


def _done_text(ws, module, mine, rnd):
    """The short status phrase of a step done in this round."""
    if module == "data":
        return "✓ %d 个集合" % sum(1 for e in mine if e["kind"] in DATASETS and ws.current(e["kind"]) == e["id"])
    if module == "train":
        return "✓ 训练门通过 · %d 个试验" % sum(1 for e in mine if e["kind"] == "candidate")
    if module == "deliver":
        ids = [e["id"] for e in mine if e["kind"] == "delivery"]
        return "✓ " + ids[0] if len(ids) == 1 else "✓ %d 个交付包" % len(ids)
    parts = []
    d = rnd["decision"]
    if ws.evaluations():
        if d is None:
            parts.append("已评测")
        else:
            parts.append("升级 %s" % d.get("version") if d.get("decision") == "promote" else "保留")
    if F.milestone_evaluations(ws):
        parts.append("里程碑评测")
    if rnd["plan_id"] is not None:
        parts.append("计划")
    return "✓ " + " · ".join(parts)


def flow_states(ws, progress):
    """The workflow diagram's nodes: key -> {"state", "text"} for task, the four stages (data, train,
    deliver, judge), check (顶层核对) and stop (停止与交付).

    state is done, current, wait (waits for the user), fail (the step's only output this round
    failed its gate), skip (not passed through this round) or todo; text is a short status phrase.
    A step is current while one of its modules runs or waits to be registered, or when next_action's
    step is it; done, fail and skip follow the round's registered artifacts; a step
    before the round's entry, or behind the step the round has reached with no output of its own,
    is skipped (with the attempts used up, ④ is still to come: the judge writes the plan of the
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
        elif step == "refresh":
            position = STEP_INDEX["deliver"]  # the new evalset comes before ③ (or ④)
        elif step in NEXT_STEPS:
            position = STEP_INDEX[step]
        elif step == "close":
            position = STEP_INDEX["judge"]  # ③ is not passed through; the judge still writes the plan
        else:
            position = AFTER_STEPS
        nxt = NEXT_STEPS.get(step, ())
        asked = bool(nxt) and key != "question" and bool(_state_questions(ws, key, action))
        writing_plan = action.get("step") == "write_plan"
    out = {"task": _node("done", "✓ 完成")}
    records = bool(ws.dispatches())
    answering = _stages(action_modules(ws, action)) if answer_going_back(ws, action) else []
    for m in MODULES:
        p = progress[m]
        mine = _step_artifacts(ws, m, arts)
        status = _registered_status(ws, m, arts)
        if m == "judge" and rnd["plan_id"] is not None:
            status = "done"  # the judge's plan is in (also for a round closed without evaluation)
        if p["state"] == "running" and not stopped:
            out[m] = _node("current", "● 运行中 · %s 起" % _hm(datetime.datetime.fromisoformat(p["since"])))
        elif p["state"] == "running" and records:  # the stop has not reached its agent or long job yet
            out[m] = _node("current", "● 停止后仍在运行")
        elif p["state"] == "pending" and not stopped:
            out[m] = _node("current", "● 待启动")
        elif p["state"] == "unregistered" and not stopped:
            out[m] = _node("current", "● 送回你的回答" if m in answering else "● 待登记")
        elif m == "judge" and writing_plan:
            out[m] = _node("current", "● 写计划")
        elif m == "data" and step == "refresh":
            out[m] = _node("current", "● 刷新评测集")  # also when ① is done in this round
        elif status == "done":
            out[m] = _node("done", _done_text(ws, m, mine, rnd))
        elif status == "fail":
            out[m] = _node("fail", "门没过 · 可重派" if m in nxt else "门没过")
        elif p["state"] == "pending":
            out[m] = _node("skip", "停止时未启动")
        elif p["state"] == "unregistered" and records:  # handed back, not registered, and the run has stopped
            held = held_handbacks(ws, m)
            out[m] = _node("skip", ("停止时回答还没送回" if all(F.question_answered(ws, d) for d in held)
                                    else "停止时还在等你回答") if held else "交回未登记")
        elif p["state"] in BUSY:
            out[m] = _node("skip", "停止时未交回")
        elif status == "closed":
            out[m] = (_node("done", "✓ 本轮已关闭 · 里程碑评测") if F.milestone_evaluations(ws)
                      else _node("skip", "本轮已关闭"))
        elif status == "partial":  # e.g. ② trials whose search is owed: under way until the round is decided
            out[m] = (_node("todo", "停止时未完成") if stopped else _node("todo", "本轮已判定，未完成")
                      if rnd["decision"] is not None else _node("current", "● 进行中"))
        elif m in nxt:
            late = m == "deliver" and F.own_delivery(ws, arts) and F.redeliver_pending(ws)
            out[m] = (_node("wait", "⏸ 等你批准") if asked
                      else _node("current", "● 待重新交付最佳版本" if late else "● 下一步"))
        elif STEP_INDEX[m] < entry or STEP_INDEX[m] < position:
            out[m] = _node("skip", "本轮不经过")
        else:
            out[m] = _node("todo")
    if stopped:
        out["check"] = _node("done", "✓ 已核对")
    elif step == "plan":
        out["check"] = _node("wait", "⏸ 等你批准计划")
    elif _milestone_open(ws):
        out["check"] = _node("wait", "⏸ 等你定里程碑")
    elif step == "promote":
        out["check"] = _node("current", "● 等 rsi promote")
    elif step == "close":
        out["check"] = _node("current", "● 等关闭本轮")
    elif step == "milestone" or _milestone_running(ws):
        out["check"] = _node("current", "● 里程碑发布中")
    elif step == "decided" and not writing_plan:
        out["check"] = _node("current", "● 等 rsi next")
    elif rnd["decision"] is not None:
        out["check"] = _node("done", "✓ 已判定")
    else:
        out["check"] = _node("todo")
    if stopped:
        at = _utc(stopped.get("at"))
        out["stop"] = _node("current", "■ %s 停止" % _hm(at) if at else "■ 已停止")
    else:
        out["stop"] = _node("todo")
        notes = bool(_notes_questions(ws))
        asker = None if notes else _asking_module(ws, key, action)
        if key == "question" or notes:  # the run waits for the user before going on
            for k, node in out.items():
                if node["state"] == "current" and (asker is None or k == asker):
                    running = k in progress and progress[k]["state"] == "running"
                    out[k] = _node("wait", "⏸ 运行中 · 等你回答" if running else "⏸ 等你回答")
            if asker is not None and out[asker]["state"] != "wait":
                out[asker] = _node("wait", "⏸ 等你回答")
    return out


# ---- the overview's lead ----------------------------------------------------------------------

# Short phrases for the orchestrator's own commands (_status keys); a next step leads with its name.
LEADS = {"promote": "顶层核对：等 rsi promote 判定本轮", "close": "等关闭本轮（rsi close-round）",
         "milestone": "里程碑发布进行中：等里程碑的交付和评测登记"}
# The same with the engine's records: rsi run carries these steps out itself.
ENGINE_LEADS = dict(LEADS, promote="顶层核对：rsi run 判定本轮（升级或保留）", close="rsi run 关闭本轮（rsi close-round 的效果）")


def now_lead(ws, progress=None, questions=None, view=None):
    """What the run waits for in a few words: the title of the overview's 「现在在等什么」 (its detail is
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
        return "等你批准计划 %s" % rnd["plan_id"]
    if questions:
        return "等你在 Claude 里回答上方的问题（%d 条）" % len(questions)
    if key in ("running", "handed"):
        state = "running" if key == "running" else "unregistered"
        if ws.dispatches():  # the engine's records: the dispatches the action names, of any round
            mods = [m for m in action_modules(ws, action) if stage_of(m)]
        else:
            mods = [m for m in MODULES if progress[m]["state"] == state]
        where = " / ".join(STEP_NAMES[m] for m in _stages(mods))
        if answer_going_back(ws, action):
            return "%s：把你的回答送回 %s" % (where, "、".join(_skill(m) for m in mods))
        if key == "running" and ws.dispatches():  # an agent not launched yet is no running one
            ds = _action_dispatches(ws, action)
            run = [m for m in mods if any(d.get("module") == m and d.get("status") != "pending" for d in ds)]
            queued = [m for m in mods if m not in run]
            parts = (["%s 运行中" % "、".join(_skill(m) for m in run)] if run else []) + (
                ["%s 待启动" % "、".join(_skill(m) for m in queued)] if queued else [])
            return "%s：%s" % (where, "，".join(parts))
        return "%s：%s %s" % (where, "、".join(_skill(m) for m in mods), "运行中" if key == "running" else "已交回，待登记")
    if key in NEXT_STEPS:
        return "下一步：%s" % where
    records = bool(ws.dispatches())  # the engine's run: rsi run does promote, close_round and next itself
    if key in LEADS:
        return (ENGINE_LEADS if records else LEADS)[key]
    if ws.stop_check()[0]:
        return "停止条件已满足：rsi run 结束运行" if records else "停止条件已满足：等 rsi next 结束运行"
    if rnd["plan_id"] is None:
        return "%s：等 rsi-judge 写计划" % STEP_NAMES["judge"]
    return ("rsi run 进入第 %d 轮" if records else "等进入第 %d 轮（rsi next）") % (s["round"] + 1)


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
