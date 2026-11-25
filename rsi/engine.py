"""The `rsi run` engine: carries out rsi.flow.next_action until the run has to wait.

run_status(ws, runner, max_steps=200) loops: next_action -> log the action to <ws>/work/run.jsonl -> carry it
out -> log its result -> rewrite the progress pages (rsi.dashboard.write_dashboard; a failure only warns) ->
next_action again. It returns (status, reason): why it stopped as one of STATUSES and one Chinese line. run()
returns the line only. All it needs is in the workspace, so a run cut off at any point (the process killed, the
machine restarted) is picked up by running it again.

While it waits (wait_agent's polls, and wait_long_job with check-ins on) every poll ends with Engine.tick: once
flow.page_refresh_seconds (default 60) have passed since the pages were last written, it takes a GPU snapshot
(rsi.gpus.snapshot, written to <ws>/work/gpus.json: the only place nvidia-smi runs) and rewrites the pages. Given a
status stream (`rsi run` gives stderr) the engine prints status lines (rsi.live.now_line, e.g. `20:23 第1轮 ② 训练调优
d-0001 · 第 3 批：3 个训练在跑 · 配置 3/3 · 下次检查 20:46`): after a step when rsi.live.now_key differs from the last
line's (the first step always prints; a run that starts with a wait prints before it), and while it waits when it has
printed none for flow.status_minutes (default 5: a heartbeat). The pages, the snapshot and the status line only warn
when they fail (`warning: <what>: <error>` on stderr, each kind of failure once a run: Engine.warn); stdout keeps only
the reason. The actions:

  dispatch
        the dispatch records first, in one write, status pending (purpose, and a deliver's candidate_id, copied
        from the action), then each output directory work/round-<n>/<module>[-k]/ with brief.md and brief.json,
        then the launches. A record is marked launched (status running and launched_at together, with the new
        session's id when the runner makes one up) just before its runner is called: a pending record is
        launched by the next run, a launched one is never launched twice. A record closed before its launch (the
        run stopped) does not count that launch.
  wait_long_job
        without check-ins (no checkin_at): stop, the run waits for `rsi run` once the marker is there. With
        check-ins on (flow.checkin_minutes): wait in this process until a long job's marker appears, the
        check-in is due (checkin_at) or the run is stopped; the next action then goes on (a check-in is a
        resume, mode checkin).
  wait_agent
        poll the agents until one ends (with the manual runner: stop at once). A running record this process
        did not launch is attached (the runner finds its process again); once its process is gone it counts as
        handed back when its output directory holds a valid handback.json, else as failed. A pending record is
        launched first.
  process_handback
        a hand-back with questions registers nothing: the questions become STATE waiting (the record's `asked`
        notes the launch) and the run stops; once the user has answered, the hand-back goes back to the same
        session with the answers (resume, mode answer) and the module's next hand-back is registered. A hand-back
        whose candidates would take the compute used over budget.compute_hours while the task requires the
        over_budget approval and the round has none (Workspace.budget_overrun) is held the same way (hold_budget):
        nothing of it is registered, STATE waiting asks over_budget (kind approval, ref None, dispatch_id: this
        dispatch; options approve, stop) and the record's `held_budget` notes the launch; once approved (`rsi answer`
        or `rsi approve <ws> over_budget`) the same hand-back is registered, no retraining; a stop ends the hold
        unregistered and its candidates' compute counts (Workspace.stop; so does that of any hand-back a stop
        closes unprocessed, also one found when settling after the stop). Order: questions, a long job's
        continuation, the budget hold, registration. Otherwise
        register each artifact subdirectory the hand-back lists, in dependency order (KIND_ORDER: task, corpus,
        testset, evalset, valset, pool, trainset first), skipping those already registered (each registered entry
        of the record notes its launch). Once a task, corpus, set or pool of the hand-back is refused (or a
        subdirectory whose kind cannot be told), the ones after it are refused too, unregistered (DATA_CHAIN); a
        record still registers. A subdirectory with task.json and no manifest.json is a new task version:
        only rsi-data hands one back (`dset sets` writes it when it fills in the split rule), and it registers only
        as a fill of the split rule, checked on the staged copy (Workspace.register fill_only), so it never uses a
        task_change approval; the manifests after it carry the new version. A milestone artifact (test predictions,
        an evaluation on the testset) gets its use_testset approval right before its registration, quoting the
        user's milestone consent. Refusals are kept in the record (with the launch) and fail the dispatch, and the
        decision point handback_refused is decided right there (rsi.decisions.apply: by default back to the same
        session with the registration's error text). A long job whose marker is there (the record's marker_seen
        notes it, ok or not: that job has ended, so no stop or close looks for it again; a later long_job
        hand-back clears it) and says ok: the artifacts it lists, or,
        with none listed, its session is continued to collect them, at most flow.max_continues times (then the
        dispatch fails as "capped" and the user decides); a marker that says anything else fails the dispatch.
  resume
        the same session gets feedback-<n>.md: a return (mode resume, returns + 1), the continuation after a
        long job (continue), the second part of a two-part hand-back (followup: train's search, the judge's plan;
        followups + 1, no return), the user's answers to its questions (answer: answered + 1, no return) or a long
        job's check-in (checkin: checkins + 1, no return; the feedback starts with 「# 定时检查」); with
        no session, or one the runner says cannot be resumed, a fresh session gets history-<n>.md (the previous
        brief and hand-backs and the feedback). The user's answers to the dispatch's questions go into the
        feedback. A followup's and a return's feedback list what the dispatch registered since the round's latest
        search (over all its launches: the first part of a two-part hand-back), each hand-back subdirectory with its
        id, and registered-<n>.json beside it holds the same as [{"dir", "kind", "id"}].
  decide        rsi.decisions.apply: the code's default (or a decision agent's choice), recorded once per
                decision_key, so a run cut off before carrying it out, and the run after the user answered the
                question a decision asked, reuse the latest record; then carried out.
  ask           STATE waiting, then stop (rsi answer / rsi approve record the answer).

The user may stop the run (`rsi stop`, or `stop` as an answer) from another process while a step is under way.
Before anything is launched or resumed, and before a decision is made or carried out, the engine reads STATE
again (halt_if_stopped): once it is stopped nothing is started, the dispatch that would have been is closed
(status done, "closed"), and the step ends with StoppedError; a registration refused because the run stopped is
no refusal of the module's work. The loop then goes on to the report, which first settles every open record
(settle_after_stop): agents still running are stopped, a long job without its marker has its process (named in
<marker>.pid, LONG_JOB_PID_SUFFIX) terminated when it is confirmed to be the job, and so has each process an orphan
pid file of its output directory names (orphan_pid_files: a long job its session started and has not handed back
yet, e.g. a check-in's new watch) (stop_long_job; the record's long_job_stop says what became of each, and done's
reason lists the long jobs terminated and those left running). A registration cut short by the stop does the same
for its dispatch. A decision or `rsi close-round` that closes a failed dispatch ends its long job the same way, after
its transaction is saved (rsi.decisions.terminate_after_commit; long_job_stop.context "close"); one cut off between
that save and the termination (long_job_stop "pending") is finished at the start of the run's next step, by the next
close, or at the stop (end_long_job; the start-time check makes a repeated termination harmless).
  promote, close_round, next, report
                the Workspace methods (the report goes to <ws>/REPORT.md); done stops.

While it runs the engine holds <ws>/work/run.lock (its pid), so a second `rsi run` on the workspace is refused; the
pages show "● 运行中" only while a live process holds it, and are written once more after the lock is let go.
Besides each action and its result, run.jsonl records launch, recover, agent_end, register, resume (a return,
or the continuation after a long job: its mode says which), followup (the second part of a two-part hand-back),
checkin (a long job's check-in),
resume_with_answer (the user's answers to a hand-back's questions), asked (a hand-back's questions, its artifacts
held), budget_held (a hand-back held for over_budget: dispatch_id, launch, question_id, dirs, cost, used, budget),
budget_released (that hand-back, approved, goes to registration: dispatch_id, launch, by), resume_failed,
fresh_session, session_mismatch, clear_waiting, decision (every decision, with who made
it), decision_reused, decision_agent, decision_agent_failed, approval (a use_testset approval quoting the
milestone consent), continue_capped, closed, agent_stopped, long_job_stopped, long_job_gone, long_job_left (a
long job still running at a stop: terminated, already gone, or left running) and exit; `rsi answer` and `rsi
approve` append answer and approve (log_event).

Files of a dispatch (n numbers its launches): brief.md, brief.json, prompt-<n>.md (claude, manual),
agent-<n>.log, feedback-<n>.md (why it was sent back, or, for a dispatch a decision made, why it was made),
registered-<n>.json (a followup's or a return's: what the dispatch registered since the round's latest search), history-<n>.md, handback.json (the current one)
and handback-<n>.json (the one launch n wrote, kept when the next launch starts).
"""
import glob
import json
import os
import re
import sys
import time

try:
    import fcntl
except ImportError:  # no fcntl (not POSIX): run.lock is a pid file created with O_EXCL
    fcntl = None

from rsi import briefs as B
from rsi import decisions as DP
from rsi import flow as F
from rsi import gpus as G
from rsi import live as L
from rsi.runners import attach_runner
from rsi.runners.base import SESSION_MODES, launch_failed, pid_alive, process_info, stop_process
from rsi.runners.base import launched_epoch
from rsi.report import _num, long_job_text
from rsi.rules import flow_settings
from rsi.workspace import RsiError, StoppedError, Workspace, now_iso, parse_utc

RUN_LOG = os.path.join("work", "run.jsonl")
RUN_LOCK = os.path.join("work", "run.lock")
# Why a run stopped: done (stopped and reported), waiting for the user, for a long job's marker, for a manual
# hand-back, or max_steps used up; `rsi run` exits with STATUS_EXIT_CODES.
STATUSES = ("done", "waiting_user", "waiting_long_job", "waiting_manual", "max_steps")
STATUS_EXIT_CODES = {"done": 0, "waiting_user": 10, "waiting_long_job": 11, "waiting_manual": 12, "max_steps": 13}
# Registration order inside one hand-back: an artifact's checks refer to the kinds before it (data's: the filled-in
# task version first, so the manifests after it may carry its version; the corpus before the sets it vouches for).
KIND_ORDER = ("task", "corpus", "testset", "evalset", "valset", "pool", "trainset", "candidate", "search", "delivery",
              "feedback", "evaluation", "plan", "record")
# The only module that hands back a task version (a fill of the split rule), and what a refusal of one says.
TASK_MODULE = "data"
TASK_REFUSED = "rsi-data 只能交回补全切分规则的任务新版本，这份不是："
# The kinds of data's hand-back that depend on each other (KIND_ORDER): once one is refused (or a subdirectory whose
# kind cannot be told), the later ones are refused unregistered, since registering them would skip the checks against
# it (the sets come from the corpus) and fix the split rule; a record still registers.
DATA_CHAIN = ("task", "corpus", "testset", "evalset", "valset", "pool", "trainset")
# A long job's process: the module writes <marker>.pid next to its completion marker, first line the pid, an optional
# second line the process's start time (Unix seconds, e.g. `date +%s`). On a stop the report step terminates that
# process (its process group) when it is confirmed to be the job (stop_long_job).
LONG_JOB_PID_SUFFIX = ".pid"
# The claude runner's own pid file in a dispatch's output directory (rsi.runners.claude): never a long job's.
AGENT_PID_FILE = re.compile(r"^agent-\d+\.pid$")
# How far the start time `ps` shows (or a marker's time) may be off, in whole seconds: rsi.flow.LONG_JOB_SLACK_S.
LONG_JOB_SLACK_S = F.LONG_JOB_SLACK_S
# The counter of a dispatch record each launch mode in the same session adds one to, and its run.jsonl event.
MODE_COUNTERS = {"resume": "returns", "continue": "continued", "followup": "followups", "answer": "answered",
                 "checkin": "checkins"}
MODE_EVENTS = {"resume": "resume", "continue": "resume", "followup": "followup", "answer": "resume_with_answer",
               "checkin": "checkin"}
# The decisions live in rsi.decisions (these names stay for callers of the engine).
decision_key, apply_decision, ask_reason = DP.decision_key, DP.apply, DP.ask_reason


def run(ws, runner, max_steps=200, pages=True, poll_s=None, status=None):
    """run_status, returning only the reason line."""
    return run_status(ws, runner, max_steps=max_steps, pages=pages, poll_s=poll_s, status=status)[1]


def run_status(ws, runner, max_steps=200, pages=True, poll_s=None, status=None):
    """Run the main flow of workspace `ws` (a Workspace or its path) with `runner` until it has to wait, at most
    `max_steps` actions; return (status, reason). pages=False skips rewriting the progress pages (and the GPU
    snapshots); poll_s overrides the runner's seconds between polls; `status` (a text stream: `rsi run` gives
    sys.stderr; None prints none) gets the status lines."""
    root = ws.root if isinstance(ws, Workspace) else Workspace(ws).root
    if max_steps < 1:
        raise RsiError("--max-steps must be at least 1 (got %d)" % max_steps)
    lock = _lock(root)
    eng = Engine(root, runner, pages=pages, poll_s=poll_s, status=status)
    try:
        return eng.run(max_steps)
    finally:
        _unlock(lock)
        eng.write_pages()  # once more without the lock: the pages no longer say a run is under way


def _lock_owner(path):
    try:
        return int(_read(path, default="").strip() or 0)
    except ValueError:
        return 0


def _lock(root):
    """Take <ws>/work/run.lock so that two `rsi run`s never work on a workspace side by side; returns what
    _unlock needs. With fcntl: an exclusive non-blocking flock held for the run's whole life (the kernel drops it
    when the process dies, so a lock left by a killed run is no lock and no takeover can race); the file holds
    the pid for the message, and it is checked to be the file at the path (not one a finishing run just
    removed). Without fcntl: the file is created with O_CREAT|O_EXCL; a file whose pid is gone is moved aside
    with an atomic rename (only one process can) and creation is retried."""
    path = os.path.join(root, RUN_LOCK)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    busy = "another `rsi run` (pid %s) is working on this workspace: wait for it to stop (it holds %s)"
    if fcntl is not None:
        for _attempt in range(20):
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise RsiError(busy % (_lock_owner(path) or "?", path))
            try:
                same = os.fstat(fd).st_ino == os.stat(path).st_ino
            except OSError:
                same = False
            if not same:  # the file was removed or replaced between open and flock: take the one at the path
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, ("%d\n" % os.getpid()).encode())
            return (path, fd)
        raise RsiError("cannot take %s" % path)
    for _attempt in range(20):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pid = _lock_owner(path)
            if pid and pid != os.getpid() and pid_alive(pid):
                raise RsiError(busy % (pid, path))
            aside = "%s.stale-%d-%s" % (path, os.getpid(), os.urandom(3).hex())
            try:
                os.rename(path, aside)  # atomic: of two runs taking over, one moves it
            except OSError:
                continue
            os.remove(aside)
            continue
        os.write(fd, ("%d\n" % os.getpid()).encode())
        return (path, fd)
    raise RsiError("cannot take %s" % path)


def _unlock(lock):
    path, fd = lock
    try:
        os.remove(path)  # removed while still held: a run that opened the old file re-checks the path
    except OSError:
        pass
    os.close(fd)


def _answer_item(x):
    return {"dispatch_id": x.get("dispatch_id"), "question": x.get("question"), "by": x.get("by"), "quote": x.get("quote")}


def _answer_lines(items):
    return "".join("- %s（%s）\n  用户（%s）的回答：%s\n" % (x["question"], x["dispatch_id"], x["by"], x["quote"])
                   for x in items)


def brief_changes(old, new):
    """What differs between two brief.json dicts, in words: the ids, the inputs (added, gone, another artifact),
    the attempts and the budget left."""
    if not old:
        return []
    out = []
    oi, ni = old.get("ids") or {}, new.get("ids") or {}
    for k in sorted(set(oi) | set(ni)):
        if oi.get(k) != ni.get(k):
            out.append("`ids.%s`：%s → %s" % (k, oi.get(k), ni.get(k)))
    oin, nin = old.get("inputs") or {}, new.get("inputs") or {}
    notes = new.get("input_notes") or {}
    onotes = old.get("input_notes") or {}
    for k in sorted(set(oin) | set(nin)):
        if k not in oin:
            out.append("新的输入 `%s`：%s（%s）" % (k, notes.get(k, ""), nin[k]))
        elif k not in nin:
            out.append("不再给的输入 `%s`（%s）" % (k, onotes.get(k, oin[k])))
        elif oin[k] != nin[k]:
            out.append("输入 `%s` 换成了 %s（%s）" % (k, notes.get(k, ""), nin[k]))
    for k, label in (("attempts", "本轮已用尝试"), ("budget_left", "剩余预算")):
        if old.get(k) != new.get(k):
            out.append("%s：%s → %s" % (label, json.dumps(old.get(k), ensure_ascii=False),
                                       json.dumps(new.get(k), ensure_ascii=False)))
    return out


def log_event(root, event, **fields):
    """Append one line to <root>/work/run.jsonl: {"at", "event", fields...} (also `rsi answer` and `rsi approve`)."""
    rec = {"at": now_iso(), "event": event}
    rec.update(fields)
    path = os.path.join(root, RUN_LOG)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def registered_file(n):
    """The file beside feedback-<n>.md of a followup or a return that maps what the dispatch registered since the
    round's latest search: registered-<n>.json."""
    return "registered-%d.json" % n


def registered_since_search(ws, d):
    """What dispatch `d` registered, over all its launches, after its round's latest search (the first part of a
    two-part hand-back, even when it took more than one launch), in registration order, as [{"dir", "kind", "id"}]."""
    arts = ws.state["rounds"][str(d.get("round"))]["artifacts"]
    last = max([i for i, a in enumerate(arts) if ws.entry(a)["kind"] == "search"] + [-1])
    order = dict((a, i) for i, a in enumerate(arts) if i > last)
    mine = sorted((x for x in d.get("registered") or [] if x.get("id") in order), key=lambda x: order[x["id"]])
    return [{"dir": x.get("dir"), "kind": x.get("kind"), "id": x.get("id")} for x in mine]


def long_job_since(d):
    """When dispatch `d`'s long job may have started at the earliest (Unix seconds): the launch of the hand-back that
    first named its marker (the record's long_job_since; a check-in that hands back the same marker keeps it), else
    the record's launch. A long job's process started before it is not the job (long_job_identity)."""
    since = parse_utc(d.get("long_job_since"))
    return since.timestamp() if since is not None else launched_epoch(d)


def long_job_open(ws, d):
    """True when dispatch `d` may still have a long job running: its record names a completion marker that is not
    there (a long job, a check-in's session working or ended, a failed return of a long-job dispatch) and that is not
    the one a processed hand-back read (marker_seen: that job has ended, whatever became of the file since; a later
    long_job hand-back clears it, even under the same marker path).
    Whoever stops the run or closes the dispatch terminates it first (terminate_long_job)."""
    marker = d.get("marker")
    return bool(marker) and d.get("marker_seen") != marker and not F.marker_done(ws, d)


def orphan_pid_files(ws, d):
    """The pid files (<name>.pid, absolute, sorted) in dispatch `d`'s output directory whose <name> does not exist or
    is older than the pid file (beyond LONG_JOB_SLACK_S: an earlier job's marker, the path reused), other than its
    record's marker's (and marker_seen's) and the runner's agent-<n>.pid: long jobs its session started and has not
    handed back yet (e.g. a check-in that relaunched the watch). terminate_long_job ends them too."""
    out = os.path.join(ws.root, d["outdir"]) if d.get("outdir") else None
    if not out or not os.path.isdir(out):
        return []
    markers = (F.marker_path(ws, d), F.marker_path(ws, {"marker": d.get("marker_seen")}))
    known = set(os.path.realpath(long_job_pid_file(m)) for m in markers if m)
    found = []
    for path in sorted(glob.glob(os.path.join(out, "*" + LONG_JOB_PID_SUFFIX))):
        if os.path.realpath(path) in known or AGENT_PID_FILE.match(os.path.basename(path)) or not os.path.isfile(path):
            continue
        try:
            ended = os.lstat(path[:-len(LONG_JOB_PID_SUFFIX)]).st_mtime >= os.path.getmtime(path) - LONG_JOB_SLACK_S
        except OSError:
            ended = False
        if not ended:
            found.append(path)
    return found


def long_job_note(st):
    """A long_job_stop in words, for the record's `closed`: over its pid files (the marker's and the others)."""
    every = [st] + list(st.get("others") or [])
    left = [x for x in every if x["result"] not in ("terminated", "gone")]
    words = ["长任务已终止"] if any(x["result"] == "terminated" for x in every) else []
    if left:
        words.append("长任务没能终止，可能仍在运行（%s）" % "；".join(x["why"] for x in left))
    return "；".join(words) or "长任务的进程已不在"


def terminate_long_job(ws, d, context="stop"):
    """Dispatch `d`'s long job has not written its marker and the run stops (context "stop") or the dispatch is
    closed by a decision or `rsi close-round` ("close"): terminate the job's process (its process group: SIGTERM, then
    SIGKILL) when <marker>.pid names a process confirmed to be the job (long_job_identity, from long_job_since:
    check-ins relaunch the session, not the job), and likewise each orphan_pid_files (confirmed from the dispatch's
    start: any launch of it may have started one). Returns the record's long_job_stop: {"pid", "pid_file", "result"
    (terminated, gone, or one of rsi.report.LONG_JOB_LEFT: unconfirmed, no_pid_file, bad_pid_file: it may still be
    running), "why"} of <marker>.pid, with "others" (the orphans, each {"pid", "pid_file", "result", "why"}), "at" and
    "context". run.jsonl: long_job_stopped, long_job_gone or long_job_left for each pid file."""
    marker = F.marker_path(ws, d)
    out = _stop_pid_file(ws, d, long_job_pid_file(marker) if marker else None, long_job_since(d), context)
    since = parse_utc(d.get("started_at"))
    since = min(since.timestamp(), long_job_since(d)) if since is not None else long_job_since(d)
    out["others"] = [_stop_pid_file(ws, d, path, since, context, orphan=True) for path in orphan_pid_files(ws, d)]
    out.update(at=now_iso(), context=context)
    return out


def _stop_pid_file(ws, d, path, since, context, orphan=False):
    """Terminate the process pid file `path` names when it is confirmed (long_job_identity from `since`; a file
    without a start time line is checked against its modification time: the job wrote it after it started) to be a
    long job of dispatch `d`: {"pid", "pid_file", "result", "why"}, logged to run.jsonl (`orphan`: one of
    orphan_pid_files)."""
    out = {"pid": None, "pid_file": path}
    if path is None or not os.path.exists(path):
        out.update(result="no_pid_file", why="没有 pid 文件 %s，不知道长任务的进程" % (path or "（没有完成标记）"))
    else:
        pid, started, error = read_pid_file(path)
        out["pid"] = pid
        if error is None and started is None:
            try:
                started = os.path.getmtime(path)
            except OSError:
                pass
        if error:
            out.update(result="bad_pid_file", why=error)
        else:
            state, why = long_job_identity(pid, started, since)
            if state == "ours":
                stop_process(pid)
                out.update(result="terminated", why="%s，已终止" % why)
            else:
                out.update(result=state, why=why)
    event = {"terminated": "long_job_stopped", "gone": "long_job_gone"}.get(out["result"], "long_job_left")
    log_event(ws.root, event, dispatch_id=d["id"], pid=out["pid"], result=out["result"], why=out["why"], pid_file=path,
              context=context, orphan=orphan)
    return out


def end_long_job(ws, d, context):
    """Outside any transaction: terminate_long_job for dispatch `d` (closed already, its long_job_stop maybe "pending":
    rsi.decisions.close_dispatch) and record its long_job_stop, with the note appended to `closed`, in a save of its
    own; returns it. When the job is no longer open (its marker came meanwhile) a pending long_job_stop goes and None
    is returned. An error while terminating is recorded as result "error" (it may still run), never raised."""
    if not long_job_open(ws, d):
        if (d.get("long_job_stop") or {}).get("result") == "pending":
            ws.update_dispatch(d["id"], long_job_stop=None)
        return None
    try:
        st = terminate_long_job(ws, d, context)
    except Exception as e:  # recorded: the user ends it by hand; the other dispatches are still dealt with
        marker = F.marker_path(ws, d)
        st = {"pid": None, "pid_file": long_job_pid_file(marker) if marker else None, "result": "error",
              "why": "终止长任务时出错：%s: %s" % (type(e).__name__, e), "others": [], "at": now_iso(),
              "context": context}
        log_event(ws.root, "long_job_left", dispatch_id=d["id"], pid=None, result="error", why=st["why"],
                  pid_file=st["pid_file"], context=context, orphan=False)
    ws.update_dispatch(d["id"], long_job_stop=st, closed="%s；%s" % (d.get("closed"), long_job_note(st)))
    return st


def pending_long_job_stops(ws):
    """The dispatches, whatever their status, whose long_job_stop is still "pending": a close saved, and its
    termination (end_long_job) cut off before it ran."""
    return [d for d in ws.dispatches() if (d.get("long_job_stop") or {}).get("result") == "pending"]


def long_job_pid_file(marker):
    """The pid file of the long job whose completion marker is `marker` (absolute): <marker>.pid."""
    return marker + LONG_JOB_PID_SUFFIX


def read_pid_file(path):
    """(pid, start time in Unix seconds or None, error or None) from a long job's pid file: the first line is the
    pid, an optional second line the process's start time (Unix seconds, or UTC ISO 8601 like 2026-11-23T20:15:00Z)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [x.strip() for x in f.read().splitlines()]
    except OSError as e:
        return None, None, "读不了 %s：%s" % (path, e)
    try:
        pid = int(lines[0])
        if pid <= 1:
            raise ValueError(pid)
    except (IndexError, ValueError):
        return None, None, "%s 的第一行不是进程号" % path
    started = None
    if len(lines) > 1 and lines[1]:
        try:
            started = float(lines[1])
        except ValueError:
            dt = parse_utc(lines[1])
            if dt is None:
                return pid, None, "%s 的第二行不是启动时间（Unix 秒或 ISO 8601 UTC）：%s" % (path, lines[1][:80])
            started = dt.timestamp()
    return pid, started, None


def long_job_identity(pid, started_file, launched):
    """(state, why): "ours" when process `pid` lives (a zombie does not: pid_alive), `ps` gives its start time, that
    start is not before the dispatch's launch `launched` (Unix seconds; whole seconds, and LONG_JOB_SLACK_S of slack:
    on Linux `ps` can put the start a second early) and, when the pid file gives a start time `started_file`, not
    later than it (beyond LONG_JOB_SLACK_S: a process started after the job did reuses its pid; one started before
    the time the file gives is the job, which wrote the file late); "gone" when there is no such process; else
    "unconfirmed"."""
    if not pid_alive(pid):
        return "gone", "进程 %d 已不在" % pid
    info = process_info(pid)
    if info is None or info[1] is None:
        return "unconfirmed", "进程 %d 还在，但查不到它的启动时间，确认不了它是这个长任务" % pid
    started = info[1]
    if started < int(launched) - LONG_JOB_SLACK_S:
        return "unconfirmed", ("进程 %d 的启动时间（%s）早于这次派出的启动（%s），不是这个长任务启动的（可能是复用的进程号）"
                               % (pid, _local(started), _local(launched)))
    if started_file is not None and started - started_file > LONG_JOB_SLACK_S:
        return "unconfirmed", ("进程 %d 的启动时间（%s）晚于 pid 文件的时间（%s），不是这个长任务（可能是复用的进程号）"
                               % (pid, _local(started), _local(started_file)))
    return "ours", "进程 %d 是这个长任务（%s 启动）" % (pid, _local(started))


def _local(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


# ---- the engine ------------------------------------------------------------------------------------

def _read(path, default=None):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        if default is None:
            raise
        return default


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class Engine(object):
    def __init__(self, root, runner, pages=True, poll_s=None, status=None, clock=time.time):
        self.root, self.runner, self.pages = root, runner, pages
        self.poll_s = runner.poll_interval if poll_s is None else poll_s
        self.handles = {}  # dispatch id -> runner handle, for the agents this process launched or attached
        self.status, self.clock = status, clock  # status: the stream of the status lines (None: none)
        self.pages_at = self.said_at = clock()  # when the pages were last written, a status line last printed
        self.said_key = self.said_line = None  # rsi.live.now_key and the text of the last status line
        self.warned = set()  # the warnings printed so far (warn: each kind of failure once)

    def ws(self):
        """The workspace as it is on disk now (the user may approve or answer while the engine waits)."""
        return Workspace(self.root)

    def outdir(self, d):
        return os.path.join(self.root, d["outdir"])

    # ---- the loop ----------------------------------------------------------------------------------

    def run(self, max_steps):
        for step in range(1, max_steps + 1):
            ws = self.ws()
            repaired = ws.repair_registry()
            if repaired:  # registrations cut off between registry.jsonl and STATE
                self.log("repair", artifacts=repaired)
            pending = pending_long_job_stops(ws)
            for d in pending:  # a close saved, its termination cut off (e.g. `rsi answer` killed): finished first
                end_long_job(ws, d, d["long_job_stop"].get("context") or "close")
            if pending:
                ws = self.ws()
            action = F.next_action(ws)
            if ws.waiting() and action["type"] != "ask":
                self.log("clear_waiting", waiting=ws.waiting(), action=action["type"])
                ws.clear_waiting()  # an approval wait already approved, or a run that has stopped
            self.log("action", step=step, action=action)
            if action["type"] in ("wait_agent", "wait_long_job"):
                self.say(ws)  # a run that starts with a wait says so before it
            try:
                result, stop = self.execute(ws, action)
            except StoppedError as e:  # stopped from another process during the step: the next action reports
                self.log("result", step=step, type=action["type"], stopped=e.errors)
                self.write_pages()
                self.say(self.ws())
                continue
            except RsiError as e:
                self.log("result", step=step, type=action["type"], error=e.errors)
                self.write_pages()
                raise
            except BaseException as e:
                self.log("result", step=step, type=action["type"], error=["%s: %s" % (type(e).__name__, e)])
                raise
            self.log("result", step=step, type=action["type"], result=result)
            self.write_pages()
            self.say(self.ws())
            if stop is not None:
                self.log("exit", status=stop[0], reason=stop[1])
                return stop
        stop = ("max_steps", "已执行 %d 步，到了 --max-steps 的上限；再运行 rsi run 接着走。" % max_steps)
        self.log("exit", status=stop[0], reason=stop[1])
        return stop

    def execute(self, ws, a):
        """Carry out one action; return (result for the run log, (status, reason) to stop, or None)."""
        t = a["type"]
        if t == "dispatch":
            return self.dispatch(ws, [a]), None
        if t == "resume":
            return self.resume(ws, ws.dispatch(a["dispatch_id"]), a.get("feedback") or a.get("reason") or "",
                               mode=a.get("mode") or "resume"), None
        if t == "wait_agent":
            return self.wait(a["dispatch_ids"])
        if t == "wait_long_job":
            if a.get("checkin_at"):
                return self.wait_long_job(a["checkin_at"]), None
            d = ws.dispatch(a["dispatch_id"])
            return {}, ("waiting_long_job", "在等长任务：%s（rsi-%s）完成后会写标记 %s；标记出现后再运行 rsi run。"
                        % (d["id"], d.get("module"), F.marker_path(ws, d)))
        if t == "process_handback":
            return self.process(ws, ws.dispatch(a["dispatch_id"]))
        if t == "decide":
            return DP.apply(self, ws, a)
        if t == "ask":
            return self.ask(ws, a)
        if t == "promote":
            decision, version, reason = ws.promote(a["evaluation_id"])
            return {"decision": decision, "version": version, "reason": reason}, None
        if t == "close_round":  # as `rsi close-round`: the round's failed dispatches with a long job closed first
            decision, _version, reason = DP.close_round(ws, a["reason"])
            return {"decision": decision, "reason": reason}, None
        if t == "next":
            n, text = ws.next_round()
            return {"round": n, "text": text}, None
        if t == "report":
            from rsi.report import write_report
            closed = self.settle_after_stop(ws)
            return {"report": write_report(self.ws(), os.path.join(self.root, F.REPORT_FILE)), "closed": closed}, None
        if t == "done":
            closed = self.settle_after_stop(ws)  # never done while a record is still running
            if closed:
                self.write_pages()
            stopped = ws.state["stopped"] or {}
            reason = "运行已结束（%s），报告在 %s。" % (stopped.get("reason"), os.path.join(self.root, F.REPORT_FILE))
            reason += long_job_text(self.ws())
            return {}, ("done", reason)
        raise RsiError("unknown action type %r" % t)

    def ask(self, ws, a):
        w = ws.waiting()
        if not (w and (w.get("kind"), w.get("what"), w.get("ref")) == (a.get("kind"), a.get("what"), a.get("ref"))):
            w = ws.set_waiting({"kind": a["kind"], "what": a.get("what"), "ref": a.get("ref"),
                                "question": a["question"], "options": list(a["options"])})
        return {"waiting": w}, ("waiting_user", DP.ask_reason(w))

    def stopped(self):
        """True when STATE (as on disk now) says the run has stopped."""
        return bool(self.ws().state["stopped"])

    def settle_after_stop(self, ws):
        """The run has stopped (the report step, and done): no record may stay running. Each running record's
        agent (this process's handle, or found again through its runner) gets its record when it has ended;
        otherwise it is stopped when the runner can vouch for it (its own process, or one `ours()` recognises) and
        the record is closed (a manual or lost agent: the record is only closed). Then every record left open is
        closed; a hand-back's `record` artifacts (still allowed after a stop) are registered first. In both, a record
        whose long job may still run (long_job_open: its marker is not there; a long job, or a check-in's session,
        working or ended) has the job terminated (terminate_long_job). First, a long_job_stop still "pending" (a
        close whose termination was cut off), whatever its dispatch's status, is finished (end_long_job). Returns the
        closed ids."""
        reason = (ws.state["stopped"] or {}).get("reason")
        for d in pending_long_job_stops(self.ws()):  # a close saved, its termination cut off: finished now
            end_long_job(self.ws(), d, d["long_job_stop"].get("context") or "close")
        closed = []
        for d in self.ws().dispatches():
            if d.get("status") not in ("pending", "running"):
                continue
            did = d["id"]
            if self._close_unlaunched(did):  # never launched: no agent, its launch not counted
                closed.append(did)
                continue
            h = self.handles.pop(did, None)
            if h is None and d.get("launched_at"):
                h = self.attach(self.ws(), d)
                self.handles.pop(did, None)
            stopped = False
            if h is not None:
                r = h["runner"].poll(h)
                if r is not None:  # it had ended: its record says how, closed below
                    try:
                        self.finish(did, r)
                    except StoppedError:  # a resume that could not go on: finish closed the record
                        pass
                    continue
                stopped = h["runner"].stop(h)
            why = "agent 已终止" if stopped else "这个运行确认不了它的 agent 进程（手动完成的，或进程已不在），只关闭记录"
            # A hand-back it wrote before its process was seen to end is this launch's (the previous one is renamed
            # before a relaunch): its candidates were trained, so their compute counts (once).
            self.ws().count_stopped_handback(d)
            extra = {}
            if long_job_open(self.ws(), d):  # a check-in's agent: the long job it looks after runs on
                extra["long_job_stop"] = self.stop_long_job(self.ws(), d)
                why += "；%s" % long_job_note(extra["long_job_stop"])
            self.ws().update_dispatch(did, status="done", pid=None, capped=None, ended_at=now_iso(),
                                      closed="运行已停止（%s）：%s" % (reason, why), **extra)
            self.log("agent_stopped" if stopped else "closed", dispatch_id=did, why="stopped", pid=d.get("pid"))
            closed.append(did)
        ws = self.ws()
        for d in ws.dispatches():
            status = d.get("status")
            if status not in ("handed_back", "failed", "long_job"):
                continue
            note = "这次派出的交回不再登记，也不再退回"
            extra = {}
            if status in ("handed_back", "long_job"):  # its candidates were trained: their compute counts (once)
                ws.count_stopped_handback(d)
            if status == "handed_back":
                done, left = self.register_records(ws, d)
                if done:
                    note = "登记了交回里的实验记录 %s；其余（%s）不再登记" % ("、".join(done), "、".join(left) or "无")
            if long_job_open(ws, d):  # a long job, or a check-in that ended (e.g. failed) while it runs on
                extra["long_job_stop"] = self.stop_long_job(ws, d)
                note += "；%s" % long_job_note(extra["long_job_stop"])
            ws.update_dispatch(d["id"], status="done", capped=None, ended_at=d.get("ended_at") or now_iso(),
                               closed="运行已停止（%s）：%s" % (reason, note), **extra)
            self.log("closed", dispatch_id=d["id"], why="stopped", status=status)
            closed.append(d["id"])
        return closed

    def stop_long_job(self, ws, d):
        """terminate_long_job for dispatch `d` at a stop (context "stop"; a closing decision and `rsi close-round` end
        theirs after their transaction: rsi.decisions.terminate_after_commit)."""
        return terminate_long_job(ws, d, "stop")

    def register_records(self, ws, d):
        """After a stop: register the `record` artifacts a hand-back lists (the one kind still allowed); returns
        (registered ids, the subdirectories not registered)."""
        hb = B.read_handback(self.outdir(d))
        out = os.path.realpath(self.outdir(d))
        sources = dict((e["source"], e) for e in ws.entries())
        done, left = [], []
        for name in hb["artifacts"] if hb["ok"] else []:
            src = os.path.realpath(os.path.join(out, name))
            if src in sources:
                continue
            kind, error, _testset = self._kind_of(src, d)
            if kind == "record" and error is None:
                try:
                    aid = ws.register("record", src)
                except RsiError as e:
                    self.log("register_refused", dispatch_id=d["id"], dir=name, error=e.errors)
                    left.append(name)
                else:
                    self.log("register", dispatch_id=d["id"], dir=name, kind=kind, id=aid, after_stop=True)
                    done.append(aid)
            else:
                left.append(name)
        return done, left

    def halt_if_stopped(self, d=None, what="不再启动"):
        """Read STATE from disk; when the run has stopped meanwhile, close dispatch `d` (one this step would launch
        or resume: failed, a long job to continue, or pending: not launched yet, its launch not counted) and raise
        StoppedError. Returns the workspace as read."""
        ws = self.ws()
        st = ws.state["stopped"]
        if not st:
            return ws
        if d is not None:
            cur = ws.dispatch(d["id"])
            if cur.get("status") == "pending" or (cur.get("status") == "running" and not cur.get("launched_at")):
                self._close_unlaunched(d["id"])
            elif cur.get("status") in ("failed", "running", "long_job"):
                extra, note = {}, ""
                if long_job_open(ws, cur):  # a check-in this step would launch: its long job runs on
                    extra["long_job_stop"] = self.stop_long_job(ws, cur)
                    note = "；%s" % long_job_note(extra["long_job_stop"])
                ws.update_dispatch(d["id"], status="done", closed="运行已停止（%s）：%s%s" % (st.get("reason"), what, note),
                                   capped=None, ended_at=now_iso(), **extra)
                self.log("closed", dispatch_id=d["id"], why="stopped")
        raise StoppedError("the run has stopped (%s) while this step was under way: nothing more is started"
                           % st.get("reason"))

    # ---- dispatching and launching ------------------------------------------------------------------

    def _new_outdir(self, rnd, module, taken):
        k = 1
        while True:
            rel = "work/round-%d/%s" % (rnd, module if k == 1 else "%s-%d" % (module, k))
            if rel not in taken and not os.path.exists(os.path.join(self.root, rel)):
                return rel
            k += 1

    def dispatch(self, ws, actions, why=None):
        """New dispatch records for `actions` (one write), then their output directories and briefs, then the
        launches. `why` (a dispatch a decision made: (reason, failing candidate ids)) becomes feedback-1.md."""
        ws = self.halt_if_stopped()
        rnd = ws.state["round"]
        taken = set(d.get("outdir") for d in ws.dispatches())
        records = []
        for a in actions:
            rel = self._new_outdir(rnd, a["module"], taken)
            taken.add(rel)
            r = {"round": rnd, "module": a["module"], "purpose": a["purpose"], "reason": a.get("reason"),
                 "outdir": rel, "brief": rel + "/brief.md", "runner": self.runner.name, "launches": 1,
                 "launch_mode": "first", "feedback": "feedback-1.md" if why else None, "launched_at": None,
                 "status": "pending"}
            if why:  # kept so feedback-1.md can be written again when a run is cut off before it was
                r["made"] = {"reason": why[0], "failing": list(why[1])}
            if "candidate_id" in a:
                r["candidate_id"] = a["candidate_id"]
            records.append(r)
        ids = ws.add_dispatches(records)
        for did in ids:
            d = ws.dispatch(did)
            self.prepare(ws, d)
            self.made_feedback(ws, d)
        launched = []
        try:
            for did in ids:
                self.launch(did)
                launched.append(did)
        except StoppedError:  # the ones not launched yet are closed too
            for did in ids:
                if did not in launched:
                    self._close_unlaunched(did)
            raise
        return {"dispatch_ids": ids}

    def made_feedback(self, ws, d):
        """feedback-1.md of a dispatch a decision made (why it was made), unless it is there already."""
        made = d.get("made")
        path = os.path.join(self.outdir(d), d.get("feedback") or "feedback-1.md")
        if made and d.get("launch_mode", "first") == "first" and not os.path.isfile(path):
            _write(path, self.feedback_text(ws, d, made.get("reason") or "", "made", made.get("failing") or ()))

    def _close_unlaunched(self, did, force=False):
        """Close a record whose agent was never launched because the run stopped (pending; with `force` also one
        just marked running whose runner was not called): status done, no launched_at, and the launch it was
        prepared for is not counted (launches, and the return / continuation / followup / answer / check-in it was
        for); a check-in's long job is terminated (stop_long_job). Returns True when it closed the record."""
        ws = self.ws()
        d = ws.dispatch(did)
        if not (d.get("status") == "pending" or (d.get("status") == "running" and (force or not d.get("launched_at")))):
            return False
        fields = {"status": "done", "launched_at": None, "pid": None, "capped": None, "ended_at": now_iso(),
                  "closed": "运行已停止（%s）：没有启动" % (ws.state["stopped"] or {}).get("reason"),
                  "launches": max(0, int(d.get("launches") or 1) - 1)}
        if long_job_open(ws, d):  # a check-in prepared, not launched: its long job runs on
            fields["long_job_stop"] = self.stop_long_job(ws, d)
            fields["closed"] += "；%s" % long_job_note(fields["long_job_stop"])
        counter = MODE_COUNTERS.get(d.get("launch_mode"))
        if counter:
            fields[counter] = max(0, int(d.get(counter) or 0) - 1)
        ws.update_dispatch(did, **fields)
        self.log("closed", dispatch_id=did, why="stopped", launched=False)
        return True

    def prepare(self, ws, d):
        """The output directory with brief.md and brief.json (built from the workspace as it is now)."""
        out = self.outdir(d)
        os.makedirs(out, exist_ok=True)
        brief = B.build_brief(ws, d["module"], d["id"])
        text = B.render_brief(brief)
        answers = DP.pending_answers(ws, d)
        if answers:  # the user's answers to questions this module asked before, not given to it yet
            brief["answers"] = [_answer_item(x) for x in answers]
            text = text.rstrip("\n") + "\n\n## 用户对本模块之前提问的回答\n\n" + _answer_lines(brief["answers"])
        if d.get("module") == "judge" and (F.dispatch_purpose(d) == "plan"
                                           or ws.state["rounds"][str(d.get("round"))]["decision"] is not None):
            every = DP.round_answers(ws, d.get("round"))
            if every:  # the plan sees what the user answered to every module this round
                brief["round_answers"] = [_answer_item(x) for x in every]
                text = text.rstrip("\n") + "\n\n## 本轮用户对各模块提问的回答\n\n" + _answer_lines(brief["round_answers"])
        _write(os.path.join(out, "brief.md"), text)
        if answers:
            DP.mark_delivered(ws, answers, "%s brief" % d["id"])
        # brief.json last and whole (a launch after an interruption rebuilds the brief when it is missing)
        _write(os.path.join(out, ".brief.json.tmp"), json.dumps(brief, indent=2, ensure_ascii=False) + "\n")
        os.replace(os.path.join(out, ".brief.json.tmp"), os.path.join(out, "brief.json"))
        return brief

    def prompt(self, d):
        """The prompt of the dispatch's next launch. First launch: where the module skill is and its full text,
        the brief, how to hand back, and the feedback file when a decision made the dispatch. A return to the
        session (or its continuation after a long job): the feedback file. A fresh session taking over: as a first
        launch, with the history file."""
        out = self.outdir(d)
        mode = d.get("launch_mode") or "first"
        if mode in SESSION_MODES:
            return _read(os.path.join(out, d["feedback"]))
        path = B.skill_path(d["module"])
        try:
            skill = _read(path)
        except OSError as e:
            raise RsiError("cannot read the module skill %s: %s" % (path, e))
        text = ("你是 RSI 运行里的模块 agent `%s`。下面先是这个模块的 skill 全文（文件 `%s`，skill 里的相对路径都相对它所在的"
                "目录 `%s`；全文已经在这里，不必再去读），然后是这次派出的简报。\n\n"
                "===== skill 全文开始 =====\n%s\n===== skill 全文结束 =====\n\n%s\n## 交回\n\n结构化的简报在 `%s`。%s\n"
                "写好 `handback.json` 就结束这次会话。\n"
                % (B.SKILLS[d["module"]], path, os.path.dirname(path), skill.rstrip("\n"),
                   _read(os.path.join(out, "brief.md")), os.path.join(out, "brief.json"), B.HANDBACK_FORMAT))
        if d.get("feedback"):
            text += "\n---\n\n" + _read(os.path.join(out, d["feedback"]), default="")
        return text

    def launch(self, did):
        """Launch the dispatch's agent as its record says (launch_mode, feedback, session). The record says it was
        launched (status running, launched_at, the new session's id) before the runner is called; until then it is
        pending."""
        ws = self.ws()
        d = ws.dispatch(did)
        if not os.path.isfile(os.path.join(self.outdir(d), "brief.json")):
            self.prepare(ws, d)  # interrupted after the record was written
        self.made_feedback(ws, d)
        prompt = self.prompt(d)
        mode = d.get("launch_mode") or "first"
        resume = d.get("session") if mode in SESSION_MODES else None
        new = None if resume else self.runner.new_session(d)
        fields = {"status": "running", "launched_at": now_iso(), "runner": self.runner.name}
        denied = B.denied_paths(ws, json.loads(_read(os.path.join(self.outdir(d), "brief.json"))))
        fields["denied"] = denied  # enforced by the claude runner only; never in the brief
        if getattr(self.runner, "enforces_deny", False):
            B.deny_rules(denied)  # RsiError before anything is recorded or launched: a path no safe rule can name
        if not resume:
            fields["session"] = new
        d = ws.update_dispatch(did, **fields)
        if ws.state["stopped"]:  # STATE as reloaded for that save: a stop recorded meanwhile starts nothing
            self._close_unlaunched(did, force=True)
            raise StoppedError("the run has stopped (%s) before %s was launched: nothing more is started"
                               % (ws.state["stopped"].get("reason"), did))
        stamp = F.launch_stamp(ws, d, int(d.get("launches") or 1))
        try:  # this launch's time on the output directory's clock (rsi.flow.marker_done); without it, the host's
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(fields["launched_at"] + "\n")
        except (OSError, TypeError):
            pass
        try:
            h = self.runner.launch(ws, d, prompt, resume_session=resume, new_session=new)
        except OSError as e:
            log = os.path.join(self.outdir(d), "agent-%d.log" % int(d.get("launches") or 1))
            h = launch_failed(self.runner, log, e)
        h.setdefault("runner", self.runner)
        if h.get("pid"):
            ws.update_dispatch(did, pid=h["pid"])
        self.handles[did] = h
        self.log("launch", dispatch_id=did, module=d["module"], launch=d.get("launches"), mode=mode,
                 runner=self.runner.name, resume_session=resume, session=new or resume, pid=h.get("pid"),
                 denied=denied, error=(h.get("result") or {}).get("error"))
        return h

    # ---- waiting for agents ------------------------------------------------------------------------

    def attach(self, ws, d):
        name = d.get("runner")
        runner = self.runner if name == self.runner.name else attach_runner(name, ws)
        h = runner.attach(ws, d)
        h.setdefault("runner", runner)
        self.handles[d["id"]] = h
        self.log("recover", dispatch_id=d["id"], runner=runner.name, pid=h.get("pid"), found="result" not in h)
        return h

    def wait(self, dids):
        """Poll the running agents of `dids` until at least one has ended (its record then says how); stop at
        once when only agents the engine does not wait for (manual) are left, and as soon as the run is stopped
        (STATE is read on every poll; between polls every second): the next step, the report, stops the agents.
        Every poll ends with tick (the pages, a GPU snapshot and a status line when they are due)."""
        while True:
            ws = self.ws()
            if ws.state["stopped"]:
                return {"stopped": True, "waiting_for": dids}, None
            pending, ended = [], []
            for did in dids:
                d = ws.dispatch(did)
                if d.get("status") not in ("pending", "running"):
                    continue
                h = self.handles.get(did)
                if h is None:
                    h = self.recover(ws, d)
                r = h["runner"].poll(h)
                if r is None:
                    pending.append((d, h))
                    continue
                self.handles.pop(did, None)
                ended.append(self.finish(did, r))
            if ended or not pending:
                return {"ended": ended}, None
            if not any(h.get("waits") for _d, h in pending):
                return {"waiting_for": [d["id"] for d, _h in pending]}, ("waiting_manual", "；".join(
                    "在等手动完成：%s（rsi-%s）在 %s 里写好产物和 handback.json 后，再运行 rsi run。"
                    % (d["id"], d.get("module"), self.outdir(d)) for d, _h in pending))
            self.tick(ws)
            self.sleep_unless_stopped(self.poll_s)

    def wait_long_job(self, checkin_at):
        """With check-ins on: wait until a long job's marker is there, `checkin_at` (UTC ISO) has come or the run is
        stopped (STATE and the markers are read every poll, at most every second, each poll ending with tick);
        returns what ended the wait."""
        due = parse_utc(checkin_at)
        due = due.timestamp() if due is not None else time.time()
        while True:
            ws = self.ws()
            if ws.state["stopped"]:
                return {"stopped": True}
            done = [d["id"] for d in ws.dispatches() if d.get("status") == "long_job" and F.marker_done(ws, d)]
            if done:
                return {"marker": done}
            rest = due - time.time()
            if rest <= 0:
                return {"checkin_due": checkin_at}
            self.tick(ws)
            time.sleep(max(0.0, min(due - time.time(), self.poll_s, 1.0)))

    def sleep_unless_stopped(self, seconds):
        end = time.time() + seconds
        while True:
            rest = end - time.time()
            if rest <= 0 or self.stopped():
                return
            time.sleep(min(rest, 1.0))

    def recover(self, ws, d):
        """A handle for a pending or running record this process did not launch: pending (not marked launched) →
        launch it; marked launched but its runner finds no trace of a start (cut off between the mark and the
        start) → launch it again, same launch, no return counted; else find the agent again (attach)."""
        did = d["id"]
        if d.get("status") == "pending" or not d.get("launched_at"):
            return self.launch(did)
        name = d.get("runner")
        runner = self.runner if name == self.runner.name else attach_runner(name, ws)
        if runner.started(ws, d) is False:
            self.log("relaunch", dispatch_id=did, launch=d.get("launches"), why="cut off before the agent started")
            ws.update_dispatch(did, status="pending", launched_at=None)
            return self.launch(did)
        return self.attach(ws, d)

    def finish(self, did, r):
        """An agent has ended: record whether it handed back (status handed_back or long_job) or failed; a
        session the runner says cannot be resumed gets a fresh one instead."""
        ws = self.ws()
        d = ws.dispatch(did)
        if r.get("resume_failed") and d.get("launch_mode") in SESSION_MODES:
            self.log("resume_failed", dispatch_id=did, session=d.get("session"), exit_code=r.get("exit_code"),
                     log=r.get("log"))
            self.fresh(ws, d, "会话 %s 续不上（退出码 %s）" % (d.get("session"), r.get("exit_code")))
            return {"dispatch_id": did, "status": "running", "fresh_session": True}
        hb = B.read_handback(self.outdir(d))
        fields = {"ended_at": now_iso(), "exit_code": r.get("exit_code"), "pid": None}
        if r.get("session") and r["session"] != d.get("session"):
            if d.get("session"):
                self.log("session_mismatch", dispatch_id=did, recorded=d.get("session"), reported=r["session"])
                print("warning: %s: the agent reports session %s, the record had %s; keeping the reported one"
                      % (did, r["session"], d.get("session")), file=sys.stderr)
            fields["session"] = r["session"]
        if hb["ok"] and hb["status"] == "ok":
            fields["status"] = "handed_back"
        elif hb["ok"] and hb["status"] == "long_job":
            # A long job runs (again): even under a marker path seen at its end before, it is open (marker_seen goes)
            # and, being new, started in this launch.
            fields.update(status="long_job", marker=hb["marker"], marker_seen=None)
            same = d.get("marker") == hb["marker"] and d.get("marker_seen") != hb["marker"]  # the same job runs on
            if not (d.get("long_job_since") and same):
                # the launch that started this long job: its time, and the number of its stamp (rsi.flow.marker_done)
                fields.update(long_job_since=d.get("launched_at"), long_job_launch=int(d.get("launches") or 1))
        else:
            errors = []
            if r.get("error"):
                errors.append(r["error"])
            if r.get("lost"):
                errors.append("rsi run 重新运行时，这个 agent 的进程已不在")
            if r.get("timed_out"):
                errors.append("agent 到了时间上限，已被终止")
            if hb["ok"]:
                errors.append("rsi-%s 交回的 status 是 failed：%s" % (d.get("module"), hb["notes"] or "（notes 为空）"))
            else:
                errors += hb["errors"]
            if r.get("exit_code") not in (0, None):
                errors.append("agent 的退出码是 %s" % r["exit_code"])
            if r.get("log"):
                errors.append("agent 的输出在 %s" % r["log"])
            fields.update(status="failed", errors=errors)
        ws.update_dispatch(did, **fields)
        start = parse_utc(d.get("launched_at"))
        self.log("agent_end", dispatch_id=did, status=fields["status"], exit_code=r.get("exit_code"),
                 session=fields.get("session", d.get("session")), log=r.get("log"), cost_usd=r.get("cost_usd"),
                 seconds=None if start is None else int(time.time() - start.timestamp()))
        return {"dispatch_id": did, "status": fields["status"]}

    # ---- hand-backs --------------------------------------------------------------------------------

    def process(self, ws, d):
        """Register what a hand-back lists (see the module docstring), record the outcome in the dispatch and
        decide handback_refused when registration refused something; a hand-back with questions registers nothing
        and asks the user (ask_handback), or, once they are answered, goes back to the session with the answers;
        one whose candidates would go over the compute budget without the over_budget approval registers nothing
        and asks for it (hold_budget). Returns (result, stop or None)."""
        did, out = d["id"], self.outdir(d)
        hb = B.read_handback(out)
        if d.get("status") == "long_job":
            marker = F.marker_path(ws, d)
            text = _read(marker, default="")
            first = (text.splitlines() or [""])[0].strip()
            if first != "ok":  # the job has ended (marker_seen), and failed
                ws.update_dispatch(did, status="failed", ended_at=now_iso(), marker_seen=d.get("marker"),
                                   errors=["长任务失败：完成标记 %s 写的是：%s" % (marker, text.strip()[:2000] or "（空）")])
                return {"dispatch_id": did, "status": "failed", "marker": first}, None
            if d.get("marker_seen") != d.get("marker"):  # this job has ended: no stop or close looks for it again
                d = ws.update_dispatch(did, marker_seen=d.get("marker"))
        elif not hb["ok"]:
            ws.update_dispatch(did, status="failed", ended_at=now_iso(), errors=hb["errors"])
            return {"dispatch_id": did, "status": "failed"}, None
        if hb["ok"] and hb["questions"]:
            # The module asks the user something: nothing it handed back is registered before the answer, which goes
            # back to the same session (rsi.flow gives that resume, mode answer); its next hand-back is registered.
            if not F.handback_answers(ws, d):
                return self.ask_handback(ws, d, hb)
            if d.get("status") != "long_job" or hb["artifacts"]:
                return self.resume(ws, d, "你上次交回时提了问题，用户已经回答（见下）。那次交回的产物都还没有登记：按回答修改"
                                          "或确认后，重新交回要登记的全部产物。", mode="answer"), None
            # (a long job that listed nothing: its continuation below carries the answers)
        if d.get("status") == "long_job":
            if not (hb["ok"] and hb["artifacts"]):
                cap, done = flow_settings(ws.task())["max_continues"], int(d.get("continued") or 0)
                if done >= cap:
                    ws.update_dispatch(did, status="failed", ended_at=now_iso(), capped="max_continues",
                                       errors=["长任务已结束（完成标记 %s 写的是 ok），但这次派出已续跑 %d 次，到了上限 "
                                               "flow.max_continues = %d：不再自动续跑会话" % (marker, done, cap)])
                    self.log("continue_capped", dispatch_id=did, continued=done, max_continues=cap)
                    return {"dispatch_id": did, "status": "failed", "capped": "max_continues"}, None
                return self.resume(ws, d, "长任务已结束：完成标记 `%s` 的第一行是 ok。" % F.marker_path(ws, d),
                                   mode="continue"), None
        # The candidates it lists would take the compute used over the budget, and the round has no over_budget
        # approval: nothing of it is registered before the user's answer (registration would refuse them, and a
        # return would only train again). Approved, this same hand-back is registered below.
        launch = int(d.get("launches") or 1)
        fresh = self.new_candidates(ws, d, hb["artifacts"])
        over = ws.budget_overrun([src for src, _name in fresh])
        if over is not None:
            names = dict(fresh)
            hold = self.hold_budget(ws, d, hb, dict(over, dirs=[names[src] for src in over["dirs"]]))
            if hold is not None:
                return hold
            ws = self.ws()  # approved meanwhile (another process): register on STATE as it is now
            d = ws.dispatch(did)
        held = d.get("held_budget") or {}
        if int(held.get("launch") or 0) == launch and ws.has_approval("over_budget"):
            self.log("budget_released", dispatch_id=did, launch=launch, by="approve")
        registered, refused, stopped = self.register(ws, d, hb["artifacts"])
        d = ws.dispatch(did)
        registered = [dict(x, launch=launch) for x in registered]
        refused = [dict(x, launch=launch) for x in refused]
        merged = list(d.get("registered") or [])
        known = set(x.get("id") for x in merged)
        merged += [x for x in registered if x["id"] not in known]
        fields = {"registered": merged, "refused": list(d.get("refused") or []) + refused, "ended_at": now_iso()}
        if stopped:  # the run was stopped meanwhile: what is left is not registered, and that is no refusal
            err, left = stopped
            note = ""
            if long_job_open(ws, d):  # e.g. a check-in handed back a part while the job runs on (as halt_if_stopped)
                fields["long_job_stop"] = self.stop_long_job(ws, d)
                note = "；%s" % long_job_note(fields["long_job_stop"])
            ws.update_dispatch(did, status="done", capped=None, closed="运行已停止（%s）：交回的 %s 没有登记%s"
                               % ((ws.state["stopped"] or {}).get("reason"), "、".join(left), note), **fields)
            self.log("closed", dispatch_id=did, why="stopped", unregistered=left)
            raise err
        if refused:
            fields.update(status="failed", errors=["子目录 %s（%s）登记被拒：%s" % (x["dir"], x.get("kind"), x["error"])
                                                   for x in refused])
        else:
            fields.update(status="done", errors=[])
        d = ws.update_dispatch(did, **fields)
        result = {"dispatch_id": did, "status": fields["status"], "registered": [x["id"] for x in registered],
                  "refused": [x["dir"] for x in refused]}
        if refused:  # decided here, on the refusal's own decision point (rsi.flow gives the same decide on a rerun)
            self.halt_if_stopped(d)  # STATE as it is now: a stop recorded meanwhile decides nothing
            out, stop = DP.apply(self, ws, F.refused_decision(ws, d))
            result["decision"] = out.pop("decision")
            result["carried_out"] = out
            return result, stop
        return result, None

    def ask_handback(self, ws, d, hb):
        """The hand-back of dispatch `d` asks the user something: STATE waiting (once), the record's `asked` (the
        launch, the question's id, the questions); what it handed back stays unregistered. Returns (result, stop)."""
        did, launch = d["id"], int(d.get("launches") or 1)
        w = ws.waiting()
        if not (w and w.get("what") == "handback" and w.get("ref") == did):
            w = ws.set_waiting({"kind": "question", "what": "handback", "ref": did,
                                "question": "rsi-%s（%s）交回时有要你决定的事：%s（它这次交回的产物%s要等你回答后再登记："
                                            "回答会送回它的同一个会话，它按回答修改后重新交回。）"
                                            % (d.get("module"), did, "；".join(hb["questions"]),
                                               " %s " % "、".join(hb["artifacts"]) if hb["artifacts"] else "（这次没有）"),
                                "options": ["已回答"]})
        ws.update_dispatch(did, asked={"launch": launch, "question_id": w.get("id"), "questions": list(hb["questions"])})
        self.log("asked", dispatch_id=did, launch=launch, question_id=w.get("id"), held=list(hb["artifacts"]))
        return ({"dispatch_id": did, "status": d.get("status"), "questions": hb["questions"], "held": hb["artifacts"]},
                ("waiting_user", DP.ask_reason(w)))

    def new_candidates(self, ws, d, names):
        """[(real path, name)] of the subdirectories `names` of dispatch `d`'s output directory, in order, that are
        not registered yet and that registration would take as candidates (_kind_of: a candidate of the dispatch's
        module): the ones the budget is weighed against before registering (Workspace.budget_overrun)."""
        out = os.path.realpath(self.outdir(d))
        known = set(e["source"] for e in ws.entries())
        found = []
        for name in names:
            src = os.path.realpath(os.path.join(out, name))
            if src not in known and self._kind_of(src, d)[:2] == ("candidate", None):
                found.append((src, name))
        return found

    def hold_budget(self, ws, d, hb, over):
        """The candidates dispatch `d`'s hand-back `hb` lists would take the compute used over the budget (`over`:
        Workspace.budget_overrun, its dirs the hand-back's subdirectory names) and the round has no over_budget
        approval: what it handed back stays unregistered; STATE waiting asks over_budget (kind approval, ref None,
        dispatch_id d: `rsi answer approve` and `rsi approve <ws> over_budget` both answer it) and the record's
        held_budget notes the launch, in one save; a rerun keeps the same question and logs budget_held once.
        The overrun is weighed again on STATE as reloaded under the lock: a stop recorded meanwhile raises
        StoppedError (the run goes on to its report), an approval given meanwhile holds nothing (returns None: the
        caller registers the hand-back). Returns (result, stop), or None."""
        did = d["id"]
        out = os.path.realpath(self.outdir(d))
        ws = self.ws()
        with ws.transaction():
            ws._refuse_if_stopped("holding the hand-back of %s for over_budget" % did)
            names = dict((os.path.realpath(os.path.join(out, n)), n) for n in over["dirs"])
            over = ws.budget_overrun(list(names))
            if over is None:
                return None
            over = dict(over, dirs=[names[src] for src in over["dirs"]])
            d = ws.dispatch(did)
            launch = int(d.get("launches") or 1)
            w = ws.waiting()
            if not (w and (w.get("kind"), w.get("what"), w.get("dispatch_id")) == ("approval", "over_budget", did)):
                question = ("over_budget：rsi-%s（%s）这次交回的候选（%s）要 %s 小时算力，加上已用 %s 小时，超过预算 %s "
                            "小时。approve：批准超预算，登记这次交回的产物，本轮接着做到判定为止，判定后因算力超过预算停止运行；"
                            "stop：现在停止运行（同 rsi stop），这次交回的候选不登记。"
                            % (d.get("module"), did, "、".join(over["dirs"]), _num(over["cost"]), _num(over["used"]),
                               _num(over["budget"])))
                w = ws.set_waiting({"kind": "approval", "what": "over_budget", "ref": None, "dispatch_id": did,
                                    "question": question, "options": list(F.ASK_OPTIONS["over_budget"])})
            held = {"launch": launch, "question_id": w.get("id"), "dirs": list(over["dirs"]), "cost": over["cost"],
                    "used": over["used"], "budget": over["budget"]}
            new = d.get("held_budget") != held
            if new:
                ws.update_dispatch(did, held_budget=held)
        if new:
            self.log("budget_held", dispatch_id=did, **held)
        return ({"dispatch_id": did, "status": d.get("status"), "held_budget": held, "held": list(hb["artifacts"])},
                ("waiting_user", DP.ask_reason(w)))

    def _kind_of(self, src, d):
        """(kind, error, uses the testset): the kind a hand-back subdirectory's manifest names (task for one with
        task.json and no manifest.json: only rsi-data hands one back), why it cannot be registered, and whether its
        registration needs a use_testset approval (test predictions, or an evaluation with uses_testset)."""
        if not os.path.isdir(src):
            return None, "%s 不是一个目录" % src, False
        if not os.path.exists(os.path.join(src, "manifest.json")) and os.path.isfile(os.path.join(src, "task.json")):
            if d.get("module") != TASK_MODULE:
                return "task", "任务新版本（task.json）只有 rsi-data 能交回（补全切分规则），这是 rsi-%s 的交回" \
                    % d.get("module"), False
            return "task", None, False
        try:
            with open(os.path.join(src, "manifest.json"), encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError) as e:
            return None, "读不了 %s：%s" % (os.path.join(src, "manifest.json"), e), False
        kind = m.get("kind") if isinstance(m, dict) else None
        if kind not in KIND_ORDER or kind == "task":  # a task version has no manifest
            return kind, "manifest 的 kind %r 不是模块能交回的产物类别（%s）" % (kind, "、".join(KIND_ORDER)), False
        if m.get("producer") != d.get("module"):
            return kind, "manifest 的 producer 是 %r，但这是 rsi-%s 的交回" % (m.get("producer"), d.get("module")), False
        testset = (kind == "delivery" and "predictions_test" in (m.get("files") or {})) or \
            (kind == "evaluation" and m.get("uses_testset") is True)
        return kind, None, testset

    def consent(self, ws, d, name):
        """Before registering a milestone artifact: the use_testset approval it uses, quoting the user's milestone
        consent (written here, right before the registration, so each of the two uses has its own)."""
        if ws.has_approval("use_testset"):
            return
        yes = DP.milestone_consent(ws)
        if yes is None:
            return  # no consent recorded (e.g. a run by hand): registration says what is missing
        by = yes.get("by") or "user"
        quote = yes.get("quote") or yes.get("reason") or "同意里程碑发布"
        ws.approve("use_testset", by, quote)
        self.log("approval", what="use_testset", by=by, quote=quote, dispatch_id=d["id"], dir=name,
                 consent_at=yes.get("at"))

    def register(self, ws, d, names):
        """Register the listed subdirectories of the dispatch's output directory in KIND_ORDER (the order listed
        within a kind); one already registered (an earlier, interrupted run) is not registered again. A task version
        registers only as a fill of the split rule (Workspace.register fill_only: checked on the staged copy, TASK.md
        byte for byte, so an open task_change approval is never used); a refusal says so (TASK_REFUSED). A
        subdirectory whose kind cannot be told comes first; once it, or one of DATA_CHAIN, is refused, the later ones
        of DATA_CHAIN are refused without registering (each names the first refusal); a record still registers. Returns
        (registered [{dir, kind, id}], refused [{dir, kind, error}], stopped): stopped is (StoppedError, the
        subdirectories left unregistered) when the run was stopped meanwhile, else None."""
        out = os.path.realpath(self.outdir(d))
        sources = dict((e["source"], e) for e in ws.entries())
        todo = []
        for i, name in enumerate(names):
            src = os.path.realpath(os.path.join(out, name))
            e = sources.get(src)
            kind, error, testset = (e["kind"], None, False) if e else self._kind_of(src, d)
            rank = KIND_ORDER.index(kind) if kind in KIND_ORDER else -1  # an unknown kind first: it may be the corpus
            todo.append((rank, i, name, src, kind, error, e["id"] if e else None, testset))
        registered, refused, stopped, left = [], [], None, []
        blocked = None  # the first refusal of DATA_CHAIN (or of an unknown kind): the later ones of it wait
        for _rank, _i, name, src, kind, error, aid, testset in sorted(todo, key=lambda t: (t[0], t[1])):
            if stopped is not None and aid is None and kind != "record":
                left.append(name)  # after a stop only records may still be registered
                continue
            if aid is None and error is None and blocked is not None and kind in DATA_CHAIN:
                error = "本次交回的 %s被拒，后面依赖它的集合不登记：先改好它，再一起重新交回" % blocked
            if aid is None and error is None:
                try:
                    if testset:
                        self.consent(ws, d, name)
                    aid = ws.register(kind, src, fill_only=True) if kind == "task" else ws.register(kind, src)
                except StoppedError as e:
                    stopped = e
                    left.append(name)
                    continue
                except RsiError as e:
                    # the cause on the refusal's first line: the decision's reason quotes only that line
                    error = (TASK_REFUSED if kind == "task" else "") + "\n".join(e.errors)
                else:
                    self.log("register", dispatch_id=d["id"], dir=name, kind=kind, id=aid, warnings=list(ws.warnings))
            if error is not None:
                refused.append({"dir": name, "kind": kind, "error": error})
                if blocked is None and (kind in DATA_CHAIN or kind not in KIND_ORDER):
                    blocked = "%s（子目录 %s）" % (kind or "类别不明的产物", name)
            else:
                registered.append({"dir": name, "kind": kind, "id": aid})
        return registered, refused, (stopped, left) if stopped is not None else None

    # ---- returns -----------------------------------------------------------------------------------

    def send_to(self, ws, choice, purpose, did, reason, failing=()):
        """A decision chose module `choice`: back to its session of this round made for `purpose` (the decision's
        dispatch when it is one), else a new dispatch. Either way the feedback names the gates of the `failing`
        candidates."""
        module = choice
        target = None
        if did:
            d = ws.dispatch(did)
            if d.get("module") == module and F.dispatch_purpose(d) == purpose:
                target = d
        if target is None:
            same = [d for d in ws.dispatches(rnd=ws.state["round"], module=module) if F.dispatch_purpose(d) == purpose]
            target = same[-1] if same else None
        if target is not None:
            return self.resume(ws, target, reason, failing=failing)
        return self.dispatch(ws, [{"module": module, "purpose": purpose, "reason": reason}], why=(reason, failing))

    def refresh_brief(self, ws, d, n):
        """Rebuild the dispatch's brief from the workspace as it is now, keeping the previous one as brief-<n>.md
        and brief-<n>.json (n: the launch it was for); returns the changes in words (ids, inputs, attempts,
        budget), [] when none."""
        out = self.outdir(d)
        try:
            with open(os.path.join(out, "brief.json"), encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = {}
        for ext in ("md", "json"):
            src = os.path.join(out, "brief.%s" % ext)
            if os.path.exists(src):
                os.replace(src, os.path.join(out, "brief-%d.%s" % (n, ext)))
        new = self.prepare(ws, d)
        return brief_changes(old, new)

    def _archive(self, out, n):
        """Keep the current handback.json as handback-<n>.json (the hand-back of launch n)."""
        src = os.path.join(out, "handback.json")
        if not os.path.exists(src):
            return
        dst, k = os.path.join(out, "handback-%d.json" % n), 1
        while os.path.exists(dst):
            k += 1
            dst = os.path.join(out, "handback-%d-%d.json" % (n, k))
        os.replace(src, dst)

    def _gates(self, ws, ids):
        """Lines on the gates the artifacts `ids` failed, with values and thresholds."""
        lines = []
        for aid in ids:
            e, m = ws.entry(aid), ws.manifest(aid)
            if e["kind"] == "trainset" and m["quality"]["passed"] is not True:
                checks = [c for c in m["quality"].get("checks", []) if c.get("passed") is not True] \
                    or m["quality"].get("checks", [])
                lines.append("- 训练集 %s 没过质量门：%s" % (aid, "；".join(
                    "%s = %s（阈值 %s，%s）" % (c.get("name"), c.get("value"), c.get("threshold"),
                                           "通过" if c.get("passed") else "没过") for c in checks) or "没有检查项"))
            elif e["kind"] == "candidate" and m["train_gate"]["passed"] is not True:
                lines.append("- 候选 %s（%s）没过训练门：%s（验证集指标 %s，对照组 %s）" % (
                    aid, e["producer"], m["train_gate"].get("detail"),
                    json.dumps(m.get("metrics_val"), ensure_ascii=False),
                    json.dumps(m.get("baselines_val"), ensure_ascii=False)))
        return lines

    def _last_notes(self, out, d):
        """The notes of the dispatch's last hand-back: handback.json, or the copy kept as handback-<n>.json when a
        later launch was being prepared (a run cut off after keeping it)."""
        path = os.path.join(out, "handback.json")
        if not os.path.exists(path):
            path = os.path.join(out, "handback-%d.json" % int(d.get("launches") or 1))
        try:
            notes = json.loads(_read(path, default="")).get("notes")
        except (ValueError, AttributeError):
            return ""
        return notes if isinstance(notes, str) else ""

    def feedback_text(self, ws, d, reason, mode, failing=(), changes=(), registered=()):
        """What a session is told: why it is sent back (mode resume), continued after its long job (continue),
        asked for the second part of its hand-back (followup; with a return, the same: what the dispatch
        `registered` since the round's latest search, each hand-back subdirectory and its id), given the user's answers to its questions (answer), checked in on during its long
        job (checkin) or made by a decision (made); what changed in its brief since its last launch (`changes`); the user's
        answers to its questions; the gates its artifacts and the `failing` artifacts failed, the dispatch's errors
        (refusals included) and the last hand-back's notes; then what to do."""
        out = self.outdir(d)
        if mode == "continue":
            head = ["# 长任务已结束（rsi-%s，派出 %s）" % (d.get("module"), d["id"]), "", reason]
            todo = ("读长任务的结果，把产物写到输出目录 `%s` 下的子目录（每个产物一个），然后重新写 `handback.json` 交回"
                    "（status ok）。" % out)
        elif mode == "answer":
            head = ["# 用户回答了你的问题（rsi-%s，派出 %s；不算退回）" % (d.get("module"), d["id"]), "", "原因：%s" % reason]
            todo = ("按用户的回答修改或确认产物：上次交回的子目录都还没有登记，可以原地修改，也可以写到输出目录 `%s` 下新的子目录；"
                    "然后重新写 `handback.json`，列出要登记的全部子目录；上一次的 handback.json 已改名保留。" % out)
        elif mode == "followup":
            head = ["# 继续：交回的第二部分（rsi-%s，派出 %s；不算退回）" % (d.get("module"), d["id"]), "",
                    "原因：%s" % reason]
            todo = ("写第二部分的产物，放到输出目录 `%s` 下新的子目录（已经登记过的子目录不能再用），然后重新写 "
                    "`handback.json` 交回；上一次的 handback.json 已改名保留。" % out)
        elif mode == "checkin":
            head = ["# 定时检查（rsi-%s，派出 %s，第 %d 次；不算退回）" % (d.get("module"), d["id"],
                                                             int(d.get("checkins") or 0) + 1), "", "原因：%s" % reason]
            todo = ("检查长任务和环境：它的进程还在不在（`<marker>.pid` 第一行的进程号）、日志是否在进展、有没有报错、用到的 "
                    "GPU 是否被别人占用、磁盘够不够。有异常就先处理（按你的 skill：例如结束出问题的部分、换卡重跑）。长任务还在"
                    "运行就以 long_job 交回：没有重新启动长任务，就交回同一个完成标记（现在是 `%s`）；重新启动了（新的长任务），"
                    "就用新的完成标记路径，交回新的标记。不要等待长任务结束。上一次的 handback.json 已改名保留。"
                    % (F.marker_path(ws, d) or "（未给出）"))
        elif mode == "made":
            head = ["# 为什么派你（rsi-%s，派出 %s）" % (d.get("module"), d["id"]), "", "决策点给出的原因：%s" % reason]
            todo = "按简报工作，针对上面的原因改进。产物写到输出目录 `%s` 下的子目录，然后写 `handback.json` 交回。" % out
        else:
            head = ["# 退回给 rsi-%s（派出 %s，第 %d 次退回）" % (d.get("module"), d["id"], int(d.get("returns") or 0) + 1),
                    "", "原因：%s" % reason]
            todo = ("按上面的原因改好。新产物写到输出目录 `%s` 下新的子目录（已经登记过的子目录不能再用），然后重新写 "
                    "`handback.json` 交回；上一次的 handback.json 已改名保留。" % out)
        lines = head
        if registered and mode in ("followup", "resume"):
            lines += ["", "## 已登记的产物（子目录 → 登记 id；这次派出在本轮最近一次 search 之后登记的，各次启动合在一起）", ""]
            lines += ["- `%s` → %s（%s）" % (x["dir"], x["id"], x["kind"]) for x in registered]
            lines += ["", "同样的对应关系也在 `%s`（[{\"dir\", \"kind\", \"id\"}]）。"
                      % os.path.join(out, registered_file(int(d.get("launches") or 1) + 1))]
        if changes:
            lines += ["", "## 简报有变化", "", "工作区在你上次启动之后变了，简报已按现在的工作区重写（`%s`；上一份留作 "
                      "`brief-%d.md`）。新产物引用新的 id：" % (os.path.join(out, "brief.md"), int(d.get("launches") or 1)),
                      ""] + ["- %s" % c for c in changes]
        if mode != "made":
            answers = DP.question_answers(ws, d)
            if answers:
                lines += ["", "## 你上次交回时提的问题，用户的回答", ""]
                lines += ["- %s\n  用户（%s）的回答：%s" % (x.get("question"), x.get("by"), x.get("quote")) for x in answers]
        ids = [e["id"] for e in F.products(ws, d)] if mode != "made" else []
        ids += [a for a in failing if a not in ids]
        gates = self._gates(ws, ids)
        if gates:
            lines += ["", "## 没过的门", ""] + gates
        if mode != "made":
            if d.get("errors"):
                lines += ["", "## 错误", ""] + ["- %s" % x for x in d["errors"]]
            notes = self._last_notes(out, d)
            if notes:
                lines += ["", "## 上次交回的 notes", "", notes]
        lines += ["", "## 要做的事", "", todo, "简报在 `%s`。" % os.path.join(out, "brief.md"), ""]
        return "\n".join(lines)

    def resume(self, ws, d, reason, mode="resume", failing=()):
        """Go on in the dispatch's session with feedback-<n>.md: sent back (mode "resume": returns + 1), continued
        after its long job ("continue": continued + 1), asked for the second part of its hand-back ("followup":
        followups + 1, no return; registered-<n>.json lists what the first part registered), given the user's
        answers to its questions ("answer": answered + 1, no return) or checked in on during its long job
        ("checkin": checkins + 1, no return); without a session, a fresh one takes over. Nothing on a stopped
        run."""
        ws = self.halt_if_stopped(d, "不再续")
        d = ws.dispatch(d["id"])
        did, out = d["id"], self.outdir(d)
        n = int(d.get("launches") or 1) + 1
        name = "feedback-%d.md" % n
        changes = self.refresh_brief(ws, d, n - 1)  # the brief as the workspace is now (a new trainset, a refresh...)
        answers = DP.question_answers(ws, d)
        registered = registered_since_search(ws, d) if mode in ("followup", "resume") else []
        if mode in ("followup", "resume"):
            _write(os.path.join(out, registered_file(n)), json.dumps(registered, indent=2, ensure_ascii=False) + "\n")
        _write(os.path.join(out, name), self.feedback_text(ws, d, reason, mode, failing, changes=changes,
                                                           registered=registered))
        DP.mark_delivered(ws, answers, "%s launch %d" % (did, n))
        self._archive(out, n - 1)
        fields = {"status": "pending", "launches": n, "launch_mode": mode, "feedback": name, "launched_at": None,
                  "ended_at": None, "exit_code": None, "pid": None, "capped": None}
        counter = MODE_COUNTERS[mode]
        fields[counter] = int(d.get(counter) or 0) + 1
        d = ws.update_dispatch(did, **fields)
        self.log(MODE_EVENTS[mode], dispatch_id=did, mode=mode, launch=n, session=d.get("session"), feedback=name)
        if not d.get("session"):
            return self.fresh(ws, d, "派出记录里没有会话 id")
        self.launch(did)
        return {"dispatch_id": did, "mode": mode, "launch": n}

    def history_text(self, d, why):
        out = self.outdir(d)
        lines = ["# 接手说明：之前的会话续不上（%s）" % why, "",
                 "你接手 RSI 运行里 rsi-%s 的一次派出（%s）。之前的会话做过下面这些；输出目录 `%s` 里的文件都还在。"
                 % (d.get("module"), d["id"], out), "", "## 上次的简报", "",
                 _read(os.path.join(out, "brief.md"), default="（没有简报）"), "## 上次的交回", ""]
        backs = sorted(glob.glob(os.path.join(out, "handback-*.json")), key=os.path.getmtime)
        if not backs:
            lines.append("（没有交回）")
        for p in backs:
            lines += ["`%s`：" % os.path.basename(p), "", "```json", _read(p, default="").strip(), "```", ""]
        if d.get("feedback"):
            lines += ["", "## 这次要做的", "", _read(os.path.join(out, d["feedback"]), default="")]
        return "\n".join(lines) + "\n"

    def fresh(self, ws, d, why):
        """A fresh session takes the dispatch over: history-<n>.md and a rebuilt brief, then the launch."""
        ws = self.halt_if_stopped(d, "不再新开会话")
        d = ws.dispatch(d["id"])
        did, out = d["id"], self.outdir(d)
        n = int(d.get("launches") or 1) + (1 if d.get("launched_at") else 0)
        name = "history-%d.md" % n
        _write(os.path.join(out, name), self.history_text(d, why))
        self._archive(out, n - 1)
        self.refresh_brief(ws, d, n - 1)
        ws.update_dispatch(did, status="pending", launches=n, launch_mode="fresh", feedback=name, session=None,
                           launched_at=None, ended_at=None, exit_code=None, pid=None)
        self.log("fresh_session", dispatch_id=did, launch=n, why=why, history=name)
        self.launch(did)
        return {"dispatch_id": did, "mode": "fresh", "launch": n}

    # ---- the run log and the pages -----------------------------------------------------------------

    def log(self, event, **fields):
        log_event(self.root, event, **fields)

    def write_pages(self, now=None):
        """Rewrite the progress pages (unless pages=False); a failure only warns. pages_at becomes `now` (default
        clock())."""
        if not self.pages:
            return
        self.pages_at = self.clock() if now is None else now
        try:
            from rsi.dashboard import write_dashboard
            write_dashboard(self.ws())
        except Exception as e:  # the pages are a convenience: never stop the run for them
            self.warn("dashboard not updated", e)

    # ---- while waiting: refresh and status lines ----------------------------------------------------

    def tick(self, ws, now=None):
        """Called on every poll while the engine waits (`now`: clock() seconds). When flow.page_refresh_seconds have
        passed since the pages were written: a GPU snapshot (rsi.gpus, to <ws>/work/gpus.json), then the pages; when
        flow.status_minutes have passed since the last status line: one more (a heartbeat). Each only warns when it
        fails (a snapshot whose status is error too; missing nvidia-smi is no failure)."""
        if not self.pages and self.status is None:
            return
        now = self.clock() if now is None else now
        try:
            fs = flow_settings(ws.task())
        except Exception as e:
            self.warn("flow settings not read", e)
            return
        if self.pages and now - self.pages_at >= fs["page_refresh_seconds"]:
            try:
                snap = G.snapshot()
                G.write_snapshot(self.root, snap)
                if snap.get("status") == "error":
                    self.warn("GPU snapshot failed", snap.get("error") or "?")
            except Exception as e:
                self.warn("GPU snapshot failed", e)
            self.write_pages(now)
        if self.status is not None and now - self.said_at >= fs["status_minutes"] * 60:
            self.say(ws, heartbeat=True, now=now)

    def say(self, ws, heartbeat=False, now=None):
        """Print the status line (rsi.live.now_line) to the status stream when the state changed since the last one
        (rsi.live.now_key; the first call always prints) and its words did too, or for a heartbeat; said_at becomes
        `now` (default clock()). Nothing without a status stream; a failure only warns, and a stream that cannot be
        written to (a broken pipe) gets no more lines."""
        if self.status is None:
            return
        try:
            key = L.now_key(ws)
            if key == self.said_key and not heartbeat:
                return
            line = L.now_line(ws)
            self.said_key = key
            if line == self.said_line and not heartbeat:  # the same words as the last line: nothing new
                return
        except Exception as e:
            self.warn("status line failed", e)
        else:
            try:
                print(line, file=self.status, flush=True)
                self.said_line = line
            except (OSError, ValueError) as e:  # a closed or broken stream (a pipe whose reader is gone): no more lines
                self.status = None
                self.warn("status line failed", e)
        self.said_at = self.clock() if now is None else now

    def warn(self, kind, error=None):
        """`warning: <kind>[: <error>]` on stderr, once in a run per kind and kind of error (its type and errno, not
        its text, which may name a temp file). Never raises: a broken stderr does not stop the run."""
        key = (kind,) if error is None else (kind, type(error).__name__, getattr(error, "errno", None))
        if key in self.warned:
            return
        self.warned.add(key)
        try:
            print("warning: %s" % (kind if error is None else "%s: %s" % (kind, error)), file=sys.stderr)
        except (OSError, ValueError):
            pass
