"""The interface rsi.engine launches module agents through, and helpers the runners share.

A runner starts one agent for a dispatch record and reports when it has ended:

  new_session(dispatch) -> str | None
        A session id for a new session, made up before the launch so the engine can record it first (None: the
        runner has no sessions, or learns the id only from the agent).
  launch(ws, dispatch, prompt, resume_session=None, new_session=None) -> handle
        Start the agent and return at once. With resume_session
        the agent goes on in that session, else it starts session new_session. The handle is a dict the engine
        keeps in memory: "runner" (the runner that polls it), "waits" (False: the engine does not block on it but
        stops and comes back later, the manual runner), "session" when known at launch, "pid" when an OS
        process runs it. A launch that cannot start (e.g. the program is missing) gives a handle whose result
        carries "error" (launch_failed), never an exception.
  poll(handle) -> None | result
        None while the agent runs; then {"exit_code": int or None, "session": str or None, "log": path or None,
        "resume_failed": the session given cannot be resumed, "timed_out": the runner stopped the agent at its
        time limit, "lost": the agent was launched by an earlier `rsi run` and its process was already gone,
        "error": why it could not start (only then)}.
  attach(ws, dispatch) -> handle
        A handle for a dispatch an earlier `rsi run` launched (after that process was killed or the machine
        restarted): the runner finds the agent again if it can, else the handle is already finished ("lost").
  stop(handle) -> bool
        Stop the agent of a handle (the run was stopped): True when a live agent this runner can vouch for (its
        own process, or an attached one that is still its program) was terminated.
  started(ws, dispatch) -> True | False | None
        Whether the dispatch's latest launch left any trace of starting (a pid file, a log, a prompt file); False
        means the launch was cut off before the agent could start (the engine launches it again, no return
        counted); None: the runner cannot tell.
  decide(ws, request, cancel=None) -> None | {"output": parsed JSON or None, "error": str or None, "log": path
        or None} A decision agent's answer (rsi.decisions, task decisions.<point>: "agent"): request is {"point",
        "options" (the choices allowed), "default" (the code's), "prompt", "schema" (the JSON schema {choice,
        reason} the answer follows), "dir" (a directory for the call's files)}; cancel() true (the run was
        stopped) stops the agent. None: the runner has no decision agent (the code's default is used).
  name  the dispatch record's "runner"; poll_interval  seconds between polls while the engine waits.

The dispatch record (rsi.workspace engine records) tells a runner where to work: "outdir" (relative to the
workspace), "module", "launches" (this launch's number within the dispatch: agent-<n>.log), "feedback" (a
file in the output directory a resumed or fresh session is given) and, for attach, "pid" and "launched_at".
"""
import glob
import json
import os
import re
import signal
import subprocess
import time

from rsi.gpus import run_bounded
from rsi.workspace import parse_utc

# The launch modes (a dispatch record's launch_mode) that go on in the dispatch's session (--resume): a return
# (resume), the continuation after a long job (continue), the second part of a two-part hand-back (followup), the
# user's answers to the questions of the last hand-back (answer) and a long job's check-in (checkin). "first" and
# "fresh" start a new session.
SESSION_MODES = ("resume", "continue", "followup", "answer", "checkin")


def outdir(ws, dispatch):
    """The dispatch's output directory, absolute."""
    return os.path.join(ws.root, dispatch["outdir"])


def launch_number(dispatch):
    return int(dispatch.get("launches") or 1)


_GONE = ("Z", "X")  # the /proc states of a process that has ended: zombie, dead


def pid_alive(pid, proc_root="/proc"):
    """True while a process with this pid exists and runs (one owned by another user counts). A zombie, a process
    that has ended but that its parent has not reaped yet, counts as gone: os.kill(pid, 0) still succeeds on it, so
    on Linux the state in <proc_root>/<pid>/stat decides (Z or X: gone, unless _others_alive: a thread of it, or a
    process of the group it leads, still runs); elsewhere a zombie counts as alive. `proc_root` stands for /proc."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except (OSError, ValueError, TypeError):
        return False
    state = _proc_state(int(pid), proc_root)
    if state in _GONE:
        return _others_alive(int(pid), proc_root)
    return state != "gone"


def _read_stat(path):
    """The fields of a /proc stat file after the command (which is in parentheses and may hold spaces and
    parentheses): [state, ppid, pgrp, ...]. Raises OSError when the file cannot be read."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text[text.rfind(")") + 1:].split()


def _stat_fields(path):
    """_read_stat, or None when the file cannot be read."""
    try:
        return _read_stat(path)
    except OSError:
        return None


def _proc_state(pid, proc_root="/proc"):
    """The state letter of process `pid` from <proc_root>/<pid>/stat (Linux), "gone" when /proc has no such
    process, None without /proc (macOS)."""
    try:
        rest = _read_stat(os.path.join(proc_root, str(pid), "stat"))
    except FileNotFoundError:
        return "gone" if os.path.isdir(os.path.join(proc_root, "self")) else None
    except OSError:
        return None
    return rest[0] if rest else None


def _others_alive(pid, proc_root="/proc"):
    """For process `pid` whose state is Z (or X): True when one of its threads (<proc_root>/<pid>/task/*/stat) is
    not (its main thread ended while the others run: the process shows Z), or, when it leads its process group, a
    process of that group (<proc_root>/*/stat) is not (the leader ended, its members run on: stop_process ends the
    group). Linux only; `proc_root` stands for /proc."""
    for path in glob.glob(os.path.join(proc_root, str(pid), "task", "*", "stat")):
        rest = _stat_fields(path)
        if rest and rest[0] not in _GONE:
            return True
    mine = _stat_fields(os.path.join(proc_root, str(pid), "stat"))
    if not mine or len(mine) < 3 or mine[2] != str(pid):
        return False
    for path in glob.glob(os.path.join(proc_root, "[0-9]*", "stat")):
        if os.path.basename(os.path.dirname(path)) == str(pid):
            continue
        rest = _stat_fields(path)
        if rest and len(rest) >= 3 and rest[2] == mine[2] and rest[0] not in _GONE:
            return True
    return False


_PS_LINE = re.compile(r"^\s*(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{1,2}:\d\d:\d\d\s+\d{4})\s+(.*)$")


def process_info(pid):
    """(command line, start time in seconds since the epoch or None) of process `pid` as `ps` shows them; None
    when there is no such process or `ps` cannot tell. `ps` runs bounded (rsi.gpus.run_bounded: one that does not
    end within 10 s, nor a second after its kill, is left behind): the engine calls this while it waits."""
    if not pid_alive(pid):
        return None
    try:
        out = run_bounded(["ps", "-o", "lstart=", "-o", "command=", "-p", str(int(pid))], timeout=10,
                          env=dict(os.environ, LC_ALL="C", LANG="C"))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
    m = _PS_LINE.match(out.stdout.strip()) if out.returncode == 0 else None
    if not m:
        return None
    try:
        started = time.mktime(time.strptime(m.group(1), "%a %b %d %H:%M:%S %Y"))
    except (ValueError, OverflowError):
        started = None
    return m.group(2).strip(), started


def command_of(pid):
    """The command line of process `pid` as `ps` shows it, or None (see process_info)."""
    info = process_info(pid)
    return info[0] if info else None


def stop_process(pid, proc=None, grace=5.0):
    """Stop an agent: SIGTERM to its process group (agents are started as group leaders, so their tools go too),
    then SIGKILL when it has not ended after `grace` seconds. `proc` (a Popen of this process) is reaped."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        end = time.time() + grace
        while time.time() < end:
            if (proc.poll() is not None) if proc is not None else not pid_alive(pid):
                return
            time.sleep(0.05)


def launched_epoch(dispatch):
    """When the dispatch's agent was launched (seconds since the epoch); now when unknown."""
    dt = parse_utc(dispatch.get("launched_at"))
    return dt.timestamp() if dt is not None else time.time()


def result(exit_code=None, session=None, log=None, resume_failed=False, timed_out=False, lost=False, **extra):
    out = {"exit_code": exit_code, "session": session, "log": log, "resume_failed": resume_failed,
           "timed_out": timed_out, "lost": lost}
    out.update(extra)
    return out


def last_json_object(text):
    """The last line of `text` that is a JSON object, parsed; None when there is none."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def launch_failed(runner, log, error):
    """The handle of a launch that could not start: already finished, its result says why (also in the log)."""
    text = "could not start the agent: %s" % error
    try:
        with open(log, "ab") as f:
            f.write((text + "\n").encode("utf-8"))
    except OSError:
        pass
    return {"runner": runner, "waits": True, "result": result(log=log, error=text)}


class Runner(object):
    """The runner interface (see the module docstring). This base class finds nothing to attach to."""
    name = None
    poll_interval = 10.0

    def new_session(self, dispatch):
        return None

    def launch(self, ws, dispatch, prompt, resume_session=None, new_session=None):
        raise NotImplementedError

    def poll(self, handle):
        return handle.get("result")

    def attach(self, ws, dispatch):
        return {"runner": self, "waits": True, "result": result(lost=True)}

    def stop(self, handle):
        return False

    def started(self, ws, dispatch):
        return None

    def decide(self, ws, request, cancel=None):
        return None
