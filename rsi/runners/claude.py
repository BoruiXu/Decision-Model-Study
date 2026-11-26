"""ClaudeCodeRunner: a module agent is a headless Claude Code session (`claude -p`).

The command is `claude -p --dangerously-skip-permissions --output-format json` (module agents get full
permissions: the user's decision of 2026-11-23), then the task's extra arguments, then `--session-id <uuid>`
for a new session (the engine makes the uuid up and records it before the launch) or `--resume <session>` to go
on in one. The prompt is never an argument (a prompt starting with "-", e.g. a skill's front matter, would be
taken for an option, and argv has a size limit): it is written to prompt-<n>.md in the output directory and
given on stdin, which `claude -p` reads as the prompt. The working directory is the output directory. The task
definition can change the command under `runner.claude`: `command` (the argument list before the extra
arguments), `args` (extra arguments; without --model / --effort in them the agent runs with --model claude-opus-5-5 --effort xhigh), `timeout_s` (default 6 hours, counted from the launch).

The agent runs in its own session and process group, so it survives the `rsi run` that started it; a small
shell writes its pid to agent-<n>.pid before it becomes `claude`, and stdout and stderr go to agent-<n>.log.
A later `rsi run` attaches to that pid only while the process is this runner's program (the first or second
word of its command line, so an interpreter running a `claude` script counts) and started within
START_SLACK of the launch, so a pid reused after a restart, or any other process that merely has "claude"
somewhere in its arguments, is neither waited for nor stopped.

The session id `--output-format json` prints at the end is read back as a cross-check. A resume counts as
failed (resume_failed: the engine opens a fresh session with the history) only when the CLI says the session
cannot be resumed (RESUME_GONE); any other failure of a resumed agent keeps its session.

A decision agent (decide) is the same command plus `--json-schema <schema>` (the CLI checks the answer against
it and reports it as "structured_output" in the result object), started like a module agent in the decision's
directory with the prompt on stdin (prompt.md) and polled every DECISION_POLL_S: it is stopped when the run is
stopped (cancel) or after DECISION_TIMEOUT_S (or the task's timeout_s when shorter).

stop(handle) terminates an agent (its process group: SIGTERM, then SIGKILL) only when it is this runner's own
process or an attached one `ours()` still recognises.
"""
import json
import os
import shutil
import subprocess
import time
import uuid

from rsi.briefs import decision_denied, deny_rules
from rsi.runners import base

DEFAULT_COMMAND = ("claude", "-p", "--dangerously-skip-permissions", "--output-format", "json")
DEFAULT_TIMEOUT_S = 6 * 3600
# Model and effort of every agent (module and decision) unless runner.claude.args names its own.
DEFAULT_MODEL_ARGS = (("--model", "claude-opus-5-5"), ("--effort", "xhigh"))
# Run by /bin/sh -c with $0 = the pid file and "$@" = the claude command line: write the pid, then exec.
PID_WRAPPER = 'echo $$ > "$0"; exec "$@"'
# What the CLI prints when the session given to --resume cannot be resumed (Claude Code 2.1: "No conversation
# found with session ID: <id>" for an unknown id, "Failed to resume session" when it cannot be loaded).
RESUME_GONE = ("No conversation found with session ID", "Failed to resume session")
# How far before / after the recorded launch time an attached process may have started (seconds).
START_SLACK = (5, 60)
DECISION_TIMEOUT_S = 1800
DECISION_POLL_S = 0.5


def parse_output(path):
    """(session id, result object) from the log of a `claude -p --output-format json` run: the last JSON object
    (or array element) holding a non-empty "session_id"; (None, {}) when there is none."""
    text = _read_text(path)
    chunks = [line.strip() for line in reversed(text.splitlines())]
    start = text.rfind("\n{")
    chunks.append(text[start + 1:] if start >= 0 else text)  # a pretty-printed object at the end
    for chunk in chunks:
        if not chunk.startswith(("{", "[")):
            continue
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        for o in reversed(obj if isinstance(obj, list) else [obj]):
            if isinstance(o, dict) and isinstance(o.get("session_id"), str) and o["session_id"]:
                return o["session_id"], o
    return None, {}


def structured_output(path):
    """The answer of a `--json-schema` run from its log: the result object's "structured_output", else its
    "result" text when that is a JSON object; None when there is neither."""
    _session, info = parse_output(path)
    out = info.get("structured_output")
    if isinstance(out, dict):
        return out
    text = info.get("result")
    if isinstance(text, str):
        try:
            obj = json.loads(text.strip())
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, TypeError):
        return ""


class ClaudeCodeRunner(base.Runner):
    name = "claude"
    enforces_deny = True  # the engine refuses to launch when a denied path cannot be written as a safe rule

    def __init__(self, command=None, args=None, timeout_s=None, poll_interval=10.0, setting_sources=None):
        self.command = list(command or DEFAULT_COMMAND)
        self.args = list(args or [])
        for opt, value in reversed(DEFAULT_MODEL_ARGS):
            if not any(a == opt or a.startswith(opt + "=") for a in self.args):
                self.args = [opt, value] + self.args
        if setting_sources is not None:  # which of the user's Claude Code settings an agent loads
            self.args += ["--setting-sources", ",".join(setting_sources)]
        self.timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        self.poll_interval = poll_interval

    @classmethod
    def from_task(cls, task):
        """The runner with the task definition's runner.claude settings (defaults for the rest)."""
        c = (task.get("runner") or {}).get("claude") or {}
        return cls(command=c.get("command"), args=c.get("args"), timeout_s=c.get("timeout_s"),
                   setting_sources=c.get("setting_sources"))

    def new_session(self, dispatch):
        return str(uuid.uuid4())

    def argv(self, resume_session=None, new_session=None, deny=()):
        """The command line (the prompt goes to stdin). `deny`: Claude Code deny rules, each its own argument
        after `--disallowedTools` (a variadic option: it takes arguments up to the next one starting with "-"),
        put last, right before --session-id / --resume, so that it can take nothing but the rules."""
        if deny:
            tail = ["--disallowedTools"] + list(deny)
        else:
            tail = []
        if resume_session:
            tail += ["--resume", resume_session]
        elif new_session:
            tail += ["--session-id", new_session]
        return self.command + self.args + tail

    def is_program(self, cmdline):
        """True when a `ps` command line runs this runner's program: it starts with the program (as given, as found
        on PATH, or its real path; spaces in the path allowed), or an interpreter runs it (the line after the first
        word starts so); else, for a bare program name, the first or second word's base name is it."""
        a0 = self.command[0]
        names = set()
        for p in (a0, shutil.which(a0)):
            if p:
                names.update([p, os.path.realpath(p)])
        names = [n for n in names if os.sep in n]

        def starts(text):
            return any(text == n or text.startswith(n + " ") for n in names)

        if starts(cmdline) or starts(cmdline.partition(" ")[2].lstrip()):
            return True
        return os.path.basename(a0) in [os.path.basename(w) for w in cmdline.split()[:2]]

    def ours(self, pid, launched=None):
        """True while process `pid` lives, is this runner's program (is_program) and, given the launch time
        `launched` (seconds since the epoch), started within START_SLACK of it. A process `ps` cannot describe is
        not ours."""
        info = base.process_info(pid)
        if not info or not info[0]:
            return False
        command, started = info
        if not self.is_program(command):
            return False
        if launched is None:
            return True
        return started is not None and launched - START_SLACK[0] <= started <= launched + START_SLACK[1]

    def _files(self, ws, dispatch):
        out, n = base.outdir(ws, dispatch), base.launch_number(dispatch)
        return (out, os.path.join(out, "agent-%d.log" % n), os.path.join(out, "agent-%d.pid" % n),
                os.path.join(out, "prompt-%d.md" % n))

    def launch(self, ws, dispatch, prompt, resume_session=None, new_session=None):
        out, log, pidfile, prompt_file = self._files(ws, dispatch)
        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(prompt_file, "rb") as stdin, open(log, "ab") as f:
                proc = subprocess.Popen(["/bin/sh", "-c", PID_WRAPPER, pidfile]
                                        + self.argv(resume_session, new_session,
                                                  deny_rules(dispatch.get("denied") or [])),
                                        cwd=out, stdin=stdin, stdout=f, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        except OSError as e:
            return base.launch_failed(self, log, e)
        return {"runner": self, "waits": True, "proc": proc, "pid": proc.pid, "log": log, "started": time.time(),
                "resume": resume_session, "session": new_session}

    def attach(self, ws, dispatch):
        """The agent of an earlier `rsi run` by its pid (the record's, else its pid file): a handle that waits
        while that process lives and is ours (time limit counted from launched_at), else a finished, lost one."""
        _out, log, pidfile, _prompt = self._files(ws, dispatch)
        resume = dispatch.get("session") if dispatch.get("launch_mode") in base.SESSION_MODES else None
        pid = dispatch.get("pid") or base.read_pid(pidfile)
        launched = base.launched_epoch(dispatch)
        if pid and self.ours(pid, launched):
            return {"runner": self, "waits": True, "proc": None, "pid": int(pid), "log": log, "started": launched,
                    "resume": resume, "attached": True}
        return {"runner": self, "waits": True, "pid": pid, "result": self._result(None, log, resume, lost=True)}

    def poll(self, handle):
        if "result" in handle:
            return handle["result"]
        proc, timed_out = handle.get("proc"), False
        if proc is not None:
            code = proc.poll()
            running = code is None
        else:
            code, running = None, self.ours(handle["pid"], handle["started"])
        if running:
            if time.time() - handle["started"] <= self.timeout_s:
                return None
            if proc is not None or self.ours(handle["pid"], handle["started"]):
                base.stop_process(handle["pid"], proc)
            timed_out = True
            code = proc.returncode if proc is not None else None
        handle["result"] = self._result(code, handle["log"], handle.get("resume"), timed_out=timed_out)
        return handle["result"]

    def _result(self, code, log, resumed, timed_out=False, lost=False):
        session, info = parse_output(log)
        gone = bool(resumed) and not timed_out and code != 0 and any(m in _read_text(log) for m in RESUME_GONE)
        return base.result(exit_code=code, session=None if gone else session, log=log, timed_out=timed_out,
                           lost=lost, resume_failed=gone, cost_usd=info.get("total_cost_usd"),
                           is_error=info.get("is_error"))

    def stop(self, handle):
        """Terminate the handle's agent when it is still running and ours; True when it was."""
        if "result" in handle:
            return False
        proc, pid = handle.get("proc"), handle.get("pid")
        if proc is not None:
            if proc.poll() is not None:
                return False
            base.stop_process(pid, proc)
            return True
        if pid and self.ours(pid, handle.get("started")):
            base.stop_process(pid)
            return True
        return False

    def started(self, ws, dispatch):
        _out, log, pidfile, _prompt = self._files(ws, dispatch)
        return bool(dispatch.get("pid")) or os.path.exists(pidfile) or os.path.exists(log)

    def decision_argv(self, schema, deny=()):
        """The command line of a decision agent (the prompt goes to stdin); `deny` rules last, as in argv."""
        tail = ["--disallowedTools"] + list(deny) if deny else []
        return self.command + self.args + ["--json-schema", json.dumps(schema, ensure_ascii=False)] + tail

    def decide(self, ws, request, cancel=None):
        d = request["dir"]
        prompt, log = os.path.join(d, "prompt.md"), os.path.join(d, "agent.log")
        try:
            with open(prompt, "w", encoding="utf-8") as f:
                f.write(request["prompt"])
            with open(prompt, "rb") as stdin, open(log, "wb") as f:
                proc = subprocess.Popen(self.decision_argv(request["schema"], deny_rules(decision_denied(ws))),
                                        cwd=d, stdin=stdin, stdout=f,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            return {"output": None, "error": "could not start the decision agent: %s" % e, "log": log}
        limit, started = min(self.timeout_s, DECISION_TIMEOUT_S), time.time()
        while proc.poll() is None:
            if cancel is not None and cancel():
                base.stop_process(proc.pid, proc)
                return {"output": None, "error": "the run was stopped while the decision agent worked", "log": log,
                        "cancelled": True}
            if time.time() - started > limit:
                base.stop_process(proc.pid, proc)
                return {"output": None, "error": "the decision agent ran past its time limit", "log": log}
            time.sleep(DECISION_POLL_S)
        return {"output": structured_output(log),
                "error": None if proc.returncode == 0 else "exit code %s" % proc.returncode, "log": log}
