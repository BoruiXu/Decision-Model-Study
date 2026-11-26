"""CodeRunner: the dispatch of a code module (rsi.flow.CODE_MODULES, e.g. compare) runs a fixed command, no agent.

The command is the task's `runner.code.commands.<module>` (an argv list; a relative path as its first word is taken
from the repository root, a bare name is looked up on PATH), else `<repo>/modules/<module>/run`. launch
starts argv(), `<command> --brief <outdir>/brief.json --outdir <outdir> --mode <launch mode> [--feedback
<outdir>/<dispatch feedback>]`, the way MockRunner starts a script: in the output directory, as its own process group,
stdout and stderr to agent-<n>.log, returning at once; the engine polls it. The launch mode is the dispatch record's
(first, resume, continue, followup, answer, checkin, or fresh, which the command treats as resume): every mode reruns
the same idempotent command, which reads its brief and the files of its output directory and ends by writing
handback.json (modules/<module>/SKILL.md is its contract). A small shell runs it: it first writes its own pid (the
process group's leader, the pid launch returns) to agent-<n>.pid, and when the command ends its exit status to
agent-<n>.exit. A later `rsi run` attaches to it by the record's pid, else (`rsi run` was killed before it could record
the pid) by agent-<n>.pid, while that process runs `--outdir <the dispatch's output directory>` and started within
START_SLACK of the record's launch (MockRunner.attach), and also learns how it ended; it is never launched a second
time. A command whose executable is missing gives exit code 127 at once. The
session id is made up (code-<hex>) and means nothing to the command; there is no session to lose, so no exit code
says a resume failed. The prompt the engine builds from the module's SKILL.md is ignored, and the read-deny list is
only recorded (the command reads only its brief's inputs by construction). It has no decision agent. A command still
running after `runner.code.timeout_s` seconds (default 6 hours) is stopped (timed_out).
"""
import os
import shutil
import uuid

from rsi import briefs as B
from rsi.runners import base
from rsi.runners.mock import MockRunner

DEFAULT_TIMEOUT_S = 6 * 3600
# Run by /bin/sh -c with $0 = the exit status file, $1 = the pid file and the rest = the command line: write the
# shell's pid, run the command, then keep its exit status.
EXIT_WRAPPER = 'echo $$ > "$1"; shift; "$@"; code=$?; echo $code > "$0"; exit $code'


def exit_file(log):
    """The exit status file of the launch whose log is agent-<n>.log: agent-<n>.exit beside it."""
    return os.path.splitext(log)[0] + ".exit"


def pid_file(log):
    """The pid file of the launch whose log is agent-<n>.log: agent-<n>.pid beside it (rsi.engine.AGENT_PID_FILE keeps
    it out of the long jobs' pid files)."""
    return os.path.splitext(log)[0] + ".pid"


def read_exit(path):
    """The exit status in an exit status file, or None (no file, or not a number)."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class CodeRunner(MockRunner):
    name = "code"
    enforces_deny = False
    resume_failed_exit = None  # a command has no session that could be lost

    def __init__(self, commands, timeout_s=DEFAULT_TIMEOUT_S):
        MockRunner.__init__(self, None, timeout_s)
        self.commands = dict(commands or {})

    @classmethod
    def from_task(cls, task):
        """The runner with the task definition's runner.code settings (defaults for the rest)."""
        c = (task.get("runner") or {}).get("code") or {}
        return cls(c.get("commands"), c.get("timeout_s", DEFAULT_TIMEOUT_S))

    def command_for(self, module):
        """The command of `module`: the task's runner.code.commands.<module>, else <repo>/modules/<module>/run. A
        relative path as the first word is made absolute from the repository root (the command runs in the output
        directory); a bare name stays, for PATH."""
        command = list(self.commands.get(module) or [os.path.join(B.repo_root(), "modules", module, "run")])
        if os.sep in command[0] and not os.path.isabs(command[0]):
            command[0] = os.path.normpath(os.path.join(B.repo_root(), command[0]))
        return command

    def new_session(self, dispatch):
        return "code-%s" % uuid.uuid4().hex[:12]

    def argv(self, ws, dispatch, out, session):
        """The command line of `dispatch`'s launch (output directory `out`; the session is not passed); None when the
        command's executable (its first word: a path, or a name found on PATH) does not exist."""
        command = self.command_for(dispatch["module"])
        exe = command[0]
        found = os.path.isfile(exe) if os.sep in exe else shutil.which(exe) is not None
        if not found:
            return None
        argv = command + ["--brief", os.path.join(out, "brief.json"), "--outdir", out,
                          "--mode", dispatch.get("launch_mode") or "first"]
        if dispatch.get("feedback"):
            argv += ["--feedback", os.path.join(out, dispatch["feedback"])]
        return argv

    def missing(self, dispatch):
        module = dispatch["module"]
        return "code runner: the command of module %s (%s) does not exist: %s" % (
            module, "runner.code.commands.%s" % module if module in self.commands else "the default",
            self.command_for(module)[0])

    def popen_argv(self, argv, log):
        """The command run by a small shell that writes its pid to agent-<n>.pid first and the command's exit status
        to agent-<n>.exit at its end (earlier ones removed)."""
        for path in (exit_file(log), pid_file(log)):
            try:
                os.remove(path)
            except OSError:
                pass
        return ["/bin/sh", "-c", EXIT_WRAPPER, exit_file(log), pid_file(log)] + list(argv)

    def attach(self, ws, dispatch):
        """MockRunner.attach by the record's pid, else by the pid the launch's shell wrote to agent-<n>.pid (a run
        killed between the launch and recording the pid)."""
        if not dispatch.get("pid"):
            log = os.path.join(base.outdir(ws, dispatch), "agent-%d.log" % base.launch_number(dispatch))
            dispatch = dict(dispatch, pid=base.read_pid(pid_file(log)))
        return MockRunner.attach(self, ws, dispatch)

    def poll(self, handle):
        """MockRunner.poll; for a command this process did not start (attached), the exit status it left."""
        r = MockRunner.poll(self, handle)
        if r is not None and r.get("exit_code") is None and not r.get("timed_out") and r.get("log"):
            code = read_exit(exit_file(r["log"]))
            if code is not None:
                r["exit_code"] = code
        return r

    def decide(self, ws, request, cancel=None):
        return base.Runner.decide(self, ws, request, cancel)
