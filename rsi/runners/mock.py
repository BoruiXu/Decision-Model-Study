"""MockRunner: a module agent is a script, for tests and flow simulations (no model is called).

launch starts argv(), `<python> <scripts_dir>/<module>.py --brief <outdir>/brief.json --outdir <outdir> --session
<id> [--feedback <outdir>/<dispatch feedback>]`, in the output directory as its own process group (stdout and stderr
to agent-<n>.log) and returns at once (a missing script: exit code 127 at once, the log says why), like
ClaudeCodeRunner: the engine polls it, so a stop is noticed while the script runs (stop terminates its process
group) and records dispatched together run side by side. A new session's id is made up by new_session (mock-<hex>), a
resumed one passed on; a script that exits RESUME_FAILED when given a session to resume reports that the session
cannot be resumed. A script still running after timeout_s is stopped (timed_out). Subclasses change argv, missing
(the log line when argv is None) and popen_argv (rsi.runners.code.CodeRunner).

attach (a later `rsi run`, after the one that launched the script was killed) finds the script again by the
record's pid while that process runs `--outdir <the dispatch's output directory>` and started within START_SLACK
of the record's launch; otherwise the agent is lost.

decide writes the request to <dir>/request.json and runs `<python> <scripts_dir>/decide.py --request <it>` in
that directory; the last JSON object the script prints is the answer (output to <dir>/agent.log).
"""
import json
import os
import subprocess
import sys
import time
import uuid

from rsi.runners import base

RESUME_FAILED = 75
# How far before / after the recorded launch time an attached script may have started (seconds).
START_SLACK = (5, 60)


class MockRunner(base.Runner):
    name = "mock"
    resume_failed_exit = RESUME_FAILED  # the exit code of a resumed script that says its session cannot be resumed

    def __init__(self, scripts_dir, timeout_s=600, poll_interval=0.05):
        self.scripts_dir = os.path.abspath(scripts_dir) if scripts_dir else None
        self.timeout_s = timeout_s
        self.poll_interval = poll_interval

    def new_session(self, dispatch):
        return "mock-%s" % uuid.uuid4().hex[:12]

    def script(self, module):
        return os.path.join(self.scripts_dir or "", "%s.py" % module)

    def argv(self, ws, dispatch, out, session):
        """The command line launch runs for `dispatch` (output directory `out`, absolute; session `session`); None
        when there is nothing to run (no mock module script <scripts_dir>/<module>.py)."""
        script = self.script(dispatch["module"])
        if not (self.scripts_dir and os.path.isfile(script)):
            return None
        argv = [sys.executable, script, "--brief", os.path.join(out, "brief.json"), "--outdir", out,
                "--session", session]
        if dispatch.get("feedback"):
            argv += ["--feedback", os.path.join(out, dispatch["feedback"])]
        return argv

    def missing(self, dispatch):
        """The log line of a launch whose argv is None."""
        return "mock runner: no mock module script %s" % self.script(dispatch["module"])

    def popen_argv(self, argv, log):
        """What launch starts for `argv` (the launch's log `log`): argv itself."""
        return argv

    def launch(self, ws, dispatch, prompt, resume_session=None, new_session=None):
        out = base.outdir(ws, dispatch)
        log = os.path.join(out, "agent-%d.log" % base.launch_number(dispatch))
        session = resume_session or new_session or self.new_session(dispatch)
        argv = self.argv(ws, dispatch, out, session)
        if argv is None:
            with open(log, "ab") as f:
                f.write((self.missing(dispatch) + "\n").encode("utf-8"))
            return {"runner": self, "waits": True, "session": session,
                    "result": base.result(exit_code=127, session=session, log=log)}
        try:
            with open(log, "ab") as f:
                proc = subprocess.Popen(self.popen_argv(argv, log), cwd=out, stdin=subprocess.DEVNULL, stdout=f,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            return base.launch_failed(self, log, e)
        return {"runner": self, "waits": True, "proc": proc, "pid": proc.pid, "outdir": out, "log": log,
                "started": time.time(), "resume": resume_session, "session": session}

    def poll(self, handle):
        if "result" in handle:
            return handle["result"]
        proc, timed_out = handle.get("proc"), False
        if proc is not None:
            code = proc.poll()
            running = code is None
        else:
            code, running = None, self.ours(handle["pid"], handle.get("outdir"), handle["started"])
        if running:
            if time.time() - handle["started"] <= self.timeout_s:
                return None
            base.stop_process(handle["pid"], proc)
            timed_out = True
            code = proc.returncode if proc is not None else None
        failed = bool(handle.get("resume")) and code is not None and code == self.resume_failed_exit
        session = None if failed else handle.get("session")
        handle["result"] = base.result(exit_code=code, session=session, log=handle.get("log"), resume_failed=failed,
                                       timed_out=timed_out)
        return handle["result"]

    def ours(self, pid, outdir, launched=None):
        """True while process `pid` lives, runs a mock module script for the output directory `outdir` (its
        command line holds `--outdir <outdir>`) and, given the launch time `launched` (seconds since the epoch),
        started within START_SLACK of it. A process `ps` cannot describe is not ours."""
        info = base.process_info(pid)
        if not info or not info[0] or not outdir:
            return False
        command, started = info
        if ("--outdir %s " % outdir) not in command + " ":
            return False
        if launched is None:
            return True
        return started is not None and launched - START_SLACK[0] <= started <= launched + START_SLACK[1]

    def attach(self, ws, dispatch):
        """The script of an earlier `rsi run` by the record's pid: a handle that waits while that process runs and
        is this dispatch's script (time limit counted from launched_at), else a finished, lost one."""
        out = base.outdir(ws, dispatch)
        log = os.path.join(out, "agent-%d.log" % base.launch_number(dispatch))
        pid = dispatch.get("pid")
        launched = base.launched_epoch(dispatch)
        resume = dispatch.get("session") if dispatch.get("launch_mode") in base.SESSION_MODES else None
        if pid and self.ours(pid, out, launched):
            return {"runner": self, "waits": True, "proc": None, "pid": int(pid), "outdir": out, "log": log,
                    "started": launched, "resume": resume, "session": dispatch.get("session"), "attached": True}
        return {"runner": self, "waits": True, "pid": pid, "result": base.result(log=log, lost=True)}

    def stop(self, handle):
        """Terminate the handle's script when it still runs (its own process, or an attached one that is still
        this dispatch's script); True when it was."""
        if "result" in handle:
            return False
        proc, pid = handle.get("proc"), handle.get("pid")
        if proc is not None:
            if proc.poll() is not None:
                return False
            base.stop_process(pid, proc)
            return True
        if pid and self.ours(pid, handle.get("outdir"), handle.get("started")):
            base.stop_process(pid)
            return True
        return False

    def started(self, ws, dispatch):
        return os.path.exists(os.path.join(base.outdir(ws, dispatch), "agent-%d.log" % base.launch_number(dispatch)))

    def decide(self, ws, request, cancel=None):
        d = request["dir"]
        req, log = os.path.join(d, "request.json"), os.path.join(d, "agent.log")
        with open(req, "w", encoding="utf-8") as f:
            json.dump(request, f, indent=2, ensure_ascii=False)
        script = os.path.join(self.scripts_dir or "", "decide.py")
        if not (self.scripts_dir and os.path.isfile(script)):
            return {"output": None, "error": "mock runner: no decision script %s" % script, "log": None}
        try:
            p = subprocess.run([sys.executable, script, "--request", req], cwd=d, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=self.timeout_s,
                               universal_newlines=True)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {"output": None, "error": "mock decision agent: %s" % e, "log": None}
        with open(log, "w", encoding="utf-8") as f:
            f.write(p.stdout)
        return {"output": base.last_json_object(p.stdout),
                "error": None if p.returncode == 0 else "exit code %d" % p.returncode, "log": log}
