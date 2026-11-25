"""The agent runners (rsi/runners/): ClaudeCodeRunner's command line, prompt on stdin, session ids, resume
failure, timeout and re-attaching after a restart, tested against a strict fake `claude` placed first on PATH
(the real one is never called); MockRunner's script call and sessions; ManualRunner's wait for handback.json;
the task's optional `runner` settings."""
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
import uuid
from unittest import mock

import fixtures
from rsi import gpus as G
from rsi import rows
from rsi.runners import ClaudeCodeRunner, ManualRunner, MockRunner, make_runner
from rsi.runners import base
from rsi.runners.base import pid_alive
from rsi.workspace import RsiError, now_iso

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A stand-in for `claude` that parses its options like the real CLI: an argument starting with "-" that is no
# known option is an error, the prompt comes from a positional argument or stdin, --session-id must be a UUID and
# cannot go with --resume (without --fork-session), and resuming a session listed in FAKE_CLAUDE_GONE fails with
# the real CLI's message. It records its argv, working directory and prompt in FAKE_CLAUDE_RECORD.
FAKE_CLAUDE = r'''#!@PYTHON@
import json, os, sys, time, uuid
FLAGS = {"-p", "--print", "--dangerously-skip-permissions", "--verbose", "--fork-session"}
VALUED = {"--output-format", "--resume", "-r", "--session-id", "--model", "--effort", "--append-system-prompt"}
args, opts, positional, i = sys.argv[1:], {}, [], 0
while i < len(args):
    a = args[i]
    if a.startswith("-"):
        if a == "--disallowedTools":  # variadic like the real CLI: takes every argument up to the next option
            opts[a] = []
            while i + 1 < len(args) and not args[i + 1].startswith("-"):
                opts[a].append(args[i + 1])
                i += 1
        elif a in FLAGS:
            opts[a] = True
        elif a in VALUED and i + 1 < len(args):
            opts[a] = args[i + 1]
            i += 1
        else:
            sys.stderr.write("error: unknown option '" + a.splitlines()[0] + "'\n")
            sys.exit(1)
    else:
        positional.append(a)
    i += 1
prompt = positional[0] if positional else sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_RECORD"], "a") as f:
    f.write(json.dumps({"argv": args, "cwd": os.getcwd(), "prompt": prompt}) + "\n")
if not prompt:
    sys.stderr.write("Error: Input must be provided either through stdin or as a prompt argument when using --print\n")
    sys.exit(1)
sid, resume = opts.get("--session-id"), opts.get("--resume")
if sid:
    try:
        uuid.UUID(sid)
    except ValueError:
        sys.stderr.write("Error: Invalid session ID. Must be a valid UUID.\n")
        sys.exit(1)
    if resume and "--fork-session" not in opts:
        sys.stderr.write("Error: --session-id with --continue or --resume also needs --fork-session.\n")
        sys.exit(1)
if resume and resume in os.environ.get("FAKE_CLAUDE_GONE", "").split(","):
    sys.stderr.write("No conversation found with session ID: " + resume + "\n")
    sys.exit(1)
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if mode == "sleep":
    time.sleep(30)
sys.stderr.write("some progress on stderr\n")
if mode == "fail":
    sys.stderr.write("the agent failed\n")
    sys.exit(2)
session = os.environ.get("FAKE_CLAUDE_SESSION") or sid or resume or "sess-new"
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session,
                  "result": "done", "total_cost_usd": 0.01}))
'''


def install_fake_claude(case, root):
    """Put the fake `claude` first on PATH for the test `case`; return (its path, the record file)."""
    bindir = os.path.join(root, "bin")
    os.makedirs(bindir, exist_ok=True)
    path = os.path.join(bindir, "claude")
    with open(path, "w") as f:
        f.write(FAKE_CLAUDE.replace("@PYTHON@", sys.executable))
    os.chmod(path, 0o755)
    record = os.path.join(root, "claude-calls.jsonl")
    env = mock.patch.dict(os.environ, {"PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
                                       "FAKE_CLAUDE_RECORD": record})
    env.start()
    case.addCleanup(env.stop)
    return path, record


def claude_calls(record):
    with open(record) as f:
        return [json.loads(line) for line in f if line.strip()]


def spawn(case, argv, executable=None):
    """Start `argv` as a grandchild in its own process group (never a zombie of this process); return its pid."""
    code = ("import subprocess, sys, json; a = json.loads(sys.argv[1]); print(subprocess.Popen(a[0], executable=a[1], "
            "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid)")
    out = subprocess.run([sys.executable, "-c", code, json.dumps([list(argv), executable])], capture_output=True,
                         text=True)
    pid = int(out.stdout)
    case.addCleanup(lambda: pid_alive(pid) and os.kill(pid, signal.SIGKILL))
    return pid


def wait_result(runner, handle, limit=20):
    end = time.time() + limit
    while time.time() < end:
        r = runner.poll(handle)
        if r is not None:
            return r
        time.sleep(0.02)
    raise AssertionError("runner did not finish")


class RunnerCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.ws = types.SimpleNamespace(root=os.path.join(self.root, "ws"))
        self.outdir = "work/round-1/data"
        os.makedirs(os.path.join(self.ws.root, self.outdir))

    def dispatch(self, **fields):
        d = {"id": "d-0001", "module": "data", "purpose": "sets", "round": 1, "outdir": self.outdir,
             "session": None, "launches": 1, "launch_mode": "first", "feedback": None, "launched_at": now_iso()}
        d.update(fields)
        return d

    def out(self, name=""):
        return os.path.join(self.ws.root, self.outdir, name)


class TestClaudeCodeRunner(RunnerCase):
    def setUp(self):
        super(TestClaudeCodeRunner, self).setUp()
        self.fake, self.record = install_fake_claude(self, self.root)

    def calls(self):
        return claude_calls(self.record)

    def runner(self, **claude):
        task = fixtures.binary_task()
        if claude:
            task["runner"] = {"claude": claude}
        r = ClaudeCodeRunner.from_task(task)
        r.poll_interval = 0.02
        return r

    def test_default_command_line_prompt_on_stdin_and_session(self):
        r = self.runner()
        sid = r.new_session(self.dispatch())
        self.assertEqual(str(uuid.UUID(sid)), sid)
        h = r.launch(self.ws, self.dispatch(denied=[]), "PROMPT text", new_session=sid)  # no --disallowedTools
        self.assertTrue(h["waits"])
        res = wait_result(r, h)
        self.assertEqual((res["exit_code"], res["session"], res["resume_failed"]), (0, sid, False))
        self.assertEqual(self.calls(), [{"argv": ["-p", "--dangerously-skip-permissions", "--output-format", "json",
                                                  "--model", "claude-opus-5-5", "--effort", "xhigh", "--session-id", sid], "cwd": self.out().rstrip("/"),
                                         "prompt": "PROMPT text"}])
        with open(self.out("prompt-1.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "PROMPT text")
        self.assertEqual(res["log"], self.out("agent-1.log"))
        with open(res["log"]) as f:
            text = f.read()
        self.assertIn("some progress on stderr", text)
        self.assertIn('"session_id": "%s"' % sid, text)
        with open(self.out("agent-1.pid")) as f:
            self.assertEqual(int(f.read()), h["pid"])

    def test_denied_paths_become_disallowed_tools_before_the_session_option(self):
        r = self.runner(args=["--model", "opus"])
        sid = str(uuid.uuid4())
        den = ["/w/artifacts/evalset-0001", "/w/artifacts/valset-0001/labels.jsonl", "/w/sp ace/x,y"]
        exists = os.path.exists  # every denied path is there: one rule each (a directory's /**)
        with mock.patch("os.path.isdir", side_effect=lambda p: p != den[1]), \
                mock.patch("os.path.exists", side_effect=lambda p: p in den or exists(p)):
            h = r.launch(self.ws, self.dispatch(denied=den), "P", new_session=sid)
            self.assertEqual(wait_result(r, h)["exit_code"], 0)  # the fake rejects an option it does not know
            wait_result(r, r.launch(self.ws, self.dispatch(denied=den, launches=2), "F", resume_session="sess-7"))
        rules = ["Read(//w/artifacts/evalset-0001/**)", "Edit(//w/artifacts/evalset-0001/**)",
                 "Read(//w/artifacts/valset-0001/labels.jsonl)", "Edit(//w/artifacts/valset-0001/labels.jsonl)",
                 "Read(//w/sp ace/x,y/**)", "Edit(//w/sp ace/x,y/**)"]
        self.assertEqual(self.calls()[0]["argv"],
                         ["-p", "--dangerously-skip-permissions", "--output-format", "json", "--effort", "xhigh",
                          "--model", "opus", "--disallowedTools"] + rules + ["--session-id", sid])
        # without --model / --effort in runner.claude.args every agent runs on Opus 5.5 at effort xhigh
        self.assertEqual(self.runner().args[:4], ["--model", "claude-opus-5-5", "--effort", "xhigh"])
        self.assertEqual(self.runner(args=["--effort=high"]).args, ["--model", "claude-opus-5-5", "--effort=high"])
        self.assertEqual(self.calls()[1]["argv"][-3 - len(rules):],
                         ["--disallowedTools"] + rules + ["--resume", "sess-7"])

    def test_a_prompt_that_starts_like_an_option_goes_through(self):
        prompt = "---\nname: rsi-data\ndescription: x\n---\n\n# rsi-data\n"
        bad = subprocess.run([self.fake, "-p", prompt], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)  # as an argument the CLI takes it for an option
        self.assertIn("unknown option", bad.stderr)
        r = self.runner()
        res = wait_result(r, r.launch(self.ws, self.dispatch(), prompt, new_session=str(uuid.uuid4())))
        self.assertEqual(res["exit_code"], 0)
        self.assertEqual(self.calls()[-1]["prompt"], prompt)

    def test_command_resume_and_extra_args_from_the_task(self):
        r = self.runner(command=["claude", "-p", "--output-format", "json"])
        sid = str(uuid.uuid4())
        wait_result(r, r.launch(self.ws, self.dispatch(), "P", new_session=sid))
        self.assertEqual(self.calls()[0]["argv"], ["-p", "--output-format", "json", "--model", "claude-opus-5-5", "--effort", "xhigh", "--session-id", sid])
        os.remove(self.record)
        r = self.runner(args=["--model", "opus"], timeout_s=60)
        self.assertEqual(r.timeout_s, 60)
        res = wait_result(r, r.launch(self.ws, self.dispatch(launches=2), "FEEDBACK", resume_session="sess-7"))
        self.assertEqual(self.calls()[0]["argv"], ["-p", "--dangerously-skip-permissions", "--output-format", "json",
                                                   "--effort", "xhigh", "--model", "opus", "--resume", "sess-7"])
        self.assertEqual(self.calls()[0]["prompt"], "FEEDBACK")
        self.assertEqual(res["session"], "sess-7")
        self.assertEqual(res["log"], self.out("agent-2.log"))

    def test_a_failed_resume(self):
        # The session is gone: resume_failed, no session.
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_GONE": "sess-gone"}):
            r = self.runner()
            res = wait_result(r, r.launch(self.ws, self.dispatch(launches=2), "F", resume_session="sess-gone"))
        self.assertEqual(res["exit_code"], 1)
        self.assertTrue(res["resume_failed"])
        self.assertIsNone(res["session"])
        # Any other failure of a resumed agent keeps the session.
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_MODE": "fail"}):
            r = self.runner()
            res = wait_result(r, r.launch(self.ws, self.dispatch(launches=3), "F", resume_session="sess-7"))
        self.assertEqual(res["exit_code"], 2)
        self.assertFalse(res["resume_failed"])

    def test_timeout_stops_the_agent(self):
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_MODE": "sleep"}):
            r = self.runner(timeout_s=1)
            h = r.launch(self.ws, self.dispatch(), "P")
            res = wait_result(r, h)
        self.assertTrue(res["timed_out"])
        self.assertFalse(pid_alive(h["pid"]))

    def test_attach_after_a_restart(self):
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_MODE": "sleep"}):
            first = self.runner()
            h = first.launch(self.ws, self.dispatch(), "P")
        end = time.time() + 5
        while not (os.path.exists(self.out("agent-1.pid")) and os.path.getsize(self.out("agent-1.pid"))) \
                and time.time() < end:
            time.sleep(0.02)
        # A new runner (a new `rsi run`) finds the agent by the pid file the agent's shell wrote.
        again = self.runner()
        h2 = again.attach(self.ws, self.dispatch())
        self.assertEqual(h2["pid"], h["pid"])
        self.assertIsNone(again.poll(h2))
        os.killpg(h["pid"], signal.SIGTERM)
        h["proc"].wait()
        res = wait_result(again, h2)
        self.assertFalse(res["lost"])  # it ended while watched: not lost, its exit code unknown
        self.assertIsNone(res["exit_code"])
        # A finished agent's session id is read back from its log.
        r = self.runner()
        wait_result(r, r.launch(self.ws, self.dispatch(launches=3), "P"))
        res = wait_result(r, r.attach(self.ws, self.dispatch(launches=3, pid=None)))
        self.assertTrue(res["lost"])
        self.assertEqual(res["session"], "sess-new")

    def test_attached_resume_that_failed(self):
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_GONE": "sess-gone"}):
            r = self.runner()
            wait_result(r, r.launch(self.ws, self.dispatch(launches=2), "F", resume_session="sess-gone"))
        # Every launch mode that goes on in the session (a return, a continuation, a followup, the user's answer).
        for mode in base.SESSION_MODES:
            res = r.poll(r.attach(self.ws, self.dispatch(launches=2, launch_mode=mode, session="sess-gone")))
            self.assertTrue(res["resume_failed"], mode)

    def test_attach_times_out_from_the_launch_time(self):
        launched = now_iso()
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_MODE": "sleep"}):
            pid = spawn(self, [self.fake, "-p", "P"])
        r = self.runner(timeout_s=1)
        res = wait_result(r, r.attach(self.ws, self.dispatch(pid=pid, launched_at=launched)))
        self.assertTrue(res["timed_out"])
        end = time.time() + 5
        while pid_alive(pid) and time.time() < end:
            time.sleep(0.05)
        self.assertFalse(pid_alive(pid))

    def test_attach_ignores_processes_that_are_not_the_agent(self):
        # Neither waited for nor stopped: a process that merely has "claude" in its arguments, and a `claude`
        # that started long after the recorded launch (a pid reused after a restart).
        r = self.runner(timeout_s=1)
        decoy = spawn(self, [sys.executable, "-c", "import time; time.sleep(30)", "/Users/someone/.claude/projects/x"])
        named = spawn(self, ["claude", "30"], executable="/bin/sleep")
        self.assertFalse(r.ours(decoy))
        for pid, launched in ((decoy, now_iso()), (named, "2026-01-01T00:00:00Z")):
            res = wait_result(r, r.attach(self.ws, self.dispatch(pid=pid, launched_at=launched)))
            self.assertTrue(res["lost"])
            self.assertFalse(res["timed_out"])
            self.assertTrue(pid_alive(pid))
        # The same `claude` process launched just now is the agent.
        self.assertTrue(r.ours(named, time.time()))


class TestMockRunner(RunnerCase):
    def setUp(self):
        super(TestMockRunner, self).setUp()
        self.scripts = os.path.join(self.root, "mocks")
        os.makedirs(self.scripts)
        with open(os.path.join(self.scripts, "data.py"), "w") as f:
            f.write("import json, sys\nprint(json.dumps(sys.argv[1:]))\n"
                    "sys.exit(75 if '--feedback' in sys.argv and 'fail' in sys.argv[-1] else 0)\n")
        # train.py waits for the file named by RELEASE (or 30 seconds), then exits 0: an agent still at work.
        self.release = os.path.join(self.root, "release")
        with open(os.path.join(self.scripts, "train.py"), "w") as f:
            f.write("import os, sys, time\nend = time.time() + 30\n"
                    "while not os.path.exists(%r) and time.time() < end:\n    time.sleep(0.02)\n" % self.release)

    def argv(self, res):
        with open(res["log"]) as f:
            return json.loads(f.read().splitlines()[-1])

    def test_script_call_and_sessions(self):
        r = MockRunner(self.scripts)
        h = r.launch(self.ws, self.dispatch(), "P")
        res = wait_result(r, h)
        self.assertEqual(res["exit_code"], 0)
        session = res["session"]
        self.assertTrue(session.startswith("mock-"))
        self.assertEqual(h["session"], session)
        self.assertEqual(self.argv(res), ["--brief", self.out("brief.json"), "--outdir", self.out().rstrip("/"),
                                          "--session", session])
        res2 = wait_result(r, r.launch(self.ws, self.dispatch(launches=2, feedback="feedback-2.md"), "F",
                                       resume_session=session))
        self.assertEqual(res2["session"], session)
        self.assertEqual(self.argv(res2)[-2:], ["--feedback", self.out("feedback-2.md")])
        self.assertEqual(res2["log"], self.out("agent-2.log"))
        self.assertNotEqual(wait_result(r, r.launch(self.ws, self.dispatch(), "P"))["session"], session)
        given = r.new_session(self.dispatch())
        self.assertEqual(wait_result(r, r.launch(self.ws, self.dispatch(), "P", new_session=given))["session"], given)

    def test_resume_failure_and_missing_script(self):
        r = MockRunner(self.scripts)
        res = wait_result(r, r.launch(self.ws, self.dispatch(feedback="feedback-fail.md"), "F",
                                      resume_session="mock-x"))
        self.assertTrue(res["resume_failed"])
        self.assertIsNone(res["session"])
        res = wait_result(r, r.launch(self.ws, self.dispatch(module="judge"), "P"))
        self.assertNotEqual(res["exit_code"], 0)
        with open(res["log"]) as f:
            self.assertIn("judge.py", f.read())

    def test_launch_returns_while_the_script_runs(self):
        # Like ClaudeCodeRunner: launch starts the script and returns; poll says None until it has ended.
        r = MockRunner(self.scripts)
        t0 = time.time()
        h = r.launch(self.ws, self.dispatch(module="train"), "P")
        self.assertLess(time.time() - t0, 5)
        self.assertTrue(h["waits"])
        self.assertTrue(pid_alive(h["pid"]))
        self.assertIsNone(r.poll(h))
        self.assertTrue(r.started(self.ws, self.dispatch(module="train")))
        open(self.release, "w").close()
        res = wait_result(r, h)
        self.assertEqual((res["exit_code"], res["timed_out"]), (0, False))
        self.assertFalse(r.stop(h))  # it has ended: nothing to stop

    def test_stop_terminates_a_running_script(self):
        r = MockRunner(self.scripts)
        h = r.launch(self.ws, self.dispatch(module="train"), "P")
        self.assertTrue(r.stop(h))
        self.assertFalse(pid_alive(h["pid"]))
        self.assertIsNotNone(r.poll(h))

    def test_time_limit(self):
        r = MockRunner(self.scripts, timeout_s=0.2)
        h = r.launch(self.ws, self.dispatch(module="train"), "P")
        res = wait_result(r, h)
        self.assertTrue(res["timed_out"])
        self.assertFalse(pid_alive(h["pid"]))

    def launch_elsewhere(self, d):
        """Launch dispatch `d` from another process (an earlier `rsi run` that is gone): the script is no child of
        this process; return its pid."""
        code = ("import json, sys, types; sys.path.insert(0, %r); from rsi.runners import MockRunner; "
                "d = json.loads(sys.argv[1]); ws = types.SimpleNamespace(root=sys.argv[2]); "
                "print(MockRunner(sys.argv[3]).launch(ws, d, 'P')['pid'])" % REPO)
        out = subprocess.run([sys.executable, "-c", code, json.dumps(d), self.ws.root, self.scripts],
                             capture_output=True, text=True)
        pid = int(out.stdout)
        self.addCleanup(lambda: pid_alive(pid) and os.kill(pid, signal.SIGKILL))
        return pid

    def test_attach_finds_the_script_of_an_earlier_run(self):
        # By the record's pid and launch time, a later `rsi run` (a MockRunner without the scripts directory) waits
        # for the script while it runs and can stop it; a process that is not this dispatch's script is not found.
        other = MockRunner(None)
        d = self.dispatch(module="train")
        d["pid"] = self.launch_elsewhere(d)
        a = other.attach(self.ws, d)
        self.assertNotIn("result", a)
        self.assertIsNone(other.poll(a))
        open(self.release, "w").close()
        res = wait_result(other, a)
        self.assertFalse(res["lost"])
        self.assertIsNone(res["exit_code"])  # not its parent: the exit code is unknown
        os.remove(self.release)
        d2 = dict(d, launches=2)
        d2["pid"] = self.launch_elsewhere(d2)
        self.assertTrue(other.stop(other.attach(self.ws, d2)))
        self.assertFalse(pid_alive(d2["pid"]))
        decoy = spawn(self, [sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertTrue(other.poll(other.attach(self.ws, dict(d, pid=decoy)))["lost"])
        self.assertFalse(other.stop(other.attach(self.ws, dict(d, pid=decoy))))
        self.assertTrue(pid_alive(decoy))
        # Launched long before the record says (a pid reused after a restart): not the agent.
        d3 = dict(d, launches=3)
        d3["pid"] = self.launch_elsewhere(d3)
        self.assertTrue(other.poll(other.attach(self.ws, dict(d3, launched_at="2026-01-01T00:00:00Z")))["lost"])
        open(self.release, "w").close()
        # A record without a pid: nothing running to attach to.
        res = MockRunner(self.scripts).poll(MockRunner(self.scripts).attach(self.ws, self.dispatch()))
        self.assertTrue(res["lost"])


class TestPidAlive(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir("/proc/self"), "a zombie is told by /proc/<pid>/stat (Linux)")
    def test_a_zombie_is_not_alive(self):
        # A child that has ended but is not reaped yet is a zombie: os.kill(pid, 0) still succeeds on it.
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        self.addCleanup(p.wait)
        end = time.time() + 30
        while base._proc_state(p.pid) != "Z":
            self.assertLess(time.time(), end, "the child never became a zombie")
            time.sleep(0.02)
        os.kill(p.pid, 0)  # no error: the pid still exists
        self.assertFalse(pid_alive(p.pid))
        self.assertTrue(pid_alive(os.getpid()))

    def test_pid_alive_reads_threads_and_the_group(self):
        # F18 (Linux, a fake /proc here): a zombie group leader whose group still runs, and a process whose main thread
        # ended while another thread runs, are alive; with all of them zombies it is gone.
        pid = os.getpid()  # os.kill(pid, 0) succeeds; its state comes from the fake /proc

        def stat(path, n, state):
            os.makedirs(os.path.join(path, str(n)))
            with open(os.path.join(path, str(n), "stat"), "w") as f:
                f.write("%d (sh -c a) b) %s 1 %d %d 0\n" % (n, state, pid, pid))

        for name, member, thread, want in (("group", "R", "Z", True), ("thread", "Z", "R", True),
                                           ("none", "Z", "Z", False)):
            with self.subTest(name):
                root = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, root)
                stat(root, pid, "Z")
                stat(root, 999999, member)  # another process of its group
                stat(os.path.join(root, str(pid), "task"), pid, "Z")  # its threads: the main one ended
                stat(os.path.join(root, str(pid), "task"), 999998, thread)
                self.assertEqual(pid_alive(pid, proc_root=root), want)

    def test_process_info_never_waits_on_a_stuck_ps(self):
        # `ps` runs bounded (rsi.gpus.run_bounded): one that never answers, not even after its kill, is left behind.
        killed = []

        class Stuck(object):
            stdout = stderr = None

            def __init__(self, args, **_kw):
                self.args = args

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(self.args, timeout)

            def kill(self):
                killed.append(self.args[0])

        self.addCleanup(lambda: G.ABANDONED.__setitem__(slice(None), []))
        with mock.patch.object(base, "run_bounded", functools.partial(G.run_bounded, popen=Stuck)):
            self.assertIsNone(base.process_info(os.getpid()))
        self.assertEqual(killed, ["ps"])


class TestManualRunner(RunnerCase):
    def test_waits_for_the_handback(self):
        r = ManualRunner()
        self.assertIsNone(r.new_session(self.dispatch()))
        h = r.launch(self.ws, self.dispatch(), "THE PROMPT")
        self.assertFalse(h["waits"])
        with open(self.out("prompt-1.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "THE PROMPT")
        self.assertIsNone(r.poll(h))
        self.assertIsNone(r.poll(r.attach(self.ws, self.dispatch())))
        with open(self.out("handback.json"), "w") as f:
            f.write("{}")
        res = r.poll(r.attach(self.ws, self.dispatch()))
        self.assertEqual((res["exit_code"], res["session"]), (0, None))


class TestRunnerSettings(unittest.TestCase):
    def test_task_runner_settings(self):
        r = ClaudeCodeRunner.from_task(fixtures.binary_task())  # the defaults
        self.assertEqual(r.command, ["claude", "-p", "--dangerously-skip-permissions", "--output-format", "json"])
        self.assertEqual(r.args, ["--model", "claude-opus-5-5", "--effort", "xhigh"])
        self.assertEqual(r.timeout_s, 6 * 3600)
        task = fixtures.binary_task()
        task["runner"] = {"claude": {"command": ["claude", "-p"], "args": ["--model", "opus"], "timeout_s": 600}}
        self.assertEqual(rows.check_task(task), [])
        task["runner"]["claude"]["nice"] = 1
        self.assertTrue(rows.check_task(task))
        task["runner"] = {"claude": {"command": []}}
        self.assertTrue(rows.check_task(task))
        task["runner"] = {"claude": {"setting_sources": ["project", "local"]}}
        self.assertEqual(rows.check_task(task), [])
        r = ClaudeCodeRunner.from_task(task)
        self.assertEqual(r.argv(new_session="s")[-4:], ["--setting-sources", "project,local", "--session-id", "s"])
        for bad in ([], ["everything"], "user"):
            task["runner"] = {"claude": {"setting_sources": bad}}
            self.assertTrue(rows.check_task(task), bad)

    def test_make_runner(self):
        ws = types.SimpleNamespace(task=fixtures.binary_task)
        self.assertEqual(make_runner("manual", ws).name, "manual")
        self.assertEqual(make_runner("claude", ws).name, "claude")
        with self.assertRaises(RsiError):
            make_runner("mock", ws)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(make_runner("mock", ws, mock_dir=d).name, "mock")


if __name__ == "__main__":
    unittest.main()
