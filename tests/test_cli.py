import contextlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import fixtures

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMANDS = [
    "init", "validate", "register", "status",
    "promote", "approve", "answer", "next", "check-stop", "close-round", "stop", "report", "score", "brief", "run",
    "dashboard",
]


def run_rsi(*args):
    return subprocess.run(
        [sys.executable, "-m", "rsi", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=dict(os.environ, RSI_NO_BROWSER="1"),
    )


def run_rsi_without_dashboard(*args):
    """The CLI with rsi.dashboard unimportable, as if the page code had an import-time error."""
    code = "import sys; sys.modules['rsi.dashboard'] = None; from rsi.cli import main; sys.exit(main(sys.argv[1:]))"
    return subprocess.run([sys.executable, "-c", code, *args], cwd=REPO, capture_output=True, text=True,
                          env=dict(os.environ, RSI_NO_BROWSER="1"))


class TestCli(unittest.TestCase):
    def test_every_command_is_implemented(self):
        from rsi import cli
        result = run_rsi("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in COMMANDS:
            self.assertIn(name, result.stdout)
        self.assertEqual(run_rsi("bogus").returncode, 2)  # an unknown command
        self.assertEqual(sorted(cli.ARGUMENTS), sorted(COMMANDS))
        for name in COMMANDS:
            # Missing arguments: an argparse usage error, never a placeholder.
            result = run_rsi(name)
            self.assertEqual(result.returncode, 2, name)
            self.assertIn("usage:", result.stderr)
            self.assertNotIn("not implemented", result.stdout + result.stderr)


class TestDashboardCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.task = fixtures.binary_task()
        self.task_dir = fixtures.make_task_dir(os.path.join(self.tmp, "task"), self.task)
        self.ws = os.path.join(self.tmp, "ws")
        self.page = os.path.join(self.ws, "dashboard.html")
        self.n = 0

    def init(self):
        r = run_rsi("init", self.ws, "--task", self.task_dir)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r

    def register_testset(self):
        self.n += 1
        src = fixtures.make_artifact("testset", os.path.join(self.tmp, "src%d" % self.n), self.task,
                                     ids=range(30, 34), notes="source %d" % self.n)
        return run_rsi("register", self.ws, "testset", src)

    def read_page(self, page=None):
        with open(page or self.page, "rb") as f:
            return f.read()

    def run_main(self, argv, no_browser):
        """cli.main in-process with webbrowser.open mocked; returns the mock."""
        from rsi import cli
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("webbrowser.open") as opened, mock.patch.dict(os.environ), \
                mock.patch.object(cli.sys, "platform", "darwin"), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            os.environ.pop("RSI_NO_BROWSER", None)
            if no_browser:
                os.environ["RSI_NO_BROWSER"] = "1"
            code = cli.main(argv)
        self.assertEqual(code, 0, err.getvalue())
        return opened


    def test_refresh_after_each_state_command(self):
        # Every command in DASHBOARD_COMMANDS rewrites the pages: the page's mtime grows after each.
        from rsi import cli
        from rsi.workspace import Workspace
        import test_rounds
        task = fixtures.binary_task()
        task["approvals"] = task["approvals"] + ["plan"]
        self.task_dir = fixtures.make_task_dir(os.path.join(self.tmp, "task2"), task)
        self.task = task
        self.init()
        mtime = [os.stat(self.page).st_mtime_ns]

        def grew(*args):
            r = run_rsi(*args)
            self.assertEqual(r.returncode, 0, (args, r.stderr))
            now = os.stat(self.page).st_mtime_ns
            self.assertGreater(now, mtime[0], args[0])
            mtime[0] = now
            return r

        def mk(kind, **kw):
            self.n += 1
            return fixtures.make_artifact(kind, os.path.join(self.tmp, "src%d" % self.n), task,
                                          notes="source %d" % self.n, **kw)

        for kind, ids in (("testset", range(30, 34)), ("evalset", range(20, 24)), ("valset", range(10, 14)),
                          ("trainset", range(0, 4))):
            grew("register", self.ws, kind, mk(kind, ids=ids))
        grew("register", self.ws, "candidate", mk("candidate", ids=range(10, 14), train_ids=range(0, 4)))
        grew("register", self.ws, "delivery", mk("delivery", ids=range(20, 24), val_ids=range(10, 14),
                                                 candidate_id="candidate-0001"))
        ev = dict(delivery_id="delivery-0001", candidate_id="candidate-0001", evalset_id="evalset-0001",
                  best_before=None, ids=range(20, 24), train_ids=range(0, 4))
        grew("register", self.ws, "evaluation", mk("evaluation", **ev))
        grew("promote", self.ws, "evaluation-0001")
        issue = dict(fixtures._plan_issue(), attribution="model_capacity", module="train")
        grew("register", self.ws, "plan", mk("plan", evaluation_id="evaluation-0001", entry_module="train",
                                             issues=[issue]))
        grew("approve", self.ws, "plan", "--by", "alice", "--quote", "go", "--ref", "plan-0001")
        Workspace(self.ws).set_waiting({"kind": "question", "question": "用哪台机器？", "options": ["已回答"]})
        grew("answer", self.ws, "已回答", "--by", "alice", "--quote", "a100")
        grew("next", self.ws)
        # A round closed without an evaluation (round 2: the candidate failed its train gate).
        fail = {"passed": False, "detail": "below the majority baseline"}
        grew("register", self.ws, "candidate",
             mk("candidate", round=2, ids=range(10, 14), wrong=range(10, 14), train_ids=range(0, 4),
                train_gate=fail))
        grew("close-round", self.ws, "--reason", "no candidate passed the train gate")
        grew("stop", self.ws, "--by", "alice", "--quote", "enough")
        grew("report", self.ws, "--out", os.path.join(self.tmp, "REPORT.md"))
        self.assertEqual(set(cli.DASHBOARD_COMMANDS),
                         {"init", "register", "promote", "close-round", "approve", "answer", "next", "stop", "report"})

    def test_refused_command_leaves_dashboard(self):
        self.init()
        before = self.read_page()
        mtimes = [os.stat(os.path.join(self.ws, *rel.split("/"))).st_mtime_ns
                  for rel in ("dashboard.html", "dashboard/rounds.html", "dashboard/events.html", "dashboard/report.html")]
        r = run_rsi("register", self.ws, "testset", os.path.join(self.tmp, "no-such-dir"))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(self.read_page(), before)
        self.assertEqual(mtimes, [os.stat(os.path.join(self.ws, *rel.split("/"))).st_mtime_ns
                                  for rel in ("dashboard.html", "dashboard/rounds.html", "dashboard/events.html",
                                              "dashboard/report.html")])

    def test_dashboard_failure_is_a_warning(self):
        self.init()
        os.remove(self.page)
        os.mkdir(self.page)
        r = self.register_testset()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("testset-0001"), r.stdout)
        self.assertIn("warning: dashboard not updated", r.stderr)

    def test_broken_page_code_breaks_no_command(self):
        from rsi import cli, dashboard
        self.assertEqual(cli.DASHBOARD_PAGE, dashboard.DASHBOARD_FILE)
        self.init()
        src = fixtures.make_artifact("testset", os.path.join(self.tmp, "src-broken"), self.task, ids=range(30, 34))
        r = run_rsi_without_dashboard("validate", "testset", src, "--task", self.task_dir)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "OK"), r.stderr)
        r = run_rsi_without_dashboard("register", self.ws, "testset", src)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("testset-0001"), r.stdout)
        self.assertIn("warning: dashboard not updated", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        r = run_rsi_without_dashboard("dashboard", self.ws, "--no-open")
        self.assertEqual(r.returncode, 1)
        self.assertIn("cannot write", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_dashboard_command(self):
        self.init()
        os.remove(self.page)
        r = run_rsi("dashboard", self.ws, "--no-open")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), os.path.abspath(self.page))
        self.assertTrue(os.path.isfile(self.page))
        os.remove(self.page)  # a page that cannot be written is an error of the command
        os.mkdir(self.page)
        r = run_rsi("dashboard", self.ws, "--no-open")
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn(self.page, r.stderr)
        self.assertIn("cannot write", r.stderr)
        self.assertIn("Is a directory", r.stderr)

    def test_headless_unix_never_opens_a_browser(self):
        from rsi import cli
        self.init()
        env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", "RSI_NO_BROWSER")}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(cli.sys, "platform", "linux"), \
                mock.patch("webbrowser.open") as opened:
            self.assertFalse(cli.open_in_browser(self.page))
            opened.assert_not_called()
            with mock.patch.dict(os.environ, {"DISPLAY": ":0"}):
                cli.open_in_browser(self.page)
            opened.assert_called_once()

    def test_open_rules(self):
        self.ws = os.path.join(self.tmp, "my ws 工作区")  # the URI quotes a path with spaces and CJK
        self.page = os.path.join(self.ws, "dashboard.html")
        self.init()
        for case in ("env", "flag", "open"):
            args = ["dashboard", self.ws] + (["--no-open"] if case == "flag" else [])
            opened = self.run_main(args, no_browser=(case == "env"))
            if case == "open":
                opened.assert_called_once_with(pathlib.Path(self.page).resolve().as_uri())
            else:
                opened.assert_not_called()
        for case in ("env", "flag", "open"):
            ws2 = os.path.join(self.tmp, "ws2-" + case)
            args = ["init", ws2, "--task", self.task_dir] + (["--no-open"] if case == "flag" else [])
            opened = self.run_main(args, no_browser=(case == "env"))
            self.assertTrue(os.path.isfile(os.path.join(ws2, "dashboard.html")))
            if case == "open":
                opened.assert_called_once()
                uri = opened.call_args[0][0]
                self.assertTrue(uri.startswith("file://") and uri.endswith("/dashboard.html"), uri)
            else:
                opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
