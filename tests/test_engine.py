"""The `rsi run` engine loop (rsi/engine.py) with MockRunner and the tiny mock modules in tests/mock_modules:
a round from a fresh workspace to its decision, dispatch records and output directories, the run log,
registration of hand-backs, returns to the same session (and a fresh session with the history when it
cannot be resumed), decisions with the code's defaults, questions, long jobs, the manual runner, and
picking up again after the engine was interrupted at any point without dispatching or registering twice."""
import contextlib
import datetime
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import fixtures
from rsi import cli
from rsi import decisions as DP
from rsi import engine
from rsi import flow as F
from rsi.runners import ClaudeCodeRunner, ManualRunner, MockRunner
from rsi.runners.base import pid_alive as alive
from rsi.runners.base import result as agent_result
from rsi.workspace import RsiError, StoppedError, Workspace, now_iso

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCKS = os.path.join(REPO, "tests", "mock_modules")
sys.path.insert(0, os.path.join(REPO, "modules", "data"))
from dset.cli import main as dset_main  # noqa: E402
TEST_IDS, EVAL_IDS, VAL_IDS, TRAIN_IDS = range(10, 14), range(20, 24), range(0, 4), range(40, 44)
ROUND = ["dispatch", "wait_agent", "process_handback"]  # one module dispatched and handed back
SECOND_PART = ["resume", "wait_agent", "process_handback"]  # the second part of a two-part hand-back


def task_def(max_rounds=1, **flow):
    t = fixtures.binary_task()
    t["flow"] = dict(flow)
    t["stop"]["max_rounds"] = max_rounds
    return t


class EngineCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.behavior = {}
        self.use_task(task_def())
        env = mock.patch.dict(os.environ, {"RSI_MOCK_BEHAVIOR": os.path.join(self.root, "behavior.json"),
                                           "RSI_NO_BROWSER": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.track_mock_agents()

    def track_mock_agents(self):
        """Keep every handle MockRunner.launch returns during the test, and stop and reap its agents at cleanup
        (also when an assertion failed). A run that returns while an agent it launched still works (--max-steps, a
        question) leaves the agent running on purpose, as in production; the test must not leave it running while
        its temporary directory is removed."""
        self.mock_agents = []
        orig = MockRunner.launch

        def launch(runner, ws, d, prompt, resume_session=None, new_session=None):
            h = orig(runner, ws, d, prompt, resume_session=resume_session, new_session=new_session)
            self.mock_agents.append(h)
            return h

        patch = mock.patch.object(MockRunner, "launch", launch)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.reap_mock_agents)

    def reap_mock_agents(self):
        """Stop the mock agents this test launched that still run (the runner's stop: SIGTERM to the process group,
        then SIGKILL) and wait for each, so none is left running or unreaped."""
        for h in self.mock_agents:
            proc = h.get("proc")
            if proc is None:
                continue
            if proc.poll() is None:
                h["runner"].stop(h)
            proc.wait(timeout=30)

    def use_task(self, task, name="ws"):
        self.task = task
        self.ws = os.path.join(self.root, name)
        Workspace.init(self.ws, fixtures.make_task_dir(os.path.join(self.root, "task-" + name), task))

    def behave(self, module, **knobs):
        self.behavior.setdefault(module, {}).update(knobs)
        with open(os.path.join(self.root, "behavior.json"), "w", encoding="utf-8") as f:
            json.dump(self.behavior, f)

    def flag(self, name):
        return os.path.join(self.root, name + ".flag")

    def run_engine(self, runner=None, **kw):
        return engine.run(self.ws, runner or MockRunner(MOCKS), **kw)

    def state(self):
        return Workspace(self.ws)

    def log(self):
        with open(os.path.join(self.ws, "work", "run.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def actions(self):
        return [r["action"]["type"] for r in self.log() if r["event"] == "action"]

    def calls(self, did):
        """The mock module's calls in dispatch `did`'s output directory."""
        d = self.state().dispatch(did)
        path = os.path.join(self.ws, d["outdir"], "mock-calls.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def outfile(self, did, name):
        return os.path.join(self.ws, self.state().dispatch(did)["outdir"], name)

    def assertNoDoubleRegistration(self):
        entries = self.state().entries()
        sources = [e["source"] for e in entries]
        self.assertEqual(len(sources), len(set(sources)), sources)
        self.assertEqual(len(set(e["content_sha256"] for e in entries)), len(entries))

    def assertMilestoneAsk(self, reason):
        """The run of a one-round task stops at the milestone question once V1 is promoted."""
        self.assertIn("里程碑", reason)
        ws = self.state()
        self.assertEqual([v["version"] for v in ws.state["versions"]], ["V1"])
        self.assertEqual(ws.waiting()["what"], F.MILESTONE)
        self.assertEqual(F.next_action(ws)["type"], "ask")
        self.assertNoDoubleRegistration()


class TestOneRound(EngineCase):
    def test_fresh_workspace_to_the_decision(self):
        # A run.lock left by a killed run (its pid even alive in an unrelated process) holds no flock: no lock.
        with open(os.path.join(self.ws, "work", "run.lock"), "w") as f:
            f.write("%d\n" % os.getppid())
        reason = self.run_engine()
        self.assertMilestoneAsk(reason)
        self.assertEqual(len(reason.splitlines()), 1)
        ws = self.state()
        ds = ws.dispatches()
        self.assertEqual([(d["id"], d["module"], d["purpose"], d["status"], d["outdir"], d["returns"]) for d in ds], [
            ("d-0001", "data", "sets", "done", "work/round-1/data", 0),
            ("d-0002", "train", "train", "done", "work/round-1/train", 0),
            ("d-0003", "deliver", "deliver", "done", "work/round-1/deliver", 0),
            ("d-0004", "judge", "judge", "done", "work/round-1/judge", 0)])
        for d in ds:
            self.assertEqual(d["runner"], "mock")
            self.assertTrue(d["session"].startswith("mock-"), d)
            self.assertEqual(d["brief"], d["outdir"] + "/brief.md")
            with open(os.path.join(self.ws, d["outdir"], "brief.json"), encoding="utf-8") as f:
                brief = json.load(f)
            self.assertEqual((brief["purpose"], brief["dispatch_id"]), (d["purpose"], d["id"]))
            self.assertTrue(os.path.isfile(os.path.join(self.ws, d["brief"])))
            self.assertIsNotNone(d["ended_at"])
        self.assertEqual(ds[2]["candidate_id"], "candidate-0001")
        # The data module handed its sets back out of order; they are registered testset, evalset, valset, trainset.
        self.assertEqual([e["kind"] for e in ws.entries()], ["task", "testset", "evalset", "valset", "trainset",
                                                             "candidate", "search", "delivery", "evaluation"])
        self.assertEqual([r["kind"] for r in ds[0]["registered"]], ["testset", "evalset", "valset", "trainset"])
        self.assertEqual(ds[0]["refused"], [])
        # Every action is in the run log, in order, then why the run stopped.
        self.assertEqual(self.actions(), ROUND * 2 + SECOND_PART + ROUND * 2 + ["promote", "ask"])
        log = self.log()
        self.assertEqual(log[-1]["event"], "exit")
        self.assertEqual((log[-1]["status"], log[-1]["reason"]), ("waiting_user", reason))
        # Each action is logged before it is carried out, its result after.
        steps = [(r["event"], r["step"]) for r in log if r["event"] in ("action", "result")]
        self.assertEqual(steps, [(e, n) for n in range(1, 18) for e in ("action", "result")])
        launches = [r for r in log if r["event"] == "launch"]
        self.assertEqual([r["dispatch_id"] for r in launches], ["d-0001", "d-0002", "d-0002", "d-0003", "d-0004"])
        self.assertEqual([r["event"] for r in log].count("agent_end"), 5)
        # The pages were rewritten and show the question the engine stopped at.
        with open(os.path.join(self.ws, "dashboard.html"), encoding="utf-8") as f:
            self.assertIn("里程碑发布", f.read())
        # A second run changes nothing: the question is still open.
        self.assertEqual(self.run_engine(), reason)
        self.assertEqual(len(self.state().dispatches()), 4)

    def test_record_is_written_before_the_output_directory(self):
        seen = []
        orig = os.makedirs

        def spy(path, *a, **kw):
            if "work" + os.sep + "round-1" + os.sep in path + os.sep:
                seen.append(len(Workspace(self.ws).dispatches()))
            return orig(path, *a, **kw)

        with mock.patch("rsi.engine.os.makedirs", spy):
            self.run_engine(max_steps=1)
        self.assertTrue(seen)
        self.assertGreaterEqual(min(seen), 1)

    def test_max_steps(self):
        reason = self.run_engine(max_steps=2)
        self.assertIn("--max-steps", reason)
        self.assertEqual(self.actions(), ["dispatch", "wait_agent"])
        self.assertMilestoneAsk(self.run_engine())
        self.assertEqual(len(self.state().dispatches(module="data")), 1)


class TestTwoPart(EngineCase):
    def test_train_search_and_judge_plan_follow_in_the_same_session(self):
        task = task_def(max_rounds=5)
        task["approvals"].append("plan")
        self.use_task(task, "ws-two")
        reason = self.run_engine()
        ws = self.state()
        self.assertEqual((ws.waiting()["what"], ws.waiting()["ref"]), ("plan", "plan-0001"))
        self.assertIn("plan-0001", reason)
        ds = ws.dispatches()
        # The second parts are followups of the same session, no returns.
        self.assertEqual([(d["module"], d["purpose"], d["returns"], d.get("followups", 0)) for d in ds],
                         [("data", "sets", 0, 0), ("train", "train", 0, 1), ("deliver", "deliver", 0, 0),
                          ("judge", "judge", 0, 1)])
        self.assertEqual([(r["dispatch_id"], r["launch"]) for r in self.log() if r["event"] == "followup"],
                         [("d-0002", 2), ("d-0004", 2)])
        self.assertEqual([r["dispatch_id"] for r in self.log() if r["event"] == "resume"], [])
        self.assertEqual([ds[1]["launch_mode"], ds[3]["launch_mode"]], ["followup", "followup"])
        self.assertEqual(self.actions(), ROUND * 2 + SECOND_PART + ROUND * 2 + ["promote"] + SECOND_PART + ["ask"])
        # train and the judge got their second part in the same session, with the ids they need: what the first
        # part registered, each hand-back subdirectory with its id, also as registered-2.json beside the feedback.
        train = self.calls("d-0002")
        self.assertEqual([c["session"] for c in train], [ds[1]["session"]] * 2)
        self.assertIn("- `candidate` → candidate-0001（candidate）", train[1]["feedback"])
        self.assertIn("registered-2.json", train[1]["feedback"])
        with open(self.outfile("d-0002", "registered-2.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), [{"dir": "candidate", "kind": "candidate", "id": "candidate-0001"}])
        judge = self.calls("d-0004")
        self.assertEqual([c["session"] for c in judge], [ds[3]["session"]] * 2)
        self.assertIn("evaluation-0001", judge[1]["feedback"])
        self.assertEqual([e["kind"] for e in ws.entries()][-5:], ["candidate", "search", "delivery", "evaluation",
                                                                  "plan"])
        # The hand-back of the first part was kept, the second one written anew.
        self.assertTrue(os.path.isfile(self.outfile("d-0002", "handback-1.json")))
        with open(self.outfile("d-0002", "feedback-2.md"), encoding="utf-8") as f:
            self.assertIn("继续", f.read().splitlines()[0])
        # Each registered artifact notes the launch that handed it back.
        self.assertEqual([(x["kind"], x["launch"]) for x in ds[1]["registered"]], [("candidate", 1), ("search", 2)])
        self.assertNoDoubleRegistration()

    def test_a_first_part_over_two_launches_is_listed_whole(self):
        # R14b: registration refused one trial; the return registers it fixed; the search part lists both trials of the
        # first part with their subdirectories, as registered-<n>.json does (written for the return too).
        self.behave("train", trials=2, refuse_trial=True)
        self.assertMilestoneAsk(self.run_engine())
        d = self.state().dispatch("d-0002")
        self.assertEqual((d["launches"], d["returns"], d["followups"]), (3, 1, 1))

        def registered(n):
            with open(self.outfile("d-0002", "registered-%d.json" % n), encoding="utf-8") as f:
                return [(x["dir"], x["id"]) for x in json.load(f)]

        self.assertEqual(registered(2), [("candidate", "candidate-0001")])
        self.assertEqual(registered(3), [("candidate", "candidate-0001"), ("candidate-t2-fixed", "candidate-0002")])
        feedback = self.calls("d-0002")[2]["feedback"]
        for line in ("- `candidate` → candidate-0001（candidate）", "- `candidate-t2-fixed` → candidate-0002（candidate）"):
            self.assertIn(line, feedback)


class TestAsyncMockRunner(EngineCase):
    """MockRunner starts the mock script and the engine polls it (as with ClaudeCodeRunner): a stop is noticed while
    an agent works."""

    def run_in_thread(self, **kw):
        out = {}

        def target():
            try:
                out["result"] = engine.run_status(self.ws, MockRunner(MOCKS), **kw)
            except BaseException as e:  # reported by the test
                out["error"] = e

        t = threading.Thread(target=target)
        t.start()
        self.addCleanup(t.join, 60)
        return t, out

    def wait_for(self, path, limit=30):
        end = time.time() + limit
        while not os.path.exists(path):
            self.assertLess(time.time(), end, "timed out waiting for %s" % path)
            time.sleep(0.02)

    def wait_for_pid(self, path, limit=30):
        """The pid the mock agent writes into `path`, once the file holds one (it exists before it is written: an
        empty read would give pid 0, and signal 0 to pid 0 reaches the test's own process group)."""
        end = time.time() + limit
        while True:
            try:
                with open(path) as f:
                    text = f.read().strip()
            except OSError:
                text = ""
            if text.isdigit() and int(text) > 1:
                return int(text)
            self.assertLess(time.time(), end, "timed out waiting for a pid in %s" % path)
            time.sleep(0.02)

    def test_a_stop_ends_the_agent_that_works(self):
        started = self.flag("train-started")
        self.behave("train", started=started, release=self.flag("never"))
        open(started, "w").close()  # M8: the file exists before the agent writes its pid (the race, made certain)
        t, out = self.run_in_thread(poll_s=0.05)
        pid = self.wait_for_pid(started)
        self.assertTrue(alive(pid))
        t0 = time.time()
        Workspace(self.ws).stop("bob", "不要了")
        t.join(30)
        self.assertFalse(t.is_alive())
        self.assertNotIn("error", out)
        self.assertEqual(out["result"][0], "done")
        self.assertLess(time.time() - t0, 20)
        self.assertFalse(alive(pid))
        d = self.state().dispatch("d-0002")
        self.assertEqual(d["status"], "done")
        self.assertIn("agent 已终止", d["closed"])
        self.assertIn("agent_stopped", [r["event"] for r in self.log()])
        self.assertTrue(os.path.isfile(os.path.join(self.ws, "REPORT.md")))

class TestPendingRecords(EngineCase):
    """A dispatch record is pending until its agent is launched: launched_at and status running are written together,
    right before the runner is called; a record closed before its launch counts no launch."""

    def test_records_are_pending_until_each_launch(self):
        # Two records written in one dispatch: each says running (with launched_at) only once its runner is called.
        seen = []
        orig = MockRunner.launch

        def spy(runner, ws, d, prompt, resume_session=None, new_session=None):
            now = Workspace(ws.root)
            seen.append((d["id"], [(x["id"], x["status"], bool(x.get("launched_at"))) for x in now.dispatches()]))
            return orig(runner, ws, d, prompt, resume_session=resume_session, new_session=new_session)

        eng = engine.Engine(self.ws, MockRunner(MOCKS))
        with mock.patch.object(MockRunner, "launch", spy):
            eng.dispatch(self.state(), [{"module": "data", "purpose": "sets"}, {"module": "data", "purpose": "sets"}])
        self.assertEqual(seen, [
            ("d-0001", [("d-0001", "running", True), ("d-0002", "pending", False)]),
            ("d-0002", [("d-0001", "running", True), ("d-0002", "running", True)])])

    def test_a_return_stopped_before_its_launch_counts_no_launch(self):
        # The stop lands while the return is prepared (its brief rebuilt): feedback-2.md is written, the agent is
        # never launched; the record keeps one launch and no return.
        self.behave("data", trainset="quality_fail")
        orig, ws_path = engine.Engine.prepare, self.ws

        def prepare(eng, ws, d):
            if os.path.isfile(os.path.join(ws_path, d["outdir"], "handback.json")):
                Workspace(ws_path).stop("bob", "先停")
            return orig(eng, ws, d)

        with mock.patch.object(engine.Engine, "prepare", prepare):
            status, _reason = engine.run_status(self.ws, MockRunner(MOCKS))
        self.assertEqual(status, "done")
        d = self.state().dispatch("d-0001")
        self.assertEqual(d["status"], "done")
        self.assertIn("没有启动", d["closed"])
        self.assertIsNone(d["launched_at"])
        self.assertEqual((d["launches"], d["returns"], len(self.calls("d-0001"))), (1, 0, 1))
        self.assertTrue(os.path.isfile(self.outfile("d-0001", "feedback-2.md")))


class TestReturns(EngineCase):
    def test_quality_gate_goes_back_to_the_same_data_session(self):
        self.behave("data", trainset="quality_fail")
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        d = ws.dispatch("d-0001")
        self.assertEqual(d["returns"], 1)
        self.assertEqual(d["status"], "done")
        self.assertEqual(ws.decisions(), [dict(ws.decisions()[0], point="quality_gate_failed", choice="resume_data",
                                               by="code", dispatch_id="d-0001")])
        calls = self.calls("d-0001")
        self.assertEqual([c["session"] for c in calls], [d["session"]] * 2)
        self.assertIsNone(calls[0]["feedback"])
        self.assertEqual(calls[1]["feedback_file"], "feedback-2.md")
        for word in ("dup_rate", "0.5", "0.05", "trainset-0001"):
            self.assertIn(word, calls[1]["feedback"])
        self.assertEqual(ws.current("trainset"), "trainset-0002")
        self.assertEqual(self.actions()[:6], ROUND + ["decide", "wait_agent", "process_handback"])
        # The resume's read-deny list is computed again from its rebuilt brief.
        launches = [r for r in self.log() if r["event"] == "launch" and r["dispatch_id"] == "d-0001"]
        self.assertGreaterEqual(len(launches), 2)
        self.assertEqual(launches[0]["denied"], [])  # nothing registered yet
        self.assertTrue(launches[1]["denied"])  # the sets exist now and the "sets" brief gives none of them
        self.assertTrue(all(os.path.isabs(p) for p in launches[1]["denied"]))
        self.assertEqual(ws.dispatch("d-0001")["denied"], launches[-1]["denied"])

    def test_refused_registration_goes_back_to_the_same_session(self):
        self.behave("data", trainset="overlap")
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        d = ws.dispatch("d-0001")
        self.assertEqual(d["returns"], 1)
        self.assertEqual(len(d["refused"]), 1)
        self.assertEqual((d["refused"][0]["dir"], d["refused"][0]["launch"]), ("trainset", 1))
        self.assertIn("overlaps the current evalset", d["refused"][0]["error"])
        self.assertEqual([r["kind"] for r in d["registered"]], ["testset", "evalset", "valset", "trainset"])
        d0 = ws.decisions()[0]
        self.assertEqual((d0["point"], d0["choice"], d0["by"]), ("handback_refused", "resume", "code"))
        # Decided while the hand-back was processed: no decide action, the agent went straight back.
        self.assertEqual(self.actions()[:5], ROUND + ["wait_agent", "process_handback"])
        first = [r for r in self.log() if r["event"] == "result" and r["type"] == "process_handback"][0]
        self.assertEqual(first["result"]["decision"]["point"], "handback_refused")
        calls = self.calls("d-0001")
        self.assertEqual(calls[1]["session"], calls[0]["session"])
        self.assertIn("overlaps the current evalset", calls[1]["feedback"])
        self.assertIn("新的子目录", calls[1]["feedback"])
        self.assertNotIn("没有合法的交回", calls[1]["feedback"])
        self.assertEqual(len(ws.dispatches(module="data")), 1)

    def test_a_session_that_cannot_be_resumed_gets_a_fresh_one_with_the_history(self):
        self.behave("data", trainset="quality_fail", fail_resume=self.flag("resume"))
        self.assertMilestoneAsk(self.run_engine())
        d = self.state().dispatch("d-0001")
        self.assertEqual(d["returns"], 1)
        calls = self.calls("d-0001")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1]["session"], calls[0]["session"])  # the resume, refused by the mock
        self.assertNotEqual(calls[2]["session"], calls[0]["session"])
        self.assertEqual(d["session"], calls[2]["session"])
        self.assertTrue(calls[2]["feedback_file"].startswith("history-"))
        history = calls[2]["feedback"]
        self.assertIn("你负责 RSI 运行里的一个模块", history)  # the previous brief
        self.assertIn('"status": "ok"', history)  # the previous hand-back
        self.assertIn("dup_rate", history)  # the feedback
        self.assertIn("resume_failed", [r["event"] for r in self.log()])

    def test_a_denied_path_no_safe_rule_can_name_stops_the_launch(self):
        self.use_task(task_def(), "ws(1)")
        self.behave("data", trainset="quality_fail")
        with mock.patch.object(MockRunner, "enforces_deny", True, create=True):
            with self.assertRaises(RsiError) as e:
                self.run_engine()
        self.assertIn("cannot deny reads", str(e.exception))
        launches = [r for r in self.log() if r["event"] == "launch"]
        self.assertEqual(len(launches), 1)  # the resume (denied sets exist by then) was not launched


class TestDataHandback(EngineCase):
    def data_dispatch(self, task, edit_task=None, edit_md=None, approve=None):
        """A new workspace of `task` with a round-1 data dispatch (sets) whose output directory holds what the data
        agent hands back: corpus/ (fixtures.corpus_rows and a script), and what `dset sets` (dset.cli.main) writes
        from it with the rule fixtures.TIME (split.json): task/ (version 2, the task's sizes filled in), the corpus
        manifest, the four sets, pool/ and trainset/; then a record at version 2. edit_task: task.json also changes
        these keys; edit_md: TASK.md gets this text appended; approve: an approval recorded first. Returns
        (workspace, dispatch record); self.engine is an engine on that workspace."""
        self.n = getattr(self, "n", 0) + 1
        self.use_task(task, "ws-data-%d" % self.n)
        ws = Workspace(self.ws)
        if approve:
            ws.approve(approve, "alice", "ok")
        did = ws.add_dispatches([{"module": "data", "purpose": "sets", "outdir": "work/round-1/data"}])[0]
        out = os.path.join(self.ws, "work", "round-1", "data")
        os.makedirs(os.path.join(out, "corpus"))
        fixtures.write_jsonl(os.path.join(out, "corpus", "rows.jsonl"), fixtures.corpus_rows(task))
        with open(os.path.join(out, "corpus", "preprocess.py"), "w", encoding="utf-8") as f:
            f.write(fixtures.PREPROCESS_STUB)
        with open(os.path.join(out, "split.json"), "w", encoding="utf-8") as f:
            json.dump(fixtures.TIME, f)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = dset_main(["sets", "--task", os.path.join(ws.path(ws.current("task")), "task.json"),
                              "--corpus", os.path.join(out, "corpus"), "--script", "preprocess.py",
                              "--command", "python3 preprocess.py", "--label-method", "given",
                              "--split", os.path.join(out, "split.json"), "--round", "1", "--out", out])
        self.assertEqual(code, 0, err.getvalue())
        if edit_task:
            with open(os.path.join(out, "task", "task.json"), encoding="utf-8") as f:
                new = json.load(f)
            with open(os.path.join(out, "task", "task.json"), "w", encoding="utf-8") as f:
                json.dump(dict(new, **edit_task), f)
        if edit_md:
            with open(os.path.join(out, "task", "TASK.md"), "ab") as f:
                f.write(edit_md.encode("utf-8"))
        fixtures.make_artifact("record", os.path.join(out, "record"), dict(task, version=2), artifacts=[])
        self.engine = engine.Engine(self.ws, MockRunner(MOCKS), pages=False)
        ws = Workspace(self.ws)
        return ws, ws.dispatch(did)

    def test_data_handback_registers_the_rule_fill_first(self):
        ws, d = self.data_dispatch(task=fixtures.time_task())  # round 1, sets, handed back as dset sets writes it
        names = ["record", "trainset", "pool", "valset", "evalset", "testset", "corpus", "task"]
        registered, refused, stopped = self.engine.register(ws, d, names)
        self.assertEqual([x["kind"] for x in registered],
                         ["task", "corpus", "testset", "evalset", "valset", "pool", "trainset", "record"])
        self.assertEqual((refused, stopped, ws.state["approvals"], ws.state["task_version"]), ([], None, [], 2))
        self.assertEqual(ws.entry("task-0002")["producer"], "data")  # rsi-data wrote the rule
        self.assertIsNone(F.data_missing(ws, [ws.entry(a) for a in ws.state["rounds"]["1"]["artifacts"]]))
        # A task/ that also changes eval_use_limit, or whose TASK.md says more than the rule, is refused, and an open
        # task_change approval stays unused (the fill is checked on the staged copy, TASK.md byte for byte).
        for edit in ({"edit_task": {"eval_use_limit": 9}}, {"edit_md": "\n另加的一段。\n"}):
            ws, d = self.data_dispatch(task=fixtures.time_task(), approve="task_change", **edit)
            registered, refused, _ = self.engine.register(ws, d, ["task", "corpus"])
            self.assertEqual(refused[0]["kind"], "task"); self.assertIn("rsi-data", refused[0]["error"])
            first = refused[0]["error"].splitlines()[0]  # A-4: the decision's reason quotes only the first line
            self.assertTrue(first.startswith(engine.TASK_REFUSED) and len(first) > len(engine.TASK_REFUSED), first)
            self.assertIsNone(ws.state["approvals"][0].get("used_by"))
        # Only rsi-data hands back a task/.
        self.assertIn("rsi-data", self.engine._kind_of(os.path.join(self.engine.outdir(d), "task"),
                                                       dict(d, module="train"))[1])

    def test_a_refused_corpus_keeps_the_sets_out(self):
        # The corpus is refused (its sources name no path of data.sources): the sets after it are refused unregistered,
        # so no evalset or testset fixes the rule and the corrected corpus still registers before them.
        ws, d = self.data_dispatch(task=fixtures.time_task())
        out = self.engine.outdir(d)
        fixtures.update_manifest(os.path.join(out, "corpus"), sources=["/elsewhere/raw"])
        names = ["task", "corpus", "testset", "evalset", "valset", "pool", "trainset", "record"]
        registered, refused, _ = self.engine.register(ws, d, names)
        self.assertEqual([x["kind"] for x in registered], ["task", "record"])
        self.assertEqual([x["kind"] for x in refused], ["corpus", "testset", "evalset", "valset", "pool", "trainset"])
        self.assertIn("data.sources", refused[0]["error"])
        self.assertIn("corpus（子目录 corpus）被拒，后面依赖它的集合不登记", refused[1]["error"])
        reason = F.refused_decision(ws, dict(d, launches=1, refused=refused))["reason"]  # the line once, with the kinds
        self.assertEqual(reason.count("后面依赖它的集合不登记"), 1)
        self.assertIn("（未登记：testset、evalset、valset、pool、trainset）", reason)
        ws = Workspace(self.ws)
        self.assertEqual([ws.current(k) for k in ("corpus", "testset", "evalset")], [None, None, None])
        self.assertTrue(all(os.path.isdir(os.path.join(out, k)) for k in ("testset", "evalset")))
        fixtures.update_manifest(os.path.join(out, "corpus"), sources=["/raw/rows"])  # the agent fixes it
        registered, refused, _ = self.engine.register(ws, d, names)
        self.assertEqual(([x["kind"] for x in registered], refused),
                         (["task", "corpus", "testset", "evalset", "valset", "pool", "trainset", "record"], []))


class TestDecisions(EngineCase):
    def test_search_without_a_passing_candidate_goes_back_then_closes(self):
        # Review focus 1: no trial of the search passes the train gate: back to the same train session for a new
        # search (the second attempt, a return); with the attempts used up the round is closed: no loop.
        task = task_def()
        task["stop"]["max_attempts_per_round"] = 2
        self.use_task(task, "ws-no-pass")
        self.behave("train", gate_fail="always")
        status, reason = engine.run_status(self.ws, MockRunner(MOCKS))
        self.assertEqual(status, "done", reason)
        ws = self.state()
        self.assertEqual([(x["point"], x["choice"], x["by"]) for x in ws.decisions()],
                         [(F.TRAIN_GATE_FAILED, "train", "code")])
        self.assertEqual([d["module"] for d in ws.dispatches()], ["data", "train"])
        d = ws.dispatch("d-0002")
        self.assertEqual((d["launches"], d["returns"], d["followups"]), (4, 1, 2))
        self.assertEqual(ws.attempts(rnd=1), ["search-0001", "search-0002"])
        self.assertEqual(ws.passed_candidates(rnd=1), [])
        decision = ws.state["rounds"]["1"]["decision"]
        self.assertEqual((decision["decision"], decision["evaluation_id"]), ("keep", None))
        self.assertEqual(self.actions(), ROUND * 2 + SECOND_PART + ["decide", "wait_agent", "process_handback"]
                         + SECOND_PART + ["close_round", "next", "report", "done"])
        calls = self.calls("d-0002")
        self.assertEqual([c["session"] for c in calls], [d["session"]] * 4)
        self.assertIn("没过训练门", calls[2]["feedback"])

    def test_train_gate_failed_sends_train_back_for_a_new_search(self):
        self.behave("train", gate_fail=True)  # the first search's trials fail
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        self.assertEqual([(x["point"], x["choice"], x["by"]) for x in ws.decisions()],
                         [("train_gate_failed", "train", "code")])
        ds = ws.dispatches()
        self.assertEqual([(d["module"], d["purpose"]) for d in ds],
                         [("data", "sets"), ("train", "train"), ("deliver", "deliver"), ("judge", "judge")])
        train = self.calls("d-0002")
        self.assertEqual([c["session"] for c in train], [ds[1]["session"]] * 4)  # back to the same train session
        # It is told the gate numbers of the trials that failed.
        self.assertIn("候选 candidate-0001（train）没过训练门", train[2]["feedback"])
        self.assertIn('"majority"', train[2]["feedback"])
        self.assertEqual((ds[1]["returns"], ds[1]["followups"]), (1, 2))
        self.assertEqual(ws.attempts(), ["search-0001", "search-0002"])
        self.assertEqual(ds[2]["candidate_id"], "candidate-0002")  # the second search's trial, through the gate

    def test_at_the_cap_the_decision_asks_the_user(self):
        self.use_task(task_def(max_returns=0), "ws-cap")
        self.behave("data", trainset="quality_fail")
        reason = self.run_engine()
        ws = self.state()
        self.assertEqual([(x["point"], x["choice"], x["by"]) for x in ws.decisions()],
                         [("quality_gate_failed", "ask", "code")])
        w = ws.waiting()
        self.assertEqual((w["kind"], w["what"], w["ref"], w["options"]),
                         ("question", "quality_gate_failed", "d-0001", ["resume", "new_session", "close_round", "stop"]))
        self.assertIn(w["question"], reason)
        self.assertEqual(self.actions(), ROUND + ["decide"])
        self.assertEqual(self.run_engine(), reason)  # still waiting: nothing dispatched
        self.assertEqual(len(self.state().dispatches()), 1)
        self.assertEqual(self.state().dispatch("d-0001")["returns"], 0)
        # The question names the step's current part; new_session sends a fresh data session for it.
        self.assertIn("目的 trainset", w["question"])
        DP.answer(self.state(), "new_session", "alice", "换个会话补训练集")
        self.assertMilestoneAsk(self.run_engine())
        d = self.state().dispatch("d-0002")
        self.assertEqual((d["module"], d["purpose"]), ("data", "trainset"))


class TestStatuses(EngineCase):
    def test_why_the_run_stopped(self):
        self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS), max_steps=1)[0], "max_steps")
        self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS))[0], "waiting_user")
        # The user says no to the milestone release: the run stops and reports, without using the testset.
        DP.answer(self.state(), "no", "alice", "不用 test")
        status, reason = engine.run_status(self.ws, MockRunner(MOCKS))
        self.assertEqual(status, "done")
        ws = self.state()
        d = ws.decisions()[-1]
        self.assertEqual((d["point"], d["choice"], d["by"]), (F.MILESTONE, "no", "alice"))
        self.assertEqual([a for a in ws.state["approvals"] if a["what"] == "use_testset"], [])
        self.assertIn("REPORT.md", reason)
        self.assertTrue(os.path.isfile(os.path.join(self.ws, "REPORT.md")))
        self.assertEqual(self.actions()[-3:], ["next", "report", "done"])
        self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS)), (status, reason))

    def test_a_stale_wait_is_cleared_and_logged(self):
        # (An approval given after the question clears it at once; this one was given before it was asked.)
        ws = self.state()
        ws.approve("refresh_sets", "alice", "ok")
        ws.set_waiting({"kind": "approval", "what": "refresh_sets", "question": "批准刷新？",
                        "options": ["approve", "stop"]})
        self.run_engine(max_steps=1)
        self.assertIsNone(self.state().waiting())
        cleared = [r for r in self.log() if r["event"] == "clear_waiting"]
        self.assertEqual(cleared[0]["waiting"]["what"], "refresh_sets")


class TestQuestionsAndLongJobs(EngineCase):
    def test_a_question_holds_the_handback_until_the_answer(self):
        # data hands back its sets with a question: nothing is registered before the user answers; the answer goes
        # back to the same session (no return), the module hands back again (here: a changed trainset), and that
        # hand-back is registered.
        self.behave("data", questions=["要不要多造一些短消息？"])
        reason = self.run_engine()
        self.assertIn("要不要多造一些短消息？", reason)
        ws = self.state()
        d = ws.dispatch("d-0001")
        self.assertEqual(d["status"], "handed_back")
        self.assertEqual((d["asked"]["launch"], d["asked"]["questions"]), (1, ["要不要多造一些短消息？"]))
        self.assertEqual([e["kind"] for e in ws.entries()], ["task"])  # nothing registered yet
        self.assertIsNone(ws.current("trainset"))
        w = ws.waiting()
        self.assertEqual((w["kind"], w["what"], w["ref"], w["options"]), ("question", "handback", "d-0001", ["已回答"]))
        self.assertEqual(d["asked"]["question_id"], w["id"])
        self.assertIn("登记", w["question"])  # says the hand-back waits for the answer
        self.assertEqual(self.run_engine(), reason)  # still waiting: nothing registered, nothing dispatched
        self.assertEqual((len(self.state().dispatches()), len(self.calls("d-0001"))), (1, 1))
        self.assertEqual([e["kind"] for e in self.state().entries()], ["task"])
        from rsi import decisions as DP
        rec = DP.answer(self.state(), "已回答", "alice", "要，多造短消息", ref=w["id"])
        self.assertIn("d-0001", rec["effect"])
        a = F.next_action(self.state())
        self.assertEqual((a["type"], a["dispatch_id"], a["mode"]), ("resume", "d-0001", "answer"))
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        d = ws.dispatch("d-0001")
        self.assertEqual((d["launches"], d["returns"], d["answered"], d["launch_mode"]), (2, 0, 1, "answer"))
        calls = self.calls("d-0001")
        self.assertEqual([c["session"] for c in calls], [d["session"]] * 2)
        self.assertEqual(calls[1]["feedback_file"], "feedback-2.md")
        self.assertIn("要，多造短消息", calls[1]["feedback"])
        self.assertIn("要不要多造一些短消息？", calls[1]["feedback"])
        self.assertIn("还没有登记", calls[1]["feedback"])
        # The registered trainset is the one handed back after the answer.
        cur = ws.current("trainset")
        self.assertEqual(ws.entry(cur)["source"], self.outfile("d-0001", "trainset-2"))
        self.assertIn("要，多造短消息", ws.manifest(cur)["notes"])
        self.assertEqual([r["kind"] for r in d["registered"]], ["testset", "evalset", "valset", "trainset"])
        self.assertEqual(set(r["launch"] for r in d["registered"]), {2})
        rec = [x for x in ws.decisions() if x["point"] == "handback"][0]
        self.assertEqual(rec["delivered_to"], ["d-0001 launch 2"])
        self.assertEqual([(r["dispatch_id"], r["launch"]) for r in self.log() if r["event"] == "resume_with_answer"],
                         [("d-0001", 2)])

    def test_an_answer_cut_off_after_its_delivery_mark_still_reaches_the_session(self):
        # M3 (final review): the run is cut off in resume(mode answer) after the answer was marked delivered and before
        # the record says the new launch; the rerun sends the answer back again, with the user's words.
        self.behave("data", questions=["要不要多造一些短消息？"])
        self.run_engine()
        from rsi import decisions as DP
        ws = self.state()
        DP.answer(ws, "已回答", "alice", "要，多造短消息", ref=ws.waiting()["id"])
        with mock.patch.object(engine.Engine, "_archive", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_engine()
        d = self.state().dispatch("d-0001")
        self.assertEqual((d["status"], d["launches"]), ("handed_back", 1))  # the new launch is not recorded
        rec = [x for x in self.state().decisions() if x["point"] == "handback"][0]
        self.assertEqual(rec["delivered_to"], ["d-0001 launch 2"])  # but the answer was marked delivered
        self.assertMilestoneAsk(self.run_engine())
        d = self.state().dispatch("d-0001")
        self.assertEqual((d["launches"], d["answered"], d["returns"]), (2, 1, 0))
        calls = self.calls("d-0001")
        self.assertEqual(calls[1]["feedback_file"], "feedback-2.md")
        self.assertIn("要，多造短消息", calls[1]["feedback"])
        self.assertIn("要不要多造一些短消息？", calls[1]["feedback"])
        rec = [x for x in self.state().decisions() if x["point"] == "handback"][0]
        self.assertEqual(rec["delivered_to"], ["d-0001 launch 2"])  # once

    def test_a_stop_while_a_question_waits_registers_only_records(self):
        self.behave("data", questions=["要不要多造一些短消息？"])
        self.run_engine()
        Workspace(self.ws).stop("bob", "先停")
        self.assertIn("REPORT.md", self.run_engine())
        ws = self.state()
        d = ws.dispatch("d-0001")
        self.assertEqual(d["status"], "done")
        self.assertIn("运行已停止", d["closed"])
        self.assertEqual([e["kind"] for e in ws.entries()], ["task"])
        self.assertIsNone(ws.waiting())

    def test_long_job_waits_for_its_marker_then_continues_the_session(self):
        self.behave("train", long_job=True)
        status, reason = engine.run_status(self.ws, MockRunner(MOCKS))
        self.assertEqual(status, "waiting_long_job")
        self.assertIn("长任务", reason)
        d = self.state().dispatch("d-0002")
        self.assertEqual(d["status"], "long_job")
        marker = self.outfile("d-0002", "job.marker")
        self.assertEqual(d["marker"], marker)
        self.assertEqual(self.run_engine(), reason)  # no marker yet: still waiting, nothing dispatched
        self.assertEqual(len(self.state().dispatches()), 2)
        with open(marker, "w") as f:
            f.write("ok\n")
        self.assertMilestoneAsk(self.run_engine())
        d = self.state().dispatch("d-0002")
        self.assertEqual((d["returns"], d["continued"], d["followups"]), (0, 1, 1))
        calls = self.calls("d-0002")
        self.assertEqual([c["session"] for c in calls], [d["session"]] * 3)
        self.assertIn("长任务已结束", calls[1]["feedback"])

    def test_checkins_until_the_marker(self):
        # Review focus 2: a long job that goes on: its session is checked in on every flow.checkin_minutes (counted
        # apart, no return); at flow.max_checkins the user is asked; once the marker is there the question is moot
        # and no check-in follows.
        self.use_task(task_def(checkin_minutes=0.0005, max_checkins=2), "ws-checkin")
        self.behave("train", long_job=True)
        status, reason = engine.run_status(self.ws, MockRunner(MOCKS))
        self.assertEqual(status, "waiting_user", reason)
        ws = self.state()
        d = ws.dispatch("d-0002")
        self.assertEqual((d["status"], d["launches"], d["checkins"], d.get("returns")), ("long_job", 3, 2, 0))
        w = ws.waiting()
        self.assertEqual((w["what"], w["ref"], w["options"]), ("checkin", "d-0002", ["continue", "wait", "stop"]))
        self.assertEqual([(r["dispatch_id"], r["launch"]) for r in self.log() if r["event"] == "checkin"],
                         [("d-0002", 2), ("d-0002", 3)])
        calls = self.calls("d-0002")
        self.assertEqual([c["session"] for c in calls], [d["session"]] * 3)
        self.assertTrue(calls[1]["feedback"].startswith("# 定时检查"), calls[1]["feedback"])
        with open(self.outfile("d-0002", "job.marker"), "w") as f:
            f.write("ok\n")
        self.assertMilestoneAsk(self.run_engine())
        d = self.state().dispatch("d-0002")
        self.assertEqual((d["checkins"], d["continued"], d["returns"]), (2, 1, 0))
        self.assertEqual(len([r for r in self.log() if r["event"] == "checkin"]), 2)

    def test_failed_long_job_is_a_failed_dispatch(self):
        self.behave("train", long_job=True)
        self.run_engine()
        with open(self.outfile("d-0002", "job.marker"), "w") as f:
            f.write("failed\nout of memory\n")
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        self.assertEqual(ws.decisions()[0]["point"], "handback_failed")
        calls = self.calls("d-0002")
        self.assertIn("out of memory", calls[1]["feedback"])
        self.assertEqual(ws.dispatch("d-0002")["returns"], 1)


def sleeper(case):
    """A process that sleeps 60 s, in its own process group and no child of this process; returns its pid."""
    code = ("import subprocess, sys; print(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
            "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid)")
    pid = int(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout)
    case.addCleanup(lambda: alive(pid) and os.kill(pid, signal.SIGKILL))
    return pid


def failed_check_in(case, name):
    """A long job's check-in failed with no returns left: the run asks about d-0002 (handback_failed). The job is a
    sleeper named in job.marker.pid. Returns (its pid, STATE waiting)."""
    case.use_task(task_def(checkin_minutes=0.0005, max_returns=0), name)
    case.behave("train", long_job=True, checkin={"fail": "查不了环境"})
    status, reason = engine.run_status(case.ws, MockRunner(MOCKS))
    case.assertEqual(status, "waiting_user", reason)
    w = case.state().waiting()
    case.assertEqual((w["what"], w["ref"], case.state().dispatch("d-0002")["launch_mode"]),
                     (F.HANDBACK_FAILED, "d-0002", "checkin"))
    pid = sleeper(case)
    with open(case.outfile("d-0002", "job.marker") + ".pid", "w") as f:
        f.write("%d\n%d\n" % (pid, int(time.time())))
    return pid, w


class TestLiveStatus(EngineCase):
    """While the engine waits it refreshes the pages (after a GPU snapshot) every flow.page_refresh_seconds and prints
    a status line every flow.status_minutes; each step prints one when the state changed (rsi.live.now_key)."""

    def test_pages_and_status_refresh_while_waiting(self):
        Workspace(self.ws).add_dispatches([{"module": "train", "outdir": "work/round-1/train", "purpose": "train",
                                            "status": "running", "launched_at": now_iso()}])
        buf, err = io.StringIO(), io.StringIO()
        eng = engine.Engine(self.ws, MockRunner(MOCKS), status=buf)
        missing = {"at": now_iso(), "status": "missing", "error": None, "uid": os.getuid(), "gpus": []}
        with mock.patch("rsi.dashboard.write_dashboard") as pages, \
                mock.patch("rsi.gpus.snapshot", return_value=missing) as snap, \
                mock.patch("rsi.gpus.write_snapshot") as saved, contextlib.redirect_stderr(err):
            eng.pages_at = eng.said_at = 1000.0
            eng.tick(self.state(), 1030)
            self.assertEqual((pages.call_count, snap.call_count), (0, 0))
            eng.tick(self.state(), 1061)  # page_refresh_seconds (60): a snapshot, then the pages
            self.assertEqual((pages.call_count, snap.call_count, saved.call_count), (1, 1, 1))
            self.assertEqual(saved.call_args[0], (self.ws, missing))
            self.assertEqual(buf.getvalue(), "")
            eng.tick(self.state(), 1301)  # status_minutes (5): a heartbeat line
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertRegex(lines[0], r"^\d\d:\d\d 第1轮 ② 训练调优 d-0001 · rsi-train 运行中（第 1 次启动）$")
        self.assertEqual(pages.call_count, 2)
        self.assertEqual(err.getvalue(), "")  # no nvidia-smi is no failure
        # A status stream that cannot be written to (its reader gone) ends the status lines, never the run.
        broken = mock.Mock(write=mock.Mock(side_effect=BrokenPipeError(32, "Broken pipe")))
        eng.status = broken
        with contextlib.redirect_stderr(err):
            eng.say(self.state(), heartbeat=True)
            eng.say(self.state(), heartbeat=True)
        self.assertIsNone(eng.status)
        self.assertEqual(err.getvalue().count("warning: status line failed: [Errno 32] Broken pipe"), 1, err.getvalue())

    def test_page_and_gpu_failures_only_warn_once(self):
        # Review focus 4: every refresh of a long wait fails to write the pages (an OSError naming another temp file each
        # time) and to take a snapshot (first it raises, then nvidia-smi times out); the wait still ends when the marker
        # appears, and each kind of failure is warned about once.
        self.use_task(task_def(page_refresh_seconds=0.05, checkin_minutes=10), "ws-warn")
        marker = os.path.join(self.ws, "work", "round-1", "train", "job.done")
        os.makedirs(os.path.dirname(marker))
        did = Workspace(self.ws).add_dispatches([{"module": "train", "outdir": "work/round-1/train", "marker": marker,
                                                  "status": "long_job", "ended_at": now_iso()}])[0]

        def finish():
            with open(marker, "w") as f:
                f.write("ok\n")

        timer = threading.Timer(0.4, finish)
        timer.start()
        self.addCleanup(timer.cancel)
        due = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        eng = engine.Engine(self.ws, MockRunner(MOCKS), poll_s=0.02)
        err = io.StringIO()

        def disk_full(_ws):
            raise OSError(28, "No space left on device", ".dashboard.%s.tmp" % os.urandom(4).hex())

        timed_out = {"at": now_iso(), "status": "error", "error": "nvidia-smi 超时（5 秒没有返回）", "uid": 0, "gpus": []}
        with mock.patch("rsi.dashboard.write_dashboard", side_effect=disk_full) as pages, \
                mock.patch("rsi.gpus.snapshot", side_effect=[RuntimeError("boom")] + [timed_out] * 100) as snap, \
                contextlib.redirect_stderr(err):
            self.assertEqual(eng.wait_long_job(due), {"marker": [did]})
        text = err.getvalue()
        self.assertGreaterEqual(min(pages.call_count, snap.call_count), 3)
        self.assertEqual(text.count("warning: dashboard not updated: [Errno 28] No space left on device"), 1, text)
        self.assertEqual(text.count("warning: GPU snapshot failed: boom"), 1, text)
        self.assertEqual(text.count("warning: GPU snapshot failed: nvidia-smi 超时"), 1, text)
        self.assertEqual(len(text.splitlines()), 3, text)

    def test_status_lines_on_state_changes(self):
        buf = io.StringIO()
        status, _reason = engine.run_status(self.ws, MockRunner(MOCKS), status=buf)
        self.assertEqual(status, "waiting_user")
        lines = buf.getvalue().splitlines()
        self.assertTrue(any("② 训练调优 d-" in line for line in lines), lines)
        for line in lines:
            self.assertRegex(line, r"^\d\d:\d\d 第\d+轮 ")
        # Stopped at the milestone question: the next run asks again and stops; its first step prints one line.
        buf = io.StringIO()
        engine.run_status(self.ws, MockRunner(MOCKS), status=buf)
        self.assertEqual(len(buf.getvalue().splitlines()), 1, buf.getvalue())


class TestLongJobStop(EngineCase):
    """A stop with a long job still running: the job's process, named in <marker>.pid (first line the pid, an
    optional second line its start time in Unix seconds), is terminated when it is confirmed to be the job (alive,
    started at or after the dispatch's launch, at the start time the file gives); otherwise it is listed as a long job
    that may still be running. The stop reason, the record (long_job_stop) and REPORT.md say which."""

    def long_job(self, name):
        self.use_task(task_def(), name)
        self.behave("train", long_job=True)
        self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS))[0], "waiting_long_job")
        return self.outfile("d-0002", "job.marker")

    def stop_and_report(self):
        Workspace(self.ws).stop("bob", "不要了")
        status, reason = engine.run_status(self.ws, MockRunner(MOCKS))
        self.assertEqual(status, "done")
        d = self.state().dispatch("d-0002")
        self.assertEqual(d["status"], "done")
        with open(os.path.join(self.ws, "REPORT.md"), encoding="utf-8") as f:
            return reason, d, f.read()

    def test_a_confirmed_long_job_is_terminated(self):
        marker = self.long_job("ws-lj1")
        pid = sleeper(self)
        with open(marker + ".pid", "w") as f:
            f.write("%d\n%d\n" % (pid, int(time.time())))
        # A check-in relaunched the session after the job started: the job is still told by the launch that started it.
        later = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 5))
        Workspace(self.ws).update_dispatch("d-0002", launched_at=later, launches=2, launch_mode="checkin", checkins=1)
        reason, d, report = self.stop_and_report()
        self.assertFalse(alive(pid))
        self.assertEqual((d["long_job_stop"]["result"], d["long_job_stop"]["pid"]), ("terminated", pid))
        self.assertIn("终止了长任务 d-0002", reason)
        self.assertIn(str(pid), reason)
        self.assertIn("d-0002", report)
        self.assertIn("已终止", report)
        self.assertIn("long_job_stopped", [r["event"] for r in self.log()])
        self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS)), ("done", reason))  # said again, not redone

    def run_until(self, did, want, limit=120):
        """Run the engine one action at a time until dispatch `did` has (launch_mode, status) `want`; no fixed step
        count (a check-in may first wait for its due time, up to a second)."""
        end = time.time() + limit
        while True:
            ws = self.state()
            if did in [d["id"] for d in ws.dispatches()]:
                d = ws.dispatch(did)
                if (d.get("launch_mode"), d.get("status")) == want:
                    return d
            self.assertLess(time.time(), end, "dispatch %s did not reach %s in %d s" % (did, want, limit))
            engine.run(self.ws, MockRunner(MOCKS), max_steps=1)

    def test_a_stop_during_a_check_in_terminates_the_job(self):
        # Review I1: a check-in relaunched the session; the run is stopped while its agent works, or after it failed:
        # the long job is terminated all the same, and the stop says so.
        for variant, status, how in (("working", "running", {"wait": self.flag("never")}),
                                     ("failed", "failed", {"fail": "查不了环境"})):
            with self.subTest(variant):
                self.use_task(task_def(checkin_minutes=0.0005), "ws-ci-" + variant)
                self.behave("train", long_job=True, checkin=how)
                self.run_until("d-0002", ("checkin", status))
                pid = sleeper(self)
                with open(self.outfile("d-0002", "job.marker") + ".pid", "w") as f:
                    f.write("%d\n%d\n" % (pid, int(time.time())))
                reason, d, report = self.stop_and_report()
                self.assertFalse(alive(pid))
                self.assertEqual((d["long_job_stop"]["result"], d["long_job_stop"]["pid"]), ("terminated", pid))
                self.assertIn("长任务已终止", d["closed"])
                self.assertIn("终止了长任务 d-0002", reason)
                self.assertIn("已终止（pid %d）" % pid, report)

    def test_a_decision_that_closes_a_failed_check_in_terminates_the_job(self):
        # R16: a check-in failed and, no returns left, the user answers stop or new_session: closing the failed dispatch
        # terminates its long job first; the stop's reason and REPORT.md list it.
        for option in ("stop", "new_session"):
            with self.subTest(option):
                pid, w = failed_check_in(self, "ws-cd-" + option)
                DP.answer(self.state(), option, "alice", "这次的训练不要了", ref=w["id"])
                if option == "new_session":  # carried out by the next run; the new session's check-in fails as well
                    self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS))[0], "waiting_user")
                    self.assertEqual(self.state().dispatch("d-0003")["module"], "train")
                    self.state().stop("bob", "停")
                d = self.state().dispatch("d-0002")
                self.assertFalse(alive(pid))
                self.assertEqual((d["status"], d["long_job_stop"]["result"], d["long_job_stop"]["pid"]),
                                 ("done", "terminated", pid))
                self.assertIn("长任务已终止", d["closed"])
                status, reason = engine.run_status(self.ws, MockRunner(MOCKS))
                self.assertEqual(status, "done", reason)
                self.assertIn("终止了长任务 d-0002（rsi-train，pid %d）" % pid, reason)
                with open(os.path.join(self.ws, "REPORT.md"), encoding="utf-8") as f:
                    report = f.read()
                self.assertIn("d-0002（rsi-train）：已终止（pid %d）" % pid, report)
                if option == "new_session":  # F15: ended by closing the dispatch, not by the stop
                    self.assertIn("关闭派出时的长任务 d-0002", report)
                    self.assertNotIn("停止时的长任务 d-0002", report)

    def test_close_round_terminates_a_failed_dispatchs_long_job(self):
        # F16: `rsi close-round` closes the round's failed dispatches whose long job may still run, and ends the job.
        pid, _w = failed_check_in(self, "ws-cr")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["close-round", self.ws, "--reason", "这一轮不做了"]), 0)
        d = self.state().dispatch("d-0002")
        self.assertFalse(alive(pid))
        self.assertEqual((d["status"], d["long_job_stop"]["result"], d["long_job_stop"]["context"]),
                         ("done", "terminated", "close"))

    def cut_off_close(self, name):
        """Two failed dispatches with long jobs (d-0002, d-0003) closed by `rsi close-round`, the process cut off right
        after its save: both records done, their long_job_stop pending, both jobs alive. Returns the two pids."""
        pid, _w = failed_check_in(self, name)
        out = os.path.join(self.ws, "work", "round-1", "train-x")
        os.makedirs(out)
        Workspace(self.ws).add_dispatches([{"module": "train", "outdir": "work/round-1/train-x", "purpose": "train",
                                            "status": "failed", "marker": os.path.join(out, "job.marker"),
                                            "launched_at": now_iso()}])
        other = sleeper(self)
        with open(os.path.join(out, "job.marker.pid"), "w") as f:
            f.write("%d\n%d\n" % (other, int(time.time())))
        with mock.patch.object(DP, "terminate_after_commit"):  # the process dies right after the save
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["close-round", self.ws, "--reason", "这一轮不做了"]), 0)
        ws = self.state()
        self.assertEqual([(ws.dispatch(x)["status"], ws.dispatch(x)["long_job_stop"]["result"]) for x in
                          ("d-0002", "d-0003")], [("done", "pending")] * 2)
        self.assertTrue(alive(pid) and alive(other))
        return pid, other

    def test_a_close_cut_off_after_its_save_is_finished_at_the_stop(self):
        # The save that closed the dispatches marks their long jobs pending; cut off before the termination, the stop
        # finishes them whatever the dispatches' status. An error ending one is recorded, and the next is still ended.
        _pid, other = self.cut_off_close("ws-cut")
        ws = self.state()
        real, calls = engine.terminate_long_job, []

        def flaky(ws, d, context="stop"):
            calls.append(d["id"])
            if len(calls) == 1:
                raise OSError("boom")
            return real(ws, d, context)

        ws.stop("bob", "停")
        with mock.patch.object(engine, "terminate_long_job", flaky):
            self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS))[0], "done")
        ws = self.state()
        first, second = ws.dispatch("d-0002")["long_job_stop"], ws.dispatch("d-0003")["long_job_stop"]
        self.assertEqual((first["result"], first["context"], second["result"]), ("error", "close", "terminated"))
        self.assertIn("boom", first["why"])
        self.assertFalse(alive(other))

    def test_a_close_cut_off_after_its_save_is_finished_by_the_next_run(self):
        # The run goes on (no stop): its next step finishes the pending stops before its action, so an old job does not
        # hold its GPUs while the run continues.
        pids = self.cut_off_close("ws-cut2")
        engine.run(self.ws, MockRunner(MOCKS), max_steps=1)
        ws = self.state()
        self.assertFalse(alive(pids[0]) or alive(pids[1]))
        self.assertEqual([(ws.dispatch(x)["long_job_stop"]["result"], ws.dispatch(x)["long_job_stop"]["context"])
                          for x in ("d-0002", "d-0003")], [("terminated", "close")] * 2)
        self.assertIn("长任务已终止", ws.dispatch("d-0002")["closed"])
        self.assertNotIn(self.actions()[-1], ("report", "done"))  # no stop: not the report step's settling

    def test_a_stop_during_a_check_in_relaunch_terminates_the_new_watch(self):
        # F10: a check-in relaunched the watch (watch-2.done) and has not handed back its marker yet: the record names
        # only the old one, whose process has ended. The stop ends the new watch too, never the runner's agent-<n>.pid.
        # An orphan without a start time line is checked against the file's time: a process started well after the
        # file was written has reused its pid and is left alone. A pid file whose marker is older than it belongs to
        # a job relaunched under an earlier job's marker path: an orphan too.
        marker = self.long_job("ws-lj8")
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        watch, agent, reused, again = sleeper(self), sleeper(self), sleeper(self), sleeper(self)
        files = [(marker + ".pid", dead.pid), (self.outfile("d-0002", "old.done.pid"), reused),
                 (self.outfile("d-0002", "prep.done.pid"), again),
                 (self.outfile("d-0002", "watch-2.done.pid"), watch), (self.outfile("d-0002", "agent-1.pid"), agent)]
        for path, pid in files:
            with open(path, "w") as f:
                f.write("%d\n" % pid)
        os.utime(files[1][0], (time.time() - 30, time.time() - 30))
        with open(self.outfile("d-0002", "prep.done"), "w") as f:
            f.write("ok\n")
        os.utime(self.outfile("d-0002", "prep.done"), (time.time() - 600, time.time() - 600))
        _reason, d, _report = self.stop_and_report()
        self.assertFalse(alive(watch) or alive(again))
        self.assertTrue(alive(agent) and alive(reused))
        st = d["long_job_stop"]
        self.assertEqual([(x["pid_file"], x["result"]) for x in [st] + st["others"]],
                         [(files[0][0], "gone"), (files[1][0], "unconfirmed"), (files[2][0], "terminated"),
                          (files[3][0], "terminated")])

    def test_a_seen_marker_is_no_open_job(self):
        # F11: a long job's hand-back was processed, its marker read (ok: the session continues; failed: the dispatch
        # fails): that marker is seen. Deleted afterwards, it is no long job left running at a stop.
        for first in ("ok", "failed"):
            with self.subTest(first):
                marker = self.long_job("ws-lj9-" + first)
                with open(marker, "w") as f:
                    f.write(first + "\n")
                engine.run(self.ws, MockRunner(MOCKS), max_steps=1)  # process_handback
                os.remove(marker)
                _reason, d, report = self.stop_and_report()
                self.assertNotIn("没能终止", report)
                self.assertNotIn("long_job_stop", d)
                self.assertEqual(d["marker_seen"], d["marker"])

    def test_a_seen_marker_handed_back_again_is_open(self):
        # A later launch starts a new long job under the marker path seen before and hands it back: marker_seen goes,
        # so a stop ends that job. When the old marker file is still there (not removed before the relaunch), it is
        # older than the launch that started the new job: no hand-back is processed on it, and a stop ends the job;
        # also when it was written just before that launch (reviewer A's probe: half a second before its stamp).
        for old in ("removed", "kept", "just before"):
            with self.subTest(old):
                marker = self.long_job("ws-lj11-" + old.replace(" ", "-"))
                with open(marker, "w") as f:
                    f.write("ok\n")
                engine.run(self.ws, MockRunner(MOCKS), max_steps=2)  # process_handback, then wait for the continuation
                self.assertEqual(self.state().dispatch("d-0002")["status"], "handed_back")
                if old == "removed":
                    os.remove(marker)
                elif old == "kept":
                    os.utime(marker, (time.time() - 600, time.time() - 600))  # the earlier job ended long ago
                else:
                    launch = self.state().dispatch("d-0002")["launches"]
                    stamp = os.path.getmtime(self.outfile("d-0002", ".launch-%d" % launch))
                    os.utime(marker, (stamp - 0.5, stamp - 0.5))
                pid = sleeper(self)
                with open(marker + ".pid", "w") as f:
                    f.write("%d\n%d\n" % (pid, int(time.time())))
                with open(self.outfile("d-0002", "handback.json"), "w") as f:
                    json.dump({"status": "long_job", "artifacts": [], "marker": marker, "notes": "again",
                               "questions": []}, f)
                engine.Engine(self.ws, MockRunner(MOCKS)).finish("d-0002", agent_result(exit_code=0))  # agent ended
                self.assertEqual(F.next_action(self.state())["type"], "wait_long_job")
                _reason, d, _report = self.stop_and_report()
                self.assertFalse(alive(pid))
                self.assertEqual((d["marker_seen"], d["long_job_stop"]["result"], d.get("continued")),
                                 (None, "terminated", 1))
                if old == "kept":
                    # The launch's stamp in the output directory (same filesystem, same clock as the marker) is the
                    # reference, not the host's clock: a marker written after it counts though the host runs ahead.
                    stamp = os.path.getmtime(self.outfile("d-0002", ".launch-%d" % d["launches"]))
                    os.utime(marker, (stamp + 1, stamp + 1))
                    ahead = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 600))
                    self.assertTrue(F.marker_done(self.state(), dict(d, long_job_since=ahead, launched_at=ahead)))

    def test_stop_during_registration_terminates_the_open_long_job(self):
        # F12: a hand-back processed while its long job's marker is not there yet (a check-in handed back a part), and
        # the run stopped during its registration: closing the dispatch terminates the job.
        marker = self.long_job("ws-lj10")
        pid = sleeper(self)
        with open(marker + ".pid", "w") as f:
            f.write("%d\n" % pid)
        os.makedirs(self.outfile("d-0002", "search-1"))
        with open(self.outfile("d-0002", "search-1/manifest.json"), "w") as f:
            json.dump({"kind": "search", "producer": "train"}, f)
        with open(self.outfile("d-0002", "handback.json"), "w") as f:
            json.dump({"status": "ok", "artifacts": ["search-1"], "marker": None, "notes": "a part",
                       "questions": []}, f)
        Workspace(self.ws).update_dispatch("d-0002", status="handed_back")
        ws_path = self.ws

        def register(ws, kind, src, fill_only=False):
            Workspace(ws_path).stop("bob", "停下")
            raise StoppedError("the run has stopped")

        with mock.patch.object(Workspace, "register", register):
            self.assertEqual(engine.run_status(self.ws, MockRunner(MOCKS))[0], "done")
        self.assertFalse(alive(pid))
        self.assertEqual(self.state().dispatch("d-0002")["long_job_stop"]["result"], "terminated")

    def test_a_long_job_that_cannot_be_confirmed_is_left_and_listed(self):
        pid = sleeper(self)
        marker = self.long_job("ws-lj2")
        with open(marker + ".pid", "w") as f:
            f.write("%d\n" % pid)
        # The process started well before the long job's launch (beyond LONG_JOB_SLACK_S): a reused pid, not the job.
        later = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3 * engine.LONG_JOB_SLACK_S))
        Workspace(self.ws).update_dispatch("d-0002", long_job_since=later)
        reason, d, report = self.stop_and_report()
        self.assertTrue(alive(pid))
        self.assertEqual(d["long_job_stop"]["result"], "unconfirmed")
        self.assertIn("没能终止", reason)
        self.assertIn(str(pid), reason)
        self.assertIn("可能仍在运行", report)
        self.assertIn(str(pid), report)

    def test_ps_may_show_the_start_a_second_early(self):
        # Linux: `ps` reckons a start time from the boot time and the uptime and can show it a second before the real
        # one: a job started in its launch's second is still the job; one started before the slack is not.
        launched = time.time()
        with mock.patch.object(engine, "pid_alive", return_value=True):
            for started, state in ((int(launched) - 1, "ours"),
                                   (int(launched) - engine.LONG_JOB_SLACK_S - 1, "unconfirmed")):
                with mock.patch.object(engine, "process_info", return_value=("sleep 60", float(started))):
                    self.assertEqual(engine.long_job_identity(4242, None, launched)[0], state, started)
            # F18c: a pid file written late (its time after the start `ps` shows) is the job; a process that started
            # after the time the file gives is not (a reused pid).
            started = int(launched) + 30
            with mock.patch.object(engine, "process_info", return_value=("sleep 60", float(started))):
                for written, state in ((started + 10, "ours"), (started - 10, "unconfirmed")):
                    self.assertEqual(engine.long_job_identity(4242, written, launched)[0], state, written)

    def test_a_start_time_that_differs_from_the_pid_file_is_not_the_job(self):
        marker = self.long_job("ws-lj3")
        pid = sleeper(self)
        with open(marker + ".pid", "w") as f:
            f.write("%d\n%d\n" % (pid, int(time.time()) - 3600))
        reason, d, _report = self.stop_and_report()
        self.assertTrue(alive(pid))
        self.assertEqual(d["long_job_stop"]["result"], "unconfirmed")
        self.assertIn("没能终止", reason)

    def test_no_pid_file_and_a_process_that_is_gone(self):
        marker = self.long_job("ws-lj4")
        reason, d, report = self.stop_and_report()
        self.assertEqual(d["long_job_stop"]["result"], "no_pid_file")
        self.assertIn(marker + ".pid", d["long_job_stop"]["why"])
        self.assertIn("没能终止", reason)
        self.assertIn("可能仍在运行", report)
        marker = self.long_job("ws-lj5")
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        with open(marker + ".pid", "w") as f:
            f.write("%d\n" % dead.pid)
        reason, d, _report = self.stop_and_report()
        self.assertEqual(d["long_job_stop"]["result"], "gone")
        self.assertNotIn("没能终止", reason)
        marker = self.long_job("ws-lj6")
        with open(marker + ".pid", "w") as f:
            f.write("not a pid\n")
        reason, d, _report = self.stop_and_report()
        self.assertEqual(d["long_job_stop"]["result"], "bad_pid_file")
        self.assertIn("没能终止", reason)

    def test_a_finished_long_job_is_not_touched(self):
        marker = self.long_job("ws-lj7")
        pid = sleeper(self)
        with open(marker + ".pid", "w") as f:
            f.write("%d\n" % pid)
        with open(marker, "w") as f:
            f.write("ok\n")
        Workspace(self.ws).stop("bob", "不要了")
        status, reason = engine.run_status(self.ws, MockRunner(MOCKS))
        self.assertEqual(status, "done")
        self.assertTrue(alive(pid))  # its marker says it ended: nothing to terminate
        self.assertNotIn("long_job_stop", self.state().dispatch("d-0002"))
        self.assertNotIn("长任务", reason)


class TestManualRunner(EngineCase):
    def test_brief_then_wait_then_fill_in_and_go_on(self):
        reason = self.run_engine(ManualRunner())
        self.assertIn("handback.json", reason)
        ws = self.state()
        d = ws.dispatch("d-0001")
        self.assertEqual((d["status"], d["runner"], d["session"]), ("running", "manual", None))
        out = os.path.join(self.ws, d["outdir"])
        for name in ("brief.md", "brief.json", "prompt-1.md"):
            self.assertTrue(os.path.isfile(os.path.join(out, name)), name)
        self.assertEqual(self.run_engine(ManualRunner()), reason)  # nothing new until the hand-back is there
        self.assertEqual(len(self.state().dispatches()), 1)
        for kind, ids in (("testset", TEST_IDS), ("evalset", EVAL_IDS), ("valset", VAL_IDS), ("trainset", TRAIN_IDS)):
            fixtures.make_artifact(kind, os.path.join(out, kind), self.task, ids=ids)
        with open(os.path.join(out, "handback.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "artifacts": ["testset", "evalset", "valset", "trainset"], "marker": None,
                       "notes": "by hand", "questions": []}, f)
        reason = self.run_engine(ManualRunner())
        ws = self.state()
        self.assertEqual(ws.dispatch("d-0001")["status"], "done")
        self.assertEqual(ws.current("trainset"), "trainset-0001")
        self.assertEqual(ws.dispatch("d-0002")["module"], "train")
        self.assertIn(ws.dispatch("d-0002")["outdir"], reason)


class TestInterrupted(EngineCase):
    def test_interrupted_registration_is_not_repeated(self):
        orig, seen = Workspace.register, []

        def flaky(ws, kind, src):
            seen.append(kind)
            if len(seen) == 3:
                raise KeyboardInterrupt
            return orig(ws, kind, src)

        with mock.patch.object(Workspace, "register", flaky):
            with self.assertRaises(KeyboardInterrupt):
                self.run_engine()
        self.assertEqual([e["kind"] for e in self.state().entries()], ["task", "testset", "evalset"])
        self.assertEqual(self.state().dispatch("d-0001")["status"], "handed_back")
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        self.assertEqual(len(ws.dispatches(module="data")), 1)
        self.assertEqual(len(self.calls("d-0001")), 1)
        self.assertEqual([r["kind"] for r in ws.dispatch("d-0001")["registered"]],
                         ["testset", "evalset", "valset", "trainset"])

    def test_interrupted_between_the_record_and_the_launch(self):
        # The record is written, the brief is not: nothing was started, so the restart launches that record.
        from rsi import briefs
        orig, seen = briefs.build_brief, []

        def flaky(*a, **kw):
            seen.append(1)
            if len(seen) == 1:
                raise KeyboardInterrupt
            return orig(*a, **kw)

        with mock.patch.object(briefs, "build_brief", flaky):
            with self.assertRaises(KeyboardInterrupt):
                self.run_engine()
        d = self.state().dispatch("d-0001")
        self.assertEqual((d["status"], d["launched_at"]), ("pending", None))
        self.assertFalse(os.path.exists(os.path.join(self.ws, d["outdir"], "brief.md")))
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        self.assertEqual(len(ws.dispatches(module="data")), 1)
        self.assertEqual(ws.dispatch("d-0001")["returns"], 0)
        self.assertEqual(len(self.calls("d-0001")), 1)
        self.assertIsNone(self.calls("d-0001")[0]["feedback"])

    def interrupt_first_launch(self, started):
        """Cut the first launch off inside the runner; with `started` the agent's log was already written (the
        agent may have started)."""
        orig, seen = MockRunner.launch, []

        def flaky(runner, ws, d, prompt, resume_session=None, new_session=None):
            seen.append(d["id"])
            if len(seen) == 1:
                if started:
                    open(os.path.join(ws.root, d["outdir"], "agent-1.log"), "w").close()
                raise KeyboardInterrupt
            return orig(runner, ws, d, prompt, resume_session, new_session)

        with mock.patch.object(MockRunner, "launch", flaky):
            with self.assertRaises(KeyboardInterrupt):
                self.run_engine()
        d = self.state().dispatch("d-0001")
        self.assertIsNotNone(d["launched_at"])
        self.assertTrue(d["session"].startswith("mock-"))
        return d["session"]

    def test_interrupted_before_the_agent_started(self):
        # Marked launched, but the runner finds no trace of a start (no log): it is launched again, same launch,
        # no return counted.
        self.interrupt_first_launch(started=False)
        self.assertMilestoneAsk(self.run_engine())
        d = self.state().dispatch("d-0001")
        self.assertEqual((d["returns"], d["launches"]), (0, 1))
        calls = self.calls("d-0001")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["feedback_file"])
        self.assertIn("relaunch", [r["event"] for r in self.log()])

    def test_interrupted_while_launching(self):
        # The agent may have started (its log exists), so the record is never launched twice; with no hand-back it
        # is failed and goes back, in the same record, to the session recorded before the launch.
        session = self.interrupt_first_launch(started=True)
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        self.assertEqual(len(ws.dispatches(module="data")), 1)
        self.assertEqual(ws.dispatch("d-0001")["returns"], 1)
        calls = self.calls("d-0001")
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["session"], calls[0]["feedback_file"]), (session, "feedback-2.md"))

    def cli(self, *args):
        env = dict(os.environ, RSI_NO_BROWSER="1")
        return subprocess.run([sys.executable, "-m", "rsi", "run", self.ws, "--runner", "mock", "--mock-dir", MOCKS]
                              + list(args), cwd=REPO, capture_output=True, text=True, env=env)

    def test_engine_killed_after_the_agent_handed_back(self):
        self.behave("data", kill_parent={"flag": self.flag("kill"), "when": "after_handback"})
        r = self.cli()
        self.assertEqual(r.returncode, -signal.SIGKILL, r.stdout + r.stderr)
        d = self.state().dispatch("d-0001")
        self.assertEqual(d["status"], "running")
        r = self.cli()
        self.assertEqual(r.returncode, 10, r.stderr)
        self.assertMilestoneAsk(r.stdout.strip())
        ws = self.state()
        self.assertEqual(len(ws.dispatches(module="data")), 1)
        self.assertEqual(ws.dispatch("d-0001")["returns"], 0)
        self.assertEqual(len(self.calls("d-0001")), 1)
        self.assertIn("recover", [x["event"] for x in self.log()])

    def test_engine_killed_before_the_agent_handed_back(self):
        self.behave("data", kill_parent={"flag": self.flag("kill"), "when": "before_handback"})
        self.assertEqual(self.cli().returncode, -signal.SIGKILL)
        session = self.state().dispatch("d-0001")["session"]  # recorded before the launch
        r = self.cli()
        self.assertEqual(r.returncode, 10, r.stderr)
        self.assertMilestoneAsk(r.stdout.strip())
        ws = self.state()
        d = ws.dispatch("d-0001")
        self.assertEqual(len(ws.dispatches(module="data")), 1)
        self.assertEqual(d["returns"], 1)
        self.assertEqual(ws.decisions()[0]["point"], "handback_failed")
        calls = self.calls("d-0001")
        self.assertEqual(len(calls), 2)
        self.assertEqual([c["session"] for c in calls], [session, session])  # back to the same session
        self.assertEqual(calls[1]["feedback_file"], "feedback-2.md")
        self.assertIn("进程已不在", calls[1]["feedback"])

    def test_decision_interrupted_before_it_was_carried_out(self):
        # Cut off after the decision was recorded and the hand-back kept, before the return was written: the rerun
        # reuses the decision and the feedback still carries the hand-back's notes.
        self.behave("data", trainset="quality_fail")
        orig, seen = Workspace.update_dispatch, []

        def flaky(ws, did, **fields):
            if fields.get("launch_mode") == "resume" and not seen:
                seen.append(did)
                raise KeyboardInterrupt
            return orig(ws, did, **fields)

        with mock.patch.object(Workspace, "update_dispatch", flaky):
            with self.assertRaises(KeyboardInterrupt):
                self.run_engine()
        self.assertEqual(len(self.state().decisions()), 1)
        self.assertTrue(os.path.isfile(self.outfile("d-0001", "handback-1.json")))
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        self.assertEqual(len(ws.decisions()), 1)
        self.assertEqual(ws.dispatch("d-0001")["returns"], 1)
        self.assertIn("decision_reused", [r["event"] for r in self.log()])
        feedback = self.calls("d-0001")[1]["feedback"]
        self.assertIn("上次交回的 notes", feedback)
        self.assertIn("mock", feedback)

    def test_running_dispatch_still_alive_is_waited_for(self):
        # A dispatch whose agent (here a sleeping process) is still alive after a restart is waited for,
        # never dispatched again; once it is gone without a hand-back it is failed and sent back.
        # The sleeper is a grandchild, so it is not this process's zombie once it ends.
        # Its program name is "claude", as the record's runner expects of its agent.
        spawn = ("import subprocess, sys; print(subprocess.Popen(['claude', '1.5'], executable='/bin/sleep', "
                 "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid)")
        pid = int(subprocess.run([sys.executable, "-c", spawn], capture_output=True, text=True).stdout)
        ws = self.state()
        ws.add_dispatches([{"module": "data", "purpose": "sets", "outdir": "work/round-1/data", "runner": "claude",
                            "pid": pid, "launched_at": now_iso(), "launches": 1}])
        os.makedirs(os.path.join(self.ws, "work", "round-1", "data"))
        self.assertMilestoneAsk(self.run_engine(poll_s=0.05))
        ws = self.state()
        self.assertFalse(alive(pid))
        self.assertEqual(len(ws.dispatches(module="data")), 1)
        self.assertEqual(ws.dispatch("d-0001")["returns"], 1)
        feedback = self.calls("d-0001")[0]["feedback"]
        self.assertIn("handback.json", feedback)  # it ended while watched without a hand-back
        self.assertNotIn("进程已不在", feedback)  # so it was not lost


class TestClaudeCodeRunner(EngineCase):
    """The engine with ClaudeCodeRunner against the strict fake `claude` (never the real one)."""

    def setUp(self):
        super(TestClaudeCodeRunner, self).setUp()
        import test_runners
        self.fake, self.record = test_runners.install_fake_claude(self, self.root)
        self.claude_calls = lambda: test_runners.claude_calls(self.record)

    def runner(self):
        return ClaudeCodeRunner.from_task(self.task)

    def test_the_real_prompt_goes_through_on_stdin(self):
        status, _reason = engine.run_status(self.ws, self.runner(), max_steps=2, poll_s=0.02)
        self.assertEqual(status, "max_steps")
        d = self.state().dispatch("d-0001")
        calls = self.claude_calls()
        self.assertEqual(len(calls), 1)
        argv, prompt = calls[0]["argv"], calls[0]["prompt"]
        # The session id was made up and recorded before the launch, then passed with --session-id.
        self.assertEqual(argv, ["-p", "--dangerously-skip-permissions", "--output-format", "json",
                                "--model", "claude-opus-5-5", "--effort", "xhigh", "--session-id",
                                d["session"]])
        self.assertEqual(d["exit_code"], 0)  # the fake took every argument and the prompt
        skill = os.path.join(REPO, "modules", "data", "SKILL.md")  # D8: each module's folder holds its skill
        self.assertTrue(prompt.startswith("你是 RSI 运行里的模块 agent `rsi-data`"))
        self.assertIn(skill, prompt)
        with open(skill, encoding="utf-8") as f:
            text = f.read()
        self.assertTrue(text.startswith("---"))  # front matter: as an argument it would read as an option
        self.assertIn(text.rstrip("\n"), prompt)
        self.assertIn("你负责 RSI 运行里的一个模块", prompt)  # the brief
        self.assertIn("handback.json", prompt)
        with open(self.outfile("d-0001", "prompt-1.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), prompt)
        self.assertEqual(d["status"], "failed")  # the fake writes no hand-back
        self.assertNotIn("unknown option", " ".join(d["errors"]))
        # The read-deny list is recorded and logged, not briefed.
        self.assertIsInstance(d["denied"], list)  # a first-round data module: no dataset is registered yet
        launch = [r for r in self.log() if r["event"] == "launch"][0]
        self.assertEqual(launch["denied"], d["denied"])
        with open(self.outfile("d-0001", "brief.json"), encoding="utf-8") as f:
            self.assertNotIn("denied", f.read())

    def test_a_launch_that_cannot_start_is_a_failed_dispatch(self):
        with mock.patch("rsi.runners.claude.subprocess.Popen", side_effect=OSError(7, "Argument list too long")):
            status, _reason = engine.run_status(self.ws, self.runner(), max_steps=2, poll_s=0.02)
        self.assertEqual(status, "max_steps")
        d = self.state().dispatch("d-0001")
        self.assertEqual(d["status"], "failed")
        self.assertIn("Argument list too long", " ".join(d["errors"]))
        with open(self.outfile("d-0001", "agent-1.log")) as f:  # the runner logs why it could not start
            self.assertIn("Argument list too long", f.read())

    def test_session_reported_differently_is_kept_with_a_warning(self):
        with mock.patch.dict(os.environ, {"FAKE_CLAUDE_SESSION": "other-session"}):
            engine.run_status(self.ws, self.runner(), max_steps=2, poll_s=0.02)
        self.assertEqual(self.state().dispatch("d-0001")["session"], "other-session")
        self.assertIn("session_mismatch", [r["event"] for r in self.log()])


HOLD_LOCK = """
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
os.write(fd, b"%d\\n" % os.getpid())
open(sys.argv[2], "w").close()
time.sleep(30)
"""


class TestLock(EngineCase):
    def lock(self):
        path = os.path.join(self.ws, "work", "run.lock")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def test_a_second_run_is_refused_while_one_runs(self):
        # A live `rsi run` holds run.lock (here a process holding its flock): a second run is refused, nothing is
        # dispatched. Once it is gone the lock is free (the kernel dropped it), and removed when the run stops.
        lock, ready = self.lock(), os.path.join(self.root, "held")
        holder = subprocess.Popen([sys.executable, "-c", HOLD_LOCK, lock, ready])
        self.addCleanup(holder.wait)
        self.addCleanup(holder.kill)
        end = time.time() + 20
        while not os.path.exists(ready) and time.time() < end:
            time.sleep(0.02)
        with self.assertRaises(RsiError) as e:
            self.run_engine()
        self.assertIn("another `rsi run` (pid %d)" % holder.pid, str(e.exception))
        self.assertEqual(self.state().dispatches(), [])
        holder.kill()
        holder.wait()
        self.assertMilestoneAsk(self.run_engine())
        self.assertFalse(os.path.exists(lock))

    def test_the_pages_say_running_only_while_the_run_holds_the_lock(self):
        # P9 (sim2): the pages written during the run say "● 运行中" (the lock names this live process); the run
        # writes them once more after it let go of the lock, so a run that has returned is not shown as running.
        from rsi import dashboard as D
        seen = []
        orig = D.write_dashboard

        def write(ws):
            seen.append(D.run_active(ws))
            return orig(ws)

        with mock.patch.object(D, "write_dashboard", write):
            status, _reason = engine.run_status(self.ws, MockRunner(MOCKS), max_steps=1)
        self.assertEqual(status, "max_steps")
        self.assertEqual(seen, [True, False])  # the step's pages, then the pages after the run
        with open(os.path.join(self.ws, "dashboard.html"), encoding="utf-8") as f:
            page = f.read()
        self.assertNotIn("● 运行中</span>", page)
        self.assertIn('<span class="chip idle">等待 rsi run</span>', page)
        self.assertFalse(os.path.exists(self.lock()))

    def test_without_fcntl_a_live_pid_refuses_and_a_dead_one_is_moved_aside(self):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(sleeper.wait)
        self.addCleanup(sleeper.kill)
        lock = self.lock()
        with open(lock, "w") as f:
            f.write("%d\n" % sleeper.pid)
        with mock.patch.object(engine, "fcntl", None):
            with self.assertRaises(RsiError):
                self.run_engine()
            sleeper.kill()
            sleeper.wait()
            self.assertMilestoneAsk(self.run_engine())
        self.assertFalse(os.path.exists(lock))
        self.assertEqual([n for n in os.listdir(os.path.dirname(lock)) if n.startswith("run.lock")], [])


class TestCli(EngineCase):
    def run_rsi(self, *args):
        return subprocess.run([sys.executable, "-m", "rsi", "run", self.ws] + list(args), cwd=REPO,
                              capture_output=True, text=True, env=dict(os.environ, RSI_NO_BROWSER="1"))

    def test_rsi_run_prints_the_reason(self):
        r = self.run_rsi("--runner", "manual", "--max-steps", "1")
        self.assertEqual(r.returncode, 13, r.stderr)
        self.assertIn("--max-steps", r.stdout)
        r = self.run_rsi("--runner", "manual")
        self.assertEqual(r.returncode, 12, r.stderr)  # waiting for a manual hand-back

    def test_json_stdout_is_unchanged_with_status_lines(self):
        # Review focus 5: `rsi run` prints status lines on stderr; with --json its stdout is still one JSON object.
        r = self.run_rsi("--runner", "mock", "--mock-dir", MOCKS, "--json")
        self.assertEqual(r.returncode, 10, r.stderr)  # waiting for the user
        out = json.loads(r.stdout)
        self.assertEqual(r.stdout.strip().count("\n"), 0, r.stdout)
        self.assertEqual(out["status"], "waiting_user")
        self.assertIn("里程碑", out["reason"])
        self.assertRegex(r.stderr, r"(?m)^\d\d:\d\d 第1轮 ")
        r = self.run_rsi("--runner", "mock", "--mock-dir", MOCKS)  # without --json: the reason, one line
        self.assertEqual(r.returncode, 10, r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)
        self.assertMilestoneAsk(r.stdout.strip())

    def test_refusals_exit_1(self):
        r = self.run_rsi("--runner", "mock")  # the mock runner needs its directory
        self.assertEqual(r.returncode, 1)
        self.assertIn("--mock-dir", r.stderr)
        self.assertEqual(self.state().dispatches(), [])
        r = self.run_rsi("--runner", "mock", "--json")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(json.loads(r.stdout)["status"], "error")
        shutil.rmtree(os.path.join(self.root, "nowhere"), ignore_errors=True)
        r = subprocess.run([sys.executable, "-m", "rsi", "run", os.path.join(self.root, "nowhere"), "--runner",
                            "manual"], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("not an rsi workspace", r.stderr)


def budget_task(budget=0.25, max_rounds=3):
    """A task with budget.compute_hours `budget` (over_budget in its approvals, as binary_task has it); max_rounds 3,
    so only the budget stops the run after round 1."""
    t = task_def(max_rounds=max_rounds)
    t["budget"]["compute_hours"] = budget
    return t


class TestBudgetHold(EngineCase):
    """B1 (third simulation): a hand-back whose candidates would take the compute used over budget.compute_hours is
    held unregistered and the run asks over_budget at once (no handback_refused loop); approved, the same hand-back
    is registered without retraining, the round goes on to its decision and the budget stops the run."""

    QUESTION = ("over_budget：rsi-train（%s）这次交回的候选（%s）要 0.35 小时算力，加上已用 0 小时，超过预算 0.25 小时。"
                "approve：批准超预算，登记这次交回的产物，本轮接着做到判定为止，判定后因算力超过预算停止运行；"
                "stop：现在停止运行（同 rsi stop），这次交回的候选不登记。")

    def setUp(self):
        super(TestBudgetHold, self).setUp()
        self.use_task(budget_task(), "ws-b")
        self.behave("train", cost=0.35, record=True)

    def events(self, name):
        return [r for r in self.log() if r["event"] == name]

    def go(self):
        return engine.run_status(self.ws, MockRunner(MOCKS))

    def held(self):
        """Run until the run stops for the user; return STATE waiting."""
        status, reason = self.go()
        self.assertEqual(engine.STATUS_EXIT_CODES[status], 10, reason)  # `rsi run` exits 10
        w = self.state().waiting()
        self.assertIn(" ".join(w["question"].split()), reason)
        return w

    def kinds(self, *kinds):
        return [e["kind"] for e in self.state().entries() if e["kind"] in kinds]

    def test_a_candidate_over_budget_is_held_and_asked(self):
        w = self.held()
        ws = self.state()
        d = ws.dispatch("d-0002")
        self.assertEqual((d["module"], d["status"], d["returns"], d["launches"]), ("train", "handed_back", 0, 1))
        held = d["held_budget"]
        self.assertEqual((held["launch"], held["question_id"], held["dirs"]), (1, w["id"], ["candidate"]))
        self.assertEqual((held["cost"], held["used"], held["budget"]), (0.35, 0, 0.25))
        self.assertEqual(self.kinds("candidate", "record"), [])  # nothing of the hand-back, the record neither
        self.assertEqual(d.get("registered") or [], [])
        self.assertEqual(d.get("refused") or [], [])
        self.assertEqual((w["kind"], w["what"], w["ref"], w["dispatch_id"], w["options"]),
                         ("approval", "over_budget", None, "d-0002", ["approve", "stop"]))
        self.assertEqual(w["question"], self.QUESTION % ("d-0002", "candidate"))  # the text the user sees
        self.assertNotIn(F.HANDBACK_REFUSED, [x["point"] for x in ws.decisions()])
        self.assertEqual(ws.state["budget_used"]["compute_hours"], 0)  # held, not refused: nothing counted
        self.assertEqual([dict((k, r[k]) for k in ("dispatch_id", "launch", "question_id", "dirs", "cost", "used",
                                                    "budget")) for r in self.events("budget_held")],
                         [dict(held, dispatch_id="d-0002")])
        self.assertTrue(F.budget_held(ws, d))
        self.assertEqual(F.next_action(ws)["type"], "ask")

    def finish_after_approval(self):
        """The approval given: the next run registers the held hand-back (no new launch), the round goes on through
        delivery, evaluation and its decision, and the budget stops the run (the milestone question first: V1)."""
        status, reason = self.go()
        ws = self.state()
        d = ws.dispatch("d-0002")
        self.assertEqual((d["launches"], d["returns"], d["followups"], d["status"]), (2, 0, 1, "done"))
        self.assertEqual(len(self.calls("d-0002")), 2)  # not trained again: the second launch writes the search
        self.assertEqual([(x["dir"], x["kind"], x["launch"]) for x in d["registered"]],
                         [("candidate", "candidate", 1), ("record", "record", 1), ("search-2", "search", 2)])
        self.assertEqual(ws.state["budget_used"]["compute_hours"], 0.35)
        self.assertEqual([x["module"] for x in ws.dispatches()], ["data", "train", "deliver", "judge"])
        self.assertEqual(ws.state["rounds"]["1"]["decision"]["decision"], "promote")
        self.assertFalse(F.budget_held(ws, d))
        self.assertEqual((status, ws.waiting()["what"]), ("waiting_user", F.MILESTONE), reason)
        DP.answer(ws, "no", "alice", "不做里程碑")
        status, reason = self.go()
        self.assertEqual(status, "done", reason)
        self.assertIn("compute hours used 0.35 reached budget 0.25", self.state().state["stopped"]["reason"])
        self.assertEqual([(r["dispatch_id"], r["launch"], r["by"]) for r in self.events("budget_released")],
                         [("d-0002", 1, "approve")])
        self.assertEqual(len(self.events("budget_held")), 1)
        self.assertNoDoubleRegistration()

    def test_approving_registers_the_held_handback_without_retraining(self):
        # Approved by `rsi answer approve` or by `rsi approve <ws> over_budget` (no --ref): the same release.
        for how in ("answer", "approve"):
            with self.subTest(how):
                self.use_task(budget_task(), "ws-b-" + how)
                w = self.held()
                if how == "answer":
                    rec = DP.answer(self.state(), "approve", "alice", "可以超预算，登记这次的候选", ref=w["id"])
                    self.assertEqual((rec["point"], rec["choice"], rec["dispatch_id"]),
                                     ("over_budget", "approve", "d-0002"))
                    ws = self.state()
                    self.assertIsNone(ws.waiting())
                    self.assertEqual([(a["what"], a["ref"], a["quote"]) for a in ws.state["approvals"]],
                                     [("over_budget", None, "可以超预算，登记这次的候选")])
                else:
                    self.state().approve("over_budget", "alice", "可以超")
                    ws = self.state()
                    self.assertTrue(ws.waiting() is None or F.stale_waiting(ws, ws.waiting()))
                    self.assertEqual(F.next_action(ws),
                                     {"type": "process_handback", "dispatch_id": "d-0002", "step": "handed"})
                self.finish_after_approval()

    def test_stop_instead_of_approving_registers_no_candidate_and_counts_its_compute(self):
        # R7: the held candidates were trained; a stop ends the hold without registering them, so their compute
        # counts as a refused candidate's (budget_used.refused, once per source), right with the stop.
        w = self.held()
        DP.answer(self.state(), "stop", "alice", "不超预算，先停", ref=w["id"])
        src = os.path.realpath(self.outfile("d-0002", "candidate"))
        ws = self.state()
        self.assertEqual(ws.state["budget_used"], {"compute_hours": 0.35, "refused": {src: 0.35}})
        self.assertEqual(ws.refused_compute_hours(), 0.35)
        self.assertFalse(F.budget_held(ws, ws.dispatch("d-0002")))  # the run stopped: no approval awaited
        status, reason = self.go()
        self.assertEqual(status, "done", reason)
        ws = self.state()
        self.assertEqual(ws.state["budget_used"]["refused"], {src: 0.35})
        self.assertEqual(ws.state["stopped"]["reason"], "user requested: 不超预算，先停")
        self.assertEqual(self.kinds("candidate", "record"), ["record"])  # the record, as after any stop
        d = ws.dispatch("d-0002")
        self.assertEqual(d["status"], "done")
        self.assertIn("运行已停止", d["closed"])
        self.assertEqual(self.events("budget_released"), [])
        with open(os.path.join(self.ws, F.REPORT_FILE), encoding="utf-8") as f:
            self.assertIn("0.35", f.read())
        self.assertEqual(self.go()[0], "done")  # once more: nothing counted twice
        self.assertEqual(self.state().state["budget_used"], ws.state["budget_used"])

    def test_a_handback_found_after_the_stop_counts_its_compute(self):
        # F1 (fourth simulation): train (d-0002, still running in STATE) hands back its trials only after the run was
        # stopped; the stop cannot see that hand-back, the report run finds it and counts its compute, once.
        # "lagging": the report run's poll still says train runs (its process has not exited yet), so the run
        # terminates it; the hand-back it wrote counts all the same.
        for variant in ("found", "lagging"):
            with self.subTest(variant):
                self.use_task(budget_task(budget=None), "ws-b-found-" + variant)
                self.behave("train", cost=0.075, trials=2, release=self.flag("train-go-" + variant))
                self.assertEqual(self.run_engine(max_steps=4).count("--max-steps"), 1)  # train launched, waiting
                self.state().stop("alice", "先停")
                open(self.flag("train-go-" + variant), "w").close()
                deadline = time.time() + 30
                while not os.path.exists(self.outfile("d-0002", "handback.json")) and time.time() < deadline:
                    time.sleep(0.05)
                if variant == "lagging":
                    with mock.patch.object(MockRunner, "poll", lambda runner, h: None):
                        status, reason = self.go()
                else:
                    status, reason = self.go()
                self.assertEqual(status, "done", reason)
                trials = dict((os.path.realpath(self.outfile("d-0002", t)), 0.075)
                              for t in ("candidate", "candidate-t2"))
                ws = self.state()
                self.assertEqual(ws.state["budget_used"]["refused"], trials)
                self.assertAlmostEqual(ws.state["budget_used"]["compute_hours"], 0.15)
                self.assertEqual(self.kinds("candidate"), [])
                self.assertEqual(ws.dispatch("d-0002")["status"], "done")
                self.assertEqual(self.go()[0], "done")  # once more: nothing counted twice
                self.assertEqual(self.state().state["budget_used"], ws.state["budget_used"])

    def test_a_stop_after_the_approval_counts_the_held_compute_too(self):
        # Approved, but `rsi stop` came before the next rsi run registered the hand-back: the hold ends unregistered.
        self.held()
        self.state().approve("over_budget", "alice", "可以超")
        self.state().stop("alice", "还是停吧")
        src = os.path.realpath(self.outfile("d-0002", "candidate"))
        self.assertEqual(self.state().state["budget_used"], {"compute_hours": 0.35, "refused": {src: 0.35}})
        self.assertEqual(self.go()[0], "done")
        self.assertEqual(self.kinds("candidate"), [])
        self.assertEqual(self.state().state["budget_used"]["compute_hours"], 0.35)

    def test_questions_come_before_the_budget_hold(self):
        # Review focus 2: a hand-back with questions over the budget: the module's question first; its answer goes
        # back to the session, and the hand-back after it is held for the budget.
        self.behave("train", questions=["要不要换更大的模型？"])
        w = self.held()
        self.assertEqual((w["kind"], w["what"], w["ref"]), ("question", "handback", "d-0002"))
        self.assertNotIn("held_budget", self.state().dispatch("d-0002"))
        self.assertEqual(self.events("budget_held"), [])
        DP.answer(self.state(), "已回答", "alice", "不换", ref=w["id"])
        w = self.held()
        self.assertEqual((w["kind"], w["what"], w["ref"], w["dispatch_id"]),
                         ("approval", "over_budget", None, "d-0002"))
        d = self.state().dispatch("d-0002")
        self.assertEqual((d["launches"], d["answered"], d["returns"], d["held_budget"]["launch"]), (2, 1, 0, 2))
        self.assertEqual(self.kinds("candidate", "record"), [])
        self.assertTrue(F.budget_held(self.state(), d))

    def test_a_rerun_keeps_the_same_budget_question(self):
        # Review focus 3: the run cut off right after the hold was recorded, then run again (twice): the same
        # question, no second budget_held, nothing dispatched or launched again.
        orig = engine.Engine.hold_budget

        def cut(eng, *a, **kw):
            orig(eng, *a, **kw)
            raise KeyboardInterrupt("killed")

        with mock.patch.object(engine.Engine, "hold_budget", cut):
            with self.assertRaises(KeyboardInterrupt):
                self.run_engine()
        w = self.state().waiting()
        self.assertEqual((w["what"], w["dispatch_id"]), ("over_budget", "d-0002"))
        for _ in range(2):
            self.assertEqual(self.held()["id"], w["id"])
        ws = self.state()
        self.assertEqual(ws.state["waiting_seq"], 1)
        self.assertEqual((len(ws.dispatches()), ws.dispatch("d-0002")["launches"], len(self.calls("d-0002"))),
                         (2, 1, 1))
        self.assertEqual(len(self.events("budget_held")), 1)
        self.assertEqual(ws.dispatch("d-0002")["held_budget"]["question_id"], w["id"])

    def test_no_hold_without_a_budget_approval_item(self):
        # Review focus 4: no budget, or no over_budget among the approvals: registered as before, no question.
        for name, budget, approvals in (("ws-nb", None, None), ("ws-na", 0.25, ["task_change", "use_testset"])):
            with self.subTest(name):
                t = budget_task(budget=budget, max_rounds=1)
                if approvals is not None:
                    t["approvals"] = approvals
                self.use_task(t, name)
                status, reason = self.go()
                ws = self.state()
                self.assertEqual(self.kinds("candidate", "record"), ["candidate", "record"])
                self.assertNotIn("held_budget", ws.dispatch("d-0002"))
                self.assertEqual(self.events("budget_held"), [])
                self.assertEqual(ws.state["budget_used"]["compute_hours"], 0.35)
                self.assertEqual((status, ws.waiting()["what"]), ("waiting_user", F.MILESTONE), reason)

    def test_old_state_without_refused_costs(self):
        # Review focus 5: a workspace from before budget_used.refused (and before held_budget) works as before.
        self.held()
        path = os.path.join(self.ws, "STATE.json")
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        s["budget_used"] = {"compute_hours": 0}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s, f)
        ws = self.state()
        self.assertEqual(ws.refused_compute_hours(), 0)
        self.assertFalse(ws.stop_check()[0])
        self.assertFalse(ws.stop_check_if_closed()[0])
        r = subprocess.run([sys.executable, "-m", "rsi", "status", self.ws], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("compute hours: 0 / 0.25", r.stdout)
        ws.approve("over_budget", "alice", "可以超")
        self.go()
        ws = self.state()
        self.assertEqual(self.kinds("candidate"), ["candidate"])
        self.assertEqual(ws.state["budget_used"]["compute_hours"], 0.35)
        self.assertTrue(ws.stop_check()[0])


if __name__ == "__main__":
    unittest.main()
