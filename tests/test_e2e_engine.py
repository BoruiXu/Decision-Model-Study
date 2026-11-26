"""End to end: `rsi run --runner mock --mock-dir examples/mock_modules`, driven from the command line the way an
operator works (`rsi init`, `rsi run`, `rsi answer` for a scripted user), with the example mock modules: they read
their brief.json, run the train command examples/mock_modules/bin/train, compute their numbers with `rsi score`
and hand back handback.json (see examples/mock_modules/mocklib.py for the knobs a scenario sets).

TestLongScenario (the long scenario: four rounds, about 20 seconds) goes through every turn of the main flow; each
round dispatches the train module once (rsi-train: one search, handed back in two parts, the search as a
followup of the same session): quality gate failed -> the same data session -> passed; no trial of the search
through the train gate -> back to the same train session for a new search -> passed; a registration refused -> the
same deliver session -> registered; V1; the plan revised by the user, then approved; a hand-back question (nothing
registered until the answer, which goes back to the same data session); a long job; V2; the evalset used up ->
refresh approved -> a new evalset and the best version re-delivered; a round with no candidate through the train
gate in two searches closed; patience reached -> milestone yes -> milestone delivery and evaluation -> stopped ->
REPORT.md. Every candidate's config_request lies in the task's search_space and it was trained by the task's train
command (registration checks both). At each stop it checks that the overview page's
"What the run waits for" is what next_action says, the exit code of `rsi run`, and at the end the run log, the sessions
that got each return, and that nothing was dispatched or registered twice.

TestCheckins: a long job with check-ins (flow.checkin_minutes): the run waits for it and checks its session in, in
the same session, until the job ends.

TestInterruptedRun kills `rsi run` mid-step three times (during an agent, right after a hand-back, and together
with its agent as a machine restart would) and runs it again each time: the run completes with nothing dispatched
or registered twice.

RSI_E2E_SNAPSHOTS=<dir> copies the pages at the long scenario's stops into <dir>/<stop>/ (for screenshots).
"""
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import fixtures
from rsi import dashboard as D
from rsi import dashboard_pages as P
from rsi import engine
from rsi import flow as F
from rsi.runners.base import pid_alive
from rsi.workspace import Workspace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCKS = os.path.join(REPO, "examples", "mock_modules")
TRAIN_COMMAND = os.path.join(MOCKS, "bin", "train")
# The exit codes of `rsi run` as docs/interfaces/engine.md documents them.
EXIT_CODES = {"done": 0, "waiting_user": 10, "waiting_long_job": 11, "waiting_manual": 12, "max_steps": 13}
USER = "Zoë Test"


def task_def(**stop):
    t = fixtures.binary_task()
    t["stop"] = dict({"target": None, "max_rounds": 6, "patience": 2, "max_attempts_per_round": 2}, **stop)
    t["eval_use_limit"] = 2
    t["resources"] = {"train_command": TRAIN_COMMAND}
    # What the mock train command's config_request names (examples/mock_modules/bin/train).
    t["search_space"] = {"base": ["mock-spam"], "recipe": ["default"], "seed": list(range(101)),
                         "overrides.onejev.lr": {"min": 0.001, "max": 1000},
                         "overrides.onejev.capacity": {"min": 0.5, "max": 1.0}}
    t["quality_checks"] = {"min_rows": 100, "max_near_dup_rate": 0.02, "min_label_agreement": 0.9}
    t["metrics"]["ci"] = {"resamples": 200, "seed": 0}
    return t


def wait_until(cond, timeout=60.0, what="condition"):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise AssertionError("timed out waiting for %s" % what)
        time.sleep(0.05)


class E2ECase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.ws_dir = os.path.join(self.root, "ws")
        self.scenario = os.path.join(self.root, "scenario.json")
        self.env = dict(os.environ, PYTHONPATH=REPO, RSI_NO_BROWSER="1", PYTHONDONTWRITEBYTECODE="1",
                        RSI_MOCK_SCENARIO=self.scenario)
        self.procs = []
        self.addCleanup(self._reap)

    def _reap(self):
        """Kill what is left of the background runs' process groups (their agents included)."""
        for p, out in self.procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
            p.wait()
            out.close()

    def flag(self, name):
        return os.path.join(self.root, name)

    def init(self, task, scenario):
        with open(self.scenario, "w", encoding="utf-8") as f:
            json.dump(scenario, f, ensure_ascii=False)
        fixtures.make_task_dir(os.path.join(self.root, "task"), task)
        p = self.rsi("init", self.ws_dir, "--task", os.path.join(self.root, "task"), "--no-open")
        self.assertEqual(p.returncode, 0, p.stderr)

    # ---- the command line ------------------------------------------------------------------------

    def rsi(self, *args):
        return subprocess.run([sys.executable, "-m", "rsi"] + list(args), cwd=self.root, env=self.env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=600)

    def run_argv(self, max_steps=None):
        argv = [sys.executable, "-m", "rsi", "run", self.ws_dir, "--runner", "mock", "--mock-dir", MOCKS, "--json"]
        return argv + (["--max-steps", str(max_steps)] if max_steps else [])

    def start_run(self):
        """`rsi run` as a background process, the leader of its own process group (output in run-<n>.out)."""
        out = open(os.path.join(self.root, "run-%d.out" % (len(self.procs) + 1)), "w")
        p = subprocess.Popen(self.run_argv(), cwd=self.root, env=self.env, stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append((p, out))
        return p

    def run_engine(self, expect, max_steps=None):
        """`rsi run --json` to its stop: the status must be `expect` and the exit code the documented one; returns
        the reason line."""
        p = subprocess.run(self.run_argv(max_steps), cwd=self.root, env=self.env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=600)
        out = json.loads(p.stdout)
        self.assertEqual(out["status"], expect, "%s\n%s" % (p.stdout, p.stderr))
        self.assertEqual(p.returncode, EXIT_CODES[expect], p.stderr)
        last = self.log()[-1]
        self.assertEqual((last["event"], last["status"], last["reason"]), ("exit", expect, out["reason"]))
        return out["reason"]

    def answer(self, option, quote):
        """The scripted user answers the question waiting now (by its id, as the page shows the command)."""
        w = self.ws().waiting()
        p = self.rsi("answer", self.ws_dir, option, "--ref", w["id"], "--by", USER, "--quote", quote)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("answered", p.stdout)
        self.assertIsNone(self.ws().waiting())

    # ---- reading the workspace ---------------------------------------------------------------------

    def ws(self):
        return Workspace(self.ws_dir)

    def log(self):
        with open(os.path.join(self.ws_dir, "work", "run.jsonl"), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def calls(self, did, event="start"):
        """The mock module's calls (start lines) in dispatch `did`'s output directory."""
        path = os.path.join(self.ws_dir, self.ws().dispatch(did)["outdir"], "mock-calls.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [x for x in (json.loads(line) for line in f if line.strip()) if x["event"] == event]

    def outdir(self, did):
        return os.path.join(self.ws_dir, self.ws().dispatch(did)["outdir"])

    def runs(self):
        """The action timeline of each `rsi run`: tokens per run, a run ends at its exit line (or where it was
        killed: a new run starts again at step 1)."""
        runs, cur = [], None
        for r in self.log():
            if r["event"] == "action":
                if r["step"] == 1 or cur is None:
                    cur = []
                    runs.append(cur)
                a = r["action"]
                tok = a["type"]
                if tok == "dispatch":
                    tok += ":%s/%s" % (a["module"], a["purpose"])
                elif tok == "decide":
                    tok += ":" + a["point"]
                elif tok in ("resume", "process_handback"):
                    tok += ":" + a["dispatch_id"]
                elif tok == "ask":
                    tok += ":" + a["what"]
                cur.append(tok)
            elif r["event"] == "exit" and cur is not None:
                cur.append("exit:" + r["status"])
                cur = None
        return runs

    def snapshot(self, name):
        dest = os.environ.get("RSI_E2E_SNAPSHOTS")
        if not dest:
            return
        out = os.path.join(dest, name)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        shutil.copy2(os.path.join(self.ws_dir, "dashboard.html"), out)
        shutil.copytree(os.path.join(self.ws_dir, "dashboard"), os.path.join(out, "dashboard"))

    # ---- checks --------------------------------------------------------------------------------------

    def assertPageAgrees(self, expect_type):
        """The overview on disk (written by the last command) says what next_action says now: its "What the run waits for"
        card is the one the current workspace renders, which _view derives from next_action; and it shows the
        action itself (the question and its options, the module, the long job, the stop). Returns the action."""
        ws = self.ws()
        a = F.next_action(ws)
        self.assertEqual(a["type"], expect_type, a)
        progress = D.module_progress(ws)
        _key, where, what, action = D._view(ws, progress)
        self.assertEqual(action, a)
        with open(os.path.join(self.ws_dir, "dashboard.html"), encoding="utf-8") as f:
            page = f.read()
        card = re.compile(r'<section id="now".*?</section>', re.S)
        now = card.search(page).group(0)
        self.assertEqual(now, card.search(P.render_pages(ws)["dashboard.html"]).group(0))
        if not ws.state["stopped"]:  # the card's sentence (a next step's drops its lead "Next <where>: ")
            lead = "Next %s: " % where
            self.assertIn(html.escape(what[len(lead):] if what.startswith(lead) else what), now)
        if a["type"] == "ask":
            self.assertIn(html.escape(a["question"]), page)
            for o in a["options"]:
                self.assertIn("<code>%s</code>" % html.escape(o), now)
        elif a["type"] in ("dispatch", "decide"):
            self.assertIn("rsi-%s" % (a.get("module") or ws.dispatch(a["dispatch_id"])["module"]), now)
        elif a["type"] == "wait_long_job":
            self.assertIn("long job waiting for its marker", now)
        elif a["type"] == "done":
            self.assertIn("Stopped", now)
            self.assertIn("View the run report", now)
        return a

    def assertNothingTwice(self):
        """No artifact registered twice: one registry entry per source and one register line per entry; every
        registered id in exactly one dispatch record."""
        ws = self.ws()
        entries = [e for e in ws.entries() if e["kind"] != "task"]
        self.assertEqual(len(set(e["source"] for e in entries)), len(entries))
        lines = [r["id"] for r in self.log() if r["event"] == "register"]
        self.assertEqual(sorted(lines), sorted(e["id"] for e in entries))
        by_dispatch = [x["id"] for d in ws.dispatches() for x in d.get("registered") or []]
        self.assertEqual(sorted(by_dispatch), sorted(e["id"] for e in entries))

    def assertSessions(self):
        """Every launch of a dispatch is one mock call, all in the dispatch's one session; launch k > 1 got
        feedback-k.md (a return, the continuation after a long job, the second part of a two-part hand-back, or
        the user's answers to its questions)."""
        for d in self.ws().dispatches():
            calls = self.calls(d["id"])
            self.assertEqual(len(calls), d["launches"], d["id"])
            self.assertEqual(set(c["session"] for c in calls), {d["session"]}, d["id"])
            for k, c in enumerate(calls[1:], 2):
                self.assertEqual(c["feedback_file"], "feedback-%d.md" % k, d["id"])
            self.assertEqual(d["launches"], 1 + sum(int(d.get(k) or 0) for k in
                                                    ("returns", "continued", "followups", "answered", "checkins")),
                             d["id"])

    def assertRunLog(self, killed=0):
        """Each `rsi run` logged every step: actions numbered 1..n, each followed by its result (a killed run's
        last action has none), and an exit line; each completed launch has a launch line."""
        log = self.log()
        segments, cur = [], None
        for r in log:
            if r["event"] == "action":
                if r["step"] == 1:
                    cur = {"steps": [], "results": [], "exit": False}
                    segments.append(cur)
                cur["steps"].append(r["step"])
            elif r["event"] == "result":
                cur["results"].append(r["step"])
            elif r["event"] == "exit":
                cur["exit"] = True
        cut = 0
        for seg in segments:
            self.assertEqual(seg["steps"], list(range(1, len(seg["steps"]) + 1)))
            if seg["exit"]:
                self.assertEqual(seg["results"], seg["steps"])
            else:
                cut += 1
                self.assertEqual(seg["results"], seg["steps"][:-1])
        self.assertEqual(cut, killed)
        launched = [(r["dispatch_id"], r["launch"]) for r in log if r["event"] == "launch"]
        self.assertEqual(len(launched), len(set(launched)))
        if not killed:
            self.assertEqual(sorted(launched), sorted((d["id"], k) for d in self.ws().dispatches()
                                                      for k in range(1, d["launches"] + 1)))
        return segments


# ---- the long scenario -------------------------------------------------------------------------------

class TestLongScenario(E2ECase):
    """The long scenario (four rounds through every turn of the main flow; about 20 seconds)."""

    def test_four_rounds_through_gates_returns_refresh_close_and_milestone(self):
        self.assertEqual(engine.STATUS_EXIT_CODES, EXIT_CODES)  # the code says what the interface doc says
        task = task_def()
        task["approvals"].append("plan")
        release = self.flag("release-long-job")
        self.init(task, {
            "data": {"near_dups": {"1": 0.1},
                     "questions": {"2": ["The new data holds 12 samples of unknown source: keep them or drop them?"]}},
            "train": {"trials": {"1": [{"lr": 100}, {"lr": 1000}]},
                      "config": {"2": {"capacity": 0.95}, "3": {"capacity": 0.75}, "4": {"lr": 100}},
                      "long_job": {"2": {"release": release}}},
            "deliver": {"malformed": {"1": True}},
            "judge": {"entry": {"1": "train", "2": "train", "3": "train"}}})
        self.round_1()
        self.round_2(release)
        self.round_3()
        self.round_4_and_milestone()
        # The whole run: every step logged, sessions as recorded, nothing twice.
        self.assertRunLog()
        self.assertSessions()
        self.assertNothingTwice()
        self.assertEqual(self.runs(), TIMELINE)
        exits = [r["status"] for r in self.log() if r["event"] == "exit"]
        self.assertEqual(exits, ["max_steps", "waiting_user", "waiting_user", "waiting_user", "waiting_long_job",
                                 "waiting_user", "waiting_user", "waiting_user", "waiting_user", "done"])
        # The decisions, in order: the code's defaults at the decision points and the user's answers.
        self.assertEqual([(d["round"], d["point"], d["choice"], d["by"]) for d in self.ws().decisions()], [
            (1, "quality_gate_failed", "resume_data", "code"), (1, "train_gate_failed", "train", "code"),
            (1, "handback_refused", "resume", "code"), (1, "plan", "revise", USER), (1, "plan", "approve", USER),
            (2, "handback", "answered", USER), (2, "plan", "approve", USER), (3, "refresh_sets", "approve", USER),
            (3, "plan", "approve", USER), (4, "train_gate_failed", "train", "code"), (4, "milestone", "yes", USER)])
        # One search per train dispatch and per return: every trial's configuration lies in the search space.
        ws = self.ws()
        self.assertEqual([[e["id"] for e in ws.entries("search") if e["round"] == r] for r in (1, 2, 3, 4)],
                         [["search-0001", "search-0002"], ["search-0003"], ["search-0004"], ["search-0005", "search-0006"]])
        for e in ws.entries("candidate"):
            m = ws.manifest(e["id"])
            self.assertEqual((m["producer"], m["config_request"]["base"], m["code_version"]["entry"]),
                             ("train", "mock-spam", os.path.realpath(TRAIN_COMMAND)))
        # A run after the end changes nothing.
        n = len(self.ws().dispatches())
        self.run_engine("done")
        self.assertEqual(len(self.ws().dispatches()), n)
        self.assertEqual(self.runs()[-1], ["done", "exit:done"])

    def round_1(self):
        # A first look: three steps, then --max-steps stops the run (exit 13) with the quality decision next.
        self.assertIn("--max-steps", self.run_engine("max_steps", max_steps=3))
        ws = self.ws()
        self.assertEqual([(e["kind"], e["id"]) for e in ws.entries()][1:5], [
            ("testset", "testset-0001"), ("evalset", "evalset-0001"), ("valset", "valset-0001"),
            ("trainset", "trainset-0001")])
        self.assertFalse(ws.manifest("trainset-0001")["quality"]["passed"])
        self.assertIsNone(ws.current("trainset"))
        a = self.assertPageAgrees("decide")
        self.assertEqual((a["point"], a["default"], a["dispatch_id"]), ("quality_gate_failed", "resume_data", "d-0001"))
        self.snapshot("1-quality-decision")

        reason = self.run_engine("waiting_user")
        self.assertIn("plan-0001", reason)
        ws = self.ws()
        ds = dict((d["id"], d) for d in ws.dispatches())
        self.assertEqual([(d["module"], d["purpose"]) for d in ds.values()], [
            ("data", "sets"), ("train", "train"), ("deliver", "deliver"), ("judge", "judge")])
        # Quality gate failed -> the same data session, told which check failed -> a trainset through the gate.
        self.assertEqual((ds["d-0001"]["launches"], ds["d-0001"]["returns"]), (2, 1))
        fb = self.calls("d-0001")[1]["feedback"]
        self.assertIn("failed its quality gate", fb)
        self.assertIn("max_near_dup_rate", fb)
        self.assertEqual(ws.current("trainset"), "trainset-0002")
        self.assertTrue(ws.manifest("trainset-0002")["quality"]["passed"])
        # One train session: the first search's trials (lr 100, 1000) fail the gate, their search follows given the
        # trials' ids; train_gate_failed sends the session back (a return) for a second search, which passes.
        self.assertEqual((ds["d-0002"]["launches"], ds["d-0002"]["returns"], ds["d-0002"]["followups"]), (4, 1, 2))
        calls = self.calls("d-0002")
        self.assertIn("candidate-0001, candidate-0002", calls[1]["feedback"])
        self.assertIn("`trial-1` → candidate-0001 (candidate)", calls[1]["feedback"])
        self.assertIn("train gate", calls[2]["feedback"])
        self.assertFalse(any(ws.manifest(c)["train_gate"]["passed"] for c in ("candidate-0001", "candidate-0002")))
        search = ws.manifest("search-0002")
        self.assertGreaterEqual(search["gain_vs_default"]["delta"], 0)
        v1 = ws.version("V1")["candidate_id"]
        self.assertIn(v1, [t["candidate_id"] for t in search["trials"]])
        # Registration refused the malformed delivery -> the same deliver session got the error text -> fixed.
        self.assertEqual([x["dir"] for x in ds["d-0003"]["refused"]], ["delivery"])
        self.assertIn("consistent_with_offline", self.calls("d-0003")[1]["feedback"])
        self.assertEqual(ws.entry("delivery-0001")["source"], os.path.join(self.outdir("d-0003"), "delivery-2"))
        # V1, and the judge's plan from the same session.
        self.assertEqual([(v["version"], v["candidate_id"]) for v in ws.state["versions"]], [("V1", v1)])
        self.assertEqual(ws.manifest("plan-0001")["entry_module"], "train")
        a = self.assertPageAgrees("ask")
        self.assertEqual((a["kind"], a["what"], a["ref"]), ("approval", "plan", "plan-0001"))
        self.snapshot("2-plan-approval")

        # The user wants the plan changed: the judge's session gets the user's words and writes a new plan.
        self.answer("revise", "start from data instead: widen the data coverage first")
        self.assertIn("plan-0002", self.run_engine("waiting_user"))
        ws = self.ws()
        self.assertIn("start from data instead", self.calls("d-0004")[2]["feedback"])
        # The plan was the judge's second part (a followup); the user's revision is its one return.
        self.assertEqual((ws.dispatch("d-0004")["returns"], ws.dispatch("d-0004")["followups"]), (1, 1))
        self.assertEqual(ws.manifest("plan-0002")["entry_module"], "data")
        self.assertEqual(self.assertPageAgrees("ask")["ref"], "plan-0002")
        self.answer("approve", "agreed to the new plan, the next round starts from data")

    def round_2(self, release):
        # A hand-back question from data stops the run until the user answers it; its new trainset is not
        # registered meanwhile.
        reason = self.run_engine("waiting_user")
        self.assertIn("unknown source", reason)
        ws = self.ws()
        self.assertEqual((ws.state["round"], ws.state["rounds"]["2"]["entry"]), (2, "data"))
        self.assertEqual(ws.state["rounds"]["2"]["artifacts"], [])
        self.assertEqual(ws.dispatch("d-0005")["status"], "handed_back")
        a = self.assertPageAgrees("ask")
        self.assertEqual((a["what"], a["ref"], a["options"]), ("handback", "d-0005", ["answered"]))
        self.answer("answered", "drop the samples of unknown source")
        # The answer goes back to the same data session, which drops those rows; then the trainset is registered.
        # The train module starts a long job: the run stops until its marker is there (exit 11).
        self.assertIn("search.marker", self.run_engine("waiting_long_job"))
        ws = self.ws()
        d = ws.dispatch("d-0005")
        self.assertEqual((d["launches"], d["returns"], d["answered"]), (2, 0, 1))
        self.assertIn("drop the samples of unknown source", self.calls("d-0005")[1]["feedback"])
        trainset = ws.current("trainset")
        self.assertEqual(os.path.dirname(ws.entry(trainset)["source"]), self.outdir("d-0005"))
        self.assertIn("dropped 12 rows of unknown source by the user's answer", ws.manifest(trainset)["changes"])
        self.assertFalse([r for r in ws.rows(trainset, "rows") if r["meta"]["source"] == "unknown-export"])
        self.assertEqual(self.ws().dispatch("d-0006")["status"], "long_job")
        self.assertPageAgrees("wait_long_job")
        self.snapshot("3-long-job")
        marker = os.path.join(self.outdir("d-0006"), "search.marker")
        self.assertFalse(os.path.exists(marker))
        open(release, "w").close()
        wait_until(lambda: os.path.exists(marker), what="the long job's marker")
        # The same session collects the trials, then writes their search; deliver, judge: V2.
        self.assertIn("plan-0003", self.run_engine("waiting_user"))
        ws = self.ws()
        d = ws.dispatch("d-0006")
        self.assertEqual((d["launches"], d["returns"], d["continued"], d["followups"]), (3, 0, 1, 1))
        self.assertIn("Long job finished", self.calls("d-0006")[1]["feedback"])
        self.assertEqual(ws.state["rounds"]["2"]["decision"]["version"], "V2")
        self.assertEqual(ws.state["best"], "V2")
        self.assertEqual(ws.state["eval_use_count"], 2)
        self.assertPageAgrees("ask")
        self.answer("approve", "agreed to plan-0003")

    def round_3(self):
        # The evalset is used up: before ③ the run asks for refresh_sets.
        self.assertIn("refresh_sets", self.run_engine("waiting_user"))
        self.assertTrue(self.ws().passed_candidates())
        a = self.assertPageAgrees("ask")
        self.assertEqual((a["kind"], a["what"], a["step"]), ("approval", "refresh_sets", "deliver"))
        self.snapshot("4-refresh-sets")
        self.answer("approve", "approve refreshing the evalset")
        # data makes a new evalset; deliver packs the round's candidate and re-delivers the best version on it; the
        # judge compares the two there: keep.
        self.assertIn("plan-0004", self.run_engine("waiting_user"))
        ws = self.ws()
        self.assertEqual(ws.dispatch("d-0010")["purpose"], "refresh")
        self.assertEqual(ws.current("evalset"), "evalset-0002")
        self.assertEqual(ws.best_delivery_on_current_evalset(), "delivery-0004")
        self.assertEqual([(ws.manifest(x)["candidate_id"], ws.manifest(x)["evalset_id"])
                          for x in ("delivery-0003", "delivery-0004")],
                         [(F.chosen_candidate(ws), "evalset-0002"), (ws.version("V2")["candidate_id"], "evalset-0002")])
        ev = ws.manifest("evaluation-0003")
        self.assertIn("delivery-0004", ev["parents"])
        self.assertEqual(ev["recommendation"]["decision"], "keep")
        self.assertEqual(ws.state["rounds"]["3"]["decision"]["decision"], "keep")
        self.assertPageAgrees("ask")
        self.answer("approve", "agreed to plan-0004, tune again next round")

    def round_4_and_milestone(self):
        # The search's trials all fail the train gate -> back to train, the same session -> its second search fails:
        # the attempts are used up and the round is closed; patience is reached: the milestone.
        self.assertIn("Milestone release", self.run_engine("waiting_user"))
        ws = self.ws()
        r4 = [d for d in ws.dispatches() if d["round"] == 4]
        self.assertEqual([(d["module"], d["launches"], d.get("returns"), d.get("followups")) for d in r4],
                         [("train", 4, 1, 2)])
        self.assertEqual(len(ws.attempts()), 2)
        self.assertEqual(ws.passed_candidates(), [])
        decision = ws.state["rounds"]["4"]["decision"]
        self.assertEqual((decision["decision"], decision["evaluation_id"]), ("keep", None))
        self.assertTrue(ws.stop_check()[0])
        a = self.assertPageAgrees("ask")
        self.assertEqual((a["what"], a["options"]), ("milestone", ["yes", "no"]))
        self.snapshot("5-milestone")
        self.answer("yes", "make the milestone release, verify V2 on the sealed testset")
        # The milestone delivery and evaluation on the sealed testset, each with its use_testset approval quoting the
        # user; then the run stops and writes the report (exit 0).
        self.assertIn("REPORT.md", self.run_engine("done"))
        ws = self.ws()
        self.assertIn("patience", ws.state["stopped"]["reason"])
        ms = [d for d in ws.dispatches() if d["purpose"].startswith("milestone")]
        self.assertEqual([(d["module"], d["purpose"], d["status"]) for d in ms],
                         [("deliver", "milestone_deliver", "done"), ("judge", "milestone_judge", "done")])
        uses = [a for a in ws.state["approvals"] if a["what"] == "use_testset"]
        self.assertEqual([(a["by"], a["quote"], a["used_by"]) for a in uses],
                         [(USER, "make the milestone release, verify V2 on the sealed testset", "delivery-0005"),
                          (USER, "make the milestone release, verify V2 on the sealed testset", "evaluation-0004")])
        ev = ws.manifest("evaluation-0004")
        self.assertTrue(ev["uses_testset"])
        self.assertEqual(ev["candidate_id"], ws.version("V2")["candidate_id"])
        self.assertIn("auc", ev["metrics_test"])
        self.assertTrue(os.path.isfile(os.path.join(self.ws_dir, "REPORT.md")))
        self.assertPageAgrees("done")
        self.snapshot("6-done")


# The actions of each `rsi run` of the long scenario, in order.
TIMELINE = [
    ["dispatch:data/sets", "wait_agent", "process_handback:d-0001", "exit:max_steps"],
    ["decide:quality_gate_failed", "wait_agent", "process_handback:d-0001",
     "dispatch:train/train", "wait_agent", "process_handback:d-0002",
     "resume:d-0002", "wait_agent", "process_handback:d-0002",
     "decide:train_gate_failed", "wait_agent", "process_handback:d-0002",
     "resume:d-0002", "wait_agent", "process_handback:d-0002",
     "dispatch:deliver/deliver", "wait_agent", "process_handback:d-0003", "wait_agent", "process_handback:d-0003",
     "dispatch:judge/judge", "wait_agent", "process_handback:d-0004",
     "promote", "resume:d-0004", "wait_agent", "process_handback:d-0004", "ask:plan", "exit:waiting_user"],
    ["resume:d-0004", "wait_agent", "process_handback:d-0004", "ask:plan", "exit:waiting_user"],
    ["next", "dispatch:data/trainset", "wait_agent", "process_handback:d-0005", "exit:waiting_user"],
    ["resume:d-0005", "wait_agent", "process_handback:d-0005",
     "dispatch:train/train", "wait_agent", "wait_long_job", "exit:waiting_long_job"],
    ["process_handback:d-0006", "wait_agent", "process_handback:d-0006",
     "resume:d-0006", "wait_agent", "process_handback:d-0006",
     "dispatch:deliver/deliver", "wait_agent", "process_handback:d-0007",
     "dispatch:judge/judge", "wait_agent", "process_handback:d-0008",
     "promote", "resume:d-0008", "wait_agent", "process_handback:d-0008", "ask:plan", "exit:waiting_user"],
    ["next", "dispatch:train/train", "wait_agent", "process_handback:d-0009",
     "resume:d-0009", "wait_agent", "process_handback:d-0009", "ask:refresh_sets", "exit:waiting_user"],
    ["dispatch:data/refresh", "wait_agent", "process_handback:d-0010",
     "dispatch:deliver/deliver", "wait_agent", "process_handback:d-0011",
     "dispatch:judge/judge", "wait_agent", "process_handback:d-0012",
     "promote", "resume:d-0012", "wait_agent", "process_handback:d-0012", "ask:plan", "exit:waiting_user"],
    ["next", "dispatch:train/train", "wait_agent", "process_handback:d-0013",
     "resume:d-0013", "wait_agent", "process_handback:d-0013",
     "decide:train_gate_failed", "wait_agent", "process_handback:d-0013",
     "resume:d-0013", "wait_agent", "process_handback:d-0013",
     "close_round", "ask:milestone", "exit:waiting_user"],
    ["dispatch:deliver/milestone_deliver", "wait_agent", "process_handback:d-0014",
     "dispatch:judge/milestone_judge", "wait_agent", "process_handback:d-0015",
     "next", "report", "done", "exit:done"],
]


# ---- ③ Compare ---------------------------------------------------------------------------------------

class TestCompareRound(E2ECase):
    def test_compare_between_train_and_deliver(self):
        # With a compare section, round 1 runs ③ between ② and ④: under --runner mock the mock compare (its one
        # baseline, the trainset's prior) works as a long job, then registers the baseline and, as a followup of the
        # same session, the comparison with a record. The judge reports the baseline on the evalset; at the milestone
        # (answered yes) compare scores it on the testset inputs between the milestone delivery and the milestone
        # judge, which reports it on the testset.
        task = task_def(max_rounds=1)
        task["compare"] = {"baselines": ["prior"]}
        release = self.flag("compare.release")
        self.init(task, {"compare": {"long_job": {"1": {"release": release}}}})
        self.assertIn("baselines.marker", self.run_engine("waiting_long_job"))
        d = self.ws().dispatches(module="compare")[0]
        open(release, "w").close()
        wait_until(lambda: os.path.exists(os.path.join(self.outdir(d["id"]), "baselines.marker")),
                   what="the mock compare's long job")
        self.assertIn("Milestone release", self.run_engine("waiting_user"))  # max_rounds 1 and V1: the milestone
        ws = self.ws()
        # The judge reported the round's baseline on the evalset; status and the overview show candidate vs prior.
        self.assertEqual(list(ws.manifest(ws.entries("evaluation")[0]["id"])["baselines_eval"]), ["baseline-0001"])
        status = self.rsi("status", self.ws_dir).stdout
        self.assertRegex(status, r"compare: comparison-0001 \(vs prior Δauc [+-]\d\.\d{3}\)")
        with open(os.path.join(self.ws_dir, "dashboard.html"), encoding="utf-8") as f:
            self.assertRegex(f.read(), r'<g id="n-compare"(?:(?!</g>).)*>vs prior Δ[+-]\d\.\d{3}<')
        self.answer("yes", "make the milestone release with the baselines on the testset")
        self.assertIn("REPORT.md", self.run_engine("done"))
        ws = self.ws()
        self.assertEqual([(x["module"], x["purpose"]) for x in ws.dispatches()],
                         [("data", "sets"), ("train", "train"), ("compare", "compare"), ("deliver", "deliver"),
                          ("judge", "judge"), ("deliver", "milestone_deliver"), ("compare", "milestone_compare"),
                          ("judge", "milestone_judge")])
        d = ws.dispatch(d["id"])
        self.assertEqual((d["runner"], d["launches"], d["continued"], d["followups"]), ("mock", 3, 1, 1))
        self.assertEqual([x["kind"] for x in d["registered"]], ["baseline", "comparison", "record"])
        self.assertEqual(len(ws.entries("comparison")), 1)
        # The milestone scored the comparison's baseline on the testset inputs; the milestone judge reported it.
        test_baselines = F.milestone_baselines(ws)
        self.assertEqual([ws.manifest(b)["reused_from"] for b in test_baselines], ["baseline-0001"])
        milestone = [e["id"] for e in ws.entries("evaluation") if ws.manifest(e["id"])["uses_testset"]]
        self.assertEqual(list(ws.manifest(milestone[0])["baselines_test"]), test_baselines)
        with open(os.path.join(self.ws_dir, "REPORT.md"), encoding="utf-8") as f:
            report = f.read()
        self.assertIn("### ③ Compare (round 1)", report)
        self.assertIn("Candidate vs strongest baseline (prior): Δauc ", report)
        self.assertIn("On the sealed testset: candidate vs prior Δauc ", report)
        self.assertSessions()
        self.assertNothingTwice()


# ---- check-ins ---------------------------------------------------------------------------------------

class TestCheckins(E2ECase):
    def test_check_ins_while_the_search_runs_in_the_background(self):
        # D6: with flow.checkin_minutes the run waits for the long job itself and checks its session in, in the same
        # session and no return: twice (the second sees the job end), then the continuation collects the trials.
        task = task_def(max_rounds=1)
        task["flow"] = {"checkin_minutes": 0.0005, "max_checkins": 5}
        self.init(task, {"train": {"long_job": {"1": {"checkins": 2}}}})
        self.assertIn("Milestone release", self.run_engine("waiting_user"))
        d = self.ws().dispatch("d-0002")
        self.assertEqual((d["launches"], d["checkins"], d["continued"], d["followups"], d.get("returns")),
                         (5, 2, 1, 1, 0))
        self.assertEqual([(r["dispatch_id"], r["launch"]) for r in self.log() if r["event"] == "checkin"],
                         [("d-0002", 2), ("d-0002", 3)])
        calls = self.calls("d-0002")
        self.assertEqual([c["feedback"].splitlines()[0].split(" (")[0] for c in calls[1:4]],
                         ["# Check-in", "# Check-in", "# Long job finished"])
        self.assertEqual(len(self.ws().entries("search")), 1)
        self.assertSessions()
        self.assertNothingTwice()


# ---- the user's stop ---------------------------------------------------------------------------------

class TestStop(E2ECase):
    """`rsi stop` while an agent works and while a long job runs (the second simulation's cases): the agent and the
    long job's process are terminated, and the stop reason says so."""

    def test_stop_while_an_agent_works(self):
        started = self.flag("train.started")
        self.init(task_def(max_rounds=1), {"train": {"hang": {"1": {"started": started,
                                                                     "release": self.flag("never")}}}})
        run = self.start_run()

        def agent_pid():
            try:
                with open(started) as f:
                    return int(f.read().strip())
            except (OSError, ValueError):
                return None

        wait_until(agent_pid, what="the train agent")
        agent = agent_pid()
        p = self.rsi("stop", self.ws_dir, "--by", USER, "--quote", "this run is not needed any more")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(run.wait(timeout=30), 0)  # the run notices the stop while the agent works
        wait_until(lambda: not pid_alive(agent), timeout=10, what="the train agent to be stopped")
        d = self.ws().dispatch("d-0002")
        self.assertEqual(d["status"], "done")
        self.assertIn("agent terminated", d["closed"])
        self.assertTrue(os.path.isfile(os.path.join(self.ws_dir, "REPORT.md")))
        self.assertPageAgrees("done")

    def test_stop_while_a_long_job_runs(self):
        self.init(task_def(max_rounds=1), {"train": {"long_job": {"1": {"release": self.flag("never")}}}})
        self.assertIn("search.marker", self.run_engine("waiting_long_job"))
        marker = os.path.join(self.outdir("d-0002"), "search.marker")
        with open(marker + ".pid") as f:
            job = int(f.read().split()[0])
        self.assertTrue(pid_alive(job))
        p = self.rsi("stop", self.ws_dir, "--by", USER, "--quote", "end the whole run, do not wait for the training")
        self.assertEqual(p.returncode, 0, p.stderr)
        reason = self.run_engine("done")
        self.assertIn("The stop terminated long job d-0002 (rsi-train, pid %d)" % job, reason)
        wait_until(lambda: not pid_alive(job), timeout=10, what="the long job to be stopped")
        d = self.ws().dispatch("d-0002")
        self.assertEqual((d["long_job_stop"]["result"], d["long_job_stop"]["pid"]), ("terminated", job))
        self.assertIn("long job terminated", d["closed"])
        with open(os.path.join(self.ws_dir, "REPORT.md"), encoding="utf-8") as f:
            self.assertIn("- At the stop, long job d-0002 (rsi-train): terminated (pid %d)" % job, f.read())
        self.assertFalse(os.path.exists(marker))
        self.assertPageAgrees("done")


# ---- the interrupted run -----------------------------------------------------------------------------

class TestInterruptedRun(E2ECase):
    def test_killed_during_an_agent_after_a_handback_and_with_its_agent(self):
        started_data, release_data = self.flag("data.started"), self.flag("data.release")
        started_deliver = self.flag("deliver.started")
        self.init(task_def(max_rounds=1), {
            "data": {"hang": {"1": {"started": started_data, "release": release_data}}},
            "train": {"kill_parent": {"1": "after_handback"}},
            "deliver": {"hang": {"1": {"started": started_deliver, "release": self.flag("deliver.never")}}}})

        # 1. Killed while the data agent works: the agent goes on and hands back on its own.
        run = self.start_run()
        wait_until(lambda: os.path.exists(started_data), what="the data agent to start")
        second = subprocess.run(self.run_argv(), cwd=self.root, env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(second.returncode, 1)  # one `rsi run` per workspace
        self.assertEqual(json.loads(second.stdout)["status"], "error")
        self.assertIn("another `rsi run`", second.stdout)
        os.kill(run.pid, signal.SIGKILL)
        self.assertEqual(run.wait(), -signal.SIGKILL)
        d = self.ws().dispatch("d-0001")
        self.assertEqual(d["status"], "running")
        self.assertIsNotNone(d["launched_at"])
        agent = self.calls("d-0001")[0]["pid"]
        self.assertTrue(pid_alive(agent))
        open(release_data, "w").close()
        wait_until(lambda: self.calls("d-0001", "end") and not pid_alive(agent), what="the data agent to end")

        # 2. The next run finds the hand-back (no second launch), registers it, dispatches train, whose agent kills
        # the run right after writing its hand-back.
        run = self.start_run()
        self.assertEqual(run.wait(timeout=120), -signal.SIGKILL)
        ws = self.ws()
        self.assertEqual([(x["event"], x.get("dispatch_id"), x.get("found")) for x in self.log()
                          if x["event"] == "recover"], [("recover", "d-0001", False)])
        self.assertEqual((ws.dispatch("d-0001")["launches"], len(self.calls("d-0001"))), (1, 1))
        self.assertEqual(ws.dispatch("d-0001")["status"], "done")
        self.assertEqual([e["kind"] for e in ws.entries()][1:5], ["testset", "evalset", "valset", "trainset"])
        self.assertEqual(ws.dispatch("d-0002")["status"], "running")
        self.assertTrue(os.path.isfile(os.path.join(self.outdir("d-0002"), "handback.json")))

        # 3. The next run registers train's hand-back (the first part) once, gets the search from the same session,
        # dispatches deliver and is killed together with the deliver agent (as a machine restart would; the agent
        # runs in its own process group): nothing is handed back.
        run = self.start_run()
        wait_until(lambda: os.path.exists(started_deliver), timeout=120, what="the deliver agent to start")
        agent = self.calls("d-0003")[0]["pid"]
        os.killpg(run.pid, signal.SIGKILL)
        os.killpg(agent, signal.SIGKILL)
        self.assertEqual(run.wait(), -signal.SIGKILL)
        wait_until(lambda: not pid_alive(agent), what="the deliver agent to die")
        ws = self.ws()
        self.assertEqual((ws.dispatch("d-0002")["launches"], len(self.calls("d-0002"))), (2, 2))
        self.assertEqual([x["id"] for x in ws.dispatch("d-0002")["registered"]],
                         ["candidate-0001", "candidate-0002", "candidate-0003", "record-0002", "search-0001"])
        self.assertEqual(ws.dispatch("d-0003")["status"], "running")
        self.assertFalse(os.path.exists(os.path.join(self.outdir("d-0003"), "handback.json")))

        # 4. The next run finds the deliver agent gone without a hand-back: a failed hand-back, sent back to the
        # same session by default; then judge, V1, and the milestone question (max_rounds 1).
        self.assertIn("Milestone release", self.run_engine("waiting_user"))
        ws = self.ws()
        d = ws.dispatch("d-0003")
        self.assertEqual((d["launches"], d["returns"], d["status"]), (2, 1, "done"))
        calls = self.calls("d-0003")
        self.assertEqual([c["session"] for c in calls], [d["session"]] * 2)
        self.assertIn("process was gone", calls[1]["feedback"])
        self.assertEqual([(x["point"], x["choice"], x["by"]) for x in ws.decisions()],
                         [("handback_failed", "resume", "code")])
        self.assertEqual([v["version"] for v in ws.state["versions"]], ["V1"])
        self.assertPageAgrees("ask")
        self.answer("no", "no milestone release, end now")
        self.assertIn("REPORT.md", self.run_engine("done"))

        # Nothing was dispatched or registered twice, and the run completed.
        ws = self.ws()
        self.assertEqual([(d["module"], d["launches"]) for d in ws.dispatches()],
                         [("data", 1), ("train", 2), ("deliver", 2), ("judge", 1)])
        self.assertSessions()
        self.assertNothingTwice()
        self.assertTrue(os.path.isfile(os.path.join(self.ws_dir, "REPORT.md")))
        self.assertPageAgrees("done")
        # Each killed run ends with an action that has no result (the wait for the agent it launched); every launch
        # has its line (the runner returns as soon as the agent is started), and the next run's recover line.
        self.assertRunLog(killed=3)
        self.assertEqual(self.runs()[:3], [["dispatch:data/sets", "wait_agent"],
                                           ["wait_agent", "process_handback:d-0001", "dispatch:train/train",
                                            "wait_agent"],
                                           ["wait_agent", "process_handback:d-0002", "resume:d-0002", "wait_agent",
                                            "process_handback:d-0002", "dispatch:deliver/deliver", "wait_agent"]])
        self.assertEqual([x["dispatch_id"] for x in self.log() if x["event"] == "recover"], ["d-0001", "d-0002",
                                                                                             "d-0003"])
        self.assertEqual(sorted((x["dispatch_id"], x["launch"]) for x in self.log() if x["event"] == "launch"),
                         [("d-0001", 1), ("d-0002", 1), ("d-0002", 2), ("d-0003", 1), ("d-0003", 2), ("d-0004", 1)])


if __name__ == "__main__":
    unittest.main()
