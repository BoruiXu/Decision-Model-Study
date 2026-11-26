"""Round budget and run end (Task 12): stop defaults, attempts per round, closing a round,
user stop and the run report."""
import json
import os
import unittest

from fixtures import add_predictions, binary_task
from rsi.rows import check_task
from rsi.rules import check_stop, stop_settings
from rsi.workspace import Workspace
from test_rules import LoopCase, stop_state
from test_workspace import EVAL, TEST, VAL, run_rsi

FAIL = {"passed": False, "detail": "below the majority baseline"}
# A candidate that fails its train gate: every validation prediction wrong (auc 0 < majority 0.5).
FAILING = dict(wrong=VAL, train_gate=FAIL)


class TestStopDefaults(unittest.TestCase):
    def test_stop_defaults(self):
        t = binary_task()
        del t["stop"]["patience"]
        self.assertEqual(check_task(t), [])
        s = stop_settings(t)
        self.assertEqual((s["patience"], s["max_attempts_per_round"]), (3, 3))
        self.assertEqual((s["target"], s["max_rounds"]), (None, 5))
        self.assertEqual(s["defaulted"], ["max_attempts_per_round", "patience"])
        self.assertNotIn("patience", t["stop"])  # the task itself is not changed
        # The default patience is the one the stop check uses.
        self.assertFalse(check_stop(t, stop_state(no_promotion_streak=2), None)[0])
        stop, reason = check_stop(t, stop_state(no_promotion_streak=3), None)
        self.assertTrue(stop)
        self.assertIn("patience 3", reason)
        # Explicit values override the defaults.
        t["stop"].update(patience=2, max_attempts_per_round=5)
        self.assertEqual(check_task(t), [])
        s = stop_settings(t)
        self.assertEqual((s["patience"], s["max_attempts_per_round"], s["defaulted"]), (2, 5, []))
        self.assertTrue(check_stop(t, stop_state(no_promotion_streak=2), None)[0])
        for bad in (0, 1.5, "3", True):
            t["stop"]["max_attempts_per_round"] = bad
            self.assertTrue(check_task(t), bad)


class TestAttempts(LoopCase):
    def search(self, **candidate_fields):
        """One search (an attempt) over two new trial candidates (no attempts); return its id."""
        a, b = self.candidate(**candidate_fields), self.candidate(**candidate_fields)
        return self.ws.register("search", self.mk("search", producer="train", **self.search_fields(a, b)))

    def test_attempt_limit(self):
        # D4: a search is an attempt, the trial candidates it compares are not.
        self.sets()
        self.assertEqual([self.search() for _ in range(3)], ["search-0001", "search-0002", "search-0003"])
        self.assertEqual(self.ws.attempts(), ["search-0001", "search-0002", "search-0003"])
        self.assertEqual(self.candidate(), "candidate-0007")  # a trial candidate is no attempt
        before = self.state_on_disk()
        src = self.mk("search", producer="train", **self.search_fields("candidate-0001", "candidate-0007"))
        msg = self.refused("search", src, "3 of 3 attempts used in round 1", "rsi close-round")
        self.assertIn("search-0001, search-0002, search-0003", msg)
        r = run_rsi("register", self.wsdir, "search", src)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("3 of 3 attempts used in round 1", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        out = run_rsi("status", self.wsdir).stdout
        self.assertIn("attempts: 3 of 3", out)
        # Candidates of this round passed the train gate: no need to close the round.
        self.assertNotIn("rsi close-round", out)

    def test_attempts_counted_per_round(self):
        self.task["stop"]["max_attempts_per_round"] = 1
        self.first_round()
        self.search()  # round 1's only attempt
        self.refused("search", self.mk("search", producer="train", **self.search_fields(
            "candidate-0002", "candidate-0003")), "1 of 1 attempts used in round 1")
        self.ws.next_round()
        self.assertEqual(self.ws.attempts(), [])
        self.assertEqual(self.search(), "search-0002")
        self.refused("search", self.mk("search", producer="train", **self.search_fields(
            "candidate-0004", "candidate-0005")), "1 of 1 attempts used in round 2")

    def test_status_shows_stop_settings_and_close_round_hint(self):
        del self.task["stop"]["patience"]
        self.sets()
        out = run_rsi("status", self.wsdir).stdout
        self.assertIn("stop settings: target none, max_rounds 5, patience 3 (default), "
                      "max_attempts_per_round 3 (default)", out)
        self.assertIn("attempts: 0 of 3", out)
        for n in (1, 2):
            self.search(**FAILING)
            self.assertNotIn("rsi close-round", run_rsi("status", self.wsdir).stdout)
        self.search(**FAILING)
        out = run_rsi("status", self.wsdir).stdout
        self.assertIn("attempts: 3 of 3", out)
        self.assertTrue(out.startswith("now: "), out)  # the now line (rsi.live.now_line) names the close too
        hint = [line for line in out.splitlines()[1:] if "rsi close-round" in line]
        self.assertEqual(len(hint), 1, out)
        self.assertIn("no candidate of round 1 passed the train gate", hint[0])
        self.assertIn("rsi close-round <ws> --reason", hint[0])
        # Explicit settings are shown without "(default)".
        self.task = binary_task()
        self.task["stop"]["max_attempts_per_round"] = 2
        self.wsdir = os.path.join(self.root, "ws-explicit")
        self.init()
        out = run_rsi("status", self.wsdir).stdout
        self.assertIn("stop settings: target none, max_rounds 5, patience 2, max_attempts_per_round 2", out)
        self.assertNotIn("(default)", out)


class TestCloseRound(LoopCase):
    def setUp(self):
        LoopCase.setUp(self)
        self.task["approvals"].append("plan")

    def failed_round(self):
        """The current round's attempts, all three candidates failing the train gate."""
        for _ in range(3):
            self.candidate(**FAILING)

    def null_plan(self, **fields):
        issue = {"issue": "no candidate beats the majority baseline", "evidence": "train gate failed 3 times",
                 "attribution": "data_coverage", "module": "data", "change": "add rows for the weak slice",
                 "expected": {"metric": "auc", "direction": "up", "size": 0.05}}
        fields.setdefault("evaluation_id", None)
        return self.mk("plan", issues=[issue], entry_module="data", **fields)

    def test_close_round(self):
        self.sets()
        self.failed_round()
        before = self.state_on_disk()
        self.raises(lambda: self.ws.close_round("  "), "--reason")
        self.raises(lambda: self.ws.close_round(None), "--reason")
        r = run_rsi("close-round", self.wsdir)  # --reason is a required option
        self.assertEqual(r.returncode, 2)
        self.assertIn("--reason", r.stderr)
        r = run_rsi("close-round", self.wsdir, "--reason", " ")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("--reason", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        r = run_rsi("close-round", self.wsdir, "--reason", "no candidate passed the train gate")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "keep: closed without evaluation: no candidate passed the train gate")
        s = Workspace(self.wsdir).state
        d = s["rounds"]["1"]["decision"]
        self.assertEqual({k: d[k] for k in ("decision", "version", "evaluation_id", "reason")},
                         {"decision": "keep", "version": None, "evaluation_id": None,
                          "reason": "closed without evaluation: no candidate passed the train gate"})
        self.assertTrue(d["at"])
        self.assertEqual(s["no_promotion_streak"], 1)
        self.assertEqual((s["best"], s["versions"]), (None, []))
        out = run_rsi("status", self.wsdir).stdout
        self.assertIn("decision keep, closed without evaluation", out)
        self.assertNotIn("rsi close-round", out)  # the round is decided: no hint any more
        # Once per round, like promote.
        ws = Workspace(self.wsdir)
        before = self.state_on_disk()
        self.raises(lambda: ws.close_round("again"), "round 1 is already decided")
        self.assertEqual(self.state_on_disk(), before)
        # promote and a second close name how the round was closed, never "None" (R35).
        for fn in (lambda: ws.promote("evaluation-0009"), lambda: ws.close_round("again")):
            msg = self.raises(fn, "round 1 is already decided",
                              "keep: closed without evaluation: no candidate passed the train gate")
            self.assertNotIn("None", msg)
        r = run_rsi("promote", self.wsdir, "evaluation-0009")
        self.assertEqual(r.returncode, 1)
        self.assertIn("closed without evaluation: no candidate passed the train gate", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        # The closed round counts as completed for max_rounds and as a round without promotion.
        self.assertIn("1 of max_rounds 5 completed", ws.stop_check()[1])
        self.assertIn("no-promotion streak 1 of patience 2", ws.stop_check()[1])

    def test_close_round_refused_with_evaluation(self):
        self.ready()
        self.ws.register("evaluation", self.eval_src())
        before = self.state_on_disk()
        self.raises(lambda: self.ws.close_round("give up"), "evaluation-0001", "rsi promote <ws> evaluation-0001")
        r = run_rsi("close-round", self.wsdir, "--reason", "give up")
        self.assertEqual(r.returncode, 1)
        self.assertIn("rsi promote", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        # After promote the round is decided.
        self.ws.promote("evaluation-0001")
        self.raises(lambda: self.ws.close_round("give up"), "round 1 is already decided", "promote V1")

    def test_close_round_ignores_milestone_evaluation(self):
        # A milestone evaluation decides nothing, so it does not keep a round from closing.
        self.first_round()
        self.ws.approve("plan", by="alice", quote="go", ref="plan-0001")
        self.ws.next_round()
        self.approve("use_testset")
        self.ws.register("delivery", add_predictions(
            self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
            self.task, "predictions_test", TEST, testset_id="testset-0001"))
        self.approve("use_testset")
        self.ws.register("evaluation", self.eval_src(
            delivery_id="delivery-0002", uses_testset=True, test_ids=TEST,
            best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
            recommendation={"decision": "keep", "reason": "milestone"}))
        self.assertEqual(self.ws.close_round("no new candidate")[0], "keep")

    def test_plan_after_closed_round(self):
        self.sets()
        self.failed_round()
        # Not closed yet: a plan without an evaluation is refused.
        msg = self.refused("plan", self.null_plan(), "evaluation_id is null", "rsi close-round")
        self.assertIn("round 1", msg)
        self.ws.close_round("no candidate passed the train gate")
        self.assertEqual(self.ws.register("plan", self.null_plan()), "plan-0001")
        self.assertEqual(self.ws.state["rounds"]["1"]["plan_id"], "plan-0001")
        self.raises(lambda: self.ws.next_round(), "plan-0001", "rsi approve")
        self.ws.approve("plan", by="alice", quote="try more data", ref="plan-0001")
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "round 2: data")

    def test_null_plan_refused_after_promote(self):
        self.first_round()  # round 1 decided by `rsi promote`
        self.refused("plan", self.null_plan(), "evaluation_id is null", "rsi promote")
        # A plan citing a registered evaluation still works in a closed round.
        self.ws.approve("plan", by="alice", quote="go", ref="plan-0001")
        self.ws.next_round()
        self.failed_round()
        self.ws.close_round("no candidate passed the train gate")
        self.assertEqual(self.ws.register("plan", self.null_plan(evaluation_id="evaluation-0001")), "plan-0002")

    def test_next_message_without_evaluation(self):
        self.sets()
        self.failed_round()
        before = self.state_on_disk()
        msg = self.raises(lambda: self.ws.next_round(), "round 1 has no decision",
                          "rsi promote <ws> <evaluation id>", "rsi close-round <ws> --reason")
        self.assertIn("no candidate can be evaluated", msg)
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 1)
        self.assertIn("rsi close-round", r.stderr)
        self.assertEqual(self.state_on_disk(), before)


class TestUserStop(LoopCase):
    def test_user_stop(self):
        self.first_round()
        before = self.state_on_disk()
        self.raises(lambda: self.ws.stop("", "stop now"), "--by")
        self.raises(lambda: self.ws.stop("alice", "  "), "--quote")
        r = run_rsi("stop", self.wsdir, "--by", "alice")  # --quote is a required option
        self.assertEqual(r.returncode, 2)
        self.assertIn("--quote", r.stderr)
        r = run_rsi("stop", self.wsdir, "--by", " ", "--quote", "stop now")
        self.assertEqual(r.returncode, 1)
        self.assertIn("--by", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        r = run_rsi("stop", self.wsdir, "--by", "alice", "--quote", "enough, ship V1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "stop: user requested: enough, ship V1")
        st = Workspace(self.wsdir).state["stopped"]
        self.assertEqual({k: st[k] for k in ("reason", "by", "round")},
                         {"reason": "user requested: enough, ship V1", "by": "alice", "round": 1})
        self.assertTrue(st["at"])
        self.assertIn("stopped: user requested: enough, ship V1", run_rsi("status", self.wsdir).stdout)
        self.assertTrue(run_rsi("check-stop", self.wsdir).stdout.startswith("stop: "))
        # Afterwards only `approve delete` and records (R18).
        ws = self.ws = Workspace(self.wsdir)
        ws.approve("delete", by="alice", quote="delete the checkpoints", ref="candidate-0001")
        self.assertEqual(ws.register("record", self.mk("record", artifacts=["evaluation-0001"])), "record-0001")
        before = self.state_on_disk()
        self.raises(lambda: ws.next_round(), "stopped")
        self.raises(lambda: ws.promote("evaluation-0001"), "stopped")
        self.raises(lambda: ws.close_round("late"), "stopped")
        self.raises(lambda: ws.approve("over_budget", by="alice", quote="yes"), "stopped")
        self.refused("trainset", self.mk("trainset", ids=range(40, 44)), "stopped")
        self.raises(lambda: ws.stop("alice", "stop again"), "stopped")
        r = run_rsi("stop", self.wsdir, "--by", "alice", "--quote", "stop again")
        self.assertEqual(r.returncode, 1)
        self.assertIn("stopped", r.stderr)
        self.assertEqual(self.state_on_disk(), before)


class TestStatusLines(LoopCase):
    BEST = "best delivery on current evalset: "

    def status(self):
        r = run_rsi("status", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def line(self, out, prefix):
        return [l for l in out.splitlines() if l.startswith(prefix)]

    def test_status_best_delivery_line(self):
        self.sets()
        self.assertEqual(self.line(self.status(), self.BEST), [])  # no version yet
        self.assertIsNone(Workspace(self.wsdir).best_delivery_on_current_evalset())
        self.candidate()
        self.delivery()
        self.ws.register("evaluation", self.eval_src())
        self.ws.promote("evaluation-0001")
        out = self.status()
        self.assertEqual(self.line(out, self.BEST), [self.BEST + "delivery-0001"])
        lines = out.splitlines()
        self.assertTrue(lines[lines.index(self.line(out, "best: ")[0]) + 1].startswith(self.BEST))
        self.plan()
        self.ws.next_round()
        # A refreshed evalset: V1 has no delivery on it yet.
        self.approve("refresh_sets")
        new = range(60, 64)
        self.ws.register("evalset", self.mk("evalset", ids=new))
        self.assertEqual(self.line(self.status(), self.BEST),
                         [self.BEST + "none (re-deliver the best version on the current evalset first)"])
        # Another candidate's delivery on the new evalset does not count.
        self.candidate()
        self.delivery(candidate_id="candidate-0002", evalset_id="evalset-0002", ids=new)
        self.assertTrue(self.line(self.status(), self.BEST)[0].startswith(self.BEST + "none"))
        # V1's candidate delivered on the new evalset.
        self.delivery(candidate_id="candidate-0001", evalset_id="evalset-0002", ids=new)
        self.assertEqual(self.line(self.status(), self.BEST), [self.BEST + "delivery-0003"])
        self.assertEqual(Workspace(self.wsdir).best_delivery_on_current_evalset(), "delivery-0003")

    def test_status_marks_milestone_delivery(self):
        self.task["stop"]["target"] = 0.8
        self.first_round()
        self.approve("use_testset")
        self.ws.register("delivery", add_predictions(
            self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
            self.task, "predictions_test", TEST, testset_id="testset-0001"))
        rows = [l for l in self.status().splitlines() if l.strip().startswith("delivery-")]
        self.assertEqual(len(rows), 2)
        self.assertNotIn("milestone", rows[0])
        self.assertIn("milestone", rows[1])
        # The milestone delivery is on the current evalset but is not the best delivery.
        self.assertEqual(self.line(self.status(), self.BEST), [self.BEST + "delivery-0001"])

    def test_status_stop_time_local(self):
        self.first_round()
        self.ws.stop("alice", "enough")
        line = self.line(self.status(), "stopped: ")[0]
        self.assertIn("user requested: enough (at ", line)
        when = line.rsplit("(at ", 1)[1].rstrip(")")
        self.assertNotIn("T", when)
        self.assertNotIn("Z", when)
        self.assertRegex(when, r"^\d{4}-\d\d-\d\d \d\d:\d\d$")
        from rsi import timing
        self.assertEqual(when, timing.format_full(Workspace(self.wsdir).state["stopped"]["at"]))


class TestReportGuards(LoopCase):
    """R35: --out never writes over the workspace's own files, however the path is spelled."""

    def setUp(self):
        LoopCase.setUp(self)
        self.sets()
        self.state = os.path.join(self.wsdir, "STATE.json")
        self.registry = os.path.join(self.wsdir, "registry.jsonl")
        self.arts = os.path.join(self.wsdir, "artifacts")

    def files_on_disk(self):
        out = []
        for path in (self.state, self.registry):
            with open(path, "rb") as f:
                out.append(f.read())
        return out + [sorted(os.listdir(self.arts))]

    def refused_out(self, out):
        before = self.files_on_disk()
        r = run_rsi("report", self.wsdir, "--out", out)
        self.assertEqual(r.returncode, 1, (out, r.stdout))
        self.assertIn("--out", r.stderr)
        self.assertEqual(self.files_on_disk(), before, out)
        return r.stderr

    def test_exact_paths_and_case_variants_refused(self):
        for out in (self.state, self.registry, self.arts, os.path.join(self.arts, "REPORT.md"),
                    os.path.join(self.arts, "task-0001", "task.json"), os.path.join(self.wsdir, "Artifacts", "REPORT.md")):
            self.refused_out(out)
        # On a case-insensitive filesystem these name the workspace's own files and are refused;
        # on a case-sensitive one they are other paths. Either way the workspace is untouched.
        for out in (os.path.join(self.wsdir, "state.json"), os.path.join(self.wsdir, "REGISTRY.JSONL")):
            before = self.files_on_disk()
            same = os.path.exists(out)
            r = run_rsi("report", self.wsdir, "--out", out)
            self.assertEqual(r.returncode, 1 if same else 0, (out, r.stderr))
            self.assertEqual(self.files_on_disk(), before, out)

    def test_same_file_under_another_name_refused(self):
        # A hard link names the same file under a path whose realpath differs, as a case variant
        # does on a case-insensitive filesystem: only a same-file check catches it.
        for target in (self.state, self.registry):
            link = os.path.join(self.root, "link-" + os.path.basename(target))
            os.link(target, link)
            self.assertNotEqual(os.path.realpath(link), os.path.realpath(target))
            self.assertIn("must not", self.refused_out(link))
        # A directory link into artifacts/, and a dangling link to a new file under it.
        dlink = os.path.join(self.root, "arts-link")
        os.symlink(self.arts, dlink)
        self.assertIn("must not", self.refused_out(os.path.join(dlink, "REPORT.md")))
        dangling = os.path.join(self.root, "dangling.md")
        os.symlink(os.path.join(self.arts, "new.md"), dangling)
        self.assertIn("must not", self.refused_out(dangling))
        self.assertFalse(os.path.exists(os.path.join(self.arts, "new.md")))


class TestReport(LoopCase):
    def report(self, *args):
        r = run_rsi("report", self.wsdir, *args)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def files_on_disk(self):
        out = []
        for name in ("STATE.json", "registry.jsonl"):
            with open(os.path.join(self.wsdir, name), "rb") as f:
                out.append(f.read())
        return out

    def test_report(self):
        # A promoted run: round 1 promotes V1, round 2 is closed, a milestone evaluation, a user stop.
        self.first_round()  # candidate-0001 -> V1 (auc 1), plan-0001
        self.ws.register("record", self.mk("record", artifacts=["trainset-0001"]))
        self.ws.register("record", self.mk("record", module="judge", producer="judge", artifacts=[],
                                           result={"metric": "auc", "before": None, "after": 0.85,
                                                   "conclusion": "inconclusive"}))
        self.assertEqual(self.ws.next_round(), (2, "data"))
        for _ in range(3):
            self.candidate(**FAILING)
        self.ws.register("search", self.mk("search", producer="train", **self.search_fields(
            "candidate-0002", "candidate-0003", "candidate-0004")))
        self.ws.close_round("no candidate passed the train gate")
        self.approve("use_testset")
        self.ws.register("delivery", add_predictions(
            self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
            self.task, "predictions_test", TEST, testset_id="testset-0001"))
        self.approve("use_testset")
        self.ws.register("evaluation", self.eval_src(
            delivery_id="delivery-0002", uses_testset=True, test_ids=TEST,
            best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
            recommendation={"decision": "keep", "reason": "milestone"}))
        self.ws.stop("alice", "ship V1")
        before = self.files_on_disk()
        out = self.report()
        self.assertEqual(self.files_on_disk(), before)
        self.assertTrue(out.startswith("# RSI run report: spam-check\n"), out[:80])
        self.assertNotIn("No releasable version was produced", out)
        for section in ("## Task", "## Stop status", "## Rounds", "## Versions", "## Milestone evaluations",
                        "## Experiment records", "## Attempts and budget"):
            self.assertIn(section, out)
        for text in ("- Task: spam-check v1 (task-0001)", "- Decision type: binary",
                     "- Primary metric: auc (higher is better)", "- Target: not set",
                     "- Stopped (round 2",
                     "user requested: ship V1",
                     "| 1 | data | promote V1 | no best version yet and the candidate passed its train gate "
                     "| evaluation-0001 |",
                     "| 2 | data | keep | closed without evaluation: no candidate passed the train gate | none |",
                     "| V1 | candidate-0001 | delivery-0001 | evaluation-0001 | 1 |",
                     "Current best: V1, auc 1 (evaluation-0001)",
                     "- Model package: %s" % os.path.join(self.wsdir, "artifacts", "delivery-0001", "package"),
                     "- Entry point: package/predict --inputs",
                     os.path.join(self.wsdir, "artifacts", "evaluation-0001", "report.md"),
                     "| evaluation-0002 | candidate-0001 | 2 | auc 1, log_loss 0.105361 |",
                     "- evaluation-0002: evaluation report %s; delivery delivery-0002 (%s), model package %s"
                     % (os.path.join(self.wsdir, "artifacts", "evaluation-0002", "report.md"),
                        os.path.join(self.wsdir, "artifacts", "delivery-0002"),
                        os.path.join(self.wsdir, "artifacts", "delivery-0002", "package")),
                     "### ① Data", "| record-0001 | 1 | success |",
                     "### ⑤ Judge & decide", "| record-0002 | 1 | inconclusive |",
                     "| 1 | 0 of 3 |  |",
                     "| 2 | 1 of 3 | search-0001 |",
                     "GPU hours: 6 used, no budget set", "Agent $: 0.00 used, no budget set"):
            self.assertIn(text, out)
        # --out writes the same report to a file and changes nothing else.
        dst = os.path.join(self.wsdir, "REPORT.md")
        r = run_rsi("report", self.wsdir, "--out", dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(dst, r.stdout)
        with open(dst, encoding="utf-8") as f:
            self.assertEqual(f.read(), out)
        self.assertEqual(self.files_on_disk(), before)
        # The workspace's own files are never overwritten.
        for bad in (os.path.join(self.wsdir, "STATE.json"), os.path.join(self.wsdir, "registry.jsonl"),
                    os.path.join(self.wsdir, "artifacts", "REPORT.md"), self.wsdir,
                    os.path.join(self.root, "no-such-dir", "REPORT.md")):
            r = run_rsi("report", self.wsdir, "--out", bad)
            self.assertEqual(r.returncode, 1, bad)
            self.assertIn("--out", r.stderr)
        self.assertEqual(self.files_on_disk(), before)
        self.assertFalse(os.path.exists(os.path.join(self.wsdir, "artifacts", "REPORT.md")))

    def test_report_external_package(self):
        # Task 14: a package kept elsewhere is named by its external entry's location and sha256.
        self.sets()
        self.candidate()
        self.delivery(package={"location": "external:weights", "entry": "serve --port 8080",
                               "environment": "python3"},
                      external={"weights": {"location": "gpu-box:/srv/pkg.tar", "sha256": "ab" * 32}})
        self.ws.register("evaluation", self.eval_src())
        self.ws.promote("evaluation-0001")
        self.assertIn("- Model package: gpu-box:/srv/pkg.tar (external:weights, sha256 %s)" % ("ab" * 32), self.report())

    def test_report_names_undecided_evaluation(self):
        # R35: a user stop can leave a registered evaluation undecided; the report names it.
        self.ready()
        self.ws.register("evaluation", self.eval_src())
        self.ws.stop("alice", "stop here")
        out = self.report()
        self.assertIn("| 1 | data | undecided (evaluation-0001) |", out)
        self.assertIn("- Round 1: undecided (evaluation-0001)", out[:out.index("## Task")])

    def test_report_without_promoted_version(self):
        self.task["stop"]["target"] = 0.9
        self.sets()
        for _ in range(3):
            self.candidate(**FAILING)
        self.ws.close_round("no candidate passed the train gate")
        issue = {"issue": "below baseline", "evidence": "3 train gate failures", "attribution": "data_coverage",
                 "module": "data", "change": "more rows", "expected": {"metric": "auc", "direction": "up",
                                                                       "size": 0.05}}
        self.ws.register("plan", self.mk("plan", evaluation_id=None, issues=[issue], entry_module="data"))
        self.assertEqual(self.ws.next_round(), (2, "data"))
        self.candidate(**FAILING)
        self.candidate(**FAILING)
        self.ws.register("search", self.mk("search", producer="train", **self.search_fields(
            "candidate-0004", "candidate-0005")))
        before = self.files_on_disk()
        out = self.report()
        self.assertEqual(self.files_on_disk(), before)
        # The summary comes first, before the sections, and lists each round.
        head = out[:out.index("## Task")]
        self.assertIn("No releasable version was produced", head)
        self.assertIn("- Round 1: keep; closed without evaluation: no candidate passed the train gate", head)
        self.assertIn("- Round 2: undecided", head)
        self.assertIn("progress report", head)  # not stopped yet
        self.assertIn("- Target: auc 0.9", out)
        self.assertIn("- Not stopped", out)
        self.assertIn("Current best: none", out)
        self.assertIn("| 2 | 1 of 3 | search-0001 |", out)
        self.assertIn("## Milestone evaluations\n\nNone.", out)
        self.assertIn("## Experiment records\n\nNone.", out)
        # After a stop, the same summary without the progress note.
        self.ws.stop("alice", "give up")
        out = self.report()
        head = out[:out.index("## Task")]
        self.assertIn("No releasable version was produced", head)
        self.assertNotIn("progress report", head)


if __name__ == "__main__":
    unittest.main()
