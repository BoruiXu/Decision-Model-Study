"""End-to-end walkthrough: the real `rsi` CLI over two rounds of the fictional spam-check task.

All artifacts are synthetic (built by fixtures); no real module is run.
"""
import os
import shutil
import tempfile
import unittest

import fixtures
from test_cli import run_rsi

# Row ids (q<i>) of the data sets; they never overlap.
TEST_IDS, EVAL_IDS, VAL_IDS, TRAIN_IDS = range(10, 14), range(20, 24), range(0, 4), range(40, 44)


def task_def():
    t = fixtures.binary_task()
    t["approvals"] = t["approvals"] + ["plan"]
    t["stop"]["patience"] = 1
    return t


class Walk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.task = task_def()
        self.ws = os.path.join(self.tmp, "ws")
        self.n = 0
        self.ok(run_rsi("init", self.ws, "--task", fixtures.make_task_dir(self.path("task"), self.task)))

    def path(self, name):
        self.n += 1
        return os.path.join(self.tmp, "src%02d-%s" % (self.n, name))

    def ok(self, r):
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout.strip()

    def refused(self, r, *words):
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        for w in words:
            self.assertIn(w, r.stderr)

    def register(self, kind, rnd=1, ids=None, **fields):
        d = fixtures.make_artifact(kind, self.path(kind), self.task, ids=ids, round=rnd, **fields)
        return d, run_rsi("register", self.ws, kind, d)

    def reg_ok(self, kind, **kw):
        self.ok(self.register(kind, **kw)[1])

    def evaluation(self, rnd, best_before, wrong=(), best_wrong=(), **extra):
        """The judge's evaluation of round `rnd`'s delivery on the evalset. Its numbers are what
        `rsi score` gives for the delivery's predictions (`wrong` ids wrong) against the best
        version's (`best_wrong` ids wrong); the recommendation is what the promotion rule gives."""
        return dict(delivery_id="delivery-%04d" % rnd, candidate_id="candidate-%04d" % rnd,
                    evalset_id="evalset-0001", best_before=best_before, ids=EVAL_IDS, wrong=wrong,
                    best_wrong=best_wrong, train_ids=TRAIN_IDS, **extra)

    def register_sets(self):
        # testset, evalset, valset, trainset; row ids never overlap between sets.
        self.reg_ok("testset", ids=TEST_IDS)
        self.reg_ok("evalset", ids=EVAL_IDS)
        self.reg_ok("valset", ids=VAL_IDS)
        self.reg_ok("trainset", ids=TRAIN_IDS)

    def register_chain(self, rnd, best_before, wrong=()):
        """candidate -> delivery -> evaluation of round `rnd`; return the evaluation id."""
        self.reg_ok("candidate", rnd=rnd, ids=VAL_IDS, train_ids=TRAIN_IDS)
        self.reg_ok("delivery", rnd=rnd, ids=EVAL_IDS, wrong=wrong, candidate_id="candidate-%04d" % rnd)
        self.reg_ok("evaluation", rnd=rnd, **self.evaluation(rnd, best_before, wrong))
        return "evaluation-%04d" % rnd

    def test_two_rounds(self):
        self.register_sets()
        ev1 = self.register_chain(1, None)
        self.assertTrue(self.ok(run_rsi("promote", self.ws, ev1)).startswith("promote V1"))
        self.reg_ok("plan", evaluation_id=ev1, entry_module="train",
                    issues=[dict(fixtures._plan_issue(), attribution="model_capacity", module="train")])
        self.refused(run_rsi("next", self.ws), "approv")
        self.ok(run_rsi("approve", self.ws, "plan", "--by", "alice", "--quote", "go ahead",
                        "--ref", "plan-0001"))
        self.assertEqual(self.ok(run_rsi("next", self.ws)), "round 2: train")

        # Round 2's delivery gets evalset row q20 wrong (auc 0.75 against V1's 1).
        ev2 = self.register_chain(2, {"version": "V1", "evaluation_id": ev1}, wrong=(20,))
        self.assertTrue(self.ok(run_rsi("promote", self.ws, ev2)).startswith("keep"))
        self.reg_ok("plan", rnd=2, evaluation_id=ev2, entry_module="train",
                    issues=[dict(fixtures._plan_issue(), attribution="model_capacity", module="train")])
        self.ok(run_rsi("approve", self.ws, "plan", "--by", "alice", "--quote", "again",
                        "--ref", "plan-0002"))
        self.assertTrue(self.ok(run_rsi("next", self.ws)).startswith("stop:"))
        status = self.ok(run_rsi("status", self.ws))
        self.assertIn("stopped: ", status)
        self.assertIn("patience", status.split("stopped: ")[1].split("\n")[0])

    def test_milestone_release(self):
        # R27: once the stop check says stop, the best version is scored on the sealed testset.
        self.task["stop"]["target"] = 0.8
        self.ws = os.path.join(self.tmp, "ws-milestone")
        self.ok(run_rsi("init", self.ws, "--task", fixtures.make_task_dir(self.path("task"), self.task)))
        self.register_sets()
        ev1 = self.register_chain(1, None)
        self.assertTrue(self.ok(run_rsi("promote", self.ws, ev1)).startswith("promote V1"))
        self.assertTrue(self.ok(run_rsi("check-stop", self.ws)).startswith("stop: best version auc 1 "
                                                                           "reached target 0.8"))
        # Delivery of V1's candidate on the evalset and testset inputs (first use_testset approval).
        d = fixtures.make_artifact("delivery", self.path("delivery"), self.task, ids=EVAL_IDS,
                                   candidate_id="candidate-0001")
        fixtures.add_predictions(d, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        self.refused(run_rsi("register", self.ws, "delivery", d), "predictions_test needs a use_testset approval")
        self.ok(run_rsi("approve", self.ws, "use_testset", "--by", "alice", "--quote", "run the test inputs"))
        self.assertEqual(self.ok(run_rsi("register", self.ws, "delivery", d)), "delivery-0002")
        # The judge scores it on the testset labels (second use_testset approval).
        milestone = self.evaluation(1, {"version": "V1", "evaluation_id": ev1})
        milestone.update(delivery_id="delivery-0002", candidate_id="candidate-0001", uses_testset=True,
                         test_ids=TEST_IDS)
        src, r = self.register("evaluation", **milestone)
        self.refused(r, "uses_testset needs a use_testset approval",
                     "approvals are single-use; this round's were used by delivery-0002")
        self.ok(run_rsi("approve", self.ws, "use_testset", "--by", "alice", "--quote", "score it on the test"))
        self.assertTrue(self.ok(run_rsi("register", self.ws, "evaluation", src)).startswith("evaluation-0002"))
        self.refused(run_rsi("promote", self.ws, "evaluation-0002"),
                     "milestone evaluations are not promotion decisions")
        self.assertTrue(self.ok(run_rsi("next", self.ws)).startswith("stop: best version auc 1 reached target"))
        self.assertIn("stopped: best version auc 1 reached target", self.ok(run_rsi("status", self.ws)))

    def test_three_rounds_without_promotion(self):
        # Task 12: default patience 3 and 3 attempts per round; no candidate ever passes the train gate.
        self.task = fixtures.binary_task()
        del self.task["stop"]["patience"]
        self.task["approvals"] = self.task["approvals"] + ["plan"]
        self.ws = os.path.join(self.tmp, "ws-closed")
        self.ok(run_rsi("init", self.ws, "--task", fixtures.make_task_dir(self.path("task"), self.task)))
        self.register_sets()
        self.assertIn("patience 3 (default), max_attempts_per_round 3 (default)", self.ok(run_rsi("status", self.ws)))
        fail = {"passed": False, "detail": "below the majority baseline"}
        for rnd in (1, 2, 3):
            train_ids = TRAIN_IDS if rnd == 1 else range(40 + 10 * rnd, 44 + 10 * rnd)
            if rnd > 1:  # the plan sent the round to data: a new trainset
                self.reg_ok("trainset", rnd=rnd, ids=train_ids)
            trainset = "trainset-%04d" % rnd
            for attempt in range(1, 5):
                # A search over two trials, every validation prediction wrong: auc 0, below the majority baseline.
                trials = []
                for k in (1, 2):
                    _d, r = self.register("candidate", rnd=rnd, ids=VAL_IDS, wrong=VAL_IDS, train_ids=train_ids,
                                          trainset_id=trainset, train_gate=fail,
                                          notes="round %d attempt %d trial %d" % (rnd, attempt, k))
                    trials.append(self.ok(r).split()[0])
                rows = [{"candidate_id": c, "config": {"lr": 0.001 * k}, "metric_val": 0.0, "compute_hours": 1,
                         "status": "ok"} for k, c in enumerate(trials, 1)]
                _d, r = self.register("search", rnd=rnd, producer="train", trainset_id=trainset, trials=rows,
                                      default_candidate=trials[0], best_candidate=trials[0],
                                      gain_vs_default={"metric": "auc", "delta": 0.0})
                if attempt < 4:
                    self.ok(r)
            self.refused(r, "3 of 3 attempts used in round %d" % rnd, "rsi close-round")
            status = self.ok(run_rsi("status", self.ws))
            self.assertIn("attempts: 3 of 3", status)
            self.assertIn("run `rsi close-round <ws> --reason <text>`", status)
            self.refused(run_rsi("next", self.ws), "rsi promote <ws> <evaluation id>", "rsi close-round <ws> --reason")
            self.assertEqual(self.ok(run_rsi("close-round", self.ws, "--reason", "no candidate passed the train gate")),
                             "keep: closed without evaluation: no candidate passed the train gate")
            check = self.ok(run_rsi("check-stop", self.ws))
            if rnd == 3:
                break
            self.assertTrue(check.startswith("continue:"), check)
            # The judge's plan for a closed round cites no evaluation.
            plan = "plan-%04d" % rnd
            self.reg_ok("plan", rnd=rnd, evaluation_id=None, entry_module="data", issues=[fixtures._plan_issue()])
            self.refused(run_rsi("next", self.ws), plan, "rsi approve")
            self.ok(run_rsi("approve", self.ws, "plan", "--by", "alice", "--quote", "try new data", "--ref", plan))
            self.assertEqual(self.ok(run_rsi("next", self.ws)), "round %d: data" % (rnd + 1))
        # After the third closed round the stop check says stop: no plan is needed.
        self.assertEqual(check, "stop: 3 rounds in a row without promotion reached patience 3")
        self.assertEqual(self.ok(run_rsi("next", self.ws)), check)
        self.assertIn("stopped: 3 rounds in a row without promotion", self.ok(run_rsi("status", self.ws)))
        # The report is a failure report: no version, each round's outcome first.
        out = os.path.join(self.ws, "REPORT.md")
        self.assertIn(out, self.ok(run_rsi("report", self.ws, "--out", out)))
        with open(out, encoding="utf-8") as f:
            report = f.read()
        head = report[:report.index("## 任务")]
        self.assertIn("没有产生可发布的版本", head)
        for rnd in (1, 2, 3):
            self.assertIn("- 第 %d 轮：keep，closed without evaluation: no candidate passed the train gate" % rnd,
                          head)
        self.assertIn("当前最佳：无", report)
        self.assertIn("3 rounds in a row without promotion reached patience 3", report)

    def test_promote_without_evaluation_refused(self):
        self.register_sets()
        self.refused(run_rsi("promote", self.ws, "evaluation-0001"),
                     "is not registered", "round 1 has no registered evaluation",
                     "rsi register <ws> evaluation <dir>")

    def test_wrong_recommendation_refused(self):
        self.register_sets()
        ev1 = self.register_chain(1, None)
        self.ok(run_rsi("promote", self.ws, ev1))
        self.reg_ok("plan", evaluation_id=ev1, entry_module="train",
                    issues=[dict(fixtures._plan_issue(), attribution="model_capacity", module="train")])
        self.ok(run_rsi("approve", self.ws, "plan", "--by", "alice", "--quote", "go", "--ref", "plan-0001"))
        self.ok(run_rsi("next", self.ws))
        self.reg_ok("candidate", rnd=2, ids=VAL_IDS, train_ids=TRAIN_IDS)
        self.reg_ok("delivery", rnd=2, ids=EVAL_IDS, wrong=(20,), candidate_id="candidate-0002")
        # Round 2: the primary metric got worse, so the rule gives keep; claim promote.
        bad = self.evaluation(2, {"version": "V1", "evaluation_id": ev1}, wrong=(20,),
                              recommendation={"decision": "promote", "reason": "walkthrough"})
        self.refused(self.register("evaluation", rnd=2, **bad)[1],
                     "recommendation.decision is 'promote' but the promotion rule gives 'keep'")

    def test_trainset_overlapping_evalset_refused(self):
        self.reg_ok("testset", ids=TEST_IDS)
        self.reg_ok("evalset", ids=EVAL_IDS)
        self.reg_ok("valset", ids=VAL_IDS)
        self.refused(self.register("trainset", ids=range(22, 26))[1], "overlaps the current evalset")


if __name__ == "__main__":
    unittest.main()
