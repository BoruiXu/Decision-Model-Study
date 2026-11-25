"""Promotion rule, stop conditions, approvals and round progression (Task 8)."""
import shutil
import unittest

from fixtures import add_predictions, binary_task
from rsi.rules import APPROVAL_ITEMS, check_stop, decide_promotion
from rsi.workspace import RsiError, Workspace
from test_workspace import EVAL, TEST, VAL, WsCase, run_rsi


def evaluation(delta=0.05, ci_low=0.01, ci_high=0.09, candidate=None, best="default", primary="auc"):
    """An evaluation manifest fragment: what decide_promotion reads."""
    if candidate is None:
        candidate = {primary: 0.85, "log_loss": 0.4}
    if best == "default":
        best = {primary: 0.8, "log_loss": 0.5}
    return {
        "metrics": {"candidate": candidate, "best": best, "baselines": {}},
        "deltas": {primary: {"delta": delta, "ci_low": ci_low, "ci_high": ci_high}},
    }


def lower_is_better_task():
    t = binary_task()
    t["metrics"]["primary"] = "expected_cost"
    t["metrics"]["higher_is_better"] = False
    t["promotion_rule"] = {"min_delta": 0.01, "ci_lower_min": 0.0, "guards": []}
    return t


class TestDecidePromotion(unittest.TestCase):
    def setUp(self):
        self.task = binary_task()

    def test_first_round_promotes_if_gate_passed(self):
        decision, reason = decide_promotion(self.task, evaluation(), False, True)
        self.assertEqual(decision, "promote")
        self.assertIn("train gate", reason)
        # Without a best version the metrics play no part.
        bad = evaluation(delta=-1.0, ci_low=-2.0, ci_high=-0.5, best=None)
        self.assertEqual(decide_promotion(self.task, bad, False, True)[0], "promote")
        decision, reason = decide_promotion(self.task, evaluation(), False, False)
        self.assertEqual(decision, "keep")
        self.assertIn("train gate", reason)

    def test_promote_on_improvement_above_min_delta(self):
        decision, reason = decide_promotion(self.task, evaluation(delta=0.05), True, True)
        self.assertEqual(decision, "promote")
        self.assertIn("min_delta", reason)
        self.assertEqual(decide_promotion(self.task, evaluation(delta=-0.05), True, True)[0], "keep")
        # With a best version, the candidate's train gate does not decide.
        self.assertEqual(decide_promotion(self.task, evaluation(delta=0.05), True, False)[0], "promote")
        # The improvement must be strictly above min_delta (0.01): exactly 0.01 (also 3/300 and with float noise)
        # is not above it.
        noisy = 0.7 - 0.69
        self.assertGreater(noisy, 0.01)
        for delta in (0.01, 3 / 300, noisy):
            decision, reason = decide_promotion(self.task, evaluation(delta=delta), True, True)
            self.assertEqual(decision, "keep", delta)
            self.assertIn("is not above min_delta", reason)
        self.assertEqual(decide_promotion(self.task, evaluation(delta=0.010001), True, True)[0], "promote")
        # An interval lower bound exactly at ci_lower_min passes, with noise too.
        self.task["promotion_rule"]["ci_lower_min"] = 0.01
        for low in (0.01, 0.7 - 0.69, 0.3 - 0.29):
            self.assertEqual(decide_promotion(self.task, evaluation(delta=0.05, ci_low=low), True, True)[0],
                             "promote", low)
        self.assertEqual(decide_promotion(self.task, evaluation(delta=0.05, ci_low=0.0099), True, True)[0], "keep")

    def test_keep_when_ci_below_min(self):
        self.task["promotion_rule"]["ci_lower_min"] = 0.0
        decision, reason = decide_promotion(self.task, evaluation(delta=0.05, ci_low=-0.01), True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("ci_lower_min", reason)
        self.assertEqual(decide_promotion(self.task, evaluation(delta=0.05, ci_low=0.0), True, True)[0],
                         "promote")
        # ci_lower_min null: the interval is not required.
        self.task["promotion_rule"]["ci_lower_min"] = None
        self.assertEqual(decide_promotion(self.task, evaluation(delta=0.05, ci_low=-0.5), True, True)[0],
                         "promote")

    def test_guard_regression_blocks(self):
        # log_loss: lower is better, max_regression 0.02 (best 0.5).
        ev = evaluation(candidate={"auc": 0.9, "log_loss": 0.53})
        decision, reason = decide_promotion(self.task, ev, True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("log_loss", reason)
        # Exactly max_regression is allowed (0.52 - 0.5 is not exactly 0.02 in floating point).
        ev = evaluation(candidate={"auc": 0.9, "log_loss": 0.52})
        self.assertEqual(decide_promotion(self.task, ev, True, True)[0], "promote")
        # A higher-is-better guard regresses when the candidate is lower.
        self.task["promotion_rule"]["guards"].append(
            {"metric": "recall", "higher_is_better": True, "max_regression": 0.0})
        ev = evaluation(candidate={"auc": 0.9, "log_loss": 0.4, "recall": 0.7},
                        best={"auc": 0.8, "log_loss": 0.5, "recall": 0.71})
        decision, reason = decide_promotion(self.task, ev, True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("recall", reason)
        ev["metrics"]["candidate"]["recall"] = 0.72
        self.assertEqual(decide_promotion(self.task, ev, True, True)[0], "promote")

    def test_lower_is_better_flips_sign(self):
        t = lower_is_better_task()

        def ev(delta, ci_low, ci_high):
            return evaluation(delta, ci_low, ci_high, primary="expected_cost",
                              candidate={"expected_cost": 1.0}, best={"expected_cost": 1.05})

        # Cost fell by 0.05 and the whole interval is below 0: an improvement.
        decision, reason = decide_promotion(t, ev(-0.05, -0.09, -0.01), True, True)
        self.assertEqual(decision, "promote", reason)
        # Cost rose: not an improvement, even though delta is positive.
        self.assertEqual(decide_promotion(t, ev(0.05, 0.01, 0.09), True, True)[0], "keep")
        # Improvement lower bound is -ci_high = -0.01 < ci_lower_min 0.
        decision, reason = decide_promotion(t, ev(-0.05, -0.09, 0.01), True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("ci_lower_min", reason)
        # A lower-is-better guard on the primary metric uses cand - best.
        t["promotion_rule"]["guards"] = [
            {"metric": "expected_cost", "higher_is_better": False, "max_regression": 0.0}]
        self.assertEqual(decide_promotion(t, ev(-0.05, -0.09, -0.01), True, True)[0], "promote")

    def test_missing_metric_keeps(self):
        ev = evaluation()
        ev["deltas"] = {}
        decision, reason = decide_promotion(self.task, ev, True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("deltas.auc", reason)
        decision, reason = decide_promotion(self.task, evaluation(best=None), True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("metrics.best.log_loss", reason)
        decision, reason = decide_promotion(self.task, evaluation(candidate={"auc": 0.9}), True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("metrics.candidate.log_loss", reason)
        self.task["promotion_rule"]["ci_lower_min"] = 0.0
        ev = evaluation()
        del ev["deltas"]["auc"]["ci_low"]
        decision, reason = decide_promotion(self.task, ev, True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("deltas.auc.ci_low", reason)
        # A non-finite number counts as missing.
        decision, reason = decide_promotion(self.task, evaluation(delta=float("nan")), True, True)
        self.assertEqual(decision, "keep")
        self.assertIn("deltas.auc.delta", reason)


def stop_state(**over):
    s = {
        "rounds": {"1": {"entry": "data", "artifacts": [], "decision": None, "plan_id": None}},
        "no_promotion_streak": 0,
        "stopped": None,
        "budget_used": {"compute_hours": 0},
    }
    s.update(over)
    return s


def decided(n):
    """`rounds` with rounds 1..n decided and round n+1 open."""
    rounds = dict((str(i), {"entry": "data", "artifacts": [], "decision": {"decision": "keep"},
                            "plan_id": "plan-%04d" % i}) for i in range(1, n + 1))
    rounds[str(n + 1)] = {"entry": "data", "artifacts": [], "decision": None, "plan_id": None}
    return rounds


class TestCheckStop(unittest.TestCase):
    def setUp(self):
        self.task = binary_task()

    def test_check_stop(self):
        self.task["stop"]["target"] = 0.9
        self.assertFalse(check_stop(self.task, stop_state(), None)[0])
        self.assertFalse(check_stop(self.task, stop_state(), 0.89)[0])
        stop, reason = check_stop(self.task, stop_state(), 0.9)
        self.assertTrue(stop)
        self.assertIn("target", reason)
        # Lower is better: reaching the target means going at or below it.
        t = lower_is_better_task()
        t["stop"]["target"] = 0.5
        self.assertFalse(check_stop(t, stop_state(), 0.6)[0])
        self.assertTrue(check_stop(t, stop_state(), 0.5)[0])
        # No target: never stops on the metric.
        self.task["stop"]["target"] = None
        self.assertFalse(check_stop(self.task, stop_state(), 1.0)[0])
        # The compute budget.
        self.task["budget"]["compute_hours"] = 10
        self.assertFalse(check_stop(self.task, stop_state(budget_used={"compute_hours": 9.5}), None)[0])
        stop, reason = check_stop(self.task, stop_state(budget_used={"compute_hours": 10}), None)
        self.assertTrue(stop)
        self.assertIn("compute", reason)
        # Sums of costs carry floating-point error: 0.7 + 0.1 + 0.1 + 0.1 < 1.0.
        self.task["budget"]["compute_hours"] = 1.0
        used = 0.7 + 0.1 + 0.1 + 0.1
        self.assertTrue(check_stop(self.task, stop_state(budget_used={"compute_hours": used}), None)[0])
        self.task["budget"]["compute_hours"] = None
        self.assertFalse(check_stop(self.task, stop_state(budget_used={"compute_hours": 1e6}), None)[0])
        # max_rounds counts decided rounds.
        self.task["stop"]["max_rounds"] = 2
        self.assertFalse(check_stop(self.task, stop_state(rounds=decided(1)), None)[0])
        stop, reason = check_stop(self.task, stop_state(rounds=decided(2)), None)
        self.assertTrue(stop)
        self.assertIn("max_rounds", reason)
        # No condition met: continue, saying so; a stopped run stays stopped.
        self.task = binary_task()
        stop, reason = check_stop(self.task, stop_state(), 0.8)
        self.assertFalse(stop)
        self.assertIn("no stop condition", reason)
        st = stop_state(stopped={"reason": "patience reached", "at": "2026-11-21T00:00:00Z"})
        stop, reason = check_stop(self.task, st, None)
        self.assertTrue(stop)
        self.assertIn("patience reached", reason)


ATTRIBUTION = {"data": "data_coverage", "train": "model_capacity", "deliver": "runtime"}


class LoopCase(WsCase):
    def mk(self, kind, name=None, **fields):
        ws = getattr(self, "ws", None)
        if ws is not None:
            fields.setdefault("round", ws.state["round"])
        return WsCase.mk(self, kind, name, **fields)

    def ready(self, wrong=()):
        """Data sets, candidate-0001 and its delivery-0001 (predictions of `wrong` ids wrong)."""
        self.sets()
        self.candidate()
        self.delivery(wrong=wrong)
        return self.ws

    def plan(self, module="data"):
        issue = {"issue": "misses short messages", "evidence": "slice auc 0.55",
                 "attribution": ATTRIBUTION[module], "module": module, "change": "fix it",
                 "expected": {"metric": "auc", "direction": "up", "size": 0.02}}
        return self.ws.register("plan", self.mk("plan", issues=[issue], entry_module=module))

    def first_round(self, module="data", wrong=()):
        """Round 1: evaluation, promote to V1, plan entering `module`. V1's evalset predictions
        of the `wrong` ids are wrong (perfect by default: auc 1)."""
        self.ready(wrong)
        self.ws.register("evaluation", self.eval_src(wrong=wrong))
        self.assertEqual(self.ws.promote("evaluation-0001")[:2], ("promote", "V1"))
        return self.plan(module)

    def second_round_evaluation(self, wrong=(EVAL[0],), best_wrong=()):
        """Round 2: candidate-0002, delivery-0002, an evaluation against V1 (whose evalset
        predictions of `best_wrong` are wrong). By default the new delivery gets one row wrong
        that V1 gets right: auc 0.75 against 1, so the rule gives keep."""
        self.candidate()
        self.delivery(candidate_id="candidate-0002", wrong=wrong)
        return self.ws.register("evaluation", self.eval_src(
            delivery_id="delivery-0002", candidate_id="candidate-0002", wrong=wrong, best_wrong=best_wrong,
            best_before={"version": "V1", "evaluation_id": "evaluation-0001"}))

    def raises(self, fn, *fragments):
        with self.assertRaises(RsiError) as cm:
            fn()
        msg = str(cm.exception)
        for frag in fragments:
            self.assertIn(frag, msg)
        return msg


class TestRecommendationCheck(LoopCase):
    def test_evaluation_recommendation_must_match_rule(self):
        # A candidate that failed its train gate is never delivered (R28), so the evaluated
        # candidate of the first round passed it: the rule says promote.
        self.ready()
        keep = {"decision": "keep", "reason": "not sure yet"}
        msg = self.refused("evaluation", self.eval_src(recommendation=keep), "recommendation", "promote",
                           "train gate")
        self.assertIn("'keep'", msg)
        self.assertEqual(self.ws.register("evaluation", self.eval_src()), "evaluation-0001")
        self.assertEqual(self.ws.promote("evaluation-0001")[:2], ("promote", "V1"))
        self.plan()
        self.ws.next_round()
        # The new candidate predicts the evalset as well as V1: delta 0 is not above min_delta 0.01,
        # so the rule says keep.
        self.candidate()
        self.delivery(candidate_id="candidate-0002")
        fields = dict(delivery_id="delivery-0002", candidate_id="candidate-0002",
                      best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
                      recommendation={"decision": "promote", "reason": "looks better"})
        self.refused("evaluation", self.eval_src(**fields), "recommendation", "keep", "min_delta")
        fields["recommendation"] = {"decision": "keep", "reason": "gain below min_delta"}
        self.assertEqual(self.ws.register("evaluation", self.eval_src(**fields)), "evaluation-0002")


class TestPromote(LoopCase):
    def test_promote_without_evaluation_refused(self):
        self.ready()
        before = self.state_on_disk()
        self.raises(lambda: self.ws.promote("evaluation-0001"), "evaluation-0001", "not registered",
                    "rsi register")
        self.raises(lambda: self.ws.promote("candidate-0001"), "candidate-0001", "not an evaluation")
        r = run_rsi("promote", self.wsdir, "evaluation-0001")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("rsi register", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        self.ws.register("evaluation", self.eval_src())
        r = run_rsi("promote", self.wsdir, "evaluation-0001")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("promote V1"), r.stdout)

    def test_promote_records_version(self):
        self.ready()
        self.ws.register("evaluation", self.eval_src())
        decision, version, reason = self.ws.promote("evaluation-0001")
        self.assertEqual((decision, version), ("promote", "V1"))
        s = Workspace(self.wsdir).state
        self.assertEqual(s["versions"], [{"version": "V1", "candidate_id": "candidate-0001",
                                          "delivery_id": "delivery-0001",
                                          "evaluation_id": "evaluation-0001", "round": 1}])
        self.assertEqual(s["best"], "V1")
        self.assertEqual(s["no_promotion_streak"], 0)
        d = s["rounds"]["1"]["decision"]
        self.assertEqual((d["decision"], d["version"], d["evaluation_id"]),
                         ("promote", "V1", "evaluation-0001"))
        self.assertEqual(d["reason"], reason)
        self.assertIn("decision promote V1 from evaluation-0001", run_rsi("status", self.wsdir).stdout)
        # Once per round.
        self.ws.register("evaluation", self.eval_src(best_before={"version": "V1",
                                                                  "evaluation_id": "evaluation-0001"}))
        self.raises(lambda: self.ws.promote("evaluation-0002"), "round 1", "already")
        self.raises(lambda: self.ws.promote("evaluation-0001"), "round 1", "already")
        self.assertEqual(len(Workspace(self.wsdir).state["versions"]), 1)

    def test_keep_increments_streak_and_second_promotion_adds_version(self):
        # V1 gets evalset row q20 wrong (auc 0.75); round 2 also q21 (auc 0.5): keep;
        # round 3 gets every row right (auc 1): promote.
        self.first_round(wrong=(20,))
        self.ws.next_round()
        self.raises(lambda: self.ws.promote("evaluation-0001"), "evaluation-0001", "round 1", "round 2")
        self.second_round_evaluation(wrong=(20, 21), best_wrong=(20,))
        r = run_rsi("promote", self.wsdir, "evaluation-0002")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("keep"), r.stdout)
        ws = Workspace(self.wsdir)
        s = ws.state
        self.assertEqual((s["best"], len(s["versions"]), s["no_promotion_streak"]), ("V1", 1, 1))
        self.assertEqual(s["rounds"]["2"]["decision"]["decision"], "keep")
        self.assertIsNone(s["rounds"]["2"]["decision"]["version"])
        # Round 3 promotes: V2 becomes best and the streak resets.
        self.ws = ws
        self.plan()
        self.ws.next_round()
        self.candidate()
        self.delivery(candidate_id="candidate-0003")
        self.ws.register("evaluation", self.eval_src(
            delivery_id="delivery-0003", candidate_id="candidate-0003", best_wrong=(20,),
            best_before={"version": "V1", "evaluation_id": "evaluation-0001"}))
        self.assertEqual(self.ws.promote("evaluation-0003")[:2], ("promote", "V2"))
        s = Workspace(self.wsdir).state
        self.assertEqual((s["best"], s["no_promotion_streak"]), ("V2", 0))
        self.assertEqual(s["versions"][-1], {"version": "V2", "candidate_id": "candidate-0003",
                                             "delivery_id": "delivery-0003",
                                             "evaluation_id": "evaluation-0003", "round": 3})

    def test_evaluation_on_stale_evalset_refused(self):
        self.ready()
        self.ws.register("evaluation", self.eval_src())
        self.approve("refresh_sets")
        self.ws.register("evalset", self.mk("evalset", ids=range(60, 64)))
        # Registering an evaluation of the old evalset is refused ...
        self.refused("evaluation", self.eval_src(), "evalset-0001", "evalset-0002")
        # ... and so is promoting on one registered before the refresh.
        self.raises(lambda: self.ws.promote("evaluation-0001"), "evalset-0001", "evalset-0002")
        self.assertIsNone(Workspace(self.wsdir).state["rounds"]["1"]["decision"])


class TestApprove(LoopCase):
    def test_approve_records_user_words(self):
        self.init()
        self.assertEqual(set(APPROVAL_ITEMS), {"task_change", "over_budget", "use_testset",
                                               "refresh_sets", "plan", "delete"})
        a = self.ws.approve("over_budget", by="alice", quote="OK, two more GPU hours")
        self.assertEqual({k: a[k] for k in ("what", "ref", "by", "quote", "round")},
                         {"what": "over_budget", "ref": None, "by": "alice",
                          "quote": "OK, two more GPU hours", "round": 1})
        self.assertTrue(a["at"])
        r = run_rsi("approve", self.wsdir, "delete", "--by", "alice", "--quote", "delete old checkpoints",
                    "--ref", "candidate-0001")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("delete", r.stdout)
        s = Workspace(self.wsdir).state
        self.assertEqual([(a["what"], a["ref"]) for a in s["approvals"]],
                         [("over_budget", None), ("delete", "candidate-0001")])
        # --ref is optional for items other than plan.
        r = run_rsi("approve", self.wsdir, "refresh_sets", "--by", "alice", "--quote", "refresh the valset")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(Workspace(self.wsdir).state["approvals"][-1]["what"], "refresh_sets")

    def test_approve_refusals(self):
        self.first_round()
        self.raises(lambda: self.ws.approve("bogus", by="alice", quote="yes"), "bogus")
        self.raises(lambda: self.ws.approve("over_budget", by="alice", quote="  "), "quote")
        self.raises(lambda: self.ws.approve("over_budget", by="", quote="yes"), "by")
        self.raises(lambda: self.ws.approve("plan", by="alice", quote="yes"), "--ref")
        self.raises(lambda: self.ws.approve("plan", by="alice", quote="yes", ref="plan-0009"), "plan-0009")
        self.raises(lambda: self.ws.approve("plan", by="alice", quote="yes", ref="evaluation-0001"),
                    "evaluation-0001")
        self.assertEqual(Workspace(self.wsdir).state["approvals"], [])
        r = run_rsi("approve", self.wsdir, "plan", "--by", "alice", "--ref", "plan-0001")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--quote", r.stderr)
        r = run_rsi("approve", self.wsdir, "bogus", "--by", "alice", "--quote", "yes")
        self.assertNotEqual(r.returncode, 0)
        # A plan from an earlier round cannot be approved in this one.
        self.ws.approve("plan", by="alice", quote="go", ref="plan-0001")
        self.ws.next_round()
        self.raises(lambda: self.ws.approve("plan", by="alice", quote="yes", ref="plan-0001"),
                    "plan-0001", "round 1")


class TestNext(LoopCase):
    def setUp(self):
        LoopCase.setUp(self)
        self.task["approvals"].append("plan")

    def test_next_without_approved_plan_refused(self):
        self.ready()
        before = self.state_on_disk()
        msg = self.raises(lambda: self.ws.next_round(), "rsi promote", "plan")
        self.assertEqual(self.state_on_disk(), before)
        self.ws.register("evaluation", self.eval_src())
        self.ws.promote("evaluation-0001")
        self.raises(lambda: self.ws.next_round(), "no plan", "rsi register")
        self.plan("train")
        msg = self.raises(lambda: self.ws.next_round(), "plan-0001", "rsi approve")
        self.assertNotIn("rsi promote", msg)
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("plan-0001", r.stderr)
        self.assertEqual(Workspace(self.wsdir).state["round"], 1)
        # An approval of some other item does not approve the plan.
        self.ws.approve("over_budget", by="alice", quote="fine")
        self.raises(lambda: self.ws.next_round(), "plan-0001")
        r = run_rsi("approve", self.wsdir, "plan", "--ref", "plan-0001", "--by", "alice",
                    "--quote", "go with the train plan")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "round 2: train")
        s = Workspace(self.wsdir).state
        self.assertEqual(s["round"], 2)
        self.assertIn("opened_at", s["rounds"]["2"])
        self.assertEqual({k: v for k, v in s["rounds"]["2"].items() if k != "opened_at"},
                         {"entry": "train", "artifacts": [], "decision": None, "plan_id": None})
        self.assertIsNone(s["stopped"])

    def test_next_stops_and_run_is_frozen(self):
        self.task["stop"]["patience"] = 1
        self.first_round()
        self.ws.approve("plan", by="alice", quote="go", ref="plan-0001")
        self.assertEqual(self.ws.next_round(), (2, "data"))
        self.second_round_evaluation()
        self.assertEqual(self.ws.promote("evaluation-0002")[0], "keep")
        self.plan()
        # The stop check runs first: no plan approval is needed to stop.
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("stop: "), r.stdout)
        self.assertIn("patience", r.stdout)
        s = Workspace(self.wsdir).state
        self.assertEqual(s["round"], 2)
        self.assertIn("patience", s["stopped"]["reason"])
        self.assertTrue(s["stopped"]["at"])
        self.assertIn("stopped: ", run_rsi("status", self.wsdir).stdout)
        self.assertNotIn("stopped: no", run_rsi("status", self.wsdir).stdout)
        # Once stopped, nothing moves.
        ws = Workspace(self.wsdir)
        self.ws = ws
        self.refused("trainset", self.mk("trainset", ids=range(40, 44)), "stopped")
        self.raises(lambda: ws.promote("evaluation-0002"), "stopped")
        self.raises(lambda: ws.approve("plan", by="alice", quote="go", ref="plan-0002"), "stopped")
        self.raises(lambda: ws.next_round(), "stopped")
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 1)
        self.assertIn("stopped", r.stderr)
        r = run_rsi("check-stop", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("stop: "), r.stdout)

    def test_next_refuses_undecided_evaluation_before_budget_stop(self):
        # R19: a budget stop cannot pre-empt judging a registered evaluation.
        self.task["budget"]["compute_hours"] = 1.5
        self.task["approvals"].remove("over_budget")
        self.ready()  # candidate-0001 uses the whole budget
        self.ws.register("evaluation", self.eval_src())
        self.assertTrue(self.ws.stop_check()[0])
        before = self.state_on_disk()
        self.raises(lambda: self.ws.next_round(), "evaluation-0001", "rsi promote")
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("rsi promote", r.stderr)
        self.assertIn("evaluation-0001", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        self.assertIsNone(Workspace(self.wsdir).state["stopped"])
        # Once the evaluation is judged, next stops on the budget.
        self.assertEqual(self.ws.promote("evaluation-0001")[:2], ("promote", "V1"))
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("stop: "), r.stdout)
        self.assertIn("compute hours", r.stdout)
        self.assertIn("compute hours", Workspace(self.wsdir).state["stopped"]["reason"])

    def test_after_stop_delete_approval_and_records_allowed(self):
        # R18: cleanup approvals and final experiment records stay possible after a stop.
        self.task["stop"]["max_rounds"] = 1
        self.first_round()
        r = run_rsi("next", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("stop: "), r.stdout)
        self.assertIn("max_rounds", r.stdout)
        r = run_rsi("approve", self.wsdir, "delete", "--by", "alice", "--quote", "delete the checkpoints",
                    "--ref", "candidate-0001")
        self.assertEqual(r.returncode, 0, r.stderr)
        ws = Workspace(self.wsdir)
        self.ws = ws
        self.assertEqual(ws.state["approvals"][-1]["what"], "delete")
        self.assertEqual(ws.register("record", self.mk("record", artifacts=["evaluation-0001"])),
                         "record-0001")
        r = run_rsi("register", self.wsdir, "record", self.mk("record", artifacts=["candidate-0001"]))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "record-0002")
        # Every other approval and registration stays refused.
        ws = Workspace(self.wsdir)
        self.ws = ws
        for what in ("over_budget", "task_change", "use_testset", "refresh_sets"):
            self.raises(lambda: ws.approve(what, by="alice", quote="yes"), "stopped")
        r = run_rsi("approve", self.wsdir, "over_budget", "--by", "alice", "--quote", "yes")
        self.assertEqual(r.returncode, 1)
        self.assertIn("stopped", r.stderr)
        self.refused("plan", self.mk("plan"), "stopped")
        self.assertEqual([a["what"] for a in Workspace(self.wsdir).state["approvals"]], ["delete"])


class TestCheckStopCommand(LoopCase):
    def test_check_stop_on_target_does_not_change_state(self):
        self.task["stop"]["target"] = 0.85
        self.ready()
        self.ws.register("evaluation", self.eval_src())
        before = self.state_on_disk()
        r = run_rsi("check-stop", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("continue: "), r.stdout)
        self.assertEqual(self.state_on_disk(), before)
        self.ws.promote("evaluation-0001")
        self.assertEqual(self.ws.best_primary(), 1.0)
        before = self.state_on_disk()
        r = run_rsi("check-stop", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("stop: "), r.stdout)
        self.assertIn("target", r.stdout)
        self.assertEqual(self.state_on_disk(), before)
        self.assertIsNone(Workspace(self.wsdir).state["stopped"])


class TestMilestoneEvaluation(LoopCase):
    """R27: an evaluation with uses_testset is a milestone evaluation of the best version."""

    BEST_BEFORE = {"version": "V1", "evaluation_id": "evaluation-0001"}

    def milestone_delivery(self, candidate_id="candidate-0001"):
        """Approve use_testset and register a delivery with evalset and testset predictions."""
        self.approve("use_testset")
        return self.ws.register("delivery", add_predictions(
            self.mk("delivery", ids=EVAL, candidate_id=candidate_id),
            self.task, "predictions_test", TEST, testset_id="testset-0001"))

    def milestone_src(self, **fields):
        fields.setdefault("delivery_id", "delivery-0002")
        fields.setdefault("candidate_id", "candidate-0001")
        fields.setdefault("best_before", self.BEST_BEFORE)
        fields.setdefault("uses_testset", True)
        fields.setdefault("ids", EVAL)
        fields.setdefault("test_ids", TEST)
        fields.setdefault("recommendation", {"decision": "keep", "reason": "milestone release"})
        return self.mk("evaluation", **fields)

    def test_milestone_scores_the_best_version_without_using_the_evalset(self):
        self.first_round()  # V1 is candidate-0001; evaluation-0001 used the evalset once
        # Fill the evalset's use limit with ordinary evaluations of round 1.
        for _ in range(self.task["eval_use_limit"] - 1):
            self.ws.register("evaluation", self.eval_src(best_before=self.BEST_BEFORE))
        limit = self.task["eval_use_limit"]
        self.assertEqual(self.ws.state["eval_use_count"], limit)
        self.refused("evaluation", self.eval_src(best_before=self.BEST_BEFORE), "eval_use_limit")
        self.milestone_delivery()  # delivery-0002
        self.approve("use_testset")
        # A milestone evaluation does not check or count evalset uses.
        self.assertEqual(self.ws.register("evaluation", self.milestone_src()), "evaluation-0004")
        self.assertEqual(Workspace(self.wsdir).state["eval_use_count"], limit)

    def test_milestone_needs_best_version_candidate_and_keep(self):
        self.ready()
        self.ws.register("evaluation", self.eval_src())  # evaluation-0001, not yet decided
        self.milestone_delivery()  # delivery-0002
        self.approve("use_testset")
        self.refused("evaluation", self.milestone_src(best_before=None), "no best version")
        self.ws.promote("evaluation-0001")  # V1 is candidate-0001
        self.candidate()  # candidate-0002
        self.milestone_delivery("candidate-0002")  # delivery-0003
        self.approve("use_testset")
        self.refused("evaluation", self.milestone_src(delivery_id="delivery-0003", candidate_id="candidate-0002"),
                     "candidate-0002", "V1", "candidate-0001")
        promote = {"decision": "promote", "reason": "looks better"}
        self.refused("evaluation", self.milestone_src(recommendation=promote),
                     "recommendation.decision", "keep", "milestone")
        self.assertEqual(self.ws.register("evaluation", self.milestone_src()), "evaluation-0002")

    def test_promote_refuses_milestone_and_next_ignores_it(self):
        self.task["budget"]["compute_hours"] = 3.0
        self.task["approvals"].remove("over_budget")
        self.first_round()
        self.assertEqual(self.ws.next_round(), (2, "data"))
        self.candidate()  # candidate-0002 uses the rest of the budget
        self.milestone_delivery()  # delivery-0002 of V1's candidate-0001
        self.approve("use_testset")
        self.assertEqual(self.ws.register("evaluation", self.milestone_src()), "evaluation-0002")
        before = self.state_on_disk()
        self.raises(lambda: self.ws.promote("evaluation-0002"), "evaluation-0002",
                    "milestone evaluations are not promotion decisions")
        r = run_rsi("promote", self.wsdir, "evaluation-0002")
        self.assertEqual(r.returncode, 1)
        self.assertIn("milestone evaluations are not promotion decisions", r.stderr)
        self.assertEqual(self.state_on_disk(), before)
        # The round has only a milestone evaluation: next does not ask for `rsi promote`.
        n, reason = self.ws.next_round()
        self.assertIsNone(n)
        self.assertIn("compute hours", reason)


class TestRegistrationGuards(LoopCase):
    def test_registration_round_must_match_state(self):
        self.init()
        self.refused("testset", self.mk("testset", ids=range(30, 34), round=2), "round is 2", "round 1")
        self.assertEqual(self.ws.register("testset", self.mk("testset", ids=range(30, 34))), "testset-0001")

    def test_over_budget_needs_approval(self):
        self.task["budget"]["compute_hours"] = 2.0
        self.sets()
        self.assertEqual(self.candidate(), "candidate-0001")  # 1.5 of 2.0
        self.refused("candidate", self.mk("candidate", ids=VAL), "over_budget", "compute")
        self.assertEqual(self.ws.state["budget_used"]["compute_hours"], 3.0)  # B1: the refused training counts too
        self.ws.approve("over_budget", by="alice", quote="spend up to 5 hours")
        self.assertEqual(self.candidate(), "candidate-0002")
        self.assertEqual(self.candidate(), "candidate-0003")  # not single-use
        self.assertNotIn("used_by", self.ws.state["approvals"][0])
        # over_budget not among the task's approvals: no approval needed.
        shutil.rmtree(self.wsdir)
        self.task["approvals"].remove("over_budget")
        self.sets()
        self.candidate()
        self.assertEqual(self.candidate(), "candidate-0002")


if __name__ == "__main__":
    unittest.main()
