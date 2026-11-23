"""Registration recomputes the reported numbers with rsi.metrics and refuses mismatches (Task 13, R36)."""
import json
import os

from fixtures import add_predictions, choice_task, label_row, input_row, predictions, update_manifest
from rsi import metrics as M
from rsi.workspace import Workspace, unverified_names
from test_rules import LoopCase
from test_workspace import EVAL, TEST, VAL, read_registry, run_rsi

BEST_BEFORE = {"version": "V1", "evaluation_id": "evaluation-0001"}


def manifest(src):
    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


class VerifyCase(LoopCase):
    def on_disk(self):
        return self.state_on_disk(), read_registry(self.wsdir)

    def refused_atomic(self, kind, src, *fragments):
        """Refused with every fragment in the message; STATE.json and the registry unchanged."""
        before = self.on_disk()
        msg = self.refused(kind, src, *fragments)
        self.assertEqual(self.on_disk(), before)
        return msg


class TestCandidateNumbers(VerifyCase):
    def test_wrong_metrics_val_refused(self):
        self.sets()
        msg = self.refused_atomic("candidate", self.mk("candidate", ids=VAL, metrics_val={"auc": 0.8}),
                                  "metrics_val.auc", "reported 0.8", "recomputed 1.0", "valset-0001", "rsi score")
        self.assertIn("--labels", msg)  # the error gives the `rsi score` form to use
        log_loss = M.log_loss(predictions(self.task, VAL), [label_row(i, self.task) for i in VAL], self.task)
        self.refused_atomic("candidate", self.mk("candidate", ids=VAL, metrics_val={"auc": 1.0,
                                                                                  "log_loss": log_loss + 1e-8}),
                            "metrics_val.log_loss")
        # Within the 1e-9 tolerance it registers.
        src = self.mk("candidate", ids=VAL, metrics_val={"auc": 1.0 - 5e-10, "log_loss": log_loss})
        self.assertEqual(self.ws.register("candidate", src), "candidate-0001")

    def test_wrong_baseline_refused(self):
        self.sets()
        self.refused_atomic("candidate", self.mk("candidate", ids=VAL, baselines_val={"majority": {"auc": 0.4}}),
                            "baselines_val.majority.auc", "reported 0.4", "recomputed 0.5", "trainset-0001",
                            "rsi score")

    def test_baselines_come_from_the_candidates_trainset(self):
        self.sets()
        # trainset-0002 holds three yes rows and one no: its prior differs from trainset-0001's.
        skewed = [40, 42, 44, 41]
        self.ws.register("trainset", self.mk("trainset", ids=skewed))
        self.refused_atomic("candidate", self.mk("candidate", ids=VAL, trainset_id="trainset-0002"),
                            "baselines_val.majority.log_loss", "trainset-0002")
        self.assertEqual(self.candidate(trainset_id="trainset-0002", train_ids=skewed), "candidate-0001")

    def test_train_gate_must_match_the_numbers(self):
        self.sets()
        # Every prediction wrong: auc 0 does not beat the majority baseline's 0.5.
        msg = self.refused_atomic("candidate", self.mk("candidate", ids=VAL, wrong=VAL,
                                                      train_gate={"passed": True, "detail": "looks fine"}),
                                  "train_gate.passed is true", "majority", "rsi score")
        self.assertIn("auc 0.0", msg)
        self.refused_atomic("candidate", self.mk("candidate", ids=VAL,
                                                 train_gate={"passed": False, "detail": "not sure"}),
                            "train_gate.passed is false", "beats every baseline")
        self.assertEqual(self.candidate(wrong=VAL, train_gate={"passed": False, "detail": "below"}),
                         "candidate-0001")
        # Equal to the baseline is not beating it (strictly): half the rows wrong gives auc 0.5.
        self.assertEqual(self.candidate(wrong=[10, 11]), "candidate-0002")
        self.assertFalse(self.ws.manifest("candidate-0002")["train_gate"]["passed"])
        self.assertEqual(self.ws.manifest("candidate-0002")["metrics_val"]["auc"], 0.5)

    def test_unknown_names_are_registered_and_reported(self):
        self.sets()
        src = self.mk("candidate", ids=VAL)
        m = manifest(src)
        m["metrics_val"]["recall"] = 0.7
        m["baselines_val"]["category-prior"] = {"auc": 0.6}
        update_manifest(src, metrics_val=m["metrics_val"], baselines_val=m["baselines_val"])
        r = run_rsi("register", self.wsdir, "candidate", src)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "candidate-0001 (train gate: passed; val auc 1) "
                                           "(not verified: recall, category-prior)")
        # Everything known: no note.
        r = run_rsi("register", self.wsdir, "candidate", self.mk("candidate", ids=VAL))
        self.assertEqual(r.stdout.strip(), "candidate-0002 (train gate: passed; val auc 1)", r.stderr)

    def test_gate_with_a_baseline_rsi_does_not_compute(self):
        self.task["metrics"]["baselines"] = ["majority", "category-prior"]
        self.sets()
        # Beating majority but failing the unknown baseline is possible: passed false is accepted.
        r = run_rsi("register", self.wsdir, "candidate",
                    self.mk("candidate", ids=VAL, train_gate={"passed": False, "detail": "below category-prior"}))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(not verified: category-prior)", r.stdout)
        # Passing while below majority is not.
        self.refused_atomic("candidate", self.mk("candidate", ids=VAL, wrong=VAL,
                                                 train_gate={"passed": True, "detail": "ok"}),
                            "train_gate.passed is true", "majority")

    def test_choice_task_metrics(self):
        self.task = choice_task()
        self.task["metrics"]["others"] = ["log_loss", "macro_f1"]
        self.task["metrics"]["baselines"] = ["majority", "group_prior"]
        self.task["promotion_rule"]["guards"] = []
        self.sets()
        src = self.mk("candidate", ids=VAL, wrong=[10])
        m = manifest(src)
        self.assertEqual(sorted(m["metrics_val"]), ["accuracy", "log_loss", "macro_f1"])
        self.assertEqual(sorted(m["baselines_val"]), ["group_prior", "majority"])
        bad = dict(m["metrics_val"], macro_f1=m["metrics_val"]["macro_f1"] + 0.01)
        self.refused_atomic("candidate", self.mk("candidate", ids=VAL, wrong=[10], metrics_val=bad),
                            "metrics_val.macro_f1")
        bad = dict(m["baselines_val"], group_prior={"accuracy": 0.1})
        self.refused_atomic("candidate", self.mk("candidate", ids=VAL, wrong=[10], baselines_val=bad),
                            "baselines_val.group_prior.accuracy")
        self.assertEqual(self.ws.register("candidate", src), "candidate-0001")


class TestEvaluationNumbers(VerifyCase):
    def test_wrong_candidate_metrics_refused(self):
        self.ready()
        src = self.eval_src()
        m = manifest(src)
        m["metrics"]["candidate"]["auc"] = 0.9
        msg = self.refused_atomic("evaluation", update_manifest(src, metrics=m["metrics"]),
                                  "metrics.candidate.auc", "reported 0.9", "recomputed 1.0", "delivery-0001",
                                  "evalset-0001", "rsi score")
        self.assertIn("--clusters", msg)
        self.assertEqual(self.ws.state["eval_use_count"], 0)
        self.assertEqual(self.ws.register("evaluation", self.eval_src()), "evaluation-0001")

    def test_best_version_numbers_must_be_present(self):
        # R41: with a best version (best_before set) an evaluation reports metrics.best with the primary
        # metric and deltas[primary]; registration refuses it otherwise, before any number is compared.
        self.first_round()
        self.ws.next_round()
        self.candidate()
        self.delivery(candidate_id="candidate-0002")
        fields = dict(delivery_id="delivery-0002", candidate_id="candidate-0002", best_before=BEST_BEFORE)
        src = self.eval_src(**fields)
        m = manifest(src)
        self.refused_atomic("evaluation", update_manifest(src, metrics=dict(m["metrics"], best=None)),
                            "metrics.best", "best_before")
        src = self.eval_src(**fields)
        self.refused_atomic("evaluation", update_manifest(src, deltas={}), "deltas", "'auc'")
        src = self.eval_src(**fields)
        best = dict(manifest(src)["metrics"]["best"])
        del best["auc"]
        self.refused_atomic("evaluation", update_manifest(src, metrics=dict(m["metrics"], best=best)),
                            "metrics.best", "'auc'")
        self.assertEqual(self.ws.register("evaluation", self.eval_src(**fields)), "evaluation-0002")

    def test_wrong_baselines_refused(self):
        self.ready()
        src = self.eval_src()
        m = manifest(src)
        m["metrics"]["baselines"]["majority"]["auc"] = 0.6
        self.refused_atomic("evaluation", update_manifest(src, metrics=m["metrics"]),
                            "metrics.baselines.majority.auc", "recomputed 0.5", "trainset-0001")

    def test_group_prior_takes_groups_from_evalset_inputs(self):
        # Evalset label rows carry no meta.group: registration takes it from the evalset inputs.
        self.task = choice_task()
        self.task["metrics"]["baselines"] = ["majority", "group_prior"]
        self.task["promotion_rule"]["guards"] = []
        self.ready()
        src = self.eval_src()
        m = manifest(src)
        # Both groups' training priors pick allow; one of the four evalset rows is allow.
        self.assertEqual(m["metrics"]["baselines"]["group_prior"]["accuracy"], 0.25)
        m["metrics"]["baselines"]["group_prior"]["accuracy"] = 0.5
        self.refused_atomic("evaluation", self.eval_src(metrics=m["metrics"]),
                            "metrics.baselines.group_prior.accuracy", "recomputed 0.25")
        self.assertEqual(self.ws.register("evaluation", src), "evaluation-0001")

    def second_round(self, wrong=(20,)):
        """V1 (perfect on the evalset) and round 2's candidate-0002 / delivery-0002 (`wrong` ids wrong)."""
        self.first_round()
        self.ws.next_round()
        self.candidate()
        self.delivery(candidate_id="candidate-0002", wrong=wrong)
        return dict(delivery_id="delivery-0002", candidate_id="candidate-0002", wrong=wrong,
                    best_before=BEST_BEFORE)

    def test_wrong_best_and_deltas_refused(self):
        fields = self.second_round()
        src = self.eval_src(**fields)
        good = manifest(src)
        self.assertEqual(good["recommendation"]["decision"], "keep")
        for path, fragment in ((("deltas", "auc", "ci_low"), "deltas.auc.ci_low"),
                               (("deltas", "auc", "ci_high"), "deltas.auc.ci_high"),
                               (("deltas", "auc", "delta"), "deltas.auc.delta"),
                               (("metrics", "best", "log_loss"), "metrics.best.log_loss")):
            m = manifest(src)
            obj = m
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] += 0.01
            bad = self.mk("evaluation", ids=EVAL, **dict(fields, metrics=m["metrics"], deltas=m["deltas"],
                                                        recommendation=good["recommendation"]))
            msg = self.refused_atomic("evaluation", bad, fragment, "rsi score")
            if path[0] == "deltas":
                self.assertIn("delivery-0001", msg)  # the best version's predictions
                self.assertIn("--compare", msg)
        self.assertEqual(self.ws.register("evaluation", src), "evaluation-0002")
        d = self.ws.manifest("evaluation-0002")["deltas"]["auc"]
        self.assertEqual(d["delta"], 0.75 - 1.0)
        self.assertLessEqual(d["ci_low"], d["delta"])

    def test_deltas_use_the_tasks_ci_settings(self):
        self.task["metrics"]["ci"] = {"resamples": 300, "seed": 1, "level": 0.5}
        fields = self.second_round()
        # Computed with the defaults instead of the task's settings: refused.
        labels = [label_row(i, self.task) for i in EVAL]
        clusters = M.clusters_of([input_row(i, self.task) for i in EVAL])
        a, b = predictions(self.task, EVAL, fields["wrong"]), predictions(self.task, EVAL)
        default = M.paired_delta("auc", a, b, labels, self.task, clusters, **M.CI_DEFAULTS)
        tasks = M.paired_delta("auc", a, b, labels, self.task, clusters, 300, 1, 0.5)
        self.assertNotEqual(default, tasks)
        self.refused_atomic("evaluation", self.eval_src(deltas={"auc": default}, **fields), "deltas.auc.ci_")
        src = self.eval_src(**fields)
        self.assertEqual(manifest(src)["deltas"]["auc"], tasks)
        self.assertEqual(self.ws.register("evaluation", src), "evaluation-0002")

    def test_best_version_delivery_after_refresh(self):
        self.first_round()  # V1: candidate-0001, delivery-0001 on evalset-0001
        self.ws.next_round()
        self.approve("refresh_sets")
        new = range(60, 64)
        self.ws.register("evalset", self.mk("evalset", ids=new))  # evalset-0002
        self.candidate()  # candidate-0002
        self.delivery(candidate_id="candidate-0002", evalset_id="evalset-0002", ids=new)  # delivery-0002
        fields = dict(delivery_id="delivery-0002", candidate_id="candidate-0002", evalset_id="evalset-0002",
                      ids=new, best_before=BEST_BEFORE)
        msg = self.refused_atomic("evaluation", self.eval_src(**fields), "metrics.best",
                                  "deliver the best version on the current evalset first", "parents")
        self.assertIn("delivery-0001", msg)
        # A parent delivery of another candidate does not count.
        self.refused_atomic("evaluation", self.eval_src(parents=["delivery-0002"], **fields),
                            "deliver the best version on the current evalset first")
        # V1's candidate delivered on the new evalset (one row wrong), listed in parents.
        self.delivery(candidate_id="candidate-0001", evalset_id="evalset-0002", ids=new, wrong=[60])
        self.refused_atomic("evaluation", self.eval_src(parents=["delivery-0003"], **fields), "metrics.best.auc")
        src = self.eval_src(parents=["delivery-0003"], best_wrong=[60], **fields)
        self.assertEqual(manifest(src)["recommendation"]["decision"], "promote")
        self.assertEqual(self.ws.register("evaluation", src), "evaluation-0002")
        self.assertEqual(self.ws.promote("evaluation-0002")[:2], ("promote", "V2"))

    def test_milestone_metrics_test_refused(self):
        self.first_round()
        self.approve("use_testset")
        self.ws.register("delivery", add_predictions(
            self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
            self.task, "predictions_test", TEST, wrong=[30], testset_id="testset-0001"))  # delivery-0002
        self.approve("use_testset")
        fields = dict(delivery_id="delivery-0002", candidate_id="candidate-0001", uses_testset=True,
                      best_before=BEST_BEFORE, test_ids=TEST)
        self.refused_atomic("evaluation", self.eval_src(**fields), "metrics_test.auc", "testset-0001",
                            "rsi score")
        self.assertIsNone(self.ws.state["approvals"][-1].get("used_by"))
        src = self.eval_src(wrong=[30], **fields)
        self.assertEqual(manifest(src)["metrics_test"]["auc"], 0.75)
        self.assertEqual(self.ws.register("evaluation", src), "evaluation-0002")

    def test_unknown_metric_registered_with_note(self):
        self.ready()
        src = self.eval_src()
        m = manifest(src)
        m["metrics"]["candidate"]["recall"] = 0.7
        update_manifest(src, metrics=m["metrics"])
        r = run_rsi("register", self.wsdir, "evaluation", src)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "evaluation-0001 (not verified: recall)")
        self.assertEqual(Workspace(self.wsdir).state["eval_use_count"], 1)


class TestUnknownPrimary(VerifyCase):
    def test_unknown_primary_listed_once(self):
        # R46: an unknown primary is not verified: its metrics.candidate value always, its delta only when one
        # is expected (best_before set). Either way the note names it once.
        self.task["metrics"]["primary"] = "expected_cost"
        self.task["metrics"]["higher_is_better"] = False
        self.first_round()
        self.assertEqual(unverified_names("evaluation", self.ws.manifest("evaluation-0001"), self.task),
                         ["expected_cost"])
        self.ws.next_round()
        self.candidate()
        self.delivery(candidate_id="candidate-0002")
        src = self.eval_src(delivery_id="delivery-0002", candidate_id="candidate-0002", best_before=BEST_BEFORE)
        m = manifest(src)
        self.assertIn("expected_cost", m["deltas"])
        self.assertEqual(unverified_names("evaluation", m, self.task), ["expected_cost"])
        # Without best_before nothing about deltas is listed: with the primary absent from metrics.candidate and
        # metrics.best (not a valid evaluation, only to isolate the rule) the note is empty.
        m["best_before"], m["metrics"]["candidate"], m["metrics"]["best"] = None, {"log_loss": 0.1}, None
        self.assertEqual(unverified_names("evaluation", m, self.task), [])
        m["best_before"] = BEST_BEFORE
        self.assertEqual(unverified_names("evaluation", m, self.task), ["expected_cost"])


class TestScoreThenRegister(VerifyCase):
    """The module workflow: `rsi score` on the files, its output copied into the manifest, registered."""

    def score(self, *args):
        task = os.path.join(self.ws.path(self.ws.current("task")), "task.json")
        r = run_rsi("score", "--task", task, *args)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def art(self, aid, name):
        return os.path.join(self.ws.path(aid), name)

    def test_candidate_and_evaluation_from_score_output(self):
        self.first_round()
        self.ws.next_round()
        # train: a candidate whose numbers come from `rsi score` on its own predictions_val.
        src = self.mk("candidate", ids=VAL, wrong=[11], metrics_val={}, baselines_val={})
        out = self.score("--labels", self.art("valset-0001", "rows.jsonl"),
                         "--predictions", os.path.join(src, "predictions_val.jsonl"),
                         "--train", self.art("trainset-0001", "rows.jsonl"))
        update_manifest(src, metrics_val=out["metrics"], baselines_val=out["baselines"])
        self.assertEqual(self.ws.register("candidate", src), "candidate-0002")
        # deliver: the package reproduces the candidate's validation predictions (q11 wrong).
        self.delivery(candidate_id="candidate-0002", wrong=[21], val_wrong=[11])  # delivery-0002
        # judge: metrics, baselines and delta against V1's delivery-0001, then V1's own metrics.
        ev = self.art("evalset-0001", "labels.jsonl")
        out = self.score("--labels", ev, "--predictions", self.art("delivery-0002", "predictions_eval.jsonl"),
                         "--train", self.art("trainset-0001", "rows.jsonl"),
                         "--clusters", self.art("evalset-0001", "inputs.jsonl"),
                         "--compare", self.art("delivery-0001", "predictions_eval.jsonl"))
        best = self.score("--labels", ev, "--predictions", self.art("delivery-0001", "predictions_eval.jsonl"))
        src = self.eval_src(delivery_id="delivery-0002", candidate_id="candidate-0002", best_before=BEST_BEFORE,
                            metrics={"candidate": out["metrics"], "best": best["metrics"],
                                     "baselines": out["baselines"]},
                            deltas={"auc": out["delta"]})
        self.assertEqual(self.ws.register("evaluation", src), "evaluation-0002")
        self.assertEqual(self.ws.promote("evaluation-0002")[0], "keep")
