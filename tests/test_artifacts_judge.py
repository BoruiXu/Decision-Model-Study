import json
import os
import subprocess
import sys
import tempfile
import unittest

from fixtures import add_predictions, binary_task, choice_task, make_artifact, update_manifest
from rsi import artifacts, rules
from rsi.workspace import RsiError
from test_workspace import EVAL, TEST, VAL, WsCase

KINDS = ("evaluation", "plan", "record")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _manifest(d):
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _save(d, m):
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f)


class TestJudgeArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task = binary_task()

    def mk(self, kind, **fields):
        return make_artifact(kind, os.path.join(self.tmp.name, kind), self.task, **fields)

    def errs(self, kind, d):
        return artifacts.validate_artifact(kind, d, self.task)

    def edit(self, kind, fn):
        d = self.mk(kind)
        m = _manifest(d)
        fn(m)
        _save(d, m)
        return self.errs(kind, d)

    def test_valid_directories(self):
        for task in (binary_task(), choice_task()):
            self.task = task
            for kind in KINDS:
                d = self.mk(kind)
                self.assertEqual(artifacts.validate_artifact(kind, d, task), [], kind)
        self.task = binary_task()

        def no_best(m):
            m["best_before"] = None
            m["metrics"]["best"] = None
            m["deltas"] = {}

        def orchestrator_record(m):
            m["module"] = "orchestrator"
            m["result"]["before"] = None
            m["result"]["after"] = None
            m["automation"] = "automated"
        for name, kind, change in (
                ("evaluation without a best version", "evaluation", no_best),
                ("milestone evaluation (R27)", "evaluation",
                 lambda m: m.update(uses_testset=True, metrics_test={"auc": 0.83})),
                ("orchestrator record, null numbers", "record", orchestrator_record)):
            with self.subTest(name):
                self.assertEqual(self.edit(kind, change), [])

    def test_refusals(self):
        def drop(*path):
            def change(m):
                for k in path[:-1]:
                    m = m[k]
                del m[path[-1]]
            return change

        def put(value, *path):
            def change(m):
                for k in path[:-1]:
                    m = m[k]
                m[path[-1]] = value
            return change
        # (name, kind, change of a valid manifest, words one error must contain)
        cases = (
            ("evaluation without its report", "evaluation", lambda m: m["files"].clear(), ()),
            ("candidate metrics without the primary", "evaluation", drop("metrics", "candidate", "auc"), ("auc",)),
            ("a best version needs the primary delta", "evaluation", put({}, "deltas"), ("deltas",)),
            ("a best version needs its metrics", "evaluation", put(None, "metrics", "best"), ("metrics.best",)),
            ("delta without its interval", "evaluation", drop("deltas", "auc", "ci_low"), ()),
            ("bad decision", "evaluation", put("maybe", "recommendation", "decision"), ()),
            # R27: metrics_test is required when uses_testset is true, holds the primary metric, and is not
            # allowed otherwise.
            ("milestone without metrics_test", "evaluation", lambda m: m.update(uses_testset=True),
             ("metrics_test", "required")),
            ("metrics_test without the primary", "evaluation",
             lambda m: m.update(uses_testset=True, metrics_test={"log_loss": 0.4}), ("metrics_test", "auc")),
            ("metrics_test not a number", "evaluation",
             lambda m: m.update(uses_testset=True, metrics_test={"auc": "high"}), ("metrics_test",)),
            ("metrics_test without uses_testset", "evaluation", lambda m: m.update(metrics_test={"auc": 0.83}),
             ("metrics_test", "uses_testset")),
            ("failure attributed to another module", "evaluation", put("train", "failures", 0, "module"),
             ("data_coverage",)),
            ("issue attributed to another module", "plan", put("deliver", "issues", 0, "module"), ("data_coverage",)),
            ("unknown attribution", "evaluation", put("bad_luck", "failures", 0, "attribution"), ("bad_luck",)),
            ("entry module not in the issues", "plan", put("train", "entry_module"), ("entry_module",)),
            ("bad expected direction", "plan", put("sideways", "issues", 0, "expected", "direction"), ()),
            ("entry module not a module", "plan", put("judge", "entry_module"), ("not in enum",)),
            ("record conclusion", "record", put("great", "result", "conclusion"), ()),
            ("record module", "record", put("nobody", "module"), ()),
        )
        for name, kind, change, words in cases:
            with self.subTest(name):
                errs = self.edit(kind, change)
                self.assertTrue(errs)
                self.assertTrue(any(all(w in e for w in words) for e in errs), errs)


LR = {"baseline-0001": ("lr", (10, 20))}  # the round's lr baseline: one valset and one evalset row wrong


class TestBaselinesScored(WsCase):
    """The judge reports the baselines of the round's comparison on the evalset (baselines_eval) and, at the milestone,
    the round's test baselines on the testset (baselines_test); registration recomputes every number."""

    def ws_with_comparison_and_delivery(self):
        self.sets()
        self.candidate()
        self.ws.register("baseline", make_artifact("baseline", self.src("bl-lr"), self.task, ids=VAL, eval_ids=EVAL,
                                                   wrong=(10, 20)))
        self.ws.register("comparison", self.mk("comparison", ids=VAL, baselines={"lr": ("baseline-0001", (10, 20))}))
        self.delivery()
        return self.ws

    def ws_milestone_with_test_baseline(self):
        ws = self.ws_with_comparison_and_delivery()
        ws.register("evaluation", self.eval_src(baselines=LR))
        ws.promote("evaluation-0001")
        self.approve("use_testset")
        ws.register("delivery", add_predictions(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                                                self.task, "predictions_test", TEST, testset_id="testset-0001"))
        self.approve("use_testset")
        src = make_artifact("baseline", self.src("bl-lr-test"), self.task, ids=VAL, eval_ids=EVAL, wrong=(10, 20),
                            reused_from="baseline-0001")
        ws.register("baseline", add_predictions(src, self.task, "predictions_test", TEST, wrong=(30,),
                                                testset_id="testset-0001"))
        self.approve("use_testset")
        return ws

    def milestone_src(self, **fields):
        return self.eval_src(uses_testset=True, best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
                             recommendation={"decision": "keep", "reason": "milestone release"}, test_ids=TEST,
                             delivery_id="delivery-0002", **fields)

    def rsi_score_baselines(self, ws, delivery, *baselines):
        """What the judge runs (its skill): `rsi score --baselines-on evalset --baseline <dir>...` on the registered
        files, parsed."""
        def f(aid, logical):
            return os.path.join(ws.path(aid), ws.manifest(aid)["files"][logical]["path"])
        eid = ws.current("evalset")
        argv = [sys.executable, "-m", "rsi", "score", "--task", os.path.join(ws.path("task-0001"), "task.json"),
                "--labels", f(eid, "labels"), "--predictions", f(delivery, "predictions_eval"),
                "--clusters", f(eid, "inputs"), "--baselines-on", "evalset"]
        for b in baselines:
            argv += ["--baseline", ws.path(b)]
        p = subprocess.run(argv, env=dict(os.environ, PYTHONPATH=REPO), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def test_evaluation_must_cover_the_round_baselines(self):
        ws = self.ws_with_comparison_and_delivery()          # comparison lists baseline-0001
        with self.assertRaises(RsiError) as e:
            ws.register("evaluation", self.eval_src())
        self.assertIn("baselines_eval: missing baseline-0001 (lr)", str(e.exception))
        self.assertIn("--baselines-on evalset --baseline <baseline-0001>", str(e.exception))
        # the object the fixed command prints registers as it is
        printed = self.rsi_score_baselines(ws, "delivery-0001", "baseline-0001")["baselines_eval"]
        good = update_manifest(self.eval_src(), baselines_eval=printed)
        self.assertEqual(_manifest(good)["baselines_eval"], _manifest(self.eval_src(baselines=LR))["baselines_eval"])
        shifted = json.loads(json.dumps(_manifest(good)["baselines_eval"]))
        shifted["baseline-0001"]["delta"]["delta"] += 0.01
        with self.assertRaises(RsiError) as e:
            ws.register("evaluation", update_manifest(self.eval_src(baselines=LR), baselines_eval=shifted))
        self.assertIn("baselines_eval.baseline-0001.delta", str(e.exception))
        self.assertIn("--baselines-on evalset --baseline <baseline-0001>`", str(e.exception))
        self.refused("evaluation", self.eval_src(baselines=dict(LR, **{"baseline-0009": ("gbdt", ())})),
                     "baselines_eval.baseline-0009")
        self.assertEqual(ws.register("evaluation", good), "evaluation-0001")

    def test_milestone_evaluation_must_cover_the_test_baselines(self):
        ws = self.ws_milestone_with_test_baseline()          # baseline-0002 has predictions_test
        with self.assertRaises(RsiError) as e:
            ws.register("evaluation", self.milestone_src())
        self.assertIn("baselines_test: missing baseline-0002 (lr)", str(e.exception))
        self.assertTrue(ws.register("evaluation", self.milestone_src(baselines={"baseline-0002": ("lr", (30,))})))
        # baselines_test belongs to a milestone evaluation, baselines_eval to the others
        for field, src in (("baselines_test", self.eval_src()), ("baselines_eval", self.milestone_src())):
            errors = artifacts.validate_artifact("evaluation", update_manifest(src, **{field: {}}), self.task)
            self.assertTrue(any(e.startswith(field) for e in errors), errors)


class TestRules(unittest.TestCase):
    def test_module_of(self):
        self.assertEqual(rules.MODULE_OF, {
            "data_coverage": "data", "label_noise": "data", "label_definition": "data",
            "model_capacity": "train", "calibration": "train",
            "config": "train", "runtime": "deliver"})  # tune was merged into train

if __name__ == "__main__":
    unittest.main()
