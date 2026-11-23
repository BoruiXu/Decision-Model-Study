import json
import os
import tempfile
import unittest

from fixtures import binary_task, choice_task, make_artifact
from rsi import artifacts, rules

KINDS = ("evaluation", "plan", "record")


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

    def test_evaluation_requires_report(self):
        self.assertTrue(self.edit("evaluation", lambda m: m["files"].clear()))

    def test_evaluation_without_best_before(self):
        def f(m):
            m["best_before"] = None
            m["metrics"]["best"] = None
            m["deltas"] = {}
        self.assertEqual(self.edit("evaluation", f), [])

    def test_evaluation_candidate_needs_primary(self):
        def f(m):
            del m["metrics"]["candidate"]["auc"]
        self.assertTrue(any("auc" in e for e in self.edit("evaluation", f)))

    def test_evaluation_with_best_needs_primary_delta(self):
        def f(m):
            m["deltas"] = {}
        self.assertTrue(any("deltas" in e for e in self.edit("evaluation", f)))

    def test_evaluation_with_best_needs_best_metrics(self):
        def f(m):
            m["metrics"]["best"] = None
        self.assertTrue(any("metrics.best" in e for e in self.edit("evaluation", f)))

    def test_delta_needs_ci(self):
        def f(m):
            del m["deltas"]["auc"]["ci_low"]
        self.assertTrue(self.edit("evaluation", f))

    def test_bad_decision(self):
        def f(m):
            m["recommendation"]["decision"] = "maybe"
        self.assertTrue(self.edit("evaluation", f))

    def test_milestone_metrics_test(self):
        # R27: metrics_test is required when uses_testset is true, holds the primary metric,
        # and is not allowed otherwise.
        e = self.edit("evaluation", lambda m: m.update(uses_testset=True))
        self.assertTrue(any("metrics_test" in x and "required" in x for x in e), e)
        e = self.edit("evaluation", lambda m: m.update(uses_testset=True, metrics_test={"log_loss": 0.4}))
        self.assertTrue(any("metrics_test" in x and "auc" in x for x in e), e)
        e = self.edit("evaluation", lambda m: m.update(uses_testset=True, metrics_test={"auc": "high"}))
        self.assertTrue(any("metrics_test" in x for x in e), e)
        self.assertEqual(self.edit("evaluation", lambda m: m.update(uses_testset=True,
                                                                     metrics_test={"auc": 0.83})), [])
        e = self.edit("evaluation", lambda m: m.update(metrics_test={"auc": 0.83}))
        self.assertTrue(any("metrics_test" in x and "uses_testset" in x for x in e), e)

    def test_attribution_module_mismatch(self):
        def f(m):
            m["failures"][0]["module"] = "train"
        self.assertTrue(any("data_coverage" in e for e in self.edit("evaluation", f)))

        def g(m):
            m["issues"][0]["module"] = "deliver"
        self.assertTrue(any("data_coverage" in e for e in self.edit("plan", g)))

    def test_unknown_attribution(self):
        def f(m):
            m["failures"][0]["attribution"] = "bad_luck"
        self.assertTrue(any("bad_luck" in e for e in self.edit("evaluation", f)))

    def test_entry_module_not_in_issues(self):
        def f(m):
            m["entry_module"] = "train"
        self.assertTrue(any("entry_module" in e for e in self.edit("plan", f)))

    def test_plan_bad_direction(self):
        def f(m):
            m["issues"][0]["expected"]["direction"] = "sideways"
        self.assertTrue(self.edit("plan", f))

    def test_plan_entry_module_enum(self):
        def f(m):
            m["entry_module"] = "judge"
        errs = self.edit("plan", f)
        self.assertTrue(any("not in enum" in e for e in errs), errs)

    def test_record_conclusion_enum(self):
        def f(m):
            m["result"]["conclusion"] = "great"
        self.assertTrue(self.edit("record", f))

    def test_record_null_numbers_and_orchestrator(self):
        def f(m):
            m["module"] = "orchestrator"
            m["result"]["before"] = None
            m["result"]["after"] = None
            m["automation"] = "automated"
        self.assertEqual(self.edit("record", f), [])

    def test_record_bad_module(self):
        def f(m):
            m["module"] = "nobody"
        self.assertTrue(self.edit("record", f))


class TestRules(unittest.TestCase):
    def test_module_of(self):
        self.assertEqual(rules.MODULE_OF, {
            "data_coverage": "data", "label_noise": "data", "label_definition": "data",
            "model_capacity": "train", "calibration": "train",
            "config": "tune", "runtime": "deliver"})

if __name__ == "__main__":
    unittest.main()
