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


class TestRules(unittest.TestCase):
    def test_module_of(self):
        self.assertEqual(rules.MODULE_OF, {
            "data_coverage": "data", "label_noise": "data", "label_definition": "data",
            "model_capacity": "train", "calibration": "train",
            "config": "train", "runtime": "deliver"})  # tune was merged into train

if __name__ == "__main__":
    unittest.main()
