import hashlib
import json
import os
import tempfile
import unittest

from fixtures import binary_task, choice_task, feedback_row, make_artifact, prediction_row, write_jsonl
from rsi import artifacts

KINDS = ("candidate", "search", "delivery", "feedback")


def _manifest(d):
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _save(d, m):
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f)


class TestModelArtifacts(unittest.TestCase):
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

    def test_valid_each_kind(self):
        for kind in KINDS:
            self.assertEqual(self.errs(kind, self.mk(kind)), [], kind)

    def test_valid_choice_task(self):
        self.task = choice_task()
        for kind in KINDS:
            self.assertEqual(self.errs(kind, self.mk(kind)), [], kind)

    def test_candidate_needs_primary_metric(self):
        e = self.edit("candidate", lambda m: m["metrics_val"].pop("auc"))
        self.assertTrue(any("auc" in x for x in e), e)

    def test_candidate_bad_predictions(self):
        d = self.mk("candidate")
        rows = [prediction_row(0, self.task), prediction_row(0, self.task)]
        write_jsonl(os.path.join(d, "predictions_val.jsonl"), rows)
        m = _manifest(d)
        m["files"]["predictions_val"]["rows"] = 2
        _save(d, m)
        self.assertTrue(any("duplicate" in x for x in self.errs("candidate", d)))

    def test_candidate_missing_predictions(self):
        e = self.edit("candidate", lambda m: m["files"].pop("predictions_val"))
        self.assertTrue(any("predictions_val" in x for x in e), e)

    def test_candidate_no_coverage_check(self):
        d = self.mk("candidate")
        write_jsonl(os.path.join(d, "predictions_val.jsonl"), [prediction_row(7, self.task)])
        m = _manifest(d)
        m["files"]["predictions_val"]["rows"] = 1
        _save(d, m)
        self.assertEqual(self.errs("candidate", d), [])

    def test_candidate_predict_entry(self):
        # Task 14: model.predict is an executable inside the candidate directory.
        self.assertEqual(self.errs("candidate", self.mk("candidate")), [])
        e = self.edit("candidate", lambda m: m["model"].pop("predict"))
        self.assertTrue(any("predict" in x for x in e), e)
        outside = os.path.join(self.tmp.name, "predict-outside")
        with open(outside, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(outside, 0o755)
        e = self.edit("candidate", lambda m: m["model"].update(predict="../predict-outside"))
        self.assertTrue(any("model.predict" in x and "escapes" in x for x in e), e)
        e = self.edit("candidate", lambda m: m["model"].update(predict=outside))
        self.assertTrue(any("model.predict" in x and "relative" in x for x in e), e)
        e = self.edit("candidate", lambda m: m["model"].update(predict="bin/nope"))
        self.assertTrue(any("model.predict" in x and "bin/nope" in x and "not found" in x for x in e), e)
        d = self.mk("candidate")
        os.chmod(os.path.join(d, "predict"), 0o644)
        e = self.errs("candidate", d)
        self.assertTrue(any("model.predict" in x and "not executable" in x for x in e), e)

    def test_candidate_code_version(self):
        e = self.edit("candidate", lambda m: m.pop("code_version"))
        self.assertTrue(any("code_version" in x for x in e), e)
        code = os.path.join(self.tmp.name, "train")
        with open(code, "w") as f:
            f.write("#!/bin/sh\necho train\n")
        digest = hashlib.sha256(b"#!/bin/sh\necho train\n").hexdigest()

        def cv(entry, sha):
            return lambda m: m.update(code_version={"entry": entry, "sha256": sha})
        # A local file named by an absolute path: its sha256 is checked (hex digits in any case).
        self.assertEqual(self.edit("candidate", cv(code, digest)), [])
        self.assertEqual(self.edit("candidate", cv(code, digest.upper())), [])
        e = self.edit("candidate", cv(code, "0" * 64))
        self.assertTrue(any("code_version.sha256" in x and digest in x for x in e), e)
        e = self.edit("candidate", cv(code, None))
        self.assertTrue(any("code_version.sha256 is null" in x and digest in x for x in e), e)
        # Remote locations, names and paths not on this machine are not checked.
        for entry in ("gpu-box:/opt/train", "trainer@v3", os.path.join(self.tmp.name, "nope")):
            self.assertEqual(self.edit("candidate", cv(entry, None)), [], entry)

    def test_search_best_not_in_trials(self):
        e = self.edit("search", lambda m: m.update(best_candidate="candidate-9"))
        self.assertTrue(any("best_candidate" in x for x in e), e)

    def test_search_default_not_in_trials(self):
        e = self.edit("search", lambda m: m.update(default_candidate="candidate-9"))
        self.assertTrue(any("default_candidate" in x for x in e), e)

    def test_search_over_budget(self):
        e = self.edit("search", lambda m: m["budget"].update(compute_hours=4))
        self.assertTrue(any("budget" in x for x in e), e)

    def test_search_at_budget_ok(self):
        e = self.edit("search", lambda m: m["budget"].update(compute_hours=5))
        self.assertEqual(e, [])

    def test_search_best_not_worse_than_default(self):
        # R28: the tune gate. Fixture: default candidate-0001 0.8, best candidate-0002 0.85, auc higher is better.
        def worse(m):
            m["trials"][1]["metric_val"] = 0.75
            m["gain_vs_default"]["delta"] = -0.05
        e = self.edit("search", worse)
        self.assertTrue(any("worse" in x and "candidate-0002" in x and "candidate-0001" in x for x in e), e)
        # Handing back the default as best, with zero gain, passes.
        e = self.edit("search", lambda m: (m.update(best_candidate="candidate-0001"),
                                           m["gain_vs_default"].update(delta=0)))
        self.assertEqual(e, [])
        # Lower is better: the same numbers make the best worse than the default.
        self.task["metrics"]["higher_is_better"] = False
        e = self.errs("search", self.mk("search"))
        self.assertTrue(any("worse" in x and "lower is better" in x for x in e), e)
        e = self.edit("search", lambda m: (m["trials"][1].update(metric_val=0.7),
                                           m["gain_vs_default"].update(delta=-0.1)))
        self.assertEqual(e, [])

    def test_search_gain_vs_default_must_match_trials(self):
        e = self.edit("search", lambda m: m["gain_vs_default"].update(metric="log_loss"))
        self.assertTrue(any("gain_vs_default.metric" in x and "auc" in x for x in e), e)
        e = self.edit("search", lambda m: m["gain_vs_default"].update(delta=0.06))
        self.assertTrue(any("gain_vs_default.delta" in x for x in e), e)
        # 0.85 - 0.8 is not exactly 0.05 in floating point; the check allows for that.
        self.assertEqual(self.edit("search", lambda m: m["gain_vs_default"].update(delta=0.05)), [])

    def test_search_failed_trials(self):
        # M3: a trial whose status is not "ok" may have no candidate and no metric.
        failed = {"candidate_id": None, "config": {"lr": 1.0}, "metric_val": None, "compute_hours": 1,
                  "status": "failed"}
        self.assertEqual(self.edit("search", lambda m: m["trials"].append(failed)), [])
        self.assertEqual(self.edit("search", lambda m: m["trials"].extend([failed, dict(failed)])), [])
        # An ok trial needs both.
        e = self.edit("search", lambda m: m["trials"].append(dict(failed, status="ok")))
        self.assertTrue(any("trials[2]" in x and "candidate_id" in x for x in e), e)
        self.assertTrue(any("trials[2]" in x and "metric_val" in x for x in e), e)
        # best and default must be ok trials.
        e = self.edit("search", lambda m: m["trials"][0].update(status="timeout"))
        self.assertTrue(any("default_candidate" in x and "ok trial" in x for x in e), e)

    def test_delivery_package_location(self):
        # Task 14: the package is inside the delivery directory, or an `external` entry (with a sha256).
        self.assertEqual(self.errs("delivery", self.mk("delivery")), [])
        outside = os.path.join(self.tmp.name, "pkg-outside")
        os.makedirs(outside)
        e = self.edit("delivery", lambda m: m["package"].update(location=outside))
        self.assertTrue(any("package.location" in x and "relative" in x for x in e), e)
        e = self.edit("delivery", lambda m: m["package"].update(location="../pkg-outside"))
        self.assertTrue(any("package.location" in x and "escapes" in x for x in e), e)
        e = self.edit("delivery", lambda m: m["package"].update(location="pkg/nope"))
        self.assertTrue(any("package.location" in x and "pkg/nope" in x and "not found" in x for x in e), e)
        e = self.edit("delivery", lambda m: m["package"].update(location="external:weights"))
        self.assertTrue(any("package.location" in x and "external:weights" in x and "'weights'" in x
                            for x in e), e)
        remote = {"weights": {"location": "gpu-box:/srv/pkg.tar", "sha256": "ab" * 32}}
        e = self.edit("delivery", lambda m: (m["package"].update(location="external:weights"),
                                             m.update(external=remote)))
        self.assertEqual(e, [])
        # A file inside the delivery directory is a package too.
        e = self.edit("delivery", lambda m: m["package"].update(location="package/predict"))
        self.assertEqual(e, [])

    def test_delivery_needs_valset_reproduction(self):
        # Task 14: predictions_val (the package on the valset inputs) and valset_id are required.
        e = self.edit("delivery", lambda m: m["files"].pop("predictions_val"))
        self.assertTrue(any("predictions_val" in x for x in e), e)
        e = self.edit("delivery", lambda m: m.pop("valset_id"))
        self.assertTrue(any("valset_id" in x for x in e), e)
        d = self.mk("delivery")
        write_jsonl(os.path.join(d, "predictions_val.jsonl"), [prediction_row(0, self.task)] * 2)
        m = _manifest(d)
        m["files"]["predictions_val"]["rows"] = 2
        _save(d, m)
        self.assertTrue(any("predictions_val" in x and "duplicate" in x for x in self.errs("delivery", d)))

    def test_delivery_cost_unit(self):
        self.assertEqual(self.edit("delivery", lambda m: m["runtime"].update(cost_unit="usd", cost_per_1k=0.4)), [])
        e = self.edit("delivery", lambda m: m["runtime"].update(cost_unit=3))
        self.assertTrue(any("cost_unit" in x for x in e), e)

    def test_delivery_inconsistent_runtime(self):
        e = self.edit("delivery", lambda m: m["runtime"].update(consistent_with_offline=False))
        self.assertTrue(any("consistent_with_offline" in x for x in e), e)

    def test_delivery_test_predictions_need_testset_id(self):
        d = self.mk("delivery")
        write_jsonl(os.path.join(d, "predictions_test.jsonl"), [prediction_row(0, self.task)])
        m = _manifest(d)
        m["files"]["predictions_test"] = {"path": "predictions_test.jsonl", "rows": 1}
        _save(d, m)
        self.assertTrue(any("testset_id" in x for x in self.errs("delivery", d)))
        m["testset_id"] = "testset-0001"
        _save(d, m)
        self.assertEqual(self.errs("delivery", d), [])

    def test_delivery_optional_trace_validated(self):
        d = self.mk("delivery")
        write_jsonl(os.path.join(d, "trace.jsonl"), [feedback_row(0, self.task, outcome="maybe")])
        m = _manifest(d)
        m["files"]["trace"] = {"path": "trace.jsonl", "rows": 1}
        _save(d, m)
        self.assertTrue(any("outcome" in x for x in self.errs("delivery", d)))

    def test_feedback_without_trace_refused(self):
        d = self.mk("feedback")
        m = _manifest(d)
        del m["files"]["trace"]
        _save(d, m)
        self.assertTrue(any("trace" in x for x in self.errs("feedback", d)), self.errs("feedback", d))

    def test_feedback_bad_outcome_label(self):
        d = self.mk("feedback")
        write_jsonl(os.path.join(d, "trace.jsonl"),
                    [feedback_row(i, self.task, outcome="maybe" if i == 0 else None) for i in range(3)])
        self.assertTrue(any("outcome" in x for x in self.errs("feedback", d)))

    def test_feedback_needs_a_reference(self):
        e = self.edit("feedback", lambda m: m.pop("delivery_id"))
        self.assertTrue(any("delivery_id" in x and "candidate_id" in x for x in e), e)

    def test_feedback_candidate_id_alone_ok(self):
        e = self.edit("feedback", lambda m: (m.pop("delivery_id"), m.update(candidate_id="candidate-0001")))
        self.assertEqual(e, [])

if __name__ == "__main__":
    unittest.main()
