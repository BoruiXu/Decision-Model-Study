"""The artifacts of ③ Compare, `baseline` and `comparison`: their directory checks and the numbers registration
recomputes."""
import json
import os

from fixtures import add_predictions, make_artifact, update_manifest
from rsi import engine
from rsi.artifacts import validate_artifact
from rsi.runners import ManualRunner
from rsi.workspace import _single_use_needed, unverified_names
from test_workspace import EVAL, TEST, VAL, WsCase


def _manifest(d):
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _edit(d, fn):
    m = _manifest(d)
    fn(m)
    return update_manifest(d, **m)


# Baselines of the comparisons below, on the valset rows VAL: (id, the valset ids it gets wrong). gbdt is the weakest;
# lr and mlp tie, so lr, the alphabetically first, is the strongest.
BASELINES = {"gbdt": ("baseline-0001", (10, 11)), "lr": ("baseline-0002", (10,)), "mlp": ("baseline-0003", (10,))}


class TestCompareArtifacts(WsCase):
    def baseline(self, name="lr", **fields):
        """A fresh baseline source: predictions_val on the valset rows VAL, predictions_eval on the evalset rows EVAL."""
        self.n += 1
        fields.setdefault("notes", "source %d" % self.n)
        fields.setdefault("ids", VAL)
        fields.setdefault("eval_ids", EVAL)
        return make_artifact("baseline", self.src("baseline-%d" % self.n), self.task, name=name, **fields)

    def comparison(self, **fields):
        """A fresh comparison source of candidate-0001 (right on every valset row) against BASELINES."""
        fields.setdefault("ids", VAL)
        fields.setdefault("candidate", "candidate-0001")
        fields.setdefault("baselines", BASELINES)
        return self.mk("comparison", **fields)

    def ws_with_candidate_and_baselines(self):
        self.sets()
        self.candidate()
        for name in sorted(BASELINES):
            self.ws.register("baseline", self.baseline(name, wrong=BASELINES[name][1]))

    def test_directory_checks(self):
        for kind, d in (("baseline", self.baseline()), ("comparison", self.comparison())):
            self.assertEqual(validate_artifact(kind, d, self.task), [], kind)
        # (kind, change of a valid manifest, words one error must contain)
        cases = (
            ("baseline", lambda m: m["metrics_val"].pop("auc"), ("metrics_val", "auc")),
            ("baseline", lambda m: m.update(name="Lr"), ("name", "pattern")),
            ("baseline", lambda m: m.update(testset_id="testset-0001"), ("testset_id", "predictions_test")),
            ("baseline", lambda m: m.update(model={"location": "model", "predict": "bin/predict"}),
             ("model.predict", "bin/predict", "not found")),
            ("comparison", lambda m: m["metrics_val"]["candidate"].pop("auc"), ("metrics_val.candidate", "auc")),
            ("comparison", lambda m: m["metrics_val"].pop("mlp"), ("metrics_val", "mlp")),
            ("comparison", lambda m: m["deltas_val"].update(svm=m["deltas_val"]["lr"]), ("deltas_val.svm", "baselines")),
            ("comparison", lambda m: m["deltas_val"]["lr"].update(metric="log_loss"), ("deltas_val.lr.metric", "auc")),
            ("comparison", lambda m: m.update(strongest=None), ("strongest",)),
            ("comparison", lambda m: m["skipped"].update(lr="unknown baseline lr"), ("skipped.lr", "baselines")),
        )
        for kind, change, words in cases:
            with self.subTest(kind=kind, words=words):
                d = _edit(self.baseline() if kind == "baseline" else self.comparison(), change)
                errs = validate_artifact(kind, d, self.task)
                self.assertTrue(any(all(w in e for w in words) for e in errs), errs)
        d = add_predictions(self.baseline(), self.task, "predictions_test", TEST)
        self.assertTrue(any("testset_id" in e for e in validate_artifact("baseline", d, self.task)))

    def test_baseline_registers_and_its_numbers_are_checked(self):
        self.sets()
        self.assertEqual(self.ws.register("baseline", self.baseline(wrong=(10,))), "baseline-0001")
        self.refused("baseline", update_manifest(self.baseline(wrong=(10,)), metrics_val={"auc": 0.5}),
                     "metrics_val.auc", "reported 0.5", "recomputed 0.75", "rsi score")
        self.refused("baseline", self.baseline(eval_ids=VAL), "predictions_eval", "evalset-0001")
        self.refused("baseline", self.baseline(valset_id="valset-0009"), "valset_id", "valset-0009")
        self.refused("baseline", self.baseline("gbdt", reused_from="baseline-0001"), "reused_from", "'lr'", "'gbdt'")
        # zero_shot does not depend on the trainset.
        self.assertEqual(self.ws.register("baseline", self.baseline("zero_shot", trainset_id=None)), "baseline-0002")

    def test_comparison_numbers_are_recomputed(self):
        self.ws_with_candidate_and_baselines()
        good = self.comparison()
        m = _manifest(good)
        self.assertEqual(m["strongest"], "lr")
        for name in ("gbdt", "mlp"):  # the weaker one; the tied one that is not alphabetically first
            self.refused("comparison", update_manifest(self.comparison(), strongest=name), "strongest", "'lr'")
        deltas = dict(m["deltas_val"], lr=dict(m["deltas_val"]["lr"], delta=m["deltas_val"]["lr"]["delta"] + 0.01))
        self.refused("comparison", update_manifest(self.comparison(), deltas_val=deltas),
                     "deltas_val.lr.delta", "--compare <baseline-0002>/predictions_val.jsonl")
        metrics = dict(m["metrics_val"], gbdt={"auc": 0.9})
        self.refused("comparison", update_manifest(self.comparison(), metrics_val=metrics),
                     "metrics_val.gbdt.auc", "<baseline-0001>/predictions_val.jsonl")
        # Right numbers under the wrong names: the gbdt and lr baselines swapped, or one baseline listed twice.
        swapped = {"gbdt": ("baseline-0002", (10,)), "lr": ("baseline-0001", (10, 11)), "mlp": BASELINES["mlp"]}
        self.refused("comparison", self.comparison(baselines=swapped),
                     "baselines.lr: baseline-0001 is the baseline named 'gbdt', but the comparison lists it as 'lr'")
        twice = dict(BASELINES, mlp=BASELINES["lr"])
        self.refused("comparison", self.comparison(baselines=twice), "baseline-0002 is listed under lr, mlp")
        self.assertEqual(self.ws.register("comparison", good), "comparison-0001")
        # rsi register notes what it could not verify: here nothing, with a primary metric rsi does not compute that.
        self.assertEqual(self.ws.unverified("comparison-0001"), [])
        custom = dict(self.task, metrics=dict(self.task["metrics"], primary="custom"))
        m["metrics_val"]["candidate"]["custom"] = 0.9
        self.assertEqual(unverified_names("comparison", m, custom), ["custom"])
        self.assertEqual(unverified_names("baseline", {"metrics_val": {"auc": 0.8, "custom": 0.7}}, custom), ["custom"])

    def test_a_baseline_with_test_predictions_needs_use_testset(self):
        self.sets()
        self.ws.register("baseline", self.baseline())
        src = add_predictions(self.baseline(reused_from="baseline-0001"), self.task, "predictions_test", TEST,
                              testset_id="testset-0001")
        # The engine writes the use_testset approval (quoting the milestone consent) before registering it.
        self.assertEqual(engine.Engine(self.wsdir, ManualRunner(), pages=False)._kind_of(src, {"module": "compare"}),
                         ("baseline", None, True))
        self.refused("baseline", src, "use_testset")
        self.ws.approve("use_testset", by="u", quote="yes")
        self.refused("baseline", add_predictions(self.baseline(), self.task, "predictions_test", EVAL,
                                                 testset_id="testset-0001"), "predictions_test", "testset-0001")
        self.assertEqual(self.ws.register("baseline", src), "baseline-0002")
        self.assertEqual(_single_use_needed(self.ws, "baseline", self.ws.manifest("baseline-0002"), "baseline-0002"),
                         ["use_testset"])

    def test_a_baseline_on_another_evalset_is_refused_in_a_comparison(self):
        self.sets()
        self.candidate()
        self.ws.register("baseline", self.baseline())
        self.approve("refresh_sets")
        self.ws.register("evalset", self.mk("evalset", ids=range(60, 64)))
        self.refused("comparison", self.comparison(evalset_id="evalset-0002", baselines={"lr": "baseline-0001"}),
                     "baselines.lr", "evalset_id", "evalset-0001", "evalset-0002")
