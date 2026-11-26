"""Shared metric, baseline and confidence-interval definitions (rsi.metrics, Task 13)."""
import math
import unittest

from fixtures import binary_task, choice_task
from rsi import metrics as M
from rsi.rows import check_task


def lab(rid, label, task, **meta):
    """A label row whose target is the hard label `label`."""
    target = {l: 0.0 for l in task["decision"]["labels"]}
    target[label] = 1.0
    return {"id": rid, "target": target, "meta": meta}


def soft(rid, target, **meta):
    return {"id": rid, "target": dict(target), "meta": meta}


def pred(rid, probs, decision=None):
    if decision is None:
        decision = max(sorted(probs), key=lambda l: probs[l])
    return {"id": rid, "probs": dict(probs), "decision": decision, "uncertainty": None}


class TestHardLabel(unittest.TestCase):
    def test_argmax_and_ties_by_label_order(self):
        labels = ["allow", "block", "review"]
        self.assertEqual(M.hard_label({"allow": 0.2, "block": 0.5, "review": 0.3}, labels), "block")
        # Ties go to the earlier label in task.decision.labels, whatever the map's key order.
        self.assertEqual(M.hard_label({"review": 0.4, "block": 0.4, "allow": 0.2}, labels), "block")
        self.assertEqual(M.hard_label({"review": 0.5, "allow": 0.5, "block": 0.0}, labels), "allow")
        self.assertEqual(M.hard_label({"yes": 0.5, "no": 0.5}, ["no", "yes"]), "no")


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.bt = binary_task()  # labels yes, no; positive yes
        self.ct = choice_task()  # labels allow, block, review

    def test_known_names_and_score_dispatch(self):
        self.assertEqual(M.KNOWN_METRICS, ("accuracy", "log_loss", "macro_f1", "auc", "auprc", "brier", "f1",
                                           "expected_cost"))
        self.assertEqual(M.KNOWN_BASELINES, ("majority", "group_prior"))
        self.assertEqual(sorted(M.METRICS), sorted(M.KNOWN_METRICS))
        self.assertEqual(sorted(M.BASELINES), sorted(M.KNOWN_BASELINES))
        t = self.bt
        labels = [lab("a", "yes", t), lab("b", "no", t)]
        preds = [pred("a", {"yes": 0.9, "no": 0.1}), pred("b", {"yes": 0.9, "no": 0.1})]
        self.assertEqual(M.score("accuracy", preds, labels, t), 0.5)
        with self.assertRaises(ValueError):
            M.score("recall", preds, labels, t)

    def test_accuracy_uses_decision_and_hard_label(self):
        t = self.ct
        labels = [lab("a", "allow", t), lab("b", "block", t),
                  soft("c", {"allow": 0.4, "block": 0.4, "review": 0.2}),  # hard label: allow (tie)
                  lab("d", "review", t)]
        preds = [pred("d", {"allow": 0.1, "block": 0.1, "review": 0.8}),
                 pred("a", {"allow": 0.9, "block": 0.05, "review": 0.05}),
                 pred("b", {"allow": 0.6, "block": 0.3, "review": 0.1}, decision="block"),
                 pred("c", {"allow": 0.2, "block": 0.7, "review": 0.1})]
        # Rows are matched by id; b counts as right because its decision is block.
        self.assertEqual(M.accuracy(preds, labels, t), 0.75)

    def test_log_loss(self):
        t = self.bt
        labels = [lab("a", "yes", t), lab("b", "no", t), soft("c", {"yes": 0.7, "no": 0.3})]
        preds = [pred("a", {"yes": 0.8, "no": 0.2}), pred("b", {"yes": 0.0, "no": 1.0}),
                 pred("c", {"yes": 0.0, "no": 1.0})]
        # c's hard label is yes, predicted with probability 0, clipped to 1e-12.
        want = (-math.log(0.8) + 0.0 - math.log(1e-12)) / 3
        self.assertAlmostEqual(M.log_loss(preds, labels, t), want, places=12)
        perfect = M.log_loss([pred("a", {"yes": 1.0, "no": 0.0})], [lab("a", "yes", t)], t)
        self.assertEqual(perfect, 0.0)
        self.assertEqual(math.copysign(1.0, perfect), 1.0)  # never -0.0

    def test_macro_f1_over_all_labels(self):
        t = self.ct
        labels = [lab("a", "allow", t), lab("b", "allow", t), lab("c", "block", t), lab("d", "block", t)]
        preds = [pred("a", {"allow": 1.0, "block": 0.0, "review": 0.0}),
                 pred("b", {"allow": 0.0, "block": 1.0, "review": 0.0}),
                 pred("c", {"allow": 0.0, "block": 1.0, "review": 0.0}),
                 pred("d", {"allow": 0.0, "block": 1.0, "review": 0.0})]
        # allow: tp 1, fp 0, fn 1 -> 2/3; block: tp 2, fp 1, fn 0 -> 0.8;
        # review: never true nor predicted -> 0 (it still counts: macro over all labels).
        self.assertAlmostEqual(M.macro_f1(preds, labels, t), (2 / 3 + 0.8 + 0.0) / 3, places=12)

    def test_auc_with_ties(self):
        t = self.bt
        labels = [lab("p1", "yes", t), lab("p2", "yes", t), lab("n1", "no", t), lab("n2", "no", t)]
        preds = [pred("p1", {"yes": 0.9, "no": 0.1}), pred("p2", {"yes": 0.4, "no": 0.6}),
                 pred("n1", {"yes": 0.4, "no": 0.6}), pred("n2", {"yes": 0.2, "no": 0.8})]
        # Pairs (p, n): (p1,n1) 1, (p1,n2) 1, (p2,n1) tie 0.5, (p2,n2) 1 -> 3.5 / 4.
        self.assertEqual(M.auc(preds, labels, t), 0.875)
        # The positive label comes from task.decision.positive, not from label order; with
        # probabilities that sum to 1 the other choice of positive gives the same AUC.
        t2 = binary_task()
        t2["decision"]["positive"] = "no"
        self.assertEqual(M.auc(preds, labels, t2), 0.875)
        # auc needs a binary task and both classes among the labels.
        with self.assertRaises(ValueError):
            M.auc([pred("a", {"allow": 1.0, "block": 0.0, "review": 0.0})], [lab("a", "allow", self.ct)], self.ct)
        with self.assertRaises(ValueError):
            M.auc([pred("a", {"yes": 0.9, "no": 0.1})], [lab("a", "yes", self.bt)], self.bt)

    def test_auprc_and_brier(self):
        # D2: auprc is the average precision of decision.positive (rows with tied scores are one threshold),
        # brier the mean of (p(positive) - [hard label is positive])^2; both binary only.
        t = self.bt

        def case(scores):
            """(predictions, label rows) from {id: positive probability}; ids starting with p are positive."""
            return ([pred(i, {"yes": s, "no": 1 - s}) for i, s in scores.items()],
                    [lab(i, "yes" if i[0] == "p" else "no", t) for i in scores])
        self.assertEqual(M.auprc(*case({"p1": 0.9, "p2": 0.8, "n1": 0.3, "n2": 0.1}), t), 1.0)  # perfect ranking
        # p1 at recall 1/2 with precision 1, p2 at recall 1 with precision 2/4.
        self.assertAlmostEqual(M.auprc(*case({"p1": 0.9, "n1": 0.8, "n2": 0.7, "p2": 0.6}), t), 0.75, places=12)
        # The tie p2 = p3 = n1 is one threshold: recall 1/3 -> 1 at precision 3/4 (no order of the tie gives that).
        self.assertAlmostEqual(M.auprc(*case({"p1": 0.9, "p2": 0.5, "p3": 0.5, "n1": 0.5, "n2": 0.1}), t), 5 / 6,
                               places=12)
        with self.assertRaises(ValueError):  # no positive row
            M.auprc(*case({"n1": 0.2}), t)
        preds = [pred("p1", {"yes": 0.9, "no": 0.1}), pred("n1", {"yes": 0.3, "no": 0.7}),
                 pred("s", {"yes": 0.6, "no": 0.4})]
        labels = [lab("p1", "yes", t), lab("n1", "no", t), soft("s", {"yes": 0.7, "no": 0.3})]  # s: hard label yes
        self.assertAlmostEqual(M.brier(preds, labels, t), (0.01 + 0.09 + 0.16) / 3, places=12)
        # Directions: a task must agree with them; rsi score reports both.
        self.assertEqual((M.HIGHER_IS_BETTER["auprc"], M.HIGHER_IS_BETTER["brier"]), (True, False))
        task = binary_task()
        task["metrics"].update(primary="auprc", others=["brier"])
        task["promotion_rule"]["guards"] = [{"metric": "brier", "higher_is_better": False, "max_regression": 0.005}]
        self.assertEqual(check_task(task), [])
        from rsi.cli import score_numbers
        self.assertEqual(score_numbers(task, labels, preds)["metrics"],
                         {"auprc": M.auprc(preds, labels, t), "brier": M.brier(preds, labels, t)})
        task["metrics"]["higher_is_better"] = False
        task["promotion_rule"]["guards"][0]["higher_is_better"] = True
        errs = check_task(task)
        for word in ("auprc is higher-is-better", "brier is lower-is-better"):
            self.assertTrue(any(word in e for e in errs), (word, errs))
        ct = choice_task()
        ct["metrics"]["others"] = ["auprc", "brier"]
        errs = check_task(ct)
        for name in ("auprc", "brier"):
            self.assertTrue(any("%s is defined for binary tasks only" % name in e for e in errs), (name, errs))
            with self.assertRaises(ValueError):
                M.score(name, [pred("a", {"allow": 1.0, "block": 0.0, "review": 0.0})], [lab("a", "allow", ct)], ct)

    def test_f1_and_expected_cost(self):
        t = self.bt
        hard = ["yes", "yes", "no", "yes", "no", "no"]
        decided = ["yes", "yes", "yes", "yes", "no", "no"]                  # tp 3, fp 1, fn 0
        labels = [lab("r%d" % i, y, t) for i, y in enumerate(hard)]
        # the decision counts, not the probabilities (a policy's threshold may say yes below one half)
        preds = [pred("r%d" % i, {"yes": 0.4, "no": 0.6}, decision=d) for i, d in enumerate(decided)]
        self.assertAlmostEqual(M.score("f1", preds, labels, t), 6 / 7)
        self.assertEqual(M.f1(preds[4:], labels[4:], t), 0.0)               # no tp, fp nor fn
        costs = binary_task()
        costs["metrics"]["costs"] = {"yes": {"yes": 0, "no": 10}, "no": {"yes": 1, "no": 0}}
        self.assertAlmostEqual(M.score("expected_cost", preds, labels, costs), (1 * 1 + 10 * 0) / 6)
        with self.assertRaisesRegex(ValueError, "expected_cost needs metrics.costs"):
            M.score("expected_cost", preds, labels, t)
        self.assertEqual((M.HIGHER_IS_BETTER["f1"], M.HIGHER_IS_BETTER["expected_cost"]), (True, False))
        with self.assertRaisesRegex(ValueError, "binary tasks only"):
            M.score("f1", [pred("a", {"allow": 1.0, "block": 0.0, "review": 0.0})], [lab("a", "allow", self.ct)],
                    self.ct)
        d = M.paired_delta("expected_cost", preds, [dict(p, decision="no") for p in preds], labels, costs, None,
                           100, 0, 0.9)
        self.assertAlmostEqual(d["delta"], 1 / 6 - 30 / 6)                  # never yes: 3 misses at 10

    def test_predictions_must_match_the_label_ids(self):
        t = self.bt
        labels = [lab("a", "yes", t), lab("b", "no", t)]
        with self.assertRaises(ValueError) as cm:
            M.accuracy([pred("a", {"yes": 1.0, "no": 0.0})], labels, t)
        self.assertIn("b", str(cm.exception))
        with self.assertRaises(ValueError):
            M.accuracy([pred("a", {"yes": 1.0, "no": 0.0}), pred("b", {"yes": 0.0, "no": 1.0}),
                        pred("c", {"yes": 0.0, "no": 1.0})], labels, t)
        p = pred("a", {"yes": 1.0, "no": 0.0})
        for metric in (M.accuracy, M.log_loss, M.auc):
            with self.assertRaises(ValueError) as cm:
                metric([p, p, pred("b", {"yes": 0.0, "no": 1.0})], labels, t)
            self.assertIn("duplicate prediction id", str(cm.exception))



class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.t = choice_task()

    def train(self):
        t = self.t
        return [lab("t1", "allow", t, group="web"), lab("t2", "allow", t, group="web"),
                lab("t3", "block", t, group="app"), lab("t4", "review", t, group="app")]

    def test_majority_predicts_the_training_prior(self):
        scored = [{"id": "s1", "target": {}}, {"id": "s2", "target": {}}]
        rows = M.majority(self.train(), scored, self.t)
        self.assertEqual([r["id"] for r in rows], ["s1", "s2"])
        for r in rows:
            self.assertEqual(r["probs"], {"allow": 0.5, "block": 0.25, "review": 0.25})
            self.assertEqual(r["decision"], "allow")
        # A tie in the prior goes to the earlier label.
        train = [lab("t1", "review", self.t), lab("t2", "block", self.t)]
        self.assertEqual(M.majority(train, scored, self.t)[0]["decision"], "block")
        # Soft targets are averaged as they are.
        train = [soft("t1", {"allow": 0.6, "block": 0.4, "review": 0.0}),
                 soft("t2", {"allow": 0.0, "block": 0.6, "review": 0.4})]
        self.assertEqual(M.majority(train, scored, self.t)[0]["probs"], {"allow": 0.3, "block": 0.5, "review": 0.2})

    def test_group_prior_with_unseen_group_fallback(self):
        scored = [{"id": "s1", "meta": {"group": "web"}}, {"id": "s2", "meta": {"group": "app"}},
                  {"id": "s3", "meta": {"group": "phone"}}]
        rows = M.group_prior(self.train(), scored, self.t)
        self.assertEqual(rows[0]["probs"], {"allow": 1.0, "block": 0.0, "review": 0.0})
        self.assertEqual(rows[0]["decision"], "allow")
        self.assertEqual(rows[1]["probs"], {"allow": 0.0, "block": 0.5, "review": 0.5})
        self.assertEqual(rows[1]["decision"], "block")
        # An unseen group falls back to the overall prior.
        self.assertEqual(rows[2]["probs"], {"allow": 0.5, "block": 0.25, "review": 0.25})
        # Label rows carry no meta.group: the groups come from the inputs file given.
        scored = [{"id": "s1", "target": {}}, {"id": "s2", "target": {}}]
        groups = M.groups_of([{"id": "s1", "meta": {"group": "app"}}, {"id": "s2", "meta": {"group": "web"}}])
        rows = M.group_prior(self.train(), scored, self.t, groups)
        self.assertEqual([r["decision"] for r in rows], ["block", "allow"])
        with self.assertRaises(ValueError) as cm:
            M.group_prior(self.train(), scored, self.t)
        self.assertIn("meta.group", str(cm.exception))

    def test_baseline_rows_are_valid_predictions(self):
        from rsi.rows import check_predictions
        scored = [{"id": "s1", "meta": {"group": "web"}}]
        for name in M.KNOWN_BASELINES:
            self.assertEqual(check_predictions(M.BASELINES[name](self.train(), scored, self.t), self.t), [], name)


# Accuracy of a (5 of 6 right) minus b (4 of 6 right), as floats.
DELTA = 5 / 6 - 4 / 6


class TestPairedDelta(unittest.TestCase):
    def setUp(self):
        self.t = binary_task()
        t = self.t
        self.labels = [lab("q%d" % i, "yes" if i % 2 == 0 else "no", t) for i in range(6)]
        right = lambda i: {"yes": 0.9, "no": 0.1} if i % 2 == 0 else {"yes": 0.1, "no": 0.9}
        wrong = lambda i: {"yes": 0.1, "no": 0.9} if i % 2 == 0 else {"yes": 0.9, "no": 0.1}
        self.a = [pred("q%d" % i, wrong(i) if i == 0 else right(i)) for i in range(6)]
        self.b = [pred("q%d" % i, wrong(i) if i in (0, 3) else right(i)) for i in range(6)]
        self.clusters = {"q0": "c0", "q1": "c0", "q2": "c1", "q3": "c1", "q4": "c2", "q5": "c2"}

    def delta(self, metric="accuracy", **kw):
        args = dict(clusters=self.clusters, resamples=200, seed=0, level=0.9)
        args.update(kw)
        return M.paired_delta(metric, self.a, self.b, self.labels, self.t, **args)

    def test_known_small_example(self):
        # Only cluster c1 differs (b gets q3 wrong): a draw of three clusters holding c1 m times
        # gives m / 6, so the bounds are draws such as c0 + c1 + c1 (a: 5 of 6, b: 3 of 6 right).
        self.assertEqual(self.delta(), {"delta": DELTA, "ci_low": 0.0, "ci_high": 5 / 6 - 3 / 6})
        d = self.delta("auc")
        self.assertEqual(d["delta"], M.auc(self.a, self.labels, self.t) - M.auc(self.b, self.labels, self.t))
        self.assertEqual(d["delta"], 7.5 / 9 - 6 / 9)
        # Golden values with every row its own cluster: identical under Python 3.9 and 3.14
        # (random.Random(seed).randrange, math.fsum, linear interpolation between draws).
        self.assertEqual(self.delta("log_loss", clusters=None, level=0.8),
                         {"delta": -0.3662040962227032, "ci_low": -0.7690286020676786, "ci_high": 0.0})
        self.assertEqual(self.delta("auc", clusters=None, level=0.8),
                         {"delta": 0.16666666666666674, "ci_low": 0.0, "ci_high": 0.5})
        self.assertEqual(self.delta("macro_f1", clusters=None, level=0.8),
                         {"delta": 0.16190476190476188, "ci_low": 0.0, "ci_high": 0.37777777777777766})

    def test_deterministic(self):
        self.assertEqual(self.delta(), self.delta())
        self.assertEqual(self.delta("log_loss", seed=7), self.delta("log_loss", seed=7))
        # Row order does not matter.
        flipped = M.paired_delta("accuracy", self.a[::-1], self.b, self.labels[::-1], self.t,
                                 self.clusters, 200, 0, 0.9)
        self.assertEqual(flipped, self.delta())

    def test_degenerate_cases(self):
        # Identical predictions: delta and interval 0.
        same = M.paired_delta("auc", self.a, self.a, self.labels, self.t, self.clusters, 100, 0, 0.95)
        self.assertEqual(same, {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0})
        # A single cluster is always resampled whole: the interval is the delta itself.
        one = dict((k, "all") for k in self.clusters)
        d = M.paired_delta("accuracy", self.a, self.b, self.labels, self.t, one, 100, 0, 0.95)
        self.assertEqual(d, {"delta": DELTA, "ci_low": DELTA, "ci_high": DELTA})




class TestCiSettings(unittest.TestCase):
    def test_defaults_and_override(self):
        self.assertEqual(M.CI_DEFAULTS, {"resamples": 1000, "seed": 0, "level": 0.95})
        t = binary_task()
        self.assertEqual(M.ci_settings(t), {"resamples": 1000, "seed": 0, "level": 0.95})
        t["metrics"]["ci"] = {"seed": 3}
        self.assertEqual(check_task(t), [])
        self.assertEqual(M.ci_settings(t), {"resamples": 1000, "seed": 3, "level": 0.95})
        t["metrics"]["ci"] = {"resamples": 200, "seed": 1, "level": 0.9}
        self.assertEqual(check_task(t), [])
        self.assertEqual(M.ci_settings(t), {"resamples": 200, "seed": 1, "level": 0.9})
        self.assertEqual(t["metrics"]["ci"], {"resamples": 200, "seed": 1, "level": 0.9})  # unchanged
        for bad in ({"resamples": 99}, {"resamples": 1.5}, {"seed": "0"}, {"level": 0}, {"level": 1},
                    {"level": 1.5}, {"level": "0.9"}, {"method": "bca"}):
            t = binary_task()
            t["metrics"]["ci"] = bad
            self.assertTrue(check_task(t), bad)

    def test_known_metric_direction_must_match(self):
        t = binary_task()
        t["metrics"]["primary"] = "log_loss"
        errors = check_task(t)
        self.assertTrue(any("higher_is_better" in e and "log_loss" in e for e in errors), errors)
        t["metrics"]["higher_is_better"] = False
        self.assertEqual(check_task(t), [])
        t = binary_task()
        t["promotion_rule"]["guards"][0]["higher_is_better"] = True  # log_loss guard
        errors = check_task(t)
        self.assertTrue(any("guards[0]" in e and "log_loss" in e for e in errors), errors)
        # Unknown metrics keep the direction the task gives.
        t = binary_task()
        t["metrics"]["primary"] = "net_cost"
        t["metrics"]["higher_is_better"] = False
        t["promotion_rule"]["guards"] = []
        self.assertEqual(check_task(t), [])


if __name__ == "__main__":
    unittest.main()
