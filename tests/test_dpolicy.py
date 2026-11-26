"""dpolicy (modules/common/dpolicy): threshold rules, calibrations, the self-contained apply file, the policy fitted on
the valset (fit and select splits, the identity policy and its reasons); the task fields policy and metrics.costs."""
import ast
import importlib.util
import os
import random
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modules", "common"))

from dpolicy import calibrate  # noqa: E402
from dpolicy import apply as policy_apply  # noqa: E402
from dpolicy.apply import apply_policy  # noqa: E402
from dpolicy.fit import fit_policy, split_rows  # noqa: E402
from dpolicy.rules import choose_threshold  # noqa: E402
from fixtures import binary_task, choice_task  # noqa: E402
from rsi.rows import check_task  # noqa: E402
from rsi.rules import policy_settings  # noqa: E402

INF = float("inf")


def val_row(i, label, step=None):
    """Valset row v<i> (its own group and cluster) of the hard label `label` (yes or no)."""
    meta = {"state_id": "s%d" % i, "cluster": "c%d" % i}
    if step is not None:
        meta["extra"] = {"step": step}
    return {"id": "v%d" % i, "target": {"yes": float(label == "yes"), "no": float(label == "no")}, "meta": meta}


def raw(rid, p_yes):
    """A raw prediction row (the most probable label as its decision)."""
    probs = {"yes": p_yes, "no": 1.0 - p_yes}
    d = "yes" if p_yes >= 0.5 else "no"
    return {"id": rid, "probs": probs, "decision": d, "uncertainty": 1.0 - probs[d]}


def load_by_path(path):
    spec = importlib.util.spec_from_file_location("copied_policy_apply", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRulesAndCalibrations(unittest.TestCase):
    def test_threshold_rules_on_a_small_table(self):
        p, y = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2], [1, 1, 0, 1, 0, 0]
        self.assertEqual(choose_threshold({"rule": "max_f1"}, p, y, None), 0.4)
        self.assertEqual(choose_threshold({"rule": "precision_at", "target": 1.0}, p, y, None), 0.8)
        self.assertEqual(choose_threshold({"rule": "recall_at", "target": 1.0}, p, y, None), 0.4)
        self.assertEqual(choose_threshold({"rule": "cost", "cost_fp": 1, "cost_fn": 10}, p, y, None), 0.4)
        # the costs of metrics.costs when the rule has none of its own
        self.assertEqual(choose_threshold({"rule": "cost"}, p, y, {"cost_fp": 1, "cost_fn": 10}), 0.4)
        self.assertEqual(choose_threshold({"rule": "top_frac", "frac": 0.5}, p, y, None), 0.7)    # a distinct score
        self.assertEqual(choose_threshold({"rule": "fixed", "value": 0.65}, p, y, None), 0.65)
        self.assertEqual(choose_threshold({"rule": "argmax"}, p, y, None), 0.5)
        # no cut reaches the precision, or every cut costs more than predicting none: never positive
        self.assertEqual(choose_threshold({"rule": "precision_at", "target": 1.0}, [0.9, 0.1], [0, 1], None), INF)
        self.assertEqual(choose_threshold({"rule": "cost", "cost_fp": 10, "cost_fn": 1}, [0.9, 0.1], [0, 1], None),
                         INF)

    def test_calibrations_and_the_copied_apply_file(self):
        # scores of 0.95 and 0.05 that are right only 7 times in 10: temperature softens them, platt flattens them
        p = [0.95] * 10 + [0.05] * 10
        y = [1] * 7 + [0] * 3 + [1] * 3 + [0] * 7
        t = calibrate.fit("temperature", p, y)
        self.assertGreater(t["T"], 1.0)
        q = calibrate.apply("temperature", t, [0.95, 0.05])
        self.assertAlmostEqual(q[0], 0.7, places=3)
        self.assertAlmostEqual(q[0] + q[1], 1.0, places=9)
        a = calibrate.fit("platt", p, y)
        self.assertAlmostEqual(calibrate.apply("platt", a, [0.95])[0], 0.7, places=4)
        rng = random.Random(1)
        p = [rng.random() for _ in range(60)]
        y = [int(rng.random() < x) for x in p]
        iso = calibrate.fit("isotonic", p, y)
        q = calibrate.apply("isotonic", iso, sorted(p) + [-1.0, 2.0])
        self.assertEqual(q[:-2], sorted(q[:-2]))
        self.assertEqual(q[-2:], [iso["y"][0], iso["y"][-1]])                   # clipped at both ends
        self.assertEqual(calibrate.apply("none", {}, [0.3]), [0.3])
        # a multiclass temperature: softmax(log p / T) on each row
        mt = calibrate.fit("temperature", [[0.9, 0.05, 0.05]] * 6 + [[0.05, 0.9, 0.05]] * 6,
                           [0, 0, 0, 1, 1, 2, 1, 1, 1, 0, 0, 2])
        self.assertGreater(mt["T"], 1.0)
        self.assertAlmostEqual(sum(calibrate.apply("temperature", mt, [[0.9, 0.05, 0.05]])[0]), 1.0, places=12)

        # the apply file is self-contained: only math and json, and a copy outside the repository gives the same rows
        imported = set()
        with open(policy_apply.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"math", "json"})
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        copied = load_by_path(shutil.copy(policy_apply.__file__, os.path.join(tmp, "policy_apply.py")))
        pol = {"version": 1, "labels": ["yes", "no"], "calibration": {"method": "platt", "params": a},
               "decision": {"rule": "max_f1", "threshold": 0.3, "positive": "yes"}}
        row = raw("r", 0.2)
        got = copied.apply_policy(pol, row, ["yes", "no"])
        self.assertEqual(got, apply_policy(pol, row, ["yes", "no"]))
        self.assertEqual((got["decision"], got["uncertainty"]), ("yes", 1.0 - got["probs"]["yes"]))
        self.assertLess(got["probs"]["yes"], 0.5)                               # positive below one half: the rule
        self.assertEqual(row, raw("r", 0.2))                                    # the row given is not changed
        never = dict(pol, decision={"rule": "cost", "threshold": None, "positive": "yes"})   # null: never positive
        self.assertEqual(apply_policy(never, raw("r", 0.99), ["yes", "no"])["decision"], "no")


class TestFitPolicy(unittest.TestCase):
    def test_policy_moves_the_threshold_when_it_helps(self):
        # positives score 0.30-0.45, negatives 0.05-0.25: argmax never says yes; a threshold near 0.3 is right
        rng = random.Random(3)
        task = binary_task()
        task["policy"] = {}
        rows, preds = [], []
        for i in range(120):
            label = "yes" if i % 4 == 0 else "no"
            rows.append(val_row(i, label))
            preds.append(raw("v%d" % i, rng.uniform(0.30, 0.45) if label == "yes" else rng.uniform(0.05, 0.25)))
        settings = policy_settings(task)
        pol, rep = fit_policy(task, settings, rows, preds)
        self.assertTrue(pol["accepted"], pol["reason"])
        self.assertEqual(pol["objective"], "f1")
        self.assertEqual((pol["calibration"]["method"], pol["decision"]["rule"]), ("none", "max_f1"))
        self.assertLess(pol["decision"]["threshold"], 0.5)
        self.assertGreater(pol["select"]["ci_low"], 0)
        self.assertEqual(pol["select"]["n_fit"] + pol["select"]["n_select"], 120)
        self.assertEqual(len(rep["pairs"]), 4)                                 # each calibration with max_f1
        self.assertEqual(fit_policy(task, settings, rows, preds), (pol, rep))   # deterministic
        applied = [apply_policy(pol, r, ["yes", "no"]) for r in preds]
        self.assertTrue(all(a["decision"] == ("yes" if a["probs"]["yes"] >= pol["decision"]["threshold"] else "no")
                            for a in applied))

    def test_a_policy_never_trades_the_primary_metric(self):
        # positives score 0.20-0.45 and negatives 0.02-0.30: at the fixed threshold 0.5, isotonic gains most on f1, but
        # its step function ties scores and lowers auprc, the task's primary metric; platt (monotone) is chosen
        rng = random.Random(0)
        task = binary_task()
        task["metrics"]["primary"] = "auprc"
        task["policy"] = {"calibrations": ["none", "platt", "isotonic"], "rules": [{"rule": "fixed", "value": 0.5}]}
        rows, preds = [], []
        for i in range(120):
            label = "yes" if i % 3 == 0 else "no"
            rows.append(val_row(i, label))
            preds.append(raw("v%d" % i, rng.uniform(0.2, 0.45) if label == "yes" else rng.uniform(0.02, 0.3)))
        pol, rep = fit_policy(task, policy_settings(task), rows, preds)
        none, platt, iso = rep["pairs"]
        self.assertGreater(iso["improvement"], platt["improvement"])           # isotonic wins on f1 ...
        self.assertGreater(iso["low"], 0)                                       # ... with a lower bound above 0
        self.assertFalse(iso["accepted"])
        self.assertIn("worsens the task's primary metric auprc", iso["rejected"])
        self.assertLess(iso["primary"], rep["primary"]["identity"])
        self.assertTrue(pol["accepted"])
        self.assertEqual(pol["calibration"]["method"], "platt")
        self.assertIn("the primary metric auprc is not worse", pol["reason"])
        self.assertFalse(none["accepted"])                                      # the identity itself: no gain

    def test_a_split_without_a_class_gives_the_identity_policy(self):
        # Review Focus 5: rare positives, all in the early steps; the select part (the latest steps) has none
        task = binary_task()
        task["data"]["split"] = {"by": "time", "key": "meta.extra.step"}
        task["policy"] = {}
        settings = policy_settings(task)
        self.assertEqual(settings["split"], {"by": "time", "key": "meta.extra.step", "select_fraction": 0.4})
        rows = [val_row(i, "yes" if i < 3 else "no", step=1 + i // 10) for i in range(50)]   # steps 1..5
        preds = [raw(r["id"], 0.9 if r["target"]["yes"] else 0.1) for r in rows]
        fit_ids, select_ids = split_rows(rows, settings["split"])
        self.assertEqual(len(select_ids), 20)                                   # round(5 x 0.4) steps: 4 and 5
        self.assertEqual(set(select_ids), set("v%d" % i for i in range(30, 50)))
        pol, _ = fit_policy(task, settings, rows, preds)
        self.assertFalse(pol["accepted"])
        self.assertIn("the select split lacks label yes", pol["reason"])
        self.assertEqual((pol["calibration"]["method"], pol["decision"]["rule"]), ("none", "argmax"))
        self.assertEqual([apply_policy(pol, r, ["yes", "no"])["decision"] for r in preds[:4]], ["yes"] * 3 + ["no"])
        # the positives only in the latest step: the fit part lacks them; a single step: the fit part is empty
        late = [val_row(i, "yes" if i >= 47 else "no", step=1 + i // 10) for i in range(50)]
        self.assertIn("the fit split lacks label yes", fit_policy(task, settings, late, preds)[0]["reason"])
        one = [val_row(i, "yes" if i % 2 else "no", step=7) for i in range(6)]
        pol, _ = fit_policy(task, settings, one, [raw(r["id"], 0.5) for r in one])
        self.assertFalse(pol["accepted"])
        self.assertIn("the fit split lacks label yes", pol["reason"])
        self.assertIsNone(fit_policy(task, settings, [], [])[0]["select"]["delta"])

    def test_task_policy_settings_and_checks(self):
        task = binary_task()
        self.assertIsNone(policy_settings(task))                                # no policy section: none
        task["policy"] = {}
        self.assertEqual(check_task(task), [])
        self.assertEqual(policy_settings(task), {
            "objective": "f1", "calibrations": ["none", "platt", "temperature", "isotonic"],
            "rules": [{"rule": "max_f1"}], "split": {"by": "group", "key": "meta.state_id", "select_fraction": 0.4},
            "min_lo": 0})
        task["policy"] = {"objective": "expected_cost"}
        task["metrics"]["costs"] = {"yes": {"yes": 0, "no": 10}, "no": {"yes": 1, "no": 0}}
        self.assertEqual(check_task(task), [])
        self.assertEqual(policy_settings(task)["rules"], [{"rule": "cost"}])
        ct = choice_task()
        ct["policy"] = {}
        self.assertEqual((policy_settings(ct)["objective"], policy_settings(ct)["calibrations"],
                          policy_settings(ct)["rules"]), ("log_loss", ["none", "temperature"], [{"rule": "argmax"}]))
        for t, change, error in (
                (binary_task(), {"policy": {"objective": "recall"}}, "$.policy.objective: 'recall' is not a metric"),
                (binary_task(), {"policy": {"objective": "expected_cost"}}, "expected_cost needs metrics.costs"),
                (binary_task(), {"metrics.costs": {"yes": {"yes": 0, "no": 1}}}, "$.metrics.costs: the keys"),
                (binary_task(), {"policy": {"rules": [{"rule": "top_frac"}]}}, "top_frac needs frac"),
                (choice_task(), {"policy": {"rules": [{"rule": "max_f1"}]}}, "max_f1 is a threshold rule"),
                (choice_task(), {"policy": {"calibrations": ["platt"]}}, "platt is for binary tasks only")):
            for key, value in change.items():
                if key == "metrics.costs":
                    t["metrics"]["costs"] = value
                else:
                    t[key] = value
            errors = check_task(t)
            self.assertTrue(any(error in e for e in errors), (error, errors))


if __name__ == "__main__":
    unittest.main()
