"""`rsi score`: the shared metric definitions on files, without a workspace (Task 13)."""
import json
import os
import tempfile
import unittest

from fixtures import binary_task, choice_task, full_row, input_row, label_row, prediction_row, write_jsonl
from rsi import metrics as M
from test_cli import run_rsi

VAL, TRAIN = range(10, 16), range(0, 6)


class ScoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task = binary_task()

    def write(self, name, rows):
        p = os.path.join(self.tmp.name, name)
        write_jsonl(p, rows)
        return p

    def task_file(self):
        p = os.path.join(self.tmp.name, "task.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.task, f)
        return p

    def score(self, *args):
        return run_rsi("score", "--task", self.task_file(), *args)

    def ok(self, *args):
        r = self.score(*args)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def preds(self, ids, wrong=()):
        return [prediction_row(i, self.task, wrong=i in wrong) for i in ids]


class TestScore(ScoreCase):
    def test_small_case(self):
        t = self.task
        labels = [full_row(i, t) for i in VAL]
        train = [full_row(i, t) for i in TRAIN]
        preds = self.preds(VAL, wrong=(10,))
        out = self.ok("--labels", self.write("rows.jsonl", labels), "--predictions", self.write("p.jsonl", preds),
                      "--train", self.write("train.jsonl", train))
        self.assertEqual(sorted(out), ["baselines", "metrics", "not_computed"])
        # Every known metric among primary and others, computed by rsi.metrics.
        self.assertEqual(out["metrics"], {"auc": M.auc(preds, labels, t), "log_loss": M.log_loss(preds, labels, t)})
        self.assertEqual(out["metrics"]["auc"], 7.5 / 9)
        majority = M.majority(train, labels, t)
        self.assertEqual(out["baselines"], {"majority": {"auc": 0.5, "log_loss": M.log_loss(majority, labels, t)}})
        self.assertEqual(out["not_computed"], [])
        # labels.jsonl rows give the same metrics as full rows.
        out2 = self.ok("--labels", self.write("labels.jsonl", [label_row(i, t) for i in VAL]),
                       "--predictions", self.write("p.jsonl", preds))
        self.assertEqual(out2["metrics"], out["metrics"])
        self.assertNotIn("baselines", out2)  # baselines need --train

    def test_unknown_names_are_listed_not_computed(self):
        self.task = choice_task()
        self.task["metrics"]["others"] = ["log_loss", "block_recall", "macro_f1"]
        self.task["metrics"]["baselines"] = ["majority", "category-prior"]
        t = self.task
        labels = [full_row(i, t) for i in VAL]
        out = self.ok("--labels", self.write("rows.jsonl", labels), "--predictions",
                      self.write("p.jsonl", self.preds(VAL)), "--train",
                      self.write("train.jsonl", [full_row(i, t) for i in TRAIN]))
        self.assertEqual(sorted(out["metrics"]), ["accuracy", "log_loss", "macro_f1"])
        self.assertEqual(sorted(out["baselines"]), ["majority"])
        self.assertEqual(sorted(out["baselines"]["majority"]), ["accuracy", "log_loss", "macro_f1"])
        self.assertEqual(out["not_computed"], ["block_recall", "category-prior"])

    def test_group_prior_needs_groups(self):
        self.task["metrics"]["baselines"] = ["group_prior"]
        t = self.task
        args = ["--predictions", self.write("p.jsonl", self.preds(VAL)),
                "--train", self.write("train.jsonl", [full_row(i, t) for i in TRAIN])]
        labels = self.write("labels.jsonl", [label_row(i, t) for i in VAL])
        r = self.score("--labels", labels, *args)
        self.assertEqual(r.returncode, 1)
        self.assertIn("meta.group", r.stderr)
        # The inputs file given with --clusters supplies meta.group.
        inputs = self.write("inputs.jsonl", [input_row(i, t) for i in VAL])
        out = self.ok("--labels", labels, "--clusters", inputs, *args)
        full = self.ok("--labels", self.write("rows.jsonl", [full_row(i, t) for i in VAL]), *args)
        self.assertEqual(out["baselines"], full["baselines"])
        self.assertEqual(out["baselines"]["group_prior"]["auc"], 1.0)  # groups g0/g1 follow the labels

    def test_coverage_refused(self):
        t = self.task
        labels = self.write("labels.jsonl", [label_row(i, t) for i in VAL])
        r = self.score("--labels", labels, "--predictions", self.write("p.jsonl", self.preds(list(VAL)[:-1])))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("missing predictions for 1 ids", r.stderr)
        self.assertIn("--predictions", r.stderr)
        r = self.score("--labels", labels, "--predictions", self.write("p.jsonl", self.preds(list(VAL) + [99])))
        self.assertEqual(r.returncode, 1)
        self.assertIn("unexpected predictions for 1 ids", r.stderr)
        r = self.score("--labels", labels, "--predictions", self.write("p.jsonl", self.preds(VAL)),
                       "--compare", self.write("c.jsonl", self.preds(list(VAL)[1:])))
        self.assertEqual(r.returncode, 1)
        self.assertIn("--compare", r.stderr)

    def test_compare_delta(self):
        t = self.task
        labels = [label_row(i, t) for i in VAL]
        a, b = self.preds(VAL, wrong=(10,)), self.preds(VAL, wrong=(11, 12))
        inputs = [input_row(i, t) for i in VAL]
        for r in inputs:  # two rows per cluster
            r["meta"]["cluster"] = "k%d" % (int(r["id"][1:]) // 2)
        args = ["--labels", self.write("labels.jsonl", labels), "--predictions", self.write("a.jsonl", a),
                "--compare", self.write("b.jsonl", b)]
        out = self.ok("--clusters", self.write("inputs.jsonl", inputs), *args)
        want = M.paired_delta("auc", a, b, labels, t, M.clusters_of(inputs), 1000, 0, 0.95)
        self.assertEqual(out["delta"], want)
        self.assertGreater(out["delta"]["delta"], 0)
        # Without --clusters, label rows have no meta.cluster: each row is its own cluster.
        alone = self.ok(*args)
        self.assertEqual(alone["delta"], M.paired_delta("auc", a, b, labels, t, {}, 1000, 0, 0.95))
        self.assertNotEqual(alone["delta"], out["delta"])
        # The task's metrics.ci settings are used.
        self.task["metrics"]["ci"] = {"resamples": 200, "seed": 5, "level": 0.8}
        out = self.ok("--clusters", self.write("inputs.jsonl", inputs), *args)
        self.assertEqual(out["delta"], M.paired_delta("auc", a, b, labels, t, M.clusters_of(inputs), 200, 5, 0.8))

    def test_compare_needs_known_primary(self):
        self.task["metrics"]["primary"] = "expected_cost"
        self.task["metrics"]["higher_is_better"] = False
        self.task["promotion_rule"]["guards"] = []
        t = self.task
        args = ["--labels", self.write("labels.jsonl", [label_row(i, t) for i in VAL]),
                "--predictions", self.write("a.jsonl", self.preds(VAL))]
        self.assertEqual(self.ok(*args)["not_computed"], ["expected_cost"])
        r = self.score(*(args + ["--compare", self.write("b.jsonl", self.preds(VAL))]))
        self.assertEqual(r.returncode, 1)
        self.assertIn("expected_cost", r.stderr)

    def test_bad_inputs_refused(self):
        t = self.task
        p = self.write("p.jsonl", self.preds(VAL))
        bad = [label_row(i, t) for i in VAL]
        bad[0]["target"] = "yes"
        r = self.score("--labels", self.write("bad.jsonl", bad), "--predictions", p)
        self.assertEqual(r.returncode, 1)
        self.assertIn("--labels", r.stderr)
        r = self.score("--labels", os.path.join(self.tmp.name, "nope.jsonl"), "--predictions", p)
        self.assertEqual(r.returncode, 1)
        self.assertIn("--labels", r.stderr)
        r = run_rsi("score", "--labels", p, "--predictions", p)  # --task is required
        self.assertEqual(r.returncode, 2)
        # Rows of the --clusters file without an id are ignored (those rows are their own cluster).
        labels = self.write("labels.jsonl", [label_row(i, t) for i in VAL])
        out = self.ok("--labels", labels, "--predictions", p, "--compare", p,
                      "--clusters", self.write("c.jsonl", [{"meta": {"cluster": "k", "group": "g"}}]))
        self.assertEqual(out["delta"], {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0})


if __name__ == "__main__":
    unittest.main()
