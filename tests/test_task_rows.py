import copy
import json
import os
import tempfile
import unittest

from fixtures import binary_task, choice_task, full_row, write_jsonl
from rsi import rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "skills", "rsi-orchestrator", "templates", "task.example.json")


class TestTask(unittest.TestCase):
    def test_example_task_valid(self):
        with open(EXAMPLE, encoding="utf-8") as f:
            t = json.load(f)
        self.assertEqual(t["task_id"], "txn-review")
        self.assertEqual(rows.check_task(t), [])

    def test_fixtures_valid(self):
        self.assertEqual(rows.check_task(binary_task()), [])
        self.assertEqual(rows.check_task(choice_task()), [])

    def test_binary_requires_positive(self):
        t = binary_task()
        del t["decision"]["positive"]
        self.assertTrue(rows.check_task(t))
        t["decision"]["positive"] = "maybe"
        self.assertTrue(rows.check_task(t))

    def test_choice_forbids_positive(self):
        t = choice_task()
        t["decision"]["positive"] = "allow"
        self.assertTrue(rows.check_task(t))

    def test_score_labels_must_be_ordered(self):
        t = choice_task()
        d = t["decision"]
        d["type"] = "score"
        d["labels"] = ["0", "1", "2"]
        d["label_meanings"] = {"0": "a", "1": "b", "2": "c"}
        self.assertEqual(rows.check_task(t), [])
        d["labels"] = ["0", "2", "1"]
        self.assertTrue(rows.check_task(t))

    def test_label_meanings_keys(self):
        t = choice_task()
        del t["decision"]["label_meanings"]["review"]
        self.assertTrue(rows.check_task(t))

    def test_guard_metric_must_be_known(self):
        t = binary_task()
        t["promotion_rule"]["guards"][0]["metric"] = "nope"
        self.assertTrue(rows.check_task(t))

    def test_auc_only_for_binary_tasks(self):
        # Task 14 (R41): auc is defined for binary tasks only, wherever the task names it.
        t = choice_task()
        t["metrics"]["primary"] = "auc"
        t["promotion_rule"]["guards"] = []
        errs = rows.check_task(t)
        self.assertTrue(any(e.startswith("$.metrics.primary") and "auc is defined for binary tasks only" in e
                            for e in errs), errs)
        t = choice_task()
        t["metrics"]["others"] = ["log_loss", "auc"]
        t["promotion_rule"]["guards"] = [{"metric": "auc", "higher_is_better": True, "max_regression": 0.01}]
        errs = rows.check_task(t)
        for where in ("$.metrics.others[1]", "$.promotion_rule.guards[0].metric"):
            self.assertTrue(any(e.startswith(where) and "binary tasks only" in e for e in errs), (where, errs))
        t["decision"]["type"] = "score"
        t["decision"]["labels"] = ["0", "1", "2"]
        t["decision"]["label_meanings"] = {"0": "a", "1": "b", "2": "c"}
        self.assertTrue(any("binary tasks only" in e for e in rows.check_task(t)))
        self.assertEqual(rows.check_task(binary_task()), [])  # binary_task's primary is auc

    def test_reproduction_tolerance(self):
        # R43: optional delivery.reproduction_tolerance, a number > 0 (default 1e-6, rsi.rules).
        from rsi.rules import reproduction_tolerance
        t = binary_task()
        self.assertEqual(reproduction_tolerance(t), 1e-6)
        t["delivery"] = {"reproduction_tolerance": 1e-4}
        self.assertEqual(rows.check_task(t), [])
        self.assertEqual(reproduction_tolerance(t), 1e-4)
        for bad in (0, -1e-6, "1e-4", None):
            t["delivery"] = {"reproduction_tolerance": bad}
            self.assertTrue(any("reproduction_tolerance" in e for e in rows.check_task(t)), bad)
        t["delivery"] = {"tolerance": 1e-4}
        self.assertTrue(rows.check_task(t))

    def test_max_decision_mismatch_rate(self):
        # R44: optional delivery.max_decision_mismatch_rate, 0 <= rate < 1, default 0; the cap is a count.
        from rsi.rules import max_decision_mismatches
        t = binary_task()
        self.assertEqual(max_decision_mismatches(t, 1000), 0)
        for ok in (0, 0.3, 0.29):
            t["delivery"] = {"max_decision_mismatch_rate": ok}
            self.assertEqual(rows.check_task(t), [], ok)
        self.assertEqual(max_decision_mismatches(t, 100), 29)  # floor(0.29 * 100), not 28 from rounding
        t["delivery"] = {"max_decision_mismatch_rate": 0.3}
        self.assertEqual(max_decision_mismatches(t, 4), 1)
        for bad in (-0.1, 1, 1.5, "0.1", None):
            t["delivery"] = {"max_decision_mismatch_rate": bad}
            self.assertTrue(any("max_decision_mismatch_rate" in e for e in rows.check_task(t)), bad)

    def test_fields_format_requires_fields(self):
        t = binary_task()
        t["input"] = {"format": "fields", "description": "d"}
        self.assertTrue(rows.check_task(t))
        t["input"]["fields"] = ["a"]
        self.assertEqual(rows.check_task(t), [])

    def test_unknown_property_rejected(self):
        t = binary_task()
        t["extra"] = 1
        self.assertTrue(rows.check_task(t))

    def test_quality_checks_optional_free_form_object(self):
        # R31: thresholds for the data module's quality gate; free-form, optional.
        t = binary_task()
        t["quality_checks"] = {"min_rows": 1000, "max_dup_rate": 0.01, "label_agreement": {"min": 0.8}}
        self.assertEqual(rows.check_task(t), [])
        t["quality_checks"] = ["min_rows"]
        errs = rows.check_task(t)
        self.assertTrue(any("quality_checks" in e for e in errs), errs)


class TestRows(unittest.TestCase):
    def test_target_float_tolerance(self):
        t = binary_task()
        self.assertEqual(rows.check_target({"yes": 0.1, "no": 0.9000001}, t), [])
        self.assertTrue(rows.check_target({"yes": 0.5, "no": 0.49}, t))
        self.assertTrue(rows.check_target({"yes": 0.5, "maybe": 0.5}, t))
        self.assertTrue(rows.check_target({"yes": 1.5, "no": -0.5}, t))

    def test_full_rows_duplicate_id(self):
        t = binary_task()
        errs = rows.check_full_rows([full_row(1, t), full_row(1, t)], t)
        self.assertTrue(any("duplicate" in e for e in errs))

    def test_messages_input_must_be_nonempty(self):
        t = binary_task()
        self.assertTrue(rows.check_input({"messages": []}, t))
        self.assertTrue(rows.check_input({"fields": {}}, t))
        self.assertEqual(
            rows.check_input({"messages": [{"role": "user", "content": "hi"}]}, t), []
        )

    def test_fields_input_must_cover_task_fields(self):
        t = binary_task()
        t["input"] = {"format": "fields", "description": "d", "fields": ["a", "b"]}
        self.assertTrue(rows.check_input({"fields": {"a": 1}}, t))
        self.assertEqual(rows.check_input({"fields": {"a": 1, "b": 2}}, t), [])

    def test_input_and_label_rows(self):
        t = binary_task()
        full = full_row(1, t)
        inp = {"id": full["id"], "input": full["input"],
               "meta": {"group": "g", "cluster": "c"}}
        lab = {"id": full["id"], "target": full["target"],
               "meta": {"source": "s", "label_source": "human", "n_obs": 2}}
        self.assertEqual(rows.check_input_rows([inp], t), [])
        self.assertEqual(rows.check_label_rows([lab], t), [])
        del inp["meta"]["cluster"]
        self.assertTrue(rows.check_input_rows([inp], t))
        lab["meta"]["label_source"] = "guess"
        self.assertTrue(rows.check_label_rows([lab], t))

    def test_predictions_must_cover_exactly(self):
        t = binary_task()

        def p(i, dec="yes"):
            return {"id": i, "probs": {"yes": 0.4, "no": 0.6}, "decision": dec, "uncertainty": None}

        self.assertEqual(rows.check_predictions([p("a"), p("b")], t, {"a", "b"}), [])
        self.assertTrue(rows.check_predictions([p("a")], t, {"a", "b"}))
        self.assertTrue(rows.check_predictions([p("a"), p("b"), p("c")], t, {"a", "b"}))
        self.assertTrue(rows.check_predictions([p("a", "zzz"), p("b")], t, {"a", "b"}))

    def test_feedback_rows(self):
        t = binary_task()
        fb = {
            "id": "a",
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "output": {"probs": {"yes": 0.5, "no": 0.5}, "decision": "yes"},
            "model": "candidate-0001",
            "latency_ms": 12.5,
            "timestamp": "2026-01-01T00:00:00Z",
            "outcome": None,
        }
        self.assertEqual(rows.check_feedback_rows([fb], t), [])
        fb2 = copy.deepcopy(fb)
        fb2["outcome"] = "maybe"
        self.assertTrue(rows.check_feedback_rows([fb2], t))

    def test_fingerprint_ignores_case_and_spaces(self):
        a = {"messages": [{"role": "user", "content": "Hello  World"}]}
        b = {"messages": [{"role": "user", "content": "hello world"}]}
        c = {"messages": [{"role": "user", "content": "other"}]}
        self.assertEqual(rows.input_fingerprint(a), rows.input_fingerprint(b))
        self.assertNotEqual(rows.input_fingerprint(a), rows.input_fingerprint(c))

    def test_overlap(self):
        t = binary_task()
        a = [full_row(1, t), full_row(2, t)]
        b = [full_row(2, t), full_row(3, t, id="q2x")]
        b[1]["input"] = a[0]["input"]
        o = rows.overlap(a, b)
        self.assertEqual(o["ids"], 1)
        self.assertEqual(o["inputs"], 2)
        self.assertEqual(o["clusters"], 1)
        self.assertEqual(rows.overlap([{"id": "x"}], [{"id": "y"}]),
                         {"ids": 0, "inputs": 0, "clusters": 0})

    def test_read_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl")
            write_jsonl(p, [{"id": "a"}, {"id": "b"}])
            self.assertEqual(rows.read_jsonl(p), [{"id": "a"}, {"id": "b"}])
            with open(p, "a", encoding="utf-8") as f:
                f.write("{bad\n")
            with self.assertRaises(ValueError) as cm:
                rows.read_jsonl(p)
            self.assertIn("3", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
