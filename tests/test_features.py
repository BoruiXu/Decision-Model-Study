"""features (modules/common/features): model view, edit view, task adapters, the featurizer and its label-leak check;
the task fields features and input.verbatim."""
import hashlib
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modules", "common"))

from features import FEATURES_VERSION, FeatureError  # noqa: E402
from features.flatten import edit_view, model_view, set_paths  # noqa: E402
from features.table import Featurizer, real_rows  # noqa: E402
from features.adapter import load_adapter  # noqa: E402
from features.constraints import check, repair  # noqa: E402
from fixtures import binary_task, choice_task, full_row  # noqa: E402
from rsi import rows as R  # noqa: E402
from rsi.metrics import hard_label  # noqa: E402
from rsi.rules import adapter_path, features_settings, verbatim_paths  # noqa: E402

# An adapter that records what it is given and changes it (the row must not change).
SEEN_ADAPTER = """
SEEN = []


def features(state):
    SEEN.append(sorted(state))
    state["changed"] = True
    return {"n": len(state) - 1}
"""

# An adapter whose repair records the total it is given and whose check always finds one problem.
OWN_ADAPTER = """
def repair(state):
    return dict(state, seen_total=state["s"]["total"])


def check(state):
    return ["own rule"]
"""


class TestFeatures(unittest.TestCase):
    def test_model_view_is_deterministic_and_aggregates_lists(self):
        s = {"summary": {"a": 1, "b": True, "s": "x"}, "transactions": [{"v": 1.0, "r": "in"}, {"v": 3.0, "r": "out"}]}
        want = {"summary.a": 1.0, "summary.b": 1.0, "transactions.count": 2.0, "transactions[].v.mean": 2.0,
                "transactions[].v.max": 3.0, "transactions[].v.sum": 4.0}
        self.assertEqual(model_view(s), want)
        self.assertEqual(model_view({"transactions": s["transactions"], "summary": {"s": "x", "b": True, "a": 1}}), want)
        s = {"tags": [1, 2, 3, "x"], "t": [{"d": {"w": 2}, "ins": [1, 2]}, {"ins": []}], "m": [[1, 2], [3]],
             "bad": float("nan")}
        self.assertEqual(model_view(s), {"tags.count": 4.0, "tags.mean": 2.0, "tags.max": 3.0, "tags.sum": 6.0,
                                         "t.count": 2.0, "t[].d.w.mean": 2.0, "t[].d.w.max": 2.0, "t[].d.w.sum": 2.0,
                                         "t[].ins.count.mean": 1.0, "m.count": 2.0, "m[].count.mean": 1.5})

    def test_featurizer_sees_only_the_state(self):
        task = binary_task()
        with tempfile.TemporaryDirectory() as d:
            adapter = os.path.join(d, "adapter.py")
            with open(adapter, "w", encoding="utf-8") as f:
                f.write(SEEN_ADAPTER)
            f = Featurizer(adapter)
            row = full_row(1, task)
            cols = f.fit([row])
            self.assertEqual(cols, ["n"])
            seen = f.adapter.module.SEEN
            self.assertEqual(seen, [["messages"]])
            self.assertNotIn("target", seen[0]); self.assertNotIn("id", seen[0])
            self.assertEqual(row, full_row(1, task))                              # a deep copy of the state only
            ids, X = f.transform([full_row(2, task)])
            self.assertEqual((ids, X), ([full_row(2, task)["id"]], [[1.0]]))
            with open(adapter, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            self.assertEqual(f.signature(), {"version": FEATURES_VERSION, "adapter": os.path.abspath(adapter),
                                             "adapter_sha256": sha, "columns": 1})
            not_py = os.path.join(d, "adapter.txt")
            with open(not_py, "w", encoding="utf-8") as fh:
                fh.write(SEEN_ADAPTER)
            for path in (not_py, os.path.join(d, "missing.py")):
                with self.assertRaises(FeatureError) as e:
                    load_adapter(path)
                self.assertIn(os.path.abspath(path), str(e.exception))
        variant = dict(full_row(3, task), meta=dict(full_row(3, task)["meta"], extra={"variant": "v1", "variant_of": "q1"}))
        self.assertEqual([r["id"] for r in real_rows([full_row(1, task), variant])], [full_row(1, task)["id"]])

    def test_a_column_that_equals_the_label_is_refused(self):
        task = choice_task()
        labels = task["decision"]["labels"]
        rows = [full_row(i, task) for i in range(5)]       # allow, block, review, allow, block: review is on one row
        hard_labels = [hard_label(r["target"], labels) for r in rows]
        for i, (r, y) in enumerate(zip(rows, hard_labels)):
            r["state"] = {"leak": 1 if y == "allow" else 0, "x": i}
        with self.assertRaises(FeatureError) as e:
            Featurizer().fit(rows, labels=hard_labels)
        self.assertIn("column leak ", str(e.exception))
        self.assertNotIn("column x ", str(e.exception))   # x is unique on the one review row: no leak (2+ rows needed)
        for r in rows:
            del r["state"]["leak"]
        self.assertEqual(Featurizer().fit(rows, labels=hard_labels), ["x"])

    def test_edit_view_and_set_paths(self):
        s = {"transactions": [{"v": 1.0}, {"v": 3.0}], "req": "keep"}
        self.assertEqual(edit_view(s), {"transactions.0.v": 1.0, "transactions.1.v": 3.0, "req": "keep"})
        t = set_paths(s, {"transactions.1.v": 5.0})
        self.assertEqual(t["transactions"][1]["v"], 5.0); self.assertEqual(s["transactions"][1]["v"], 3.0)
        with self.assertRaises(KeyError):
            set_paths(s, {"transactions.2.v": 1.0})
        with self.assertRaises(FeatureError):
            set_paths(s, {"transactions.0.v": "large"})
        self.assertNotIn("req", Featurizer(verbatim=["req"]).edit_view(s))

    def test_constraints_repair_and_check(self):
        c = [{"sum": "s.total", "of": ["s.a", "s.b"]}, {"order": ["s.min", "s.med", "s.max"]}, {"nonneg": ["t[].v"]}]
        st = {"s": {"a": 2, "b": 3, "total": 9, "min": 5, "med": 1, "max": 4}, "t": [{"v": -1}]}
        self.assertEqual(len(check(st, c)), 3)
        fixed = repair(st, c)
        self.assertEqual(check(fixed, c), [])
        self.assertEqual((fixed["s"]["total"], [fixed["s"][k] for k in ("min", "med", "max")], fixed["t"][0]["v"]),
                         (5, [1, 4, 5], 0))
        self.assertEqual(st["s"]["total"], 9)                                    # a copy is repaired
        # integer rounding first, then the relations twice: the sum over t's elements sees zero_if_zero's repair
        c2 = [{"integer": ["n"]}, {"sum": "total", "of": ["t[].v"]}, {"zero_if_zero": "t[].k", "then": ["t[].v"]}]
        fixed = repair({"n": 2.6, "total": 0, "t": [{"k": 0, "v": 4}, {"k": 1, "v": 2.5}]}, c2)
        self.assertEqual((fixed["n"], fixed["total"], [x["v"] for x in fixed["t"]]), (3, 2.5, [0, 2.5]))
        self.assertEqual(check(fixed, c2), [])
        missing = check({"n": 1, "total": 0, "t": [{"v": 0}]}, c2)              # no k in t's element: not repaired
        self.assertEqual(len(missing), 1); self.assertIn("t.0.k", missing[0])
        # the adapter: the task's constraints first, then its own functions; a malformed item is refused on load
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "adapter.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(OWN_ADAPTER)
            a = load_adapter(path, c)
            self.assertEqual(a.repair(st)["seen_total"], 5)
            self.assertEqual(a.check(st)[3:], ["own rule"]); self.assertEqual(len(a.check(st)), 4)
        with self.assertRaises(FeatureError) as e:
            load_adapter(None, [{"sum": "s.total"}])
        self.assertIn("constraints[0]", str(e.exception))

    def test_task_fields(self):
        task = binary_task()
        self.assertEqual((features_settings(task), adapter_path(task), verbatim_paths(task)),
                         ({"adapter": None, "constraints": []}, None, []))
        task["features"] = {"adapter": "examples/elliptic_onejev/features_adapter.py",
                            "constraints": [{"sum": "s.total", "of": ["s.a", "s.b"]}, {"order": ["s.min", "s.max"]}]}
        task["input"]["verbatim"] = ["messages[].content"]
        self.assertEqual(R.check_task(task), [])
        bad = dict(task, features={"constraints": [{"order": ["s.min"]}, {"max": ["s.a"]}]})
        self.assertEqual(len(R.check_task(bad)), 2)                   # one path to order; no constraint named max
        self.assertEqual(adapter_path(task), os.path.join(ROOT, "examples", "elliptic_onejev", "features_adapter.py"))
        state = full_row(0, task)["state"]
        self.assertEqual(Featurizer(verbatim=verbatim_paths(task)).edit_view(state), {"messages.0.role": "user"})
        task["features"]["adapter"] = "examples/no_such_adapter.py"
        errors = R.check_task(task)
        self.assertEqual(len(errors), 1)
        self.assertIn(os.path.join(ROOT, "examples", "no_such_adapter.py"), errors[0])


if __name__ == "__main__":
    unittest.main()
