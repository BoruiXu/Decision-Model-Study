"""Prompt variants (rsi.variants) as checked rewrites of their seeds, and the held-out check of generated rows when a
trainset is registered."""
import copy
import os
import tempfile
import unittest

from fixtures import FakeEngine, binary_task, full_row, make_artifact, make_task_dir
from rsi.rows import question_of
from rsi.variants import apply_spec, check_spec, check_variant_rows, invert_state
from rsi.workspace import RsiError, Workspace

SPEC = {"id": "v1", "instructions": "Does this address take part in illicit activity?", "criteria": None,
        "rename": {"summary.txs_sent": "transactions sent"}, "tables": ["transactions"]}
TRAIN, VAL, EVAL, TEST = range(0, 4), range(10, 14), range(20, 24), range(30, 34)


def object_task():
    t = binary_task()
    t["task_id"] = "address-risk"
    t["decision"].update(instructions="Is this address involved in illicit activity?",
                         label_meanings={"yes": "the address is illicit", "no": "the address is licit"})
    t["input"] = {"format": "object", "description": "An address: a summary and its transactions."}
    return t


def row(i, task):
    """q<i> of fixtures.full_row with a numeric state: a summary and two transactions."""
    return full_row(i, task, state={"summary": {"txs_sent": i + 1, "volume": 12.5 + i},
                                    "transactions": [{"amount": 1.25 + i, "step": 3}, {"amount": 7.5, "step": 4 + i}]})


def nudge(state, eps):
    """The state with eps added to every number."""
    if isinstance(state, dict):
        return dict((k, nudge(v, eps)) for k, v in state.items())
    if isinstance(state, list):
        return [nudge(v, eps) for v in state]
    if isinstance(state, (int, float)) and not isinstance(state, bool):
        return state + eps
    return state


def check_named(m, name):
    return [c for c in m["quality"]["checks"] if c["name"] == name][0]


class TestVariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.task = object_task()
        self.base = [row(i, self.task) for i in TRAIN]
        self.n = 0

    def src(self, kind, rows, **fields):
        self.n += 1
        return make_artifact(kind, os.path.join(self.root, "src", "%s-%d" % (kind, self.n)), self.task, rows=rows,
                             notes="source %d" % self.n, **fields)

    def workspace(self, sets=True):
        ws = Workspace.init(os.path.join(self.root, "ws"), make_task_dir(os.path.join(self.root, "src", "task"),
                                                                          self.task))
        for kind, ids in (("testset", TEST), ("evalset", EVAL), ("valset", VAL), ("trainset", TRAIN)) if sets else ():
            ws.register(kind, self.src(kind, [row(i, self.task) for i in ids]))
        return ws

    def test_variant_rows_register_with_their_seed(self):
        ws = self.workspace()
        v = apply_spec(self.base[0], SPEC, self.task)
        self.assertEqual((v["id"], v["meta"]["extra"]), ("q0~v1", {"variant": "v1", "variant_of": "q0"}))
        self.assertEqual(v["state"], {"summary": {"transactions sent": 1, "volume": 12.5},
                                      "transactions": {"columns": ["amount", "step"], "rows": [[1.25, 3], [7.5, 4]]}})
        self.assertEqual(v["question"], dict(question_of(self.task), instructions=SPEC["instructions"]))
        self.assertEqual((v["target"], invert_state(v["state"], SPEC)), (self.base[0]["target"], self.base[0]["state"]))
        tid = ws.register("trainset", self.src("trainset", self.base + [v], prompt_variants=[SPEC]))
        self.assertEqual((tid, ws.current("trainset")), ("trainset-0002", "trainset-0002"))
        with self.assertRaises(RsiError) as e:                               # its spec must be listed
            ws.register("trainset", self.src("trainset", self.base + [v]))
        self.assertIn("prompt_variants", str(e.exception))

    def test_a_variant_that_changes_content_is_refused(self):
        task, base = self.task, self.base
        v = apply_spec(base[0], SPEC, task)
        v["state"]["transactions"]["rows"][0][0] += 1.0
        errs = check_variant_rows(base + [v], [SPEC], task)
        self.assertIn("state is not a rewrite of its seed", errs[0])
        v2 = apply_spec(base[0], SPEC, task)
        v2["target"] = {"yes": 0.0, "no": 1.0}                               # q0 is a yes row
        self.assertIn("target", check_variant_rows(base + [v2], [SPEC], task)[0])
        routing = binary_task()
        routing["input"]["verbatim"] = ["messages[].content"]
        on_messages = {"id": "m1", "instructions": "Is this message unwanted bulk mail?",
                       "criteria": {"true": "it is unwanted bulk mail", "false": "it is a normal message"},
                       "rename": None, "tables": None}
        seed = full_row(0, routing)
        v3 = apply_spec(seed, on_messages, routing)
        v3["state"]["messages"][0]["content"] += "!"
        self.assertIn("messages[].content", check_variant_rows([seed, v3], [on_messages], routing)[0])
        self.assertIn("seed", check_variant_rows([apply_spec(base[0], SPEC, task)], [SPEC], task)[0])
        self.assertEqual(check_variant_rows(base + [apply_spec(base[1], SPEC, task)], [SPEC], task), [])
        self.assertEqual((check_spec(SPEC, task), check_spec(on_messages, routing)), ([], []))
        for spec, t, why in ((dict(SPEC, criteria={"yes": "illicit"}), task, "criteria"),
                             (dict(SPEC, rename={"transactions[]": "tx"}), task, "names no object key"),
                             (dict(SPEC, rename={"summary.txs_sent": "n", "summary.volume": "n"}), task, "unique"),
                             (dict(on_messages, rename={"messages[].role": "speaker"}), routing, "input format"),
                             (dict(on_messages, tables=["messages"]), routing, "verbatim"),
                             (dict(SPEC, instructions=None, rename=None, tables=None), task, "changes nothing")):
            self.assertIn(why, "; ".join(check_spec(spec, t)), spec)
        with self.assertRaises(ValueError):                                  # a table of items with other keys
            apply_spec(dict(base[0], state={"transactions": [{"amount": 1}, {"step": 2}]}), SPEC, task)

    def test_generated_rows_matching_heldout_inputs_fail_the_quality_gate(self):
        ws = self.workspace()
        gen = copy.deepcopy(self.base[1])                                    # dgen:smote, its state an evalset input's
        gen.update(id="syn0a1b2c_smote_000000", state=nudge(row(20, self.task)["state"], 1e-7))
        gen["meta"].update(source="dgen:smote", extra={"seed_id": "q1", "recipe": "0a1b2c3d4e5f"})
        tid = ws.register("trainset", self.src("trainset", self.base + [gen]))
        m = ws.manifest(tid)
        self.assertFalse(m["quality"]["passed"])
        self.assertEqual(check_named(m, "generated_matches_heldout"),
                         {"name": "generated_matches_heldout", "value": 1, "threshold": 0, "passed": False})
        self.assertNotEqual(ws.current("trainset"), tid)
        self.assertEqual(ws.summary(tid), "quality gate: failed (generated_matches_heldout); never current")
        gates = "\n".join(FakeEngine(ws)._gates(ws, [tid]))                  # the data module's feedback
        self.assertIn(gen["id"], gates)
        self.assertNotIn("q20", gates)
        real = copy.deepcopy(self.base[2])                                   # a real row, near-identical to a testset's
        real.update(id="q5", state=nudge(row(30, self.task)["state"], 1e-7))
        self.assertTrue(ws.manifest(ws.register("trainset", self.src("trainset", self.base + [real])))["quality"]
                        ["passed"])
        var = apply_spec(real, SPEC, self.task)                              # a variant: the state it inverts to
        tid = ws.register("trainset", self.src("trainset", self.base + [real, var], prompt_variants=[SPEC]))
        self.assertEqual((check_named(ws.manifest(tid), "generated_matches_heldout")["value"],
                          ws._generated_heldout_matches(ws.manifest(tid), ws.path(tid))), (1, ["q5~v1"]))

    def test_only_trainsets_may_hold_variant_rows(self):
        ws = self.workspace(sets=False)
        rows = [row(i, self.task) for i in VAL]
        with self.assertRaises(RsiError) as e:
            ws.register("valset", self.src("valset", rows + [apply_spec(rows[0], SPEC, self.task)]))
        self.assertIn("variant", str(e.exception))


if __name__ == "__main__":
    unittest.main()
