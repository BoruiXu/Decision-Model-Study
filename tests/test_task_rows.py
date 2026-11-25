import copy
import json
import os
import tempfile
import unittest

from fixtures import TIME, binary_task, choice_task, full_row, write_jsonl
from rsi import datasplit as DS
from rsi import rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "skills", "rsi-orchestrator", "templates", "task.example.json")


def _score_task(labels):
    t = choice_task()
    t["decision"].update(type="score", labels=labels, label_meanings={"0": "a", "1": "b", "2": "c"})
    return t


def _set(path, value):
    """A change of a task: set the nested key `path` to `value`, or delete it (DELETE)."""
    def change(t):
        *head, last = path
        for k in head:
            t = t[k]
        if value is DELETE:
            del t[last]
        else:
            t[last] = value
    return change


DELETE = object()
FIELDS_INPUT = {"format": "fields", "description": "d"}


def _object_task():
    """binary_task whose rows' state is a free-form non-empty object (input.format object, R9)."""
    t = binary_task()
    t["input"] = {"format": "object", "description": "an address: entity, summary and transactions"}
    return t


class TestTask(unittest.TestCase):
    def test_valid_tasks(self):
        with open(EXAMPLE, encoding="utf-8") as f:
            t = json.load(f)
        self.assertEqual(t["task_id"], "txn-review")
        fields = binary_task()
        fields["input"] = dict(FIELDS_INPUT, fields=["a"])
        quality = binary_task()
        quality["quality_checks"] = {"min_rows": 1000, "max_dup_rate": 0.01, "label_agreement": {"min": 0.8}}
        for name, task in (("example", t), ("binary fixture", binary_task()), ("choice fixture", choice_task()),
                           ("score labels in order", _score_task(["0", "1", "2"])), ("fields input", fields),
                           ("object input (R9)", _object_task()), ("quality_checks free-form (R31)", quality)):
            with self.subTest(name):
                self.assertEqual(rows.check_task(task), [])

    def test_invalid_tasks(self):
        cases = (
            ("binary without positive", binary_task, _set(("decision", "positive"), DELETE), None),
            ("binary positive not a label", binary_task, _set(("decision", "positive"), "maybe"), None),
            ("choice with positive", choice_task, _set(("decision", "positive"), "allow"), None),
            ("score labels out of order", lambda: _score_task(["0", "2", "1"]), lambda t: None, None),
            ("label_meanings misses a label", choice_task, _set(("decision", "label_meanings", "review"), DELETE),
             None),
            ("unknown guard metric", binary_task, lambda t: t["promotion_rule"]["guards"][0].update(metric="nope"),
             None),
            ("fields format without fields", binary_task, _set(("input",), dict(FIELDS_INPUT)), None),
            ("object format with input.fields", _object_task, _set(("input", "fields"), ["a"]),
             "only allowed when format is fields"),
            ("unknown property", binary_task, _set(("extra",), 1), None),
            ("quality_checks not an object", binary_task, _set(("quality_checks",), ["min_rows"]), "quality_checks"),
        )
        for name, make, change, word in cases:
            with self.subTest(name):
                t = make()
                change(t)
                errors = rows.check_task(t)
                self.assertTrue(errors)
                if word:
                    self.assertTrue(any(word in e for e in errors), errors)

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

    def test_search_space_and_training_settings(self):
        # D5, D9, R14a: the training-tuning module's task fields: search_space (config path -> a list of allowed
        # values, {min, max}, or {values, dead} with dead optional), search, resources.onejev (any object) and
        # resources.gpu_ids.
        t = binary_task()
        t["search_space"] = {"base": ["joey-1"], "recipe": ["recipes/a.yaml", "recipes/b.yaml"],
                             "overrides.onejev.lr": {"values": [5e-6, 1e-5], "dead": {"5e-06": "diverged in round 1"}},
                             "overrides.onejev.warmup_frac": {"values": [0.03, 0.1]},
                             "overrides.lora.r": {"min": 4, "max": 64}, "seed": [0, 1, 2],
                             "max_steps": {"min": 10, "max": 1000}}
        t["search"] = {"max_configs": 3, "seeds": [0, 1, 2], "probe_margin": 0.01, "accept_margin": 0.01,
                       "guards": {"ece": 0.02, "brier": 0.005}}
        t["resources"] = {"onejev": {"python": "/x/.venv/bin/python", "default_config": {"base": "joey-1"}},
                          "gpu_ids": [0, 3], "train_command": "modules/train/train"}
        self.assertEqual(rows.check_task(t), [])
        for name, change, word in (
                ("a key that is no config path", _set(("search_space", "lr"), [1]), "search_space.lr"),
                ("an empty list of values", _set(("search_space", "seed"), []), "search_space.seed"),
                ("a range without max", _set(("search_space", "max_steps"), {"min": 1}), "search_space.max_steps"),
                ("a range of bases", _set(("search_space", "base"), {"min": 0, "max": 1}), "search_space.base"),
                ("a range of seeds", _set(("search_space", "seed"), {"min": 0, "max": 2}), "search_space.seed"),
                ("min above max", _set(("search_space", "overrides.lora.r"), {"min": 64, "max": 4}),
                 "search_space.overrides.lora.r"),
                ("a top-level dead (dead is per key)", _set(("search_space", "dead"), {"seed": {"0": "b"}}),
                 "search_space.dead"),
                ("{values, dead} whose dead is no {value: reason}",
                 _set(("search_space", "overrides.onejev.lr", "dead"), ["5e-06"]), "search_space.overrides.onejev.lr.dead"),
                ("{min, max} with dead", _set(("search_space", "overrides.lora.r", "dead"), {}),
                 "search_space.overrides.lora.r"),
                ("seeds repeat", _set(("search", "seeds"), [0, 0]), "search.seeds"),
                ("an unknown search setting", _set(("search", "trials"), 3), "search"),
                ("a negative GPU id", _set(("resources", "gpu_ids"), [0, -1]), "gpu_ids"),
                ("onejev not an object", _set(("resources", "onejev"), "venv"), "onejev")):
            with self.subTest(name):
                bad = copy.deepcopy(t)
                change(bad)
                errors = rows.check_task(bad)
                self.assertTrue(any(word in e for e in errors), errors)

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

    def test_data_section(self):
        def task(**data):
            t = binary_task()
            t["data"] = dict({"sources": [], "label_sources": ["rule"]}, **data)
            return t
        ok = [dict(sources=["只有说明的旧写法"]), dict(sources=[{"path": "/d/raw", "about": "x"}]),
              dict(split={"sizes": {"trainset": 10}}), dict(split=dict(TIME, unit="state_id", seed=3)),
              dict(rows={"about": "a", "expect": {"rows": 10, "label_counts": {"yes": 2, "no": 8}}})]
        for d in ok:
            self.assertEqual(rows.check_task(task(**d)), [], d)
        bad = [(dict(sources=[{"path": "rel/raw"}]), "absolute"), (dict(sources=[{"path": "/d/a*b"}]), "deny"),
               (dict(split=dict(TIME, ranges=dict(TIME["ranges"], valset=[6, 8]))), "overlap"),
               (dict(split=dict(TIME, ranges=dict(TIME["ranges"], testset=[12, 11]))), "testset"),
               (dict(split=dict(TIME, fractions={"trainset": 0.5})), "fractions"),
               (dict(split={"by": "random", "key": "meta.state_id"}), "key"),
               (dict(split={"by": "group", "key": "meta.state_id",
                            "fractions": {"trainset": 0.7, "valset": 0.2, "evalset": 0.2, "testset": 0.1}}), "sum"),
               (dict(split={"sizes": {"pool": 5}}), "pool"), (dict(split={"unit": "address"}), "unit"),
               (dict(rows={"about": "a", "expect": {"label_counts": {"maybe": 1}}}), "maybe")]
        for d, frag in bad:
            errs = rows.check_task(task(**d))
            self.assertTrue(any(frag in e for e in errs), (d, errs))
        self.assertTrue(DS.is_complete(TIME))
        self.assertFalse(DS.is_complete({"sizes": {"trainset": 10}}))
        self.assertEqual(DS.filled(TIME)["unit"], "cluster")
        merged, errs = DS.merge_split({"sizes": {"trainset": 10}}, dict(TIME, sizes={"valset": 5}))
        self.assertEqual((errs, merged["sizes"]), ([], {"trainset": 10, "valset": 5}))
        self.assertIn("data.split.sizes.trainset",
                      DS.merge_split({"sizes": {"trainset": 10}}, {"sizes": {"trainset": 9}})[1][0])


class TestRows(unittest.TestCase):
    def test_rows_are_onejev_rows(self):
        # D1: dataset rows are OneJev rows whose question is the task's decision as OneJev asks it
        # (yes/no binary: noul, criteria true/false; choice: label -> meaning; score: the meanings in order),
        # and whose target follows OneJev's rule (keys among the question's labels, sum 1 within 1e-6).
        bt, ct, st, ot = binary_task(), choice_task(), _score_task(["0", "1", "2"]), _object_task()
        noul = {"type": "noul", "instructions": "Decide whether a message is spam.",
                "criteria": {"true": "the message is spam", "false": "the message is not spam"}}
        choice = {"type": "choice", "instructions": "Pick the handling for a transaction.",
                  "criteria": {"allow": "let it through", "block": "reject it", "review": "send to a human"}}
        score = {"type": "score", "instructions": "Pick the handling for a transaction.", "criteria": ["a", "b", "c"]}
        yes = {"yes": 1.0, "no": 0.0}

        def row(question, target, **over):
            r = {"id": "r1", "state": {"messages": [{"role": "user", "content": "win a prize"}]},
                 "question": question, "target": target, "split": "train",
                 "meta": {"state_id": "s1", "cluster": "c1", "source": "mail", "label_source": "rule", "n_obs": 1}}
            r.update(over)
            return dict((k, v) for k, v in r.items() if v is not None)  # a key set to None is left out

        cases = (
            ("binary yes/no: noul", bt, row(noul, yes), ()),
            ("choice: a target over some of the labels", ct, row(choice, {"block": 0.75, "review": 0.25}), ()),
            ("score: criteria in label order", st, row(score, {"2": 1.0}), ()),
            ("question type not the task's", bt, row(dict(noul, type="choice"), yes), ("question.type",)),
            ("question instructions not the task's", bt, row(dict(noul, instructions="Ham?"), yes),
             ("question.instructions",)),
            ("noul criteria keyed yes/no", bt, row(dict(noul, criteria={"yes": "spam", "no": "ham"}), yes),
             ("question.criteria",)),
            ("target key not a label of the question", bt, row(noul, {"yes": 0.5, "true": 0.5}), ("'true'",)),
            ("target sums to 1 - 1e-5", bt, row(noul, {"yes": 0.5, "no": 0.49999}), ("sum to 1",)),
            ("no state, no split", bt, row(noul, yes, state=None, split=None), ("state", "split")),
            ("an old row's input instead of state", bt,
             row(noul, yes, state=None, input={"messages": [{"role": "user", "content": "win"}]}),
             ("rows are OneJev rows now",)),
            ("format object: a free-form state (R9)", ot,
             row(noul, yes, state={"entity": "x", "summary": {}, "transactions": []}), ()),
            ("format object: an empty state", ot, row(noul, yes, state={}), ("state: expected a non-empty object",)),
            ("format object: a state that is not an object", ot, row(noul, yes, state=["x"]),
             ("state: expected a non-empty object",)),
        )
        for name, task, r, words in cases:
            with self.subTest(name):
                errs = rows.check_full_rows([r], task)
                for word in words:
                    self.assertTrue(any(word in e for e in errs), (word, errs))
                if not words:
                    self.assertEqual(errs, [])
                    self.assertEqual(rows.check_input_rows([rows.input_row_of(r)], task), [])  # its inputs.jsonl row
                    label = {"id": "r1", "target": r["target"], "meta": {"label_source": "rule"}}  # labels.jsonl
                    self.assertEqual(rows.check_label_rows([label], task), [])

    def test_old_row_fields_still_read(self):
        # D1: rows of an old workspace carry input and meta.group; they read as state and meta.state_id.
        from rsi import metrics as M
        t = binary_task()
        new = full_row(1, t)
        old = {"id": "old1", "input": new["state"], "target": {"yes": 0.0, "no": 1.0},
               "meta": {"group": "web", "cluster": "k1", "source": "s", "label_source": "rule", "n_obs": 1}}
        self.assertEqual(rows.state_of(old), new["state"])
        self.assertEqual(rows.state_of(new), new["state"])
        self.assertEqual((rows.group_of(old), rows.group_of(new)), ("web", new["meta"]["state_id"]))
        self.assertIsNone(rows.group_of({"id": "x", "meta": {}}))
        self.assertEqual(rows.overlap([old], [new]), {"ids": 0, "inputs": 1, "clusters": 0})
        self.assertEqual(M.groups_of([old, new]), {"old1": "web", "q1": new["meta"]["state_id"]})
        # group_prior learns the groups of old training rows and applies them to new rows.
        scored = [dict(new, meta=dict(new["meta"], state_id="web"))]
        self.assertEqual(M.group_prior([old], scored, t)[0]["probs"], {"yes": 0.0, "no": 1.0})

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

    def test_input_must_match_the_task_format(self):
        # The task's input section describes a row's state (D1).
        t = binary_task()  # messages: non-empty
        self.assertTrue(rows.check_state({"messages": []}, t))
        self.assertTrue(rows.check_state({"fields": {}}, t))
        self.assertEqual(
            rows.check_state({"messages": [{"role": "user", "content": "hi"}]}, t), []
        )
        t["input"] = {"format": "fields", "description": "d", "fields": ["a", "b"]}  # fields: every task field
        self.assertTrue(rows.check_state({"fields": {"a": 1}}, t))
        self.assertEqual(rows.check_state({"fields": {"a": 1, "b": 2}}, t), [])

    def test_input_and_label_rows(self):
        # D1, R8, F7a: an input row is the full row's id, state, question, meta and split, with only the meta keys
        # state_id, cluster, source and n_obs (any other key, label_source included, may carry the label); a label
        # row is id, target and meta, meta.label_source its only meta key.
        t = binary_task()
        full = full_row(1, t)
        full["meta"]["f"] = 0.9167  # e.g. a OneJev export's agreement rate, which the target follows from
        inp = rows.input_row_of(full)
        self.assertEqual(sorted(inp), ["id", "meta", "question", "split", "state"])
        self.assertEqual(sorted(inp["meta"]), ["cluster", "n_obs", "source", "state_id"])
        self.assertNotIn("label", rows.input_row_of(dict(full, label="yes")))
        lab = {"id": full["id"], "target": full["target"], "meta": {"label_source": "human"}}
        cases = (  # (name, check, row, meta change (None drops the key), top-level change, word an error starts with)
            ("input: whitelisted meta", rows.check_input_rows, inp, {}, None, None),
            ("input: meta.f", rows.check_input_rows, inp, {"f": 0.9167}, None, "meta.f"),
            ("input: meta.label_source", rows.check_input_rows, inp, {"label_source": "rule"}, None,
             "meta.label_source"),
            ("input: no meta.cluster", rows.check_input_rows, inp, {"cluster": None}, None, "meta.cluster"),
            ("label: meta.label_source", rows.check_label_rows, lab, {}, None, None),
            ("label: no meta.label_source", rows.check_label_rows, lab, {"label_source": None}, None,
             "meta.label_source"),
            ("label: another meta key", rows.check_label_rows, lab, {"source": "mail"}, None, "meta.source"),
            ("input: a top-level label", rows.check_input_rows, inp, {}, {"label": "yes"}, "label"),
            ("input: the old input next to state", rows.check_input_rows, inp, {}, {"input": inp["state"]}, "input"),
            ("label: target off 1", rows.check_label_rows, lab, {}, {"target": {"yes": 0.5}}, "target"),
            ("label: a top-level note", rows.check_label_rows, lab, {}, {"note": "x"}, "note"),
        )
        for name, check, base, change, top, word in cases:
            with self.subTest(name):
                r = copy.deepcopy(base)
                r["meta"].update(change)
                r["meta"] = dict((k, v) for k, v in r["meta"].items() if v is not None)
                r.update(top or {})
                errs = check([r], t)
                if word is None:
                    self.assertEqual(errs, [])
                else:
                    self.assertTrue(any(e.split(": ")[1] == word for e in errs), (word, errs))

    def test_predictions_must_cover_exactly(self):
        t = binary_task()

        def p(i, dec="yes"):
            return {"id": i, "probs": {"yes": 0.4, "no": 0.6}, "decision": dec, "uncertainty": None}

        self.assertEqual(rows.check_predictions([p("a"), p("b")], t, {"a", "b"}), [])
        self.assertTrue(rows.check_predictions([p("a")], t, {"a", "b"}))
        self.assertTrue(rows.check_predictions([p("a"), p("b"), p("c")], t, {"a", "b"}))
        self.assertTrue(rows.check_predictions([p("a", "zzz"), p("b")], t, {"a", "b"}))

    def test_feedback_rows(self):
        # F2: a call's input is `state` (the input row's state); a trace of an old workspace calls it `input`.
        t = binary_task()
        fb = {
            "id": "a",
            "state": {"messages": [{"role": "user", "content": "x"}]},
            "output": {"probs": {"yes": 0.5, "no": 0.5}, "decision": "yes"},
            "model": "candidate-0001",
            "latency_ms": 12.5,
            "timestamp": "2026-01-01T00:00:00Z",
            "outcome": None,
        }
        self.assertEqual(rows.check_feedback_rows([fb], t), [])
        old = dict(fb)
        old["input"] = old.pop("state")
        self.assertEqual(rows.check_feedback_rows([old], t), [])
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
        b[1]["state"] = a[0]["state"]
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
