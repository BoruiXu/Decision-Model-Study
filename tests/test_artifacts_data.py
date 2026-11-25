import hashlib
import json
import os
import tempfile
import unittest

from fixtures import (binary_task, choice_task, corpus_rows, filled_time_task, full_row, make_artifact,
                      update_manifest, write_jsonl)
from rsi import artifacts, rows
from rsi.artifacts import validate_artifact


def _manifest(d):
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _save(d, m):
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f)


class TestDataArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task = binary_task()

    def d(self, name):
        return os.path.join(self.tmp.name, name)

    def mk(self, kind, **fields):
        """A fresh source directory of `kind`."""
        self.n = getattr(self, "n", 0) + 1
        return make_artifact(kind, os.path.join(self.tmp.name, "%s-%d" % (kind, self.n)), self.task, **fields)

    def edit_manifest(self, d, fn):
        m = _manifest(d)
        fn(m)
        _save(d, m)
        return d

    def edit_rows(self, d, name, fn):
        fp = os.path.join(d, name)
        rs = rows.read_jsonl(fp)
        fn(rs)
        write_jsonl(fp, rs)
        return d

    def test_valid(self):
        ext = os.path.join(self.tmp.name, "big.bin")
        with open(ext, "wb") as f:
            f.write(b"hello")
        good = hashlib.sha256(b"hello").hexdigest()

        def blank_lines(d):
            with open(os.path.join(d, "rows.jsonl"), "a") as f:
                f.write("\n\n")  # not counted
            return d
        cases = [("%s" % kind, kind, lambda kind=kind: self.mk(kind))
                 for kind in ("trainset", "valset", "evalset", "testset")]
        cases += [
            ("trainset whose failed quality flag matches its checks", "trainset", lambda: self.mk(
                "trainset", quality={"passed": False, "checks": [
                    {"name": "b", "value": 3, "threshold": 2, "passed": False}]})),
            ("blank lines", "valset", lambda: blank_lines(self.mk("valset"))),
            ("valset rows with a meta key its inputs leave out (R8)", "valset", lambda: self.edit_rows(
                self.mk("valset"), "rows.jsonl", lambda rs: [r["meta"].update(f=0.75) for r in rs])),
            ("local external, sha256 matches", "valset",
             lambda: self.mk("valset", external={"blob": {"location": ext, "sha256": good}})),
            ("sha256 hex digits in upper case (T4)", "valset",
             lambda: self.mk("valset", external={"blob": {"location": ext, "sha256": good.upper()}})),
            ("remote external, not checked", "valset",
             lambda: self.mk("valset", external={"blob": {"location": "gpu-box:/data/x.bin", "sha256": "0" * 64}})),
        ]
        for name, kind, build in cases:
            with self.subTest(name):
                self.assertEqual(artifacts.validate_artifact(kind, build(), self.task), [])
        self.task = choice_task()
        for kind in ("trainset", "evalset"):
            with self.subTest("choice task " + kind):
                self.assertEqual(artifacts.validate_artifact(kind, self.mk(kind), self.task), [])

    def test_artifact_files_table(self):
        self.assertEqual(artifacts.ARTIFACT_FILES["trainset"], ["rows", "quality_report"])
        self.assertEqual(artifacts.ARTIFACT_FILES["valset"], ["rows", "inputs"])
        self.assertEqual(artifacts.ARTIFACT_FILES["evalset"], ["inputs", "labels"])
        self.assertEqual(artifacts.ARTIFACT_FILES["testset"], ["inputs", "labels"])
        self.assertEqual(artifacts.ARTIFACT_FILES["corpus"], ["rows", "script"])
        self.assertEqual(artifacts.ARTIFACT_FILES["pool"], ["rows"])

    def test_corpus_and_pool_validation(self):
        task = filled_time_task()
        task["data"]["rows"] = {"about": "a", "expect": {"rows": 48, "label_counts": {"yes": 24, "no": 24}}}
        rows = corpus_rows(task)
        self.assertEqual(validate_artifact("corpus", make_artifact("corpus", self.d("c0"), task, rows=rows), task), [])
        def no_step(rs): rs[5]["meta"]["extra"].pop("step"); return rs
        def text_step(rs): rs[5]["meta"]["extra"]["step"] = "3"; return rs
        for name, rs, frag in (("c1", no_step(corpus_rows(task)), "meta.extra.step"),
                               ("c2", text_step(corpus_rows(task)), "number"), ("c3", rows[:-1], "expect")):
            errs = validate_artifact("corpus", make_artifact("corpus", self.d(name), task, rows=rs), task)
            self.assertTrue(any(frag in e for e in errs), (name, errs))
        for n, sources in enumerate((["/elsewhere"], [])):                  # the task names raw data: say which
            c4 = make_artifact("corpus", self.d("c4-%d" % n), task, rows=rows); update_manifest(c4, sources=sources)
            self.assertTrue(any("data.sources" in e for e in validate_artifact("corpus", c4, task)), sources)
        rows[0]["split"] = "test"                                     # a corpus row may keep its source split
        self.assertEqual(validate_artifact("corpus", make_artifact("corpus", self.d("c5"), task, rows=rows), task), [])
        errs = validate_artifact("pool", make_artifact("pool", self.d("p"), task, rows=rows[:8]), task)
        self.assertTrue(any("split" in e for e in errs))              # pool rows feed training

    def test_valset_inputs(self):
        # Task 14: a valset also has inputs.jsonl (no labels), for deliver; same ids and inputs as rows.
        d = self.mk("valset")
        self.assertEqual(sorted(os.listdir(d)), ["inputs.jsonl", "manifest.json", "rows.jsonl"])
        self.assertEqual(artifacts.validate_artifact("valset", d, self.task), [])
        m = _manifest(d)
        del m["files"]["inputs"]
        _save(d, m)
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any("inputs" in e for e in errs), errs)

        def edited(fn):
            d = self.mk("valset")
            inp = rows.read_jsonl(os.path.join(d, "inputs.jsonl"))
            fn(inp)
            write_jsonl(os.path.join(d, "inputs.jsonl"), inp)
            m = _manifest(d)
            m["files"]["inputs"]["rows"] = len(inp)
            _save(d, m)
            return artifacts.validate_artifact("valset", d, self.task)
        errs = edited(lambda inp: inp[0].update(id="q99"))
        self.assertTrue(any("inputs" in e and "ids differ" in e for e in errs), errs)
        errs = edited(lambda inp: inp.pop())
        self.assertTrue(any("inputs" in e and "ids differ" in e for e in errs), errs)
        errs = edited(lambda inp: inp[1]["state"]["messages"][0].update(content="something else"))
        self.assertTrue(any("inputs" in e and "q1" in e and "rows.jsonl" in e for e in errs), errs)
        errs = edited(lambda inp: inp[0].update(target={"yes": 1.0, "no": 0.0}))
        self.assertTrue(any("inputs" in e and "target" in e for e in errs), errs)
        errs = edited(lambda inp: inp[0]["meta"].pop("cluster"))
        self.assertTrue(any("inputs" in e and "meta.cluster" in e for e in errs), errs)

    def test_unhashable_ids_are_errors(self):
        # R43: an id that is not a string (here a list) is a validation error, not a crash.
        def with_bad_id(kind, name):
            d = self.mk(kind)
            fp = os.path.join(d, name)
            rs = rows.read_jsonl(fp)
            rs[0]["id"] = ["x"]
            write_jsonl(fp, rs)
            return artifacts.validate_artifact(kind, d, self.task)
        for kind, name in (("valset", "inputs.jsonl"), ("valset", "rows.jsonl"), ("evalset", "inputs.jsonl"),
                           ("evalset", "labels.jsonl"), ("testset", "labels.jsonl")):
            errs = with_bad_id(kind, name)
            self.assertTrue(any("id: expected non-empty string" in e for e in errs), (kind, name, errs))

    def test_refusals(self):
        # (name, kind validated, a broken source, words one error must contain)
        outside = os.path.join(self.tmp.name, "outside.jsonl")
        write_jsonl(outside, [full_row(0, self.task)])

        def bad_json(d):
            with open(os.path.join(d, "manifest.json"), "w") as f:
                f.write("{oops")
            return d

        def bad_trainset_row(d):
            bad = full_row(9, self.task)
            bad["target"] = {"yes": 0.5, "no": 0.2}
            write_jsonl(os.path.join(d, "rows.jsonl"), [bad])
            return self.edit_manifest(d, lambda m: m["files"]["rows"].update(rows=1))

        def removed(d, name):
            os.remove(os.path.join(d, name))
            return d

        def dangling(d):
            os.symlink(os.path.join(self.tmp.name, "gone.jsonl"), os.path.join(d, "extra.jsonl"))
            return d

        def named_test(d, *names):  # onejev-harness does not read training or validation rows whose split is test*
            for name in names:
                self.edit_rows(d, name, lambda rs: rs[0].update(split="test-a"))
            return d

        cases = (
            ("missing manifest", "valset", lambda: os.path.join(self.tmp.name, "nope"), ("manifest",)),
            ("manifest not JSON", "valset", lambda: bad_json(self.mk("valset")), ()),
            ("unknown kind", "nonsense", lambda: self.mk("valset"), ("unknown kind",)),
            ("kind mismatch", "testset", lambda: self.mk("evalset"), ("kind",)),
            ("unexpected manifest field", "valset", lambda: self.mk("valset", surprise=1), ("surprise",)),
            ("missing common field", "valset",
             lambda: self.edit_manifest(self.mk("valset"), lambda m: m.pop("round")), ("round",)),
            ("missing kind field", "trainset",
             lambda: self.edit_manifest(self.mk("trainset"), lambda m: m.pop("label_method")), ("label_method",)),
            ("quality.passed against its checks", "trainset", lambda: self.mk("trainset", quality={
                "passed": True, "checks": [{"name": "a", "value": 1, "threshold": 2, "passed": True},
                                           {"name": "b", "value": 3, "threshold": 2, "passed": False}]}),
             ("quality.passed",)),
            ("bad trainset row", "trainset", lambda: bad_trainset_row(self.mk("trainset")), ()),
            ("trainset row whose split starts with test", "trainset",
             lambda: named_test(self.mk("trainset"), "rows.jsonl"), ("split",)),
            ("valset row whose split starts with test", "valset",
             lambda: named_test(self.mk("valset"), "rows.jsonl", "inputs.jsonl"), ("split",)),
            ("rows count mismatch", "valset",
             lambda: self.edit_manifest(self.mk("valset"), lambda m: m["files"]["rows"].update(rows=99)),
             ("rows", "99")),
            ("missing file", "valset", lambda: removed(self.mk("valset"), "rows.jsonl"), ("rows.jsonl",)),
            ("absolute file path", "valset",
             lambda: self.edit_manifest(self.mk("valset"), lambda m: m["files"]["rows"].update(path=outside)),
             ("relative",)),
            ("file path out of the directory", "valset", lambda: self.edit_manifest(
                self.mk("valset"), lambda m: m["files"]["rows"].update(path="../outside.jsonl")), ("escapes",)),
            ("dangling link, any kind", "valset", lambda: dangling(self.mk("valset")),
             ("extra.jsonl", "does not exist")),
            ("missing logical file", "evalset",
             lambda: self.edit_manifest(self.mk("evalset"), lambda m: m["files"].pop("labels")), ("labels",)),
            ("evalset label ids differ", "evalset", lambda: self.edit_rows(
                self.mk("evalset"), "labels.jsonl", lambda rs: rs[0].update(id="other")), ("ids",)),
            ("testset bad label row", "testset", lambda: self.edit_rows(
                self.mk("testset"), "labels.jsonl", lambda rs: rs[0].update(target={"maybe": 1.0})), ()),
            # Inputs files are what deliver gets: a target there would hand it labels.
            ("inputs carry a target", "evalset", lambda: self.edit_rows(
                self.mk("evalset"), "inputs.jsonl", lambda rs: rs[2].update(target={"yes": 0.0, "no": 1.0})),
             ("inputs", "q2", "target")),
            ("local external, sha256 differs", "valset",
             lambda: self.mk("valset", external={"blob": {"location": os.path.join(self.tmp.name, "outside.jsonl"),
                                                          "sha256": "0" * 64}}), ("sha256",)),
            ("local external missing", "valset",
             lambda: self.mk("valset", external={"blob": {"location": "/no/such/file", "sha256": "0" * 64}}),
             ("blob",)),
        )
        for name, kind, build, words in cases:
            with self.subTest(name):
                errs = artifacts.validate_artifact(kind, build(), self.task)
                self.assertTrue(errs)
                self.assertTrue(any(all(w in e for w in words) for e in errs), errs)

    def test_files_symlink_out_only_refused_for_sealed_sets(self):
        # R45: a copied kind may link to a file elsewhere (registration copies the content); an evalset or
        # testset is moved, so a link out of it, or a hard-linked file, would leave the labels behind.
        outside = os.path.join(self.tmp.name, "outside.jsonl")
        write_jsonl(outside, [full_row(i, self.task) for i in range(4)])
        d = self.mk("valset")
        os.remove(os.path.join(d, "rows.jsonl"))
        os.symlink(outside, os.path.join(d, "rows.jsonl"))
        self.assertEqual(artifacts.validate_artifact("valset", d, self.task), [])
        d = self.mk("evalset")
        kept = os.path.join(self.tmp.name, "labels-kept.jsonl")
        os.rename(os.path.join(d, "labels.jsonl"), kept)
        os.symlink(kept, os.path.join(d, "labels.jsonl"))
        errs = artifacts.validate_artifact("evalset", d, self.task)
        self.assertTrue(any("labels.jsonl" in e and "symbolic link" in e for e in errs), errs)
        d = self.mk("testset")
        os.link(os.path.join(d, "labels.jsonl"), os.path.join(self.tmp.name, "labels-second-name.jsonl"))
        errs = artifacts.validate_artifact("testset", d, self.task)
        self.assertTrue(any("labels.jsonl" in e and "2 hard links" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
