import hashlib
import json
import os
import tempfile
import unittest

from fixtures import binary_task, choice_task, full_row, make_artifact, write_jsonl
from rsi import artifacts, rows


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

    def mk(self, kind, **fields):
        return make_artifact(kind, os.path.join(self.tmp.name, kind), self.task, **fields)

    def test_valid_each_kind(self):
        for kind in ("trainset", "valset", "evalset", "testset"):
            d = self.mk(kind)
            self.assertEqual(artifacts.validate_artifact(kind, d, self.task), [], kind)

    def test_valid_choice_task(self):
        self.task = choice_task()
        for kind in ("trainset", "evalset"):
            d = self.mk(kind)
            self.assertEqual(artifacts.validate_artifact(kind, d, self.task), [], kind)

    def test_artifact_files_table(self):
        self.assertEqual(artifacts.ARTIFACT_FILES["trainset"], ["rows", "quality_report"])
        self.assertEqual(artifacts.ARTIFACT_FILES["valset"], ["rows", "inputs"])
        self.assertEqual(artifacts.ARTIFACT_FILES["evalset"], ["inputs", "labels"])
        self.assertEqual(artifacts.ARTIFACT_FILES["testset"], ["inputs", "labels"])

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
        errs = edited(lambda inp: inp[1]["input"]["messages"][0].update(content="something else"))
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

    def test_input_rows_carry_no_target(self):
        # Inputs files are what deliver gets: a target there would hand it labels.
        d = self.mk("evalset")
        inp = rows.read_jsonl(os.path.join(d, "inputs.jsonl"))
        inp[2]["target"] = {"yes": 0.0, "no": 1.0}
        write_jsonl(os.path.join(d, "inputs.jsonl"), inp)
        errs = artifacts.validate_artifact("evalset", d, self.task)
        self.assertTrue(any("inputs" in e and "q2" in e and "target" in e for e in errs), errs)

    def test_missing_manifest_is_error(self):
        errs = artifacts.validate_artifact("valset", os.path.join(self.tmp.name, "nope"), self.task)
        self.assertTrue(errs)
        self.assertIn("manifest", errs[0])

    def test_invalid_json_manifest_is_error(self):
        d = self.mk("valset")
        with open(os.path.join(d, "manifest.json"), "w") as f:
            f.write("{oops")
        self.assertTrue(artifacts.validate_artifact("valset", d, self.task))

    def test_unknown_kind(self):
        d = self.mk("valset")
        errs = artifacts.validate_artifact("nonsense", d, self.task)
        self.assertTrue(any("unknown kind" in e for e in errs))

    def test_kind_mismatch(self):
        d = self.mk("evalset")
        errs = artifacts.validate_artifact("testset", d, self.task)
        self.assertTrue(any("kind" in e for e in errs), errs)

    def test_unexpected_manifest_field(self):
        d = self.mk("valset", surprise=1)
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any("surprise" in e for e in errs))

    def test_missing_common_field(self):
        d = self.mk("valset")
        m = _manifest(d)
        del m["round"]
        _save(d, m)
        self.assertTrue(any("round" in e for e in artifacts.validate_artifact("valset", d, self.task)))

    def test_missing_kind_field(self):
        d = self.mk("trainset")
        m = _manifest(d)
        del m["label_method"]
        _save(d, m)
        self.assertTrue(any("label_method" in e for e in artifacts.validate_artifact("trainset", d, self.task)))

    def test_trainset_quality_flag_mismatch(self):
        d = self.mk("trainset", quality={"passed": True, "checks": [
            {"name": "a", "value": 1, "threshold": 2, "passed": True},
            {"name": "b", "value": 3, "threshold": 2, "passed": False}]})
        errs = artifacts.validate_artifact("trainset", d, self.task)
        self.assertTrue(any("quality.passed" in e for e in errs))

    def test_trainset_quality_failed_consistent(self):
        d = self.mk("trainset", quality={"passed": False, "checks": [
            {"name": "b", "value": 3, "threshold": 2, "passed": False}]})
        self.assertEqual(artifacts.validate_artifact("trainset", d, self.task), [])

    def test_trainset_bad_rows(self):
        d = self.mk("trainset")
        bad = full_row(9, self.task)
        bad["target"] = {"yes": 0.5, "no": 0.2}
        write_jsonl(os.path.join(d, "rows.jsonl"), [bad])
        m = _manifest(d)
        m["files"]["rows"]["rows"] = 1
        _save(d, m)
        self.assertTrue(artifacts.validate_artifact("trainset", d, self.task))

    def test_rows_count_mismatch(self):
        d = self.mk("valset")
        m = _manifest(d)
        m["files"]["rows"]["rows"] = 99
        _save(d, m)
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any("rows" in e and "99" in e for e in errs))

    def test_blank_lines_not_counted(self):
        d = self.mk("valset")
        with open(os.path.join(d, "rows.jsonl"), "a") as f:
            f.write("\n\n")
        self.assertEqual(artifacts.validate_artifact("valset", d, self.task), [])

    def test_missing_file(self):
        d = self.mk("valset")
        os.remove(os.path.join(d, "rows.jsonl"))
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any("rows.jsonl" in e for e in errs))

    def _escape(self, spec_path, expect):
        d = self.mk("valset")
        m = _manifest(d)
        m["files"]["rows"]["path"] = spec_path
        _save(d, m)
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any(expect in e for e in errs), errs)

    def test_files_absolute_path_rejected(self):
        outside = os.path.join(self.tmp.name, "outside.jsonl")
        write_jsonl(outside, [full_row(0, self.task)])
        self._escape(outside, "relative")

    def test_files_dotdot_path_rejected(self):
        write_jsonl(os.path.join(self.tmp.name, "outside.jsonl"), [full_row(0, self.task)])
        self._escape("../outside.jsonl", "escapes")

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
        d = os.path.join(self.tmp.name, "testset")
        self.mk("testset")
        os.link(os.path.join(d, "labels.jsonl"), os.path.join(self.tmp.name, "labels-second-name.jsonl"))
        errs = artifacts.validate_artifact("testset", d, self.task)
        self.assertTrue(any("labels.jsonl" in e and "2 hard links" in e for e in errs), errs)

    def test_dangling_link_refused_for_every_kind(self):
        d = self.mk("valset")
        os.symlink(os.path.join(self.tmp.name, "gone.jsonl"), os.path.join(d, "extra.jsonl"))
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any("extra.jsonl" in e and "does not exist" in e for e in errs), errs)

    def test_missing_required_logical_file(self):
        d = self.mk("evalset")
        m = _manifest(d)
        del m["files"]["labels"]
        _save(d, m)
        errs = artifacts.validate_artifact("evalset", d, self.task)
        self.assertTrue(any("labels" in e for e in errs))

    def test_evalset_ids_must_match(self):
        d = self.mk("evalset")
        lab = rows.read_jsonl(os.path.join(d, "labels.jsonl"))
        lab[0]["id"] = "other"
        write_jsonl(os.path.join(d, "labels.jsonl"), lab)
        errs = artifacts.validate_artifact("evalset", d, self.task)
        self.assertTrue(any("ids" in e for e in errs))

    def test_testset_bad_label_row(self):
        d = self.mk("testset")
        lab = rows.read_jsonl(os.path.join(d, "labels.jsonl"))
        lab[0]["target"] = {"maybe": 1.0}
        write_jsonl(os.path.join(d, "labels.jsonl"), lab)
        self.assertTrue(artifacts.validate_artifact("testset", d, self.task))

    def test_external_sha_mismatch_local(self):
        ext = os.path.join(self.tmp.name, "big.bin")
        with open(ext, "wb") as f:
            f.write(b"hello")
        good = hashlib.sha256(b"hello").hexdigest()
        d = self.mk("valset", external={"blob": {"location": ext, "sha256": good}})
        self.assertEqual(artifacts.validate_artifact("valset", d, self.task), [])
        d = self.mk("valset", external={"blob": {"location": ext, "sha256": "0" * 64}})
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any("sha256" in e for e in errs))
        # T4: hex digits compare case-insensitively.
        d = self.mk("valset", external={"blob": {"location": ext, "sha256": good.upper()}})
        self.assertEqual(artifacts.validate_artifact("valset", d, self.task), [])

    def test_external_local_missing(self):
        d = self.mk("valset", external={"blob": {"location": "/no/such/file", "sha256": "0" * 64}})
        errs = artifacts.validate_artifact("valset", d, self.task)
        self.assertTrue(any("blob" in e for e in errs))

    def test_external_remote_not_checked(self):
        d = self.mk("valset", external={"blob": {"location": "gpu-box:/data/x.bin", "sha256": "0" * 64}})
        self.assertEqual(artifacts.validate_artifact("valset", d, self.task), [])

if __name__ == "__main__":
    unittest.main()
