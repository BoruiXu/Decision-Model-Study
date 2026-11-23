import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from fixtures import (add_predictions, binary_task, full_row, make_artifact, make_task_dir,
                      update_manifest, write_jsonl)
from rsi import cli
from rsi.artifacts import validate_artifact
from rsi.workspace import RsiError, Workspace, content_sha256

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Disjoint row-id ranges (q<i>) for the four data sets.
TRAIN, VAL, EVAL, TEST = range(0, 4), range(10, 14), range(20, 24), range(30, 34)
# Tuning trial candidates: not attempts (their search is).
TUNE = {"producer": "tune", "config_source": "tune"}


def run_rsi(*args):
    return subprocess.run([sys.executable, "-m", "rsi"] + list(args),
                          cwd=REPO, capture_output=True, text=True,
                          env=dict(os.environ, RSI_NO_BROWSER="1"))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_registry(ws):
    with open(os.path.join(ws, "registry.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class WsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.task = binary_task()
        self.wsdir = os.path.join(self.root, "ws")
        self.n = 0

    def src(self, name):
        return os.path.join(self.root, "src", name)

    def mk(self, kind, name=None, **fields):
        """Build a fresh source directory; `notes` varies so contents never repeat. A delivery
        reproduces the candidate's predictions on the valset rows VAL (val_ids)."""
        self.n += 1
        fields.setdefault("notes", "source %d" % self.n)
        if kind == "delivery":
            fields.setdefault("val_ids", VAL)
        return make_artifact(kind, self.src(name or "%s-%d" % (kind, self.n)), self.task, **fields)

    def init(self):
        self.ws = Workspace.init(self.wsdir, make_task_dir(self.src("task"), self.task))
        return self.ws

    def sets(self):
        self.init()
        for kind, ids in (("testset", TEST), ("evalset", EVAL), ("valset", VAL), ("trainset", TRAIN)):
            self.ws.register(kind, self.mk(kind, ids=ids))
        return self.ws

    def approve(self, what, ref=None):
        self.ws.state["approvals"].append({
            "what": what, "ref": ref, "by": "tester", "quote": "go ahead",
            "at": "2026-11-21T00:00:00Z", "round": self.ws.state["round"]})
        self.ws.save_state()

    def refused(self, kind, src, *fragments):
        with self.assertRaises(RsiError) as cm:
            self.ws.register(kind, src)
        msg = str(cm.exception)
        for frag in fragments:
            self.assertIn(frag, msg)
        return msg

    def state_on_disk(self):
        with open(os.path.join(self.wsdir, "STATE.json"), encoding="utf-8") as f:
            return f.read()

    def candidate(self, **fields):
        fields.setdefault("ids", VAL)
        return self.ws.register("candidate", self.mk("candidate", **fields))

    def delivery(self, **fields):
        fields.setdefault("ids", EVAL)
        fields.setdefault("candidate_id", "candidate-0001")
        return self.ws.register("delivery", self.mk("delivery", **fields))

    def search_fields(self, default, best, *more):
        """Search fields over registered candidates: ok trials for `default`, `best` and `more`, each
        trial's metric_val copied from its candidate's metrics_val (as rsi-tune does), and the gain."""
        primary = self.task["metrics"]["primary"]
        ids = [default, best] + [c for c in more if c not in (default, best)]
        trials = [{"candidate_id": c, "config": {"lr": 0.001 * (i + 1)},
                   "metric_val": self.ws.manifest(c)["metrics_val"][primary], "compute_hours": 1, "status": "ok"}
                  for i, c in enumerate(ids)]
        return dict(trials=trials, default_candidate=default, best_candidate=best,
                    gain_vs_default={"metric": primary, "delta": trials[1]["metric_val"] - trials[0]["metric_val"]})

    def eval_src(self, **fields):
        """An evaluation of the evalset rows EVAL; its numbers come from the fixture's predictions
        (`wrong`, and `best_wrong` for the best version), as a judge would compute them."""
        fields.setdefault("ids", EVAL)
        fields.setdefault("delivery_id", "delivery-0001")
        fields.setdefault("candidate_id", "candidate-0001")
        fields.setdefault("best_before", None)
        return self.mk("evaluation", **fields)


class TestInit(WsCase):
    def test_init_creates_state_and_registers_task(self):
        ws = self.init()
        self.assertTrue(os.path.isfile(os.path.join(self.wsdir, "STATE.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.wsdir, "artifacts", "task-0001", "task.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.wsdir, "artifacts", "task-0001", "TASK.md")))
        s = ws.state
        self.assertEqual(s["task_id"], "spam-check")
        self.assertEqual(s["task_version"], 1)
        self.assertEqual(s["round"], 1)
        self.assertEqual(s["current"], {"task": "task-0001", "trainset": None, "valset": None,
                                        "evalset": None, "testset": None})
        self.assertEqual(s["versions"], [])
        self.assertIsNone(s["best"])
        self.assertEqual(s["eval_use_count"], 0)
        self.assertEqual(s["approvals"], [])
        self.assertRegex(s["rounds"]["1"]["opened_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual({n: {k: v for k, v in r.items() if k != "opened_at"} for n, r in s["rounds"].items()},
                         {"1": {"entry": "data", "artifacts": ["task-0001"],
                                             "decision": None, "plan_id": None}})
        self.assertEqual(s["no_promotion_streak"], 0)
        self.assertIsNone(s["stopped"])
        self.assertEqual(s["budget_used"], {"compute_hours": 0})
        reg = read_registry(self.wsdir)
        self.assertEqual(len(reg), 1)
        e = reg[0]
        self.assertEqual(sorted(e), sorted(["id", "kind", "round", "producer", "source", "content_sha256",
                                            "path", "files", "parents", "registered_at"]))
        self.assertEqual((e["id"], e["kind"], e["round"]), ("task-0001", "task", 1))
        self.assertEqual(e["source"], self.src("task"))
        self.assertEqual(e["content_sha256"], content_sha256(self.src("task")))
        self.assertEqual(e["files"]["task.json"],
                         sha(os.path.join(self.wsdir, "artifacts", "task-0001", "task.json")))
        self.assertEqual(ws.current("task"), "task-0001")
        self.assertEqual(ws.manifest("task-0001")["task_id"], "spam-check")
        self.assertEqual(ws.task()["version"], 1)
        self.assertEqual(Workspace(self.wsdir).state, s)

    def test_init_refuses_existing_workspace(self):
        self.init()
        with self.assertRaises(RsiError):
            Workspace.init(self.wsdir, make_task_dir(self.src("task2"), self.task))

    def test_init_refuses_invalid_task_and_creates_nothing(self):
        t = binary_task()
        del t["decision"]["positive"]
        with self.assertRaises(RsiError) as cm:
            Workspace.init(self.wsdir, make_task_dir(self.src("bad"), t))
        self.assertIn("positive", str(cm.exception))
        self.assertFalse(os.path.exists(self.wsdir))

    def test_init_requires_task_md(self):
        d = make_task_dir(self.src("task"), self.task)
        os.remove(os.path.join(d, "TASK.md"))
        with self.assertRaises(RsiError) as cm:
            Workspace.init(self.wsdir, d)
        self.assertIn("TASK.md", str(cm.exception))

    def test_open_non_workspace_refused(self):
        with self.assertRaises(RsiError):
            Workspace(self.root)

    def test_save_state_is_atomic_and_persists(self):
        ws = self.init()
        ws.state["no_promotion_streak"] = 2
        ws.save_state()
        self.assertEqual(Workspace(self.wsdir).state["no_promotion_streak"], 2)
        self.assertEqual(sorted(os.listdir(self.wsdir)), ["STATE.json", "artifacts", "registry.jsonl"])


class TestTaskVersion(WsCase):
    def test_task_new_version_needs_approval(self):
        self.init()
        t2 = binary_task()
        t2["version"] = 2
        t2["eval_use_limit"] = 5
        d = make_task_dir(self.src("task-v2"), t2)
        self.refused("task", d, "task_change")
        self.approve("task_change")
        self.assertEqual(self.ws.register("task", d), "task-0002")
        self.assertEqual(self.ws.state["approvals"][0]["used_by"], "task-0002")
        self.assertEqual(self.ws.state["task_version"], 2)
        self.assertEqual(self.ws.current("task"), "task-0002")
        self.assertEqual(self.ws.task()["eval_use_limit"], 5)
        # Artifacts must now cite the new version.
        self.refused("testset", self.mk("testset", ids=TEST), "task_version")
        self.assertEqual(self.ws.register("testset", self.mk("testset", ids=TEST, task_version=2)),
                         "testset-0001")

    def test_task_version_must_increase_by_one_and_keep_id(self):
        self.init()
        self.approve("task_change")
        t3 = binary_task()
        t3["version"] = 3
        self.refused("task", make_task_dir(self.src("t3"), t3), "version")
        other = binary_task()
        other["version"] = 2
        other["task_id"] = "other-task"
        self.refused("task", make_task_dir(self.src("other"), other), "task_id")

    def test_approval_from_earlier_round_does_not_count(self):
        self.init()
        self.approve("task_change")
        self.ws.state["approvals"][0]["round"] = 0
        self.ws.save_state()
        t2 = binary_task()
        t2["version"] = 2
        self.refused("task", make_task_dir(self.src("t2"), t2), "task_change")


class TestRegister(WsCase):
    def test_register_assigns_sequential_ids(self):
        self.sets()
        reg = read_registry(self.wsdir)
        self.assertEqual([e["id"] for e in reg],
                         ["task-0001", "testset-0001", "evalset-0001", "valset-0001", "trainset-0001"])
        self.assertEqual(self.ws.register("trainset", self.mk("trainset", ids=range(40, 44))),
                         "trainset-0002")
        self.assertEqual(self.candidate(trainset_id="trainset-0002"), "candidate-0001")
        self.assertEqual(self.candidate(trainset_id="trainset-0002"), "candidate-0002")
        self.assertEqual(self.ws.state["rounds"]["1"]["artifacts"],
                         ["task-0001", "testset-0001", "evalset-0001", "valset-0001", "trainset-0001",
                          "trainset-0002", "candidate-0001", "candidate-0002"])
        self.assertEqual(self.ws.state["current"]["trainset"], "trainset-0002")
        self.assertEqual(self.ws.current("trainset"), "trainset-0002")

    def test_register_copies_and_writes_id(self):
        self.init()
        src = self.mk("valset", ids=VAL)
        aid = self.ws.register("valset", src)
        dst = os.path.join(self.wsdir, "artifacts", aid)
        self.assertEqual(self.ws.path(aid), dst)
        self.assertEqual(self.ws.manifest(aid)["id"], "valset-0001")
        with open(os.path.join(src, "manifest.json"), encoding="utf-8") as f:
            self.assertIsNone(json.load(f)["id"])
        self.assertEqual(sorted(os.listdir(dst)), ["inputs.jsonl", "manifest.json", "rows.jsonl"])
        e = read_registry(self.wsdir)[-1]
        self.assertEqual(e["path"], "artifacts/valset-0001")
        self.assertEqual(e["source"], src)
        self.assertEqual(e["producer"], "data")
        self.assertEqual(e["parents"], [])
        self.assertEqual(e["content_sha256"], content_sha256(src))
        self.assertEqual(e["files"], {"rows": sha(os.path.join(dst, "rows.jsonl")),
                                      "inputs": sha(os.path.join(dst, "inputs.jsonl"))})
        self.assertEqual(self.ws.entry(aid), e)
        self.assertEqual(self.ws.state["current"]["valset"], aid)

    def test_sealed_sets_are_moved_in(self):
        # Task 14 (R37): an evalset or testset is handed over: its source directory moves into the
        # workspace, so the producing module keeps no copy of the labels. Other kinds are copied.
        self.init()
        for kind, ids in (("testset", TEST), ("evalset", EVAL)):
            src = self.mk(kind, ids=ids)
            digest = content_sha256(src)
            aid = self.ws.register(kind, src)
            self.assertFalse(os.path.exists(src), kind)
            self.assertEqual(sorted(os.listdir(self.ws.path(aid))), ["inputs.jsonl", "labels.jsonl", "manifest.json"])
            e = read_registry(self.wsdir)[-1]
            self.assertEqual((e["source"], e["content_sha256"]), (src, digest))
            self.assertEqual(self.ws.state["current"][kind], aid)
        # Nothing else is left beside the sources (no hidden staging copies).
        self.assertEqual(sorted(os.listdir(self.src(""))), ["task"])
        for kind, ids in (("valset", VAL), ("trainset", TRAIN)):
            src = self.mk(kind, ids=ids)
            self.ws.register(kind, src)
            self.assertTrue(os.path.isfile(os.path.join(src, "manifest.json")), kind)

    def test_refused_sealed_set_stays_in_place(self):
        self.init()
        self.ws.register("testset", self.mk("testset", ids=TEST))
        src = self.mk("evalset", ids=[20, 21, 31])  # overlaps the testset
        digest = content_sha256(src)
        self.refused("evalset", src, "testset-0001")
        self.assertEqual(content_sha256(src), digest)
        src = self.mk("evalset", ids=EVAL, task_id="other-task")
        self.refused("evalset", src, "task_id")
        self.assertTrue(os.path.isfile(os.path.join(src, "labels.jsonl")))

    def test_sealed_set_hand_over_is_atomic(self):
        self.init()
        before = (self.state_on_disk(), read_registry(self.wsdir))
        # The commit fails (a stray artifacts/testset-0001): the source is put back as it was.
        os.makedirs(os.path.join(self.wsdir, "artifacts", "testset-0001"))
        src = self.mk("testset", ids=TEST)
        digest = content_sha256(src)
        self.refused("testset", src, "testset-0001", "repair the workspace")
        self.assertEqual(content_sha256(src), digest)
        self.assertEqual(sorted(os.listdir(os.path.dirname(src))), sorted(["task", os.path.basename(src)]))
        os.rmdir(os.path.join(self.wsdir, "artifacts", "testset-0001"))
        # The source cannot be moved (its parent is read-only): refused before anything is recorded.
        parent = os.path.dirname(src)
        os.chmod(parent, 0o555)
        self.addCleanup(os.chmod, parent, 0o755)
        if os.access(parent, os.W_OK):
            self.skipTest("running with permissions that ignore a read-only directory")
        self.refused("testset", src, "cannot move the source", "nothing was registered")
        os.chmod(parent, 0o755)
        self.assertEqual(content_sha256(src), digest)
        self.assertEqual((self.state_on_disk(), read_registry(self.wsdir)), before)
        self.assertEqual(sorted(os.listdir(os.path.join(self.wsdir, "artifacts"))), ["task-0001"])
        self.assertEqual(self.ws.register("testset", src), "testset-0001")

    def test_symlink_out_of_the_source_refused(self):
        # R43: copying follows symbolic links, so a labels.jsonl linked from elsewhere would register as a
        # file and the move would leave the real labels behind. Links leaving the source are refused.
        self.init()
        before = (self.state_on_disk(), read_registry(self.wsdir))
        src = self.mk("testset", ids=TEST)
        elsewhere = os.path.join(self.root, "kept-labels.jsonl")
        os.rename(os.path.join(src, "labels.jsonl"), elsewhere)
        os.symlink(elsewhere, os.path.join(src, "labels.jsonl"))
        digest = content_sha256(src)
        self.refused("testset", src, "labels.jsonl", "symbolic link", elsewhere)
        self.assertTrue(os.path.islink(os.path.join(src, "labels.jsonl")))
        self.assertEqual(content_sha256(src), digest)
        self.assertEqual((self.state_on_disk(), read_registry(self.wsdir)), before)
        self.assertEqual(os.listdir(os.path.join(self.wsdir, "artifacts")), ["task-0001"])
        # A link that stays inside the source is fine (the copy takes the file it points to).
        src = self.mk("testset", ids=TEST)
        os.rename(os.path.join(src, "labels.jsonl"), os.path.join(src, "labels.real.jsonl"))
        os.symlink("labels.real.jsonl", os.path.join(src, "labels.jsonl"))
        self.assertEqual(self.ws.register("testset", src), "testset-0001")
        self.assertFalse(os.path.exists(src))

    def test_copied_kinds_follow_links(self):
        # R45: a copied kind may link out of its directory; the registered copy holds the content (e.g. a
        # virtualenv inside a delivery's package/). Validation and registration agree.
        self.sets()
        src = self.mk("candidate", ids=VAL)
        outside = os.path.join(self.root, "predict-elsewhere")
        os.rename(os.path.join(src, "predict"), outside)
        os.symlink(outside, os.path.join(src, "predict"))
        self.assertEqual(validate_artifact("candidate", src, self.task), [])
        self.assertEqual(self.ws.register("candidate", src), "candidate-0001")
        copied = os.path.join(self.ws.path("candidate-0001"), "predict")
        self.assertFalse(os.path.islink(copied))
        self.assertTrue(os.access(copied, os.X_OK))
        src = self.mk("delivery", ids=EVAL, candidate_id="candidate-0001")
        pkg = os.path.join(self.root, "pkg-elsewhere")
        os.rename(os.path.join(src, "package"), pkg)
        os.symlink(pkg, os.path.join(src, "package"))
        os.makedirs(os.path.join(pkg, "venv", "bin"))
        os.symlink(sys.executable, os.path.join(pkg, "venv", "bin", "python3"))
        self.assertEqual(validate_artifact("delivery", src, self.task), [])
        self.assertEqual(self.ws.register("delivery", src), "delivery-0001")
        registered = os.path.join(self.ws.path("delivery-0001"), "package")
        self.assertTrue(os.path.isfile(os.path.join(registered, "predict")))
        self.assertFalse(os.path.islink(os.path.join(registered, "venv", "bin", "python3")))
        # A dangling link is refused for every kind, with a message instead of a copy error.
        src = self.mk("delivery", ids=EVAL, candidate_id="candidate-0001")
        os.symlink(os.path.join(self.root, "no-such-file"), os.path.join(src, "package", "weights.bin"))
        self.refused("delivery", src, "package/weights.bin", "does not exist")


    def test_hand_over_cleanup_failure_is_a_warning(self):
        # R43: the registration succeeded; a leftover source copy is reported on stderr, exit 0.
        self.init()
        src = self.mk("testset", ids=TEST)
        real = shutil.rmtree

        def flaky(path, *args, **kwargs):
            if ".rsi-handed-over-" in str(path):
                raise OSError("permission denied")
            return real(path, *args, **kwargs)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("rsi.workspace.shutil.rmtree", side_effect=flaky), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["register", self.wsdir, "testset", src])
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual(out.getvalue(), "testset-0001\n")
        self.assertTrue(err.getvalue().startswith("warning: testset-0001 is registered, but removing its source"),
                        err.getvalue())
        self.assertIn("permission denied", err.getvalue())
        self.assertEqual(Workspace(self.wsdir).current("testset"), "testset-0001")
        left = [n for n in os.listdir(os.path.dirname(src)) if ".rsi-handed-over-" in n]
        self.assertEqual(len(left), 1)
        self.assertIn(left[0], err.getvalue())

    def test_moved_source_path_stays_registered(self):
        # R4 still holds after the move: a new directory at the old path is refused.
        self.init()
        src = self.mk("testset", "ts", ids=TEST)
        self.ws.register("testset", src)
        self.refused("testset", src, "not a directory")
        make_artifact("testset", src, self.task, ids=range(40, 44), notes="rewritten")
        self.refused("testset", src, "already registered", "testset-0001")
        self.assertTrue(os.path.isdir(src))

    def test_register_same_source_twice_refused(self):
        self.init()
        src = self.mk("valset", "vs", ids=VAL)
        self.ws.register("valset", src)
        self.refused("valset", src, "already registered", "valset-0001")
        # Modifying the source after registration does not make it new.
        update_manifest(src, notes="edited after registration")
        self.refused("valset", src, "already registered")
        # Same content from another directory is the same artifact.
        twin = make_artifact("valset", self.src("twin"), self.task, ids=VAL, notes="source 1")
        self.assertEqual(content_sha256(twin), read_registry(self.wsdir)[-1]["content_sha256"])
        self.refused("valset", twin, "same content", "valset-0001")
        # The registered copy itself cannot be registered again.
        self.refused("valset", self.ws.path("valset-0001"), "valset-0001")
        self.assertEqual(len(read_registry(self.wsdir)), 2)

    def test_refused_registration_leaves_no_trace(self):
        self.init()
        before = (json.dumps(self.ws.state, sort_keys=True), read_registry(self.wsdir))
        self.refused("testset", self.mk("testset", ids=TEST, task_id="other-task"), "task_id")
        self.assertEqual((json.dumps(Workspace(self.wsdir).state, sort_keys=True),
                          read_registry(self.wsdir)), before)
        self.assertEqual(os.listdir(os.path.join(self.wsdir, "artifacts")), ["task-0001"])

    def test_invalid_artifact_refused(self):
        self.init()
        src = self.mk("testset", ids=TEST)
        os.remove(os.path.join(src, "labels.jsonl"))
        self.refused("testset", src, "labels")
        self.refused("bogus", src, "unknown kind")

    def test_common_checks(self):
        self.init()
        self.refused("testset", self.mk("testset", ids=TEST, task_id="other-task"), "task_id")
        self.refused("testset", self.mk("testset", ids=TEST, parents=["valset-0009"]), "valset-0009")
        self.refused("testset", self.mk("testset", ids=TEST, id="testset-0007"), "must be null")
        ts = self.ws.register("testset", self.mk("testset", ids=TEST))
        self.assertEqual(self.ws.register("evalset", self.mk("evalset", ids=EVAL, parents=[ts])),
                         "evalset-0001")

    def test_unknown_id_refused(self):
        self.init()
        for fn in (self.ws.path, self.ws.manifest, self.ws.entry):
            with self.assertRaises(RsiError):
                fn("trainset-0001")


class TestDataSets(WsCase):
    def test_trainset_overlap_with_evalset_refused(self):
        self.init()
        self.ws.register("testset", self.mk("testset", ids=TEST))
        self.ws.register("evalset", self.mk("evalset", ids=EVAL))
        self.ws.register("valset", self.mk("valset", ids=VAL))
        self.refused("trainset", self.mk("trainset", ids=[0, 1, 20]), "evalset-0001")
        self.refused("trainset", self.mk("trainset", ids=[0, 1, 30]), "testset-0001")
        self.refused("trainset", self.mk("trainset", ids=[0, 1, 10]), "valset-0001")
        # Same input under another id, case and spacing still overlaps (Review Focus 2).
        src = self.mk("trainset", ids=TRAIN)
        row = full_row(20, self.task, id="other-id")
        row["input"]["messages"][0]["content"] = "  MESSAGE   20 "
        row["meta"]["cluster"] = "fresh-cluster"
        write_jsonl(os.path.join(src, "rows.jsonl"), [full_row(i, self.task) for i in TRAIN] + [row])
        files = {"rows": {"path": "rows.jsonl", "rows": 5},
                 "quality_report": {"path": "quality_report.json"}}
        update_manifest(src, files=files)
        msg = self.refused("trainset", src, "evalset-0001")
        self.assertIn("inputs 1", msg)
        self.assertEqual(self.ws.register("trainset", self.mk("trainset", ids=TRAIN)), "trainset-0001")

    def test_held_out_sets_must_be_disjoint(self):
        self.init()
        self.ws.register("testset", self.mk("testset", ids=TEST))
        self.refused("evalset", self.mk("evalset", ids=[20, 21, 31]), "testset-0001")
        self.ws.register("evalset", self.mk("evalset", ids=EVAL))
        self.refused("valset", self.mk("valset", ids=[10, 11, 22]), "evalset-0001")

    def test_refresh_sets_needs_approval(self):
        self.sets()
        new_val = self.mk("valset", ids=range(50, 54))
        self.refused("valset", new_val, "refresh_sets", "valset-0001")
        self.approve("refresh_sets")
        self.assertEqual(self.ws.register("valset", new_val), "valset-0002")
        self.assertEqual(self.ws.current("valset"), "valset-0002")

    def test_failed_quality_trainset_is_evidence_not_current(self):
        # R28: a trainset that failed its quality gate registers but never becomes current.
        failed = {"passed": False,
                  "checks": [{"name": "dup_rate", "value": 0.2, "threshold": 0.05, "passed": False},
                             {"name": "rows", "value": 4, "threshold": 2, "passed": True}]}
        self.init()
        self.ws.register("valset", self.mk("valset", ids=VAL))
        self.assertEqual(self.ws.register("trainset", self.mk("trainset", ids=TRAIN, quality=failed)),
                         "trainset-0001")
        self.assertIsNone(self.ws.current("trainset"))
        self.refused("candidate", self.mk("candidate", ids=VAL), "trainset-0001", "no current trainset")
        self.assertEqual(self.ws.register("trainset", self.mk("trainset", ids=range(40, 44))), "trainset-0002")
        self.assertEqual(self.ws.current("trainset"), "trainset-0002")
        self.assertEqual(self.ws.register("trainset", self.mk("trainset", ids=range(50, 54), quality=failed)),
                         "trainset-0003")
        self.assertEqual(Workspace(self.wsdir).current("trainset"), "trainset-0002")
        self.refused("candidate", self.mk("candidate", ids=VAL, trainset_id="trainset-0003"),
                     "trainset-0003 is not the current trainset trainset-0002")
        r = run_rsi("register", self.wsdir, "trainset", self.mk("trainset", ids=range(60, 64), quality=failed))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "trainset-0004 (quality gate: failed (dup_rate); never current)")
        self.assertEqual(Workspace(self.wsdir).current("trainset"), "trainset-0002")

    def test_parent_trainset_must_be_registered_trainset(self):
        self.sets()
        self.refused("trainset", self.mk("trainset", ids=range(40, 44), parent_trainset="trainset-0009"),
                     "parent_trainset")
        self.assertEqual(self.ws.register("trainset", self.mk("trainset", ids=range(40, 44),
                                                              parent_trainset="trainset-0001")),
                         "trainset-0002")


class TestModelArtifacts(WsCase):
    def test_candidate_predictions_must_match_valset(self):
        self.sets()
        self.refused("candidate", self.mk("candidate", ids=TRAIN), "predictions_val", "valset-0001")
        self.refused("candidate", self.mk("candidate", ids=[10, 11, 12]), "predictions_val")
        self.refused("candidate", self.mk("candidate", ids=VAL, trainset_id="trainset-0009"), "trainset_id")
        self.refused("candidate", self.mk("candidate", ids=VAL, valset_id="evalset-0001"), "valset_id")
        self.assertEqual(self.ws.state["budget_used"]["compute_hours"], 0)
        self.assertEqual(self.candidate(), "candidate-0001")
        self.assertEqual(self.ws.state["budget_used"]["compute_hours"], 1.5)
        self.candidate(cost={"compute_hours": 2})
        self.assertEqual(self.ws.state["budget_used"]["compute_hours"], 3.5)

    def test_candidate_must_use_current_trainset(self):
        self.sets()
        self.ws.register("trainset", self.mk("trainset", ids=range(40, 44)))
        self.refused("candidate", self.mk("candidate", ids=VAL), "trainset-0002")
        self.assertEqual(self.candidate(trainset_id="trainset-0002"), "candidate-0001")

    def test_candidate_needs_current_valset(self):
        self.init()
        self.refused("candidate", self.mk("candidate", ids=VAL), "valset")

    def test_search_trials_must_be_registered(self):
        self.sets()
        self.candidate(**TUNE)
        self.refused("search", self.mk("search"), "candidate-0002")
        self.candidate(**TUNE)
        # M3: trials that did not finish ok are not checked against the registry.
        fields = self.search_fields("candidate-0001", "candidate-0002")
        fields["trials"] += [{"candidate_id": None, "config": {"lr": 1.0}, "metric_val": None, "compute_hours": 1,
                              "status": "failed"},
                             {"candidate_id": "candidate-0009", "config": {"lr": 2.0}, "metric_val": None,
                              "compute_hours": 1, "status": "timeout"}]
        self.assertEqual(self.ws.register("search", self.mk("search", **fields)), "search-0001")

    def test_search_trials_share_one_code_version(self):
        # Task 14: the ok trials' candidates (default included) ran one version of the training code.
        self.sets()
        self.candidate(**TUNE)  # candidate-0001: the fixture code version, sha256 1111...
        self.candidate(code_version={"entry": "fixture-trainer", "sha256": "2" * 64}, **TUNE)
        self.candidate(**TUNE)  # candidate-0003: same code as candidate-0001
        before = (self.state_on_disk(), read_registry(self.wsdir))
        msg = self.refused("search", self.mk("search", **self.search_fields("candidate-0001", "candidate-0003",
                                                                            "candidate-0002")),
                           "different code versions", "code_version.sha256",
                           "1" * 64 + " (candidate-0001, candidate-0003)", "2" * 64 + " (candidate-0002)")
        self.assertIn("between rounds", msg)
        self.refused("search", self.mk("search", **self.search_fields("candidate-0002", "candidate-0001")),
                     "candidate-0001", "candidate-0002")
        self.assertEqual((self.state_on_disk(), read_registry(self.wsdir)), before)
        # A trial that did not finish ok is not compared.
        fields = self.search_fields("candidate-0001", "candidate-0003")
        fields["trials"].append({"candidate_id": "candidate-0002", "config": {"lr": 1.0}, "metric_val": None,
                                 "compute_hours": 1, "status": "failed"})
        self.assertEqual(self.ws.register("search", self.mk("search", **fields)), "search-0001")
        # Without a sha256 the trials cannot show one code version.
        for _ in range(2):
            self.candidate(code_version={"entry": "fixture-trainer", "sha256": None}, **TUNE)
        self.refused("search", self.mk("search", **self.search_fields("candidate-0004", "candidate-0005")),
                     "code_version.sha256 is null", "candidate-0004, candidate-0005")

    def test_search_trial_metric_val_comes_from_its_candidate(self):
        # Task 14 (R40): an ok trial's metric_val is its candidate's registered metrics_val[primary].
        self.sets()
        self.candidate(wrong=[10], **TUNE)  # candidate-0001: val auc 0.75
        self.candidate(**TUNE)  # candidate-0002: val auc 1
        fields = self.search_fields("candidate-0001", "candidate-0002")
        self.assertEqual([t["metric_val"] for t in fields["trials"]], [0.75, 1.0])
        fields["trials"][1]["metric_val"] = 0.9
        fields["gain_vs_default"]["delta"] = 0.9 - 0.75  # consistent with the trials, so validate passes
        src = self.mk("search", **fields)
        self.assertEqual(validate_artifact("search", src, self.task), [])
        before = (self.state_on_disk(), read_registry(self.wsdir))
        self.refused("search", src, "trials[1].metric_val is 0.9", "candidate-0002", "metrics_val.auc 1.0")
        self.assertEqual((self.state_on_disk(), read_registry(self.wsdir)), before)
        # Within 1e-9 it registers.
        fields = self.search_fields("candidate-0001", "candidate-0002")
        fields["trials"][1]["metric_val"] -= 5e-10
        fields["gain_vs_default"]["delta"] -= 5e-10
        self.assertEqual(self.ws.register("search", self.mk("search", **fields)), "search-0001")

    def test_delivery_needs_candidate_that_passed_train_gate(self):
        # R28: a candidate that failed its train gate is never delivered.
        self.sets()
        self.candidate(wrong=VAL, train_gate={"passed": False, "detail": "below the majority baseline"})
        self.refused("delivery", self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                     "candidate-0001", "train gate", "below the majority baseline")
        self.candidate()
        self.assertEqual(self.delivery(candidate_id="candidate-0002"), "delivery-0001")

    def test_register_prints_gate_results(self):
        # R28: `rsi register` prints the module gate result after the id.
        self.init()
        for kind, ids in (("testset", TEST), ("evalset", EVAL), ("valset", VAL)):
            self.ws.register(kind, self.mk(kind, ids=ids))
        r = run_rsi("register", self.wsdir, "trainset", self.mk("trainset", ids=TRAIN))
        self.assertEqual(r.stdout.strip(), "trainset-0001 (quality gate: passed)", r.stderr)
        r = run_rsi("register", self.wsdir, "candidate", self.mk("candidate", ids=VAL))
        self.assertEqual(r.stdout.strip(), "candidate-0001 (train gate: passed; val auc 1)", r.stderr)
        # Every validation prediction wrong: auc 0, below the majority baseline's 0.5.
        r = run_rsi("register", self.wsdir, "candidate", self.mk(
            "candidate", ids=VAL, wrong=VAL, train_gate={"passed": False, "detail": "below majority"}))
        self.assertEqual(r.stdout.strip(),
                         "candidate-0002 (train gate: failed (below majority); val auc 0; cannot be delivered)",
                         r.stderr)
        self.ws = Workspace(self.wsdir)
        r = run_rsi("register", self.wsdir, "search", self.mk("search", **self.search_fields(
            "candidate-0002", "candidate-0001")))
        self.assertEqual(r.stdout.strip(), "search-0001 (tune gate: passed; gain vs default auc +1 "
                                           "(best candidate-0001, default candidate-0002))", r.stderr)
        r = run_rsi("register", self.wsdir, "valset", self.mk("valset", ids=range(50, 54)))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")

    def test_delivery_test_predictions_need_approval(self):
        self.sets()
        self.candidate()
        src = add_predictions(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                              self.task, "predictions_test", TEST, testset_id="testset-0001")
        self.refused("delivery", src, "use_testset")
        self.approve("use_testset")
        self.assertEqual(self.ws.register("delivery", src), "delivery-0001")

    def test_delivery_test_predictions_must_match_testset(self):
        self.sets()
        self.candidate()
        self.approve("use_testset")
        src = add_predictions(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                              self.task, "predictions_test", EVAL, testset_id="testset-0001")
        self.refused("delivery", src, "predictions_test")
        src = add_predictions(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                              self.task, "predictions_test", TEST, testset_id="testset-0009")
        self.refused("delivery", src, "testset_id")

    def test_delivery_checks(self):
        self.sets()
        self.candidate()
        self.refused("delivery", self.mk("delivery", ids=EVAL, candidate_id="candidate-0009"),
                     "candidate-0009")
        self.refused("delivery", self.mk("delivery", ids=VAL, candidate_id="candidate-0001"),
                     "predictions_eval", "evalset-0001")
        self.refused("delivery", self.mk("delivery", ids=EVAL, candidate_id="candidate-0001",
                                         evalset_id="testset-0001"), "evalset_id")
        self.assertEqual(self.delivery(), "delivery-0001")

    def test_feedback_references_must_be_registered(self):
        self.sets()
        self.candidate()
        self.refused("feedback", self.mk("feedback", delivery_id="delivery-0001"), "delivery-0001")
        self.delivery()
        self.assertEqual(self.ws.register("feedback", self.mk("feedback")), "feedback-0001")


def shift_probs(src, rid, by):
    """Move `by` of probability from `no` to `yes` in the predictions_val row `rid` of a delivery source."""
    fp = os.path.join(src, "predictions_val.jsonl")
    with open(fp, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        if r["id"] == rid:
            r["probs"] = {"yes": r["probs"]["yes"] + by, "no": r["probs"]["no"] - by}
    write_jsonl(fp, rows)
    return src


class TestDeliveryReproduction(WsCase):
    """Task 14 (R38): the package's predictions on the valset inputs reproduce the candidate's."""

    def setUp(self):
        WsCase.setUp(self)
        self.sets()
        self.candidate()  # candidate-0001: predictions_val on VAL, all right

    def refused_atomic(self, src, *fragments):
        before = (self.state_on_disk(), read_registry(self.wsdir))
        msg = self.refused("delivery", src, *fragments)
        self.assertEqual((self.state_on_disk(), read_registry(self.wsdir)), before)
        return msg

    def test_matching_predictions_register(self):
        self.assertEqual(self.delivery(), "delivery-0001")
        # Within 1e-6 of the candidate's probabilities still reproduces it.
        self.assertEqual(self.ws.register("delivery", shift_probs(self.mk("delivery", ids=EVAL,
                                                                          candidate_id="candidate-0001"),
                                                                  "q11", 5e-7)), "delivery-0002")

    def test_missing_or_wrong_ids_refused(self):
        src = self.mk("delivery", ids=EVAL, candidate_id="candidate-0001")
        with open(os.path.join(src, "manifest.json"), encoding="utf-8") as f:
            files = json.load(f)["files"]
        del files["predictions_val"]
        update_manifest(src, files=files)
        self.refused_atomic(src, "predictions_val")
        self.refused_atomic(self.mk("delivery", ids=EVAL, val_ids=[10, 11, 12], candidate_id="candidate-0001"),
                            "predictions_val", "valset-0001", "missing predictions for 1 ids", "q13")
        self.refused_atomic(self.mk("delivery", ids=EVAL, val_ids=TRAIN, candidate_id="candidate-0001"),
                            "predictions_val", "valset-0001")

    def test_row_that_does_not_reproduce_refused(self):
        # One probability off by more than 1e-6; the flag still says consistent.
        msg = self.refused_atomic(shift_probs(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                                              "q11", 2e-6),
                                  "predictions_val", "1 of 4 rows", "candidate-0001", "q11", "probs.yes", "1e-6")
        self.assertIn("consistent_with_offline", msg)
        # A different decision (and so different probabilities).
        self.refused_atomic(self.mk("delivery", ids=EVAL, val_wrong=[12], candidate_id="candidate-0001"),
                            "predictions_val", "q12 (")
        # Only the decision differs, the probabilities are the candidate's.
        src = self.mk("delivery", ids=EVAL, candidate_id="candidate-0001")
        fp = os.path.join(src, "predictions_val.jsonl")
        with open(fp, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows[2]["decision"] = "no"  # q12: the candidate says yes with probability 0.9
        write_jsonl(fp, rows)
        self.refused_atomic(src, "predictions_val", "q12 (decision no, candidate yes)")

    def flipped(self, *rids, by=0.0):
        """A delivery source whose predictions_val give the other decision for `rids`, with probs.yes moved by `by`."""
        src = self.mk("delivery", ids=EVAL, candidate_id="candidate-0001")
        fp = os.path.join(src, "predictions_val.jsonl")
        with open(fp, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if r["id"] in rids:
                r["decision"] = "no" if r["decision"] == "yes" else "yes"
                r["probs"] = {"yes": r["probs"]["yes"] + by, "no": r["probs"]["no"] - by}
        write_jsonl(fp, rows)
        return src

    def test_decision_mismatch_allowance(self):
        # R44: near-boundary rows can flip at any tolerance; the task may allow a capped number of flips.
        self.refused_atomic(self.flipped("q12"), "q12 (decision no, candidate yes)",
                            "delivery.max_decision_mismatch_rate", "allows 0")
        self.task["delivery"] = {"reproduction_tolerance": 1e-3, "max_decision_mismatch_rate": 0.3}
        self.wsdir = os.path.join(self.root, "ws-flips")
        self.sets()
        self.candidate()  # 4 valset rows: floor(0.3 * 4) = 1 flip allowed
        # A flipped row whose probabilities moved beyond the tolerance does not count as a flip.
        self.refused_atomic(self.flipped("q12", by=-2e-3), "q12 (decision no, candidate yes; probs.yes")
        # More flips than allowed.
        self.refused_atomic(self.flipped("q11", "q12", by=5e-4), "2 rows", "q11", "q12", "allows 1",
                            "max_decision_mismatch_rate 0.3")
        r = run_rsi("register", self.wsdir, "delivery", self.flipped("q12", by=-5e-4))
        self.assertEqual((r.returncode, r.stdout), (0, "delivery-0001 (decision mismatches: 1)\n"), r.stderr)
        r = run_rsi("register", self.wsdir, "delivery", self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"))
        self.assertEqual((r.returncode, r.stdout), (0, "delivery-0002\n"), r.stderr)
        self.assertEqual(Workspace(self.wsdir).decision_mismatches("delivery-0001"), 1)

    def test_task_reproduction_tolerance(self):
        # R43: a task may allow a larger difference (e.g. bf16 at another batch size); the default is 1e-6
        # (above), and the decisions must still be equal.
        self.refused_atomic(shift_probs(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"), "q11", 1e-4),
                            "q11", "within 1e-6", "delivery.reproduction_tolerance")
        self.task["delivery"] = {"reproduction_tolerance": 1e-3}
        self.wsdir = os.path.join(self.root, "ws-tolerant")
        self.sets()
        self.candidate()
        self.assertEqual(self.ws.register("delivery", shift_probs(self.mk("delivery", ids=EVAL,
                                                                          candidate_id="candidate-0001"),
                                                                  "q11", 1e-4)), "delivery-0001")
        self.refused_atomic(shift_probs(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"), "q11", 2e-3),
                            "q11", "probs.yes", "within 0.001")
        src = self.mk("delivery", ids=EVAL, candidate_id="candidate-0001")
        fp = os.path.join(src, "predictions_val.jsonl")
        with open(fp, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows[2]["decision"] = "no"
        write_jsonl(fp, rows)
        self.refused_atomic(src, "q12 (decision no, candidate yes)")

    def test_valset_id_is_the_candidates_valset(self):
        self.refused_atomic(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001", valset_id="valset-0009"),
                            "valset_id", "valset-0009")
        self.refused_atomic(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001", valset_id="evalset-0001"),
                            "valset_id", "evalset-0001")
        # After a valset refresh the candidate is still reproduced on the valset it was validated on.
        self.approve("refresh_sets")
        self.ws.register("valset", self.mk("valset", ids=range(50, 54)))
        self.refused_atomic(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001", valset_id="valset-0002",
                                    val_ids=range(50, 54)),
                            "valset_id", "valset-0002", "candidate-0001", "valset-0001")
        self.assertEqual(self.delivery(), "delivery-0001")


class TestJudgeArtifacts(WsCase):
    def ready(self):
        self.sets()
        self.candidate()
        self.delivery()

    def test_evaluation_use_limit(self):
        self.ready()
        limit = self.task["eval_use_limit"]
        for i in range(limit):
            self.assertEqual(self.ws.register("evaluation", self.eval_src()), "evaluation-%04d" % (i + 1))
            self.assertEqual(self.ws.state["eval_use_count"], i + 1)
        self.refused("evaluation", self.eval_src(), "eval_use_limit")
        self.assertEqual(self.ws.state["eval_use_count"], limit)
        # Refreshing the evaluation set (approved) resets the count.
        self.approve("refresh_sets")
        self.ws.register("evalset", self.mk("evalset", ids=range(60, 64)))
        self.assertEqual(self.ws.state["eval_use_count"], 0)

    def test_evaluation_must_match_delivery(self):
        self.ready()
        self.refused("evaluation", self.eval_src(delivery_id="delivery-0009"), "delivery-0009")
        self.candidate()
        self.refused("evaluation", self.eval_src(candidate_id="candidate-0002"), "candidate_id")

    def test_evaluation_best_before_must_match_state(self):
        self.ready()
        bb = {"version": "V1", "evaluation_id": "evaluation-0001"}
        self.refused("evaluation", self.eval_src(best_before=bb), "best_before")
        self.ws.register("evaluation", self.eval_src())
        self.ws.state["versions"].append({"version": "V1", "candidate_id": "candidate-0001",
                                          "delivery_id": "delivery-0001",
                                          "evaluation_id": "evaluation-0001", "round": 1})
        self.ws.state["best"] = "V1"
        self.ws.save_state()
        self.refused("evaluation", self.eval_src(), "best_before")
        self.refused("evaluation", self.eval_src(best_before={"version": "V1", "evaluation_id": "evaluation-0009"}),
                     "evaluation-0009")
        self.refused("evaluation", self.eval_src(best_before={"version": "V2", "evaluation_id": "evaluation-0001"}),
                     "V2")
        self.assertEqual(self.ws.register("evaluation", self.eval_src(best_before=bb)), "evaluation-0002")

    def best_v1(self):
        """candidate-0001, delivery-0001 and evaluation-0001 promoted to V1."""
        self.ready()
        self.ws.register("evaluation", self.eval_src())
        self.ws.promote("evaluation-0001")

    def milestone_src(self, **fields):
        fields.setdefault("best_before", {"version": "V1", "evaluation_id": "evaluation-0001"})
        fields.setdefault("recommendation", {"decision": "keep", "reason": "milestone release"})
        fields.setdefault("test_ids", TEST)
        return self.eval_src(uses_testset=True, **fields)

    def test_evaluation_uses_testset_needs_approval(self):
        self.best_v1()
        self.approve("use_testset")
        self.ws.register("delivery", add_predictions(
            self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
            self.task, "predictions_test", TEST, testset_id="testset-0001"))
        self.assertEqual(self.ws.state["approvals"][0]["used_by"], "delivery-0002")
        # The delivery consumed that approval; the evaluation needs its own.
        src = self.milestone_src(delivery_id="delivery-0002")
        self.refused("evaluation", src, "use_testset", "delivery-0002")
        self.approve("use_testset")
        self.assertEqual(self.ws.register("evaluation", src), "evaluation-0002")
        self.assertEqual([a.get("used_by") for a in Workspace(self.wsdir).state["approvals"]],
                         ["delivery-0002", "evaluation-0002"])

    def test_evaluation_uses_testset_needs_test_predictions(self):
        self.best_v1()
        self.approve("use_testset")
        self.refused("evaluation", self.milestone_src(), "predictions_test")

    def test_plan_sets_round_plan_id(self):
        self.ready()
        self.refused("plan", self.mk("plan"), "evaluation-0001")
        self.ws.register("evaluation", self.eval_src())
        self.assertEqual(self.ws.register("plan", self.mk("plan")), "plan-0001")
        self.assertEqual(self.ws.state["rounds"]["1"]["plan_id"], "plan-0001")

    def test_record_artifacts_must_be_registered(self):
        self.sets()
        self.refused("record", self.mk("record", artifacts=["trainset-0002"]), "trainset-0002")
        self.assertEqual(self.ws.register("record", self.mk("record", artifacts=["trainset-0001"])),
                         "record-0001")


class TestApprovalsAndRefresh(WsCase):
    def ready(self):
        self.sets()
        self.candidate()
        self.delivery()

    def test_evalset_refresh_with_same_rows_refused(self):
        # Reviewer repro: re-registering evalset-0001's rows as a "refreshed" evalset
        # would reset eval_use_count and lift the use cap.
        self.ready()
        for _ in range(self.task["eval_use_limit"]):
            self.ws.register("evaluation", self.eval_src())
        self.approve("refresh_sets")
        self.refused("evalset", self.mk("evalset", ids=EVAL), "evalset-0001", "ids 4, inputs 4")
        self.assertEqual(self.ws.state["eval_use_count"], self.task["eval_use_limit"])
        # Same inputs under new ids are still the same evaluation rows.
        src = self.mk("evalset", ids=EVAL)
        for name in ("inputs.jsonl", "labels.jsonl"):
            fp = os.path.join(src, name)
            with open(fp, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            for r in rows:
                r["id"] = "new-" + r["id"]
                if "meta" in r and "cluster" in r["meta"]:
                    r["meta"]["cluster"] = "new-" + r["meta"]["cluster"]
            write_jsonl(fp, rows)
        self.refused("evalset", src, "evalset-0001", "ids 0, inputs 4")
        # Genuinely new rows are accepted; earlier evalsets stay off limits.
        self.assertEqual(self.ws.register("evalset", self.mk("evalset", ids=range(60, 64))), "evalset-0002")
        self.approve("refresh_sets")
        self.refused("evalset", self.mk("evalset", ids=EVAL), "evalset-0001")
        self.assertEqual(self.ws.register("evalset", self.mk("evalset", ids=range(70, 74))), "evalset-0003")

    def test_refresh_approval_is_single_use(self):
        self.sets()
        self.approve("refresh_sets")
        self.assertEqual(self.ws.register("valset", self.mk("valset", ids=range(50, 54))), "valset-0002")
        self.assertEqual(self.ws.state["approvals"][0]["used_by"], "valset-0002")
        self.assertEqual(Workspace(self.wsdir).state["approvals"][0]["used_by"], "valset-0002")
        self.assertFalse(self.ws.has_approval("refresh_sets"))
        self.refused("valset", self.mk("valset", ids=range(54, 58)), "refresh_sets", "valset-0002")
        self.approve("refresh_sets")
        self.assertEqual(self.ws.register("valset", self.mk("valset", ids=range(54, 58))), "valset-0003")
        self.assertEqual([a.get("used_by") for a in self.ws.state["approvals"]], ["valset-0002", "valset-0003"])

    def test_refused_registration_leaves_approval_unconsumed(self):
        self.sets()
        self.approve("refresh_sets")
        # Overlaps the current evalset: refused after the approval check passed.
        self.refused("valset", self.mk("valset", ids=[50, 51, 20]), "evalset-0001")
        self.assertNotIn("used_by", self.ws.state["approvals"][0])
        self.assertNotIn("used_by", Workspace(self.wsdir).state["approvals"][0])
        self.assertTrue(self.ws.has_approval("refresh_sets"))
        self.assertEqual(self.ws.register("valset", self.mk("valset", ids=range(50, 54))), "valset-0002")
        self.assertEqual(self.ws.state["approvals"][0]["used_by"], "valset-0002")

    def test_task_change_approval_is_single_use(self):
        self.init()
        self.approve("task_change")
        for version in (2, 3):
            t = binary_task()
            t["version"] = version
            d = make_task_dir(self.src("task-v%d" % version), t)
            if version == 3:
                self.refused("task", d, "task_change", "task-0002")
                self.approve("task_change")
            self.assertEqual(self.ws.register("task", d), "task-%04d" % version)


class TestCliCommands(WsCase):
    def test_cli_init_and_register(self):
        r = run_rsi("init", self.wsdir, "--task", make_task_dir(self.src("task"), self.task))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "task-0001")
        r = run_rsi("register", self.wsdir, "testset", self.mk("testset", ids=TEST))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "testset-0001")
        r = run_rsi("register", self.wsdir, "evalset", self.mk("evalset", ids=[20, 30]))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout, "")
        self.assertIn("testset-0001", r.stderr)
        r = run_rsi("init", self.wsdir, "--task", self.src("task"))
        self.assertEqual(r.returncode, 1)
        self.assertTrue(r.stderr.strip())

    def test_cli_validate(self):
        task_dir = make_task_dir(self.src("task"), self.task)
        task_json = os.path.join(task_dir, "task.json")
        r = run_rsi("validate", "task", task_dir)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "OK"), r.stderr)
        src = self.mk("trainset")
        r = run_rsi("validate", "trainset", src, "--task", task_json)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "OK"), r.stderr)
        os.remove(os.path.join(src, "rows.jsonl"))
        update_manifest(src, quality={"passed": True, "checks": []}, label_method=3)
        r = run_rsi("validate", "trainset", src, "--task", task_json)
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("OK", r.stdout)
        self.assertIn("label_method", r.stderr)
        r = run_rsi("validate", "trainset", src)
        self.assertEqual(r.returncode, 1)
        self.assertIn("--task", r.stderr)
        r = run_rsi("validate", "task", task_dir, "--task", task_json)
        self.assertEqual(r.returncode, 1)
        self.assertIn("--task", r.stderr)

    def test_cli_validate_task_errors_one_per_line(self):
        t = binary_task()
        t["eval_use_limit"] = 0
        t["stop"]["max_rounds"] = 0
        r = run_rsi("validate", "task", make_task_dir(self.src("bad"), t))
        self.assertEqual(r.returncode, 1)
        lines = r.stderr.strip().splitlines()
        self.assertEqual(len(lines), 2, r.stderr)

    def test_status_output(self):
        self.sets()
        self.candidate()
        self.delivery()
        self.ws.register("evaluation", self.eval_src())
        self.ws.register("plan", self.mk("plan"))
        r = run_rsi("status", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        for line in ("task: spam-check v1 (task-0001)", "round: 1", "trainset: trainset-0001",
                     "valset: valset-0001", "evalset: evalset-0001", "testset: testset-0001",
                     "best: none", "eval uses: 1/3", "unapproved plans: plan-0001", "stopped: no",
                     "compute hours: 1.5"):
            self.assertIn(line, out)
        self.approve("plan", "plan-0001")
        self.ws.state["versions"].append({"version": "V1", "candidate_id": "candidate-0001",
                                          "delivery_id": "delivery-0001",
                                          "evaluation_id": "evaluation-0001", "round": 1})
        self.ws.state["best"] = "V1"
        self.ws.state["stopped"] = {"reason": "patience reached", "at": "2026-11-21T00:00:00Z"}
        self.ws.save_state()
        out = run_rsi("status", self.wsdir).stdout
        for line in ("best: V1 (candidate-0001, evaluation-0001)", "unapproved plans: none",
                     "stopped: patience reached"):
            self.assertIn(line, out)

    def test_status_lists_round_artifacts(self):
        # T9: the current round's registered artifacts, with module gates and the val primary metric.
        failed = {"passed": False, "checks": [{"name": "dup_rate", "value": 0.2, "threshold": 0.05,
                                               "passed": False}]}
        self.sets()
        self.ws.register("trainset", self.mk("trainset", ids=range(40, 44), quality=failed))
        self.candidate()
        self.candidate(wrong=VAL, train_gate={"passed": False, "detail": "below majority"})
        self.delivery()
        self.ws.register("evaluation", self.eval_src())
        r = run_rsi("status", self.wsdir)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        start = lines.index("round 1 artifacts (10):")
        self.assertEqual(lines[start + 1:], [
            "  task-0001 task",
            "  testset-0001 testset",
            "  evalset-0001 evalset",
            "  valset-0001 valset",
            "  trainset-0001 trainset (quality gate: passed)",
            "  trainset-0002 trainset (quality gate: failed (dup_rate); never current)",
            "  candidate-0001 candidate (train gate: passed; val auc 1)",
            "  candidate-0002 candidate (train gate: failed (below majority); val auc 0; cannot be delivered)",
            "  delivery-0001 delivery",
            "  evaluation-0001 evaluation",
        ])
        self.assertIn("trainset: trainset-0001", r.stdout)

    def test_ab_draft_refusal_says_drafts_are_not_registered(self):
        # M22: a data module's A/B candidate (trainset not registered, producer data) is a draft.
        self.sets()
        msg = self.refused("candidate", self.mk("candidate", ids=VAL, trainset_id="unregistered", producer="data"),
                           "trainset_id", "unregistered")
        self.assertIn("A/B drafts are not registered", msg)
        self.assertIn("record", msg)
        # An ordinary wrong trainset id keeps its plain message.
        msg = self.refused("candidate", self.mk("candidate", ids=VAL, trainset_id="trainset-0009"), "trainset_id")
        self.assertNotIn("drafts", msg)

    def test_nan_compute_hours_refused(self):
        # M4: NaN and Infinity are not numbers; they would poison budget_used.
        self.sets()
        for bad in (float("nan"), float("inf")):
            self.refused("candidate", self.mk("candidate", ids=VAL, cost={"compute_hours": bad}),
                         "cost.compute_hours")
        self.assertEqual(Workspace(self.wsdir).state["budget_used"]["compute_hours"], 0)

    def test_status_on_missing_workspace(self):
        r = run_rsi("status", os.path.join(self.root, "nope"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("STATE.json", r.stderr)


if __name__ == "__main__":
    unittest.main()
