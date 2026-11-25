import contextlib
import copy
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

from fixtures import (TIME, add_predictions, binary_task, corpus_rows, filled_time_task, full_row, make_artifact,
                      make_task_dir, update_manifest, write_jsonl)
from rsi import cli
from rsi import datasplit as DS
from rsi.artifacts import validate_artifact
from rsi.workspace import RsiError, StoppedError, Workspace, _single_use_needed, content_sha256

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Disjoint row-id ranges (q<i>) for the four data sets.
TRAIN, VAL, EVAL, TEST = range(0, 4), range(10, 14), range(20, 24), range(30, 34)


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

    def init(self, task=None, root=None):
        """A workspace for `task` (default: self.task); `root` names a further one beside the first."""
        if task is not None:
            self.task = task
        if root is not None:
            self.wsdir = os.path.join(self.root, root)
        self.ws = Workspace.init(self.wsdir, make_task_dir(self.src("task" if root is None else "task-" + root),
                                                           self.task))
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
        trial's metric_val copied from its candidate's metrics_val (as rsi-train does), and the gain."""
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
                                        "evalset": None, "testset": None, "corpus": None, "pool": None})
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
        # work/ holds the run's files (run.jsonl, the dispatches' output directories, the operator's notes.md).
        self.assertEqual(os.listdir(os.path.join(self.wsdir, "work")), [])

    def test_init_and_open_refusals(self):
        t = binary_task()
        del t["decision"]["positive"]
        with self.assertRaises(RsiError) as cm:
            Workspace.init(self.wsdir, make_task_dir(self.src("bad"), t))
        self.assertIn("positive", str(cm.exception))
        self.assertFalse(os.path.exists(self.wsdir))  # an invalid task creates nothing
        d = make_task_dir(self.src("no-md"), self.task)
        os.remove(os.path.join(d, "TASK.md"))
        with self.assertRaises(RsiError) as cm:
            Workspace.init(self.wsdir, d)
        self.assertIn("TASK.md", str(cm.exception))
        with self.assertRaises(RsiError):
            Workspace(self.root)  # not a workspace
        self.init()
        with self.assertRaises(RsiError):
            Workspace.init(self.wsdir, make_task_dir(self.src("task2"), self.task))  # already one

    def test_save_state_is_atomic_and_persists(self):
        ws = self.init()
        ws.state["no_promotion_streak"] = 2
        ws.save_state()
        self.assertEqual(Workspace(self.wsdir).state["no_promotion_streak"], 2)
        # No temp file is left; STATE.lock is the lock every STATE change holds (Workspace.transaction).
        self.assertEqual(sorted(os.listdir(self.wsdir)),
                         ["STATE.json", "STATE.lock", "artifacts", "registry.jsonl", "work"])


class TestTaskVersion(WsCase):
    def test_task_new_version_needs_approval(self):
        self.init()
        t2 = binary_task()
        t2["version"] = 2
        t2["eval_use_limit"] = 5
        d = make_task_dir(self.src("task-v2"), t2)
        self.refused("task", d, "task_change")
        self.approve("task_change")
        self.ws.state["approvals"][0]["round"] = 0  # an approval from an earlier round does not count
        self.ws.save_state()
        self.refused("task", d, "task_change")
        self.ws.state["approvals"][0]["round"] = 1
        self.ws.save_state()
        self.assertEqual(self.ws.register("task", d), "task-0002")
        self.assertEqual(self.ws.state["approvals"][0]["used_by"], "task-0002")
        self.assertEqual(self.ws.state["task_version"], 2)
        self.assertEqual(self.ws.current("task"), "task-0002")
        self.assertEqual(self.ws.task()["eval_use_limit"], 5)
        # Artifacts must now cite the new version.
        self.refused("testset", self.mk("testset", ids=TEST), "task_version")
        self.assertEqual(self.ws.register("testset", self.mk("testset", ids=TEST, task_version=2)),
                         "testset-0001")
        # The approval is single use: the next version needs a new one.
        t3 = binary_task()
        t3["version"] = 3
        d = make_task_dir(self.src("task-v3"), t3)
        self.refused("task", d, "task_change", "task-0002")
        self.approve("task_change")
        self.assertEqual(self.ws.register("task", d), "task-0003")

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

    def test_split_fill_needs_no_approval_and_freezes_with_the_sealed_sets(self):
        t1 = binary_task()
        t1["data"]["split"] = {"sizes": {"trainset": 4}}
        self.init(task=t1)
        t2 = copy.deepcopy(t1)
        t2["version"] = 2
        t2["data"]["split"] = dict(DS.filled(dict(TIME, sizes={"trainset": 4})))
        with open(os.path.join(self.ws.path("task-0001"), "TASK.md"), "rb") as f:
            raw = f.read()
        text, md = raw.decode("utf-8"), DS.fill_task_md(raw, t2["data"]["split"]).decode("utf-8")
        self.assertEqual(self.ws.split_fill_errors(t2, md), [])
        changed = copy.deepcopy(t2)
        changed["data"]["split"]["sizes"]["trainset"] = 5  # a fill keeps what the current rule writes
        self.assertIn("sizes.trainset", " ".join(self.ws.split_fill_errors(changed)))
        msg = self.refused("task", make_task_dir(self.src("v2-md"), dict(copy.deepcopy(t1), version=2), md),
                           "task_change")  # TASK.md changed only: no fill reasons
        self.assertNotIn("data.split", msg)
        for n, other in enumerate(("label 'yes' now means the opposite\n",  # a fill keeps TASK.md ...
                                   text + "\n## 切分规则\n\n标签 yes 现在表示相反\n")):  # ... and appends only the rule
            self.refused("task", make_task_dir(self.src("v2-text-%d" % n), t2, other), "task_change", "TASK.md")
        self.assertEqual(self.ws.register("task", make_task_dir(self.src("v2"), t2, md)), "task-0002")
        self.assertEqual(self.ws.state["approvals"], [])  # no task_change asked or used
        for n, change in enumerate(({"eval_use_limit": 9},  # not a fill: another field
                                    {"data": dict(t2["data"], split=dict(t2["data"]["split"], seed=5))})):  # complete
            t3 = dict(copy.deepcopy(t2), version=3, **change)
            self.refused("task", make_task_dir(self.src("v3-%d" % n), t3), "task_change")
        self.ws.register("testset", self.mk("testset", ids=TEST, task_version=2))
        self.approve("task_change")
        t3 = copy.deepcopy(t2)
        t3["version"] = 3
        t3["data"]["split"]["seed"] = 5
        self.refused("task", make_task_dir(self.src("v3-split"), t3), "fixed")
        t3 = copy.deepcopy(t2)
        t3["version"] = 3
        t3["eval_use_limit"] = 9
        self.assertEqual(self.ws.register("task", make_task_dir(self.src("v3-ok"), t3)), "task-0003")
        # A repair of a registration cut off before its STATE save asks the same: the fill used no approval.
        self.ws.state["current"]["task"] = "task-0001"
        self.assertEqual(_single_use_needed(self.ws, "task", t2, "task-0002"), [])
        self.assertEqual(_single_use_needed(self.ws, "task", t3, "task-0003"), ["task_change"])


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
        for fn in (self.ws.path, self.ws.manifest, self.ws.entry):  # an id that is not registered
            with self.assertRaises(RsiError):
                fn("trainset-0001")

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
        # A refused sealed set stays in place, unchanged.
        src = self.mk("evalset", ids=range(60, 64), task_id="other-task")
        digest = content_sha256(src)
        self.refused("evalset", src, "task_id")
        self.assertEqual(content_sha256(src), digest)
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
        # R4 still holds after a sealed set's move: a new directory at the old path is refused.
        src = self.mk("testset", "ts", ids=TEST)
        self.ws.register("testset", src)
        self.refused("testset", src, "not a directory")
        make_artifact("testset", src, self.task, ids=range(40, 44), notes="rewritten")
        self.refused("testset", src, "already registered", "testset-0001")
        self.assertTrue(os.path.isdir(src))

    def test_common_checks(self):
        self.init()
        src = self.mk("testset", ids=TEST)
        os.remove(os.path.join(src, "labels.jsonl"))
        self.refused("testset", src, "labels")  # an invalid artifact
        self.refused("bogus", src, "unknown kind")
        self.refused("testset", self.mk("testset", ids=TEST, task_id="other-task"), "task_id")
        self.refused("testset", self.mk("testset", ids=TEST, parents=["valset-0009"]), "valset-0009")
        self.refused("testset", self.mk("testset", ids=TEST, id="testset-0007"), "must be null")
        ts = self.ws.register("testset", self.mk("testset", ids=TEST))
        self.assertEqual(self.ws.register("evalset", self.mk("evalset", ids=EVAL, parents=[ts])),
                         "evalset-0001")


class TestDataSets(WsCase):
    def test_sets_must_be_disjoint(self):
        self.init()
        self.ws.register("testset", self.mk("testset", ids=TEST))
        self.refused("evalset", self.mk("evalset", ids=[20, 21, 31]), "testset-0001")
        self.ws.register("evalset", self.mk("evalset", ids=EVAL))
        self.refused("valset", self.mk("valset", ids=[10, 11, 22]), "evalset-0001")
        self.ws.register("valset", self.mk("valset", ids=VAL))
        self.refused("trainset", self.mk("trainset", ids=[0, 1, 20]), "evalset-0001")
        self.refused("trainset", self.mk("trainset", ids=[0, 1, 30]), "testset-0001")
        self.refused("trainset", self.mk("trainset", ids=[0, 1, 10]), "valset-0001")
        # Same input under another id, case and spacing still overlaps (Review Focus 2).
        src = self.mk("trainset", ids=TRAIN)
        row = full_row(20, self.task, id="other-id")
        row["state"]["messages"][0]["content"] = "  MESSAGE   20 "
        row["meta"]["cluster"] = "fresh-cluster"
        write_jsonl(os.path.join(src, "rows.jsonl"), [full_row(i, self.task) for i in TRAIN] + [row])
        files = {"rows": {"path": "rows.jsonl", "rows": 5},
                 "quality_report": {"path": "quality_report.json"}}
        update_manifest(src, files=files)
        msg = self.refused("trainset", src, "evalset-0001")
        self.assertIn("inputs 1", msg)
        self.assertEqual(self.ws.register("trainset", self.mk("trainset", ids=TRAIN)), "trainset-0001")

    def test_corpus_and_pool_registration(self):
        task = filled_time_task()
        self.init(task=task)
        rows = corpus_rows(task); by_step = lambda lo, hi: [r for r in rows if lo <= r["meta"]["extra"]["step"] <= hi]
        del self.ws.state["current"]["pool"]; self.ws.save_state()    # an old workspace's STATE
        self.assertIsNone(Workspace(self.wsdir).current("pool"))
        # the task names its raw data: no set before the corpus
        self.refused("testset", make_artifact("testset", self.src("t0"), task, rows=by_step(11, 12)),
                     "register the corpus")
        src = make_artifact("corpus", self.src("corpus"), task, rows=rows)
        self.assertEqual(self.ws.register("corpus", src), "corpus-0001"); self.assertFalse(os.path.exists(src))
        test = by_step(11, 12); bad = copy.deepcopy(test)
        bad[0]["target"] = dict((k, 1.0 - v) for k, v in bad[0]["target"].items())   # not the corpus row's label
        self.refused("testset", make_artifact("testset", self.src("t-bad"), task, rows=bad), "corpus-0001")
        self.ws.register("testset", make_artifact("testset", self.src("t"), task, rows=test))
        pool = by_step(1, 6)
        fresh = corpus_rows(task, steps=13)[-1]                                    # r13_3: no row of the corpus
        moved = copy.deepcopy(pool); moved[0]["meta"]["cluster"] = "k0"; moved[1]["meta"]["state_id"] = "a99"
        for name, rs, frag in (("p-bad", pool + test[:1] + [fresh], "testset-0001"),
                               ("p-new", pool + [fresh], "1 rows of the pool are not rows of the current corpus"),
                               ("p-moved", moved, "2 rows of the pool have another state_id or cluster")):
            msg = self.refused("pool", make_artifact("pool", self.src(name), task, rows=rs), frag)
            self.assertEqual("corpus-0001" in msg, name != "p-bad", msg)  # the corpus is read once the rest passed
        self.assertEqual(self.ws.register("pool", make_artifact("pool", self.src("p"), task, rows=pool)), "pool-0001")
        self.ws.register("trainset", make_artifact("trainset", self.src("tr"), task, rows=pool[:6]))  # in the pool
        self.refused("pool", make_artifact("pool", self.src("p2"), task, rows=pool[:-1]), "refresh_sets")
        # unit state_id: a set sharing an address with the testset is refused (second workspace, its task names no raw
        # data, so its sets need no corpus; steps 7-8 and 11-12 use the same 8 addresses)
        t2 = filled_time_task(unit="state_id"); t2["data"]["sources"] = []; self.init(task=t2, root="ws2")
        self.ws.register("testset", make_artifact("testset", self.src("t2"), t2, rows=by_step(11, 12)))
        self.refused("valset", make_artifact("valset", self.src("v2"), t2, rows=by_step(7, 8)), "state_ids")
        # a corpus vouches only for the sets registered after it
        self.refused("corpus", make_artifact("corpus", self.src("c2"), t2, rows=rows), "testset-0001", "corpus first")

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
        # B1: the four refused candidates (four sources, 1.5 hours each) were trained: their compute counts.
        self.assertEqual(self.ws.state["budget_used"]["compute_hours"], 6)
        self.assertEqual(self.candidate(), "candidate-0001")
        self.assertEqual(self.ws.state["budget_used"]["compute_hours"], 7.5)
        self.candidate(cost={"compute_hours": 2})
        self.assertEqual(self.ws.state["budget_used"]["compute_hours"], 9.5)

    def test_candidate_needs_the_current_sets(self):
        self.init()
        self.refused("candidate", self.mk("candidate", ids=VAL), "valset")
        shutil.rmtree(self.wsdir)
        self.sets()
        self.ws.register("trainset", self.mk("trainset", ids=range(40, 44)))
        self.refused("candidate", self.mk("candidate", ids=VAL), "trainset-0002")
        self.assertEqual(self.candidate(trainset_id="trainset-0002"), "candidate-0001")

    def test_config_outside_search_space_is_refused(self):
        # D5: in a task with a search_space, a candidate says which configuration it trained (config_request): a key
        # of the space takes an allowed value, a key outside it null or the default configuration's value; and it was
        # trained by the task's train command.
        self.task["search_space"] = {"overrides.onejev.lr": {"values": [5e-6, 1e-5, 2e-5],
                                                             "dead": {"5e-06": "diverged before"}},
                                     "overrides.onejev.warmup_frac": {"min": 0.0, "max": 0.1}, "seed": [0, 1, 2]}
        self.task["resources"] = {"train_command": "modules/train/train", "onejev": {"default_config": {
            "base": "joey-1", "recipe": "recipes/joey_full_1gpu.yaml", "overrides": {}, "seed": 0, "max_steps": None}}}
        self.sets()
        ok = {"base": "joey-1", "recipe": None,  # base: the default's; recipe: not given (the default)
              "overrides": {"onejev": {"lr": 1e-5, "warmup_frac": 0.03}}, "seed": 1, "max_steps": None}
        entry = {"entry": "modules/train/train", "sha256": "1" * 64}
        for name, request, cv, words in (
                ("a key the space does not have", dict(ok, overrides={"onejev": {"lr": 1e-5, "weight_decay": 0.1}}),
                 entry, ["config_request", "overrides.onejev.weight_decay", "search_space"]),
                ("a value not in the list", dict(ok, seed=3), entry, ["config_request", "seed", "3"]),
                ("a value outside the range", dict(ok, overrides={"onejev": {"warmup_frac": 0.2}}), entry,
                 ["config_request", "overrides.onejev.warmup_frac", "0.2"]),
                ("a key outside the space off its default", dict(ok, base="joey-2"), entry,
                 ["config_request", "base", "joey-2", "joey-1"]),
                ("no config_request", None, entry, ["config_request", "search_space"]),
                ("not the task's train command", ok, {"entry": "/elsewhere/train", "sha256": "1" * 64},
                 ["code_version.entry", "/elsewhere/train", "modules/train/train"])):
            with self.subTest(name):
                src = self.mk("candidate", ids=VAL, code_version=cv, config_request=request)
                if request is None:
                    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as f:
                        m = json.load(f)
                    del m["config_request"]
                    with open(os.path.join(src, "manifest.json"), "w", encoding="utf-8") as f:
                        json.dump(m, f)
                self.refused("candidate", src, *words)
        self.assertEqual(self.candidate(config_request=ok, code_version=entry), "candidate-0001")
        # A flat override key names the same path; a dead value is still in the space (trsi decides about dead ends).
        self.assertEqual(self.candidate(config_request=dict(ok, overrides={"onejev.lr": 2e-5}), code_version=entry),
                         "candidate-0002")
        self.assertEqual(self.candidate(config_request=dict(ok, overrides={"onejev": {"lr": 5e-6}}), code_version=entry),
                         "candidate-0003")

    def test_search_trials_must_be_registered(self):
        self.sets()
        self.candidate()
        self.refused("search", self.mk("search"), "candidate-0002")
        self.candidate()
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
        self.candidate()  # candidate-0001: the fixture code version, sha256 1111...
        self.candidate(code_version={"entry": "fixture-trainer", "sha256": "2" * 64})
        self.candidate()  # candidate-0003: same code as candidate-0001
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
            self.candidate(code_version={"entry": "fixture-trainer", "sha256": None})
        self.refused("search", self.mk("search", **self.search_fields("candidate-0004", "candidate-0005")),
                     "code_version.sha256 is null", "candidate-0004, candidate-0005")

    def test_search_trial_metric_val_comes_from_its_candidate(self):
        # Task 14 (R40): an ok trial's metric_val is its candidate's registered metrics_val[primary].
        self.sets()
        self.candidate(wrong=[10])  # candidate-0001: val auc 0.75
        self.candidate()  # candidate-0002: val auc 1
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

    def test_delivery_test_predictions(self):
        self.sets()
        self.candidate()
        src = add_predictions(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                              self.task, "predictions_test", TEST, testset_id="testset-0001")
        self.refused("delivery", src, "use_testset")
        self.approve("use_testset")
        bad = add_predictions(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                              self.task, "predictions_test", EVAL, testset_id="testset-0001")
        self.refused("delivery", bad, "predictions_test")
        bad = add_predictions(self.mk("delivery", ids=EVAL, candidate_id="candidate-0001"),
                              self.task, "predictions_test", TEST, testset_id="testset-0009")
        self.refused("delivery", bad, "testset_id")
        self.assertEqual(self.ws.register("delivery", src), "delivery-0001")

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
        # R43: a task may allow a larger difference (e.g. bf16 at another batch size); the default is 1e-6, and the
        # decisions must still be equal.
        self.assertEqual(self.delivery(), "delivery-0001")
        # Within 1e-6 of the candidate's probabilities still reproduces it.
        self.assertEqual(self.ws.register("delivery", shift_probs(self.mk("delivery", ids=EVAL,
                                                                          candidate_id="candidate-0001"),
                                                                  "q11", 5e-7)), "delivery-0002")
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
        # A milestone evaluation needs a delivery with test predictions.
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
        # A registration refused after the approval check leaves the approval unconsumed.
        self.approve("refresh_sets")
        self.refused("valset", self.mk("valset", ids=[60, 61, 20]), "evalset-0001")  # overlaps the evalset
        self.assertNotIn("used_by", self.ws.state["approvals"][2])
        self.assertNotIn("used_by", Workspace(self.wsdir).state["approvals"][2])
        self.assertTrue(self.ws.has_approval("refresh_sets"))
        self.assertEqual(self.ws.register("valset", self.mk("valset", ids=range(60, 64))), "valset-0004")
        self.assertEqual(self.ws.state["approvals"][2]["used_by"], "valset-0004")


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
        t = binary_task()  # task errors: one per line
        t["eval_use_limit"] = 0
        t["stop"]["max_rounds"] = 0
        r = run_rsi("validate", "task", make_task_dir(self.src("bad"), t))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(r.stderr.strip().splitlines()), 2, r.stderr)

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
        r = run_rsi("status", os.path.join(self.root, "nope"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("STATE.json", r.stderr)

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

    def test_a_data_module_candidate_is_refused(self):
        # 2026-11-25: the data module does not train; a candidate it produced is refused even on the current trainset.
        self.sets()
        self.refused("candidate", self.mk("candidate", ids=VAL, producer="data"), "producer data", "does not train")
        # An ordinary wrong trainset id keeps its plain message.
        msg = self.refused("candidate", self.mk("candidate", ids=VAL, trainset_id="trainset-0009"), "trainset_id")
        self.assertNotIn("producer data", msg)


class TestRefusedCompute(WsCase):
    """B1: the compute of a candidate refused at registration counts toward budget_used.compute_hours, once per
    source directory (R2): the same source at the same cost is the same training; another cost is a new training;
    the later registration of a counted source does not add the same cost again."""

    def used(self):
        return Workspace(self.wsdir).state["budget_used"]

    def cand(self, name, cost, **fields):
        fields.setdefault("ids", VAL)
        return self.mk("candidate", name, cost={"compute_hours": cost}, **fields)

    def redo(self, src, cost, ids=VAL, note="redone"):
        """The candidate rewritten in place (fixed or retrained): same directory, new content."""
        return make_artifact("candidate", src, self.task, ids=ids, cost={"compute_hours": cost}, notes=note)

    def test_a_refused_candidate_costs_compute_once(self):
        self.sets()
        a = self.cand("cand-a", 0.1, ids=TRAIN)  # predictions on the trainset rows, not the valset's: refused
        self.refused("candidate", a, "predictions_val")
        self.assertAlmostEqual(self.used()["compute_hours"], 0.1)
        self.assertEqual(self.used()["refused"], {os.path.realpath(a): 0.1})
        self.assertAlmostEqual(Workspace(self.wsdir).refused_compute_hours(), 0.1)
        self.refused("candidate", a, "predictions_val")  # handed back again as it was: the same training
        self.assertAlmostEqual(self.used()["compute_hours"], 0.1)
        self.redo(a, 0.1, note="fixed in place")  # fixed without retraining: same cost
        self.assertEqual(self.ws.register("candidate", a), "candidate-0001")
        self.assertAlmostEqual(self.used()["compute_hours"], 0.1)
        self.assertEqual(self.used()["refused"], {})
        self.assertEqual(Workspace(self.wsdir).refused_compute_hours(), 0)
        b = self.cand("cand-b", 0.1, ids=TRAIN)
        self.refused("candidate", b, "predictions_val")
        self.assertAlmostEqual(self.used()["compute_hours"], 0.2)
        self.redo(b, 0.2, note="retrained")  # a new training: its cost counts too, apart from cand-a
        self.assertEqual(self.ws.register("candidate", b), "candidate-0002")
        self.assertAlmostEqual(self.used()["compute_hours"], 0.4)  # cand-a 0.1, cand-b 0.1 + 0.2 = 0.3
        self.assertEqual(self.used()["refused"], {})
        # Refused again after a retraining at another cost: counted again, the latest cost kept per source.
        c = self.cand("cand-c", 0.1, ids=TRAIN)
        self.refused("candidate", c, "predictions_val")
        self.redo(c, 0.3, ids=TRAIN, note="retrained, still wrong")
        self.refused("candidate", c, "predictions_val")
        self.assertAlmostEqual(self.used()["compute_hours"], 0.8)
        self.assertEqual(self.used()["refused"], {os.path.realpath(c): 0.3})
        self.assertAlmostEqual(Workspace(self.wsdir).refused_compute_hours(), 0.3)

    def test_refusals_that_count_nothing(self):
        self.sets()
        broken = self.cand("broken", 0.1)
        with open(os.path.join(broken, "manifest.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.refused("candidate", broken)  # the manifest cannot be read
        for i, bad in enumerate(("two", float("nan"), float("inf"))):  # the cost is no number (M4: nor NaN, inf)
            self.refused("candidate", self.mk("candidate", "notnum-%d" % i, ids=VAL, cost={"compute_hours": bad}),
                         "compute_hours")
        self.refused("candidate", self.mk("trainset", "a-trainset", ids=range(40, 44)))  # not a candidate
        reg = self.cand("registered", 0.1)
        self.assertEqual(self.ws.register("candidate", reg), "candidate-0001")
        self.refused("candidate", reg, "already registered")  # its cost counted once, at its registration
        same = os.path.join(self.root, "src", "same-content")
        shutil.copytree(reg, same)
        self.refused("candidate", same, "same content")  # a copy of a registered candidate: no new training
        self.ws.stop("alice", "先停")
        late = self.cand("late", 0.1, ids=TRAIN)
        with self.assertRaises(StoppedError):  # refused because the run stopped: not counted
            self.ws.register("candidate", late)
        self.assertAlmostEqual(self.used()["compute_hours"], 0.1)
        self.assertEqual(self.used().get("refused") or {}, {})

    def test_a_counted_refusal_is_not_weighed_against_the_budget_again(self):
        # cand-a's 0.2 hours are in the hours used once it is refused; fixed in place, registering it adds nothing,
        # so it needs no over_budget approval (0.2 of 0.25), and a new candidate of 0.1 does (0.3 > 0.25).
        self.task["budget"]["compute_hours"] = 0.25
        self.sets()
        a = self.cand("cand-a", 0.2, ids=TRAIN)
        self.refused("candidate", a, "predictions_val")
        self.redo(a, 0.2)
        self.assertIsNone(Workspace(self.wsdir).budget_overrun([a]))
        self.assertEqual(self.ws.register("candidate", a), "candidate-0001")
        self.assertAlmostEqual(self.used()["compute_hours"], 0.2)
        b = self.cand("cand-b", 0.1)
        self.refused("candidate", b, "over_budget")
        # The budget refusal is a refusal too: cand-b's training counts (0.3), and its registration once approved
        # adds nothing more.
        self.assertAlmostEqual(self.used()["compute_hours"], 0.3)
        self.ws.approve("over_budget", "alice", "可以超")
        self.assertEqual(self.ws.register("candidate", b), "candidate-0002")
        self.assertAlmostEqual(self.used()["compute_hours"], 0.3)

    def test_budget_overrun(self):
        self.task["budget"]["compute_hours"] = 0.25
        self.sets()
        a, b = self.cand("cand-a", 0.2), self.cand("cand-b", 0.1)
        rec = self.mk("record", "rec", artifacts=["trainset-0001"])
        ws = Workspace(self.wsdir)
        self.assertIsNone(ws.budget_overrun([]))
        self.assertIsNone(ws.budget_overrun([a, rec]))  # 0.2 of 0.25
        over = ws.budget_overrun([rec, a, b])
        self.assertEqual((over["dirs"], over["used"], over["budget"]), ([a, b], 0, 0.25))
        self.assertAlmostEqual(over["cost"], 0.3)
        self.approve("over_budget")
        self.assertIsNone(Workspace(self.wsdir).budget_overrun([a, b]))  # the round has the approval
        self.assertEqual(self.used(), {"compute_hours": 0})  # it only reads
        # No overrun without a budget, or without over_budget among the approvals.
        for budget, approvals in ((None, None), (0.25, ["task_change", "use_testset"])):
            with self.subTest((budget, approvals)):
                shutil.rmtree(self.wsdir, ignore_errors=True)
                self.task = binary_task()
                self.task["budget"]["compute_hours"] = budget
                if approvals is not None:
                    self.task["approvals"] = approvals
                self.sets()
                self.assertIsNone(self.ws.budget_overrun([self.cand("big-%s" % budget, 5)]))

    def handed_back(self, out, *names):
        """The hand-back (handback.json) a module wrote into output directory `out`, listing `names`."""
        with open(os.path.join(self.wsdir, out, "handback.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "artifacts": list(names), "marker": None, "notes": "", "questions": []}, f)

    def test_a_stop_counts_the_candidates_of_a_held_handback(self):
        # R7, F1: a stop ends a hand-back not processed yet (held for over_budget, or handed back) unregistered: the
        # candidates it lists count in that save, keyed by source as for a refusal (R2). A dispatch the stop does not
        # close (its hand-back processed: done) counts nothing.
        self.sets()
        ws = Workspace(self.wsdir)
        outs = ["work/round-1/train-3", "work/round-1/train", "work/round-1/train-2"]
        for out in outs:
            os.makedirs(os.path.join(self.wsdir, out))
        tune, old, done = ws.add_dispatches([{"module": m, "outdir": o, "status": st, "launches": n} for m, o, st, n in
                                             (("train", outs[0], "handed_back", 1), ("train", outs[1], "handed_back", 2),
                                              ("train", outs[2], "done", 1))])
        trials = [make_artifact("candidate", os.path.join(self.wsdir, outs[0], t), self.task, ids=VAL, notes=t,
                                cost={"compute_hours": c}) for t, c in (("trial-1", 0.2), ("trial-2", 0.3))]
        for did, out in ((old, outs[1]), (done, outs[2])):
            make_artifact("candidate", os.path.join(self.wsdir, out, "candidate"), self.task, ids=VAL, notes=out)
        held = {"launch": 1, "question_id": "q-0001", "cost": 0.5, "used": 0, "budget": 0.25}
        ws.update_dispatch(tune, held_budget=dict(held, dirs=["trial-1", "trial-2"]))
        ws.update_dispatch(old, held_budget=dict(held, dirs=["candidate"]))  # held for launch 1; handed back again
        ws.update_dispatch(done, held_budget=dict(held, dirs=["candidate"]))  # processed after its approval
        for out, names in ((outs[0], ("trial-1", "trial-2")), (outs[1], ("candidate",)), (outs[2], ("candidate",))):
            self.handed_back(out, *names)
        Workspace(self.wsdir).stop("alice", "停")
        used = self.used()
        self.assertAlmostEqual(used["compute_hours"], 2.0)  # 0.2 + 0.3, and the fixture candidate's 1.5
        self.assertEqual(used["refused"], {trials[0]: 0.2, trials[1]: 0.3,
                                           os.path.join(self.wsdir, outs[1], "candidate"): 1.5})

    def test_a_stop_racing_the_registration_of_a_held_candidate_counts_it_once(self):
        # Review fix 1: `rsi stop` loaded its Workspace (and the registry) before the engine registered the approved
        # held candidate, and takes the STATE lock after that commit, while the dispatch is still handed_back. The
        # stop reads the registry again under the lock: the candidate is registered, so it is no held compute.
        self.task["budget"]["compute_hours"] = 0.25
        self.sets()
        ws = Workspace(self.wsdir)
        out = "work/round-1/train"
        os.makedirs(os.path.join(self.wsdir, out))
        (did,) = ws.add_dispatches([{"module": "train", "outdir": out, "status": "handed_back", "launches": 1}])
        src = make_artifact("candidate", os.path.join(self.wsdir, out, "candidate"), self.task, ids=VAL,
                            cost={"compute_hours": 0.35})
        ws.update_dispatch(did, held_budget={"launch": 1, "question_id": "q-0001", "dirs": ["candidate"],
                                             "cost": 0.35, "used": 0, "budget": 0.25})
        self.handed_back(out, "candidate")
        Workspace(self.wsdir).approve("over_budget", "alice", "可以超")
        stopper, stale = Workspace(self.wsdir), Workspace(self.wsdir)  # loaded before the engine's commit
        self.assertEqual(Workspace(self.wsdir).register("candidate", src), "candidate-0001")  # the engine's commit
        stale._count_refused(src)  # a refusal counted from a stale registry view: the source is registered now
        self.assertEqual(self.used(), {"compute_hours": 0.35})
        stopper.stop("alice", "停")
        self.assertEqual(self.used(), {"compute_hours": 0.35})
        self.assertEqual(Workspace(self.wsdir).state["stopped"]["reason"], "user requested: 停")

    def test_stop_check_if_closed(self):
        self.sets()
        stop, why = Workspace(self.wsdir).stop_check_if_closed()
        self.assertFalse(stop)
        self.assertIn("no-promotion streak 1 of patience 2", why)
        self.assertIn("1 of max_rounds 5 completed", why)
        self.ws.state["no_promotion_streak"] = 1
        self.ws.save_state()
        before = self.state_on_disk()
        stop, why = Workspace(self.wsdir).stop_check_if_closed()
        self.assertTrue(stop)
        self.assertEqual(why, "2 rounds in a row without promotion reached patience 2")
        self.assertEqual(self.state_on_disk(), before)  # it changes nothing
        self.assertFalse(Workspace(self.wsdir).stop_check()[0])


if __name__ == "__main__":
    unittest.main()
