"""dset, the data module's split tool (modules/data/dset): sets, grow, check, refresh and repair, run in-process."""
import contextlib
import copy
import filecmp
import io
import itertools
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modules", "data"))

from dset import quality as Q  # noqa: E402
from dset.cli import main  # noqa: E402
from fixtures import (TIME, binary_task, corpus_rows, full_row, make_artifact, make_task_dir, time_task,  # noqa: E402
                      write_jsonl)
from rsi import datasplit as DS  # noqa: E402
from rsi import rows as R  # noqa: E402
from rsi.artifacts import validate_artifact  # noqa: E402
from rsi.variants import apply_spec  # noqa: E402
from rsi.workspace import RsiError, Workspace  # noqa: E402

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def ids(rows):
    return set(r["id"] for r in rows)


def step(r):
    """The time step of corpus row r<step>_<j>."""
    return int(r["id"][1:].split("_")[0])


def sets_args(base, out, *extra):
    """`dset sets` on <base>/task/task.json and <base>/corpus (relative to the temporary directory)."""
    return ("sets", "--task", os.path.join(base, "task", "task.json"), "--corpus", os.path.join(base, "corpus"),
            "--script", "preprocess.py", "--command", "python3 preprocess.py", "--label-method", "given",
            "--split", "split.json", "--round", "1", "--out", out) + extra


class TestDset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.task = time_task()
        self.write_case(self.tmp, self.task, corpus_rows(self.task))
        self.write_json("split.json", TIME)

    def p(self, *parts):
        return os.path.join(self.tmp, *parts)

    def write_json(self, name, obj):
        with open(self.p(name), "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return name

    @staticmethod
    def write_case(base, task, rows):
        make_task_dir(os.path.join(base, "task"), task)
        os.makedirs(os.path.join(base, "corpus"))
        write_jsonl(os.path.join(base, "corpus", "rows.jsonl"), rows)
        with open(os.path.join(base, "corpus", "preprocess.py"), "w", encoding="utf-8") as f:
            f.write("# writes rows.jsonl from the raw data\n")

    def fresh(self, name, edit_task=None, edit_rows=None, pre=()):
        """Another task and corpus (no manifest) under <name>/, each with one edit, and <name>/out/ holding `pre`."""
        task, rows = time_task(), corpus_rows(time_task())
        for edit, obj in ((edit_task, task), (edit_rows, rows)):
            if edit:
                edit(obj)
        self.write_case(self.p(name), task, rows)
        os.makedirs(self.p(name, "out"))
        for d in pre:
            os.makedirs(self.p(name, "out", d))
        return types.SimpleNamespace(out=self.p(name, "out"), corpus=self.p(name, "corpus"),
                                     sets_args=lambda *extra: sets_args(name, os.path.join(name, "out"), *extra))

    def dset(self, *args):
        """(exit code, the stdout summary or None, stderr) of dset run in the temporary directory."""
        out, err, cwd = io.StringIO(), io.StringIO(), os.getcwd()
        os.chdir(self.tmp)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(list(args))
        finally:
            os.chdir(cwd)
        return code, (json.loads(out.getvalue()) if out.getvalue().strip() else None), err.getvalue()

    def sets(self, out, *extra):
        return self.dset(*sets_args("", out, *extra))

    def test_sets_fills_the_rule_and_writes_valid_disjoint_reproducible_sets(self):
        code, summary, _ = self.sets("out")
        self.assertEqual(code, 0)
        new = load(self.p("out/task/task.json"))
        rule = new["data"]["split"]
        self.assertEqual((new["version"], rule["sizes"]["trainset"], rule["unit"], rule["seed"]), (2, 6, "cluster", 0))
        self.assertEqual(DS.fill_errors(self.task, new), [])
        self.assertIn(b"## Split rule (`data.split`)", read(self.p("out/task/TASK.md")))
        self.assertEqual(read(self.p("out/task/TASK.md")), DS.fill_task_md(read(self.p("task/TASK.md")), rule))
        dirs = dict((k, self.p("out", k)) for k in ("testset", "evalset", "valset", "pool", "trainset"))
        dirs["corpus"] = self.p("corpus")
        for kind, d in dirs.items():
            self.assertEqual(validate_artifact(kind, d, new), [], kind)
            self.assertEqual(load(os.path.join(d, "manifest.json"))["task_version"], 2, kind)
        rows = dict((k, R.read_jsonl(os.path.join(d, "inputs.jsonl" if k in ("evalset", "testset") else "rows.jsonl")))
                    for k, d in dirs.items() if k != "corpus")
        self.assertEqual([len(rows[k]) for k in ("trainset", "valset", "evalset", "testset")], [6, 3, 3, 4])
        for k, (lo, hi) in rule["ranges"].items():
            self.assertTrue(all(lo <= step(r) <= hi for r in rows[k]), k)
        self.assertTrue(all(1 <= step(r) <= 6 for r in rows["pool"]))
        self.assertLessEqual(ids(rows["trainset"]), ids(rows["pool"]))
        for a, b in itertools.combinations(("testset", "evalset", "valset", "pool"), 2):
            self.assertEqual(R.overlap(rows[a], rows[b]), {"ids": 0, "inputs": 0, "clusters": 0}, (a, b))
        report = load(self.p("out/trainset/quality_report.json"))["split"]
        self.assertEqual(report["shared_state_ids"]["testset/trainset"],
                         R.shared_state_ids(rows["testset"], rows["trainset"]))
        self.assertGreater(report["shared_state_ids"]["testset/pool"], 0)              # reported, not refused
        self.assertNotIn("label_counts", report["sets"]["testset"])
        self.assertIn("label_counts", report["sets"]["valset"])
        self.assertEqual(summary["shared_state_ids"], report["shared_state_ids"])
        self.assertIn("rows 4", load(os.path.join(dirs["testset"], "manifest.json"))["notes"])
        for kind in ("evalset", "testset"):  # final review A-3: manifest.json is no sealed file, its notes no label counts
            self.assertNotIn("labels", load(os.path.join(dirs[kind], "manifest.json"))["notes"], kind)
        self.assertIn("labels", load(os.path.join(dirs["valset"], "manifest.json"))["notes"])
        # a corpus with a manifest is only read: a second run writes the same sets and leaves corpus/ as it is
        before = dict((n, read(self.p("corpus", n))) for n in os.listdir(self.p("corpus")))
        self.assertEqual(self.sets("again")[0], 0)
        self.assertEqual(dict((n, read(self.p("corpus", n))) for n in os.listdir(self.p("corpus"))), before)
        for kind in ("testset", "evalset", "valset", "pool", "trainset"):
            for name in os.listdir(dirs[kind]):
                if name != "manifest.json":                                            # created_at differs
                    self.assertTrue(filecmp.cmp(os.path.join(dirs[kind], name), self.p("again", kind, name),
                                                shallow=False), (kind, name))
        self.assertEqual(self.sets("dry", "--dry-run")[0], 0)
        self.assertFalse(os.path.exists(self.p("dry")))
        code, s2, _ = self.sets("by-address", "--split", self.write_json("s2.json", dict(TIME, unit="state_id")))
        self.assertEqual((code, set(s2["shared_state_ids"].values())), (0, {0}))
        # the whole hand-back registers in KIND_ORDER without any approval
        ws = Workspace.init(self.p("ws"), self.p("task"))
        for kind, d in [("task", self.p("out/task"))] + [(k, dirs[k]) for k in ("corpus", "testset", "evalset",
                                                                              "valset", "pool", "trainset")]:
            ws.register(kind, d)
        self.assertEqual((ws.current("trainset"), ws.state["approvals"], ws.state["task_version"]),
                         ("trainset-0001", [], 2))
        with self.assertRaisesRegex(RsiError, "holds no rows; an evalset needs"):       # an empty set never registers
            ws.register("evalset", make_artifact("evalset", self.p("empty"), new, rows=[]))
        corpora = [make_artifact("corpus", self.p("c-empty"), new, rows=[]),
                   make_artifact("corpus", self.p("c-left"), new, rows=corpus_rows(new))]
        open(os.path.join(corpora[1], ".dset-manifest.json.tmp"), "w").close()         # a stopped dset run's leftover
        for d, frag in zip(corpora, ("a corpus needs at least one", ".dset-manifest.json.tmp")):
            self.assertTrue(any(frag in e for e in validate_artifact("corpus", d, new)), d)

    def test_sets_errors_write_nothing(self):
        def min_pos(t):
            t["quality_checks"]["min_positive_rows"] = 99

        def big_test(t):
            t["data"]["split"]["sizes"]["testset"] = 5                                  # step 12 holds 4 rows

        def no_step(rs):
            rs[5]["meta"]["extra"].pop("step")

        def text_step(rs):
            rs[5]["meta"]["extra"]["step"] = "3"

        def all_eval(t):
            del t["data"]["split"]["sizes"]["evalset"]                                # null: the whole range
        short = dict(TIME, ranges=dict(TIME["ranges"], testset=[12, 12], evalset=[9, 11]))
        none = dict(TIME, ranges=dict(TIME["ranges"], evalset=[13, 14]))             # no rows in steps 13-14
        no_val = dict(TIME, ranges=dict(TIME["ranges"], valset=[13, 14]))
        cases = [  # name, task edit, corpus edit, extra args, pre-made dirs, exit code, fragment
            ("gate", min_pos, None, [], [], 2, "min_positive_rows"),
            ("conflict", None, None, ["--split", self.write_json("c.json", dict(TIME, sizes={"trainset": 7}))], [], 1,
             "data.split.sizes.trainset"),
            ("no-step", None, no_step, [], [], 1, "meta.extra.step"),
            ("text-step", None, text_step, [], [], 1, "number"),
            ("short", big_test, None, ["--split", self.write_json("s.json", short)], [], 1, "testset"),
            ("empty", all_eval, None, ["--split", self.write_json("e.json", none)], [], 1, "no empty set"),
            ("empty-val", None, None, ["--split", self.write_json("v.json", no_val)], [], 1, "valset: its range"),
            ("exists", None, None, [], ["trainset"], 1, "exists"),
            ("left-over", None, None, [], [".dset-x1"], 1, ".dset-x1")]                  # a stopped run's staging
        for name, edit_task, edit_rows, extra, pre, want, frag in cases:
            case = self.fresh(name, edit_task, edit_rows, pre)        # its own task/, corpus/ (no manifest) and out/
            code, summary, err = self.dset(*case.sets_args(*extra))
            self.assertEqual(code, want, (name, err))
            self.assertIn(frag, err if want == 1 else json.dumps(summary["gate"]), name)
            self.assertEqual(sorted(os.listdir(case.out)), pre, name)
            self.assertFalse(os.path.exists(os.path.join(case.corpus, "manifest.json")), name)

    def test_grow_and_check(self):
        self.sets("out")
        t2 = "out/task/task.json"
        old, pool = R.read_jsonl(self.p("out/trainset/rows.jsonl")), R.read_jsonl(self.p("out/pool/rows.jsonl"))
        code, s, _ = self.dset("grow", "--task", t2, "--pool", "out/pool", "--trainset", "out/trainset",
                               "--parent", "trainset-0001", "--add", "2", "--round", "2", "--out", "g")
        new = R.read_jsonl(self.p("g/rows.jsonl"))
        self.assertEqual((code, len(new)), (0, 8))
        self.assertLessEqual(ids(old), ids(new))
        self.assertLessEqual(ids(new), ids(pool))
        self.assertEqual(load(self.p("g/manifest.json"))["parent_trainset"], "trainset-0001")
        self.assertEqual(validate_artifact("trainset", self.p("g"), load(self.p(t2))), [])
        self.assertEqual(self.dset("grow", "--task", t2, "--pool", "out/pool", "--trainset", "out/trainset", "--parent",
                                   "trainset-0001", "--add", "999", "--round", "2", "--out", "g2")[0], 1)
        rows = corpus_rows(self.task)[:8]
        broken = copy.deepcopy(rows[3])
        del broken["target"]
        for name, again, want in (("repeated id", rows[3], 0), ("repeated id, broken", broken, 1)):
            write_jsonl(self.p("rep.jsonl"), rows + [again])                             # row 8 repeats row 3's id
            code, s, err = self.dset("check", "--task", "task/task.json", "--rows", "rep.jsonl")
            self.assertEqual((code, s and s["duplicate_ids"]), (want, 1 if want == 0 else None), (name, err))
            self.assertEqual("row 8 (id=%s): target" % rows[3]["id"] in err, want == 1, name)
        twin = dict(copy.deepcopy(rows[1]), id="twin")
        twin["target"] = dict((k, 1.0 - v) for k, v in twin["target"].items())           # same input, other label
        write_jsonl(self.p("dirty.jsonl"), rows + [twin])
        code, s, _ = self.dset("check", "--task", "task/task.json", "--rows", "dirty.jsonl", "--write", "clean",
                               "--mode", "merge", "--round", "1")
        self.assertEqual((code, s["duplicate_inputs"], s["conflicting_inputs"]), (0, 1, 1))
        merged = [r for r in R.read_jsonl(self.p("clean/rows.jsonl")) if r["id"] == rows[1]["id"]]
        self.assertEqual((len(merged), merged[0]["target"], merged[0]["meta"]["n_obs"]),
                         (1, {"yes": 0.5, "no": 0.5}, 2))

    def test_grow_takes_several_with_files(self):
        self.sets("out")
        t2 = "out/task/task.json"
        shas, n = ("a1b2c3d4e5f6", "0f9e8d7c6b5a"), 0
        for sha in shas:                                    # two arms' verified rows, two rows each, new ids and inputs
            os.makedirs(self.p("arms", sha))
            rows = [full_row(100 + n + j, self.task) for j in range(2)]
            for r in rows:
                r["meta"]["extra"] = {"step": 3}
            n += 2
            write_jsonl(self.p("arms", sha, "verified.jsonl"), rows)
        code, s, err = self.dset("grow", "--task", t2, "--pool", "out/pool", "--trainset", "out/trainset", "--parent",
                                 "trainset-0001", "--with", "arms/%s/verified.jsonl" % shas[0], "--with",
                                 "arms/%s/verified.jsonl" % shas[1], "--round", "2", "--out", "g")
        self.assertEqual((code, s and s["with"]), (0, 4), err)
        self.assertEqual(len(R.read_jsonl(self.p("g/rows.jsonl"))), 6 + 4)
        changes = load(self.p("g/manifest.json"))["changes"]
        for sha in shas:
            self.assertIn("arms/%s" % sha, changes)

    def test_prompt_variants_follow_their_seeds(self):
        self.sets("out")
        t2 = "out/task/task.json"
        task = load(self.p(t2))
        spec = {"id": "p1", "instructions": "Is this message unwanted bulk mail?", "criteria": None, "rename": None,
                "tables": None}
        self.write_json("p1.json", [spec])
        rows = R.read_jsonl(self.p("out/trainset/rows.jsonl"))
        rows.insert(1, apply_spec(rows[0], spec, task))                    # the same state as its seed
        write_jsonl(self.p("v.jsonl"), rows)
        code, s, err = self.dset("check", "--task", t2, "--rows", "v.jsonl")
        self.assertEqual(code, 1)
        self.assertIn("variant", err)                                       # without its spec
        code, s, err = self.dset("check", "--task", t2, "--rows", "v.jsonl", "--variants", "p1.json", "--write", "ts",
                                 "--mode", "drop", "--round", "2", "--parent", "trainset-0001")
        self.assertEqual((code, s and s["duplicate_inputs"]), (0, 0), err)  # a variant is no duplicate of its seed
        self.assertEqual([r["id"] for r in R.read_jsonl(self.p("ts/rows.jsonl"))][:2], [rows[0]["id"], rows[1]["id"]])
        self.assertEqual(load(self.p("ts/manifest.json"))["prompt_variants"], [spec])
        self.assertEqual(validate_artifact("trainset", self.p("ts"), task), [])
        code, s, err = self.dset("grow", "--task", t2, "--pool", "out/pool", "--trainset", "ts", "--parent",
                                 "trainset-0002", "--add", "2", "--round", "3", "--out", "g")
        self.assertEqual(code, 0, err)                                      # the next round keeps them
        self.assertEqual(load(self.p("g/manifest.json"))["prompt_variants"], [spec])
        self.assertEqual(validate_artifact("trainset", self.p("g"), task), [])

    def test_check_reports_sources_and_synthetic_share(self):
        task = binary_task()
        task["quality_checks"] = {"max_synthetic_share": 0.1}
        self.write_json("st.json", task)
        rows = [full_row(i, task) for i in range(10)]                     # yes on even i; source "synthetic", rule
        for r in rows[0], rows[2]:
            r["meta"].update(source="dgen:smote", label_source="human")
        write_jsonl(self.p("ten.jsonl"), rows)
        code, s, err = self.dset("check", "--task", "st.json", "--rows", "ten.jsonl", "--by-source")
        self.assertEqual((code, s and s["synthetic_share"]), (0, 0.2), err)
        self.assertEqual(s["by_source"], {"dgen|human|yes": 2, "synthetic|rule|no": 5, "synthetic|rule|yes": 3})
        self.assertEqual(Q.quality_checks(rows, task),
                         [{"name": "max_synthetic_share", "value": 0.2, "threshold": 0.1, "passed": False}])

    def test_repair_applies_task_constraints(self):
        task = binary_task()
        task["input"]["format"] = "object"
        task["features"] = {"constraints": [{"nonneg": ["t[].v"]}]}
        self.write_json("ct.json", task)
        rows = [full_row(i, task, state={"t": [{"v": v}]}) for i, v in enumerate((-1, 2))]
        write_jsonl(self.p("r.jsonl"), rows)
        code, s, err = self.dset("repair", "--task", "ct.json", "--rows", "r.jsonl", "--out", "fixed.jsonl")
        self.assertEqual((code, s and s["repaired"], s and s["violations_left"]), (0, 1, 0), err)
        self.assertEqual(s["by_rule"], {"nonneg t[].v": {"broken": 1, "left": 0}})
        self.assertEqual([r["state"]["t"][0]["v"] for r in R.read_jsonl(self.p("fixed.jsonl"))], [0, 2])
        rows[1]["state"] = {"t": [{"w": 2}]}                               # a leaf repair cannot make: exit 1
        write_jsonl(self.p("r2.jsonl"), rows)
        code, s, err = self.dset("repair", "--task", "ct.json", "--rows", "r2.jsonl", "--out", "left.jsonl")
        self.assertEqual((code, s["violations_left"], [x["id"] for x in s["left"]]), (1, 1, [rows[1]["id"]]))
        self.assertIn("still break", err)
        self.assertEqual(len(R.read_jsonl(self.p("left.jsonl"))), 2)      # written all the same, for the record

    def test_refresh_draws_unused_eval_rows(self):
        self.sets("out")
        args = ["refresh", "--task", "out/task/task.json", "--corpus", "corpus", "--round", "1"]
        code, s, _ = self.dset(*args, "--exclude", "out/evalset/inputs.jsonl", "--out", "e2")
        e2 = R.read_jsonl(self.p("e2/inputs.jsonl"))
        self.assertEqual((code, len(e2)), (0, 3))
        self.assertNotIn("labels", load(self.p("e2/manifest.json"))["notes"])
        self.assertTrue(all(9 <= step(r) <= 10 for r in e2))
        for f in ("out/evalset/inputs.jsonl", "out/testset/inputs.jsonl", "out/valset/rows.jsonl",
                  "out/pool/rows.jsonl"):
            self.assertEqual(R.overlap(e2, R.read_jsonl(self.p(f))), {"ids": 0, "inputs": 0, "clusters": 0}, f)
        code, _, err = self.dset(*args, "--exclude", "out/evalset/inputs.jsonl", "--exclude", "e2/inputs.jsonl",
                                 "--out", "e3")
        self.assertEqual(code, 1)
        self.assertIn("evalset", err)                                                    # 8 rows in steps 9-10: 2 left


if __name__ == "__main__":
    unittest.main()
