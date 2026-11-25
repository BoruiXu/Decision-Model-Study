"""Code-built briefs (rsi/briefs.py): the inputs of every module and purpose follow the main-flow table,
paths stay inside what the module may read, and handback.json is checked without raising."""
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

import fixtures
import test_dashboard as TD
import test_skills as TS
from rsi import briefs as B
from rsi.workspace import RsiError, Workspace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_IDS, TRAIN_IDS, EVAL_IDS, TEST_IDS = TD.VAL_IDS, TD.TRAIN_IDS, TD.EVAL_IDS, TD.TEST_IDS
REFRESH_IDS = range(60, 64)  # the evalset of a refresh (BriefCase.refreshed)


def strings(x):
    """Every string in a nested structure (keys and values)."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            for s in strings(k):
                yield s
            for s in strings(v):
                yield s
    elif isinstance(x, (list, tuple)):
        for v in x:
            for s in strings(v):
                yield s


class BriefCase(TD.DashboardCase):
    def use_task(self, task, name="ws"):
        """Every task here names its raw data: data.sources [{"path": <tmp>/raw}] (never created). So its sets come
        from a corpus (sets())."""
        task["data"]["sources"] = [{"path": os.path.join(self.root, "raw")}]
        super(BriefCase, self).use_task(task, name)

    def corpus(self):
        """Register the corpus: the full row of every id the scenarios use, so every set registered later is from it."""
        ids = sorted(set(TEST_IDS) | set(EVAL_IDS) | set(VAL_IDS) | set(TRAIN_IDS) | set(REFRESH_IDS))
        return self.reg("corpus", rows=[fixtures.full_row(i, self.task) for i in ids])

    def sets(self):
        """The corpus (the task names raw data, so it comes first), then the four sets."""
        self.corpus()
        super(BriefCase, self).sets()

    def register_corpus_and_pool(self):
        """sets() (corpus-0001 first) and pool-0001 (the trainset's ids)."""
        self.sets()
        self.reg("pool", ids=TRAIN_IDS)

    def brief(self, module, **kw):
        return B.build_brief(Workspace(self.ws), module, **kw)

    def keys(self, module, **kw):
        return set(self.brief(module, **kw)["inputs"])

    def files(self, module, **kw):
        return sorted(os.path.basename(p) for p in self.brief(module, **kw)["inputs"].values() if os.path.isfile(p))

    def check_paths(self, brief):
        """Every path in the brief is the output directory, the repo root's command, or inside an artifact
        directory named by inputs; the workspace root never appears and no label or valset rows leak."""
        root = Workspace(self.ws).root
        for s in strings(brief):
            self.assertNotEqual(s.rstrip("/"), root, s)
            if s.startswith(root + os.sep):
                ok = s.startswith(brief["outdir"]) or any(
                    s == p or s.startswith(p + os.sep) for p in brief["inputs"].values())
                self.assertTrue(ok, "%s is not an input or the output directory" % s)
        for p in brief["inputs"].values():
            self.assertTrue(os.path.exists(p), p)
            self.assertNotIn(os.sep + "work" + os.sep, p)

    def milestone(self):
        self.target_task()
        self.promoted_round()

    def scan(self, module, **kw):
        """The rendered brief of `module`: it names no workspace path but its inputs and output directory."""
        b = self.brief(module, **kw)
        text = B.render_brief(b)
        root = Workspace(self.ws).root
        self.assertNotIn(root + "\n", text)
        allowed = list(b["inputs"].values()) + [b["outdir"]]
        for m in re.findall(r"(?<![\w<])/[\w./-]+", text):
            m = m.rstrip(".")
            if m.startswith(root + os.sep) or m == root:
                self.assertTrue(any(m == p or m.startswith(p + os.sep) for p in allowed), m)
        return text

    def target_task(self):
        task = fixtures.binary_task()
        task["stop"]["target"] = 0.8
        task["approvals"] = [x for x in task["approvals"] if x != "plan"]
        self.use_task(task, "ws-ms")

    def refreshed(self, data_issue=False):
        # a new workspace: its sets() registers a corpus first (BriefCase.sets), but no pool
        task = fixtures.binary_task()
        task["eval_use_limit"] = 1
        self.use_task(task, "ws-refresh")
        ev = self.promoted_round()
        if data_issue:
            self.reg("plan", 1, evaluation_id=ev, entry_module="data",
                     issues=[dict(fixtures._plan_issue(), issue="stale eval rows", attribution="data_coverage",
                                  module="data", change="new evalset")])
        else:
            self.plan(ev)
        Workspace(self.ws).next_round()
        self.candidate(2)
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh")
        self.reg("evalset", 2, ids=REFRESH_IDS)


class TestInputs(BriefCase):
    def test_first_round_data_gets_only_the_task(self):
        b = self.brief("data")
        self.assertEqual(b["purpose"], "sets")
        self.assertEqual(list(b["inputs"]), ["task"])
        self.assertEqual(b["round"], 1)
        self.assertEqual((b["task_id"], b["task_version"]), (self.task["task_id"], self.task["version"]))
        self.assertTrue(b["task_json"].endswith("task.json"))
        self.assertIn(self.task["decision"]["instructions"], b["goal"])
        self.assertEqual(b["rsi"], "PYTHONPATH=%s python3 -m rsi" % REPO)
        self.check_paths(b)

    def test_train_gets_only_trainset_and_valset_of_the_sets(self):
        self.sets()
        b = self.brief("train")
        self.assertEqual(set(b["inputs"]), {"task", "trainset", "valset"})
        self.assertNotIn("evalset", b["ids"])
        self.assertNotIn("testset", b["ids"])
        text = json.dumps(b)
        self.assertNotIn("evalset", text)
        self.assertNotIn("testset", text)
        self.assertIn("train_command", b)
        self.assertEqual(b["progress_file"], os.path.join(b["outdir"], "progress.json"))
        self.assertIn("progress.json", B.render_brief(b))
        self.check_paths(b)
        self.assertNotIn("train_command", self.brief("deliver"))

    def test_train_gets_its_search_inputs(self):
        # D9 (and the old tune brief): every registered candidate and search by id, the best version's candidate,
        # the plan that opened the round, the task's search_space, search, resources.onejev and resources.gpu_ids;
        # every brief names its dispatch.
        task = fixtures.binary_task()
        task["search_space"] = {"seed": [0, 1]}
        task["search"] = {"max_configs": 3}
        task["resources"] = {"onejev": {"python": "/x/python"}, "gpu_ids": [0, 2], "train_command": "fixture-trainer"}
        task["flow"] = {"checkin_minutes": 30}
        self.use_task(task, "ws-train-brief")
        ev = self.promoted_round()
        plan = self.plan(ev)
        Workspace(self.ws).next_round()
        trials, search = self.tune_trials(2)
        ws = Workspace(self.ws)
        best = ws.version("V1")["candidate_id"]
        b = self.brief("train")
        self.assertEqual((b["inputs"]["best_candidate"], b["inputs"]["plan"]), (ws.path(best), ws.path(plan)))
        for a in [best] + trials + [search]:
            self.assertEqual(b["inputs"][a], ws.path(a))
        self.assertEqual(set(b["inputs"]) - set(trials) - {search, best},
                         {"task", "trainset", "valset", "best_candidate", "plan"})
        self.assertEqual((b["search_space"], b["search"], b["resources"], b["checkin_minutes"], b["train_command"]),
                         (task["search_space"], task["search"], {"onejev": {"python": "/x/python"}, "gpu_ids": [0, 2]},
                          30, "fixture-trainer"))
        text = B.render_brief(b)
        for word in ('"seed": [0, 1]', '"gpu_ids": [0, 2]', "定时检查"):
            self.assertIn(word, text)
        # The train skill's 输入 names every input this brief gives (a registered one by its kind) and its fields.
        section = TS.skill_section(TS.module_path("train"), "输入")
        names = set(re.sub(r"-\d{4}$", "", k) for k in b["inputs"]) | {
            "search_space", "search", "checkin_minutes", "budget_left", "dispatch_id", "train_command"}
        self.assertEqual(names, {"task", "trainset", "valset", "best_candidate", "plan", "candidate", "search",
                                 "search_space", "checkin_minutes", "budget_left", "dispatch_id", "train_command"})
        for name in names:
            self.assertIn("`%s`" % name, section, name)
        for key in b["resources"]:
            self.assertIn("`resources.%s`" % key, section, key)
        self.check_paths(b)
        self.assertIsNone(b["dispatch_id"])
        did = ws.add_dispatches([{"module": "train", "purpose": "train", "outdir": "work/round-2/train"}])[0]
        self.assertEqual(B.build_brief(Workspace(self.ws), "train", did)["dispatch_id"], did)

    def test_train_command_from_task_or_default(self):
        self.sets()
        self.assertEqual(self.brief("train")["train_command"],
                         os.path.join(REPO, "modules", "train", "train"))
        task = fixtures.binary_task()
        task["resources"] = {"train_command": "/opt/x/train"}
        self.use_task(task, "ws-tc")
        self.sets()
        self.assertEqual(self.brief("train")["train_command"], "/opt/x/train")
        # The data module does not train: no purpose of its brief names the train command.
        for purpose in ("sets", "trainset"):
            self.assertNotIn("train_command", self.brief("data", purpose=purpose))

    def test_deliver_gets_only_input_files(self):
        self.sets()
        cand = self.candidate()
        b = self.brief("deliver")
        self.assertEqual(set(b["inputs"]), {"task", "candidate", "evalset_inputs", "valset_inputs"})
        self.assertEqual(self.files("deliver"), ["inputs.jsonl", "inputs.jsonl", "task.json"])
        ws = Workspace(self.ws)
        self.assertEqual(b["inputs"]["candidate"], ws.path(cand))
        self.assertEqual(b["inputs"]["evalset_inputs"],
                         os.path.join(ws.path(ws.current("evalset")), "inputs.jsonl"))
        self.assertEqual(b["inputs"]["valset_inputs"], os.path.join(ws.path(ws.current("valset")), "inputs.jsonl"))
        for p in b["inputs"].values():
            self.assertNotIn("labels", p)
            self.assertNotIn("rows.jsonl", p)
        self.assertNotIn("testset", json.dumps(b))
        self.check_paths(b)
        # The read-deny list: only the inputs files of the evalset and the valset are open.
        den = B.denied_paths(ws, b)
        art = lambda aid, *parts: os.path.join(ws.path(aid), *parts)
        self.assertIn(art("evalset-0001", "labels.jsonl"), den)
        self.assertNotIn(art("evalset-0001", "inputs.jsonl"), den)
        self.assertNotIn(art("evalset-0001"), den)
        self.assertNotIn(art("valset-0001", "inputs.jsonl"), den)
        self.assertIn(art("valset-0001", "rows.jsonl"), den)
        self.assertIn(art("testset-0001"), den)
        self.assertIn(art("trainset-0001"), den)

    def test_judge_gets_labels_and_trainset_rows(self):
        self.sets()
        cand = self.candidate()
        dlv = self.delivery(cand)
        b = self.brief("judge")
        ws = Workspace(self.ws)
        self.assertEqual(set(b["inputs"]), {"task", "candidate", "delivery", "evalset_inputs", "evalset_labels",
                                            "trainset_rows"})
        self.assertEqual(b["inputs"]["delivery"], ws.path(dlv))
        self.assertTrue(b["inputs"]["evalset_labels"].endswith("labels.jsonl"))
        self.assertTrue(b["inputs"]["trainset_rows"].endswith("rows.jsonl"))
        self.assertIn(ws.manifest(cand)["trainset_id"], b["inputs"]["trainset_rows"])
        self.assertIsNone(b["best"])
        self.assertNotIn("testset", json.dumps(b))
        self.check_paths(b)
        den = B.denied_paths(ws, b)  # the evalset with its labels is open, the testset is not
        self.assertIn(ws.path("testset-0001"), den)
        self.assertFalse([p for p in den if p.startswith(ws.path("evalset-0001"))])

    def test_judge_gets_best_version_after_promotion(self):
        self.two_promoted_rounds()
        cand2 = self.reg("candidate", 2, ids=VAL_IDS, train_ids=TRAIN_IDS, notes="round 2")
        dlv = self.reg("delivery", 2, ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand2, wrong=(20,))
        b = self.brief("judge")
        ws = Workspace(self.ws)
        best = ws.state["best"]
        v = ws.version(best)
        self.assertEqual(b["best"]["version"], best)
        self.assertEqual(b["best"]["evaluation_id"], v["evaluation_id"])
        self.assertEqual(b["best"]["delivery_id"], ws.manifest(v["evaluation_id"])["delivery_id"])
        self.assertEqual(b["inputs"]["best_evaluation"], ws.path(v["evaluation_id"]))
        self.assertEqual(b["inputs"]["best_delivery"], ws.path(b["best"]["delivery_id"]))
        self.assertEqual(b["inputs"]["delivery"], ws.path(dlv))
        self.assertNotIn("best_delivery_current_evalset", b["inputs"])
        self.check_paths(b)

    def test_deliver_and_judge_after_refresh(self):
        # deliver also gets the best version's candidate (re-delivered on the new evalset); the judge gets that
        # re-delivery beside the round's delivery.
        self.refreshed()
        ws = Workspace(self.ws)
        b = self.brief("deliver")
        self.assertEqual(b["inputs"]["candidate"], ws.path(ws.passed_candidates()[0]))
        self.assertEqual(b["inputs"]["best_candidate"], ws.path(ws.version("V1")["candidate_id"]))
        self.assertEqual(b["inputs"]["evalset_inputs"], os.path.join(ws.path("evalset-0002"), "inputs.jsonl"))
        self.assertNotIn("labels", json.dumps(b["inputs"]))
        self.check_paths(b)
        best_cand = ws.version("V1")["candidate_id"]
        cand2 = ws.passed_candidates()[0]
        self.reg("delivery", 2, ids=range(60, 64), val_ids=VAL_IDS, candidate_id=best_cand, evalset_id="evalset-0002")
        redelivery = Workspace(self.ws).best_delivery_on_current_evalset()
        dlv = self.reg("delivery", 2, ids=range(60, 64), val_ids=VAL_IDS, candidate_id=cand2, evalset_id="evalset-0002")
        b = self.brief("judge")
        ws = Workspace(self.ws)
        self.assertEqual(b["inputs"]["best_delivery_current_evalset"], ws.path(redelivery))
        self.assertEqual(b["best"]["delivery_on_current_evalset"], redelivery)
        self.assertEqual(b["inputs"]["delivery"], ws.path(dlv))
        self.assertEqual(b["inputs"]["candidate"], ws.path(cand2))
        self.check_paths(b)

    def test_later_round_briefs(self):
        # Round 2: data gets no evalset or testset; the goal is the plan's issues for the module; the rendered text
        # hides the workspace root and the held-out sets.
        self.two_promoted_rounds()
        b = self.brief("data")
        ws = Workspace(self.ws)
        self.assertEqual(b["purpose"], "trainset")
        self.assertEqual(set(b["inputs"]), {"task", "trainset", "valset", "plan"})
        self.assertEqual(b["inputs"]["plan"], ws.path(ws.state["rounds"]["1"]["plan_id"]))
        text = json.dumps(b)
        self.assertNotIn("evalset", text)
        self.assertNotIn("testset", text)
        self.check_paths(b)
        plan = ws.state["rounds"]["1"]["plan_id"]
        issue = ws.manifest(plan)["issues"][0]
        b = self.brief("train")
        self.assertIn(issue["issue"], b["goal"])
        self.assertIn(issue["change"], b["goal"])
        self.assertIn(plan, b["goal"])
        self.assertNotIn(issue["issue"], self.brief("data")["goal"])  # not assigned to data
        self.assertIn(self.task["decision"]["instructions"], self.brief("data")["goal"])
        for m in ("train", "deliver", "judge"):
            self.scan(m)
        for m in ("train", "data"):
            t = self.scan(m)
            self.assertNotIn("evalset", t)
            self.assertNotIn("testset", t)

    def test_data_briefs_by_purpose(self):
        raw = self.task["data"]["sources"][0]["path"]
        b = self.brief("data", purpose="sets")
        self.assertEqual(b["sources"], [raw]); self.assertTrue(b["dset"].endswith("python3 -m dset"))
        self.assertIn("原始数据", B.render_brief(b)); self.assertIn(raw, B.render_brief(b))
        self.register_corpus_and_pool()                     # sets() plus corpus-0001 and pool-0001
        self.assertLessEqual({"corpus", "pool"}, set(self.brief("data", purpose="sets")["inputs"]))   # a resumed sets
        tb = self.brief("data", purpose="trainset")
        self.assertIn("pool", tb["inputs"]); self.assertNotIn("corpus", tb["inputs"]); self.assertNotIn("sources", tb)
        self.assertNotIn("原始数据", B.render_brief(tb)); self.assertIn("切分工具：`%s`" % tb["dset"], B.render_brief(tb))
        self.refreshed()                                     # evalset-0002 registered after an approved refresh
        rb = self.brief("data", purpose="refresh")
        self.assertIn("corpus", rb["inputs"]); self.assertNotIn("pool", rb["inputs"])
        self.assertEqual(sorted(k for k in rb["inputs"] if k.startswith("evalset_inputs_")),
                         ["evalset_inputs_evalset-0001", "evalset_inputs_evalset-0002"])
        self.assertTrue(all(os.path.basename(rb["inputs"][k]) == "inputs.jsonl"
                            for k in rb["inputs"] if k.startswith("evalset_")))
        self.assertNotIn("dset", self.brief("train"))

    def test_data_refresh_gets_the_task_and_plan_issues(self):
        self.refreshed(data_issue=True)
        b = self.brief("data", purpose="refresh")
        self.assertEqual(set(b["inputs"]), {"task", "trainset", "valset", "plan", "corpus",
                                            "evalset_inputs_evalset-0001", "evalset_inputs_evalset-0002"})
        self.assertNotIn("evalset", b["inputs"])
        self.assertNotIn("labels", json.dumps(b))
        self.check_paths(b)
        self.assertIn("stale eval rows", b["goal"])

    def test_plan_after_close_round_gets_candidates_searches_records(self):
        self.sets()
        cand = self.reg("candidate", 1, ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=TD.FAIL)
        trials, search = self.tune_trials()
        rec = self.reg("record", 1, artifacts=[cand])
        Workspace(self.ws).close_round("nothing passed")
        b = self.brief("judge", purpose="plan")
        ws = Workspace(self.ws)
        self.assertEqual(set(b["inputs"]) - {"task"}, {cand, search, rec} | set(trials))
        self.assertNotIn("evalset_labels", b["inputs"])
        self.assertNotIn("labels", json.dumps(b["inputs"]))
        self.assertEqual(b["inputs"][cand], ws.path(cand))
        self.check_paths(b)

    def test_milestone_briefs(self):
        # deliver adds the testset's inputs (its labels denied); the judge gets the test labels.
        self.milestone()
        ws = Workspace(self.ws)
        b = self.brief("deliver", purpose="milestone_deliver")
        self.assertEqual(set(b["inputs"]), {"task", "candidate", "valset_inputs", "evalset_inputs", "testset_inputs"})
        self.assertEqual(b["inputs"]["candidate"], ws.path(ws.version("V1")["candidate_id"]))
        self.assertEqual(b["inputs"]["testset_inputs"], os.path.join(ws.path("testset-0001"), "inputs.jsonl"))
        self.assertNotIn("labels", json.dumps(b["inputs"]))
        self.check_paths(b)
        den = B.denied_paths(ws, b)
        self.assertIn(os.path.join(ws.path("testset-0001"), "labels.jsonl"), den)
        self.assertNotIn(os.path.join(ws.path("testset-0001"), "inputs.jsonl"), den)
        den = B.denied_paths(ws, B.build_brief(ws, "judge", purpose="milestone_judge"))
        self.assertFalse([p for p in den if p.startswith(ws.path("testset-0001"))])
        cand = ws.version("V1")["candidate_id"]
        d = fixtures.make_artifact("delivery", self.src("delivery"), self.task, ids=EVAL_IDS, val_ids=VAL_IDS,
                                   candidate_id=cand, round=1)
        fixtures.add_predictions(d, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        ws.approve("use_testset", "alice", "go")
        dlv = ws.register("delivery", d)
        b = self.brief("judge", purpose="milestone_judge")
        self.assertEqual(b["inputs"]["delivery"], ws.path(dlv))
        self.assertTrue(b["inputs"]["testset_labels"].endswith("labels.jsonl"))
        self.assertTrue(b["inputs"]["evalset_labels"].endswith("labels.jsonl"))
        self.check_paths(b)

    def test_dispatch_purpose_and_outdir(self):
        self.sets()
        ws = Workspace(self.ws)
        did = ws.add_dispatches([{"module": "judge", "outdir": "work/round-1/judge-2", "purpose": "plan"}])[0]
        b = B.build_brief(ws, "judge", did)
        self.assertEqual(b["purpose"], "plan")
        self.assertEqual(b["outdir"], os.path.join(ws.root, "work", "round-1", "judge-2"))
        self.assertNotEqual(b["outdir"], ws.root)
        self.assertEqual(self.brief("judge")["outdir"], os.path.join(ws.root, "work", "round-1", "judge"))
        with self.assertRaises(RsiError):
            B.build_brief(ws, "data", did)  # a judge dispatch
        with self.assertRaises(RsiError):
            self.brief("train", purpose="plan")

    def test_budget_attempts_and_ids(self):
        task = fixtures.binary_task()
        task["budget"]["compute_hours"] = 100
        self.use_task(task, "ws-budget")
        self.sets()
        self.search_over([self.candidate()])
        b = self.brief("train")
        self.assertEqual(b["attempts"], "1 / 3")
        used = Workspace(self.ws).state["budget_used"]["compute_hours"]
        self.assertEqual(b["budget_left"]["compute_hours"], 100 - used)
        self.assertEqual(b["ids"], {"trainset": "trainset-0001", "valset": "valset-0001"})
        self.assertEqual(self.brief("judge")["ids"]["evalset"], "evalset-0001")
        self.assertNotIn("testset", self.brief("judge")["ids"])


class TestOwnRecordsAndPlan(BriefCase):
    """P14: a module's brief gives its own records of earlier rounds (no other module's, none of this round); the
    judge's evaluation brief gives the plan that opened the round (the previous round's), to judge whether it
    worked. The read-deny list opens exactly these and still denies every other record and plan."""

    def next_round(self, plan, n):
        if "plan" in self.task["approvals"]:
            Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        self.assertEqual(Workspace(self.ws).next_round(), (n, "train"))

    def three_rounds(self):
        """Round 1: V1, a record of every module, plan-0001; round 2: keep, a train record, plan-0002; round 3 (the
        current round): a train record. Returns (records of round 1 by module, of round 2, of round 3, plans)."""
        if Workspace(self.ws).current("testset") is None:
            self.sets()
        ev1 = self.chain()
        Workspace(self.ws).promote(ev1)
        r1 = self.module_records(1)
        plan1 = self.plan(ev1)
        self.next_round(plan1, 2)
        c2 = self.candidate(2)
        ev2 = self.evaluation(c2, self.delivery(c2, 2), 2, best_before={"version": "V1", "evaluation_id": ev1})
        Workspace(self.ws).promote(ev2)
        r2 = self.module_records(2, ("train",))
        plan2 = self.plan(ev2, 2)
        self.next_round(plan2, 3)
        r3 = self.module_records(3, ("train",))
        return r1, r2, r3, (plan1, plan2)

    def test_each_module_gets_only_its_own_records_of_earlier_rounds(self):
        r1, r2, r3, _plans = self.three_rounds()
        ws = Workspace(self.ws)
        records = set(e["id"] for e in ws.entries("record"))
        own = {"data": {r1["data"]}, "train": {r1["train"], r1["tune"], r2["train"]},  # an old tune record: train's
               "deliver": {r1["deliver"]}}
        for module, want in own.items():
            b = self.brief(module)
            self.assertEqual(set(b["inputs"]) & records, want, module)
            self.assertNotIn(r3["train"], b["inputs"])  # this round's record is no earlier round's
            for rid in want:
                self.assertEqual(b["inputs"][rid], ws.path(rid))
                self.assertIn("实验记录", b["input_notes"][rid])
                self.assertIn(rid, B.render_brief(b))
            self.check_paths(b)
        self.assertLessEqual(records, set(self.brief("judge")["inputs"]))  # the judge reads every record, as before

    def test_the_judges_evaluation_brief_gets_the_plan_that_opened_the_round(self):
        self.sets()
        self.assertNotIn("plan", self.brief("judge")["inputs"])  # round 1: no plan before it
        _r1, _r2, _r3, (plan1, plan2) = self.three_rounds()
        ws = Workspace(self.ws)
        b = self.brief("judge")
        self.assertEqual(b["purpose"], "judge")
        self.assertEqual(b["inputs"]["plan"], ws.path(plan2))
        self.assertIn("计划", b["input_notes"]["plan"])
        self.assertNotIn(ws.path(plan1), b["inputs"].values())
        self.check_paths(b)
        for purpose in ("plan", "milestone_judge"):  # only the evaluation brief
            try:
                other = self.brief("judge", purpose=purpose)
            except RsiError:
                continue
            self.assertNotIn(ws.path(plan2), other["inputs"].values(), purpose)

    def test_the_deny_list_opens_exactly_the_given_records_and_plan(self):
        r1, r2, _r3, (plan1, plan2) = self.three_rounds()
        ws = Workspace(self.ws)
        records = [e["id"] for e in ws.entries("record")]
        given = {("data", "trainset"): {r1["data"], plan2},  # data already got the plan that opened the round
                 ("train", "train"): {r1["train"], r1["tune"], r2["train"], plan2},  # and the round's plan
                 ("deliver", "deliver"): {r1["deliver"]}, ("judge", "judge"): set(records) | {plan2},
                 ("judge", "plan"): set(records)}
        for (module, purpose), opened in given.items():
            den = B.denied_paths(ws, B.build_brief(ws, module, purpose=purpose))
            for aid in records + [plan1, plan2]:
                if aid in opened:
                    self.assertNotIn(self.art(aid), den, (module, purpose, aid))
                else:
                    self.assertIn(self.art(aid), den, (module, purpose, aid))
            if module in ("data", "train"):
                self.assertIn(self.art("evalset-0001"), den, (module, purpose))
            self.assertIn(self.art("testset-0001"), den, (module, purpose))

    def art(self, aid, *parts):
        return os.path.join(Workspace(self.ws).path(aid), *parts)


class TestDenied(BriefCase):
    def deny(self, module, **kw):
        ws = Workspace(self.ws)
        return B.denied_paths(ws, B.build_brief(ws, module, **kw))

    def art(self, aid, *parts):
        return os.path.join(Workspace(self.ws).path(aid), *parts)

    def test_train_is_denied_every_dataset_it_is_not_given(self):
        self.sets()
        self.reg("trainset", 1, ids=range(100, 110))  # trainset-0002 is now the current one
        den = self.deny("train")
        self.assertIn(self.art("evalset-0001"), den)
        self.assertIn(self.art("testset-0001"), den)
        self.assertIn(self.art("trainset-0001"), den)  # not the current training set
        self.assertNotIn(self.art("trainset-0002"), den)
        self.assertNotIn(self.art("valset-0001"), den)
        self.assertTrue(all(os.path.isabs(p) for p in den))

    def test_sources_corpus_and_pool_are_denied_outside_their_dispatches(self):
        raw = self.task["data"]["sources"][0]["path"]; self.register_corpus_and_pool()
        ws = Workspace(self.ws); corpus, pool = self.art("corpus-0001"), self.art("pool-0001")
        for (module, purpose), allowed in {("data", "sets"): {raw, corpus, pool}, ("data", "trainset"): {pool},
                                           ("data", "refresh"): {corpus}, ("train", "train"): set(),
                                           ("deliver", "deliver"): set(), ("judge", "judge"): set()}.items():
            den = B.denied_paths(ws, B.build_brief(ws, module, purpose=purpose))
            for p in (raw, corpus, pool):
                (self.assertNotIn if p in allowed else self.assertIn)(p, den, (module, purpose, p))
        self.assertTrue({raw, corpus, pool} <= set(B.decision_denied(ws)))
        rules = B.deny_rules(["/gone/raw"])                 # cannot tell whether it is a directory: both rules
        self.assertIn("Read(//gone/raw)", rules); self.assertIn("Read(//gone/raw/**)", rules)
        # A task version that moves the raw data: the sets brief gives the new path only; every task version's raw
        # data stays denied to the other dispatches and to the decision agent.
        moved = os.path.join(self.root, "raw2")
        task = dict(self.task, version=2, data=dict(self.task["data"], sources=[{"path": moved}]))
        ws.approve("task_change", "alice", "the raw data moved")
        ws.register("task", fixtures.make_task_dir(self.src("task"), task))
        ws = Workspace(self.ws)
        self.assertEqual(B.build_brief(ws, "data", purpose="sets")["sources"], [moved])
        self.assertLessEqual({raw, moved}, set(B.denied_paths(ws, B.build_brief(ws, "train"))))
        self.assertLessEqual({raw, moved}, set(B.decision_denied(ws)))

    def test_data_output_directories_are_denied_but_ones_own(self):
        # Final review A-2: a data dispatch's output directory may keep unregistered sealed sets (a refused hand-back,
        # a forgotten redo/): every other dispatch, and the decision agent, is denied it.
        self.sets()
        ws = Workspace(self.ws)
        first, second = ws.add_dispatches([{"module": "data", "purpose": "sets", "outdir": "work/round-1/data"},
                                           {"module": "data", "purpose": "sets", "outdir": "work/round-1/data-2"}])
        ws = Workspace(self.ws)
        mine, other = (os.path.join(self.ws, "work", "round-1", x) for x in ("data-2", "data"))
        den = B.denied_paths(ws, B.build_brief(ws, "data", dispatch_id=second))
        self.assertIn(other, den)
        self.assertNotIn(mine, den)
        for den in (self.deny("train"), B.decision_denied(ws)):
            self.assertLessEqual({mine, other}, set(den))

    def test_deny_rules(self):
        # The double-slash absolute form; spaces and commas stay inside one argument.
        d, f = "/a b/c", "/a/rows,1.jsonl"
        with mock.patch("os.path.isdir", side_effect=lambda p: p == d), \
                mock.patch("os.path.exists", side_effect=lambda p: p in (d, f)):
            self.assertEqual(B.deny_rules([d, f]), ["Read(//a b/c/**)", "Edit(//a b/c/**)",
                                                    "Read(//a/rows,1.jsonl)", "Edit(//a/rows,1.jsonl)"])
        # A path the rule parser would mangle is refused.
        for bad in ("/w/a(b/x", "/w/a)b", "/w/a*b", "/w/a?b", "/w/a[b", "!/w"):
            with self.assertRaises(RsiError, msg=bad):
                B.deny_rules([bad])

    def test_every_registered_artifact_the_brief_does_not_give_is_denied(self):
        self.sets()
        cand = self.candidate()
        dlv = self.delivery(cand)
        ev = self.evaluation(cand, dlv)
        ws = Workspace(self.ws)
        den = self.deny("train")
        for aid in (dlv, ev):
            self.assertIn(self.art(aid), den)
        self.assertNotIn(self.art(cand), den)  # train gets every registered candidate
        den = self.deny("deliver")
        self.assertIn(self.art("trainset-0001"), den)
        self.assertNotIn(os.path.dirname(self.art(ws.current("task"), "task.json")), den)

    def test_nothing_a_brief_gives_is_denied_for_any_module_and_purpose(self):
        self.milestone()
        ws = Workspace(self.ws)
        self.tune_trials()
        checked = 0
        for module, purposes in B.MODULE_PURPOSES.items():
            for purpose in purposes:
                try:
                    brief = B.build_brief(ws, module, purpose=purpose)
                except RsiError:
                    continue
                checked += 1
                den = [os.path.realpath(p) for p in B.denied_paths(ws, brief)]
                for name, path in brief["inputs"].items():
                    real = os.path.realpath(path)
                    for d in den:
                        self.assertFalse(real == d or real.startswith(d + os.sep) or d.startswith(real + os.sep)
                                         and os.path.isdir(real),
                                         "%s/%s: input %s is denied by %s" % (module, purpose, name, d))
        self.assertGreaterEqual(checked, 8)

    def test_a_symlinked_workspace_root_gets_rules_for_both_spellings(self):
        self.sets()
        link = os.path.join(self.root, "link")
        os.symlink(self.ws, link)
        ws = Workspace(link)
        den = B.denied_paths(ws, B.build_brief(ws, "train"))
        self.assertIn(os.path.join(link, "artifacts", "evalset-0001"), den)
        rules = B.deny_rules(den)
        self.assertIn("Read(//%s/**)" % os.path.join(link, "artifacts", "evalset-0001").lstrip("/"), rules)
        self.assertIn("Read(//%s/**)" % os.path.join(self.ws, "artifacts", "evalset-0001").lstrip("/"), rules)

    def test_decision_agent_denied_list(self):
        self.sets()
        cand = self.candidate()
        dlv = self.delivery(cand)
        ev = self.evaluation(cand, dlv)
        ws = Workspace(self.ws)
        den = B.decision_denied(ws)
        for aid in ("trainset-0001", "valset-0001", "evalset-0001", "testset-0001", dlv, ev):
            self.assertIn(self.art(aid), den)
        self.assertNotIn(self.art(cand), den)

    def test_the_list_is_not_in_the_brief(self):
        self.sets()
        self.delivery(self.candidate())
        b = self.brief("train")
        text = B.render_brief(b) + json.dumps(b)
        self.assertTrue(self.deny("train"))
        for p in self.deny("train"):
            self.assertNotIn(p, text)
        self.assertNotIn("denied", b)


class TestFixes(BriefCase):
    def test_plan_judge_skips_milestone_artifacts(self):
        self.target_task()
        self.promoted_round()
        ws = Workspace(self.ws)
        cand = ws.version("V1")["candidate_id"]
        d = fixtures.make_artifact("delivery", self.src("delivery"), self.task, ids=EVAL_IDS, val_ids=VAL_IDS,
                                   candidate_id=cand, round=1)
        fixtures.add_predictions(d, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        ws.approve("use_testset", "alice", "go")
        mdlv = ws.register("delivery", d)
        ws.approve("use_testset", "alice", "go again")
        ev = self.reg("evaluation", 1, ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=mdlv, candidate_id=cand,
                      evalset_id="evalset-0001", uses_testset=True, test_ids=TEST_IDS,
                      best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
                      recommendation={"decision": "keep", "reason": "milestone release"})
        ws = Workspace(self.ws)
        self.assertTrue(ws.manifest(ev)["uses_testset"])
        b = self.brief("judge", purpose="plan")
        self.assertNotIn(mdlv, b["inputs"])
        self.assertNotIn(ev, b["inputs"])
        self.assertTrue(any(k.startswith("evaluation-") for k in b["inputs"]))
        self.assertNotIn("test", " ".join(os.path.basename(p) for p in b["inputs"].values()))

    def test_missing_sets_are_clear_errors(self):
        with self.assertRaises(RsiError) as c:
            self.brief("train")
        self.assertIn("no current trainset registered", str(c.exception))
        with self.assertRaises(RsiError) as c:
            self.brief("deliver")
        self.assertIn("no current evalset registered", str(c.exception))
        self.sets_without_test = None
        self.corpus()
        self.reg("evalset", ids=EVAL_IDS)
        self.reg("valset", ids=VAL_IDS)
        self.reg("trainset", ids=TRAIN_IDS)
        self.reg("candidate", 1, ids=VAL_IDS, train_ids=TRAIN_IDS)
        with self.assertRaises(RsiError) as c:
            self.brief("deliver", purpose="milestone_deliver")
        self.assertIn("no testset registered", str(c.exception))



class TestRender(BriefCase):
    def test_render_follows_the_template(self):
        self.sets()
        cand = self.candidate()
        self.delivery(cand)
        text = B.render_brief(self.brief("judge"))
        for part in ("rsi-judge", "task.json", "PYTHONPATH=%s python3 -m rsi validate" % REPO,
                     "python3 -m rsi score", "不要运行 `rsi register`", "handback.json", "输出目录", "当前最佳版本"):
            self.assertIn(part, text)
        self.assertNotIn(Workspace(self.ws).root + "\n", text)
        self.assertNotIn("训练命令", text)
        self.assertIn("训练命令：`%s/modules/train/train`" % REPO, B.render_brief(self.brief("train")))
        # R18: the train module's trials go through trsi launch (its skill); the data module does not train.
        self.assertIn("trsi launch", B.render_brief(self.brief("train")))
        data = B.render_brief(self.brief("data"))
        self.assertNotIn("训练命令", data)
        self.assertIn("本模块不训练", data)

    def test_cli_brief(self):
        self.sets()
        env = dict(os.environ, PYTHONPATH=REPO)
        out = subprocess.run([sys.executable, "-m", "rsi", "brief", self.ws, "train"], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout, B.render_brief(self.brief("train")))
        bad = subprocess.run([sys.executable, "-m", "rsi", "brief", self.ws, "train", "--purpose", "plan"], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("does not belong", bad.stderr)


class TestHandback(BriefCase):
    def write(self, obj, raw=None):
        d = os.path.join(self.root, "out")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "handback.json"), "w", encoding="utf-8") as f:
            f.write(raw if raw is not None else json.dumps(obj))
        return d

    def test_read_handback(self):
        r = B.read_handback(self.write({"status": "ok", "artifacts": ["trainset-a", "record"], "marker": None,
                                        "notes": "done", "questions": ["more budget?"]}))
        self.assertEqual((r["ok"], r["errors"], r["status"]), (True, [], "ok"))
        self.assertEqual((r["artifacts"], r["marker"], r["notes"], r["questions"]),
                         (["trainset-a", "record"], None, "done", ["more budget?"]))
        r = B.read_handback(self.write({"status": "long_job", "artifacts": [], "marker": "/x/done", "notes": "",
                                        "questions": []}))
        self.assertTrue(r["ok"])
        self.assertEqual(r["marker"], "/x/done")
        self.assertTrue(B.read_handback(self.write({"status": "failed", "notes": "no GPU"}))["ok"])  # defaults
        cases = [
            ({"status": "long_job", "marker": None}, "marker"),
            ({"status": "long_job", "marker": "rel/done"}, "marker"),
            ({"status": "ok", "marker": "rel"}, "marker"),
            ({"status": "failed"}, "notes"),
            ({"status": "failed", "notes": "  "}, "notes"),
            ({"status": "maybe"}, "status"),
            ({}, "status"),
            ({"status": "ok", "artifacts": "a"}, "artifacts"),
            ({"status": "ok", "artifacts": ["../x"]}, "plain subdirectory"),
            ({"status": "ok", "artifacts": ["a/b"]}, "plain subdirectory"),
            ({"status": "ok", "artifacts": ["a", "a"]}, "twice"),
            ({"status": "ok", "notes": 3}, "notes"),
            ({"status": "ok", "questions": [1]}, "questions"),
            ({"status": "ok", "questions": [""]}, "questions"),
            ({"status": "ok", "marker": 5}, "marker"),
            ({"status": "ok", "extra": 1}, "unknown field"),
            ([1], "object"),
        ]
        for obj, word in cases:
            r = B.read_handback(self.write(obj))
            self.assertFalse(r["ok"], obj)
            self.assertTrue(any(word in e for e in r["errors"]), (obj, r["errors"]))
        for raw in ("", "{not json", "null", "[" * 200000):  # the last one: hostile nesting
            r = B.read_handback(self.write(None, raw=raw))
            self.assertFalse(r["ok"])
            self.assertTrue(r["errors"])
        for d in (os.path.join(self.root, "nothing"), self.root):  # no handback.json
            r = B.read_handback(d)
            self.assertFalse(r["ok"])
            self.assertIn("handback.json", r["errors"][0])


# What each module produces (the flow registers these from its hand-back; `record` comes from every module and
# train's search compares its trial candidates), checked against the producer the fixtures give each kind.
EXPECTED_OUTPUTS = {"data": {"task", "corpus", "testset", "evalset", "valset", "pool", "trainset", "record"},
                    "train": {"candidate", "search", "record"}, "deliver": {"delivery", "feedback", "record"},
                    "judge": {"evaluation", "plan", "record"}}
# Keys of brief.json by module: the optional ones are named in the field table of templates/brief.md as well.
ALWAYS_KEYS = ("module", "skill", "purpose", "round", "dispatch_id", "task_id", "task_version", "task_json", "ids",
               "inputs", "input_notes", "outdir", "progress_file", "rsi", "budget_left", "attempts", "goal", "handback")


class TestSkillsMatchBriefs(BriefCase):
    """The module skills and templates/brief.md say what rsi/briefs.py really builds."""

    def collect(self):
        """{(module, purpose): brief} over scenarios that reach every purpose of every module."""
        out = {}
        self.reach = {}

        def grab(*pairs):
            for module, purpose in pairs:
                out[(module, purpose)] = self.brief(module, purpose=purpose)
                self.reach.setdefault(module, set()).update(self.kinds_of(out[(module, purpose)]))

        self.use_task(fixtures.binary_task(), "ws-sets")
        self.register_corpus_and_pool()  # a resumed sets gets the corpus and the pool
        grab(("data", "sets"))
        self.use_task(fixtures.binary_task(), "ws-rounds")
        self.two_promoted_rounds(records=True)  # round 2's briefs give each module its own round-1 record
        cand = self.reg("candidate", 2, ids=VAL_IDS, train_ids=TRAIN_IDS, notes="round 2 again")
        self.tune_trials(2)
        self.reg("record", 2, artifacts=[cand])
        self.reg("feedback", 2)
        self.delivery(cand, 2)
        grab(("data", "trainset"), ("train", "train"), ("deliver", "deliver"), ("judge", "judge"), ("judge", "plan"))
        self.refreshed(data_issue=True)
        grab(("data", "refresh"), ("deliver", "deliver"))
        self.milestone()
        ws = Workspace(self.ws)
        cand1 = ws.version("V1")["candidate_id"]
        d = fixtures.make_artifact("delivery", self.src("delivery"), self.task, ids=EVAL_IDS, val_ids=VAL_IDS,
                                   candidate_id=cand1, round=1)
        fixtures.add_predictions(d, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        ws.approve("use_testset", "alice", "go")
        ws.register("delivery", d)
        grab(("deliver", "milestone_deliver"), ("judge", "milestone_judge"))
        return out

    def kinds_of(self, brief):
        ws = Workspace(self.ws)
        dirs = [(os.path.realpath(ws.path(e["id"])), e["kind"]) for e in ws.entries()]
        kinds = set()
        for p in brief["inputs"].values():
            p = os.path.realpath(p)
            found = [k for d, k in dirs if p == d or p.startswith(d + os.sep)]
            self.assertTrue(found, "%s is in no registered artifact" % p)
            kinds.add(found[0])
        return kinds

    def test_briefs_of_every_purpose_match_the_template_and_the_skills(self):
        briefs = self.collect()
        # Every purpose of every module is reached.
        want = set((m, p) for m, ps in B.MODULE_PURPOSES.items() for p in ps)
        self.assertEqual(set(briefs), want)
        # Labels only go to the judge (not for its plan).
        for (module, purpose), brief in briefs.items():
            labels = [p for p in brief["inputs"].values() if os.path.basename(p) == "labels.jsonl"]
            if module == "judge" and purpose != "plan":
                self.assertTrue(labels, (module, purpose))
            else:
                self.assertEqual(labels, [], (module, purpose))
        # The input kinds are the ones each skill's 输入 section names.
        for module in B.MODULE_PURPOSES:
            reach = self.reach[module]
            skill = set(t for t in TS.lower_tokens(TS.skill_section(TS.module_path(module), "输入"))
                        if t in TS.KINDS)
            self.assertEqual(reach, skill, "%s: brief inputs vs the skill's 输入 section" % module)
        # brief.json keys are in templates/brief.md's field table and the skills.
        template = TS.read(TS.BRIEF)
        table = [l for l in template.splitlines() if l.startswith("| `")]
        documented = set(re.findall(r"`(\w+)`", "".join(l.split("|")[1] for l in table)))
        real = set()
        for brief in briefs.values():
            real |= set(brief)
        self.assertEqual(real - documented, set(), "brief.json keys missing from templates/brief.md's field table")
        self.assertEqual(documented - real, set(), "templates/brief.md documents keys the briefs do not have")
        for key in ALWAYS_KEYS:
            self.assertIn(key, real)
        listed = {"module", "purpose", "round", "task_json", "ids", "inputs", "outdir", "budget_left", "goal"}
        for name in TS.MODULE_SKILLS:
            text = TS.skill_section(TS.module_path(name), "输入")
            named = set(re.findall(r"`(\w+)`", text.split("只读下面列出")[0])) & (real | listed)
            self.assertTrue(listed <= named, name)
            self.assertLessEqual(listed, real)

    def test_output_kinds_match_the_skills(self):
        for module, kinds in EXPECTED_OUTPUTS.items():
            skill = set(t for t in TS.lower_tokens(TS.skill_section(TS.module_path(module), "输出"))
                        if t in TS.KINDS)
            self.assertEqual(skill, kinds, module)
        for kinds in EXPECTED_OUTPUTS.values():
            self.assertLessEqual(kinds, set(TS.KINDS))
        from rsi import engine
        for kind in set().union(*EXPECTED_OUTPUTS.values()):
            self.assertIn(kind, engine.KIND_ORDER)
        for kind, producer in (("candidate", "train"), ("search", "train"), ("delivery", "deliver"),
                               ("evaluation", "judge"), ("trainset", "data"), ("plan", "judge")):
            self.assertIn(kind, EXPECTED_OUTPUTS[producer])

    def handback_examples(self):
        """The handback.json example of the format text, and of each skill (its 输出 paragraph)."""
        yield "briefs.HANDBACK_FORMAT", B.HANDBACK_FORMAT
        for name in TS.MODULE_SKILLS + (TS.TEMPLATE_NAME,):
            yield name, TS.skill_section(TS.module_path(name), "输出")

    def test_handback_format_texts_and_example_pass_read_handback(self):
        for where, text in self.handback_examples():
            for key in B.HANDBACK_KEYS:
                self.assertIn("`%s`" % key if where != "briefs.HANDBACK_FORMAT" else '"%s"' % key, text, where)
            for status in B.HANDBACK_STATUSES:
                self.assertIn(status, text, where)
        d = os.path.join(self.root, "hb")
        os.makedirs(d)

        def read(obj):
            with open(os.path.join(d, "handback.json"), "w", encoding="utf-8") as f:
                json.dump(obj, f)
            return B.read_handback(d)

        # an example built from the keys the texts name (the refusals are in TestHandback.test_read_handback)
        self.assertTrue(read({"status": "ok", "artifacts": ["a"], "marker": None, "notes": "n", "questions": []})["ok"])
        self.assertTrue(read({"status": "long_job", "artifacts": [], "marker": "/abs/done", "notes": "",
                              "questions": []})["ok"])


if __name__ == "__main__":
    unittest.main()
