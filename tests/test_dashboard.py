"""The progress pages (rsi/dashboard.py, rsi/dashboard_pages.py): module output directories, what the
run waits for, the workflow diagram, the overview and the three secondary pages."""
import builtins
import datetime
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import fixtures
from rsi import dashboard as D
from rsi.workspace import Workspace

# Row ids (q<i>) of the data sets; they never overlap.
TEST_IDS, EVAL_IDS, VAL_IDS, TRAIN_IDS = range(10, 14), range(20, 24), range(0, 4), range(40, 44)
ALLOWED = {"STATE.json", "registry.jsonl", "manifest.json", "task.json", "REPORT.md", "notes.md", "run.jsonl",
           "run.lock",  # run.lock: the pid of the `rsi run` that works now (P9: "● 运行中" only then)
           "gpus.json", "progress.json", "handback.json"}  # the live panels (handback-<n>.json too: HANDBACK_COPY)
HANDBACK_COPY = re.compile(r"^handback-\d+(-\d+)?\.json$")
OVERVIEW, ROUNDS, EVENTS, REPORT = ("dashboard.html", "dashboard/rounds.html", "dashboard/events.html",
                                    "dashboard/report.html")
TABS = ("概览", "轮次与产物", "事件记录", "运行报告")
FAIL = {"passed": False, "detail": "below the majority baseline"}


def pages(ws, now=None):
    """All four pages of workspace `ws` (a path or a Workspace): relative path -> HTML."""
    from rsi.dashboard_pages import render_pages
    return render_pages(Workspace(ws) if isinstance(ws, str) else ws, now)


def section(page, sec):
    """The HTML of <section id="sec"> (sections are not nested)."""
    start = page.index('id="%s"' % sec)
    return page[start:page.index("</section>", start)]


def round_block(page, n):
    """The HTML of round `n`'s <details> on the rounds page, up to the next round's block."""
    start = page.index('<details class="round" id="round-%d"' % n)
    nxt = page.find('<details class="round"', start + 1)
    return page[start:] if nxt < 0 else page[start:nxt]


def decision_row(block):
    return re.search(r'<div class="body decision">(.*?)</div>', block, re.S).group(1)


def utc(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class DashboardCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.n = 0
        self.use_task(fixtures.binary_task())

    def use_task(self, task, name="ws"):
        """A new workspace for `task` at <tmp>/<name>; later helpers register into it."""
        self.task = task
        self.ws = os.path.join(self.root, name)
        Workspace.init(self.ws, fixtures.make_task_dir(self.src("task"), task))

    def plan_task(self, name, **stop):
        """A new workspace whose task needs plan approval (and `stop` settings)."""
        task = fixtures.binary_task()
        task["approvals"] = task["approvals"] + ["plan"]
        task["stop"].update(stop)
        self.use_task(task, name)

    def src(self, name):
        """A fresh source directory outside the workspace."""
        self.n += 1
        return os.path.join(self.root, "src", "%02d-%s" % (self.n, name))

    def reg(self, kind, rnd=1, **fields):
        d = fixtures.make_artifact(kind, self.src(kind), self.task, round=rnd, **fields)
        return Workspace(self.ws).register(kind, d)

    def sets(self):
        self.reg("testset", ids=TEST_IDS)
        self.reg("evalset", ids=EVAL_IDS)
        self.reg("valset", ids=VAL_IDS)
        self.reg("trainset", ids=TRAIN_IDS)

    def candidate(self, rnd=1):
        return self.reg("candidate", rnd, ids=VAL_IDS, train_ids=TRAIN_IDS)

    def delivery(self, cand, rnd=1):
        return self.reg("delivery", rnd, ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand)

    def evaluation(self, cand, dlv, rnd=1, best_before=None):
        return self.reg("evaluation", rnd, ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=dlv,
                        candidate_id=cand, evalset_id="evalset-0001", best_before=best_before)

    def chain(self):
        """Round 1's candidate, delivery and evaluation; return the evaluation id."""
        cand = self.candidate()
        return self.evaluation(cand, self.delivery(cand))

    def promoted_round(self):
        self.sets()
        ev = self.chain()
        self.assertEqual(Workspace(self.ws).promote(ev)[:2], ("promote", "V1"))
        return ev

    def plan(self, ev, rnd=1):
        return self.reg("plan", rnd, evaluation_id=ev, entry_module="train",
                        issues=[dict(fixtures._plan_issue(), attribution="model_capacity", module="train")])

    def tune_trials(self, rnd=1, n=3):
        """`n` tuning trial candidates and their search in round `rnd`; return (trial ids, search id)."""
        trials = [self.reg("candidate", rnd, ids=VAL_IDS, train_ids=TRAIN_IDS, producer="tune",
                           config_source="tune", notes="trial %d" % i) for i in range(n)]
        ws = Workspace(self.ws)
        primary = self.task["metrics"]["primary"]
        rows = [{"candidate_id": c, "config": {"lr": 0.001 * (i + 1)}, "compute_hours": 1, "status": "ok",
                 "metric_val": ws.manifest(c)["metrics_val"][primary]} for i, c in enumerate(trials)]
        search = self.reg("search", rnd, trials=rows, default_candidate=trials[0], best_candidate=trials[1],
                          budget={"compute_hours": max(10, n)},
                          gain_vs_default={"metric": primary,
                                           "delta": rows[1]["metric_val"] - rows[0]["metric_val"]})
        return trials, search

    def search_over(self, cands, rnd=1):
        """A search of the training-tuning module (producer train) over the registered candidates `cands` (the first
        the default; the best by the validation primary metric); return its id."""
        ws = Workspace(self.ws)
        primary = self.task["metrics"]["primary"]
        rows = [{"candidate_id": c, "config": {"lr": 0.001 * (i + 1)}, "compute_hours": 1, "status": "ok",
                 "metric_val": ws.manifest(c)["metrics_val"][primary]} for i, c in enumerate(cands)]
        best = max(rows, key=lambda r: r["metric_val"])
        return self.reg("search", rnd, producer="train", trials=rows, default_candidate=cands[0],
                        best_candidate=best["candidate_id"], budget={"compute_hours": max(10, len(cands))},
                        gain_vs_default={"metric": primary, "delta": best["metric_val"] - rows[0]["metric_val"]})

    def failed_search(self, rnd=1):
        """A search whose one trial failed the train gate: an attempt with no candidate through the gate."""
        cand = self.reg("candidate", rnd, ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        return self.search_over([cand], rnd)

    def module_records(self, rnd=1, modules=("data", "train", "tune", "deliver", "judge")):
        """One record of each of `modules` in round `rnd` (the current round); return module -> record id."""
        return dict((m, self.reg("record", rnd, module=m, producer=m, artifacts=["trainset-0001"],
                                 notes="%s record of round %d" % (m, rnd))) for m in modules)

    def two_promoted_rounds(self, records=False):
        """Round 1 (one evalset row wrong, tuning trials and their search) promotes V1, round 2 promotes V2.
        records: each module also registers a record in round 1 (module_records)."""
        self.sets()
        c1 = self.candidate()
        self.tune_trials()
        d1 = self.reg("delivery", 1, ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=c1, wrong=(20,))
        ev1 = self.reg("evaluation", 1, ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=d1, candidate_id=c1,
                       evalset_id="evalset-0001", best_before=None, wrong=(20,))
        self.assertEqual(Workspace(self.ws).promote(ev1)[:2], ("promote", "V1"))
        if records:
            self.module_records()
        plan = self.plan(ev1)
        if "plan" in self.task["approvals"]:
            Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        self.assertEqual(Workspace(self.ws).next_round(), (2, "train"))
        c2 = self.candidate(2)
        d2 = self.delivery(c2, 2)
        ev2 = self.reg("evaluation", 2, ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=d2, candidate_id=c2,
                       evalset_id="evalset-0001", best_before={"version": "V1", "evaluation_id": ev1},
                       best_wrong=(20,))
        self.assertEqual(Workspace(self.ws).promote(ev2)[:2], ("promote", "V2"))

    def refresh_task(self, name):
        """Round 1 promoted V1 and used up the evalset (eval_use_limit 1); round 2 has a candidate. Returns the
        evaluation that used it up."""
        task = fixtures.binary_task()
        task["eval_use_limit"] = 1
        task["approvals"] = task["approvals"] + ["plan"]
        self.use_task(task, name)
        ev = self.promoted_round()
        plan = self.plan(ev)
        Workspace(self.ws).approve("plan", "alice", "go", plan)
        Workspace(self.ws).next_round()
        self.candidate(2)
        return ev

    def waiting(self):
        return D.waiting_for(Workspace(self.ws), D.module_progress(Workspace(self.ws)))

    def overview(self):
        return pages(self.ws)[OVERVIEW]

    def now_parts(self):
        """(lead, detail) of the now card."""
        now = section(self.overview(), "now")
        return (re.search(r'<div class="wait-now">(.*?)</div>', now).group(1),
                re.search(r'<div class="detail">(.*?)</div>', now, re.S).group(1))

    def assertFlow(self, **expect):
        """flow_states and the overview's diagram both give each named node its expected state."""
        ws = Workspace(self.ws)
        states = D.flow_states(ws, D.module_progress(ws))
        page = pages(ws)[OVERVIEW]
        for key, state in expect.items():
            self.assertEqual(states[key]["state"], state, (key, states[key]))
            self.assertIn('id="n-%s" data-state="%s"' % (key, state), page)
        return states


class TestModuleProgress(DashboardCase):
    def test_module_progress(self):
        out = os.path.join(self.ws, "work", "round-1", "data")
        os.makedirs(out)
        p = D.module_progress(Workspace(self.ws))
        self.assertEqual(p["data"]["state"], "running")
        self.assertIsNotNone(p["data"]["since"])
        datetime.datetime.fromisoformat(p["data"]["since"])  # ISO format
        self.assertEqual(p["train"]["state"], "idle")
        src = fixtures.make_artifact("testset", os.path.join(out, "testset"), self.task, ids=TEST_IDS, round=1)
        self.assertEqual(D.module_progress(Workspace(self.ws))["data"],
                         {"state": "unregistered", "since": None, "pending": ["testset"], "superseded": {}})
        Workspace(self.ws).register("testset", src)  # hand-over moves it away
        self.assertEqual(D.module_progress(Workspace(self.ws))["data"]["state"], "done")
        w = D.waiting_for(Workspace(self.ws), D.module_progress(Workspace(self.ws)))
        self.assertIn("rsi-data", w)  # round 1 still lacks evalset/valset/trainset

    def test_module_progress_counts_artifact_directories(self):
        # Output directories: train-2 counts as train's, trainer/ and train.log do not.
        r1 = os.path.join(self.ws, "work", "round-1")
        os.makedirs(os.path.join(r1, "trainer"))
        open(os.path.join(r1, "train.log"), "w").close()
        os.makedirs(os.path.join(r1, "train-2"))
        p = D.module_progress(Workspace(self.ws))
        self.assertEqual(p["train"]["state"], "running")
        # Only a subdirectory holding a manifest.json is handed-back output: drafts (rsi-data's ab/),
        # files and hidden directories are not, so a module that has handed back nothing is running.
        out = os.path.join(self.ws, "work", "round-1", "data")
        os.makedirs(os.path.join(out, "ab", "candidate-a"))
        os.makedirs(os.path.join(out, ".testset.rsi-handed-over-0000"))
        open(os.path.join(out, "report.md"), "w").close()
        self.assertEqual(D.module_progress(Workspace(self.ws))["data"]["state"], "running")
        fixtures.make_artifact("valset", os.path.join(out, "valset"), self.task, ids=VAL_IDS, round=1)
        self.assertEqual(D.module_progress(Workspace(self.ws))["data"],
                         {"state": "unregistered", "since": None, "pending": ["valset"], "superseded": {}})
        Workspace(self.ws).register("valset", os.path.join(out, "valset"))  # copied: the source stays
        self.assertEqual(D.module_progress(Workspace(self.ws))["data"]["state"], "done")
        # A second dispatch of the same module in the round, not handed back yet.
        os.makedirs(os.path.join(self.ws, "work", "round-1", "data_2"))
        self.assertEqual(D.module_progress(Workspace(self.ws))["data"]["state"], "running")

    def test_a_dispatchs_unregistered_hand_back_is_what_its_handback_lists(self):
        # Final review B-5: with dispatch records, 交回未登记 lists the subdirectories the hand-back names (the task
        # fill task/ too), not drafts such as a resumed sets' redo/; the task fill is no artifact of ① 数据's
        # row on the rounds page (it shows under 任务接入).
        ws = Workspace(self.ws)
        out = os.path.join(self.ws, "work", "round-1", "data")
        ws.add_dispatches([{"module": "data", "purpose": "sets", "outdir": "work/round-1/data",
                            "status": "handed_back"}])
        fixtures.make_artifact("valset", os.path.join(out, "valset"), self.task, ids=VAL_IDS, round=1)
        fixtures.make_artifact("valset", os.path.join(out, "redo"), self.task, ids=VAL_IDS, round=1)
        os.makedirs(os.path.join(out, "task"))
        with open(os.path.join(out, "task", "task.json"), "w") as f:
            json.dump(self.task, f)
        with open(os.path.join(out, "handback.json"), "w") as f:
            json.dump({"status": "ok", "artifacts": ["task", "valset"], "marker": None, "notes": "x",
                       "questions": []}, f)
        self.assertEqual(D.module_progress(Workspace(self.ws))["data"]["pending"], ["task", "valset"])
        fill = {"id": "task-0002", "kind": "task", "producer": "data"}
        self.assertEqual(D._step_artifacts(ws, "data", [fill]), [])
        from rsi.dashboard_pages import KIND_NAMES  # B-4: every kind has its name (the corpus and the pool too)
        from rsi.workspace import KINDS
        self.assertEqual(set(KINDS) - set(KIND_NAMES), set())


class TestWaitingFor(DashboardCase):
    def test_waiting_for_follows_the_round(self):
        task = fixtures.binary_task()
        task["approvals"] = task["approvals"] + ["plan"]
        self.use_task(task, "ws-plan")
        self.assertIn("rsi-data", self.waiting())
        self.sets()
        w = self.waiting()
        self.assertIn("rsi-train", w)
        self.assertNotIn("rsi-tune", w)  # one module trains and tunes
        cand = self.candidate()
        w = self.waiting()
        self.assertIn("rsi-deliver", w)
        self.assertIn("（本轮通过训练门的候选：%s）" % cand, w)
        dlv = self.delivery(cand)
        self.assertIn("rsi-judge", self.waiting())
        ev = self.evaluation(cand, dlv)
        self.assertIn("rsi promote", self.waiting())
        Workspace(self.ws).promote(ev)
        w = self.waiting()
        self.assertIn("rsi-judge", w)
        self.assertIn("计划", w)
        plan = self.plan(ev)
        w = self.waiting()
        self.assertIn("批准计划", w)
        self.assertIn(plan, w)
        Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        self.assertIn("rsi next", self.waiting())
        Workspace(self.ws).stop("alice", "enough for now")
        w = self.waiting()
        self.assertTrue(w.startswith("已停止："), w)
        self.assertIn("enough for now", w)

    def test_waiting_for_module_output(self):
        self.sets()
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        w = self.waiting()
        self.assertIn("rsi-train", w)
        self.assertIn("交回", w)
        fixtures.make_artifact("candidate", os.path.join(self.ws, "work", "round-1", "train", "cand"), self.task,
                               ids=VAL_IDS, train_ids=TRAIN_IDS)
        w = self.waiting()
        self.assertIn("rsi-train", w)
        self.assertIn("登记", w)
        self.assertIn("cand", w)

    def test_waiting_for_deliver_names_few_candidates(self):
        # Twelve tuning trials and a train candidate pass the train gate: the sentence gives the count
        # and at most three ids, the train candidate first; the rounds page still lists them all.
        self.sets()
        trials, _search = self.tune_trials(n=12)
        cand = self.candidate()
        w = self.waiting()
        self.assertIn("rsi-deliver", w)
        self.assertIn("本轮通过训练门的候选 13 个：%s、%s、%s 等" % (cand, trials[0], trials[1]), w)
        self.assertIn("按验证集主指标选最好的送评", w)
        self.assertEqual(len(re.findall(r"candidate-\d{4}", w)), 3, w)
        p = pages(self.ws)
        self.assertEqual(len(re.findall(r"candidate-\d{4}", section(p[OVERVIEW], "now"))), 3)
        for c in trials + [cand]:
            self.assertIn(c, p[ROUNDS])


class TestFlow(DashboardCase):
    def test_four_stage_nodes(self):
        st = self.assertFlow(task="done", data="current", train="todo", deliver="todo", judge="todo", check="todo",
                             stop="todo")
        self.assertEqual(set(st), {"task", "data", "train", "deliver", "judge", "check", "stop"})
        page = self.overview()
        for name in ("① 数据", "② 训练调优", "③ 交付", "④ 评测与决策"):
            self.assertIn(name, page)
        for stale in ("③ 调优", "④ 交付", "⑤", "rsi-tune", "② 训练 / ③ 调优"):
            self.assertNotIn(stale, page)

    def test_flow_states_follow_the_round(self):
        self.plan_task("ws-flow")
        self.sets()
        self.assertFlow(data="done", train="current")
        cand = self.candidate()
        st = self.assertFlow(data="done", train="done", deliver="current")
        self.assertEqual(st["train"]["text"], "✓ 训练门通过 · 1 个试验")
        out = os.path.join(self.ws, "work", "round-1", "deliver")
        os.makedirs(out)  # rsi-deliver dispatched, nothing handed back yet
        st = self.assertFlow(deliver="current", judge="todo")
        self.assertIn("运行中", st["deliver"]["text"])
        src = fixtures.make_artifact("delivery", os.path.join(out, "delivery"), self.task, round=1, ids=EVAL_IDS,
                                     val_ids=VAL_IDS, candidate_id=cand)
        dlv = Workspace(self.ws).register("delivery", src)
        self.assertFlow(deliver="done", judge="current")
        ev = self.evaluation(cand, dlv)
        st = self.assertFlow(judge="done", check="current")
        self.assertIn("rsi promote", st["check"]["text"])
        Workspace(self.ws).promote(ev)
        self.assertFlow(judge="current", check="done")  # the judge writes the plan
        plan = self.plan(ev)
        self.assertFlow(judge="done", check="wait")
        Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        st = self.assertFlow(check="current")
        self.assertIn("rsi next", st["check"]["text"])
        Workspace(self.ws).stop("alice", "enough for now")
        st = self.assertFlow(task="done", data="done", train="done", deliver="done", judge="done", check="done",
                             stop="current")
        self.assertRegex(st["stop"]["text"], r"\d\d:\d\d")
        self.assertIn("停止", st["stop"]["text"])

    def test_flow_failed_gate(self):
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        st = self.assertFlow(train="fail")  # attempts left: ② again
        self.assertEqual(st["train"]["text"], "门没过 · 可重派")
        self.assertIn("回到 rsi-train 再搜一次", self.waiting())
        r1 = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("训练门没过", r1)
        self.assertIn('class="bad"', r1)
        self.assertNotIn("次尝试已用完", r1)
        # Any candidate of ② through the gate (here an old workspace's tune trial) makes ② done.
        self.reg("candidate", ids=VAL_IDS, train_ids=TRAIN_IDS, producer="tune", config_source="tune", notes="trial")
        self.assertEqual(self.assertFlow(train="done")["train"]["text"], "✓ 训练门通过 · 2 个试验")

    def test_flow_attempts_used_up(self):
        # Nothing to deliver; ④ is still to come (the judge writes the closed round's plan). The rounds page and
        # the wait say the attempts are used up, in one wording; no question is asked.
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-flow-close")
        self.sets()
        self.failed_search()
        st = self.assertFlow(train="fail", deliver="skip", judge="todo", check="current")
        self.assertEqual(st["train"]["text"], "门没过")
        ws = Workspace(self.ws)
        r1 = round_block(pages(ws)[ROUNDS], 1)
        self.assertIn('<p class="warn">', r1)
        self.assertIn("rsi close-round", r1)
        self.assertIn("次尝试已用完", self.waiting())
        self.assertIn("rsi close-round", self.waiting())
        self.assertEqual([], D.pending_questions(ws, D.module_progress(ws)))
        ws.close_round("no candidate passed")
        self.assertFlow(deliver="skip", judge="current", check="done")
        self.assertNotIn('class="warn"', round_block(pages(self.ws)[ROUNDS], 1))
        self.assertNotIn("rsi close-round", self.waiting())

    def test_decided_round_leaves_no_step_in_progress(self):
        # A search under way (train's trial candidates registered, their search still owed), then the round is
        # closed: ② no longer shows as under way.
        self.sets()
        out = "work/round-1/train"
        os.makedirs(os.path.join(self.ws, out))
        Workspace(self.ws).add_dispatches([{"module": "train", "purpose": "train", "outdir": out, "status": "done",
                                            "session": "s-1"}])
        src = fixtures.make_artifact("candidate", os.path.join(self.ws, out, "trial-1"), self.task, ids=VAL_IDS,
                                     wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        Workspace(self.ws).register("candidate", src)
        st = self.assertFlow(train="current")
        self.assertEqual(st["train"]["text"], "● 进行中")
        Workspace(self.ws).close_round("no candidate passed")
        st = self.assertFlow(train="todo", check="done")
        self.assertEqual(st["train"]["text"], "本轮已判定，未完成")

    def test_flow_entry_after_data(self):
        # Round 2 enters at train: data is not passed through, even with a refreshed evalset.
        task = fixtures.binary_task()
        task["approvals"] = task["approvals"] + ["refresh_sets"]
        self.use_task(task, "ws-flow-entry")
        self.plan(self.promoted_round())
        self.assertEqual(Workspace(self.ws).next_round(), (2, "train"))
        st = self.assertFlow(data="skip", train="current")
        self.assertEqual(st["data"]["text"], "本轮不经过")
        Workspace(self.ws).approve("refresh_sets", "alice", "new evalset")
        self.reg("evalset", 2, ids=range(30, 34))
        self.assertFlow(data="skip")

    def test_stopped_run_shows_no_running_module(self):
        self.sets()
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        st = self.assertFlow(train="current")
        self.assertIn("运行中", st["train"]["text"])
        Workspace(self.ws).stop("alice", "stop now")
        st = self.assertFlow(train="skip", stop="current")
        self.assertIn("停止时未交回", st["train"]["text"])
        self.assertNotIn("运行中", st["train"]["text"])


class TestOverview(DashboardCase):
    def test_overview_is_a_summary(self):
        self.sets()
        cand = self.candidate()
        trials, _search = self.tune_trials()
        record = self.reg("record", artifacts=["trainset-0001"])
        self.assertEqual(record, "record-0001")
        ev = self.evaluation(cand, self.delivery(cand))
        Workspace(self.ws).promote(ev)
        self.plan(ev)
        p = pages(self.ws)
        for sec in ("flow", "best", "stop", "now"):
            self.assertIn('id="%s"' % sec, p[OVERVIEW])
        overview = p[OVERVIEW].replace(section(p[OVERVIEW], "recent"), "")  # 最近事件 quotes the events page
        self.assertNotIn("record-0001", overview)
        for t in trials:
            self.assertNotIn(t, overview)
        self.assertIn("record-0001", p[ROUNDS])
        for t in trials:
            self.assertIn(t, p[ROUNDS])

    def test_milestone_question_hides_plan_question(self):
        self.plan_task("ws-ms-plan", target=0.8)
        plan = self.plan(self.promoted_round())  # registered, not approved
        ws = Workspace(self.ws)
        self.assertTrue(ws.stop_check()[0])
        items = D.pending_questions(ws, D.module_progress(ws))
        self.assertTrue(any("里程碑" in i["text"] for i in items), items)
        self.assertFalse(any("批准计划" in i["text"] or plan in i["text"] for i in items), items)
        self.assertIn("里程碑", self.waiting())
        self.assertFlow(check="wait")

    def test_best_card(self):
        self.assertIn("还没有版本", section(self.overview(), "best"))
        self.promoted_round()
        best = section(self.overview(), "best")
        self.assertIn("V1", best)
        self.assertIn(D._num(Workspace(self.ws).best_primary()), best)
        self.assertNotIn('<svg class="spark"', best)
        task = fixtures.binary_task()
        task["stop"]["target"] = 0.8
        self.use_task(task, "ws-target-reached")
        self.promoted_round()  # auc 1 reaches the target 0.8
        best = section(self.overview(), "best")
        self.assertIn("距目标", best)
        self.assertNotIn("meter hot", best)
        self.assertIn('style="width:100%"', best)
        self.use_task(fixtures.binary_task(), "ws-two-versions")
        self.two_promoted_rounds()
        best = section(self.overview(), "best")
        self.assertIn("V2", best)
        self.assertIn('<svg class="spark"', best)
        # Unknown numbers: a lower-is-better metric rsi does not know, no target, no budget.
        task = fixtures.binary_task()
        task["metrics"]["primary"] = "expected_cost"
        task["metrics"]["higher_is_better"] = False
        task["budget"]["compute_hours"] = None
        task["stop"]["target"] = None
        self.use_task(task, "ws-unknown")
        page = self.overview()  # no version yet: the best value is unknown
        self.assertIn("还没有版本", section(page, "best"))
        self.assertIn("目标未设", section(page, "best"))
        self.promoted_round()
        page = self.overview()
        best = section(page, "best")
        self.assertIn("expected_cost", best)
        self.assertIn("0.85", best)
        self.assertIn("目标未设", best)  # the null target
        self.assertNotIn("距目标", best)
        self.assertIn("无上限", section(page, "stop"))  # the null budget

    def test_stop_card(self):
        stop = section(self.overview(), "stop")
        for label in ("已完成轮数", "连续未升级", "本轮尝试", "评测集使用", "算力"):
            self.assertIn(label, stop)
        self.assertIsNone(self.task["budget"]["compute_hours"])
        self.assertIn("无上限", stop)
        self.assertNotIn("停止检查", stop)  # no raw check-stop sentence
        self.assertNotIn("no stop condition", stop)
        self.assertNotIn("停止条件已满足", stop)
        task = fixtures.binary_task()
        task["budget"]["compute_hours"] = 10
        self.use_task(task, "ws-budget")
        self.assertNotIn("无上限", section(self.overview(), "stop"))
        task = fixtures.binary_task()
        task["stop"]["target"] = 0.8
        self.use_task(task, "ws-stop-met")
        self.promoted_round()
        stop = section(self.overview(), "stop")
        self.assertIn("停止条件已满足", stop)
        self.assertNotIn("reached target", stop)
        Workspace(self.ws).stop("alice", "stop now")
        stop = section(self.overview(), "stop")
        self.assertNotIn("停止条件已满足", stop)
        self.assertNotIn("stop now", stop)  # the reason is in the now card, not repeated here

    def test_now_card(self):
        self.sets()
        lead, detail = self.now_parts()
        self.assertEqual(lead, "下一步：② 训练调优")
        self.assertTrue(detail.startswith("派 rsi-train（"), detail)  # the action, muted
        self.assertIn("已用尝试 0 / 3", detail)
        self.assertRegex(section(self.overview(), "now"), r"自 \d{4}-\d\d-\d\d \d\d:\d\d:\d\d 起")
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        lead, detail = self.now_parts()
        self.assertEqual(lead, "② 训练调优：rsi-train 运行中")
        self.assertIn("输出目录自", detail)  # the sentence names when the module started, once
        self.assertEqual(len(re.findall(r"\d\d:\d\d:\d\d", section(self.overview(), "now"))), 1)
        Workspace(self.ws).stop("alice", "stop now")
        lead, detail = self.now_parts()
        self.assertEqual(lead, "已停止：user requested: stop now")
        self.assertEqual(detail, "运行 rsi run 写运行报告。")  # no link to an empty report page; rsi run writes it
        self.assertNotIn("dashboard/report.html", section(self.overview(), "now"))
        with open(os.path.join(self.ws, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# 报告\n")
        self.assertIn('href="dashboard/report.html"', self.now_parts()[1])
        # A plan to approve, then attempts used up.
        self.plan_task("ws-now-plan")
        plan = self.plan(self.promoted_round())
        lead, detail = self.now_parts()
        self.assertEqual(lead, "等你批准计划 %s" % plan)
        self.assertIn("rsi answer", detail)  # the same answer command as every question; rsi approve the alternative
        now = section(self.overview(), "now")
        self.assertIn("answer %s approve" % self.ws, now)
        self.assertIn("approve %s plan --ref %s" % (self.ws, plan), now)
        self.assertNotIn(section(self.overview(), "ask").split("<li>")[1].split("<")[0], detail)  # not the banner
        row = decision_row(round_block(pages(self.ws)[ROUNDS], 1))  # the rounds page: not approved yet
        self.assertIn(plan, row)
        self.assertIn("未批准", row)
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-now-close")
        self.sets()
        self.failed_search()
        lead, detail = self.now_parts()
        self.assertEqual(lead, "等关闭本轮（rsi close-round）")
        self.assertIn("rsi run 关闭本轮（rsi close-round 的效果）", detail)  # rsi run does it, not the operator


class TestSecondaryPages(DashboardCase):
    def test_rounds_page(self):
        self.plan_task("ws-rounds")
        self.two_promoted_rounds()
        page = pages(self.ws)[ROUNDS]
        rounds = section(page, "rounds")
        tags = re.findall(r'<details class="round"[^>]*>', rounds)
        self.assertEqual(len(tags), 2, tags)
        self.assertIn('id="round-2"', tags[0])
        self.assertIn(" open", tags[0])
        self.assertIn('id="round-1"', tags[1])
        self.assertNotIn(" open", tags[1])
        r1 = round_block(rounds, 1)
        trial = 'href="../artifacts/candidate-0002/manifest.json"'  # a trial candidate of round 1
        nested = r1.index('<details class="trials"')
        self.assertLess(nested, r1.index(trial))
        self.assertLess(r1.index(trial), r1.index("</details>", nested))
        self.assertIn('href="../artifacts/candidate-0001/manifest.json"', page)
        row = decision_row(r1)
        self.assertIn("升级 V1", row)
        self.assertIn("plan-0001", row)
        self.assertIn("已批准", row)
        self.assertNotIn("未批准", row)
        self.assertIn("升级 V2", decision_row(round_block(rounds, 2)))

    def test_events_page(self):
        self.plan_task("ws-events")
        self.sets()
        cand = self.candidate()
        trials, _search = self.tune_trials()
        ev = self.evaluation(cand, self.delivery(cand))
        Workspace(self.ws).promote(ev)
        plan = self.plan(ev)
        Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        events = section(pages(self.ws)[EVENTS], "events")
        self.assertLess(events.index("evaluation-0001"), events.index("task-0001"))  # newest first
        self.assertIn("登记 %d 个试验候选" % (1 + len(trials)), events)  # train's candidate and the tune trials
        self.assertIn(cand, events)
        self.assertIn(trials[-1], events)
        self.assertNotIn(trials[0], events)  # one line for the run of trials
        self.assertIn("批准 plan", events)
        self.assertIn("第 1 轮判定", events)
        self.assertIn('href="../artifacts/evaluation-0001/manifest.json"', events)

    def test_report_page(self):
        self.assertIn("还没有运行报告", section(pages(self.ws)[REPORT], "report"))
        from rsi.report import write_report
        self.promoted_round()
        path = os.path.join(self.ws, "REPORT.md")
        write_report(Workspace(self.ws), path)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n<b>x</b> & 结论\n")
        page = pages(self.ws)[REPORT]
        report = section(page, "report")
        self.assertIn("<table>", report)
        self.assertIn("<h2>", report)
        self.assertFalse([line for line in page.splitlines() if line.startswith("|")])
        self.assertIn("&lt;b&gt;x&lt;/b&gt; &amp; 结论", report)
        self.assertNotIn("<b>x</b>", page)
        self.assertIn('href="../REPORT.md"', page)


class TestProgressTruth(DashboardCase):
    """Fixes from the simulation: refused hand-backs, a milestone release in progress, tune before
    search, and smaller page fixes."""

    def questions(self):
        ws = Workspace(self.ws)
        return D.pending_questions(ws, D.module_progress(ws))

    def waiting_and_flow(self):
        ws = Workspace(self.ws)
        progress = D.module_progress(ws)
        return D.waiting_for(ws, progress), D.flow_states(ws, progress)

    def pending_delivery(self, cand, name="delivery-a", subdir="deliver", age=3600):
        """A delivery handed back in work/round-1/<subdir>/<name> and never registered (refused);
        its directory is `age` seconds old, as if the module had written it some time ago."""
        d = fixtures.make_artifact("delivery", os.path.join(self.ws, "work", "round-1", subdir, name),
                                   self.task, ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand, round=1)
        past = datetime.datetime.now().timestamp() - age
        os.utime(d, (past, past))
        return d

    def test_superseded_refused_handback(self):
        # S2: a refused delivery stays in deliver/; a later registration of the same kind by the same
        # module (from deliver-2/) supersedes it.
        self.sets()
        cand = self.candidate()
        self.pending_delivery(cand)
        ws = Workspace(self.ws)
        self.assertEqual(D.module_progress(ws)["deliver"]["state"], "unregistered")
        w, flow = self.waiting_and_flow()
        self.assertIn("登记", w)
        self.assertEqual(flow["deliver"]["state"], "current")
        dlv = self.delivery(cand)  # registered from elsewhere afterwards
        ws = Workspace(self.ws)
        p = D.module_progress(ws)["deliver"]
        self.assertEqual(p["state"], "done")
        self.assertEqual(p["pending"], [])
        self.assertEqual(p["superseded"], {"delivery-a": dlv})
        w, flow = self.waiting_and_flow()
        self.assertNotIn("登记", w)
        self.assertEqual(flow["deliver"]["state"], "done")
        block = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("已被 %s 取代" % dlv, block)
        self.assertNotIn("交回未登记", block)

    def test_refused_handback_without_later_registration_stays_pending(self):
        self.sets()
        cand = self.candidate()
        self.pending_delivery(cand)
        self.reg("candidate", ids=VAL_IDS, wrong=(0,), train_ids=TRAIN_IDS)  # another kind: no supersede
        self.assertEqual(D.module_progress(Workspace(self.ws))["deliver"]["state"], "unregistered")
        self.assertIn("登记", self.waiting())
        # A delivery registered BEFORE the refused directory was written does not supersede it either.
        self.delivery(cand)
        self.pending_delivery(cand, "delivery-b", "deliver-2", age=-3600)
        p = D.module_progress(Workspace(self.ws))["deliver"]
        self.assertEqual(p["state"], "unregistered")
        self.assertIn("delivery-b", p["pending"])
        self.assertIn("交回未登记", round_block(pages(self.ws)[ROUNDS], 1))

    def milestone_round(self):
        """A promoted round whose stop check says stop (target met), V1 best; return the candidate."""
        self.target_task()
        self.promoted_round()
        return Workspace(self.ws).version("V1")["candidate_id"]

    def target_task(self):
        task = fixtures.binary_task()
        task["stop"]["target"] = 0.8
        task["approvals"] = [x for x in task["approvals"] if x != "plan"]
        self.use_task(task, "ws-ms")

    def milestone_delivery(self, cand):
        d = fixtures.make_artifact("delivery", self.src("delivery"), self.task, ids=EVAL_IDS, val_ids=VAL_IDS,
                                   candidate_id=cand, round=1)
        fixtures.add_predictions(d, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        return Workspace(self.ws).register("delivery", d)

    def test_milestone_in_progress(self):
        # S3: once the user said yes (use_testset), `rsi next` must not be suggested until the
        # milestone evaluation is registered.
        cand = self.milestone_round()
        w, _flow = self.waiting_and_flow()
        self.assertIn("rsi next", w + " " + "".join(q["text"] for q in self.questions()))  # question time
        Workspace(self.ws).approve("use_testset", "alice", "run the test inputs")
        w, flow = self.waiting_and_flow()
        self.assertIn("里程碑发布进行中", w)
        self.assertIn("rsi-deliver", w)
        self.assertNotIn("rsi next", w)
        self.assertEqual(flow["check"]["state"], "current")
        self.assertIn("里程碑发布中", flow["check"]["text"])
        page = self.overview()
        self.assertIn("里程碑发布进行中", section(page, "now"))
        self.assertNotIn("rsi next", section(page, "now"))
        self.milestone_delivery(cand)
        w, flow = self.waiting_and_flow()
        self.assertIn("里程碑发布进行中", w)
        self.assertIn("rsi-judge", w)
        self.assertNotIn("rsi next", w)
        self.assertIn("里程碑发布中", flow["check"]["text"])
        Workspace(self.ws).approve("use_testset", "alice", "run the test inputs")
        self.reg("evaluation", 1, ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id="delivery-0002",
                 candidate_id=cand, evalset_id="evalset-0001", uses_testset=True, test_ids=TEST_IDS,
                 best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
                 recommendation={"decision": "keep", "reason": "milestone release"})
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi next", w)
        self.assertIn("停止条件已满足", w)
        self.assertNotIn("里程碑发布进行中", w)
        self.assertNotIn("里程碑发布中", flow["check"]["text"])
        self.assertIn("升级 V1 · 里程碑评测", flow["judge"]["text"])  # M19, the round's verdict stays
        self.assertNotIn("先问用户是否做里程碑发布", w)

    def test_trials_without_search_wait_for_the_search(self):
        # S4: a train dispatch registered its search's trials: the search comes before ③.
        self.sets()
        out = "work/round-1/train"
        os.makedirs(os.path.join(self.ws, out))
        Workspace(self.ws).add_dispatches([{"module": "train", "purpose": "train", "outdir": out, "status": "done",
                                            "session": "s-1"}])
        src = fixtures.make_artifact("candidate", os.path.join(self.ws, out, "trial-1"), self.task, ids=VAL_IDS,
                                     train_ids=TRAIN_IDS)
        trial = Workspace(self.ws).register("candidate", src)
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-train 写 search", w)
        self.assertNotIn("③ 交付", w)
        self.assertNotEqual(flow["deliver"]["state"], "current")
        self.assertIn("rsi-train", section(self.overview(), "now"))
        self.search_over([trial])
        self.assertIn("rsi-deliver", self.waiting())

    def test_events_same_second_follow_registry_order(self):
        # M11: entries registered in one second are listed newest first, also within that second.
        self.sets()
        ws = Workspace(self.ws)
        path = os.path.join(self.ws, "registry.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            r["registered_at"] = "2026-11-22T10:00:00Z"
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        events = section(pages(self.ws)[EVENTS], "events")
        names = re.findall(r"登记(测试集|评测集|验证集|训练集)", events)
        self.assertEqual(names, ["训练集", "验证集", "评测集", "测试集"])

    def test_decided_current_round_is_not_running(self):
        # M12
        self.promoted_round()
        head = round_block(pages(self.ws)[ROUNDS], 1).split("</summary>")[0]
        self.assertIn("升级 V1", head)
        self.assertNotIn("进行中", head)
        self.assertRegex(head, r">已 [^<]*分")

    def test_data_node_counts_current_sets_only(self):
        # M13
        self.reg("testset", ids=TEST_IDS)
        self.reg("evalset", ids=EVAL_IDS)
        self.reg("valset", ids=VAL_IDS)
        self.reg("trainset", ids=range(50, 54),
                 quality={"passed": False, "checks": [{"name": "dup_rate", "value": 0.5, "threshold": 0.05,
                                                       "passed": False}]})
        self.reg("trainset", ids=TRAIN_IDS)
        _w, flow = self.waiting_and_flow()
        self.assertEqual(flow["data"]["text"], "✓ 4 个集合")

    def test_redelivery_of_best_does_not_finish_step_four(self):
        # M16: after a refresh the best version needs a re-delivery on the new evalset; that re-delivery (an earlier
        # round's candidate) is not the round's delivery.
        self.refresh_task("ws-refresh-step4")
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        self.assertIsNotNone(Workspace(self.ws).best_delivery_on_current_evalset())
        self.assertNotIn("重新交付", self.waiting())
        self.reg("evalset", 2, ids=range(60, 64))  # refreshed evalset: V1 has no delivery on it
        self.assertIsNone(Workspace(self.ws).best_delivery_on_current_evalset())
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-deliver", w)
        self.assertIn("重新交付", w)
        self.assertIn("V1", w)
        best_cand = Workspace(self.ws).version("V1")["candidate_id"]
        self.reg("delivery", 2, ids=range(60, 64), val_ids=VAL_IDS, candidate_id=best_cand, evalset_id="evalset-0002")  # re-delivery of V1
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-deliver", w)
        self.assertNotIn("重新交付", w)
        self.assertNotIn("rsi-judge", w)
        self.assertEqual(flow["deliver"]["state"], "current")
        self.assertEqual(flow["judge"]["state"], "todo")
        cand2 = [a for a in Workspace(self.ws).passed_candidates()][0]
        self.reg("delivery", 2, ids=range(60, 64), val_ids=VAL_IDS, candidate_id=cand2, evalset_id="evalset-0002", wrong=(0,))
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-judge", w)
        self.assertEqual(flow["deliver"]["state"], "done")

    def test_tune_trials_and_siblings_of_one_handback_stay_pending(self):
        # Registering one artifact of a hand-back from its own directory does not supersede its siblings. An old
        # workspace's tune/ output directory is ②'s.
        self.sets()
        for i in range(3):
            fixtures.make_artifact("candidate", os.path.join(self.ws, "work", "round-1", "tune", "trial-%d" % i),
                                   self.task, ids=VAL_IDS, train_ids=TRAIN_IDS, producer="tune",
                                   config_source="tune", round=1, notes="t%d" % i)
        Workspace(self.ws).register("candidate", os.path.join(self.ws, "work", "round-1", "tune", "trial-0"))
        p = D.module_progress(Workspace(self.ws))["train"]
        self.assertEqual(p["pending"], ["trial-1", "trial-2"])
        self.assertEqual(p["superseded"], {})
        self.assertEqual(p["state"], "unregistered")

    def test_deliver_entry_round_finishes_step_four_with_its_delivery(self):
        # A round whose plan enters at deliver has no candidate of its own.
        ev = self.promoted_round()
        self.reg("plan", 1, evaluation_id=ev, entry_module="deliver",
                 issues=[dict(fixtures._plan_issue(), attribution="runtime", module="deliver")])
        Workspace(self.ws).next_round()
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-deliver", w)
        cand = Workspace(self.ws).version("V1")["candidate_id"]
        self.reg("delivery", 2, ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand, wrong=(0,))
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-judge", w)
        self.assertEqual(flow["deliver"]["state"], "done")

    def test_round_delivery_does_not_hide_the_missing_redelivery(self):
        # After a refresh the round's own delivery is in but the best version still has none on the new evalset.
        self.refresh_task("ws-refresh-late")
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        self.reg("evalset", 2, ids=range(60, 64))
        cand2 = Workspace(self.ws).passed_candidates()[0]
        self.reg("delivery", 2, ids=range(60, 64), val_ids=VAL_IDS, candidate_id=cand2, evalset_id="evalset-0002")
        self.assertIsNone(Workspace(self.ws).best_delivery_on_current_evalset())
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-deliver", w)
        self.assertIn("重新交付", w)
        self.assertNotIn("rsi-judge", w)
        self.assertEqual(flow["deliver"]["state"], "current")
        self.assertIn("待重新交付最佳版本", flow["deliver"]["text"])

    def test_attempts_meter_is_hot_only_without_a_passing_candidate(self):
        # M17: the first search passes, a second one adds trials (attempts 2 / 3 here)
        self.sets()
        self.search_over([self.candidate()])
        self.tune_trials(n=2)
        ws = Workspace(self.ws)
        self.assertEqual((len(ws.attempts()), ws.attempt_limit()), (2, 3))
        self.assertNotIn("hot", self.meter("本轮尝试"))
        self.search_over([self.reg("candidate", ids=VAL_IDS, wrong=(1,), train_ids=TRAIN_IDS)])  # third, passing
        self.assertEqual(len(Workspace(self.ws).attempts()), 3)
        self.assertNotIn("hot", self.meter("本轮尝试"))
        task = fixtures.binary_task()
        task["stop"]["max_attempts_per_round"] = 1
        self.use_task(task, "ws-hot")
        self.sets()
        self.failed_search()
        self.assertIn("hot", self.meter("本轮尝试"))

    def meter(self, label):
        stop = section(self.overview(), "stop")
        return re.search(r'<div class="meter[^"]*">(?:(?!<div class="meter).)*?%s' % label, stop, re.S).group(0)


class TestPages(DashboardCase):
    def test_tabs_and_links(self):
        self.promoted_round()
        p = pages(self.ws, now=datetime.datetime(2026, 11, 22, 9, 30, 15))
        self.assertEqual(set(p), set(D.PAGES))
        for rel, text in p.items():
            self.assertIn("更新于 2026-11-22 09:30:15", text, rel)  # the given generation time
        for rel in (ROUNDS, EVENTS, REPORT):
            self.assertIn('href="dashboard/%s"' % rel.split("/")[1], p[OVERVIEW])
            self.assertIn('href="../dashboard.html"', p[rel])
            for other in (ROUNDS, EVENTS, REPORT):
                self.assertIn('href="%s"' % other.split("/")[1], p[rel])
        for (rel, text), label in zip(sorted(p.items(), key=lambda kv: D.PAGES.index(kv[0])), TABS):
            for other in TABS:
                self.assertIn(other, text)
            self.assertEqual(re.findall(r'<a class="on" href="[^"]*">([^<]*)</a>', text), [label], rel)

    def test_escaping(self):
        self.sets()
        Workspace(self.ws).close_round('<b>x</b> & "y"')
        p = pages(self.ws)
        for rel in (ROUNDS, EVENTS):
            self.assertIn("&lt;b&gt;x&lt;/b&gt; &amp;", p[rel])
        for rel, text in p.items():
            self.assertNotIn("<b>x</b>", text, rel)

    def test_rendering_reads_only_allowed_files_and_writes_nothing(self):
        self.sets()
        self.chain()
        with open(os.path.join(self.ws, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# 报告\n")
        # Module output in work/ is listed, never read.
        out = fixtures.make_artifact("candidate", os.path.join(self.ws, "work", "round-1", "train", "cand"),
                                     self.task, ids=VAL_IDS)
        os.makedirs(os.path.join(self.ws, "work", "round-1", "judge"))
        with open(os.path.join(self.ws, "work", "notes.md"), "w", encoding="utf-8") as f:
            f.write("repo root\n")
        with open(os.path.join(self.ws, "work", "run.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"at": "2026-11-23T08:00:00Z", "event": "ask", "question": "q"}) + "\n")
        self.assertTrue(os.path.isfile(os.path.join(out, "predictions_val.jsonl")))
        # A dispatch working in that directory: its progress.json and hand-backs are read, nothing else there.
        Workspace(self.ws).add_dispatches([{"module": "train", "outdir": "work/round-1/train"}])
        for name in ("progress.json", "handback.json", "handback-1.json"):
            with open(os.path.join(self.ws, "work", "round-1", "train", name), "w", encoding="utf-8") as f:
                f.write("{}")

        def snapshot():
            out = []
            for dirpath, dirs, files in os.walk(self.ws):
                for name in dirs + files:
                    p = os.path.join(dirpath, name)
                    out.append((os.path.relpath(p, self.ws), os.stat(p).st_mtime_ns))
            return sorted(out)

        before = snapshot()
        ws = Workspace(self.ws)
        with mock.patch("builtins.open", wraps=builtins.open) as opened:
            p = pages(ws)
        self.assertEqual(set(p), set(D.PAGES))
        names = set(os.path.basename(str(c[0][0])) for c in opened.call_args_list)
        self.assertLessEqual({"REPORT.md", "run.jsonl", "gpus.json", "progress.json", "handback.json", "handback-1.json"},
                             names)
        self.assertLessEqual(set(n for n in names if not HANDBACK_COPY.match(n)), ALLOWED)
        self.assertEqual(snapshot(), before)

    def test_write_dashboard_writes_four_pages(self):
        self.assertEqual(D.PAGES, (OVERVIEW, ROUNDS, EVENTS, REPORT))
        path = D.write_dashboard(Workspace(self.ws))
        self.assertEqual(path, os.path.join(os.path.realpath(self.ws), "dashboard.html"))  # absolute
        for rel in D.PAGES:
            p = os.path.join(self.ws, *rel.split("/"))
            with open(p, encoding="utf-8") as f:
                text = f.read()
            self.assertTrue(text.lstrip().lower().startswith("<!doctype html>"), rel)
            self.assertIn('<meta http-equiv="refresh" content="5">', text)
            self.assertIn("spam-check", text)  # the task
            for label in TABS:
                self.assertIn(label, text)
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o644, rel)
        for d in (self.ws, os.path.join(self.ws, "dashboard")):
            self.assertEqual([n for n in os.listdir(d) if n.endswith(".tmp")], [], d)
        # A failed write leaves no temp file behind and raises.
        os.remove(path)
        os.makedirs(os.path.join(path, "occupied"))
        with self.assertRaises(OSError):
            D.write_dashboard(Workspace(self.ws))
        for d in (self.ws, os.path.join(self.ws, "dashboard")):
            self.assertEqual([n for n in os.listdir(d) if n.endswith(".tmp")], [], d)

    def test_renders_in_every_state(self):
        states = {"fresh": self.ws}
        # A module running mid-round.
        self.use_task(fixtures.binary_task(), "ws-running")
        self.sets()
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        states["running"] = self.ws
        # A round closed with close-round: keep, null version and evaluation ids.
        self.use_task(fixtures.binary_task(), "ws-closed")
        self.sets()
        Workspace(self.ws).close_round("no candidate passed the train gate")
        self.assertIsNone(Workspace(self.ws).state["rounds"]["1"]["decision"]["evaluation_id"])
        states["closed"] = self.ws
        # Stopped by the user before any version.
        self.use_task(fixtures.binary_task(), "ws-stopped")
        Workspace(self.ws).stop("alice", "stop now")
        states["stopped without version"] = self.ws
        # Stopped after a promoted round and a milestone evaluation on the testset.
        task = fixtures.binary_task()
        task["stop"]["target"] = 0.8
        self.use_task(task, "ws-milestone")
        ev1 = self.promoted_round()
        Workspace(self.ws).approve("use_testset", "alice", "run the test inputs")
        d = fixtures.make_artifact("delivery", self.src("delivery"), self.task, ids=EVAL_IDS, val_ids=VAL_IDS,
                                   candidate_id="candidate-0001")
        fixtures.add_predictions(d, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        self.assertEqual(Workspace(self.ws).register("delivery", d), "delivery-0002")
        Workspace(self.ws).approve("use_testset", "alice", "score it on the test")
        self.assertEqual(self.reg("evaluation", ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id="delivery-0002",
                                  candidate_id="candidate-0001", evalset_id="evalset-0001",
                                  best_before={"version": "V1", "evaluation_id": ev1}, uses_testset=True,
                                  test_ids=TEST_IDS), "evaluation-0002")
        self.assertIsNone(Workspace(self.ws).next_round()[0])
        states["stopped after milestone"] = self.ws
        ids = {OVERVIEW: ("flow", "best", "stop", "now", "doing", "judgement", "gpus", "recent"), ROUNDS: ("rounds",),
               EVENTS: ("events",),
               REPORT: ("report",)}
        for name, ws in states.items():
            with self.subTest(name):
                p = pages(ws)
                self.assertEqual(set(p), set(D.PAGES))
                for rel, secs in ids.items():
                    for sec in secs:
                        self.assertIn('id="%s"' % sec, p[rel])
                self.assertTrue(D.waiting_for(Workspace(ws), D.module_progress(Workspace(ws))))


class TestQuestions(DashboardCase):
    def questions(self):
        ws = Workspace(self.ws)
        return D.pending_questions(ws, D.module_progress(ws))

    def state_items(self):
        return [i for i in self.questions() if i["source"] == "state"]

    def test_plan_approval_is_a_question(self):
        self.plan_task("ws-q-plan")
        plan = self.plan(self.promoted_round())
        items = self.questions()
        self.assertEqual([i["source"] for i in items], ["state"])
        self.assertIn(plan, items[0]["text"])
        self.assertIsNotNone(items[0]["since"])
        p = pages(self.ws)
        page = p[OVERVIEW]
        self.assertIn('id="ask"', page)
        self.assertLess(page.index('id="ask"'), page.index('id="flow"'))
        self.assertIn("⏸ 等你在 Claude 里回答", page)
        self.assertIn(plan, section(page, "ask"))
        for rel, text in p.items():
            self.assertRegex(text, r"<title>⏸ 需要你回答", rel)
            self.assertIn("⏸ 等你回答", text)  # the status label of the top bar
        self.write_notes("## 待用户回答\n- 笔记问题\n")  # STATE's questions come before the notes' ones
        self.assertEqual([i["source"] for i in self.questions()], ["state", "notes"])
        self.write_notes("## 待用户回答\n")
        Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        self.assertEqual(self.questions(), [])
        page = self.overview()
        self.assertNotIn('id="ask"', page)
        self.assertNotIn("⏸", re.search(r"<title>.*?</title>", page).group(0))
        self.assertIn('<span class="chip idle">等待 rsi run</span>', page)  # "● 运行中" only while rsi run works

    def target_task(self, name):
        task = fixtures.binary_task()
        task["stop"]["target"] = 0.8
        task["approvals"] = [x for x in task["approvals"] if x != "plan"]
        self.use_task(task, name)

    def test_milestone_question_is_asked_before_next(self):
        self.target_task("ws-q-ms")
        self.promoted_round()
        ws = Workspace(self.ws)
        self.assertTrue(ws.stop_check()[0])
        # A use_testset approval given before the round's decision does not count.
        ws.state["approvals"].append({"what": "use_testset", "ref": None, "by": "alice", "quote": "ok",
                                      "at": "2020-01-01T00:00:00Z", "round": 1})
        ws.save_state()
        items = self.state_items()
        self.assertEqual(len(items), 1)
        self.assertIn("里程碑", items[0]["text"])
        page = pages(ws)[OVERVIEW]
        self.assertIn("里程碑", section(page, "ask"))
        w = self.waiting()
        self.assertIn("等用户回答", w)
        self.assertNotIn("派 rsi-", w)
        self.assertIn("等你在 Claude 里回答上方的问题（1 条）", section(page, "now"))
        self.assertFlow(check="wait")
        # The user says yes: the use_testset approval ends the question.
        Workspace(self.ws).approve("use_testset", "alice", "run the test inputs")
        self.assertEqual(self.state_items(), [])

    def test_milestone_question_gone_once_stopped(self):
        self.target_task("ws-q-ms-stopped")
        self.promoted_round()
        self.assertEqual(len(self.state_items()), 1)
        self.assertIsNone(Workspace(self.ws).next_round()[0])  # the user declined: rsi next stops
        self.assertEqual(self.state_items(), [])
        page = self.overview()
        self.assertNotIn('id="ask"', page)
        self.assertIn("■ 已停止", page)

    def test_no_milestone_question_without_version(self):
        # A round closed without a candidate leaves no version. max_rounds 1 makes the stop check say stop,
        # but there is no best version to validate on the test inputs: no milestone question.
        task = fixtures.binary_task()
        task["stop"]["max_rounds"] = 1
        self.use_task(task, "ws-q-nover")
        self.sets()
        Workspace(self.ws).close_round("no candidate passed")
        ws = Workspace(self.ws)
        self.assertTrue(ws.stop_check()[0])
        self.assertIsNone(ws.state["best"])
        self.assertEqual(self.questions(), [])

    def test_over_budget_question(self):
        # Used compute reaching the budget ends the run at the next `rsi next`, so the state where
        # more training would go over budget is a round still waiting for a candidate: here round 1
        # after a candidate that failed its train gate used up the budget.
        probe = fixtures.binary_task()
        probe["budget"]["compute_hours"] = None
        self.use_task(probe, "ws-q-budget-probe")
        self.sets()
        cost = self.manifest_cost(self.candidate())
        for needs, expect in ((True, 1), (False, 0)):
            task = fixtures.binary_task()
            task["approvals"] = [a for a in task["approvals"] if a != "over_budget"] + (
                ["over_budget"] if needs else [])
            task["budget"]["compute_hours"] = cost
            self.use_task(task, "ws-q-budget-%s" % needs)
            self.sets()
            self.assertEqual(self.state_items(), [])  # nothing used yet
            self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
            self.assertEqual(Workspace(self.ws).state["budget_used"]["compute_hours"], cost)
            items = self.state_items()
            self.assertEqual(len(items), expect, items)
            if expect:
                self.assertIn("over_budget", items[0]["text"])
                self.assertIn("over_budget", section(self.overview(), "ask"))
                self.assertFlow(train="fail")  # (the node does not show the wait yet: Task 3)
                Workspace(self.ws).approve("over_budget", "alice", "go over budget")
                self.assertEqual(self.state_items(), [])
                self.assertNotIn("等用户回答", self.waiting())
            else:
                self.assertFlow(train="fail")

    def manifest_cost(self, cand):
        return Workspace(self.ws).manifest(cand)["cost"]["compute_hours"]

    def test_refresh_question(self):
        # No question while the round is decided or round 2 still waits for a candidate; then one, whose since (M15)
        # is the registration time of the evaluation that used up the evalset; the approval ends it.
        task = fixtures.binary_task()
        task["eval_use_limit"] = 1
        task["approvals"] = task["approvals"] + ["plan"]
        self.use_task(task, "ws-q-refresh")
        ev = self.promoted_round()
        self.assertEqual(self.state_items(), [])  # decided round: nothing to refresh yet
        self.assertEqual(Workspace(self.ws).state["eval_use_count"], 1)
        plan = self.plan(ev)
        Workspace(self.ws).approve("plan", "alice", "go", plan)
        Workspace(self.ws).next_round()
        self.assertEqual(self.state_items(), [])  # round 2 still waits for a candidate
        self.candidate(2)
        items = self.state_items()
        self.assertEqual(len(items), 1)
        self.assertIn("refresh_sets", items[0]["text"])
        self.assertEqual(items[0]["since"], D._utc_iso(Workspace(self.ws).entry(ev)["registered_at"]))
        self.assertIn("自 ", section(self.overview(), "ask"))
        w = self.waiting()
        self.assertIn("等用户回答", w)
        self.assertNotIn("派 rsi-deliver", w)
        st = self.assertFlow(deliver="wait")  # delivering waits for the user
        self.assertEqual(st["deliver"]["text"], "⏸ 等你批准")
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        self.assertEqual(self.state_items(), [])
        self.assertIn("rsi-data", self.waiting())  # rsi.flow: the new evalset comes first, then ③
        self.assertFlow(data="current", deliver="todo")

    def test_plan_question_waits_for_running_judge(self):
        # The judge is dispatched again to rewrite the plan: the page shows it running, not a plan to approve.
        self.plan_task("ws-q-plan-rerun")
        self.plan(self.promoted_round())
        self.assertEqual(len(self.state_items()), 1)
        os.makedirs(os.path.join(self.ws, "work", "round-1", "judge-2"))
        self.assertEqual(self.state_items(), [])
        w = self.waiting()
        self.assertIn("rsi-judge", w)
        self.assertNotIn("批准计划", w)

    def test_notes_that_ask_nothing(self):
        self.assertEqual(self.questions(), [])  # no notes.md
        for text in ("# 笔记\n- 只是笔记\n## 其他\n- 不是问题\n", "# 笔记\n## 待用户回答\n\n   \n## 其他\n- x\n",
                     "## 待用户回答 （旧）\n- 不算\n"):  # no section, an empty one, another heading
            self.write_notes(text)
            self.assertEqual(self.questions(), [], text)
        os.remove(os.path.join(self.ws, "work", "notes.md"))
        # Unreadable notes do not break the page.
        os.makedirs(os.path.join(self.ws, "work", "notes.md"))  # a directory where the file should be
        self.assertEqual(self.questions(), [])
        self.assertIn('id="now"', self.overview())
        os.rmdir(os.path.join(self.ws, "work", "notes.md"))
        self.write_notes("## 待用户回答\n- x\n")
        os.chmod(os.path.join(self.ws, "work", "notes.md"), 0)
        self.addCleanup(os.chmod, os.path.join(self.ws, "work", "notes.md"), 0o644)
        self.assertEqual(self.questions(), [])  # unreadable (or readable when running as root)

    def write_notes(self, text):
        path = os.path.join(self.ws, "work", "notes.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_notes_questions(self):
        self.write_notes("# 笔记\nrepo root: /x\n\n## 待用户回答\n- 质量门没过，要重做还是放宽？\n"
                         "* <删除> 旧候选？\n\n## 其他\n- 不是问题\n")
        q = self.questions()
        self.assertEqual([i["text"] for i in q if i["source"] == "notes"],
                         ["质量门没过，要重做还是放宽？", "<删除> 旧候选？"])
        self.assertIsNotNone(q[-1]["since"])
        for rel, text in pages(self.ws).items():
            self.assertNotIn("不是问题", text, rel)
            self.assertNotIn("<删除>", text, rel)
        self.assertIn("&lt;删除&gt; 旧候选？", section(self.overview(), "ask"))

    def test_notes_question_pauses_the_overview(self):
        self.sets()
        self.write_notes("## 待用户回答\n- 训练用哪台机器？\n")
        page = self.overview()
        now = section(page, "now")
        lead = re.search(r'<div class="wait-now">(.*?)</div>', now).group(1)
        detail = re.search(r'<div class="detail">(.*?)</div>', now).group(1)
        self.assertEqual(lead, "等你在 Claude 里回答上方的问题（1 条）")
        self.assertIn("派 rsi-train", detail)  # the step sentence moves to the detail
        st = self.assertFlow(train="wait", data="done")
        self.assertEqual(st["train"]["text"], "⏸ 等你回答")
        self.assertIn('<span class="chip ask">⏸ 等你回答</span>', page)
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        st = self.assertFlow(train="wait")
        self.assertEqual(st["train"]["text"], "⏸ 运行中 · 等你回答")
        self.write_notes("## 待用户回答\n")
        st = self.assertFlow(train="current")
        self.assertIn("运行中", st["train"]["text"])

    def test_stopped_label_and_title(self):
        Workspace(self.ws).stop("alice", "stop now")
        page = self.overview()
        self.assertIn('<span class="chip stop">■ 已停止</span>', page)
        self.assertNotIn("⏸", re.search(r"<title>.*?</title>", page).group(0))
        self.write_notes("## 待用户回答\n- 要删掉旧候选吗？\n")
        for rel, text in pages(self.ws).items():
            self.assertIn('<span class="chip ask">⏸ 等你回答（已停止）</span>', text, rel)
            self.assertRegex(text, r"<title>⏸ 需要你回答", rel)
            self.assertNotIn("■ 已停止", text, rel)
        self.assertFlow(stop="current", check="done")  # a stopped run waits in no step


SESSION = "3f2a9c1e-5b7d-4e0a-9c3d-0123456789ab"


class EngineStateCase(DashboardCase):
    """Helpers for pages over the engine's records: dispatch records and run.jsonl lines."""

    def dispatch(self, module="train", **fields):
        rec = {"module": module, "outdir": "work/round-1/%s" % module, "runner": "claude", "session": SESSION,
               "purpose": module, "started_at": "2026-11-23T08:30:00Z", "launches": 1}
        rec.update(fields)
        return Workspace(self.ws).add_dispatches([rec])[0]

    def log(self, *events):
        path = os.path.join(self.ws, "work", "run.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for e in events:
                f.write((e if isinstance(e, str) else json.dumps(e, ensure_ascii=False)) + "\n")


class TestEngineState(EngineStateCase):
    """The pages show the engine's state: running agents, the question STATE waiting holds, the run log."""

    def test_running_agent_on_the_overview(self):
        self.sets()
        self.dispatch("train")
        self.dispatch("tune", session=None, runner="mock", started_at="2026-11-23T08:45:00Z")
        agents = D.running_agents(Workspace(self.ws))
        self.assertEqual([a["module"] for a in agents], ["train", "tune"])
        self.assertEqual(agents[0]["session"], SESSION)
        now = section(self.overview(), "now")
        self.assertIn("现在在等什么", now)
        self.assertIn("rsi-train", now)
        self.assertIn("rsi-tune", now)
        self.assertIn("3f2a9c1e", now)
        self.assertNotIn(SESSION, now)  # the short form only
        self.assertIn("claude", now)
        self.assertIn("mock", now)
        started = datetime.datetime.strptime("2026-11-23T08:30:00Z", "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).astimezone()
        self.assertIn(started.strftime("%H:%M"), now)
        st = self.assertFlow(train="current")
        self.assertIn("● 运行中", st["train"]["text"])

    def test_old_tune_records_show_under_training_tuning(self):
        # An old workspace's tune dispatch, trials, search and record: all under ② (rsi.dashboard.stage_of). Its
        # dispatches keep their own skill names: the running one and the one not launched yet, in one stage.
        self.sets()
        self.dispatch("tune", status="running", purpose="tune")
        self.dispatch("train", status="pending", session=None, runner="mock")
        _trials, search_id = self.tune_trials(n=2)
        record_id = self.reg("record", module="tune", producer="tune", artifacts=[search_id])
        ws = Workspace(self.ws)
        p = D.module_progress(ws)
        self.assertNotIn("tune", p)
        self.assertEqual(p["train"]["state"], "running")
        self.assertEqual(D.flow_states(ws, p)["train"]["state"], "current")
        self.assertEqual(self.now_parts()[0], "② 训练调优：rsi-tune 运行中，rsi-train 待启动")
        block = round_block(pages(ws)[ROUNDS], 1)
        self.assertIn("rsi-tune 进行中", block)
        self.assertIn("2 个试验候选", block)
        self.assertIn(search_id, block)
        self.assertNotIn("其他", block)
        self.assertIn("登记 2 个试验候选", pages(ws)[EVENTS])
        from rsi.report import RECORD_SECTIONS, report_markdown
        md = report_markdown(ws)
        self.assertIn("### ② 训练调优", md)
        self.assertLess(md.index("### ② 训练调优"), md.index(record_id))
        for title, modules in RECORD_SECTIONS[:4]:  # the report's sections are the pages' stages
            for m in modules:
                self.assertEqual(D.STEP_NAMES[D.stage_of(m)], title, m)

    def test_long_job_or_finished_dispatch_is_not_a_running_agent(self):
        self.sets()
        marker = os.path.join(self.root, "marker-never")
        self.dispatch("train", status="long_job", marker=marker)
        ws = Workspace(self.ws)
        self.assertEqual(D.running_agents(ws), [])
        self.assertEqual([a["id"] for a in D.waiting_long_jobs(ws)], ["d-0001"])
        now = section(self.overview(), "now")
        self.assertIn("长任务等待标记", now)
        self.use_task(fixtures.binary_task(), "ws-finished")
        self.sets()
        self.dispatch("train", status="handed_back")
        self.dispatch("tune", status="failed")
        self.assertEqual(D.running_agents(Workspace(self.ws)), [])
        self.assertNotIn('class="agents"', section(self.overview(), "now"))

    def test_run_log_tail_only(self):
        self.sets()
        old = json.dumps({"at": "2026-11-23T08:00:00Z", "event": "fresh_session", "dispatch_id": "d-OLD", "why": "x"})
        with open(os.path.join(self.ws, "work", "run.jsonl") if os.path.isdir(os.path.join(self.ws, "work"))
                  else self._mk(), "w", encoding="utf-8") as f:
            f.write(old + "\n")
            f.write(("x" * 1000 + "\n") * (D.RUN_LOG_TAIL // 1000 + 5))
            f.write(json.dumps({"at": "2026-11-23T09:00:00Z", "event": "fresh_session", "dispatch_id": "d-NEW",
                                "why": "y"}) + "\n")
        ids = [e.get("dispatch_id") for e in D.run_events(Workspace(self.ws))]
        self.assertEqual(ids, ["d-NEW"])

    def _mk(self):
        os.makedirs(os.path.join(self.ws, "work"), exist_ok=True)
        return os.path.join(self.ws, "work", "run.jsonl")

    def test_waiting_question_with_options_and_command(self):
        self.sets()
        self.assertNotIn("rsi answer", section(self.overview(), "now"))  # no question, no answer command
        did = self.dispatch("train", status="failed")  # the question is about an open failure (not stale)
        w = Workspace(self.ws).set_waiting({"kind": "question", "what": "handback_failed", "ref": did,
                                            "question": "交回被拒 <b>x</b>，选一个做法", "options": ["resume", "stop"]})
        page = self.overview()
        now = section(page, "now")
        self.assertIn("交回被拒 &lt;b&gt;x&lt;/b&gt;", now)
        self.assertNotIn("<b>x</b>", page)
        from rsi.briefs import rsi_command
        for opt in ("resume", "stop"):  # the full command, naming the question it answers
            self.assertIn("%s answer %s %s --ref %s --by " % (rsi_command(), self.ws, opt, w["id"]), now)
        self.assertIn("--quote", now)
        self.assertIn("&lt;你的名字&gt;", now)
        self.assertIn("&lt;用户原话&gt;", now)

    def test_events_page_merges_the_run_log(self):
        self.sets()
        ask = {"type": "ask", "question": "要不要关闭本轮？", "options": ["close_round", "stop"]}
        self.log(
            {"at": "2026-11-23T08:00:00Z", "event": "launch", "dispatch_id": "d-0001", "module": "train",
             "launch": 1, "mode": "first", "runner": "claude", "session": SESSION},
            {"at": "2026-11-23T08:10:00Z", "event": "resume", "dispatch_id": "d-0001", "mode": "resume",
             "launch": 2, "session": SESSION, "feedback": "feedback-2.md"},
            {"at": "2026-11-23T08:20:00Z", "event": "resume", "dispatch_id": "d-0001", "mode": "resume",
             "launch": 3, "session": SESSION, "feedback": "feedback-3.md"},
            {"at": "2026-11-23T08:21:00Z", "event": "decision", "point": "handback_refused", "choice": "resume",
             "by": "code", "reason": "默认 <原因>", "dispatch_id": "d-0001"},
            {"at": "2026-11-23T08:22:00Z", "event": "decision", "point": "train_gate_failed", "choice": "tune",
             "by": "agent", "reason": "配置问题", "dispatch_id": "d-0001"},
            {"at": "2026-11-23T08:23:00Z", "event": "action", "step": 3, "action": ask},
            {"at": "2026-11-23T08:24:00Z", "event": "action", "step": 4, "action": ask},
            {"at": "2026-11-23T08:25:00Z", "event": "answer", "point": "handback_failed", "choice": "stop",
             "by": "alice", "quote": "<停>", "ref": "d-0001", "effect": "the run is stopped"},
            {"at": "2026-11-23T08:26:00Z", "event": "action", "step": 5, "action": {"type": "dispatch"}},
            "not json at all", "{broken", "[1, 2]")
        events = section(pages(self.ws)[EVENTS], "events")
        self.assertIn("派出 rsi-train", events)
        self.assertIn("退回 d-0001（第 1 次", events)
        self.assertIn("退回 d-0001（第 2 次", events)
        self.assertIn("决策 handback_refused → resume", events)
        self.assertIn("代码", events)
        self.assertIn("默认 &lt;原因&gt;", events)
        self.assertIn("决策 agent", events)
        self.assertIn("用户 alice", events)
        self.assertIn("&lt;停&gt;", events)
        self.assertNotIn("<停>", events)
        self.assertEqual(events.count("向用户提问：要不要关闭本轮？"), 1)  # the same ask repeated: one line
        self.assertIn("task-0001", events)  # registrations are still there

    def test_rounds_page_lists_the_dispatches(self):
        self.sets()
        self.dispatch("train", returns=2, status="running", purpose="train")
        self.dispatch("data", status="done", purpose="sets", session=None)
        block = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("d-0001", block)
        self.assertIn("rsi-train", block)
        self.assertIn("3f2a9c1e", block)
        self.assertNotIn(SESSION, block)
        self.assertIn("退回 2 次", block)
        self.assertIn("运行中", block)
        self.assertIn("d-0002", block)
        self.assertIn("完成", block)
        self.assertIn("sets", block)


class TestSim2Pages(EngineStateCase):
    """Page fixes from the second simulation (P3, P5, P6, P7, P8, P9, P15) and the engine's new statuses and
    run-log events."""

    LAUNCHED = "2026-11-23T08:30:00Z"

    def label(self, page=None):
        """The status chip of the top bar."""
        page = self.overview() if page is None else page
        return re.search(r'<span class="chip">第 \d+ 轮</span>(<span class="chip[^"]*">[^<]*</span>)',
                         page).group(1)

    def hold_lock(self, pid=None):
        """Write <ws>/work/run.lock as a running `rsi run` does (its pid; default: this live test process)."""
        path = os.path.join(self.ws, "work", "run.lock")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("%d\n" % (os.getpid() if pid is None else pid))
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

    def dead_pid(self):
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        return p.pid

    # ---- P5 / review: a pending record is not a running agent ---------------------------------------

    def test_a_pending_record_is_shown_as_not_yet_launched(self):
        self.sets()
        self.dispatch("train", launched_at=self.LAUNCHED)
        self.dispatch("deliver", status="pending", session=None, runner="mock", purpose="deliver")
        ws = Workspace(self.ws)
        progress = D.module_progress(ws)
        self.assertEqual(progress["train"]["state"], "running")
        self.assertEqual(progress["deliver"]["state"], "pending")
        self.assertEqual([a["module"] for a in D.running_agents(ws)], ["train"])
        st = self.assertFlow(train="current", deliver="current")
        self.assertIn("● 运行中", st["train"]["text"])
        self.assertEqual(st["deliver"]["text"], "● 待启动")
        lead, detail = self.now_parts()
        self.assertEqual(lead, "② 训练调优 / ③ 交付：rsi-train 运行中，rsi-deliver 待启动")
        self.assertIn("rsi-deliver 待启动", detail)
        now = section(self.overview(), "now")
        self.assertIn("d-0002 待启动", now)
        self.assertNotIn("rsi-deliver 运行中", now)
        block = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("待启动", block)
        self.assertIn("rsi-deliver 待启动", block)

    # ---- the user's answer on its way back to the asking session ------------------------------------

    def asked(self, module="data", answered=False, **fields):
        rec = dict(status="handed_back", purpose="sets" if module == "data" else module, launches=1,
                   launched_at=self.LAUNCHED, asked={"launch": 1, "question_id": "q-0001", "questions": ["删不删那 12 封？"]})
        rec.update(fields)
        did = self.dispatch(module, **rec)
        ws = Workspace(self.ws)
        if answered:
            ws.add_decision({"point": "handback", "choice": "已回答", "by": "alice", "quote": "删掉那 12 封",
                             "dispatch_id": did, "launches": 1, "question_id": "q-0001"})
        else:
            ws.set_waiting({"kind": "question", "what": "handback", "ref": did, "options": ["已回答"],
                            "question": "rsi-%s（%s）交回时有要你决定的事：删不删那 12 封？" % (module, did)})
        return did

    def test_a_held_handback_and_the_answer_on_its_way_back(self):
        # The rounds page names a held hand-back and the launches that are no returns.
        self.asked("data")
        self.dispatch("tune", status="done", purpose="tune", returns=1, followups=1, answered=1, continued=2)
        block = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("交回未登记（等你回答它的提问）", block)
        self.assertIn("退回 1 次 · 继续 1 次 · 长任务收尾 2 次 · 送回回答 1 次", block)
        # Answered (rsi answer), not sent back yet: the pages say the answer is on its way back.
        self.use_task(fixtures.binary_task(), "ws-answered")
        did = self.asked("data", answered=True)
        block = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("交回未登记（已回答，等 rsi run 送回）", block)
        self.assertNotIn("等你回答它的提问", block)
        a = D.next_action(Workspace(self.ws))
        self.assertEqual((a["type"], a.get("mode")), ("resume", "answer"))
        lead, detail = self.now_parts()
        self.assertEqual(lead, "① 数据：把你的回答送回 rsi-data")
        self.assertIn("把你的回答送回 rsi-data 的同一个会话（%s）" % did, detail)
        self.assertIn("那时才登记", detail)
        self.assertNotIn("起运行", detail)
        self.assertIn("送回", self.waiting())
        self.assertEqual(self.assertFlow(data="current")["data"]["text"], "● 送回你的回答")

    def long_job_asked(self, answered=False):
        """train's long job ended (marker ok) and its hand-back asks something."""
        self.sets()
        marker = os.path.join(self.root, "train.marker")
        with open(marker, "w") as f:
            f.write("ok\n")
        return self.asked("train", answered=answered, status="long_job", marker=marker)

    def test_a_long_jobs_question_and_its_answer(self):
        # M2 (final review): a long job's hand-back may ask too (asked once its marker is there); the answer goes
        # back to the session with the long job's results.
        did = self.long_job_asked()
        st = self.assertFlow(train="wait")
        self.assertEqual(st["train"]["text"], "⏸ 等你回答")
        self.assertIn("交回未登记（等你回答它的提问）", round_block(pages(self.ws)[ROUNDS], 1))
        self.use_task(fixtures.binary_task(), "ws-long-answered")
        did = self.long_job_asked(answered=True)
        a = D.next_action(Workspace(self.ws))
        self.assertEqual((a["type"], a["dispatch_id"]), ("process_handback", did))  # the engine sends the answer
        lead, detail = self.now_parts()
        self.assertEqual(lead, "② 训练调优：把你的回答送回 rsi-train")
        self.assertIn("长任务已结束，把你的回答送回 rsi-train 的同一个会话（%s）" % did, detail)
        self.assertNotIn("等校验、登记", detail)
        self.assertIn("交回未登记（已回答，等 rsi run 送回）", round_block(pages(self.ws)[ROUNDS], 1))
        self.assertEqual(self.assertFlow(train="current")["train"]["text"], "● 送回你的回答")

    def test_after_a_stop_the_pages_say_what_still_runs_and_what_waited(self):
        # I2, M1, M7 (final review): the rounds page and the agents list agree with the diagram after a stop.
        self.dispatch("train", launched_at=self.LAUNCHED)
        self.dispatch("deliver", status="pending", session=None, runner="mock", purpose="deliver")
        self.asked("data", launches=1)  # a held hand-back: registration waits for the user's answer
        Workspace(self.ws).clear_waiting()
        Workspace(self.ws).stop("alice", "stop now")
        block = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("rsi-train 停止后仍在运行", block)
        self.assertIn("停止时未启动", block)
        self.assertIn("交回未登记（停止时还在等你回答）", block)
        self.assertNotIn("停止时未交回", block)
        now = section(self.overview(), "now")
        self.assertIn("d-0002 停止时未启动（rsi run 关闭它，不再启动）", now)
        self.assertNotIn("待启动", now)
        st = self.assertFlow(train="current", deliver="skip", data="skip")
        self.assertEqual(st["train"]["text"], "● 停止后仍在运行")
        self.assertEqual(st["deliver"]["text"], "停止时未启动")
        self.assertEqual(st["data"]["text"], "停止时还在等你回答")
        # a hand-back that did not wait for an answer: handed back, not registered, and the run has stopped
        self.use_task(fixtures.binary_task(), "ws-stopped-handback")
        self.sets()
        self.dispatch("deliver", status="handed_back", purpose="deliver", launched_at=self.LAUNCHED)
        Workspace(self.ws).stop("alice", "stop now")
        self.assertIn("交回未登记（运行已停止）", round_block(pages(self.ws)[ROUNDS], 1))
        self.assertEqual(self.assertFlow(deliver="skip")["deliver"]["text"], "交回未登记")

    def test_the_leads_name_rsi_run_for_its_own_steps(self):
        # M6 (final review): with the engine's records the overview's lead says rsi run does promote, close_round and
        # next; the manual flow keeps the commands.
        self.sets()
        self.dispatch("data", status="done", purpose="sets")
        ev = self.chain()
        self.assertEqual(self.now_parts()[0], "顶层核对：rsi run 判定本轮（升级或保留）")
        Workspace(self.ws).promote(ev)
        self.plan(ev)
        self.assertEqual(self.now_parts()[0], "rsi run 进入第 2 轮")
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-lead-close")
        self.sets()
        self.dispatch("data", status="done", purpose="sets")
        self.failed_search()
        self.assertEqual(self.now_parts()[0], "rsi run 关闭本轮（rsi close-round 的效果）")
        task = fixtures.binary_task()
        task["stop"]["target"] = 0.8
        self.use_task(task, "ws-lead-stop")
        self.sets()
        self.dispatch("data", status="done", purpose="sets")
        ev = self.chain()
        Workspace(self.ws).promote(ev)
        Workspace(self.ws).add_decision({"point": "milestone", "options": ["yes", "no"], "choice": "no", "by": "alice"})
        self.assertEqual(self.now_parts()[0], "停止条件已满足：rsi run 结束运行")
        self.use_task(fixtures.binary_task(), "ws-lead-manual")  # no records: the commands stay named
        self.sets()
        self.chain()
        self.assertEqual(self.now_parts()[0], "顶层核对：等 rsi promote 判定本轮")

    # ---- P6: a module's question pauses that module's node only ------------------------------------------

    def test_a_modules_question_pauses_only_the_asking_module(self):
        self.sets()
        self.dispatch("train", launched_at=self.LAUNCHED)
        self.asked("data")  # ① asks: only ① waits
        st = self.assertFlow(data="wait", train="current")
        self.assertEqual(st["data"]["text"], "⏸ 等你回答")
        self.assertIn("● 运行中", st["train"]["text"])
        self.assertNotIn("⏸", st["train"]["text"])

    # ---- P9: "● 运行中" only while an rsi run holds run.lock -----------------------------------------------

    def test_the_running_label_needs_a_live_rsi_run(self):
        from rsi import engine
        self.assertEqual(D.RUN_LOCK, engine.RUN_LOCK)
        self.sets()
        self.assertEqual(self.label(), '<span class="chip idle">等待 rsi run</span>')
        self.assertFalse(D.run_active(Workspace(self.ws)))
        self.hold_lock()
        self.assertTrue(D.run_active(Workspace(self.ws)))
        for rel, text in pages(self.ws).items():
            self.assertEqual(self.label(text), '<span class="chip run">● 运行中</span>', rel)
        self.hold_lock(self.dead_pid())  # a killed run leaves its file behind: no run
        self.assertFalse(D.run_active(Workspace(self.ws)))
        self.assertEqual(self.label(), '<span class="chip idle">等待 rsi run</span>')
        # Without a live run the label names what the run waits for.
        self.use_task(fixtures.binary_task(), "ws-long-job")
        self.sets()
        self.dispatch("train", status="long_job", marker=os.path.join(self.root, "never"))
        self.assertEqual(self.label(), '<span class="chip idle">等长任务完成</span>')
        self.use_task(fixtures.binary_task(), "ws-manual")
        self.sets()
        self.dispatch("train", runner="manual", launched_at=self.LAUNCHED)
        self.assertEqual(self.label(), '<span class="chip idle">等手动交回</span>')
        self.hold_lock()  # a run that works: running, whatever it waits for next
        self.assertEqual(self.label(), '<span class="chip run">● 运行中</span>')

    # ---- P3: stopped while agents or long jobs still run ---------------------------------------------------

    def test_a_stop_with_agents_still_running(self):
        self.sets()
        self.dispatch("train", launched_at=self.LAUNCHED)
        self.dispatch("tune", status="long_job", purpose="tune", marker=os.path.join(self.root, "never"),
                      launched_at=self.LAUNCHED)
        Workspace(self.ws).stop("alice", "stop now")
        lead, detail = self.now_parts()
        self.assertEqual(lead, "已停止，还有 1 个 agent 和 1 个长任务在运行")
        self.assertIn("运行 rsi run 终止它们并写报告", detail)
        self.assertIn("停止原因：user requested: stop now", detail)
        self.assertIn("运行 rsi run 终止它们并写报告", self.waiting())
        now = section(self.overview(), "now")
        self.assertIn("rsi-train", now)  # which ones
        self.assertIn("长任务等待标记", now)
        st = self.assertFlow(train="current", stop="current")
        self.assertEqual(st["train"]["text"], "● 停止后仍在运行")
        self.hold_lock()  # the run that holds the lock is ending them itself
        lead, detail = self.now_parts()
        self.assertIn("rsi run 正在终止它们并写报告", detail)
        self.assertNotIn("运行 rsi run", detail)
        ws = Workspace(self.ws)
        ws.update_dispatch("d-0002", status="done", closed="运行已停止")
        self.assertIn("rsi run 正在终止它并写报告", self.now_parts()[1])  # one left

    def test_a_stop_reports_the_long_jobs_it_ended_and_left(self):
        from rsi import live as L
        self.sets()
        a = self.dispatch("train", status="long_job", marker=os.path.join(self.root, "m1"), launched_at=self.LAUNCHED)
        b = self.dispatch("tune", status="long_job", marker=os.path.join(self.root, "m2"), purpose="tune",
                          launched_at=self.LAUNCHED)
        # Final review B-3: a failed check-in whose long job runs on, and a close whose termination is still pending,
        # are long jobs the report step ends too; the status line says the run has stopped.
        c = self.dispatch("train", status="failed", marker=os.path.join(self.root, "m3"), launched_at=self.LAUNCHED)
        d = self.dispatch("train", status="done", marker=os.path.join(self.root, "m4"), launched_at=self.LAUNCHED,
                          long_job_stop={"result": "pending", "context": "close", "at": self.LAUNCHED})
        ws = Workspace(self.ws)
        ws.stop("alice", "stop now")
        lead, detail = self.now_parts()
        self.assertEqual(lead, "已停止，还有 4 个长任务在运行")
        self.assertRegex(L.now_line(Workspace(self.ws)), r"^\d\d:\d\d 第1轮 已停止，还有 4 个长任务在运行 ｜ ")
        ws.update_dispatch(a, status="done", closed="运行已停止", long_job_stop={
            "pid": 4242, "pid_file": "x", "result": "terminated", "why": "已终止", "at": self.LAUNCHED})
        ws.update_dispatch(b, status="done", closed="运行已停止", long_job_stop={
            "pid": 4343, "pid_file": "y", "result": "unconfirmed", "why": "启动时间对不上", "at": self.LAUNCHED})
        ws.update_dispatch(c, status="done", closed="运行已停止", long_job_stop={
            "pid": 4141, "pid_file": "w", "result": "gone", "why": "进程已不在", "at": self.LAUNCHED, "context": "stop"})
        ws.update_dispatch(d, long_job_stop={"pid": 4444, "pid_file": "z", "result": "terminated", "why": "已终止",
                                             "at": self.LAUNCHED, "context": "close"})
        lead, detail = self.now_parts()
        self.assertEqual(lead, "已停止：user requested: stop now")
        self.assertIn("运行 rsi run 写运行报告", detail)
        with open(os.path.join(self.ws, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# 报告\n")
        lead, detail = self.now_parts()
        self.assertIn('href="dashboard/report.html"', detail)
        # B-7: what the stop ended apart from what an earlier close ended.
        self.assertIn("停止时终止了长任务 d-0001（rsi-train，pid 4242）。关闭派出时终止了长任务 d-0004（rsi-train，pid 4444）",
                      detail)
        self.assertIn("没能终止的长任务（可能仍在运行，请手动结束）：d-0002（rsi-tune，pid 4343）：启动时间对不上", detail)
        self.assertIn("没能终止的长任务", self.waiting())

    # ---- P7: the evalset of the best score, and the score on the current evalset after a refresh ------------

    def refreshed_keep(self):
        """V1 on evalset-0001 (auc 1); the evalset refreshed in round 2; V1 re-delivered on evalset-0002 (one row
        wrong); round 2's candidate is worse there: keep. Returns (ev1, redelivery, ev2)."""
        task = fixtures.binary_task()
        task["eval_use_limit"] = 1
        task["stop"]["target"] = 2  # never reached
        self.use_task(task, "ws-refresh-best")
        ev1 = self.promoted_round()
        plan = self.plan(ev1)
        self.assertEqual(Workspace(self.ws).next_round()[0], 2)
        cand2 = self.candidate(2)
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh")
        self.reg("evalset", 2, ids=range(60, 64))
        best = section(self.overview(), "best")
        self.assertIn("evalset-0001", best)
        self.assertIn("当前评测集 evalset-0002 上还没有它的成绩", best)
        v1 = Workspace(self.ws).version("V1")["candidate_id"]
        red = self.reg("delivery", 2, ids=range(60, 64), val_ids=VAL_IDS, candidate_id=v1, evalset_id="evalset-0002",
                       wrong=(60,))
        d2 = self.reg("delivery", 2, ids=range(60, 64), val_ids=VAL_IDS, candidate_id=cand2,
                      evalset_id="evalset-0002", wrong=(60, 61))
        ev2 = self.reg("evaluation", 2, ids=range(60, 64), train_ids=TRAIN_IDS, delivery_id=d2, candidate_id=cand2,
                       evalset_id="evalset-0002", best_before={"version": "V1", "evaluation_id": ev1},
                       wrong=(60, 61), best_wrong=(60,), parents=[red])
        self.assertEqual(Workspace(self.ws).promote(ev2)[0], "keep")
        return ev1, red, ev2, plan

    def test_the_best_card_names_the_evalset_and_the_current_score(self):
        self.promoted_round()
        best = section(self.overview(), "best")
        self.assertIn("auc（评测集 evalset-0001，越高越好）", best)
        self.assertNotIn("当前评测集", best)
        ev1, red, ev2, _plan = self.refreshed_keep()
        ws = Workspace(self.ws)
        now_value = ws.manifest(ev2)["metrics"]["best"]["auc"]
        self.assertLess(now_value, 1)
        best = section(self.overview(), "best")
        self.assertIn("V1 · 1", best)
        self.assertIn("auc（评测集 evalset-0001，越高越好）", best)
        self.assertIn("在当前评测集 evalset-0002 上：%s" % D._num(now_value), best)
        self.assertIn(ev2, best)
        self.assertIn(red, best)
        self.assertIn("目标仍按 evalset-0001 上的成绩判断", best)
        from rsi.report import report_markdown  # the report says the same
        text = report_markdown(ws)
        self.assertIn("当前最佳：V1，auc 1（evaluation-0001），在评测集 evalset-0001 上", text)
        self.assertIn("- 在当前评测集 evalset-0002 上：auc %s（%s，用重新交付的 %s 算）"
                      % (D._num(ws.manifest(ev2)["metrics"]["best"]["auc"]), ev2, red), text)

    # ---- P15 and the engine's new run-log events --------------------------------------------------------

    def test_steps_rsi_run_did_are_credited_to_it(self):
        ev = self.promoted_round()
        self.log({"at": "2026-11-23T08:00:00Z", "event": "action", "step": 1,
                  "action": {"type": "promote", "evaluation_id": ev, "step": "promote"}})
        events = section(pages(self.ws)[EVENTS], "events")
        self.assertIn("第 1 轮判定：升级 V1", events)
        self.assertIn("· rsi run</span>", events)
        self.assertNotIn("· rsi promote", events)
        self.assertNotIn("· 引擎", events)
        # a round closed and a stop by the stop condition, both carried out by rsi run
        self.use_task(fixtures.binary_task(), "ws-closed-by-run")
        self.sets()
        ws = Workspace(self.ws)
        reason = "第 1 轮的 3 次尝试已用完，没有候选通过训练门"
        self.log({"at": "2026-11-23T08:00:00Z", "event": "action", "step": 1,
                  "action": {"type": "close_round", "reason": reason, "step": "close"}})
        ws.close_round(reason)
        events = section(pages(self.ws)[EVENTS], "events")
        self.assertIn("第 1 轮判定：关闭（没有评测）", events)
        self.assertIn("· rsi run</span>", events)
        self.assertNotIn("· rsi close-round", events)
        # without the run log (a round closed by hand) the command stays named
        self.use_task(fixtures.binary_task(), "ws-closed-by-hand")
        self.sets()
        Workspace(self.ws).close_round("by hand")
        self.assertIn("· rsi close-round", section(pages(self.ws)[EVENTS], "events"))

    def test_the_engines_new_events_are_listed(self):
        self.sets()
        did = self.dispatch("tune", status="done", purpose="tune")
        at = "2026-11-23T08:%02d:00Z"
        self.log(
            {"at": at % 0, "event": "launch", "dispatch_id": did, "module": "tune", "launch": 1, "mode": "first",
             "runner": "mock", "session": SESSION},
            {"at": at % 1, "event": "followup", "dispatch_id": did, "mode": "followup", "launch": 2,
             "session": SESSION, "feedback": "feedback-2.md"},
            {"at": at % 2, "event": "resume", "dispatch_id": did, "mode": "continue", "launch": 3,
             "session": SESSION, "feedback": "feedback-3.md"},
            {"at": at % 3, "event": "resume", "dispatch_id": did, "mode": "resume", "launch": 4,
             "session": SESSION, "feedback": "feedback-4.md"},
            {"at": at % 4, "event": "asked", "dispatch_id": did, "launch": 4, "question_id": "q-0001",
             "held": ["trial-a", "trial-b"]},
            {"at": at % 5, "event": "resume_with_answer", "dispatch_id": did, "mode": "answer", "launch": 5,
             "session": SESSION, "feedback": "feedback-5.md"},
            {"at": at % 6, "event": "agent_stopped", "dispatch_id": did, "why": "stopped", "pid": 99},
            {"at": at % 7, "event": "long_job_stopped", "dispatch_id": did, "pid": 4242, "result": "terminated",
             "why": "进程 4242 是这个长任务，已终止", "pid_file": "/m.pid"},
            {"at": at % 8, "event": "long_job_gone", "dispatch_id": did, "pid": 4343, "result": "gone",
             "why": "进程 4343 已不在", "pid_file": "/m.pid"},
            {"at": at % 9, "event": "long_job_left", "dispatch_id": did, "pid": None, "result": "no_pid_file",
             "why": "没有 pid 文件", "pid_file": "/m.pid"},
            {"at": at % 10, "event": "closed", "dispatch_id": did, "why": "stopped", "launched": False},
            {"at": at % 11, "event": "long_job_stopped", "dispatch_id": did, "pid": 4444, "result": "terminated",
             "why": "已终止", "pid_file": "/n.pid", "context": "close", "orphan": False},
            {"at": at % 12, "event": "long_job_gone", "dispatch_id": did, "pid": 4545, "result": "gone",
             "why": "进程 4545 已不在", "pid_file": "/old.pid", "context": "stop", "orphan": True})
        events = section(pages(self.ws)[EVENTS], "events")
        self.assertIn("继续 %s（交回的第二部分，不算退回，会话 3f2a9c1e，feedback-2.md）" % did, events)
        self.assertIn("续跑 %s（长任务收尾" % did, events)
        self.assertIn("退回 %s（第 1 次" % did, events)  # the second part and the continuation are no returns
        self.assertNotIn("第 2 次", events)
        self.assertIn("rsi-tune（%s）交回时提问：等你回答，交回的 trial-a、trial-b 暂不登记" % did, events)
        self.assertIn("把你的回答送回 %s（会话 3f2a9c1e，feedback-5.md）" % did, events)
        self.assertIn("停止时终止了 agent %s" % did, events)
        self.assertIn("停止时终止了长任务 %s（pid 4242）" % did, events)
        self.assertIn("停止时长任务 %s 的进程已不在（pid 4343）" % did, events)
        self.assertIn("停止时没能终止长任务 %s：没有 pid 文件" % did, events)
        self.assertIn("关闭派出 %s（运行已停止，没有启动）" % did, events)
        self.assertIn("关闭派出时终止了长任务 %s（pid 4444）" % did, events)  # a decision or close-round closed it
        self.assertNotIn("4545", events)  # an orphan pid file whose process had ended: no line



class TestNowLine(EngineStateCase):
    """The status line `rsi run` prints and `rsi status` starts with (rsi.live.now_line, spec §6)."""

    def test_now_line_forms(self):
        from rsi import live as L
        task = fixtures.binary_task()
        task["flow"] = {"checkin_minutes": 30}
        self.use_task(task, "ws-now")
        now = datetime.datetime(2026, 11, 24, 20, 23).astimezone()
        ago = lambda minutes: utc((now - datetime.timedelta(minutes=minutes)).astimezone(datetime.timezone.utc))  # noqa: E731
        outdir = os.path.join(self.ws, "work", "round-1", "train")
        os.makedirs(outdir)
        did = self.dispatch("train", status="long_job", launched_at=ago(20), ended_at=ago(7),
                            marker=os.path.join(outdir, "batch-3.done"))
        line = lambda: L.now_line(Workspace(self.ws), now)  # noqa: E731
        self.assertEqual(line(), "20:23 第1轮 ② 训练调优 d-0001 · 长任务在跑，等完成标记 batch-3.done · 下次检查 20:46")

        def progress(updated_at):
            with open(os.path.join(outdir, "progress.json"), "w", encoding="utf-8") as f:
                json.dump({"version": 1, "updated_at": updated_at, "summary": "第 3 批：3 个训练在跑（卡 0、4、6）",
                           "facts": [{"label": "配置", "value": "3/3"}, {"label": "GPU 小时", "value": "1.3/8"},
                                     {"label": "当前方案", "value": "c43"}, {"label": "第四个", "value": "x"}]}, f)

        progress(ago(1))
        self.assertEqual(line(), "20:23 第1轮 ② 训练调优 d-0001 · 第 3 批：3 个训练在跑（卡 0、4、6） · 配置 3/3 · "
                                 "GPU 小时 1.3/8 · 当前方案 c43 · 下次检查 20:46")
        progress(ago(10))  # the dispatch is alive and its progress is older than PROGRESS_STALE_SECONDS
        self.assertIn("第 3 批：3 个训练在跑（卡 0、4、6）（进度 10 分钟没有更新） · 配置 3/3", line())
        # R5: a question STATE waits on that is tied to the dispatch (ref or dispatch_id, any what) comes before its
        # progress; its facts may follow, the next check-in does not.
        facts = " · 配置 3/3 · GPU 小时 1.3/8 · 当前方案 c43"
        # The check-in cap's question is live only while a check-in is due by the real clock (rsi.flow.checkin_moot):
        # its last hand-back a day back, so it is due in every time zone.
        asks = (({"status": "handed_back"}, {"kind": "question", "what": "handback", "ref": did, "options": ["已回答"]}),
                ({"status": "handed_back"}, {"kind": "approval", "what": "over_budget", "ref": None, "dispatch_id": did,
                                             "options": ["approve", "stop"]}),
                ({"status": "long_job", "checkins": 48, "ended_at": ago(60 * 24)},  # the check-in cap (48)
                 {"kind": "question", "what": "checkin", "ref": did, "options": ["continue", "wait", "stop"]}))
        for n, (fields, w) in enumerate(asks, 1):
            Workspace(self.ws).update_dispatch(did, **fields)
            Workspace(self.ws).set_waiting(dict(w, question="要你决定的事"))
            self.assertEqual(line(), "20:23 第1轮 ② 训练调优 d-0001 · 等你回答（q-%04d）%s" % (n, facts), w["what"])
            Workspace(self.ws).clear_waiting()
        os.remove(os.path.join(outdir, "progress.json"))
        ws = Workspace(self.ws)
        for fields, want in (({"status": "pending"}, "待启动"),
                             ({"status": "running", "launches": 9, "launch_mode": "checkin"},
                              "rsi-train 运行中（第 9 次启动 · 定时检查）"),
                             ({"status": "long_job"}, "长任务已完成，等 rsi run 收尾"),  # its marker is there
                             ({"status": "handed_back"}, "已交回，等登记")):
            ws.update_dispatch(did, **fields)
            if fields["status"] == "long_job":
                with open(os.path.join(outdir, "batch-3.done"), "w") as f:
                    f.write("ok\n")
            self.assertEqual(line(), "20:23 第1轮 ② 训练调优 d-0001 · %s" % want, fields)
        self.dispatch("data", status="pending", outdir="work/round-1/data")
        self.assertEqual(line().split(" ｜ "), ["20:23 第1轮 ② 训练调优 d-0001 · 已交回，等登记", "① 数据 d-0002 · 待启动"])
        for d in ("d-0001", "d-0002"):
            Workspace(self.ws).update_dispatch(d, status="done")
        # No open dispatch: the question STATE waits on (one without a ref), the notes' questions (not the page's
        # 「上方」), else the overview's lead.
        Workspace(self.ws).set_waiting({"kind": "approval", "what": "refresh_sets", "ref": None, "question": "刷新？",
                                        "options": ["approve", "close_round", "stop"]})
        self.assertEqual(line(), "20:23 第1轮 等你回答（q-0004）")
        Workspace(self.ws).clear_waiting()
        notes = os.path.join(self.ws, "work", "notes.md")
        with open(notes, "w", encoding="utf-8") as f:
            f.write("## 待用户回答\n- 要不要多用一张卡？\n")
        self.assertEqual(line(), "20:23 第1轮 等你在 Claude 里回答 1 个问题（进度页面上列着）")
        os.remove(notes)
        self.assertEqual(line(), "20:23 第1轮 " + D.now_lead(Workspace(self.ws)))


class TestLivePanels(EngineStateCase):
    """The overview's live panels (spec §4, §5): 现在在做什么, agent 的判断, GPU, 最近事件, and the cards' new lines."""

    NOW = datetime.datetime(2026, 11, 24, 20, 23, tzinfo=datetime.timezone.utc)

    def setUp(self):
        super().setUp()
        task = fixtures.binary_task()
        task["flow"] = {"checkin_minutes": 30}
        task["resources"] = {"gpu_ids": [0, 2, 5]}
        self.use_task(task, "ws-live")
        self.out = os.path.join(self.ws, "work", "round-1", "train")
        os.makedirs(self.out)

    def ago(self, seconds):
        return utc(self.NOW - datetime.timedelta(seconds=seconds))

    def long_job(self, **fields):
        return self.dispatch("train", status="long_job", launched_at=self.ago(1200), ended_at=self.ago(420),
                             marker=os.path.join(self.out, "batch-3.done"), **fields)

    def write(self, name, doc, age=None):
        """Write `doc` (JSON, or text as is) to <outdir>/<name>; `age`: its modification time that many seconds
        before NOW."""
        path = os.path.join(self.out, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False))
        if age is not None:
            t = (self.NOW - datetime.timedelta(seconds=age)).timestamp()
            os.utime(path, (t, t))
        return path

    def ov(self):
        return pages(self.ws, self.NOW)[OVERVIEW]

    def test_doing_panel_shows_the_modules_progress(self):
        self.assertIn("现在没有在跑的模块", section(self.ov(), "doing"))
        self.long_job()
        self.write("progress.json", {
            "version": 1, "updated_at": self.ago(60), "summary": "第 3 批：3 个训练在跑",
            "facts": [{"label": "配置", "value": "3/3"}, {"label": "GPU 小时", "value": "1.3/8"},
                      {"label": "当前方案", "value": "c43"}],
            "table": {"title": "本次搜索的试验", "columns": ["试验", "验证集 auc"],
                      "rows": [["c43-s1", 0.61], ["c43-s2", None]]},
            "notes": ["next：补 seed 1、2"]})
        ov = self.ov()
        doing = section(ov, "doing")
        for text in ("第 3 批：3 个训练在跑", "配置", "3/3", "验证集 auc", "c43-s1", "next：补 seed 1、2", "1 分钟前",
                     "定时检查 0 / 48", "完成标记"):
            self.assertIn(text, doing)
        self.assertIn("配置 3/3", re.search(r'<g id="n-train".*?</g>', ov, re.S).group(0))

    def test_stale_and_broken_progress_never_break_the_page(self):
        did = self.long_job()
        stale = {"updated_at": self.ago(600), "summary": "第 3 批"}
        self.write("progress.json", stale)  # (a) the job is alive, its marker not there
        self.assertIn("10 分钟没有更新", section(self.ov(), "doing"))
        self.write("progress.json", "{not json")  # (b)
        doing = section(self.ov(), "doing")
        self.assertIn("progress.json 不是合法的 JSON", doing)
        self.assertNotIn("Traceback", doing)
        self.write("progress.json", stale)  # (c) handed back: not alive, never stale
        Workspace(self.ws).update_dispatch(did, status="handed_back")
        self.assertNotIn("没有更新", section(self.ov(), "doing"))

    def test_judgement_panel_lists_latest_notes_and_questions(self):
        from rsi import live as L
        self.long_job(launches=2)
        self.write("handback-1.json", {"status": "long_job", "notes": "first", "questions": []}, age=900)
        questions = ["要不要放宽门槛？", "问" * 600] + ["q%d" % i for i in range(5)]  # 7: the first 5, each clipped
        self.write("handback.json", {"status": "ok", "notes": "长" * 300 + "余" * 100, "questions": questions}, age=60)
        j = section(self.ov(), "judgement")
        self.assertLess(j.index("要不要放宽门槛？"), j.index("first"))  # the newer first
        self.assertRegex(j, r'长{300}…</div><details[^>]*><summary>[^<]*</summary><div class="hbn">余{100}</div>'
                            r'</details>')  # only the rest inside <details>
        self.assertIn("问" * 500 + "…<", j)
        self.assertNotIn("问" * 501, j)
        self.assertIn("q2", j)
        self.assertNotIn("q3", j)
        self.assertIn("另有 2 条", j)
        for n in range(2, 8):  # 8 hand-backs (one unreadable, skipped): the latest 6
            self.write("handback-%d.json" % n, "{bad" if n == 7 else {"status": "ok", "notes": "n%d" % n}, age=800 - n)
        ws = Workspace(self.ws)
        notes = L.handback_notes(ws)
        self.assertEqual([h["notes"][:2] for h in notes], ["长长", "n6", "n5", "n4", "n3", "n2"])
        self.assertEqual((notes[0]["dispatch_id"], notes[0]["module"], notes[0]["launch"]), ("d-0001", "train", 2))
        self.assertEqual(section(self.ov(), "judgement").count('<li class="hb"'), 6)

    def test_gpu_panel_rules(self):
        from rsi import gpus as G
        cards = [{"index": i, "uuid": "GPU-%d" % i, "name": "A100", "memory_used_mib": 20480,
                  "memory_total_mib": 81920, "utilization_pct": 50, "processes": []} for i in range(3)]
        cards[0]["processes"] = [{"pid": 11, "memory_mib": 10240, "owner": "ours", "user": "me"}]
        cards[2]["processes"] = [{"pid": 12, "memory_mib": 8192, "owner": "other", "user": "alice"},
                                 {"pid": 13, "memory_mib": None, "owner": "unknown", "user": None}]

        def snap(status="ok", age=30, error=None, gpus=cards):
            return {"at": self.ago(age), "status": status, "error": error, "uid": 1000, "gpus": gpus}

        cases = (("none", None, ["还没有 GPU 信息", "每 60 秒"], ["卡 0"]),
                 ("broken", "{half", ["GPU 快照读不了"], ["还没有 GPU 信息"]),
                 ("missing", snap("missing", gpus=[]), ["这台机器没有 nvidia-smi"], ["卡 0"]),
                 ("ok", snap(), ["卡 0", "卡 2", "卡 5", "没有这张卡", "本用户", "alice", "未知", "20.0 / 80.0 GB"],
                  ["卡 1", "已过时"]),
                 ("error", snap("error", error="nvidia-smi 超时（5 秒没有返回）", gpus=[]),
                  ["nvidia-smi 超时（5 秒没有返回）"], []),
                 ("old", snap(age=181), ["已过时"], []))
        for name, doc, has, lacks in cases:
            with self.subTest(name):
                if isinstance(doc, str):
                    os.makedirs(os.path.join(self.ws, "work"), exist_ok=True)
                    with open(os.path.join(self.ws, G.GPU_FILE), "w", encoding="utf-8") as f:
                        f.write(doc)
                elif doc is not None:
                    G.write_snapshot(self.ws, doc)
                gp = section(self.ov(), "gpus")
                for text in has:
                    self.assertIn(text, gp)
                for text in lacks:
                    self.assertNotIn(text, gp)

    def test_overview_cards_show_the_round_in_progress(self):
        from rsi import live as L
        self.sets()
        low = self.reg("candidate", ids=VAL_IDS, wrong=(0,), train_ids=TRAIN_IDS)
        high = self.candidate()
        ws = Workspace(self.ws)
        vals = dict((c, ws.manifest(c)["metrics_val"]["auc"]) for c in (low, high))
        self.assertGreater(vals[high], vals[low])
        self.assertEqual(L.round_best_val(ws), (high, vals[high]))
        self.dispatch("train")
        self.log(*[{"at": self.ago(60 * (12 - i)), "event": "launch", "mode": "first", "dispatch_id": "d-%04d" % i,
                    "module": "data", "runner": "claude"} for i in range(12)])
        ov = self.ov()
        best = section(ov, "best")
        self.assertIn("本轮验证集最好", best)
        self.assertIn("auc %s" % D._num(vals[high]), best)
        self.assertIn(high, best)
        self.assertIn("（1 次进行中）", section(ov, "stop"))
        recent = section(ov, "recent")
        self.assertEqual(recent.count("<li"), 10)
        self.assertIn('href="dashboard/events.html"', recent)


if __name__ == "__main__":
    unittest.main()
