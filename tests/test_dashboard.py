"""The progress pages (rsi/dashboard.py, rsi/dashboard_pages.py): module output directories, what the
run waits for, the workflow diagram, the overview and the three secondary pages."""
import builtins
import datetime
import json
import os
import re
import stat
import tempfile
import unittest
from unittest import mock

import fixtures
from rsi import dashboard as D
from rsi.workspace import Workspace

# Row ids (q<i>) of the data sets; they never overlap.
TEST_IDS, EVAL_IDS, VAL_IDS, TRAIN_IDS = range(10, 14), range(20, 24), range(0, 4), range(40, 44)
ALLOWED = {"STATE.json", "registry.jsonl", "manifest.json", "task.json", "REPORT.md", "notes.md"}
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

    def two_promoted_rounds(self):
        """Round 1 (one evalset row wrong, tuning trials and their search) promotes V1, round 2 promotes V2."""
        self.sets()
        c1 = self.candidate()
        self.tune_trials()
        d1 = self.reg("delivery", 1, ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=c1, wrong=(20,))
        ev1 = self.reg("evaluation", 1, ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=d1, candidate_id=c1,
                       evalset_id="evalset-0001", best_before=None, wrong=(20,))
        self.assertEqual(Workspace(self.ws).promote(ev1)[:2], ("promote", "V1"))
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

    def waiting(self):
        return D.waiting_for(Workspace(self.ws), D.module_progress(Workspace(self.ws)))

    def overview(self):
        return pages(self.ws)[OVERVIEW]

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
    def test_fresh_workspace(self):
        page = self.overview()
        self.assertIn('<meta http-equiv="refresh" content="5">', page)
        self.assertIn("spam-check", page)
        for sec in ("flow", "best", "stop", "now"):
            self.assertIn('id="%s"' % sec, page)
        self.assertNotIn('id="ask"', page)
        self.assertIn("rsi-data", D.waiting_for(Workspace(self.ws), D.module_progress(Workspace(self.ws))))

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

    def test_module_progress_ignores_strays(self):
        r1 = os.path.join(self.ws, "work", "round-1")
        os.makedirs(os.path.join(r1, "trainer"))
        open(os.path.join(r1, "train.log"), "w").close()
        os.makedirs(os.path.join(r1, "train-2"))
        p = D.module_progress(Workspace(self.ws))
        self.assertEqual(p["train"]["state"], "running")  # train-2 counts, trainer and train.log do not
        self.assertEqual(p["tune"]["state"], "idle")
        self.assertEqual(p["data"]["state"], "idle")

    def test_module_progress_counts_artifact_directories(self):
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


class TestWaitingFor(DashboardCase):
    def test_waiting_for_follows_the_round(self):
        task = fixtures.binary_task()
        task["approvals"] = task["approvals"] + ["plan"]
        self.use_task(task, "ws-plan")
        self.assertIn("rsi-data", self.waiting())
        self.sets()
        w = self.waiting()
        self.assertIn("rsi-train", w)
        self.assertIn("rsi-tune", w)
        cand = self.candidate()
        self.assertIn("rsi-deliver", self.waiting())
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

    def test_waiting_for_deliver_one_candidate(self):
        self.sets()
        cand = self.candidate()
        self.assertIn("（本轮通过训练门的候选：%s）" % cand, self.waiting())


    def test_waiting_for_stop_condition(self):
        # The stop check says stop after the decision: no plan is needed, `rsi next` ends the run.
        self.task["stop"]["target"] = 0.8
        self.use_task(self.task, "ws-target")
        self.promoted_round()
        # The milestone question comes first (its sentence also names `rsi next`); once the user said
        # yes the release is in progress (TestProgressTruth); the stop-condition sentence is what the
        # run waits for when the stop check says stop and no milestone is asked or under way.
        self.assertIn("等用户回答", self.waiting())
        Workspace(self.ws).approve("use_testset", "alice", "run the test inputs")
        self.assertIn("里程碑发布进行中", self.waiting())


class TestFlow(DashboardCase):
    def test_flow_states_follow_the_round(self):
        self.plan_task("ws-flow")
        st = self.assertFlow(task="done", data="current", train="todo", tune="todo", deliver="todo",
                             judge="todo", check="todo", stop="todo")
        self.assertEqual(set(st), {"task", "data", "train", "tune", "deliver", "judge", "check", "stop"})
        self.sets()
        self.assertFlow(data="done", train="current", tune="current")
        cand = self.candidate()
        st = self.assertFlow(data="done", train="done", tune="skip", deliver="current")
        self.assertEqual(st["tune"]["text"], "本轮不经过")
        out = os.path.join(self.ws, "work", "round-1", "deliver")
        os.makedirs(out)  # rsi-deliver dispatched, nothing handed back yet
        st = self.assertFlow(deliver="current", judge="todo")
        self.assertIn("进行中", st["deliver"]["text"])
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
        st = self.assertFlow(task="done", data="done", train="done", tune="skip", deliver="done", judge="done",
                             check="done", stop="current")
        self.assertRegex(st["stop"]["text"], r"\d\d:\d\d")
        self.assertIn("停止", st["stop"]["text"])

    def test_flow_failed_gate(self):
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        st = self.assertFlow(train="fail", tune="current")  # attempts left: ② or ③ again
        self.assertEqual(st["train"]["text"], "门没过 · 可重派")
        self.assertIn("rsi-train 或 rsi-tune", self.waiting())

    def test_flow_attempts_used_up(self):
        # Nothing to deliver; ⑤ is still to come (the judge writes the closed round's plan).
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-flow-close")
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        st = self.assertFlow(train="fail", tune="skip", deliver="skip", judge="todo", check="current")
        self.assertEqual(st["train"]["text"], "门没过")
        Workspace(self.ws).close_round("no candidate passed")
        self.assertFlow(deliver="skip", judge="current", check="done")

    def test_flow_entry_after_data(self):
        # Round 2 enters at train: data is not passed through, even with a refreshed evalset.
        task = fixtures.binary_task()
        task["approvals"] = task["approvals"] + ["refresh_sets"]
        self.use_task(task, "ws-flow-entry")
        self.plan(self.promoted_round())
        self.assertEqual(Workspace(self.ws).next_round(), (2, "train"))
        st = self.assertFlow(data="skip", train="current", tune="current")
        self.assertEqual(st["data"]["text"], "本轮不经过")
        Workspace(self.ws).approve("refresh_sets", "alice", "new evalset")
        self.reg("evalset", 2, ids=range(30, 34))
        self.assertFlow(data="skip")

    def test_stopped_run_shows_no_running_module(self):
        self.sets()
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        st = self.assertFlow(train="current")
        self.assertIn("进行中", st["train"]["text"])
        Workspace(self.ws).stop("alice", "stop now")
        st = self.assertFlow(train="skip", stop="current")
        self.assertIn("停止时未交回", st["train"]["text"])
        self.assertNotIn("进行中", st["train"]["text"])


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
        self.assertNotIn("record-0001", p[OVERVIEW])
        for t in trials:
            self.assertNotIn(t, p[OVERVIEW])
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

    def now_parts(self):
        """(lead, detail) of the now card."""
        now = section(self.overview(), "now")
        return (re.search(r'<div class="wait-now">(.*?)</div>', now).group(1),
                re.search(r'<div class="detail">(.*?)</div>', now).group(1))

    def test_now_card(self):
        self.sets()
        lead, detail = self.now_parts()
        self.assertEqual(lead, "下一步：② 训练 / ③ 调优")
        self.assertTrue(detail.startswith("派 rsi-train 或 rsi-tune（"), detail)  # the action, muted
        self.assertIn("已用尝试 0 / 3", detail)
        self.assertRegex(section(self.overview(), "now"), r"自 \d{4}-\d\d-\d\d \d\d:\d\d:\d\d 起")
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        lead, detail = self.now_parts()
        self.assertEqual(lead, "② 训练：rsi-train 运行中")
        self.assertIn("输出目录自", detail)  # the sentence names when the module started, once
        self.assertEqual(len(re.findall(r"\d\d:\d\d:\d\d", section(self.overview(), "now"))), 1)
        Workspace(self.ws).stop("alice", "stop now")
        lead, detail = self.now_parts()
        self.assertEqual(lead, "已停止：user requested: stop now")
        self.assertEqual(detail, "运行报告由顶层在停止后生成。")  # no link to an empty report page
        self.assertNotIn("dashboard/report.html", section(self.overview(), "now"))
        with open(os.path.join(self.ws, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# 报告\n")
        self.assertIn('href="dashboard/report.html"', self.now_parts()[1])

    def test_now_card_plan_and_close(self):
        self.plan_task("ws-now-plan")
        plan = self.plan(self.promoted_round())
        lead, detail = self.now_parts()
        self.assertEqual(lead, "等你批准计划 %s" % plan)
        self.assertIn("rsi approve", detail)
        self.assertNotIn(section(self.overview(), "ask").split("<li>")[1].split("<")[0], detail)  # not the banner
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-now-close")
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        lead, detail = self.now_parts()
        self.assertEqual(lead, "等关闭本轮（rsi close-round）")
        self.assertIn("rsi close-round &lt;ws&gt; --reason", detail)

    def test_missing_numbers_show_none(self):
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
        trial = 'href="../artifacts/candidate-0002/manifest.json"'  # the first tuning trial of round 1
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

    def test_plan_approval_on_rounds_page(self):
        self.plan_task("ws-plan-cell")
        plan = self.plan(self.promoted_round())
        row = decision_row(round_block(pages(self.ws)[ROUNDS], 1))
        self.assertIn(plan, row)
        self.assertIn("未批准", row)
        Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        row = decision_row(round_block(pages(self.ws)[ROUNDS], 1))
        self.assertIn("已批准", row)
        self.assertNotIn("未批准", row)

    def test_failed_gate_candidate_on_rounds_page(self):
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        r1 = round_block(pages(self.ws)[ROUNDS], 1)
        self.assertIn("训练门没过", r1)
        self.assertIn('class="bad"', r1)
        self.assertNotIn("次尝试已用完", r1)

    def test_attempts_used_up_page_and_wait(self):
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-attempts-page")
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        ws = Workspace(self.ws)
        r1 = round_block(pages(ws)[ROUNDS], 1)
        self.assertIn('<p class="warn">', r1)
        self.assertIn("rsi close-round", r1)
        self.assertIn("次尝试已用完", self.waiting())  # one wording for both
        self.assertEqual([], D.pending_questions(ws, D.module_progress(ws)))
        ws.close_round("no candidate passed")
        self.assertNotIn('class="warn"', round_block(pages(self.ws)[ROUNDS], 1))
        self.assertNotIn("rsi close-round", self.waiting())

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
        self.assertIn("%d 个调优试验候选" % len(trials), events)
        self.assertIn(trials[0], events)
        self.assertIn(trials[-1], events)
        self.assertNotIn(trials[1], events)  # one line for the run of trials
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

    def test_tune_trials_without_search_wait_for_the_search(self):
        # S4
        self.sets()
        self.reg("candidate", ids=VAL_IDS, train_ids=TRAIN_IDS, producer="tune", config_source="tune")
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-tune", w)
        self.assertIn("search", w)
        self.assertNotIn("④", w)
        self.assertEqual(flow["tune"]["state"], "current")
        self.assertNotEqual(flow["deliver"]["state"], "current")
        self.assertIn("rsi-tune", section(self.overview(), "now"))
        self.tune_search_only()
        self.assertIn("rsi-deliver", self.waiting())

    def tune_search_only(self):
        ws = Workspace(self.ws)
        trial = [e["id"] for e in ws.entries("candidate") if e["producer"] == "tune"][0]
        primary = self.task["metrics"]["primary"]
        row = {"candidate_id": trial, "config": {"lr": 0.1}, "compute_hours": 1, "status": "ok",
               "metric_val": ws.manifest(trial)["metrics_val"][primary]}
        self.reg("search", trials=[row], default_candidate=trial, best_candidate=trial,
                 budget={"compute_hours": 10}, gain_vs_default={"metric": primary, "delta": 0.0})

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

    def refresh_task(self, name):
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

    def test_refresh_question_has_since(self):
        # M15: since = the registration time of the evaluation that used up the evalset
        ev = self.refresh_task("ws-refresh-since")
        ws = Workspace(self.ws)
        item = [q for q in D.pending_questions(ws, D.module_progress(ws)) if "refresh_sets" in q["text"]][0]
        self.assertEqual(item["since"], D._utc_iso(ws.entry(ev)["registered_at"]))
        self.assertIn("自 ", section(self.overview(), "ask"))

    def test_redelivery_hint_after_refresh(self):
        # M16
        self.refresh_task("ws-refresh-redeliver")
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        ws = Workspace(self.ws)
        self.assertIsNotNone(ws.best_delivery_on_current_evalset())
        self.assertNotIn("重新交付", self.waiting())
        self.reg("evalset", 2, ids=range(60, 64))  # refreshed evalset: V1 has no delivery on it
        ws = Workspace(self.ws)
        self.assertIsNone(ws.best_delivery_on_current_evalset())
        w = self.waiting()
        self.assertIn("rsi-deliver", w)
        self.assertIn("重新交付", w)
        self.assertIn("V1", w)

    def test_redelivery_of_best_does_not_finish_step_four(self):
        # After a refresh the best version's re-delivery (an earlier round's candidate) is not the round's delivery.
        self.refresh_task("ws-refresh-step4")
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        self.reg("evalset", 2, ids=range(60, 64))
        w, flow = self.waiting_and_flow()
        self.assertIn("rsi-deliver", w)
        self.assertIn("重新交付", w)
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
        # Registering one artifact of a hand-back from its own directory does not supersede its siblings.
        self.sets()
        for i in range(3):
            fixtures.make_artifact("candidate", os.path.join(self.ws, "work", "round-1", "tune", "trial-%d" % i),
                                   self.task, ids=VAL_IDS, train_ids=TRAIN_IDS, producer="tune",
                                   config_source="tune", round=1, notes="t%d" % i)
        Workspace(self.ws).register("candidate", os.path.join(self.ws, "work", "round-1", "tune", "trial-0"))
        p = D.module_progress(Workspace(self.ws))["tune"]
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
        # M17: train passes in one attempt and tune adds trials (attempts 2 / 3 here)
        self.sets()
        self.candidate()
        self.tune_trials(n=2)
        ws = Workspace(self.ws)
        self.assertEqual((len(ws.attempts()), ws.attempt_limit()), (2, 3))
        self.assertNotIn("hot", self.meter("本轮尝试"))
        self.reg("candidate", ids=VAL_IDS, wrong=(1,), train_ids=TRAIN_IDS)  # third attempt, also passing
        self.assertEqual(len(Workspace(self.ws).attempts()), 3)
        self.assertNotIn("hot", self.meter("本轮尝试"))
        task = fixtures.binary_task()
        task["stop"]["max_attempts_per_round"] = 1
        self.use_task(task, "ws-hot")
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        self.assertIn("hot", self.meter("本轮尝试"))

    def meter(self, label):
        stop = section(self.overview(), "stop")
        return re.search(r'<div class="meter[^"]*">(?:(?!<div class="meter).)*?%s' % label, stop, re.S).group(0)


class TestPages(DashboardCase):
    def test_tabs_and_links(self):
        self.promoted_round()
        p = pages(self.ws)
        self.assertEqual(set(p), set(D.PAGES))
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

    def test_reads_only_allowed_files(self):
        self.sets()
        self.chain()
        with open(os.path.join(self.ws, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# 报告\n")
        # Module output in work/ is listed, never read.
        out = fixtures.make_artifact("candidate", os.path.join(self.ws, "work", "round-1", "train", "cand"),
                                     self.task, ids=VAL_IDS)
        with open(os.path.join(self.ws, "work", "notes.md"), "w", encoding="utf-8") as f:
            f.write("repo root\n")
        self.assertTrue(os.path.isfile(os.path.join(out, "predictions_val.jsonl")))
        ws = Workspace(self.ws)
        with mock.patch("builtins.open", wraps=builtins.open) as opened:
            p = pages(ws)
        self.assertEqual(set(p), set(D.PAGES))
        names = set(os.path.basename(str(c[0][0])) for c in opened.call_args_list)
        self.assertIn("REPORT.md", names)
        self.assertLessEqual(names, ALLOWED)

    def test_render_writes_nothing(self):
        self.sets()
        self.chain()
        os.makedirs(os.path.join(self.ws, "work", "round-1", "judge"))

        def snapshot():
            out = []
            for dirpath, dirs, files in os.walk(self.ws):
                for name in dirs + files:
                    p = os.path.join(dirpath, name)
                    out.append((os.path.relpath(p, self.ws), os.stat(p).st_mtime_ns))
            return sorted(out)

        before = snapshot()
        self.assertEqual(set(pages(self.ws)), set(D.PAGES))
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
            for label in TABS:
                self.assertIn(label, text)
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o644, rel)
        for d in (self.ws, os.path.join(self.ws, "dashboard")):
            self.assertEqual([n for n in os.listdir(d) if n.endswith(".tmp")], [], d)

    def test_write_dashboard_failure(self):
        path = D.write_dashboard(Workspace(self.ws))
        # A failed write leaves no temp file behind and raises.
        os.remove(path)
        os.makedirs(os.path.join(path, "occupied"))
        with self.assertRaises(OSError):
            D.write_dashboard(Workspace(self.ws))
        for d in (self.ws, os.path.join(self.ws, "dashboard")):
            self.assertEqual([n for n in os.listdir(d) if n.endswith(".tmp")], [], d)

    def test_fixed_generation_time(self):
        now = datetime.datetime(2026, 11, 22, 9, 30, 15)
        for rel, text in pages(self.ws, now=now).items():
            self.assertIn("更新于 2026-11-22 09:30:15", text, rel)

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
        ids = {OVERVIEW: ("flow", "best", "stop", "now"), ROUNDS: ("rounds",), EVENTS: ("events",),
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
        Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        self.assertEqual(self.questions(), [])
        page = self.overview()
        self.assertNotIn('id="ask"', page)
        self.assertNotIn("⏸", re.search(r"<title>.*?</title>", page).group(0))
        self.assertIn("● 运行中", page)

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

    def test_milestone_approval_before_the_decision_does_not_count(self):
        self.target_task("ws-q-ms-old")
        self.promoted_round()
        ws = Workspace(self.ws)
        ws.state["approvals"].append({"what": "use_testset", "ref": None, "by": "alice", "quote": "ok",
                                      "at": "2020-01-01T00:00:00Z", "round": 1})
        ws.save_state()
        self.assertEqual(len(self.state_items()), 1)


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
                self.assertFlow(train="fail", tune="wait")  # more training waits for the user
            else:
                self.assertFlow(train="fail", tune="current")

    def manifest_cost(self, cand):
        return Workspace(self.ws).manifest(cand)["cost"]["compute_hours"]

    def test_refresh_question(self):
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

    def test_over_budget_question_gone_after_approval(self):
        probe = fixtures.binary_task()
        probe["budget"]["compute_hours"] = None
        self.use_task(probe, "ws-q-ob-probe")
        self.sets()
        cost = self.manifest_cost(self.candidate())
        task = fixtures.binary_task()
        task["approvals"] = [a for a in task["approvals"] if a != "over_budget"] + ["over_budget"]
        task["budget"]["compute_hours"] = cost
        self.use_task(task, "ws-q-ob-approved")
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS,
                 train_gate={"passed": False, "detail": "x"})
        self.assertEqual(len(self.state_items()), 1)
        Workspace(self.ws).approve("over_budget", "alice", "go over budget")
        self.assertEqual(self.state_items(), [])
        self.assertNotIn("等用户回答", self.waiting())

    def refresh_state(self, name):
        task = fixtures.binary_task()
        task["eval_use_limit"] = 1
        task["approvals"] = task["approvals"] + ["plan"]
        self.use_task(task, name)
        ev = self.promoted_round()
        plan = self.plan(ev)
        Workspace(self.ws).approve("plan", "alice", "go", plan)
        Workspace(self.ws).next_round()
        self.candidate(2)

    def test_refresh_question_gone_after_approval(self):
        self.refresh_state("ws-q-refresh-approved")
        self.assertEqual(len(self.state_items()), 1)
        w = self.waiting()
        self.assertIn("等用户回答", w)
        self.assertNotIn("派 rsi-deliver", w)
        st = self.assertFlow(deliver="wait")  # delivering waits for the user
        self.assertEqual(st["deliver"]["text"], "⏸ 等你批准")
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        self.assertEqual(self.state_items(), [])
        self.assertIn("rsi-deliver", self.waiting())
        self.assertFlow(deliver="current")

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

    def test_unreadable_notes_do_not_break_the_page(self):
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
        self.assertIn("派 rsi-train 或 rsi-tune", detail)  # the step sentence moves to the detail
        st = self.assertFlow(train="wait", tune="wait", data="done")
        self.assertEqual(st["train"]["text"], "⏸ 等你回答")
        self.assertIn('<span class="chip ask">⏸ 等你回答</span>', page)
        os.makedirs(os.path.join(self.ws, "work", "round-1", "train"))
        st = self.assertFlow(train="wait", tune="todo")
        self.assertEqual(st["train"]["text"], "⏸ 运行中 · 等你回答")
        self.write_notes("## 待用户回答\n")
        st = self.assertFlow(train="current")
        self.assertIn("进行中", st["train"]["text"])

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

    def test_notes_without_section(self):
        self.assertEqual(self.questions(), [])  # no notes.md
        self.write_notes("# 笔记\n- 只是笔记\n## 其他\n- 不是问题\n")
        self.assertEqual(self.questions(), [])
        self.write_notes("# 笔记\n## 待用户回答\n\n   \n## 其他\n- x\n")
        self.assertEqual(self.questions(), [])
        self.write_notes("## 待用户回答 （旧）\n- 不算\n")
        self.assertEqual(self.questions(), [])

    def test_state_questions_come_first(self):
        self.plan_task("ws-q-order")
        plan = self.plan(self.promoted_round())
        self.write_notes("## 待用户回答\n- 笔记问题\n")
        self.assertEqual([i["source"] for i in self.questions()], ["state", "notes"])
        self.assertIn(plan, self.questions()[0]["text"])

    def test_close_round_wait(self):
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-q-close")
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)
        self.assertIn("rsi close-round", self.waiting())
        self.assertEqual(self.questions(), [])
        self.assertFlow(train="fail", check="current")


if __name__ == "__main__":
    unittest.main()
