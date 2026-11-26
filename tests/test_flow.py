"""The main flow as a pure function (rsi/flow.py): next_action for every transition of a run, the new
STATE fields (dispatches, decisions, waiting), the task's optional `flow` settings, and the pages
taking what the run waits for from next_action."""
import builtins
import copy
import datetime
import json
import os
import unittest
from unittest import mock

import fixtures
import test_dashboard as TD
from rsi import briefs as B
from rsi import cli
from rsi import dashboard as D
from rsi import flow as F
from rsi import rows
from rsi.rules import MODULE_OF, compare_settings, flow_settings
from rsi.workspace import Workspace, now_iso

VAL_IDS, TRAIN_IDS, EVAL_IDS, TEST_IDS = TD.VAL_IDS, TD.TRAIN_IDS, TD.EVAL_IDS, TD.TEST_IDS
FAIL = TD.FAIL
QUALITY_FAIL = {"passed": False, "checks": [{"name": "dup_rate", "value": 0.5, "threshold": 0.05, "passed": False}]}


class FlowCase(TD.DashboardCase):
    """DashboardCase's builders plus the records the engine keeps in STATE (dispatches, decisions,
    waiting), written with the Workspace writers the engine uses."""

    def action(self):
        return F.next_action(Workspace(self.ws))

    def unique(self):
        """A workspace name not used yet in this test."""
        self.k = getattr(self, "k", 0) + 1
        return "ws-u%03d" % self.k

    def edit_state(self, fn):
        """Change STATE.json by hand (only for a state the writers do not make: an old workspace)."""
        path = os.path.join(self.ws, "STATE.json")
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        fn(s)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)

    def add_dispatch(self, module, status="running", rnd=None, returns=0, marker=None, purpose=None):
        """A dispatch record of `module` (and its output directory); return its id. purpose defaults to the
        module's usual one."""
        ws = Workspace(self.ws)
        rnd = ws.state["round"] if rnd is None else rnd
        same = len(ws.dispatches(rnd=rnd, module=module))
        outdir = "work/round-%d/%s%s" % (rnd, module, "-%d" % (same + 1) if same else "")
        os.makedirs(os.path.join(self.ws, outdir), exist_ok=True)
        return ws.add_dispatches([{
            "round": rnd, "module": module, "outdir": outdir, "brief": outdir + "/brief.md", "runner": "mock",
            "purpose": purpose or F.dispatch_purpose({"module": module, "round": rnd}),
            "session": "s-%d" % (len(ws.dispatches()) + 1), "status": status, "returns": returns,
            "ended_at": None if status == "running" else now_iso(), "marker": marker}])[0]

    def set_status(self, did, status, **fields):
        Workspace(self.ws).update_dispatch(did, status=status, **fields)

    def handback_into(self, did, kind, name=None, **fields):
        """Write an artifact into dispatch `did`'s output directory (subdirectory `name`, default the kind) and
        register it (as the engine would on processing its hand-back); return the id."""
        d = Workspace(self.ws).dispatch(did)
        src = fixtures.make_artifact(kind, os.path.join(self.ws, d["outdir"], name or kind), self.task,
                                     round=d["round"], **fields)
        return Workspace(self.ws).register(kind, src)

    def decide(self, point, choice, options=("yes", "no")):
        Workspace(self.ws).add_decision({"point": point, "options": list(options), "choice": choice, "by": "alice",
                                         "reason": "user answer"})

    def set_waiting(self, **w):
        record = {"kind": "question", "question": "q?", "options": ["answered"]}
        record.update(w)
        Workspace(self.ws).set_waiting(record)

    def flow_task(self, name, **flow):
        task = fixtures.binary_task()
        task["flow"] = flow
        self.use_task(task, name)

    def round_two(self, entry="train", **task_changes):
        """Round 1 promoted V1 and its plan (entry `entry`) took the run to round 2; return the evaluation."""
        ev = self.promoted_round()
        attribution = {"data": "data_coverage", "train": "model_capacity", "deliver": "runtime"}[entry]
        plan = self.reg("plan", 1, evaluation_id=ev, entry_module=entry,
                        issues=[dict(fixtures._plan_issue(), attribution=attribution, module=entry)])
        if "plan" in self.task["approvals"]:
            Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        self.assertEqual(Workspace(self.ws).next_round(), (2, entry))
        return ev

    def search_from(self, did, cands):
        """Register, from dispatch `did`'s output directory, the search over candidates `cands` (the first the default;
        the best by the validation auc); return its id."""
        ws = Workspace(self.ws)
        rows_ = [{"candidate_id": c, "config": {"lr": 0.001 * (i + 1)}, "compute_hours": 1, "status": "ok",
                  "metric_val": ws.manifest(c)["metrics_val"]["auc"]} for i, c in enumerate(cands)]
        best = max(rows_, key=lambda r: r["metric_val"])
        n = len(ws.entries("search")) + 1
        return self.handback_into(did, "search", "search-%d" % n, producer="train", trials=rows_,
                                  default_candidate=cands[0], best_candidate=best["candidate_id"],
                                  gain_vs_default={"metric": "auc",
                                                   "delta": best["metric_val"] - rows_[0]["metric_val"]})

    def assertAction(self, expect, action=None):
        a = self.action() if action is None else action
        for k, v in expect.items():
            self.assertEqual(a.get(k), v, (k, a))
        self.assertIn(a["type"], F.ACTION_TYPES)
        return a


# ---- the transition table ---------------------------------------------------------------------
# Each entry: (name, builder, expected fields). A builder gets a FlowCase with a fresh workspace
# named after the entry and brings it into the state; next_action must return the expected fields.

def b_fresh(c):
    pass


def b_four_sets(c):
    c.sets()


def b_running(c):
    c.sets()
    c.add_dispatch("train")
    c.add_dispatch("data", purpose="trainset")


def b_handed_back(c):
    c.sets()
    c.add_dispatch("train", "handed_back")


def b_handed_back_before_running(c):
    c.sets()
    c.add_dispatch("train", "running")
    c.add_dispatch("data", "handed_back", purpose="trainset")


def b_long_job(c):
    c.sets()
    c.add_dispatch("train", "long_job", marker="work/round-1/train/DONE")


def b_long_job_marker_present(c):
    b_long_job(c)
    open(os.path.join(c.ws, "work", "round-1", "train", "DONE"), "w").close()


def b_long_job_absolute_marker(c):
    c.sets()
    marker = os.path.join(c.root, "markers", "job.done")
    os.makedirs(os.path.dirname(marker))
    open(marker, "w").close()
    c.add_dispatch("train", "long_job", marker=marker)


def checkin_task(c, **flow):
    """A task with check-ins every 30 minutes (and `flow`), its four sets registered."""
    task = fixtures.binary_task()
    task["flow"] = dict({"checkin_minutes": 30}, **flow)
    c.use_task(task, c.unique())
    c.sets()


LONG_AGO = "2026-11-24T00:00:00Z"  # a hand-back long before the test runs


def b_long_job_checkin_due(c):
    checkin_task(c)
    did = c.add_dispatch("train", "long_job", marker="work/round-1/train/DONE")
    Workspace(c.ws).update_dispatch(did, ended_at=LONG_AGO)


def b_long_job_checkin_later(c):
    checkin_task(c)
    c.add_dispatch("train", "long_job", marker="work/round-1/train/DONE")  # handed back just now


def b_long_job_checkins_used_up(c):
    checkin_task(c, max_checkins=2)
    did = c.add_dispatch("train", "long_job", marker="work/round-1/train/DONE")
    Workspace(c.ws).update_dispatch(did, ended_at=LONG_AGO, checkins=2, launches=3, launch_mode="checkin")


def b_failed_dispatch(c):
    c.sets()
    c.add_dispatch("train", "failed")


def b_failed_dispatch_returns_used(c):
    c.sets()
    c.add_dispatch("train", "failed", returns=2)


def b_failed_dispatch_superseded(c):
    # A failed dispatch followed by a later one of the same module is no longer open.
    c.sets()
    c.add_dispatch("train", "failed")
    c.add_dispatch("train", "done")
    c.candidate()


def b_train_gate_failed(c):
    c.sets()
    c.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL)


def failed_search_from(c, did):
    """A trial that failed its train gate and its search, both from dispatch `did`; return the search."""
    n = len(Workspace(c.ws).entries("candidate")) + 1
    cand = c.handback_into(did, "candidate", "trial-%d" % n, ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS,
                           train_gate=FAIL, notes="trial %d" % n)
    return c.search_from(did, [cand])


def b_train_gate_failed_after_search(c, train_returns=0):
    # train (d-0001) handed back a trial that failed its gate, then its search.
    c.sets()
    failed_search_from(c, c.add_dispatch("train", "done", returns=train_returns))


def b_train_gate_failed_at_the_cap(c):
    b_train_gate_failed_after_search(c, train_returns=2)


def b_train_handed_back_nothing(c, first_returns=0):
    # After the gate failed, a fresh train dispatch (d-0002) handed back without registering anything.
    b_train_gate_failed_after_search(c, train_returns=first_returns)
    c.add_dispatch("train", "done")


def b_redispatch_keeps_the_cap(c):
    b_train_handed_back_nothing(c, first_returns=1)


def b_refused(c, returns=0, launches=1):
    # The registration refused the trainset rsi-data's first launch handed back (its latest unless `launches` > 1).
    did = c.add_dispatch("data", "failed", returns=returns)
    Workspace(c.ws).update_dispatch(did, launches=launches, refused=[{"dir": "trainset", "kind": "trainset",
                                                                      "error": "overlaps the evalset", "launch": 1}])


def b_refused_returns_used(c):
    b_refused(c, returns=2)


def b_refused_earlier_launch(c):
    b_refused(c, launches=2)


def b_failed_and_running(c):
    c.sets()
    c.add_dispatch("train", "failed")
    c.add_dispatch("data", "running", purpose="trainset")


def b_running_and_long_job(c):
    c.sets()
    c.add_dispatch("train", "long_job", marker="work/round-1/train/DONE")
    c.add_dispatch("data", "running", purpose="trainset")


def b_attempts_used_up(c):
    c.task["stop"]["max_attempts_per_round"] = 1
    c.use_task(c.task, c.unique())
    b_train_gate_failed_after_search(c)


def b_over_budget(c):
    task = fixtures.binary_task()
    task["budget"]["compute_hours"] = 1.5  # one fixture candidate costs 1.5 hours
    c.use_task(task, c.unique())
    b_train_gate_failed(c)


def b_quality_gate_failed(c):
    c.reg("testset", ids=TEST_IDS)
    c.reg("evalset", ids=EVAL_IDS)
    c.reg("valset", ids=VAL_IDS)
    c.reg("trainset", ids=TRAIN_IDS, quality=QUALITY_FAIL)


def b_quality_gate_failed_from_dispatch(c):
    did = c.add_dispatch("data", "done")
    c.handback_into(did, "testset", ids=TEST_IDS)
    c.handback_into(did, "evalset", ids=EVAL_IDS)
    c.handback_into(did, "valset", ids=VAL_IDS)
    c.handback_into(did, "trainset", ids=TRAIN_IDS, quality=QUALITY_FAIL)


def b_quality_gate_failed_after_step_one(c):
    # Round 2 enters at train; the train gate failed and a new trainset from data (d-0002) failed its quality gate.
    c.round_two(entry="train")
    failed_search_from(c, c.add_dispatch("train", "done"))
    c.handback_into(c.add_dispatch("data", "done", purpose="trainset"), "trainset", ids=range(70, 80),
                    quality=QUALITY_FAIL)


def b_data_handed_back_short(c):
    # rsi-data handed back three of the four sets: the step is not done after its dispatch.
    did = c.add_dispatch("data", "done")
    c.handback_into(did, "testset", ids=TEST_IDS)
    c.handback_into(did, "valset", ids=VAL_IDS)


def b_trials_without_search(c):
    # train (d-0001) handed back its search's trials (the first part); the search is still owed.
    c.sets()
    did = c.add_dispatch("train", "done")
    c.handback_into(did, "candidate", ids=VAL_IDS, train_ids=TRAIN_IDS)


def b_passed_candidate(c):
    c.sets()
    c.reg("candidate", ids=VAL_IDS, wrong=(1,), train_ids=TRAIN_IDS)  # passes, one row wrong
    c.candidate()  # passes with every row right: the better one on the valset


def b_delivery_without_evaluation(c):
    c.sets()
    cand = c.candidate()
    c.delivery(cand)


def b_judge_handed_back_nothing(c):
    b_delivery_without_evaluation(c)
    c.add_dispatch("judge", "done")


def b_evaluation_undecided(c):
    c.sets()
    c.chain()


def b_decided_no_plan(c):
    c.promoted_round()


def b_decided_judge_to_resume(c):
    c.sets()
    cand = c.candidate()
    dlv = c.delivery(cand)
    did = c.add_dispatch("judge", "done")
    ev = c.handback_into(did, "evaluation", ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=dlv, candidate_id=cand,
                         evalset_id="evalset-0001", best_before=None)
    Workspace(c.ws).promote(ev)


def b_decided_judge_returns_used(c):
    # The judge's session already got its plan part (a followup) and came back without a plan; its returns are
    # used up.
    b_decided_judge_to_resume(c)
    Workspace(c.ws).update_dispatch(Workspace(c.ws).dispatches()[-1]["id"], returns=2, launches=2,
                                    launch_mode="followup", followups=1)


def b_closed_round(c):
    c.sets()
    Workspace(c.ws).close_round("nothing to evaluate")


def b_plan_needs_approval(c):
    c.plan_task(c.unique())
    c.plan(c.promoted_round())


def b_plan_revise(c, launches=1):
    # The judge (d-0001) wrote the evaluation and the plan; the user asked for a revision of the plan.
    c.plan_task(c.unique())
    b_decided_judge_to_resume(c)
    issue = dict(fixtures._plan_issue(), attribution="model_capacity", module="train")
    c.handback_into("d-0001", "plan", evaluation_id="evaluation-0001", entry_module="train", issues=[issue])
    Workspace(c.ws).add_decision({"point": "plan", "options": ["approve", "revise"], "choice": "revise", "by": "alice",
                                  "quote": "add a slice for short messages", "ref": "plan-0001", "dispatch_id": "d-0001",
                                  "launches": 1})
    Workspace(c.ws).update_dispatch("d-0001", launches=launches)


def b_plan_revise_done(c):
    # The judge's session was resumed for the revision (launch 2) and came back without a new plan.
    b_plan_revise(c, launches=2)


def b_plan_approved(c):
    b_plan_needs_approval(c)
    Workspace(c.ws).approve("plan", "alice", "go ahead", "plan-0001")


def b_plan_without_approval_item(c):
    c.plan(c.promoted_round())


def target_task(c):
    task = fixtures.binary_task()
    task["stop"]["target"] = 0.8
    c.use_task(task, c.unique())


def b_milestone_not_asked(c):
    target_task(c)
    c.promoted_round()


def b_milestone_no(c):
    b_milestone_not_asked(c)
    c.decide("milestone", "no")


def b_milestone_yes(c):
    b_milestone_not_asked(c)
    c.decide("milestone", "yes")


def milestone_delivery(c):
    cand = Workspace(c.ws).version("V1")["candidate_id"]
    Workspace(c.ws).approve("use_testset", "alice", "yes, validate it on the test inputs")
    d = fixtures.make_artifact("delivery", c.src("delivery"), c.task, ids=EVAL_IDS, val_ids=VAL_IDS,
                               candidate_id=cand, round=1)
    fixtures.add_predictions(d, c.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
    return Workspace(c.ws).register("delivery", d), cand


def b_milestone_delivered(c):
    b_milestone_yes(c)
    milestone_delivery(c)


def b_milestone_evaluated(c):
    b_milestone_yes(c)
    dlv, cand = milestone_delivery(c)
    Workspace(c.ws).approve("use_testset", "alice", "yes, validate it on the test inputs")
    c.reg("evaluation", 1, ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=dlv, candidate_id=cand,
          evalset_id="evalset-0001", uses_testset=True, test_ids=TEST_IDS,
          best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
          recommendation={"decision": "keep", "reason": "milestone release"})


def b_milestone_legacy_approval(c):
    # The manual flow recorded the user's yes as a use_testset approval after the decision.
    b_milestone_not_asked(c)
    Workspace(c.ws).approve("use_testset", "alice", "yes, validate it on the test inputs")


def b_stop_without_version(c):
    task = fixtures.binary_task()
    task["stop"]["max_rounds"] = 1
    c.use_task(task, c.unique())
    b_closed_round(c)


def refresh_round(c):
    task = fixtures.binary_task()
    task["eval_use_limit"] = 1
    c.use_task(task, c.unique())
    c.round_two(entry="train")
    c.candidate(2)


def b_compare_due(c):
    task = fixtures.binary_task()
    task["compare"] = {"baselines": ["lr"]}
    c.use_task(task, c.unique())
    c.sets()
    c.candidate()


def b_compare_failed_at_the_cap(c):
    b_compare_due(c)
    c.add_dispatch("compare", "failed", returns=2)


def b_evalset_used_up(c):
    refresh_round(c)


def b_refresh_approved(c):
    b_evalset_used_up(c)
    Workspace(c.ws).approve("refresh_sets", "alice", "refresh the evalset")


def b_refreshed_evalset(c):
    b_refresh_approved(c)
    c.reg("evalset", 2, ids=range(60, 64))


def b_refresh_brought_a_trainset_only(c):
    b_refresh_approved(c)
    c.handback_into(c.add_dispatch("data", "done", purpose="refresh"), "trainset", ids=range(80, 84))


def b_stale_waiting_approved(c):
    b_waiting(c)
    Workspace(c.ws).approve("plan", "alice", "go ahead", "plan-0001")


def b_stale_waiting_used(c):
    # The refresh_sets approval the question asked for was given and then used by the new evalset.
    b_evalset_used_up(c)
    c.set_waiting(kind="approval", what="refresh_sets", question="refresh_sets?", options=["approve", "reject"])
    Workspace(c.ws).approve("refresh_sets", "alice", "refresh the evalset")
    c.reg("evalset", 2, ids=range(60, 64))


def b_stopped_no_report(c):
    Workspace(c.ws).stop("alice", "enough")


def b_stopped_report(c):
    b_stopped_no_report(c)
    with open(os.path.join(c.ws, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Report\n")


def b_stopped_with_waiting_and_agents(c):
    c.sets()
    c.add_dispatch("train")
    c.set_waiting()
    Workspace(c.ws).stop("alice", "enough")


def b_waiting(c):
    c.plan_task(c.unique())
    c.plan(c.promoted_round())
    c.set_waiting(kind="approval", what="plan", ref="plan-0001", question="Approve plan plan-0001?",
                  options=["approve", "reject"])


def b_waiting_before_agents(c):
    c.sets()
    c.add_dispatch("train", "handed_back")
    c.set_waiting(question="Which machine to use?", options=["a", "b"])


def b_old_workspace(c):
    c.edit_state(lambda s: [s.pop(k) for k in ("dispatches", "decisions", "waiting")])


TABLE = (
    ("fresh workspace", b_fresh, {"type": "dispatch", "module": "data", "purpose": "sets", "step": "data"}),
    ("old workspace without the new fields", b_old_workspace, {"type": "dispatch", "module": "data"}),
    ("four sets", b_four_sets, {"type": "dispatch", "module": "train", "purpose": "train", "step": "train"}),
    ("agents running", b_running, {"type": "wait_agent", "dispatch_ids": ["d-0001", "d-0002"]}),
    ("hand-back", b_handed_back, {"type": "process_handback", "dispatch_id": "d-0001"}),
    ("hand-back while another agent runs", b_handed_back_before_running,
     {"type": "process_handback", "dispatch_id": "d-0002"}),
    ("long job", b_long_job, {"type": "wait_long_job", "dispatch_id": "d-0001", "marker": "work/round-1/train/DONE"}),
    ("long job whose marker exists", b_long_job_marker_present, {"type": "process_handback", "dispatch_id": "d-0001"}),
    ("long job, absolute marker exists", b_long_job_absolute_marker,
     {"type": "process_handback", "dispatch_id": "d-0001"}),
    ("long job, its check-in due", b_long_job_checkin_due,
     {"type": "resume", "dispatch_id": "d-0001", "mode": "checkin", "step": "running"}),
    ("long job, its check-in not due yet", b_long_job_checkin_later,
     {"type": "wait_long_job", "dispatch_id": "d-0001", "step": "running"}),
    ("long job, check-ins at flow.max_checkins", b_long_job_checkins_used_up,
     {"type": "ask", "kind": "question", "what": "checkin", "ref": "d-0001", "options": ["continue", "wait", "stop"],
      "step": "running"}),
    ("failed dispatch", b_failed_dispatch,
     {"type": "decide", "point": "handback_failed", "options": ["resume", "ask"], "default": "resume",
      "dispatch_id": "d-0001"}),
    ("failed dispatch, returns used up", b_failed_dispatch_returns_used,
     {"type": "decide", "point": "handback_failed", "default": "ask"}),
    ("failed dispatch followed by another", b_failed_dispatch_superseded, {"type": "dispatch", "module": "deliver"}),
    ("train gate failed, no dispatch record", b_train_gate_failed,
     {"type": "decide", "point": "train_gate_failed", "options": ["train", "data", "ask"], "default": "train",
      "dispatch_id": None, "step": "train"}),
    ("train gate failed after a search: the dispatch behind it", b_train_gate_failed_after_search,
     {"type": "decide", "point": "train_gate_failed", "default": "train", "dispatch_id": "d-0001"}),
    ("train gate failed, train's returns used up", b_train_gate_failed_at_the_cap,
     {"type": "decide", "point": "train_gate_failed", "default": "ask", "dispatch_id": "d-0001"}),
    ("fresh train dispatch registered nothing", b_train_handed_back_nothing,
     {"type": "decide", "point": "handback_failed", "default": "resume", "dispatch_id": "d-0002", "step": "train"}),
    ("re-dispatching does not reset the cap", b_redispatch_keeps_the_cap,
     {"type": "decide", "point": "handback_failed", "default": "ask", "dispatch_id": "d-0002"}),
    ("refused registration", b_refused,
     {"type": "decide", "point": "handback_refused", "options": ["resume", "ask"], "default": "resume",
      "dispatch_id": "d-0001", "step": "data"}),
    ("refused registration, returns used up", b_refused_returns_used,
     {"type": "decide", "point": "handback_refused", "default": "ask", "dispatch_id": "d-0001"}),
    ("refusal of an earlier launch: the latest launch failed otherwise", b_refused_earlier_launch,
     {"type": "decide", "point": "handback_failed", "dispatch_id": "d-0001"}),
    ("failed dispatch beats running agents", b_failed_and_running,
     {"type": "decide", "point": "handback_failed", "dispatch_id": "d-0001"}),
    ("running agents beat a long job", b_running_and_long_job, {"type": "wait_agent", "dispatch_ids": ["d-0002"]}),
    ("attempts used up", b_attempts_used_up, {"type": "close_round", "step": "close"}),
    ("compute budget used up", b_over_budget,
     {"type": "ask", "kind": "approval", "what": "over_budget", "options": ["approve", "stop"], "step": "train"}),
    ("quality gate failed, no dispatch record", b_quality_gate_failed,
     {"type": "decide", "point": "quality_gate_failed", "options": ["resume_data", "ask"], "default": "ask",
      "dispatch_id": None, "step": "data"}),
    ("quality gate failed from a dispatch", b_quality_gate_failed_from_dispatch,
     {"type": "decide", "point": "quality_gate_failed", "default": "resume_data", "dispatch_id": "d-0001"}),
    ("new trainset failed its quality gate after step one", b_quality_gate_failed_after_step_one,
     {"type": "decide", "point": "quality_gate_failed", "default": "resume_data", "dispatch_id": "d-0002",
      "step": "data"}),
    ("data handed back three sets", b_data_handed_back_short,
     {"type": "decide", "point": "handback_failed", "default": "resume", "dispatch_id": "d-0001", "step": "data"}),
    ("trials registered without their search", b_trials_without_search,
     {"type": "resume", "dispatch_id": "d-0001", "step": "train", "mode": "followup"}),
    ("candidates through the train gate", b_passed_candidate,
     {"type": "dispatch", "module": "deliver", "candidate_id": "candidate-0002", "step": "deliver"}),
    ("candidate through the train gate, ③ Compare on", b_compare_due,
     {"type": "dispatch", "module": "compare", "purpose": "compare", "candidate_id": "candidate-0001",
      "step": "compare"}),
    ("compare failed at the cap", b_compare_failed_at_the_cap,
     {"type": "decide", "point": "handback_failed", "options": ["resume", "skip", "ask"], "default": "skip",
      "dispatch_id": "d-0001", "step": "compare"}),
    ("delivery without evaluation", b_delivery_without_evaluation, {"type": "dispatch", "module": "judge"}),
    ("judge handed back no evaluation", b_judge_handed_back_nothing,
     {"type": "decide", "point": "handback_failed", "dispatch_id": "d-0001", "step": "judge"}),
    ("evaluation not decided", b_evaluation_undecided, {"type": "promote", "evaluation_id": "evaluation-0001"}),
    ("decided, no plan", b_decided_no_plan,
     {"type": "dispatch", "module": "judge", "purpose": "plan", "step": "write_plan"}),
    ("decided, the judge's session writes the plan", b_decided_judge_to_resume,
     {"type": "resume", "dispatch_id": "d-0001", "step": "write_plan", "mode": "followup"}),
    ("decided, the judge's returns used up", b_decided_judge_returns_used,
     {"type": "decide", "point": "handback_failed", "default": "ask", "step": "write_plan"}),
    ("closed round, no plan", b_closed_round, {"type": "dispatch", "module": "judge", "step": "write_plan"}),
    ("plan waits for approval", b_plan_needs_approval,
     {"type": "ask", "kind": "approval", "what": "plan", "ref": "plan-0001", "options": ["approve", "revise"]}),
    ("plan revision asked of the judge's session", b_plan_revise,
     {"type": "resume", "dispatch_id": "d-0001", "step": "write_plan", "mode": "resume"}),
    ("plan revision carried out, the same plan came back", b_plan_revise_done,
     {"type": "ask", "kind": "approval", "what": "plan", "ref": "plan-0001"}),
    ("plan approved", b_plan_approved, {"type": "next"}),
    ("plan needs no approval", b_plan_without_approval_item, {"type": "next"}),
    ("stop condition met, milestone not asked", b_milestone_not_asked,
     {"type": "ask", "kind": "question", "what": "milestone", "options": ["yes", "no"]}),
    ("milestone declined", b_milestone_no, {"type": "next"}),
    ("milestone agreed", b_milestone_yes,
     {"type": "dispatch", "module": "deliver", "purpose": "milestone_deliver", "candidate_id": "candidate-0001",
      "step": "milestone"}),
    ("milestone agreed in the manual flow", b_milestone_legacy_approval,
     {"type": "dispatch", "module": "deliver", "step": "milestone"}),
    ("milestone delivery registered", b_milestone_delivered,
     {"type": "dispatch", "module": "judge", "purpose": "milestone_judge", "step": "milestone"}),
    ("milestone evaluation registered", b_milestone_evaluated, {"type": "next"}),
    ("stop condition met, no version", b_stop_without_version, {"type": "next"}),
    ("evalset used up", b_evalset_used_up,
     {"type": "ask", "kind": "approval", "what": "refresh_sets", "options": ["approve", "close_round", "stop"],
      "step": "deliver"}),
    ("refresh approved", b_refresh_approved,
     {"type": "dispatch", "module": "data", "purpose": "refresh", "step": "refresh"}),
    ("refresh dispatch brought a trainset only", b_refresh_brought_a_trainset_only,
     {"type": "decide", "point": "handback_failed", "dispatch_id": "d-0001", "step": "refresh"}),
    ("waiting approval already given", b_stale_waiting_approved, {"type": "next"}),
    ("waiting approval given and used", b_stale_waiting_used, {"type": "dispatch", "module": "deliver"}),
    ("evalset refreshed", b_refreshed_evalset, {"type": "dispatch", "module": "deliver", "step": "deliver"}),
    ("stopped, no report", b_stopped_no_report, {"type": "report"}),
    ("stopped, report written", b_stopped_report, {"type": "done"}),
    ("stopped beats waiting and agents", b_stopped_with_waiting_and_agents, {"type": "report"}),
    ("waiting for an answer", b_waiting,
     {"type": "ask", "kind": "approval", "what": "plan", "ref": "plan-0001", "question": "Approve plan plan-0001?",
      "options": ["approve", "reject"], "step": "plan"}),
    ("waiting beats agents", b_waiting_before_agents,
     {"type": "ask", "kind": "question", "what": None, "options": ["a", "b"]}),
)


FIELDS = {"dispatch": ("module", "purpose", "reason"), "resume": ("dispatch_id", "feedback", "mode"), "wait_agent": ("dispatch_ids",),
          "wait_long_job": ("dispatch_id", "marker", "checkin_at"), "process_handback": ("dispatch_id",),
          "decide": ("point", "options", "default", "dispatch_id", "reason"),
          "ask": ("kind", "what", "ref", "question", "options"), "promote": ("evaluation_id",),
          "close_round": ("reason",), "next": ("reason",), "report": (), "done": ()}


def snapshot(root):
    """Every path under `root` with its modification time and size: a write anywhere changes it."""
    out = []
    for dirpath, dirs, files in os.walk(root):
        for name in dirs + files:
            p = os.path.join(dirpath, name)
            st = os.stat(p)
            out.append((os.path.relpath(p, root), st.st_mtime_ns, st.st_size))
    return sorted(out)


class TestTransitions(FlowCase):
    def test_table(self):
        """For every entry: next_action gives the expected fields and those of its type plus a step; it
        writes nothing, leaves the state as it was and opens no file but manifests (task.json for the
        task); the pages render the state."""
        seen = set()
        for i, (name, builder, expect) in enumerate(TABLE):
            with self.subTest(name):
                self.use_task(fixtures.binary_task(), "ws-%02d" % i)
                builder(self)
                ws = Workspace(self.ws)
                state = copy.deepcopy(ws.state)
                before = snapshot(self.root)
                with mock.patch("builtins.open", wraps=builtins.open) as opened:
                    a = F.next_action(ws)
                self.assertAction(expect, a)
                seen.add(a["type"])
                for f in FIELDS[a["type"]] + ("step",):
                    self.assertIn(f, a)
                self.assertIn(a["step"], F.STEPS)
                if a["type"] == "decide":
                    self.assertIn(a["default"], a["options"])
                self.assertIn(a.get("purpose"), F.PURPOSES + (None,))
                json.dumps(a)  # the engine logs it to run.jsonl
                self.assertEqual(snapshot(self.root), before)
                self.assertEqual(ws.state, state)
                names = set(os.path.basename(str(c[0][0])) for c in opened.call_args_list)
                self.assertLessEqual(names, {"manifest.json", "task.json"})
                self.assertIn('id="now"', TD.pages(self.ws)[TD.OVERVIEW])
                self.assertTrue(self.waiting())
        self.assertEqual(seen, set(F.ACTION_TYPES))  # the table reaches every action type


class TestTransitionDetails(FlowCase):
    def test_walk_one_round_with_the_engine_records(self):
        # A round as the engine drives it: every module dispatched, handed back and registered.
        self.plan_task("ws-walk")
        self.assertAction({"type": "dispatch", "module": "data"})
        data = self.add_dispatch("data")
        self.assertAction({"type": "wait_agent", "dispatch_ids": [data]})
        self.set_status(data, "handed_back")
        self.assertAction({"type": "process_handback", "dispatch_id": data})
        for kind, ids in (("testset", TEST_IDS), ("evalset", EVAL_IDS), ("valset", VAL_IDS), ("trainset", TRAIN_IDS)):
            self.handback_into(data, kind, ids=ids)
        self.set_status(data, "done")
        self.assertAction({"type": "dispatch", "module": "train", "purpose": "train"})
        train = self.add_dispatch("train")
        self.assertAction({"type": "wait_agent", "dispatch_ids": [train]})
        self.set_status(train, "done")
        cand = self.handback_into(train, "candidate", ids=VAL_IDS, train_ids=TRAIN_IDS)
        a = self.assertAction({"type": "resume", "dispatch_id": train, "mode": "followup"})  # its search: part two
        self.assertIn(cand, a["feedback"])
        self.search_from(train, [cand])
        self.assertAction({"type": "dispatch", "module": "deliver", "candidate_id": cand})
        deliver = self.add_dispatch("deliver", "done")
        dlv = self.handback_into(deliver, "delivery", ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand)
        self.assertAction({"type": "dispatch", "module": "judge"})
        judge = self.add_dispatch("judge", "done")
        ev = self.handback_into(judge, "evaluation", ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=dlv,
                                candidate_id=cand, evalset_id="evalset-0001", best_before=None)
        self.assertAction({"type": "promote", "evaluation_id": ev})
        Workspace(self.ws).promote(ev)
        a = self.assertAction({"type": "resume", "dispatch_id": judge, "step": "write_plan"})
        self.assertIn(ev, a["feedback"])
        plan = self.handback_into(judge, "plan", evaluation_id=ev, entry_module="train",
                                  issues=[dict(fixtures._plan_issue(), attribution="model_capacity", module="train")])
        self.assertAction({"type": "ask", "kind": "approval", "what": "plan", "ref": plan})
        Workspace(self.ws).approve("plan", "alice", "go ahead", plan)
        self.assertAction({"type": "next"})
        self.assertEqual(Workspace(self.ws).next_round(), (2, "train"))
        self.assertAction({"type": "dispatch", "module": "train"})  # round 2 enters at train

    def test_deliver_dispatch_after_refresh_is_new(self):
        # The deliver dispatch of the old evalset is spent: the refreshed evalset needs a new dispatch, and the
        # best version's re-delivery is part of it.
        task = fixtures.binary_task()
        task["eval_use_limit"] = 1
        self.use_task(task, "ws-redeliver")
        self.round_two(entry="train")
        cand = self.candidate(2)
        d = self.add_dispatch("deliver", "done")
        self.handback_into(d, "delivery", ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand)
        self.assertAction({"type": "ask", "what": "refresh_sets", "step": "judge"})
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        self.assertAction({"type": "dispatch", "module": "data"})
        self.reg("evalset", 2, ids=range(60, 64))
        a = self.assertAction({"type": "dispatch", "module": "deliver", "candidate_id": cand})
        self.assertIn("V1", a["reason"])  # the best version is re-delivered on the new evalset too

    def test_milestone_dispatches_are_not_the_rounds_own(self):
        # The round's deliver and judge dispatches produced its delivery and evaluation; the milestone
        # release needs new ones, and a milestone dispatch that brought nothing is a failed hand-back.
        target_task(self)
        self.sets()
        cand = self.candidate()
        d1 = self.add_dispatch("deliver", "done")
        dlv = self.handback_into(d1, "delivery", ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand)
        j1 = self.add_dispatch("judge", "done")
        ev = self.handback_into(j1, "evaluation", ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=dlv,
                                candidate_id=cand, evalset_id="evalset-0001", best_before=None)
        Workspace(self.ws).promote(ev)
        self.decide("milestone", "yes")
        self.assertAction({"type": "dispatch", "module": "deliver", "step": "milestone"})
        d2 = self.add_dispatch("deliver", "done", purpose="milestone_deliver")
        self.assertAction({"type": "decide", "point": "handback_failed", "dispatch_id": d2, "step": "milestone"})
        Workspace(self.ws).approve("use_testset", "alice", "yes")
        src = fixtures.make_artifact("delivery", os.path.join(self.ws, "work", "round-1", "deliver-2", "m"),
                                     self.task, ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand, round=1)
        fixtures.add_predictions(src, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        Workspace(self.ws).register("delivery", src)
        self.assertAction({"type": "dispatch", "module": "judge", "step": "milestone"})
        j2 = self.add_dispatch("judge", "done", purpose="milestone_judge")
        self.assertAction({"type": "decide", "point": "handback_failed", "dispatch_id": j2, "step": "milestone"})

    def test_search_resume_is_bounded(self):
        # The search is the second part of train's hand-back: given once after the trials, whatever the returns.
        b_trials_without_search(self)
        Workspace(self.ws).update_dispatch("d-0001", returns=2)
        self.assertAction({"type": "resume", "dispatch_id": "d-0001", "mode": "followup", "step": "train"})
        # Its followup came back without a search: asking again is a return, capped by max_returns.
        Workspace(self.ws).update_dispatch("d-0001", returns=0, launches=2, launch_mode="followup", followups=1)
        self.assertAction({"type": "resume", "dispatch_id": "d-0001", "mode": "resume", "step": "train"})
        Workspace(self.ws).update_dispatch("d-0001", returns=2)
        self.assertAction({"type": "decide", "point": "handback_failed", "default": "ask", "step": "train"})

    def test_second_part_follows_the_launch_that_registered_the_first(self):
        # With launches noted in the record's registered entries: the trials of the latest launch call for the
        # search (a followup); trials from an earlier launch do not (a return).
        b_trials_without_search(self)
        trial = Workspace(self.ws).entries("candidate")[-1]["id"]
        Workspace(self.ws).update_dispatch("d-0001", launches=2, launch_mode="resume", returns=1,
                                           registered=[{"dir": "candidate", "kind": "candidate", "id": trial,
                                                        "launch": 2}])
        self.assertAction({"type": "resume", "mode": "followup"})
        Workspace(self.ws).update_dispatch("d-0001", launches=3, returns=2, registered=[
            {"dir": "candidate", "kind": "candidate", "id": trial, "launch": 2}])
        self.assertAction({"type": "decide", "point": "handback_failed", "default": "ask", "step": "train"})

    def test_an_answered_question_sends_the_handback_back_with_the_answer(self):
        # data handed back with a question (its artifacts wait unregistered): the question first; once the user has
        # answered it for this launch, the same session gets the answer (a resume with mode answer, no return).
        did = self.add_dispatch("data", "handed_back")
        Workspace(self.ws).update_dispatch(did, launches=1, asked={"launch": 1, "question_id": "q-0001",
                                                                   "questions": ["Keep them or drop them?"]})
        self.set_waiting(what="handback", ref=did, question="rsi-data (d-0001) handed back something for the user to "
                                                             "decide: Keep them or drop them?")
        self.assertAction({"type": "ask", "what": "handback", "ref": did})
        Workspace(self.ws).add_decision({"point": "handback", "choice": "answered", "by": "alice", "quote": "drop them",
                                         "dispatch_id": did, "launches": 1, "question_id": "q-0001"})
        Workspace(self.ws).clear_waiting()
        self.assertAction({"type": "resume", "dispatch_id": did, "mode": "answer", "step": "handed"})
        self.assertIn("Nothing of that hand-back is registered yet", self.action()["feedback"])
        # An answer to an earlier launch's question does not count for this hand-back: it is processed (and asks).
        Workspace(self.ws).update_dispatch(did, launches=2, asked={"launch": 2, "question_id": "q-0002",
                                                                   "questions": ["One more question?"]})
        self.assertAction({"type": "process_handback", "dispatch_id": did})

    def test_a_handback_question_of_a_closed_dispatch_is_stale(self):
        did = self.add_dispatch("data", "handed_back")
        self.set_waiting(what="handback", ref=did)
        ws = Workspace(self.ws)
        self.assertFalse(F.stale_waiting(ws, ws.waiting()))
        Workspace(self.ws).update_dispatch(did, status="done", closed="run stopped")
        ws = Workspace(self.ws)
        self.assertTrue(F.stale_waiting(ws, ws.waiting()))

    def test_the_judges_plan_is_no_return(self):
        # The judge's returns are used up (e.g. a refused evaluation sent back twice): its plan part still follows.
        b_decided_judge_to_resume(self)
        Workspace(self.ws).update_dispatch("d-0001", returns=2)
        self.assertAction({"type": "resume", "dispatch_id": "d-0001", "mode": "followup", "step": "write_plan"})
        self.assertEqual(F.returns_used(Workspace(self.ws), "judge"), 2)

    def test_no_search_owed_once_attempts_are_used_up(self):
        # Trials without a search when the attempts are used up: a search could not be registered any more,
        # so a trial through the train gate goes on to ③.
        self.task["stop"]["max_attempts_per_round"] = 1
        self.use_task(self.task, "ws-no-search")
        self.sets()
        did = self.add_dispatch("train", "done")
        failed_search_from(self, did)
        trial = self.handback_into(did, "candidate", "trial-2", ids=VAL_IDS, train_ids=TRAIN_IDS)
        self.assertAction({"type": "dispatch", "module": "deliver", "candidate_id": trial})

    def test_failed_dispatch_of_an_earlier_round_is_not_open(self):
        self.promoted_round()
        self.add_dispatch("train", "failed", rnd=1)
        self.plan(Workspace(self.ws).evaluations()[-1])
        Workspace(self.ws).next_round()
        self.assertAction({"type": "dispatch", "module": "train"})

    def test_checkin_times_and_answers(self):
        # D6: the next check-in is flow.checkin_minutes after the long job's latest hand-back; at the cap the user's
        # continue allows flow.max_checkins more, wait ends the check-ins of the dispatch; the marker ends them too.
        b_long_job_checkin_later(self)
        ws = Workspace(self.ws)
        a = self.assertAction({"type": "wait_long_job", "dispatch_id": "d-0001"})
        self.assertEqual(F.parse_utc(a["checkin_at"]) - F.parse_utc(ws.dispatch("d-0001")["ended_at"]),
                         datetime.timedelta(minutes=30))
        self.assertIn("check-ins of long job d-0001 (rsi-train): 0 of 48, every 30 minutes; next due",
                      "\n".join(cli.status_lines(ws)))
        b_long_job_checkins_used_up(self)
        q = self.assertAction({"type": "ask", "what": "checkin"})
        self.assertEqual(q["question"], "The long job of rsi-train (d-0001) has not finished (completion marker %s is "
                         "not there yet) after 2 check-ins, reaching the cap flow.max_checkins = 2. continue: 2 more "
                         "check-ins; wait: no more check-ins, only wait for the completion marker; stop: stop the run "
                         "(as rsi stop, which terminates the long job)"
                         % os.path.join(self.ws, "work/round-1/train/DONE"))
        self.set_waiting(what="checkin", ref="d-0001", question=q["question"], options=q["options"])
        self.assertFalse(F.stale_waiting(Workspace(self.ws), Workspace(self.ws).waiting()))
        Workspace(self.ws).clear_waiting()
        Workspace(self.ws).add_decision({"point": "checkin", "choice": "continue", "by": "alice", "quote": "keep checking",
                                         "dispatch_id": "d-0001", "options": list(F.CHECKIN_OPTIONS)})
        self.assertAction({"type": "resume", "dispatch_id": "d-0001", "mode": "checkin"})
        Workspace(self.ws).add_decision({"point": "checkin", "choice": "wait", "by": "alice", "quote": "just wait",
                                         "dispatch_id": "d-0001", "options": list(F.CHECKIN_OPTIONS)})
        self.assertAction({"type": "wait_long_job", "dispatch_id": "d-0001", "checkin_at": None})
        self.assertIn("check-ins of long job d-0001 (rsi-train): 2 of -, every 30 minutes; no more (the user said wait)",
                      cli.status_lines(Workspace(self.ws)))
        self.set_waiting(what="checkin", ref="d-0001", question=q["question"], options=q["options"])
        open(os.path.join(self.ws, "work", "round-1", "train", "DONE"), "w").close()
        self.assertTrue(F.stale_waiting(Workspace(self.ws), Workspace(self.ws).waiting()))
        self.assertAction({"type": "process_handback", "dispatch_id": "d-0001"})

    def test_old_tune_records_still_render(self):
        # Review focus 5 (flow and status; the pages and the report: test_dashboard's
        # test_old_tune_records_show_under_training_tuning): an old workspace whose round 1 had a
        # train and a tune dispatch, tune's trials and search (producer tune) and a plan entering round 2 at tune.
        task = fixtures.binary_task()
        task["flow"] = {"tune": "every_round"}  # still valid, no effect
        self.assertEqual(rows.check_task(task), [])
        self.use_task(task, "ws-old-tune")
        self.sets()
        train, tune = self.add_dispatch("train", "done"), self.add_dispatch("tune", "done", purpose="tune")
        cand = self.handback_into(train, "candidate", ids=VAL_IDS, train_ids=TRAIN_IDS)
        trials = [self.handback_into(tune, "candidate", "trial-%d" % i, ids=VAL_IDS, train_ids=TRAIN_IDS,
                                     producer="tune", config_source="tune", notes="trial %d" % i) for i in range(2)]
        ws = Workspace(self.ws)
        rows_ = [{"candidate_id": c, "config": {"lr": 0.001 * (i + 1)}, "compute_hours": 1, "status": "ok",
                  "metric_val": ws.manifest(c)["metrics_val"]["auc"]} for i, c in enumerate(trials)]
        search = self.handback_into(tune, "search", trials=rows_, default_candidate=trials[0],
                                    best_candidate=trials[1], gain_vs_default={"metric": "auc", "delta": 0.0})
        ev = self.evaluation(cand, self.delivery(cand))
        Workspace(self.ws).promote(ev)
        with mock.patch.dict(MODULE_OF, {"config": "tune"}):  # registered before the merge
            self.reg("plan", 1, evaluation_id=ev, entry_module="tune",
                     issues=[dict(fixtures._plan_issue(), attribution="config", module="tune")])
        ws = Workspace(self.ws)
        self.assertEqual(ws.attempts(rnd=1), [search])  # the train candidate is no attempt now
        self.assertEqual(ws.next_round(), (2, "tune"))
        # The plan's entry tune opens round 2 at the merged train step.
        self.assertAction({"type": "dispatch", "module": "train", "purpose": "train", "step": "train"})
        lines = cli.status_lines(Workspace(self.ws))
        self.assertRegex(lines[0], r"^now: \d\d:\d\d round 2 next: ② Train & tune \| budget: GPU hours 4\.5/–$")
        self.assertIn("round: 2 (entry tune, decision none, plan none)", lines)
        self.assertIn("attempts: 0 of 3", lines)
        ws = Workspace(self.ws)
        self.assertEqual((F.dispatch_purpose(ws.dispatch(tune)), ws.summary(search).split(";")[0]),
                         ("tune", "tune gate: passed"))

    def test_ask_texts(self):
        b_plan_needs_approval(self)
        self.assertIn("plan-0001", self.action()["question"])
        b_milestone_not_asked(self)
        q = self.action()["question"]
        self.assertIn("V1", q)
        self.assertIn("rsi next", q)
        b_evalset_used_up(self)
        self.assertIn("refresh_sets", self.action()["question"])


class TestBudgetHoldFlow(FlowCase):
    """B1 (third simulation): a hand-back the engine holds for an over_budget approval (rsi.flow.budget_held), and
    its STATE waiting going stale with its dispatch."""

    HELD = {"launch": 1, "question_id": "q-0001", "dirs": ["candidate"], "cost": 0.35, "used": 0, "budget": 0.25}

    def held_dispatch(self):
        self.sets()
        did = self.add_dispatch("train", "handed_back")
        Workspace(self.ws).update_dispatch(did, launches=1, held_budget=dict(self.HELD))
        return did

    def test_budget_held(self):
        self.sets()
        plain = self.add_dispatch("train", "handed_back")
        ws = Workspace(self.ws)
        self.assertFalse(F.budget_held(ws, ws.dispatch(plain)))  # no hold (a record from before holds, too)
        self.use_task(fixtures.binary_task(), self.unique())
        did = self.held_dispatch()
        ws = Workspace(self.ws)
        self.assertTrue(F.budget_held(ws, ws.dispatch(did)))
        ws.update_dispatch(did, launches=2)  # held for an earlier launch: not this hand-back
        self.assertFalse(F.budget_held(Workspace(self.ws), Workspace(self.ws).dispatch(did)))
        ws.update_dispatch(did, launches=1, status="done",
                           closed="run stopped (user requested: stop): this dispatch's hand-back is no longer registered")
        self.assertFalse(F.budget_held(Workspace(self.ws), Workspace(self.ws).dispatch(did)))
        ws.update_dispatch(did, status="handed_back", closed=None)
        Workspace(self.ws).approve("over_budget", "alice", "going over is fine")
        ws = Workspace(self.ws)
        self.assertFalse(F.budget_held(ws, ws.dispatch(did)))  # approved: the next process_handback registers it
        self.assertEqual(ws.dispatch(did)["held_budget"], self.HELD)  # kept as history

    def test_held_budget_question_goes_stale_with_its_dispatch(self):
        did = self.held_dispatch()
        self.set_waiting(kind="approval", what="over_budget", ref=None, dispatch_id=did, question="over_budget: …",
                         options=["approve", "stop"])
        ws = Workspace(self.ws)
        self.assertFalse(F.stale_waiting(ws, ws.waiting()))
        self.assertAction({"type": "ask", "what": "over_budget", "ref": None})
        self.set_status(did, "done", closed="run stopped (user requested: stop): this dispatch's hand-back is no longer "
                                            "registered or sent back")
        ws = Workspace(self.ws)
        self.assertTrue(F.stale_waiting(ws, ws.waiting()))
        ws.set_waiting(dict(ws.waiting(), dispatch_id="d-0099"))  # a dispatch the workspace does not know
        ws = Workspace(self.ws)
        self.assertTrue(F.stale_waiting(ws, ws.waiting()))
        ws.set_waiting(dict(ws.waiting(), dispatch_id=None))  # the over_budget ask before a dispatch: as before
        ws = Workspace(self.ws)
        self.assertFalse(F.stale_waiting(ws, ws.waiting()))


class TestStopTexts(FlowCase):
    """S1 (third simulation): a question's options say only what will happen, the stop that follows included."""

    OVER_BUDGET = ("over_budget: GPU hours used 0.25, which reaches the budget of 0.25; more training needs your "
                   "approval to go over budget. approve: approve going over budget (the round goes on until it is "
                   "decided, then the run stops because the budget is reached); stop: stop the run (as rsi stop)")

    def test_over_budget_before_dispatch_text(self):
        task = fixtures.binary_task()
        task["budget"]["compute_hours"] = 0.25
        self.use_task(task, self.unique())
        self.sets()
        self.reg("candidate", ids=VAL_IDS, wrong=VAL_IDS, train_ids=TRAIN_IDS, train_gate=FAIL,
                 cost={"compute_hours": 0.25})  # used exactly the budget
        self.assertAction({"type": "ask", "kind": "approval", "what": "over_budget", "ref": None, "step": "train",
                           "question": self.OVER_BUDGET, "options": ["approve", "stop"]})

    def test_counts_in_the_texts(self):
        # One of a thing in the singular, more in the plural: attempts, evalset uses, check-ins, returns.
        from rsi import decisions as DP
        b_attempts_used_up(self)
        self.assertEqual(self.action()["reason"],
                         "round 1 used up its 1 attempt with no candidate through the train gate")
        b_evalset_used_up(self)
        self.assertIn("refresh_sets: the evalset has been used 1 time, reaching its limit of 1;",
                      self.action()["question"])
        checkin_task(self, max_checkins=1)
        did = self.add_dispatch("train", "long_job", marker="DONE")
        Workspace(self.ws).update_dispatch(did, ended_at=LONG_AGO, checkins=1)
        self.assertIn("after 1 check-in, reaching the cap flow.max_checkins = 1. continue: 1 more check-in;",
                      self.action()["question"])
        self.use_task(fixtures.binary_task(), self.unique())
        b_failed_dispatch(self)
        for returns, words in ((0, "0 times"), (1, "1 time"), (2, "2 times")):
            Workspace(self.ws).update_dispatch("d-0001", returns=returns)
            self.assertEqual(DP._returns(Workspace(self.ws), "train"),
                             "rsi-train was sent back %s this round (cap flow.max_returns = 2)" % words)

    def test_refresh_close_round_says_whether_the_run_stops(self):
        b_evalset_used_up(self)
        q = self.action()["question"]
        self.assertIn("close_round: no refresh, close the round (no evaluation, recorded as keep), then rsi-judge "
                      "writes the plan; ", q)
        self.edit_state(lambda s: s.__setitem__("no_promotion_streak", 1))  # closing round 2 reaches patience 2
        stop, why = Workspace(self.ws).stop_check_if_closed()
        self.assertTrue(stop)
        q = self.action()["question"]
        self.assertIn("close_round: no refresh, close the round (no evaluation, recorded as keep), then the run stops "
                      "(%s); " % why, q)
        self.assertNotIn("writes the plan", q)


class TestStateFields(FlowCase):
    def test_read_helpers(self):
        ws = Workspace(self.ws)
        self.assertEqual((ws.state["dispatches"], ws.state["decisions"], ws.state["waiting"]), ([], [], None))
        self.assertEqual((ws.dispatches(), ws.decisions(), ws.waiting()), ([], [], None))
        self.sets()
        a = self.add_dispatch("train")
        b = self.add_dispatch("deliver", "done")
        self.decide("milestone", "no")
        ws = Workspace(self.ws)
        self.assertEqual([d["id"] for d in ws.dispatches()], [a, b])
        self.assertEqual([d["id"] for d in ws.dispatches(module="deliver")], [b])
        self.assertEqual(ws.dispatches(rnd=2), [])
        self.assertEqual(ws.dispatch(b)["module"], "deliver")
        self.assertEqual([d["choice"] for d in ws.decisions(rnd=1, point="milestone")], ["no"])
        self.assertEqual(ws.decisions(point="plan"), [])
        from rsi.workspace import RsiError
        with self.assertRaises(RsiError):
            ws.dispatch("d-9999")

    def test_old_workspace_reads_as_empty_and_takes_writes(self):
        b_old_workspace(self)
        ws = Workspace(self.ws)
        self.assertNotIn("dispatches", ws.state)
        self.assertEqual((ws.dispatches(), ws.decisions(), ws.waiting()), ([], [], None))
        self.assertIn('id="now"', TD.pages(self.ws)[TD.OVERVIEW])
        self.assertEqual(ws.add_dispatches([{"module": "data", "outdir": "work/round-1/data"}]), ["d-0001"])
        ws.add_decision({"point": "p", "choice": "c"})
        self.assertEqual((len(Workspace(self.ws).dispatches()), len(Workspace(self.ws).decisions())), (1, 1))


class TestWriters(FlowCase):
    """The Workspace writers the engine uses: each changes one STATE field and saves once."""

    def test_add_dispatches(self):
        ws = Workspace(self.ws)
        with mock.patch.object(Workspace, "save_state", wraps=ws.save_state) as saved:
            ids = ws.add_dispatches([{"module": "train", "purpose": "train", "outdir": "work/round-1/train"},
                                     {"module": "data", "purpose": "trainset", "outdir": "work/round-1/data", "id": "x"}])
        self.assertEqual(ids, ["d-0001", "d-0002"])
        self.assertEqual(saved.call_count, 1)  # two records together: one save
        d = Workspace(self.ws).dispatch("d-0002")  # persisted
        self.assertEqual((d["round"], d["module"], d["purpose"], d["status"], d["returns"], d["ended_at"]),
                         (1, "data", "trainset", "running", 0, None))
        self.assertRegex(d["started_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(Workspace(self.ws).add_dispatches([{"module": "deliver", "outdir": "work/round-1/deliver"}]),
                         ["d-0003"])
        # A bad record refuses the whole write: nothing is saved.
        ws = Workspace(self.ws)
        from rsi.workspace import RsiError
        for bad in ({"outdir": "work/x"}, {"module": "train"}, {"module": "train", "outdir": "w", "status": "lost"}):
            with self.assertRaises(RsiError):
                ws.add_dispatches([{"module": "data", "outdir": "work/round-1/data"}, bad])
        self.assertEqual(len(Workspace(self.ws).dispatches()), 3)

    def test_failed_save_leaves_state_unchanged(self):
        ws = Workspace(self.ws)
        with mock.patch.object(Workspace, "save_state", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                ws.add_dispatches([{"module": "train", "outdir": "work/round-1/train"}])
            with self.assertRaises(OSError):
                ws.set_waiting({"kind": "question", "question": "q?", "options": ["a"]})
        self.assertEqual((ws.dispatches(), ws.waiting()), ([], None))

    def test_update_dispatch(self):
        did = self.add_dispatch("train")
        ws = Workspace(self.ws)
        d = ws.update_dispatch(did, status="handed_back", session="s-9", returns=1)
        self.assertEqual((d["status"], d["session"], d["returns"]), ("handed_back", "s-9", 1))
        self.assertEqual(Workspace(self.ws).dispatch(did)["status"], "handed_back")
        from rsi.workspace import RsiError
        for call in (lambda: ws.update_dispatch("d-0042", status="done"), lambda: ws.update_dispatch(did, id="d-9"),
                     lambda: ws.update_dispatch(did, status="lost")):
            with self.assertRaises(RsiError):
                call()
        self.assertEqual(Workspace(self.ws).dispatch(did)["status"], "handed_back")

    def test_add_decision(self):
        ws = Workspace(self.ws)
        d = ws.add_decision({"point": "train_gate_failed", "options": ["train", "data"], "choice": "data",
                             "reason": "a new trainset"})
        self.assertEqual((d["round"], d["by"]), (1, "code"))
        self.assertEqual(Workspace(self.ws).decisions(point="train_gate_failed")[0]["choice"], "data")
        from rsi.workspace import RsiError
        for bad in ({"choice": "data"}, {"point": "p"}, {"point": "p", "options": ["a"], "choice": "b"}):
            with self.assertRaises(RsiError):
                ws.add_decision(bad)
        self.assertEqual(len(Workspace(self.ws).decisions()), 1)

    def test_set_and_clear_waiting(self):
        ws = Workspace(self.ws)
        w = ws.set_waiting({"kind": "approval", "what": "plan", "ref": "plan-0001", "question": "Approve?",
                            "options": ["approve", "reject"]})
        self.assertRegex(w["since"], r"Z$")
        self.assertEqual(Workspace(self.ws).waiting()["ref"], "plan-0001")
        from rsi.workspace import RsiError
        for bad in ({"kind": "chat", "question": "q", "options": ["a"]}, {"kind": "question", "options": ["a"]},
                    {"kind": "question", "question": "q", "options": []}):
            with self.assertRaises(RsiError):
                ws.set_waiting(bad)
        Workspace(self.ws).clear_waiting()
        self.assertIsNone(Workspace(self.ws).waiting())

    def test_returns_used_counts_returns_and_extra_dispatches(self):
        self.sets()
        ws = Workspace(self.ws)
        self.assertEqual(F.returns_used(ws, "train"), 0)
        self.add_dispatch("train", "done", returns=1)
        self.assertEqual(F.returns_used(Workspace(self.ws), "train"), 1)
        self.add_dispatch("train", "done")  # a fresh dispatch counts as one more
        self.assertEqual(F.returns_used(Workspace(self.ws), "train"), 2)
        self.assertFalse(F.may_return(Workspace(self.ws), "train"))
        self.assertTrue(F.may_return(Workspace(self.ws), "deliver"))


class TestTaskFlowField(unittest.TestCase):
    def test_defaults(self):
        task = fixtures.binary_task()
        self.assertEqual(flow_settings(task), {"max_returns": 2, "max_continues": 10, "checkin_minutes": None,
                                               "max_checkins": 48, "page_refresh_seconds": 60, "status_minutes": 5,
                                               "defaulted": ["checkin_minutes", "max_checkins", "max_continues",
                                                             "max_returns", "page_refresh_seconds", "status_minutes"]})
        task["flow"] = {"tune": "on_demand", "max_returns": 0, "max_continues": 3, "checkin_minutes": 30,
                        "max_checkins": 5, "page_refresh_seconds": 15, "status_minutes": 0.5}  # flow.tune: no effect (D3)
        self.assertEqual(flow_settings(task), {"max_returns": 0, "max_continues": 3, "checkin_minutes": 30,
                                               "max_checkins": 5, "page_refresh_seconds": 15, "status_minutes": 0.5,
                                               "defaulted": []})
        task["flow"] = {"max_continues": 0}
        self.assertEqual(flow_settings(task)["max_continues"], 0)

    def test_check_task(self):
        task = fixtures.binary_task()
        for flow in ({}, {"tune": "every_round"}, {"tune": "on_demand", "max_returns": 5}, {"max_returns": 0},
                     {"max_continues": 0}, {"max_continues": 25, "max_returns": 1},
                     {"checkin_minutes": 30, "max_checkins": 0}, {"checkin_minutes": 0.5}, {"checkin_minutes": None},
                     {"page_refresh_seconds": 0.5, "status_minutes": 10}):
            task["flow"] = flow
            self.assertEqual(rows.check_task(task), [], flow)
        for flow, word in (({"tune": "sometimes"}, "enum"), ({"max_returns": -1}, "minimum"),
                           ({"max_returns": 1.5}, "integer"), ({"max_returns": True}, "integer"),
                           ({"max_continues": -1}, "minimum"), ({"max_continues": 1.5}, "integer"),
                           ({"max_continues": True}, "integer"), ({"checkin_minutes": 0}, "minimum"),
                           ({"checkin_minutes": "30"}, "number"), ({"max_checkins": -1}, "minimum"),
                           ({"max_checkins": 2.5}, "integer"), ({"page_refresh_seconds": 0}, "minimum"),
                           ({"status_minutes": None}, "number"),
                           ({"retries": 2}, "retries"), ("every_round", "object")):
            task["flow"] = flow
            errors = rows.check_task(task)
            self.assertTrue(errors, flow)
            self.assertTrue(any(word in e and "flow" in e for e in errors), (flow, errors))


class TestPagesFollowFlow(FlowCase):
    def test_waiting_for_comes_from_next_action(self):
        self.sets()
        fake = {"type": "promote", "evaluation_id": "evaluation-0042", "step": "promote"}
        with mock.patch.object(D, "next_action", return_value=fake):
            self.assertIn("evaluation-0042", self.waiting())
            self.assertIn("evaluation-0042", TD.section(TD.pages(self.ws)[TD.OVERVIEW], "now"))

    def test_engine_dispatches_shown_from_state(self):
        # With dispatch records the pages show what the records say, not the directory listing.
        self.sets()
        train = self.add_dispatch("train")
        os.makedirs(os.path.join(self.ws, "work", "round-1", "judge"))  # no dispatch record behind it: ignored
        self.assertEqual(D.module_progress(Workspace(self.ws))["judge"]["state"], "idle")
        w = self.waiting()
        self.assertIn("rsi-train", w)
        self.assertNotIn("rsi-tune", w)
        self.assertNotIn("rsi-judge", w)
        st = self.assertFlow(train="current")
        self.assertIn("running", st["train"]["text"])
        self.set_status(train, "handed_back")
        self.assertIn("registered", self.waiting())
        self.assertFlow(train="current")
        # A failed dispatch with an empty output directory is no running module.
        self.set_status(train, "failed")
        self.assertEqual(D.module_progress(Workspace(self.ws))["train"]["state"], "done")
        w = self.waiting()
        self.assertIn("handback_failed", w)
        self.assertEqual(train, Workspace(self.ws).dispatches()[0]["id"])

    def test_refresh_in_a_data_entry_round_makes_data_current(self):
        # ① is done in this round (entry data); the refresh dispatch is still ①'s module: its node is current.
        task = fixtures.binary_task()
        task["eval_use_limit"] = 1
        self.use_task(task, "ws-refresh-data-entry")
        self.round_two(entry="data")
        self.handback_into(self.add_dispatch("data", "done"), "trainset", ids=range(80, 84))
        self.reg("candidate", 2, ids=VAL_IDS, train_ids=range(80, 84), trainset_id="trainset-0002")
        Workspace(self.ws).approve("refresh_sets", "alice", "refresh it")
        self.assertAction({"type": "dispatch", "module": "data", "purpose": "refresh"})
        st = self.assertFlow(data="current", deliver="todo")
        self.assertEqual(st["data"]["text"], "● refreshing evalset")
        self.assertIn("a new evalset", self.waiting())

    def test_agent_of_an_earlier_round_is_named(self):
        self.round_two(entry="train")
        self.add_dispatch("judge", "running", rnd=1)
        self.assertAction({"type": "wait_agent", "dispatch_ids": ["d-0001"]})
        self.assertIn("rsi-judge", self.waiting())
        now = TD.section(TD.pages(self.ws)[TD.OVERVIEW], "now")
        self.assertIn("⑤ Judge &amp; decide: rsi-judge running", now)
        self.assertFlow(judge="current")
        self.set_status("d-0001", "handed_back")
        self.assertIn("rsi-judge has handed back", TD.section(TD.pages(self.ws)[TD.OVERVIEW], "now"))

    def test_waiting_question_on_the_page(self):
        self.sets()
        self.set_waiting(question="Which machine to train on?", options=["a100", "h100"])
        ws = Workspace(self.ws)
        q = D.pending_questions(ws, D.module_progress(ws))
        self.assertEqual([i["text"] for i in q], ["Which machine to train on?"])
        self.assertIsNotNone(q[0]["since"])
        self.assertIn("Which machine to train on?", TD.section(TD.pages(self.ws)[TD.OVERVIEW], "ask"))
        self.assertIn("Waiting for the user", self.waiting())
        st = self.assertFlow(data="done", train="wait")
        self.assertEqual(st["train"]["text"], "⏸ waiting for the user")


class CompareCase(FlowCase):
    """Workspaces at ③ Compare, with or without a `compare` section in the task. Artifacts are registered through the
    Workspace object a test holds, so its next_action sees them."""

    def ws_passed_candidate(self, compare=None, **task_changes):
        """A new workspace (its task with `compare` when given) holding the four sets and a candidate through the
        train gate: ② is done."""
        task = fixtures.binary_task()
        task.update(task_changes)
        if compare is not None:
            task["compare"] = compare
        self.use_task(task, self.unique())
        self.sets()
        self.candidate()
        return Workspace(self.ws)

    def put(self, ws, kind, src=None, **fields):
        """Register a `kind` artifact from `src` (default a fresh source directory) into `ws`; return its id."""
        return ws.register(kind, fixtures.make_artifact(kind, src or self.src(kind), self.task,
                                                        round=ws.state["round"], **fields))

    def register_comparison(self, ws, out=None, name="lr"):
        """A baseline `name` (wrong on q0) and the comparison of the chosen candidate with it, from subdirectories of
        `out` (a dispatch's output directory) when given; return the baseline's id."""
        b = self.put(ws, "baseline", out and os.path.join(out, "bl-" + name), ids=VAL_IDS, eval_ids=EVAL_IDS,
                     wrong=(0,), name=name)
        self.put(ws, "comparison", out and os.path.join(out, "comparison"), ids=VAL_IDS,
                 candidate=F.chosen_candidate(ws), baselines={name: (b, (0,))})
        return b

    def ws_compare_dispatch_registered(self, kinds):
        """A workspace at ③ whose compare dispatch handed back and registered `kinds` (["baseline"], or with
        "comparison" too); returns (ws, the dispatch record)."""
        self.ws_passed_candidate({"baselines": ["lr"]})
        did = self.add_dispatch("compare", "done", purpose="compare")
        ws = Workspace(self.ws)
        out = os.path.join(self.ws, ws.dispatch(did)["outdir"])
        if "comparison" in kinds:
            self.register_comparison(ws, out)
        else:
            self.put(ws, "baseline", os.path.join(out, "bl-lr"), ids=VAL_IDS, eval_ids=EVAL_IDS, wrong=(0,))
        return ws, ws.dispatch(did)

    def ws_milestone_asked(self, compare=None, skipped=False, name="lr"):
        """Round 1 compared and promoted its candidate (V1) and the stop check says stop: the milestone is asked.
        skipped: the comparison lists no baseline (each was skipped); else its one baseline is `name`."""
        ws = self.ws_passed_candidate(compare, stop=dict(fixtures.binary_task()["stop"], target=0.8))
        cand = F.chosen_candidate(ws)
        scored = {}  # the judge reports the comparison's baseline on the evalset (baselines_eval)
        if skipped:
            self.put(ws, "comparison", ids=VAL_IDS, candidate=cand, baselines={},
                     skipped={"zero_shot": "needs 0.1 GPU hours, 0 left"})
        else:
            scored[self.register_comparison(ws, name=name)] = (name, (0,))
        dlv = self.put(ws, "delivery", ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand)
        ev = self.put(ws, "evaluation", ids=EVAL_IDS, train_ids=TRAIN_IDS, delivery_id=dlv, candidate_id=cand,
                      evalset_id="evalset-0001", best_before=None, baselines=scored)
        self.assertEqual(ws.promote(ev)[:2], ("promote", "V1"))
        return ws

    def milestone_yes_with_delivery(self):
        """The user says yes to the milestone and the milestone delivery is registered."""
        self.decide("milestone", "yes")
        milestone_delivery(self)
        return Workspace(self.ws)

    def register_milestone_baseline(self, ws):
        """The comparison's baseline lr scored on the testset inputs (its use_testset approval given, as the engine
        writes it); returns its id."""
        ws.approve("use_testset", "alice", "yes, validate it on the test inputs")
        src = fixtures.make_artifact("baseline", self.src("baseline"), self.task, round=ws.state["round"], ids=VAL_IDS,
                                     eval_ids=EVAL_IDS, wrong=(0,), reused_from="baseline-0001")
        fixtures.add_predictions(src, self.task, "predictions_test", TEST_IDS, testset_id="testset-0001")
        return ws.register("baseline", src)


class TestCompareStep(CompareCase):
    def test_compare_comes_after_train_and_before_deliver(self):
        ws = self.ws_passed_candidate(compare={"baselines": ["lr"]})
        a = F.next_action(ws)
        self.assertEqual((a["type"], a["module"], a["purpose"], a["step"]),
                         ("dispatch", "compare", "compare", "compare"))
        self.assertEqual(a["candidate_id"], F.chosen_candidate(ws))
        self.register_comparison(ws)                                  # baseline + comparison of the chosen candidate
        self.assertEqual(F.next_action(ws)["module"], "deliver")
        self.assertEqual(F.next_action(self.ws_passed_candidate())["module"], "deliver")   # no compare section

    def test_the_compare_section_of_the_task(self):
        # Any lower-case name passes (the mock compare's prior); defaults are filled in by compare_settings only.
        task = fixtures.binary_task()
        self.assertIsNone(compare_settings(task))
        for compare in ({"baselines": ["prior"]},
                        {"baselines": ["lr", "zero_shot"], "team_recipe": dict(fixtures.CONFIG_REQUEST)}):
            task["compare"] = compare
            self.assertEqual(rows.check_task(task), [], compare)
        self.assertEqual(compare_settings(task)["gpu_hours"], {"zero_shot": 0.1, "few_shot": 0.1, "predict": 0.1})
        for compare, word in (({"baselines": []}, "minItems"), ({"baselines": ["LR"]}, "pattern"),
                              ({"baselines": ["lr", "lr"]}, "named once"), ({"seed": 0}, "baselines"),
                              ({"baselines": ["lr"], "few_shot": {"k": 0}}, "minimum")):
            task["compare"] = compare
            errors = rows.check_task(task)
            self.assertTrue(any(word in e and "compare" in e for e in errors), (compare, errors))

    def test_compare_baselines_then_comparison_in_the_same_session(self):
        ws, d = self.ws_compare_dispatch_registered(["baseline"])
        a = F.next_action(ws)
        self.assertEqual((a["type"], a["mode"], a["dispatch_id"]), ("resume", "followup", d["id"]))

    def test_milestone_compare_between_milestone_deliver_and_judge(self):
        q = F.next_action(self.ws_milestone_asked(compare={"baselines": ["lr"]}))["question"]
        self.assertIn("up to 3 times", q)  # deliver, the comparison's one baseline, judge
        self.assertIn("rsi-compare scores 1 baseline of the best version's comparison comparison-0001", q)
        self.assertNotIn("GPU hours", q)                                  # lr is no GPU baseline
        ws = self.milestone_yes_with_delivery()   # best version has a comparison
        a = F.next_action(ws)
        self.assertEqual((a["module"], a["purpose"], a["step"]), ("compare", "milestone_compare", "milestone"))
        b = B.build_brief(ws, "compare", purpose="milestone_compare")
        self.assertTrue(b["inputs"]["testset_inputs"].endswith("inputs.jsonl"))
        self.assertIn(fixtures.labels_path(ws, "testset"), B.denied_paths(ws, b))
        self.register_milestone_baseline(ws)
        self.assertEqual(F.next_action(ws)["purpose"], "milestone_judge")
        q = F.next_action(self.ws_milestone_asked(compare={"baselines": ["team_recipe"]}, name="team_recipe"))
        self.assertIn("may use GPU hours beyond the compute budget", q["question"])   # predict-only, not held by it

    def test_milestone_has_no_compare_step_when_the_comparison_lists_no_baseline(self):
        q = F.next_action(self.ws_milestone_asked(compare={"baselines": ["zero_shot"]}, skipped=True))["question"]
        self.assertIn("twice", q)
        self.assertNotIn("rsi-compare", q)
        self.assertEqual(F.next_action(self.milestone_yes_with_delivery())["purpose"], "milestone_judge")


if __name__ == "__main__":
    unittest.main()
