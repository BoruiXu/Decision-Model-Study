"""Decision points, returns and asking the user (rsi/decisions.py): the code's default at each decision point and
its caps, an agent's choice (task decisions.<point>: "agent", MockRunner's decide.py), a refused registration
decided inside the hand-back, the exits every question offers (each answer changes the state), approvals with
explicit options, `rsi answer` and `rsi approve`, the milestone consent recorded as two use_testset approvals,
answers to a module's questions reaching its next resume, the long-job continuation cap, and the STATE lock that
lets `rsi approve` / `rsi answer` / `rsi stop` write while `rsi run` works."""
import contextlib
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import fixtures
import test_engine as TE
import test_flow as TF
from rsi import cli
from rsi import decisions as DP
from rsi import engine
from rsi import flow as F
from rsi import rows
from rsi import workspace as W
from rsi.rules import flow_settings
from rsi.runners import ClaudeCodeRunner, ManualRunner, MockRunner
from rsi.runners.claude import structured_output
from rsi.workspace import RsiError, Workspace

MOCKS, REPO, ROUND = TE.MOCKS, TE.REPO, TE.ROUND
EXITS = ["resume", "new_session", "close_round", "stop"]


# ---- the code's default at each decision point -----------------------------------------------------

class TestDefaults(TF.FlowCase):
    def default(self, a=None):
        a = a or self.action()
        return DP.default_choice(Workspace(self.ws), a["point"], {"dispatch_id": a["dispatch_id"]})

    def fresh(self, builder, **flow):
        task = fixtures.binary_task()
        if flow:
            task["flow"] = flow
        self.use_task(task, self.unique())
        return builder(self)

    def test_decision_points(self):
        self.assertEqual(sorted(DP.DECISION_POINTS), sorted([F.TRAIN_GATE_FAILED, F.QUALITY_GATE_FAILED,
                                                             F.HANDBACK_REFUSED, F.HANDBACK_FAILED]))
        self.assertEqual(DP.DECISION_POINTS[F.TRAIN_GATE_FAILED]["options"], ("train", "data", "ask"))
        self.assertEqual(DP.DECISION_POINTS[F.QUALITY_GATE_FAILED]["options"], ("resume_data", "ask"))
        self.assertEqual(DP.DECISION_POINTS[F.HANDBACK_REFUSED]["options"], ("resume", "ask"))
        self.assertEqual(DP.DECISION_POINTS[F.HANDBACK_FAILED]["options"], ("resume", "ask"))
        for p in DP.DECISION_POINTS.values():
            self.assertTrue(p["text"])

    def test_train_gate_failed(self):
        # Back to train for a new search (also with no dispatch record); train at its cap -> ask.
        for builder, did, want in ((TF.b_train_gate_failed, None, "train"),
                                   (TF.b_train_gate_failed_after_search, "d-0001", "train"),
                                   (TF.b_train_gate_failed_at_the_cap, "d-0001", "ask")):
            with self.subTest(builder.__name__):
                self.fresh(builder)
                a = self.action()
                self.assertEqual((a["point"], a["dispatch_id"]), (F.TRAIN_GATE_FAILED, did))
                choice, why = self.default(a)
                self.assertEqual((choice, a["default"]), (want, want))
                self.assertIn({"train": "another search", "ask": "at the cap"}[want], why)

    def test_train_gate_failed_asks_after_a_record_only_return(self):
        # F17: train's latest launch registered only an experiment record (trsi said STOP): another return would only
        # hand back a record again, so the code asks.
        self.fresh(TF.b_train_gate_failed_after_search)
        Workspace(self.ws).update_dispatch("d-0001", launches=2, registered=[
            {"dir": "record", "kind": "record", "id": "record-0001", "launch": 2}])
        choice, why = DP.default_choice(Workspace(self.ws), F.TRAIN_GATE_FAILED, {"dispatch_id": "d-0001"})
        self.assertEqual(choice, "ask")
        self.assertIn("experiment records", why)

    def test_quality_gate_failed(self):
        self.fresh(TF.b_quality_gate_failed)  # no dispatch record to send it back to
        self.assertEqual(self.default()[0], "ask")
        self.fresh(TF.b_quality_gate_failed_from_dispatch)
        self.assertEqual(self.default(), ("resume_data", self.default()[1]))
        self.assertIn("rsi-data", self.default()[1])
        self.fresh(TF.b_quality_gate_failed_from_dispatch, max_returns=0)
        choice, why = self.default()
        self.assertEqual(choice, "ask")
        self.assertIn("max_returns", why)

    def test_handback_failed(self):
        self.fresh(TF.b_failed_dispatch)
        self.assertEqual(self.default()[0], "resume")
        self.fresh(TF.b_failed_dispatch_returns_used)
        self.assertEqual(self.default()[0], "ask")
        # A long job stopped at the continuation cap is the user's call, whatever the returns.
        self.fresh(TF.b_failed_dispatch)
        Workspace(self.ws).update_dispatch("d-0001", capped="max_continues")
        a = self.action()
        choice, why = self.default(a)
        self.assertEqual((choice, a["default"]), ("ask", "ask"))
        self.assertIn("max_continues", why)

    def test_allowed_choices_and_exit_options(self):
        self.fresh(TF.b_train_gate_failed_after_search)
        ws = Workspace(self.ws)
        self.assertEqual(DP.allowed(ws, F.TRAIN_GATE_FAILED, "d-0001"), ["train", "data", "ask"])
        self.fresh(TF.b_train_gate_failed_at_the_cap)  # train sent back twice: at its cap
        self.assertEqual(DP.allowed(Workspace(self.ws), F.TRAIN_GATE_FAILED, "d-0001"), ["data", "ask"])
        self.fresh(TF.b_failed_dispatch_returns_used)
        self.assertEqual(DP.allowed(Workspace(self.ws), F.HANDBACK_FAILED, "d-0001"), ["ask"])
        self.fresh(TF.b_quality_gate_failed)
        self.assertEqual(DP.allowed(Workspace(self.ws), F.QUALITY_GATE_FAILED, None), ["ask"])
        # The exits every question about a decision offers.
        self.fresh(TF.b_failed_dispatch)
        ws = Workspace(self.ws)
        self.assertEqual(DP.exit_options(ws, F.HANDBACK_FAILED, ws.dispatch("d-0001")), EXITS)
        self.fresh(TF.b_quality_gate_failed)
        ws = Workspace(self.ws)
        self.assertEqual(DP.exit_options(ws, F.QUALITY_GATE_FAILED, None), ["new_session", "close_round", "stop"])
        self.fresh(TF.b_train_gate_failed_after_search)
        ws = Workspace(self.ws)
        self.assertEqual(DP.exit_options(ws, F.TRAIN_GATE_FAILED, ws.dispatch("d-0001")),
                         ["train", "data", "new_session", "close_round", "stop"])
        # A decided round cannot be closed: the judge's plan failing offers no close_round.
        self.fresh(TF.b_decided_judge_returns_used)
        ws = Workspace(self.ws)
        self.assertEqual(DP.exit_options(ws, F.HANDBACK_FAILED, ws.dispatch("d-0001")),
                         ["resume", "new_session", "stop"])


class TestSkipCompare(TF.CompareCase):
    def ws_failed_compare(self, returns=0):
        """A workspace at ③ whose compare dispatch failed after `returns` returns."""
        self.ws_passed_candidate({"baselines": ["lr"]})
        did = self.add_dispatch("compare", "failed", purpose="compare", returns=returns)
        ws = Workspace(self.ws)
        return ws, ws.dispatch(did)

    def test_a_failed_compare_is_skipped_at_the_cap(self):
        ws, d = self.ws_failed_compare(returns=2)                     # flow.max_returns default 2
        a = F.next_action(ws)
        self.assertEqual((a["point"], a["default"]), ("handback_failed", "skip"))
        self.assertEqual(a["options"], ["resume", "skip", "ask"])
        self.assertEqual(DP.exit_options(ws, a["point"], d), ["resume", "skip", "new_session", "close_round", "stop"])
        DP.apply(fixtures.FakeEngine(ws), ws, a)
        d = Workspace(ws.root).dispatch(d["id"])
        self.assertEqual((d["status"], d["closed"]), ("done", "skipped: ③ Compare is report only"))
        self.assertEqual(F.next_action(Workspace(ws.root))["module"], "deliver")
        # Capped by flow.max_continues, it is skipped too: ③ never stops the run to ask.
        ws, d = self.ws_failed_compare()
        ws.update_dispatch(d["id"], capped="max_continues")
        self.assertEqual(F.next_action(ws)["default"], "skip")
        # Asked about a done dispatch (baselines only, its followup spent, at the cap), the user's skip closes it at once.
        ws, d = self.ws_compare_dispatch_registered(["baseline"])
        ws.update_dispatch(d["id"], launch_mode="followup", returns=2)
        a = F.next_action(ws)
        DP.carry_out(fixtures.FakeEngine(ws), ws, a, {"choice": "ask", "by": "agent", "reason": "the user decides"})
        self.assertIn("skip", Workspace(ws.root).waiting()["options"])
        self.assertIn("closed as skipped", DP.answer(Workspace(ws.root), "skip", "alice", "no comparison")["effect"])
        d = Workspace(ws.root).dispatch(d["id"])
        self.assertEqual((d["status"], d["closed"]), ("done", "skipped: ③ Compare is report only"))
        self.assertEqual(F.next_action(Workspace(ws.root))["module"], "deliver")


# ---- the engine with decisions, answers and approvals ----------------------------------------------

class DecisionCase(TE.EngineCase):
    def answer(self, option, quote="do it this way", by="alice"):
        return DP.answer(Workspace(self.ws), option, by, quote)

    def waiting(self):
        return Workspace(self.ws).waiting()

    def decisions(self):
        return [(d["point"], d["choice"], d["by"]) for d in Workspace(self.ws).decisions()]

    def events(self, name):
        return [r for r in self.log() if r["event"] == name]

    def status(self, runner=None, **kw):
        return engine.run_status(self.ws, runner or MockRunner(MOCKS), **kw)

    def train_fails(self, **flow):
        """A one-round task whose train module hands back status failed, with no returns allowed: the run asks."""
        self.use_task(TE.task_def(max_returns=0, **flow), "ws-tf")
        self.behave("train", fail="out of GPU memory, the training did not finish")
        status, reason = self.status()
        self.assertEqual(status, "waiting_user")
        w = self.waiting()
        self.assertEqual((w["kind"], w["what"], w["ref"], w["options"]), ("question", F.HANDBACK_FAILED, "d-0002", EXITS))
        self.assertIn("out of GPU memory", w["question"])
        for option in EXITS:
            self.assertIn(option + ": ", w["question"])  # every option says what it does
        self.assertEqual(self.decisions(), [(F.HANDBACK_FAILED, "ask", "code")])
        return reason


class TestAgentDecisions(DecisionCase):
    def agent_task(self, name, **flow):
        t = TE.task_def(**flow)
        t["decisions"] = {F.QUALITY_GATE_FAILED: "agent", F.TRAIN_GATE_FAILED: "agent"}
        self.use_task(t, name)

    def requests(self):
        out = []
        for p in sorted(glob.glob(os.path.join(self.ws, "work", "decisions", "*", "request.json"))):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        return out

    def test_task_decisions_setting(self):
        task = fixtures.binary_task()
        for value in ({}, {F.HANDBACK_REFUSED: "agent"}, {F.TRAIN_GATE_FAILED: "code", F.HANDBACK_FAILED: "agent"}):
            task["decisions"] = value
            self.assertEqual(rows.check_task(task), [], value)
        for value, word in (({F.HANDBACK_FAILED: "llm"}, "enum"), ({"milestone": "agent"}, "milestone"),
                            ("agent", "object")):
            task["decisions"] = value
            errors = rows.check_task(task)
            self.assertTrue(any(word in e and "decisions" in e for e in errors), (value, errors))

    def test_agent_asks_the_user(self):
        self.agent_task("ws-a1")
        self.behave("data", trainset="quality_fail")
        self.behave("decide", **{F.QUALITY_GATE_FAILED: {"choice": "ask", "reason": "too many duplicates, have a look"}})
        status, reason = self.status()
        self.assertEqual(status, "waiting_user")
        ws = Workspace(self.ws)
        d = ws.decisions()[0]
        self.assertEqual((d["point"], d["choice"], d["by"]), (F.QUALITY_GATE_FAILED, "ask", "agent"))
        self.assertIn("too many duplicates", d["reason"])
        w = ws.waiting()
        self.assertEqual((w["what"], w["ref"], w["options"]), (F.QUALITY_GATE_FAILED, "d-0001", EXITS))
        req = self.requests()
        self.assertEqual(len(req), 1)
        self.assertEqual((req[0]["point"], req[0]["options"], req[0]["default"]),
                         (F.QUALITY_GATE_FAILED, ["resume_data", "ask"], "resume_data"))
        self.assertEqual(req[0]["schema"]["properties"]["choice"]["enum"], ["resume_data", "ask"])
        self.assertEqual(req[0]["schema"]["required"], ["choice", "reason"])
        for word in ("dup_rate", "0.05", "resume_data", "d-0001"):
            self.assertIn(word, req[0]["prompt"])
        self.assertIn(F.QUALITY_GATE_FAILED, [r["point"] for r in self.events("decision")])

    def test_agent_choice_is_carried_out(self):
        # The agent sends the trials back to train's session, with its reason.
        self.agent_task("ws-a2")
        self.behave("train", gate_fail=True)
        self.behave("decide", **{F.TRAIN_GATE_FAILED: {"choice": "train", "reason": "search again with other configs",
                                                       "cost_usd": 0.5}})
        self.assertMilestoneAsk(self.run_engine())
        self.assertEqual(self.decisions(), [(F.TRAIN_GATE_FAILED, "train", "agent")])
        ws = Workspace(self.ws)
        self.assertEqual(ws.used("agent_usd"), 0.5)  # the decision agent's session, counted once
        self.assertEqual([e["cost_usd"] for e in self.events("decision_agent")], [0.5])
        self.assertEqual([d["module"] for d in ws.dispatches()], ["data", "train", "deliver", "judge"])
        self.assertEqual(ws.dispatch("d-0002")["returns"], 1)
        self.assertIn("search again with other configs", self.calls("d-0002")[2]["feedback"])

    def test_invalid_agent_answers_fall_back_to_the_code_default(self):
        for i, answer in enumerate(({"raw": "I think it should be redone"}, {"choice": "train", "reason": "x"},
                                    {"choice": "resume_data", "reason": ""})):
            with self.subTest(answer):
                self.agent_task("ws-bad%d" % i)
                self.behave("data", trainset="quality_fail")
                self.behave("decide", **{F.QUALITY_GATE_FAILED: answer})
                self.assertMilestoneAsk(self.run_engine())
                d = Workspace(self.ws).decisions()[0]
                self.assertEqual((d["choice"], d["by"]), ("resume_data", "code"))
                self.assertIn("agent", d["reason"])
                self.assertTrue(self.events("decision_agent_failed"))

    def test_agent_cannot_pass_the_caps(self):
        # One return each: data was sent back once (the quality gate), so after the failed search only train or ask
        # remain. The agent's "data" is refused and the code's default (train) taken.
        self.agent_task("ws-a5", max_returns=1)
        self.behave("data", trainset="quality_fail")
        self.behave("train", gate_fail=True)
        self.behave("decide", **{F.TRAIN_GATE_FAILED: {"choice": "data", "reason": "more data"}})
        self.assertMilestoneAsk(self.run_engine())
        self.assertEqual(self.decisions(), [(F.QUALITY_GATE_FAILED, "resume_data", "agent"),
                                            (F.TRAIN_GATE_FAILED, "train", "code")])
        self.assertIn("agent", Workspace(self.ws).decisions()[1]["reason"])
        req = self.requests()
        self.assertEqual([r["options"] for r in req], [["resume_data", "ask"], ["train", "ask"]])
        self.assertEqual(Workspace(self.ws).dispatch("d-0002")["returns"], 1)

    def test_runner_without_a_decision_agent(self):
        self.agent_task("ws-a4")
        self.behave("data", trainset="quality_fail")
        self.status(max_steps=3)  # dispatch, wait, hand-back: the decision is next
        ws = Workspace(self.ws)
        a = F.next_action(ws)
        self.assertEqual(a["point"], F.QUALITY_GATE_FAILED)
        eng = engine.Engine(self.ws, ManualRunner(), pages=False)
        choice, by, reason = DP.choose(eng, ws, a)
        self.assertEqual((choice, by), ("resume_data", "code"))
        self.assertIn("manual", reason)

    def test_claude_runner_decision_call(self):
        # Only the command line and the output parsing: the real `claude` is never run.
        r = ClaudeCodeRunner(args=["--model", "opus"])
        schema = DP.choice_schema(["train", "ask"])
        argv = r.decision_argv(schema)
        self.assertEqual(argv[:-2], ["claude", "-p", "--dangerously-skip-permissions", "--output-format", "json",
                                     "--effort", "xhigh", "--model", "opus"])
        self.assertEqual((argv[-2], json.loads(argv[-1])), ("--json-schema", schema))
        denied = r.decision_argv(schema, ["Read(//a/**)", "Edit(//a/**)"])
        self.assertEqual(denied[-3:], ["--disallowedTools", "Read(//a/**)", "Edit(//a/**)"])
        self.assertEqual(denied[-5], "--json-schema")
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        log = os.path.join(tmp, "agent.log")
        with open(log, "w") as f:
            f.write(json.dumps({"type": "result", "subtype": "success", "session_id": "s1", "result": "done",
                                "structured_output": {"choice": "train", "reason": "r"}}) + "\n")
        self.assertEqual(structured_output(log), {"choice": "train", "reason": "r"})
        with open(log, "w") as f:
            f.write(json.dumps({"type": "result", "session_id": "s1", "result": '{"choice": "ask", "reason": "q"}'}))
        self.assertEqual(structured_output(log), {"choice": "ask", "reason": "q"})
        with open(log, "w") as f:
            f.write("no json at all\n")
        self.assertIsNone(structured_output(log))


class TestRefused(DecisionCase):
    def test_refusal_at_the_cap_asks_then_the_answer_resumes(self):
        self.use_task(TE.task_def(max_returns=0), "ws-rc")
        self.behave("data", trainset="overlap")
        self.assertEqual(self.status()[0], "waiting_user")
        w = self.waiting()
        self.assertEqual((w["what"], w["ref"], w["options"]), (F.HANDBACK_REFUSED, "d-0001", EXITS))
        self.assertIn("overlaps the current evalset", w["question"])
        self.assertEqual(Workspace(self.ws).dispatch("d-0001")["status"], "failed")
        self.answer("resume", "trainset: take rows that do not overlap the evalset")
        self.assertMilestoneAsk(self.run_engine())
        self.assertEqual(self.decisions()[:2], [(F.HANDBACK_REFUSED, "ask", "code"),
                                                (F.HANDBACK_REFUSED, "resume", "alice")])
        calls = self.calls("d-0001")
        self.assertEqual(calls[1]["session"], calls[0]["session"])
        self.assertIn("trainset: take rows that do not overlap the evalset", calls[1]["feedback"])
        self.assertIn("overlaps the current evalset", calls[1]["feedback"])
        self.assertEqual(Workspace(self.ws).dispatch("d-0001")["returns"], 1)


class TestExits(DecisionCase):
    def test_resume_sends_the_users_words_to_the_same_session(self):
        self.train_fails()
        rec = self.answer("resume", "run again with a smaller batch")
        self.assertEqual((rec["by"], rec["quote"], rec["dispatch_id"]), ("alice", "run again with a smaller batch", "d-0002"))
        self.assertIsNone(self.waiting())
        # The flow still meets the same decision; the answer, not the earlier ask, is what the engine carries out.
        self.assertEqual(F.next_action(Workspace(self.ws))["point"], F.HANDBACK_FAILED)
        self.assertMilestoneAsk(self.run_engine())
        self.assertEqual(self.decisions()[:2], [(F.HANDBACK_FAILED, "ask", "code"), (F.HANDBACK_FAILED, "resume", "alice")])
        self.assertEqual(len(Workspace(self.ws).decisions()), 2)
        d = Workspace(self.ws).dispatch("d-0002")
        self.assertEqual((d["status"], d["returns"]), ("done", 1))
        calls = self.calls("d-0002")  # the failed launch, the return, the search's second part
        self.assertEqual([c["session"] for c in calls], [d["session"]] * 3)
        self.assertIn("run again with a smaller batch", calls[1]["feedback"])
        self.assertIn("out of GPU memory", calls[1]["feedback"])
        self.assertTrue(self.events("answer"))

    def test_new_session_closes_the_failed_dispatch(self):
        self.train_fails()
        self.answer("new_session", "start over in a new session")
        self.status()
        ws = Workspace(self.ws)
        old, new = ws.dispatch("d-0002"), ws.dispatch("d-0003")
        self.assertEqual(old["status"], "done")
        self.assertIn("new_session", old["closed"])
        self.assertEqual((new["module"], new["purpose"]), ("train", "train"))
        self.assertNotEqual(new["session"], old["session"])
        call = self.calls("d-0003")[0]
        self.assertEqual(call["feedback_file"], "feedback-1.md")
        self.assertIn("start over in a new session", call["feedback"])
        self.assertIn("out of GPU memory", call["feedback"])
        # The mock fails again in its new directory: the question is now about the new dispatch.
        self.assertEqual(self.waiting()["ref"], "d-0003")

    def test_close_round(self):
        self.train_fails()
        self.answer("close_round", "drop this round")
        ws = Workspace(self.ws)
        decision = ws.state["rounds"]["1"]["decision"]
        self.assertEqual((decision["decision"], decision["evaluation_id"]), ("keep", None))
        self.assertIn("drop this round", decision["reason"])
        self.assertEqual(ws.dispatch("d-0002")["status"], "done")
        self.assertIsNone(ws.waiting())
        status, reason = self.status()  # one round: the stop check stops, no version to release
        self.assertEqual(status, "done")
        self.assertEqual(self.actions()[-3:], ["next", "report", "done"])

    def test_a_failed_answer_transaction_leaves_the_job_alive(self):
        # F14: the answer's transaction fails after it closed the dispatch: nothing is saved, and the long job, ended
        # only once the answer is saved, still runs.
        pid, w = TE.failed_check_in(self, "ws-ft")
        with mock.patch.object(Workspace, "close_round", side_effect=RsiError("boom")):
            with self.assertRaises(RsiError):
                DP.answer(Workspace(self.ws), "close_round", "alice", "drop this round", ref=w["id"])
        self.assertTrue(TE.alive(pid))
        self.assertEqual(Workspace(self.ws).dispatch("d-0002")["status"], "failed")

    def gate_cap(self):
        # No returns allowed: the first search's trials all fail the gate; the question at the cap is about
        # rsi-train's dispatch (d-0002).
        self.use_task(TE.task_def(max_returns=0), "ws-gc")
        self.behave("train", gate_fail=True)
        self.assertEqual(self.status()[0], "waiting_user")
        ws = Workspace(self.ws)
        self.assertEqual([(x["point"], x["choice"], x["by"], x["dispatch_id"]) for x in ws.decisions()],
                         [(F.TRAIN_GATE_FAILED, "ask", "code", "d-0002")])
        self.assertEqual(ws.dispatch("d-0002")["module"], "train")
        w = self.waiting()
        self.assertEqual((w["what"], w["ref"], w["options"]),
                         (F.TRAIN_GATE_FAILED, "d-0002", ["train", "data", "new_session", "close_round", "stop"]))
        self.assertIn("new_session: dispatch a new rsi-train session", w["question"])

    def test_train_gate_answer_dispatches_with_the_choice_purpose(self):
        self.gate_cap()
        self.answer("data", "the trainset is too small, add data")
        self.status()
        ws = Workspace(self.ws)
        d = ws.dispatch("d-0003")
        self.assertEqual((d["module"], d["purpose"]), ("data", F.CHOICE_PURPOSE["data"]))
        call = self.calls("d-0003")[0]
        self.assertEqual(call["purpose"], "trainset")
        self.assertIn("the trainset is too small, add data", call["feedback"])
        self.assertEqual(ws.current("trainset"), "trainset-0002")
        # The new trainset changed the state, so the gate's next failure is a new question.
        self.assertEqual(self.decisions()[1:], [(F.TRAIN_GATE_FAILED, "data", "alice"),
                                                (F.TRAIN_GATE_FAILED, "ask", "code")])
        self.assertEqual(self.waiting()["what"], F.TRAIN_GATE_FAILED)

    def test_train_gate_answer_train_goes_back_to_the_same_session(self):
        self.gate_cap()
        self.answer("train", "halve the learning rate")
        self.assertEqual(self.status()[0], "waiting_user")
        ws = Workspace(self.ws)
        d = ws.dispatch("d-0002")
        self.assertEqual(d["returns"], 1)  # past the cap: the user's choice
        self.assertEqual(ws.passed_candidates(), ["candidate-0002"])
        calls = self.calls("d-0002")
        self.assertEqual(calls[2]["session"], calls[0]["session"])
        self.assertIn("halve the learning rate", calls[2]["feedback"])
        # Each search's second part is a followup of the session and no return, so it is asked for although no
        # returns are allowed; the run goes on to the milestone question.
        self.assertEqual((d["followups"], d["launch_mode"]), (2, "followup"))
        self.assertEqual(ws.attempts(), ["search-0001", "search-0002"])
        self.assertEqual(ws.waiting()["what"], F.MILESTONE)

    def test_answer_given_before_other_progress_is_asked_again(self):
        # The answer belongs to the state it was given in: after other progress the question comes back.
        self.train_fails()
        self.answer("resume", "try once more")
        src = fixtures.make_artifact("record", os.path.join(self.root, "rec"), self.task, artifacts=["trainset-0001"])
        Workspace(self.ws).register("record", src)
        self.assertEqual(self.status()[0], "waiting_user")
        self.assertEqual(self.waiting()["ref"], "d-0002")
        self.assertEqual(Workspace(self.ws).dispatch("d-0002")["returns"], 0)
        self.assertEqual(self.decisions()[-1], (F.HANDBACK_FAILED, "ask", "code"))


class TestAnswer(DecisionCase):
    def test_validation(self):
        with self.assertRaises(RsiError) as e:
            self.answer("resume")
        self.assertIn("nothing is waiting", str(e.exception))
        self.train_fails()
        with self.assertRaises(RsiError) as e:
            self.answer("approve")
        for option in EXITS:
            self.assertIn(option, str(e.exception))
        for by, quote in (("alice", "  "), ("", "ok")):
            with self.assertRaises(RsiError):
                self.answer("resume", quote, by)
        self.assertEqual(self.waiting()["what"], F.HANDBACK_FAILED)  # nothing recorded
        self.assertEqual(len(Workspace(self.ws).decisions()), 1)

    def test_stale_approval_is_refused(self):
        ws = Workspace(self.ws)
        ws.set_waiting({"kind": "approval", "what": "refresh_sets", "question": "q", "options": ["approve", "stop"]})
        ws.approve("refresh_sets", "bob", "ok")  # Workspace.approve clears the matching wait
        self.assertIsNone(Workspace(self.ws).waiting())
        ws.set_waiting({"kind": "approval", "what": "refresh_sets", "question": "q", "options": ["approve", "stop"],
                        "since": "2000-01-01T00:00:00Z"})
        with self.assertRaises(RsiError) as e:
            self.answer("approve")
        self.assertIn("already", str(e.exception))

    def test_cli(self):
        # The stop exit, answered from the command line: the run stops and the failed dispatch is closed.
        self.train_fails()
        r = subprocess.run([sys.executable, "-m", "rsi", "answer", self.ws, "stop", "--by", "alice", "--quote",
                            "stop it"], cwd=REPO, capture_output=True, text=True, env=dict(os.environ, RSI_NO_BROWSER="1"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)
        self.assertIn("stop", r.stdout)
        ws = Workspace(self.ws)
        self.assertEqual((ws.state["stopped"]["reason"], ws.state["stopped"]["by"]), ("user requested: stop it", "alice"))
        self.assertEqual(ws.dispatch("d-0002")["status"], "done")
        last = self.log()[-1]
        self.assertEqual((last["event"], last["choice"], last["by"], last["quote"]), ("answer", "stop", "alice", "stop it"))
        self.assertEqual(self.status()[0], "done")
        r = subprocess.run([sys.executable, "-m", "rsi", "answer", self.ws, "stop", "--by", "alice", "--quote", "x"],
                           cwd=REPO, capture_output=True, text=True, env=dict(os.environ, RSI_NO_BROWSER="1"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("nothing is waiting", r.stderr)


class TestApprovals(DecisionCase):
    def plan_task(self):
        t = TE.task_def(max_rounds=2)
        t["approvals"].append("plan")
        self.use_task(t, "ws-plan")
        self.assertEqual(self.status()[0], "waiting_user")
        w = self.waiting()
        self.assertEqual((w["kind"], w["what"], w["ref"], w["options"]), ("approval", "plan", "plan-0001",
                                                                          ["approve", "revise"]))

    def test_plan_revise_then_approve(self):
        self.plan_task()
        judge = Workspace(self.ws).dispatch("d-0004")
        self.assertEqual((judge["module"], judge["returns"], judge["followups"]), ("judge", 0, 1))  # its plan part
        rec = self.answer("revise", "add a slice for short messages to the plan")
        self.assertEqual((rec["point"], rec["choice"], rec["ref"], rec["dispatch_id"]),
                         ("plan", "revise", "plan-0001", "d-0004"))
        self.assertIsNone(self.waiting())
        self.assertEqual(Workspace(self.ws).state["approvals"], [])
        self.assertEqual(self.status()[0], "waiting_user")
        ws = Workspace(self.ws)
        self.assertEqual(ws.waiting()["ref"], "plan-0002")  # the revised plan, asked again
        judge = ws.dispatch("d-0004")  # the revision is the judge's only return (its plan part is a followup)
        self.assertEqual((judge["launches"], judge["returns"], judge["followups"]), (3, 1, 1))
        self.assertEqual(F.returns_used(ws, "judge"), 1)
        self.assertTrue(F.may_return(ws, "judge"))
        calls = self.calls("d-0004")
        self.assertEqual([c["session"] for c in calls], [judge["session"]] * 3)
        self.assertIn("add a slice for short messages to the plan", calls[2]["feedback"])
        self.assertIn("plan-0001", calls[2]["feedback"])
        self.answer("approve", "this version is fine")
        ws = Workspace(self.ws)
        a = ws.state["approvals"][-1]
        self.assertEqual((a["what"], a["ref"], a["by"], a["quote"]), ("plan", "plan-0002", "alice", "this version is fine"))
        self.assertEqual(self.decisions()[-2:], [("plan", "revise", "alice"), ("plan", "approve", "alice")])
        self.status(max_steps=1)
        self.assertEqual(Workspace(self.ws).state["round"], 2)

    def test_rsi_approve_clears_a_matching_waiting(self):
        self.plan_task()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["approve", self.ws, "over_budget", "--by", "alice", "--quote",
                                       "going over is fine"]), 0)
        self.assertEqual(self.waiting()["what"], "plan")  # another item: the plan question stays
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["approve", self.ws, "plan", "--ref", "plan-0001", "--by", "alice", "--quote",
                                       "agreed"]), 0)
        self.assertIsNone(self.waiting())
        self.assertEqual(self.decisions()[-1], ("plan", "approve", "alice"))
        self.assertTrue([r for r in self.events("approve") if r["what"] == "plan" and r["answered"]])
        self.assertEqual(F.next_action(Workspace(self.ws))["type"], "next")

    def test_milestone_yes_records_each_use_testset_approval_before_its_registration(self):
        self.assertMilestoneAsk(self.run_engine())
        self.answer("yes", "check it on the sealed testset")
        status, reason = self.status()
        self.assertEqual(status, "done", reason)
        ws = Workspace(self.ws)
        approvals = [a for a in ws.state["approvals"] if a["what"] == "use_testset"]
        self.assertEqual([(a["by"], a["quote"]) for a in approvals], [("alice", "check it on the sealed testset")] * 2)
        self.assertEqual([a["used_by"] for a in approvals], ["delivery-0002", "evaluation-0002"])
        order = [(r["event"], r.get("what") or r.get("id")) for r in self.log() if r["event"] in ("approval", "register")]
        self.assertEqual(order[-4:], [("approval", "use_testset"), ("register", "delivery-0002"),
                                      ("approval", "use_testset"), ("register", "evaluation-0002")])
        self.assertEqual([d["purpose"] for d in ws.dispatches()][-2:], ["milestone_deliver", "milestone_judge"])
        self.assertEqual(ws.decisions()[-1]["point"], F.MILESTONE)



class TestApprovalAsks(TF.FlowCase):
    def test_refresh_sets_and_over_budget(self):
        for builder, what, options in ((TF.b_evalset_used_up, "refresh_sets", ["approve", "close_round", "stop"]),
                                       (TF.b_over_budget, "over_budget", ["approve", "stop"])):
            for option in options:
                with self.subTest((what, option)):
                    self.use_task(fixtures.binary_task(), self.unique())
                    builder(self)
                    ws = Workspace(self.ws)
                    a = F.next_action(ws)
                    self.assertEqual((a["type"], a["what"], a["options"]), ("ask", what, options))
                    for word in options:
                        self.assertIn(word + ": ", a["question"])
                    ws.set_waiting(dict((k, a[k]) for k in ("kind", "what", "ref", "question", "options")))
                    DP.answer(Workspace(self.ws), option, "alice", "do as I say")
                    ws = Workspace(self.ws)
                    self.assertIsNone(ws.waiting())
                    if option == "stop":
                        self.assertEqual(ws.state["stopped"]["reason"], "user requested: do as I say")
                        self.assertEqual(F.next_action(ws)["type"], "report")
                    elif option == "close_round":  # no new evalset: the round is closed, the judge writes the plan
                        self.assertIn("do as I say", ws.state["rounds"][str(ws.state["round"])]["decision"]["reason"])
                        self.assertEqual(self.action()["purpose"], "plan")
                    else:
                        self.assertTrue(ws.has_approval(what))
                        self.assertEqual(ws.state["approvals"][-1]["quote"], "do as I say")
                        self.assertNotEqual(F.next_action(ws)["type"], "ask")
                    self.assertEqual(ws.decisions()[-1]["choice"], option)

    def test_plan_revision_of_a_plan_registered_by_hand(self):
        TF.b_plan_needs_approval(self)
        a = self.action()
        self.assertEqual((a["what"], a["options"]), ("plan", ["approve", "revise"]))
        Workspace(self.ws).set_waiting(dict((k, a[k]) for k in ("kind", "what", "ref", "question", "options")))
        rec = DP.answer(Workspace(self.ws), "revise", "alice", "make the goals more specific")
        self.assertIsNone(rec["dispatch_id"])
        a = self.assertAction({"type": "dispatch", "module": "judge", "purpose": "plan", "step": "write_plan"})
        self.assertIn("make the goals more specific", a["reason"])
        self.add_dispatch("judge", "done", purpose="plan")  # it came back without a new plan: asked again
        self.assertAction({"type": "ask", "what": "plan", "ref": "plan-0001"})


class TestTextsAndAttribution(TF.FlowCase):
    """S1: the close_round exit says the stop when closing the round meets a stop condition; M3: a train_gate_failed
    decision is about the round's latest train dispatch."""

    def close_text(self, ws):
        return DP._exit_text(ws, "close_round", ws.dispatch("d-0001"))

    def test_close_round_option_names_the_stop(self):
        TF.b_train_gate_failed_after_search(self)  # patience 2, max_rounds 5
        ws = Workspace(self.ws)
        self.assertFalse(ws.stop_check_if_closed()[0])
        old = "close_round: close round 1 (no evaluation, recorded as keep), then rsi-judge writes the plan"
        self.assertEqual(self.close_text(ws), old)
        self.assertIn(old, DP.question(ws, self.action(), {"by": "code"})["question"])
        self.edit_state(lambda s: s.__setitem__("no_promotion_streak", 1))  # closing round 1 reaches patience 2
        ws = Workspace(self.ws)
        stop, why = ws.stop_check_if_closed()
        self.assertTrue(stop)
        text = "close_round: close round %d (no evaluation, recorded as keep), then the run stops (%s)" % (1, why)
        self.assertEqual(self.close_text(ws), text)
        q = DP.question(ws, self.action(), {"by": "code"})["question"]
        self.assertIn(text, q)
        self.assertNotIn("writes the plan", q)

    def test_train_gate_failed_names_the_latest_train_dispatch(self):
        self.sets()
        TF.failed_search_from(self, self.add_dispatch("train", "done"))
        latest = self.add_dispatch("train", "done")  # a new session took over and ran a search too
        TF.failed_search_from(self, latest)
        self.assertAction({"type": "decide", "point": F.TRAIN_GATE_FAILED, "dispatch_id": latest})
        # Without a train dispatch in the round (candidates registered by hand): no dispatch.
        self.use_task(fixtures.binary_task(), self.unique())
        TF.b_train_gate_failed(self)
        self.assertAction({"type": "decide", "point": F.TRAIN_GATE_FAILED, "dispatch_id": None})


class TestContinuations(DecisionCase):
    def marker(self, n):
        with open(self.outfile("d-0002", "job-%d.marker" % n), "w") as f:
            f.write("ok\n")

    def test_cap_asks_then_the_answer_goes_on(self):
        self.use_task(TE.task_def(max_continues=1), "ws-lj")
        self.behave("train", long_jobs=2)
        self.assertEqual(self.status()[0], "waiting_long_job")
        self.marker(1)
        self.assertEqual(self.status()[0], "waiting_long_job")  # continued once, the second phase runs
        self.assertEqual(Workspace(self.ws).dispatch("d-0002")["continued"], 1)
        self.marker(2)
        status, reason = self.status()
        self.assertEqual(status, "waiting_user")
        w = self.waiting()
        self.assertEqual((w["what"], w["ref"], w["options"]), (F.HANDBACK_FAILED, "d-0002", EXITS))
        self.assertIn("max_continues", w["question"])
        d = Workspace(self.ws).dispatch("d-0002")
        self.assertEqual((d["status"], d["capped"], d["continued"]), ("failed", "max_continues", 1))
        self.answer("resume", "go on collecting the results")
        self.assertMilestoneAsk(self.run_engine())
        d = Workspace(self.ws).dispatch("d-0002")
        self.assertEqual((d["returns"], d["continued"], d.get("capped")), (1, 1, None))
        self.assertIn("go on collecting the results", self.calls("d-0002")[2]["feedback"])


# ---- fix round 1 (R12): a stop mid-step, new_session's purpose, questions, question ids, stale questions ---

def stop_from_elsewhere(ws_path, quote="stop for now"):
    """`rsi stop` from another process (the user, while `rsi run` works)."""
    r = subprocess.run([sys.executable, "-m", "rsi", "stop", ws_path, "--by", "bob", "--quote", quote], cwd=REPO,
                       capture_output=True, text=True, env=dict(os.environ, RSI_NO_BROWSER="1"))
    assert r.returncode == 0, r.stderr


class TestStopMidStep(DecisionCase):
    def assertStoppedCleanly(self, status):
        self.assertEqual(status, "done")
        ws = Workspace(self.ws)
        self.assertTrue(ws.state["stopped"])
        self.assertEqual([d["id"] for d in ws.dispatches() if d["status"] != "done"], [])
        self.assertTrue([r for r in self.log() if r["event"] == "result" and r.get("stopped")] or
                        [r for r in self.log() if r["event"] == "closed"])

    def test_stop_while_the_decision_agent_thinks(self):
        # Whatever the agent answers (its default resume_data, or ask), the run was stopped meanwhile: nothing is
        # sent back, no question is set, the run reports.
        for answer in (None, {"choice": "ask", "reason": "ask someone"}):
            with self.subTest(answer):
                t = TE.task_def()
                t["decisions"] = {F.QUALITY_GATE_FAILED: "agent"}
                self.use_task(t, "ws-sd-%s" % bool(answer))
                self.behave("data", trainset="quality_fail")
                if answer:
                    self.behave("decide", **{F.QUALITY_GATE_FAILED: answer})
                orig, ws_path = MockRunner.decide, self.ws

                def decide(runner, ws, request, cancel=None):
                    stop_from_elsewhere(ws_path)
                    return orig(runner, ws, request, cancel=cancel)

                with mock.patch.object(MockRunner, "decide", decide):
                    status, _reason = self.status()
                self.assertStoppedCleanly(status)
                self.assertEqual(len(self.calls("d-0001")), 1)  # not sent back after the stop
                self.assertEqual(Workspace(self.ws).dispatch("d-0001")["returns"], 0)
                self.assertEqual(self.actions()[-3:], ["decide", "report", "done"])
                self.assertIsNone(Workspace(self.ws).waiting())

    def test_stop_before_the_refusal_is_decided(self):
        self.behave("data", trainset="overlap")
        orig, ws_path = engine.Engine.register, self.ws

        def register(eng, ws, d, names):
            out = orig(eng, ws, d, names)
            stop_from_elsewhere(ws_path)
            return out

        with mock.patch.object(engine.Engine, "register", register):
            status, _reason = self.status()
        self.assertStoppedCleanly(status)
        ws = Workspace(self.ws)
        self.assertEqual(ws.decisions(), [])  # nothing decided on a stopped run
        d = ws.dispatch("d-0001")
        self.assertEqual((d["status"], len(d["refused"]), d["returns"]), ("done", 1, 0))
        self.assertEqual(len(self.calls("d-0001")), 1)

    def test_registration_refused_by_the_stop_is_no_refusal(self):
        orig, ws_path, fired = Workspace.register, self.ws, []

        def register(ws, kind, src):
            if kind == "candidate" and not fired:
                fired.append(1)
                Workspace(ws_path).stop("bob", "stop here")
            return orig(ws, kind, src)

        with mock.patch.object(Workspace, "register", register):
            status, _reason = self.status()
        self.assertStoppedCleanly(status)
        ws = Workspace(self.ws)
        d = ws.dispatch("d-0002")
        self.assertEqual((d["status"], d["refused"], d["returns"]), ("done", [], 0))
        self.assertIn("not registered from the hand-back", d["closed"])
        self.assertEqual(ws.decisions(), [])  # no handback_refused
        self.assertEqual(len(self.calls("d-0002")), 1)

    def test_stop_between_the_record_and_the_launch(self):
        orig, ws_path = engine.Engine.prepare, self.ws

        def prepare(eng, ws, d):
            orig(eng, ws, d)
            if d["module"] == "train":
                stop_from_elsewhere(ws_path)

        with mock.patch.object(engine.Engine, "prepare", prepare):
            status, _reason = self.status()
        self.assertStoppedCleanly(status)
        d = Workspace(self.ws).dispatch("d-0002")
        self.assertEqual(d["status"], "done")
        self.assertIn("never launched", d["closed"])
        self.assertEqual(self.calls("d-0002"), [])  # never launched

    def test_stop_during_the_launches_of_one_write(self):
        # Two records written together: the first launched before the stop (its agent runs, waiting), the second
        # after it: the second never starts and counts no launch; the report stops the first agent and closes it.
        self.behave("data", release=os.path.join(self.root, "never"))
        orig, ws_path, fired = MockRunner.launch, self.ws, []

        def launch(runner, ws, d, prompt, resume_session=None, new_session=None):
            if not fired:
                fired.append(1)
                stop_from_elsewhere(ws_path)
            return orig(runner, ws, d, prompt, resume_session=resume_session, new_session=new_session)

        eng = engine.Engine(self.ws, MockRunner(MOCKS), pages=False)
        with mock.patch.object(MockRunner, "launch", launch):
            with self.assertRaises(W.StoppedError):
                eng.dispatch(Workspace(self.ws), [{"module": "data", "purpose": "sets"},
                                                  {"module": "data", "purpose": "sets"}])
        status, _reason = self.status()
        self.assertStoppedCleanly(status)
        ws = Workspace(self.ws)
        first, second = ws.dispatch("d-0001"), ws.dispatch("d-0002")
        self.assertEqual(self.calls("d-0002"), [])
        self.assertIn("agent terminated", first["closed"])
        self.assertIsNotNone(first["launched_at"])
        self.assertEqual(first["launches"], 1)
        self.assertIn("never launched", second["closed"])
        self.assertEqual((second["status"], second["launched_at"], second["launches"]), ("done", None, 0))
        self.assertEqual([e["kind"] for e in ws.entries()], ["task"])  # nothing registered

    def test_resume_on_a_stopped_run_starts_nothing(self):
        self.train_fails()
        Workspace(self.ws).stop("bob", "stop")
        eng = engine.Engine(self.ws, MockRunner(MOCKS), pages=False)
        ws = Workspace(self.ws)
        with self.assertRaises(W.StoppedError):
            eng.resume(ws, ws.dispatch("d-0002"), "try again")
        d = Workspace(self.ws).dispatch("d-0002")
        self.assertEqual((d["status"], d["returns"], d["launches"]), ("done", 0, 1))
        self.assertEqual(len(self.calls("d-0002")), 1)


class TestFreshPurpose(TF.FlowCase):
    def test_the_steps_current_part(self):
        TF.b_decided_judge_returns_used(self)  # the judge evaluated; the round is decided: its plan is owed
        ws = Workspace(self.ws)
        d = ws.dispatch("d-0001")
        self.assertEqual(DP.fresh_purpose(ws, "write_plan", d), "plan")
        q = DP.question(ws, self.action(), {"by": "code"})
        self.assertIn("rsi-judge session (purpose plan)", q["question"])
        self.use_task(fixtures.binary_task(), self.unique())
        TF.b_quality_gate_failed_from_dispatch(self)  # round 1: three sets in, the trainset failed
        ws = Workspace(self.ws)
        self.assertEqual(DP.fresh_purpose(ws, "data", ws.dispatch("d-0001")), "trainset")
        self.use_task(fixtures.binary_task(), self.unique())
        TF.b_trials_without_search(self)
        ws = Workspace(self.ws)
        self.assertEqual(DP.fresh_purpose(ws, "train", ws.dispatch("d-0001")), "train")
        self.use_task(fixtures.binary_task(), self.unique())
        self.assertEqual(DP.fresh_purpose(Workspace(self.ws), "data", None), "sets")


class TestR12Answers(DecisionCase):
    def test_questions_come_before_registration(self):
        # The hand-back would be refused (its trainset overlaps the evalset), but it asks a question: nothing is
        # registered, so nothing is refused, until the answer; the answer goes back to the same session, whose new
        # hand-back (a changed trainset) is registered.
        self.behave("data", trainset="overlap", questions=["Make more short messages?"])
        self.assertEqual(self.status()[0], "waiting_user")
        ws = Workspace(self.ws)
        w, d = ws.waiting(), ws.dispatch("d-0001")
        self.assertEqual((w["what"], w["ref"], w["options"], d["status"], d["launches"]),
                         ("handback", "d-0001", ["answered"], "handed_back", 1))
        self.assertEqual((ws.decisions(), d.get("refused"), [e["kind"] for e in ws.entries()]), ([], None, ["task"]))
        rec = self.answer("answered", "yes, make more short messages")
        self.assertEqual((rec["point"], rec["dispatch_id"], rec["launches"]), ("handback", "d-0001", 1))
        self.assertMilestoneAsk(self.run_engine())
        self.assertEqual(self.decisions()[:1], [("handback", "answered", "alice")])
        self.assertNotIn(F.HANDBACK_REFUSED, [x[0] for x in self.decisions()])
        feedback = self.calls("d-0001")[1]["feedback"]
        for word in ("Make more short messages?", "yes, make more short messages", "registered yet"):
            self.assertIn(word, feedback)
        ws = Workspace(self.ws)
        self.assertEqual(ws.entry(ws.current("trainset"))["source"], self.outfile("d-0001", "trainset-2"))

    def test_answer_goes_back_to_the_asking_session_not_to_other_dispatches(self):
        self.use_task(TE.task_def(max_returns=0), "ws-br")
        self.behave("data", questions=["Deduplicate the trainset?"])
        self.behave("train", gate_fail=True)
        self.assertEqual(self.status()[0], "waiting_user")
        self.answer("answered", "yes, deduplicate")
        self.assertEqual(self.status()[0], "waiting_user")
        self.assertEqual(self.waiting()["what"], F.TRAIN_GATE_FAILED)
        self.assertIn("yes, deduplicate", self.calls("d-0001")[1]["feedback"])
        self.answer("data", "add a batch of data")
        self.status()
        with open(self.outfile("d-0003", "brief.md"), encoding="utf-8") as f:
            self.assertNotIn("yes, deduplicate", f.read())  # delivered once, to the session that asked
        rec = [x for x in Workspace(self.ws).decisions() if x["point"] == "handback"][0]
        self.assertEqual(rec["delivered_to"], ["d-0001 launch 2"])

    def test_question_ids_and_ref(self):
        self.train_fails()
        ws = Workspace(self.ws)
        qid = ws.waiting()["id"]
        self.assertRegex(qid, r"^q-\d{4}$")
        self.assertIn(qid, self.status()[1])  # the same question again keeps its id, named in the reason
        self.assertEqual(Workspace(self.ws).waiting()["id"], qid)
        with self.assertRaises(RsiError) as e:
            DP.answer(Workspace(self.ws), "resume", "alice", "x", ref="q-9999")
        self.assertIn("not the question waiting now", str(e.exception))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["answer", self.ws, "resume", "--ref", qid, "--by", "alice", "--quote", "again"]), 0)
        self.assertEqual(Workspace(self.ws).decisions()[-1]["question_id"], qid)
        nxt = Workspace(self.ws).set_waiting({"kind": "question", "question": "q", "options": ["x"]})
        self.assertNotEqual(nxt["id"], qid)

    def test_decision_question_goes_stale_when_the_round_or_its_dispatch_is_closed(self):
        self.train_fails()
        Workspace(self.ws).update_dispatch("d-0002", status="done", closed="by hand")
        ws = Workspace(self.ws)
        self.assertTrue(F.stale_waiting(ws, ws.waiting()))
        self.use_task(TE.task_def(max_returns=0), "ws-closed-round")  # the round closed by hand instead
        self.behave("train", fail="out of GPU memory, the training did not finish")
        self.assertEqual(self.status()[0], "waiting_user")
        Workspace(self.ws).close_round("close it, try another idea")
        ws = Workspace(self.ws)
        self.assertTrue(F.stale_waiting(ws, ws.waiting()))
        self.assertNotEqual(F.next_action(ws)["type"], "ask")
        with self.assertRaises(RsiError) as e:
            self.answer("resume")
        self.assertIn("no longer applies", str(e.exception))
        self.assertEqual(self.status()[0], "done")  # the failed train dispatch is moot in the decided round
        self.assertIn("clear_waiting", [r["event"] for r in self.log()])

    def test_manual_milestone_consent_covers_both_uses(self):
        # The user agreed by `rsi approve use_testset` after the round's decision: before the question was asked, or
        # after (the question goes stale). One approval covers the delivery and the evaluation.
        for asked in (False, True):
            with self.subTest(asked=asked):
                self.use_task(TE.task_def(), "ws-ms-%s" % asked)
                if asked:
                    self.assertMilestoneAsk(self.run_engine())
                else:
                    self.assertEqual(self.status(max_steps=16)[0], "max_steps")
                    self.assertEqual(self.actions()[-1], "promote")
                Workspace(self.ws).approve("use_testset", "alice", "check it on the testset")
                self.assertEqual(self.status()[0], "done")
                approvals = [a for a in Workspace(self.ws).state["approvals"] if a["what"] == "use_testset"]
                self.assertEqual([(a["quote"], a["used_by"]) for a in approvals],
                                 [("check it on the testset", "delivery-0002"),
                                  ("check it on the testset", "evaluation-0002")])

    def test_closing_or_replacing_clears_capped(self):
        for option in ("new_session", "close_round"):
            with self.subTest(option):
                self.use_task(TE.task_def(max_continues=0), "ws-cc-" + option)
                self.behave("train", long_jobs=1)
                self.assertEqual(self.status()[0], "waiting_long_job")
                with open(self.outfile("d-0002", "job-1.marker"), "w") as f:
                    f.write("ok\n")
                self.assertEqual(self.status()[0], "waiting_user")
                self.assertEqual(Workspace(self.ws).dispatch("d-0002")["capped"], "max_continues")
                self.answer(option, "answer")
                self.status()
                d = Workspace(self.ws).dispatch("d-0002")
                self.assertEqual((d["status"], d.get("capped")), ("done", None))

    def test_answer_line_is_written_with_the_save(self):
        self.train_fails()
        seen = []
        orig = Workspace.save_state

        def save(ws):
            seen.append([r["event"] for r in self.log()][-1])
            return orig(ws)

        with mock.patch.object(Workspace, "save_state", save):
            self.answer("resume", "again")
        self.assertEqual(seen, ["answer"])  # already in run.jsonl when STATE is saved
        ws = Workspace(self.ws)
        ws.set_waiting({"kind": "question", "question": "q", "options": ["x"]})
        with mock.patch.object(Workspace, "save_state", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.answer("x", "y")
        self.assertEqual([r["event"] for r in self.log()][-2:], ["answer", "answer_failed"])
        self.assertEqual(Workspace(self.ws).waiting()["question"], "q")  # nothing saved


# ---- final review (R15): a stop stops the agents ---------------------------------------------------------

FAKE_SLEEPER = r"""#!@PYTHON@
import os, sys, time
sys.stdin.read()
with open(os.environ["FAKE_SLEEPER_PIDS"], "a") as f:
    f.write("%d\n" % os.getpid())
time.sleep(60)
"""


class TestStopStopsAgents(DecisionCase):
    """ClaudeCodeRunner against the strict fake `claude` of test_runners in sleep mode (never the real one)."""

    def setUp(self):
        super(TestStopStopsAgents, self).setUp()
        import test_runners
        self.fake, self.record = test_runners.install_fake_claude(self, self.root)
        env = mock.patch.dict(os.environ, {"FAKE_CLAUDE_MODE": "sleep"})
        env.start()
        self.addCleanup(env.stop)

    def start_run(self):
        p = subprocess.Popen([sys.executable, "-m", "rsi", "run", self.ws, "--runner", "claude"], cwd=REPO,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                             env=dict(os.environ, RSI_NO_BROWSER="1"))
        self.addCleanup(lambda: p.poll() is None and p.kill())
        end = time.time() + 30
        while time.time() < end:
            ds = Workspace(self.ws).dispatches()
            if ds and ds[0].get("pid"):
                return p, ds[0]["pid"]
            time.sleep(0.1)
        p.kill()
        self.fail("the agent never started: %s" % (p.communicate()[1],))

    def assertGone(self, pid, within=10):
        from rsi.runners.base import pid_alive
        end = time.time() + within
        while pid_alive(pid) and time.time() < end:
            time.sleep(0.1)
        self.assertFalse(pid_alive(pid), "the agent %d is still alive" % pid)

    def assertClosedAndNotShown(self):
        ws = Workspace(self.ws)
        d = ws.dispatch("d-0001")
        self.assertEqual(d["status"], "done")
        self.assertIn("agent terminated", d["closed"])
        from rsi import dashboard as D
        self.assertEqual(D.running_agents(ws), [])
        self.assertEqual([x for x in ws.dispatches() if x["status"] == "running"], [])

    def test_stop_while_rsi_run_waits_stops_the_agent(self):
        p, pid = self.start_run()
        t0 = time.time()
        stop_from_elsewhere(self.ws, "stop")
        out, err = p.communicate(timeout=30)
        self.assertEqual(p.returncode, 0, err)  # done, with the report
        self.assertLess(time.time() - t0, 10)  # promptly, not after the agent's 30 s
        self.assertIn("The run has ended", out)
        self.assertGone(pid)
        self.assertClosedAndNotShown()
        self.assertIn("agent_stopped", [r["event"] for r in self.log()])

    def test_killed_run_then_stop_then_run(self):
        import signal
        from rsi.runners.base import pid_alive
        p, pid = self.start_run()
        p.send_signal(signal.SIGKILL)
        p.communicate()
        self.assertTrue(pid_alive(pid))  # the agent outlives the killed `rsi run`
        stop_from_elsewhere(self.ws, "stop")
        r = subprocess.run([sys.executable, "-m", "rsi", "run", self.ws, "--runner", "claude"], cwd=REPO,
                           capture_output=True, text=True, timeout=60, env=dict(os.environ, RSI_NO_BROWSER="1"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGone(pid)
        self.assertClosedAndNotShown()
        with open(os.path.join(self.ws, "dashboard.html"), encoding="utf-8") as f:
            self.assertNotIn('class="agents"', f.read())

    def test_done_never_leaves_a_running_record(self):
        # Stopped and reported already, yet a record still says running (left by an earlier run): done closes it.
        ws = Workspace(self.ws)
        ws.add_dispatches([{"module": "data", "purpose": "sets", "outdir": "work/round-1/data", "runner": "mock"}])
        ws.stop("bob", "stop")
        with open(os.path.join(self.ws, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write("# Report\n")
        self.assertEqual(self.status()[0], "done")
        self.assertEqual(self.actions(), ["done"])
        d = Workspace(self.ws).dispatch("d-0001")
        self.assertEqual(d["status"], "done")
        self.assertIn("run stopped", d["closed"])

    def test_decision_agent_is_stopped_with_the_run(self):
        from rsi.runners.base import pid_alive
        path = os.path.join(self.root, "sleeper")
        with open(path, "w") as f:
            f.write(FAKE_SLEEPER.replace("@PYTHON@", sys.executable))
        os.chmod(path, 0o755)
        pids = os.path.join(self.root, "sleeper.pids")
        workdir = os.path.join(self.root, "decision")
        os.makedirs(workdir)
        runner = ClaudeCodeRunner(command=[path])
        t0 = time.time()

        def cancel():  # the run stops once the agent works (its pid is written), not after a fixed second
            return (os.path.isfile(pids) and os.path.getsize(pids) > 0) or time.time() - t0 > 8.0
        with mock.patch.dict(os.environ, {"FAKE_SLEEPER_PIDS": pids}):
            r = runner.decide(Workspace(self.ws), {"prompt": "p", "schema": DP.choice_schema(["ask"]), "dir": workdir},
                              cancel=cancel)
        self.assertLess(time.time() - t0, 10)
        self.assertTrue(r.get("cancelled"))
        with open(pids) as f:
            pid = int(f.read().split()[0])
        self.assertFalse(pid_alive(pid))


class TestR15Engine(DecisionCase):
    def search_cap(self, name="sc"):
        """train's trials are in, its followup came back without the search and no returns are allowed: the
        question is about a dispatch whose status is done (its step incomplete)."""
        self.use_task(TE.task_def(max_returns=0), "ws-" + name)
        self.behave("train", skip_search=1)
        self.assertEqual(self.status()[0], "waiting_user")
        ws = Workspace(self.ws)
        w = ws.waiting()
        self.assertEqual((w["what"], w["ref"], ws.dispatch(w["ref"])["status"]), (F.HANDBACK_FAILED, "d-0002", "done"))
        self.assertFalse(F.stale_waiting(ws, w))
        return w

    def test_question_at_the_cap_on_a_done_dispatch(self):
        for option in ("resume", "new_session"):
            with self.subTest(option):
                w = self.search_cap(option)
                if option == "resume":
                    self.answer("resume", "write the search")
                    self.assertMilestoneAsk(self.run_engine())
                    d = Workspace(self.ws).dispatch("d-0002")
                    self.assertEqual((d["returns"], d["followups"]), (1, 1))
                    self.assertIn("write the search", self.calls("d-0002")[2]["feedback"])
                    self.assertIn("search", [e["kind"] for e in Workspace(self.ws).entries()])
                    self.assertNotEqual(Workspace(self.ws).state["waiting"]["id"], w["id"])
                else:
                    self.answer("new_session", "write the search in a new session")
                    self.status()
                    ws = Workspace(self.ws)
                    new = [d for d in ws.dispatches() if d["module"] == "train"][-1]
                    self.assertNotEqual(new["id"], "d-0002")
                    self.assertEqual(new["purpose"], "train")
                    self.assertIn("write the search in a new session", self.calls(new["id"])[0]["feedback"])

    def test_resume_rebuilds_the_brief(self):
        # train is sent back after data made a new trainset: its brief names the new one and the candidate it
        # hands back is registered against it.
        self.use_task(TE.task_def(max_returns=0), "ws-rb")
        self.behave("train", gate_fail=True)
        self.assertEqual(self.status()[0], "waiting_user")
        self.answer("data", "add data")
        self.assertEqual(self.status()[0], "waiting_user")
        self.answer("train", "train again on the new trainset")
        self.status()
        ws = Workspace(self.ws)
        with open(self.outfile("d-0002", "brief.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["ids"]["trainset"], "trainset-0002")
        self.assertTrue(os.path.isfile(self.outfile("d-0002", "brief-1.md")))
        feedback = self.calls("d-0002")[2]["feedback"]
        self.assertIn("The brief has changed", feedback)
        self.assertIn("trainset-0001 → trainset-0002", feedback)
        cand = [e["id"] for e in ws.entries("candidate") if e["source"].startswith(self.outfile("d-0002", ""))][-1]
        self.assertEqual(ws.manifest(cand)["trainset_id"], "trainset-0002")

    def test_answers_are_delivered_once_and_the_plan_sees_them(self):
        t = TE.task_def(max_rounds=2)
        t["approvals"].append("plan")
        self.use_task(t, "ws-ad")
        self.behave("data", questions=["Make more short messages?"])
        self.assertEqual(self.status()[0], "waiting_user")
        self.answer("answered", "yes, make more short messages")
        self.assertEqual(self.status()[0], "waiting_user")  # at the plan approval
        ws = Workspace(self.ws)
        self.assertEqual(ws.waiting()["what"], "plan")
        # The answer went back to the asking data session with its next launch, once.
        self.assertIn("yes, make more short messages", self.calls("d-0001")[1]["feedback"])
        rec = [d for d in ws.decisions() if d["point"] == "handback"][0]
        self.assertEqual(rec["delivered_to"], ["d-0001 launch 2"])
        with open(self.outfile("d-0004", "brief.md"), encoding="utf-8") as f:  # the judge's plan brief (rebuilt)
            self.assertIn("yes, make more short messages", f.read())
        # A data dispatch made later (any round, the same purpose) does not get it again.
        nd = engine.Engine(self.ws, MockRunner(MOCKS), pages=False)
        ws.add_dispatches([{"module": "data", "purpose": "sets", "outdir": "work/round-1/data-x", "launches": 1}])
        d = Workspace(self.ws).dispatches()[-1]
        nd.prepare(Workspace(self.ws), d)
        with open(os.path.join(self.ws, d["outdir"], "brief.md"), encoding="utf-8") as f:
            self.assertNotIn("yes, make more short messages", f.read())
        self.assertEqual(DP.pending_answers(Workspace(self.ws), dict(d, id="d-9999")), [])

    def test_answer_reaches_the_module_in_a_later_round(self):
        # An answer that never went back (its dispatch was closed first) reaches the next dispatch of the same module
        # and purpose, in any round; not one made for another purpose.
        ws = Workspace(self.ws)
        ws.add_dispatches([{"module": "data", "purpose": "trainset", "outdir": "work/round-1/data", "status": "done"}])
        ws.add_decision({"point": "handback", "choice": "answered", "by": "alice", "quote": "use the newer data",
                         "question": "q?", "question_id": "q-0001", "dispatch_id": "d-0001", "launches": 1})
        ws.state["round"] = 2
        ws.state["rounds"]["2"] = dict(ws.state["rounds"]["1"], artifacts=[])
        ws.save_state()
        ws = Workspace(self.ws)
        later = {"id": "d-0002", "module": "data", "round": 2, "purpose": "trainset"}
        self.assertEqual([x["quote"] for x in DP.pending_answers(ws, later)], ["use the newer data"])
        self.assertEqual(DP.pending_answers(ws, dict(later, purpose="refresh")), [])
        self.assertEqual(DP.pending_answers(ws, dict(later, module="train")), [])

    def test_made_feedback_is_written_again_after_an_interruption(self):
        # A decision agent sends the failed search's round to data (a dispatch the decision makes); then, data at its
        # cap, the code's default sends train back.
        t = TE.task_def(max_returns=1)
        t["decisions"] = {F.TRAIN_GATE_FAILED: "agent"}
        self.use_task(t, "ws-mf")
        self.behave("train", gate_fail=True)
        self.behave("decide", **{F.TRAIN_GATE_FAILED: {"choice": "data", "reason": "add data"}})
        orig, seen = engine.Engine.made_feedback, []

        def flaky(eng, ws, d):
            if d.get("made") and not seen:
                seen.append(d["id"])
                raise KeyboardInterrupt
            return orig(eng, ws, d)

        with mock.patch.object(engine.Engine, "made_feedback", flaky):
            with self.assertRaises(KeyboardInterrupt):
                self.status()
        self.assertFalse(os.path.exists(self.outfile(seen[0], "feedback-1.md")))
        self.assertMilestoneAsk(self.run_engine())
        call = self.calls(seen[0])[0]
        self.assertEqual(call["feedback_file"], "feedback-1.md")
        self.assertIn("Why you were dispatched", call["feedback"])

    def test_registration_cut_off_before_state_is_repaired(self):
        orig, fired = Workspace.save_state, []

        def save(ws):
            if not fired and ws._entries and ws._entries[-1]["kind"] == "trainset":
                fired.append(1)
                raise KeyboardInterrupt
            return orig(ws)

        with mock.patch.object(Workspace, "save_state", save):
            with self.assertRaises(KeyboardInterrupt):
                self.status()
        ws = Workspace(self.ws)
        self.assertEqual([e["id"] for e in ws.unlisted()], ["trainset-0001"])
        self.assertEqual(F.products(ws, ws.dispatch("d-0001"))[-1]["id"], "trainset-0001")  # counted already
        self.assertMilestoneAsk(self.run_engine())
        ws = Workspace(self.ws)
        self.assertEqual(ws.unlisted(), [])
        self.assertEqual(ws.current("trainset"), "trainset-0001")
        self.assertIn("repair", [r["event"] for r in self.log()])

    def test_records_are_registered_after_a_stop(self):
        self.status(runner=ManualRunner())
        out = self.outfile("d-0001", "")
        fixtures.make_artifact("record", os.path.join(out, "rec"), self.task, artifacts=["task-0001"])
        fixtures.make_artifact("testset", os.path.join(out, "testset"), self.task, ids=TE.TEST_IDS)
        with open(os.path.join(out, "handback.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "ok", "artifacts": ["testset", "rec"], "marker": None, "notes": "n",
                       "questions": []}, f)
        Workspace(self.ws).stop("bob", "stop")
        self.assertEqual(self.status(runner=ManualRunner())[0], "done")
        ws = Workspace(self.ws)
        self.assertEqual([e["kind"] for e in ws.entries()], ["task", "record"])
        d = ws.dispatch("d-0001")
        self.assertEqual(d["status"], "done")
        self.assertIn("experiment records from the hand-back: record-0001", d["closed"])

    def test_program_match_with_spaces(self):
        from rsi.runners.base import pid_alive
        d = os.path.join(self.root, "my tools")
        os.makedirs(d)
        prog = os.path.join(d, "claude")
        with open(prog, "w") as f:
            f.write("import time\ntime.sleep(20)\n")
        r = ClaudeCodeRunner(command=[prog])
        self.assertTrue(r.is_program("%s -p" % prog))
        self.assertTrue(r.is_program("/usr/bin/python3 %s -p" % prog))
        self.assertFalse(r.is_program("/usr/bin/python3 %s-other -p" % prog))
        p = subprocess.Popen([sys.executable, prog])
        self.addCleanup(p.wait)
        self.addCleanup(p.kill)
        time.sleep(0.3)
        self.assertTrue(pid_alive(p.pid))
        self.assertTrue(r.ours(p.pid, time.time()))


class TestR15Pages(TF.FlowCase):
    def test_last_evalset_use_on_the_action_and_the_page(self):
        task = fixtures.binary_task()
        task["eval_use_limit"] = 2
        self.use_task(task, self.unique())
        self.round_two(entry="train")
        self.candidate(2)
        a = self.action()
        self.assertEqual((a["type"], a["module"]), ("dispatch", "deliver"))
        self.assertIn("last evaluation (uses 1 / 2)", a["reason"])
        import test_dashboard as TDB
        self.assertIn("last evaluation", TDB.section(TDB.pages(self.ws)[TDB.OVERVIEW], "now"))

    def test_refresh_card_says_the_data_refresh_comes_next(self):
        TF.b_evalset_used_up(self)
        import test_dashboard as TDB
        now = TDB.section(TDB.pages(self.ws)[TDB.OVERVIEW], "now")
        self.assertIn("refresh the evalset", now)
        self.assertNotIn("④ Deliver: Dispatch rsi-deliver", now)
        self.assertIn("The engine has not recorded this question yet", now)  # the flow asks; rsi run has not recorded it yet
        self.assertIn("answer %s close_round" % self.ws, now)

    def test_closed_round_judge_node_names_the_milestone_evaluation(self):
        task = fixtures.binary_task()
        task["stop"]["max_rounds"] = 2
        self.use_task(task, self.unique())
        self.round_two(entry="train")
        Workspace(self.ws).close_round("nothing to evaluate")
        self.decide(F.MILESTONE, "yes")
        cand = Workspace(self.ws).version("V1")["candidate_id"]
        Workspace(self.ws).approve("use_testset", "alice", "yes")
        src = fixtures.make_artifact("delivery", self.src("ms-delivery"), self.task, ids=TF.EVAL_IDS,
                                     val_ids=TF.VAL_IDS, candidate_id=cand, round=2)
        fixtures.add_predictions(src, self.task, "predictions_test", TF.TEST_IDS, testset_id="testset-0001")
        dlv = Workspace(self.ws).register("delivery", src)
        Workspace(self.ws).approve("use_testset", "alice", "yes")
        self.reg("evaluation", 2, ids=TF.EVAL_IDS, train_ids=TF.TRAIN_IDS, delivery_id=dlv, candidate_id=cand,
                 evalset_id="evalset-0001", uses_testset=True, test_ids=TF.TEST_IDS,
                 best_before={"version": "V1", "evaluation_id": "evaluation-0001"},
                 recommendation={"decision": "keep", "reason": "milestone release"})
        from rsi import dashboard as D
        ws = Workspace(self.ws)
        node = D.flow_states(ws, D.module_progress(ws))["judge"]
        self.assertEqual(node["state"], "done")
        self.assertIn("milestone", node["text"])

    def test_events_follow_the_run_log_order(self):
        e = TE.EngineCase("run_engine")
        e.setUp()
        self.addCleanup(e.doCleanups)
        e.run_engine()
        import test_dashboard as TDB
        page = TDB.pages(e.ws)[TDB.EVENTS]
        # Newest first: train's candidate was registered after rsi-train was dispatched, in the same second or not.
        self.assertLess(page.index("candidate-0001"), page.index("Dispatched rsi-train"))
        self.assertLess(page.index("Dispatched rsi-train"), page.index("trainset-0001"))


# ---- the STATE lock ---------------------------------------------------------------------------------

CHILD_HOLDS = """
import sys, time
sys.path.insert(0, sys.argv[1])
from rsi.workspace import Workspace
ws = Workspace(sys.argv[2])
with ws.transaction():
    ws.approve("over_budget", "child", "go on")
    open(sys.argv[3], "w").close()
    time.sleep(0.8)
"""

CHILD_LOOP = """
import os, sys, time
sys.path.insert(0, sys.argv[1])
from rsi.workspace import Workspace
ws = Workspace(sys.argv[2])
open(sys.argv[3] + ".ready", "w").close()
while not os.path.exists(sys.argv[3]):
    time.sleep(0.002)
for i in range(int(sys.argv[4])):
    ws.add_decision({"point": "child", "choice": "c%d" % i})
    time.sleep(0.002)
"""


class TestStateLock(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.ws = os.path.join(self.root, "ws")
        task = fixtures.binary_task()
        Workspace.init(self.ws, fixtures.make_task_dir(os.path.join(self.root, "task"), task))

    def test_a_writer_waits_for_the_lock_and_keeps_the_other_write(self):
        stale = Workspace(self.ws)  # loaded before the other process writes
        ready = os.path.join(self.root, "ready")
        p = subprocess.Popen([sys.executable, "-c", CHILD_HOLDS, REPO, self.ws, ready])
        self.addCleanup(p.wait)
        end = time.time() + 20
        while not os.path.exists(ready) and time.time() < end:
            time.sleep(0.01)
        self.assertTrue(os.path.exists(ready))
        t0 = time.time()
        stale.add_decision({"point": "p", "choice": "c"})
        waited = time.time() - t0
        self.assertEqual(p.wait(), 0)
        self.assertGreater(waited, 0.3)
        ws = Workspace(self.ws)
        self.assertEqual([a["by"] for a in ws.state["approvals"]], ["child"])
        self.assertEqual([d["point"] for d in ws.decisions()], ["p"])
        self.assertTrue(os.path.isfile(os.path.join(self.ws, "STATE.lock")))

    def test_two_processes_lose_no_update(self):
        # Both processes load STATE before either writes, then write alternately, each through one Workspace
        # object whose in-memory STATE is stale between its writes.
        n = 40
        go = os.path.join(self.root, "go")
        p = subprocess.Popen([sys.executable, "-c", CHILD_LOOP, REPO, self.ws, go, str(n)])
        self.addCleanup(p.wait)
        ws = Workspace(self.ws)
        end = time.time() + 20
        while not os.path.exists(go + ".ready") and time.time() < end:
            time.sleep(0.005)
        open(go, "w").close()
        for i in range(n):
            ws.approve("over_budget", "parent", "q%d" % i)
            time.sleep(0.002)
        self.assertEqual(p.wait(), 0)
        ws = Workspace(self.ws)
        self.assertEqual(len([a for a in ws.state["approvals"] if a["by"] == "parent"]), n)
        self.assertEqual(len(ws.decisions(point="child")), n)

    def test_nested_writes_save_once(self):
        ws = Workspace(self.ws)
        with mock.patch.object(Workspace, "save_state", wraps=ws.save_state) as saved:
            with ws.transaction():
                ws.add_decision({"point": "a", "choice": "x"})
                ws.set_waiting({"kind": "question", "question": "q", "options": ["x"]})
        self.assertEqual(saved.call_count, 1)
        with self.assertRaises(RsiError):
            with ws.transaction():
                ws.clear_waiting()
                ws.add_decision({"point": "b"})  # refused: nothing of the transaction is saved
        ws = Workspace(self.ws)
        self.assertEqual(([d["point"] for d in ws.decisions()], ws.waiting()["question"]), (["a"], "q"))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root can write anywhere")
    def test_lock_that_cannot_be_opened_is_a_refusal(self):
        os.remove(os.path.join(self.ws, "STATE.lock"))
        os.chmod(self.ws, 0o555)
        self.addCleanup(os.chmod, self.ws, 0o755)
        with self.assertRaises(RsiError) as e:
            Workspace(self.ws).add_decision({"point": "p", "choice": "c"})
        self.assertIn("STATE lock", str(e.exception))

    def test_without_fcntl_the_writes_still_work(self):
        ws2 = os.path.join(self.root, "ws2")
        with mock.patch.object(W, "fcntl", None):
            Workspace.init(ws2, os.path.join(self.root, "task"))
            Workspace(ws2).add_decision({"point": "p", "choice": "c"})
        self.assertEqual(len(Workspace(ws2).decisions()), 1)
        self.assertFalse(os.path.exists(os.path.join(ws2, "STATE.lock")))


if __name__ == "__main__":
    unittest.main()
