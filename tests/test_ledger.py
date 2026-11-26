"""The spend ledger (spec §4.9): GPU hours, data-LLM dollars, agent dollars and annotation items, each against its own
optional budget, the two dollar accounts never added together; a dispatch's spend file, counted once; the stop check
and the over_budget question for every resource; the brief, the report, the pages and `rsi status`; an old workspace
resumed after the upgrade."""
import json
import os
import unittest

import fixtures
import test_dashboard as TD
import test_engine as TE
import test_flow as TF
from rsi import briefs as B
from rsi import cli
from rsi import flow as F
from rsi.dashboard_pages import EVENTS, OVERVIEW
from rsi.live import ledger_text
from rsi.report import report_markdown
from rsi.workspace import Workspace

T = "2026-11-26T00:00:00Z"
NONE = {"compute_hours": 0, "api_usd": 0, "annotation_items": 0}


def write_lines(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class LedgerCase(TF.CompareCase):
    def spend_file(self, did):
        return os.path.join(self.ws, Workspace(self.ws).dispatch(did)["outdir"], "spend.jsonl")

    def events(self, name):
        with open(os.path.join(self.ws, "work", "run.jsonl"), encoding="utf-8") as f:
            return [r for r in (json.loads(line) for line in f if line.strip()) if r["event"] == name]


class TestLedger(LedgerCase):
    def test_spend_lines_and_agent_costs_add_up_once(self):
        did = self.add_dispatch("data", "done")
        write_lines(self.spend_file(did), [{"at": T, "source": "llm:gen", "api_usd": 1.0},
                                           {"at": T, "source": "compare:team_recipe", "compute_hours": 0.25},
                                           {"bad": 1},
                                           {"source": "llm:gen", "api_usd": 0.5},  # no "at": counted all the same
                                           {"at": T, "source": "llm:gen", "api_usd": -1, "annotation_items": 2}])
        with open(self.spend_file(did), "a", encoding="utf-8") as f:
            f.write("not json\n")
        ws = Workspace(self.ws)
        d = ws.dispatch(did)
        self.assertEqual(ws.count_spend(d), {"api_usd": 1.5, "compute_hours": 0.25, "annotation_items": 2})
        self.assertEqual(ws.count_spend(d), NONE)  # the record as it was: STATE is read again under the lock
        self.assertEqual(ws.dispatch(did)["spend_lines"], 6)
        self.assertEqual([(e["line"], e["counted"]) for e in self.events("spend_invalid")],
                         [(3, {}), (4, {"api_usd": 0.5}), (5, {"annotation_items": 2}), (6, {})])
        self.assertEqual(ws.spend_problems(), 4)
        self.assertIn("· spend-file problems 4", ledger_text(ws))  # rsi status, and the other displays
        self.assertIn("- Spend-file problems: 4 lines", report_markdown(ws))
        pages = TD.pages(ws)
        self.assertIn("Spend-file problems: 4 lines", TD.section(pages[OVERVIEW], "stop"))
        self.assertIn("Spend file of %s, line 4: at must be a non-empty string (counted: api_usd 0.5)" % did,
                      pages[EVENTS])
        # A resumed session reports its cumulative cost: each launch adds the increase, once.
        for launch, total in ((1, 1.0), (2, 1.5), (3, 2.25)):
            self.assertTrue(ws.add_agent_cost(did, launch, total, "s-1"))
        self.assertFalse(ws.add_agent_cost(did, 3, 2.25, "s-1"))  # once per launch
        self.assertFalse(ws.add_agent_cost(did, 4, None, "s-1"))  # no result (killed): nothing, the total stays
        self.assertAlmostEqual(Workspace(self.ws).used("agent_usd"), 2.25)
        self.assertTrue(ws.add_agent_cost(did, 5, 0.5, "s-2"))  # a fresh session counts its own total
        self.assertTrue(ws.add_decision_cost("1|handback_failed|d-0001|1|1|0", 0.25))
        self.assertFalse(ws.add_decision_cost("1|handback_failed|d-0001|1|1|0", 0.25))  # once per decision key
        ws = Workspace(self.ws)
        self.assertAlmostEqual(ws.used("api_usd"), 1.5)  # data LLM calls only
        self.assertAlmostEqual(ws.used("agent_usd"), 3.0)  # agent sessions only
        self.assertAlmostEqual(ws.used("compute_hours"), 0.25)
        self.assertEqual(ws.dispatch(did)["spent_sessions"], {"s-1": 2.25, "s-2": 0.5})

    def test_any_budget_stops_the_run_and_asks_over_budget(self):
        task = fixtures.binary_task()
        task["budget"] = {"compute_hours": None, "annotation_items": None, "api_cost": 2, "agent_cost": 10}
        self.use_task(task, "ws-dollars")
        self.sets()
        did = self.add_dispatch("data", "done")
        write_lines(self.spend_file(did), [{"at": T, "source": "llm:gen", "api_usd": 2.0}])
        ws = Workspace(self.ws)
        ws.count_spend(ws.dispatch(did))
        ws.add_agent_cost(did, 1, 3.0)
        ws = Workspace(self.ws)
        stop, why = ws.stop_check()
        self.assertTrue(stop)
        self.assertIn("data LLM $ used 2 reached budget 2", why)
        self.assertNotIn("agent $", why)  # 3 of 10: the accounts are apart
        self.assertEqual(ws.budget_exhausted(), [("data LLM $", 2.0, 2)])
        b = B.build_brief(ws, "train")
        self.assertEqual(b["budget_left"], {"compute_hours": None, "api_usd": 0, "agent_usd": 7.0,
                                            "annotation_items": None})
        self.assertEqual(b["spend_file"], os.path.join(b["outdir"], "spend.jsonl"))
        a = F.next_action(ws)  # before ② (the task lists over_budget)
        self.assertEqual((a["type"], a["what"], a["step"]), ("ask", "over_budget", "train"))
        self.assertIn("data LLM $", a["question"])
        self.assertNotIn("agent $", a["question"])
        # The displays keep the two dollar accounts on separate fields.
        self.assertIn("GPU hours 0/– · data LLM $ 2.00/2 · agent $ 3.00/10 · annotation 0/–", ledger_text(ws))
        self.assertIn("budget: " + ledger_text(ws), cli.status_lines(ws))
        report = report_markdown(ws)
        self.assertIn("- Data LLM $: 2.00 used, budget 2", report)
        self.assertIn("- Agent $: 3.00 used, budget 10", report)
        stop_card = TD.section(self.overview(), "stop")
        for label in ("GPU hours", "Data LLM $", "Agent $"):
            self.assertIn(label, stop_card)
        # The agent account alone stops the run.
        self.use_task(task, "ws-agents")
        did = self.add_dispatch("train", "done")
        Workspace(self.ws).add_agent_cost(did, 1, 10.0)
        self.assertIn("agent $ used 10 reached budget 10", Workspace(self.ws).stop_check()[1])

    def test_compare_asks_only_before_a_new_dispatch(self):
        task_budget = {"compute_hours": 1.5, "annotation_items": None, "api_cost": None}  # the candidate's 1.5
        ws = self.ws_passed_candidate({"baselines": ["lr"]}, budget=task_budget)
        a = F.next_action(ws)
        self.assertEqual((a["type"], a.get("what"), a["step"]), ("ask", "over_budget", "compare"))
        did = self.add_dispatch("compare", "done", purpose="compare")  # dispatched before the budget was reached
        ws = Workspace(self.ws)
        self.put(ws, "baseline", os.path.join(self.ws, ws.dispatch(did)["outdir"], "bl-lr"), ids=TF.VAL_IDS,
                 eval_ids=TF.EVAL_IDS, wrong=(0,))
        a = F.next_action(ws)  # its second part goes on: ③ is report only
        self.assertEqual((a["type"], a.get("mode"), a.get("dispatch_id")), ("resume", "followup", did))

    def test_old_workspace_runs_on_unchanged(self):
        # Review Focus 1: a task without compare, policy or features, STATE's budget_used with compute_hours only, a
        # dispatch record without spend_lines.
        self.sets()
        did = self.add_dispatch("train", "done")
        self.candidate()
        self.edit_state(lambda s: s.update(budget_used={"compute_hours": 1.0}))
        ws = Workspace(self.ws)
        self.assertNotIn("spend_lines", ws.dispatch(did))
        self.assertEqual((ws.used("api_usd"), ws.used("agent_usd")), (0, 0))
        self.assertEqual(ws.count_spend(ws.dispatch(did)), NONE)  # no spend file
        self.assertFalse(ws.stop_check()[0])
        self.assertEqual(F.next_action(ws)["module"], "deliver")  # ③ is off: ② -> ④
        self.assertIn("GPU hours 1/", ledger_text(ws))
        self.assertEqual(B.build_brief(ws, "deliver")["budget_left"]["agent_usd"], None)


class TestEngineSpend(TE.EngineCase):
    def test_the_engine_counts_a_dispatch_spend_file_once(self):
        self.behave("data", spend=[{"source": "llm:gen", "api_usd": 1.5}, {"source": "llm:review", "api_usd": 0.5}])
        self.assertMilestoneAsk(self.run_engine())
        ws = self.state()
        self.assertEqual(ws.dispatch("d-0001")["spend_lines"], 2)
        self.assertAlmostEqual(ws.used("api_usd"), 2.0)
        self.assertEqual(ws.used("agent_usd"), 0)  # the mock agents report no cost


if __name__ == "__main__":
    unittest.main()
