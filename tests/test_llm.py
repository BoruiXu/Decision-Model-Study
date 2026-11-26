"""llm (modules/common/llm): one fresh `claude -p` process per call with structured output, no tools and no ambient
context; every call logged and billed to the shared spend file (a killed call never at $0); the caps checked before a
call; a bad reply is a rejection, never a crash; a stop ends the call's process and still bills it.
tests/fake_claude.py stands in for claude."""
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modules", "common"))
sys.path.insert(0, os.path.join(ROOT, "tests"))  # fixtures, also when run as tests.test_llm from the root

from fixtures import binary_task  # noqa: E402
from llm.claude import ClaudeCall, LlmBudgetExceeded, schema_problems  # noqa: E402
from rsi.runners.base import pid_alive  # noqa: E402
from rsi.workspace import spend_line_problems  # noqa: E402

FAKE = os.path.join(ROOT, "tests", "fake_claude.py")
SCHEMA = {"type": "object", "required": ["answer"], "additionalProperties": False,
          "properties": {"answer": {"type": "string"}}}
LOG_KEYS = {"at", "tag", "session", "cost_usd", "billed_usd", "ok", "error", "prompt_sha256", "call"}


class Stop(Exception):
    pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def session_of(call):
    return call["argv"][call["argv"].index("--session-id") + 1]


def option(call, name):
    return call["argv"][call["argv"].index(name) + 1]


class TestClaudeCall(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.log, self.script_file = os.path.join(self.dir, "calls.jsonl"), os.path.join(self.dir, "script.json")
        env = mock.patch.dict(os.environ, {"FAKE_CLAUDE_LOG": self.log, "FAKE_CLAUDE_SCRIPT": self.script_file})
        env.start()
        self.addCleanup(env.stop)
        self.task = binary_task()

    def script(self, replies):
        with open(self.script_file, "w", encoding="utf-8") as f:
            json.dump(replies, f)

    def call(self, work, spend="spend.jsonl", task=None, **kw):
        return ClaudeCall(task or self.task, os.path.join(self.dir, work), os.path.join(self.dir, spend),
                          command=[sys.executable, FAKE], **kw)

    def test_each_call_is_a_fresh_process_and_costs_are_logged(self):
        self.script([{"answer": "a"}] * 9)
        spend = os.path.join(self.dir, "spend.jsonl")
        c = self.call("work", usd_cap=0.3)
        a, b = c.run("p1", SCHEMA, "gen"), c.run("p2", SCHEMA, "gen")
        self.assertEqual((a.output, a.error, a.cost_usd), ({"answer": "a"}, None, 0.12))
        calls = read_jsonl(self.log)
        self.assertNotEqual(session_of(calls[0]), session_of(calls[1]))
        self.assertEqual((a.session, b.session), (session_of(calls[0]), session_of(calls[1])))
        argv = calls[0]["argv"]
        self.assertNotIn("--resume", argv)
        self.assertEqual([argv.count(o) for o in ("-p", "--output-format", "--model", "--effort", "--json-schema",
                                                  "--tools", "--session-id", "--strict-mcp-config",
                                                  "--setting-sources")], [0, 0, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([option(calls[0], o) for o in ("--model", "--effort", "--tools", "--setting-sources")],
                         ["claude-opus-5-5", "xhigh", "", "project"])
        self.assertEqual(json.loads(option(calls[0], "--json-schema")), SCHEMA)
        self.assertEqual([x["stdin"] for x in calls], ["p1", "p2"])  # the prompt on stdin
        self.assertEqual([x["files"] for x in calls], [[], []])       # a new empty directory per call,
        self.assertNotEqual(calls[0]["cwd"], calls[1]["cwd"])
        for x in calls:                                               # outside the workspace, removed after it
            self.assertFalse(os.path.realpath(x["cwd"]).startswith(os.path.realpath(self.dir) + os.sep))
            self.assertFalse(os.path.exists(x["cwd"]))
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "work", "1", "out.json")))
        logged = read_jsonl(os.path.join(self.dir, "work", "llm_calls.jsonl"))
        self.assertEqual([set(x) for x in logged], [LOG_KEYS] * 2)
        self.assertEqual([(x["tag"], x["ok"], x["session"]) for x in logged], [("gen", True, a.session),
                                                                                ("gen", True, b.session)])
        lines = read_jsonl(spend)
        self.assertAlmostEqual(sum(x["api_usd"] for x in lines), 0.24)
        self.assertEqual([(x["source"], spend_line_problems(x)[1]) for x in lines], [("llm:gen", [])] * 2)
        c.run("p3", SCHEMA, "gen")                                    # spent 0.36 >= 0.3 after it
        with self.assertRaises(LlmBudgetExceeded):
            c.run("p4", SCHEMA, "gen")
        self.assertEqual(len(read_jsonl(self.log)), 3)                # a refused call never starts
        own = dict(self.task, runner={"claude": {"args": ["--mcp-config", "a.json", "b.json", "--setting-sources",
                                                          "user", "--mcp-config=c.json", "--verbose"],
                                                 "setting_sources": ["user", "local"]}})
        d = self.call("work2", task=own, usd_cap=None, api_left=0.4, model="claude-sonnet-5", effort="high")
        d.run("p5", SCHEMA, "rev")                                    # the spend file holds 0.36 < 0.4 before it
        last = read_jsonl(self.log)[-1]
        self.assertEqual([option(last, o) for o in ("--model", "--effort", "--setting-sources")],
                         ["claude-sonnet-5", "high", "project"])      # replaced in place, never twice:
        self.assertEqual([last["argv"].count(o) for o in ("--model", "--setting-sources", "--strict-mcp-config",
                                                          "--verbose")], [1, 1, 1, 1])
        self.assertFalse([a for a in last["argv"] if a.startswith("--mcp-config") or a.endswith(".json")
                          or a in ("user", "user,local")])            # no task MCP config, no user settings
        with self.assertRaises(LlmBudgetExceeded):                    # 0.48 >= 0.4: one account for every call
            d.run("p6", SCHEMA, "rev")
        g = self.call("g", spend="other.jsonl", usd_cap=0.2)          # a recipe's generator and reviewer: one cap
        r = self.call("r", spend="other.jsonl", usd_cap=0.2, shared=g)
        g.shared = r
        g.run("p7", SCHEMA, "gen")
        r.run("p8", SCHEMA, "review")
        with self.assertRaises(LlmBudgetExceeded):
            g.run("p9", SCHEMA, "gen")
        self.assertAlmostEqual(g.spent + r.spent, 0.24)
        bad = {"type": "object", "$defs": {"D": {"type": "object"}}, "properties": {
            "m": {"type": "object", "additionalProperties": {}}, "u": {"type": ["number", "string"]},
            "p": {"properties": {"a": {"type": "string"}}, "additionalProperties": False},
            "r": {"$ref": "#/$defs/D"}, "x": {"$ref": "#/$defs/Gone"},
            "q": {"type": "object", "additionalProperties": False, "patternProperties": {"^k": {"type": "object"}}},
            "t": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "object"}]}}}
        self.assertEqual(sorted(x.split(":")[0] for x in schema_problems(bad)), [
            "$", "$.$defs.D", "$.m", "$.p", "$.q.patternProperties.^k", "$.t.prefixItems[1]", "$.u", "$.x"])
        good = {"type": "object", "additionalProperties": False, "$defs": {"V": {"anyOf": [{"type": "number"},
                                                                                           {"type": "string"}]}},
                "properties": {"a": {"type": "object", "additionalProperties": {"$ref": "#/$defs/V"}}}}
        self.assertEqual(schema_problems(good), [])
        with self.assertRaises(ValueError):                           # refused before anything is spent
            g.run("p", bad, "gen")
        self.assertEqual(len(read_jsonl(self.log)), 6)                # p1-p3, p5, p7, p8: nothing else started

    def test_a_bad_reply_is_a_rejection_not_a_crash(self):
        bad_exit = {"type": "result", "is_error": True, "total_cost_usd": 0.07, "session_id": "s",
                    "structured_output": {"answer": "a"}}
        child = os.path.join(self.dir, "child.pid")
        self.script([{"raw": "not json", "exit": 0}, {"raw": "", "exit": 1}, {"answer": 3, "_cost": 0.05},
                     {"raw": json.dumps(bad_exit), "exit": 1}, {"answer": "late", "_sleep": 30, "_child": child},
                     {"answer": "late", "_sleep": 30}, {"answer": "late", "_sleep": 30}])
        c = self.call("work", usd_cap=None, timeout_s=1)
        results = [c.run("p", SCHEMA, "gen") for _ in range(5)]
        with open(child) as f:                                        # a group member ignoring SIGTERM is killed
            pid = int(f.read())
        end = time.time() + 5
        while pid_alive(pid) and time.time() < end:                   # (init reaps it once killed)
            time.sleep(0.05)
        self.assertFalse(pid_alive(pid))
        self.assertEqual([r.output for r in results], [None] * 5)
        self.assertTrue(all(r.error for r in results))
        self.assertIn("schema", results[2].error)
        self.assertIn("time", results[4].error)
        self.assertEqual([r.cost_usd for r in results], [None, None, 0.05, 0.07, None])
        logged = read_jsonl(os.path.join(self.dir, "work", "llm_calls.jsonl"))
        self.assertEqual([x["ok"] for x in logged], [False] * 5)
        lines = read_jsonl(os.path.join(self.dir, "spend.jsonl"))
        self.assertEqual(len(lines), 5)                               # every call billed;
        self.assertTrue(all(spend_line_problems(x)[1] == [] for x in lines))
        self.assertEqual([x["api_usd"] for x in lines], [0, 0, 0.05, 0.07, 0.07])   # a killed one never at $0
        self.assertIn("not reported", lines[0]["note"])
        self.assertIn("highest", lines[4]["note"])
        self.assertAlmostEqual(c.spent, 0.19)
        d = self.call("work2", usd_cap=None, timeout_s=1)
        d.run("p", SCHEMA, "review")                                  # nothing recorded for llm:review yet
        self.assertEqual(read_jsonl(os.path.join(self.dir, "spend.jsonl"))[-1]["api_usd"], 1.0)

        def stop(signum, frame):                                      # as dgen's handler: SIGTERM raises
            raise Stop("stopped")
        old = signal.signal(signal.SIGTERM, stop)
        self.addCleanup(signal.signal, signal.SIGTERM, old)
        threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM)).start()
        e = self.call("work3", usd_cap=None, timeout_s=60)
        t0 = time.time()
        with self.assertRaises(Stop):
            e.run("p", SCHEMA, "gen")
        self.assertLess(time.time() - t0, 20)
        self.assertFalse(pid_alive(read_jsonl(self.log)[-1]["pid"]))  # the in-flight claude is ended,
        last = read_jsonl(os.path.join(self.dir, "spend.jsonl"))[-1]  # and billed
        self.assertEqual((last["source"], last["api_usd"]), ("llm:gen", 0.07))
        self.assertIn("stopped", read_jsonl(os.path.join(self.dir, "work3", "llm_calls.jsonl"))[0]["error"])


if __name__ == "__main__":
    unittest.main()
