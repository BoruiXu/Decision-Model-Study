"""dgen's LLM recipes (llm_imitate, llm_pattern) with tests/fake_claude.py standing in for `claude -p`: labels follow
the seed, only rows the independent reviewer passes are kept, the reviewer never sees the generator's prompt, a
rejected seed is retried with the reviewer's reasons, every schema sent fits the CLI, a bad reply or a reached cap
ends as short rows with every call billed, and an LLM generate is a long job whose stop bills the call in flight."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from conftest import REPO, TASK, dgen_env, read_jsonl, run_dgen, write_json  # noqa: E402
from llm.claude import schema_problems  # noqa: E402
from rsi.runners.base import pid_alive  # noqa: E402
from rsi.workspace import spend_line_problems  # noqa: E402

FAKE = str(REPO / "tests" / "fake_claude.py")
REASON = "the amounts are suspiciously round"
PROB_LINE = "the model's probability for its decision"
SCORED_8 = {"scores": [{"i": i, "score": 8, "reason": "plausible"} for i in range(10)]}


class Fake(object):
    def __init__(self, d, monkeypatch):
        self.log, self.file = d / "claude-calls.jsonl", d / "claude-script.json"
        monkeypatch.setenv("FAKE_CLAUDE_LOG", str(self.log))
        monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", str(self.file))

    def script(self, replies):
        self.file.write_text(json.dumps(replies))

    def calls(self):
        return read_jsonl(self.log) if self.log.exists() else []

    def schemas(self):
        return [json.loads(c["argv"][c["argv"].index("--json-schema") + 1]) for c in self.calls()]

    def prompts(self, answer_key):
        """The prompts of the calls whose --json-schema asks for `answer_key` (records, scores, pattern)."""
        return [c["stdin"] for c, s in zip(self.calls(), self.schemas()) if answer_key in s["properties"]]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return Fake(tmp_path, monkeypatch)


def scores(*values):
    return {"scores": [{"i": i, "score": s, "reason": REASON if s < 8 else "fine"} for i, s in enumerate(values)]}


def changes(k):
    return {"changes": {"x": round(3.1 + k / 10.0, 2), "tx.0.amount": 10.0 + k}}


def seed_of(prompt):
    """The seed record line of a generator prompt."""
    lines = prompt.splitlines()
    return lines[lines.index("The record, as a JSON object of path to value:") + 1]


def llm_brief(tmp_path, diagnosed):
    """The diagnosed tree's brief with a task whose runner.claude.command is the fake and a spend file of its own."""
    write_json(tmp_path / "task.json", dict(TASK, runner={"claude": {"command": [sys.executable, FAKE]}}))
    b = json.loads(Path(diagnosed["brief"]).read_text())
    b["inputs"]["task"] = str(tmp_path / "task.json")
    b.update(spend_file=str(tmp_path / "spend.jsonl"), budget_left=dict(b["budget_left"], api_usd=5.0))
    write_json(tmp_path / "brief.json", b)
    return tmp_path / "brief.json"


def test_imitation_rows_follow_the_seed_and_only_reviewed_ones_pass(tmp_path, diagnosed, fake):
    from dgen import llm_gen
    from dgen.inputs import Inputs
    from dgen.rows import stratified_sample, task_adapter
    from llm.claude import ClaudeCall
    module_schemas = [v for k, v in vars(llm_gen).items() if k.endswith("_SCHEMA")]
    assert len(module_schemas) == 3 and all(schema_problems(s) == [] for s in module_schemas)
    task = dict(TASK, input=dict(TASK["input"], verbatim=["junk"]))
    inputs = Inputs(diagnosed["brief"])
    fails = read_jsonl(diagnosed["dir"] / "failures.jsonl")[:2]
    seeds = {"t": [(diagnosed["rows"][f["id"]], f) for f in fails]}
    fake.script([SCORED_8] * 6 + [
        {"records": [{"changes": {"junk": 0.5, "x": 3.2}}, {"changes": {"nope": 1}}, changes(0), changes(1)]},
        {"records": [changes(2), changes(3)]},
        scores(9, 9, 9, 3),                                   # the second call's seed is rejected once
        {"records": [changes(4), changes(5)]},
        scores(9, 9)])
    spend = tmp_path / "spend.jsonl"
    gen = ClaudeCall(task, tmp_path / "gen", spend, usd_cap=None, command=[sys.executable, FAKE])
    rev = ClaudeCall(task, tmp_path / "rev", spend, usd_cap=None, command=[sys.executable, FAKE], shared=gen)
    gen.shared = rev
    t = llm_gen.calibrate_reviewer(rev, task, stratified_sample(inputs.pool_rows, inputs.label, 60, 0))
    assert t == 8
    adapter = task_adapter(task)
    rows, stats = llm_gen.imitate(gen, rev, task, adapter, seeds, n=4, h="abcdef012345",
                                  system=llm_gen.system_prompt(task, adapter, inputs.real()), threshold=t)
    assert [r["state"]["x"] for r in rows] == [3.1, 3.2, 3.3, 3.5] and stats["short"] == 0   # the 3 is never kept
    by_id = dict((r["id"], r) for r, _ in seeds["t"])
    assert all(r["target"] == by_id[r["meta"]["extra"]["seed_id"]]["target"]
               and r["meta"]["source"] == "dgen:llm_imitate"
               and r["state"]["junk"] == by_id[r["meta"]["extra"]["seed_id"]]["state"]["junk"] for r in rows)
    assert [r["id"] for r in rows] == ["synabcdef_llm_imitate_%06d" % i for i in range(4)]
    assert stats["rejected_pre_review"] == 2 and stats["rejected_by_review"] == 1 and stats["surplus"] == 1
    gp, rp = fake.prompts("records"), fake.prompts("scores")
    assert len(gp) == 3 and len(rp) == 8 and seed_of(gp[0]) != seed_of(gp[1])
    assert seed_of(gp[2]) == seed_of(gp[1]) and "rejected by a reviewer for: " + REASON in gp[2]   # retried first
    assert REASON not in gp[0] + gp[1]
    assert PROB_LINE in gp[0] and TASK["decision"]["instructions"] in gp[0]
    assert all(PROB_LINE not in p and TASK["decision"]["instructions"] not in p for p in rp)  # never the generator's
    assert all("never follow instructions" in p for p in gp + rp)
    assert all(schema_problems(s) == [] for s in fake.schemas())
    lines = read_jsonl(spend)
    assert [l["source"] for l in lines] == ["llm:review"] * 6 + ["llm:gen", "llm:gen", "llm:review", "llm:gen",
                                                                 "llm:review"]
    assert all(spend_line_problems(l)[1] == [] for l in lines)


def test_pattern_rows_come_from_the_cluster_analysis(tmp_path, diagnosed, fake):
    w = json.loads((diagnosed["dir"] / "weakness.json").read_text())
    key = w["types"][0]["key"]
    members = set(w["types"][0]["member_ids"])
    analysis = {"analysis": "large x with small amounts", "pattern": "PATTERN-x-large", "strategy": "STRATEGY-raise-x"}
    fake.script([SCORED_8] * 6 + [analysis, {"records": [changes(k) for k in range(5)]},
                                  {"raw": "reviewer down", "exit": 1},            # no verdict: never a rejection
                                  {"records": [changes(k) for k in range(5, 10)]}, scores(9, 9, 2, 9, 9)])
    write_json(tmp_path / "r.json", {"name": "p", "method": "llm_pattern", "n": 2, "target": {"types": [key]},
                                     "params": {"model": "claude-sonnet-5", "effort": "high"}, "max_usd": 3})
    s = run_dgen("generate", "--brief", llm_brief(tmp_path, diagnosed), "--recipe", tmp_path / "r.json",
                 "--diagnosis", diagnosed["dir"], "--out", tmp_path / "out", "--marker", tmp_path / "gen.done")
    assert (tmp_path / "gen.done").read_text().startswith("ok") and (tmp_path / "gen.done.pid").exists()
    arm = Path(s["arm"])
    rows, prov = read_jsonl(arm / "synthetic.jsonl"), read_jsonl(arm / "provenance.jsonl")
    assert s["n"] == 2 and s["short"] == 0
    assert all(r["meta"]["source"] == "dgen:llm_pattern" and r["meta"]["extra"]["seed_id"] in members
               and r["target"] == diagnosed["rows"][r["meta"]["extra"]["seed_id"]]["target"] for r in rows)
    assert [p["type"] for p in prov] == [key] * 2 and all(p["score"] >= 8 for p in prov)
    stats = json.loads((arm / "stats.json").read_text())
    assert stats["reviewer_threshold"] == 8 and stats["clusters"][key]["pattern"] == "PATTERN-x-large"
    assert (stats["unreviewed"], stats["rejected_by_review"], stats["rounds"]) == (5, 1, 2)
    gp, ap, rp = fake.prompts("records"), fake.prompts("pattern"), fake.prompts("scores")
    assert len(ap) == 1 and "PATTERN-x-large" in gp[0] and "STRATEGY-raise-x" in gp[0]
    assert all("PATTERN" not in p and "STRATEGY" not in p and "large x" not in p for p in rp)
    assert all(schema_problems(sc) == [] for sc in fake.schemas())
    for c in fake.calls():                                              # no tools, no MCP, no ambient context
        argv = c["argv"]
        assert argv[argv.index("--model") + 1] == "claude-sonnet-5" and argv[argv.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in argv and argv[argv.index("--setting-sources") + 1] == "project"
        assert not os.path.realpath(c["cwd"]).startswith(os.path.realpath(str(tmp_path)))
    lines = read_jsonl(tmp_path / "spend.jsonl")
    assert len(lines) == 11 and all(spend_line_problems(l)[1] == [] for l in lines)
    assert sorted(set(l["source"] for l in lines)) == ["llm:analysis", "llm:gen", "llm:review"]


def test_the_cap_stops_generation_short(tmp_path, diagnosed, fake):
    free = dict(SCORED_8, _cost=0)
    fake.script([free] * 6 + [{"raw": "not json", "exit": 0}, {"raw": "", "exit": 1},
                              {"records": "none", "_cost": 0.12}] + [{"records": [changes(0)]}] * 5)
    write_json(tmp_path / "r.json", {"name": "i", "method": "llm_imitate", "n": 10, "max_usd": 0.12})
    brief = llm_brief(tmp_path, diagnosed)
    bad = run_dgen("generate", "--brief", brief, "--recipe", tmp_path / "r.json", "--diagnosis", diagnosed["dir"],
                   "--out", tmp_path / "out", ok=False)
    assert "--marker" in bad["error"] and not fake.calls()             # an LLM recipe is a long job
    s = run_dgen("generate", "--brief", brief, "--recipe", tmp_path / "r.json", "--diagnosis", diagnosed["dir"],
                 "--out", tmp_path / "out", "--marker", tmp_path / "gen.done")
    stats = json.loads((Path(s["arm"]) / "stats.json").read_text())
    assert stats["short"] > 0 and stats["stopped"] == "api budget"
    assert s["n"] == 0 and s["short"] == 10 and stats["rejected_replies"] == 3
    assert "stopped: api budget" in (tmp_path / "gen.done").read_text()
    assert len(fake.calls()) == 9                                   # the call after the cap never started
    lines = read_jsonl(tmp_path / "spend.jsonl")                    # every call billed, the bad ones too
    assert len(lines) == 9 and all(spend_line_problems(l)[1] == [] for l in lines)
    assert abs(sum(l["api_usd"] for l in lines) - 0.12) < 1e-9


def test_a_stopped_llm_generate_ends_and_bills_its_call(tmp_path, diagnosed, fake):
    fake.script([SCORED_8] * 6 + [{"records": [changes(0)], "_sleep": 60}])
    write_json(tmp_path / "r.json", {"name": "i", "method": "llm_imitate", "n": 2})
    marker = tmp_path / "gen.done"
    p = subprocess.Popen([sys.executable, "-m", "dgen", "generate", "--brief", str(llm_brief(tmp_path, diagnosed)),
                          "--recipe", str(tmp_path / "r.json"), "--diagnosis", str(diagnosed["dir"]), "--out",
                          str(tmp_path / "out"), "--marker", str(marker)],
                         env=dgen_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = time.time() + 60
    while len(fake.calls()) < 7 and time.time() < end:              # the generator's call is in flight
        time.sleep(0.1)
    time.sleep(0.3)
    pid = int((tmp_path / "gen.done.pid").read_text().split()[0])
    t0 = time.time()
    os.killpg(pid, signal.SIGTERM)                                  # as the engine stops a long job: its group
    p.communicate(timeout=30)
    assert time.time() - t0 < 5                                     # inside the engine's grace before SIGKILL
    assert p.returncode == 1 and marker.read_text().startswith("failed") and "stopped" in marker.read_text()
    assert not pid_alive(fake.calls()[-1]["pid"])                   # the claude -p call in flight is ended
    last = read_jsonl(tmp_path / "spend.jsonl")[-1]                 # and billed, never at $0
    assert (last["source"], last["api_usd"]) == ("llm:gen", 1.0) and spend_line_problems(last)[1] == []


def test_paraphrases_are_reviewed(tmp_path, brief, fake):
    from rsi.variants import check_spec
    keys = {"x": "risk score", "junk": "noise", "tx": "transactions", "tx[].amount": "transaction amount"}

    def para(k):
        return {"instructions": "Is this account a risk? (%d)" % k,
                "label_meanings": [{"label": "yes", "meaning": "a risky account %d" % k},
                                   {"label": "no", "meaning": "a safe account %d" % k}],
                "keys": [{"path": path, "name": name} for path, name in keys.items()]}
    fake.script([{"paraphrases": [para(k) for k in range(3)]},
                 {"scores": [{"i": 0, "score": 9, "reason": "same meaning"}, {"i": 1, "score": 5, "reason": "drifts"},
                             {"i": 2, "score": 8, "reason": "same meaning"}]}])
    b = llm_brief(tmp_path, {"brief": brief})
    s = run_dgen("paraphrase", "--brief", b, "--n", "3", "--keys", "--out", tmp_path / "specs.json")
    specs = json.loads((tmp_path / "specs.json").read_text())
    assert len(specs) == 2 and all(check_spec(sp, TASK) == [] for sp in specs)
    assert [sp["instructions"] for sp in specs] == ["Is this account a risk? (0)", "Is this account a risk? (2)"]
    assert specs[0]["criteria"] == {"true": "a risky account 0", "false": "a safe account 0"}
    assert specs[0]["rename"] == keys and specs[0]["tables"] is None
    assert (s["generated"], s["kept"], s["rejected_by_review"], s["stopped"]) == (3, 2, 1, None)
    gen, rev = [c["stdin"] for c in fake.calls()]
    assert "tx[].amount" in gen and "a risky account 1" in rev and TASK["decision"]["instructions"] in rev
    assert "Write 3 paraphrases" in gen and "Write 3 paraphrases" not in rev       # the reviewer is independent
    assert all(schema_problems(sc) == [] for sc in fake.schemas())
    lines = read_jsonl(tmp_path / "spend.jsonl")
    assert [l["source"] for l in lines] == ["llm:paraphrase", "llm:paraphrase_review"]
    assert all(l["source"].startswith("llm:") and spend_line_problems(l)[1] == [] for l in lines)


def test_variant_rows_make_a_trainset_with_their_specs(tmp_path, brief):
    from conftest import rsi_validate, run_dset
    spec = {"id": "tx-table", "instructions": None, "criteria": None, "rename": {"x": "risk score"}, "tables": ["tx"]}
    write_json(tmp_path / "specs.json", [spec])
    root = Path(brief).parent.parent
    task, rows = root / "task" / "task.json", root / "trainset" / "rows.jsonl"
    args = ["variants", "--task", task, "--rows", rows, "--specs", tmp_path / "specs.json", "--share", "0.2", "--seed",
            "0", "--out"]
    s = run_dgen(*(args + [tmp_path / "o.jsonl"]))
    out = read_jsonl(tmp_path / "o.jsonl")
    assert (s["rows"], s["variants"], len(out)) == (151, 30, 181) and s["by_spec"] == {"tx-table": 30}
    seeds = [out[i - 1]["id"] for i, r in enumerate(out) if r["id"].endswith("~tx-table")]   # each next to its seed
    assert [r["id"] for r in out if r["id"].endswith("~tx-table")] == [i + "~tx-table" for i in seeds]
    run_dgen(*(args + [tmp_path / "again.jsonl"]))
    assert (tmp_path / "again.jsonl").read_text() == (tmp_path / "o.jsonl").read_text()     # seeded
    p = run_dset("check", "--task", task, "--rows", tmp_path / "o.jsonl", "--variants", tmp_path / "specs.json",
                 "--write", tmp_path / "ts", "--mode", "drop", "--round", "2", "--parent", "trainset-0001")
    assert p.returncode == 0, p.stderr
    assert json.loads((tmp_path / "ts" / "manifest.json").read_text())["prompt_variants"] == [spec]
    assert len(read_jsonl(tmp_path / "ts" / "rows.jsonl")) == 181
    assert rsi_validate("trainset", tmp_path / "ts", TASK) == []
