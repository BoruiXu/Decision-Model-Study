import json
import sys
from pathlib import Path

import yaml

from harness import cli, status
from harness.config import EvaluatorSpec
from harness.finalize import finalize
from harness.result import render_result_md

sys.path.insert(0, str(Path(__file__).parent))
from test_finalize import FakeScorer, make_run  # noqa: E402


def finalized(tmp_path, **kw):
    run_dir, lock = make_run(tmp_path, **kw)
    finalize(run_dir, scorer=FakeScorer())
    return run_dir, lock, json.loads((run_dir / "result.json").read_text()), (run_dir / "RESULT.md").read_text()


def test_result_json_contents(tmp_path):
    run_dir, lock, res, _ = finalized(tmp_path)
    assert res["run_id"] == lock["run_id"] and res["status"] in ("done", "failed")
    d = res["data"]
    assert d["rows"]["train"] == 80 and d["rows"]["select"] == 30 and d["rows"]["report"] == {"a": 15, "b": 15}
    assert d["rows"]["select_by_set"] == {"a": 15, "b": 15}
    assert d["label_version"] == {"train": "lv1", "reference": "lv1"}
    assert d["sha256"]["train.jsonl"] == lock["data"]["sha256"]["train.jsonl"]
    assert res["base"]["name"] == "joey-1" and res["recipe"]["path"].endswith("joey_full_1gpu.yaml")
    assert res["recipe"]["mode"] == "full" and res["recipe"]["sha256"] == lock["recipe"]["sha256"]
    assert res["gpus"]["n"] == 1 and res["gpus"]["cards"][0]["uuid"] == "GPU-0"
    assert res["wall_time"]["seconds"] == 1800.0 and res["wall_time"]["gpu_hours"] == 0.5
    s = res["seeds"]["42"]
    assert [p["step"] for p in s["curve"]] == [0, 4, 8, 10] and s["selected_step"] == 8
    assert s["checkpoint_sha256"] and s["checkpoint"] == "runs/exp-s42/ckpt-step8"
    m = s["metrics"]
    assert m["set"] == "pooled_select" and m["questions"] == 30
    assert list(m["methods"])[:1] == ["model"]
    assert {"model", "tfidf", "group_prior", "length", "all_positive", "oracle"} <= set(m["methods"])
    mm = m["methods"]["model"]
    assert {"auc", "auc_ci95", "within_group_auc", "within_group_auc_ci95", "share_at_budget_10",
            "share_at_budget_10_ci95", "jaccard"} <= set(mm)
    assert set(s["thresholds"]) == {"best_jaccard", "budget_10", "budget_15"}
    assert {"threshold", "small_share", "jaccard"} <= set(s["thresholds"]["budget_10"]["pooled"])
    assert s["gate"]["enabled"] is True and isinstance(s["gate"]["reasons"], list) and s["gate"]["reasons"]
    assert res["bootstrap"]["level"] == "group-then-item"
    cmds = {n["command"]: n for n in res["next"]}
    assert any("eval-report" in c and "--allow-report" in c for c in cmds)
    assert any(c.endswith("--confirm") and "cleanup" in c for c in cmds)
    assert all(n["needs_approval"] for c, n in cmds.items() if "eval-report" in c or "--confirm" in c)


def test_result_md_sections(tmp_path):
    run_dir, lock, res, md = finalized(tmp_path)
    assert md.startswith("# ")
    for text in ("lv1", "joey-1", "joey_full_1gpu.yaml", "GPU-0", "0.50 GPU-hours", "| 8 |", "←",
                 res["seeds"]["42"]["checkpoint_sha256"], "| model |", "| tfidf |", "| length |", "| oracle |",
                 "best_jaccard", "budget_10", "eval-report", "--allow-report", "cleanup", "--confirm",
                 "needs user approval"):
        assert text in md, text
    assert "[" in md and "]" in md  # CIs
    assert "## Gate" in md and "## Warnings" in md and "## Next" in md


def test_gate_failure_reasons_and_warnings(tmp_path):
    run_dir, lock, res, md = finalized(tmp_path, evaluator=EvaluatorSpec(bootstrap_reps=10), drop=True)
    assert res["warnings"]  # dropped ids at least
    assert any("dropped" in w for w in res["warnings"])


def test_render_item_level_and_plugin_shapes():
    res = {"run_id": "r1", "experiment": "e", "run_dir": "/x", "status": "failed",
           "data": {"rows": {"train": 3, "select": 2, "report": {}, "select_by_set": {"a": 2}},
                    "label_version": {"train": "v", "reference": "v"}, "sha256": {"train.jsonl": "ab"},
                    "adapter": {"name": "onejev_rows", "sha256": "cd"}, "allow_test": False},
           "base": {"name": "joey-1", "revision": "rev"}, "recipe": {"path": "/r.yaml", "mode": "full",
                                                                      "parallel": "single", "sha256": "ef"},
           "gpus": {"n": 1, "cards": [], "explicit": None},
           "wall_time": {"seconds": 60.0, "gpu_hours": 1 / 60, "per_seed": {}},
           "seeds": {"1": {"curve": [], "selected_step": None, "checkpoint": None, "checkpoint_sha256": None,
                           "metrics": {"set": "pooled_select", "questions": 2,
                                       "methods": {"model": {"nll": 0.5, "accuracy": 1.0}}},
                           "thresholds": {}, "gate": {"enabled": False, "passed": None, "reasons": []}}},
           "bootstrap": {"reps": 10, "seed": 0, "level": "item", "ci": 0.95, "note": "no group field: item level"},
           "warnings": ["no group field: item level"], "failures": ["seed 1: boom"], "next": []}
    md = render_result_md(res)
    assert "| model | 0.5000 | 1.0000 |" in md.replace("nll | accuracy", "nll | accuracy")
    assert "item level" in md and "seed 1: boom" in md


def test_status_prints_result_path(tmp_path, capsys):
    run_dir, lock = make_run(tmp_path)
    assert status.status(run_dir)["result"] is None
    finalize(run_dir, scorer=FakeScorer())
    (run_dir / "DONE").write_text(json.dumps({"exit_code": 0}))
    assert status.status(run_dir)["result"] == str(run_dir / "RESULT.md")
    assert cli.main(["status", str(run_dir)]) == 0
    assert f"result: {run_dir / 'RESULT.md'}" in capsys.readouterr().out


def test_lock_records_recipe_path_and_sha(tmp_path):
    run_dir, lock = make_run(tmp_path)
    import hashlib
    assert lock["recipe"]["path"].endswith("joey_full_1gpu.yaml")
    assert lock["recipe"]["sha256"] == hashlib.sha256(Path(lock["recipe"]["path"]).read_bytes()).hexdigest()
    assert yaml.safe_load((run_dir / "run.lock.yaml").read_text())["recipe"] == lock["recipe"]


def test_per_eval_set_tables_with_cis(tmp_path):
    run_dir, lock, res, md = finalized(tmp_path)
    by_set = res["seeds"]["42"]["metrics_by_set"]
    assert set(by_set) == {"a", "b"} and by_set["a"]["questions"] == 15
    assert by_set["a"]["methods"]["model"]["auc_ci95"] is not None and "tfidf" in by_set["b"]["methods"]
    assert "### Metrics on eval set a (select part, 15 questions)" in md
    assert "### Metrics on eval set b (select part, 15 questions)" in md
