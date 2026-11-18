import json
import sys
from pathlib import Path

import pytest
import yaml

from harness import cleanup, cli, report_eval
from harness.report_eval import ReportRefused
from harness.resolve import Infeasible

sys.path.insert(0, str(Path(__file__).parent))
from test_finalize import make_run, read_jsonl  # noqa: E402

SUBCOMMANDS = ["inspect", "make-split", "run", "setup", "fetch-base", "prepare", "resolve", "launch", "status", "finalize", "eval-report",
               "table", "cleanup", "note"]


def test_cli_help_lists_all_subcommands(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in out


def _locked(tmp_path, seeds=(42,)):
    run_dir, lock = make_run(tmp_path, seeds=seeds)
    seed_dir = run_dir / "runs" / "exp-s42"
    ck = seed_dir / "final"
    ck.mkdir(parents=True, exist_ok=True)
    thr = {"best_jaccard": {"threshold": 0.5}, "budget_10": {"threshold": 0.7}}
    (run_dir / "thresholds.json").write_text(json.dumps({str(s): thr for s in seeds}))
    (run_dir / "comparison.json").write_text(json.dumps(
        {str(s): {"checkpoint": "runs/exp-s42/final", "selected_step": 8} for s in seeds}))
    return run_dir, lock


def fake_scorer(calls):
    def scorer(ckpt, rows, *, base_dir=None, **kw):
        calls.append((ckpt, [r["id"] for r in rows], kw))
        return {r["id"]: {"id": r["id"], "labels": ["yes", "no"], "probs": [0.8, 0.2], "target": [1.0, 0.0]}
                for r in rows}
    return scorer


def test_report_refused_without_flag(tmp_path, capsys):
    run_dir, _ = _locked(tmp_path)
    with pytest.raises(ReportRefused):
        report_eval.run(run_dir, allow_report=False)
    calls = []
    assert cli.main(["eval-report", str(run_dir)]) == 2
    assert "allow-report" in capsys.readouterr().err
    assert not (run_dir / "report_eval.json").exists() and not calls


def test_report_eval_scores_report_rows_with_locked_thresholds(tmp_path):
    run_dir, lock = _locked(tmp_path)
    calls = []
    out = report_eval.run(run_dir, allow_report=True, scorer=fake_scorer(calls))
    ckpt, ids, kw = calls[0]
    assert ckpt == (run_dir / "runs/exp-s42/final").resolve() and kw["dtype"] == "float32"
    select = {r["id"] for r in read_jsonl(lock["data"]["select"])}
    assert len(ids) == 30 and not set(ids) & select  # only report halves (15 per set)
    assert out["seed"] == 42 and out["allow_test"] is False
    assert set(out["sets"]) == {"a", "b", "pooled_report"}
    pooled = out["sets"]["pooled_report"]
    assert pooled["questions"] == 30 and "within_group_auc" in pooled and "ece" in pooled
    assert set(pooled["thresholds"]) == {"best_jaccard", "budget_10"}
    assert pooled["thresholds"]["budget_10"]["threshold"] == 0.7  # locked, not refitted
    assert {"small_share", "jaccard", "inconsistency"} <= set(pooled["thresholds"]["budget_10"])
    assert json.loads((run_dir / "report_eval.json").read_text())["seed"] == 42


def test_report_eval_refuses_test_rows(tmp_path):
    run_dir, lock = _locked(tmp_path)
    path = Path(lock["eval_sets"]["a"]["train_labels"])
    rows = read_jsonl(path)
    rows[0]["split"] = "test"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from harness.prepare import TestSplitRefused
    with pytest.raises(TestSplitRefused):
        report_eval.run(run_dir, allow_report=True, scorer=fake_scorer([]))
    out = report_eval.run(run_dir, allow_report=True, allow_test=True, scorer=fake_scorer([]))
    assert out["allow_test"] is True


def test_cleanup_lists_but_keeps_without_confirm(tmp_path, capsys):
    run_dir = tmp_path / "run"
    d = run_dir / "runs" / "exp-s42" / "checkpoint-4"
    d.mkdir(parents=True)
    (d / "w.bin").write_bytes(b"x" * 10)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (run_dir / "deletion_candidates.json").write_text(json.dumps({"candidates": [
        {"path": str(d), "bytes": 10}, {"path": str(outside), "bytes": 1},
        {"path": str(run_dir / ".." / "elsewhere"), "bytes": 1}]}))
    assert cleanup.delete(run_dir, confirm=False) == []
    assert cli.main(["cleanup", str(run_dir)]) == 0
    assert str(d) in capsys.readouterr().out and d.exists()
    assert [c["path"] for c in cleanup.delete(run_dir, confirm=True)] == [str(d)]
    assert not d.exists() and outside.exists()


def test_resolve_infeasible_exits_2(monkeypatch, capsys):
    def boom(exp, machine, **kw):
        raise Infeasible("too big", ["use 8 gpus"])
    monkeypatch.setattr(cli, "load_experiment", lambda p: None)
    monkeypatch.setattr(cli, "find_machine", lambda p: (None, {"source": "--machine", "path": p}))
    monkeypatch.setattr(cli.resolve, "resolve", boom)
    assert cli.main(["resolve", "e.yaml", "--machine", "m.yaml"]) == 2
    assert "use 8 gpus" in capsys.readouterr().err


def test_launch_prints_json_and_config_error_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(cli.launch, "launch", lambda lock: {"session": "s", "log": "l", "done": "d",
                                                            "run_dir": "r"})
    assert cli.main(["launch", "x.yaml"]) == 0
    assert json.loads(capsys.readouterr().out)["session"] == "s"
    assert cli.main(["finalize", "/nonexistent-run"]) == 1  # no run.lock.yaml: ConfigError
    assert "run.lock.yaml" in capsys.readouterr().err


def test_note_and_table(tmp_path, monkeypatch, capsys):
    run_dir, lock = make_run(tmp_path)
    seen = {}
    monkeypatch.setattr(cli.registry, "note", lambda path, rid, text: seen.update(a=(path, rid, text)) or 2)
    assert cli.main(["note", str(run_dir), "my text"]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert seen["a"] == (lock["machine"]["registry"] or str(Path(lock["machine"]["out_root"]) / "registry.jsonl"),
                         lock["run_id"], "my text")
    monkeypatch.setattr(cli.registry, "table", lambda path, include_incomplete: f"T {path} {include_incomplete}")
    assert cli.main(["table", "--registry", "r.jsonl", "--all"]) == 0
    assert capsys.readouterr().out.strip() == "T r.jsonl True"
    assert cli.main(["table"]) == 1


def test_finalize_and_eval_report_acquire_one_gpu(tmp_path, monkeypatch, capsys):
    run_dir, _ = _locked(tmp_path)
    log = []

    class Lease:
        uuids = ["GPU-9"]

        def release(self):
            log.append("release")

    monkeypatch.setattr(cli.gpu, "acquire", lambda machine, n: log.append(("acquire", n)) or Lease())
    monkeypatch.setattr(cli.finalize, "finalize", lambda rd, devices=None: log.append(("fin", devices)) or
                        {"run_id": "r", "status": "done", "failures": []})
    assert cli.main(["finalize", str(run_dir)]) == 0
    assert log == [("acquire", 1), ("fin", ["GPU-9"]), "release"]
    log.clear()
    monkeypatch.setattr(cli.report_eval, "run", lambda rd, **kw: log.append(kw) or {"seed": 42})
    assert cli.main(["eval-report", str(run_dir), "--allow-report", "--allow-test"]) == 0
    assert log[1]["devices"] == ["GPU-9"] and log[1]["allow_test"] and log[1]["seed"] is None
    assert log[-1] == "release"


def test_cleanup_symlinks_and_protected_paths(tmp_path):
    run_dir = tmp_path / "run"
    seed = run_dir / "runs" / "exp-s42"
    keep, link, out_link = seed / "final", seed / "checkpoint-4", seed / "checkpoint-8"
    keep.mkdir(parents=True)
    (keep / "w.bin").write_bytes(b"k")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "w.bin").write_bytes(b"o")
    link.symlink_to(keep, target_is_directory=True)
    out_link.symlink_to(outside, target_is_directory=True)
    (run_dir / "run.lock.yaml").write_text("x")
    (run_dir / "deletion_candidates.json").write_text(json.dumps({"candidates": [
        {"path": str(link), "bytes": 1}, {"path": str(out_link), "bytes": 1},
        {"path": str(run_dir / "run.lock.yaml"), "bytes": 1}, {"path": str(run_dir / "runs"), "bytes": 1},
        {"path": str(run_dir / "runs" / "exp-s42" / ".." / ".." / "run.lock.yaml"), "bytes": 1}]}))
    done, skipped = cleanup.delete_report(run_dir, confirm=True)
    assert [c["path"] for c in done] == [str(link), str(out_link)]
    assert not link.is_symlink() and not out_link.is_symlink()
    assert (keep / "w.bin").exists() and (outside / "w.bin").exists()
    assert (run_dir / "run.lock.yaml").exists() and (run_dir / "runs").is_dir()
    assert len(skipped) == 3 and all(c["reason"] for c in skipped)


def test_report_eval_keeps_hard_negatives(tmp_path):
    run_dir, lock = _locked(tmp_path)
    path = Path(lock["eval_sets"]["a"]["train_labels"])
    rows = read_jsonl(path)
    assert rows[1]["id"] == "a1"  # a report row
    rows[1]["target"] = {"no": 1.0}
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = report_eval.run(run_dir, allow_report=True, scorer=fake_scorer([]))
    assert out["dropped"] == {} and out["sets"]["pooled_report"]["questions"] == 30


def test_report_eval_honors_the_runs_allow_test(tmp_path):
    from test_finalize import make_run as mk_run
    run_dir, lock = mk_run(tmp_path, allow_test=True)
    (run_dir / "runs" / "exp-s42" / "final").mkdir(parents=True, exist_ok=True)
    (run_dir / "thresholds.json").write_text(json.dumps({"42": {"best_jaccard": {"threshold": 0.5}}}))
    (run_dir / "comparison.json").write_text(json.dumps({"42": {"checkpoint": "runs/exp-s42/final"}}))
    out = report_eval.run(run_dir, allow_report=True, scorer=fake_scorer([]))
    assert out["allow_test"] is True and out["sets"]["pooled_report"]["questions"] == 30


def _machine_yaml(tmp_path, **over):
    d = {"python": "py", "model_dir": str(tmp_path / "models"), "out_root": str(tmp_path / "out"), **over}
    p = tmp_path / "machine.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


def test_prepare_refuses_forbidden_out_root(tmp_path, capsys):
    (tmp_path / "r").mkdir()
    run_dir, lock = make_run(tmp_path / "r")
    exp = tmp_path / "e.yaml"
    exp.write_text(yaml.safe_dump({"name": "fresh", "base": "joey-1", "recipe": "r.yaml",
                                   "data": {"train": lock["data"]["source"], "eval": {"a": lock["eval_sets"]["a"]},
                                            "train_sha256": lock["data"]["source_sha256"]},
                                   "fields": {"label": "target.yes"}}))
    m = _machine_yaml(tmp_path, forbidden_write_roots=[str(tmp_path / "out")])
    assert cli.main(["prepare", str(exp), "--machine", m]) == 1
    err = capsys.readouterr().err
    assert "ConfigError" in err and str(tmp_path / "out") in err and "fresh" in err
    assert not (tmp_path / "out").exists()


def test_fetch_base_refuses_forbidden_model_dir(tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(cli.fetch_base, "fetch_base", lambda name, d: called.append(name))
    m = _machine_yaml(tmp_path, forbidden_write_roots=[str(tmp_path / "models")])
    assert cli.main(["fetch-base", "joey-1", "--machine", m]) == 1
    assert called == [] and str(tmp_path / "models") in capsys.readouterr().err
