import json
import os
import random
import stat
import sys
from pathlib import Path

import pytest
import yaml

from harness import cli, oneshot, resolve as resolve_mod
from harness.catalog import ROOT, CatalogError
from harness.config import ConfigError
from harness.resolve import Infeasible

sys.path.insert(0, str(Path(__file__).parent))
from test_finalize import mk, write_jsonl  # noqa: E402

CARDS = "0, GPU-a, NVIDIA A100-SXM4-80GB, 80000\n1, GPU-b, NVIDIA A100-SXM4-80GB, 4000\n"
APPS = "GPU-b, 99, python, 76000\n"


def workspace(tmp_path, **exp_extra):
    rng = random.Random(0)
    ws = tmp_path / "ws"
    ws.mkdir()
    write_jsonl(ws / "train.jsonl", [mk(f"t{i}", rng, "train", f"g{i % 3}") for i in range(40)])
    write_jsonl(ws / "val.jsonl", [mk(f"v{i}", rng, "val", f"g{i % 3}") for i in range(12)])
    exp = {"name": "one", "data": {"train": "train.jsonl", "eval": {"val": {"train_labels": "val.jsonl"}}},
           "fields": {"label": "target.yes", "group": "meta.cat", "text": "messages"}, **exp_extra}
    (ws / "exp.yaml").write_text(yaml.safe_dump(exp))
    m = ws / "machine.yaml"
    m.write_text(yaml.safe_dump({"python": "/venv/bin/python", "model_dir": str(tmp_path / "models"),
                                 "out_root": str(tmp_path / "out")}))
    return ws


class Tmux:
    def __init__(self):
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)


def go(ws, tmux=None, **kw):
    kw.setdefault("machine", str(ws / "machine.yaml"))
    return oneshot.run(ws / "exp.yaml", query=lambda uuid=None: CARDS, apps_query=lambda: APPS,
                       plan_runner=lambda lock: f"PLAN_OK\n10 steps for {lock['n_gpus']} ranks\n",
                       fingerprint=lambda: {"pip_freeze_sha256": "ab" * 32, "driver_version": "570"},
                       tmux=tmux or Tmux(), **kw)


def test_run_prepares_resolves_and_launches(tmp_path):
    ws = workspace(tmp_path)
    tmux = Tmux()
    out = go(ws, tmux=tmux, base="joey-1", gpus=[0])
    assert len(tmux.calls) == 1 and tmux.calls[0][:4] == ["tmux", "new-session", "-d", "-s"]
    run_dir = Path(out["run_dir"])
    lock = yaml.safe_load((run_dir / "run.lock.yaml").read_text())
    assert out["session"] == tmux.calls[0][4] and out["log"] == str(run_dir / "runner.log")
    assert out["done"] == str(run_dir / "DONE")
    assert out["base"] == "joey-1" and lock["base"]["name"] == "joey-1"
    assert out["recipe"] == {"path": str(ROOT / "examples/recipes/joey_full_1gpu.yaml"),
                             "source": "catalog default", "mode": "full"}
    assert out["gpus"] == [{"index": 0, "uuid": "GPU-a", "name": "NVIDIA A100-SXM4-80GB", "free_mib": 80000,
                            "need_mib": lock["feasibility"]["need_mib_per_card"], "processes": []}]
    assert out["data_sha256"]["train.jsonl"] == lock["data"]["sha256"]["train.jsonl"]
    assert out["data_sha256"]["source"] == lock["data"]["source_sha256"]
    assert out["machine"]["source"] == "--machine" and lock["machine"]["origin"]["source"] == "--machine"
    assert lock["machine"]["gpu"]["source"] == "explicit" and lock["machine"]["gpu"]["explicit"] == [0]
    assert out["result"] == str(run_dir / "RESULT.md")


def test_missing_base_lists_catalog_and_writes_nothing(tmp_path):
    ws = workspace(tmp_path)
    with pytest.raises(ConfigError) as e:
        go(ws, gpus=[0])
    assert "--base" in str(e.value) and "joey-1" in str(e.value) and "mammoth-1" in str(e.value)
    assert not (tmp_path / "out").exists()
    with pytest.raises(CatalogError, match="unknown base"):
        go(ws, base="nope", gpus=[0])


def test_missing_gpus_is_an_error(tmp_path):
    ws = workspace(tmp_path)
    with pytest.raises(ConfigError, match="--gpus"):
        go(ws, base="joey-1")


def test_flags_override_the_experiment(tmp_path):
    ws = workspace(tmp_path, base="elephant-1", gpus=[0, 1], recipe="examples/recipes/elephant_full_fsdp.yaml")
    out = go(ws, base="joey-1", gpus=[0], recipe=str(ROOT / "examples/recipes/joey_lora_1gpu.yaml"))
    assert out["base"] == "joey-1" and out["recipe"]["source"] == "--recipe" and out["recipe"]["mode"] == "lora"
    assert [g["index"] for g in out["gpus"]] == [0]


def test_experiment_supplies_base_gpus_and_recipe(tmp_path):
    ws = workspace(tmp_path, base="joey-1", gpus=[0], recipe="examples/recipes/joey_lora_1gpu.yaml")
    out = go(ws)
    assert out["recipe"]["source"] == "experiment" and out["recipe"]["mode"] == "lora"


def test_no_default_recipe_for_the_gpu_count(tmp_path):
    ws = workspace(tmp_path)
    tmux = Tmux()
    with pytest.raises(CatalogError, match="no default recipe for 1 GPU"):
        go(ws, tmux=tmux, base="elephant-1", gpus=[0])
    assert tmux.calls == []


def test_infeasible_cards_never_launch(tmp_path):
    ws = workspace(tmp_path)
    tmux = Tmux()
    with pytest.raises(Infeasible) as e:
        go(ws, tmux=tmux, base="joey-1", gpus=[1])
    assert "GPU 1: 4000 MiB free" in str(e.value) and "pid 99" in str(e.value)
    assert any("GPU 0" in a for a in e.value.alternatives)
    assert tmux.calls == []


def test_cli_run_prints_summary_and_exits_2_when_infeasible(tmp_path, monkeypatch, capsys):
    ws = workspace(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text("#!/bin/sh\ncase \"$*\" in\n"
                      f"  *query-compute-apps*) printf '{APPS}' ;;\n  *) printf '{CARDS}' ;;\nesac\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(resolve_mod, "_default_plan_runner", lambda lock: "PLAN_OK\n10 steps for 1 ranks\n")
    monkeypatch.setattr(resolve_mod, "_fingerprint", lambda m: {"pip_freeze_sha256": "x", "driver_version": "1"})
    calls = []
    monkeypatch.setattr(oneshot.launch, "_tmux", calls.append)
    m = str(ws / "machine.yaml")
    assert cli.main(["run", str(ws / "exp.yaml"), "--base", "joey-1", "--gpus", "1", "--machine", m]) == 2
    err = capsys.readouterr().err
    assert "GPU 1" in err and "GPU 0" in err and calls == []
    assert cli.main(["run", str(ws / "exp.yaml"), "--base", "joey-1", "--gpus", "0", "--machine", m]) == 0
    summary = json.loads(capsys.readouterr().out)
    for k in ("run_dir", "session", "log", "done", "base", "recipe", "gpus", "data_sha256"):
        assert k in summary
    assert len(calls) == 1
    assert cli.main(["run", str(ws / "exp.yaml"), "--base", "joey-1", "--gpus", "0,x", "--machine", m]) == 1


def test_busy_cards_are_checked_after_data_and_plan(tmp_path):
    from harness.config import load_experiment
    from harness.resolve import resolve
    ws = workspace(tmp_path)
    m = __import__("harness.config", fromlist=["load_machine"]).load_machine(ws / "machine.yaml")
    exp = load_experiment(ws / "exp.yaml")
    exp.base, exp.gpus = "joey-1", [1]
    exp.recipe = str(ROOT / "examples/recipes/joey_full_1gpu.yaml")
    plans = []
    # no prepared data: the data problem is reported, not the busy card
    with pytest.raises(ConfigError, match="no prepared data"):
        resolve(exp, m, query=lambda uuid=None: CARDS, apps_query=lambda: APPS,
                plan_runner=lambda lock: plans.append(1) or "PLAN_OK\n10 steps for 1 ranks\n",
                fingerprint=lambda: {"driver_version": "570"})
    from harness.prepare import prepare
    prepare(exp, Path(m.out_root) / exp.name / "prepared")
    with pytest.raises(Infeasible, match="GPU 1"):
        resolve(exp, m, query=lambda uuid=None: CARDS, apps_query=lambda: APPS,
                plan_runner=lambda lock: plans.append(1) or "PLAN_OK\n10 steps for 1 ranks\n",
                fingerprint=lambda: {"driver_version": "570"})
    assert plans == [1]  # plan-only ran before the cards were judged


def test_check_validates_everything_but_the_cards(tmp_path):
    ws = workspace(tmp_path)
    fps = []
    out = oneshot.check(ws / "exp.yaml", base="joey-1", gpus=[1], machine=str(ws / "machine.yaml"),
                        query=lambda uuid=None: CARDS, apps_query=lambda: APPS,
                        plan_runner=lambda lock: f"PLAN_OK\n10 steps for {lock['n_gpus']} ranks\n",
                        fingerprint=lambda: fps.append(1) or {"packages_sha256": "x", "driver_version": "570"})
    assert fps == [1] and out["plan"] == {"steps": 10, "ranks": 1, "interval": 2}
    assert out["cards_checked"] is False and out["n_gpus"] == 1 and out["base"] == "joey-1"
    assert out["recipe"]["source"] == "catalog default" and out["fingerprint"]["driver_version"] == "570"
    assert out["estimate_gib_per_card"] == 32.0 and out["data_sha256"]["train.jsonl"]
    exp_dir = tmp_path / "out" / "one"
    assert sorted(p.name for p in exp_dir.iterdir()) == ["prepared"]  # no run dir, no lock


def test_cli_check(tmp_path, monkeypatch, capsys):
    ws = workspace(tmp_path)
    monkeypatch.setattr(resolve_mod, "_default_plan_runner", lambda lock: "PLAN_OK\n10 steps for 1 ranks\n")
    monkeypatch.setattr(resolve_mod, "_fingerprint", lambda m: {"packages_sha256": "x", "driver_version": "1"})
    assert cli.main(["check", str(ws / "exp.yaml"), "--base", "joey-1", "--gpus", "5",
                     "--machine", str(ws / "machine.yaml")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cards_checked"] is False and out["plan"]["steps"] == 10
