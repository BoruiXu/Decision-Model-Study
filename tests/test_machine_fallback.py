import os
from pathlib import Path

import pytest
import yaml

from harness import cli
from harness.catalog import ROOT
from harness.config import ConfigError, find_machine


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for k in ("OJH_MACHINE", "OJH_OUT_ROOT", "OJH_MODEL_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    return home


def machine_file(path, out_root):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"python": "py", "model_dir": "/models", "out_root": str(out_root)}))
    return path


def test_explicit_path_wins(clean_env, tmp_path, monkeypatch):
    a = machine_file(tmp_path / "a.yaml", "/out-a")
    monkeypatch.setenv("OJH_MACHINE", str(machine_file(tmp_path / "b.yaml", "/out-b")))
    machine_file(clean_env / ".config" / "onejev-harness" / "machine.yaml", "/out-c")
    m, origin = find_machine(str(a))
    assert m.out_root == "/out-a" and origin == {"source": "--machine", "path": str(a)}


def test_env_var_before_user_config(clean_env, tmp_path, monkeypatch):
    b = machine_file(tmp_path / "b.yaml", "/out-b")
    monkeypatch.setenv("OJH_MACHINE", str(b))
    machine_file(clean_env / ".config" / "onejev-harness" / "machine.yaml", "/out-c")
    m, origin = find_machine(None)
    assert m.out_root == "/out-b" and origin["source"] == "$OJH_MACHINE"


def test_user_config_before_inferred(clean_env):
    c = machine_file(clean_env / ".config" / "onejev-harness" / "machine.yaml", "/out-c")
    m, origin = find_machine(None)
    assert m.out_root == "/out-c" and origin == {"source": "user config", "path": str(c)}


def test_missing_env_file_is_an_error(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("OJH_MACHINE", str(tmp_path / "nope.yaml"))
    with pytest.raises(ConfigError, match="OJH_MACHINE"):
        find_machine(None)


def test_inferred_defaults(clean_env, tmp_path):
    m, origin = find_machine(None)
    assert origin == {"source": "inferred", "path": None}
    assert m.python == str(ROOT / ".venv" / "bin" / "python")
    assert m.out_root == str(clean_env / "ojh_runs")  # ~/ojh_runs, never the cwd (which may be the repo)
    assert m.model_dir == str(clean_env / ".cache" / "onejev-harness" / "models")
    assert m.registry_path == str(clean_env / "ojh_runs" / "registry.jsonl")
    assert m.lock_dir is None and m.gpu_source == "explicit"


def test_inferred_defaults_from_env(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("OJH_OUT_ROOT", "runs-here")
    monkeypatch.setenv("OJH_MODEL_DIR", str(tmp_path / "weights"))
    m, _ = find_machine(None)
    assert m.out_root == str(tmp_path / "runs-here") and os.path.isabs(m.out_root)
    assert m.model_dir == str(tmp_path / "weights")


def test_prepare_works_without_machine(clean_env, tmp_path):
    import json
    row = {"id": "a", "split": "train", "state": {"messages": [{"role": "user", "content": "hi"}]},
           "question": {"type": "noul", "instructions": "?", "criteria": {"yes": "y", "no": "n"}},
           "target": {"yes": 1.0}}
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "val.jsonl").write_text(json.dumps(dict(row, id="b", split="val")) + "\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"name": "e", "data": {"train": "train.jsonl",
                                                         "eval": {"val": {"train_labels": "val.jsonl"}}},
                                   "fields": {"label": "target.yes"}}))
    assert cli.main(["prepare", str(exp)]) == 0
    assert (clean_env / "ojh_runs" / "e" / "prepared" / "prepared.json").is_file()


def test_lock_records_machine_origin_and_missing_base_lists_catalog(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_resolve import run, setup
    exp, machine = setup(tmp_path)
    lock = yaml.safe_load(run(exp, machine, machine_origin={"source": "inferred", "path": None}).read_text())
    assert lock["machine"]["origin"] == {"source": "inferred", "path": None}
    assert lock["machine"]["gpu"]["source"] == "scan"
    exp.base = None
    with pytest.raises(ConfigError, match="joey-1.*mammoth-1|elephant-1"):
        run(exp, machine)


def test_inferred_out_root_is_not_the_repo_even_when_run_there(clean_env, monkeypatch):
    monkeypatch.chdir(ROOT)
    m, _ = find_machine(None)
    assert m.out_root == str(clean_env / "ojh_runs")


def test_out_root_or_model_dir_inside_the_repo_is_refused(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("OJH_OUT_ROOT", str(ROOT / "ojh_runs"))
    with pytest.raises(ConfigError, match="inside the repo"):
        find_machine(None)
    monkeypatch.delenv("OJH_OUT_ROOT")
    monkeypatch.setenv("OJH_MODEL_DIR", str(ROOT / "models"))
    with pytest.raises(ConfigError, match="model_dir.*inside the repo"):
        find_machine(None)
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"python": "py", "model_dir": "/models", "out_root": str(ROOT / "runs")}))
    with pytest.raises(ConfigError, match="out_root.*inside the repo"):
        find_machine(str(p))


def test_gitignore_covers_ojh_runs():
    assert "ojh_runs/" in (ROOT / ".gitignore").read_text().split()
