"""The CPU baselines end to end: trained, validated, deterministic, reused; and the launcher."""
import json
import subprocess
import sys

import pytest

pytest.importorskip("sklearn")

from conftest import RUN, handback, manifest, rsi_validate, write_json  # noqa: E402
from compare.cpu import gbdt_library  # noqa: E402
from compare.cli import main  # noqa: E402
from compare.inputs import Inputs  # noqa: E402
from compare.reuse import plan  # noqa: E402

try:
    gbdt_library()
except ValueError as e:  # neither xgboost nor lightgbm imports
    pytest.skip(str(e), allow_module_level=True)


def test_cpu_baselines_on_a_small_task(tmp_path, brief, task):
    assert main(["run", "--brief", brief, "--outdir", str(tmp_path), "--mode", "first"]) == 0
    hb = json.loads((tmp_path / "handback.json").read_text())
    assert (hb["status"], hb["artifacts"]) == ("ok", ["bl-lr", "bl-gbdt", "bl-mlp"])
    for d in hb["artifacts"]:
        assert rsi_validate("baseline", tmp_path / d, task) == []
    assert manifest(tmp_path / "bl-lr")["metrics_val"]["auc"] > 0.9
    assert manifest(tmp_path / "bl-gbdt")["config"]["library"] in ("xgboost", "lightgbm")
    assert json.loads((tmp_path / "skipped.json").read_text()) == {}
    first = (tmp_path / "bl-lr" / "predictions_val.jsonl").read_bytes()
    main(["run", "--brief", brief, "--outdir", str(tmp_path / "again"), "--mode", "resume"])
    assert (tmp_path / "again" / "bl-lr" / "predictions_val.jsonl").read_bytes() == first
    made = manifest(tmp_path / "bl-mlp")["created_at"]
    main(["run", "--brief", brief, "--outdir", str(tmp_path), "--mode", "resume"])   # finished work is kept
    assert manifest(tmp_path / "bl-mlp")["created_at"] == made
    assert handback(tmp_path)["artifacts"] == ["bl-lr", "bl-gbdt", "bl-mlp"]


def test_a_registered_baseline_is_reused(tmp_path, brief_with_registered_lr):
    main(["run", "--brief", brief_with_registered_lr, "--outdir", str(tmp_path), "--mode", "first"])
    assert not (tmp_path / "bl-lr").exists()
    assert plan(Inputs.load(brief_with_registered_lr))[0].action == "reuse"
    assert handback(tmp_path)["status"] == "ok"
    assert handback(tmp_path)["artifacts"] == ["comparison"]   # every baseline reused: the comparison at once
    assert manifest(tmp_path / "comparison")["baselines"] == {"lr": "baseline-0001"}


def test_the_launcher_picks_its_python(tmp_path, brief, task):
    b = json.loads(open(brief).read())
    b["compare"]["python"] = None
    write_json(tmp_path / "none.json", b)
    args = ["--outdir", str(tmp_path), "--mode", "first"]
    subprocess.run([str(RUN), "--brief", str(tmp_path / "none.json")] + args, check=True)
    assert handback(tmp_path) == {
        "status": "failed", "artifacts": [], "marker": None, "questions": [],
        "notes": "compare needs a Python with scikit-learn: set compare.python or resources.onejev.python"}
    task["compare"] = {"baselines": ["foo"]}
    write_json(tmp_path / "task.json", task)
    b["resources"] = {"onejev": {"python": sys.executable}}
    b["inputs"]["task"] = b["task_json"] = str(tmp_path / "task.json")
    write_json(tmp_path / "foo.json", b)
    subprocess.run([str(RUN), "--brief", str(tmp_path / "foo.json")] + args, check=True)
    assert (handback(tmp_path)["status"], handback(tmp_path)["artifacts"]) == ("ok", ["comparison"])
    assert manifest(tmp_path / "comparison")["skipped"] == {"foo": "unknown baseline foo"}
