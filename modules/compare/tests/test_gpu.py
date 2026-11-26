"""The GPU baselines as one long job (zero_shot, few_shot, team_recipe), their re-prediction on a refreshed evalset,
and the milestone's scoring on the testset inputs. No GPU: a fake nvidia-smi, the fake scorer fake_ojh_score.py
(COMPARE_TEST_SCORER) and a fake train command, whose candidate's predict gives every row the trainset's prior; the
fake train command and that predict append "train" and "predict" to $FAKE_CALLS_LOG. Like the real one, the fake train
command fails at packing when its config_request is outside the task's search_space (rsi.space, as `rsi validate
candidate` checks it); the GPU task's search_space covers a tuning dimension only, not the team recipe."""
import json
import os
import random
import shutil
import stat
import subprocess
import sys
import time

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("yaml")   # trsi (the card, model.predict's code) needs it

from conftest import (HERE, TASK, handback, make_brief, make_rows, make_set, manifest, rsi_validate,  # noqa: E402
                      workspace, write_json)
from compare.cli import main  # noqa: E402
from compare.gpu import base_scoring, few_shot_rows  # noqa: E402
from dpolicy.apply import apply_policy  # noqa: E402
from fake_ojh_score import score_checkpoint  # noqa: E402

TEAM = {"base": "tiny-1", "recipe": "tiny_lora.yaml", "overrides": {"onejev.lr": 5e-05}, "seed": 0,
        "max_steps": None}

FAKE_TRAIN = '''#!%s
import argparse, json, os, shutil, stat, subprocess
ap = argparse.ArgumentParser()
for name in ("task", "trainset", "trainset-id", "valset", "valset-id", "config", "config-source", "round", "producer",
             "out", "marker"):
    ap.add_argument("--" + name, required=True)
a = ap.parse_args()
with open(os.environ["FAKE_CALLS_LOG"], "a") as f:
    f.write("train\\n")
rows = [json.loads(x) for x in open(os.path.join(a.trainset, "rows.jsonl")) if x.strip()]
os.makedirs(os.path.join(a.out, "model"))
json.dump({"yes": sum(r["target"]["yes"] for r in rows) / len(rows)}, open(os.path.join(a.out, "model", "prior.json"), "w"))
shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "fake_predict.py"), os.path.join(a.out, "predict.py"))
entry = os.path.join(a.out, "predict")
open(entry, "w").write('#!/bin/sh\\nexec %s "$(dirname "$0")/predict.py" "$@"\\n')
os.chmod(entry, 0o755)
subprocess.run([entry, "--inputs", os.path.join(a.valset, "inputs.jsonl"), "--out",
                os.path.join(a.out, "predictions_val.jsonl")], check=True)
files = {"predictions_val": {"path": "predictions_val.jsonl"}}
if "policy" in json.load(open(a.task)):   # a candidate fits its own policy (its predict would apply it)
    json.dump({"fake": "the candidate's own policy"}, open(os.path.join(a.out, "policy.json"), "w"))
    files["policy"] = {"path": "policy.json"}
request = dict({"seed": 0}, **json.load(open(a.config)))   # the train command's default seed
json.dump({"kind": "candidate", "files": files,
           "model": {"location": "model", "predict": "predict"}, "cost": {"compute_hours": 0.25},
           "config_request": request}, open(os.path.join(a.out, "manifest.json"), "w"))
task = json.load(open(a.task))
if task.get("search_space") is not None:   # packing checks the candidate as `rsi validate candidate` does
    from rsi import space
    bad = space.request_errors(request, task["search_space"], space.default_request(task))
    if bad:
        open(a.marker, "w").write("failed\\npacking the candidate failed: rsi validate candidate: " + "; ".join(bad))
        raise SystemExit(1)
open(a.marker, "w").write("ok\\n")
''' % (sys.executable, sys.executable)

FAKE_PREDICT = '''import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(os.environ["FAKE_CALLS_LOG"], "a") as f:
    f.write("predict\\n")
if os.environ.get("FAKE_PREDICT_OOM") and "evalset" in a.inputs:
    sys.exit("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
p = json.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "prior.json")))["yes"]
with open(a.out, "w") as f:
    for line in open(a.inputs):
        if line.strip():
            probs = {"yes": p, "no": 1 - p}
            d = max(probs, key=probs.get)
            f.write(json.dumps({"id": json.loads(line)["id"], "probs": probs, "decision": d,
                                "uncertainty": 1 - probs[d]}) + "\\n")
'''


def executable(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def wait_for(path, timeout=60):
    end = time.time() + timeout
    while not os.path.exists(path):
        assert time.time() < end, "no %s after %d s" % (path, timeout)
        time.sleep(0.05)


def run_until_done(brief, outdir, mode="first"):
    """`mode` (first), then continue after each long job; the last hand-back."""
    main(["run", "--brief", brief, "--outdir", str(outdir), "--mode", mode])
    hb = handback(outdir)
    while hb["status"] == "long_job":
        wait_for(hb["marker"])
        main(["run", "--brief", brief, "--outdir", str(outdir), "--mode", "continue"])
        hb = handback(outdir)
    return hb


def spend_lines(outdir):
    with open(os.path.join(str(outdir), "spend.jsonl")) as f:
        return [json.loads(x) for x in f if x.strip()]


def read_calls_log():
    path = os.environ["FAKE_CALLS_LOG"]
    return open(path).read().split() if os.path.exists(path) else []


def read_json(path):
    with open(str(path)) as f:
        return json.load(f)


def ids_of(path):
    with open(str(path)) as f:
        return sorted(json.loads(x)["id"] for x in f if x.strip())


@pytest.fixture
def world(tmp_path_factory, monkeypatch):
    """A fake GPU machine: nvidia-smi with one idle card, an ojh repository whose catalog knows the base tiny-1, a
    machine config whose model_dir holds it, and the fake train command."""
    root = tmp_path_factory.mktemp("gpu")
    (root / "bin").mkdir()
    executable(root / "bin" / "nvidia-smi", "#!/bin/sh\necho '0, GPU-fake-0, 3, 81917'\n")
    executable(root / "bin" / "fake_train", FAKE_TRAIN)
    (root / "bin" / "fake_predict.py").write_text(FAKE_PREDICT)
    (root / "ojh" / "catalog" / "bases").mkdir(parents=True)
    (root / "ojh" / "catalog" / "bases" / "tiny-1.yaml").write_text(
        "name: tiny-1\nmultimodal: false\ndefault_recipes:\n  1: examples/recipes/tiny_full_1gpu.yaml\n")
    (root / "ojh" / "examples" / "recipes").mkdir(parents=True)
    (root / "ojh" / "examples" / "recipes" / "tiny_full_1gpu.yaml").write_text(
        "mode: full\ngpus: 1\nonejev:\n  dtype: bfloat16\n  lr: 1.0e-5\n")
    (root / "models" / "tiny-1").mkdir(parents=True)
    (root / "machine.yaml").write_text("python: python3\nmodel_dir: models\nout_root: runs\nenv:\n  HF_HUB_OFFLINE: 1\n")
    monkeypatch.setenv("PATH", "%s%s%s" % (root / "bin", os.pathsep, os.environ["PATH"]))
    monkeypatch.setenv("COMPARE_TEST_SCORER", str(HERE / "fake_ojh_score.py"))
    monkeypatch.setenv("FAKE_CALLS_LOG", str(root / "calls.log"))
    return root


def gpu_task(world, baselines):
    task = json.loads(json.dumps(TASK))
    task["compare"] = {"baselines": list(baselines), "python": sys.executable, "team_recipe": TEAM,
                       "few_shot": {"k": 4, "max_example_chars": 200}}
    task["resources"] = {"onejev": {"python": sys.executable, "ojh_bin": str(world / "ojh" / ".venv" / "bin" / "ojh"),
                                    "machine": str(world / "machine.yaml"),
                                    "default_config": dict(TEAM, recipe=None, overrides={}),
                                    "estimate": {"tiny-1": 0.5}},
                         "gpu_ids": [0]}
    task["search_space"] = {"overrides.onejev.lr": [1e-05, 2e-05], "seed": [0, 1, 2]}
    return task


def fields(world, task, outdir, compute_hours=None):
    return {"resources": task["resources"], "train_command": str(world / "bin" / "fake_train"),
            "spend_file": str(outdir / "spend.jsonl"),
            "budget_left": {"compute_hours": compute_hours, "api_usd": None, "agent_usd": None,
                            "annotation_items": None}}


def register(src, root, bid):
    """Copy a baseline directory into the workspace as registration does, with its id."""
    shutil.copytree(str(src), str(root / bid))
    m = manifest(root / bid)
    m["id"] = bid
    write_json(root / bid / "manifest.json", m)
    return root / bid


@pytest.fixture
def gpu_brief(tmp_path, tmp_path_factory, world):
    task = gpu_task(world, ["zero_shot", "few_shot", "team_recipe"])
    return workspace(tmp_path_factory.mktemp("ws"), task, **fields(world, task, tmp_path))


@pytest.fixture
def gpu_brief_no_hours(tmp_path, tmp_path_factory, world):
    task = gpu_task(world, ["zero_shot", "few_shot", "team_recipe"])
    return workspace(tmp_path_factory.mktemp("ws"), task, **fields(world, task, tmp_path, compute_hours=0))


@pytest.fixture
def milestone_brief(tmp_path_factory, world):
    """A milestone_compare brief: the comparison-0001 of lr, zero_shot, few_shot and team_recipe (trained by the
    module, then registered as baseline-0001..4), the rows of their trainset and valset, the testset's inputs. The task
    has a policy section whose best calibration and rule is always accepted (min_lo -1)."""
    task = gpu_task(world, ["lr", "zero_shot", "few_shot", "team_recipe"])
    task["policy"] = {"min_lo": -1}
    root = tmp_path_factory.mktemp("ws")
    first = workspace(root, task, **fields(world, task, root / "out"))
    scratch = tmp_path_factory.mktemp("first")
    regs = {}
    for i, sub in enumerate(run_until_done(first, scratch)["artifacts"], 1):
        regs[manifest(scratch / sub)["name"]] = register(scratch / sub, root, "baseline-%04d" % i).name
    test = make_set(root, task, "testset", 30, random.Random(3))
    (root / "comparison-0001").mkdir()
    write_json(root / "comparison-0001" / "manifest.json", {"kind": "comparison", "id": "comparison-0001",
                                                            "baselines": regs})
    path = make_brief(root, task, dict((bid, root / bid) for bid in regs.values()), ids={"testset": "testset-0001"},
                      name="milestone.json", purpose="milestone_compare", **fields(world, task, root / "out"))
    b = json.loads(open(path).read())
    for key in ("trainset", "valset", "evalset_inputs"):
        del b["inputs"][key]
    b["inputs"].update({"comparison": str(root / "comparison-0001"),
                        "trainset_rows_trainset-0001": str(root / "trainset" / "rows.jsonl"),
                        "valset_rows_valset-0001": str(root / "valset" / "rows.jsonl"),
                        "testset_inputs": str(test / "inputs.jsonl")})
    write_json(path, b)
    os.remove(os.environ["FAKE_CALLS_LOG"])
    return path


@pytest.fixture
def brief_with_team_baseline_on_old_evalset(tmp_path, tmp_path_factory, world):
    """The team recipe trained by the module on evalset-0001 and registered as baseline-0003; a brief on the
    refreshed evalset-0002."""
    task = gpu_task(world, ["team_recipe"])
    root = tmp_path_factory.mktemp("ws")
    first = workspace(root, task, **fields(world, task, root / "out"))
    scratch = tmp_path_factory.mktemp("first")
    assert run_until_done(first, scratch)["artifacts"] == ["bl-team_recipe"]
    register(scratch / "bl-team_recipe", root, "baseline-0003")
    make_set(root, task, "evalset", 100, random.Random(11), sid="evalset-0002")
    os.remove(os.environ["FAKE_CALLS_LOG"])
    # 0.2 GPU hours left: enough for a prediction (compare.gpu_hours.predict 0.1), not a training (0.5)
    return make_brief(root, task, {"baseline-0003": root / "baseline-0003"}, ids={"evalset": "evalset-0002"},
                      name="refresh.json", **fields(world, task, tmp_path, compute_hours=0.2))


def test_gpu_baselines_run_as_a_long_job(tmp_path, gpu_brief):
    main(["run", "--brief", gpu_brief, "--outdir", str(tmp_path), "--mode", "first"])
    hb = handback(tmp_path)
    assert hb["status"] == "long_job"
    assert os.path.isfile(hb["marker"] + ".pid")
    wait_for(hb["marker"])
    assert open(hb["marker"]).readline().strip() == "ok"
    main(["run", "--brief", gpu_brief, "--outdir", str(tmp_path), "--mode", "continue"])
    assert handback(tmp_path)["artifacts"] == ["bl-zero_shot", "bl-few_shot", "bl-team_recipe"]
    task = json.loads(open(json.loads(open(gpu_brief).read())["task_json"]).read())
    for d in handback(tmp_path)["artifacts"]:
        assert rsi_validate("baseline", tmp_path / d, task) == []
    assert [x["source"] for x in spend_lines(tmp_path)] == [   # team_recipe: its training, then its prediction
        "compare:zero_shot", "compare:few_shot", "compare:team_recipe", "compare:team_recipe"]
    team = manifest(tmp_path / "bl-team_recipe")
    assert os.access(str(tmp_path / "bl-team_recipe" / team["model"]["predict"]), os.X_OK)
    assert read_calls_log() == ["train", "predict", "predict"]   # valset at packing, then the evalset inputs
    # the train command got the task with a search_space that also allows the team recipe's config, else the same
    derived = read_json(tmp_path / "gpu-1" / "team_recipe" / "team" / "task.json")
    assert dict(derived, search_space=task["search_space"]) == task
    assert derived["search_space"] == {"overrides.onejev.lr": [1e-05, 2e-05, 5e-05], "seed": [0, 1, 2],
                                       "base": ["tiny-1"], "recipe": ["tiny_lora.yaml"]}
    # the base is scored as trsi scores a candidate of its default recipe (a full fine-tune on 1 GPU: float32)
    scoring = manifest(tmp_path / "bl-zero_shot")["config"]["scoring"]
    assert (scoring["kwargs"]["dtype"], scoring["kwargs"]["autocast"]) == ("float32", True)


def test_few_shot_prompt_keeps_the_row_and_adds_k_examples():
    task = json.loads(json.dumps(TASK))
    rng = random.Random(1)
    train_rows, rows = make_rows(task, "trainset", 40, rng), make_rows(task, "valset", 3, rng)
    out = few_shot_rows(rows, train_rows, 4, 200, 0, task)
    text = out[0]["question"]["instructions"]
    assert text.startswith("Here are 4 labelled examples.\n\nExample 1:\nInput: {")
    assert text.endswith("\n\nNow answer for the input below.\n\n" + rows[0]["question"]["instructions"])
    assert out[0] == dict(rows[0], question=dict(rows[0]["question"], instructions=text))
    assert {"yes": text.count("\nAnswer: yes"), "no": text.count("\nAnswer: no")} == {"yes": 2, "no": 2}
    assert out[1]["question"]["instructions"].startswith(text[:-len(rows[0]["question"]["instructions"])])


def test_gpu_baselines_skipped_without_compute(tmp_path, gpu_brief_no_hours):
    main(["run", "--brief", gpu_brief_no_hours, "--outdir", str(tmp_path), "--mode", "first"])
    assert handback(tmp_path)["status"] == "ok"                 # no long job
    assert handback(tmp_path)["artifacts"] == ["comparison"]
    assert manifest(tmp_path / "comparison")["skipped"]["team_recipe"] == "needs 0.5 GPU hours, 0 left"
    assert json.loads((tmp_path / "skipped.json").read_text()) == {
        "zero_shot": "needs 0.1 GPU hours, 0 left", "few_shot": "needs 0.1 GPU hours, 0 left",
        "team_recipe": "needs 0.5 GPU hours, 0 left"}


def test_checkin_reports_a_dead_job_as_failed(tmp_path, gpu_brief):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    (tmp_path / "gpu-1.done.pid").write_text("%d\n%d\n" % (p.pid, int(time.time())))
    main(["run", "--brief", gpu_brief, "--outdir", str(tmp_path), "--mode", "checkin"])
    hb = handback(tmp_path)
    assert (hb["status"], hb["notes"]) == ("failed", "the GPU job %d ended without its marker" % p.pid)


def test_milestone_scores_the_baselines_on_the_test_inputs(tmp_path, milestone_brief):
    hb = run_until_done(milestone_brief, tmp_path)
    assert hb["artifacts"] == ["bl-lr-test", "bl-zero_shot-test", "bl-few_shot-test", "bl-team_recipe-test"]
    b = json.loads(open(milestone_brief).read())
    task = json.loads(open(b["task_json"]).read())
    for d in hb["artifacts"]:
        m = manifest(tmp_path / d)
        assert m["testset_id"] == "testset-0001" and m["reused_from"]
        src = os.path.join(b["inputs"][m["reused_from"]], "predictions_val.jsonl")
        assert open(str(tmp_path / d / "predictions_val.jsonl")).read() == open(src).read()
        assert ids_of(tmp_path / d / "predictions_test.jsonl") == ids_of(b["inputs"]["testset_inputs"])
        assert rsi_validate("baseline", tmp_path / d, task) == []
        assert read_json(tmp_path / d / "policy.json") == read_json(os.path.join(b["inputs"][m["reused_from"]],
                                                                                 "policy.json"))
    assert read_calls_log() == ["predict"]                      # the stored team model; nothing trained
    # lr's refit reproduced its predictions_val with its policy applied (else it would be skipped); the team keeps its
    # candidate's policy; zero_shot's new testset scores get its recorded policy
    assert read_json(tmp_path / "bl-team_recipe-test" / "policy.json") == {"fake": "the candidate's own policy"}
    for name in ("lr", "zero_shot", "few_shot"):                # each fit's per-pair report is kept beside its policy
        assert manifest(tmp_path / ("bl-%s-test" % name))["files"]["policy_report"] == {"path": "policy_report.json"}
        assert read_json(tmp_path / ("bl-%s-test" % name) / "policy_report.json")["pairs"]
    policy = read_json(tmp_path / "bl-zero_shot-test" / "policy.json")
    assert policy["accepted"] and policy["settings_sha256"]
    inputs = [dict(json.loads(x), target={}) for x in open(b["inputs"]["testset_inputs"]) if x.strip()]
    raw = score_checkpoint(str(tmp_path), inputs)
    with open(str(tmp_path / "bl-zero_shot-test" / "predictions_test.jsonl")) as f:
        got = dict((r["id"], r) for r in map(json.loads, f))
    for i, rec in raw.items():
        want = apply_policy(policy, {"id": i, "probs": dict(zip(rec["labels"], rec["probs"]))}, ["yes", "no"])
        assert got[i]["decision"] == want["decision"] and got[i]["probs"] == pytest.approx(want["probs"])


def test_a_trained_team_model_survives_a_failed_prediction(tmp_path, tmp_path_factory, world, monkeypatch):
    task = gpu_task(world, ["team_recipe"])
    brief = workspace(tmp_path_factory.mktemp("ws"), task, **fields(world, task, tmp_path))
    monkeypatch.setenv("FAKE_PREDICT_OOM", "1")                 # the evalset prediction runs out of memory
    assert run_until_done(brief, tmp_path)["artifacts"] == ["comparison"]   # of no baseline
    assert "OOM" in json.loads((tmp_path / "skipped.json").read_text())["team_recipe"]
    assert "CUDA out of memory" in (tmp_path / "gpu-1.log").read_text()   # predict's output is in the job log
    monkeypatch.delenv("FAKE_PREDICT_OOM")
    b = json.loads(open(brief).read())
    b["budget_left"]["compute_hours"] = 0.2                     # short of a training (0.5), enough for a prediction
    write_json(brief, b)
    assert run_until_done(brief, tmp_path, "resume")["artifacts"] == ["bl-team_recipe"]
    assert read_calls_log().count("train") == 1                 # the trained model of gpu-1 is used, not retrained
    # the training is billed once, on its own line as soon as it ends; then the failed and the good prediction
    assert [round(x["compute_hours"], 2) for x in spend_lines(tmp_path)] == [0.25, 0.0, 0.0]


def test_base_scoring_follows_the_base_default_recipe(tmp_path):
    """A LoRA default recipe's lora.autocast reaches the scoring kwargs, as ojh turns it into lora_autocast."""
    (tmp_path / "catalog" / "bases").mkdir(parents=True)
    (tmp_path / "catalog" / "bases" / "big-1.yaml").write_text(
        "name: big-1\nmultimodal: true\ndefault_recipes:\n  2-8: examples/big_lora_ddp.yaml\n")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "big_lora_ddp.yaml").write_text(
        "mode: lora\ngpus: 2\nonejev:\n  dtype: bfloat16\nlora:\n  r: 32\n  autocast: true\n")
    got = base_scoring({"ojh_bin": str(tmp_path / ".venv" / "bin" / "ojh")}, "big-1")
    assert got["recipe"] == "examples/big_lora_ddp.yaml"       # no one-card default: the first one
    assert (got["kwargs"]["dtype"], got["kwargs"]["autocast"], got["kwargs"]["multimodal"]) == ("bfloat16", True, True)


def test_refreshed_evalset_repredicts_without_training(tmp_path, brief_with_team_baseline_on_old_evalset):
    run_until_done(brief_with_team_baseline_on_old_evalset, tmp_path)
    m = manifest(tmp_path / "bl-team_recipe")
    assert m["reused_from"] == "baseline-0003" and m["evalset_id"] == "evalset-0002"
    assert read_calls_log() == ["predict"]                      # the fake train command never ran
    assert ids_of(tmp_path / "bl-team_recipe" / "predictions_eval.jsonl")[0].startswith("evalset-0002-")
