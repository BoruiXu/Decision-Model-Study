"""dgen on the conftest tree: pool scoring and diagnosis, the hard/smote/borderline recipes, the quality filters (dedup
against the valset, CLLM), and a trainset built from verified rows by dset grow."""
import copy
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import redirect_stdout

import pytest

pytest.importorskip("sklearn")

from conftest import (DGEN_ID, OLD_RECIPE, TASK, dgen_env, fake_smi, generate, read_jsonl, rsi_validate,  # noqa: E402
                      run_dgen, run_dset, write_json, write_jsonl)
from rsi.rows import question_of  # noqa: E402
from rsi.workspace import spend_line_problems  # noqa: E402


def spend_file(brief):
    return json.loads(brief.read_text())["spend_file"]


def test_diagnosis_finds_the_errors_by_kind(tmp_path, brief):
    s = run_dgen("score-pool", "--brief", brief, "--out", tmp_path, "--marker", tmp_path / "s.done")
    assert (tmp_path / "s.done").read_text().startswith("ok")
    assert (tmp_path / "s.done.pid").read_text().split()[0].isdigit()
    lines = read_jsonl(spend_file(brief))
    assert [l["source"] for l in lines] == ["dgen:score-pool"] and spend_line_problems(lines[0])[1] == []
    scored = [r["id"] for r in read_jsonl(tmp_path / "inputs.jsonl")]
    assert s["rows"] == len(scored) == 150 and "target" not in read_jsonl(tmp_path / "inputs.jsonl")[0]
    assert not set(scored) & set(r["id"] for r in read_jsonl(brief.parent.parent / "trainset" / "rows.jsonl"))
    run_dgen("diagnose", "--brief", brief, "--scores", tmp_path / "pool_predictions.jsonl", "--out", tmp_path)
    d = json.loads((tmp_path / "diagnosis.json").read_text())
    assert set(d["errors"]) == {"yes→no"} and d["types"][0]["signature"][0]["feature"] == "x"
    fails = read_jsonl(tmp_path / "failures.jsonl")
    assert len(fails) == d["errors"]["yes→no"]["n"] and all(f["kind"] == "yes→no" for f in fails)
    assert set(f["type"] for f in fails) == set(t["key"] for t in d["types"])
    assert "FN" in (tmp_path / "diagnosis.md").read_text()
    assert d["findings"] and all(set(f) >= {"issue", "evidence", "try"} for f in d["findings"])
    learn = json.loads((tmp_path / "learnability.json").read_text())
    assert learn["by_kind"]["yes→no"]["n"] == len(fails)
    other = json.loads(brief.read_text())                       # the scores must be the brief's best candidate's
    cand = brief.parent.parent / "candidate-2"
    shutil.copytree(other["inputs"]["best_candidate"], cand)
    write_json(cand / "manifest.json", dict(json.loads((cand / "manifest.json").read_text()), id="candidate-0002"))
    other["inputs"]["best_candidate"] = str(cand)
    write_json(tmp_path / "b2.json", other)
    bad = run_dgen("diagnose", "--brief", tmp_path / "b2.json", "--scores", tmp_path / "pool_predictions.jsonl",
                   "--out", tmp_path / "d2", ok=False)
    assert "candidate-0001" in bad["error"] and "candidate-0002" in bad["error"]


def stopped_run(tmp_path, brief, name, env, ready, *extra):
    """Start score-pool into tmp_path/<name>, wait until ready(), stop it as the engine stops a long job (SIGTERM to
    its process group) and return its marker's text."""
    p = subprocess.Popen([sys.executable, "-m", "dgen", "score-pool", "--brief", str(brief), "--out",
                          str(tmp_path / name), "--marker", str(tmp_path / (name + ".done"))] + list(extra), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end = time.time() + 30
    while not ready() and time.time() < end:
        time.sleep(0.1)
    assert ready(), "score-pool did not get there within 30 s"
    time.sleep(0.2)
    os.killpg(p.pid, signal.SIGTERM)
    p.communicate(timeout=30)
    return (tmp_path / (name + ".done")).read_text()


def test_score_pool_takes_an_idle_card_refuses_without_hours_and_is_billed_when_stopped(tmp_path, brief, monkeypatch):
    from dgen import DgenError, card
    path = fake_smi(tmp_path / "bin", ["0, GPU-busy, 30000", "1, GPU-idle, 10", "2, GPU-other, 0"])
    s = run_dgen("score-pool", "--brief", brief, "--out", tmp_path / "a", "--marker", tmp_path / "a.done",
                 env=dgen_env(PATH=path))
    log = (tmp_path / "a" / "score-pool.log").read_text()
    assert s["card"] == "GPU-idle" and "CUDA_VISIBLE_DEVICES=GPU-idle" in log
    s = run_dgen("score-pool", "--brief", brief, "--out", tmp_path / "own", "--marker", tmp_path / "own.done",
                 env=dgen_env(PATH=path, CUDA_VISIBLE_DEVICES="7"))       # the caller's choice stands, recorded
    meta = json.loads((tmp_path / "own" / "score_pool.json").read_text())
    assert (s["card"], s["cuda_visible_devices"], meta["cuda_visible_devices"]) == (None, "7", "7")
    busy = fake_smi(tmp_path / "bin2", ["0, GPU-busy, 30000", "2, GPU-other, 0"])
    bad = run_dgen("score-pool", "--brief", brief, "--out", tmp_path / "b", "--marker", tmp_path / "b.done",
                   "--card-wait", "0", ok=False, env=dgen_env(PATH=busy))
    assert "no idle card" in bad["error"] and (tmp_path / "b.done").read_text().startswith("failed")
    hang = tmp_path / "hang"                                    # a hung nvidia-smi is an error, never "no card"
    hang.mkdir()
    (hang / "nvidia-smi").write_text("#!/bin/sh\nexec /bin/sleep 5\n")
    os.chmod(hang / "nvidia-smi", 0o755)
    monkeypatch.setattr(card, "SMI_TIMEOUT_S", 0.3)
    monkeypatch.setenv("PATH", str(hang))
    with pytest.raises(DgenError, match="did not answer"):
        card.pick_card([0], 0)
    monkeypatch.setenv("PATH", str(tmp_path / "none"))         # no nvidia-smi at all: no card chosen
    assert card.pick_card([0], 0) is None
    monkeypatch.undo()                                          # the host's PATH again
    zero = json.loads(brief.read_text())
    zero["budget_left"]["compute_hours"] = 0
    write_json(tmp_path / "zero.json", zero)
    bad = run_dgen("score-pool", "--brief", tmp_path / "zero.json", "--out", tmp_path / "z", "--marker",
                   tmp_path / "z.done", ok=False)
    assert "no GPU hours left" in bad["error"] and (tmp_path / "z.done").read_text().startswith("failed")
    asked = tmp_path / "asked"                                  # stopped while waiting for a card: nothing billed
    (asked / "nvidia-smi").parent.mkdir()
    (asked / "nvidia-smi").write_text("#!/bin/sh\ntouch %s/called\nprintf '0, GPU-busy, 30000\\n'\n" % asked)
    os.chmod(asked / "nvidia-smi", 0o755)
    text = stopped_run(tmp_path, brief, "w", dgen_env(PATH=str(asked) + os.pathsep + os.environ["PATH"]),
                       (asked / "called").exists, "--card-wait", "60")
    assert text.startswith("failed") and "while waiting for a card" in text
    log = tmp_path / "c" / "score-pool.log"                     # stopped while scoring: billed, "(stopped)"
    text = stopped_run(tmp_path, brief, "c", dgen_env(PATH=path, FAKE_PREDICT_SLEEP="60"),
                       lambda: log.exists() and "CUDA_VISIBLE_DEVICES" in log.read_text())
    assert text.startswith("failed") and "while scoring" in text
    lines = read_jsonl(spend_file(brief))
    assert [l["source"] for l in lines] == ["dgen:score-pool"] * 3 and "(stopped)" in lines[2]["note"]
    assert "(stopped)" not in lines[0]["note"] + lines[1]["note"] and lines[2]["compute_hours"] > 0


def test_recipes_keep_the_seed_label_and_provenance(tmp_path, diagnosed):
    from dgen.inputs import Inputs
    rows_by_id = diagnosed["rows"]
    fails = set(f["id"] for f in read_jsonl(diagnosed["dir"] / "failures.jsonl"))
    inputs = Inputs(diagnosed["brief"])               # a row generated in an earlier round: known, never real
    assert DGEN_ID not in [r["id"] for r in inputs.real()] and DGEN_ID in [r["id"] for r in inputs.known()]
    anchors = sum(1 for r in inputs.trainset_rows if r["target"]["yes"] == 1.0)
    for method in ("hard", "smote", "borderline"):
        arm = generate(diagnosed, {"name": method, "method": method, "n": 20, "seed": 0}, tmp_path)
        rows = read_jsonl(arm / "synthetic.jsonl")
        stats = json.loads((arm / "stats.json").read_text())
        assert len(rows) == stats["n"] == 20, stats
        for r in rows:
            seed = rows_by_id[r["meta"]["extra"]["seed_id"]] if method != "hard" else rows_by_id[r["id"]]
            assert r["target"] == seed["target"] and r["meta"]["cluster"] == seed["meta"]["cluster"]
            assert r["question"] == question_of(TASK)
            assert method == "hard" or r["meta"]["source"] == "dgen:" + method
        if method == "hard":
            assert set(r["id"] for r in rows) <= fails and rows == [rows_by_id[r["id"]] for r in rows]
        else:
            assert all(r["id"].startswith("syn%s_%s_" % (arm.name[:6], method)) and r["split"] == "train"
                       and r["meta"]["extra"]["recipe"] == arm.name for r in rows)
            partners = [rows_by_id[r["meta"]["extra"]["partner_id"]] for r in rows]
            assert all(len(p["state"]["tx"]) == len(r["state"]["tx"]) and p["target"] == r["target"]
                       for p, r in zip(partners, rows))
            assert stats["anchors"] == anchors and DGEN_ID not in [x for r in rows for x in (
                r["meta"]["extra"]["seed_id"], r["meta"]["extra"]["partner_id"])]
    (tmp_path / "typo.json").write_text(json.dumps({"name": "typo", "method": "smote", "n": 5, "sed": 1}))
    bad = run_dgen("generate", "--brief", diagnosed["brief"], "--recipe", tmp_path / "typo.json", "--out", tmp_path,
                   ok=False)
    assert "sed" in bad["error"]                                     # an unknown key is an input error
    write_json(tmp_path / "old.json", OLD_RECIPE)                    # its ids are in the trainset already
    bad = run_dgen("generate", "--brief", diagnosed["brief"], "--recipe", tmp_path / "old.json", "--out", tmp_path,
                   ok=False)
    assert DGEN_ID in bad["error"] and "seed" in bad["error"]
    again = run_dgen("generate", "--brief", diagnosed["brief"], "--recipe", tmp_path / "hard.json", "--diagnosis",
                     diagnosed["dir"], "--out", tmp_path)
    assert again["existing"]
    newer = tmp_path / "newer"                                       # another diagnosis: never the old arm
    shutil.copytree(diagnosed["dir"], newer)
    write_json(newer / "diagnosis.json", dict(json.loads((newer / "diagnosis.json").read_text()), seed=1))
    bad = run_dgen("generate", "--brief", diagnosed["brief"], "--recipe", tmp_path / "hard.json", "--diagnosis",
                   newer, "--out", tmp_path, ok=False)
    assert "another diagnosis" in bad["error"]


def test_missing_libraries_are_an_input_error(tmp_path, diagnosed, monkeypatch):
    from dgen import cli
    (tmp_path / "r.json").write_text(json.dumps({"name": "s", "method": "smote", "n": 5}))
    monkeypatch.setitem(__import__("sys").modules, "sklearn", None)
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main(["generate", "--brief", str(diagnosed["brief"]), "--recipe", str(tmp_path / "r.json"),
                         "--out", str(tmp_path)])
    assert code == 1 and len(out.getvalue().splitlines()) == 1
    assert "scikit-learn" in json.loads(out.getvalue())["error"]


@pytest.fixture
def arm_with_a_valset_copy_and_flipped_label_rows(tmp_path, diagnosed):
    """A smote arm plus a row whose state is a valset row's (LEAK_ID) and five rows near trainset rows labelled no
    whose target says yes (FLIPPED_IDS)."""
    arm = generate(diagnosed, {"name": "s", "method": "smote", "n": 20, "seed": 1}, tmp_path)
    rows = read_jsonl(arm / "synthetic.jsonl")
    prov = read_jsonl(arm / "provenance.jsonl")
    val = read_jsonl(diagnosed["root"] / "valset" / "rows.jsonl")[0]
    old = [r for r in read_jsonl(diagnosed["root"] / "trainset" / "rows.jsonl") if r["id"] == DGEN_ID][0]
    train_no = [r for r in read_jsonl(diagnosed["root"] / "trainset" / "rows.jsonl")
                if r["target"]["no"] == 1.0 and r["state"]["x"] < 0.3][:5]
    extra = []
    leak = copy.deepcopy(rows[0])
    leak.update(id="leak-0", state=copy.deepcopy(val["state"]))
    extra.append(leak)
    copy_old = copy.deepcopy(rows[1])
    copy_old.update(id="old-copy", state=copy.deepcopy(old["state"]))
    extra.append(copy_old)
    for i, seed in enumerate(train_no):
        r = copy.deepcopy(rows[0])
        state = copy.deepcopy(seed["state"])
        state["x"] = round(state["x"] + 0.001, 6)
        r.update(id="flip-%d" % i, state=state, target={"yes": 1.0, "no": 0.0})
        r["meta"]["extra"] = {"seed_id": seed["id"], "recipe": arm.name}
        extra.append(r)
    write_jsonl(arm / "synthetic.jsonl", rows + extra)
    write_jsonl(arm / "provenance.jsonl", prov + [{"id": r["id"], "method": "smote",
                                                   "seed_id": r["meta"]["extra"]["seed_id"]} for r in extra])
    return arm, "leak-0", ["flip-%d" % i for i in range(len(train_no))]   # and old-copy: the generated trainset row


def test_filters_drop_leaks_and_cllm_hard(diagnosed, arm_with_a_valset_copy_and_flipped_label_rows):
    arm, leak_id, flipped_ids = arm_with_a_valset_copy_and_flipped_label_rows
    assert len(flipped_ids) == 5
    s = run_dgen("filter", "--brief", diagnosed["brief"], "--arm", arm)
    flags = dict((f["id"], f) for f in read_jsonl(arm / "flags.jsonl"))
    assert flags[leak_id]["leak_val"] and flags[leak_id]["removed"]
    assert flags["old-copy"]["copy_of_real"] and flags["old-copy"]["removed"]   # generated rows stay known rows
    assert all(flags[i]["cllm_group"] == "hard" and flags[i]["removed"] for i in flipped_ids)
    verified = read_jsonl(arm / "verified.jsonl")
    assert [r["id"] for r in verified] == [i for i, f in flags.items() if not f["removed"]]
    assert s["kept"] == len(verified) >= 14
    summary = json.loads((arm / "quality_summary.json").read_text())
    assert summary["leak_val"] == 1 and summary["cllm"]["by_label"]["yes"]["hard"] > 0
    assert summary["discriminability"]["yes"]["n"] == summary["by_label"]["yes"]
    assert 0 < len(read_jsonl(arm / "sample.jsonl")) <= 50 and (arm / "quality_report.md").exists()


def test_verified_rows_build_a_trainset_through_dset(tmp_path, diagnosed):
    arm = generate(diagnosed, {"name": "b", "method": "borderline", "n": 20, "seed": 2}, tmp_path)
    run_dgen("filter", "--brief", diagnosed["brief"], "--arm", arm)
    root = diagnosed["root"]
    out = run_dset("grow", "--task", root / "task" / "task.json", "--pool", root / "pool", "--trainset",
                   root / "trainset", "--parent", "trainset-0001", "--with", arm / "verified.jsonl", "--out",
                   tmp_path / "ts", "--round", "2")
    assert out.returncode == 0, out.stderr
    assert rsi_validate("trainset", tmp_path / "ts", TASK) == []
    rows = read_jsonl(tmp_path / "ts" / "rows.jsonl")
    assert len(rows) == 151 + len(read_jsonl(arm / "verified.jsonl"))
    assert all(r["meta"]["source"] == "dgen:borderline" for r in rows[151:])


def test_neardup_groups_close_rows(tmp_path, brief):
    rows = read_jsonl(brief.parent.parent / "trainset" / "rows.jsonl")
    near = copy.deepcopy(rows[0])
    near["id"], near["state"]["junk"] = "near-copy", near["state"]["junk"] + 1e-6           # a leaf moved by 1e-6
    write_jsonl(tmp_path / "rows.jsonl", rows + [near])
    out = run_dgen("neardup", "--brief", brief, "--rows", tmp_path / "rows.jsonl", "--eps-ratio", "0.05", "--write",
                   tmp_path / "kept.jsonl")
    # the copy and its row; and the trainset's generated row DGEN_ID, its seed's x moved by 0.01, with its seed
    seed = [r for r in rows if r["id"] == DGEN_ID][0]["meta"]["extra"]["seed_id"]
    assert out["groups"] == [[rows[0]["id"], "near-copy"], [seed, DGEN_ID]]
    kept = read_jsonl(tmp_path / "kept.jsonl")
    assert [r["id"] for r in kept] == [r["id"] for r in rows if r["id"] != DGEN_ID]   # each group's first row stays
    assert "already exists" in run_dgen("neardup", "--brief", brief, "--rows", tmp_path / "rows.jsonl", "--write",
                                        tmp_path / "kept.jsonl", ok=False)["error"]
