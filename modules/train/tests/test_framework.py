"""trsi inside the framework: init from a brief, candidates packed by collect, adopt, watch, report --search, the
prediction entry and the train command (fake ojh, fake scorer)."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from conftest import EL, MM, MODULE, REPO, wait_for, write_jsonl
from rsi.runners.base import stop_process          # what `rsi stop` does to a long job (engine.stop_long_job)
from trsi import archive
from trsi.config import load_run
from trsi.metrics import ece


def _jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _trial(env, seed=0, proposal=None, gpus=0):
    args = ["launch", env.run] + ([proposal] if proposal else ["--start"]) + ["--seed", seed, "--gpus", gpus]
    code, out, err = env.trsi(*args)
    assert code == 0, err
    return json.loads(out[:out.rindex("}") + 1])["run_dir"]


def _tid(env, seed=0):
    return next(t for t, r in archive.trials(load_run(env.run)).items() if r["start"] and r["seed"] == seed)


def _outside_space(req, task):
    """The keys of a config_request registration refuses: a key the task's search_space lists must take an allowed
    value (a list, {values} or {min, max}); any other key must be null or equal resources.onejev.default_config's."""
    space, default = task.get("search_space") or {}, task["resources"]["onejev"]["default_config"]
    flat = {**{k: req[k] for k in ("base", "recipe", "seed", "max_steps")},
            **{f"overrides.{k}": v for k, v in req["overrides"].items()}}
    bad = []
    for k, v in flat.items():
        if k in space:
            a = space[k]
            ok = (a["min"] <= v <= a["max"]) if isinstance(a, dict) and "values" not in a else \
                v in (a["values"] if isinstance(a, dict) else a)
        else:
            d = (default.get("overrides") or {}).get(k[len("overrides."):]) if k.startswith("overrides.") \
                else default.get(k)
            ok = v is None or v == d
        if not ok:
            bad.append(k)
    return bad


def _watch(env, marker):
    p = subprocess.run([sys.executable, "-m", "trsi", "watch", str(env.run), "--marker", str(marker), "--poll", "0.05"],
                       cwd=MODULE, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(MODULE)})
    assert p.returncode == 0, p.stderr
    lines = Path(str(marker) + ".pid").read_text().split()
    pid, started = int(lines[0]), int(lines[1])
    assert abs(started - time.time()) < 30
    return pid


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()[:1] \
        not in ("Z", "")


def test_init_from_brief(env):
    run = yaml.safe_load(env.run.read_text())
    task = json.loads(env.task_json.read_text())
    oj = task["resources"]["onejev"]
    assert run["run_name"] == "d-0002" and run["out_dir"] == str(env.out / "trsi")
    assert run["ojh"] == {"bin": oj["ojh_bin"], "machine": oj["machine"], "root": str(env.ojh_root)}
    assert run["start"]["base"] == "elephant-1" and run["start"]["recipe"] == EL and run["start"]["overrides"] == {}
    assert run["space"] == {"bases": ["elephant-1", "mammoth-1"], "recipes": [EL, MM], "overrides": {
        "onejev.lr": {"values": [5.0e-5, 1.0e-4, 2.0e-4]},
        "onejev.brier_weight": {"values": [0, 1, 3], "dead": {"3": "Elliptic: 3 seeds mean -0.082"}},
        "lora.r": {"range": [8, 128]}}}
    assert run["gpus"] == {"ids": [0, 1, 2, 3]}
    assert run["metric"] == {"primary": "auprc", "guards": {"ece": 0.02, "brier": 0.005}}
    assert run["rules"] == {"seeds": [0, 1, 2], "probe_margin": 0.035, "accept_margin": 0.035, "max_configs": 3}
    assert run["budget"] == {"gpu_hours": 100, "estimate": {"elephant-1": 8, "mammoth-1": 23}}
    assert run["framework"] == {
        "task": str(env.task_json), "trainset": str(env.trainset), "trainset_id": "trainset-0001",
        "valset": str(env.valset), "valset_id": "valset-0001", "outdir": str(env.out), "round": 1,
        "producer": "train", "python": sys.executable, "train_command": str(MODULE / "train"),
        "progress_file": str(env.out / "progress.json")}     # a brief without progress_file: the default
    exp = yaml.safe_load((env.run.parent / run["exp"]).read_text())
    assert exp["data"] == {"adapter": "onejev_rows", "train": str(env.trainset / "rows.jsonl"),
                           "eval": {"val": {"train_labels": str(env.valset / "rows.jsonl")}}}
    assert exp["fields"] == {"label": "target.yes", "group": "meta.state_id"}
    assert exp["evaluator"]["type"] == "binary_decision" and exp["overrides"] == {}

    # round 2: the best version's config is the start; no gpu_ids (every card), no compute budget, default guards
    best = env.root / "artifacts" / "candidate-0003"
    best.mkdir(parents=True)
    (best / "manifest.json").write_text(json.dumps({"id": "candidate-0003", "config_request": {
        "base": "mammoth-1", "recipe": MM, "overrides": {"onejev.lr": 5e-5}, "seed": 1, "max_steps": None}}))
    t = env.write_task(search={"max_configs": 2})
    del t["resources"]["gpu_ids"]
    env.task_json.write_text(json.dumps(t))
    out2 = env.root / "work" / "round-2" / "train"
    env.write_brief(round=2, outdir=str(out2), dispatch_id=None, budget_left={"compute_hours": None},
                    inputs={"task": str(env.task_json), "trainset": str(env.trainset), "valset": str(env.valset),
                            "best_candidate": str(best)}, progress_file=str(env.root / "p" / "progress.json"))
    env.init()
    run2 = yaml.safe_load((out2 / "trsi" / "run.yaml").read_text())
    assert run2["run_name"] == "r2-train"
    assert run2["framework"]["progress_file"] == str(env.root / "p" / "progress.json")
    assert {k: run2["start"][k] for k in ("base", "recipe")} == {"base": "mammoth-1", "recipe": MM}
    assert {k: o["value"] for k, o in run2["start"]["overrides"].items()} == {"onejev.lr": 5e-5}
    assert run2["gpus"] == {"ids": None} and run2["budget"]["gpu_hours"] is None
    assert run2["metric"]["guards"] == {"ece": 0.02, "brier": 0.005} and run2["rules"]["seeds"] == [0, 1, 2]


def test_init_refusal_leaves_no_run_yaml(env):
    # T5: a refused configuration leaves no run.yaml; one left without incumbent.yaml is written again
    t = env.write_task()
    t["search_space"]["overrides.onejev.attention"] = ["causal"]
    env.task_json.write_text(json.dumps(t))
    out = env.root / "o"
    env.write_brief(outdir=str(out))
    code, _, err = env.trsi("init", "--brief", env.brief)
    assert code == 2 and "derives" in err and not (out / "trsi" / "run.yaml").exists()
    (out / "trsi" / "run.yaml").write_text("run_name: stale\n")
    env.write_task()
    code, _, err = env.trsi("init", "--brief", env.brief)
    assert code == 0, err


def test_collect_packs_a_valid_candidate(env, monkeypatch):
    rd = Path(_trial(env, seed=0, gpus=2))
    code, _, err = env.trsi("collect", env.run, "--gpu", 4)
    assert code == 2 and "not in the task's resources.gpu_ids" in err
    calls = env.root / "scorer_calls.jsonl"
    monkeypatch.setenv("FAKE_SCORER_CALLS", str(calls))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")            # the agent's shell: not what scoring uses
    code, out, err = env.trsi("collect", env.run)
    assert code == 0, err
    assert [c["cuda"] for c in _jsonl(calls)] == ["GPU-fake-2"]  # the idle card of gpu_ids, by uuid
    tid = _tid(env)
    cand = env.out / f"cand-{tid}"
    assert str(cand) in out
    v = env.rsi("validate", "candidate", cand, "--task", env.task_json)
    assert v.returncode == 0, v.stdout + v.stderr
    m = json.loads((cand / "manifest.json").read_text())
    lock = json.loads(json.dumps(yaml.safe_load((rd / "run.lock.yaml").read_text())))
    assert (m["config"]["flags"], m["config"]["data"]["sha256"]) == (lock["flags"], lock["data"]["sha256"])
    shown = json.dumps(m["config"])
    assert str(env.valset) not in shown and str(env.trainset) not in shown   # T9: no path to labelled data
    assert m["config_request"] == {"base": "elephant-1", "recipe": EL, "overrides": {}, "seed": 0, "max_steps": None}
    assert _outside_space(m["config_request"], json.loads(env.task_json.read_text())) == []
    assert 2.0 < m["cost"]["compute_hours"] < 2.01                    # 2 GPU hours + the scoring time
    train = MODULE / "train"
    assert m["code_version"] == {"entry": str(train), "sha256": hashlib.sha256(train.read_bytes()).hexdigest()}
    assert (m["trainset_id"], m["valset_id"], m["producer"], m["round"]) == ("trainset-0001", "valset-0001", "train", 1)
    assert not json.loads((cand / "model" / "fake_model.json").read_text()).get("adapter_only")  # LoRA: merged model
    # R3: packed, so collect deleted what ojh's cleanup offers (the unselected checkpoint, the merged model now in the
    # candidate); ojh keeps the selected checkpoint
    assert [p.name for p in rd.glob("runs/*/*-step*")] == ["adapter-step800"]
    # the comparison uses the candidate's numbers: metrics_val and the ECE of predictions_val
    rec = archive.trial_record(load_run(env.run), tid)
    assert rec["status"] == "done" and rec["result"]["metrics"]["auprc"] == m["metrics_val"]["auprc"]
    assert rec["result"]["metrics"]["brier"] == m["metrics_val"]["brier"]
    rows = {r["id"]: r for r in _jsonl(env.valset / "rows.jsonl")}
    preds = _jsonl(cand / "predictions_val.jsonl")
    y = [int(rows[p["id"]]["target"]["yes"] == 1) for p in preds]
    assert rec["result"]["metrics"]["ece"] == pytest.approx(ece(y, [p["probs"]["yes"] for p in preds]))
    assert not (rd / "predictions_select.jsonl").exists()


def test_collect_skips_a_broken_trial(env, monkeypatch):
    """A trial that cannot be packed stays done but unpacked: not compared, not retrained; a later collect packs it;
    launch --force trains it again instead (T11)."""
    monkeypatch.setenv("FAKE_SCORER_FAIL", "elephant-1#1,elephant-1#2")
    for s in (0, 1, 2):
        _trial(env, seed=s)
    code, out, err = env.trsi("collect", env.run)
    assert code == 0, err
    run = load_run(env.run)
    broken = archive.trial_record(run, _tid(env, 2))
    assert broken["status"] == "done" and "the checkpoint does not load" in broken["result"]["pack_error"]
    assert list(Path(broken["run_dir"]).glob("runs/*/merged-step800"))   # kept: the next collect packs it
    assert broken["result"]["candidate_dir"] is None and not (env.out / f"cand-{_tid(env, 2)}").exists()
    assert (env.out / f"cand-{_tid(env, 0)}").is_dir() and "unpacked" in out and "the checkpoint does not load" in out
    from trsi.cli import spent
    assert spent(run) > 6.0                                # its GPU hours are spent all the same
    out = env.trsi("next", env.run)[1]
    assert "incumbent seeds [1, 2] not finished" in out and "the checkpoint does not load" in out
    n = env.n_calls()
    assert "already part of this run" in env.trsi("launch", env.run, "--start", "--seed", 2, "--gpus", 0)[1]
    old = Path(archive.trial_record(run, _tid(env, 1))["run_dir"])
    assert env.trsi("launch", env.run, "--start", "--seed", 1, "--gpus", 0, "--force")[0] == 0
    assert env.n_calls() == n + 1 and [e["gpu_h"] for e in archive.events(run, "retry") if e["charged"]] == [2.0]
    assert not list(old.glob("runs/*/*-step*"))                           # the old run is never packed now
    monkeypatch.delenv("FAKE_SCORER_FAIL")
    env.trsi("collect", env.run)
    assert (env.out / f"cand-{_tid(env, 2)}").is_dir() and env.n_calls() == n + 1
    assert "incumbent seeds" not in env.trsi("next", env.run)[1]


def test_adopt_registered_candidate(env):
    _trial(env, seed=1)
    env.trsi("collect", env.run)
    reg = env.root / "artifacts" / "candidate-0003"
    shutil.copytree(env.out / f"cand-{_tid(env, 1)}", reg)
    m = json.loads((reg / "manifest.json").read_text())
    (reg / "manifest.json").write_text(json.dumps({**m, "id": "candidate-0003"}))
    out2 = env.root / "work" / "round-2" / "train"
    env.write_brief(round=2, outdir=str(out2), dispatch_id="d-0005",
                    inputs={"task": str(env.task_json), "trainset": str(env.trainset), "valset": str(env.valset),
                            "best_candidate": str(reg)})
    env.init()
    env.run = out2 / "trsi" / "run.yaml"
    n = env.n_calls()
    code, out, err = env.trsi("adopt", env.run, "--candidate", reg)
    assert code == 0, err
    assert env.n_calls() == n                              # no retraining
    run = load_run(env.run)
    (tid, rec), = archive.trials(run).items()
    assert rec["seed"] == 1 and rec["status"] == "done" and rec["adopted"] and not rec["charged"]
    assert rec["registered_id"] == "candidate-0003" and rec["result"]["metrics"]["auprc"] == m["metrics_val"]["auprc"]
    assert "incumbent seeds [0, 2] not finished" in env.trsi("next", env.run)[1]
    other = env.root / "artifacts" / "candidate-0004"
    shutil.copytree(reg, other)
    (other / "manifest.json").write_text(json.dumps({**m, "id": "candidate-0004", "config_request": {
        **m["config_request"], "overrides": {"onejev.lr": 5e-5}}}))
    code, _, err = env.trsi("adopt", env.run, "--candidate", other)
    assert code == 2 and "not the start" in err


def test_watch_marker_and_pid(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    rds = [Path(_trial(env, seed=s)) for s in (0, 1)]
    marker = env.out / "trsi" / "watch.marker"
    pid = _watch(env, marker)
    assert _alive(pid) and not marker.exists()
    (rds[0] / "DONE").write_text(json.dumps({"exit_code": 0}))
    time.sleep(0.3)
    assert not marker.exists()
    (rds[1] / "DONE").write_text(json.dumps({"exit_code": 1, "stage": "train", "error": "x"}))   # a failure: still ok
    assert wait_for(marker.exists) and marker.read_text().splitlines()[0] == "ok"
    assert wait_for(lambda: not _alive(pid))


def test_watch_stop_kills_the_trials(env, monkeypatch):
    # T17: the trials launched after the watch started too
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    _trial(env, seed=0)
    marker = env.out / "trsi" / "watch.marker"
    pid = _watch(env, marker)
    _trial(env, seed=1)
    sessions = sorted(r["session"] for r in archive.trials(load_run(env.run)).values())
    stop_process(pid)
    assert wait_for(marker.exists) and marker.read_text().splitlines()[0] == "failed"
    assert sorted(env.tmux_log.read_text().splitlines()) == [f"kill-session -t {s}" for s in sessions]
    assert not _alive(pid)


def test_watch_error_still_writes_a_marker(env, monkeypatch):
    # R4: an error in the background watcher (a DONE that is not a JSON object) still writes a failed marker, with
    # the traceback in its log: the engine never waits for a marker that cannot come. Final review C-3: it ends the
    # run's ojh trainings too (that failed marker ends the long job: nobody would watch them any more)
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    (Path(_trial(env)) / "DONE").write_text("[1]")
    marker = env.out / "trsi" / "watch.marker"
    _watch(env, marker)
    assert wait_for(marker.exists)
    lines = marker.read_text().splitlines()
    assert lines[0] == "failed" and "AttributeError" in lines[1]
    assert "Traceback" in Path(f"{marker}.log").read_text()
    session, = [r["session"] for r in archive.trials(load_run(env.run)).values()]
    assert env.tmux_log.read_text().splitlines() == [f"kill-session -t {session}"]


def test_watch_rides_out_a_transient_read_error(env, monkeypatch):
    # Final review (re-review): one OSError reading a run directory (a flaky NFS read) is retried at the next poll;
    # only WATCH_READ_ERRORS such polls in a row fail the watch. A good poll follows: ok, no training ended
    from trsi import harness
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    _trial(env)
    calls = []

    def flaky(run_dir, session=None):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(116, "Stale file handle")
        return "done"

    monkeypatch.setattr(harness, "state", flaky)
    marker = env.out / "trsi" / "watch.marker"
    code, _, err = env.trsi("watch", env.run, "--marker", marker, "--poll", "0.01", "--foreground")
    assert code == 0 and marker.read_text().splitlines() == ["ok"], err
    assert "Stale file handle" in err
    assert not env.tmux_log.exists() or "kill-session" not in env.tmux_log.read_text()


def test_report_writes_search(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1": 0.8, "elephant-1|onejev.lr=5e-05": 2.5,
                                                       "elephant-1|onejev.lr=0.0002": "fail"}))
    for s in (0, 1, 2):
        _trial(env, seed=s)
    env.trsi("collect", env.run)

    ids = {}

    def report(name):                                    # registration numbers the candidates in hand-back order
        for d in sorted(env.out.glob("cand-*"), key=lambda d: d.stat().st_mtime):
            ids.setdefault(d.name, f"candidate-{len(ids) + 1:04d}")
        if name == "search-a":                           # a JSON object; keys may be paths
            f = env.root / "ids.json"
            f.write_text(json.dumps({str(env.out / d): i for d, i in ids.items()}))
        else:                                            # the engine's registered-<k>.json
            f = env.out / "registered-2.json"
            f.write_text(json.dumps([{"dir": d, "kind": "candidate", "id": i} for d, i in ids.items()]
                                    + [{"dir": "record-1", "kind": "record", "id": "record-0001"}]))
        code, out, err = env.trsi("report", env.run, "--search", env.out / name, "--ids", f)
        assert code == 0, err
        v = env.rsi("validate", "search", env.out / name, "--task", env.task_json)
        assert v.returncode == 0, v.stdout + v.stderr
        return dict(ids), json.loads((env.out / name / "manifest.json").read_text())

    ids, s = report("search-a")                         # only the start: best is the default
    default = ids[f"cand-{_tid(env, 0)}"]
    assert s["default_candidate"] == s["best_candidate"] == default and s["gain_vs_default"]["delta"] == 0
    p = env.proposal("lr-half", overrides={"onejev.lr": 5e-5})
    for seed in (0, 1, 2):
        _trial(env, seed=seed, proposal=p)
        env.trsi("collect", env.run)
        env.trsi("next", env.run)
    assert archive.incumbent(load_run(env.run))[0].overrides["onejev.lr"].value == 5e-5
    _trial(env, seed=0, proposal=env.proposal("lr-double", overrides={"onejev.lr": 2e-4}))
    env.trsi("collect", env.run)
    ids, s = report("search-b")
    assert (s["kind"], s["trainset_id"], s["valset_id"]) == ("search", "trainset-0001", "valset-0001")
    by_dir = {}
    for t in s["trials"]:
        assert set(t) == {"candidate_id", "config", "metric_val", "compute_hours", "status"}
        by_dir[t["candidate_id"]] = t
    assert len(s["trials"]) == 7 and [t["status"] for t in s["trials"]].count("failed") == 1
    assert all(_outside_space(t["config"], json.loads(env.task_json.read_text())) == [] for t in s["trials"])
    failed = next(t for t in s["trials"] if t["status"] == "failed")
    assert failed["candidate_id"] is None and failed["metric_val"] is None and failed["config"]["overrides"] == {
        "onejev.lr": 2e-4} and failed["compute_hours"] == 1.0
    for d, cid in ids.items():
        m = json.loads((env.out / d / "manifest.json").read_text())
        t = by_dir[cid]
        assert t["config"] == m["config_request"] and t["metric_val"] == m["metrics_val"]["auprc"]
        assert t["compute_hours"] == m["cost"]["compute_hours"] and t["status"] == "ok"
    accepted = [cid for cid, t in by_dir.items() if t["config"]["overrides"] == {"onejev.lr": 5e-5}]
    best = max(accepted, key=lambda c: by_dir[c]["metric_val"])
    assert s["default_candidate"] == default and s["best_candidate"] == best
    assert s["gain_vs_default"] == {"metric": "auprc",
                                    "delta": by_dir[best]["metric_val"] - by_dir[default]["metric_val"]}
    assert s["gain_vs_default"]["delta"] > 0


def test_predict_runs_from_candidate_dir_only(env, tmp_path):
    _trial(env, seed=0)
    env.trsi("collect", env.run)
    cand = tmp_path / "elsewhere" / "cand"
    shutil.copytree(env.out / f"cand-{_tid(env)}", cand)
    shutil.rmtree(tmp_path / "ojh_runs")
    shutil.rmtree(env.out)
    assert sys.executable in (cand / "predict").read_text()   # the interpreter written at training time
    keep = ("state_id", "cluster", "source", "n_obs")          # the meta keys input rows carry
    rows = [{**r, "meta": {k: r["meta"][k] for k in keep}} for r in _jsonl(env.valset / "inputs.jsonl")]
    write_jsonl(tmp_path / "inputs.jsonl", rows)
    calls = tmp_path / "scorer_calls.jsonl"
    p = subprocess.run([str(cand / "predict"), "--inputs", str(tmp_path / "inputs.jsonl"), "--out",
                        str(tmp_path / "p.jsonl")], capture_output=True, text=True, cwd=tmp_path,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "3", "FAKE_SCORER_CALLS": str(calls)})
    assert p.returncode == 0, p.stderr
    got = _jsonl(tmp_path / "p.jsonl")
    assert got == _jsonl(cand / "predictions_val.jsonl")
    for r in got:
        assert set(r) == {"id", "probs", "decision", "uncertainty"}
        assert r["decision"] == max(r["probs"], key=r["probs"].get)
        assert r["uncertainty"] == pytest.approx(1 - max(r["probs"].values()))
    call, = _jsonl(calls)
    assert call["model_dir"] == str(cand / "model") and call["base_dir"] is None
    assert (call["cuda"], call["device"], call["kw"]["multimodal"]) == ("3", "cuda:0", True)


def test_train_command_contract(env, tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_TRAIN_GPUS", "1")
    monkeypatch.setenv("RSI_TRAIN_POLL", "0.05")
    task = env.write_task(search_space={"seed": [0, 1, 2], "max_steps": [30]})   # the default: joey-1, recipe null
    task["resources"]["onejev"]["default_config"] = {"base": "joey-1", "recipe": None, "overrides": {}, "seed": 0,
                                                     "max_steps": None}
    env.task_json.write_text(json.dumps(task))

    def train(config, name, producer="train"):
        cfg = tmp_path / f"{name}.json"
        cfg.write_text(json.dumps(config))
        out, marker = tmp_path / name, tmp_path / f"{name}.marker"
        p = subprocess.run(env.train_argv(cfg, out, marker, producer), capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(REPO)})
        return p, out, marker.read_text().splitlines() if marker.exists() else None

    calls = tmp_path / "scorer_calls.jsonl"
    monkeypatch.setenv("FAKE_SCORER_CALLS", str(calls))
    monkeypatch.setenv("FAKE_TMUX_GONE", "1")                           # ojh's session ends with its run
    monkeypatch.setenv("FAKE_GPU_BUSY_ONCE", str(tmp_path / "busy"))     # I2: no card can score at the first try
    p, out, marker = train({"seed": 1, "max_steps": 30}, "c1")
    assert p.returncode == 0 and marker == ["ok"], p.stderr
    assert "can score this candidate now" in p.stdout                    # tried again once a card was free
    monkeypatch.delenv("FAKE_GPU_BUSY_ONCE")
    assert [r[3] for r in json.loads((tmp_path / "progress.json").read_text())["table"]["rows"]] == ["完成"]
    assert [c["cuda"] for c in _jsonl(calls)] == ["GPU-fake-1"]   # scored on the card it trained on
    v = env.rsi("validate", "candidate", out, "--task", env.task_json)
    assert v.returncode == 0, v.stdout + v.stderr
    m = json.loads((out / "manifest.json").read_text())
    assert m["config_request"] == {"base": "joey-1", "recipe": None, "overrides": {}, "seed": 1, "max_steps": 30}
    assert _outside_space(m["config_request"], task) == []
    assert _outside_space({**m["config_request"], "base": "elephant-1", "max_steps": 40}, task) == ["base", "max_steps"]
    assert m["config"]["flags"]["max_steps"] == 30 and m["config"]["base"]["name"] == "joey-1"
    assert m["config"]["recipe"]["path"] == str(env.ojh_root / "examples" / "recipes" / "joey_full_1gpu.yaml")
    assert (m["config_source"], m["producer"], m["code_version"]["entry"]) == ("default", "train", str(MODULE / "train"))
    assert env.calls.read_text().splitlines()[-1].endswith(f"--gpus 1 --machine {env.root / 'machine.yaml'}")
    rd, = Path(os.environ["FAKE_OJH_OUT"]).glob("*/r0")
    assert [p.name for p in rd.glob("runs/*/*-step*")] == ["ckpt-step800"]    # R3: the unselected one deleted
    selected, = rd.glob("runs/*/ckpt-step800/fake_model.json")
    assert os.stat(out / "model" / "fake_model.json").st_ino == os.stat(selected).st_ino   # T7: a hard link
    assert train({"seed": 1, "max_steps": 30}, "c1b")[2] == ["ok"]          # T4: another --out, another ojh name
    names = [x.split()[0] for x in env.calls.read_text().splitlines()]
    assert len(names) == 2 and names[0] != names[1]
    for producer in ("tune", "data"):                                       # (c): train only; data does not train
        p, _, marker = train({}, "c0", producer=producer)
        assert p.returncode == 2 and marker is None and "must be train" in p.stderr, producer

    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"joey-1": "fail"}))
    p, out, marker = train({}, "c2")
    assert p.returncode != 0 and marker[0] == "failed" and "CUDA out of memory" in "\n".join(marker)
    assert not (out / "manifest.json").exists()

    monkeypatch.delenv("FAKE_TMUX_GONE")
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")                  # stopped while training: ends its ojh session
    (tmp_path / "c3.json").write_text("{}")
    marker = tmp_path / "c3.marker"
    proc = subprocess.Popen(env.train_argv(tmp_path / "c3.json", tmp_path / "c3", marker),
                            env={**os.environ, "PYTHONPATH": str(REPO)}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    assert wait_for(lambda: list((tmp_path / "c3.trsi").glob("trsi/trials/*.json")))
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=20) != 0
    lines = marker.read_text().splitlines()
    assert lines[0] == "failed" and "SIGTERM" in lines[1]
    assert env.tmux_log.read_text().startswith("kill-session -t ojh-test-")
