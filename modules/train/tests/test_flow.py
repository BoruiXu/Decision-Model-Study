"""End to end with the fake ojh and the fake scorer: probe -> more seeds -> accept, reuse, budget, failures."""
import errno
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import fake_ojh
from conftest import MM, REPO, age, wait_for
from rsi.progress import read_progress
from trsi import archive, harness, progress
from trsi.cli import spent
from trsi.config import load_run


def _start(env):
    for s in (0, 1, 2):
        code, out, err = env.trsi("launch", env.run, "--start", "--seed", s, "--gpus", 0)
        assert code == 0, err
    assert env.trsi("collect", env.run)[0] == 0


def test_probe_more_seeds_then_accept(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1": 0.8, "mammoth-1|onejev.lr=5e-05": 2.5}))
    _start(env)
    p = env.proposal("mm1-lr-half", base="mammoth-1", recipe=MM, overrides={"onejev.lr": 5e-5},
                     prediction={"auprc": 0.91, "p_accept": 0.3})
    code, _, err = env.trsi("launch", env.run, p, "--seed", 1, "--gpus", 0)   # seed 1 before the probe
    assert code == 2 and "comes after the probe" in err
    assert env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 1)[0] == 0
    env.trsi("collect", env.run)
    code, out, _ = env.trsi("next", env.run)
    assert "probe passed" in out and "launch seeds [1, 2]" in out
    for s in (1, 2):
        assert env.trsi("launch", env.run, p, "--seed", s, "--gpus", 2)[0] == 0
    env.trsi("collect", env.run)
    code, out, _ = env.trsi("next", env.run)
    assert "ACCEPTED" in out
    run = load_run(env.run)
    inc, cid = archive.incumbent(run)
    assert inc.base == "mammoth-1" and inc.overrides["onejev.lr"].value == 5e-5
    assert [e["event"] for e in archive.events(run)].count("incumbent") == 2
    assert env.n_calls() == 6
    rep = run.home / "report.md"
    assert env.trsi("report", env.run)[0] == 0 and cid in rep.read_text() and "cand-" in rep.read_text()


def test_probe_reject(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1": 1.5, "elephant-1|onejev.lr=0.0002": 0.3}))
    _start(env)
    p = env.proposal("lr-double", overrides={"onejev.lr": 2e-4})   # weaker than the start
    env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    env.trsi("collect", env.run)
    code, out, _ = env.trsi("next", env.run)
    assert "rejected (probe seed 0 gain" in out
    code, _, err = env.trsi("launch", env.run, p, "--seed", 1, "--gpus", 0)
    assert code == 2 and "comes after the probe" in err


def test_reuse_without_training(env):
    _start(env)
    p = env.proposal("lr-half", overrides={"onejev.lr": 5e-5})
    env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    env.trsi("collect", env.run)
    n = env.n_calls()
    code, out, _ = env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    assert code == 0 and "already part of this run" in out and env.n_calls() == n
    env.write_run(run_name="r2")                       # another run under the same out_dir
    env.trsi("init", env.run)
    code, out, _ = env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    assert "reused" in out and env.n_calls() == n
    code, out, _ = env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    assert "reused" in out and env.n_calls() == n
    run = load_run(env.run)
    assert archive.trials(run) and spent(run) > 0      # the reused trials are charged to run r2


def test_budget(env):
    env.write_run(budget={"gpu_hours": 10, "estimate": {"elephant-1": 8, "mammoth-1": 23}})
    _start(env)                                         # the start's trials are candidates too: charged
    run = load_run(env.run)
    assert 6.0 < spent(run) < 6.1                       # 3 x (2 GPU h + the scoring time)
    code, _, err = env.trsi("launch", env.run, env.proposal("mm", base="mammoth-1", recipe=MM), "--seed", 0, "--gpus", 0)
    assert code == 2 and "budget" in err
    code, _, err = env.trsi("launch", env.run, env.proposal("lr", overrides={"onejev.lr": 5e-5}), "--seed", 0, "--gpus", 0)
    assert code == 0, err                               # about 2 GPU h measured: 6 + 2 < 10


def test_failed_run_is_charged_and_not_compared(env, monkeypatch):
    _start(env)
    before = spent(load_run(env.run))
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1|onejev.lr=0.0002": "fail"}))
    p = env.proposal("lr-double", overrides={"onejev.lr": 2e-4})
    env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
    code, out, _ = env.trsi("collect", env.run)
    assert "failed" in out and "CUDA out of memory" in out
    rd = Path(next(r for r in archive.trials(load_run(env.run)).values() if r["status"] == "failed")["run_dir"])
    assert (rd / "DONE").is_file() and not list(rd.glob("runs/*/ckpt-step*"))   # R3: its checkpoints deleted
    assert spent(load_run(env.run)) - before == 1.0     # 3600 s x 1 card
    code, out, _ = env.trsi("next", env.run)
    assert "probe seed 0 failed" in out and "ACCEPTED" not in out and "rejected" not in out
    monkeypatch.setenv("FAKE_OJH_QUALITY", "{}")
    real, calls = harness.launch, []

    def busy_once(*args, **kw):  # final review C-1: a relaunch ojh refuses charges nothing
        calls.append(1)
        if len(calls) == 1:
            raise harness.HarnessError("ojh run exited with code 1: card busy")
        return real(*args, **kw)

    monkeypatch.setattr(harness, "launch", busy_once)
    assert env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)[0] == 2
    assert env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)[0] == 0   # relaunch; the failed hour stays charged
    assert len(archive.events(load_run(env.run), "retry")) == 1
    env.trsi("collect", env.run)
    assert 3.0 < spent(load_run(env.run)) - before < 3.01


def test_pending_run_and_read_result(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    code, out, _ = env.trsi("collect", env.run)
    assert "running" in out
    rd = next(iter(archive.trials(load_run(env.run)).values()))["run_dir"]
    assert harness.state(rd) == "running"
    monkeypatch.delenv("FAKE_OJH_PENDING")
    env.trsi("launch", env.run, "--start", "--seed", 1, "--gpus", 0)
    rd = next(r for r in archive.trials(load_run(env.run)).values() if r["seed"] == 1)["run_dir"]
    r = harness.read_result(rd, 1)
    assert r.status == "done" and r.gpu_hours == 2.0 and r.gpu_names == ["NVIDIA A100"]
    assert r.selected_step == 800 and r.last_step == 1200 and r.checkpoint.endswith("adapter-step800")


def test_lost_session_is_failed(env, monkeypatch):
    # T1, I1: an ojh run whose tmux session is gone and that has written no DONE, and changed nothing, for
    # LOST_GRACE_S has failed: a watch ends, collect records it and the same launch trains it again; the checkpoints of
    # the replaced run go once its DONE turns up
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    monkeypatch.setenv("FAKE_TMUX_GONE", "1")
    assert "running" in env.trsi("collect", env.run)[1]                  # just gone: ojh may still write DONE
    old = Path(next(iter(archive.trials(load_run(env.run)).values()))["run_dir"])
    age(old, harness.LOST_GRACE_S + 60)
    marker = env.out / "trsi" / "watch.marker"
    assert env.trsi("watch", env.run, "--marker", marker, "--foreground", "--poll", 0.01)[0] == 0
    assert marker.read_text().splitlines()[0] == "ok"
    code, out, _ = env.trsi("collect", env.run)
    assert "failed" in out and "ended without DONE" in out
    n = env.n_calls()
    monkeypatch.delenv("FAKE_OJH_PENDING")
    assert env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)[0] == 0 and env.n_calls() == n + 1
    fake_ojh.finish_pending(old, "fail")                                  # its DONE comes late, a checkpoint with it
    assert list(old.glob("runs/*/ckpt-step*"))
    env.trsi("collect", env.run)
    assert not list(old.glob("runs/*/ckpt-step*"))


def test_cleanup_left_is_said_once(env, monkeypatch):
    # M3: what ojh's cleanup leaves (and why) is reported once, not tried again at every collect
    monkeypatch.setenv("FAKE_OJH_CLEANUP_SKIP", "1")
    env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    out = env.trsi("collect", env.run)[1]
    assert "ojh cleanup left 2 path(s)" in out and "not strictly below" in out
    assert "ojh cleanup left" not in env.trsi("collect", env.run)[1]
    assert archive.events(load_run(env.run), "cleanup") == []


def test_collect_retries_after_oom(env, monkeypatch):
    # (a), T22: the card is taken between training and packing (out of memory): packed again once on the next card
    # that qualifies, which the result records; --gpu is checked like any other card
    calls = env.root / "scorer.jsonl"
    monkeypatch.setenv("FAKE_SCORER_CALLS", str(calls))
    monkeypatch.setenv("FAKE_SCORER_OOM_CARD", "GPU-fake-2")                 # card 2 has the most free memory
    env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    code, out, err = env.trsi("collect", env.run)
    assert code == 0, err
    assert [json.loads(x)["cuda"] for x in calls.read_text().splitlines()] == ["GPU-fake-2", "GPU-fake-1"]
    run = load_run(env.run)
    (tid, rec), = archive.trials(run).items()
    assert archive.state_of(rec) == "done" and rec["result"]["scoring_card"] == "GPU-fake-1"
    assert archive.events(run, "result")[-1]["scoring_card"] == "GPU-fake-1"
    monkeypatch.delenv("FAKE_SCORER_OOM_CARD")
    env.trsi("launch", env.run, "--start", "--seed", 1, "--gpus", 0)
    code, out, err = env.trsi("collect", env.run, "--gpu", 0)             # card 0 runs another user's process
    assert code == 0 and "card 0" in out and "another process" in out, err
    assert json.loads(calls.read_text().splitlines()[-1])["cuda"] == "GPU-fake-2"


def test_max_configs_per_dispatch(env, monkeypatch):
    # (b): search.max_configs counts the configurations of the whole dispatch, returns included, and says so
    env.write_run(rules={"seeds": [0, 1, 2], "max_configs": 1})
    monkeypatch.setenv("FAKE_OJH_QUALITY", json.dumps({"elephant-1|onejev.lr=5e-05": 0.1}))   # rejected at the probe
    _start(env)
    assert env.trsi("launch", env.run, env.proposal("a", overrides={"onejev.lr": 5e-5}), "--seed", 0, "--gpus", 0)[0] == 0
    code, _, err = env.trsi("launch", env.run, env.proposal("b", overrides={"onejev.lr": 2e-4}), "--seed", 0, "--gpus", 0)
    assert code == 2 and "this dispatch (returns included)" in err
    env.trsi("collect", env.run)
    assert "STOP: this dispatch (returns included)" in env.trsi("next", env.run)[1]


def test_stop_during_launch_kills_the_training(env, tmp_path, monkeypatch):
    # T12: a SIGTERM while ojh starts the training (its tmux session exists, the train command has not recorded it yet)
    # is handled once the session is recorded: the training is ended
    for k, v in (("FAKE_OJH_PENDING", "1"), ("FAKE_OJH_SLEEP", "1"), ("RSI_TRAIN_GPUS", "1")):
        monkeypatch.setenv(k, v)
    (tmp_path / "c.json").write_text("{}")
    marker = tmp_path / "c.marker"
    proc = subprocess.Popen(env.train_argv(tmp_path / "c.json", tmp_path / "c", marker),
                            env={**os.environ, "PYTHONPATH": str(REPO)}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    assert wait_for((Path(os.environ["FAKE_OJH_OUT"]) / "sleeping").exists)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=20) != 0 and marker.read_text().splitlines()[0] == "failed"
    assert env.tmux_log.read_text().startswith("kill-session -t ojh-test-")


def test_train_command_error_while_ending_the_training_still_writes_a_marker(env, tmp_path, monkeypatch):
    # Final review C-2: the training's state cannot be read (a DONE that is no JSON object): the train command fails,
    # and ending the training fails the same way; the failed marker is written all the same
    for k, v in (("FAKE_OJH_PENDING", "1"), ("RSI_TRAIN_GPUS", "1"), ("RSI_TRAIN_POLL", "0.05")):
        monkeypatch.setenv(k, v)
    (tmp_path / "c.json").write_text("{}")
    marker = tmp_path / "c.marker"
    proc = subprocess.Popen(env.train_argv(tmp_path / "c.json", tmp_path / "c", marker),
                            env={**os.environ, "PYTHONPATH": str(REPO)}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    assert wait_for(lambda: list((tmp_path / "c.trsi").glob("trsi/trials/*.json")))
    trial, = (tmp_path / "c.trsi").glob("trsi/trials/*.json")
    (Path(json.loads(trial.read_text())["run_dir"]) / "DONE").write_text("[1]")
    assert proc.wait(timeout=20) != 0
    lines = marker.read_text().splitlines()
    assert lines[0] == "failed" and "ended the ojh training" in lines[1]
    assert env.tmux_log.read_text().startswith("kill-session -t ojh-test-")


def test_check(env):
    code, out, _ = env.trsi("check", env.run, "--ojh")
    assert code == 0 and "ojh check" in out and "mode lora" in out


def test_env_file_sets_the_ojh_path(env, monkeypatch):
    monkeypatch.delenv("TRSI_TEST_OJH_DIR", raising=False)
    (env.root / "env.sh").write_text(f"export TRSI_TEST_OJH_DIR={env.ojh_root / '.venv' / 'bin'}\n")
    env.write_run(ojh={"bin": "$TRSI_TEST_OJH_DIR/ojh", "env_file": str(env.root / "env.sh"),
                       "root": str(env.ojh_root)})
    code, out, err = env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)
    assert code == 0, err
    assert env.n_calls() == 1


def test_progress_rows_and_facts(env, monkeypatch):
    monkeypatch.setenv("FAKE_OJH_PENDING", "1")
    code, _, err = env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 2)
    assert code == 0, err
    path = env.out / "progress.json"
    p = json.loads(path.read_text())
    assert [f["label"] for f in p["facts"]] == ["配置", "GPU 小时", "当前方案"]
    (tid, rec), = archive.trials(load_run(env.run)).items()
    (row,) = p["table"]["rows"]
    assert row[:5] == [tid, "起点", 0, "训练 5/32 步 · 剩约 12 分", "2"]
    assert "1 个训练在跑" in p["summary"]
    assert read_progress(str(path))["problems"] == []
    monkeypatch.delenv("FAKE_OJH_PENDING")
    fake_ojh.finish_pending(rec["run_dir"])
    launched = progress._epoch(archive.events(load_run(env.run), "launch")[-1]["t"])
    os.utime(Path(rec["run_dir"]) / "DONE", (launched + 360, launched + 360))   # trained 6 minutes
    assert env.trsi("collect", env.run)[0] == 0
    (row,) = json.loads(path.read_text())["table"]["rows"]
    assert row[3] == "完成" and isinstance(row[6], float)
    assert row[5] == "6 分"        # to when ojh wrote DONE, not to when collect packed it


def test_progress_failure_never_stops_the_watch(env, monkeypatch):
    # Review Focus 2: when progress.json cannot be written the watcher still writes its marker; the error is logged.
    assert env.trsi("launch", env.run, "--start", "--seed", 0, "--gpus", 0)[0] == 0   # a trial that has ended

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(progress, "build", boom)
    marker = env.out / "trsi" / "watch.marker"
    code, out, err = env.trsi("watch", env.run, "--marker", marker, "--foreground", "--poll", 0.01)
    assert code == 0 and marker.read_text().splitlines()[0] == "ok"
    assert "boom" in out + err

    class FullDisk:                                 # the disk is full: reporting the error fails too
        def write(self, s):
            raise OSError(errno.ENOSPC, "No space left on device")

        flush = write

    with monkeypatch.context() as mp:
        mp.setattr(sys, "stderr", FullDisk())
        marker = env.out / "trsi" / "watch-2.marker"
        code = env.trsi("watch", env.run, "--marker", marker, "--foreground", "--poll", 0.01)[0]
    assert code == 0 and marker.read_text().splitlines()[0] == "ok"
