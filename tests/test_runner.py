import getpass
import hashlib
import json
import signal
from pathlib import Path

import pytest
import yaml

from harness.launch import launch
from harness.runner import run


class FakeLease:
    uuids = ["GPU-a", "GPU-b"]
    names = ["A100", "A100"]

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


DATA_FILES = ("train.jsonl", "select.jsonl", "select_index.json")


def make_lock(tmp_path, seeds=(42, 7)):
    run_dir = tmp_path / "exp" / "plan"
    run_dir.mkdir(parents=True)
    prep = tmp_path / "exp" / "prepared"
    prep.mkdir()
    for f in DATA_FILES:
        (prep / f).write_text(f"{f}\n")
    data = {"train": str(prep / "train.jsonl"), "select": str(prep / "select.jsonl"), "prepared_dir": str(prep),
            "sha256": {f: hashlib.sha256((prep / f).read_bytes()).hexdigest() for f in DATA_FILES}}
    lock = {"experiment": "exp", "repo_root": str(tmp_path / "repo"), "run_id": "20261117T101500Z-abcdef12",
            "run_dir": str(run_dir), "n_gpus": 2, "seeds": list(seeds),
            "flags": {"epochs": 1}, "plan": {"steps": 10, "ranks": 2, "interval": 5},
            "base": {"name": "b"}, "data": data,
            "machine": {"python": "/venv/bin/python", "out_root": str(tmp_path), "model_dir": "/m",
                        "registry": None, "env": {"FOO": "bar"},
                        "gpu": {"include": None, "exclude": [], "min_free_mib": 0, "lock_dir": None,
                                "order": "free_desc", "wait_timeout_s": 5, "source": "scan"},
                        "forbidden_write_roots": []}}
    path = run_dir / "run.lock.yaml"
    path.write_text(yaml.safe_dump(lock))
    return path, lock, run_dir


class FakePopen:
    calls = []

    def __init__(self, cmd, **kw):
        FakePopen.calls.append((cmd, kw))
        self.rc = FakePopen.codes.pop(0)

    pid = 4242

    def wait(self, timeout=None):
        return self.rc


@pytest.fixture(autouse=True)
def _reset():
    FakePopen.calls, FakePopen.codes = [], []


def test_launch_returns_immediately(tmp_path):
    path, lock, run_dir = make_lock(tmp_path)
    seen = []
    out = launch(path, tmux=lambda argv: seen.append(argv))
    assert len(seen) == 1
    argv = seen[0]
    assert argv[:4] == ["tmux", "new-session", "-d", "-s"]
    session = argv[4]
    assert argv[5:7] == ["-c", lock["repo_root"]]
    assert f"PYTHONPATH={lock['repo_root']} /venv/bin/python" in argv[7]
    assert run_dir.is_dir()
    assert out["session"] == session and getpass.getuser().replace(".", "_") in session
    assert session.startswith("ojh-") and session.endswith("20261117T101500Z-abcdef12")
    assert "CUDA_VISIBLE_DEVICES" not in argv[7]  # scan: the runner picks and locks the cards
    assert all(c.isalnum() or c in "_-" for c in session)
    assert "/venv/bin/python -m harness.runner" in argv[7] and "runner.log" in argv[7]
    assert out["done"] == str(run_dir / "DONE") and out["run_dir"] == str(run_dir)


def test_launch_refuses_missing_and_finished(tmp_path):
    with pytest.raises(FileNotFoundError):
        launch(tmp_path / "nope.yaml", tmux=lambda a: None)
    path, _, run_dir = make_lock(tmp_path)
    (run_dir / "DONE").write_text("{}")
    with pytest.raises(RuntimeError):
        launch(path, tmux=lambda a: None)


def test_runner_writes_done_on_child_failure(tmp_path):
    path, _, run_dir = make_lock(tmp_path)
    lease = FakeLease()
    FakePopen.codes = [3]
    fin = []
    rc = run(path, acquire=lambda m, n: lease, popen=FakePopen, finalize=fin.append)
    assert rc == 3 and fin == [] and lease.released
    done = json.loads((run_dir / "DONE").read_text())
    assert done["exit_code"] == 3 and done["stage"] == "train" and done["end"].endswith("Z")
    assert "seed 42" in done["error"] and "3" in done["error"] and "train-s42.log" in done["error"]
    assert "\n" not in done["error"]
    assert (run_dir / "status").read_text().strip() == "failed"
    assert len(FakePopen.calls) == 1  # stops after the first failing seed


def test_runner_runs_seeds_in_order(tmp_path):
    path, lock, run_dir = make_lock(tmp_path)
    FakePopen.codes = [0, 0]
    fin = []
    rc = run(path, acquire=lambda m, n: FakeLease(), popen=FakePopen, finalize=fin.append)
    assert rc == 0 and fin == [run_dir]
    cmds = [c for c, _ in FakePopen.calls]
    assert [c[c.index("--seed") + 1] for c in cmds] == ["42", "7"]
    env = FakePopen.calls[0][1]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b" and env["FOO"] == "bar"
    assert env["PYTHONPATH"] == f"{lock['repo_root']}/vendor/OneJev:{lock['repo_root']}"
    assert env["HF_HUB_OFFLINE"] == "1" and env["OMP_NUM_THREADS"] == "4"
    assert env["WANDB_MODE"] == "disabled" and env["TOKENIZERS_PARALLELISM"] == "false"
    timing = json.loads((run_dir / "timing.json").read_text())
    assert timing["n_gpus"] == 2 and list(timing["seeds"]) == ["42", "7"]
    assert timing["seeds"]["42"]["exit_code"] == 0 and "seconds" in timing["seeds"]["42"]
    assert json.loads((run_dir / "gpus.json").read_text()) == [
        {"uuid": "GPU-a", "name": "A100"}, {"uuid": "GPU-b", "name": "A100"}]
    done = json.loads((run_dir / "DONE").read_text())
    assert done["exit_code"] == 0 and done["stage"] == "finalize" and "error" not in done
    assert (run_dir / "status").read_text().strip() == "running"  # finalize owns the final status


def test_runner_acquire_failure_and_finalize_failure(tmp_path):
    path, _, run_dir = make_lock(tmp_path)

    def boom(m, n):
        raise TimeoutError("no gpus")

    assert run(path, acquire=boom, popen=FakePopen, finalize=lambda d: None) != 0
    done = json.loads((run_dir / "DONE").read_text())
    assert done["stage"] == "acquire" and done["exit_code"] != 0
    assert done["error"] == "TimeoutError: no gpus"
    assert (run_dir / "status").read_text().strip() == "failed"

    (run_dir / "DONE").unlink()
    FakePopen.codes = [0, 0]
    lease = FakeLease()

    def bad_fin(d):
        raise RuntimeError("x\nsecond line")

    assert run(path, acquire=lambda m, n: lease, popen=FakePopen, finalize=bad_fin) != 0
    done = json.loads((run_dir / "DONE").read_text())
    assert done["stage"] == "finalize" and done["exit_code"] != 0 and lease.released
    assert done["error"] == "RuntimeError: x second line"  # one line


def test_interrupt_terminates_child_before_release_and_writes_done(tmp_path):
    path, _, run_dir = make_lock(tmp_path)
    events = []

    class Lease(FakeLease):
        def release(self):
            events.append("release")
            super().release()

    class Hung:
        pid = 777

        def wait(self, timeout=None):
            if timeout is None and "term" not in events:
                raise KeyboardInterrupt
            events.append("reaped")
            return -15

    def killpg(pid, sig):
        events.append(("killpg", pid, sig))
        events.append("term")

    lease = Lease()
    rc = run(path, acquire=lambda m, n: lease, popen=lambda c, **kw: Hung(),
             finalize=lambda d: events.append("finalize"), killpg=killpg)
    import signal
    assert rc != 0 and "finalize" not in events
    assert events == [("killpg", 777, signal.SIGTERM), "term", "reaped", "release"]
    done = json.loads((run_dir / "DONE").read_text())
    assert done["exit_code"] != 0 and done["stage"] == "train"
    assert (run_dir / "status").read_text().strip() == "failed"


def test_child_started_in_own_session(tmp_path):
    path, _, _ = make_lock(tmp_path)
    FakePopen.codes = [0, 0]
    run(path, acquire=lambda m, n: FakeLease(), popen=FakePopen, finalize=lambda d: None)
    assert FakePopen.calls[0][1]["start_new_session"] is True


def test_launch_relative_lock_path_made_absolute(tmp_path, monkeypatch):
    path, lock, run_dir = make_lock(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen = []
    launch(path.relative_to(tmp_path), tmux=lambda argv: seen.append(argv))
    assert f"-m harness.runner {path} " in seen[0][7]  # the runner's cwd is the repo, not the caller's cwd


@pytest.mark.parametrize("changed", DATA_FILES)
def test_runner_refuses_changed_prepared_data(tmp_path, capsys, changed):
    path, lock, run_dir = make_lock(tmp_path)
    (Path(lock["data"]["prepared_dir"]) / changed).write_text("re-prepared\n")  # prepare re-ran in place
    acquired = []
    rc = run(path, acquire=lambda m, n: acquired.append(n) or FakeLease(), popen=FakePopen, finalize=lambda d: None)
    assert rc == 1 and acquired == [] and FakePopen.calls == []
    done = json.loads((run_dir / "DONE").read_text())
    assert done["exit_code"] == 1 and done["stage"] == "verify" and changed in done["error"]
    assert (run_dir / "status").read_text().strip() == "failed"
    err = capsys.readouterr().err  # runner.log in a launched run
    assert changed in err and "sha256" in err and "resolve" in err


def test_second_signal_during_cleanup_is_ignored(tmp_path):
    path, _, run_dir = make_lock(tmp_path)
    events = []

    class Lease(FakeLease):
        def release(self):
            signal.raise_signal(signal.SIGHUP)  # yet another signal while DONE is being written
            events.append("release")
            super().release()

    class Hung:
        pid = 777

        def wait(self, timeout=None):
            if timeout is None:
                signal.raise_signal(signal.SIGTERM)  # the first interrupt
            events.append("reaped")
            return -15

    def killpg(pid, sig):
        events.append(("killpg", sig))
        signal.raise_signal(signal.SIGTERM)  # a second SIGTERM while the child is being stopped

    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    lease = Lease()
    rc = run(path, acquire=lambda m, n: lease, popen=lambda c, **kw: Hung(), finalize=lambda d: None,
             killpg=killpg)
    assert rc != 0 and lease.released
    assert events == [("killpg", signal.SIGTERM), "reaped", "release"]
    done = json.loads((run_dir / "DONE").read_text())
    assert done["stage"] == "train" and done["exit_code"] != 0 and "Interrupted" in done["error"]
    assert {s: signal.getsignal(s) for s in before} == before  # handlers restored after the run


def test_interrupt_inside_popen_still_writes_done(tmp_path):
    path, _, run_dir = make_lock(tmp_path)
    lease = FakeLease()

    def popen(cmd, **kw):
        signal.raise_signal(signal.SIGTERM)
        raise AssertionError("unreachable")

    assert run(path, acquire=lambda m, n: lease, popen=popen, finalize=lambda d: None) != 0
    done = json.loads((run_dir / "DONE").read_text())
    assert done["stage"] == "train" and "Interrupted" in done["error"] and lease.released


def test_command_written_before_each_seed_starts(tmp_path):
    path, lock, run_dir = make_lock(tmp_path)
    seen = []

    class Recording(FakePopen):
        def __init__(self, cmd, **kw):
            seed = cmd[cmd.index("--seed") + 1]
            seen.append(json.loads((run_dir / f"command-s{seed}.json").read_text()))
            super().__init__(cmd, **kw)

    FakePopen.codes = [0, 0]
    assert run(path, acquire=lambda m, n: FakeLease(), popen=Recording, finalize=lambda d: None) == 0
    assert [s["seed"] for s in seen] == [42, 7]
    for s, (cmd, kw) in zip(seen, FakePopen.calls):
        assert s["argv"] == cmd and s["cwd"] == kw["cwd"] == lock["repo_root"]
        assert s["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b" and s["env"]["FOO"] == "bar"
        assert s["env"]["PYTHONPATH"] == kw["env"]["PYTHONPATH"]


@pytest.mark.parametrize("text", ["{{{ not yaml", "{}\n", "- a list\n"])
def test_unreadable_lock_writes_done(tmp_path, text, capsys):
    run_dir = tmp_path / "exp" / "20261117T000000Z-0123abcd"
    run_dir.mkdir(parents=True)
    path = run_dir / "run.lock.yaml"
    path.write_text(text)
    acquired = []
    assert run(path, acquire=lambda m, n: acquired.append(n), popen=FakePopen, finalize=lambda d: None) == 1
    done = json.loads((run_dir / "DONE").read_text())
    assert done["exit_code"] == 1 and done["stage"] == "verify" and done["error"]
    assert (run_dir / "status").read_text().strip() == "failed" and acquired == []
    assert "runner error in stage verify" in capsys.readouterr().err


def test_two_runs_of_one_experiment_on_one_day_get_distinct_sessions(tmp_path):
    sessions = []
    for i, run_id in enumerate(("20261117T101500Z-abcdef12", "20261117T113000Z-0f1e2d3c")):
        path, lock, run_dir = make_lock(tmp_path / str(i))
        lock["run_id"] = run_id
        path.write_text(yaml.safe_dump(lock))
        sessions.append(launch(path, tmux=lambda a: None)["session"])
    assert sessions[0] != sessions[1] and sessions[1].endswith("0f1e2d3c")


def _preassigned(tmp_path):
    path, lock, run_dir = make_lock(tmp_path)
    lock["machine"]["gpu"]["source"] = "preassigned"
    path.write_text(yaml.safe_dump(lock))
    return path, run_dir


def test_preassigned_devices_passed_to_the_session(tmp_path, monkeypatch):
    path, _ = _preassigned(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-x,GPU-y")
    seen = []
    launch(path, tmux=seen.append)
    # an existing tmux server would otherwise start the session with its own environment
    assert seen[0][7].startswith("CUDA_VISIBLE_DEVICES=GPU-x,GPU-y PYTHONPATH=")


@pytest.mark.parametrize("value", [None, "", "  "])
def test_preassigned_without_devices_refused(tmp_path, monkeypatch, value):
    from harness.config import ConfigError
    path, run_dir = _preassigned(tmp_path)
    if value is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    seen = []
    with pytest.raises(ConfigError, match="CUDA_VISIBLE_DEVICES"):
        launch(path, tmux=seen.append)
    assert seen == [] and not (run_dir / "runner.log").exists()
