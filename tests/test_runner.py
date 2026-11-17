import getpass
import json
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


def make_lock(tmp_path, seeds=(42, 7)):
    run_dir = tmp_path / "exp" / "plan"
    run_dir.mkdir(parents=True)
    lock = {"experiment": "exp", "repo_root": str(tmp_path / "repo"), "run_id": "abcdef1234567890",
            "run_dir": str(run_dir), "n_gpus": 2, "seeds": list(seeds),
            "flags": {"epochs": 1}, "plan": {"steps": 10, "ranks": 2, "interval": 5},
            "base": {"name": "b"}, "data": {"train": "t", "select": "s"},
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
    assert session.startswith("ojh-") and session.endswith("abcdef12")
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
    assert done["exit_code"] == 0 and done["stage"] == "finalize"
    assert (run_dir / "status").read_text().strip() == "running"  # finalize owns the final status


def test_runner_acquire_failure_and_finalize_failure(tmp_path):
    path, _, run_dir = make_lock(tmp_path)

    def boom(m, n):
        raise TimeoutError("no gpus")

    assert run(path, acquire=boom, popen=FakePopen, finalize=lambda d: None) != 0
    done = json.loads((run_dir / "DONE").read_text())
    assert done["stage"] == "acquire" and done["exit_code"] != 0
    assert (run_dir / "status").read_text().strip() == "failed"

    (run_dir / "DONE").unlink()
    FakePopen.codes = [0, 0]
    lease = FakeLease()

    def bad_fin(d):
        raise RuntimeError("x")

    assert run(path, acquire=lambda m, n: lease, popen=FakePopen, finalize=bad_fin) != 0
    done = json.loads((run_dir / "DONE").read_text())
    assert done["stage"] == "finalize" and done["exit_code"] != 0 and lease.released


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
