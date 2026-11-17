"""Execute a run lock: take GPUs, train each seed, finalize, always leave a DONE marker."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from harness.config import MachineConfig
from harness.resolve import build_command


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _machine(snap: dict) -> MachineConfig:
    g = snap["gpu"]
    return MachineConfig(
        python=snap["python"], model_dir=snap["model_dir"], out_root=snap["out_root"],
        registry=snap.get("registry"), gpu_include=g["include"], gpu_exclude=g["exclude"],
        min_free_mib=g["min_free_mib"], lock_dir=g["lock_dir"], order=g["order"],
        wait_timeout_s=g["wait_timeout_s"], gpu_source=g["source"],
        forbidden_write_roots=list(snap["forbidden_write_roots"]), env=dict(snap["env"]))


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _default_finalize(run_dir):
    from harness.finalize import finalize
    return finalize(run_dir)


class Interrupted(KeyboardInterrupt):
    """Raised by the SIGTERM/SIGHUP handlers so the cleanup in run() still executes."""


def _raise_interrupted(signum, frame):
    raise Interrupted(f"signal {signum}")


def _terminate_group(proc, killpg, timeout=30) -> None:
    """SIGTERM the child's process group, wait, escalate to SIGKILL, and reap it."""
    try:
        killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.wait()


def run(lock_path, *, acquire=None, popen=subprocess.Popen, finalize=_default_finalize,
        killpg=os.killpg) -> int:
    old = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            old[sig] = signal.signal(sig, _raise_interrupted)
    try:
        return _run(lock_path, acquire, popen, finalize, killpg)
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def _run(lock_path, acquire, popen, finalize, killpg) -> int:
    lock = yaml.safe_load(Path(lock_path).read_text())
    run_dir = Path(lock["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status").write_text("running\n")
    if acquire is None:
        from harness.gpu import acquire
    stage, code, lease = "acquire", 1, None
    try:
        lease = acquire(_machine(lock["machine"]), lock["n_gpus"])
        _write_json(run_dir / "gpus.json", [{"uuid": u, "name": n} for u, n in zip(lease.uuids, lease.names)])
        repo = lock["repo_root"]
        env = {**os.environ, **lock["machine"]["env"],
               "CUDA_VISIBLE_DEVICES": ",".join(lease.uuids),
               "PYTHONPATH": f"{repo}/vendor/OneJev:{repo}", "HF_HUB_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "4"}
        stage = "train"
        timing = {"n_gpus": lock["n_gpus"], "seeds": {}}
        ok = True
        for seed in lock["seeds"]:
            start, t0 = _now(), time.monotonic()
            with open(run_dir / f"train-s{seed}.log", "ab") as log:
                proc = popen(build_command(lock, seed), env=env, cwd=repo, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = proc.wait()
                except BaseException:
                    _terminate_group(proc, killpg)
                    raise
            timing["seeds"][str(seed)] = {"start": start, "end": _now(),
                                          "seconds": round(time.monotonic() - t0, 1), "exit_code": rc}
            _write_json(run_dir / "timing.json", timing)
            if rc:
                code, ok = rc, False
                break
        if ok:
            stage = "finalize"
            finalize(run_dir)
            code = 0
    except BaseException as e:  # code stays non-zero; DONE is written below
        print(f"runner error in stage {stage}: {e!r}", file=sys.stderr)
    finally:
        if lease is not None:
            lease.release()
        if code != 0:
            (run_dir / "status").write_text("failed\n")
        _write_json(run_dir / "DONE", {"exit_code": code, "end": _now(), "stage": stage})
    return code


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
