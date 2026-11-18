"""Execute a run lock: take GPUs, train each seed, finalize, always leave a DONE marker."""
from __future__ import annotations

import json
import math
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
from harness.resolve import build_command, check_data


def lock_allows_sharing(lock: dict) -> bool:
    """Whether the run's explicit cards may carry other processes: the lock's ``machine.gpu.allow_shared``
    (ojh run --allow-shared). Locks written before that key existed keep their behaviour of the time, when
    explicit cards were checked for memory only: sharing allowed."""
    return bool(lock["machine"]["gpu"].get("allow_shared", True))


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


def _ignore_signals() -> None:
    """Cleanup (stop and reap the child, release the cards, write DONE) must finish once it has started, so
    later SIGTERM/SIGHUP are ignored; ``run`` restores the previous handlers when it returns."""
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)


def _one_line(e: BaseException) -> str:
    return " ".join(f"{type(e).__name__}: {e}".split())


def _run(lock_path, acquire, popen, finalize, killpg) -> int:
    lock_path = Path(lock_path)
    run_dir = lock_path.parent  # where DONE goes until the lock names the run dir (resolve writes it there)
    stage, code, lease, error = "verify", 1, None, None
    try:
        lock = yaml.safe_load(lock_path.read_text())
        if not isinstance(lock, dict):
            raise ValueError(f"{lock_path} is not a run lock (expected a mapping)")
        run_dir = Path(lock["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "status").write_text("running\n")
        check_data(lock)  # before taking cards: a re-run of prepare must not train on data the lock never saw
        if acquire is None:
            from harness.gpu import acquire
        stage = "acquire"
        g = lock["machine"]["gpu"]
        kw = ({"explicit": list(g["explicit"]), "need_mib": math.ceil(float(lock["estimate_gib"]) * 1024),
               "allow_shared": lock_allows_sharing(lock)}
              if g["source"] == "explicit" else {})
        lease = acquire(_machine(lock["machine"]), lock["n_gpus"], **kw)
        _write_json(run_dir / "gpus.json", [{"uuid": u, "name": n} for u, n in zip(lease.uuids, lease.names)])
        repo = lock["repo_root"]
        set_env = {**lock["machine"]["env"],
                   "CUDA_VISIBLE_DEVICES": ",".join(lease.uuids),
                   "PYTHONPATH": f"{repo}/vendor/OneJev:{repo}", "HF_HUB_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "4"}
        env = {**os.environ, **set_env}
        stage = "train"
        timing = {"n_gpus": lock["n_gpus"], "seeds": {}}
        ok = True
        for seed in lock["seeds"]:
            cmd = build_command(lock, seed)
            # the executed command (design section 11): argv, cwd and the variables the runner sets
            _write_json(run_dir / f"command-s{seed}.json", {"seed": seed, "argv": cmd, "cwd": repo, "env": set_env})
            start, t0 = _now(), time.monotonic()
            with open(run_dir / f"train-s{seed}.log", "ab") as log:
                proc = None
                try:  # entered before Popen returns its handle, so an interrupt right after the start is handled
                    proc = popen(cmd, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
                    rc = proc.wait()
                except BaseException:
                    _ignore_signals()
                    if proc is not None:
                        _terminate_group(proc, killpg)
                    raise
            timing["seeds"][str(seed)] = {"start": start, "end": _now(),
                                          "seconds": round(time.monotonic() - t0, 1), "exit_code": rc}
            _write_json(run_dir / "timing.json", timing)
            if rc:
                code, ok = rc, False
                error = f"seed {seed}: training exited with code {rc} (see train-s{seed}.log)"
                break
        if ok:
            stage = "finalize"
            finalize(run_dir)
            code = 0
    except BaseException as e:  # code stays non-zero; DONE is written below
        error = _one_line(e)
        print(f"runner error in stage {stage}: {e!r}", file=sys.stderr)
    finally:
        _ignore_signals()
        if lease is not None:
            lease.release()
        run_dir.mkdir(parents=True, exist_ok=True)
        if code != 0:
            (run_dir / "status").write_text("failed\n")
        done = {"exit_code": code, "end": _now(), "stage": stage}
        if code != 0:
            done["error"] = error or f"exit code {code}"
        _write_json(run_dir / "DONE", done)
    return code


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
