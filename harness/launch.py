"""Start a run in a detached tmux session and return at once."""
from __future__ import annotations

import getpass
import os
import re
import shlex
import subprocess
from pathlib import Path

import yaml

from harness.config import ConfigError


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def _tmux(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def launch(lock_path, *, tmux=_tmux) -> dict:
    # absolute: the runner starts in the repo, not in the caller's cwd
    lock_path = Path(os.path.abspath(os.path.expanduser(str(lock_path))))
    if not lock_path.is_file():
        raise FileNotFoundError(f"lock not found: {lock_path}")
    lock = yaml.safe_load(lock_path.read_text())
    run_dir = Path(lock["run_dir"])
    done = run_dir / "DONE"
    if done.exists():
        raise RuntimeError(f"run already finished ({done} exists); resolve a new run instead")
    env = ""
    if lock["machine"]["gpu"]["source"] == "preassigned":
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not devices:
            raise ConfigError("gpu_source is 'preassigned' but CUDA_VISIBLE_DEVICES is not set in this shell; "
                              "set it to the assigned cards and launch again")
        # explicit: a session started by an already running tmux server inherits that server's environment
        env = f"CUDA_VISIBLE_DEVICES={shlex.quote(devices)} "
    # the whole run id (<UTC time>-<hash>): its first 8 characters are only the date
    session = _sanitize(f"ojh-{getpass.getuser()}-{lock['experiment']}-{lock['run_id']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "runner.log"
    repo = str(lock["repo_root"])
    inner = (f"{env}PYTHONPATH={shlex.quote(repo)} {shlex.quote(lock['machine']['python'])} -m harness.runner "
             f"{shlex.quote(str(lock_path))} >> {shlex.quote(str(log))} 2>&1")
    tmux(["tmux", "new-session", "-d", "-s", session, "-c", repo, inner])
    return {"session": session, "log": str(log), "done": str(done), "run_dir": str(run_dir)}
