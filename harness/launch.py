"""Start a run in a detached tmux session and return at once."""
from __future__ import annotations

import getpass
import re
import shlex
import subprocess
from pathlib import Path

import yaml


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def _tmux(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def launch(lock_path, *, tmux=_tmux) -> dict:
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"lock not found: {lock_path}")
    lock = yaml.safe_load(lock_path.read_text())
    run_dir = Path(lock["run_dir"])
    done = run_dir / "DONE"
    if done.exists():
        raise RuntimeError(f"run already finished ({done} exists); resolve a new run instead")
    session = _sanitize(f"ojh-{getpass.getuser()}-{lock['experiment']}-{str(lock['run_id'])[:8]}")
    log = run_dir / "runner.log"
    inner = (f"{shlex.quote(lock['machine']['python'])} -m harness.runner "
             f"{shlex.quote(str(lock_path))} >> {shlex.quote(str(log))} 2>&1")
    tmux(["tmux", "new-session", "-d", "-s", session, inner])
    return {"session": session, "log": str(log), "done": str(done), "run_dir": str(run_dir)}
