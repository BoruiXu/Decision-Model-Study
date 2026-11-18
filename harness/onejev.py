"""Clone, patch and verify the pinned OneJev checkout. Imports without torch."""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from harness.catalog import ROOT, sha256_file

LOCK_PATH = ROOT / "onejev.lock"


class OneJevMismatch(RuntimeError):
    """The OneJev checkout differs from the pinned revision or patched file hashes."""


def load_lock(path: Path = LOCK_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _git(dest: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(dest), *args], check=check, capture_output=True, text=True)


def ensure_onejev(dest: Path, lock: dict, patch_path: Path) -> Path:
    """Return ``dest`` holding OneJev at the locked revision with the patch applied exactly once."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", lock["repo"], str(dest)], check=True)
        _git(dest, "checkout", "-q", "--detach", lock["revision"])
    head = _git(dest, "rev-parse", "HEAD").stdout.strip()
    if head != lock["revision"]:
        raise OneJevMismatch(f"OneJev at {dest} is {head}, expected {lock['revision']}")
    patch = str(Path(patch_path).resolve())
    if _git(dest, "apply", "--reverse", "--check", patch, check=False).returncode:
        _git(dest, "apply", patch)
    for rel, expected in lock["patched_sha256"].items():
        target = dest / rel
        if not target.is_file() or sha256_file(target) != expected:
            raise OneJevMismatch(f"Unexpected SHA-256 for {target}")
    return dest


def verify_onejev(dest: Path, lock: dict) -> list[str]:
    """Read-only check of a checkout (``ojh doctor``): HEAD at the locked revision and every patched file
    at its locked sha256. Returns the problems; nothing is cloned, checked out or patched."""
    dest = Path(dest)
    if not dest.is_dir():
        return [f"{dest} is missing: run setup.sh (or ojh setup)"]
    problems = []
    got = _git(dest, "rev-parse", "HEAD", check=False)
    head = got.stdout.strip() if got.returncode == 0 else "unknown"
    if head != lock["revision"]:
        problems.append(f"HEAD {head[:12]} but expected {lock['revision'][:12]}")
    for rel, expected in lock["patched_sha256"].items():
        target = dest / rel
        if not target.is_file():
            problems.append(f"{rel}: missing (sha256 expected {expected[:12]}…)")
        elif sha256_file(target) != expected:
            problems.append(f"{rel}: sha256 differs from the lock (expected {expected[:12]}…)")
    return problems
