"""List and delete the checkpoint dirs that finalize marked as deletion candidates. Imports without torch."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def candidates(run_dir) -> list[dict]:
    """The ``{path, bytes, ...}`` entries of ``deletion_candidates.json`` (empty when finalize has not run)."""
    p = Path(run_dir) / "deletion_candidates.json"
    return list(json.loads(p.read_text()).get("candidates", [])) if p.is_file() else []


def _refusal(run_dir: Path, path) -> str | None:
    """Why ``path`` may not be deleted, or None. Only paths strictly below ``<run_dir>/runs`` qualify."""
    runs = Path(os.path.abspath(run_dir / "runs"))
    lexical = Path(os.path.abspath(path))
    if lexical == runs or runs not in lexical.parents:
        return "not strictly below the run's runs/ directory"
    # the entry itself may be a symlink (it is unlinked, never followed), but its parents must not escape
    if runs.resolve() not in lexical.parent.resolve().parents and lexical.parent.resolve() != runs.resolve():
        return "parent directory resolves outside runs/"
    if not os.path.lexists(lexical):
        return "already gone"
    return None


def delete_report(run_dir, confirm: bool) -> tuple[list[dict], list[dict]]:
    """``(deleted, skipped)``; skipped entries carry a ``reason``. Without ``confirm`` nothing is touched."""
    run_dir = Path(run_dir)
    if not confirm:
        return [], []
    done, skipped = [], []
    for c in candidates(run_dir):
        why = _refusal(run_dir, c["path"])
        if why:
            skipped.append({**c, "reason": why})
            continue
        target = Path(os.path.abspath(c["path"]))
        if target.is_symlink() or not target.is_dir():
            target.unlink()
        else:
            shutil.rmtree(target)
        done.append(c)
    return done, skipped


def delete(run_dir, confirm: bool) -> list[dict]:
    """Delete the candidates below ``<run_dir>/runs``; returns the entries that were removed."""
    return delete_report(run_dir, confirm)[0]
