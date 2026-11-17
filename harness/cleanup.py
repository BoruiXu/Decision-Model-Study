"""List and delete the checkpoint dirs that finalize marked as deletion candidates. Imports without torch."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def candidates(run_dir) -> list[dict]:
    """The ``{path, bytes, ...}`` entries of ``deletion_candidates.json`` (empty when finalize has not run)."""
    p = Path(run_dir) / "deletion_candidates.json"
    return list(json.loads(p.read_text()).get("candidates", [])) if p.is_file() else []


def _inside(run_dir: Path, path) -> Path | None:
    """The resolved path when it lies strictly inside ``run_dir``, else None."""
    target = Path(path).resolve()
    root = run_dir.resolve()
    return target if target != root and root in target.parents else None


def delete(run_dir, confirm: bool) -> list[dict]:
    """Delete the candidates that resolve inside ``run_dir``; without ``confirm`` delete nothing.

    Returns the entries that were removed.
    """
    run_dir = Path(run_dir)
    if not confirm:
        return []
    done = []
    for c in candidates(run_dir):
        target = _inside(run_dir, c["path"])
        if target is None or not target.exists():
            continue
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
        done.append(c)
    return done
