"""Download a catalog base model and verify its hashes. Imports without torch."""
from __future__ import annotations

from pathlib import Path

from harness.catalog import load_base, verify_files


def fetch_base(name: str, model_dir: Path) -> Path:
    """Download base ``name`` into ``<model_dir>/<name>`` at its pinned revision and verify it."""
    from huggingface_hub import snapshot_download

    spec = load_base(name)
    target = Path(model_dir) / name
    snapshot_download(repo_id=spec["hf_repo"], revision=spec["revision"], local_dir=target,
                      allow_patterns=list(spec["sha256"]))
    verify_files(target, spec["sha256"])
    return target
