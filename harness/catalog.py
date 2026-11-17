"""Base-model catalog: one yaml per base under ``catalog/bases``. Imports without torch."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "catalog" / "bases"


class CatalogError(ValueError):
    """Unknown base, or downloaded weights that do not match the catalog hashes."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(directory: Path, hashes: dict[str, str]) -> None:
    for filename, expected in hashes.items():
        target = Path(directory) / filename
        if not target.is_file() or sha256_file(target) != expected:
            raise CatalogError(f"Unexpected SHA-256 for {target}")


def load_base(name: str) -> dict:
    path = CATALOG_DIR / f"{name}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in CATALOG_DIR.glob("*.yaml"))
        raise CatalogError(f"unknown base {name!r}; known: {known}")
    return yaml.safe_load(path.read_text())
