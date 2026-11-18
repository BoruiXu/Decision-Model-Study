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


def parse_gpu_range(key) -> tuple[int, int | None]:
    """A ``default_recipes`` key: ``N``, ``A-B`` or ``A+`` (open-ended) -> (low, high or None)."""
    text = str(key).strip()
    try:
        if text.endswith("+"):
            return int(text[:-1]), None
        if "-" in text:
            lo, hi = text.split("-", 1)
            return int(lo), int(hi)
        return int(text), int(text)
    except ValueError:
        raise CatalogError(f"bad default_recipes key {key!r}: use N, A-B or A+") from None


def recipe_options(base: dict) -> list[str]:
    """Human-readable choices when no default fits: the base's defaults and every example recipe."""
    defaults = [f"{k}: {v}" for k, v in (base.get("default_recipes") or {}).items()]
    examples = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "examples" / "recipes").glob("*.yaml"))
    return ([f"{base['name']} default_recipes: " + "; ".join(defaults)] if defaults else []) + \
        ["example recipes: " + ", ".join(examples), "or pass --recipe <path> (your own recipe file)"]


def default_recipe(name: str, n_gpus: int) -> str:
    """The absolute recipe path that ``catalog/bases/<name>.yaml`` names for ``n_gpus`` cards.

    Raises CatalogError listing the options when the base has no default for that count."""
    base = load_base(name)
    table = base.get("default_recipes") or {}
    for key, path in table.items():
        lo, hi = parse_gpu_range(key)
        if lo <= n_gpus and (hi is None or n_gpus <= hi):
            return str(ROOT / path)
    head = (f"base {name} has no default recipe for {n_gpus} GPU(s)" if table
            else f"base {name} has no default_recipes")
    raise CatalogError(head + "; options:\n- " + "\n- ".join(recipe_options(base)))
