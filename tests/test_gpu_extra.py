import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Packages the reference training setup installs (torch/torchvision come from the cu128 index in setup.sh).
SOURCE_PACKAGES = {
    "torch", "torchvision", "transformers", "accelerate", "pillow", "jinja2", "pydantic",
    "fastapi", "uvicorn", "pyyaml", "safetensors", "flash-linear-attention", "peft",
}


def _names(specs):
    return {re.split(r"[<>=!~\[ ;]", s, maxsplit=1)[0].lower() for s in specs}


def test_gpu_extra_lists_every_source_package():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    listed = _names(meta["optional-dependencies"]["gpu"]) | _names(meta["dependencies"])
    assert SOURCE_PACKAGES <= listed, sorted(SOURCE_PACKAGES - listed)
