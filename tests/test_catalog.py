import hashlib
import re
import sys

import pytest
import yaml

from harness import catalog
from harness.catalog import CatalogError, load_base, verify_files
from harness.fetch_base import fetch_base

REQUIRED = {"name", "hf_repo", "revision", "params_b", "multimodal", "attention",
            "readout", "sha256", "published_recipe"}
NAMES = ["joey-1", "elephant-1", "mammoth-1", "onejev-qwen3.5-0.8b", "onejev-qwen3.5-4b",
         "onejev-qwen3.5-9b", "onejev-27b"]


def test_catalog_entries_valid():
    files = sorted(catalog.CATALOG_DIR.glob("*.yaml"))
    assert [f.stem for f in files] == sorted(NAMES)
    for f in files:
        d = yaml.safe_load(f.read_text())
        assert REQUIRED <= set(d), f.name
        assert d["name"] == f.stem
        assert re.fullmatch(r"[0-9a-f]{40}", d["revision"]), f.name
        assert isinstance(d["params_b"], (int, float))
        assert isinstance(d["multimodal"], bool)
        assert d["attention"] in ("causal", "noncausal")
        assert d["readout"] in ("slot_lm", "head")
        assert d["sha256"] and all(re.fullmatch(r"[0-9a-f]{64}", s) for s in d["sha256"].values())
        assert isinstance(d["published_recipe"], str)


def test_load_base_unknown():
    assert load_base("joey-1")["params_b"] == 2
    with pytest.raises(CatalogError):
        load_base("nope")


def test_verify_files_detects_corruption(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"abc")
    good = hashlib.sha256(b"abc").hexdigest()
    verify_files(tmp_path, {"f.bin": good})
    with pytest.raises(CatalogError):
        verify_files(tmp_path, {"f.bin": "0" * 64})


def test_fetch_base_downloads_then_verifies(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, revision, local_dir, allow_patterns):
        calls.update(repo_id=repo_id, revision=revision, allow=allow_patterns)
        for name in allow_patterns:
            (local_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (local_dir / name).write_bytes(b"wrong")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        type(sys)("huggingface_hub"))
    sys.modules["huggingface_hub"].snapshot_download = (
        lambda **kw: fake_download(kw["repo_id"], kw["revision"], kw["local_dir"], kw["allow_patterns"]))
    spec = load_base("joey-1")
    with pytest.raises(CatalogError):
        fetch_base("joey-1", tmp_path)
    assert calls["repo_id"] == spec["hf_repo"] and calls["revision"] == spec["revision"]
    assert calls["allow"] == list(spec["sha256"])


def test_default_recipes_point_at_existing_recipes():
    from harness.catalog import ROOT, parse_gpu_range
    for f in catalog.CATALOG_DIR.glob("*.yaml"):
        d = yaml.safe_load(f.read_text())
        for key, path in (d.get("default_recipes") or {}).items():
            lo, hi = parse_gpu_range(key)
            assert 1 <= lo <= (hi or lo), (f.name, key)
            assert (ROOT / path).is_file(), (f.name, path)


@pytest.mark.parametrize("base,n,recipe", [("joey-1", 1, "joey_full_1gpu"), ("elephant-1", 4, "elephant_full_fsdp"),
                                           ("elephant-1", 2, "elephant_full_fsdp"),
                                           ("mammoth-1", 8, "mammoth_lora_ddp")])
def test_default_recipe_lookup(base, n, recipe):
    from harness.catalog import ROOT, default_recipe
    assert default_recipe(base, n) == str(ROOT / "examples" / "recipes" / f"{recipe}.yaml")


def test_no_default_recipe_lists_the_options():
    from harness.catalog import default_recipe
    with pytest.raises(CatalogError) as e:
        default_recipe("elephant-1", 1)
    msg = str(e.value)
    assert "elephant-1" in msg and "1 GPU" in msg and "2-8: examples/recipes/elephant_full_fsdp.yaml" in msg
    assert "joey_lora_1gpu.yaml" in msg and "--recipe" in msg
    with pytest.raises(CatalogError, match="no default_recipes"):
        default_recipe("onejev-qwen3.5-4b", 1)


def test_parse_gpu_range():
    from harness.catalog import parse_gpu_range
    assert parse_gpu_range(1) == (1, 1) and parse_gpu_range("2-8") == (2, 8) and parse_gpu_range("2+") == (2, None)
    with pytest.raises(CatalogError):
        parse_gpu_range("x")
