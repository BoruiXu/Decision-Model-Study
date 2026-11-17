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
