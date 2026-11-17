import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from harness import registry

ROOT = Path(__file__).resolve().parent.parent


def record(run_id, seed=42, conclusion="beats tfidf on both sets (comparison.json)", **over):
    rec = {"exp": "exp", "run_id": run_id, "seed": seed, "train_label_version": "v1",
           "reference_label_version": "v1", "data_sha256": "d" * 64, "base": "joey-1",
           "base_revision": "r" * 40, "recipe": {"flags": {"lr": 1e-5}}, "mode": "full",
           "onejev": {"revision": "o" * 40}, "gpu_name": "A100", "n_gpus": 1, "gpu_hours": 0.5,
           "selected_step": 8, "checkpoint_sha256": "c" * 64,
           "metrics": {"auc": 0.71, "within_group_auc": 0.6}, "baselines": {"tfidf": {"auc": 0.6}},
           "gate": {"enabled": True, "passed": True}, "overrides": {}, "conclusion": conclusion}
    rec.update(over)
    return rec


def test_registry_incomplete_hidden(tmp_path):
    path = tmp_path / "registry.jsonl"
    registry.append(path, record("run-complete"))
    registry.append(path, record("run-open", conclusion=""))
    registry.append(path, record("run-missing", metrics=None))
    recs = registry.read(path)
    assert [r["complete"] for r in recs] == [True, False, False]
    assert set(registry.REQUIRED) <= set(recs[0])
    shown = registry.table(path)
    assert "run-complete" in shown and "run-open" not in shown and "run-missing" not in shown
    assert shown.lstrip().startswith("|")
    every = registry.table(path, include_incomplete=True)
    assert "run-open" in every and "run-missing" in every
    # a note fills the conclusion and recomputes completeness for every record of that run
    assert registry.note(path, "run-open", "worse than tfidf; see comparison.json") == 1
    assert registry.note(path, "nope", "x") == 0
    recs = registry.read(path)
    assert recs[1]["conclusion"].startswith("worse") and recs[1]["complete"] is True
    assert recs[2]["complete"] is False
    assert "run-open" in registry.table(path)


def test_registry_upsert_by_run_and_seed(tmp_path):
    path = tmp_path / "registry.jsonl"
    registry.append(path, record("r1", seed=42, conclusion=""))
    registry.append(path, record("r1", seed=7, conclusion=""))
    registry.append(path, record("r1", seed="summary", conclusion=""))
    assert registry.note(path, "r1", "noted by the agent") == 3
    # re-finalize: same (run_id, seed) keys, empty conclusions, new metrics
    registry.append(path, record("r1", seed=42, conclusion="", metrics={"auc": 0.8}))
    registry.append(path, record("r1", seed="summary", conclusion="", metrics={"auc": {"mean": 0.8, "std": 0.0}}))
    registry.append(path, record("r2", seed=42, conclusion=""))
    recs = registry.read(path)
    assert [(r["run_id"], r["seed"]) for r in recs] == [("r1", 42), ("r1", 7), ("r1", "summary"), ("r2", 42)]
    assert recs[0]["metrics"] == {"auc": 0.8} and recs[0]["conclusion"] == "noted by the agent"
    assert recs[0]["complete"] is True and recs[2]["conclusion"] == "noted by the agent"
    assert recs[2]["metrics"]["auc"]["mean"] == 0.8 and recs[3]["complete"] is False
    # a non-empty new conclusion wins
    registry.append(path, record("r1", seed=7, conclusion="rewritten"))
    assert registry.read(path)[1]["conclusion"] == "rewritten"
    assert len(path.read_text().splitlines()) == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_registry_summary_rendered(tmp_path):
    path = tmp_path / "registry.jsonl"
    registry.append(path, record("r", seed="summary", selected_step=[8, 12],
                                 metrics={"auc": {"mean": 0.7, "std": 0.01}, "selected_steps": [8, 12]}))
    out = registry.table(path)
    assert "summary" in out and "0.700" in out and "0.010" in out


WRITER = """
import sys
from harness import registry
path, tag = sys.argv[1], sys.argv[2]
for i in range(60):
    registry.append(path, {"run_id": tag, "seed": i, "i": i, "pad": "x" * 20_000})
"""


def test_registry_concurrent_append(tmp_path):
    path = tmp_path / "registry.jsonl"
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, str(path), tag], cwd=ROOT) for tag in ("a", "b")]
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    lines = path.read_text().splitlines()
    assert len(lines) == 120
    recs = [json.loads(line) for line in lines]
    for tag in ("a", "b"):
        assert sorted(r["i"] for r in recs if r["run_id"] == tag) == list(range(60))
    assert all(r["complete"] is False for r in recs)


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_registry_files_group_writable(tmp_path):
    old = os.umask(0o022)
    try:
        path = tmp_path / "registry.jsonl"
        lock = tmp_path / "registry.jsonl.lock"
        registry.append(path, record("r1", conclusion=""))
        assert mode(path) == 0o664 and mode(lock) == 0o664
        registry.note(path, "r1", "noted")  # atomic rewrite
        assert mode(path) == 0o664
        os.chmod(path, 0o644)  # a registry created before this fix
        registry.append(path, record("r1", conclusion=""))  # upsert rewrite
        assert mode(path) == 0o664
        os.chmod(lock, 0o444)  # a lock file this user cannot write, as for a second user
        registry.append(path, record("r2"))
        assert [r["run_id"] for r in registry.read(path)] == ["r1", "r2"] and "r2" in registry.table(path)
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        os.umask(old)
