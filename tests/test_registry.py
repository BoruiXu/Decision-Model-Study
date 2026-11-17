import json
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
    registry.append(path, {"run_id": tag, "i": i, "pad": "x" * 100_000})
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
