import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from harness.config import ConfigError, MachineConfig, load_experiment
from harness.prepare import adapter_file, prepare
from harness.resolve import resolve

sys.path.insert(0, str(Path(__file__).parent))
from test_resolve import fake_fp, plan_ok  # noqa: E402

ADAPTER = '''"""csv text+label -> OneJev rows; keeps rows whose split equals the `split` argument."""
import csv


def adapt(path, split=None):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            p = float(r["label"])
            yield {"id": r["id"], "split": "train" if r["split"] == "train" else r["split"],
                   "state": {"messages": [{"role": "user", "content": r["text"]}]},
                   "question": {"type": "noul", "instructions": "ok?", "criteria": {"true": "y", "false": "n"}},
                   "target": {"yes": p, "no": 1 - p}, "meta": {"grp": r["grp"]}}
'''


def workspace(tmp_path, adapter_ref="adapters_local/my_adapter.py"):
    ws = tmp_path / "ws"
    (ws / "adapters_local").mkdir(parents=True)
    (ws / "adapters_local" / "my_adapter.py").write_text(ADAPTER)
    with open(ws / "all.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label", "grp", "split"])
        for i in range(30):
            w.writerow([f"r{i}", f"text {i}", i % 2, f"g{i % 3}", "val" if i % 3 == 0 else "train"])
    exp = {"name": "csvexp",
           "data": {"adapter": adapter_ref, "train": "all.csv", "train_adapter_args": {"split": "train"},
                    "eval": {"val": {"train_labels": "all.csv", "adapter_args": {"split": "val"}}}},
           "fields": {"label": "target.yes", "group": "meta.grp"}}
    p = ws / "exp.yaml"
    p.write_text(yaml.safe_dump(exp))
    return ws, p


def machine(tmp_path):
    return MachineConfig(python="/venv/bin/python", model_dir=str(tmp_path / "models"), out_root=str(tmp_path / "out"))


def test_adapter_path_is_relative_to_the_experiment(tmp_path):
    ws, p = workspace(tmp_path)
    e = load_experiment(p)
    assert e.data.adapter == str(ws / "adapters_local" / "my_adapter.py")
    assert e.data.train_adapter_args == {"split": "train"} and e.data.eval["val"].adapter_args == {"split": "val"}
    assert load_experiment(workspace(tmp_path / "b", adapter_ref="onejev_rows")[1]).data.adapter == "onejev_rows"


def test_adapter_args_must_be_a_mapping(tmp_path):
    ws, p = workspace(tmp_path)
    d = yaml.safe_load(p.read_text())
    d["data"]["adapter_args"] = ["x"]
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="adapter_args"):
        load_experiment(p)


def test_prepare_with_file_adapter_and_args(tmp_path):
    ws, p = workspace(tmp_path)
    exp = load_experiment(p)
    out = tmp_path / "out" / "csvexp" / "prepared"
    info = prepare(exp, out)
    assert info["counts"]["train"] == 20 and info["counts"]["select"] == 10
    sha = hashlib.sha256((ws / "adapters_local" / "my_adapter.py").read_bytes()).hexdigest()
    assert info["adapter"] == {"name": exp.data.adapter, "file": exp.data.adapter, "sha256": sha}
    assert info["adapter_args"] == {"global": {}, "train": {"split": "train"}, "eval": {"val": {"split": "val"}}}


def test_module_adapter_file_is_found():
    from harness.catalog import ROOT
    assert adapter_file("onejev_rows") == ROOT / "adapters" / "onejev_rows.py"
    with pytest.raises(ConfigError, match="nope"):
        adapter_file("nope")


def resolve_ws(tmp_path):
    ws, p = workspace(tmp_path)
    exp = load_experiment(p)
    exp.base, exp.recipe = "joey-1", str(Path(__file__).resolve().parents[1] / "examples/recipes/joey_full_1gpu.yaml")
    m = machine(tmp_path)
    prepare(exp, Path(m.out_root) / exp.name / "prepared")
    return ws, exp, m


def test_resolve_records_and_copies_the_adapter(tmp_path):
    ws, exp, m = resolve_ws(tmp_path)
    lock_path = resolve(exp, m, query=lambda uuid=None: "0, GPU-0, A100, 81920\n", plan_runner=plan_ok(5, 1),
                        fingerprint=fake_fp)
    lock = yaml.safe_load(lock_path.read_text())
    src = ws / "adapters_local" / "my_adapter.py"
    assert lock["data"]["adapter_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    copy = Path(lock["data"]["adapter_copy"])
    assert copy.parent.parent == lock_path.parent and copy.read_text() == ADAPTER
    assert lock["data"]["train_adapter_args"] == {"split": "train"}
    assert lock["eval_sets"]["val"]["adapter_args"] == {"split": "val"}
    from harness.finalize import exp_from_lock
    from harness.prepare import read_eval, read_train
    again = exp_from_lock(lock)
    assert len(read_train(again.data)) == 20 and len(read_eval(again.data, "val", again.data.eval["val"].train_labels)) == 10


def test_adapter_changed_after_prepare_is_refused(tmp_path):
    ws, exp, m = resolve_ws(tmp_path)
    (ws / "adapters_local" / "my_adapter.py").write_text(ADAPTER + "\n# edited\n")
    with pytest.raises(ConfigError, match="adapter changed after prepare"):
        resolve(exp, m, query=lambda uuid=None: "0, GPU-0, A100, 81920\n", plan_runner=plan_ok(5, 1),
                fingerprint=fake_fp)


def test_run_card_names_the_adapter(tmp_path):
    from test_finalize import FakeScorer, make_run
    from harness.finalize import finalize
    run_dir, lock = make_run(tmp_path)
    finalize(run_dir, scorer=FakeScorer())
    card = (run_dir / "RUN_CARD.md").read_text()
    assert lock["data"]["adapter_sha256"] in card and "onejev_rows" in card


def test_finalize_reads_through_the_run_dirs_adapter_copy(tmp_path):
    from harness.finalize import exp_from_lock
    ws, exp, m = resolve_ws(tmp_path)
    lock = yaml.safe_load(resolve(exp, m, query=lambda uuid=None: "0, GPU-0, A100, 81920\n", plan_runner=plan_ok(5, 1),
                                  fingerprint=fake_fp).read_text())
    (ws / "adapters_local" / "my_adapter.py").write_text("def adapt(path, **kw):\n    raise RuntimeError('edited')\n")
    again = exp_from_lock(lock)
    assert again.data.adapter == lock["data"]["adapter_copy"]
    from harness.prepare import read_train
    assert len(read_train(again.data)) == 20  # the original was edited; the copy is what the run used


def test_tampered_adapter_copy_fails_finalize_and_refuses_eval_report(tmp_path):
    from test_finalize import FakeScorer, make_run
    from harness import report_eval
    from harness.finalize import finalize
    run_dir, lock = make_run(tmp_path)
    copy = Path(lock["data"]["adapter_copy"])
    copy.write_text(copy.read_text() + "\n# tampered\n")
    out = finalize(run_dir, scorer=FakeScorer())
    assert out["status"] == "failed" and any("adapter" in f and "sha256" in f for f in out["failures"])
    (run_dir / "runs" / "exp-s42" / "final").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ConfigError, match="adapter"):
        report_eval.run(run_dir, allow_report=True, scorer=FakeScorer())


@pytest.mark.parametrize("change", ["global", "train", "eval"])
def test_adapter_args_changed_after_prepare_is_refused(tmp_path, change):
    ws, exp, m = resolve_ws(tmp_path)
    if change == "global":
        exp.data.adapter_args = {"extra": 1}
    elif change == "train":
        exp.data.train_adapter_args = {"split": "val"}
    else:
        exp.data.eval["val"].adapter_args = {"split": "train"}
    with pytest.raises(ConfigError, match="adapter_args changed after prepare.*ojh prepare"):
        resolve(exp, m, query=lambda uuid=None: "0, GPU-0, A100, 81920\n", plan_runner=plan_ok(5, 1),
                fingerprint=fake_fp)
