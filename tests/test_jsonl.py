"""JSONL files survive U+2028/U+2029 inside strings (C1)."""
import json
import os
from pathlib import Path

import pytest
import yaml

from harness import finalize, registry
from harness.config import DataSpec, EvalSetSpec, ExperimentConfig, FieldsSpec
from harness.prepare import prepare
from harness.status import status

SEP = "line sep para"


def row(i, content="hi", split="train"):
    return {"id": i, "split": split, "state": {"messages": [{"role": "user", "content": content}]},
            "question": {"type": "noul", "instructions": "x", "criteria": {"true": "y", "false": "n"}},
            "target": {"yes": 1.0, "no": 0.0}, "meta": {"f": 0.5}}


def write_raw(path, rows):
    """Write as a source export would: UTF-8, separators left raw."""
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


@pytest.mark.filterwarnings("ignore:data.train_sha256 not set")
def test_prepare_then_finalize_reader_keeps_separators(tmp_path):
    train = write_raw(tmp_path / "train.jsonl", [row("t0"), row("t1", SEP), row("t2")])
    val = write_raw(tmp_path / "val.jsonl", [row("v0", SEP, "val"), row("v1", split="val")])
    exp = ExperimentConfig(name="e", base="b", recipe="r",
                           data=DataSpec(train=train, eval={"val": EvalSetSpec(train_labels=val)}),
                           fields=FieldsSpec(label="target.yes"))
    out = tmp_path / "out"
    prepare(exp, out)
    rows = finalize._jsonl(out / "train.jsonl")
    assert [r["id"] for r in rows] == ["t0", "t1", "t2"]
    assert rows[1]["state"]["messages"][0]["content"] == SEP
    sel = finalize._jsonl(out / "select.jsonl")
    assert [r["id"] for r in sel] == ["v0", "v1"] and sel[0]["state"]["messages"][0]["content"] == SEP
    # the written files hold no raw separator, so line-splitting readers elsewhere are safe too
    for f in ("train.jsonl", "select.jsonl"):
        text = (out / f).read_text(encoding="utf-8")
        assert " " not in text and " " not in text
        assert len(text.splitlines()) == len(finalize._jsonl(out / f))


def test_registry_conclusion_with_separators(tmp_path):
    path = tmp_path / "registry.jsonl"
    rec = {"run_id": "r1", "seed": 42, "exp": "e", "conclusion": ""}
    registry.append(path, dict(rec))
    registry.append(path, dict(rec, run_id="r2", conclusion=f"beats tfidf{SEP}"))
    assert registry.note(path, "r1", f"first{SEP}") == 1
    registry.append(path, dict(rec, run_id="r3"))  # an append after a rewrite still reads the file
    recs = registry.read(path)
    assert [r["run_id"] for r in recs] == ["r1", "r2", "r3"]
    assert recs[0]["conclusion"] == f"first{SEP}" and recs[1]["conclusion"] == f"beats tfidf{SEP}"
    assert "r1" in registry.table(path, include_incomplete=True)


def test_status_reads_event_with_separator(tmp_path):
    run_dir = tmp_path / "plan"
    d = run_dir / "runs" / "exp-s42"
    d.mkdir(parents=True)
    (run_dir / "run.lock.yaml").write_text(yaml.safe_dump({
        "experiment": "exp", "run_dir": str(run_dir), "seeds": [42], "flags": {}, "plan": {"steps": 50}}))
    recs = [{"step": 10, "sec_per_step": 1.0},
            {"event": "harness_eval", "step": 10, "set": "select", "n": 5, "nll": 0.4, "path": f"p{SEP}"}]
    (d / "log.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
                                 + '{"step": 11, "sec', encoding="utf-8")  # half-written last line
    s = status(run_dir)
    assert s["last_eval"]["path"] == f"p{SEP}" and s["step"] == 10


def test_onejev_rows_adapter_keeps_separators(tmp_path):
    from adapters import onejev_rows
    path = write_raw(tmp_path / "rows.jsonl", [row("a", "汉字" + SEP), row("b")])
    got = list(onejev_rows.adapt(Path(path)))
    assert [r["id"] for r in got] == ["a", "b"] and got[0]["state"]["messages"][0]["content"] == "汉字" + SEP


V1 = os.environ.get("HARNESS_V1_DIR")


@pytest.mark.skipif(not V1, reason="HARNESS_V1_DIR not set")
def test_v1_train_export_round_trips(tmp_path):
    b = Path(V1) / "full/export_majority75/binary"
    exp = ExperimentConfig(
        name="v1", base="b", recipe="r",
        data=DataSpec(train=str(b / "train.jsonl"),
                      train_sha256="fe83846076cb6c72d80a048548f38edae751bb9df9865f11409992289c3515f2",
                      eval={"val_id": EvalSetSpec(str(b / "val_id.jsonl")),
                            "val_catood": EvalSetSpec(str(b / "val_catood.jsonl"))},
                      split_file=str(Path(V1) / "train/splits.json")),
        fields=FieldsSpec(label="target.yes"))
    prepare(exp, tmp_path / "p")
    assert len(finalize._jsonl(tmp_path / "p" / "train.jsonl")) == 5695
