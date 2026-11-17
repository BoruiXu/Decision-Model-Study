import hashlib
import json
import os
import warnings
from pathlib import Path

import pytest

from harness.config import DataSpec, EvalSetSpec, ExperimentConfig, FieldsSpec, SubsetSpec
from harness.prepare import (EvalParts, IdOverlap, ShaMismatch, TestSplitRefused,
                             apply_subset, build_eval_parts, prepare, validate_rows)


def row(i, y=1.0, split="train", f=0.5, cat="a"):  # noqa: default type noul
    return {"id": i, "state": {"messages": [{"role": "user", "content": i}]},
            "question": {"type": "noul", "instructions": "x", "criteria": {"true": "t", "false": "f"}},
            "target": {"yes": y, "no": 1 - y}, "meta": {"f": f, "cat": cat}, "split": split}


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def make_exp(tmp_path, train_rows, evals=None, assignment=None, **kw):
    train = write(tmp_path / "train.jsonl", train_rows)
    ev, split = {}, None
    for name, (labels, ref) in (evals or {}).items():
        ev[name] = EvalSetSpec(train_labels=write(tmp_path / f"{name}.jsonl", labels),
                               reference=write(tmp_path / f"{name}_ref.jsonl", ref) if ref is not None else None)
    if assignment is not None:
        p = tmp_path / "splits.json"
        p.write_text(json.dumps({k: {"assignment": v} for k, v in assignment.items()}))
        split = str(p)
    data = DataSpec(train=train, eval=ev, split_file=split, **kw)
    return ExperimentConfig(name="e", base="b", recipe="r", data=data, fields=FieldsSpec(label="meta.f"))


def test_validate_rows_errors():
    good = row("a")
    assert validate_rows([good]) == []
    errs = validate_rows([good, row("a")])
    assert any("duplicate" in e for e in errs)
    bad_key = row("b"); bad_key["target"] = {"maybe": 1.0}
    assert any("labels" in e for e in validate_rows([bad_key]))
    bad_sum = row("c"); bad_sum["target"] = {"true": 0.5, "false": 0.2}
    assert any("sum" in e for e in validate_rows([bad_sum]))
    no_state = row("d"); del no_state["state"]
    assert any("state" in e for e in validate_rows([no_state]))
    # noul labels are yes/no; true/false are only descriptions
    v1 = row("e"); v1["question"]["type"] = "noul"; v1["target"] = {"yes": 1.0, "no": 0.0}
    assert validate_rows([v1]) == []
    wrong = row("f"); wrong["question"]["type"] = "noul"; wrong["target"] = {"true": 1.0}
    assert any("labels" in e for e in validate_rows([wrong]))
    neg = row("g"); neg["question"]["type"] = "noul"; neg["target"] = {"yes": 1.5, "no": -0.5}
    assert any("[0, 1]" in e for e in validate_rows([neg]))
    ch = row("h"); ch["question"] = {"type": "choice", "criteria": {"a": "x", "b": "y"}}
    ch["target"] = {"a": 0.5, "b": 0.5}
    assert validate_rows([ch]) == []
    ch["target"] = {"yes": 1.0}
    assert validate_rows([ch])
    sc = row("i"); sc["question"] = {"type": "score", "criteria": ["lo", "hi"]}
    sc["target"] = {"0": 0.25, "1": 0.75}
    assert validate_rows([sc]) == []
    sc["target"] = {"2": 1.0}
    assert validate_rows([sc])
    unk = row("j"); unk["question"]["type"] = "weird"
    assert any("unknown question type" in e for e in validate_rows([unk]))


def test_subset_deterministic():
    rows = [row(f"r{i}", cat="a" if i % 2 else "b") for i in range(40)]
    s = SubsetSpec(where={"meta.cat": "a"}, n=5, seed=1)
    a, b = apply_subset(rows, s), apply_subset(list(reversed(rows)), s)
    assert [r["id"] for r in a] == [r["id"] for r in b] and len(a) == 5
    assert all(r["meta"]["cat"] == "a" for r in a)
    c = apply_subset(rows, SubsetSpec(where={"meta.cat": "a"}, n=5, seed=2))
    assert [r["id"] for r in c] != [r["id"] for r in a]
    assert apply_subset(rows, SubsetSpec()) == rows


def test_split_assignment(tmp_path):
    ids = [f"v{i}" for i in range(6)]
    assign = {i: ("select" if n < 3 else "report") for n, i in enumerate(ids)}
    assign["extra"] = "select"
    exp = make_exp(tmp_path, [row("t1")],
                   {"val": ([row(i, split="val") for i in ids] + [row("stray", split="val")],
                            [row(i) for i in ids + ["extra"]])},
                   {"val": assign})
    p = build_eval_parts(exp)
    assert [r["id"] for r in p.select_rows] == ids[:3]
    assert p.select_index == {i: "val" for i in ids[:3]}
    assert p.report_ids == {"val": ids[3:]}
    assert set(p.reference) == set(ids)
    assert "stray" in p.dropped["val"] and "extra" in p.dropped["val"]
    exp2 = make_exp(tmp_path, [row("t1")], {"val": ([row("a"), row("stray")], None)}, {"val": {"a": "select"}})
    assert build_eval_parts(exp2).dropped["unassigned"] == ["stray"]


def test_invalid_eval_rows_named(tmp_path):
    bad = row("a"); bad["target"] = {"true": 1.0}
    exp = make_exp(tmp_path, [row("t1")], {"val": ([bad], None)}, {"val": {"a": "select"}})
    with pytest.raises(ValueError, match="val.*a"):
        build_eval_parts(exp)


def test_test_split_refused(tmp_path):
    exp = make_exp(tmp_path, [row("t1")], {"val": ([row("a", split="test_id")], None)}, {"val": {"a": "select"}})
    with pytest.raises(TestSplitRefused):
        build_eval_parts(exp)
    assert build_eval_parts(exp, allow_test=True).select_index == {"a": "val"}


@pytest.mark.filterwarnings("ignore:data.train_sha256 not set")
def test_rejects_id_overlap(tmp_path):
    exp = make_exp(tmp_path, [row("a")], {"val": ([row("a")], None)}, {"val": {"a": "select"}})
    with pytest.raises(IdOverlap):
        prepare(exp, tmp_path / "out")
    exp = make_exp(tmp_path, [row("t")], {"v1": ([row("a")], None), "v2": ([row("a")], None)},
                   {"v1": {"a": "select"}, "v2": {"a": "select"}})
    with pytest.raises(IdOverlap):
        prepare(exp, tmp_path / "out")


def test_sha_mismatch_stops(tmp_path):
    exp = make_exp(tmp_path, [row("t")], train_sha256="ab" * 32)
    with pytest.raises(ShaMismatch):
        prepare(exp, tmp_path / "out")
    assert not (tmp_path / "out" / "train.jsonl").exists()


def test_prepare_writes(tmp_path):
    rows = [row(f"t{i}") for i in range(10)]
    exp = make_exp(tmp_path, rows, {"val": ([row("a", split="val"), row("b", split="val")], None)},
                   {"val": {"a": "select", "b": "report"}})
    exp.subset = SubsetSpec(n=4, seed=3)
    with pytest.warns(UserWarning, match="train_sha256"):
        info = prepare(exp, tmp_path / "out")
    out = tmp_path / "out"
    assert len((out / "train.jsonl").read_text().splitlines()) == 4
    assert [json.loads(l)["id"] for l in (out / "select.jsonl").read_text().splitlines()] == ["a"]
    assert json.loads((out / "select_index.json").read_text()) == {"a": "val"}
    assert json.loads((out / "prepared.json").read_text()) == info
    assert info["train_sha256_source"] == hashlib.sha256(Path(exp.data.train).read_bytes()).hexdigest()
    assert info["train_sha256_prepared"] == hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    assert info["counts"]["train"] == 4 and info["counts"]["select"] == 1
    # matching sha: no warning
    exp.data.train_sha256 = info["train_sha256_source"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare(exp, tmp_path / "out2")


def test_empty_subset_noop_and_adapter_template(tmp_path):
    from adapters import template
    with pytest.raises(NotImplementedError):
        template.adapt(Path("x"))
    exp = make_exp(tmp_path, [row("t1"), row("t2")])
    exp.subset = SubsetSpec()
    with pytest.warns(UserWarning):
        assert prepare(exp, tmp_path / "o")["counts"]["train"] == 2


V1 = os.environ.get("HARNESS_V1_DIR")


@pytest.mark.skipif(not V1, reason="HARNESS_V1_DIR not set")
def test_v1_select_matches(tmp_path):
    v1 = Path(V1)
    b = v1 / "full/export_majority75/binary"
    data = DataSpec(train=str(b / "train.jsonl"),
                    train_sha256="fe83846076cb6c72d80a048548f38edae751bb9df9865f11409992289c3515f2",
                    eval={"val_id": EvalSetSpec(str(b / "val_id.jsonl")),
                          "val_catood": EvalSetSpec(str(b / "val_catood.jsonl"))},
                    split_file=str(v1 / "train/splits.json"))
    exp = ExperimentConfig(name="v1", base="b", recipe="r", data=data, fields=FieldsSpec(label="meta.f"))
    parts = build_eval_parts(exp)
    want = {json.loads(l)["id"] for l in (v1 / "train/data/select.jsonl").read_text().splitlines()}
    assert {r["id"] for r in parts.select_rows} == want and len(want) == 380
    info = prepare(exp, tmp_path / "p", allow_test=False)
    assert info["train_sha256_source"].startswith("fe83846076cb6c72")


def test_no_split_file_all_select(tmp_path):
    exp = make_exp(tmp_path, [row("t1")], {"val": ([row("a"), row("b")], None)})
    p = build_eval_parts(exp)
    assert [r["id"] for r in p.select_rows] == ["a", "b"]
    assert p.select_index == {"a": "val", "b": "val"}
    assert p.report_ids.get("val", []) == []
    assert not p.dropped.get("unassigned")


def test_split_file_missing_set_raises(tmp_path):
    from harness.config import ConfigError
    exp = make_exp(tmp_path, [row("t1")], {"val": ([row("a")], None), "other": ([row("b")], None)},
                   {"val": {"a": "select"}})
    with pytest.raises(ConfigError, match="other"):
        build_eval_parts(exp)


def test_split_file_present_set_unassigned_dropped(tmp_path):
    exp = make_exp(tmp_path, [row("t1")], {"val": ([row("a"), row("z")], None)}, {"val": {"a": "select"}})
    p = build_eval_parts(exp)
    assert [r["id"] for r in p.select_rows] == ["a"]
    assert p.dropped["unassigned"] == ["z"]


@pytest.mark.filterwarnings("ignore:data.train_sha256 not set")
def test_test_rows_in_train_refused(tmp_path):
    exp = make_exp(tmp_path, [row("t1"), row("t2", split="test_catood")], {"val": ([row("a", split="val")], None)},
                   {"val": {"a": "select"}})
    with pytest.raises(TestSplitRefused, match="t2"):
        prepare(exp, tmp_path / "out")
    assert not (tmp_path / "out" / "train.jsonl").exists()
    info = prepare(exp, tmp_path / "out", allow_test=True)
    assert info["allow_test"] is True and info["counts"]["train"] == 2


@pytest.mark.filterwarnings("ignore:data.train_sha256 not set")
def test_prepared_json_records_allow_test(tmp_path):
    exp = make_exp(tmp_path, [row("t1")], {"val": ([row("a", split="val")], None)}, {"val": {"a": "select"}})
    assert prepare(exp, tmp_path / "out")["allow_test"] is False
    assert json.loads((tmp_path / "out" / "prepared.json").read_text())["allow_test"] is False
