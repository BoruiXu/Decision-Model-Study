import json
import random
from pathlib import Path

import pytest

from harness import cli
from harness.config import ConfigError
from harness.inspect_data import split_file_shape
from harness.jsonl import read_jsonl
from harness.make_split import holdout, select_report
from harness.prepare import TestSplitRefused


@pytest.fixture
def tmp_path(tmp_path):
    """A work dir below pytest's tmp dir, whose own name (test_<name>0) would read as a test path."""
    d = tmp_path / "work"
    d.mkdir()
    return d


def rows(prefix, n, n_groups, split="val", seed=0):
    rng = random.Random(seed)
    return [{"id": f"{prefix}{i}", "split": split, "meta": {"cat": f"c{rng.randrange(n_groups)}"}, "x": i}
            for i in range(n)]


def write(path, rs):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rs))
    return str(path)


def sides_by_group(rs, assignment):
    out = {}
    for r in rs:
        out.setdefault(r["meta"]["cat"], set()).add(assignment[r["id"]])
    return out


def test_select_report_is_deterministic_and_group_aware(tmp_path):
    a = rows("a", 200, 25)
    b = rows("b", 60, 6, seed=1)
    files = [("val_id", write(tmp_path / "val_id.jsonl", a)), ("ood", write(tmp_path / "b.jsonl", b))]
    one = select_report(files, group="meta.cat")
    two = select_report(files, group="meta.cat")
    assert one == two
    assert set(one) == {"val_id", "ood"}
    for name, rs in (("val_id", a), ("ood", b)):
        asg = one[name]["assignment"]
        assert set(asg) == {r["id"] for r in rs}
        assert all(len(s) == 1 for s in sides_by_group(rs, asg).values())  # whole groups only
    share = sum(v == "select" for v in one["val_id"]["assignment"].values()) / 200
    assert abs(share - 0.5) < 0.1
    assert one["val_id"]["seed"] == 20261115 and one["val_id"]["group"] == "meta.cat"
    assert split_file_shape(one) is not None  # the existing split_file format
    other = select_report(files, group="meta.cat", seed=7)
    assert other["val_id"]["assignment"] != one["val_id"]["assignment"]


def test_frac_controls_select_share(tmp_path):
    a = rows("a", 400, 80)
    out = select_report([("v", write(tmp_path / "v.jsonl", a))], group="meta.cat", frac=0.25)
    share = sum(v == "select" for v in out["v"]["assignment"].values()) / 400
    assert abs(share - 0.25) < 0.05


def test_refuses_test_rows_and_test_files(tmp_path):
    t = write(tmp_path / "test_id.jsonl", rows("t", 10, 2))
    with pytest.raises(TestSplitRefused):
        select_report([("test_id", t)], group="meta.cat")
    mixed = rows("m", 10, 2)
    mixed[3]["split"] = "test_ood"
    m = write(tmp_path / "m.jsonl", mixed)
    with pytest.raises(TestSplitRefused):
        select_report([("m", m)], group="meta.cat")
    assert select_report([("m", m)], group="meta.cat", allow_test=True)["m"]["assignment"]


def test_missing_group_or_id_or_duplicate_is_an_error(tmp_path):
    rs = rows("a", 10, 2)
    del rs[4]["meta"]["cat"]
    with pytest.raises(ConfigError, match="meta.cat"):
        select_report([("a", write(tmp_path / "a.jsonl", rs))], group="meta.cat")
    rs = rows("a", 10, 2)
    del rs[0]["id"]
    with pytest.raises(ConfigError, match="id"):
        select_report([("a", write(tmp_path / "a.jsonl", rs))], group="meta.cat")
    a, b = write(tmp_path / "a.jsonl", rows("x", 5, 2)), write(tmp_path / "b.jsonl", rows("x", 5, 2))
    with pytest.raises(ConfigError, match="x0"):
        select_report([("a", a), ("b", b)], group="meta.cat")


def test_holdout_writes_train_val_and_manifest(tmp_path):
    src = rows("t", 300, 30, split="train")
    path = write(tmp_path / "train_all.jsonl", src)
    out = tmp_path / "split"
    m = holdout(path, out, group="meta.cat", frac=0.2)
    train, val = read_jsonl(out / "train.jsonl"), read_jsonl(out / "val.jsonl")
    assert len(train) + len(val) == 300 and {r["id"] for r in train}.isdisjoint(r["id"] for r in val)
    assert {r["meta"]["cat"] for r in train}.isdisjoint(r["meta"]["cat"] for r in val)
    assert abs(len(val) / 300 - 0.2) < 0.07
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest == m and manifest["seed"] == 20261115 and manifest["frac"] == 0.2
    assert set(manifest["groups"]["val"]) == {r["meta"]["cat"] for r in val}
    import hashlib
    assert manifest["sha256"]["val.jsonl"] == hashlib.sha256((out / "val.jsonl").read_bytes()).hexdigest()
    assert manifest["source_sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
    out2 = tmp_path / "split2"
    holdout(path, out2, group="meta.cat", frac=0.2)
    assert (out2 / "val.jsonl").read_bytes() == (out / "val.jsonl").read_bytes()


def test_cli_make_split(tmp_path, capsys):
    a = write(tmp_path / "val_id.jsonl", rows("a", 40, 8))
    b = write(tmp_path / "other.jsonl", rows("b", 40, 8, seed=3))
    out = tmp_path / "splits.json"
    assert cli.main(["make-split", "--out", str(out), "--group", "meta.cat", a, f"ood={b}"]) == 0
    obj = json.loads(out.read_text())
    assert set(obj) == {"val_id", "ood"}
    capsys.readouterr()
    # an existing, different split file is never overwritten
    assert cli.main(["make-split", "--out", str(out), "--group", "meta.cat", "--seed", "1", a]) == 1
    assert "exists" in capsys.readouterr().err and json.loads(out.read_text()) == obj
    t = write(tmp_path / "test.jsonl", rows("t", 4, 2))
    assert cli.main(["make-split", "--out", str(tmp_path / "s2.json"), "--group", "meta.cat", t]) == 2
    d = tmp_path / "hold"
    assert cli.main(["make-split", "--mode", "holdout", "--frac", "0.25", "--out", str(d), "--group", "meta.cat",
                     a]) == 0
    assert (d / "train.jsonl").is_file() and (d / "val.jsonl").is_file() and (d / "manifest.json").is_file()


def test_holdout_out_must_be_a_directory(tmp_path):
    path = write(tmp_path / "train.jsonl", rows("t", 20, 4, split="train"))
    with pytest.raises(ConfigError, match="directory"):
        holdout(path, tmp_path / "splits.json", group="meta.cat", frac=0.2)
    assert not (tmp_path / "splits.json").exists()


def test_holdout_refuses_test_rows_in_any_split_like_column(tmp_path):
    rs = rows("t", 20, 4, split="train")
    for r in rs:
        del r["split"]
        r["set"] = "train"
    rs[5]["set"] = "Test"
    path = write(tmp_path / "all.jsonl", rs)
    with pytest.raises(TestSplitRefused):
        holdout(path, tmp_path / "out", group="meta.cat", frac=0.2)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("rel", ["test/part-0.jsonl", "testset.jsonl", "Test-ood.jsonl"])
def test_refuses_test_like_paths(tmp_path, rel):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    write(p, rows("t", 10, 2))
    with pytest.raises(TestSplitRefused):
        select_report([("x", str(p))], group="meta.cat")
    assert select_report([("x", str(p))], group="meta.cat", allow_test=True)


def test_id_field_is_dotted_and_kept_as_text(tmp_path):
    rs = [{"meta": {"uid": i, "cat": f"c{i % 5}"}, "x": i} for i in range(30)]
    out = select_report([("v", write(tmp_path / "v.jsonl", rs))], group="meta.cat", id_field="meta.uid")
    assert set(out["v"]["assignment"]) == {str(i) for i in range(30)} and out["v"]["id_field"] == "meta.uid"
    del rs[3]["meta"]["uid"]
    with pytest.raises(ConfigError, match="meta.uid.*--id"):
        select_report([("v", write(tmp_path / "v.jsonl", rs))], group="meta.cat", id_field="meta.uid")


def test_raw_csv_split_feeds_a_file_adapter_through_prepare(tmp_path, capsys):
    import csv
    import yaml
    from harness.config import load_experiment
    from harness.prepare import prepare
    with open(tmp_path / "val.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "text", "label", "source"])
        w.writerows([[f"{i:04d}", f"text {i}", i % 2, f"s{i % 6}"] for i in range(36)])
    with open(tmp_path / "train.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "text", "label", "source"])
        w.writerows([[f"t{i}", f"text {i}", i % 2, f"s{i % 6}"] for i in range(20)])
    assert cli.main(["make-split", "--out", str(tmp_path / "splits.json"), "--group", "source", "--id", "qid",
                     f"val={tmp_path / 'val.csv'}"]) == 0
    assert "0007" in json.loads((tmp_path / "splits.json").read_text())["val"]["assignment"]  # not 7
    (tmp_path / "adapter.py").write_text(
        "import csv\n\ndef adapt(path, split='train'):\n"
        "    for r in csv.DictReader(open(path)):\n"
        "        y = float(r['label'])\n"
        "        yield {'id': r['qid'], 'split': split, 'state': {'messages': [{'role': 'user', 'content': r['text']}]},\n"
        "               'question': {'type': 'noul', 'instructions': 'ok?', 'criteria': {'true': 'y', 'false': 'n'}},\n"
        "               'target': {'yes': y, 'no': 1 - y}, 'meta': {'source': r['source']}}\n")
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump({
        "name": "csv", "data": {"adapter": "adapter.py", "train": "train.csv", "split_file": "splits.json",
                                "eval": {"val": {"train_labels": "val.csv", "adapter_args": {"split": "val"}}}},
        "fields": {"label": "target.yes", "group": "meta.source"}}))
    info = prepare(load_experiment(tmp_path / "exp.yaml"), tmp_path / "prepared")
    assert info["counts"]["select"] + info["counts"]["report"]["val"] == 36 and info["counts"]["select"] > 0
    assert info["dropped"]["unassigned"] == []
