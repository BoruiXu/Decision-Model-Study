import csv
import json
from pathlib import Path

import pytest

from harness import cli
from harness.inspect_data import inspect_path, render


@pytest.fixture
def tmp_path(tmp_path):
    """A work dir below pytest's tmp dir, whose own name (test_<name>0) would read as a test path."""
    d = tmp_path / "work"
    d.mkdir()
    return d


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return Path(path)


def onejev_row(i, split="train"):
    p = (i % 5) / 5
    return {"id": f"r{i}", "split": split,
            "state": {"messages": [{"role": "user", "content": f"question number {i} about something long"}]},
            "question": {"type": "noul", "instructions": "agree?", "criteria": {"yes": "y", "no": "n"}},
            "target": {"yes": p, "no": 1 - p}, "meta": {"f": 0.5 + p / 2, "cat": f"g{i % 3}"}}


def chat_row(i):
    return {"conversation": [{"role": "user", "content": f"please help me with task {i} " * 3},
                             {"role": "assistant", "content": "sure"}],
            "label": "good" if i % 2 else "bad", "score": (i % 10) / 10, "topic": ["math", "code"][i % 2],
            "uid": i}


def files_of(report):
    return {Path(f["path"]).name: f for f in report["files"]}


def test_onejev_rows_pass_the_schema(tmp_path):
    write_jsonl(tmp_path / "rows.jsonl", [onejev_row(i) for i in range(12)])
    f = files_of(inspect_path(tmp_path))["rows.jsonl"]
    assert f["type"] == "jsonl" and f["rows"] == 12 and f["size_bytes"] == (tmp_path / "rows.jsonl").stat().st_size
    assert f["onejev"]["passes"] is True and f["onejev"]["errors"] == 0
    meta_f = f["fields"]["meta.f"]
    assert meta_f["types"] == {"float": 12} and meta_f["fill"] == 1.0 and meta_f["unit_interval"] is True
    cats = {v["value"]: v["count"] for v in f["fields"]["meta.cat"]["values"]}
    assert cats == {"g0": 4, "g1": 4, "g2": 4}
    assert "values" not in f["fields"]["id"]  # one value per row: not low-cardinality


def test_chat_label_jsonl_facts(tmp_path):
    write_jsonl(tmp_path / "chat.jsonl", [chat_row(i) for i in range(10)])
    f = files_of(inspect_path(tmp_path / "chat.jsonl", rows=2))["chat.jsonl"]
    assert f["onejev"]["passes"] is False and f["onejev"]["errors"] > 0
    fields = f["fields"]
    assert {v["value"] for v in fields["label"]["values"]} == {"good", "bad"}
    assert {v["value"] for v in fields["conversation[].role"]["values"]} == {"user", "assistant"}
    assert fields["conversation[].content"]["text_like"] is True
    assert fields["conversation[].content"]["avg_len"] > 20
    assert fields["conversation"]["types"] == {"list": 10}
    assert fields["score"]["unit_interval"] is True
    assert fields["uid"]["unit_interval"] is False
    assert len(f["samples"]) == 2


def test_text_label_csv_numbers_and_text(tmp_path):
    with open(tmp_path / "data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label", "group", "conf"])
        for i in range(8):
            w.writerow([f"some fairly long request text number {i}", i % 2, f"grp{i % 2}", "" if i == 0 else 0.25])
    f = files_of(inspect_path(tmp_path))["data.csv"]
    assert f["type"] == "csv" and f["rows"] == 8
    assert f["fields"]["text"]["text_like"] is True
    assert f["fields"]["label"]["types"] == {"int": 8}
    assert {v["value"] for v in f["fields"]["label"]["values"]} == {0, 1}
    assert f["fields"]["conf"]["fill"] == pytest.approx(7 / 8) and f["fields"]["conf"]["unit_interval"] is True
    assert len(f["samples"]) == 3


def test_split_column_and_name_hints(tmp_path):
    rows = [onejev_row(i, split=["train", "val_id", "test_id"][i % 3]) for i in range(9)]
    write_jsonl(tmp_path / "all.jsonl", rows)
    write_jsonl(tmp_path / "train.jsonl", [onejev_row(i) for i in range(3)])
    write_jsonl(tmp_path / "my-validation.jsonl", [onejev_row(i) for i in range(2)])
    write_jsonl(tmp_path / "test.jsonl", [onejev_row(i, "test") for i in range(4)])
    fs = files_of(inspect_path(tmp_path))
    assert fs["all.jsonl"]["split_columns"] == {"split": {"train": 3, "val_id": 3, "test_id": 3}}
    assert all(s["split"] != "test_id" for s in fs["all.jsonl"]["samples"])  # no test rows shown
    assert fs["train.jsonl"]["name_hint"] == "train"
    assert fs["my-validation.jsonl"]["name_hint"] == "validation"
    t = fs["test.jsonl"]
    assert t["name_hint"] == "test" and t["rows"] == 4
    assert t["fields"] is None and t["samples"] == [] and "test" in t["note"]


def test_split_file_shape(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps({
        "val_id": {"assignment": {"a": "select", "b": "report", "c": "select"}},
        "val_ood": {"assignment": {"d": "report"}}}))
    (tmp_path / "other.json").write_text(json.dumps({"x": 1}))
    fs = files_of(inspect_path(tmp_path))
    assert fs["splits.json"]["split_file"] == {"val_id": {"select": 2, "report": 1},
                                               "val_ood": {"select": 0, "report": 1}}
    assert fs["other.json"]["split_file"] is None


def test_json_list_of_rows(tmp_path):
    (tmp_path / "rows.json").write_text(json.dumps([chat_row(i) for i in range(4)]))
    f = files_of(inspect_path(tmp_path))["rows.json"]
    assert f["rows"] == 4 and "label" in f["fields"]


def test_inspect_writes_nothing_and_cli(tmp_path, capsys):
    write_jsonl(tmp_path / "chat.jsonl", [chat_row(i) for i in range(5)])
    (tmp_path / "notes.txt").write_text("not data")
    before = sorted((p.name, p.stat().st_mtime_ns) for p in tmp_path.iterdir())
    assert cli.main(["inspect", str(tmp_path), "--json", "--rows", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [Path(f["path"]).name for f in out["files"]] == ["chat.jsonl"]
    assert len(out["files"][0]["samples"]) == 1
    assert cli.main(["inspect", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "chat.jsonl" in text and "label" in text and "conversation[].role" in text
    assert sorted((p.name, p.stat().st_mtime_ns) for p in tmp_path.iterdir()) == before


def test_samples_are_truncated(tmp_path):
    write_jsonl(tmp_path / "long.jsonl", [{"id": "a", "text": "x" * 5000}])
    s = files_of(inspect_path(tmp_path))["long.jsonl"]["samples"][0]
    assert len(s["text"]) < 400 and s["text"].endswith("…")


def test_render_mentions_key_facts(tmp_path):
    write_jsonl(tmp_path / "rows.jsonl", [onejev_row(i) for i in range(6)])
    text = render(inspect_path(tmp_path))
    assert "rows.jsonl" in text and "OneJev rows: yes" in text and "meta.cat" in text


def test_test_rows_never_reach_fields_values_or_samples(tmp_path):
    rows = [dict(chat_row(i), set="train") for i in range(6)]
    rows += [dict(chat_row(i), set="Test", label="TESTONLYLABEL", secret="S" * 30) for i in range(4)]
    rows += [dict(chat_row(i), kind="test", label="KINDLABEL") for i in range(2)]
    for r in rows[:6]:
        r["kind"] = "train"
    write_jsonl(tmp_path / "mixed.jsonl", rows)
    report = inspect_path(tmp_path)
    dumped = json.dumps(report)
    assert "TESTONLYLABEL" not in dumped and "KINDLABEL" not in dumped and "SSSS" not in dumped
    f = files_of(report)["mixed.jsonl"]
    assert f["rows"] == 12 and f["test_rows"] == 6
    assert f["split_columns"]["set"] == {"train": 6, "Test": 4} and f["split_columns"]["kind"] == {"train": 6,
                                                                                                    "test": 2}
    assert {v["value"] for v in f["fields"]["label"]["values"]} <= {"good", "bad"}
    assert "secret" not in f["fields"] and "TESTONLYLABEL" not in render(report)


@pytest.mark.parametrize("rel", ["test/part-0.jsonl", "testset.jsonl", "a/TestOOD/x.jsonl"])
def test_test_paths_are_counted_only(tmp_path, rel):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(p, [dict(chat_row(i), label="HIDDEN") for i in range(3)])
    f = inspect_path(tmp_path)["files"][0]
    assert f["rows"] == 3 and f["fields"] is None and f["samples"] == [] and "test" in f["note"]
    assert "HIDDEN" not in json.dumps(f)
