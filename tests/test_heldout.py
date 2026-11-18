from pathlib import Path

from harness.heldout import is_test_path, row_is_test, split_values


def test_split_like_keys_any_case_and_depth():
    assert row_is_test({"split": "test"})
    assert row_is_test({"Set": "Test_OOD"})
    assert row_is_test({"meta": {"partition": "testing"}})
    assert row_is_test({"FOLD": " test"})
    assert not row_is_test({"split": "train", "set": "val", "note": "test"})  # note is not split-like
    assert not row_is_test({"split": None, "subset": 3})
    assert not row_is_test({"items": [{"split": "test"}]})  # list items are not columns
    assert row_is_test({"kind": "test"}, extra_paths=["kind"])


def test_split_values_lists_split_like_columns():
    assert dict(split_values({"split": "a", "meta": {"set": "b"}, "x": "c"})) == {"split": "a", "meta.set": "b"}


def test_test_paths(tmp_path):
    root = tmp_path / "data"
    assert is_test_path(root / "test" / "part-0.jsonl", root)
    assert is_test_path(root / "testset.jsonl", root)
    assert is_test_path(root / "a" / "Test_ood.jsonl", root)
    assert not is_test_path(root / "train" / "contest.jsonl", root)
    assert not is_test_path(root / "latest.jsonl", root)
    assert is_test_path(tmp_path / "test" / "x.jsonl", tmp_path / "test")  # the root's own name counts
    # without a root: the file name and its directory
    assert is_test_path(Path("/x/test/part-0.jsonl")) and is_test_path(Path("/x/testset.jsonl"))
    assert not is_test_path(Path("/testing/home/data/train.jsonl"))
