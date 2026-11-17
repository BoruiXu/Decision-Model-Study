import pytest
import yaml

from harness.config import ConfigError, get_field, load_experiment, load_machine


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(obj))
    return p


def test_get_field_dotted():
    assert get_field({"meta": {"f": 0.5}}, "meta.f") == 0.5
    with pytest.raises(KeyError):
        get_field({"meta": {"f": 0.5}}, "meta.g")
    with pytest.raises(KeyError):
        get_field({"meta": 1}, "meta.f")


def test_machine_defaults(tmp_path):
    p = _write(tmp_path, "m.yaml", {"python": "py", "model_dir": "/m", "out_root": "/o"})
    m = load_machine(p)
    assert m.gpu_source == "scan"
    assert m.lock_dir is None
    assert m.wait_timeout_s == 3600
    assert m.registry is None
    assert m.registry_path.endswith("registry.jsonl")
    assert m.registry_path.startswith("/o")
    assert m.gpu_exclude == [] and m.env == {} and m.forbidden_write_roots == []


def test_experiment_eval_sets(tmp_path):
    p = _write(tmp_path, "e.yaml", {
        "name": "x", "base": "b", "recipe": "r.yaml",
        "data": {"train": "t.jsonl",
                 "eval": {"v": {"train_labels": "v.jsonl"},
                          "w": {"train_labels": "w.jsonl", "reference": "wr.jsonl"}}},
        "fields": {"label": "target.yes"},
        "evaluator": {"budgets": [0.10, 0.15]},
        "subset": {"where": {"meta.pair_design": "full"}, "n": 10, "seed": 7},
    })
    e = load_experiment(p)
    assert e.data.eval["v"].reference is None
    assert e.data.eval["w"].reference == str(tmp_path / "wr.jsonl")
    assert e.evaluator.budgets == [0.10, 0.15]
    assert e.evaluator.type == "binary_decision" and e.evaluator.gate is True
    assert e.subset.n == 10 and e.subset.where == {"meta.pair_design": "full"}
    assert e.seeds == [42]
    assert e.merge is False and e.overrides == {}


def test_missing_key_named(tmp_path):
    p = _write(tmp_path, "e.yaml", {"name": "x", "base": "b", "recipe": "r", "data": {},
                                    "fields": {"label": "l"}})
    with pytest.raises(ConfigError, match="data.train"):
        load_experiment(p)
    p = _write(tmp_path, "m.yaml", {"python": "py", "out_root": "/o"})
    with pytest.raises(ConfigError, match="model_dir"):
        load_machine(p)


def _base(**over):
    d = {"name": "x", "base": "b", "recipe": "r",
         "data": {"train": "t.jsonl"}, "fields": {"label": "l"}}
    d.update(over)
    return d


@pytest.mark.parametrize("exp, key", [
    (_base(seed=[1]), "seed"),
    (_base(data={"train": "t", "train_sha": "ab"}), "data.train_sha"),
    (_base(data={"train": "t", "eval": {"v": {"train_labels": "a", "refrence": "b"}}}),
     "data.eval.v.refrence"),
    (_base(fields={"label": "l", "grp": "g"}), "fields.grp"),
    (_base(subset={"n": 1, "sed": 2}), "subset.sed"),
    (_base(evaluator={"budget": [0.1]}), "evaluator.budget"),
    (_base(evaluator=["binary_decision"]), "evaluator"),
    (_base(subset="full"), "subset"),
])
def test_unknown_or_malformed_key_is_config_error(tmp_path, exp, key):
    with pytest.raises(ConfigError, match=key):
        load_experiment(_write(tmp_path, "e.yaml", exp))


ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_machine_paths_expanded_and_absolute(tmp_path, monkeypatch):
    home, shared, cfg = tmp_path / "home", tmp_path / "shared", tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OJH_TEST_SHARED", str(shared))
    p = _write(cfg, "m.yaml", {"python": ".venv/bin/python", "model_dir": "~/models",
                               "out_root": "$OJH_TEST_SHARED/out", "registry": "reg/registry.jsonl",
                               "lock_dir": "~/locks", "forbidden_write_roots": ["${OJH_TEST_SHARED}/ro", "data"]})
    monkeypatch.chdir(tmp_path)  # the caller's cwd is not the machine file's directory
    m = load_machine(p)
    assert m.python == str(cfg / ".venv" / "bin" / "python")
    assert m.model_dir == str(home / "models") and m.lock_dir == str(home / "locks")
    assert m.out_root == str(shared / "out") and m.registry == str(cfg / "reg" / "registry.jsonl")
    assert m.forbidden_write_roots == [str(shared / "ro"), str(cfg / "data")]
    assert m.registry_path == str(cfg / "reg" / "registry.jsonl")
    # a bare interpreter name is looked up on PATH, not made relative to the file
    m = load_machine(_write(cfg, "m2.yaml", {"python": "python3", "model_dir": "/m", "out_root": "o"}))
    assert m.python == "python3" and m.out_root == str(cfg / "o")
    assert m.registry_path == str(cfg / "o" / "registry.jsonl")


def test_experiment_paths_relative_to_file(tmp_path, monkeypatch):
    cfg = tmp_path / "exps"
    cfg.mkdir()
    (cfg / "my_recipe.yaml").write_text("mode: full\n")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OJH_TEST_DATA", str(tmp_path / "data"))
    p = _write(cfg, "e.yaml", {
        "name": "x", "base": "b", "recipe": "my_recipe.yaml",
        "data": {"train": "$OJH_TEST_DATA/train.jsonl", "split_file": "../splits.json",
                 "eval": {"v": {"train_labels": "~/v.jsonl", "reference": "refs/v.jsonl"}}},
        "fields": {"label": "target.yes"}, "evaluator": {"type": "plugin", "path": "plug.py"}})
    monkeypatch.chdir(ROOT)
    e = load_experiment(p)
    assert e.recipe == str(cfg / "my_recipe.yaml")
    assert e.data.train == str(tmp_path / "data" / "train.jsonl")
    assert e.data.split_file == str(tmp_path / "splits.json")
    assert e.data.eval["v"].train_labels == str(tmp_path / "home" / "v.jsonl")
    assert e.data.eval["v"].reference == str(cfg / "refs" / "v.jsonl")
    assert e.evaluator.path == str(cfg / "plug.py")


def test_recipe_falls_back_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e = load_experiment(_write(tmp_path, "e.yaml", _base(recipe="examples/recipes/joey_full_1gpu.yaml")))
    assert e.recipe == str(ROOT / "examples" / "recipes" / "joey_full_1gpu.yaml")
    # a recipe that exists next to the experiment file wins over the repo root
    local = tmp_path / "examples" / "recipes"
    local.mkdir(parents=True)
    (local / "joey_full_1gpu.yaml").write_text("mode: lora\n")
    e = load_experiment(_write(tmp_path, "e.yaml", _base(recipe="examples/recipes/joey_full_1gpu.yaml")))
    assert e.recipe == str(local / "joey_full_1gpu.yaml")
    # neither exists: relative to the experiment file (the error names that path later)
    e = load_experiment(_write(tmp_path, "e.yaml", _base(recipe="nope.yaml")))
    assert e.recipe == str(tmp_path / "nope.yaml")
