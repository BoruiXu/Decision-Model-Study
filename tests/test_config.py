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
    assert e.data.eval["w"].reference == "wr.jsonl"
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
