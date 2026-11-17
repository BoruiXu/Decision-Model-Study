import json
import re
from pathlib import Path

import pytest
import yaml

from harness.config import load_experiment, load_machine
from harness.resolve import resolve

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
RECIPES = sorted((EXAMPLES / "recipes").glob("*.yaml"))
BASE_OF = {"elephant_full_fsdp": "elephant-1", "joey_full_1gpu": "joey-1",
           "joey_lora_1gpu": "joey-1", "mammoth_lora_ddp": "mammoth-1"}
V1_SHA = "fe83846076cb6c72d80a048548f38edae751bb9df9865f11409992289c3515f2"
CARD_MIB = 81920  # 80 GiB: enough for every example recipe (mammoth LoRA needs ~62 GiB)


def fill(src: Path, dst: Path, tmp: Path) -> Path:
    """Copy a template, replacing every <PLACEHOLDER> with a path below tmp."""
    text = re.sub(r"<([A-Z_]+)>", lambda m: str(tmp / m.group(1).lower()), src.read_text())
    dst.write_text(text)
    return dst


def prepared(out_root: Path, name: str) -> None:
    prep = out_root / name / "prepared"
    prep.mkdir(parents=True)
    for f in ("train.jsonl", "select.jsonl", "select_index.json"):
        (prep / f).write_text(f"{f}\n")
    (prep / "prepared.json").write_text(json.dumps({
        "train_sha256_source": V1_SHA, "train_sha256_prepared": "p" * 64,
        "subset": None, "counts": {}, "dropped": {}}))


def resolve_with(exp, machine, n_gpus):
    return resolve(
        exp, machine,
        query=lambda uuid=None: "".join(
            f"{i}, GPU-{i}, NVIDIA A100-SXM4-80GB, {CARD_MIB}\n" for i in range(8)),
        plan_runner=lambda lock: f"PLAN_OK\n176 steps for {n_gpus} ranks\n",
        fingerprint=lambda: {"pip_freeze_sha256": "ab" * 32, "driver_version": "570.00"})


def test_examples_load(tmp_path):
    m = load_machine(fill(EXAMPLES / "machine.example.yaml", tmp_path / "m.yaml", tmp_path))
    assert m.gpu_source == "scan" and m.order == "free_desc"
    exp = load_experiment(fill(EXAMPLES / "dynamicserve_v1_repro.yaml", tmp_path / "e.yaml", tmp_path))
    assert exp.data.train_sha256 == V1_SHA
    assert exp.recipe == "examples/recipes/elephant_full_fsdp.yaml"
    assert exp.seeds == [42] and exp.base == "elephant-1"


def test_machine_example_lists_every_key():
    text = (EXAMPLES / "machine.example.yaml").read_text()
    from harness.config import MachineConfig
    for key in MachineConfig.__dataclass_fields__:
        assert re.search(rf"^#?\s*{key}:", text, re.M), key


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_examples_resolve(tmp_path, recipe, monkeypatch):
    monkeypatch.chdir(ROOT)  # the v1 example names its recipe relative to the repo root
    machine = load_machine(fill(EXAMPLES / "machine.example.yaml", tmp_path / "m.yaml", tmp_path))
    exp = load_experiment(fill(EXAMPLES / "dynamicserve_v1_repro.yaml", tmp_path / "e.yaml", tmp_path))
    prepared(Path(machine.out_root), exp.name)
    if recipe.stem != "elephant_full_fsdp":
        exp.recipe, exp.base, exp.gpus = str(recipe), BASE_OF[recipe.stem], None
    n = exp.gpus or yaml.safe_load(recipe.read_text())["gpus"]
    lock = yaml.safe_load(resolve_with(exp, machine, n).read_text())
    assert lock["mode"] == yaml.safe_load(recipe.read_text())["mode"]
    assert lock["base"]["name"] == BASE_OF[recipe.stem]
    assert lock["plan"]["steps"] == 176 and lock["seeds"] == [42]


def test_skill_symlinks_resolve():
    for name in ("training-with-onejev", "adapting-data-for-onejev"):
        link = ROOT / ".claude" / "skills" / name
        assert link.is_symlink()
        assert not Path(link.readlink()).is_absolute()
        assert (link / "SKILL.md").is_file()
        assert (ROOT / "skills" / name / "SKILL.md").samefile(link / "SKILL.md")
        head = (link / "SKILL.md").read_text().split("---")[1]
        assert yaml.safe_load(head)["name"] == name
