import json
import os
import shlex
from pathlib import Path

import pytest
import yaml

from harness.config import (ConfigError, DataSpec, EvalSetSpec, ExperimentConfig, FieldsSpec,
                            MachineConfig)
from harness.resolve import (Infeasible, build_command, estimate_gib, eval_interval, parallel_for,
                             parse_plan_only, resolve)

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "examples" / "recipes"
V1_DIR = os.environ.get("HARNESS_V1_DIR")


def smi(n, free_mib=81920):
    return "".join(f"{i}, GPU-{i}, NVIDIA A100-SXM4-80GB, {free_mib}\n" for i in range(n))


def plan_ok(steps=176, ranks=4):
    return lambda lock: f"PLAN_OK\n{steps} steps for {ranks} ranks\n"


def fake_fp():
    return {"pip_freeze_sha256": "ab" * 32, "driver_version": "570.00"}


def setup(tmp_path, base="elephant-1", recipe="elephant_full_fsdp", n_gpus_free=4, **kw):
    import hashlib
    out = tmp_path / "out"
    d = tmp_path / "d"
    d.mkdir(parents=True)
    for name in ("train.jsonl", "val.jsonl", "val_ref.jsonl", "splits.json"):
        (d / name).write_text(f"{name}\n")
    prep = out / "exp" / "prepared"
    prep.mkdir(parents=True)
    for name in ("train.jsonl", "select.jsonl", "select_index.json"):
        (prep / name).write_text(f"{name}\n")
    (prep / "prepared.json").write_text(json.dumps({  # as prepare writes it for d/train.jsonl
        "train_sha256_source": hashlib.sha256((d / "train.jsonl").read_bytes()).hexdigest(),
        "train_sha256_prepared": "p" * 64, "subset": None, "counts": {}, "dropped": {}}))
    machine = MachineConfig(python="/venv/bin/python", model_dir=str(tmp_path / "models"),
                            out_root=str(out), env={"HF_HUB_OFFLINE": "1"})
    exp = ExperimentConfig(
        name="exp", base=base, recipe=str(RECIPES / f"{recipe}.yaml"),
        data=DataSpec(train=str(d / "train.jsonl"),
                      eval={"val": EvalSetSpec(train_labels=str(d / "val.jsonl"))}, split_file=str(d / "splits.json")),
        fields=FieldsSpec(label="meta.f"), **kw)
    return exp, machine


def run(exp, machine, n=4, free=81920, steps=176, **kw):
    return resolve(exp, machine, query=lambda uuid=None: smi(n, free),
                   plan_runner=plan_ok(steps, exp.gpus or 4), fingerprint=fake_fp, **kw)


def v1_flags(path):
    text = Path(path).read_text().replace("\\\n", " ")
    seg = text.rsplit("publisher_native_train.py\"", 1)[1].split(">>", 1)[0]
    return parse_flags(shlex.split(seg))


def parse_flags(tokens, skip=("--model", "--train", "--runs-dir", "--name", "--eval")):
    out, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        assert t.startswith("--"), t
        j = i + 1
        vals = []
        while j < len(tokens) and not tokens[j].startswith("--"):
            vals.append(tokens[j])
            j += 1
        i = j
        if t in skip:
            continue
        v = True
        if vals:
            v = vals[0] if len(vals) == 1 else vals
            try:
                v = float(v)
            except (TypeError, ValueError):
                pass
        out[t] = v
    return out


@pytest.mark.skipif(not V1_DIR, reason="HARNESS_V1_DIR not set")
def test_elephant_command_matches_v1(tmp_path):
    exp, machine = setup(tmp_path)
    lock = yaml.safe_load(run(exp, machine).read_text())
    cmd = build_command(lock, 42)
    assert cmd[:6] == ["/venv/bin/python", "-m", "torch.distributed.run", "--standalone",
                       "--nproc_per_node", "4"]
    assert cmd[6:8] == ["-m", "harness.train_entry"]
    mine = parse_flags(cmd[8:])
    assert mine == v1_flags(Path(V1_DIR) / "train" / "command_elephant.sh")
    assert mine["--eval-every"] == 22.0


def test_mode_required(tmp_path):
    exp, machine = setup(tmp_path)
    rec = tmp_path / "r.yaml"
    rec.write_text("gpus: 4\nonejev: {lr: 1.0e-5}\n")
    exp.recipe = str(rec)
    with pytest.raises(ConfigError, match="mode"):
        run(exp, machine)


def test_lora_fsdp_infeasible():
    with pytest.raises(Infeasible):
        parallel_for("lora", 9, 4, True)
    assert parallel_for("lora", 9, 4, None) == "ddp"
    assert parallel_for("lora", 9, 1, None) == "single"
    assert parallel_for("full", 9, 4, None) == "fsdp"
    assert parallel_for("full", 2, 4, None) == "ddp"
    assert parallel_for("full", 2, 1, None) == "single"


def test_mammoth_full_infeasible_lists_lora(tmp_path):
    exp, machine = setup(tmp_path, base="mammoth-1", recipe="elephant_full_fsdp")
    with pytest.raises(Infeasible) as e:
        run(exp, machine)
    alts = " ".join(e.value.alternatives)
    assert "lora" in alts.lower() and "GPUs" in alts and "smaller base" in alts


def test_lora_override_makes_mammoth_feasible(tmp_path):
    exp, machine = setup(tmp_path, base="mammoth-1", recipe="mammoth_lora_ddp")
    lock = yaml.safe_load(run(exp, machine).read_text())
    cmd = build_command(lock, 1)
    assert "--full-finetune" not in cmd and "--fsdp" not in cmd
    assert "--lora-autocast" in cmd and cmd[cmd.index("--lora-r") + 1] == "32"


def test_override_needs_reason(tmp_path):
    exp, machine = setup(tmp_path, overrides={"onejev.lr": {"value": 1e-6}})
    with pytest.raises(ConfigError, match="reason"):
        run(exp, machine)
    exp.overrides = {"onejev.lr": {"value": 1e-6, "reason": ""}}
    with pytest.raises(ConfigError, match="reason"):
        run(exp, machine)


def test_override_applied_and_recorded(tmp_path):
    ov = {"onejev.lr": {"value": 1e-6, "reason": "lower lr"},
          "onejev.disable_fla_kernels": {"value": True, "reason": "fla crash"}}
    exp, machine = setup(tmp_path, overrides=ov)
    lock = yaml.safe_load(run(exp, machine).read_text())
    assert lock["overrides"] == ov
    cmd = build_command(lock, 42)
    assert cmd[cmd.index("--lr") + 1] == "1e-06"
    assert "--disable-fla-kernels" in cmd


def test_lock_has_all_fields(tmp_path):
    exp, machine = setup(tmp_path, seeds=[42, 7])
    path = run(exp, machine)
    assert path.name == "run.lock.yaml" and path.parent.parent == Path(machine.out_root) / "exp"
    lock = yaml.safe_load(path.read_text())
    for k in ("run_id", "run_dir", "experiment", "repo_root", "machine", "mode", "lora", "flags",
              "n_gpus", "seeds", "parallel", "plan", "data", "eval_sets", "split_file", "fields",
              "evaluator", "merge", "base", "onejev", "fingerprint", "overrides", "estimate_gib"):
        assert k in lock, k
    assert lock["mode"] == "full" and lock["parallel"] == "fsdp" and lock["n_gpus"] == 4
    assert lock["seeds"] == [42, 7] and lock["merge"] is False
    assert lock["plan"] == {"steps": 176, "ranks": 4, "interval": 22}
    assert lock["flags"]["eval_every"] == 22 and lock["flags"]["save_every"] == 22
    assert lock["data"]["sha256"]["train.jsonl"] and lock["base"]["revision"]
    assert lock["data"]["adapter"] == "onejev_rows"
    assert lock["onejev"]["patch"]["sha256"] and lock["fingerprint"]["driver_version"] == "570.00"
    assert lock["machine"]["python"] == "/venv/bin/python"
    assert lock["eval_sets"] == {"val": {"train_labels": str(tmp_path / "d" / "val.jsonl"), "reference": None,
                                         "adapter_args": {}}}
    assert lock["run_id"] in str(path)
    cmd = build_command(lock, 7)
    assert cmd[cmd.index("--name") + 1] == "exp-s7" and cmd[cmd.index("--seed") + 1] == "7"
    assert cmd[cmd.index("--runs-dir") + 1] == str(path.parent / "runs")


def test_single_gpu_command_has_no_torchrun(tmp_path):
    exp, machine = setup(tmp_path, base="joey-1", recipe="joey_full_1gpu")
    lock = yaml.safe_load(resolve(exp, machine, query=lambda uuid=None: smi(1),
                                  plan_runner=plan_ok(80, 1), fingerprint=fake_fp).read_text())
    cmd = build_command(lock, 42)
    assert cmd[:3] == ["/venv/bin/python", "-m", "harness.train_entry"]
    assert "--full-finetune" in cmd and "--fsdp" not in cmd and lock["plan"]["interval"] == 10


def test_exp_gpus_overrides_recipe(tmp_path):
    exp, machine = setup(tmp_path, gpus=2)
    lock = yaml.safe_load(run(exp, machine).read_text())
    assert lock["n_gpus"] == 2


def test_not_enough_cards_and_preassigned(tmp_path, monkeypatch):
    exp, machine = setup(tmp_path)
    with pytest.raises(Infeasible):
        run(exp, machine, n=2)
    machine.gpu_source = "preassigned"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-1,GPU-2,GPU-3")
    with pytest.raises(Infeasible):  # 36 GiB/card needed, 30 GiB free
        run(exp, machine, free=30 * 1024)
    assert run(exp, machine).exists()


def test_estimate_and_plan_parsing():
    assert estimate_gib(9, "full", 4, True) == 36
    assert estimate_gib(9, "full", 4, False) == 144
    assert estimate_gib(27, "lora", 4, False) == pytest.approx(62.1)
    assert parse_plan_only("PLAN_OK\n176 steps for 4 ranks\n") == 176
    assert eval_interval(176) == 22 and eval_interval(177) == 23
    with pytest.raises(ValueError):
        parse_plan_only("176 steps for 4 ranks")


def test_missing_prepared_dir(tmp_path):
    exp, machine = setup(tmp_path)
    exp.name = "other"
    with pytest.raises(ConfigError, match="prepare"):
        run(exp, machine)


ONEJEV_DIR = os.environ.get("ONEJEV_DIR")


@pytest.mark.skipif(not ONEJEV_DIR, reason="ONEJEV_DIR not set")
def test_emitted_flags_exist_in_onejev(tmp_path):
    import re
    src = (Path(ONEJEV_DIR) / "train" / "sft.py").read_text()
    known = set()
    for call in re.findall(r"add_argument\((.*?)\)\s*\n", src, re.S):
        known.update(re.findall(r"\"(--[a-z0-9-]+)\"", call))
    assert "--lora-r" in known and "--target-modules" in known
    for recipe, base in (("elephant_full_fsdp", "elephant-1"), ("joey_full_1gpu", "joey-1"),
                         ("joey_lora_1gpu", "joey-1"), ("mammoth_lora_ddp", "mammoth-1")):
        exp, machine = setup(tmp_path / recipe, base=base, recipe=recipe)
        exp.overrides = {"lora.target_modules": {"value": ["q_proj", "v_proj"], "reason": "t"}} \
            if recipe == "joey_lora_1gpu" else {}
        lock = yaml.safe_load(resolve(exp, machine, query=lambda uuid=None: smi(4, 10**6),
                                      plan_runner=plan_ok(80, exp.gpus or yaml.safe_load(
                                          (RECIPES / f"{recipe}.yaml").read_text())["gpus"]),
                                      fingerprint=fake_fp).read_text())
        cmd = build_command(lock, 1)
        emitted = {t for t in cmd if t.startswith("--")} - {"--standalone", "--nproc_per_node"}
        assert emitted <= known, emitted - known
        if recipe == "joey_lora_1gpu":
            assert cmd[cmd.index("--target-modules") + 1] == "q_proj,v_proj"


def test_reserved_flag_in_recipe_rejected(tmp_path):
    exp, machine = setup(tmp_path)
    rec = tmp_path / "r.yaml"
    rec.write_text("mode: lora\ngpus: 1\nonejev: {full_finetune: true}\n")
    exp.recipe = str(rec)
    with pytest.raises(ConfigError, match="full_finetune"):
        run(exp, machine)


@pytest.mark.parametrize("key", ["onejev.eval_every", "onejev.fsdp", "onejev.full_finetune", "onejev.seed",
                                 "onejev.config", "onejev.plan_only", "onejev.device", "onejev.merge",
                                 "onejev.merged_dir", "onejev.worst_case", "onejev.dump_first_grad",
                                 "onejev.eval_limit", "onejev.dist_backend"])
def test_reserved_override_rejected(tmp_path, key):
    exp, machine = setup(tmp_path, overrides={key: {"value": 5, "reason": "x"}})
    with pytest.raises(ConfigError, match="set by the harness"):
        run(exp, machine)


def test_max_steps_override_allowed(tmp_path):
    exp, machine = setup(tmp_path, overrides={"onejev.max_steps": {"value": 10, "reason": "smoke"}})
    lock = yaml.safe_load(run(exp, machine).read_text())
    cmd = build_command(lock, 1)
    assert cmd[cmd.index("--max-steps") + 1] == "10"


def test_lora_override_on_full_recipe_rejected(tmp_path):
    exp, machine = setup(tmp_path, overrides={"lora.r": {"value": 8, "reason": "x"}})
    with pytest.raises(ConfigError, match="lora"):
        run(exp, machine)


def test_preassigned_too_few_devices(tmp_path, monkeypatch):
    exp, machine = setup(tmp_path)
    machine.gpu_source = "preassigned"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-1")
    with pytest.raises(Infeasible, match="need 4"):
        run(exp, machine)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(Infeasible, match="need 4"):
        run(exp, machine)


def test_preassigned_unmatched_id_records_unchecked(tmp_path, monkeypatch):
    exp, machine = setup(tmp_path)
    machine.gpu_source = "preassigned"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-1,GPU-2,GPU-9")
    lock = yaml.safe_load(run(exp, machine).read_text())
    assert lock["feasibility"]["checked"] is False and "GPU-9" in lock["feasibility"]["reason"]


def test_feasibility_recorded_when_checked(tmp_path):
    exp, machine = setup(tmp_path)
    lock = yaml.safe_load(run(exp, machine).read_text())
    assert lock["feasibility"]["checked"] is True and lock["feasibility"]["min_free_gib"] == 80.0


@pytest.mark.parametrize("what", ["out_root", "registry", "lock_dir"])
def test_resolve_refuses_forbidden_write_roots(tmp_path, what):
    exp, machine = setup(tmp_path)
    ro = tmp_path / "ro"
    if what == "out_root":
        machine.forbidden_write_roots = [str(tmp_path / "out")]
    elif what == "registry":
        machine.registry, machine.forbidden_write_roots = str(ro / "registry.jsonl"), [str(ro)]
    else:
        machine.lock_dir, machine.forbidden_write_roots = str(ro / "locks"), [str(ro)]
    with pytest.raises(ConfigError, match="forbidden_write_roots"):
        run(exp, machine)
    assert not [p for p in (tmp_path / "out" / "exp").iterdir() if p.name != "prepared"]


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_lock_records_eval_and_split_sha256(tmp_path):
    exp, machine = setup(tmp_path)
    d = tmp_path / "d"
    exp.data.eval["ood"] = EvalSetSpec(train_labels=str(d / "val.jsonl"), reference=str(d / "val_ref.jsonl"))
    lock = yaml.safe_load(run(exp, machine).read_text())
    assert lock["data"]["eval_sha256"] == {
        "val": {"train_labels": _sha(d / "val.jsonl"), "reference": None},
        "ood": {"train_labels": _sha(d / "val.jsonl"), "reference": _sha(d / "val_ref.jsonl")}}
    assert lock["data"]["split_file_sha256"] == _sha(d / "splits.json")
    exp.data.split_file = None
    assert yaml.safe_load(run(exp, machine).read_text())["data"]["split_file_sha256"] is None
    (d / "val_ref.jsonl").unlink()
    with pytest.raises(ConfigError, match="val_ref.jsonl"):
        run(exp, machine)


def test_merge_with_full_mode_rejected(tmp_path):
    exp, machine = setup(tmp_path, merge=True)
    with pytest.raises(ConfigError, match="merge"):
        run(exp, machine)
    exp, machine = setup(tmp_path / "lora", base="joey-1", recipe="joey_lora_1gpu", merge=True)
    lock = yaml.safe_load(resolve(exp, machine, query=lambda uuid=None: smi(1), plan_runner=plan_ok(80, 1),
                                  fingerprint=fake_fp).read_text())
    assert lock["merge"] is True and lock["mode"] == "lora"


def test_plan_only_runs_in_the_repo(tmp_path, monkeypatch):
    import subprocess
    from harness import resolve as resolve_mod
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0, "PLAN_OK\n8 steps for 1 ranks\n", "")
    monkeypatch.setattr(resolve_mod.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)  # a local train/ package here must not shadow OneJev's
    lock = {"machine": {"python": "/venv/bin/python", "env": {}, "model_dir": "/m"}, "n_gpus": 1,
            "repo_root": "/repo", "flags": {}, "seeds": [1], "experiment": "e", "run_dir": "/r",
            "base": {"name": "b"}, "data": {"select": "/s", "train": "/t"}}
    resolve_mod._default_plan_runner(lock)
    assert seen["cwd"] == "/repo" and seen["cmd"][1:3] == ["-m", "train.sft"]


def _two_experiments(tmp_path):
    from harness.config import load_experiment
    from harness.prepare import prepare
    def row(i, split, y):
        return {"id": i, "split": split, "state": {"messages": [{"role": "user", "content": i}]},
                "question": {"type": "noul", "instructions": "?", "criteria": {"true": "a", "false": "b"}},
                "target": {"yes": y, "no": 1 - y}, "meta": {}}
    for d, y in (("a", 1.0), ("b", 0.0)):
        (tmp_path / d).mkdir()
        (tmp_path / d / "train.jsonl").write_text("".join(json.dumps(row(f"{d}t{i}", "train", y)) + "\n"
                                                          for i in range(3)))
        (tmp_path / d / "val.jsonl").write_text(json.dumps(row(f"{d}v", "val", y)) + "\n")
        (tmp_path / d / "exp.yaml").write_text(yaml.safe_dump({
            "name": "same", "base": "joey-1", "recipe": str(RECIPES / "joey_full_1gpu.yaml"), "gpus": 1,
            "data": {"train": "train.jsonl", "eval": {"v": {"train_labels": "val.jsonl"}}},
            "fields": {"label": "target.yes"}}))
    machine = MachineConfig(python="/venv/bin/python", model_dir=str(tmp_path / "models"),
                            out_root=str(tmp_path / "out"))
    a, b = load_experiment(tmp_path / "a" / "exp.yaml"), load_experiment(tmp_path / "b" / "exp.yaml")
    prepare(a, tmp_path / "out" / "same" / "prepared", source=str(tmp_path / "a" / "exp.yaml"))
    return a, b, machine


def _resolve(exp, machine, **kw):
    return resolve(exp, machine, query=lambda uuid=None: smi(1), plan_runner=plan_ok(3, 1), fingerprint=fake_fp, **kw)


def test_resolve_refuses_data_prepared_from_another_experiment(tmp_path):
    a, b, machine = _two_experiments(tmp_path)
    with pytest.raises(ConfigError) as e:
        _resolve(b, machine, source=str(tmp_path / "b" / "exp.yaml"))
    msg = str(e.value)
    assert str(tmp_path / "a" / "exp.yaml") in msg and "prepare" in msg and "--force" in msg
    lock = yaml.safe_load(_resolve(a, machine, source=str(tmp_path / "a" / "exp.yaml")).read_text())
    assert lock["data"]["source"] == str(tmp_path / "a" / "train.jsonl")


def test_resolve_refuses_a_train_file_that_is_not_the_prepared_one(tmp_path):
    a, b, machine = _two_experiments(tmp_path)
    with pytest.raises(ConfigError, match="train"):  # no experiment path given: the train file still differs
        _resolve(b, machine)
    (tmp_path / "a" / "train.jsonl").write_text((tmp_path / "a" / "train.jsonl").read_text() + "\n")
    with pytest.raises(ConfigError, match="changed since prepare"):
        _resolve(a, machine, source=str(tmp_path / "a" / "exp.yaml"))
