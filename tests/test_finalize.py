import hashlib
import json
import os
import random
import shutil
import statistics
import sys
from pathlib import Path

import pytest
import yaml

from harness import registry
from harness.config import (DataSpec, EvalSetSpec, EvaluatorSpec, ExperimentConfig, FieldsSpec,
                            MachineConfig)
from harness.eval_records import nll
from harness.evaluator import binary_decision
from harness.finalize import finalize
from harness.prepare import _sha256, prepare
from harness.resolve import resolve

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "examples" / "recipes"
WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]
STEPS = [0, 4, 8, 10]  # saves at 4 and 8; step 10 is the end of training (final/ or adapter/)
NOISE = {0: 0.45, 4: 0.25, 8: 0.05, 10: 0.15}  # step 8 has the lowest select NLL
FILES = ["predictions_select.jsonl", "thresholds.json", "comparison.json", "timing.json", "checkpoint.sha256",
         "dropped.json",
         "deletion_candidates.json", "RUN_CARD.md", "status"]


def mk(rid, rng, split, group):
    rate = rng.random()
    return {"id": rid, "split": split,
            "state": {"messages": [{"role": "user", "content": " ".join(rng.choices(WORDS, k=6))}]},
            "question": {"type": "noul", "instructions": "agree?", "criteria": {"yes": "y", "no": "n"}},
            "target": {"yes": rate, "no": 1 - rate},
            "meta": {"f": min(1.0, 0.5 + rate / 2), "cat": group, "label_version": "lv1"}}


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_run(tmp_path, recipe="joey_full_1gpu", seeds=(42,), merge=False, evaluator=None, overrides=None,
             drop=False, allow_test=False, extra_split=False):
    rng = random.Random(0)
    src = tmp_path / "src"
    src.mkdir()
    train_rows = [mk(f"t{i}", rng, "train", f"g{i % 3}") for i in range(80)]
    if extra_split:  # rows OneJev skips (--split train)
        x_rng = random.Random(1)
        train_rows += [mk(f"x{i}", x_rng, "other", "g3") for i in range(6)]
    train = write_jsonl(src / "train.jsonl", train_rows)
    evals, assign = {}, {}
    for name in ("a", "b"):
        rows = [mk(f"{name}{i}", rng, "test_id" if allow_test and name == "a" else "val", f"g{i % 4}")
                for i in range(30)]
        evals[name] = EvalSetSpec(train_labels=write_jsonl(src / f"{name}.jsonl", rows))
        assign[name] = {"assignment": {r["id"]: ("select" if i % 2 == 0 else "report") for i, r in enumerate(rows)}}
        if drop and name == "a":  # a1 gets no assignment
            del assign[name]["assignment"]["a1"]
        if drop and name == "b":  # the reference lacks b2, b4's reference has no group, b6 is a hard negative
            ref = [dict(r, meta={k: v for k, v in r["meta"].items() if k != "cat"}) if r["id"] == "b4"
                   else dict(r, target={"no": 1.0}) if r["id"] == "b6" else r for r in rows if r["id"] != "b2"]
            evals[name].reference = write_jsonl(src / "b_ref.jsonl", ref)
    split = src / "splits.json"
    split.write_text(json.dumps(assign))
    exp = ExperimentConfig(
        name="exp", base="joey-1", recipe=str(RECIPES / f"{recipe}.yaml"),
        data=DataSpec(train=train, train_sha256=_sha256(train), eval=evals, split_file=str(split)),
        fields=FieldsSpec(label="target.yes", group="meta.cat", consistency="meta.f", text="messages"),
        evaluator=evaluator or EvaluatorSpec(), seeds=list(seeds), merge=merge, overrides=overrides or {})
    out = tmp_path / "out"
    prepare(exp, out / "exp" / "prepared", allow_test=allow_test)
    machine = MachineConfig(python="/venv/bin/python", model_dir=str(tmp_path / "models"), out_root=str(out))
    lock_path = resolve(exp, machine, query=lambda uuid=None: "0, GPU-0, NVIDIA A100-SXM4-80GB, 81920\n",
                        plan_runner=lambda lock: "PLAN_OK\n32 steps for 1 ranks\n",
                        fingerprint=lambda: {"pip_freeze_sha256": "ab" * 32, "driver_version": "570.00"})
    lock = yaml.safe_load(lock_path.read_text())
    run_dir = lock_path.parent
    select_rows = read_jsonl(lock["data"]["select"])
    timing = {"n_gpus": 1, "seeds": {}}
    for seed in seeds:
        populate(run_dir, seed, select_rows, lock["mode"])
        timing["seeds"][str(seed)] = {"start": "s", "end": "e", "seconds": 1800.0, "exit_code": 0}
    (run_dir / "timing.json").write_text(json.dumps(timing))
    (run_dir / "gpus.json").write_text(json.dumps([{"uuid": "GPU-0", "name": "NVIDIA A100-SXM4-80GB"}]))
    return run_dir, lock


def populate(run_dir, seed, select_rows, mode):
    seed_dir = run_dir / "runs" / f"exp-s{seed}"
    (seed_dir / "preds").mkdir(parents=True)
    rng = random.Random(seed)
    log = []
    for step in STEPS:
        recs = []
        for r in sorted(select_rows, key=lambda r: r["id"]):
            t = r["target"]["yes"]
            p = min(0.99, max(0.01, t + rng.uniform(-1, 1) * NOISE[step]))
            recs.append({"id": r["id"], "labels": ["yes", "no"], "probs": [p, 1 - p], "target": [t, 1 - t]})
        path = seed_dir / "preds" / f"select-step{step}.jsonl"
        write_jsonl(path, recs)
        if step:
            log.append({"step": step, "loss": 0.5, "sec_per_step": 1.0})
        log.append({"event": "harness_eval", "step": step, "set": "select", "n": len(recs), "nll": nll(recs),
                    "path": str(path)})
    log.append({"event": "done", "steps": STEPS[-1]})
    write_jsonl(seed_dir / "log.jsonl", log)
    prefix = "ckpt-step" if mode == "full" else "adapter-step"
    for name in (f"{prefix}4", f"{prefix}8", "final" if mode == "full" else "adapter"):
        d = seed_dir / name
        d.mkdir()
        (d / "model.safetensors").write_bytes(name.encode() * 50)
        (d / "readout.json").write_text('{"readout": "slot_lm"}')


class FakeScorer:
    """Returns the training-time per-row probabilities of the step a checkpoint dir belongs to."""

    def __init__(self, shift=0.0, merged_shift=0.0):
        self.shift, self.merged_shift, self.calls = shift, merged_shift, []

    def __call__(self, model_dir, rows, *, base_dir=None, multimodal, **kw):
        model_dir = Path(model_dir)
        self.calls.append({"model_dir": model_dir, "base_dir": base_dir, "multimodal": multimodal,
                           "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), **kw})
        seed_dir, name = model_dir.parent, model_dir.name
        if "step" in name:
            step = int(name.rsplit("step", 1)[1])
        else:
            step = max(e["step"] for e in read_jsonl(seed_dir / "log.jsonl") if e.get("event") == "harness_eval")
        shift = self.merged_shift if name.startswith("merged") else self.shift
        out = {}
        for r in read_jsonl(seed_dir / "preds" / f"select-step{step}.jsonl"):
            p = min(1.0, max(0.0, r["probs"][0] + shift))
            out[r["id"]] = {**r, "probs": [p, 1 - p]}
        assert set(out) == {r["id"] for r in rows}
        return out


class FakeMerger:
    def __init__(self):
        self.calls, self.kw = [], []

    def __call__(self, base_dir, adapter_dir, out_dir, **kw):
        self.calls.append((Path(base_dir), Path(adapter_dir), Path(out_dir)))
        self.kw.append(kw)
        out = Path(out_dir)
        out.mkdir(parents=True)
        (out / "model.safetensors").write_bytes(b"merged" * 100)
        (out / "config.json").write_text("{}")
        return out


def no_merger(*a, **k):
    raise AssertionError("merger must not be called")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry_path(lock):
    return Path(lock["machine"]["registry"])


def test_finalize_writes_all_files(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run_dir, lock = make_run(tmp_path)
    scorer = FakeScorer()
    out = finalize(run_dir, scorer=scorer, merger=no_merger)
    assert out["status"] == "done", out["failures"]
    assert (run_dir / "status").read_text().strip() == "done"
    for f in FILES:
        assert (run_dir / f).is_file(), f
    seed_dir = run_dir / "runs" / "exp-s42"

    # selected step 8 was reloaded from its own checkpoint dir on the leased card
    (call,) = scorer.calls
    assert call["model_dir"] == seed_dir / "ckpt-step8" and call["base_dir"] is None
    assert call["multimodal"] is True and call["autocast"] is True and call["cuda"] == "GPU-0"
    # OneJev loads a full fine-tune without FSDP in float32 (sft.py load_dtype); the reload follows it
    assert call["dtype"] == "float32" and call["disable_fla_kernels"] is False
    assert "CUDA_VISIBLE_DEVICES" not in os.environ

    preds = read_jsonl(run_dir / "predictions_select.jsonl")
    assert len(preds) == 30 and {p["seed"] for p in preds} == {42} and {p["step"] for p in preds} == {8}
    assert all(p["positive"] == p["probs"][p["labels"].index("yes")] for p in preds)

    thr = json.loads((run_dir / "thresholds.json").read_text())
    assert set(thr["42"]) == {"best_jaccard", "budget_10", "budget_15"}
    comp = json.loads((run_dir / "comparison.json").read_text())["42"]
    assert set(comp["sets"]) == {"a", "b", "pooled_select"}
    pooled = comp["sets"]["pooled_select"]
    assert {"router", "tfidf", "group_prior", "all_positive", "oracle"} <= set(pooled)
    assert comp["gate"]["passed"] is True and comp["selected_step"] == 8
    assert comp["rescore"]["abs_diff"] < 1e-9

    timing = json.loads((run_dir / "timing.json").read_text())
    assert timing["gpu_hours"] == pytest.approx(0.5) and timing["seeds"]["42"]["seconds"] == 1800.0

    manifest = (run_dir / "checkpoint.sha256").read_text().splitlines()
    entries = dict(reversed(line.split("  ", 1)) for line in manifest if not line.startswith("#"))
    for f in ("model.safetensors", "readout.json"):
        rel = f"runs/exp-s42/ckpt-step8/{f}"
        assert entries[rel] == sha(run_dir / rel)
    assert not any("ckpt-step4" in k for k in entries)

    cands = json.loads((run_dir / "deletion_candidates.json").read_text())
    paths = {Path(c["path"]).name: c for c in cands["candidates"]}
    assert set(paths) == {"ckpt-step4", "final"}
    assert all(c["bytes"] > 0 and c["seed"] == 42 for c in paths.values())
    assert cands["total_bytes"] == sum(c["bytes"] for c in cands["candidates"])
    assert (seed_dir / "ckpt-step4").is_dir()  # nothing is deleted

    card = (run_dir / "RUN_CARD.md").read_text()
    for s in (lock["run_id"], "exp", "joey-1", "pooled_select", "tfidf", "best_jaccard", "ckpt-step8", "done",
              "## "):
        assert s in card, s

    (rec,) = registry.read(registry_path(lock))
    assert set(registry.REQUIRED) <= set(rec)
    assert rec["seed"] == 42 and rec["selected_step"] == 8 and rec["run_id"] == lock["run_id"]
    assert rec["conclusion"] == "" and rec["complete"] is False and rec["status"] == "done"
    assert rec["train_label_version"] == "lv1" and rec["reference_label_version"] == "lv1"
    assert rec["data_sha256"] == lock["data"]["sha256"]["train.jsonl"]
    assert rec["base"] == "joey-1" and rec["base_revision"] == lock["base"]["revision"]
    assert rec["mode"] == "full" and rec["n_gpus"] == 1 and rec["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    assert rec["gpu_hours"] == pytest.approx(0.5) and len(rec["checkpoint_sha256"]) == 64
    assert rec["metrics"]["auc"] == pooled["router"]["auc"] and rec["baselines"]["tfidf"] == pooled["tfidf"]
    assert rec["gate"]["passed"] is True and rec["onejev"]["revision"] == lock["onejev"]["revision"]
    assert lock["run_id"] not in registry.table(registry_path(lock))
    registry.note(registry_path(lock), lock["run_id"], "beats tfidf and group prior on a and b")
    assert registry.read(registry_path(lock))[0]["complete"] is True


def test_finalize_nll_mismatch_fails(tmp_path):
    run_dir, lock = make_run(tmp_path)
    out = finalize(run_dir, scorer=FakeScorer(shift=0.05), merger=no_merger)
    assert out["status"] == "failed" and any("NLL" in f for f in out["failures"])
    assert (run_dir / "status").read_text().strip() == "failed"
    (rec,) = registry.read(registry_path(lock))
    assert rec["status"] == "failed" and rec["failures"]
    assert "failed" in (run_dir / "RUN_CARD.md").read_text()


@pytest.mark.parametrize("merged_shift,status", [(0.0, "done"), (0.01, "failed")])
def test_finalize_lora_merge_compared(tmp_path, merged_shift, status):
    run_dir, lock = make_run(tmp_path, recipe="joey_lora_1gpu", merge=True)
    scorer, merger = FakeScorer(merged_shift=merged_shift), FakeMerger()
    out = finalize(run_dir, scorer=scorer, merger=merger)
    assert out["status"] == status, out["failures"]
    seed_dir = run_dir / "runs" / "exp-s42"
    base_dir = Path(lock["machine"]["model_dir"]) / "joey-1"
    assert merger.calls == [(base_dir, seed_dir / "adapter-step8", seed_dir / "merged-step8")]
    first, second = scorer.calls
    assert first["model_dir"] == seed_dir / "adapter-step8" and Path(first["base_dir"]) == base_dir
    assert first["dtype"] == second["dtype"] == merger.kw[0]["dtype"] == "bfloat16"
    assert merger.kw[0]["disable_fla_kernels"] is False
    assert second["model_dir"] == seed_dir / "merged-step8" and second["base_dir"] is None
    if status == "failed":
        assert any("merge" in f for f in out["failures"])
    comp = json.loads((run_dir / "comparison.json").read_text())["42"]
    assert comp["merge"]["max_prob_diff"] == pytest.approx(merged_shift, abs=1e-9)
    manifest = (run_dir / "checkpoint.sha256").read_text()
    assert lock["base"]["revision"] in manifest and "runs/exp-s42/merged-step8/model.safetensors" in manifest
    cands = json.loads((run_dir / "deletion_candidates.json").read_text())["candidates"]
    kinds = {Path(c["path"]).name: c["kind"] for c in cands}
    if status == "done":
        assert kinds == {"adapter-step4": "checkpoint", "adapter": "checkpoint", "merged-step8": "merged"}
    else:  # a failed run offers nothing for deletion
        assert kinds == {}
    (rec,) = registry.read(registry_path(lock))
    assert rec["mode"] == "lora" and rec["base_revision"] == lock["base"]["revision"]
    assert len(rec["merged_sha256"]) == 64 and rec["merged_sha256"] != rec["checkpoint_sha256"]


def test_multi_seed_summary(tmp_path):
    run_dir, lock = make_run(tmp_path, seeds=(42, 7))
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "done", out["failures"]
    recs = registry.read(registry_path(lock))
    assert [r["seed"] for r in recs] == [42, 7, "summary"]
    summary = recs[-1]
    assert set(registry.REQUIRED) <= set(summary)
    aucs = [r["metrics"]["auc"] for r in recs[:2]]
    assert summary["metrics"]["auc"]["mean"] == pytest.approx(statistics.fmean(aucs))
    assert summary["metrics"]["auc"]["std"] == pytest.approx(statistics.pstdev(aucs))
    assert summary["metrics"]["selected_steps"] == [8, 8] and summary["selected_step"] == [8, 8]
    assert summary["gpu_hours"] == pytest.approx(1.0) and summary["gate"]["passed"] is True
    assert json.loads((run_dir / "timing.json").read_text())["gpu_hours"] == pytest.approx(1.0)
    preds = read_jsonl(run_dir / "predictions_select.jsonl")
    assert len(preds) == 60 and {p["seed"] for p in preds} == {42, 7}
    assert set(json.loads((run_dir / "comparison.json").read_text())) == {"42", "7"}
    card = (run_dir / "RUN_CARD.md").read_text()
    assert "42" in card and "7" in card


PLUGIN = '''
CALLS = []


def validate(rows):
    CALLS.append(len(rows))
    return []


def select_rule(events):
    return max(e["step"] for e in events if e.get("set") == "select")


def calibrate(preds, rows):
    return {"cut": 0.5, "n": len(rows)}


def baselines(train_rows, eval_rows):
    return {"constant": {"n": len(eval_rows)}}


def acceptance(preds, rows):
    assert all(isinstance(p["probs"], list) for p in preds.values())
    return {"marker": "plugin-ok", "n": len(rows), "gate": {"passed": True}}
'''


def test_finalize_uses_plugin_hooks(tmp_path, monkeypatch):
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text(PLUGIN)
    run_dir, lock = make_run(tmp_path, evaluator=EvaluatorSpec(type="plugin", path=str(plugin)))

    def boom(*a, **k):
        raise AssertionError("binary_decision.evaluate must not run for a plugin evaluator")
    monkeypatch.setattr(binary_decision, "evaluate", boom)
    scorer = FakeScorer()
    out = finalize(run_dir, scorer=scorer, merger=no_merger)
    assert out["status"] == "done", out["failures"]
    seed_dir = run_dir / "runs" / "exp-s42"
    assert scorer.calls[0]["model_dir"] == seed_dir / "final"  # last step has no ckpt-step dir
    comp = json.loads((run_dir / "comparison.json").read_text())["42"]
    assert comp["selected_step"] == 10
    assert comp["sets"]["pooled_select"]["router"]["marker"] == "plugin-ok"
    assert comp["sets"]["a"]["router"]["n"] == 15 and comp["gate"]["passed"] is True
    assert comp["sets"]["pooled_select"]["constant"] == {"n": 30} and "tfidf" not in comp["sets"]["pooled_select"]
    assert json.loads((run_dir / "thresholds.json").read_text())["42"] == {"cut": 0.5, "n": 30}
    (mod,) = [m for m in list(sys.modules.values()) if getattr(m, "__file__", None) == str(plugin)]
    assert mod.CALLS == [110]  # validate saw train + select rows
    (rec,) = registry.read(registry_path(lock))
    assert rec["metrics"]["marker"] == "plugin-ok" and rec["selected_step"] == 10
    assert "plugin-ok" in (run_dir / "RUN_CARD.md").read_text()


def test_plugin_defaults_report_nll_accuracy_ece(tmp_path):
    plugin = tmp_path / "empty_plugin.py"
    plugin.write_text("# no hooks\n")
    run_dir, lock = make_run(tmp_path, evaluator=EvaluatorSpec(type="plugin", path=str(plugin)))
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    router = json.loads((run_dir / "comparison.json").read_text())["42"]["sets"]["pooled_select"]["router"]
    assert {"nll", "accuracy", "ece"} <= set(router)
    assert router["nll"] == pytest.approx(out["seeds"]["42"]["rescore"]["rescored_nll"])
    assert out["seeds"]["42"]["selected_step"] == 8 and out["status"] == "done"
    sets = json.loads((run_dir / "comparison.json").read_text())["42"]["sets"]
    assert {"tfidf", "group_prior"} <= set(sets["pooled_select"]) and set(sets) == {"a", "b", "pooled_select"}
    assert json.loads((run_dir / "thresholds.json").read_text())["42"] == {}


def test_plugin_validate_errors_fail(tmp_path):
    plugin = tmp_path / "strict_plugin.py"
    plugin.write_text("def validate(rows):\n    return ['row t1: bad label']\n")
    run_dir, lock = make_run(tmp_path, evaluator=EvaluatorSpec(type="plugin", path=str(plugin)))
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "failed" and any("plugin validate" in f for f in out["failures"])


def test_finalize_gate_failure_marks_failed(tmp_path, monkeypatch):
    run_dir, lock = make_run(tmp_path)
    real = binary_decision.evaluate

    def failing(*a, **k):
        res = real(*a, **k)
        res["gate"]["passed"] = False
        return res
    monkeypatch.setattr(binary_decision, "evaluate", failing)
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "failed" and any("gate" in f for f in out["failures"])
    (rec,) = registry.read(registry_path(lock))
    assert rec["gate"]["passed"] is False and rec["metrics"]  # the record is written anyway


def test_finalize_select_ids_changed_fails(tmp_path):
    run_dir, lock = make_run(tmp_path)
    split = Path(lock["split_file"])
    assign = json.loads(split.read_text())
    assign["a"]["assignment"]["a0"] = "report"
    split.write_text(json.dumps(assign))
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "failed"
    assert any("changed after prepare" in f for f in out["failures"])
    assert (run_dir / "status").read_text().strip() == "failed"


def test_finalize_missing_checkpoint_fails(tmp_path):
    run_dir, lock = make_run(tmp_path)
    shutil.rmtree(run_dir / "runs" / "exp-s42" / "ckpt-step8")
    scorer = FakeScorer()
    out = finalize(run_dir, scorer=scorer, merger=no_merger)
    assert out["status"] == "failed" and any("checkpoint" in f for f in out["failures"])
    assert scorer.calls == []
    (rec,) = registry.read(registry_path(lock))
    assert rec["complete"] is False and rec["selected_step"] == 8


def test_finalize_without_eval_events_fails(tmp_path):
    run_dir, lock = make_run(tmp_path)
    (run_dir / "runs" / "exp-s42" / "log.jsonl").write_text('{"step": 1, "loss": 0.5}\n')
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "failed" and any("cannot select" in f for f in out["failures"])
    cands = json.loads((run_dir / "deletion_candidates.json").read_text())
    assert cands["candidates"] == []  # an unverified seed offers no checkpoint for deletion
    assert (run_dir / "RUN_CARD.md").is_file()


def test_refinalize_upserts_registry_and_keeps_note(tmp_path):
    run_dir, lock = make_run(tmp_path, seeds=(42, 7))
    reg = registry_path(lock)
    finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert [r["seed"] for r in registry.read(reg)] == [42, 7, "summary"]
    finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert [r["seed"] for r in registry.read(reg)] == [42, 7, "summary"]  # one record per seed
    assert registry.note(reg, lock["run_id"], "beats both baselines on a and b") == 3
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    recs = registry.read(reg)
    assert [r["seed"] for r in recs] == [42, 7, "summary"] and out["status"] == "done"
    assert all(r["conclusion"] == "beats both baselines on a and b" and r["complete"] for r in recs)
    assert lock["run_id"] in registry.table(reg)


def testscore_kwargs_follow_onejev_load_rule():
    from harness.finalize import score_kwargs
    base = {"base": {"multimodal": False}}
    assert score_kwargs({**base, "flags": {"full_finetune": True}})["dtype"] == "float32"
    assert score_kwargs({**base, "flags": {"full_finetune": True, "fsdp": True, "dtype": "bfloat16"}})["dtype"] \
        == "bfloat16"
    assert score_kwargs({**base, "flags": {"dtype": "float16"}})["dtype"] == "float16"  # LoRA: flags.dtype
    assert score_kwargs({**base, "flags": {}})["dtype"] == "bfloat16"
    assert score_kwargs({**base, "flags": {"disable_fla_kernels": True}})["disable_fla_kernels"] is True


def test_finalize_forwards_fla_override(tmp_path):
    ov = {"onejev.disable_fla_kernels": {"value": True, "reason": "fla gated backward broken on this triton"}}
    run_dir, lock = make_run(tmp_path, overrides=ov)
    scorer = FakeScorer()
    finalize(run_dir, scorer=scorer, merger=no_merger)
    assert scorer.calls[0]["disable_fla_kernels"] is True
    assert "onejev.disable_fla_kernels" in (run_dir / "RUN_CARD.md").read_text()


def test_finalize_pins_leased_cards(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-unleased")
    run_dir, lock = make_run(tmp_path)
    scorer = FakeScorer()
    finalize(run_dir, scorer=scorer, merger=no_merger)
    assert scorer.calls[0]["cuda"] == "GPU-0"  # gpus.json wins over an inherited value
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-unleased"
    scorer = FakeScorer()
    finalize(run_dir, scorer=scorer, merger=no_merger, devices=["GPU-7", "GPU-8"])
    assert scorer.calls[0]["cuda"] == "GPU-7,GPU-8"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-unleased"
    (run_dir / "gpus.json").unlink()
    with pytest.raises(RuntimeError, match="gpus.json"):
        finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-unleased"


def test_finalize_records_dropped_ids(tmp_path):
    run_dir, lock = make_run(tmp_path, drop=True)
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "done", out["failures"]
    dropped = json.loads((run_dir / "dropped.json").read_text())
    assert dropped["eval_parts"] == {"a": [], "unassigned": ["a1"], "b": ["b2"]}
    assert dropped["evaluate"] == {"42": {"b": ["b4"]}}  # b6 ({"no": 1.0}) is a negative, not unlabeled
    card = (run_dir / "RUN_CARD.md").read_text()
    section = card.split("## 被去掉的 id", 1)[1].split("\n## ", 1)[0]
    for s in ("unassigned：1", "a1", "b2", "b4", "dropped.json"):
        assert s in section, s


def test_allow_test_carried_through_finalize(tmp_path):
    run_dir, lock = make_run(tmp_path, allow_test=True)
    assert lock["allow_test"] is True
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "done", out["failures"]
    (rec,) = registry.read(registry_path(lock))
    assert rec["allow_test"] is True
    assert "test 部分：已读" in (run_dir / "RUN_CARD.md").read_text()


def test_allow_test_off_recorded(tmp_path):
    run_dir, lock = make_run(tmp_path)
    assert lock["allow_test"] is False
    finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert registry.read(registry_path(lock))[0]["allow_test"] is False
    assert "test 部分：未读" in (run_dir / "RUN_CARD.md").read_text()


def test_baselines_train_on_the_recipe_split_only(tmp_path):
    plugin = tmp_path / "count_plugin.py"
    plugin.write_text("def baselines(train_rows, eval_rows):\n"
                      "    splits = sorted({r['split'] for r in train_rows})\n"
                      "    return {'seen': {'n': len(train_rows), 'splits': splits}}\n")
    run_dir, lock = make_run(tmp_path, extra_split=True, evaluator=EvaluatorSpec(type="plugin", path=str(plugin)))
    assert lock["flags"]["split"] == "train" and lock["data"]["counts"]["train"] == 86
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "done", out["failures"]
    pooled = json.loads((run_dir / "comparison.json").read_text())["42"]["sets"]["pooled_select"]
    assert pooled["seen"] == {"n": 80, "splits": ["train"]}  # OneJev's --split train skipped the 6 others


@pytest.mark.parametrize("changed", ["train.jsonl", "select.jsonl", "select_index.json"])
def test_finalize_refuses_changed_prepared_data(tmp_path, changed):
    run_dir, lock = make_run(tmp_path)
    p = Path(lock["data"]["prepared_dir"]) / changed
    p.write_text(p.read_text() + "\n")  # same rows, different bytes: prepare re-ran after resolve
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "failed"
    assert any(changed in f and "sha256" in f for f in out["failures"]), out["failures"]
    assert json.loads((run_dir / "comparison.json").read_text())["42"]["sets"] is None  # not evaluated
    assert (run_dir / "status").read_text().strip() == "failed"


def test_failed_run_offers_no_deletion_candidates(tmp_path):
    run_dir, lock = make_run(tmp_path)
    out = finalize(run_dir, scorer=FakeScorer(shift=0.05), merger=no_merger)  # NLL mismatch
    assert out["status"] == "failed"
    cands = json.loads((run_dir / "deletion_candidates.json").read_text())
    assert cands["candidates"] == [] and cands["total_bytes"] == 0 and "failed" in cands["note"]
    section = (run_dir / "RUN_CARD.md").read_text().split("## 删除候选", 1)[1].split("\n## ", 1)[0]
    assert "失败" in section and "ckpt-step4" not in section


def test_partial_dirs_listed_as_deletion_candidates(tmp_path):
    run_dir, lock = make_run(tmp_path)
    seed_dir = run_dir / "runs" / "exp-s42"
    for name in ("merged-step4.partial", "scratch.partial"):
        (seed_dir / name).mkdir()
        (seed_dir / name / "w").write_bytes(b"x" * 7)
    out = finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    assert out["status"] == "done", out["failures"]
    kinds = {Path(c["path"]).name: c["kind"] for c in
             json.loads((run_dir / "deletion_candidates.json").read_text())["candidates"]}
    assert kinds["merged-step4.partial"] == kinds["scratch.partial"] == "partial"


def test_run_card_states_the_gate_rule(tmp_path):
    run_dir, lock = make_run(tmp_path)
    finalize(run_dir, scorer=FakeScorer(), merger=no_merger)
    section = (run_dir / "RUN_CARD.md").read_text().split("## 门", 1)[1].split("\n## ", 1)[0]
    assert "没有任何对照基线" in section and "全部被去掉" in section and "不通过" in section
