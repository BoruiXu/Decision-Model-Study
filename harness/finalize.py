"""Finalize a trained run: select, reload and rescore, optionally merge, evaluate, write run files, register.

Called by the runner after every seed trained. Per seed: select the step from the ``harness_eval`` events,
reload the checkpoint and rescore the select rows (its NLL must match the training NLL within 1e-3), for
``merge: true`` LoRA runs merge the adapter and check the merged model's per-row probabilities against
base + adapter within 1e-3, then evaluate against the baselines. The run is ``failed`` on any mismatch,
a missing checkpoint, select rows that changed after prepare, plugin validation errors or a gate that is
on and not passed; it is ``done`` otherwise. Registry records are written either way.

Imports without torch: the GPU work lives in ``score`` and ``merge``, which import torch lazily.
"""
from __future__ import annotations

import contextlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from harness import registry, run_files
from harness.catalog import sha256_file
from harness.config import (ConfigError, DataSpec, EvalSetSpec, EvaluatorSpec, ExperimentConfig, FieldsSpec,
                            get_field)
from harness.eval_records import nll
from harness.jsonl import read_jsonl
from harness.evaluator import binary_decision
from harness.merge import merge_lora
from harness.plugins import evaluate_plugin, hook, load_plugin
from harness.prepare import build_eval_parts, read_eval
from harness.resolve import data_changes
from harness.score import score_checkpoint
from harness.select import checkpoint_dir, eval_events, select_step

TOL = 1e-3


def exp_from_lock(lock: dict) -> ExperimentConfig:
    """The parts of the experiment config that evaluation needs, rebuilt from the run lock."""
    d = lock["data"]
    return ExperimentConfig(
        name=lock["experiment"], base=lock["base"]["name"], recipe="",
        data=DataSpec(train=d["source"], adapter=d["adapter"], split_file=lock.get("split_file"),
                      adapter_args=dict(d.get("adapter_args") or {}),
                      train_adapter_args=dict(d.get("train_adapter_args") or {}),
                      eval={n: EvalSetSpec(**s) for n, s in lock["eval_sets"].items()}),
        fields=FieldsSpec(**lock["fields"]), evaluator=EvaluatorSpec(**lock["evaluator"]),
        seeds=list(lock["seeds"]), merge=bool(lock.get("merge")))


def _jsonl(path) -> list[dict]:
    return read_jsonl(path)


def baseline_train_rows(lock: dict, rows: list[dict]) -> list[dict]:
    """The prepared train rows OneJev trained on: with a ``split`` flag, only rows whose split equals it."""
    split = lock["flags"].get("split")
    if split is None or isinstance(split, bool):
        return rows
    return [r for r in rows if r.get("split") == str(split)]


def label_version(rows: list[dict], path) -> str:
    """The rows' shared ``meta.label_version``, else the first 12 hex chars of the file's sha256."""
    values = set()
    for r in rows:
        try:
            values.add(str(get_field(r, "meta.label_version")))
        except KeyError:
            values.add(None)
    if len(values) == 1 and None not in values:
        return values.pop()
    return sha256_file(path)[:12]


def label_versions(lock: dict, exp: ExperimentConfig, train_rows: list[dict]) -> dict:
    refs = []
    for name, spec in exp.data.eval.items():
        path = spec.reference or spec.train_labels
        refs.append(label_version(read_eval(exp.data, name, path), path))
    return {"train": label_version(train_rows, lock["data"]["train"]),
            "reference": "+".join(dict.fromkeys(refs)) or None}


def _plugin(exp: ExperimentConfig):
    t = exp.evaluator.type
    if t == "binary_decision":
        return None
    if t == "plugin":
        if not exp.evaluator.path:
            raise ConfigError("evaluator.type plugin needs evaluator.path")
        return load_plugin(exp.evaluator.path)
    raise ConfigError(f"unknown evaluator.type {t!r}: use binary_decision or plugin")


def score_kwargs(lock: dict) -> dict:
    """Forward-pass settings of the in-training eval, so the rescored NLL compares with it.

    Mirrors OneJev ``sft.py``: load dtype float32 for a full fine-tune without FSDP, else ``--dtype``;
    autocast as ``use_autocast``; fla kernels blocked when ``--disable-fla-kernels`` was set.
    """
    f = lock["flags"]
    full = bool(f.get("full_finetune"))
    return {"multimodal": bool(lock["base"].get("multimodal")),
            "dtype": "float32" if full and not f.get("fsdp") else (f.get("dtype") or "bfloat16"),
            "autocast": bool(f.get("lora_autocast") or (full and not f.get("no_autocast"))),
            "disable_fla_kernels": bool(f.get("disable_fla_kernels")),
            "chat_template": f.get("chat_template"), "media_root": f.get("media_root")}


@contextlib.contextmanager
def _gpu_env(run_dir: Path, lock: dict, devices: list[str] | None = None):
    """While scoring in this process: the machine env, offline HF, and CUDA_VISIBLE_DEVICES pinned to
    ``devices`` or else the run's leased cards in gpus.json (the runner still holds that lease), even
    when a value was inherited. The previous environment is restored afterwards."""
    if not devices:
        gpus = run_dir / "gpus.json"
        devices = [g["uuid"] for g in json.loads(gpus.read_text())] if gpus.is_file() else []
    if not devices:
        raise RuntimeError(f"no cards to score on: {run_dir / 'gpus.json'} is missing or empty and no devices "
                           "were given")
    want = {**{k: str(v) for k, v in (lock["machine"].get("env") or {}).items()}, "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(devices)}
    saved = {k: os.environ.get(k) for k in want}
    os.environ.update(want)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _max_prob_diff(a: dict, b: dict) -> float:
    if set(a) != set(b):
        return float("inf")
    return max((abs(x - y) for rid in a for x, y in zip(a[rid]["probs"], b[rid]["probs"])), default=0.0)


def _evaluate(recs: dict, parts, train_rows, exp: ExperimentConfig, plugin) -> dict:
    if plugin is not None:
        return evaluate_plugin(plugin, recs, parts, train_rows, exp.fields, exp.evaluator)
    k = exp.fields.label.rsplit(".", 1)[-1]
    preds = {}
    for rid, r in recs.items():
        if k not in r["labels"]:
            raise ValueError(f"fields.label {exp.fields.label!r}: key {k!r} is not a scored label {r['labels']}")
        preds[rid] = r["probs"][r["labels"].index(k)]
    return binary_decision.evaluate(preds, parts, train_rows, exp.fields, exp.evaluator)


def _seed(run_dir, lock, exp, seed, select_rows, train_rows, parts, plugin, scorer, merger) -> dict:
    seed_dir = run_dir / "runs" / f"{lock['experiment']}-s{seed}"
    out = {"seed": seed, "selected_step": None, "checkpoint": None, "rescore": None, "merge": None,
           "records": None, "evaluation": None, "failures": []}

    def fail(msg):
        out["failures"].append(f"seed {seed}: {msg}")

    log = seed_dir / "log.jsonl"
    events = _jsonl(log) if log.is_file() else []
    evals = eval_events(events)
    rule = hook(plugin, "select_rule")
    try:
        step = rule([e for e in events if e.get("event") == "harness_eval"]) if rule else select_step(events)
    except ValueError as e:
        fail(f"cannot select a step from {log}: {e}")
        return out
    out["selected_step"] = step
    hits = [e for e in evals if e["step"] == step]
    if not hits:
        fail(f"no harness_eval select event at step {step}")
        return out
    lora = lock["mode"] == "lora"
    ckpt = checkpoint_dir(seed_dir, step, lock["mode"], max(e["step"] for e in evals))
    if ckpt is None:
        fail(f"no checkpoint dir for step {step} under {seed_dir}")
        return out
    out["checkpoint"] = ckpt
    if hits[-1].get("n") not in (None, len(select_rows)):
        fail(f"training scored {hits[-1]['n']} select rows but the prepared select.jsonl has {len(select_rows)}")
        return out
    base_dir =Path(lock["machine"]["model_dir"]) / lock["base"]["name"] if lora else None
    kw = score_kwargs(lock)
    recs = scorer(ckpt, select_rows, base_dir=base_dir, **kw)
    if set(recs) != {r["id"] for r in select_rows}:
        fail(f"rescoring returned {len(recs)} rows for {len(select_rows)} select rows")
        return out
    train_nll, rescored = hits[-1]["nll"], nll([recs[r["id"]] for r in select_rows])
    diff = abs(rescored - train_nll)
    out["rescore"] = {"train_nll": train_nll, "rescored_nll": rescored, "abs_diff": diff, "tolerance": TOL}
    if diff > TOL:
        fail(f"rescored NLL {rescored:.6f} differs from training NLL {train_nll:.6f} by {diff:.2e} (> {TOL})")
    if lora and lock.get("merge"):
        merged = seed_dir / f"merged-step{step}"
        if not merged.is_dir():  # a re-finalize reuses the merged dir it wrote before
            merged = Path(merger(base_dir, ckpt, merged, multimodal=kw["multimodal"], dtype=kw["dtype"],
                                 chat_template=kw["chat_template"],
                                 disable_fla_kernels=kw["disable_fla_kernels"]))
        m_recs = scorer(merged, select_rows, base_dir=None, **kw)
        m_diff = _max_prob_diff(recs, m_recs)
        out["merge"] = {"dir": merged, "max_prob_diff": m_diff, "tolerance": TOL,
                        "nll": nll(list(m_recs.values())) if set(m_recs) == set(recs) else None}
        if m_diff > TOL:
            fail(f"merged model probabilities differ from base + adapter by up to {m_diff:.2e} (> {TOL})")
    out["records"] = recs
    if parts is not None:
        ev = out["evaluation"] = _evaluate(recs, parts, train_rows, exp, plugin)
        if ev["gate"].get("enabled") and ev["gate"].get("passed") is False:
            fail("gate enabled and not passed")
    return out


def _comparison(run_dir: Path, s: dict) -> dict:
    ev = s["evaluation"] or {}
    merge = dict(s["merge"], dir=run_files.rel(s["merge"]["dir"], run_dir)) if s["merge"] else None
    return {"selected_step": s["selected_step"],
            "checkpoint": run_files.rel(s["checkpoint"], run_dir) if s["checkpoint"] else None,
            "rescore": s["rescore"], "merge": merge, "sets": ev.get("sets"), "gate": ev.get("gate"),
            "tfidf_strength": ev.get("tfidf_strength"), "dropped": ev.get("dropped"), "failures": s["failures"]}


def _gpu_name(run_dir: Path) -> str | None:
    p = run_dir / "gpus.json"
    names = [g.get("name") for g in json.loads(p.read_text())] if p.is_file() else []
    return "+".join(dict.fromkeys(n for n in names if n)) or None


def _mean_std(dicts: list[dict]) -> dict:
    out = {}
    for key in dict.fromkeys(k for d in dicts for k in d):
        vals = [d[key] for d in dicts
                if isinstance(d.get(key), (int, float)) and not isinstance(d.get(key), bool)]
        if vals:
            out[key] = {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals), "n": len(vals)}
    return out


def _summary(recs: list[dict], lock: dict, failures: list[str]) -> dict:
    scored = [r for r in recs if isinstance(r.get("metrics"), dict)]
    names = dict.fromkeys(n for r in scored for n in (r.get("baselines") or {}))
    passed = [r["gate"].get("passed") for r in scored if r.get("gate")]
    decided = [p for p in passed if p is not None]
    steps = [r["selected_step"] for r in recs]
    shas = [r["checkpoint_sha256"] for r in recs]
    hours = [r["gpu_hours"] for r in recs]
    rec = {k: v for k, v in recs[0].items() if k not in ("complete", "merged_sha256")}
    rec.update({
        "seed": "summary", "seeds": [r["seed"] for r in recs], "selected_step": steps,
        "checkpoint_sha256": None if None in shas else shas,
        "gpu_hours": None if None in hours else sum(hours),
        "metrics": {**_mean_std([r["metrics"] for r in scored]), "selected_steps": steps} if scored else None,
        "baselines": {n: _mean_std([r["baselines"][n] for r in scored if n in r["baselines"]]) for n in names}
        if scored else None,
        "gate": {"enabled": bool(lock["evaluator"].get("gate")),
                 "passed": (all(decided) if decided else None) if lock["evaluator"].get("gate") else None,
                 "per_seed": {str(r["seed"]): r["gate"].get("passed") if r.get("gate") else None for r in recs}},
        "failures": failures,
    })
    merged = [r.get("merged_sha256") for r in recs]
    if any(merged):
        rec["merged_sha256"] = merged
    return rec


def _records(run_dir, lock, seeds, comparison, manifests, timing, versions, status, run_failures,
             failures) -> list[dict]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data, base, onejev = lock["data"], lock["base"], lock["onejev"]
    gpu_name = _gpu_name(run_dir)
    recs = []
    for key, s in seeds.items():
        c = comparison[key]
        pooled = (c.get("sets") or {}).get(binary_decision.POOLED)
        secs = (timing.get("seeds", {}).get(key) or {}).get("seconds")
        rec = {
            "exp": lock["experiment"], "run_id": lock["run_id"], "seed": s["seed"],
            "train_label_version": versions["train"], "reference_label_version": versions["reference"],
            "data_sha256": data["sha256"]["train.jsonl"], "data_source_sha256": data.get("source_sha256"),
            "select_sha256": data["sha256"].get("select.jsonl"),
            "base": base["name"], "base_revision": base.get("revision"),
            "recipe": {"parallel": lock.get("parallel"), "lora": lock.get("lora"), "flags": lock["flags"]},
            "mode": lock["mode"],
            "onejev": {"revision": onejev.get("revision"), "patch_sha256": (onejev.get("patch") or {}).get("sha256")},
            "gpu_name": gpu_name, "n_gpus": lock["n_gpus"],
            "gpu_hours": None if secs is None else float(secs) * lock["n_gpus"] / 3600,
            "selected_step": s["selected_step"],
            "checkpoint_sha256": (manifests.get(key) or {}).get("checkpoint_sha256"),
            "metrics": pooled.get("router") if pooled else None,
            "baselines": {k: v for k, v in pooled.items() if k != "router" and isinstance(v, dict)}
            if pooled else None,
            "gate": c.get("gate"), "overrides": lock.get("overrides") or {}, "conclusion": "",
            "allow_test": bool(lock.get("allow_test")),
            "evaluator": lock["evaluator"]["type"], "status": status,
            "failures": run_failures + s["failures"], "finalized_at": now,
        }
        if (manifests.get(key) or {}).get("merged_sha256"):
            rec["merged_sha256"] = manifests[key]["merged_sha256"]
        recs.append(rec)
    if len(recs) >= 2:
        recs.append(_summary(recs, lock, failures))
    return recs


def finalize(run_dir, *, scorer=score_checkpoint, merger=merge_lora, devices: list[str] | None = None) -> dict:
    """Finalize ``run_dir``. ``devices`` (CUDA uuids) overrides gpus.json for a manual re-finalize."""
    run_dir = Path(run_dir).absolute()
    lock = yaml.safe_load((run_dir / "run.lock.yaml").read_text())
    exp = exp_from_lock(lock)
    plugin = _plugin(exp)
    train_rows, select_rows = _jsonl(lock["data"]["train"]), _jsonl(lock["data"]["select"])
    run_failures: list[str] = []

    parts = build_eval_parts(exp, allow_test=bool(lock.get("allow_test")))
    dropped = {"eval_parts": parts.dropped}
    changed = data_changes(lock)
    if changed:
        run_failures.append("prepared data changed after resolve (sha256 differs from the lock; the run did "
                            "not train on these files): " + "; ".join(changed))
        parts = None
    elif sorted(r["id"] for r in parts.select_rows) != sorted(r["id"] for r in select_rows):
        run_failures.append("select ids rebuilt from the source files differ from the prepared select.jsonl: "
                            "the source files changed after prepare (re-run prepare and resolve)")
        parts = None
    validate = hook(plugin, "validate")
    if validate:
        errors = list(validate(train_rows + select_rows) or [])
        if errors:
            run_failures.append(f"plugin validate: {len(errors)} error(s): " + "; ".join(map(str, errors[:5])))

    seeds = {}
    baseline_rows = baseline_train_rows(lock, train_rows)
    with _gpu_env(run_dir, lock, devices):
        for seed in lock["seeds"]:
            seeds[str(seed)] = _seed(run_dir, lock, exp, seed, select_rows, baseline_rows, parts, plugin,
                                     scorer, merger)
    failures = run_failures + [f for s in seeds.values() for f in s["failures"]]
    status = "failed" if failures else "done"

    comparison = {k: _comparison(run_dir, s) for k, s in seeds.items()}
    thresholds = {k: (s["evaluation"] or {}).get("thresholds") for k, s in seeds.items()}
    timing = run_files.write_timing(run_dir, lock)
    manifests = run_files.write_checkpoint_manifest(run_dir, lock, seeds)
    candidates = run_files.write_deletion_candidates(run_dir, lock, seeds, failed=status == "failed")
    run_files.write_predictions(run_dir, seeds, None if plugin else exp.fields.label.rsplit(".", 1)[-1])
    run_files.write_json(run_dir / "thresholds.json", thresholds)
    run_files.write_json(run_dir / "comparison.json", comparison)
    dropped["evaluate"] = {k: (s["evaluation"] or {}).get("dropped") or {} for k, s in seeds.items()}
    run_files.write_json(run_dir / "dropped.json", dropped)
    versions = label_versions(lock, exp, train_rows)
    records = _records(run_dir, lock, seeds, comparison, manifests, timing, versions, status, run_failures,
                       failures)
    for rec in records:
        registry.append(lock["machine"]["registry"], rec)
    run_files.write_run_card(run_dir, lock, status=status, failures=failures, comparison=comparison,
                             thresholds=thresholds, timing=timing, candidates=candidates, versions=versions,
                             manifests=manifests, dropped=dropped)
    (run_dir / "status").write_text(status + "\n")
    return {"run_id": lock["run_id"], "status": status, "failures": failures, "seeds": comparison,
            "records": records}
