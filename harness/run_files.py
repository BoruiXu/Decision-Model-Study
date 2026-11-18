"""Run-directory files written by finalize: timing, checkpoint manifest, deletion candidates, predictions,
thresholds/comparison and the one-page RUN_CARD.md. Imports without torch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from harness.catalog import sha256_file
from harness.evaluator.binary_decision import POOLED, TRAIN_DROPPED


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def rel(path, run_dir) -> str:
    return os.path.relpath(path, run_dir)


def dir_bytes(path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.lstat(os.path.join(root, f)).st_size
    return total


def file_manifest(directory) -> list[tuple[str, str]]:
    """(sha256, path relative to ``directory``) for every file below it, sorted by path."""
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return [(sha256_file(p), p.relative_to(directory).as_posix()) for p in files]


def manifest_digest(entries) -> str:
    """One sha256 for a checkpoint dir: the digest of its sorted ``<sha256>  <relpath>`` lines."""
    return hashlib.sha256("".join(f"{s}  {p}\n" for s, p in entries).encode()).hexdigest()


def write_timing(run_dir: Path, lock: dict) -> dict:
    """Add ``gpu_hours = sum(train seconds) * n_gpus / 3600`` to the runner's timing.json."""
    path = run_dir / "timing.json"
    timing = json.loads(path.read_text()) if path.is_file() else {"n_gpus": lock["n_gpus"], "seeds": {}}
    n = timing.get("n_gpus") or lock["n_gpus"]
    seconds = sum(float(s.get("seconds") or 0) for s in timing.get("seeds", {}).values())
    timing["gpu_hours"] = seconds * n / 3600
    write_json(path, timing)
    return timing


def write_checkpoint_manifest(run_dir: Path, lock: dict, seeds: dict) -> dict:
    """checkpoint.sha256 (sha256sum format, paths relative to the run dir); returns per-seed digests."""
    lora = lock["mode"] == "lora"
    base = lock["base"]
    lines = ["# sha256 of every file in each seed's selected checkpoint (and merged model); paths relative "
             "to the run dir"]
    out: dict[str, dict] = {}
    for key, s in seeds.items():
        if s.get("checkpoint") is None:
            continue
        entries = file_manifest(s["checkpoint"])
        where = rel(s["checkpoint"], run_dir)
        note = f" (LoRA adapter; base {base['name']} revision {base.get('revision')})" if lora else ""
        lines.append(f"# seed {key} step {s['selected_step']}: {where}{note}")
        lines += [f"{sha}  {where}/{p}" for sha, p in entries]
        out[key] = {"checkpoint_sha256": manifest_digest(entries)}
        merged = (s.get("merge") or {}).get("dir")
        if merged is not None:
            m_entries = file_manifest(merged)
            m_where = rel(merged, run_dir)
            lines.append(f"# seed {key} merged model: {m_where}")
            lines += [f"{sha}  {m_where}/{p}" for sha, p in m_entries]
            out[key]["merged_sha256"] = manifest_digest(m_entries)
    (run_dir / "checkpoint.sha256").write_text("\n".join(lines) + "\n")
    return out


def _ckpt_kind(name: str) -> str | None:
    if name.endswith(".partial"):  # a merge that died (finalize's own temp dir)
        return "partial"
    if name.startswith("merged-step"):
        return "merged"
    if name.startswith(("ckpt-step", "adapter-step")) or name in ("final", "adapter"):
        return "checkpoint"
    return None


FAILED_NOTE = ("the run failed: nothing is offered for deletion until a re-finalize succeeds; "
               "nothing is deleted without `ojh cleanup --confirm`")


def write_deletion_candidates(run_dir: Path, lock: dict, seeds: dict, *, failed: bool = False) -> dict:
    """Unselected checkpoint dirs, every merged dir and any ``.partial`` dir, with sizes. Nothing is deleted
    here. A failed run lists nothing: its selection or verification is not trustworthy.
    """
    cands = []
    for seed in ([] if failed else lock["seeds"]):
        seed_dir = run_dir / "runs" / f"{lock['experiment']}-s{seed}"
        keep = seeds.get(str(seed), {}).get("checkpoint")
        if not seed_dir.is_dir():
            continue
        for d in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            kind = _ckpt_kind(d.name)
            if kind is None or (kind == "checkpoint" and (keep is None or d == Path(keep))):
                continue
            cands.append({"path": str(d), "seed": seed, "kind": kind, "bytes": dir_bytes(d)})
    obj = {"note": FAILED_NOTE if failed else "nothing is deleted without `ojh cleanup --confirm`", "candidates": cands,
           "total_bytes": sum(c["bytes"] for c in cands)}
    write_json(run_dir / "deletion_candidates.json", obj)
    return obj


def write_predictions(run_dir: Path, seeds: dict, positive: str | None) -> None:
    """predictions_select.jsonl: the rescored per-row predictions of every seed's selected checkpoint."""
    lines = []
    for s in seeds.values():
        for rid in sorted(s.get("records") or {}):
            r = s["records"][rid]
            row = {"seed": s["seed"], "step": s["selected_step"], "id": rid, "labels": r["labels"],
                   "probs": r["probs"], "target": r["target"]}
            if positive is not None:
                row["positive"] = r["probs"][r["labels"].index(positive)]
            lines.append(json.dumps(row) + "\n")
    (run_dir / "predictions_select.jsonl").write_text("".join(lines))


# ---- RUN_CARD.md ----

def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_f(x)}" for k, x in v.items())
    return str(v).replace("|", "\\|").replace("\n", " ")


def _e(v) -> str:
    return "-" if v is None else f"{v:.1e}"


def _metric_table(entry: dict) -> list[str]:
    methods = [k for k, v in entry.items() if isinstance(v, dict)]
    methods.sort(key=lambda k: k != "router")
    cols = list(dict.fromkeys(c for m in methods for c in entry[m] if not isinstance(entry[m][c], dict)))
    out = ["| 方法 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    out += [f"| {m} | " + " | ".join(_f(entry[m].get(c)) for c in cols) + " |" for m in methods]
    return out


def _set_order(sets: dict) -> list[str]:
    return ([POOLED] if POOLED in sets else []) + [n for n in sets if n != POOLED]


def _gate_text(gate: dict | None) -> str:
    if not gate:
        return "未评测"
    passed = gate.get("passed")
    verdicts = {True: "通过", False: "未通过", None: "未定义"}
    head = "关闭（只报告）" if not gate.get("enabled") else verdicts.get(passed, _f(passed))
    detail = "; ".join(f"{k}: {_f(v)}" for k, v in gate.items() if isinstance(v, dict))
    return head + (f"（{detail}）" if detail else "")


DROPPED_SHOWN = 20


def _dropped_lines(dropped: dict) -> list[str]:
    """Counts per key and the first ids; the full lists are in dropped.json."""
    out = []

    def add(label, ids):
        shown = ", ".join(map(str, ids[:DROPPED_SHOWN])) + (" …" if len(ids) > DROPPED_SHOWN else "")
        out.append(f"- {label}：{len(ids)}" + (f"（{shown}）" if ids else ""))
    for key, ids in (dropped.get("eval_parts") or {}).items():
        add(key, ids)
    for seed, per_set in (dropped.get("evaluate") or {}).items():
        for key, ids in per_set.items():
            if key == TRAIN_DROPPED:
                add(f"基线训练行缺标签或分组（seed {seed}）", ids)
            else:
                add(f"评测时缺标签（seed {seed}）· {key}", ids)
    return out + ["- 完整列表见 dropped.json"]


def write_run_card(run_dir: Path, lock: dict, *, status: str, failures: list[str], comparison: dict,
                   thresholds: dict, timing: dict, candidates: dict, versions: dict, manifests: dict,
                   dropped: dict) -> None:
    base, data = lock["base"], lock["data"]
    seeds = ", ".join(str(s) for s in lock["seeds"])
    ev = lock["evaluator"]
    lines = [f"# 运行卡片：{lock['experiment']} / {lock['run_id']}", "", "## 概要",
             f"- 实验：{lock['experiment']}；run id：`{lock['run_id']}`；seed：{seeds}",
             f"- 基座：{base['name']}（revision `{base.get('revision')}`）；训练方式：{lock['mode']}"
             f"（{lock.get('parallel')}，{lock['n_gpus']} 卡）",
             f"- 数据：train.jsonl sha256 `{data['sha256']['train.jsonl']}`；"
             f"源文件 sha256 `{data['source_sha256']}`",
             f"- 适配器：{data.get('adapter')}（sha256 `{data.get('adapter_sha256')}`）"
             + (f"；参数 train {data.get('train_adapter_args') or {}}，全局 {data.get('adapter_args') or {}}"
                if data.get("adapter_args") or data.get("train_adapter_args") else ""),
             f"- 标签版本：训练 {versions['train']}；参照 {versions['reference']}",
             "- test 部分：" + ("已读（prepare --allow-test，须经用户批准）" if lock.get("allow_test") else "未读"),
             f"- 评测器：{ev['type']}{'（' + ev['path'] + '）' if ev.get('path') else ''}",
             f"- GPU 时间：{timing.get('gpu_hours', 0):.2f} h",
             f"- 状态：**{status}**", "", "## 选中的 checkpoint",
             "| seed | step | 目录 | 训练时 NLL | 复评 NLL | 差 | 合并后最大概率差 | sha256 |",
             "|---|---|---|---|---|---|---|---|"]
    for key, c in comparison.items():
        r, m = c.get("rescore") or {}, c.get("merge") or {}
        sha = (manifests.get(key) or {}).get("checkpoint_sha256", "-")[:16]
        lines.append(f"| {key} | {_f(c.get('selected_step'))} | {_f(c.get('checkpoint'))} "
                     f"| {_f(r.get('train_nll'))} | {_f(r.get('rescored_nll'))} | {_e(r.get('abs_diff'))} "
                     f"| {_e(m.get('max_prob_diff'))} | `{sha}` |")
    lines += ["", "## 对照表"]
    for key, c in comparison.items():
        sets = c.get("sets") or {}
        if not sets:
            lines += [f"- seed {key}：未评测"]
        for name in _set_order(sets):
            lines += ["", f"### seed {key} · {name}（{sets[name].get('questions', '-')} 题）", ""]
            lines += _metric_table(sets[name])
    lines += ["", "## 阈值（select 合并集上锁定）"]
    for key, thr in thresholds.items():
        if not thr:
            lines.append(f"- seed {key}：无")
            continue
        for name, t in thr.items():
            if not isinstance(t, dict):
                lines.append(f"- seed {key} · **{name}**：{_f(t)}")
                continue
            own = ", ".join(f"{k} {_f(v)}" for k, v in t.items() if k != "per_set")
            per = "; ".join(f"{s}: {_f(v)}" for s, v in (t.get("per_set") or {}).items())
            lines.append(f"- seed {key} · **{name}**：{own}" + (f"；分集 {per}" if per else ""))
    lines += ["", "## 被去掉的 id（参照连接、未分配、评测时缺标签）"] + _dropped_lines(dropped)
    lines += ["", "## 门"]
    if ev["type"] == "binary_decision":
        lines.append("- 规则：每个评测集上，路由器的 AUC（有分组时还有组内 AUC）都要高于可用基线中最好的一个"
                     "（tfidf、group_prior、length 及注册的其他基线）；"
                     "某个集没有任何对照基线，或它的 select 行全部被去掉时，门不通过（没有可比的对象不能算通过）")
    lines += [f"- seed {k}：{_gate_text(c.get('gate'))}" for k, c in comparison.items()]
    lines += ["", "## overrides"]
    ov = lock.get("overrides") or {}
    lines += [f"- {k} = {_f(v.get('value'))}（原因：{v.get('reason')}）" for k, v in ov.items()] or ["- 无"]
    lines += ["", "## 删除候选（需 `ojh cleanup --confirm`）"]
    if status == "failed":
        lines.append("- 无：run 失败时不列出任何删除候选；修好后重新 finalize 成功才会列出")
    else:
        lines += [f"- {rel(c['path'], run_dir)}（{c['kind']}，{c['bytes'] / 2**30:.2f} GiB）"
                  for c in candidates["candidates"]] or ["- 无"]
    lines += ["", "## 状态", f"- **{status}**"] + [f"- {f}" for f in failures]
    (run_dir / "RUN_CARD.md").write_text("\n".join(lines) + "\n")
