"""The run report of `rsi report`: fixed-code Markdown with Chinese headings.

Everything comes from STATE.json, the registry and the registered manifests;
building the report writes nothing. When no version was ever promoted the
report opens with a "没有产生可发布的版本" summary listing each round's outcome.
Before a stop it is a progress report. The orchestrating agent adds the
interpretation (what was learned, what to try next) on top of it.
"""
import datetime
import os

from rsi import timing
from rsi.rules import is_number, stop_settings
from rsi.workspace import RsiError

# Order of the module sections under 实验记录 (the record schema's modules).
RECORD_MODULES = ("data", "train", "tune", "deliver", "judge", "orchestrator")


def _num(x):
    return "%.6g" % x if is_number(x) else "无"


def _cell(text):
    """A Markdown table cell: one line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(header, rows):
    lines = ["| %s |" % " | ".join(header), "|" + "---|" * len(header)]
    return lines + ["| %s |" % " | ".join(_cell(c) for c in row) for row in rows]


def _rounds(ws):
    """(round number, STATE entry) in round order."""
    return sorted((int(k), r) for k, r in ws.state["rounds"].items())


def _outcome(ws, n, r):
    """(decision, reason, evaluation id) of round `n` (STATE entry `r`).

    An undecided round gives 未判定, followed by its registered evaluations (for
    example one left undecided by a user stop).
    """
    d = r["decision"]
    if d is None:
        pending = ws.evaluations(n)
        return ("未判定（%s）" % "、".join(pending) if pending else "未判定"), "", ""
    decision = d.get("decision") + (" " + d["version"] if d.get("version") else "")
    return decision, d.get("reason") or "", d.get("evaluation_id") or "无"


def _metrics(metrics, primary):
    """`name value` pairs, primary metric first."""
    keys = [primary] if primary in metrics else []
    keys += [k for k in metrics if k != primary]
    return "，".join("%s %s" % (k, _num(metrics[k])) for k in keys)


def _summary(ws):
    if ws.state["versions"]:
        return []
    lines = ["**没有产生可发布的版本。** 没有候选被升级为版本，各轮的结论：", ""]
    for n, r in _rounds(ws):
        decision, reason, _eval = _outcome(ws, n, r)
        lines.append("- 第 %d 轮：%s%s" % (n, decision, "，" + reason if reason else ""))
    return lines + [""]


def _task(ws, task):
    s = ws.state
    d, m = task["decision"], task["metrics"]
    st = stop_settings(task)
    settings = ["max_rounds %d" % st["max_rounds"]]
    for key in ("patience", "max_attempts_per_round"):
        settings.append("%s %d%s" % (key, st[key], "（默认）" if key in st["defaulted"] else ""))
    return [
        "## 任务", "",
        "- 任务：%s v%d（%s）" % (s["task_id"], s["task_version"], ws.current("task")),
        "- 决策类型：%s（标签：%s）" % (d["type"], "、".join(d["labels"])),
        "- 主指标：%s（%s）" % (m["primary"], "越高越好" if m["higher_is_better"] else "越低越好"),
        "- 目标：%s" % ("未设" if st["target"] is None else "%s %s" % (m["primary"], _num(st["target"]))),
        "- 停止设置：%s" % "，".join(settings),
        "",
    ]


def _time_line(ws, now):
    """The run's total time as a report line, or None when its start is unknown."""
    start, end, approx = timing.run_window(ws)
    if start is None:
        return None
    if timing.stopped_unknown(ws):
        return "- 总用时：未知（%s 开始）" % timing.format_full(start)
    dur = ("约 " if approx else "") + timing.format_duration(timing.seconds_between(start, end, now))
    if end is None:
        return "- 已运行：%s（%s 开始，截至报告生成）" % (dur, timing.format_full(start))
    return "- 总用时：%s（%s 开始，%s 停止）" % (dur, timing.format_full(start), timing.format_full(end))


def _stop(ws, now):
    st = ws.state["stopped"]
    if st:
        when = ([] if st.get("round") is None else ["第 %d 轮" % st["round"]]) + [timing.format_full(st["at"]) if timing._good(st.get("at")) else str(st.get("at"))]
        line = "- 已停止（%s）：%s" % ("，".join(when), st.get("reason"))
        if st.get("by"):
            line += "；由 %s 提出" % st["by"]
    else:
        stop, why = ws.stop_check()
        line = "- 未停止。停止检查：%s: %s" % ("stop" if stop else "continue", why)
    extra = _time_line(ws, now)
    return ["## 停止状态", "", line] + ([extra] if extra else []) + [""]


def _rounds_table(ws, now):
    rows = []
    wins = timing.round_windows(ws)
    for n, r in _rounds(ws):
        decision, reason, evaluation = _outcome(ws, n, r)
        rows.append((n, r["entry"], decision, reason, evaluation, timing.round_text(wins[n], now, r["decision"] is not None) or "无"))
    return ["## 各轮", ""] + _table(("轮次", "入口模块", "判定", "理由", "评测", "用时"), rows) + [""]


def _versions(ws, primary):
    s = ws.state
    lines = ["## 版本", ""]
    if not s["versions"]:
        return lines + ["没有版本。", "", "当前最佳：无", ""]
    lines += _table(("版本", "候选", "交付", "评测", "轮次"),
                    [(v["version"], v["candidate_id"], v["delivery_id"], v["evaluation_id"], v["round"])
                     for v in s["versions"]])
    v = ws.version(s["best"])
    best = "当前最佳：%s，%s %s（%s）" % (s["best"], primary, _num(ws.best_primary()), v["evaluation_id"])
    lines += ["", best, ""]
    package = ws.manifest(v["delivery_id"])["package"]
    report = ws.manifest(v["evaluation_id"])["files"]["report"]["path"]
    lines += [
        "- 模型包：%s" % ws.package_location(v["delivery_id"]),
        "- 启动方式：%s" % package["entry"],
        "- 运行环境：%s" % package["environment"],
        "- 交付目录：%s" % ws.path(v["delivery_id"]),
        "- 评测报告：%s" % os.path.join(ws.path(v["evaluation_id"]), report),
        "",
    ]
    return lines


def _milestones(ws, primary):
    """Milestone evaluations with their testset metrics, then the files each one hands over."""
    rows, files = [], []
    for e in ws.entries("evaluation"):
        m = ws.manifest(e["id"])
        if not m["uses_testset"]:
            continue
        rows.append((e["id"], m["candidate_id"], e["round"], _metrics(m["metrics_test"], primary)))
        files.append("- %s：评测报告 %s；交付 %s（%s），模型包 %s"
                     % (e["id"], os.path.join(ws.path(e["id"]), m["files"]["report"]["path"]),
                        m["delivery_id"], ws.path(m["delivery_id"]), ws.package_location(m["delivery_id"])))
    lines = ["## 里程碑评测", ""]
    if not rows:
        return lines + ["无。", ""]
    return lines + _table(("评测", "候选", "轮次", "test 指标"), rows) + [""] + files + [""]


def _records(ws):
    groups = {}
    for e in ws.entries("record"):
        m = ws.manifest(e["id"])
        groups.setdefault(m["module"], []).append((e["id"], e["round"], m["result"]["conclusion"]))
    lines = ["## 实验记录", ""]
    if not groups:
        return lines + ["无。", ""]
    order = [k for k in RECORD_MODULES if k in groups] + sorted(k for k in groups if k not in RECORD_MODULES)
    for module in order:
        lines += ["### %s" % module, ""] + _table(("记录", "轮次", "结论"), groups[module]) + [""]
    return lines


def _budget(ws, task):
    s = ws.state
    limit = ws.attempt_limit()
    st = stop_settings(task)
    lines = ["## 尝试次数和预算", "",
             "每轮最多尝试 %d 次（max_attempts_per_round%s；训练产出的候选和 search 各算一次）。"
             % (limit, "，默认" if "max_attempts_per_round" in st["defaulted"] else ""), ""]
    rows = []
    for n, _r in _rounds(ws):
        done = ws.attempts(n)
        rows.append((n, "%d of %d" % (len(done), limit), ", ".join(done)))
    lines += _table(("轮次", "尝试", "产物"), rows) + [""]
    budget = task["budget"]

    def cap(x):
        return "预算未设" if x is None else "预算 %s" % _num(x)

    lines += [
        "- 算力小时：已用 %s，%s（只统计已登记候选的 cost.compute_hours）"
        % (_num(s["budget_used"]["compute_hours"]), cap(budget["compute_hours"])),
        "- 标注条数：%s；API 费用：%s（这两项的用量由顶层按模块报告累计，代码不统计）"
        % (cap(budget["annotation_items"]), cap(budget["api_cost"])),
        "- 评测集使用：%d / %d（%s）"
        % (s["eval_use_count"], task["eval_use_limit"], ws.current("evalset") or "无"),
        "",
    ]
    return lines


def report_markdown(ws, now=None):
    """The run report of workspace `ws` as Markdown text. Reads only. `now` (a datetime) is the
    report time the running durations are measured to (default: the current time)."""
    now = datetime.datetime.now() if now is None else now
    task = ws.task()
    primary = task["metrics"]["primary"]
    lines = ["# RSI 运行报告：%s" % ws.state["task_id"], ""]
    lines += _summary(ws)
    if not ws.state["stopped"]:
        lines += ["运行还没有停止，这是进度报告。", ""]
    lines += _task(ws, task)
    lines += _stop(ws, now)
    lines += _rounds_table(ws, now)
    lines += _versions(ws, primary)
    lines += _milestones(ws, primary)
    lines += _records(ws)
    lines += _budget(ws, task)
    return "\n".join(lines).rstrip("\n") + "\n"


def _same(a, b):
    return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)


def _protected(ws, path):
    """True when writing `path` would write STATE.json, registry.jsonl or anything under artifacts/.

    Paths are compared with os.path.samefile, not as strings, so another
    spelling of the same file is caught too: a case variant on a
    case-insensitive filesystem, a hard link, a symbolic link. The path as
    given and its resolved form (which follows a dangling link) are both
    checked, each with all its ancestors against artifacts/.
    """
    own = [os.path.join(ws.root, name) for name in ("STATE.json", "registry.jsonl")]
    arts = os.path.join(ws.root, "artifacts")
    for p in (os.path.abspath(path), os.path.realpath(path)):
        if any(_same(p, o) for o in own):
            return True
        while True:
            if _same(p, arts):
                return True
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
    return False


def write_report(ws, out):
    """Write the report to `out`; return its absolute path.

    Refuses the workspace's own files (STATE.json, registry.jsonl) and any
    path under artifacts/, which only registration writes, under any spelling
    (see _protected).
    """
    path = os.path.abspath(out)
    if _protected(ws, path):
        raise RsiError("--out %s: the report must not replace STATE.json or registry.jsonl or go under "
                       "artifacts/; write it elsewhere, e.g. <ws>/REPORT.md" % out)
    if os.path.isdir(path):
        raise RsiError("--out %s is a directory; give a file path, e.g. <ws>/REPORT.md" % out)
    text = report_markdown(ws)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        raise RsiError("--out %s: cannot write the report: %s" % (out, e))
    return path
