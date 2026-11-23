"""The four progress pages: the overview <ws>/dashboard.html and, under <ws>/dashboard/,
rounds.html (轮次与产物), events.html (事件记录) and report.html (运行报告).

The overview holds only the pauses waiting for the user, the workflow diagram (rsi.dashboard.
flow_states) and three cards: the best version, how far the stop conditions are, and what the
run waits for. render_pages builds every page from rsi.dashboard's derivations, reads only what
that module allows and writes nothing (rsi.dashboard.write_dashboard writes the pages). Each page
is one self-contained HTML file with the same top bar and tabs; it reloads itself and keeps its
scroll position and opened sections across reloads. Workspace text is escaped; the run report
goes through rsi.markdown.to_html, which escapes it itself.
"""
import datetime
import html
import os

from rsi import dashboard as D
from rsi import timing
from rsi.markdown import to_html
from rsi.report import _num, _rounds
from rsi.rules import completed_rounds, is_number, stop_settings

OVERVIEW, ROUNDS, EVENTS, REPORT = D.PAGES
TABS = ((OVERVIEW, "概览"), (ROUNDS, "轮次与产物"), (EVENTS, "事件记录"), (REPORT, "运行报告"))
EVENTS_SHOWN = 200  # most recent events listed on the events page

KIND_NAMES = {"task": "任务定义", "trainset": "训练集", "valset": "验证集", "evalset": "评测集",
              "testset": "测试集", "candidate": "候选模型", "search": "调优搜索", "delivery": "交付包",
              "feedback": "执行反馈", "evaluation": "评测报告", "plan": "下一轮计划", "record": "实验记录"}
PRODUCER_NAMES = {"orchestrator": "顶层", "data": "数据", "train": "训练", "tune": "调优", "deliver": "交付",
                  "judge": "评测与决策"}

# Workflow diagram: node key, left x, centre y, width; title and second line per key.
NODES = (("task", 10, 90, 104), ("data", 144, 90, 116), ("train", 294, 50, 128), ("tune", 294, 130, 128),
         ("deliver", 456, 90, 124), ("judge", 610, 90, 132), ("check", 772, 90, 132), ("stop", 944, 90, 106))
NODE_H, FLOW_W, FLOW_H, LOOP_Y = 64, 1060, 236, 204
TITLES = dict(D.STEP_NAMES, task="任务接入", check="顶层核对", stop="停止与交付")
SUBTITLES = dict(((m, D._skill(m)) for m in D.MODULES), task="顶层", check="升级 · 停止 · 批准", stop="运行报告")
FILL = {"done": "#CEE5CC", "current": "#FBE5D6", "wait": "#FFF3C0", "fail": "#FFE3F1", "skip": "#FFFFFF",
        "todo": "#FFFFFF"}
LEGEND = (("done", "本轮已完成"), ("current", "当前步骤"), ("wait", "等你回答"), ("fail", "门没过"),
          ("todo", "未开始 / 本轮不经过"))


def _e(x):
    return html.escape("" if x is None else str(x))


def _href(page, target):
    """A link from `page` to `target`, both relative to the workspace root (PAGES are at most one
    directory deep): the bare name inside the same directory, else through ../ to the root."""
    if "/" in page:
        here = page.rsplit("/", 1)[0] + "/"
        if target.startswith(here):
            return target[len(here):]
        return "../" + target
    return target


def _art(ws, page, aid):
    """A registered id linked to its manifest (task.json for a task), relative to `page`."""
    name = "task.json" if ws.entry(aid)["kind"] == "task" else "manifest.json"
    return '<a href="%s">%s</a>' % (_e(_href(page, "artifacts/%s/%s" % (aid, name))), _e(aid))


def _when(ts):
    """A UTC time from STATE or the registry as a local datetime, or None."""
    dt = D._utc(ts)
    return None if dt is None else dt.astimezone()


def _last_change(ws):
    """Local time of the latest registration, approval or round decision, or None."""
    s = ws.state
    times = [e.get("registered_at") for e in ws.entries()] + [a.get("at") for a in s["approvals"]]
    times += [r["decision"].get("at") for r in s["rounds"].values() if r.get("decision")]
    times = [t for t in times if D._utc(t) is not None]
    return D._local(max(times)) if times else None


# ---- the overview ------------------------------------------------------------------------------

def _ask(questions):
    items = "".join("<li>%s%s</li>" % (_e(q["text"]), ' <span class="since">%s</span>'
                                       % _e("（自 %s 起）" % D._since(q["since"])) if q["since"] else "")
                    for q in questions)
    return ('<section id="ask" class="banner"><h2>%s</h2><ul>%s</ul><p class="since">%s</p></section>'
            % (_e("⏸ 等你在 Claude 里回答"), items, _e("页面只读，回答请回到 Claude 会话。")))


def _flow(states):
    box = dict((k, (x, y, w)) for k, x, y, w in NODES)

    def left(k):
        return box[k][0]

    def right(k):
        return box[k][0] + box[k][2]

    def centre(k):
        return box[k][0] + box[k][2] / 2.0

    def line(points, head=True):
        return '<polyline class="ln" points="%s"%s/>' % (" ".join("%g,%g" % p for p in points),
                                                         ' marker-end="url(#ah)"' if head else "")

    mid, top, low = box["data"][1], box["train"][1], box["tune"][1]
    fork, join = (right("data") + left("train")) / 2.0, (right("train") + left("deliver")) / 2.0
    out = ['<svg class="flow" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (FLOW_W, FLOW_H, _e("工作流")),
           '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker></defs>',
           line([(right("task"), mid), (left("data"), mid)]),
           line([(right("data"), mid), (fork, mid), (fork, top), (left("train"), top)]),
           line([(fork, mid), (fork, low), (left("tune"), low)]),
           line([(right("train"), top), (join, top), (join, mid), (left("deliver"), mid)]),
           line([(right("tune"), low), (join, low), (join, mid)], head=False),
           line([(right("deliver"), mid), (left("judge"), mid)]),
           line([(right("judge"), mid), (left("check"), mid)]),
           line([(right("check"), mid), (left("stop"), mid)]),
           '<text class="note" x="%g" y="%g">%s</text>' % ((right("check") + left("stop")) / 2.0, mid - 8, _e("停止")),
           line([(centre("check"), mid + NODE_H / 2.0), (centre("check"), LOOP_Y), (centre("data"), LOOP_Y),
                 (centre("data"), mid + NODE_H / 2.0)]),
           '<text class="note" x="%g" y="%d">%s</text>' % ((centre("check") + centre("data")) / 2.0, LOOP_Y + 18,
                                                           _e("继续：下一轮从计划指定的入口模块开始"))]
    for k, x, y, w in NODES:
        state, text = states[k]["state"], states[k]["text"]
        cx = x + w / 2.0
        out.append('<g id="n-%s" data-state="%s"><rect x="%g" y="%g" width="%g" height="%d" rx="10" fill="%s" '
                   'stroke="#222" stroke-width="%s"/>' % (k, state, x, y - NODE_H / 2.0, w, NODE_H, FILL[state],
                                                          "2.2" if state in ("current", "wait") else "1"))
        out.append('<text class="t1" x="%g" y="%g">%s</text>' % (cx, y - 9, _e(TITLES[k])))
        out.append('<text class="t2" x="%g" y="%g">%s</text>' % (cx, y + 7, _e(SUBTITLES[k])))
        out.append('<text class="t3" x="%g" y="%g">%s</text></g>' % (cx, y + 23, _e(text)))
    out.append("</svg>")
    legend = "".join('<span><i class="sw %s"></i>%s</span>' % (k, _e(label)) for k, label in LEGEND)
    return ('<section id="flow" class="card"><h2>工作流</h2><div class="flowwrap">%s</div>'
            '<div class="legend">%s</div></section>' % ("".join(out), legend))


def _meter(label, value, cap, text, alarm=True):
    """A labelled progress bar of value / cap; with `alarm`, highlighted once the cap (a limit) is
    reached. The target meter has no alarm: reaching the target is good news."""
    capped = is_number(cap) and cap > 0 and is_number(value)
    pct = max(0.0, min(100.0, 100.0 * value / cap)) if capped else 0.0
    return ('<div class="meter%s"><div class="lbl"><span>%s</span><span>%s</span></div>'
            '<div class="track"><div class="fill" style="width:%.0f%%"></div></div></div>'
            % (" hot" if alarm and capped and value >= cap else "", _e(label), _e(text), pct))


def _toward(value, target, higher):
    """How far `value` has come toward `target` (0..1)."""
    if (value >= target) if higher else (value <= target):
        return 1.0
    if higher:
        return value / target if target > 0 and value > 0 else 0.0
    return target / value if value > 0 and target > 0 else 0.0


def _spark(points, target):
    """A small line chart of the versions' primary metric [(version, value or None)]."""
    values = [v for _n, v in points if is_number(v)]
    if len(values) < 2:
        return ""
    span = values + ([target] if is_number(target) else [])
    lo, hi = min(span), max(span)
    pad = (hi - lo) * 0.15 or abs(hi) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    w, h, side, top, bottom = 280, 64, 14, 8, 18
    n = len(points)

    def x(i):
        return side + (w - 2 * side) * i / (n - 1.0)

    def y(v):
        return top + (hi - v) / (hi - lo) * (h - top - bottom)

    out = ['<svg class="spark" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (w, h, _e("各版本的主指标"))]
    if is_number(target):
        out.append('<line class="target" x1="%g" y1="%.1f" x2="%g" y2="%.1f"/>' % (side, y(target), w - side, y(target)))
    coords = [(x(i), y(v), name, v) for i, (name, v) in enumerate(points) if is_number(v)]
    out.append('<polyline class="ln" points="%s"/>' % " ".join("%.1f,%.1f" % (cx, cy) for cx, cy, _n, _v in coords))
    for cx, cy, name, v in coords:
        out.append('<circle cx="%.1f" cy="%.1f" r="3.2"><title>%s</title></circle>' % (cx, cy, _e("%s：%s" % (name, _num(v)))))
    for i in sorted(set([0, n - 1])):
        out.append('<text x="%.1f" y="%d">%s</text>' % (x(i), h - 4, _e(points[i][0])))
    out.append("</svg>")
    return "".join(out)


def _best_card(ws, task):
    s = ws.state
    m = task["metrics"]
    primary, higher = m["primary"], m["higher_is_better"]
    target = stop_settings(task)["target"]
    measure = "%s（评测集，%s）" % (primary, "越高越好" if higher else "越低越好")
    goal = "目标 %s" % _num(target) if is_number(target) else "目标未设"
    if s["best"] is None:
        body = '<div class="big none">还没有版本</div><div class="sub">%s</div>' % _e(
            "第一次升级后产生 V1 · %s · %s" % (measure, goal))
    else:
        value = ws.best_primary()
        best = ws.version(s["best"])
        body = '<div class="big">%s</div>' % _e("%s · %s" % (s["best"], _num(value)))
        body += '<div class="sub">%s %s</div>' % (_e("%s · %s ·" % (measure, goal)), _art(ws, OVERVIEW, best["evaluation_id"]))
        if is_number(value) and is_number(target):
            body += _meter("距目标", _toward(value, target, higher), 1, "%s / %s" % (_num(value), _num(target)),
                           alarm=False)
        points = []
        for v in s["versions"]:
            x = ws.manifest(v["evaluation_id"])["metrics"]["candidate"].get(primary)
            points.append((v["version"], x if is_number(x) else None))
        body += _spark(points, target)
    return '<section id="best" class="card"><h2>当前最佳</h2>%s</section>' % body


def _stop_card(ws, task):
    s = ws.state
    st = stop_settings(task)
    used, budget = s["budget_used"]["compute_hours"], task["budget"]["compute_hours"]
    attempts, limit = len(ws.attempts()), ws.attempt_limit()
    done = completed_rounds(s)
    body = _meter("已完成轮数", done, st["max_rounds"], "%d / %d" % (done, st["max_rounds"]))
    body += _meter("连续未升级", s["no_promotion_streak"], st["patience"],
                   "%d / %d" % (s["no_promotion_streak"], st["patience"]))
    body += _meter("本轮尝试", attempts, limit, "%d / %d" % (attempts, limit),
                   alarm=not ws.passed_candidates())  # used up with a candidate through the gate is no alarm
    body += _meter("评测集使用", s["eval_use_count"], task["eval_use_limit"],
                   "%d / %d" % (s["eval_use_count"], task["eval_use_limit"]))
    body += _meter("算力", used, budget, "%s / %s 小时" % (_num(used), _num(budget)) if is_number(budget)
                   else "%s 小时 · 无上限" % _num(used))
    if not s["stopped"] and ws.stop_check()[0]:
        body += '<div class="sub">%s</div>' % _e("停止条件已满足")
    return '<section id="stop" class="card"><h2>离停止还有多远</h2>%s</section>' % body


# Short phrases for the orchestrator's own commands (_status keys); a next step leads with its name.
LEADS = {"promote": "顶层核对：等 rsi promote 判定本轮", "close": "等关闭本轮（rsi close-round）",
         "milestone": "里程碑发布进行中：等里程碑的交付和评测登记"}


def _lead(ws, key, where, progress, questions):
    """What the run waits for in a few words; _status's sentence follows as the detail."""
    s = ws.state
    rnd = D._round(ws)
    if s["stopped"]:
        return "已停止：%s" % s["stopped"].get("reason")
    if key == "plan":
        return "等你批准计划 %s" % rnd["plan_id"]
    if questions:
        return "等你在 Claude 里回答上方的问题（%d 条）" % len(questions)
    if key in ("running", "handed"):
        state = "running" if key == "running" else "unregistered"
        mods = [m for m in D.MODULES if progress[m]["state"] == state]
        return "%s：%s %s" % (" / ".join(D.STEP_NAMES[m] for m in mods), "、".join(D._skill(m) for m in mods),
                             "运行中" if key == "running" else "已交回，待登记")
    if key in D.NEXT_STEPS:
        return "下一步：%s" % where
    if key in LEADS:
        return LEADS[key]
    if ws.stop_check()[0]:
        return "停止条件已满足：等 rsi next 结束运行"
    if rnd["plan_id"] is None:
        return "⑤ 评测与决策：等 rsi-judge 写计划"
    return "等进入第 %d 轮（rsi next）" % (s["round"] + 1)


def _now_card(ws, progress, questions):
    s = ws.state
    key, where, what = D._status(ws, progress)
    asked = [q["since"] for q in questions if q["since"]]
    if s["stopped"]:
        since = D._local(s["stopped"].get("at"))
        if os.path.isfile(os.path.join(ws.root, D.REPORT_FILE)):
            detail = '<a href="%s">查看运行报告 →</a>' % _href(OVERVIEW, REPORT)
        else:
            detail = _e("运行报告由顶层在停止后生成。")
    else:
        prefix = "下一步 %s：" % where  # a next step's sentence repeats the lead first: keep the action
        detail = _e(what[len(prefix):] if key in D.NEXT_STEPS and not questions and what.startswith(prefix)
                    else what)
        if key == "plan" or questions:
            since = D._since(min(asked, key=datetime.datetime.fromisoformat)) if asked else _last_change(ws)
        elif key in ("running", "handed"):
            since = None  # a running module's sentence names its start; a hand-back time is unknown
        else:
            since = _last_change(ws)
    return ('<section id="now" class="card"><h2>现在在等什么</h2><div class="wait-now">%s</div>'
            '<div class="detail">%s</div><div class="sub">%s</div></section>'
            % (_e(_lead(ws, key, where, progress, questions)), detail, _e("自 %s 起" % since) if since else ""))


def _overview(ws, task, progress, questions):
    parts = [_ask(questions)] if questions else []
    parts.append(_flow(D.flow_states(ws, progress)))
    parts.append('<div class="grid">%s%s%s</div>' % (_best_card(ws, task), _stop_card(ws, task),
                                                      _now_card(ws, progress, questions)))
    return "\n".join(parts)


# ---- rounds and artifacts -----------------------------------------------------------------------

def _gate(ws, e, primary):
    """(css class, short Chinese result) of an artifact's gate or role, ("", "") when it has none."""
    kind = e["kind"]
    if kind not in ("trainset", "candidate", "search", "evaluation", "plan"):
        return "", ""
    m = ws.manifest(e["id"])
    if kind == "trainset":
        if m["quality"]["passed"] is True:
            return "ok", "质量门通过"
        failed = [c["name"] for c in m["quality"]["checks"] if not c["passed"]]
        return "bad", "质量门没过" + ("（%s）" % "、".join(failed) if failed else "")
    if kind == "candidate":
        val = "val %s %s" % (primary, _num(m["metrics_val"].get(primary)))
        return ("ok", "训练门通过 · " + val) if m["train_gate"]["passed"] is True else ("bad", "训练门没过 · " + val)
    if kind == "search":
        delta = m["gain_vs_default"]["delta"]
        return "ok", "调优门通过 · 增益 %s" % ("%+.4g" % delta if is_number(delta) else "无")
    if kind == "evaluation":
        if m.get("uses_testset") is True:
            return "", "里程碑评测（test）"
        return "", "评测集 %s %s" % (primary, _num(m["metrics"]["candidate"].get(primary)))
    return "", "入口 %s" % m.get("entry_module")


def _chip(ws, e, primary):
    cls, text = _gate(ws, e, primary)
    note = ws.summary(e["id"])
    return '<span class="art"%s>%s<span class="k">%s</span>%s</span>' % (
        ' title="%s"' % _e(note) if note else "", _art(ws, ROUNDS, e["id"]), _e(KIND_NAMES.get(e["kind"], e["kind"])),
        '<span class="%s">%s</span>' % (cls, _e(text)) if text else "")


def _row(name, cell, cls=""):
    return '<div class="step"><div class="name">%s</div><div class="body%s">%s</div></div>' % (
        _e(name), " " + cls if cls else "", cell)


def _module_note(ws, module, progress):
    """In the current round: the module's output directory that runs or waits to be registered."""
    p = progress[module]
    stopped = ws.state["stopped"]
    if p["state"] == "running":
        text = "停止时未交回" if stopped else "%s 进行中：输出目录自 %s 起还没有交回产物" % (
            D._skill(module), D._since(p["since"]))
    elif p["state"] == "unregistered":
        text = ("停止时未交回：" if stopped else "交回未登记：") + "、".join(p["pending"])
    else:
        text = ""
    old = "；".join("未登记（已被 %s 取代）：%s" % (aid, name) for name, aid in sorted(p["superseded"].items()))
    text = "；".join(t for t in (text, old) if t)
    return '<div class="note">%s</div>' % _e(text) if text else ""


def _verdict(d):
    if d.get("decision") == "promote":
        return "升级 %s" % d.get("version")
    return "关闭（没有评测）" if d.get("evaluation_id") is None else "保留"


def _decision(ws, n, r, need_approval):
    d = r["decision"]
    if d is None:
        pending = ws.evaluations(n)
        text = "<b>%s</b>" % _e("未判定" + ("（%s 待判定）" % "、".join(pending) if pending else ""))
    else:
        bits = [_e(d.get("reason") or "—"), _e(D._local(d.get("at")))]
        ev = d.get("evaluation_id")
        if ev is not None:
            bits.append("评测 " + _art(ws, ROUNDS, ev))
        text = '<b>%s</b> <span class="sub">%s</span>' % (_e(_verdict(d)), " · ".join(bits))
    plan = r.get("plan_id")
    if plan is None:
        plan_html = "无"
    else:
        approval = ("已批准" if D._plan_approved(ws, plan) else "未批准") if need_approval else "无需批准"
        plan_html = "%s%s" % (_art(ws, ROUNDS, plan), _e("（%s）" % approval))
    return '%s<br><span class="sub">下一轮计划：%s</span>' % (text, plan_html)


def _timed(prefix, dur, suffix, start, cls):
    """A duration label. While it is still running it carries data-start (UTC ISO); the page script
    recomputes it from the browser clock on load, because the file is only rewritten by rsi commands."""
    if start is None:
        return '<span class="%s">%s</span>' % (cls, _e(prefix + dur + suffix))
    approx = dur.startswith("约")
    return '<span class="%s" data-start="%s" data-prefix="%s" data-suffix="%s"%s>%s</span>' % (
        cls, _e(start), _e(prefix), _e(suffix), ' data-approx="1"' if approx else "",
        _e(prefix + dur + suffix))


def _elapsed_label(ws, now):
    p = timing.elapsed_parts(ws, now)
    return "" if p is None else _timed(p[0], p[1], p[2], p[3], "chip time")


def _round_time_label(win, now, running_chip=False):
    """`running_chip`: the round's status chip already says 进行中 or the round is decided (it stays the
    current round until `rsi next`), so only "已 N 分钟" is shown."""
    p = timing.round_parts(win, now)
    if p is None:
        return ""
    prefix = "已 " if running_chip and p[0].startswith("进行中") else p[0]
    return _timed(prefix, p[1], "", p[2], "chip time")


def _round_block(ws, n, r, progress, primary, need_approval, latest, win, now):
    s = ws.state
    current = n == s["round"]
    arts = [ws.entry(a) for a in r["artifacts"]]
    d = r["decision"]
    if d is not None:
        verdict = ("ok" if d.get("decision") == "promote" else "", _verdict(d).split("（")[0])
    elif current:
        verdict = ("stop", "已停止") if s["stopped"] else ("run", "进行中")
    else:
        verdict = ("", "未判定")
    summary = ('<summary><span class="rn">第 %d 轮</span><span class="chip">%s</span><span class="chip %s">%s</span>'
               '<span class="chip">%s</span>%s</summary>'
               % (n, _e("入口 %s" % r.get("entry")), verdict[0], _e(verdict[1]),
                  _e("尝试 %d / %d" % (len(ws.attempts(n)), ws.attempt_limit())),
                  _round_time_label(win, now, d is not None or verdict[1] == "进行中")))
    body = []
    if current and D._attempts_exhausted(ws, r):
        body.append('<p class="warn">%s</p>' % _e(D._close_round_text(ws)))
    tasks = [e for e in arts if e["kind"] == "task"]
    if tasks:
        body.append(_row("任务接入", "".join(_chip(ws, e, primary) for e in tasks)))
    for m in D.MODULES:
        mine = D._step_artifacts(ws, m, arts)
        trials = [e for e in mine if m == "tune" and e["kind"] == "candidate"]
        cell = "".join(_chip(ws, e, primary) for e in mine if e not in trials)
        if trials:
            ends = trials[0]["id"] + ("" if len(trials) == 1 else " … " + trials[-1]["id"])
            cell += ('<details class="trials" id="round-%d-trials"><summary>%s</summary><div>%s</div></details>'
                     % (n, _e("%d 个调优试验候选（%s）" % (len(trials), ends)),
                        "".join(_chip(ws, e, primary) for e in trials)))
        if current:
            cell += _module_note(ws, m, progress)
        body.append(_row(D.STEP_NAMES[m], cell or '<span class="sub">本轮没有</span>'))
    others = [e for e in arts if e["kind"] not in ("task", "record") and e["producer"] not in D.MODULES]
    if others:
        body.append(_row("其他", "".join(_chip(ws, e, primary) for e in others)))
    records = [e for e in arts if e["kind"] == "record"]
    if records:
        body.append(_row("实验记录", "".join(_chip(ws, e, primary) for e in records)))
    body.append(_row("判定", _decision(ws, n, r, need_approval), "decision"))
    return '<details class="round" id="round-%d"%s>%s<div class="round-body">%s</div></details>' % (
        n, " open" if latest else "", summary, "".join(body))


def _rounds_page(ws, task, progress, now):
    rounds = _rounds(ws)
    wins = timing.round_windows(ws)
    need = "plan" in task["approvals"]
    primary = task["metrics"]["primary"]
    blocks = [_round_block(ws, n, r, progress, primary, need, n == rounds[-1][0], wins[n], now) for n, r in reversed(rounds)]
    return '<section id="rounds">%s</section>' % "\n".join(blocks)


# ---- events and the run report ------------------------------------------------------------------

def _is_trial(e):
    return e["kind"] == "candidate" and e["producer"] == "tune"


def _events_page(ws, task):
    """Registrations (a run of consecutive tuning trial candidates as one line), approvals, round
    decisions and the stop, newest first, grouped by local date."""
    s = ws.state
    primary = task["metrics"]["primary"]
    known = set(e["id"] for e in ws.entries())
    events = []  # (UTC time, dot class, HTML text, source)
    entries = ws.entries()
    i = 0
    while i < len(entries):
        e = entries[i]
        j = i
        while _is_trial(e) and j + 1 < len(entries) and _is_trial(entries[j + 1]):
            j += 1
        if j > i:
            run = entries[i:j + 1]
            text = "%s%s … %s%s" % (_e("登记 %d 个调优试验候选（" % len(run)), _art(ws, EVENTS, run[0]["id"]),
                                    _art(ws, EVENTS, run[-1]["id"]), _e("）"))
            events.append((run[-1].get("registered_at") or "", "trial", text, PRODUCER_NAMES["tune"]))
        else:
            _cls, gate = _gate(ws, e, primary)
            text = "%s %s%s" % (_e("登记" + KIND_NAMES.get(e["kind"], e["kind"])), _art(ws, EVENTS, e["id"]),
                                _e("（%s）" % gate) if gate else "")
            events.append((e.get("registered_at") or "", "record" if e["kind"] == "record" else "reg", text,
                           PRODUCER_NAMES.get(e["producer"], e["producer"])))
        i = j + 1
    for a in s["approvals"]:
        ref = a.get("ref")
        ref = " " + (_art(ws, EVENTS, ref) if ref in known else _e(ref)) if ref else ""
        text = "%s%s%s" % (_e("批准 %s" % a.get("what")), ref, _e("（%s：“%s”）%s" % (
            a.get("by"), a.get("quote"), "；用于 %s" % a["used_by"] if a.get("used_by") else "")))
        events.append((a.get("at") or "", "approve", text, "审批"))
    for n, r in _rounds(ws):
        d = r["decision"]
        if d is not None:
            text = "第 %d 轮判定：%s（%s）" % (n, _verdict(d), d.get("reason") or "—")
            events.append((d.get("at") or "", "decide", _e(text),
                           "rsi promote" if d.get("evaluation_id") is not None else "rsi close-round"))
    stopped = s["stopped"]
    if stopped:
        events.append((stopped.get("at") or "", "stop", _e("运行停止：%s" % stopped.get("reason")),
                       "rsi stop" if stopped.get("by") else "rsi next"))
    events.sort(key=lambda t: t[0])  # stable: same-second events keep their order
    events.reverse()
    items, day = [], None
    for at, cls, text, source in events[:EVENTS_SHOWN]:
        dt = _when(at)
        date = dt.strftime("%Y-%m-%d") if dt else "时间不明"
        if date != day:
            items.append('<li class="day">%s</li>' % _e(date))
            day = date
        items.append('<li><span class="t">%s</span><span class="dot %s"></span><span>%s <span class="who">· %s</span>'
                     '</span></li>' % (_e(dt.strftime("%H:%M:%S") if dt else "—"), cls, text, _e(source)))
    note = "" if len(events) <= EVENTS_SHOWN else '<p class="sub">%s</p>' % _e(
        "共 %d 条，只列最近的 %d 条。" % (len(events), EVENTS_SHOWN))
    return '<section id="events" class="card"><h2>事件记录（新的在前）</h2><ul class="tl">%s</ul>%s</section>' % (
        "".join(items), note)


def _report_page(ws):
    path = os.path.join(ws.root, D.REPORT_FILE)
    if not os.path.isfile(path):
        return '<section id="report" class="card"><p class="empty">还没有运行报告（停止时生成）。</p></section>'
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return ('<section id="report" class="card"><p class="sub">原文：<a href="%s">%s</a></p><div class="md">%s</div>'
            '</section>' % (_e(_href(REPORT, D.REPORT_FILE)), _e(D.REPORT_FILE), to_html(text)))


# ---- the page frame -----------------------------------------------------------------------------

STYLE = """
:root{--bg:#F5F6F8;--card:#FFF;--line:#E3E6EA;--text:#1F2328;--muted:#6B7280;--accent:#2878B5;
--ok:#2F7D32;--bad:#B42318}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 -apple-system,BlinkMacSystemFont,
"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC","Helvetica Neue",Arial,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
header.top{background:var(--card);border-bottom:1px solid var(--line)}
.bar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:18px 0 10px}
.title{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.title h1{font-size:22px;margin:0;font-weight:650;overflow-wrap:anywhere}
.ver{color:var(--muted);font-size:13px}
.chip{font-size:13px;padding:2px 10px;border-radius:999px;border:1px solid var(--line);color:#374151;
background:#FAFBFC;white-space:nowrap;font-weight:normal}
.chip.run,.chip.ok{background:#EAF5E9;border-color:#BFDDBB}.chip.stop{background:#F1F2F4}
.chip.ask{background:#FFF3C0;border-color:#E6D58A}
.chip.time{background:none;border-color:transparent;color:var(--muted);padding:2px 0}
.updated{color:var(--muted);font-size:13px;white-space:nowrap}
nav.tabs{display:flex;gap:26px;overflow-x:auto}
nav.tabs a{padding:10px 0 12px;color:#4B5563;border-bottom:2px solid transparent;font-size:14.5px;white-space:nowrap}
nav.tabs a.on{color:var(--text);border-bottom-color:var(--accent);font-weight:600}
nav.tabs a:hover{text-decoration:none;color:var(--text)}
main{padding:22px 0 36px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:18px}
.card h2{font-size:15px;margin:0 0 12px;color:#374151;font-weight:600}
.banner{background:#FFF3C0;border:1px solid #E6D58A;border-radius:12px;padding:14px 18px;margin-bottom:18px}
.banner h2{font-size:16px;margin:0 0 6px}
.banner ul{margin:0;padding-left:20px}.banner li{font-weight:600;margin:3px 0;overflow-wrap:anywhere}
.banner p{margin:6px 0 0}
.since{color:#6B5E1F;font-size:13px;font-weight:normal}
.flowwrap{overflow-x:auto}
svg.flow{width:100%;min-width:760px;height:auto;display:block}
svg.flow text{font-family:inherit;text-anchor:middle}
svg.flow .t1{font-size:14.5px;font-weight:600;fill:#1F2328}
svg.flow .t2{font-size:11.5px;fill:#555}
svg.flow .t3{font-size:11.5px;fill:#222}
svg.flow .note{font-size:12.5px;fill:#444}
svg .ln{fill:none;stroke:#222;stroke-width:1.6}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:#4B5563;margin-top:8px}
.sw{display:inline-block;width:14px;height:12px;border:1px solid #222;border-radius:3px;vertical-align:-1px;margin-right:5px}
.sw.done{background:#CEE5CC}.sw.current{background:#FBE5D6}.sw.wait{background:#FFF3C0}.sw.fail{background:#FFE3F1}
.sw.todo{background:#FFF}
.grid{display:grid;grid-template-columns:1fr 1.25fr 1fr;gap:18px}
.grid .card{margin-bottom:0}
.big{font-size:30px;font-weight:650;line-height:1.3}
.big.none{color:#9CA3AF}
.sub{color:var(--muted);font-size:13px;overflow-wrap:anywhere}
.meter{margin:9px 0}
.meter .lbl{display:flex;justify-content:space-between;gap:10px;font-size:13px;color:#374151}
.meter .track{height:7px;background:#EEF0F3;border-radius:99px;overflow:hidden;margin-top:4px}
.meter .fill{height:100%;background:#9DBFDF;border-radius:99px}
.meter.hot .fill{background:#E9A38F}
svg.spark{width:100%;max-width:280px;height:auto;display:block;margin-top:12px}
svg.spark .ln{stroke:#2878B5;stroke-width:2}
svg.spark circle{fill:#2878B5}
svg.spark .target{stroke:#2F7D32;stroke-dasharray:4 4}
svg.spark text{font-size:11px;fill:#6B7280;text-anchor:middle}
.wait-now{font-size:16px;font-weight:600;margin:2px 0 6px;overflow-wrap:anywhere}
.detail{color:var(--muted);font-size:13.5px;margin-bottom:6px;overflow-wrap:anywhere}
details.round{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:14px}
details.round>summary{cursor:pointer;list-style:none;padding:14px 20px;display:flex;gap:10px;align-items:center;
flex-wrap:wrap}
details.round>summary::-webkit-details-marker{display:none}
details.round>summary:before{content:"▸";color:var(--muted);width:10px}
details.round[open]>summary:before{content:"▾"}
.rn{font-weight:600}
.round-body{padding:0 20px 12px}
.step{display:grid;grid-template-columns:150px 1fr;gap:12px;padding:10px 0;border-top:1px solid var(--line)}
.step .name{font-weight:600;color:#374151}
.art{display:inline-flex;flex-wrap:wrap;gap:6px;align-items:center;border:1px solid var(--line);border-radius:8px;
padding:3px 9px;margin:0 6px 6px 0;font-size:13.5px;background:#FCFCFD}
.art .k{color:var(--muted)}
.ok{color:var(--ok)}.bad{color:var(--bad)}
.note{font-size:13px;color:#8A4B12;margin:2px 0 4px}
details.trials>summary{cursor:pointer;color:var(--muted);font-size:13px;margin:2px 0 6px}
.warn{background:#FFF3C0;border:1px solid #E6D58A;border-radius:8px;padding:8px 12px;margin:0 0 10px;font-size:14px}
.tl{list-style:none;margin:0;padding:0}
.tl li{display:grid;grid-template-columns:70px 14px 1fr;gap:10px;padding:8px 0;border-bottom:1px dashed var(--line);
overflow-wrap:anywhere}
.tl li.day{display:block;border:0;font-weight:600;color:#374151;padding:12px 0 2px}
.tl .t{color:var(--muted);font-variant-numeric:tabular-nums}
.tl .dot{width:10px;height:10px;border-radius:50%;margin-top:7px;border:1px solid #222}
.dot.reg{background:#CEE5CC}.dot.record{background:#EDEDED}.dot.trial{background:#D2E2FB}
.dot.approve{background:#FFF3C0}.dot.decide{background:#FBE5D6}.dot.stop{background:#C5C5D0}
.tl .who{color:var(--muted);font-size:13px}
.md{overflow-wrap:anywhere}
.md h1{font-size:22px;margin:4px 0 14px}
.md h2{font-size:17px;margin:24px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.md h3{font-size:15px;margin:18px 0 8px}.md h4{font-size:14px;margin:14px 0 6px}
.md code{background:#F1F3F5;padding:1px 5px;border-radius:4px;font-size:13px}
.md pre{background:#F6F8FA;border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow-x:auto}
.md pre code{background:none;padding:0}
.md ul{padding-left:22px}.md li{margin:3px 0}.md p{margin:8px 0}
.md .tablewrap{overflow-x:auto}
.md table{border-collapse:collapse;width:100%;min-width:760px;font-size:14px;margin:8px 0 14px}
.md th,.md td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
.md th{color:#4B5563;font-weight:600;background:#FAFBFC}
.empty{color:var(--muted);margin:4px 0}
footer{color:var(--muted);font-size:12.5px;text-align:center;padding:4px 0 30px;overflow-wrap:anywhere}
@media (max-width:860px){.grid{grid-template-columns:1fr}.step{grid-template-columns:1fr;gap:4px}
.bar{flex-direction:column;align-items:flex-start;gap:6px}.updated{white-space:normal}}
@media (max-width:600px){.wrap{padding:0 16px}.card{padding:16px}}
"""

# Keeps the scroll position and the sections the reader opened or closed across the automatic
# reloads (a section left as rendered follows the page: the newest round opens by itself), and
# scrolls a diagram wider than its card (phone width) to the first current or waiting node.
SCRIPT = """
(function(){
var path=location.pathname,key="rsi-dashboard-scroll:"+path;
function get(k){try{return sessionStorage.getItem(k);}catch(e){return null;}}
function put(k,v){try{if(v===null){sessionStorage.removeItem(k);}else{sessionStorage.setItem(k,v);}}catch(e){}}
var ds=document.querySelectorAll("details[id]");
for(var i=0;i<ds.length;i++){(function(d){
var k="rsi-dashboard-open:"+path+"#"+d.id,rendered=d.open,v=get(k);
if(v!==null){d.open=(v==="1");}
d.addEventListener("toggle",function(){put(k,d.open===rendered?null:(d.open?"1":"0"));});
})(ds[i]);}
try{var w=document.querySelector(".flowwrap"),n=w&&w.querySelector('[data-state="current"],[data-state="wait"]');
if(n&&w.scrollWidth>w.clientWidth){var a=w.getBoundingClientRect(),b=n.getBoundingClientRect();
if(b.left<a.left||b.right>a.right){w.scrollLeft+=b.left-a.left-(a.width-b.width)/2;}}}catch(e){}
function restore(){var y=get(key);if(y!==null){window.scrollTo(0,parseInt(y,10)||0);}}
function save(){put(key,String(window.scrollY||window.pageYOffset||0));}
function fmt(s){s=Math.max(0,Math.floor(s));if(s<60){return "不到 1 分钟";}
if(s<3600){return Math.floor(s/60)+" 分钟";}
if(s<86400){var h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h+" 小时"+(m?" "+m+" 分":"");}
return Math.floor(s/86400)+" 天 "+Math.floor(s%86400/3600)+" 小时";}
try{var ts=document.querySelectorAll("[data-start]");
for(var j=0;j<ts.length;j++){var t=ts[j],b=Date.parse(t.getAttribute("data-start"));
if(!isNaN(b)){t.textContent=(t.getAttribute("data-prefix")||"")+(t.getAttribute("data-approx")?"约 ":"")
+fmt((Date.now()-b)/1000)+(t.getAttribute("data-suffix")||"");}}}catch(e){}
restore();window.addEventListener("load",restore);
window.addEventListener("scroll",save);window.addEventListener("pagehide",save);
})();
"""


def _frame(ws, page, title, body, now, questions):
    """A whole page: head, top bar with the status label and tabs, `body`, footer."""
    s = ws.state
    if questions:  # once stopped only the notes can hold questions
        label = '<span class="chip ask">%s</span>' % ("⏸ 等你回答（已停止）" if s["stopped"] else "⏸ 等你回答")
    elif s["stopped"]:
        label = '<span class="chip stop">■ 已停止</span>'
    else:
        label = '<span class="chip run">● 运行中</span>'
    tabs = "".join('<a%s href="%s">%s</a>' % (' class="on"' if rel == page else "", _e(_href(page, rel)), _e(name))
                   for rel, name in TABS)
    head = "%s%s · %s" % ("⏸ 需要你回答 · " if questions else "", title, s["task_id"])
    return "".join([
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n",
        '<meta http-equiv="refresh" content="%d">\n' % D.REFRESH_SECONDS,
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
        "<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n" % (_e(head), STYLE),
        '<header class="top"><div class="wrap"><div class="bar"><div class="title"><h1>%s</h1>'
        '<span class="ver">%s</span><span class="chip">%s</span>%s%s</div><div class="updated">%s</div></div>'
        '<nav class="tabs">%s</nav></div></header>\n'
        % (_e(s["task_id"]), _e("v%d" % s["task_version"]), _e("第 %d 轮" % s["round"]), label, _elapsed_label(ws, now),
           _e("更新于 %s · 每 %d 秒自动刷新" % (D._show(now), D.REFRESH_SECONDS)), tabs),
        '<main><div class="wrap">\n%s\n</div></main>\n' % body,
        "<footer><div class=\"wrap\">%s</div></footer>\n" % _e(
            "由 rsi 生成 · 只读页面，回答和操作都在 Claude 里进行 · 工作区 %s" % ws.root),
        "<script>%s</script>\n</body>\n</html>\n" % SCRIPT,
    ])


def render_pages(ws, now=None):
    """The four pages of workspace `ws`: PAGES entry -> HTML text. `now` (a datetime) is the
    generation time shown (default: now). Reads only (see rsi.dashboard), writes nothing."""
    now = datetime.datetime.now() if now is None else now
    task = ws.task()
    progress = D.module_progress(ws)
    questions = D.pending_questions(ws, progress)
    bodies = ((OVERVIEW, "RSI 进度", _overview(ws, task, progress, questions)),
              (ROUNDS, "轮次与产物", _rounds_page(ws, task, progress, now)),
              (EVENTS, "事件记录", _events_page(ws, task)),
              (REPORT, "运行报告", _report_page(ws)))
    return dict((rel, _frame(ws, rel, title, body, now, questions)) for rel, title, body in bodies)
