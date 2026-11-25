"""What the run is doing now, for the status lines of `rsi run`, the first line of `rsi status` and the pages.

open_dispatches(ws) lists the dispatches still open (OPEN_STATUSES: not launched yet, an agent running, a long job, a
hand-back not processed yet) with what the pages and the status line show of each: its stage, launch, check-ins, long
job and the module's progress.json (rsi.progress). now_line(ws) says it in one line (spec §6):

  20:23 第1轮 ② 训练调优 d-0001 · 第 3 批：3 个训练在跑（卡 0、4、6） · 配置 3/3 · GPU 小时 1.3/8 · 当前方案 c43 · 下次检查 20:46

local HH:MM and the round, then for each open dispatch 「stage id · what it does · its first 3 facts · next check-in
HH:MM」, joined by 「 ｜ 」; after a stop, the overview's stopped lead comes first (what the stop has yet to end).
What it does: 「等你回答（q-NNNN）」 while STATE waits on a question tied to the dispatch
(its ref or dispatch_id: a hand-back's question, an over_budget hold, the check-in cap; then no next check-in: ruling
R5, the line must show that the run waits on the user), else the progress summary (marked when stale), else said from
its status. With no open dispatch: 「等你回答（q-NNNN）」 while STATE waits on a question, the number of questions the
pages list (notes.md's too) while there are some, else the overview's lead (rsi.dashboard.now_lead; its question form
points at the page). now_key(ws) is what a status line changes with: the engine prints one after a step only when
it differs from the last line's (a heartbeat brings the numbers of progress.json, which are no change).

For the overview's other live panels: handback_notes(ws) lists the latest hand-backs (each dispatch's handback.json and
the copies handback-<n>.json the engine keeps, newest file first) with their notes and questions (「agent 的判断」);
round_best_val(ws) is the current round's best registered candidate on the validation set (「当前最佳」). Reads only,
never raises on a module's file.
"""
import datetime
import json
import os
import re
import stat

from rsi import dashboard as D
from rsi import flow as F
from rsi import progress as P
from rsi.rules import flow_settings, is_number
from rsi.workspace import parse_utc

OPEN_STATUSES = ("pending", "running", "long_job", "handed_back")
# The launch modes (a dispatch record's launch_mode) in words.
MODE_WORDS = {"first": "首次", "resume": "退回", "continue": "长任务收尾", "followup": "第二部分", "answer": "送回回答",
              "checkin": "定时检查", "fresh": "新会话"}
FACTS_SHOWN = 3  # the facts of progress.json a status line shows (the module puts the important ones first)
SEP = " ｜ "
# A dispatch's hand-backs in its output directory: handback.json (the current launch's) and the copies the engine
# keeps when the next launch starts (handback-<n>.json, handback-<n>-<k>.json for a second copy of launch n).
HANDBACK_FILE = re.compile(r"^handback(?:-(\d+)(?:-\d+)?)?\.json$")
HANDBACK_MAX_BYTES = 256 * 1024  # a larger hand-back is not read (skipped)


def _waiting(ws):
    """STATE waiting while the run really waits on it (it has an id, the run has not stopped and it is not stale:
    rsi.flow.stale_waiting), else None."""
    w = ws.waiting()
    if not w or not w.get("id") or ws.state["stopped"] or F.stale_waiting(ws, w):
        return None
    return w


def _question_id(w, d):
    """The id of question `w` (_waiting) when it is tied to dispatch `d` (its ref or dispatch_id is d's id: a
    hand-back's question, an over_budget hold, the check-in cap, a decision asked about it), else None."""
    if w and d.get("id") and d.get("id") in (w.get("ref"), w.get("dispatch_id")):
        return w["id"]
    return None


def open_dispatches(ws, now=None):
    """The open dispatches (OPEN_STATUSES), in dispatch order: [{"id", "round", "module", "stage", "stage_name",
    "skill", "status", "launches", "mode" (launch_mode), "since" (when its latest launch started, aware; None while
    pending), "returns", "checkins", "checkin_limit" (None without check-ins, or once the user answered wait),
    "checkin_due" (aware; only for a long job whose marker is not there), "marker" (absolute), "marker_done" (it is
    there: rsi.flow.marker_done), "question_id" (of the question STATE waits on when it is tied to the dispatch),
    "progress" (rsi.progress.dispatch_progress at `now`, or None)}]."""
    checkins_on = bool(flow_settings(ws.task())["checkin_minutes"])
    w = _waiting(ws)
    out = []
    for d in ws.dispatches():
        status = d.get("status")
        if status not in OPEN_STATUSES:
            continue
        module = d.get("module")
        stage = F.stage_of(module)
        done = F.marker_done(ws, d)
        job_open = status == "long_job" and not done
        out.append({
            "id": d.get("id"), "round": d.get("round"), "module": module, "stage": stage, "stage_name": D.STEP_NAMES.get(stage, module),
            "skill": "rsi-%s" % module, "status": status, "launches": int(d.get("launches") or 1),
            "mode": d.get("launch_mode") or "first",
            "since": None if status == "pending" else parse_utc(d.get("launched_at")) or parse_utc(d.get("started_at")),
            "returns": int(d.get("returns") or 0), "checkins": int(d.get("checkins") or 0),
            "checkin_limit": F.checkin_limit(ws, d) if checkins_on else None,
            "checkin_due": F.checkin_due(ws, d) if job_open else None,
            "marker": F.marker_path(ws, d), "marker_done": done, "question_id": _question_id(w, d),
            "progress": P.dispatch_progress(ws, d, now)})
    return out


def now_key(ws):
    """What a status line changes with: (round, stopped, the id of the question STATE waits on, next_action's type,
    step and dispatch id(s), (id, status, launches) of each open dispatch)."""
    a = F.next_action(ws)
    w = ws.waiting() or {}
    return (ws.state["round"], bool(ws.state["stopped"]), w.get("id"), a["type"], a.get("step"),
            a.get("dispatch_id") or tuple(a.get("dispatch_ids") or ()),
            tuple((d.get("id"), d.get("status"), d.get("launches")) for d in ws.dispatches()
                  if d.get("status") in OPEN_STATUSES))


def _doing(x):
    """What open dispatch `x` does, in words: the question it waits on, else its progress summary, else said from its
    status."""
    if x["question_id"]:
        return "等你回答（%s）" % x["question_id"]
    p = x["progress"]
    if p and p.get("summary"):
        return p["summary"] + ("（进度 %d 分钟没有更新）" % int(p["age_s"] // 60) if p.get("stale") else "")
    if x["status"] == "pending":
        return "待启动"
    if x["status"] == "running":
        mode = "" if x["mode"] == "first" else " · %s" % MODE_WORDS.get(x["mode"], x["mode"])
        return "%s 运行中（第 %d 次启动%s）" % (x["skill"], x["launches"], mode)
    if x["status"] == "long_job" and x["marker"]:
        if not x["marker_done"]:
            return "长任务在跑，等完成标记 %s" % os.path.basename(x["marker"])
        return "长任务已完成，等 rsi run 收尾"
    return "已交回，等登记"


def _segment(x):
    parts = ["%s %s" % (x["stage_name"], x["id"]), _doing(x)]
    p = x["progress"]
    parts += ["%s %s" % fact for fact in ((p or {}).get("facts") or [])[:FACTS_SHOWN]]
    if x["checkin_due"] is not None and not x["question_id"]:
        parts.append("下次检查 %s" % x["checkin_due"].astimezone().strftime("%H:%M"))
    return " · ".join(parts)


def _idle(ws):
    """The line's text when no dispatch is open: the question STATE waits on, else the questions the pages list (the
    milestone before it is asked, notes.md's), else the overview's lead."""
    w = _waiting(ws)
    if w:
        return "等你回答（%s）" % w["id"]
    progress = D.module_progress(ws)
    questions = D.pending_questions(ws, progress)
    view = D._view(ws, progress)
    if questions and not ws.state["stopped"] and view[0] != "plan":  # the lead's 「上方的问题」 points at the page
        return "等你在 Claude 里回答 %d 个问题（进度页面上列着）" % len(questions)
    return D.now_lead(ws, progress, questions, view)


def now_line(ws, now=None):
    """The status line (spec §6; see the module docstring) at `now` (a datetime, local when naive; default now): one
    line, whatever the texts it quotes hold."""
    now = (now or datetime.datetime.now()).astimezone()
    head = "%s 第%d轮 " % (now.strftime("%H:%M"), ws.state["round"])
    ds = open_dispatches(ws, now)
    text = SEP.join(_segment(x) for x in ds) if ds else _idle(ws)
    if ds and ws.state["stopped"]:  # the open dispatches are what the stop has yet to end: say it has stopped
        text = D.stopped_texts(ws, F.next_action(ws))[0] + SEP + text
    return head + re.sub(r"\s*[\r\n]+\s*", " ", text)


def _handback(path):
    """The hand-back file at `path` as {"status", "notes", "questions"} (a field of the wrong type: None, "" or []),
    or None when it cannot be read, is larger than HANDBACK_MAX_BYTES or is no JSON object."""
    try:
        with open(path, "rb") as f:
            raw = f.read(HANDBACK_MAX_BYTES + 1)
        if len(raw) > HANDBACK_MAX_BYTES:
            return None
        h = json.loads(raw.decode("utf-8"))
    except Exception:  # unreadable, not UTF-8, not JSON, nested too deep: skipped
        return None
    if not isinstance(h, dict):
        return None
    notes, questions = h.get("notes"), h.get("questions")
    return {"status": h.get("status") if isinstance(h.get("status"), str) else None,
            "notes": notes if isinstance(notes, str) else "",
            "questions": [q for q in questions if isinstance(q, str) and q] if isinstance(questions, list) else []}


def handback_notes(ws, limit=6):
    """The latest `limit` hand-backs of all dispatches, newest file first: [{"dispatch_id", "module", "launch" (the
    launch that wrote it: n of handback-<n>.json, the dispatch's launches for handback.json), "at" (the file's
    modification time, aware UTC), "status" (ok, long_job, failed as written, or None), "notes", "questions"}]. Only
    the directory listing of each output directory and the hand-back files themselves are read; a file that cannot
    be read is skipped."""
    found = []
    for d in ws.dispatches():
        if not d.get("outdir"):
            continue
        base = os.path.join(ws.root, d["outdir"])
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for name in names:
            m = HANDBACK_FILE.match(name)
            if not m:
                continue
            path = os.path.join(base, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                launch = int(m.group(1)) if m.group(1) else int(d.get("launches") or 1)
                found.append((st.st_mtime, launch, d, path))
    found.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out = []
    for mtime, launch, d, path in found:
        if len(out) >= limit:
            break
        h = _handback(path)
        if h is not None:
            h.update({"dispatch_id": d.get("id"), "module": d.get("module"), "launch": launch,
                      "at": datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc)})
            out.append(h)
    return out


def round_best_val(ws):
    """(candidate id, value) of the candidate registered in the current round whose validation primary metric
    (metrics_val) is the best by the task's direction, among those where it is a number; None without one. On a
    tie the earlier registered wins."""
    m = ws.task()["metrics"]
    primary, higher = m["primary"], m["higher_is_better"]
    best = None
    s = ws.state
    for aid in s["rounds"][str(s["round"])]["artifacts"]:
        if ws.entry(aid)["kind"] != "candidate":
            continue
        v = (ws.manifest(aid).get("metrics_val") or {}).get(primary)
        if is_number(v) and (best is None or (v > best[1] if higher else v < best[1])):
            best = (aid, v)
    return best
