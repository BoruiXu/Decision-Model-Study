"""The progress pages: where the run is, what it waits for and what happened so far.

PAGES are the overview <ws>/dashboard.html and three secondary pages under <ws>/dashboard/
(rounds, events, run report); rsi.dashboard_pages renders them from the derivations here
(module_progress, waiting_for, pending_questions, flow_states). Rendering reads only STATE.json,
registry.jsonl, registered manifests (task.json for a task), <ws>/REPORT.md, the `## 待用户回答`
section of <ws>/work/notes.md, the directory listing of the current round's work/round-<n>/ and the
manifest.json of each unregistered hand-back there (for its kind); it opens no data row file and
writes nothing. write_dashboard writes each page like
Workspace.save_state (temp file in the page's directory, then os.replace), so a browser never
reads half a page. The pages reload themselves every REFRESH_SECONDS seconds.

Module output directories (module_progress): the orchestrator gives each module the output
directory <ws>/work/round-<n>/<module>/, with a suffix when the module is dispatched again in
the round (train-2). Inside it each artifact is a subdirectory holding a manifest.json. A
directory from which nothing was handed back yet (no artifact subdirectory, no registered
source inside it) means the module is running; an artifact subdirectory that is no registered
artifact's source is handed back but not registered; otherwise the module is done (an evalset
or testset leaves an empty directory behind once registration has moved it away).
"""
import datetime
import json
import os
import tempfile

from rsi.report import _num
from rsi.workspace import DATASETS

# The pages, relative to the workspace root (forward slashes): the overview first.
PAGES = ("dashboard.html", "dashboard/rounds.html", "dashboard/events.html", "dashboard/report.html")
DASHBOARD_FILE = PAGES[0]
PAGES_DIR = "dashboard"
REFRESH_SECONDS = 5
MODULES = ("data", "train", "tune", "deliver", "judge")
REPORT_FILE = "REPORT.md"
NOTES_FILE = os.path.join("work", "notes.md")
NOTES_SECTION = "## 待用户回答"

STEP_NAMES = {"data": "① 数据", "train": "② 训练", "tune": "③ 调优", "deliver": "④ 交付", "judge": "⑤ 评测与决策"}
# Main-flow position of each module (② and ③ share one) and of a round's entry module.
STEP_INDEX = {"data": 0, "train": 1, "tune": 1, "deliver": 2, "judge": 3}
AFTER_STEPS = max(STEP_INDEX.values()) + 1  # the position of a round past ⑤ (waiting for 顶层核对)
BUSY = ("running", "unregistered")  # module_progress states of a module whose output is not in yet


def _skill(module):
    return "rsi-" + module


# ---- module output directories ---------------------------------------------------------------

def _module_of(name):
    """The module whose output directory `name` is: the module name, or the module name followed
    by a character that is not a letter (train-2, train_2, train2; not trainer); else None."""
    for m in MODULES:
        if name == m or (name.startswith(m) and not name[len(m)].isalpha()):
            return m
    return None


def _subdirs(path):
    """Names of the visible subdirectories of `path`, sorted. Hidden ones (for example a hand-over
    leftover .<name>.rsi-handed-over-<hex>) are no module output."""
    return sorted(n for n in os.listdir(path) if not n.startswith(".") and os.path.isdir(os.path.join(path, n)))


def _inside(child, parent):
    return os.path.commonpath([child, parent]) == parent


def _iso(mtime):
    """A file time as local ISO 8601 with its UTC offset, to the second."""
    return datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def module_progress(ws):
    """Each module's output in the current round, from the directories under work/round-<n>/.

    Returns module -> {"state", "since", "pending", "superseded"}. state: "running" when one of its
    directories has handed nothing back yet (since: the earliest such directory's modification time,
    ISO); else "unregistered" when an artifact subdirectory is no registered artifact's source
    (paths compared with os.path.realpath) and not superseded; else "done" when it has a directory;
    else "idle". pending lists those subdirectory names, sorted. superseded maps the name of an
    unregistered subdirectory (a refused hand-back) to the id of the artifact of the same kind
    (its manifest's kind) this module registered in the round afterwards: that registration replaced
    it, so it no longer waits to be registered. "Afterwards" is read from the directory's mtime: the
    entry's registered_at (whole seconds) is not before the mtime's second. A refused directory is
    never touched again, so its mtime is when the module handed it back; an unreadable manifest or
    an artifact registered earlier never supersedes.
    """
    base = os.path.join(ws.root, "work", "round-%d" % ws.state["round"])
    sources = set(os.path.realpath(e["source"]) for e in ws.entries())
    dirs = dict((m, []) for m in MODULES)
    if os.path.isdir(base):
        for name in _subdirs(base):
            m = _module_of(name)
            if m is not None:
                dirs[m].append(os.path.join(base, name))
    rnd = _round(ws)
    out = {}
    for m in MODULES:
        waiting, pending, superseded = [], [], {}
        for d in dirs[m]:
            real = os.path.realpath(d)
            arts = [n for n in _subdirs(d) if os.path.isfile(os.path.join(d, n, "manifest.json"))]
            for n in arts:
                if os.path.realpath(os.path.join(d, n)) in sources:
                    continue
                later = _superseding(ws, rnd, m, os.path.join(d, n), real)
                if later is None:
                    pending.append(n)
                else:
                    superseded[n] = later
            if not arts and not any(_inside(s, real) for s in sources):
                waiting.append(os.path.getmtime(d))
        if waiting:
            state, since = "running", _iso(min(waiting))
        else:
            state, since = ("unregistered" if pending else "done" if dirs[m] else "idle"), None
        out[m] = {"state": state, "since": since, "pending": sorted(pending), "superseded": superseded}
    return out


def _superseding(ws, rnd, module, path, outdir):
    """The id of the first artifact of the manifest's kind that `module` registered in the round after
    the directory `path` was written (see module_progress), or None. Only an artifact registered from
    outside `outdir` (the real path of the module output directory holding `path`) counts: it came from a
    re-dispatch. A sibling registered from the same directory is part of the same hand-back."""
    try:
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            kind = json.load(f).get("kind")
        written = int(os.path.getmtime(path))
    except (OSError, ValueError, AttributeError):
        return None
    for a in rnd["artifacts"]:
        e = ws.entry(a)
        at = _utc(e.get("registered_at"))
        if e["producer"] == module and e["kind"] == kind and at is not None and at.timestamp() >= written \
                and os.path.dirname(os.path.realpath(e["source"])) != outdir:
            return a
    return None


# ---- what the run waits for -------------------------------------------------------------------

def _round(ws):
    s = ws.state
    return s["rounds"][str(s["round"])]


def _plan_approved(ws, plan):
    """True when STATE records a plan approval with ref `plan` (the rule of `rsi status`)."""
    return any(a.get("what") == "plan" and a.get("ref") == plan for a in ws.state["approvals"])


def _decision_text(d):
    """promote V1（evaluation-0001）, keep（evaluation-0002）, or keep（关闭，没有评测）."""
    if d.get("evaluation_id") is None:
        return "%s（关闭，没有评测）" % d.get("decision")
    return "%s%s（%s）" % (d.get("decision"), " " + d["version"] if d.get("version") else "", d["evaluation_id"])


def _show(dt):
    """A datetime as local time; a naive one is taken as local already."""
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _local(ts):
    """A UTC time from STATE or the registry (2026-11-22T01:02:03Z) as local time."""
    try:
        dt = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError):
        return "无" if not ts else str(ts)
    return _show(dt)


def _since(iso):
    return _show(datetime.datetime.fromisoformat(iso))


def _data_missing(ws, arts):
    """Why step ① is not complete in the current round, or None: in round 1 a current
    testset, evalset, valset or trainset is missing; in any round no trainset of this round
    passed its quality gate (one that failed is evidence only, never current)."""
    if ws.state["round"] == 1:
        missing = [k for k in reversed(DATASETS) if ws.current(k) is None]
        if missing:
            return "第一轮还缺 %s" % "、".join(missing)
    if not any(e["kind"] == "trainset" and ws.manifest(e["id"])["quality"]["passed"] is True for e in arts):
        return "本轮还没有通过质量门的训练集"
    return None


def _attempts_exhausted(ws, rnd):
    """True when the round's attempts are used up with no candidate through the train gate, the
    round is undecided, has no evaluation and the run is not stopped: only `rsi close-round` is left."""
    return (len(ws.attempts()) >= ws.attempt_limit() and not ws.passed_candidates() and rnd["decision"] is None
            and not ws.evaluations() and not ws.state["stopped"])


def _close_round_text(ws):
    return ("第 %d 轮的 %d 次尝试已用完，没有候选通过训练门：等顶层运行 rsi close-round <ws> --reason <理由>，"
            "再派 rsi-judge 依据本轮的候选和记录写计划。" % (ws.state["round"], ws.attempt_limit()))


PASSED_SHOWN = 3  # candidate ids named in the deliver step's sentence (the rounds page lists them all)


def _passed_text(ws):
    """The round's candidates through the train gate, for the deliver step: train-produced first,
    the count and at most PASSED_SHOWN ids; "" when there is none."""
    passed = sorted(ws.passed_candidates(), key=lambda a: ws.entry(a)["producer"] != "train")
    if not passed:
        return ""
    if len(passed) == 1:
        return "（本轮通过训练门的候选：%s）" % passed[0]
    return "（本轮通过训练门的候选 %d 个：%s%s；按验证集主指标选最好的送评）" % (
        len(passed), "、".join(passed[:PASSED_SHOWN]), " 等" if len(passed) > PASSED_SHOWN else "")


def _own_delivery(ws, arts):
    """True when this round has the delivery step ④ asks for: a delivery of one of the round's own
    candidates (the best version's re-delivery on a refreshed evalset, whose candidate is from an
    earlier round, does not count); in a round with no candidate of its own (entry deliver) any
    delivery; a milestone delivery (predictions_test) always."""
    cands = set(e["id"] for e in arts if e["kind"] == "candidate")
    for e in arts:
        if e["kind"] == "delivery":
            m = ws.manifest(e["id"])
            if not cands or m["candidate_id"] in cands or "predictions_test" in m["files"]:
                return True
    return False


def _redeliver_pending(ws):
    """After a refreshed evalset the best version has no delivery on the new evalset yet (the rule of
    `rsi status`): ④ also re-delivers it, the evaluation compares the candidate with it there."""
    return ws.state["best"] is not None and ws.best_delivery_on_current_evalset() is None


def _delivered(ws, arts):
    """True when step ④ is done in an undecided round: the round's own delivery is registered and
    no re-delivery of the best version is owed."""
    return _own_delivery(ws, arts) and not _redeliver_pending(ws)


def _redeliver_sentence(ws, own):
    """The ④ sentence when the best version must be re-delivered on the refreshed evalset first."""
    return ("%s评测集已刷新，先把当前最佳版本 %s 在新评测集上重新交付（它在新评测集上还没有 delivery）。"
            % ("本轮候选的交付已登记，" if own else "", ws.state["best"]))


def _next_step(ws, rnd):
    """(key, step, sentence) for an undecided round with no module running or waiting to be
    registered and attempts left; key is the module whose step is next (data, train, tune, deliver,
    judge) or "promote" when the round's evaluation waits for `rsi promote`. tune: the round's tuning
    trial candidates are registered and their search is not (rsi-tune writes it before ④)."""
    evaluations = ws.evaluations()
    if evaluations:
        return ("promote", "顶层核对", "本轮评测已登记：等顶层运行 rsi promote <ws> %s 判定本轮（升级或保留）。"
                % evaluations[-1])
    arts = [ws.entry(a) for a in rnd["artifacts"]]
    start = STEP_INDEX.get(rnd["entry"], 0)
    if start == 0:
        why = _data_missing(ws, arts)
        if why:
            return "data", STEP_NAMES["data"], "下一步 ① 数据：派 rsi-data（%s）。" % why
    if start <= 1 and not ws.passed_candidates():
        return ("train", "② 训练 / ③ 调优", "下一步 ② 训练 / ③ 调优：派 rsi-train 或 rsi-tune（本轮还没有通过训练门的候选，"
                "已用尝试 %d / %d）。" % (len(ws.attempts()), ws.attempt_limit()))
    trials = any(e["kind"] == "candidate" and e["producer"] == "tune" for e in arts)
    if start <= 1 and trials and not any(e["kind"] == "search" for e in arts):
        return ("tune", STEP_NAMES["tune"], "下一步 ③ 调优：等 rsi-tune 写 search（试验候选已登记，本轮的 search 还没有登记），"
                "写好后再交付。")
    if start <= 2 and not _delivered(ws, arts):
        own = _own_delivery(ws, arts)
        pending = _redeliver_pending(ws)
        head = "派 rsi-deliver：" if own else "派 rsi-deliver 打包送评的候选%s。" % _passed_text(ws)
        return ("deliver", STEP_NAMES["deliver"], "下一步 ④ 交付：%s%s"
                % (head, _redeliver_sentence(ws, own) if pending else ""))
    return "judge", STEP_NAMES["judge"], "下一步 ⑤ 评测与决策：派 rsi-judge 在评测集上评测本轮的交付。"


def _status(ws, progress):
    """(key, where the run is, what it waits for): the first that holds of stopped, a module
    running, a module's output waiting to be registered, the round's plan waiting for approval
    (not once the round is decided and the stop check says stop: `rsi next` then stops without a
    plan), the attempts used up (`rsi close-round`), the next main-flow step of an undecided round,
    and for a decided round the plan and `rsi next`. key names the case: stopped, plan, running,
    handed, close, one of _next_step's keys, milestone (a milestone release under way) or decided."""
    s = ws.state
    stopped = s["stopped"]
    if stopped:
        if os.path.isfile(os.path.join(ws.root, REPORT_FILE)):
            then = "运行报告见 REPORT.md。"
        else:
            then = "等顶层生成运行报告（rsi report <ws> --out <ws>/REPORT.md）并交付。"
        return "stopped", "已停止", "已停止：%s。%s" % (stopped.get("reason"), then)
    rnd = _round(ws)
    plan = rnd["plan_id"]
    running = [m for m in MODULES if progress[m]["state"] == "running"]
    if running:
        first = min(datetime.datetime.fromisoformat(progress[m]["since"]) for m in running)
        return ("running", " / ".join(STEP_NAMES[m] for m in running), "等 %s 交回（输出目录自 %s 起还没有交回产物）。"
                % ("、".join(_skill(m) for m in running), _show(first)))
    handed = [m for m in MODULES if progress[m]["state"] == "unregistered"]
    if handed:
        return ("handed", " / ".join(STEP_NAMES[m] for m in handed), "%s：等顶层校验、抽查后登记（rsi register）。"
                % "；".join("%s 交回了 %s" % (_skill(m), "、".join(progress[m]["pending"])) for m in handed))
    if plan is not None and "plan" in ws.task()["approvals"] and not _plan_approved(ws, plan) \
            and not (rnd["decision"] is not None and ws.stop_check()[0]):
        return ("plan", "审批计划", "等用户批准计划 %s：顶层把计划讲给用户，用户同意后运行 rsi approve <ws> plan --ref %s。"
                % (plan, plan))
    if _attempts_exhausted(ws, rnd):
        return "close", "关闭本轮", _close_round_text(ws)
    if rnd["decision"] is None:
        return _next_step(ws, rnd)
    stop, why = ws.stop_check()
    if stop and _milestone_running(ws):
        if not _milestone_deliveries(ws):
            return ("milestone", "里程碑发布", "里程碑发布进行中：用户已同意用 test 验证最佳版本 %s，等 rsi-deliver "
                    "跑 test 预测并交回登记（这时不要结束运行）。" % ws.state["best"])
        return ("milestone", "里程碑发布", "里程碑发布进行中：里程碑 delivery 已登记，等 rsi-judge 用 test 打分并交回登记"
                "（这时不要结束运行）。")
    if stop:
        return ("decided", "停止检查", "本轮已判定，停止条件已满足（%s）：不再需要计划，等顶层运行 rsi next 结束运行"
                "%s。" % (why, "" if _milestone_evaluations(ws) else "（有最佳版本时先问用户是否做里程碑发布）"))
    if plan is None:
        return ("decided", "⑤ 评测与决策", "本轮已判定 %s：等 rsi-judge 写本轮的计划，交回后由顶层登记。"
                % _decision_text(rnd["decision"]))
    approval = "已批准" if _plan_approved(ws, plan) else "已登记"
    return ("decided", "推进轮次", "计划 %s %s：等顶层运行 rsi next 进入第 %d 轮（从 %s 开始）。"
            % (plan, approval, s["round"] + 1, ws.manifest(plan)["entry_module"]))


def _wait(ws, progress):
    """_status with (where, what) replaced by a wait for the user while a state-derived question
    is open (the plan wait already says so)."""
    key, where, what = _status(ws, progress)
    asked = _state_questions(ws, key)
    if asked and key != "plan":
        where = "等用户回答"
        what = "等用户回答：%s回答之前不派模块。" % "".join(q["text"].rstrip("。？?") + "。" for q in asked)
    return key, where, what


def waiting_for(ws, progress):
    """What the run waits for now, one Chinese sentence (see _status for the order)."""
    return _wait(ws, progress)[2]


# ---- pauses waiting for the user --------------------------------------------------------------

def _utc(ts):
    """A UTC time from STATE (2026-11-22T01:02:03Z) as an aware datetime, or None."""
    try:
        return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError):
        return None


def _utc_iso(ts):
    dt = _utc(ts)
    return None if dt is None else _iso(dt.timestamp())


def _milestone_stage(ws):
    """None unless the milestone release is on the table (not stopped, the round decided, the stop
    check says stop, a best version exists); then True when this round has a use_testset approval made
    at or after the round's decision (the user said yes), else False."""
    s = ws.state
    decision = _round(ws)["decision"]
    if s["stopped"] or decision is None or s["best"] is None or not ws.stop_check()[0]:
        return None
    at = _utc(decision.get("at"))
    for a in s["approvals"]:
        if a.get("what") == "use_testset" and a.get("round") == s["round"]:
            when = _utc(a.get("at"))
            if at is None or (when is not None and when >= at):
                return True
    return False


def _milestone_open(ws):
    """True while the orchestrator asks the milestone question: the release is on the table and the
    user has not approved it yet (the question comes before `rsi next`, which stops)."""
    return _milestone_stage(ws) is False


def _milestone_deliveries(ws):
    """This round's registered deliveries carrying predictions_test (milestone deliveries)."""
    return [a for a in _round(ws)["artifacts"]
            if ws.entry(a)["kind"] == "delivery" and "predictions_test" in ws.manifest(a)["files"]]


def _milestone_evaluations(ws):
    return [a for a in _round(ws)["artifacts"]
            if ws.entry(a)["kind"] == "evaluation" and ws.manifest(a)["uses_testset"] is True]


def _milestone_running(ws):
    """True while a milestone release is under way: approved by the user, its evaluation not
    registered yet."""
    return _milestone_stage(ws) is True and not _milestone_evaluations(ws)


def _notes_questions(ws):
    """The lines of the `## 待用户回答` section of <ws>/work/notes.md (a leading "- " or "* " dropped)."""
    path = os.path.join(ws.root, NOTES_FILE)
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        since = _iso(os.path.getmtime(path))
    except (OSError, ValueError):  # missing or unreadable notes: no questions, the page still renders
        return []
    out, inside = [], False
    for line in lines:
        if inside and line.startswith("## "):
            break
        if line == NOTES_SECTION:
            inside = True
        elif inside and line.strip():
            text = line.strip()
            if text[:2] in ("- ", "* "):
                text = text[2:].strip()
            if text:
                out.append({"source": "notes", "text": text, "since": since})
    return out


def _state_questions(ws, key):
    """The questions derived from the state; key is the case _status names."""
    s = ws.state
    task = ws.task()
    out = []
    if key == "plan":
        plan = _round(ws)["plan_id"]
        out.append({"source": "state", "since": _utc_iso(ws.entry(plan).get("registered_at")),
                    "text": "批准计划 %s：顶层把计划讲给你，你同意后它记下 plan 审批。" % plan})
    if _milestone_open(ws):
        out.append({"source": "state", "since": _utc_iso(_round(ws)["decision"].get("at")),
                    "text": "里程碑发布：是否用封存的 test 验证当前最佳版本 %s（要用两次 test，各记一次 use_testset 审批；"
                            "不做就运行 rsi next 结束运行）？" % s["best"]})
    budget = task["budget"]["compute_hours"]
    if key == "train" and "over_budget" in task["approvals"] and budget is not None \
            and s["budget_used"]["compute_hours"] >= budget - 1e-9 and not ws.has_approval("over_budget"):
        out.append({"source": "state", "since": None,
                    "text": "over_budget：已用算力 %s 小时，达到预算 %s 小时，继续派训练要你批准超预算。"
                            % (_num(s["budget_used"]["compute_hours"]), _num(budget))})
    if key in ("deliver", "judge") and s["eval_use_count"] >= task["eval_use_limit"] \
            and not ws.has_approval("refresh_sets"):
        used = [e for e in ws.entries("evaluation") if not ws.manifest(e["id"])["uses_testset"]]
        out.append({"source": "state", "since": _utc_iso(used[-1].get("registered_at")) if used else None,
                    "text": "refresh_sets：评测集已用 %d 次，到了上限 %d，要换新的评测集（需要你批准 refresh_sets）才能再送评。"
                            % (s["eval_use_count"], task["eval_use_limit"])})
    return out


def pending_questions(ws, progress):
    """The pauses where the orchestrator waits for the user: [{"source": "state" | "notes",
    "text", "since"}], those derived from the state first (plan approval, milestone release,
    over_budget, refresh_sets), then the notes' lines in order. since: ISO time or None."""
    return _state_questions(ws, _status(ws, progress)[0]) + _notes_questions(ws)


# ---- the workflow diagram ---------------------------------------------------------------------

def _step_artifacts(ws, module, arts):
    """This round's registered artifacts produced by `module` (records aside)."""
    return [e for e in arts if e["producer"] == module and e["kind"] != "record"]


def _registered_status(ws, module, arts):
    """A step's status from this round's registered artifacts: "done", "fail" (its only output
    failed its gate), "partial" (registration under way: some data sets, or tuning trials without
    their search), "closed" (⑤ in a round closed by `rsi close-round`) or "idle"."""
    mine = _step_artifacts(ws, module, arts)
    kinds = [e["kind"] for e in mine]
    if module == "data":
        if not any(k in DATASETS for k in kinds):
            return "idle"
        if STEP_INDEX.get(_round(ws)["entry"], 0) > 0 and "trainset" not in kinds:
            return "idle"  # a refreshed evalset or valset alone is no data step
        if _data_missing(ws, arts) is None:
            return "done"
        trainsets = [e for e in mine if e["kind"] == "trainset"]
        if trainsets and ws.current("trainset") not in [e["id"] for e in trainsets]:
            return "fail"
        return "partial"
    if module == "train":
        cands = [e["id"] for e in mine if e["kind"] == "candidate"]
        if not cands:
            return "idle"
        passed = ws.passed_candidates()
        return "done" if any(c in passed for c in cands) else "fail"
    if module == "tune":
        if "search" in kinds:
            return "done"
        return "partial" if "candidate" in kinds else "idle"
    if module == "deliver":
        done = _own_delivery(ws, arts) and not (_round(ws)["decision"] is None and _redeliver_pending(ws))
        return "done" if done else "idle"
    if ws.evaluations():
        return "done"
    if ws.closed():
        return "closed"
    return "idle"


# The steps each next-step key of _status sends the run to (② and ③ are alternatives).
NEXT_STEPS = {"data": ("data",), "train": ("train", "tune"), "tune": ("tune",), "deliver": ("deliver",),
              "judge": ("judge",)}


def _node(state, text=""):
    return {"state": state, "text": text}


def _hm(dt):
    return dt.astimezone().strftime("%H:%M")


def _done_text(ws, module, mine, rnd):
    """The short status phrase of a step done in this round."""
    if module == "data":
        return "✓ %d 个集合" % sum(1 for e in mine if e["kind"] in DATASETS and ws.current(e["kind"]) == e["id"])
    if module == "train":
        return "✓ 训练门通过"
    if module == "tune":
        trials = sum(1 for e in mine if e["kind"] == "candidate")
        return "✓ %d 个试验" % trials if trials else "✓ 完成"
    if module == "deliver":
        ids = [e["id"] for e in mine if e["kind"] == "delivery"]
        return "✓ " + ids[0] if len(ids) == 1 else "✓ %d 个交付包" % len(ids)
    parts = []
    d = rnd["decision"]
    if ws.evaluations():
        if d is None:
            parts.append("已评测")
        else:
            parts.append("升级 %s" % d.get("version") if d.get("decision") == "promote" else "保留")
    if _milestone_evaluations(ws):
        parts.append("里程碑评测")
    if rnd["plan_id"] is not None:
        parts.append("计划")
    return "✓ " + " · ".join(parts)


def flow_states(ws, progress):
    """The workflow diagram's nodes: key -> {"state", "text"} for task, the five steps (data, train,
    tune, deliver, judge), check (顶层核对) and stop (停止与交付).

    state is done, current, wait (waits for the user), fail (the step's only output this round
    failed its gate), skip (not passed through this round) or todo; text is a short status phrase.
    A step is current while its module runs or waits to be registered, or when it is the next step
    of _status (② and ③ both); done, fail and skip follow the round's registered artifacts; a step
    before the round's entry, or behind the step the round has reached with no output of its own,
    is skipped (with the attempts used up, ⑤ is still to come: the judge writes the plan of the
    closed round). check waits while the user is asked to approve the plan or the milestone release,
    is current while the round waits for `rsi promote`, `rsi close-round` or `rsi next` and done
    once the round is decided. While the notes hold questions for the user, every current node waits
    instead. Once stopped, stop is current and the other nodes keep their state from the round's
    artifacts (check done)."""
    s = ws.state
    rnd = _round(ws)
    arts = [ws.entry(a) for a in rnd["artifacts"]]
    stopped = s["stopped"]
    key = _status(ws, progress)[0]
    entry = STEP_INDEX.get(rnd["entry"], 0)
    if stopped:
        reached = [STEP_INDEX[m] for m in MODULES if _step_artifacts(ws, m, arts)]
        position, nxt, asked, writing_plan = max(reached + [-1]), (), False, False
    else:
        busy = [STEP_INDEX[m] for m in MODULES if progress[m]["state"] in BUSY]
        if busy:
            position = max(busy)
        elif key in NEXT_STEPS:
            position = STEP_INDEX[key]
        elif key == "close":
            position = STEP_INDEX["judge"]  # ④ is not passed through; the judge still writes the plan
        else:
            position = AFTER_STEPS
        nxt = NEXT_STEPS.get(key, ())
        asked = bool(nxt) and bool(_state_questions(ws, key))
        writing_plan = rnd["decision"] is not None and rnd["plan_id"] is None and not ws.stop_check()[0]
    out = {"task": _node("done", "✓ 完成")}
    for m in MODULES:
        p = progress[m]
        mine = _step_artifacts(ws, m, arts)
        status = _registered_status(ws, m, arts)
        if m == "judge" and rnd["plan_id"] is not None:
            status = "done"  # the judge's plan is in (also for a round closed without evaluation)
        if p["state"] == "running" and not stopped:
            out[m] = _node("current", "● 进行中 · %s 起" % _hm(datetime.datetime.fromisoformat(p["since"])))
        elif p["state"] == "unregistered" and not stopped:
            out[m] = _node("current", "● 待登记")
        elif m == "judge" and writing_plan:
            out[m] = _node("current", "● 写计划")
        elif status == "done":
            out[m] = _node("done", _done_text(ws, m, mine, rnd))
        elif status == "fail":
            out[m] = _node("fail", "门没过 · 可重派" if m in nxt else "门没过")
        elif p["state"] in BUSY:
            out[m] = _node("skip", "停止时未交回")
        elif status == "closed":
            out[m] = _node("skip", "本轮已关闭")
        elif status == "partial":
            out[m] = _node("todo", "停止时未完成") if stopped else _node("current", "● 进行中")
        elif m in nxt:
            late = m == "deliver" and _own_delivery(ws, arts) and _redeliver_pending(ws)
            out[m] = (_node("wait", "⏸ 等你批准") if asked
                      else _node("current", "● 待重新交付最佳版本" if late else "● 下一步"))
        elif STEP_INDEX[m] < entry or STEP_INDEX[m] < position:
            out[m] = _node("skip", "本轮不经过")
        else:
            out[m] = _node("todo")
    if stopped:
        out["check"] = _node("done", "✓ 已核对")
    elif key == "plan":
        out["check"] = _node("wait", "⏸ 等你批准计划")
    elif _milestone_open(ws):
        out["check"] = _node("wait", "⏸ 等你定里程碑")
    elif key == "promote":
        out["check"] = _node("current", "● 等 rsi promote")
    elif key == "close":
        out["check"] = _node("current", "● 等关闭本轮")
    elif key == "milestone" or _milestone_running(ws):
        out["check"] = _node("current", "● 里程碑发布中")
    elif key == "decided" and not writing_plan:
        out["check"] = _node("current", "● 等 rsi next")
    elif rnd["decision"] is not None:
        out["check"] = _node("done", "✓ 已判定")
    else:
        out["check"] = _node("todo")
    if stopped:
        at = _utc(stopped.get("at"))
        out["stop"] = _node("current", "■ %s 停止" % _hm(at) if at else "■ 已停止")
    else:
        out["stop"] = _node("todo")
        if _notes_questions(ws):  # the orchestrator waits for the user before going on
            for k, node in out.items():
                if node["state"] == "current":
                    running = k in progress and progress[k]["state"] == "running"
                    out[k] = _node("wait", "⏸ 运行中 · 等你回答" if running else "⏸ 等你回答")
    return out


# ---- writing the pages -----------------------------------------------------------------------

def _write_atomic(path, text):
    """Write `text` to `path` through a temp file in the same directory and os.replace, mode 0644.
    On failure the temp file is removed and the error raised."""
    fd, tmp = tempfile.mkstemp(prefix=".dashboard.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)  # mkstemp makes it 0600; the pages are no secret
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_dashboard(ws):
    """Write the four pages (PAGES, under <ws>/dashboard/ for the secondary ones), each atomically,
    and return the overview's absolute path. The overview is written last, so the pages it links
    to exist when it appears. Any failure is raised."""
    from rsi.dashboard_pages import render_pages
    pages = render_pages(ws)
    os.makedirs(os.path.join(ws.root, PAGES_DIR), exist_ok=True)
    for rel in reversed(PAGES):
        _write_atomic(os.path.join(ws.root, *rel.split("/")), pages[rel])
    return os.path.join(ws.root, DASHBOARD_FILE)
