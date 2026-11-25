"""Code-built briefs and the hand-back contract.

build_brief(ws, module, dispatch_id=None) assembles what a module may read and write as a dict; the
inputs, per purpose (rsi.flow.PURPOSES):

  data     sets         task; corpus and pool once registered (a resumed sets); the brief also carries `sources`,
                        the task's raw data (rsi.datasplit.source_paths: data.sources), read in this dispatch only
           trainset     task, trainset (current), valset, plan (the one that opened the round), feedback, pool
           refresh      task, plan, trainset, valset, corpus, and the inputs (no labels) of every registered evalset
                        (evalset_inputs_<id>): the new evalset is drawn from the corpus clear of them (the testset
                        and the evalsets' labels are never given back to data)
  train    train        task, trainset, valset, best_candidate (the best version's candidate, when there is one),
                        every registered candidate and search (by id: candidate-NNNN, search-NNNN), plan (the one
                        that opened the round, when there is one); the brief also carries the task's search_space,
                        search, resources.onejev, resources.gpu_ids and flow.checkin_minutes (null: no check-ins)
  deliver  deliver      task, candidate (the one to deliver), evalset_inputs, valset_inputs (of the
                        candidate's valset); after a refresh also best_candidate with its valset inputs
           milestone_deliver
                        task, candidate (the best version's), evalset_inputs, valset_inputs, testset_inputs
  judge    judge        task, candidate, delivery (the evaluated one), evalset_inputs, evalset_labels,
                        trainset_rows (of the candidate's trainset), plan (the one that opened the round: did it
                        work?), the round's searches, every feedback and record, best_evaluation and
                        best_delivery, best_delivery_current_evalset after a refresh
           plan         task, the round's evaluation (if any), candidates, deliveries, searches, every
                        feedback and record, best_evaluation (no labels: a closed round has no evaluation)
           milestone_judge
                        the judge inputs for the milestone delivery, plus testset_inputs and testset_labels

Every data brief also carries `dset` (dset_command: the split tool's command prefix); only the train brief carries
`train_command`: the data module does not train (whether a new trainset helps is judged by the round's train and judge
dispatches), so its brief names no train command. Every purpose of data, train
and deliver also gives the module's own records of earlier rounds (records whose module is this module, for train
also an old workspace's tune records, registered in a round before the brief's); the judge already reads every
record. The read-deny list (denied_paths) keeps every module out of the raw data (raw_data_paths: that of every
registered task version) but data's sets, and out of every data dispatch's output directory but its own.
Only the judge gets labels. Every path is absolute and lies in a registered artifact directory the module
may read or in its output directory; the workspace root is never given. Every brief names its dispatch
(dispatch_id, null for a brief built without one) and its progress_file (<outdir>/progress.json, which the module may
write: docs/interfaces/progress.md). read_handback(outdir) checks the handback.json a module writes into its output
directory.
"""
import json
import os
import shlex

import rsi
from rsi import datasplit as DS
from rsi import flow as F
from rsi.progress import PROGRESS_FILE
from rsi.rules import flow_settings
from rsi.workspace import RsiError

MODULE_PURPOSES = {
    "data": ("sets", "trainset", "refresh"),
    "train": ("train",),
    "deliver": ("deliver", "milestone_deliver"),
    "judge": ("judge", "plan", "milestone_judge"),
}
SKILLS = {m: "rsi-" + m for m in MODULE_PURPOSES}
DECISION_GOAL_MODULES = ("data", "train")  # their goal restates the task's decision when the plan gives them no issue
DEFAULT_TRAIN_COMMAND = os.path.join("modules", "train", "train")
LONG_JOB_MINUTES = 10

PURPOSE_TEXT = {
    "sets": ("第一轮：写预处理脚本，把原始数据做成 corpus；定切分规则，用 dset sets 产出 testset、evalset、valset、pool、"
             "trainset（任务的切分规则需要补全时，还有 task 的补全版本）。"),
    "trainset": "产出新版本的 trainset（带数据质量报告）。",
    "refresh": "评测集已用到上限并获批刷新：产出新的 evalset。",
    "train": ("训练调优：在任务的配置空间（search_space）里做一次搜索。先交回本次搜索所有成功试验的候选和一份 record；引擎登记后"
              "同一会话收到「# 继续：交回的第二部分」，再交回 search。"),
    "deliver": "把送评候选打包成交付，在评测集输入和验证集输入上跑预测。",
    "milestone_deliver": "里程碑发布：用当前最佳版本的候选，在 test 输入上另跑预测。",
    "judge": "在评测集上评测送评候选，写 evaluation（本轮判定后再写 plan）。",
    "plan": "本轮已判定：根据本轮的评测（关闭的轮次没有评测，则看本轮的候选、search 和记录）写下一轮的计划。",
    "milestone_judge": "里程碑发布：用 test 标签给里程碑交付打分，写里程碑 evaluation（不写 plan）。",
}

HANDBACK_FORMAT = (
    "交回时在输出目录写 `handback.json`：{\"status\": \"ok\" | \"failed\" | \"long_job\", \"artifacts\": "
    "[输出目录下的子目录名，每个产物一个], \"marker\": 长任务完成标记的绝对路径（status 为 long_job 时必填，"
    "其他写 null）, \"notes\": 报告（门数字、花费、遇到的问题）, \"questions\": [需要用户决定的事，没有写 []]}。"
    "status 为 failed 时在 notes 里写原因。questions 不为空时，这次交回的产物一个都不登记：用户回答后，回答送回同一个会话"
    "（反馈开头是「# 用户回答了你的问题」，不算退回），按回答改好或确认后重新交回要登记的全部子目录；long_job 交回里的问题等完成标记"
    "出现后才问，没列产物时回答随「# 长任务已结束」的反馈送回。长任务（预计超过 %d 分钟）"
    "在后台启动后立刻以 long_job 交回，任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`；不要等待或轮询。要交回 long_job 时"
    "完成标记已经在了（长任务已经结束）就不要以 long_job 交回：早于这次启动的标记引擎永远不认，直接读结果接着做完再交回。交回 long_job "
    "之前在完成标记旁写 `<marker>.pid`：第一行是长任务进程自己的进程号（由长任务自己写，例如 shell 里的 `$$` 或 Python 的 "
    "`os.getpid()`），第二行是它的启动时间（Unix 秒，`date +%%s`）；运行被停止时引擎按它终止长任务。每个长任务用新的标记路径"
    "（例如 `<名>-<k>.done`，`<k>` 每次加一），启动前 `rm -f <marker> <marker>.pid`：比这次启动早的标记文件不算这个长任务结束。"
    % LONG_JOB_MINUTES)

HANDBACK_STATUSES = ("ok", "failed", "long_job")
HANDBACK_KEYS = ("status", "artifacts", "marker", "notes", "questions")


def repo_root():
    """The directory containing rsi/cli.py (the repository root)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(rsi.__file__)))


def skill_path(module):
    """The module's skill, which the engine gives its agent: modules/<module>/SKILL.md under the repository root (each
    module keeps its skill, code and references in its own folder)."""
    return os.path.join(repo_root(), "modules", module, "SKILL.md")


def rsi_command():
    return "PYTHONPATH=%s python3 -m rsi" % shlex.quote(repo_root())


def dset_command():
    """The command prefix of the data module's split tool (modules/data/dset), given to every data brief."""
    return "PYTHONPATH=%s python3 -m dset" % shlex.quote(os.path.join(repo_root(), "modules", "data"))


def train_command(task):
    c = (task.get("resources") or {}).get("train_command")
    return c if c else os.path.join(repo_root(), DEFAULT_TRAIN_COMMAND)


def _purpose(ws, module, dispatch_id, purpose):
    d = None
    if dispatch_id is not None:
        d = ws.dispatch(dispatch_id)
        if d.get("module") != module:
            raise RsiError("dispatch %s is a %s dispatch, not %s" % (dispatch_id, d.get("module"), module))
        purpose = purpose or F.dispatch_purpose(d)
    purpose = purpose or F.dispatch_purpose({"module": module, "round": ws.state["round"]})
    if purpose not in MODULE_PURPOSES[module]:
        raise RsiError("purpose %r does not belong to module %s (one of %s)"
                       % (purpose, module, ", ".join(MODULE_PURPOSES[module])))
    return d, purpose


class _Inputs(object):
    """The inputs of one brief: name -> absolute path, with a one-line description each."""

    def __init__(self, ws):
        self.ws = ws
        self.paths, self.notes = {}, {}

    def art(self, name, aid, note, logical=None):
        """The artifact directory of `aid`, or a file of it (a manifest `files` entry); never twice."""
        if name in self.paths:
            return
        p = self.ws.path(aid)
        if logical == "task.json":
            p = os.path.join(p, logical)
        elif logical is not None:
            p = os.path.join(p, self.ws.manifest(aid)["files"][logical]["path"])
        self.paths[name] = os.path.realpath(p) if os.path.islink(p) else p
        self.notes[name] = "%s（%s）" % (note, aid)


def _round_ids(ws, kind, rnd=None):
    rnd = ws.state["round"] if rnd is None else rnd
    return [a for a in ws.state["rounds"][str(rnd)]["artifacts"] if ws.entry(a)["kind"] == kind]


def _plan_of_round(ws):
    """The plan that opened the current round (the previous round's), or None (round 1)."""
    prev = ws.state["rounds"].get(str(ws.state["round"] - 1))
    return prev["plan_id"] if prev else None


def _issue_text(i):
    e = i.get("expected") or {}
    return "- %s（证据：%s）；要改：%s；预期：%s %s %s" % (
        i.get("issue"), i.get("evidence"), i.get("change"), e.get("metric"), e.get("direction"), e.get("size"))


def _goal(ws, module, purpose, task):
    lead = PURPOSE_TEXT[purpose]
    plan = _plan_of_round(ws)
    issues = [i for i in ws.manifest(plan).get("issues", []) if i.get("module") == module] if plan else []
    if issues:
        return "%s\n本轮计划 %s 分给本模块的问题：\n%s" % (lead, plan, "\n".join(_issue_text(i) for i in issues))
    if module in DECISION_GOAL_MODULES:
        d = task["decision"]
        return "%s\n按任务定义：%s（输入：%s）" % (lead, d["instructions"], task["input"]["description"])
    return lead


def _budget_left(ws, task):
    b, used = task["budget"], ws.state["budget_used"]["compute_hours"]
    return {"compute_hours": None if b.get("compute_hours") is None else max(0, b["compute_hours"] - used),
            "annotation_items": b.get("annotation_items"), "api_cost": b.get("api_cost")}


def _best(ws):
    name = ws.state["best"]
    v = ws.version(name) if name is not None else None
    if v is None:
        return None
    return {"version": name, "candidate_id": v["candidate_id"], "evaluation_id": v["evaluation_id"],
            "delivery_id": ws.manifest(v["evaluation_id"])["delivery_id"],
            "delivery_on_current_evalset": ws.best_delivery_on_current_evalset()}


def _ids(module, purpose, ws):
    """The current set ids a module writes into manifests: trainset and valset; the evalset only for deliver
    and judge, the testset only for the milestone purposes."""
    kinds = ["trainset", "valset"]
    if module in ("deliver", "judge"):
        kinds.append("evalset")
    if purpose in ("milestone_deliver", "milestone_judge"):
        kinds.append("testset")
    return dict((k, ws.current(k)) for k in kinds)


def _cur(ws, kind):
    aid = ws.current(kind)
    if aid is None:
        raise RsiError("no testset registered" if kind == "testset" else "no current %s registered" % kind)
    return aid


def _own_records(ws, module, rnd, inp):
    """The records of `module` (the record's module) registered in rounds before `rnd`: what it tried before and
    what came of it. Only its own: another module's records stay out of its brief (and on its read-deny list); an
    old workspace's tune records are train's (rsi.flow.stage_of: tune was merged into train)."""
    for e in ws.entries("record"):
        if e["round"] < rnd and F.stage_of(ws.manifest(e["id"])["module"]) == module:
            inp.art(e["id"], e["id"], "本模块第 %d 轮的实验记录" % e["round"])


def _data_inputs(ws, purpose, inp):
    if purpose == "sets":  # a resumed sets: what this dispatch registered already (dset reads it, never rewrites it)
        for kind, note in (("corpus", "已登记的 corpus（预处理脚本和全部带标签的行）"), ("pool", "已登记的训练池")):
            if ws.current(kind):
                inp.art(kind, ws.current(kind), note)
        return
    if purpose == "trainset" and ws.current("trainset"):
        inp.art("trainset", ws.current("trainset"), "当前训练集")
    if purpose == "refresh":
        inp.art("trainset", _cur(ws, "trainset"), "当前训练集（新评测集不能与它重叠）")
    inp.art("valset", _cur(ws, "valset"), "验证集")
    plan = _plan_of_round(ws)
    if plan:
        inp.art("plan", plan, "开启本轮的计划")
    if purpose == "trainset":
        for f in ws.entries("feedback"):
            inp.art(f["id"], f["id"], "执行反馈")
        if ws.current("pool"):
            inp.art("pool", ws.current("pool"), "训练池（训练时段的全部带标签行）：新训练集从这里抽")
    if purpose == "refresh":
        if ws.current("corpus"):
            inp.art("corpus", ws.current("corpus"), "corpus（全部带标签的行）：新评测集从这里抽")
        for e in ws.entries("evalset"):
            inp.art("evalset_inputs_" + e["id"], e["id"], "以前的评测集的输入（不带标签）：新评测集不能与它重叠", "inputs")


def _train_inputs(ws, inp):
    inp.art("trainset", _cur(ws, "trainset"), "当前训练集")
    inp.art("valset", _cur(ws, "valset"), "验证集")
    best = _best(ws)
    if best is not None:
        inp.art("best_candidate", best["candidate_id"], "当前最佳版本 %s 的候选：本次搜索的起点" % best["version"])
    for kind in ("candidate", "search"):
        for e in ws.entries(kind):
            inp.art(e["id"], e["id"], "已登记的" + ("候选" if kind == "candidate" else " search"))
    plan = _plan_of_round(ws)
    if plan:
        inp.art("plan", plan, "开启本轮的计划")


def _deliver_inputs(ws, d, purpose, inp):
    best = _best(ws)
    arts = [ws.entry(a) for a in ws.state["rounds"][str(ws.state["round"])]["artifacts"]]
    cand = (d or {}).get("candidate_id")
    if purpose == "milestone_deliver":
        cand = cand or (best or {}).get("candidate_id")
        best_cand = None
    else:
        if cand is None and not F.own_delivery(ws, arts):
            cand = F.chosen_candidate(ws)
        best_cand = best["candidate_id"] if best and F.redeliver_pending(ws) else None
        if cand is None:  # only the best version's re-delivery is owed
            cand, best_cand = best_cand, None
    if cand is not None:
        inp.art("candidate", cand, "送评的候选")
        inp.art("valset_inputs", ws.manifest(cand)["valset_id"], "候选所用验证集的输入（不带标签），用来复现验证集预测", "inputs")
    if best_cand is not None and best_cand != cand:
        inp.art("best_candidate", best_cand, "当前最佳版本的候选：在新评测集上重新交付")
        inp.art("best_valset_inputs", ws.manifest(best_cand)["valset_id"], "最佳版本候选所用验证集的输入（不带标签）", "inputs")
    inp.art("evalset_inputs", _cur(ws, "evalset"), "评测集的输入", "inputs")
    if purpose == "milestone_deliver":
        inp.art("testset_inputs", _cur(ws, "testset"), "封存 test 的输入（里程碑发布已获批）", "inputs")


def _eval_delivery(ws, milestone):
    """The delivery the judge evaluates: the milestone delivery of the round (milestone), else the round's
    delivery on the current evalset of its own candidate (any candidate when the round has none)."""
    if milestone:
        ds = F.milestone_deliveries(ws)
        return ds[-1] if ds else None
    own = set(_round_ids(ws, "candidate"))
    cur = _cur(ws, "evalset")
    ds = [a for a in _round_ids(ws, "delivery")
          if ws.manifest(a)["evalset_id"] == cur and "predictions_test" not in ws.manifest(a)["files"]]
    mine = [a for a in ds if ws.manifest(a)["candidate_id"] in own]
    if mine or not own:
        pick = mine or ds
        return pick[-1] if pick else None
    return None


def _judge_inputs(ws, purpose, inp):
    best = _best(ws)
    if purpose == "plan":
        for kind, name in (("evaluation", "evaluation"), ("candidate", "候选"), ("delivery", "交付")):
            for a in _round_ids(ws, kind):
                m = ws.manifest(a) if kind in ("evaluation", "delivery") else {}
                if m.get("uses_testset") is True or "predictions_test" in m.get("files", {}):
                    continue  # milestone artifacts carry test results: no plan reads them
                inp.art(a, a, "本轮的" + name)
    else:
        milestone = purpose == "milestone_judge"
        dlv = _eval_delivery(ws, milestone)
        if dlv is not None:
            cand = ws.manifest(dlv)["candidate_id"]
            inp.art("delivery", dlv, "里程碑交付（含 test 预测）" if milestone else "本轮送评候选的交付")
            inp.art("candidate", cand, "被评测的候选")
            inp.art("trainset_rows", ws.manifest(cand)["trainset_id"], "被评测候选所用训练集的行，只用来算对照组", "rows")
        inp.art("evalset_inputs", _cur(ws, "evalset"), "评测集的输入", "inputs")
        inp.art("evalset_labels", _cur(ws, "evalset"), "评测集的标签（只有 judge 拿标签）", "labels")
        plan = None if milestone else _plan_of_round(ws)
        if plan:
            inp.art("plan", plan, "开启本轮的计划（上一轮写的）：对照本轮的评测，判断它是否奏效")
        if milestone:
            inp.art("testset_inputs", _cur(ws, "testset"), "封存 test 的输入", "inputs")
            inp.art("testset_labels", _cur(ws, "testset"), "封存 test 的标签（里程碑发布已获批）", "labels")
        if not milestone and best is not None and best["delivery_on_current_evalset"] != best["delivery_id"]:
            if best["delivery_on_current_evalset"]:
                inp.art("best_delivery_current_evalset", best["delivery_on_current_evalset"],
                        "最佳版本在当前评测集上的交付：算最佳版本一侧的指标，并写进 evaluation 的 parents")
        if milestone and best is not None:
            inp.art("best_evaluation", best["evaluation_id"], "当前最佳版本的评测")
    for a in _round_ids(ws, "search"):
        inp.art(a, a, "本轮的 search")
    if best is not None:
        inp.art("best_evaluation", best["evaluation_id"], "当前最佳版本 %s 的评测" % best["version"])
        inp.art("best_delivery", best["delivery_id"], "当前最佳版本 %s 的交付" % best["version"])
    for kind, note in (("feedback", "执行反馈"), ("record", "实验记录")):
        for e in ws.entries(kind):
            inp.art(e["id"], e["id"], note)


def build_brief(ws, module, dispatch_id=None, purpose=None):
    """The structured brief of a dispatch (see the module docstring). The purpose is the dispatch record's
    when dispatch_id is given, else `purpose`, else the one rsi.flow.dispatch_purpose infers."""
    if module not in MODULE_PURPOSES:
        raise RsiError("unknown module '%s' (one of %s)" % (module, ", ".join(MODULE_PURPOSES)))
    d, purpose = _purpose(ws, module, dispatch_id, purpose)
    task, s = ws.task(), ws.state
    rnd = s["round"] if d is None else d.get("round", s["round"])
    tid = ws.current("task")
    inp = _Inputs(ws)
    inp.art("task", tid, "任务定义", "task.json")
    if module == "data":
        _data_inputs(ws, purpose, inp)
    elif module == "train":
        _train_inputs(ws, inp)
    elif module == "deliver":
        _deliver_inputs(ws, d, purpose, inp)
    else:
        _judge_inputs(ws, purpose, inp)
    if module != "judge":  # the judge reads every module's records already
        _own_records(ws, module, rnd, inp)
    if d is not None and d.get("outdir"):
        outdir = os.path.join(ws.root, d["outdir"])
    else:
        outdir = os.path.join(ws.root, "work", "round-%d" % rnd, module)
    brief = {
        "module": module, "skill": SKILLS[module], "purpose": purpose, "round": rnd,
        "dispatch_id": d["id"] if d is not None else None,
        "task_id": task["task_id"], "task_version": task["version"],
        "task_json": inp.paths["task"],
        "ids": _ids(module, purpose, ws),
        "inputs": inp.paths, "input_notes": inp.notes, "outdir": outdir,
        "progress_file": os.path.join(outdir, PROGRESS_FILE),  # optional, docs/interfaces/progress.md
        "rsi": rsi_command(), "budget_left": _budget_left(ws, task),
        "attempts": "%d / %d" % (len(ws.attempts()), ws.attempt_limit()),
        "goal": _goal(ws, module, purpose, task), "handback": HANDBACK_FORMAT,
    }
    if module == "data":
        brief["dset"] = dset_command()
        if purpose == "sets":
            brief["sources"] = DS.source_paths(task)
    if module == "train":
        res = task.get("resources") or {}
        brief["train_command"] = train_command(task)
        brief["search_space"] = task.get("search_space")
        brief["search"] = task.get("search") or {}
        brief["resources"] = dict((k, res[k]) for k in ("onejev", "gpu_ids") if k in res)
        brief["checkin_minutes"] = flow_settings(task)["checkin_minutes"]  # null: no check-ins
    if module == "judge":
        brief["best"] = _best(ws)
    return brief


DATASET_KINDS = ("trainset", "valset", "evalset", "testset", "corpus", "pool")
DECISION_DENIED_KINDS = DATASET_KINDS + ("delivery", "evaluation", "feedback")
RULE_UNSAFE = DS.UNSAFE_PATH_CHARS


def denied_paths(ws, brief):
    """Absolute paths (artifact directories or single files) a module with this brief must not read or edit:
    every registered artifact of any kind (the brief is the only legitimate input) with no brief input in its
    directory, as a whole directory. A dataset (DATASET_KINDS: the four sets, the corpus, the pool) of which the
    brief gives only some files (e.g. deliver gets only `inputs.jsonl`) has its other manifest `files` listed one by
    one; a directory the brief gives whole, or any other kind of which it gives a file (the task's task.json), is not
    listed. Only registered artifacts are covered; the rest of the workspace is outside this list, but for the output
    directories of the data dispatches (data_outdirs) other than the brief's own: unregistered sets with their labels
    may be left there (a refused hand-back, a redo/ not removed). The raw data (raw_data_paths) is denied too, but for
    data's sets: the one dispatch that reads it (`sources`)."""
    given = [os.path.realpath(p) for p in brief["inputs"].values()]
    out = []
    for e in ws.entries():
        d = os.path.abspath(ws.path(e["id"]))
        real = os.path.realpath(d)
        inside = [p for p in given if p == real or p.startswith(real + os.sep)]
        if not inside:
            out.append(d)
        elif real not in inside and e["kind"] in DATASET_KINDS:
            for spec in ws.manifest(e["id"])["files"].values():
                f = os.path.join(d, spec["path"])
                if os.path.realpath(f) not in inside:
                    out.append(f)
    own = os.path.realpath(brief["outdir"]) if brief.get("outdir") else None
    out += [p for p in data_outdirs(ws) if own is None or not _related(os.path.realpath(p), own)]
    if (brief.get("module"), brief.get("purpose")) != ("data", "sets"):
        out += [p for p in raw_data_paths(ws) if p not in out]
    return out


def data_outdirs(ws):
    """The output directories (absolute, each once, in dispatch order) of every data dispatch."""
    out = []
    for d in ws.dispatches(module="data"):
        p = os.path.abspath(os.path.join(ws.root, d["outdir"])) if d.get("outdir") else None
        if p and p not in out:
            out.append(p)
    return out


def _related(a, b):
    """True when real path `a` is `b`, lies inside it or holds it."""
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


def raw_data_paths(ws):
    """The raw data: the data.sources paths of every registered task version, in order, each once (a version that
    moved the raw data leaves the old path denied too). The sets brief's `sources` is the current version's only."""
    out = []
    for e in ws.entries("task"):
        out += [p for p in DS.source_paths(ws.manifest(e["id"])) if p not in out]
    return out


def decision_denied(ws):
    """What a decision agent must not read: every dataset directory (the corpus and the pool too), every delivery,
    evaluation and feedback, the data dispatches' output directories (data_outdirs) and the raw data
    (raw_data_paths)."""
    return [os.path.abspath(ws.path(e["id"])) for e in ws.entries() if e["kind"] in DECISION_DENIED_KINDS] + \
        data_outdirs(ws) + raw_data_paths(ws)


def deny_rules(paths):
    """Claude Code deny rules (`--disallowedTools`) for `paths`: Read and Edit of `//<absolute path>` (a leading
    single slash would be relative to the session's directory), `/**` after a directory; a path whose type cannot
    be told (os.path.exists is false: gone, or under a directory made unreadable with chmod 000) gets both, the path
    itself and `/**`, so whatever lies under it stays covered. A path whose real path differs (a symlinked workspace
    root, /tmp on macOS) gets the rules for both spellings. Refuses (RsiError) a path the rule parser would mangle:
    one with `(`, `)`, `*`, `?`, `[` or starting with `!`."""
    rules = []
    for p in paths:
        a = os.path.abspath(p)
        for q in (p, a, os.path.realpath(a)):
            if any(c in q for c in RULE_UNSAFE) or q.startswith("!"):
                raise RsiError("cannot deny reads of %r: a path with any of %s or a leading ! cannot be written "
                               "as a safe deny rule; move the workspace" % (q, " ".join(RULE_UNSAFE)))
        if os.path.isdir(a):
            tails = ("/**",)
        elif os.path.exists(a):
            tails = ("",)
        else:  # cannot tell a file from a directory: deny both
            tails = ("", "/**")
        for q in dict.fromkeys((a, os.path.realpath(a))):
            for tail in tails:
                spec = "//" + q.lstrip("/") + tail
                rules += ["Read(%s)" % spec, "Edit(%s)" % spec]
    return rules


def _cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def render_brief(brief):
    """The brief as Markdown, in the structure of skills/rsi-orchestrator/templates/brief.md."""
    b, ids = brief, brief["ids"]
    L = ["你负责 RSI 运行里的一个模块。先加载 skill `%s`，按它的「输入」「输出」「门」工作。" % b["skill"], ""]
    L.append("- skill：`%s`" % b["skill"])
    L.append("- 任务定义：`%s`，人读版是同一目录的 `TASK.md`" % b["task_json"])
    L.append("- 本轮轮次：`%d`。每个产物 manifest 的 `round` 写这个数，`task_id` 写 `%s`，`task_version` 写 `%s`，"
             "`id` 写 null。" % (b["round"], b["task_id"], b["task_version"]))
    L.append("- 当前登记 id：" + "，".join("%s `%s`" % (k, ids[k] or "无") for k in ids)
             + "。manifest 里引用这些集合时用这些 id。")
    if "best" in b:
        best = b["best"]
        if best is None:
            L.append("- 当前最佳版本：无（`best_before` 写 null）。")
        else:
            L.append("- 当前最佳版本：版本名 `%s`，evaluation `%s`，delivery `%s`。`best_before` 按这些值写。评测集刷新过时，"
                     "最佳版本在当前评测集上的 delivery：`%s`（用它算最佳版本一侧的指标，并把它的 id 写进 evaluation 的 `parents`）。"
                     % (best["version"], best["evaluation_id"], best["delivery_id"],
                        best["delivery_on_current_evalset"] or "无"))
    if "train_command" in b:
        L.append("- 训练命令：`%s`。调用它时环境里带 `PYTHONPATH=%s`。" % (b["train_command"], repo_root()))
    if b.get("sources"):
        L.append("- 原始数据（任务的 `data.sources`，只在本次派出读，之后的派出都禁读）：%s。"
                 % "、".join("`%s`" % p for p in b["sources"]))
    if "dset" in b:
        L.append("- 切分工具：`%s`，用法见本模块 skill。" % b["dset"])
    if b.get("module") == "train":
        L.append("- 配置空间（任务的 `search_space`，只在这里面搜；候选 manifest 的 `config_request` 必须落在里面，"
                 "`code_version.entry` 必须是上面的训练命令）：%s" % (_json(b.get("search_space")) if b.get("search_space")
                                                         is not None else "任务没有写，按训练命令的默认配置"))
        L.append("- 搜索设置（任务的 `search`）：%s；资源（任务的 `resources.onejev`、`resources.gpu_ids`）：%s。"
                 % (_json(b.get("search") or {}), _json(b.get("resources") or {})))
        if b.get("checkin_minutes"):
            L.append("- 长任务期间引擎每 %s 分钟让本会话做一次定时检查（反馈开头「# 定时检查」，不算退回）：检查环境，有异常先处理，"
                     "长任务还在运行就以 long_job 交回。如果没重启试验，就交回同一个完成标记；如果重启了试验、开了新的 trsi "
                     "watch，就交回新的完成标记。" % b["checkin_minutes"])
    L.append("- 本次要做的事：" + b["goal"])
    bl = b["budget_left"]
    L.append("- 可用预算（剩余）：算力小时 %s，标注条数 %s，调用费用 %s；本轮尝试次数 %s。"
             % (_lim(bl["compute_hours"]), _lim(bl["annotation_items"]), _lim(bl["api_cost"]), b["attempts"]))
    L += ["- 输入产物（只读这些，没列出的产物和目录不要找、不要读）：", "", "  | 名字 | 路径 | 说明 |", "  |---|---|---|"]
    for k, p in b["inputs"].items():
        L.append("  | `%s` | `%s` | %s |" % (k, p, _cell(b["input_notes"].get(k, ""))))
    L += ["", "- 输出目录：`%s`，每个产物一个子目录。" % b["outdir"],
          "- 进度文件（可选）：`%s`。想让进度页面和 `rsi run` 的状态行显示你做到哪一步，就随时整个重写它（先写临时文件再 "
          "`os.replace`），格式见 `%s`；不写也可以。"
          % (b["progress_file"], os.path.join(repo_root(), "docs", "interfaces", "progress.md")),
          "- 要报告的门数字：该模块 skill「门」一节列出的各项，逐项给出数值、阈值和是否通过，另报告实际花费。",
          "- 报告的指标、对照组、差值和区间：用 `%s score --task %s --labels <标签行文件> --predictions <预测文件>` 算"
          "（需要对照组时加 `--train <训练集 rows.jsonl>`；需要和另一份预测比时加 `--compare <另一份预测>`，同时加 "
          "`--clusters <inputs.jsonl>` 给出区间用的簇；标签行没有 `meta.state_id` 而要算 `group_prior` 时也加 `--clusters`；"
          "完整写法见模块 skill），把它打印的数原样写进 manifest。这些数的定义固定在代码里，登记时会按同样的定义重算，不一致就被拒。"
          % (b["rsi"], b["task_json"]),
          "- 交回前：对每个产物运行 `%s validate <kind> <dir> --task %s`，通过再交回。你只能运行 `rsi validate` 和 "
          "`rsi score` 这两个 `rsi` 命令；不要运行 `rsi register` 或其他 `rsi` 命令，登记由引擎做。" % (b["rsi"], b["task_json"]),
          "- 一个产物要引用本次另一个产物的登记 id 时（例如 `search` 引用 `candidate`，`plan` 引用 `evaluation`），"
          "先交回被引用的那个；引擎登记后让同一个会话接着做（反馈开头是「# 继续：交回的第二部分」，不算退回），登记 id 在反馈里。",
          "- 遇到需要用户决定的事就停下报告（写进 `questions`），例如要超出预算、要改任务定义、要用封存 test、要刷新验证集或评测集、"
          "要删除产物、输入有问题。不要自己决定，也不要绕过。",
          "- 交回格式：" + b["handback"],
          "- 不要派别的 subagent" + _training_tail(b)]
    return "\n".join(L) + "\n"


def _training_tail(b):
    """Who trains: the train module's trials go through `trsi launch` as its skill says (its brief names the train
    command for its candidates' code_version.entry and the owner's single runs); the data module does not train (the
    round's train and judge dispatches judge its trainset); the other modules have no say on training."""
    if b.get("module") == "train":
        return ("；试验一律按本模块 skill 经 `trsi launch` 训练（上面的训练命令只供参考：候选的 `code_version.entry` 写它，"
                "负责人手动单次训练时调用它）。")
    if b.get("module") == "data":
        return "；本模块不训练：新训练集的好坏由本轮的训练调优训练、评测与决策和当前最佳版本比较来判断。"
    return "。"


def _json(x):
    return "`%s`" % json.dumps(x, ensure_ascii=False, sort_keys=True)


def _lim(x):
    return "不限" if x is None else x


def read_handback(outdir):
    """Check <outdir>/handback.json: {"ok": bool, "errors": [str], "status", "artifacts", "marker", "notes",
    "questions"} (the fields as read, defaults when absent or wrong). Never raises."""
    out = {"ok": False, "errors": [], "status": None, "artifacts": [], "marker": None, "notes": "", "questions": []}
    errs = out["errors"]
    path = os.path.join(outdir, "handback.json")
    try:
        with open(path, encoding="utf-8") as f:
            h = json.load(f)
    except (OSError, ValueError, RecursionError, MemoryError) as e:
        errs.append("handback.json: cannot read %s: %s" % (path, e))
        return out
    if not isinstance(h, dict):
        errs.append("handback.json must be a JSON object")
        return out
    for k in h:
        if k not in HANDBACK_KEYS:
            errs.append("handback.json: unknown field '%s'" % k)
    st = h.get("status")
    if st not in HANDBACK_STATUSES:
        errs.append("status must be one of %s (got %r)" % (", ".join(HANDBACK_STATUSES), st))
    else:
        out["status"] = st
    arts = h.get("artifacts", [])
    if not isinstance(arts, list) or not all(isinstance(a, str) for a in arts):
        errs.append("artifacts must be a list of subdirectory names")
    else:
        for a in arts:
            if not a or a in (".", "..") or "/" in a or "\\" in a or os.path.isabs(a):
                errs.append("artifacts: %r is not a plain subdirectory name" % a)
        if len(set(arts)) != len(arts):
            errs.append("artifacts lists a subdirectory twice")
        out["artifacts"] = list(arts)
    marker = h.get("marker")
    if marker is not None and not (isinstance(marker, str) and os.path.isabs(marker)):
        errs.append("marker must be an absolute path string or null")
    else:
        out["marker"] = marker
    if st == "long_job" and not out["marker"]:
        errs.append("status long_job needs a marker (the completion marker path)")
    notes = h.get("notes", "")
    if not isinstance(notes, str):
        errs.append("notes must be a string")
    else:
        out["notes"] = notes
        if st == "failed" and not notes.strip():
            errs.append("status failed needs notes (the reason)")
    qs = h.get("questions", [])
    if not isinstance(qs, list) or not all(isinstance(q, str) and q for q in qs):
        errs.append("questions must be a list of non-empty strings")
    else:
        out["questions"] = list(qs)
    out["ok"] = not errs
    return out
