"""Mock rsi-train, the training-tuning module (see mocklib for the call, the task and the knobs).

One dispatch runs one search, handed back in two parts. First part: one trial per config through the train command
(producer train), each trial a candidate in trial-<i>/ (trial-<search>-<i>/ from the second search of the dispatch
on), handed back with a record. Every trial starts from the best version's candidate config (brief input
best_candidate; config_source its id) or the train command's defaults, takes its trial config, the round's config
knob on top and a seed of its own; the first trial is the search's default. Second part, in the same session: the
feedback names the trials' registered ids (in the order handed back); the search (the best trial by the
validation primary metric, never worse than the default) is handed back. Sent back after the second part (no
trial through the train gate), the session runs a new search with the default trials. With the long_job knob the
trials run in the background and the hand-back is long_job with the marker search.marker; a check-in (feedback
「# 定时检查」) looks at the job and hands back long_job again; the session continued after the marker says ok
collects the trials. A train gate that fails is reported as it is (passed false), for the engine to decide.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mocklib as M  # noqa: E402

DEFAULT_TRIALS = [{}, {"lr": 0.1}, {"lr": 0.3}]
MARKER = "search.marker"
RELEASE = ".release"  # the check-in knob's own release file of the background trials


def start(c):
    """(config the trials start from, config_source, words on where it came from)."""
    best = c.inputs.get("best_candidate")
    if best:
        cfg = dict(M.read_json(os.path.join(best, "manifest.json"))["config"])
        cfg.pop("seed", None)
        return cfg, M.art_id(best), "当前最佳版本的候选 %s 的配置" % M.art_id(best)
    return {}, "default", "训练命令的默认配置"


def plan_trials(c, st):
    """The trials of a new search: [(name, config)], and the words on what was tried."""
    k = len(st.get("searches") or []) + 1  # this dispatch's k-th search
    base, source, why = start(c)
    given = c.knob("trials") if k == 1 else None
    over = c.knob("config") or {}
    out = []
    for i, delta in enumerate(given or DEFAULT_TRIALS, 1):
        cfg = dict(base)
        cfg.update(delta)
        cfg.update(over)
        cfg["seed"] = 10 * (k - 1) + i - 1  # a seed of its own: no two trials train the same model
        out.append(("trial-%d" % i if k == 1 else "trial-%d-%d" % (k, i), cfg))
    why += "；%d 个试验%s" % (len(out), "，本轮改了 %s" % json.dumps(over, sort_keys=True) if over else "")
    if k > 1:
        why += "；上一次搜索没有候选过训练门，换默认试验重新搜索"
    return out, source, why


def argv_of(c, name, cfg_path, source):
    return (c.inputs["trainset"], c.brief["ids"]["trainset"], c.inputs["valset"], c.brief["ids"]["valset"],
            cfg_path, source, "train", os.path.join(c.outdir, name), os.path.join(c.outdir, name + ".marker"))


def collect(c, st):
    """The trials the train command wrote: each checked; handed back with a record (the first part)."""
    s = st["searches"][-1]
    primary = c.task["metrics"]["primary"]
    done, dirs = [], []
    for name, cfg in s["trials"]:
        first = (M.read_text(os.path.join(c.outdir, name + ".marker")).splitlines() or [""])[0].strip()
        if first != "ok":
            done.append({"dir": None, "config": cfg, "metric_val": None, "compute_hours": 0.0, "status": "failed"})
            continue
        cand = os.path.join(c.outdir, name)
        c.check("candidate", cand)
        m = M.read_json(os.path.join(cand, "manifest.json"))
        done.append({"dir": name, "config": m["config"], "metric_val": m["metrics_val"][primary],
                     "compute_hours": m["cost"]["compute_hours"], "status": "ok", "gate": m["train_gate"]["passed"]})
        dirs.append(cand)
    if not dirs:
        c.fail("没有一个试验成功：%s" % json.dumps(done, ensure_ascii=False))
    passed = [d for d in done if d.get("gate")]
    best = max((d for d in done if d["status"] == "ok"), key=lambda d: d["metric_val"])
    r = M.write_record(c, "record", c.reason("这一轮要一个过训练门的候选"),
                       "在配置空间里换配置会改变验证集上的 %s" % primary, s["why"], "val %s" % primary,
                       done[0]["metric_val"], best["metric_val"], "success" if passed else "failure",
                       "%d 个试验，%d 个过训练门" % (len(dirs), len(passed)),
                       artifacts=[c.brief["ids"]["trainset"], c.brief["ids"]["valset"]], automation="automated")
    s["done"] = done
    st.pop("long_job", None)
    c.save_state(st)
    c.handback(dirs + [r], notes="试验（验证集 %s，训练门）：%s；登记后请把 id 发回，按交回的顺序" % (primary, "；".join(
        "%s %s %s" % (d["dir"], d["metric_val"], "通过" if d.get("gate") else "没过") for d in done)))


def search(c, st, ids):
    """The second part: the search over the trials registered as `ids`."""
    s = st["searches"][-1]
    done = s["done"]
    ok = [d for d in done if d["status"] == "ok"]
    if len(ids) != len(ok):
        c.fail("反馈里有 %d 个候选 id，成功的试验有 %d 个" % (len(ids), len(ok)))
    it = iter(ids)
    rows = []
    for d in done:
        d["candidate_id"] = next(it) if d["status"] == "ok" else None
        rows.append({"candidate_id": d["candidate_id"], "config": d["config"], "metric_val": d["metric_val"],
                     "compute_hours": d["compute_hours"], "status": d["status"]})
    higher = c.task["metrics"]["higher_is_better"]
    default = ok[0]
    best = default
    for d in ok:
        if (d["metric_val"] > best["metric_val"]) if higher else (d["metric_val"] < best["metric_val"]):
            best = d
    spent = sum(d["compute_hours"] for d in done)
    budget = c.brief["budget_left"]["compute_hours"]
    primary = c.task["metrics"]["primary"]
    tid, vid = c.brief["ids"]["trainset"], c.brief["ids"]["valset"]
    out = c.new_dir("search")
    m = M.base_manifest(c.brief, "search", "train", [tid, vid] + ids)
    m.update({"trainset_id": tid, "valset_id": vid, "strategy": "grid over the given configs (mock)",
              "budget": {"compute_hours": budget if budget is not None else round(max(1.0, 2 * spent), 4)},
              "trials": rows, "default_candidate": default["candidate_id"], "best_candidate": best["candidate_id"],
              "gain_vs_default": {"metric": primary, "delta": best["metric_val"] - default["metric_val"]}})
    M.write_json(os.path.join(out, "manifest.json"), m)
    c.check("search", out)
    s["search"] = os.path.basename(out)
    c.save_state(st)
    c.handback([out], notes="调优门：最佳 %s 相对默认 %s 的增益 %+.4f；花费 %.4f 算力小时"
                            % (best["candidate_id"], default["candidate_id"], best["metric_val"] - default["metric_val"],
                               spent))


def checkin(c, st):
    """A check-in during the long job: look at the background process; with the check-in knob, the n-th check-in
    lets the trials run (the job ends) and waits for the marker. Hands back long_job with the same marker."""
    job = st["long_job"]
    job["checkins"] = job.get("checkins", 0) + 1
    c.save_state(st)
    alive = M.pid_alive(job["pid"])
    if job.get("checkins_until") and job["checkins"] >= job["checkins_until"]:
        open(os.path.join(c.outdir, RELEASE), "w").close()
        M.wait_for(job["marker"], 120, within=c.outdir)
    return c.handback([], status="long_job", marker=job["marker"],
                      notes="定时检查 %d：后台进程 pid %d %s；完成标记%s" % (
                          job["checkins"], job["pid"], "在运行" if alive else "已结束",
                          "已出现" if os.path.exists(job["marker"]) else "还没有出现"))


def main():
    c = M.Call("train")
    if c.rehand_if_answered():
        return
    st = c.state()
    job = st.get("long_job")
    if job and c.feedback.startswith("# 定时检查"):
        return checkin(c, st)
    if job and "长任务已结束" in c.feedback:
        first = (M.read_text(job["marker"]).splitlines() or [""])[0].strip()
        if first != "ok":
            c.fail("后台试验的完成标记 %s 不是 ok：%s" % (job["marker"], first or "（没有标记）"))
        return collect(c, st)
    s = (st.get("searches") or [None])[-1]
    ids = re.findall(r"candidate-\d{4}", c.reason())
    if s and s.get("done") and not s.get("search") and ids:
        return search(c, st, ids)
    trials, source, why = plan_trials(c, st)
    st.setdefault("searches", []).append({"trials": trials, "why": why})
    argvs = []
    for name, cfg in trials:
        cfg_path = os.path.join(c.outdir, name + ".config.json")
        M.write_json(cfg_path, cfg)
        argvs.append(M.train_argv(c.brief, *argv_of(c, name, cfg_path, source)))
    knob = c.knob("long_job") if c.first else None
    if knob:
        marker = os.path.join(c.outdir, MARKER)
        log = os.path.join(c.outdir, "search.log")
        until = knob.get("checkins")
        release = os.path.join(c.outdir, RELEASE) if until else knob.get("release")
        pid = M.start_background(argvs, marker, log, release=release, wait_s=knob.get("seconds", 1))
        st["long_job"] = {"marker": marker, "pid": pid, "checkins_until": until}
        c.save_state(st)
        return c.handback([], status="long_job", marker=marker,
                          notes="%d 个试验在后台运行（pid %d，日志 %s），完成后写标记 %s" % (len(argvs), pid, log, marker))
    for argv in argvs:
        M.run_argv(c.brief, argv)
    c.save_state(st)
    collect(c, st)


if __name__ == "__main__":
    main()
