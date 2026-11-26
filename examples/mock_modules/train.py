"""Mock rsi-train, the train module (see mocklib for the call, the task and the knobs).

One dispatch runs one search, handed back in two parts. First part: one trial per config through the train command
(producer train), each trial a candidate in trial-<i>/ (trial-<search>-<i>/ from the second search of the dispatch
on), handed back with a record. Every trial starts from the best version's candidate config (brief input
best_candidate; config_source its id) or the train command's defaults, takes its trial config, the round's config
knob on top and a seed of its own; the first trial is the search's default. Second part, in the same session: the
feedback names the trials' registered ids (in the order handed back); the search (the best trial by the
validation primary metric, never worse than the default) is handed back. Sent back after the second part (no
trial through the train gate), the session runs a new search with the default trials. With the long_job knob the
trials run in the background and the hand-back is long_job with the marker search.marker; a check-in (feedback
`# Check-in`) looks at the job and hands back long_job again; the session continued after the marker says ok
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
        return cfg, M.art_id(best), "the config of the current best version's candidate %s" % M.art_id(best)
    return {}, "default", "the train command's default config"


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
    why += "; %d trials%s" % (len(out), ", this round changed %s" % json.dumps(over, sort_keys=True) if over else "")
    if k > 1:
        why += "; no candidate of the last search passed the train gate, so a new search with the default trials"
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
        c.fail("no trial succeeded: %s" % json.dumps(done, ensure_ascii=False))
    passed = [d for d in done if d.get("gate")]
    best = max((d for d in done if d["status"] == "ok"), key=lambda d: d["metric_val"])
    r = M.write_record(c, "record", c.reason("this round needs a candidate that passes the train gate"),
                       "another config in the config space changes %s on the valset" % primary, s["why"],
                       "val %s" % primary,
                       done[0]["metric_val"], best["metric_val"], "success" if passed else "failure",
                       "%d trials, %d passed the train gate" % (len(dirs), len(passed)),
                       artifacts=[c.brief["ids"]["trainset"], c.brief["ids"]["valset"]], automation="automated")
    s["done"] = done
    st.pop("long_job", None)
    c.save_state(st)
    trials = "; ".join("%s %s %s" % (d["dir"], d["metric_val"], "passed" if d.get("gate") else "failed") for d in done)
    c.handback(dirs + [r], notes="trials (valset %s, train gate): %s; after registration, send the ids back in the "
                                 "order handed back" % (primary, trials))


def search(c, st, ids):
    """The second part: the search over the trials registered as `ids`."""
    s = st["searches"][-1]
    done = s["done"]
    ok = [d for d in done if d["status"] == "ok"]
    if len(ids) != len(ok):
        c.fail("the feedback has %d candidate ids, and %d trials succeeded" % (len(ids), len(ok)))
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
    c.handback([out], notes="tune gate: best %s against default %s, gain %+.4f; cost %.4f compute hours"
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
                      notes="check-in %d: background process pid %d %s; the completion marker %s" % (
                          job["checkins"], job["pid"], "running" if alive else "ended",
                          "is there" if os.path.exists(job["marker"]) else "has not appeared yet"))


def main():
    c = M.Call("train")
    if c.rehand_if_answered():
        return
    st = c.state()
    job = st.get("long_job")
    if job and c.feedback.startswith("# Check-in"):
        return checkin(c, st)
    if job and "Long job finished" in c.feedback:
        first = (M.read_text(job["marker"]).splitlines() or [""])[0].strip()
        if first != "ok":
            c.fail("the completion marker %s of the background trials is not ok: %s" % (job["marker"],
                                                                                      first or "(no marker)"))
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
                          notes="%d trials running in the background (pid %d, log %s); the marker %s is written when "
                                "they end" % (len(argvs), pid, log, marker))
    for argv in argvs:
        M.run_argv(c.brief, argv)
    c.save_state(st)
    collect(c, st)


if __name__ == "__main__":
    main()
