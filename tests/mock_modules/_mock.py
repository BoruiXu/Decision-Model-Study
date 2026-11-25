"""Tiny mock modules for the engine tests: rsi.runners.MockRunner runs <module>.py from this directory as

    <module>.py --brief <outdir>/brief.json --outdir <outdir> --session <id> [--feedback <file>]

Each script reads the brief, writes valid artifacts with tests/fixtures.py into its output directory and a
handback.json, as a module agent would, and appends one line per call to <outdir>/mock-calls.jsonl
({"session", "feedback_file", "feedback", "purpose"}). A call is a "first" call when the output directory
holds no archived hand-back (handback-<n>.json) yet: then the module does its whole job; later calls (a
return to the session, or a fresh session given the history) fix what the feedback asks for.

RSI_MOCK_BEHAVIOR names a JSON file {module: {knob: value}}; the knobs (first calls only, unless said):

  trainset      data: "quality_fail" (the trainset fails its quality gate) or "overlap" (it overlaps the
                evalset, so registration refuses it)
  trials        train, any call: how many trial candidates a search hands back (default 1): "candidate" (call 1) or
                "candidate-<call>", then "<that>-t2", ...
  gate_fail     train: the first search's trials fail their train gate (every validation prediction wrong);
                "always": every search's trials fail it
  cost          train, any call: the cost.compute_hours of each candidate it writes (default the fixture's 1.5)
  record        train: the first call also hands back an experiment record (subdirectory "record")
  questions     the hand-back's questions
  questions_only
                the first call hands back its questions and no artifact
  long_job      train: hand back status long_job, marker <outdir>/job.marker (the test writes it); a check-in
                (feedback 「# 定时检查」, any call) hands back long_job with the same marker
  checkin       train, any check-in: {"wait": file} waits for the file first (at most WAIT_S seconds), {"fail": notes}
                hands back status failed with these notes
  refuse_trial  train: the first call's last trial names an unregistered valset, so registration refuses it; a return
                whose feedback names the refused subdirectory hands back a fixed copy (<dir>-fixed)
  long_jobs     train, any call: calls 1..n hand back status long_job, marker <outdir>/job-<call>.marker
  fail          train: hand back status failed with these notes
  skip_search   train, n: the next n calls asked for the search (the followup, ...) hand back nothing
  fail_resume   any call given a feedback-<n>.md (a resumed session): exit RESUME_FAILED once (the value is
                a flag file that marks it done)
  kill_parent   {"flag": file, "when": "after_handback" | "before_handback"}: kill the process that ran the
                script (the engine) once, after or instead of writing handback.json
  started       a file the first call writes (its pid) as soon as it starts
  release       a file the first call waits for (at most WAIT_S seconds) before it works
  after_registered
                an artifact kind: the first call waits (at most WAIT_S seconds) until the workspace has registered
                one (e.g. tune waits for train's candidate, so the two hand-backs are processed in a fixed order)

A call whose feedback carries the user's answers to the last hand-back's questions (the engine registers
nothing until they are answered) hands back the same subdirectories again, data with a new trainset whose notes
quote the answer; after a hand-back of questions only, it does the whole job.

train runs one search per dispatch, in two parts: a call whose feedback's reason line (原因：…) names registered
candidates writes the search over them (search-<call>, producer train; the default is the first, the best the one
with the best validation primary metric); any other call writes a new search's trial candidates.

Milestone purposes: deliver writes a delivery with predictions_test on the testset inputs, judge a milestone
evaluation (uses_testset). The judge's later plans go to new directories (plan-<call>), as a revised plan
must. decide.py is the mock decision agent (see there).
"""
import argparse
import glob
import json
import os
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]

import fixtures  # noqa: E402  (tests/fixtures.py)

TEST_IDS, EVAL_IDS, VAL_IDS, TRAIN_IDS = range(10, 14), range(20, 24), range(0, 4), range(40, 44)
QUALITY_FAIL = {"passed": False, "checks": [{"name": "dup_rate", "value": 0.5, "threshold": 0.05, "passed": False}]}
RESUME_FAILED = 75
WAIT_S = 30


def registered_kinds(registry):
    with open(registry, encoding="utf-8") as f:
        return set(json.loads(line).get("kind") for line in f if line.strip())


class Call(object):
    def __init__(self, module):
        p = argparse.ArgumentParser()
        for opt in ("--brief", "--outdir", "--session"):
            p.add_argument(opt, required=True)
        p.add_argument("--feedback")
        a = p.parse_args()
        self.module, self.outdir, self.session = module, a.outdir, a.session
        with open(a.brief, encoding="utf-8") as f:
            self.brief = json.load(f)
        with open(self.brief["task_json"], encoding="utf-8") as f:
            self.task = json.load(f)
        self.purpose, self.round, self.ids = self.brief["purpose"], self.brief["round"], self.brief["ids"]
        self.feedback_file = os.path.basename(a.feedback) if a.feedback else None
        self.feedback = None
        if a.feedback:
            with open(a.feedback, encoding="utf-8") as f:
                self.feedback = f.read()
        self.first = not glob.glob(os.path.join(self.outdir, "handback-*.json"))
        behavior = {}
        if os.path.isfile(os.environ.get("RSI_MOCK_BEHAVIOR") or ""):
            with open(os.environ["RSI_MOCK_BEHAVIOR"], encoding="utf-8") as f:
                behavior = json.load(f).get(module, {})
        self.behavior = behavior
        calls = os.path.join(self.outdir, "mock-calls.jsonl")
        self.n = 1
        if os.path.exists(calls):
            with open(calls, encoding="utf-8") as f:
                self.n += sum(1 for line in f if line.strip())
        with open(calls, "a", encoding="utf-8") as f:
            f.write(json.dumps({"session": self.session, "feedback_file": self.feedback_file,
                                "feedback": self.feedback, "purpose": self.purpose, "pid": os.getpid()},
                               ensure_ascii=False) + "\n")
        if self.knob("started"):
            with open(self.knob("started"), "w") as f:
                f.write("%d\n" % os.getpid())
        if self.knob("release"):
            self.wait(lambda: os.path.exists(self.knob("release")))
        kind = self.knob("after_registered")
        if kind:
            registry = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.brief["task_json"]))),
                                    "registry.jsonl")
            self.wait(lambda: kind in registered_kinds(registry))

    def wait(self, cond):
        """Wait until cond() holds; exit 1 after WAIT_S seconds, or once the output directory is gone."""
        end = time.time() + WAIT_S
        while not cond():
            if time.time() > end or not os.path.isdir(self.outdir):
                sys.exit(1)
            time.sleep(0.02)

    def knob(self, name):
        return self.behavior.get(name) if self.first else None

    def once(self, flag):
        if os.path.exists(flag):
            return False
        open(flag, "w").close()
        return True

    def make(self, kind, name, **fields):
        return fixtures.make_artifact(kind, os.path.join(self.outdir, name), self.task, round=self.round, **fields)

    def candidate_fields(self, producer, wrong=()):
        out = dict(ids=VAL_IDS, train_ids=TRAIN_IDS, wrong=wrong, producer=producer,
                   config_source="tune" if producer == "tune" else "default", trainset_id=self.ids["trainset"],
                   valset_id=self.ids["valset"])
        if self.behavior.get("cost") is not None:
            out["cost"] = {"compute_hours": self.behavior["cost"]}
        return out

    def handback(self, artifacts, status="ok", marker=None, notes="mock"):
        h = {"status": status, "artifacts": artifacts, "marker": marker, "notes": notes,
             "questions": self.knob("questions") or []}
        kill = self.behavior.get("kill_parent")
        if kill and kill.get("when") == "before_handback" and self.once(kill["flag"]):
            os.kill(os.getppid(), signal.SIGKILL)
            return
        with open(os.path.join(self.outdir, "handback.json"), "w", encoding="utf-8") as f:
            json.dump(h, f, ensure_ascii=False)
        if kill and kill.get("when") == "after_handback" and self.once(kill["flag"]):
            os.kill(os.getppid(), signal.SIGKILL)


def _data(c):
    if c.purpose == "sets" and c.first:
        for kind, ids in (("testset", TEST_IDS), ("evalset", EVAL_IDS), ("valset", VAL_IDS)):
            c.make(kind, kind, ids=ids)
        bad = c.knob("trainset")
        if bad == "quality_fail":
            c.make("trainset", "trainset", ids=TRAIN_IDS, quality=QUALITY_FAIL)
        elif bad == "overlap":
            c.make("trainset", "trainset", ids=range(22, 26))
        else:
            c.make("trainset", "trainset", ids=TRAIN_IDS)
        return c.handback(["trainset", "valset", "evalset", "testset"])  # registered in dependency order
    name = "trainset-%d" % c.n
    c.make("trainset", name, ids=TRAIN_IDS, notes="call %d" % c.n)
    return c.handback([name])


TRIALS_FILE = ".trials.json"  # the latest search's trial subdirectories, in the order handed back


def _trials(c):
    """A new search's trial candidates (the first part); the record with them when the record knob says so."""
    base = "candidate" if c.n == 1 else "candidate-%d" % c.n
    names = [base if i == 1 else "%s-t%d" % (base, i) for i in range(1, int(c.behavior.get("trials") or 1) + 1)]
    fail = c.knob("gate_fail") or c.behavior.get("gate_fail") == "always"
    for i, name in enumerate(names):
        fields = c.candidate_fields("train", VAL_IDS if fail else ())
        if c.knob("refuse_trial") and name == names[-1]:
            fields["valset_id"] = "valset-0009"
        c.make("candidate", name, notes="train call %d trial %d" % (c.n, i + 1), config={"lr": 0.001 * (i + 1)},
               **fields)
    with open(os.path.join(c.outdir, TRIALS_FILE), "w", encoding="utf-8") as f:
        json.dump(names, f)
    out = list(names)
    if c.knob("record"):
        c.make("record", "record", producer="train", module="train", artifacts=[c.ids["trainset"]],
               notes="train call %d" % c.n)
        out.append("record")
    return c.handback(out)


def _search(c, ids):
    """The second part: the search over the registered trials `ids` (in the order handed back)."""
    primary = c.task["metrics"]["primary"]
    with open(os.path.join(c.outdir, TRIALS_FILE), encoding="utf-8") as f:
        names = json.load(f)
    rows = []
    for i, (cid, name) in enumerate(zip(ids, names)):
        with open(os.path.join(c.outdir, name, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        rows.append({"candidate_id": cid, "config": m["config"], "compute_hours": m["cost"]["compute_hours"],
                     "status": "ok", "metric_val": m["metrics_val"][primary]})
    best = max(rows, key=lambda r: r["metric_val"])  # auc: higher is better
    c.make("search", "search-%d" % c.n, producer="train", trials=rows, default_candidate=rows[0]["candidate_id"],
           best_candidate=best["candidate_id"], trainset_id=c.ids["trainset"], valset_id=c.ids["valset"],
           budget={"compute_hours": 10}, gain_vs_default={"metric": primary,
                                                          "delta": best["metric_val"] - rows[0]["metric_val"]})
    return c.handback(["search-%d" % c.n])


def _train(c):
    if (c.feedback or "").startswith("# 定时检查"):
        how = c.behavior.get("checkin") or {}
        if how.get("wait"):
            c.wait(lambda: os.path.exists(how["wait"]))
        if how.get("fail"):
            return c.handback([], status="failed", notes=how["fail"])
        return c.handback([], status="long_job", marker=os.path.join(c.outdir, "job.marker"), notes="checked: it runs")
    if c.knob("long_job"):
        return c.handback([], status="long_job", marker=os.path.join(c.outdir, "job.marker"), notes="started")
    if c.n <= int(c.behavior.get("long_jobs") or 0):
        return c.handback([], status="long_job", marker=os.path.join(c.outdir, "job-%d.marker" % c.n),
                          notes="phase %d started" % c.n)
    if c.knob("fail"):
        return c.handback([], status="failed", notes=c.knob("fail"))
    reason = re.search(r"^原因：(.*)$", c.feedback or "", re.M)
    ids = re.findall(r"candidate-\d{4}", reason.group(1)) if reason else []
    if ids:
        flag = os.path.join(c.outdir, ".skipped-searches")
        skipped = len(open(flag).read()) if os.path.exists(flag) else 0
        if skipped < int(c.behavior.get("skip_search") or 0):
            with open(flag, "a") as f:
                f.write("x")
            return c.handback([], notes="no search this time (call %d)" % c.n)
        return _search(c, ids)
    refused = re.findall(r"子目录 (\S+)（candidate）登记被拒", c.feedback or "")
    if refused:  # fix what registration refused, hand back only that
        fixed = []
        for name in refused:
            c.make("candidate", name + "-fixed", notes="fixed %s" % name, config={"lr": 0.002},
                   **c.candidate_fields("train"))
            fixed.append(name + "-fixed")
        return c.handback(fixed, notes="fixed %s" % ", ".join(refused))
    return _trials(c)


def _deliver(c):
    cand = os.path.basename(c.brief["inputs"]["candidate"])
    if c.purpose == "milestone_deliver":
        path = c.make("delivery", "milestone-delivery", ids=EVAL_IDS, val_ids=VAL_IDS, candidate_id=cand,
                      evalset_id=c.ids["evalset"], valset_id=c.ids["valset"])
        fixtures.add_predictions(path, c.task, "predictions_test", TEST_IDS, testset_id=c.ids["testset"])
        return c.handback(["milestone-delivery"])
    c.make("delivery", "delivery" if c.n == 1 else "delivery-%d" % c.n, ids=EVAL_IDS, val_ids=VAL_IDS,
           candidate_id=cand, evalset_id=c.ids["evalset"], valset_id=c.ids["valset"])
    return c.handback(["delivery" if c.n == 1 else "delivery-%d" % c.n])


def _judge(c):
    best = c.brief["best"]
    if c.purpose == "milestone_judge":
        c.make("evaluation", "milestone-evaluation", ids=EVAL_IDS, train_ids=TRAIN_IDS, test_ids=TEST_IDS,
               uses_testset=True, delivery_id=os.path.basename(c.brief["inputs"]["delivery"]),
               candidate_id=os.path.basename(c.brief["inputs"]["candidate"]), evalset_id=c.ids["evalset"],
               best_before={"version": best["version"], "evaluation_id": best["evaluation_id"]},
               recommendation={"decision": "keep", "reason": "milestone release"})
        return c.handback(["milestone-evaluation"])
    ev = re.findall(r"evaluation-\d{4}", c.feedback or "")
    if c.purpose == "plan" or ev:
        issue = dict(fixtures._plan_issue(), attribution="model_capacity", module="train")
        name = "plan" if not os.path.exists(os.path.join(c.outdir, "plan")) else "plan-%d" % c.n
        c.make("plan", name, evaluation_id=ev[0] if ev else None, entry_module="train", issues=[issue],
               notes="judge call %d" % c.n)
        return c.handback([name])
    c.make("evaluation", "evaluation", ids=EVAL_IDS, train_ids=TRAIN_IDS,
           delivery_id=os.path.basename(c.brief["inputs"]["delivery"]),
           candidate_id=os.path.basename(c.brief["inputs"]["candidate"]), evalset_id=c.ids["evalset"],
           best_before=None if best is None else {"version": best["version"], "evaluation_id": best["evaluation_id"]})
    return c.handback(["evaluation"])


WORK = {"data": _data, "train": _train, "deliver": _deliver, "judge": _judge}
ANSWER = re.compile(r"用户（[^）]*）的回答：(.*)")


def _answered(c, quote):
    """The user answered the last hand-back's questions (nothing of it was registered): hand the same
    subdirectories back, data with a trainset changed by the answer; after questions only, do the whole job."""
    backs = sorted(glob.glob(os.path.join(c.outdir, "handback-*.json")), key=os.path.getmtime)
    with open(backs[-1], encoding="utf-8") as f:
        names = json.load(f).get("artifacts") or []
    c.behavior = {}  # the knobs were the first call's
    if not names:
        c.first = True
        return WORK[c.module](c)
    if c.module == "data" and "trainset" in names:
        new = "trainset-%d" % c.n
        c.make("trainset", new, ids=TRAIN_IDS, notes="按用户的回答修改：%s" % quote)
        names = [new if n == "trainset" else n for n in names]
    return c.handback(names, notes="按用户的回答重新交回")


def main(module):
    c = Call(module)
    flag = c.behavior.get("fail_resume")
    if flag and (c.feedback_file or "").startswith("feedback-") and c.once(flag):
        sys.stderr.write("mock: session %s cannot be resumed\n" % c.session)
        sys.exit(RESUME_FAILED)
    quotes = ANSWER.findall(c.feedback or "")
    if quotes and not c.first:
        return _answered(c, quotes[-1])
    if c.knob("questions_only"):
        return c.handback([], notes="questions first")
    WORK[module](c)
