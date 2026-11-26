"""Code-built briefs and the hand-back contract.

build_brief(ws, module, dispatch_id=None) assembles what a module may read and write as a dict; the
inputs, per purpose (rsi.flow.PURPOSES):

  data     sets         task; corpus and pool once registered (a resumed sets); the brief also carries `sources`,
                        the task's raw data (rsi.datasplit.source_paths: data.sources), read in this dispatch only
           trainset     task, trainset (current), valset, plan (the one that opened the round), feedback, pool,
                        best_candidate (the best version's candidate, when there is one: dgen score-pool runs its
                        model.predict on the pool); never anything of the evalset or the testset
           refresh      task, plan, trainset, valset, corpus, and the inputs (no labels) of every registered evalset
                        (evalset_inputs_<id>): the new evalset is drawn from the corpus clear of them (the testset
                        and the evalsets' labels are never given back to data)
  train    train        task, trainset, valset, best_candidate (the best version's candidate, when there is one),
                        every registered candidate and search (by id: candidate-NNNN, search-NNNN), plan (the one
                        that opened the round, when there is one); the brief also carries the task's search_space,
                        search, resources.onejev, resources.gpu_ids and flow.checkin_minutes (null: no check-ins)
  compare  compare      task, candidate (the one to compare: the round's chosen candidate), trainset (current) and
                        valset, whole (rows with labels: the baselines are fitted on the trainset and chosen and
                        compared on the valset), evalset_inputs (no labels: each baseline predicts them for the
                        judge), every registered baseline (by id: baseline-NNNN; reused when its spec and sets match)
                        but those with testset predictions
           milestone_compare
                        task, candidate (the best version's), comparison (the best version's: best_comparison), each
                        baseline of it (by id), the rows of the trainset and the valset each one names
                        (trainset_rows_<id>, valset_rows_<id>, for refits) and testset_inputs (never its labels)
  deliver  deliver      task, candidate (the one to deliver), evalset_inputs, valset_inputs (of the
                        candidate's valset); after a refresh also best_candidate with its valset inputs
           milestone_deliver
                        task, candidate (the best version's), evalset_inputs, valset_inputs, testset_inputs
  judge    judge        task, candidate, delivery (the evaluated one), evalset_inputs, evalset_labels,
                        trainset_rows (of the candidate's trainset), plan (the one that opened the round: did it
                        work?), the round's searches, every feedback and record, best_evaluation and
                        best_delivery, best_delivery_current_evalset after a refresh; with ③ Compare, the round's
                        comparison of the candidate on the current evalset and each of its baselines
                        (baseline_eval_<id>: the judge reports them on the evalset in baselines_eval)
           plan         task, the round's evaluation (if any), candidates, deliveries, searches, every
                        feedback and record, best_evaluation (no labels: a closed round has no evaluation)
           milestone_judge
                        the judge inputs for the milestone delivery, plus testset_inputs and testset_labels; with ③
                        Compare, the comparison the milestone scored and this round's baselines with testset
                        predictions (baseline_test_<id>: reported on the testset in baselines_test)

Every data brief also carries `dset` (dset_command: the split tool's command prefix), and the trainset purpose also
`dgen` (dgen_command: the data generation tool's), `features` (as compare's) and `resources` (resources.onejev and
resources.gpu_ids: dgen score-pool runs model.predict on an idle card of gpu_ids); the train and compare briefs carry
`train_command` (compare trains the team's recipe as a baseline with it): the data module does not train (whether a
new trainset helps is judged by the round's train and judge dispatches), so its brief names no train command. The
compare brief also carries `compare` (rsi.rules.compare_settings: null when ③ is off), `policy` (the task's policy
section or null), `features` (rsi.rules.features_settings, the adapter as an absolute path) and `resources`
(resources.onejev and resources.gpu_ids). Every judge brief carries `best`, and `comparison` (the comparison whose
baselines its evaluation reports, null when none) and `strongest` ({"name", "id"} of that comparison's strongest
baseline on the valset, or null). Every purpose of data, train
and deliver also gives the module's own records of earlier rounds (records whose module is this module, for train
also an old workspace's tune records, registered in a round before the brief's); the judge already reads every
record. The read-deny list (denied_paths) keeps every module out of the raw data (raw_data_paths: that of every
registered task version) but data's sets, and out of every data dispatch's output directory but its own.
Only the judge gets the labels of the evalset and the testset; compare gets the trainset's and the valset's rows (with
their labels) as train does. Every path is absolute and lies in a registered artifact directory the module
may read or in its output directory; the workspace root is never given. Every brief names its dispatch
(dispatch_id, null for a brief built without one), its progress_file (<outdir>/progress.json, which the module may
write: docs/interfaces/progress.md), its spend_file (<outdir>/spend.jsonl, where the spend the module causes outside its
own agent session goes: docs/interfaces/spend.md) and budget_left (what remains of each resource of
rsi.rules.RESOURCES: compute_hours, api_usd, agent_usd, annotation_items; null without a budget). read_handback(outdir)
checks the handback.json a module writes into its output directory.
"""
import json
import os
import shlex

import rsi
from rsi import datasplit as DS
from rsi import flow as F
from rsi.progress import PROGRESS_FILE
from rsi.rules import RESOURCES, adapter_path, amount_text, compare_settings, features_settings, flow_settings, ledger
from rsi.timing import plural
from rsi.workspace import SPEND_FILE, RsiError

MODULE_PURPOSES = {
    "data": ("sets", "trainset", "refresh"),
    "train": ("train",),
    "compare": ("compare", "milestone_compare"),
    "deliver": ("deliver", "milestone_deliver"),
    "judge": ("judge", "plan", "milestone_judge"),
}
SKILLS = {m: "rsi-" + m for m in MODULE_PURPOSES}
DECISION_GOAL_MODULES = ("data", "train")  # their goal restates the task's decision when the plan gives them no issue
DEFAULT_TRAIN_COMMAND = os.path.join("modules", "train", "train")
LONG_JOB_MINUTES = 10

PURPOSE_TEXT = {
    "sets": ("Round 1: write the preprocessing script that turns the raw data into the corpus; set the split rule and "
             "produce the testset, evalset, valset, pool and trainset with dset sets (and, when the task's split rule "
             "needs filling in, the filled-in task version)."),
    "trainset": "Produce a new trainset version (with its data quality report).",
    "refresh": "The evalset has reached its use limit and the refresh is approved: produce a new evalset.",
    "train": ("Train & tune: run one search in the task's search space (search_space). First hand back the candidates "
              "of every successful trial of this search and a record; once the engine has registered them, the same "
              "session gets `# Continue: the second part of the hand-back` and then hands back the search."),
    "compare": ("③ Compare (report only, never a gate): fit each baseline of the task's compare section on the "
                "trainset (or reuse a registered one whose spec and sets match), choose its settings on the valset and "
                "predict the evalset inputs with it; hand back the baselines first, then, in the same session, the "
                "comparison of the candidate with them on the valset."),
    "milestone_compare": ("Milestone release: score each baseline of the best version's comparison on the testset "
                          "inputs (a new baseline with predictions_test for each; never the testset labels)."),
    "deliver": ("Package the candidate for evaluation as a delivery and run its predictions on the evalset inputs and "
                "the valset inputs."),
    "milestone_deliver": ("Milestone release: run separate predictions on the testset inputs with the current best "
                          "version's candidate."),
    "judge": ("Evaluate the candidate on the evalset and write the evaluation (the plan comes after the round is "
              "decided)."),
    "plan": ("The round is decided: write the next round's plan from the round's evaluation (a closed round has no "
             "evaluation: then from the round's candidates, searches and records)."),
    "milestone_judge": ("Milestone release: score the milestone delivery with the testset labels and write the "
                        "milestone evaluation (no plan)."),
}

HANDBACK_FORMAT = (
    "To hand back, write `handback.json` in the output directory: {\"status\": \"ok\" | \"failed\" | \"long_job\", "
    "\"artifacts\": [subdirectory names in the output directory, one per artifact], \"marker\": the absolute path of "
    "the long job's completion marker (required when status is long_job, else null), \"notes\": your report (gate "
    "numbers, cost, problems met), \"questions\": [what the user must decide, [] when nothing]}. With status failed, "
    "write the reason in notes. When questions is not empty, none of the artifacts of this hand-back is registered: "
    "once the user answers, the answer goes back to the same session (the feedback starts with `# The user answered "
    "your questions`, not a return); change or confirm by the answer, then hand back again every subdirectory to "
    "register. The questions of a long_job hand-back are asked only once the completion marker is there; when it lists "
    "no artifact, the answer comes with the `# Long job finished` feedback. A long job (expected to take more than %d "
    "minutes) is started in the background and handed back at once as long_job; when it ends, succeeded or failed, "
    "it must write the completion marker, whose first line is `ok` or `failed`; do not wait or poll. "
    "If the completion marker is already there when you would hand back long_job (the long job has already ended), do "
    "not hand back long_job: the engine never accepts a marker older than this launch; read the results, finish the "
    "work, then hand back. Before handing back long_job, write `<marker>.pid` next to the completion marker: the first "
    "line is the long job process's own pid (written by the long job itself, e.g. `$$` in a shell or `os.getpid()` in "
    "Python), the second line its start time (Unix seconds, `date +%%s`); when the run is stopped the engine "
    "terminates the long job by it. Use a new marker path for every long job (e.g. `<name>-<k>.done`, `<k>` one more "
    "each time), and run `rm -f <marker> <marker>.pid` before starting it: a marker file older than this launch does "
    "not count as this long job's end."
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


def dgen_command(task):
    """The command prefix of the data module's generation tool (modules/data/dgen), given to the data brief of the
    trainset purpose. It runs in the task's resources.onejev.python (the ojh venv, where install_deps.sh puts numpy and
    scikit-learn; ~ and $VAR expanded, as trsi does), else python3."""
    root = repo_root()
    path = os.pathsep.join([os.path.join(root, "modules", "data"), os.path.join(root, "modules", "common"), root])
    python = ((task.get("resources") or {}).get("onejev") or {}).get("python") or "python3"
    python = os.path.expandvars(os.path.expanduser(str(python)))
    return "PYTHONPATH=%s %s -m dgen" % (shlex.quote(path), shlex.quote(python))


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
        self.notes[name] = "%s (%s)" % (note, aid)


def _round_ids(ws, kind, rnd=None):
    rnd = ws.state["round"] if rnd is None else rnd
    return [a for a in ws.state["rounds"][str(rnd)]["artifacts"] if ws.entry(a)["kind"] == kind]


def _plan_of_round(ws):
    """The plan that opened the current round (the previous round's), or None (round 1)."""
    prev = ws.state["rounds"].get(str(ws.state["round"] - 1))
    return prev["plan_id"] if prev else None


def _issue_text(i):
    e = i.get("expected") or {}
    return "- %s (evidence: %s); change: %s; expected: %s %s %s" % (
        i.get("issue"), i.get("evidence"), i.get("change"), e.get("metric"), e.get("direction"), e.get("size"))


def _goal(ws, module, purpose, task):
    lead = PURPOSE_TEXT[purpose]
    plan = _plan_of_round(ws)
    issues = [i for i in ws.manifest(plan).get("issues", []) if i.get("module") == module] if plan else []
    if issues:
        return "%s\nIssues that plan %s assigns to this module this round:\n%s" % (
            lead, plan, "\n".join(_issue_text(i) for i in issues))
    if module in DECISION_GOAL_MODULES:
        d = task["decision"]
        return "%s\nBy the task definition: %s (input: %s)" % (lead, d["instructions"], task["input"]["description"])
    return lead


def _budget_left(ws, task):
    """What remains of each resource (rsi.rules.RESOURCES, by its used key), never below 0; None without a budget."""
    return dict((key, None if budget is None else max(0, budget - used))
                for key, _label, used, budget in ledger(task, ws.state))


def _best(ws):
    name = ws.state["best"]
    v = ws.version(name) if name is not None else None
    if v is None:
        return None
    return {"version": name, "candidate_id": v["candidate_id"], "evaluation_id": v["evaluation_id"],
            "delivery_id": ws.manifest(v["evaluation_id"])["delivery_id"],
            "delivery_on_current_evalset": ws.best_delivery_on_current_evalset()}


def _ids(module, purpose, ws):
    """The current set ids a module writes into manifests: trainset and valset; the evalset only for compare,
    deliver and judge, the testset only for the milestone purposes."""
    kinds = ["trainset", "valset"]
    if module in ("compare", "deliver", "judge"):
        kinds.append("evalset")
    if purpose in ("milestone_deliver", "milestone_compare", "milestone_judge"):
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
            inp.art(e["id"], e["id"], "This module's experiment record of round %d" % e["round"])


def _data_inputs(ws, purpose, inp):
    if purpose == "sets":  # a resumed sets: what this dispatch registered already (dset reads it, never rewrites it)
        for kind, note in (("corpus", "Registered corpus (the preprocessing script and every labelled row)"),
                           ("pool", "Registered pool")):
            if ws.current(kind):
                inp.art(kind, ws.current(kind), note)
        return
    if purpose == "trainset" and ws.current("trainset"):
        inp.art("trainset", ws.current("trainset"), "Current trainset")
    if purpose == "refresh":
        inp.art("trainset", _cur(ws, "trainset"), "Current trainset (the new evalset must not overlap it)")
    inp.art("valset", _cur(ws, "valset"), "Valset")
    plan = _plan_of_round(ws)
    if plan:
        inp.art("plan", plan, "Plan that opened this round")
    if purpose == "trainset":
        for f in ws.entries("feedback"):
            inp.art(f["id"], f["id"], "Execution feedback")
        if ws.current("pool"):
            inp.art("pool", ws.current("pool"), "Pool (every labelled row of the training period): the new trainset is "
                    "drawn from it")
        best = _best(ws)
        if best is not None:
            inp.art("best_candidate", best["candidate_id"], "Candidate of the current best version %s: dgen score-pool "
                    "runs its model.predict on the pool rows outside the trainset, and dgen diagnose takes its "
                    "errors from those decisions" % best["version"])
    if purpose == "refresh":
        if ws.current("corpus"):
            inp.art("corpus", ws.current("corpus"), "Corpus (every labelled row): the new evalset is drawn from it")
        for e in ws.entries("evalset"):
            inp.art("evalset_inputs_" + e["id"], e["id"], "Unlabelled inputs of an earlier evalset: the new evalset "
                    "must not overlap them", "inputs")


def _train_inputs(ws, inp):
    inp.art("trainset", _cur(ws, "trainset"), "Current trainset")
    inp.art("valset", _cur(ws, "valset"), "Valset")
    best = _best(ws)
    if best is not None:
        inp.art("best_candidate", best["candidate_id"], "Candidate of the current best version %s: the starting point "
                "of this search" % best["version"])
    for kind in ("candidate", "search"):
        for e in ws.entries(kind):
            inp.art(e["id"], e["id"], "Registered " + kind)
    plan = _plan_of_round(ws)
    if plan:
        inp.art("plan", plan, "Plan that opened this round")


def _compare_inputs(ws, d, purpose, inp):
    """compare: the candidate (the dispatch's, else the chosen one), the trainset and the valset whole, the evalset's
    inputs and every registered baseline but the milestone's (they hold testset predictions); milestone_compare: the
    best version's candidate and comparison, each baseline of it with the trainset and valset rows it names (for
    refits) and the testset's inputs. Never a label file of the evalset or the testset."""
    cand = (d or {}).get("candidate_id")
    if purpose == "milestone_compare":
        best = _best(ws)
        cand = cand or (best or {}).get("candidate_id")
        if cand is not None:
            inp.art("candidate", cand, "Candidate of the current best version")
        comp = F.best_comparison(ws)
        if comp is not None:
            inp.art("comparison", comp, "Comparison of the best version's candidate: score each of its baselines on "
                    "the testset inputs")
            for name, bid in sorted(ws.manifest(comp)["baselines"].items()):
                inp.art(bid, bid, "Baseline %s of that comparison" % name)
                m = ws.manifest(bid)
                for kind in ("trainset", "valset"):
                    sid = m.get(kind + "_id")
                    if sid:
                        inp.art("%s_rows_%s" % (kind, sid), sid, "Rows of the %s the baselines of that comparison "
                                "were fitted on (for refits)" % kind, "rows")
        inp.art("testset_inputs", _cur(ws, "testset"), "Inputs of the sealed testset (the milestone release is "
                "approved; never its labels)", "inputs")
        return
    cand = cand or F.chosen_candidate(ws)
    if cand is not None:
        inp.art("candidate", cand, "Candidate to compare with the baselines")
    inp.art("trainset", _cur(ws, "trainset"), "Current trainset: the baselines are fitted on it")
    inp.art("valset", _cur(ws, "valset"), "Valset: each baseline's settings are chosen and the comparison is made "
            "on it")
    inp.art("evalset_inputs", _cur(ws, "evalset"), "Evalset inputs (no labels): every baseline predicts them for the "
            "judge", "inputs")
    for e in ws.entries("baseline"):
        if "predictions_test" not in ws.manifest(e["id"])["files"]:
            inp.art(e["id"], e["id"], "Registered baseline (reused when its spec, trainset, valset and evalset match)")


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
        inp.art("candidate", cand, "Candidate for evaluation")
        inp.art("valset_inputs", ws.manifest(cand)["valset_id"], "Unlabelled inputs of the candidate's valset, to "
                "reproduce its valset predictions", "inputs")
    if best_cand is not None and best_cand != cand:
        inp.art("best_candidate", best_cand, "Candidate of the current best version: re-deliver it on the new evalset")
        inp.art("best_valset_inputs", ws.manifest(best_cand)["valset_id"], "Unlabelled inputs of the valset of the "
                "best version's candidate", "inputs")
    inp.art("evalset_inputs", _cur(ws, "evalset"), "Evalset inputs", "inputs")
    if purpose == "milestone_deliver":
        inp.art("testset_inputs", _cur(ws, "testset"), "Inputs of the sealed testset (the milestone release is "
                "approved)", "inputs")


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


def _judge_baselines(ws, purpose, cand, inp):
    """③ Compare's part of a judge brief; returns the comparison id (None: none). judge: the round's comparison of the
    evaluated candidate on the current evalset (rsi.flow.comparison_of) and each of its baselines (baseline_eval_<id>,
    the baseline directory: the judge reports its predictions_eval in baselines_eval); milestone_judge: each baseline
    of this round with predictions_test (baseline_test_<id>: for baselines_test, the milestone's scoring of the best
    version's comparison on the testset inputs) and that comparison (rsi.flow.best_comparison)."""
    if purpose == "milestone_judge":
        comp = F.best_comparison(ws)
        for bid in F.milestone_baselines(ws):
            m = ws.manifest(bid)
            inp.art("baseline_test_" + bid, bid, "Baseline %s scored on the testset inputs (made from %s): report it "
                    "in baselines_test, its predictions_test against the testset labels" % (m["name"], m["reused_from"]))
    else:
        comp = F.comparison_of(ws, [ws.entry(a) for a in _round_ids(ws, "comparison")], cand)
        for name, bid in sorted(ws.manifest(comp)["baselines"].items()) if comp is not None else []:
            inp.art("baseline_eval_" + bid, bid, "Baseline %s of %s: report it in baselines_eval, its predictions_eval "
                    "against the evalset labels" % (name, comp))
    if comp is not None:
        inp.art("comparison", comp, "③ Compare's comparison of the candidate with the baselines on the valset (report "
                "only, never a gate; it may inform the failure analysis)")
    return comp


def _judge_inputs(ws, purpose, inp):
    """The judge's inputs (see the module docstring); returns the comparison its evaluation reports the baselines of
    (_judge_baselines), or None."""
    best = _best(ws)
    comp = None
    if purpose == "plan":
        for kind in ("evaluation", "candidate", "delivery"):
            for a in _round_ids(ws, kind):
                m = ws.manifest(a) if kind in ("evaluation", "delivery") else {}
                if m.get("uses_testset") is True or "predictions_test" in m.get("files", {}):
                    continue  # milestone artifacts carry test results: no plan reads them
                inp.art(a, a, "This round's " + kind)
    else:
        milestone = purpose == "milestone_judge"
        dlv = _eval_delivery(ws, milestone)
        if dlv is not None:
            cand = ws.manifest(dlv)["candidate_id"]
            inp.art("delivery", dlv, "Milestone delivery (with the testset predictions)" if milestone
                    else "Delivery of the round's candidate for evaluation")
            inp.art("candidate", cand, "Candidate under evaluation")
            inp.art("trainset_rows", ws.manifest(cand)["trainset_id"], "Rows of the trainset of the candidate under "
                    "evaluation, only to compute the baselines", "rows")
            comp = _judge_baselines(ws, purpose, cand, inp)
        inp.art("evalset_inputs", _cur(ws, "evalset"), "Evalset inputs", "inputs")
        inp.art("evalset_labels", _cur(ws, "evalset"), "Evalset labels (only the judge gets labels)", "labels")
        plan = None if milestone else _plan_of_round(ws)
        if plan:
            inp.art("plan", plan, "Plan that opened this round (written last round): check it against this round's "
                    "evaluation to judge whether it worked")
        if milestone:
            inp.art("testset_inputs", _cur(ws, "testset"), "Inputs of the sealed testset", "inputs")
            inp.art("testset_labels", _cur(ws, "testset"), "Labels of the sealed testset (the milestone release is "
                    "approved)", "labels")
        if not milestone and best is not None and best["delivery_on_current_evalset"] != best["delivery_id"]:
            if best["delivery_on_current_evalset"]:
                inp.art("best_delivery_current_evalset", best["delivery_on_current_evalset"],
                        "Delivery of the best version on the current evalset: compute the best version's side of "
                        "the metrics from it and list it in the evaluation's parents")
        if milestone and best is not None:
            inp.art("best_evaluation", best["evaluation_id"], "Evaluation of the current best version")
    for a in _round_ids(ws, "search"):
        inp.art(a, a, "This round's search")
    if best is not None:
        inp.art("best_evaluation", best["evaluation_id"], "Evaluation of the current best version %s" % best["version"])
        inp.art("best_delivery", best["delivery_id"], "Delivery of the current best version %s" % best["version"])
    for kind, note in (("feedback", "Execution feedback"), ("record", "Experiment record")):
        for e in ws.entries(kind):
            inp.art(e["id"], e["id"], note)
    return comp


def _strongest(ws, comp):
    """{"name", "id"} of comparison `comp`'s strongest baseline (on the valset), or None."""
    m = ws.manifest(comp) if comp is not None else {}
    name = m.get("strongest")
    return {"name": name, "id": m["baselines"][name]} if name is not None else None


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
    inp.art("task", tid, "Task definition", "task.json")
    comp = None
    if module == "data":
        _data_inputs(ws, purpose, inp)
    elif module == "train":
        _train_inputs(ws, inp)
    elif module == "compare":
        _compare_inputs(ws, d, purpose, inp)
    elif module == "deliver":
        _deliver_inputs(ws, d, purpose, inp)
    else:
        comp = _judge_inputs(ws, purpose, inp)
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
        "spend_file": os.path.join(outdir, SPEND_FILE),  # docs/interfaces/spend.md
        "rsi": rsi_command(), "budget_left": _budget_left(ws, task),
        "attempts": "%d / %d" % (len(ws.attempts()), ws.attempt_limit()),
        "goal": _goal(ws, module, purpose, task), "handback": HANDBACK_FORMAT,
    }
    if module == "data":
        brief["dset"] = dset_command()
        if purpose == "sets":
            brief["sources"] = DS.source_paths(task)
        if purpose == "trainset":
            res = task.get("resources") or {}
            brief["dgen"] = dgen_command(task)
            brief["features"] = dict(features_settings(task), adapter=adapter_path(task))
            brief["resources"] = dict((k, res[k]) for k in ("onejev", "gpu_ids") if k in res)  # score-pool's card
    if module == "train":
        res = task.get("resources") or {}
        brief["train_command"] = train_command(task)
        brief["search_space"] = task.get("search_space")
        brief["search"] = task.get("search") or {}
        brief["resources"] = dict((k, res[k]) for k in ("onejev", "gpu_ids") if k in res)
        brief["checkin_minutes"] = flow_settings(task)["checkin_minutes"]  # null: no check-ins
    if module == "compare":
        res = task.get("resources") or {}
        brief["compare"] = compare_settings(task)
        brief["policy"] = task.get("policy")
        brief["features"] = dict(features_settings(task), adapter=adapter_path(task))
        brief["resources"] = dict((k, res[k]) for k in ("onejev", "gpu_ids") if k in res)
        brief["train_command"] = train_command(task)
    if module == "judge":
        brief["best"] = _best(ws)
        brief["comparison"] = comp  # ③ Compare: the comparison whose baselines the evaluation reports (null: none)
        brief["strongest"] = _strongest(ws, comp)
    return brief


DATASET_KINDS = ("trainset", "valset", "evalset", "testset", "corpus", "pool")
# baseline and comparison hold evalset (and at the milestone testset) predictions, like a delivery.
DECISION_DENIED_KINDS = DATASET_KINDS + ("delivery", "evaluation", "feedback", "baseline", "comparison")
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
    evaluation, feedback, baseline and comparison, the data dispatches' output directories (data_outdirs) and the raw
    data (raw_data_paths)."""
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
    L = ["You are in charge of one module of an RSI run. Load the skill `%s` first and work by its \"Inputs\", "
         "\"Outputs\" and \"Gates\" sections." % b["skill"], ""]
    L.append("- Skill: `%s`" % b["skill"])
    L.append("- Task definition: `%s`; the version for people is `TASK.md` in the same directory" % b["task_json"])
    L.append("- Round: `%d`. In every artifact's manifest write this number as `round`, `%s` as `task_id`, `%s` as "
             "`task_version` and null as `id`." % (b["round"], b["task_id"], b["task_version"]))
    L.append("- Current registered ids: " + ", ".join("%s `%s`" % (k, ids[k] or "none") for k in ids)
             + ". Use these ids when a manifest refers to these sets.")
    if "best" in b:
        best = b["best"]
        if best is None:
            L.append("- Current best version: none (write null as `best_before`).")
        else:
            L.append("- Current best version: version name `%s`, evaluation `%s`, delivery `%s`. Write `best_before` "
                     "from these values. When the evalset was refreshed, the best version's delivery on the current "
                     "evalset: `%s` (compute the best version's side of the metrics from it and write its id into the "
                     "evaluation's `parents`)."
                     % (best["version"], best["evaluation_id"], best["delivery_id"],
                        best["delivery_on_current_evalset"] or "none"))
    if b.get("comparison"):
        st = b.get("strongest")
        milestone = b.get("purpose") == "milestone_judge"
        on = "testset" if milestone else "evalset"
        L.append("- ③ Compare (report only, never a gate): comparison `%s`, strongest baseline on the valset %s. The "
                 "evaluation's `%s` is the object `rsi score` prints with `--clusters <%s>/inputs.jsonl --baselines-on "
                 "%s` and one `--baseline <dir>` per input `%s<id>`; write it as it is, as this module's skill says."
                 % (b["comparison"], "`%s` (`%s`)" % (st["name"], st["id"]) if st else "none",
                    "baselines_test" if milestone else "baselines_eval", on, on,
                    "baseline_test_" if milestone else "baseline_eval_"))
    if "train_command" in b:
        L.append("- Train command: `%s`. Call it with `PYTHONPATH=%s` in the environment." % (b["train_command"],
                                                                                           repo_root()))
    if b.get("sources"):
        L.append("- Raw data (the task's `data.sources`; read it in this dispatch only, every later dispatch is denied "
                 "it): %s." % ", ".join("`%s`" % p for p in b["sources"]))
    if "dset" in b:
        L.append("- Split tool: `%s`; see this module's skill for its use." % b["dset"])
    if "dgen" in b:
        L.append("- Data generation tool: `%s`; see this module's skill for its use; it reads only the inputs below."
                 % b["dgen"])
    if b.get("module") == "train":
        L.append("- Search space (the task's `search_space`; search only inside it; a candidate manifest's "
                 "`config_request` must lie in it and its `code_version.entry` must be the train command above): %s"
                 % (_json(b.get("search_space")) if b.get("search_space") is not None
                    else "not given by the task: use the train command's default config"))
        L.append("- Search settings (the task's `search`): %s; resources (the task's `resources.onejev`, "
                 "`resources.gpu_ids`): %s." % (_json(b.get("search") or {}), _json(b.get("resources") or {})))
        if b.get("checkin_minutes"):
            L.append("- During a long job the engine has this session do a check-in every %s (the feedback starts with "
                     "`# Check-in`, not a return): check the environment, deal with any problem first, and while the "
                     "long job is still running hand back long_job. If you did not restart the trials, hand back the "
                     "same completion marker; if you restarted trials and started a new trsi watch, hand back the new "
                     "completion marker." % plural(b["checkin_minutes"], "minute"))
    L.append("- What to do this time: " + b["goal"])
    bl = b["budget_left"]
    L.append("- Budget left: %s; attempts this round %s."
             % (", ".join("%s %s" % (label, _lim(key, bl.get(key))) for key, _bkey, label in RESOURCES),
                b["attempts"]))
    L.append("- Spend file: `%s`. Append one JSON line for each spend this dispatch causes outside your own session "
             "(the engine counts your session's cost itself): `{\"at\", \"source\", \"compute_hours\"?, \"api_usd\"?, "
             "\"annotation_items\"?, \"note\"?}`, `api_usd` only for LLM calls that generate or review data (source "
             "`llm:<tag>`). Never write the hours a candidate manifest records in `cost.compute_hours`: they are counted "
             "from the manifest. The format is in `%s`." % (b["spend_file"], os.path.join(repo_root(), "docs",
                                                                                         "interfaces", "spend.md")))
    L += ["- Input artifacts (read only these; do not look for or read any artifact or directory not listed):", "",
          "  | Name | Path | Description |", "  |---|---|---|"]
    for k, p in b["inputs"].items():
        L.append("  | `%s` | `%s` | %s |" % (k, p, _cell(b["input_notes"].get(k, ""))))
    L += ["", "- Output directory: `%s`, one subdirectory per artifact." % b["outdir"],
          "- Progress file (optional): `%s`. To have the progress page and the status line of `rsi run` show how far "
          "you are, rewrite it whole at any time (write a temporary file, then `os.replace`); the format is in `%s`. "
          "You may leave it out."
          % (b["progress_file"], os.path.join(repo_root(), "docs", "interfaces", "progress.md")),
          "- Gate numbers to report: every item the \"Gates\" section of the module's skill lists, each with its "
          "value, threshold and whether it passed; also report the actual cost.",
          "- Metrics, baselines, deltas and intervals to report: compute them with `%s score --task %s --labels <label "
          "rows file> --predictions <predictions file>` (add `--train <trainset rows.jsonl>` for the baselines; to "
          "compare with another set of predictions add `--compare <other predictions>` together with `--clusters "
          "<inputs.jsonl>` for the clusters of the intervals; also add `--clusters` to compute `group_prior` when the "
          "label rows have no `meta.state_id`; the full form is in the module skill), and write the numbers it prints "
          "into the manifest as they are. Their definitions are fixed in code: registration recomputes them the same "
          "way and refuses any mismatch."
          % (b["rsi"], b["task_json"]),
          "- Before handing back: run `%s validate <kind> <dir> --task %s` on every artifact and hand it back only "
          "once it passes. The only `rsi` commands you may run are `rsi validate` and `rsi score`; do not run `rsi "
          "register` or any other `rsi` command: the engine registers." % (b["rsi"], b["task_json"]),
          "- When an artifact must refer to the registered id of another artifact of this dispatch (e.g. `search` "
          "refers to `candidate`, `plan` to `evaluation`), hand back the one referred to first; once it is registered "
          "the engine has the same session go on (the feedback starts with `# Continue: the second part of the "
          "hand-back`, not a return), and the registered ids are in the feedback.",
          "- When something needs the user's decision, stop and report it (in `questions`), e.g. going over budget, "
          "changing the task definition, using the sealed test data, refreshing a validation or evaluation set, "
          "deleting artifacts, a problem with the inputs. Do not decide it yourself and do not work around it.",
          "- Hand-back format: " + b["handback"],
          "- Do not dispatch other subagents" + _training_tail(b)]
    return "\n".join(L) + "\n"


def _training_tail(b):
    """Who trains: the train module's trials go through `trsi launch` as its skill says (its brief names the train
    command for its candidates' code_version.entry and the owner's single runs); the data module does not train (the
    round's train and judge dispatches judge its trainset); the other modules have no say on training."""
    if b.get("module") == "train":
        return ("; train every trial through `trsi launch` as this module's skill says (the train command above is "
                "for reference only: candidates write it as `code_version.entry`, and the owner calls it for a single "
                "run by hand).")
    if b.get("module") == "data":
        return ("; this module does not train: whether a new trainset is better is judged by this round's train module "
                "training on it and the judge module comparing the result with the current best version.")
    return "."


def _json(x):
    return "`%s`" % json.dumps(x, ensure_ascii=False, sort_keys=True)


def _lim(key, x):
    """What is left of resource `key` for brief.md: rounded (rsi.rules.amount_text), "no limit" without a budget."""
    return "no limit" if x is None else amount_text(key, x)


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
