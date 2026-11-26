from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import webbrowser

from rsi import __version__
from rsi import metrics as M
from rsi import rows as R
from rsi import timing
from rsi.artifacts import load_manifest, validate_artifact
from rsi.report import report_markdown, round_comparison, write_report
from rsi.rules import APPROVAL_ITEMS, compare_settings, flow_settings, stop_settings
from rsi.workspace import DATASETS, KINDS, RsiError, Workspace, check_task_dir

# (name, one-line help); ARGUMENTS below maps every name to its argument setup.
COMMANDS = [
    ("init", "create a workspace and write the task definition"),
    ("validate", "check an artifact directory against its kind schema"),
    ("register", "register an artifact into the workspace registry"),
    ("status", "show workspace state and the current round"),
    ("promote", "apply the promotion rule to this round's evaluation: promote or keep"),
    ("approve", "record a user approval with the user's own words (it also answers a waiting question for it)"),
    ("answer", "answer the question the run waits on: one of its options, with the user's own words"),
    ("next", "run the stop check, then open the next round at the plan's entry module"),
    ("check-stop", "print whether a stop condition holds (changes nothing)"),
    ("close-round", "close a round that has no evaluation (no candidate could be evaluated): keep"),
    ("stop", "record a user-requested stop with the user's own words"),
    ("report", "print the run report (Markdown), or write it with --out; also rewrites the progress pages"),
    ("score", "compute the task's known metrics, baselines and the primary metric's delta (no workspace)"),
    ("brief", "print the Markdown brief of a module's dispatch (no dispatch record needed)"),
    ("run", "run the main flow: dispatch the module agents, register their hand-backs and decide, until the run "
            "waits (for the user, a long job or a manual hand-back) or is done; prints why it stopped"),
    ("dashboard", "rewrite the progress pages (<ws>/dashboard.html and <ws>/dashboard/), print the overview's "
                  "path and open it in the browser"),
]

# Commands that change the workspace: after one succeeds, main rewrites the progress pages.
DASHBOARD_COMMANDS = ("init", "register", "promote", "close-round", "approve", "answer", "next", "stop", "report")
# The overview page (rsi.dashboard.DASHBOARD_FILE). rsi.dashboard is imported only where the pages are
# written, so an error in the page code can never break another command (modules' validate and score).
DASHBOARD_PAGE = "dashboard.html"


def _load_task_file(path: str) -> dict:
    """Load and check the task definition given by --task (task.json or its directory)."""
    if os.path.isdir(path):
        path = os.path.join(path, "task.json")
    try:
        with open(path, encoding="utf-8") as f:
            task = json.load(f)
    except (OSError, ValueError) as e:
        raise RsiError("--task: cannot load %s: %s" % (path, e))
    errors = R.check_task(task)
    if errors:
        raise RsiError(["--task: " + e for e in errors])
    return task


def open_in_browser(path: str) -> bool:
    """Open the page in the default browser; False when RSI_NO_BROWSER=1, a Unix host has no display, or opening failed."""
    if os.environ.get("RSI_NO_BROWSER") == "1":
        return False
    if sys.platform not in ("darwin", "win32") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False  # headless Unix: webbrowser would run a console browser in the foreground
    try:
        return bool(webbrowser.open(pathlib.Path(path).resolve().as_uri()))
    except Exception:
        return False


def _cmd_init(args: argparse.Namespace) -> int:
    print(Workspace.init(args.ws, args.task).current("task"))
    return 0


def _cmd_brief(args: argparse.Namespace) -> int:
    from rsi import briefs
    print(briefs.render_brief(briefs.build_brief(Workspace(args.ws), args.module, purpose=args.purpose)), end="")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Print why the run stopped (one line, or with --json {"status", "reason"}) and exit with
    engine.STATUS_EXIT_CODES: 0 done, 10 waiting for the user, 11 for a long job, 12 for a manual hand-back,
    13 --max-steps used up; a refusal or error exits 1 (with --json also printed as status "error"). The status lines
    (rsi.live.now_line) go to stderr, so stdout keeps only the reason."""
    from rsi import engine
    from rsi.runners import make_runner
    try:
        ws = Workspace(args.ws)
        status, reason = engine.run_status(ws, make_runner(args.runner, ws, mock_dir=args.mock_dir),
                                           max_steps=args.max_steps, status=sys.stderr)
    except RsiError as e:
        if not args.json:
            raise
        print(json.dumps({"status": "error", "reason": "; ".join(e.errors)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": status, "reason": reason}, ensure_ascii=False) if args.json else reason)
    return engine.STATUS_EXIT_CODES[status]


def _cmd_dashboard(args: argparse.Namespace) -> int:
    ws = Workspace(args.ws)
    try:
        from rsi.dashboard import write_dashboard
        path = write_dashboard(ws)
    except Exception as e:  # the pages are a convenience: any failure is a plain error, never a traceback
        raise RsiError("cannot write %s: %s" % (os.path.join(ws.root, DASHBOARD_PAGE), e))
    print(path)
    if not args.no_open:
        open_in_browser(path)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.kind == "task":
        if args.task is not None:
            raise RsiError("--task is not used with kind task: the directory itself is the task")
        _, errors = check_task_dir(args.dir)
    else:
        if args.task is None:
            raise RsiError("--task <task.json> is required for kind %s" % args.kind)
        errors = validate_artifact(args.kind, args.dir, _load_task_file(args.task))
    if errors:
        raise RsiError(errors)
    print("OK")
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    """Print the new id, then the module gate result in parentheses (trainset, candidate, search),
    then `(not verified: <names>)` for reported metrics and baselines rsi does not compute, then for a
    delivery `(decision mismatches: <n>)` when the task allowed flipped near-boundary rows. Warnings
    about a registration that succeeded (e.g. a handed-over source left behind) go to stderr as
    `warning: ...`; the exit code stays 0."""
    ws = Workspace(args.ws)
    aid = ws.register(args.kind, args.dir)
    line, note, skipped = aid, ws.summary(aid), ws.unverified(aid)
    if note is not None:
        line += " (%s)" % note
    if skipped:
        line += " (not verified: %s)" % ", ".join(skipped)
    flips = ws.decision_mismatches(aid)
    if flips:
        line += " (decision mismatches: %d)" % flips
    print(line)
    for w in ws.warnings:
        print("warning: %s" % w, file=sys.stderr)
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    decision, version, reason = Workspace(args.ws).promote(args.evaluation_id)
    print("%s: %s" % ("promote " + version if decision == "promote" else "keep", reason))
    return 0


def _cmd_close_round(args: argparse.Namespace) -> int:
    from rsi import decisions  # the round's failed dispatches whose long job may still run are closed, the jobs ended
    decision, _version, reason = decisions.close_round(Workspace(args.ws), args.reason)
    print("%s: %s" % (decision, reason))
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    print("stop: %s" % Workspace(args.ws).stop(args.by, args.quote))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    ws = Workspace(args.ws)
    if args.out is None:
        sys.stdout.write(report_markdown(ws))
    else:
        print("report written to %s" % write_report(ws, args.out))
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    from rsi import decisions
    a = decisions.approve(Workspace(args.ws), args.what, args.by, args.quote, args.ref)
    print("approved %s%s in round %d (by %s)"
          % (a["what"], " " + a["ref"] if a["ref"] else "", a["round"], a["by"]))
    return 0


def _cmd_answer(args: argparse.Namespace) -> int:
    """Print `answered <what>: <option> (by <name>; <what happens>)`; then `rsi run` goes on."""
    from rsi import decisions
    rec = decisions.answer(Workspace(args.ws), args.option, args.by, args.quote, ref=args.ref)
    print("answered %s%s: %s (by %s; %s)" % (rec["point"], " " + rec["ref"] if rec.get("ref") else "", rec["choice"],
                                            rec["by"], rec["effect"]))
    return 0


def _cmd_next(args: argparse.Namespace) -> int:
    n, text = Workspace(args.ws).next_round()
    print("stop: %s" % text if n is None else "round %d: %s" % (n, text))
    return 0


def _cmd_check_stop(args: argparse.Namespace) -> int:
    stop, reason = Workspace(args.ws).stop_check()
    print("%s: %s" % ("stop" if stop else "continue", reason))
    return 0


def _read_rows(option: str, path: str) -> list[dict]:
    try:
        return R.read_jsonl(path)
    except (OSError, ValueError) as e:
        raise RsiError("%s: cannot read %s: %s" % (option, path, e))


def _checked(option: str, errors: list[str]) -> None:
    if errors:
        raise RsiError(["%s: %s" % (option, e) for e in errors])


def _metric_values(names: list[str], preds: list[dict], labels: list[dict], task: dict) -> dict:
    try:
        return dict((n, M.score(n, preds, labels, task)) for n in names)
    except ValueError as e:
        raise RsiError(str(e))


def score_numbers(task: dict, labels: list[dict], preds: list[dict], train: list[dict] | None = None,
                  context: list[dict] | None = None, compare: list[dict] | None = None) -> dict:
    """What `rsi score` prints: metrics, baselines (with train), delta (with compare), not_computed.

    `context` rows (an inputs file) supply meta.cluster for the interval and meta.state_id (rsi.rows.group_of)
    for group_prior; without them both come from the label rows where those carry them.
    """
    m = task["metrics"]
    wanted = []
    for name in [m["primary"]] + list(m["others"]):
        if name not in wanted:
            wanted.append(name)
    names = [n for n in wanted if n in M.KNOWN_METRICS]
    out = {"metrics": _metric_values(names, preds, labels, task)}
    if train is not None:
        groups = M.groups_of(context if context is not None else labels)
        out["baselines"] = {}
        for b in m["baselines"]:
            if b in M.KNOWN_BASELINES:
                try:
                    rows = M.baseline_predictions(b, train, labels, task, groups)
                except ValueError as e:
                    raise RsiError("baseline %s: %s" % (b, e))
                out["baselines"][b] = _metric_values(names, rows, labels, task)
    if compare is not None:
        primary = m["primary"]
        if primary not in M.KNOWN_METRICS:
            raise RsiError("--compare: the primary metric %r is not one rsi computes (%s); rsi cannot compute "
                           "its delta" % (primary, ", ".join(M.KNOWN_METRICS)))
        clusters = M.clusters_of(context if context is not None else labels)
        try:
            out["delta"] = M.paired_delta(primary, preds, compare, labels, task, clusters, **M.ci_settings(task))
        except ValueError as e:
            raise RsiError("--compare: %s" % e)
    out["not_computed"] = [n for n in wanted if n not in M.KNOWN_METRICS] + \
        [b for b in m["baselines"] if b not in M.KNOWN_BASELINES]
    return out


# --baselines-on: the set -> (the evaluation field it prints, the baseline's prediction file of that set).
BASELINES_FIELDS = {"evalset": ("baselines_eval", "predictions_eval"),
                    "testset": ("baselines_test", "predictions_test")}


def baselines_numbers(task: dict, labels: list[dict], preds: list[dict], context: list[dict] | None, on: str,
                      dirs: list[str]) -> dict:
    """What `rsi score --baselines-on <set> --baseline <dir>...` adds: {field: the evaluation's baselines_eval (on
    "evalset") or baselines_test (on "testset")}, exactly as registration recomputes it (rsi.metrics.baselines_scored:
    each baseline's metrics and its delta, `preds` minus its predictions, clusters from `context`). The id is each
    directory's manifest id (its directory name while unregistered)."""
    field, logical = BASELINES_FIELDS[on]
    if context is None:
        raise RsiError("--baselines-on needs --clusters <%s>/inputs.jsonl: the clusters of the intervals" % on)
    if not dirs:
        raise RsiError("--baselines-on needs a --baseline <dir> for each baseline")
    m = task["metrics"]
    if m["primary"] not in M.KNOWN_METRICS:
        raise RsiError("--baselines-on: the primary metric %r is not one rsi computes (%s); rsi cannot compute the "
                       "baselines' deltas" % (m["primary"], ", ".join(M.KNOWN_METRICS)))
    ids = [r["id"] for r in labels]
    baselines = []
    for d in dirs:
        try:
            man = load_manifest(d)
            name, path = man["name"], os.path.join(d, man["files"][logical]["path"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise RsiError("--baseline %s: not a baseline with %s (%s)" % (d, logical, e))
        rows = _read_rows("--baseline", path)
        _checked("--baseline %s" % d, R.check_predictions(rows, task, ids))
        baselines.append((man.get("id") or os.path.basename(os.path.normpath(d)), name, rows))
    names = [n for n in dict.fromkeys([m["primary"]] + list(m["others"])) if n in M.KNOWN_METRICS]
    try:
        return {field: M.baselines_scored(task, labels, preds, M.clusters_of(context), baselines, names)}
    except ValueError as e:
        raise RsiError("--baselines-on: %s" % e)


def _cmd_score(args: argparse.Namespace) -> int:
    task = _load_task_file(args.task)
    labels = _read_rows("--labels", args.labels)
    _checked("--labels", R.check_target_rows(labels, task))
    ids = [r["id"] for r in labels]
    preds = _read_rows("--predictions", args.predictions)
    _checked("--predictions", R.check_predictions(preds, task, ids))
    train = compare = context = None
    if args.train is not None:
        train = _read_rows("--train", args.train)
        _checked("--train", R.check_target_rows(train, task))
    if args.clusters is not None:
        context = _read_rows("--clusters", args.clusters)
    if args.compare is not None:
        compare = _read_rows("--compare", args.compare)
        _checked("--compare", R.check_predictions(compare, task, ids))
    out = score_numbers(task, labels, preds, train, context, compare)
    if args.baselines_on is not None or args.baseline:
        if args.baselines_on is None:
            raise RsiError("--baseline needs --baselines-on evalset or testset")
        out.update(baselines_numbers(task, labels, preds, context, args.baselines_on, args.baseline))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _fmt(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _num(x) -> str:
    return ("%.4f" % x).rstrip("0").rstrip(".")


def _decision(d) -> str:
    if isinstance(d, dict):
        if d.get("evaluation_id") is None:
            return "%s, closed without evaluation" % d.get("decision")
        return "%s%s from %s" % (d.get("decision"), " " + d["version"] if d.get("version") else "",
                                 d.get("evaluation_id"))
    return _fmt(d)


def _stop_settings_line(task: dict) -> str:
    """The effective stop settings; a field that took its default is marked "(default)"."""
    st = stop_settings(task)
    parts = ["target %s" % ("none" if st["target"] is None else "%.6g" % st["target"])]
    for key in ("max_rounds", "patience", "max_attempts_per_round"):
        parts.append("%s %d%s" % (key, st[key], " (default)" if key in st["defaulted"] else ""))
    return "stop settings: " + ", ".join(parts)


def _attempt_lines(ws: Workspace, rnd: dict) -> list[str]:
    """Attempts used in the current round, and the close-round hint when a round cannot go on.

    The hint appears when the attempts are used up, no candidate of the round
    passed the train gate, and the round has no decision and no evaluation.
    """
    s = ws.state
    used, limit = len(ws.attempts()), ws.attempt_limit()
    lines = ["attempts: %d of %d" % (used, limit)]
    if used >= limit and not ws.passed_candidates() and rnd.get("decision") is None \
            and not ws.evaluations() and not s["stopped"]:
        lines.append("all %d attempts of round %d are used and no candidate of round %d passed the train gate: "
                     "run `rsi close-round <ws> --reason <text>`, then have the judge write the plan from the "
                     "round's candidates and records" % (limit, s["round"], s["round"]))
    return lines


def _checkin_lines(ws: Workspace) -> list[str]:
    """One line per long job whose marker is not there while check-ins are on (flow.checkin_minutes): how many
    check-ins it had of how many, and when the next is due (local time), or that the user ended them (wait)."""
    from rsi import flow as F
    minutes = flow_settings(ws.task())["checkin_minutes"]
    lines = []
    for d in ws.dispatches():
        if not minutes or d.get("status") != "long_job" or F.marker_done(ws, d):
            continue
        limit, due = F.checkin_limit(ws, d), F.checkin_due(ws, d)
        when = "no more (the user said wait)" if limit is None else "next due %s" % timing.format_full(
            due.strftime("%Y-%m-%dT%H:%M:%SZ"))
        lines.append("check-ins of long job %s (rsi-%s): %d of %s, every %s minutes; %s"
                     % (d["id"], d.get("module"), int(d.get("checkins") or 0), "-" if limit is None else limit,
                        _num(minutes), when))
    return lines


def _now_line(ws: Workspace) -> str:
    """`now: ` and the status line `rsi run` prints (rsi.live.now_line)."""
    from rsi import live
    try:
        return "now: %s" % live.now_line(ws)
    except Exception as e:  # the rest of `rsi status` still shows
        return "now: (unavailable: %s: %s)" % (type(e).__name__, e)


def _compare_status(ws: Workspace, task: dict) -> str | None:
    """`compare: <comparison id> (vs <strongest> Δ<primary> <delta>)` for the current round's comparison (③ Compare,
    report only; the delta on the valset); None for a task without a compare section and no comparison (an old
    workspace)."""
    comp = round_comparison(ws, ws.state["round"])
    if comp is None:
        return "compare: none this round" if compare_settings(task) is not None else None
    m = ws.manifest(comp)
    name = m["strongest"]
    if name is None:
        return "compare: %s (no baseline compared)" % comp
    d = m["deltas_val"].get(name)
    return "compare: %s (vs %s Δ%s %s)" % (comp, name, task["metrics"]["primary"],
                                           "%+.3f" % d["delta"] if d else "n/a")


def status_lines(ws: Workspace) -> list[str]:
    s = ws.state
    task = ws.task()
    rnd = s["rounds"].get(str(s["round"]), {})
    lines = [
        _now_line(ws),
        "workspace: %s" % ws.root,
        "task: %s v%d (%s)" % (s["task_id"], s["task_version"], ws.current("task")),
        "round: %d (entry %s, decision %s, plan %s)"
        % (s["round"], _fmt(rnd.get("entry")), _decision(rnd.get("decision")), _fmt(rnd.get("plan_id"))),
    ]
    lines += ["%s: %s" % (k, _fmt(ws.current(k))) for k in DATASETS]
    best = ws.version(s["best"]) if s["best"] else None
    if best:
        lines.append("best: %s (%s, %s)" % (s["best"], best["candidate_id"], best["evaluation_id"]))
    else:
        lines.append("best: %s" % _fmt(s["best"]))
    if best:
        bd = ws.best_delivery_on_current_evalset()
        lines.append("best delivery on current evalset: %s"
                     % (bd or "none (re-deliver the best version on the current evalset first)"))
    lines.append("versions: %d, no-promotion streak: %d" % (len(s["versions"]), s["no_promotion_streak"]))
    lines.append(_stop_settings_line(task))
    lines += _attempt_lines(ws, rnd)
    lines += _checkin_lines(ws)
    lines.append("eval uses: %d/%d" % (s["eval_use_count"], task["eval_use_limit"]))
    compare = _compare_status(ws, task)
    if compare:
        lines.append(compare)
    approved = set(a.get("ref") for a in s["approvals"] if a.get("what") == "plan")
    pending = [aid for aid in rnd.get("artifacts", [])
               if ws.entry(aid)["kind"] == "plan" and aid not in approved]
    lines.append("unapproved plans: %s" % (", ".join(pending) or "none"))
    stopped = s["stopped"]
    if stopped:
        at = stopped["at"]  # local time, as the pages and the report show it
        lines.append("stopped: %s (at %s)" % (stopped["reason"], timing.format_full(at) if timing._good(at) else at))
    else:
        lines.append("stopped: no")
    from rsi import live
    lines.append("budget: %s" % live.ledger_text(ws))  # used/budget of each resource, the dollar accounts apart
    # The round's registered artifacts in order, each with its module gate or role (Workspace.summary).
    arts = rnd.get("artifacts", [])
    lines.append("round %d artifacts (%d):" % (s["round"], len(arts)))
    for aid in arts:
        note = ws.summary(aid)
        if ws.entry(aid)["kind"] == "delivery" and "predictions_test" in ws.manifest(aid)["files"]:
            note = "milestone delivery: predictions on the testset inputs"  # the only note a delivery gets
        lines.append("  %s %s%s" % (aid, ws.entry(aid)["kind"], " (%s)" % note if note else ""))
    return lines


def _cmd_status(args: argparse.Namespace) -> int:
    print("\n".join(status_lines(Workspace(args.ws))))
    return 0


def _args_init(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory to create (must not exist or be empty)")
    p.add_argument("--task", required=True, metavar="DIR",
                   help="task directory holding task.json and TASK.md")
    p.add_argument("--no-open", action="store_true", help="do not open the progress page in the browser")
    p.set_defaults(handler=_cmd_init)


def _args_validate(p: argparse.ArgumentParser) -> None:
    p.add_argument("kind", choices=KINDS)
    p.add_argument("dir", help="artifact directory")
    p.add_argument("--task", metavar="TASK_JSON",
                   help="task definition the artifact must follow (not used for kind task)")
    p.set_defaults(handler=_cmd_validate)


def _args_register(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("kind", choices=KINDS)
    p.add_argument("dir", help="source artifact directory; copied into the workspace (an evalset, testset or "
                                "corpus is moved: its source directory is gone after registration)")
    p.set_defaults(handler=_cmd_register)


def _args_status(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.set_defaults(handler=_cmd_status)


def _args_promote(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("evaluation_id", help="registered evaluation of the current round")
    p.set_defaults(handler=_cmd_promote)


def _args_approve(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("what", choices=APPROVAL_ITEMS, help="approval item")
    p.add_argument("--by", required=True, help="who approved")
    p.add_argument("--quote", required=True, help="the user's own words")
    p.add_argument("--ref", help="what is approved (required for plan: a plan id of this round)")
    p.set_defaults(handler=_cmd_approve)


def _args_answer(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("option", help="one of the options of the question STATE waits on (rsi run prints them)")
    p.add_argument("--by", required=True, help="who answers")
    p.add_argument("--quote", required=True, help="the user's own words")
    p.add_argument("--ref", metavar="ID", help="the id of the question answered (q-NNNN, on the page and in rsi run's "
                                              "line): refused when another question waits now")
    p.set_defaults(handler=_cmd_answer)


def _args_close_round(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("--reason", required=True, help="why the round closes without an evaluation")
    p.set_defaults(handler=_cmd_close_round)


def _args_stop(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("--by", required=True, help="who asked to stop")
    p.add_argument("--quote", required=True, help="the user's own words")
    p.set_defaults(handler=_cmd_stop)


def _args_report(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("--out", metavar="FILE", help="write the report to FILE instead of printing it")
    p.set_defaults(handler=_cmd_report)


def _args_score(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, metavar="TASK_JSON", help="task definition (task.json or its directory)")
    p.add_argument("--labels", required=True, metavar="ROWS",
                   help="label rows: a trainset/valset rows.jsonl or an evalset/testset labels.jsonl")
    p.add_argument("--predictions", required=True, metavar="ROWS",
                   help="prediction rows covering exactly the label ids")
    p.add_argument("--train", metavar="ROWS", help="training rows (trainset rows.jsonl): adds the baselines")
    p.add_argument("--clusters", metavar="ROWS",
                   help="rows with meta.cluster and meta.state_id, e.g. the evalset inputs.jsonl: clusters for the "
                        "interval, groups for group_prior (default: the label rows)")
    p.add_argument("--compare", metavar="ROWS",
                   help="other predictions on the same rows: adds the primary metric's delta and interval")
    p.add_argument("--baselines-on", choices=BASELINES_FIELDS, metavar="SET",
                   help="evalset or testset: adds ③ Compare's baselines (each --baseline) as an evaluation reports "
                        "them, baselines_eval or baselines_test (needs --clusters, the set's inputs.jsonl)")
    p.add_argument("--baseline", action="append", default=[], metavar="DIR",
                   help="a baseline artifact directory (repeat for each): its predictions_eval (evalset) or "
                        "predictions_test (testset) are scored with --baselines-on")
    p.set_defaults(handler=_cmd_score)


def _args_brief(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("module", help="the module to brief: data, train, deliver or judge")
    p.add_argument("--purpose", help="what the dispatch is for (default: the module's usual purpose)")
    p.set_defaults(handler=_cmd_brief)


def _args_run(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("--runner", choices=("claude", "mock", "manual"), default="claude",
                   help="how module agents run: claude (headless Claude Code sessions, the default), mock (the "
                        "scripts in --mock-dir) or manual (write the brief and wait for handback.json)")
    p.add_argument("--mock-dir", metavar="DIR", help="directory of the mock module scripts <module>.py (--runner mock)")
    p.add_argument("--max-steps", type=int, default=200, metavar="N", help="stop after N actions (default 200)")
    p.add_argument("--json", action="store_true",
                   help='print {"status": ..., "reason": ...} instead of the reason line')
    p.set_defaults(handler=_cmd_run)


def _args_dashboard(p: argparse.ArgumentParser) -> None:
    p.add_argument("ws", help="workspace directory")
    p.add_argument("--no-open", action="store_true", help="only rewrite the page, do not open the browser")
    p.set_defaults(handler=_cmd_dashboard)


def _args_ws_only(handler):
    def setup(p: argparse.ArgumentParser) -> None:
        p.add_argument("ws", help="workspace directory")
        p.set_defaults(handler=handler)
    return setup


ARGUMENTS = {
    "init": _args_init,
    "validate": _args_validate,
    "register": _args_register,
    "status": _args_status,
    "promote": _args_promote,
    "approve": _args_approve,
    "answer": _args_answer,
    "next": _args_ws_only(_cmd_next),
    "check-stop": _args_ws_only(_cmd_check_stop),
    "close-round": _args_close_round,
    "stop": _args_stop,
    "report": _args_report,
    "score": _args_score,
    "brief": _args_brief,
    "run": _args_run,
    "dashboard": _args_dashboard,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsi",
        description="Decision Model RSI harness command line.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True
    for name, help_text in COMMANDS:
        ARGUMENTS[name](sub.add_parser(name, help=help_text, description=help_text))
    return parser


def _refresh_dashboard(args: argparse.Namespace) -> None:
    """Rewrite the progress pages after a successful state change; never fails the command (an
    error importing the page code included). `rsi init` opens the overview afterwards, unless
    --no-open or RSI_NO_BROWSER=1."""
    try:
        from rsi.dashboard import write_dashboard
        path = write_dashboard(Workspace(args.ws))
    except Exception as e:
        print("warning: dashboard not updated: %s" % e, file=sys.stderr)
        return
    if args.command == "init" and not args.no_open:
        open_in_browser(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.handler(args)
    except RsiError as e:
        for line in e.errors:
            print(line, file=sys.stderr)
        return 1
    if code == 0 and args.command in DASHBOARD_COMMANDS:
        _refresh_dashboard(args)
    return code
