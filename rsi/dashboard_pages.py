"""The four progress pages: the overview <ws>/dashboard.html and, under <ws>/dashboard/,
rounds.html (Rounds & artifacts), events.html (Events) and report.html (Run report).

The overview holds the pauses waiting for the user, the workflow diagram (rsi.dashboard.
flow_states; the current stage's node adds the first fact of its module's progress.json, and ③'s node, once done, the
round's strongest baseline and the candidate's delta over it on the valset), three cards: the best version
(and the round's best candidate on the validation set), how far the stop conditions are, and what the
run waits for (with the agents running and, when the run asks the user, the options and the exact
`rsi answer` commands), then the live panels (spec §4, §5): What is running now (each open dispatch,
rsi.live.open_dispatches, and its progress.json), Agent judgement (the latest hand-backs' notes and questions,
rsi.live.handback_notes), GPU (the latest snapshot <ws>/work/gpus.json; rendering never runs nvidia-smi) and Recent
events (the events page's latest lines).
The rounds page lists each round's dispatches and the events page merges work/run.jsonl (dispatch, return, decision,
answer, ask) with the registrations. render_pages builds every page from rsi.dashboard's derivations, reads only what
that module allows and writes nothing (rsi.dashboard.write_dashboard writes the pages). Each page is one self-contained
HTML file with the same top bar and tabs; it reloads itself and keeps its scroll position and opened sections across
reloads. Workspace text is escaped; the run report goes through rsi.markdown.to_html, which escapes it itself.
"""
import datetime
import html
import os
import shlex

from rsi import dashboard as D
from rsi import flow as F
from rsi import gpus as G
from rsi import live as L
from rsi import timing
from rsi.markdown import to_html
from rsi.report import _num, _rounds, best_scores, round_comparison
from rsi.rules import amount_text, completed_rounds, flow_settings, is_number, ledger, stop_settings
from rsi.timing import plural
from rsi.workspace import parse_utc

OVERVIEW, ROUNDS, EVENTS, REPORT = D.PAGES
TABS = ((OVERVIEW, "Overview"), (ROUNDS, "Rounds & artifacts"), (EVENTS, "Events"), (REPORT, "Run report"))
EVENTS_SHOWN = 200  # most recent events listed on the events page
RECENT_SHOWN = 10  # most recent events listed on the overview (Recent events)
NOTES_SHOWN = 300  # characters of a hand-back's notes shown before its <details> (Agent judgement)
NOTES_MAX = 20000  # characters of a hand-back's notes shown at most when expanded (after the first NOTES_SHOWN)
QUESTIONS_SHOWN, QUESTION_CHARS = 5, 500  # a hand-back's questions listed, and the characters of each
HANDBACK_STATUS = {"ok": "hand-back", "long_job": "long job", "failed": "failed"}
GPU_STALE_REFRESHES = 3  # a GPU snapshot older than this many flow.page_refresh_seconds is marked stale
NODE_LINE_CHARS = 18  # characters of a node's fourth line (the stage's progress fact)
CELL_NOWRAP = 40  # a progress table cell up to this long stays on one line (the table scrolls sideways)
DISPATCH_STATUS = {"pending": "not launched yet", "running": "running", "handed_back": "handed back",
                   "long_job": "long job", "failed": "failed", "done": "done"}
DECIDERS = {"code": "code", "agent": "decision agent"}

KIND_NAMES = {"task": "task definition", "trainset": "trainset", "valset": "valset", "evalset": "evalset",
              "testset": "testset", "candidate": "candidate model", "search": "tuning search",
              "baseline": "baseline model", "comparison": "baseline comparison", "delivery": "delivery package",
              "feedback": "execution feedback", "evaluation": "evaluation report", "plan": "next-round plan",
              "record": "experiment record", "corpus": "corpus", "pool": "pool"}
# Who registered an artifact, by its producer's stage (rsi.dashboard.stage_of; the orchestrator has none).
PRODUCER_NAMES = {"orchestrator": "orchestrator", "data": "data module", "train": "train module",
                  "compare": "compare module", "deliver": "deliver module", "judge": "judge module"}

# Workflow diagram: node key, left x, width; one row of nodes centred at FLOW_Y; title and second line per key.
NODES = (("task", 10, 100), ("data", 126, 152), ("train", 294, 152), ("compare", 462, 152), ("deliver", 630, 152),
         ("judge", 798, 164), ("check", 978, 150), ("stop", 1156, 120))
NODE_H, FLOW_W, FLOW_H, FLOW_Y, LOOP_Y = 64, 1288, 140, 46, 104
NODE_H4 = 80  # a node with a fourth line (the current stage's progress: _node_lines; ③'s _compare_node_line)
TITLES = dict(D.STEP_NAMES, task="Task intake", check="Top-level check", stop="Stop & deliver")
SUBTITLES = dict(((m, D._skill(m)) for m in D.MODULES), task="orchestrator", check="promote · stop · approve",
                 stop="run report")
FILL = {"done": "#CEE5CC", "current": "#FBE5D6", "wait": "#FFF3C0", "fail": "#FFE3F1", "skip": "#FFFFFF",
        "todo": "#FFFFFF"}
LEGEND = (("done", "done this round"), ("current", "current step"), ("wait", "waiting for the user"),
          ("fail", "gate failed"), ("todo", "not started / skipped this round"))


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
                                       % _e("(since %s)" % D._since(q["since"])) if q["since"] else "")
                    for q in questions)
    return ('<section id="ask" class="banner"><h2>%s</h2><ul>%s</ul><p class="since">%s</p></section>'
            % (_e("⏸ Waiting for the user to answer in Claude"), items,
               _e("This page is read-only; answer in the Claude session.")))


def _short(session):
    """The first 8 characters of a session id, "—" without one."""
    return str(session)[:8] if session else "—"


def _agents_html(ws):
    """The agents the engine has running, one line each: module, session, start, runner; a long job whose
    marker is not there yet is shown as waiting for it, not as a running agent."""
    def line(a, what):
        return '<li><b>%s</b> <span class="sub">%s</span></li>' % (
            _e("rsi-%s" % a["module"]),
            _e("%s%s · session %s · since %s · runner %s" % (
                a["id"], what, _short(a["session"]), D._since(a["started"]), a["runner"] or "—")))
    items = "".join(line(a, "") for a in D.running_agents(ws))
    items += "".join(line(a, " long job waiting for its marker") for a in D.waiting_long_jobs(ws))
    for a in D.pending_agents(ws):  # after a stop rsi run closes them instead of launching them
        what = ("%s not launched at the stop (rsi run closes it and does not launch it)" % a["id"]
                if ws.state["stopped"] else "%s not launched yet · runner %s" % (a["id"], a["runner"] or "—"))
        items += '<li><b>%s</b> <span class="sub">%s</span></li>' % (_e("rsi-%s" % a["module"]), _e(what))
    return '<ul class="agents">%s</ul>' % items if items else ""


def _answer_html(ws, questions):
    """The options of the question the run asks and the exact command that answers it, the same form for every
    question: `rsi answer ... --ref <id>` (the id of STATE waiting once the engine has recorded the question);
    for a plan `rsi approve` is named as an alternative."""
    qs = [q for q in questions if q["source"] == "state" and q.get("options")]
    if not qs:
        return ""
    q, w = qs[0], ws.waiting()
    opts = list(q["options"])
    recorded = bool(w) and (w.get("what"), w.get("ref")) == (q.get("what"), q.get("ref")) and w.get("id")
    from rsi.briefs import rsi_command
    ref = " --ref %s" % shlex.quote(w["id"]) if recorded else ""
    cmds = "\n".join("%s answer %s %s%s --by <your name> --quote \"<user's words>\"" % (
        rsi_command(), shlex.quote(ws.root), shlex.quote(o), ref) for o in opts)
    notes = [] if recorded else ["The engine has not recorded this question yet: run rsi run first; when it stops at "
                                 "this question it gives the question's id (--ref)."]
    if q.get("what") == "plan" and q.get("ref"):
        notes.append("The plan can also be approved with %s approve %s plan --ref %s --by <your name> --quote "
                     "\"<user's words>\" (same effect as approve)." % (rsi_command(), shlex.quote(ws.root), q["ref"]))
    notes.append("After answering, run rsi run to go on.")
    return ('<div class="answer"><div class="sub">%s</div><ul class="opts">%s</ul><pre class="cmd">%s</pre>%s</div>'
            % (_e("Options:"), "".join("<li><code>%s</code></li>" % _e(o) for o in opts), _e(cmds),
               "".join('<div class="sub">%s</div>' % _e(n) for n in notes)))


def _node_lines(states, ds):
    """The fourth line of the current (or waiting) stage nodes: the first fact of the progress.json of an open
    dispatch of the stage (rsi.live.open_dispatches `ds`), else for an open long job "long job · c check-ins"."""
    out = {}
    for want_fact in (True, False):
        for x in ds:
            k = x["stage"]
            if k is None or k in out or states.get(k, {}).get("state") not in ("current", "wait"):
                continue
            facts = (x["progress"] or {}).get("facts") or []
            if want_fact and facts:
                text = "%s %s" % facts[0]
                out[k] = text if len(text) <= NODE_LINE_CHARS else text[:NODE_LINE_CHARS] + "…"
            elif not want_fact and x["status"] == "long_job":
                out[k] = "long job · " + plural(x["checkins"], "check-in")
    return out


def _compare_node_line(ws, states):
    """The fourth line of ③'s node once it is done this round: the strongest baseline of the round's comparison
    (rsi.report.round_comparison) and the candidate's delta over it on the valset ("vs gbdt Δ+0.031"), at most
    NODE_LINE_CHARS long (a long name is cut, never the delta); None without one (③ off, skipped, or no baseline
    compared)."""
    comp = round_comparison(ws, ws.state["round"]) if states.get("compare", {}).get("state") == "done" else None
    m = ws.manifest(comp) if comp is not None else {}
    d = (m.get("deltas_val") or {}).get(m.get("strongest"))
    if not d:
        return None
    delta, name = "Δ%+.3f" % d["delta"], m["strongest"]
    room = NODE_LINE_CHARS - len(delta) - len("vs  ")
    if len(name) > room:
        name = name[:max(room - 1, 1)] + "…"
    return "vs %s %s" % (name, delta)


def _flow(states, lines=None, titles=None):
    """The workflow diagram; `lines`: stage -> the fourth line of its node (_node_lines); `titles`: node key -> its
    title where it differs from TITLES (③ Compare (off): rsi.dashboard.stage_title)."""
    lines = lines or {}
    titles = dict(TITLES, **(titles or {}))
    box = dict((k, (x, w)) for k, x, w in NODES)

    def height(k):
        return NODE_H4 if k in lines else NODE_H

    def left(k):
        return box[k][0]

    def right(k):
        return box[k][0] + box[k][1]

    def centre(k):
        return box[k][0] + box[k][1] / 2.0

    def line(points):
        return '<polyline class="ln" points="%s" marker-end="url(#ah)"/>' % " ".join("%g,%g" % p for p in points)

    mid, keys = FLOW_Y, [k for k, _x, _w in NODES]
    out = ['<svg class="flow" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (FLOW_W, FLOW_H, _e("Workflow")),
           '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker></defs>']
    out += [line([(right(a), mid), (left(b), mid)]) for a, b in zip(keys, keys[1:])]
    out += ['<text class="note" x="%g" y="%g">%s</text>' % ((right("check") + left("stop")) / 2.0, mid - 8, _e("stop")),
            line([(centre("check"), mid + height("check") / 2.0), (centre("check"), LOOP_Y), (centre("data"), LOOP_Y),
                  (centre("data"), mid + height("data") / 2.0)]),
            '<text class="note" x="%g" y="%d">%s</text>' % ((centre("check") + centre("data")) / 2.0, LOOP_Y + 18,
                                                            _e("Continue: the next round starts at the entry module "
                                                               "the plan names"))]
    for k, x, w in NODES:
        state, text = states[k]["state"], states[k]["text"]
        cx, h = x + w / 2.0, height(k)
        up = (h - NODE_H) / 2.0  # the lines move up by half the extra height
        out.append('<g id="n-%s" data-state="%s"><rect x="%g" y="%g" width="%g" height="%d" rx="10" fill="%s" '
                   'stroke="#222" stroke-width="%s"/>' % (k, state, x, mid - h / 2.0, w, h, FILL[state],
                                                          "2.2" if state in ("current", "wait") else "1"))
        out.append('<text class="t1" x="%g" y="%g">%s</text>' % (cx, mid - 9 - up, _e(titles[k])))
        out.append('<text class="t2" x="%g" y="%g">%s</text>' % (cx, mid + 7 - up, _e(SUBTITLES[k])))
        out.append('<text class="t3" x="%g" y="%g">%s</text>' % (cx, mid + 23 - up, _e(text)))
        if k in lines:
            out.append('<text class="t4" x="%g" y="%g">%s</text>' % (cx, mid + 39 - up, _e(lines[k])))
        out.append("</g>")
    out.append("</svg>")
    legend = "".join('<span><i class="sw %s"></i>%s</span>' % (k, _e(label)) for k, label in LEGEND)
    return ('<section id="flow" class="card"><h2>Workflow</h2><div class="flowwrap">%s</div>'
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

    out = ['<svg class="spark" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (w, h, _e("Primary metric by version"))]
    if is_number(target):
        out.append('<line class="target" x1="%g" y1="%.1f" x2="%g" y2="%.1f"/>' % (side, y(target), w - side, y(target)))
    coords = [(x(i), y(v), name, v) for i, (name, v) in enumerate(points) if is_number(v)]
    out.append('<polyline class="ln" points="%s"/>' % " ".join("%.1f,%.1f" % (cx, cy) for cx, cy, _n, _v in coords))
    for cx, cy, name, v in coords:
        out.append('<circle cx="%.1f" cy="%.1f" r="3.2"><title>%s</title></circle>' % (cx, cy, _e("%s: %s" % (name, _num(v)))))
    for i in sorted(set([0, n - 1])):
        out.append('<text x="%.1f" y="%d">%s</text>' % (x(i), h - 4, _e(points[i][0])))
    out.append("</svg>")
    return "".join(out)


def _best_card(ws, task):
    s = ws.state
    m = task["metrics"]
    primary, higher = m["primary"], m["higher_is_better"]
    target = stop_settings(task)["target"]
    direction = "higher is better" if higher else "lower is better"
    goal = "target %s" % _num(target) if is_number(target) else "no target set"
    rb = L.round_best_val(ws)  # the round in progress, on the validation set
    round_best = "" if rb is None else '<div class="sub">%s%s%s</div>' % (
        _e("Best on the valset this round: %s %s (" % (primary, _num(rb[1]))), _art(ws, OVERVIEW, rb[0]), _e(")"))
    if s["best"] is None:
        body = '<div class="big none">No version yet</div><div class="sub">%s</div>' % _e(
            "The first promotion makes V1 · %s (evalset, %s) · %s" % (primary, direction, goal))
        body += round_best
    else:
        value = ws.best_primary()
        best = ws.version(s["best"])
        sc = best_scores(ws)
        measure = "%s (evalset %s, %s)" % (primary, sc["evalset_id"], direction)
        body = '<div class="big">%s</div>' % _e("%s · %s" % (s["best"], _num(value)))
        body += '<div class="sub">%s %s</div>' % (_e("%s · %s ·" % (measure, goal)), _art(ws, OVERVIEW, best["evaluation_id"]))
        if sc["refreshed"]:  # its score is on an earlier evalset: say what it scores on the current one
            cur = sc["current"]
            if cur is None or cur["value"] is None:
                body += '<div class="sub">%s</div>' % _e("No score on the current evalset %s yet (waiting for the "
                                                         "evaluation after its re-delivery)" % sc["current_evalset"])
            else:
                body += '<div class="sub">%s%s%s%s%s</div>' % (
                    _e("On the current evalset %s: %s (" % (sc["current_evalset"], _num(cur["value"]))),
                    _art(ws, OVERVIEW, cur["evaluation_id"]), _e(", computed with the re-delivered "),
                    _art(ws, OVERVIEW, cur["delivery_id"]) if cur["delivery_id"] else _e("delivery"), _e(")"))
            if is_number(target):
                body += '<div class="sub">%s</div>' % _e("The stop target is still judged by the score on %s"
                                                         % sc["evalset_id"])
        body += round_best
        if is_number(value) and is_number(target):
            body += _meter("Toward the target", _toward(value, target, higher), 1,
                           "%s / %s" % (_num(value), _num(target)), alarm=False)
        points = []
        for v in s["versions"]:
            x = ws.manifest(v["evaluation_id"])["metrics"]["candidate"].get(primary)
            points.append((v["version"], x if is_number(x) else None))
        body += _spark(points, target)
    return '<section id="best" class="card"><h2>Current best</h2>%s</section>' % body


def _stop_card(ws, task, ds):
    """`ds`: the open dispatches (rsi.live.open_dispatches); the round's of ② are attempts in progress."""
    s = ws.state
    st = stop_settings(task)
    attempts, limit = len(ws.attempts()), ws.attempt_limit()
    going = sum(1 for x in ds if x["stage"] == "train" and x["round"] == s["round"])
    done = completed_rounds(s)
    body = _meter("Rounds completed", done, st["max_rounds"], "%d / %d" % (done, st["max_rounds"]))
    body += _meter("Rounds without promotion", s["no_promotion_streak"], st["patience"],
                   "%d / %d" % (s["no_promotion_streak"], st["patience"]))
    body += _meter("Attempts this round", attempts, limit,
                   "%d / %d" % (attempts, limit) + (" (%d in progress)" % going if going else ""),
                   alarm=not ws.passed_candidates())  # used up with a candidate through the gate is no alarm
    body += _meter("Evalset uses", s["eval_use_count"], task["eval_use_limit"],
                   "%d / %d" % (s["eval_use_count"], task["eval_use_limit"]))
    for key, label, used, budget in ledger(task, s):  # the spend ledger: GPU hours, and what has a budget or a use
        if key == "compute_hours" or budget is not None or used:
            body += _meter(label[:1].upper() + label[1:], used, budget,
                           "%s / %s" % (amount_text(key, used), _num(budget)) if budget is not None
                           else "%s · no limit" % amount_text(key, used))
    if ws.spend_problems():
        body += '<div class="sub">%s</div>' % _e("Spend-file problems: %s (see the events page)"
                                                 % plural(ws.spend_problems(), "line"))
    if not s["stopped"] and ws.stop_check()[0]:
        body += '<div class="sub">%s</div>' % _e("A stop condition is met")
    return '<section id="stop" class="card"><h2>How far from the stop</h2>%s</section>' % body


def _now_card(ws, progress, questions):
    s = ws.state
    key, where, what, action = D._view(ws, progress)
    asked = [q["since"] for q in questions if q["since"]]
    if s["stopped"]:
        since = D._local(s["stopped"].get("at"))
        text = D.stopped_texts(ws, action)[1]
        if os.path.isfile(os.path.join(ws.root, D.REPORT_FILE)):
            detail = '<a href="%s">View the run report →</a>%s' % (_href(OVERVIEW, REPORT), " " + _e(text) if text else "")
        else:
            detail = _e(text)
    else:
        prefix = "Next %s: " % where  # a next step's sentence repeats the lead first: keep the action
        detail = _e(what[len(prefix):] if key in D.NEXT_STEPS and not questions and what.startswith(prefix)
                    else what)
        if key == "plan" or questions:
            since = D._since(min(asked, key=datetime.datetime.fromisoformat)) if asked else _last_change(ws)
        elif key in ("running", "handed"):
            since = None  # a running module's sentence names its start; a hand-back time is unknown
        else:
            since = _last_change(ws)
    return ('<section id="now" class="card"><h2>What the run waits for</h2><div class="wait-now">%s</div>'
            '<div class="detail">%s</div>%s%s<div class="sub">%s</div></section>'
            % (_e(D.now_lead(ws, progress, questions, (key, where, what, action))), detail,
               _agents_html(ws), _answer_html(ws, questions),
               _e("since %s" % since) if since else ""))


# ---- the live panels (spec §4, §5): read while `rsi run` waits and rewrites the pages ----------------

def _clock(dt, now):
    """Aware time `dt` as local HH:MM:SS on the day of `now`, else MM-DD HH:MM."""
    t = dt.astimezone()
    return t.strftime("%H:%M:%S") if t.date() == now.astimezone().date() else t.strftime("%m-%d %H:%M")


def _ago(seconds):
    return timing.format_duration(seconds) + " ago"


def _progress_html(p, now):
    """A module's progress.json as shown (rsi.progress.dispatch_progress, None when there is none): summary, facts,
    table, notes, the stale warning, when it was updated and the file's problems."""
    if p is None:
        return '<div class="sub">%s</div>' % _e("The module has not written its progress yet (no progress.json in the "
                                                "output directory).")
    out = []
    if p.get("summary"):
        out.append('<div class="psum">%s</div>' % _e(p["summary"]))
    if p.get("facts"):
        out.append('<div class="facts">%s</div>' % "".join(
            '<span class="fact"><span class="k">%s</span>%s</span>' % (_e(label), _e(value))
            for label, value in p["facts"]))
    t = p.get("table")
    if t:
        head = "".join("<th>%s</th>" % _e(c) for c in t["columns"])
        rows = "".join("<tr>%s</tr>" % "".join("<td%s>%s</td>" % (' class="long"' if c and len(c) > CELL_NOWRAP else "",
                                                                  _e("—" if c is None else c)) for c in r)
                       for r in t["rows"])
        out.append('<div class="tablewrap"><table class="ptable">%s<thead><tr>%s</tr></thead><tbody>%s</tbody>'
                   '</table></div>' % ("<caption>%s</caption>" % _e(t["title"]) if t.get("title") else "", head, rows))
    if p.get("notes"):
        out.append('<ul class="pnotes">%s</ul>' % "".join("<li>%s</li>" % _e(n) for n in p["notes"]))
    if p.get("stale"):
        out.append('<div class="warn">%s</div>' % _e("No progress update for %s: the module may be stuck, or no longer "
                                                     "writes progress.json." % plural(int(p["age_s"] // 60), "minute")))
    if p.get("updated_at") is not None:
        out.append('<div class="sub">%s</div>' % _e("Progress updated %s (%s)" % (
            _clock(p["updated_at"], now), _ago(p["age_s"]))))
    if p.get("problems"):
        out.append('<div class="note">%s</div>' % _e("Problems in progress.json: " + "; ".join(p["problems"])))
    return "".join(out)


def _doing_block(x, now):
    """One open dispatch (rsi.live.open_dispatches) and its progress."""
    status = x["status"]
    head = '<div class="dhead"><b>%s</b> %s <span class="chip%s">%s</span></div>' % (
        _e(x["stage_name"]), _e("%s · %s" % (x["id"], x["skill"])),
        " run" if status in ("running", "long_job") else "", _e(DISPATCH_STATUS.get(status, status)))
    bits = ["launch %d (%s)" % (x["launches"], L.MODE_WORDS.get(x["mode"], x["mode"]))]
    if x["since"] is not None:
        bits.append("this launch at %s" % _clock(x["since"], now))
    bits.append(plural(x["returns"], "return"))
    if x["checkin_limit"] is not None:
        bits.append("check-ins %d / %d" % (x["checkins"], x["checkin_limit"]))
    elif x["checkins"]:
        bits.append("%s (no more)" % plural(x["checkins"], "check-in"))
    if x["checkin_due"] is not None and not x["question_id"]:
        bits.append("next check-in %s" % _clock(x["checkin_due"], now))
    if status == "long_job" and x["marker"]:
        bits.append("completion marker %s (%s)" % (os.path.basename(x["marker"]),  # the now card names the whole path
                                                   "present" if x["marker_done"] else "not yet written"))
    out = [head, '<div class="sub">%s</div>' % _e(" · ".join(bits))]
    if x["question_id"]:
        out.append('<div class="warn">%s</div>' % _e("⏸ Waiting for the user (%s): see the question above and \"What "
                                                     "the run waits for\"." % x["question_id"]))
    out.append(_progress_html(x["progress"], now))
    return '<div class="dblock">%s</div>' % "".join(out)


def _doing_card(ds, now):
    body = "".join(_doing_block(x, now) for x in ds) or '<p class="empty">%s</p>' % _e("No module is running now.")
    return ('<section id="doing" class="card"><h2>What is running now<span class="hint">%s</span></h2>%s</section>'
            % (_e("open dispatches and the progress their modules write (progress.json)"), body))


def _judgement_card(ws, now):
    items = []
    for h in L.handback_notes(ws):
        notes = h["notes"]
        head = '<div class="hbh"><span class="t">%s</span> <b>%s</b> %s <span class="chip">%s</span></div>' % (
            _e(_clock(h["at"], now)), _e("rsi-%s" % h["module"]), _e("%s · hand-back %d" % (h["dispatch_id"], h["launch"])),
            _e(HANDBACK_STATUS.get(h["status"], h["status"] or "status unknown")))
        if not notes.strip():
            body = '<div class="sub">%s</div>' % _e("(no notes written)")
        elif len(notes) <= NOTES_SHOWN:
            body = '<div class="hbn">%s</div>' % _e(notes)
        else:  # the first NOTES_SHOWN characters, then the rest inside <details>
            rest = notes[NOTES_SHOWN:]
            if len(rest) > NOTES_MAX:
                rest = rest[:NOTES_MAX] + "… (too long; only the first %d characters are shown)" % (NOTES_SHOWN + NOTES_MAX)
            body = ('<div class="hbn">%s</div><details id="%s"><summary>%s</summary><div class="hbn">%s</div></details>'
                    % (_e(notes[:NOTES_SHOWN] + "…"),
                       _e("hb-%s-%d-%d" % (h["dispatch_id"], h["launch"], int(h["at"].timestamp()))),
                       _e("Show the other %s (%d in all)" % (plural(len(notes) - NOTES_SHOWN, "character"),
                                                              len(notes))), _e(rest)))
        qs = h["questions"]
        if qs:
            more = len(qs) - QUESTIONS_SHOWN
            body += '<div class="qhead">%s</div><ul class="qs">%s</ul>%s' % (
                _e("Questions:"), "".join("<li>%s</li>" % _e(q if len(q) <= QUESTION_CHARS else q[:QUESTION_CHARS] + "…")
                                          for q in qs[:QUESTIONS_SHOWN]),
                '<div class="sub">%s</div>' % _e("%d more" % more) if more > 0 else "")
        items.append('<li class="hb">%s%s</li>' % (head, body))
    body = '<ul class="hbs">%s</ul>' % "".join(items) if items else '<p class="empty">%s</p>' % _e("No hand-backs yet.")
    return ('<section id="judgement" class="card"><h2>Agent judgement<span class="hint">%s</span></h2>%s</section>'
            % (_e("the latest %s, newest first" % plural(len(items), "hand-back") if items
                  else "the notes and questions modules write in their hand-backs"), body))


def _gb(mib):
    return "%.1f" % (mib / 1024.0) if is_number(mib) else "?"


def _gpu_tile(g):
    """One card of a snapshot (rsi.gpus.snapshot): memory, utilisation, a memory bar of this user's and the others'
    processes, and the processes by owner."""
    used, total, util = g.get("memory_used_mib"), g.get("memory_total_mib"), g.get("utilization_pct")
    procs = [p for p in g.get("processes") or [] if isinstance(p, dict)]

    def mib(ps):
        return sum(p["memory_mib"] for p in ps if is_number(p.get("memory_mib")))

    def held(ps):  # the memory processes `ps` hold, in words ([N/A] for every one of them: unknown)
        return "%s GB" % _gb(mib(ps)) if any(is_number(p.get("memory_mib")) for p in ps) else "memory unknown"

    ours = [p for p in procs if p.get("owner") == "ours"]
    others = [p for p in procs if p.get("owner") == "other"]
    unknown = [p for p in procs if p.get("owner") not in ("ours", "other")]
    bar = ""
    if is_number(used) and is_number(total) and total > 0:
        mine = max(0.0, min(100.0, 100.0 * mib(ours) / total))
        rest = max(0.0, min(100.0 - mine, 100.0 * (used - mib(ours)) / total))
        bar = '<span class="ours" style="width:%.1f%%"></span><span class="other" style="width:%.1f%%"></span>' % (
            mine, rest)
    lines = ["GPU memory %s / %s GB · utilization %s" % (_gb(used), _gb(total),
                                                          "%d%%" % util if is_number(util) else "unknown")]
    if not procs:
        lines.append("no compute processes")
    if ours:
        lines.append("this user: %s · %s" % (plural(len(ours), "process", "processes"), held(ours)))
    if others:
        users = {}
        for p in others:
            users.setdefault(str(p.get("user") or "?"), []).append(p)
        lines.append("other users: " + ", ".join("%s %s" % (u, held(ps)) for u, ps in sorted(users.items())))
    if unknown:
        lines.append("unknown owner: %s · %s" % (plural(len(unknown), "process", "processes"), held(unknown)))
    return '<div class="gpu%s"><b>%s</b> <span class="sub">%s</span><div class="track">%s</div>%s</div>' % (
        " busy" if others or unknown else "", _e("GPU %s" % g.get("index")), _e(g.get("name") or ""), bar,
        "".join('<div>%s</div>' % _e(t) for t in lines))


def _gpu_body(snap, problem, task, now):
    """(hint, body) of the GPU panel for snapshot `snap` (rsi.gpus.load_snapshot: None with `problem` None when there
    is none yet, with `problem` the reason the file cannot be read)."""
    refresh = flow_settings(task)["page_refresh_seconds"]
    every = "rsi run takes a snapshot every %s while it waits" % plural(refresh, "second")
    if problem is not None:
        return "", '<p class="empty">%s</p>' % _e("The GPU snapshot cannot be read (%s %s): the next one rewrites it."
                                                  % (G.GPU_FILE, problem))
    if snap is None:
        return "", '<p class="empty">%s</p>' % _e("No GPU information yet: %s." % every)
    at = parse_utc(snap.get("at"))
    age = None if at is None else (now - at).total_seconds()
    when = "time taken unknown" if at is None else "taken %s (%s)" % (_clock(at, now), _ago(age))
    if snap.get("status") == "missing":
        return _e(when), '<p class="empty">%s</p>' % _e("This machine has no nvidia-smi; no GPU information.")
    out = []
    if age is not None and age > GPU_STALE_REFRESHES * refresh:
        out.append('<div class="warn">%s</div>' % _e("Stale: the latest snapshot is from %s (%s)." % (_ago(age), every)))
    if snap.get("status") != "ok":
        out.append('<div class="note">%s</div>' % _e("nvidia-smi failed (%s): %s" % (when, snap.get("error")
                                                                                     or "reason unknown")))
    cards = dict((g["index"], g) for g in snap.get("gpus") or []
                 if isinstance(g, dict) and isinstance(g.get("index"), int) and not isinstance(g.get("index"), bool))
    ids = (task.get("resources") or {}).get("gpu_ids")
    shown = list(ids) if ids else sorted(cards)
    if cards or snap.get("status") == "ok":
        tiles = [_gpu_tile(cards[i]) if i in cards else
                 '<div class="gpu off"><b>%s</b><div class="track"></div>%s</div>' % (_e("GPU %s" % i), _e("no such GPU"))
                 for i in shown]
        out.append('<div class="gpus">%s</div>' % "".join(tiles) if tiles else
                   '<p class="empty">%s</p>' % _e("No GPUs in the snapshot."))
        out.append('<div class="gkey"><span><i class="ours"></i>this user (the one running rsi run)</span>'
                   '<span><i class="other"></i>other users and unknown</span></div>')
    hint = "%s · %s" % ("the task's resources.gpu_ids: %s" % ", ".join(str(i) for i in ids) if ids else "all GPUs",
                        when)
    return _e(hint), "".join(out)


def _gpu_card(ws, task, now):
    try:
        snap, problem = G.load_snapshot(ws.root)
        hint, body = _gpu_body(snap, problem, task, now)
    except Exception as e:  # a snapshot cut off or edited by hand never breaks the page
        hint, body = "", '<p class="empty">%s</p>' % _e("The GPU snapshot cannot be read (%s: %s)." % (type(e).__name__, e))
    return '<section id="gpus" class="card"><h2>GPU<span class="hint">%s</span></h2>%s</section>' % (hint, body)


def _recent_card(ws, task, now):
    items = []
    for at, cls, text, source, _seq in _timeline(ws, task, OVERVIEW)[:RECENT_SHOWN]:
        dt = parse_utc(at)
        items.append('<li><span class="t">%s</span><span class="dot %s"></span><span>%s <span class="who">· %s</span>'
                     '</span></li>' % (_e(_clock(dt, now) if dt else "—"), cls, text, _e(source)))
    body = '<ul class="tl recent">%s</ul>' % "".join(items) if items else '<p class="empty">%s</p>' % _e("No events yet.")
    return ('<section id="recent" class="card"><h2>Recent events</h2>%s<p class="more"><a href="%s">%s</a></p></section>'
            % (body, _e(_href(OVERVIEW, EVENTS)), _e("All events →")))


def _overview(ws, task, progress, questions, now):
    """The overview (spec §4): the questions, the workflow, the three cards, then What is running now, Agent judgement
    and GPU, and Recent events. `now`: the generation time (naive: local)."""
    now = now.astimezone()
    ds = L.open_dispatches(ws, now)
    states = D.flow_states(ws, progress)
    parts = [_ask(questions)] if questions else []
    lines = _node_lines(states, ds)
    compare = _compare_node_line(ws, states)
    if compare and "compare" not in lines:
        lines["compare"] = compare
    parts.append(_flow(states, lines, dict((m, D.stage_title(ws, m)) for m in D.MODULES)))
    parts.append('<div class="grid">%s%s%s</div>' % (_best_card(ws, task), _stop_card(ws, task, ds),
                                                      _now_card(ws, progress, questions)))
    parts.append(_doing_card(ds, now))
    parts.append('<div class="two">%s%s</div>' % (_judgement_card(ws, now), _gpu_card(ws, task, now)))
    parts.append(_recent_card(ws, task, now))
    return "\n".join(parts)


# ---- rounds and artifacts -----------------------------------------------------------------------

def _gate(ws, e, primary):
    """(css class, short result) of an artifact's gate or role, ("", "") when it has none."""
    kind = e["kind"]
    if kind not in ("trainset", "candidate", "search", "evaluation", "plan"):
        return "", ""
    m = ws.manifest(e["id"])
    if kind == "trainset":
        if m["quality"]["passed"] is True:
            return "ok", "quality gate passed"
        failed = [c["name"] for c in m["quality"]["checks"] if not c["passed"]]
        return "bad", "quality gate failed" + (" (%s)" % ", ".join(failed) if failed else "")
    if kind == "candidate":
        val = "val %s %s" % (primary, _num(m["metrics_val"].get(primary)))
        return ("ok", "train gate passed · " + val) if m["train_gate"]["passed"] is True \
            else ("bad", "train gate failed · " + val)
    if kind == "search":
        delta = m["gain_vs_default"]["delta"]
        return "ok", "tune gate passed · gain %s" % ("%+.4g" % delta if is_number(delta) else "n/a")
    if kind == "evaluation":
        if m.get("uses_testset") is True:
            return "", "milestone evaluation (test)"
        return "", "evalset %s %s" % (primary, _num(m["metrics"]["candidate"].get(primary)))
    return "", "entry %s" % m.get("entry_module")


def _chip(ws, e, primary):
    cls, text = _gate(ws, e, primary)
    note = ws.summary(e["id"])
    return '<span class="art"%s>%s<span class="k">%s</span>%s</span>' % (
        ' title="%s"' % _e(note) if note else "", _art(ws, ROUNDS, e["id"]), _e(KIND_NAMES.get(e["kind"], e["kind"])),
        '<span class="%s">%s</span>' % (cls, _e(text)) if text else "")


def _row(name, cell, cls=""):
    return '<div class="step"><div class="name">%s</div><div class="body%s">%s</div></div>' % (
        _e(name), " " + cls if cls else "", cell)


def _busy_skills(ws, module, rows):
    """The skills of stage `module`'s dispatches among `rows` (rsi.dashboard's agent rows), as _skills_of names them
    (an old workspace's rsi-tune as it is); the stage's own skill when there is none (the manual flow)."""
    return D._skills_of([a for a in rows if D.stage_of(a["module"]) == module]) or D._skill(module)


def _module_note(ws, module, progress):
    """In the current round: the module's output directory that runs or waits to be registered."""
    p = progress[module]
    stopped = ws.state["stopped"]
    records = bool(ws.dispatches())  # the manual flow (no records) keeps its wording
    if p["state"] == "running":
        who = _busy_skills(ws, module, D.running_agents(ws) + D.waiting_long_jobs(ws))
        if stopped:
            text = "%s still running after the stop" % who if records else "not handed back at the stop"
        else:
            text = "%s in progress: no artifact in the output directory since %s" % (who, D._since(p["since"]))
    elif p["state"] == "pending":
        text = "not launched at the stop" if stopped else "%s not launched yet" % _busy_skills(ws, module,
                                                                                               D.pending_agents(ws))
    elif p["state"] == "unregistered":
        held = D.held_handbacks(ws, module)
        # answered: the flow's own test (the overview's answer going back comes from it)
        answered = bool(held) and all(F.question_answered(ws, d) for d in held)
        if stopped and not records:
            head = "not handed back at the stop"
        elif stopped:
            head = "handed back, not registered (%s)" % (
                "the user's answer was not sent back at the stop" if answered
                else "still waiting for the user at the stop" if held else "the run has stopped")
        else:
            head = "handed back, not registered" + (" (answered; waiting for rsi run to send the answer back)"
                                                    if answered else " (waiting for the user to answer its questions)"
                                                    if held else "")
        text = head + (": " + ", ".join(p["pending"]) if p["pending"] else "")
    else:
        text = ""
    old = "; ".join("not registered (superseded by %s): %s" % (aid, name) for name, aid in sorted(p["superseded"].items()))
    text = "; ".join(t for t in (text, old) if t)
    return '<div class="note">%s</div>' % _e(text) if text else ""


def _verdict(d):
    if d.get("decision") == "promote":
        return "promote %s" % d.get("version")
    return "closed (no evaluation)" if d.get("evaluation_id") is None else "keep"


def _decision(ws, n, r, need_approval):
    d = r["decision"]
    if d is None:
        pending = ws.evaluations(n)
        text = "<b>%s</b>" % _e("undecided" + (" (%s to decide)" % ", ".join(pending) if pending else ""))
    else:
        bits = [_e(d.get("reason") or "—"), _e(D._local(d.get("at")))]
        ev = d.get("evaluation_id")
        if ev is not None:
            bits.append("evaluation " + _art(ws, ROUNDS, ev))
        text = '<b>%s</b> <span class="sub">%s</span>' % (_e(_verdict(d)), " · ".join(bits))
    plan = r.get("plan_id")
    if plan is None:
        plan_html = "none"
    else:
        approval = ("approved" if F.plan_approved(ws, plan) else "not approved") if need_approval else "no approval needed"
        plan_html = "%s%s" % (_art(ws, ROUNDS, plan), _e(" (%s)" % approval))
    return '%s<br><span class="sub">Next-round plan: %s</span>' % (text, plan_html)


def _timed(prefix, dur, suffix, start, cls, approx=False):
    """A duration label. While it is still running it carries data-start (UTC ISO), and data-approx when its
    start is approximated (`approx`); the page script recomputes it from the browser clock on load, because the
    file is only rewritten by rsi commands."""
    if start is None:
        return '<span class="%s">%s</span>' % (cls, _e(prefix + dur + suffix))
    return '<span class="%s" data-start="%s" data-prefix="%s" data-suffix="%s"%s>%s</span>' % (
        cls, _e(start), _e(prefix), _e(suffix), ' data-approx="1"' if approx else "",
        _e(prefix + dur + suffix))


def _elapsed_label(ws, now):
    p = timing.elapsed_parts(ws, now)
    return "" if p is None else _timed(p[0], p[1], p[2], p[3], "chip time", p[4])


def _round_time_label(win, now, running_chip=False):
    """`running_chip`: the round's status chip already says "in progress" or the round is decided (it stays the
    current round until `rsi next`), so only "N minutes so far" is shown."""
    p = timing.round_parts(win, now)
    if p is None:
        return ""
    if running_chip and p[0] == timing.OPEN_ROUND:
        return _timed("", p[1], timing.SO_FAR, p[2], "chip time", p[3])
    return _timed(p[0], p[1], "", p[2], "chip time", p[3])


# The launches in the same session that are no return (a dispatch record's counters), as the rounds page names them.
SESSION_COUNTERS = (("followups", "follow-up", "follow-ups"), ("continued", "long-job wrap-up", "long-job wrap-ups"),
                    ("answered", "answer sent back", "answers sent back"))


def _dispatch_line(d):
    """One dispatch of the round: id, module, purpose, session, returns (and the other launches in its session),
    status."""
    status = d.get("status")
    more = "".join(" · " + plural(int(d.get(k) or 0), one, many) for k, one, many in SESSION_COUNTERS if d.get(k))
    return ('<div class="disp"><b>%s</b> %s <span class="sub">%s</span> <span class="chip%s">%s</span></div>'
            % (_e(d.get("id")), _e("rsi-%s" % d.get("module")),
               _e("purpose %s · session %s · %s%s" % (d.get("purpose") or "—", _short(d.get("session")),
                                                     plural(int(d.get("returns") or 0), "return"), more)),
               " run" if status in ("running", "long_job") else " ok" if status == "done" else "",
               _e(DISPATCH_STATUS.get(status, status))))


def _round_block(ws, n, r, progress, primary, need_approval, latest, win, now):
    s = ws.state
    current = n == s["round"]
    arts = [ws.entry(a) for a in r["artifacts"]]
    d = r["decision"]
    if d is not None:
        verdict = ("ok" if d.get("decision") == "promote" else "", _verdict(d).split(" (")[0])
    elif current:
        verdict = ("stop", "stopped") if s["stopped"] else ("run", "in progress")
    else:
        verdict = ("", "undecided")
    summary = ('<summary><span class="rn">Round %d</span><span class="chip">%s</span><span class="chip %s">%s</span>'
               '<span class="chip">%s</span>%s</summary>'
               % (n, _e("entry %s" % r.get("entry")), verdict[0], _e(verdict[1]),
                  _e("attempts %d / %d" % (len(ws.attempts(n)), ws.attempt_limit())),
                  _round_time_label(win, now, d is not None or verdict[1] == "in progress")))
    body = []
    if current and F.attempts_exhausted(ws, r):
        body.append('<p class="warn">%s</p>' % _e(D._close_round_text(ws)))
    tasks = [e for e in arts if e["kind"] == "task"]
    if tasks:
        body.append(_row("Task intake", "".join(_chip(ws, e, primary) for e in tasks)))
    for m in D.MODULES:
        mine = D._step_artifacts(ws, m, arts)
        trials = [e for e in mine if _is_trial(e)]
        cell = "".join(_chip(ws, e, primary) for e in mine if e not in trials)
        if trials:
            ends = trials[0]["id"] + ("" if len(trials) == 1 else " … " + trials[-1]["id"])
            cell += ('<details class="trials" id="round-%d-trials"><summary>%s</summary><div>%s</div></details>'
                     % (n, _e("%s (%s)" % (plural(len(trials), "trial candidate"), ends)),
                        "".join(_chip(ws, e, primary) for e in trials)))
        if current:
            cell += _module_note(ws, m, progress)
        body.append(_row(D.stage_title(ws, m), cell or '<span class="sub">none this round</span>'))
    dispatches = ws.dispatches(rnd=n)
    if dispatches:
        body.append(_row("Dispatched agents", "".join(_dispatch_line(d) for d in dispatches)))
    others = [e for e in arts if e["kind"] not in ("task", "record") and D.stage_of(e["producer"]) is None]
    if others:
        body.append(_row("Other", "".join(_chip(ws, e, primary) for e in others)))
    records = [e for e in arts if e["kind"] == "record"]
    if records:
        body.append(_row("Experiment records", "".join(_chip(ws, e, primary) for e in records)))
    body.append(_row("Decision", _decision(ws, n, r, need_approval), "decision"))
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
    """A candidate of ② (Train & tune): every one is a trial of a search (an old workspace's tune trials too)."""
    return e["kind"] == "candidate" and D.stage_of(e["producer"]) == "train"


def _who(by, quote=None):
    """Who decided: code, decision agent or user <name> (with the user's words)."""
    if by in DECIDERS:
        return DECIDERS[by]
    return "user %s%s" % (by, ": “%s”" % quote if quote else "")


def _log_order(evs):
    """Where the run log records the events the other sources show (registrations, approvals, round decisions, a
    stop answered): {key: line index}, so that events of the same second keep the run log's order."""
    seq = {}
    for i, ev in enumerate(evs):
        kind = ev["event"]
        if kind == "register" and ev.get("id"):
            seq.setdefault(("reg", ev["id"]), i)
        elif kind in ("approval", "approve"):
            seq.setdefault(("approve", ev.get("what"), ev.get("quote")), i)
        elif kind == "answer":
            seq.setdefault(("approve", ev.get("point"), ev.get("quote")), i)
            if ev.get("choice") == "stop":
                seq.setdefault(("stop",), i)
            if ev.get("choice") == "close_round":
                seq.setdefault(("close", ev.get("quote")), i)
        elif kind == "action" and isinstance(ev.get("action"), dict):
            a = ev["action"]
            if a.get("type") == "promote":
                seq.setdefault(("decide", a.get("evaluation_id")), i)
            elif a.get("type") == "close_round":
                seq.setdefault(("close", None, a.get("reason")), i)
            elif a.get("type") == "next":
                seq.setdefault(("next",), i)
    return seq


def _module_of_dispatch(ws, did):
    d = D._dispatch_or_none(ws, did) if did else None
    return d.get("module") if d else None


def _when_ended(ev):
    """How a long_job_* event came about: "at the stop" (a stop; older events have no context) or "when its dispatch
    was closed" (a decision or `rsi close-round` closed its dispatch)."""
    return "when its dispatch was closed" if ev.get("context") == "close" else "at the stop"


def _run_events(ws, evs=None):
    """The run log's events worth a line, as (UTC time, dot class, HTML text, source: rsi run, or rsi answer for the
    user's answers, line index): dispatches,
    returns (counted per dispatch: a continuation after a long job, the second part of a two-part hand-back and the
    user's answers going back are no returns), fresh sessions, a hand-back's questions (its artifacts held), decisions
    (not a reused one: its first record is shown), the user's answers, the questions asked (the same question again
    without an answer between: one line), and what a stop did to agents, long jobs and records (and what closing a
    dispatch did to its long job; an orphan pid file whose process had ended gets no line, as in REPORT.md)."""
    out, returns, last_ask = [], {}, None
    for i, ev in enumerate(D.run_events(ws) if evs is None else evs):
        kind, did, at = ev["event"], ev.get("dispatch_id"), ev.get("at") or ""
        text = cls = None
        if kind == "launch" and ev.get("mode") == "first":
            text, cls = "Dispatched rsi-%s (%s, %s, session %s)" % (ev.get("module"), did, ev.get("runner") or "—",
                                                                   _short(ev.get("session"))), "agent"
        elif kind == "resume":
            if ev.get("mode") == "continue":
                text = "Resumed %s (long-job wrap-up, session %s)" % (did, _short(ev.get("session")))
            else:
                returns[did] = returns.get(did, 0) + 1
                text = "Returned %s (return %d, session %s, %s)" % (did, returns[did], _short(ev.get("session")),
                                                                    ev.get("feedback"))
            cls = "agent"
        elif kind == "followup":
            text, cls = "Continued %s (second part of the hand-back, not a return, session %s, %s)" % (
                did, _short(ev.get("session")), ev.get("feedback")), "agent"
        elif kind == "resume_with_answer":
            text, cls = "Sent the user's answer back to %s (session %s, %s)" % (did, _short(ev.get("session")),
                                                                               ev.get("feedback")), "agent"
        elif kind == "asked":
            held = ev.get("held") or []
            text = "rsi-%s (%s) asked in its hand-back: waiting for the user; %s" % (
                _module_of_dispatch(ws, did) or "?", did,
                "held unregistered: %s" % ", ".join(str(h) for h in held) if held
                else "no artifact was handed back this time")
            cls = "approve"
        elif kind == "agent_stopped":
            text, cls = "Terminated agent %s at the stop" % did, "stop"
        elif kind == "long_job_stopped":
            text, cls = "Terminated long job %s (pid %s) %s" % (did, ev.get("pid"), _when_ended(ev)), "stop"
        elif kind == "long_job_gone" and not ev.get("orphan"):  # an orphan pid file whose process ended: no line
            text, cls = "Long job %s: its process was already gone (pid %s) %s" % (did, ev.get("pid"),
                                                                                  _when_ended(ev)), "stop"
        elif kind == "long_job_left":
            text, cls = "Could not terminate long job %s %s: %s; it may still be running" % (
                did, _when_ended(ev), D._clause(ev.get("why"))), "stop"
        elif kind == "closed":
            text = "Closed dispatch %s (%s%s)" % (did, "the run has stopped" if ev.get("why") == "stopped"
                                                  else D._clause(ev.get("why") or "—"),
                                                  ", never launched" if ev.get("launched") is False else "")
            cls = "stop"
        elif kind == "fresh_session":
            text, cls = "Fresh session took over %s: %s" % (did, ev.get("why")), "agent"
        elif kind == "decision" and not ev.get("reused"):
            by = ev.get("by")
            text = "Decision %s → %s (%s%s)" % (ev.get("point"), ev.get("choice"), _who(by, ev.get("quote")),
                                               "; " + D._clause(ev["reason"]) if ev.get("reason") else "")
            cls = "decide"
        elif kind == "spend_invalid":
            counted = sorted((ev.get("counted") or {}).items())
            text, cls = "Spend file of %s, line %s: %s (counted: %s)" % (
                did, ev.get("line"), ev.get("error"), ", ".join("%s %s" % (k, _num(v)) for k, v in counted) or "nothing"
            ), "stop"
        elif kind == "decision_agent_failed":
            text, cls = "The decision agent gave no usable answer (%s %s): using the code's default" % (
                ev.get("point"), ev.get("error")), "decide"
        elif kind == "answer":
            text = "Answer %s: %s (%s)" % (ev.get("point"), ev.get("choice"), _who(ev.get("by"), ev.get("quote")))
            cls, last_ask = "approve", None
        elif kind == "action" and isinstance(ev.get("action"), dict) and ev["action"].get("type") == "ask":
            question = ev["action"].get("question")
            if question != last_ask:
                text, cls = "Asked the user: %s" % question, "approve"
            last_ask = question
        if text is not None:
            out.append((at, cls, _e(text), "rsi answer" if kind == "answer" else "rsi run", i))
    return out


def _decision_source(d, seq):
    """(source, run log line index) of a round decision: rsi run when its run log holds the promote or close_round
    action that made it, rsi answer for a round closed by the user's close_round answer, else the command of the
    flow run by hand (rsi promote, rsi close-round)."""
    if d.get("evaluation_id") is not None:
        where = seq.get(("decide", d["evaluation_id"]))
        return ("rsi run" if where is not None else "rsi promote"), where
    hit = next(((k, v) for k, v in sorted(seq.items(), key=lambda kv: kv[1])
                if k[0] == "close" and (k[-1] or "") in (d.get("reason") or "")), None)
    if hit is None:
        return "rsi close-round", None
    return ("rsi run" if len(hit[0]) == 3 else "rsi answer"), hit[1]  # ("close", None, reason): the engine's action


def _timeline(ws, task, page):
    """The events of the events page, newest first: [(UTC time, dot class, HTML text, source, run log line index or
    None)]: registrations (a run of consecutive trial candidates of ② as one line), approvals, the run log's events
    (_run_events), round decisions and the stop. Artifact links are relative to `page`."""
    s = ws.state
    primary = task["metrics"]["primary"]
    known = set(e["id"] for e in ws.entries())
    evs = D.run_events(ws)
    seq = _log_order(evs)
    events = []  # (UTC time, dot class, HTML text, source, run log line index or None)
    entries = ws.entries()
    i = 0
    while i < len(entries):
        e = entries[i]
        j = i
        while _is_trial(e) and j + 1 < len(entries) and _is_trial(entries[j + 1]):
            j += 1
        if j > i:
            run = entries[i:j + 1]
            text = "%s%s … %s%s" % (_e("Registered %d trial candidates (" % len(run)), _art(ws, page, run[0]["id"]),
                                    _art(ws, page, run[-1]["id"]), _e(")"))
            events.append((run[-1].get("registered_at") or "", "trial", text, PRODUCER_NAMES["train"],
                           seq.get(("reg", run[-1]["id"]))))
        else:
            _cls, gate = _gate(ws, e, primary)
            text = "%s %s%s" % (_e("Registered " + KIND_NAMES.get(e["kind"], e["kind"])), _art(ws, page, e["id"]),
                                _e(" (%s)" % gate) if gate else "")
            who = D.stage_of(e["producer"]) or e["producer"]
            events.append((e.get("registered_at") or "", "record" if e["kind"] == "record" else "reg", text,
                           PRODUCER_NAMES.get(who, e["producer"]), seq.get(("reg", e["id"]))))
        i = j + 1
    for a in s["approvals"]:
        ref = a.get("ref")
        ref = " " + (_art(ws, page, ref) if ref in known else _e(ref)) if ref else ""
        text = "%s%s%s" % (_e("Approved %s" % a.get("what")), ref, _e(" (%s: “%s”)%s" % (
            a.get("by"), a.get("quote"), "; used by %s" % a["used_by"] if a.get("used_by") else "")))
        events.append((a.get("at") or "", "approve", text, "approval", seq.get(("approve", a.get("what"), a.get("quote")))))
    events += _run_events(ws, evs)
    for n, r in _rounds(ws):
        d = r["decision"]
        if d is not None:
            text = "Round %d decided: %s (%s)" % (n, _verdict(d), D._clause(d.get("reason") or "—"))
            events.append((d.get("at") or "", "decide", _e(text)) + _decision_source(d, seq))
    stopped = s["stopped"]
    if stopped:
        if seq.get(("stop",)) is not None:
            source = "rsi answer"  # the user answered a question with stop
        elif stopped.get("by"):
            source = "rsi stop"
        else:  # the stop condition, found by the next step: rsi run's when its run log holds one
            source = "rsi run" if seq.get(("next",)) is not None else "rsi next"
        events.append((stopped.get("at") or "", "stop", _e("Run stopped: %s" % stopped.get("reason")), source,
                       seq.get(("stop",)) or seq.get(("next",))))
    # By time, and within a second by the run log's order (events it does not record go after those it does).
    big = len(evs) + 1
    events.sort(key=lambda t: (t[0], big if t[4] is None else t[4]))
    events.reverse()
    return events


def _events_page(ws, task):
    """The timeline (_timeline), grouped by local date."""
    events = _timeline(ws, task, EVENTS)
    items, day = [], None
    for at, cls, text, source, _seq in events[:EVENTS_SHOWN]:
        dt = _when(at)
        date = dt.strftime("%Y-%m-%d") if dt else "time unknown"
        if date != day:
            items.append('<li class="day">%s</li>' % _e(date))
            day = date
        items.append('<li><span class="t">%s</span><span class="dot %s"></span><span>%s <span class="who">· %s</span>'
                     '</span></li>' % (_e(dt.strftime("%H:%M:%S") if dt else "—"), cls, text, _e(source)))
    note = "" if len(events) <= EVENTS_SHOWN else '<p class="sub">%s</p>' % _e(
        "%d events in all; only the latest %d are listed." % (len(events), EVENTS_SHOWN))
    return '<section id="events" class="card"><h2>Events (newest first)</h2><ul class="tl">%s</ul>%s</section>' % (
        "".join(items), note)


def _report_page(ws):
    path = os.path.join(ws.root, D.REPORT_FILE)
    if not os.path.isfile(path):
        return '<section id="report" class="card"><p class="empty">No run report yet (it is written at the stop).</p></section>'
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return ('<section id="report" class="card"><p class="sub">Source: <a href="%s">%s</a></p><div class="md">%s</div>'
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
.chip.run,.chip.ok{background:#EAF5E9;border-color:#BFDDBB}.chip.stop,.chip.idle{background:#F1F2F4}
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
.card h2 .hint{font-weight:normal;color:var(--muted);font-size:13px;margin-left:8px}
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
svg.flow .t4{font-size:11.5px;fill:#8A4B12}
svg.flow .note{font-size:12.5px;fill:#444}
svg .ln{fill:none;stroke:#222;stroke-width:1.6}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:#4B5563;margin-top:8px}
.sw{display:inline-block;width:14px;height:12px;border:1px solid #222;border-radius:3px;vertical-align:-1px;margin-right:5px}
.sw.done{background:#CEE5CC}.sw.current{background:#FBE5D6}.sw.wait{background:#FFF3C0}.sw.fail{background:#FFE3F1}
.sw.todo{background:#FFF}
.grid,.two{display:grid;grid-template-columns:1fr 1.25fr 1fr;gap:18px;margin-bottom:18px}
.two{grid-template-columns:1.15fr 1fr}
.grid .card,.two .card{margin-bottom:0;min-width:0}
.dblock{padding:12px 0;border-top:1px solid var(--line)}.dblock:first-of-type{border-top:0;padding-top:0}
.dblock:last-child{padding-bottom:0}
.dhead{font-size:15px;overflow-wrap:anywhere}
.psum{font-size:15px;font-weight:600;margin:10px 0 4px;overflow-wrap:anywhere}
.facts{display:flex;flex-wrap:wrap;gap:6px 8px;margin:6px 0}
.fact{border:1px solid var(--line);border-radius:8px;padding:2px 9px;font-size:13px;background:#FCFCFD;
overflow-wrap:anywhere}
.fact .k{color:var(--muted);margin-right:6px}
.tablewrap{overflow-x:auto}
table.ptable{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0;font-variant-numeric:tabular-nums}
.ptable caption{text-align:left;color:#374151;font-weight:600;padding:2px 0 4px}
.ptable th,.ptable td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top;
white-space:nowrap}
.ptable th{color:#4B5563;font-weight:600;background:#FAFBFC}.ptable td.long{white-space:normal;min-width:16em}
.pnotes{margin:6px 0;padding-left:20px;font-size:13.5px;overflow-wrap:anywhere}
.hbs{list-style:none;margin:0;padding:0}
.hb{padding:10px 0;border-bottom:1px dashed var(--line)}.hb:first-child{padding-top:0}.hb:last-child{border-bottom:0}
.hbh{overflow-wrap:anywhere}.hbh .t{color:var(--muted);font-variant-numeric:tabular-nums;margin-right:4px}
.hbn{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13.5px;margin:4px 0}
.hb details>summary{cursor:pointer;color:var(--muted);font-size:13px}
.qhead{font-size:13px;color:#6B5E1F;margin-top:4px}
.qs{margin:2px 0;padding-left:20px;font-size:13.5px;font-weight:600;overflow-wrap:anywhere}
.gpus{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.gpu{border:1px solid var(--line);border-radius:10px;padding:8px 10px;font-size:12.5px;overflow-wrap:anywhere}
.gpu .track{height:7px;background:#EEF0F3;border-radius:99px;overflow:hidden;margin:6px 0 4px;display:flex}
.gpu.off{opacity:.6}
.ours{background:#E9A38F}.other{background:#C9CED6}
.gkey{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}
.gkey i{display:inline-block;width:12px;height:8px;border-radius:2px;margin-right:4px}
.tl.recent li{grid-template-columns:92px 14px 1fr}
.more{margin:10px 0 0;font-size:13.5px}
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
.dot.agent{background:#D2E2FB}.dot.approve{background:#FFF3C0}.dot.decide{background:#FBE5D6}.dot.stop{background:#C5C5D0}
.tl .who{color:var(--muted);font-size:13px}
.agents{list-style:none;margin:8px 0;padding:0}.agents li{margin:3px 0;overflow-wrap:anywhere}
.answer{margin:8px 0}.opts{margin:4px 0;padding-left:20px}
pre.cmd{background:#F6F8FA;border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:6px 0;
font-size:12.5px;white-space:pre-wrap;overflow-wrap:anywhere}
.disp{margin:2px 0;overflow-wrap:anywhere}
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
@media (max-width:860px){.grid,.two{grid-template-columns:1fr}.step{grid-template-columns:1fr;gap:4px}
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
function pl(k,w){return k+" "+w+(k===1?"":"s");}
function fmt(s){s=Math.max(0,Math.floor(s));if(s<60){return "under a minute";}
if(s<3600){return pl(Math.floor(s/60),"minute");}
if(s<86400){var h=Math.floor(s/3600),m=Math.floor(s%3600/60);return pl(h,"hour")+(m?" "+pl(m,"minute"):"");}
return pl(Math.floor(s/86400),"day")+" "+pl(Math.floor(s%86400/3600),"hour");}
try{var ts=document.querySelectorAll("[data-start]");
for(var j=0;j<ts.length;j++){var t=ts[j],b=Date.parse(t.getAttribute("data-start"));
if(!isNaN(b)){t.textContent=(t.getAttribute("data-prefix")||"")+(t.getAttribute("data-approx")&&Date.now()-b>=60000?"about ":"")
+fmt((Date.now()-b)/1000)+(t.getAttribute("data-suffix")||"");}}}catch(e){}
restore();window.addEventListener("load",restore);
window.addEventListener("scroll",save);window.addEventListener("pagehide",save);
})();
"""


def _status_label(ws, questions):
    """The top bar's status chip: the questions waiting for the user, the stop, "● running" only while an `rsi run`
    works on the workspace (holds work/run.lock), else what the run waits for (rsi.dashboard.idle_label)."""
    s = ws.state
    if questions:  # once stopped only the notes can hold questions
        return '<span class="chip ask">%s</span>' % ("⏸ waiting for the user (stopped)" if s["stopped"]
                                                     else "⏸ waiting for the user")
    if s["stopped"]:
        return '<span class="chip stop">■ stopped</span>'
    if D.run_active(ws):
        return '<span class="chip run">● running</span>'
    return '<span class="chip idle">%s</span>' % _e(D.idle_label(ws))


def _frame(ws, page, title, body, now, questions, label):
    """A whole page: head, top bar with the status label and tabs, `body`, footer."""
    s = ws.state
    tabs = "".join('<a%s href="%s">%s</a>' % (' class="on"' if rel == page else "", _e(_href(page, rel)), _e(name))
                   for rel, name in TABS)
    head = "%s%s · %s" % ("⏸ Waiting for the user · " if questions else "", title, s["task_id"])
    return "".join([
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n",
        '<meta http-equiv="refresh" content="%d">\n' % D.REFRESH_SECONDS,
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
        "<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n" % (_e(head), STYLE),
        '<header class="top"><div class="wrap"><div class="bar"><div class="title"><h1>%s</h1>'
        '<span class="ver">%s</span><span class="chip">%s</span>%s%s</div><div class="updated">%s</div></div>'
        '<nav class="tabs">%s</nav></div></header>\n'
        % (_e(s["task_id"]), _e("v%d" % s["task_version"]), _e("round %d" % s["round"]), label, _elapsed_label(ws, now),
           _e("Updated %s · reloads every %s" % (D._show(now), plural(D.REFRESH_SECONDS, "second"))), tabs),
        '<main><div class="wrap">\n%s\n</div></main>\n' % body,
        "<footer><div class=\"wrap\">%s</div></footer>\n" % _e(
            "Generated by rsi · read-only page; answers and actions happen in Claude · workspace %s" % ws.root),
        "<script>%s</script>\n</body>\n</html>\n" % SCRIPT,
    ])


def render_pages(ws, now=None):
    """The four pages of workspace `ws`: PAGES entry -> HTML text. `now` (a datetime) is the
    generation time shown (default: now). Reads only (see rsi.dashboard), writes nothing."""
    now = datetime.datetime.now() if now is None else now
    task = ws.task()
    progress = D.module_progress(ws)
    questions = D.pending_questions(ws, progress)
    bodies = ((OVERVIEW, "RSI progress", _overview(ws, task, progress, questions, now)),
              (ROUNDS, "Rounds & artifacts", _rounds_page(ws, task, progress, now)),
              (EVENTS, "Events", _events_page(ws, task)),
              (REPORT, "Run report", _report_page(ws)))
    label = _status_label(ws, questions)
    return dict((rel, _frame(ws, rel, title, body, now, questions, label)) for rel, title, body in bodies)
