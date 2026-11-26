"""Run and round durations, shared by the run report and the progress pages.

The run starts at the first `rsi run` (first_run: `rsi init` may come hours before it), else when round 1 was
opened (made_at: STATE rounds["1"]["opened_at"]; a workspace made before that field existed falls back to the
task's registration time), and ends at `stopped.at`, or is still going. A round starts at its `opened_at` (round 1
at the first `rsi run`, when that is before its end) and ends where the next round starts (the last one at
`stopped.at`). Where `opened_at` is missing the start is approximated and the duration is shown with a leading
"about". plural() builds the counted words the run report, the pages and the status line share. Reads only.
"""
import datetime
import json
import os

_FMT = "%Y-%m-%dT%H:%M:%SZ"


def utc(ts):
    """Parse a STATE/registry time (2026-11-22T01:02:03Z) as an aware UTC datetime."""
    return datetime.datetime.strptime(ts, _FMT).replace(tzinfo=datetime.timezone.utc)


def _iso(dt):
    return dt.astimezone(datetime.timezone.utc).strftime(_FMT)


def _good(ts):
    try:
        utc(ts)
        return True
    except (TypeError, ValueError):
        return False


def _rounds(ws):
    return sorted((int(k), r) for k, r in ws.state["rounds"].items())


def _task_time(ws):
    for e in ws.entries("task"):
        if _good(e.get("registered_at")):
            return e["registered_at"]
    return None


def first_run_at(ws):
    """The time (ISO) of the first `rsi run`'s first step: the earliest `action` or `launch` event in
    <ws>/work/run.jsonl (lines are appended in time order); None when there is none or the log cannot be read."""
    try:
        with open(os.path.join(ws.root, "work", "run.jsonl"), encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if isinstance(ev, dict) and ev.get("event") in ("action", "launch") and _good(ev.get("at")):
                    return ev["at"]
    except OSError:
        return None
    return None


def stopped_unknown(ws):
    """True when the run is stopped but `stopped.at` is missing or malformed: no live duration then."""
    stopped = ws.state.get("stopped")
    return bool(stopped) and not _good(stopped.get("at"))


def made_at(ws):
    """(when the workspace's run was set up: round 1's opening, else the task's registration, ISO or None; approx)."""
    rounds = _rounds(ws)
    opened = rounds[0][1].get("opened_at") if rounds else None
    if _good(opened):
        return opened, False
    return _task_time(ws), True


def _end(ws):
    stopped = ws.state.get("stopped")
    return stopped.get("at") if stopped and _good(stopped.get("at")) else None


def first_run(ws):
    """first_run_at when the run is counted from it (it is not after the stop), else None."""
    first, end = first_run_at(ws), _end(ws)
    return first if first is not None and (end is None or utc(first) <= utc(end)) else None


def run_window(ws):
    """(start ISO or None, end ISO or None, approx): from the first `rsi run` (first_run), else from made_at. End
    is None while the run is going."""
    start, approx = made_at(ws)
    first = first_run(ws)
    if first is not None:
        start, approx = first, False
    return start, _end(ws), approx


def round_windows(ws):
    """Round number -> {"start": ISO or None, "end": ISO or None, "approx": bool, "unknown": bool}. The first round
    starts at the first `rsi run` (first_run) when that is before its end, as the run does."""
    rounds = _rounds(ws)
    run_end = _end(ws)
    made = made_at(ws)[0]
    starts = {}
    for i, (n, r) in enumerate(rounds):
        if _good(r.get("opened_at")):
            starts[n] = (r["opened_at"], False)
        elif i == 0:
            starts[n] = (made, True)
        else:
            pn, pr = rounds[i - 1]
            d = pr.get("decision") or {}
            starts[n] = (d["at"], True) if _good(d.get("at")) else (starts[pn][0], True)
    out = {}
    for i, (n, _r) in enumerate(rounds):
        start, approx = starts[n]
        if i + 1 < len(rounds):
            end, end_approx = starts[rounds[i + 1][0]]
            approx = approx or end_approx
        else:
            end = run_end
        out[n] = {"start": start, "end": end, "approx": approx,
                  "unknown": end is None and i + 1 == len(rounds) and stopped_unknown(ws)}
    first = first_run(ws)
    if out and first is not None:
        w = out[rounds[0][0]]
        if w["end"] is None or utc(first) <= utc(w["end"]):
            w["start"] = first  # never longer than the run
    return out


def plural(n, word, words=None):
    """`n` followed by `word`, or by its plural `words` (default: `word` + "s") unless `n` is 1: "1 minute",
    "3 minutes", "0 GPUs". An integer shows as is, any other number with %.6g."""
    return "%s %s" % ("%d" % n if isinstance(n, int) else "%.6g" % n, word if n == 1 else words or word + "s")


def format_duration(seconds):
    """Four tiers, rounded down: under a minute / N minutes / H hours M minutes / D days H hours."""
    s = max(0, int(seconds))
    if s < 60:
        return "under a minute"
    if s < 3600:
        return plural(s // 60, "minute")
    if s < 86400:
        h, m = s // 3600, s % 3600 // 60
        return plural(h, "hour") if m == 0 else "%s %s" % (plural(h, "hour"), plural(m, "minute"))
    return "%s %s" % (plural(s // 86400, "day"), plural(s % 86400 // 3600, "hour"))


def approx_duration(seconds, approx):
    """format_duration with a leading "about " when `approx` (not before "under a minute")."""
    return ("about " if approx and seconds >= 60 else "") + format_duration(seconds)


def _local(dt):
    return dt.astimezone()


def format_start(start_iso, now):
    """Local time: HH:MM on the day of `now`, otherwise MM-DD HH:MM. A naive `now` is local."""
    dt = _local(utc(start_iso))
    if dt.date() == _local(now).date():
        return dt.strftime("%H:%M")
    return dt.strftime("%m-%d %H:%M")


def format_full(ts):
    """Local YYYY-MM-DD HH:MM of a STATE time."""
    return _local(utc(ts)).strftime("%Y-%m-%d %H:%M")


def seconds_between(start_iso, end, now):
    """Seconds from `start_iso` to the ISO `end`, or to `now` when `end` is None."""
    stop = _local(now) if end is None else utc(end)
    return max(0.0, (stop - utc(start_iso)).total_seconds())


def _dur(win_start, win_end, approx, now):
    return approx_duration(seconds_between(win_start, win_end, now), approx)


def elapsed_parts(ws, now):
    """(prefix, duration text, suffix, start ISO if still running else None, approx), or None when the
    run's start is unknown. The text is prefix + duration + suffix; approx: the start is approximated (the
    duration text leaves "about" out under a minute, so a live label takes it from here)."""
    start, end, approx = run_window(ws)
    if start is None:
        return None
    if stopped_unknown(ws):
        return "total time: ", "unknown", "", None, False
    dur = _dur(start, end, approx, now)
    if end is None:
        return "running for ", dur, " (started %s)" % format_start(start, now), start, approx
    return "total time: ", dur, "", None, approx


def elapsed_text(ws, now):
    p = elapsed_parts(ws, now)
    return "" if p is None else p[0] + p[1] + p[2]


OPEN_ROUND = "in progress for "  # round_parts' prefix while the round is open
SO_FAR = " so far"  # the suffix that replaces it once something else says the round is open or decided


def round_parts(win, now):
    """(prefix, duration text, start ISO if the round is still open else None, approx), or None."""
    if win["start"] is None:
        return None
    if win.get("unknown"):
        return "", "unknown", None, False
    dur = _dur(win["start"], win["end"], win["approx"], now)
    if win["end"] is None:
        return OPEN_ROUND, dur, win["start"], win["approx"]
    return "", dur, None, win["approx"]


def round_text(win, now, decided=False):
    """The round's time as text; `decided`: the round is decided but still the current one (until
    `rsi next`), so it reads "N minutes so far" rather than "in progress for N minutes"."""
    p = round_parts(win, now)
    if p is None:
        return ""
    if decided and p[0] == OPEN_ROUND:
        return p[1] + SO_FAR
    return p[0] + p[1]
