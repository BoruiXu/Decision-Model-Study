"""Run and round durations, shared by the run report and the progress pages.

The run starts when round 1 was opened (STATE rounds["1"]["opened_at"]; a workspace
made before that field existed falls back to the task's registration time) and ends
at `stopped.at`, or is still going. A round starts at its `opened_at` and ends where
the next round starts (the last one at `stopped.at`). Where `opened_at` is missing the
start is approximated and the duration is shown with a leading 约. Reads only.
"""
import datetime

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


def stopped_unknown(ws):
    """True when the run is stopped but `stopped.at` is missing or malformed: no live duration then."""
    stopped = ws.state.get("stopped")
    return bool(stopped) and not _good(stopped.get("at"))


def run_window(ws):
    """(start ISO or None, end ISO or None, approx). End is None while the run is going."""
    rounds = _rounds(ws)
    opened = rounds[0][1].get("opened_at") if rounds else None
    if _good(opened):
        start, approx = opened, False
    else:
        start, approx = _task_time(ws), True
    stopped = ws.state.get("stopped")
    end = stopped.get("at") if stopped and _good(stopped.get("at")) else None
    return start, end, approx


def round_windows(ws):
    """Round number -> {"start": ISO or None, "end": ISO or None, "approx": bool}."""
    rounds = _rounds(ws)
    run_start, run_end, run_approx = run_window(ws)
    starts = {}
    for i, (n, r) in enumerate(rounds):
        if _good(r.get("opened_at")):
            starts[n] = (r["opened_at"], False)
        elif i == 0:
            starts[n] = (run_start, True)
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
    return out


def format_duration(seconds):
    """Four tiers, rounded down: 不到 1 分钟 / N 分钟 / H 小时 M 分 / D 天 H 小时."""
    s = max(0, int(seconds))
    if s < 60:
        return "不到 1 分钟"
    if s < 3600:
        return "%d 分钟" % (s // 60)
    if s < 86400:
        h, m = s // 3600, s % 3600 // 60
        return "%d 小时" % h if m == 0 else "%d 小时 %d 分" % (h, m)
    return "%d 天 %d 小时" % (s // 86400, s % 86400 // 3600)


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
    return ("约 " if approx else "") + format_duration(seconds_between(win_start, win_end, now))


def elapsed_parts(ws, now):
    """(prefix, duration text, suffix, start ISO if still running else None), or None when the
    run's start is unknown. The text is prefix + duration + suffix."""
    start, end, approx = run_window(ws)
    if start is None:
        return None
    if stopped_unknown(ws):
        return "共运行 ", "未知", "", None
    dur = _dur(start, end, approx, now)
    if end is None:
        return "已运行 ", dur, "（%s 开始）" % format_start(start, now), start
    return "共运行 ", dur, "", None


def elapsed_text(ws, now):
    p = elapsed_parts(ws, now)
    return "" if p is None else p[0] + p[1] + p[2]


def round_parts(win, now):
    """(prefix, duration text, start ISO if the round is still open else None), or None."""
    if win["start"] is None:
        return None
    if win.get("unknown"):
        return "", "未知", None
    dur = _dur(win["start"], win["end"], win["approx"], now)
    if win["end"] is None:
        return "进行中 · 已 ", dur, win["start"]
    return "", dur, None


def round_text(win, now, decided=False):
    """The round's time as text; `decided`: the round is decided but still the current one (until
    `rsi next`), so it reads "已 N 分钟" rather than "进行中 · 已 N 分钟"."""
    p = round_parts(win, now)
    if p is None:
        return ""
    prefix = "已 " if decided and p[0].startswith("进行中") else p[0]
    return prefix + p[1]
