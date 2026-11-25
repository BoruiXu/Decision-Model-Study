"""A module's progress file: <output directory>/progress.json (docs/interfaces/progress.md).

A module that wants the pages and the command line to show how far it is writes progress.json into its dispatch's
output directory; the framework shows what it gives and never reads a module's own files (logs, trsi or ojh
records). read_progress(path, now) checks one file and keeps what is valid, recording every rule it breaks in
"problems"; it never raises. dispatch_progress(ws, d, now) reads the file of dispatch `d` and marks it stale when
the dispatch is alive (alive: running, or a long job whose completion marker has not appeared) and the file was
last updated more than PROGRESS_STALE_SECONDS ago.
"""
import datetime
import json
import os

from rsi import flow as F

PROGRESS_FILE = "progress.json"
PROGRESS_STALE_SECONDS = 300
MAX_BYTES = 256 * 1024
SUMMARY_CHARS = 200
LABEL_CHARS, VALUE_CHARS, MAX_FACTS = 40, 80, 12
MAX_COLUMNS, MAX_ROWS, CELL_CHARS = 12, 100, 200
MAX_NOTES, NOTE_CHARS = 20, 500
UTC = datetime.timezone.utc


def _clip(s, n):
    return s if len(s) <= n else s[:n] + "…"


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _parse_time(s):
    """An aware datetime from UTC ISO 8601 with `Z` or an offset, or None (naive times are refused too)."""
    if not isinstance(s, str):
        return None
    t = s.strip()
    if t.endswith(("Z", "z")):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(t)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None and dt.utcoffset() is not None else None


def _num(x):
    """A number with %.6g, or None when it is out of a float's range (an integer of hundreds of digits)."""
    try:
        return "%.6g" % x
    except OverflowError:
        return None


def _cell(x):
    """A table cell as shown: a string as is (clipped), a number with %.6g, a boolean 是 or 否, null None; else
    False (not a cell, or a number out of range)."""
    if x is None:
        return None
    if isinstance(x, bool):
        return "是" if x else "否"
    if _number(x):
        s = _num(x)
        return False if s is None else s
    if isinstance(x, str):
        return _clip(x, CELL_CHARS)
    return False


def _facts(v, problems):
    if not isinstance(v, list):
        problems.append("facts 不是列表，没有用")
        return []
    out = []
    for i, f in enumerate(v[:MAX_FACTS], 1):
        label, value = (f.get("label"), f.get("value")) if isinstance(f, dict) else (None, None)
        if isinstance(value, str):
            text = value
        elif _number(value):
            text = _num(value)
        else:
            text = None
        if not isinstance(label, str) or text is None:
            problems.append("facts 第 %d 项不是 {label: 字符串, value: 字符串或数}（或数超出范围），已跳过" % i)
            continue
        out.append((_clip(label, LABEL_CHARS), _clip(text, VALUE_CHARS)))
    return out


def _table(v, problems):
    if not isinstance(v, dict):
        problems.append("table 不是对象，没有用")
        return None
    cols = v.get("columns")
    if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
        problems.append("table.columns 不是 1–%d 个字符串，表格没有用" % MAX_COLUMNS)
        return None
    if len(cols) > MAX_COLUMNS:
        problems.append("table.columns 有 %d 个，只用前 %d 个" % (len(cols), MAX_COLUMNS))
        cols = cols[:MAX_COLUMNS]
    title = v.get("title")
    if title is not None and not isinstance(title, str):
        problems.append("table.title 不是字符串，没有用")
        title = None
    rows_in = v.get("rows", [])
    if not isinstance(rows_in, list):
        problems.append("table.rows 不是列表，没有用")
        rows_in = []
    rows, bad_rows, bad_cells = [], 0, 0
    for r in rows_in[:MAX_ROWS]:
        if not isinstance(r, list):
            bad_rows += 1
            continue
        r = (r + [None] * len(cols))[:len(cols)]
        cells = [_cell(x) for x in r]
        bad_cells += sum(1 for c in cells if c is False)
        rows.append([None if c is False else c for c in cells])
    if bad_rows:
        problems.append("table.rows 有 %d 行不是列表，已跳过" % bad_rows)
    if bad_cells:
        problems.append("table 有 %d 格不是字符串、数、布尔或 null（或数超出范围），显示为空" % bad_cells)
    return {"title": _clip(title, CELL_CHARS) if title else None,
            "columns": [_clip(c, CELL_CHARS) for c in cols], "rows": rows}


def _notes(v, problems):
    if not isinstance(v, list):
        problems.append("notes 不是列表，没有用")
        return []
    out = [_clip(n, NOTE_CHARS) for n in v[:MAX_NOTES] if isinstance(n, str)]
    if len(out) < len(v[:MAX_NOTES]):
        problems.append("notes 有 %d 条不是字符串，已跳过" % (len(v[:MAX_NOTES]) - len(out)))
    return out


def read_progress(path, now=None):
    """The progress file at `path` as shown (None when it does not exist): {"path", "updated_at" (aware datetime or
    None), "age_s" (seconds since updated_at, or None), "summary", "facts" [(label, value)], "table" ({"title",
    "columns", "rows"} or None), "notes", "problems"}. Never raises: an unreadable file, one larger than MAX_BYTES,
    one that is not UTF-8, not JSON or not an object, or of another version than 1 shows nothing but one problem."""
    out = {"path": path, "updated_at": None, "age_s": None, "summary": None, "facts": [], "table": None, "notes": [],
           "problems": []}
    problems = out["problems"]
    try:
        try:
            with open(path, "rb") as f:
                raw = f.read(MAX_BYTES + 1)
        except FileNotFoundError:
            return None
        except OSError as e:
            problems.append("读不了 progress.json：%s" % (e.strerror or e))
            return out
        if len(raw) > MAX_BYTES:
            problems.append("progress.json 超过 %d KiB，没有读" % (MAX_BYTES // 1024))
            return out
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            problems.append("progress.json 不是 UTF-8")
            return out
        try:
            doc = json.loads(text)
        except ValueError as e:
            problems.append("progress.json 不是合法的 JSON（可能正写到一半）：%s" % e)
            return out
        if not isinstance(doc, dict):
            problems.append("progress.json 不是 JSON 对象")
            return out
        version = doc.get("version", 1)
        if version != 1 or isinstance(version, bool):
            problems.append("progress.json 的 version 是 %s，不认识（只认 1），整个文件没有用"
                            % json.dumps(version, ensure_ascii=False)[:40])
            return out
        when = _parse_time(doc.get("updated_at"))
        if when is None:
            problems.append("updated_at 缺失或不是带时区的 ISO 8601 时间，不判断进度是否过时" if "updated_at" in doc
                            else "没有 updated_at，不判断进度是否过时")
        else:
            now = now or datetime.datetime.now(UTC)
            if now.tzinfo is None:
                now = now.replace(tzinfo=UTC)
            try:
                out["updated_at"], out["age_s"] = when.astimezone(UTC), (now - when).total_seconds()
            except OverflowError:  # e.g. 0001-01-01T00:00:00+01:00 is before year 1 in UTC
                problems.append("updated_at 超出能表示的时间范围，不判断进度是否过时")
        s = doc.get("summary")
        if isinstance(s, str):
            out["summary"] = _clip(s, SUMMARY_CHARS)
        elif s is not None:
            problems.append("summary 不是字符串，没有用")
        if doc.get("facts") is not None:
            out["facts"] = _facts(doc["facts"], problems)
        if doc.get("table") is not None:
            out["table"] = _table(doc["table"], problems)
        if doc.get("notes") is not None:
            out["notes"] = _notes(doc["notes"], problems)
    except Exception as e:  # never let the pages or the engine fail on a module's file
        problems.append("读 progress.json 出错：%s: %s" % (type(e).__name__, e))
    return out


def alive(ws, d):
    """Whether dispatch `d` is still at work: its agent is running, or it is a long job whose completion marker has
    not appeared."""
    return d.get("status") == "running" or (d.get("status") == "long_job" and not F.marker_done(ws, d))


def dispatch_progress(ws, d, now=None):
    """read_progress of <ws>/<d["outdir"]>/progress.json plus "stale": the dispatch is alive and the file is older
    than PROGRESS_STALE_SECONDS. None when there is no such file."""
    if not d.get("outdir"):
        return None
    p = read_progress(os.path.join(ws.root, d["outdir"], PROGRESS_FILE), now)
    if p is not None:
        p["stale"] = bool(alive(ws, d) and p["age_s"] is not None and p["age_s"] > PROGRESS_STALE_SECONDS)
    return p
