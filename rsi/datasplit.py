"""The data section of a task definition (data.sources, data.rows, data.split) and the split rule (stdlib only).

docs/interfaces/task.md (`data`) defines them. A split rule assigns the rows of a corpus to the four sets: `by` time (a
numeric `key`, closed `ranges` per set), group (whole groups of the `key`'s value, `fractions` per set) or random (by
unit, `fractions` per set); `sizes` per set (none or null: every row in range), `stratify`, `unit` and `seed` have
defaults (SPLIT_DEFAULTS). A task may write part of the rule; the data module fills in the rest from the data and
records the filled rule as a new task version. That version is a fill (fill_errors): it registers without a
task_change approval, as long as no evalset or testset is registered (rsi.workspace). corpus_errors checks the rows of a
corpus (all the labelled rows, docs/interfaces/datasets.md) against the rule and data.rows.expect.
"""
import copy
import json
import os

from .metrics import hard_label
from .rules import is_number
from .schema import load_schema, validate

SET_KINDS = ("testset", "evalset", "valset", "trainset")  # the order the sets are drawn in
SPLIT_DEFAULTS = {"stratify": True, "unit": "cluster", "seed": 0}
# A path with any of these, or a leading !, cannot be written as a safe Claude Code deny rule (rsi.briefs.deny_rules).
UNSAFE_PATH_CHARS = "()*?["
PER_SET = ("ranges", "fractions", "sizes")  # the keys that hold one value per set
WHERE = "$.data.split"
_KEY_ORDER = ("by", "key", "ranges", "fractions", "sizes", "stratify", "unit", "seed")
_SET_ORDER = tuple(reversed(SET_KINDS))  # as a rule is written: trainset, valset, evalset, testset
_MISSING = object()


def _data(task):
    data = task.get("data") if isinstance(task, dict) else None
    return data if isinstance(data, dict) else {}


def source_paths(task):
    """The paths of the object items of data.sources (the raw data), in order; a string item only describes."""
    return [s["path"] for s in _data(task).get("sources") or [] if isinstance(s, dict) and "path" in s]


def task_split(task):
    """The task's data.split (a copy), {} when it has none."""
    return copy.deepcopy(_data(task).get("split") or {})


def value_at(row, key):
    """(True, value) at the dotted path `key` of `row` (e.g. meta.extra.step), (False, None) when it is not there."""
    cur = row
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _split_schema():
    return load_schema("task")["properties"]["data"]["properties"]["split"]


def _missing(split):
    """What a rule lacks to be complete: `by`; a `key` for time and group; four `ranges` for time, four `fractions`
    for group and random."""
    by = split.get("by")
    if by not in ("time", "group", "random"):
        return ["%s.by: missing (time, group or random)" % WHERE]
    out = []
    if by != "random" and not isinstance(split.get("key"), str):
        out.append("%s.key: by %s needs a key" % (WHERE, by))
    per = "ranges" if by == "time" else "fractions"
    given = split.get(per) if isinstance(split.get(per), dict) else {}
    lack = [k for k in _SET_ORDER if k not in given]
    if lack:
        out.append("%s.%s: by %s needs one for each set (missing: %s)" % (WHERE, per, by, ", ".join(lack)))
    return out


def is_complete(split):
    """True when the rule names everything the split needs; the other keys take their defaults (filled)."""
    return isinstance(split, dict) and not _missing(split)


def split_errors(split, partial=False):
    """Why the rule `split` is malformed ([] when it is fine): its keys and values (the task schema's data.split), and
    that the keys agree: time takes ranges, group and random take fractions, random takes no key; each range is
    [lo, hi] with lo <= hi and no two ranges overlap; the fractions sum to at most 1. `partial`: only what is written
    must agree (a task may leave part of the rule to the data module); otherwise the rule must also be complete."""
    errors = validate(split, _split_schema(), WHERE)
    if errors:
        return errors
    by = split.get("by")
    if "ranges" in split and "fractions" in split:
        errors.append("%s: ranges (by time) and fractions (by group or random) exclude each other" % WHERE)
    if by in ("group", "random") and "ranges" in split:
        errors.append("%s.ranges: by %s splits by fractions; only by time takes ranges" % (WHERE, by))
    if by == "time" and "fractions" in split:
        errors.append("%s.fractions: by time splits by ranges; only by group or random take fractions" % WHERE)
    if by == "random" and "key" in split:
        errors.append("%s.key: by random draws by unit and takes no key" % WHERE)
    ranges = split.get("ranges", {})
    ok = {}
    for k in _SET_ORDER:
        if k not in ranges:
            continue
        r = ranges[k]
        if len(r) != 2:
            errors.append("%s.ranges.%s: expected [lo, hi] (got %s)" % (WHERE, k, r))
        elif r[0] > r[1]:
            errors.append("%s.ranges.%s: lo %s is above hi %s" % (WHERE, k, r[0], r[1]))
        else:
            ok[k] = r
    named = list(ok)
    for i, a in enumerate(named):
        for b in named[i + 1:]:
            if ok[a][0] <= ok[b][1] and ok[b][0] <= ok[a][1]:
                errors.append("%s.ranges: %s %s and %s %s overlap (each range is a closed interval)"
                              % (WHERE, a, ok[a], b, ok[b]))
    total = sum(split.get("fractions", {}).values())
    if total > 1 + 1e-9:
        errors.append("%s.fractions: the fractions sum to %g, more than 1" % (WHERE, total))
    if not partial:
        errors += _missing(split)
    return errors


def filled(split):
    """The rule with its defaults written out (SPLIT_DEFAULTS; `sizes` {} when not written), keys in a fixed order."""
    out = dict(copy.deepcopy(SPLIT_DEFAULTS), sizes={})
    out.update(copy.deepcopy(split))
    return _ordered(out)


def split_section_md(split):
    """The section a fill of the split rule appends to TASK.md, rendered from the filled rule (filled(split)): it opens
    with one newline (the separator: a TASK.md ending in a newline gets one blank line before the heading), then the
    heading 「## 切分规则（`data.split`）」, the line saying the data module filled it in, and the rule as JSON, one
    top-level key per line. The same rule renders the same bytes. Write the new TASK.md with fill_task_md (bytes)."""
    rule = filled(split)
    body = ",\n".join("  %s: %s" % (json.dumps(k), json.dumps(v, ensure_ascii=False)) for k, v in rule.items())
    return ("\n## 切分规则（`data.split`）\n\n由 rsi-data 按数据填写，理由见那次派出的 record。\n\n"
            "```json\n{\n%s\n}\n```\n" % body)


def fill_task_md(current_md, split):
    """The TASK.md of a fill of the split rule, as bytes: the current TASK.md's bytes `current_md` plus
    split_section_md(split) in UTF-8. A fill registers only with exactly these bytes (rsi.workspace compares them
    byte for byte), so a writer must build the file with this function from the bytes it read (open(..., "rb")),
    never from a text-mode read, which may turn CRLF into LF."""
    return current_md + split_section_md(split).encode("utf-8")


def _leaves(split):
    """(dotted path, value) for every value the rule writes: per set under ranges, fractions and sizes."""
    for k, v in split.items():
        if k in PER_SET and isinstance(v, dict):
            for s, sv in v.items():
                yield "%s.%s" % (k, s), sv
        else:
            yield k, v


def _ordered(split):
    def order(d, first):
        return dict((k, d[k]) for k in sorted(d, key=lambda k: (first.index(k) if k in first else len(first), k)))
    out = order(split, _KEY_ORDER)
    for k in PER_SET:
        if isinstance(out.get(k), dict):
            out[k] = order(out[k], _SET_ORDER)
    return out


def merge_split(given, chosen):
    """(merged rule, errors): `chosen` (the data module's rule) under `given` (the task's data.split). Every value
    `given` writes stays as it is, ranges, fractions and sizes set by set; a value `chosen` writes differently is a
    conflict, an error naming its path (e.g. $.data.split.sizes.trainset)."""
    merged = copy.deepcopy(chosen)
    errors = []
    for path, v in _leaves(given):
        found, c = value_at(chosen, path)
        if found and c != v:
            errors.append("%s.%s: the task writes %s, the rule %s (a value the task writes stays)"
                          % (WHERE, path, _show(v), _show(c)))
        head, _, tail = path.partition(".")
        if tail:
            if not isinstance(merged.get(head), dict):
                merged[head] = {}
            merged[head][tail] = copy.deepcopy(v)
        else:
            merged[head] = copy.deepcopy(v)
    return _ordered(merged), errors


def _show(v):
    return "null" if v is None else ("true" if v is True else ("false" if v is False else str(v)))


def check_data(task):
    """Errors in a (schema-valid) task's data section: each data.sources path is absolute and can be written as a
    deny rule (no UNSAFE_PATH_CHARS, no leading !); the label_counts of data.rows.expect name the task's labels; what
    data.split writes agrees (split_errors with partial)."""
    data = task["data"]
    errors = []
    for i, s in enumerate(data["sources"]):
        if not isinstance(s, dict):
            continue
        at, p = "$.data.sources[%d].path" % i, s["path"]
        if not os.path.isabs(p):
            errors.append("%s: %r must be an absolute path on this machine" % (at, p))
        if any(c in p for c in UNSAFE_PATH_CHARS) or p.startswith("!"):
            errors.append("%s: %r has one of %s or a leading !, so no deny rule can keep the modules out of it; "
                          "move or link the data to a plain path" % (at, p, " ".join(UNSAFE_PATH_CHARS)))
    labels = task["decision"]["labels"]
    for k in data.get("rows", {}).get("expect", {}).get("label_counts", {}):
        if k not in labels:
            errors.append("$.data.rows.expect.label_counts: %r is not a label of the task (labels: %s)"
                          % (k, ", ".join(labels)))
    return errors + split_errors(task_split(task), partial=True)


def _rest(task):
    """The task without its version and data.split: what a fill leaves unchanged."""
    t = copy.deepcopy(task)
    t.pop("version", None)
    if isinstance(t.get("data"), dict):
        t["data"].pop("split", None)
    return t


def _changed(a, b):
    """The top-level keys (data.<key> inside data) where `a` and `b` differ."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, _MISSING), b.get(k, _MISSING)
        if x == y:
            continue
        if k == "data" and isinstance(x, dict) and isinstance(y, dict):
            out += ["data.%s" % s for s in _changed(x, y)]
        else:
            out.append(k)
    return out


def changes_outside_split(old, new):
    """What differs between two task definitions besides version and data.split: the top-level keys, data.<key> inside
    data ([] when only those two differ)."""
    return _changed(_rest(old), _rest(new))


def fill_errors(old, new):
    """Why task definition `new` is not a fill of the split rule of `old` ([] when it is one), comparing the two only:
    the same task_id, version one higher, nothing changed but data.split, the old rule incomplete and the new one
    complete and valid, every value the old rule writes kept. (A fill must also come before any evalset or testset
    is registered, and its TASK.md must be fill_task_md(the current TASK.md, the new rule):
    rsi.workspace.Workspace.split_fill_errors.)"""
    errors = []
    if new.get("version") != old.get("version", 0) + 1:
        errors.append("version is %s, expected %s (the current version + 1)"
                      % (new.get("version"), old.get("version", 0) + 1))
    changed = changes_outside_split(old, new)
    if changed:
        errors.append("it changes more than data.split (also %s)" % ", ".join(changed))
    a, b = task_split(old), task_split(new)
    if is_complete(a):
        errors.append("the current data.split is complete already; a fill completes an incomplete rule")
    errors += split_errors(b)
    for path, v in _leaves(a):
        found, nv = value_at(b, path)
        if not found or nv != v:
            errors.append("%s.%s: the current version writes %s; a fill keeps it (the new version has %s)"
                          % (WHERE, path, _show(v), _show(nv) if found else "none"))
    return errors


def corpus_errors(rows, task):
    """Why `rows`, all the labelled rows of a corpus, do not fit the task's data section ([] when they do). With a
    complete rule (is_complete), every row has the rule's key (by time or group), and by time its value is a number (not
    a bool); with data.rows.expect, the row count and the count of each hard label it names (rsi.metrics.hard_label)
    equal it. One error per kind of problem, with its row count (and the first id of a bad row)."""
    rows = [r for r in rows if isinstance(r, dict)]  # a row that is not an object is reported by the row checks
    errors = []
    split = task_split(task)
    if is_complete(split) and split["by"] != "random":
        key, missing, bad = split["key"], [], []
        for r in rows:
            found, v = value_at(r, key)
            if not found:
                missing.append(r.get("id"))
            elif split["by"] == "time" and not is_number(v):
                bad.append(r.get("id"))
        if missing:
            errors.append("%d rows have no %s (the key of the split rule %s), e.g. %s"
                          % (len(missing), key, WHERE, missing[0]))
        if bad:
            errors.append("%d rows have a %s that is not a number (a split by time takes a number; text, a bool and "
                          "null are not), e.g. %s" % (len(bad), key, bad[0]))
    expect = (_data(task).get("rows") or {}).get("expect") or {}
    if "rows" in expect and expect["rows"] != len(rows):
        errors.append("$.data.rows.expect.rows is %d, the corpus has %d rows" % (expect["rows"], len(rows)))
    want = expect.get("label_counts") or {}
    if want:
        labels = task["decision"]["labels"]
        got = {}
        for r in rows:
            if isinstance(r.get("target"), dict):
                h = hard_label(r["target"], labels)
                got[h] = got.get(h, 0) + 1
        named = [l for l in labels if l in want] + sorted(l for l in want if l not in labels)
        off = ["%s %d rows, expected %d" % (l, got.get(l, 0), want[l]) for l in named if got.get(l, 0) != want[l]]
        if off:
            errors.append("$.data.rows.expect.label_counts: the hard labels of the corpus differ (%s)" % "; ".join(off))
    return errors
