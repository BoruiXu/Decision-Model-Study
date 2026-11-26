"""Prompt variants: checkable rewrites of a decision row's prompt (standard library only).

A prompt variant changes how a row's input is worded and laid out for the decision model, never what it says. A spec
(docs/interfaces/datasets.md, "Prompt-variant rows"; the item schema of the trainset's `prompt_variants`) is
{"id": ^[a-z0-9][a-z0-9-]*$, "instructions": str | null, "criteria": object | list | null,
 "rename": {<object-key path>: <new key>} | null, "tables": [<list path>] | null}:
- instructions replaces the question's instructions (the task's decision.instructions) with a paraphrase;
- criteria replaces the question's criteria (the label meanings, in the question's form: {"true", "false"} of a noul
  question, {label: meaning} of a choice question, [meaning of "0", ...] of a score question); the labels never change;
- rename gives keys of the state's objects new names. A path names one key in the seed's naming: keys joined by dots,
  "[]" for every item of a list ("summary.txs_sent", "transactions[].amount");
- tables turns each named list of objects with the same keys into {"columns": [key, ...], "rows": [[value, ...], ...]}
  (columns in the first item's key order, after the renames inside the items).
Paths always name places in the seed's state, so the transforms commute: apply_state walks the seed's state,
invert_state the variant's state back to the seed's. A path the state lacks changes nothing in it.

check_spec refuses a spec whose rename or table would touch a path of the task's input.verbatim (a row's own request
text stays where it is, as it is) or a key the task's input format needs (messages[].role and messages[].content;
the fields), and a spec that changes nothing. Variant rows live only in a trainset, next to their seed;
check_variant_rows checks each against its seed (rsi.artifacts runs it on every trainset).
"""
import copy
import re

from . import rows as R
from .rules import verbatim_paths
from .schema import load_schema, validate

SEPARATOR = "~"  # a variant row's id: <seed id>~<spec id>
_SEGMENT = re.compile(r"^([^.\[\]]+)((?:\[\])*)$")


def spec_schema():
    """The schema of one spec: the item schema of the trainset manifest's prompt_variants."""
    return load_schema("trainset")["properties"]["prompt_variants"]["items"]


def parse_path(path):
    """A state path as a tuple of parts, None for every item of a list ("a.b[].c" -> ("a", "b", None, "c")); None
    when `path` is not one (keys joined by dots, each key followed by any number of "[]")."""
    if not isinstance(path, str) or not path:
        return None
    parts = []
    for seg in path.split("."):
        m = _SEGMENT.match(seg)
        if not m:
            return None
        parts.append(m.group(1))
        parts.extend([None] * (len(m.group(2)) // 2))
    return tuple(parts)


def path_text(parts):
    out = ""
    for p in parts:
        out += "[]" if p is None else ("." if out else "") + p
    return out or "the state"


def _overlap(a, b):
    """Whether one of two parsed paths is the other or lies under it."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _format_paths(task):
    """The paths the task's input format needs (rsi.rows.check_state): a rename or table must leave them in place."""
    inp = task["input"]
    if inp["format"] == "messages":
        return [("messages",), ("messages", None, "role"), ("messages", None, "content")]
    if inp["format"] == "fields":
        return [("fields",)] + [("fields", f) for f in inp.get("fields", [])]
    return []


def question(spec, task):
    """The question of a variant row: rsi.rows.question_of(task) with the spec's instructions and criteria (those
    that are not null)."""
    q = R.question_of(task)
    if spec.get("instructions") is not None:
        q["instructions"] = spec["instructions"]
    if spec.get("criteria") is not None:
        q["criteria"] = copy.deepcopy(spec["criteria"])
    return q


def check_spec(spec, task):
    """Why `spec` is no valid prompt-variant spec for `task` ([] when it is one)."""
    errors = validate(spec, spec_schema())
    if errors:
        return errors
    if spec["instructions"] is not None and not spec["instructions"].strip():
        errors.append("instructions: empty (null keeps the task's)")
    want, crit = R.question_of(task)["criteria"], spec["criteria"]
    if crit is not None:
        if isinstance(want, dict) and not (isinstance(crit, dict) and set(crit) == set(want)):
            errors.append("criteria: expected an object with exactly the keys %s (the question's criteria; the labels "
                          "never change)" % ", ".join(want))
        elif isinstance(want, list) and not (isinstance(crit, list) and len(crit) == len(want)):
            errors.append("criteria: expected a list of %d meanings (the question's criteria, one per label)"
                          % len(want))
        elif any(not isinstance(v, str) or not v.strip() for v in (crit.values() if isinstance(crit, dict) else crit)):
            errors.append("criteria: every meaning is a non-empty string")
    renamed, siblings = [], {}
    for path, new in sorted((spec["rename"] or {}).items()):
        parts = parse_path(path)
        if parts is None:
            errors.append("rename: %r is not a state path (keys joined by dots, [] for the items of a list)" % path)
        elif parts[-1] is None:
            errors.append("rename: %s ends in [] and names no object key" % path)
        else:
            renamed.append((path, parts))
            siblings.setdefault(parts[:-1], []).append(new)
        if not new.strip():
            errors.append("rename: the new name of %s is empty" % path)
    for parent, names in sorted(siblings.items(), key=lambda kv: path_text(kv[0])):
        if len(set(names)) != len(names):
            errors.append("rename: the new names of the keys under %s are not unique (%s)"
                          % (path_text(parent), ", ".join(sorted(names))))
    tabled = []
    for path in spec["tables"] or []:
        parts = parse_path(path)
        if parts is None:
            errors.append("tables: %r is not a state path (keys joined by dots, [] for the items of a list)" % path)
        else:
            tabled.append((path, parts))
    if len(set(spec["tables"] or [])) != len(spec["tables"] or []):
        errors.append("tables: a path is named twice")
    for what, paths in (("rename", renamed), ("tables", tabled)):
        for path, parts in paths:
            for v in verbatim_paths(task):
                vp = parse_path(v)
                if vp is not None and _overlap(parts, vp):
                    errors.append("%s: %s touches the input.verbatim path %s: a row's own request text stays where it "
                                  "is, as it is" % (what, path, v))
            moved = [path_text(fp) for fp in _format_paths(task) if fp[:len(parts)] == parts]
            if moved:
                errors.append("%s: %s would move %s, which the task's input format (%s) needs"
                              % (what, path, ", ".join(moved), task["input"]["format"]))
    if not errors and question(spec, task) == R.question_of(task) and not spec["rename"] and not spec["tables"]:
        errors.append("the spec changes nothing: give a paraphrase (instructions, criteria), renames or tables")
    return errors


# ---- the transforms --------------------------------------------------------------------------------------------------

def _maps(spec):
    """({parent: {old key: new key}}, {parent: {new key: old key}}, {table paths}) of a spec, paths parsed."""
    fwd, back = {}, {}
    for path, new in (spec.get("rename") or {}).items():
        parts = parse_path(path)
        fwd.setdefault(parts[:-1], {})[parts[-1]] = new
        back.setdefault(parts[:-1], {})[new] = parts[-1]
    return fwd, back, set(parse_path(p) for p in spec.get("tables") or [])


def _table(items, at):
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        raise ValueError("%s: a table needs a list of objects" % path_text(at))
    columns = list(items[0]) if items else []
    for i, x in enumerate(items):
        if set(x) != set(columns):
            raise ValueError("%s: item %d has the keys %s and item 0 %s: a table needs items with the same keys"
                             % (path_text(at), i, sorted(x), sorted(columns)))
    return {"columns": columns, "rows": [[x[c] for c in columns] for x in items]}


def _untable(value, at):
    ok = isinstance(value, dict) and set(value) == {"columns", "rows"}
    columns, rows = (value["columns"], value["rows"]) if ok else (None, None)
    ok = ok and isinstance(columns, list) and all(isinstance(c, str) for c in columns) \
        and len(set(columns)) == len(columns) and isinstance(rows, list) \
        and all(isinstance(r, list) and len(r) == len(columns) for r in rows)
    if not ok:
        raise ValueError("%s: expected a table {\"columns\": [key, ...], \"rows\": [[value, ...], ...]}"
                         % path_text(at))
    return [dict(zip(columns, r)) for r in rows]


def _apply(value, at, fwd, tables):
    if isinstance(value, dict):
        names, out = fwd.get(at, {}), {}
        for k, v in value.items():
            new = names.get(k, k)
            if new in out:
                raise ValueError("%s: the renames give two keys named %r" % (path_text(at), new))
            out[new] = _apply(v, at + (k,), fwd, tables)
        value = out
    elif isinstance(value, list):
        value = [_apply(v, at + (None,), fwd, tables) for v in value]
    return _table(value, at) if at in tables else value


def _invert(value, at, fwd, back, tables):
    if at in tables:
        value = _untable(value, at)
    if isinstance(value, dict):
        names, olds, out = back.get(at, {}), fwd.get(at, {}), {}
        for k, v in value.items():
            if k in names:
                old = names[k]
            elif k in olds:
                raise ValueError("%s: the key %r is renamed to %r by the spec" % (path_text(at), k, olds[k]))
            else:
                old = k
            if old in out:
                raise ValueError("%s: two keys go back to %r" % (path_text(at), old))
            out[old] = _invert(v, at + (old,), fwd, back, tables)
        return out
    if isinstance(value, list):
        return [_invert(v, at + (None,), fwd, back, tables) for v in value]
    return value


def apply_state(state, spec):
    """The state of a variant of a row whose state is `state` (a new object; ValueError when a table's list is not a
    list of objects with the same keys, or two keys would get one name)."""
    fwd, _, tables = _maps(spec)
    return _apply(state, (), fwd, tables)


def invert_state(state, spec):
    """The seed's state a variant's `state` is the rewrite of (ValueError when `state` is no rewrite under `spec`)."""
    fwd, back, tables = _maps(spec)
    return _invert(state, (), fwd, back, tables)


def apply_spec(row, spec, task):
    """The variant of full row `row` (no prompt variant itself) under `spec`: the id "<seed id>~<spec id>", the
    spec's question, the state rewritten (apply_state), meta.extra with variant (the spec's id) and variant_of (the
    seed's id); everything else is the seed's. ValueError as apply_state, or when `row` is a variant."""
    if R.is_variant(row):
        raise ValueError("row %s is a prompt variant itself: a variant's seed is a row that is not one" % row.get("id"))
    out = copy.deepcopy(row)
    out["id"] = "%s%s%s" % (row["id"], SEPARATOR, spec["id"])
    out["question"] = question(spec, task)
    out["state"] = apply_state(R.state_of(row), spec)
    meta = out.setdefault("meta", {})
    meta["extra"] = dict(meta.get("extra") or {}, variant=spec["id"], variant_of=row["id"])
    return out


# ---- the checks of a trainset's variant rows -------------------------------------------------------------------------

def _at(value, parts, here=()):
    """[(concrete path, value)] of the values at the parsed path `parts` ("[]": every item)."""
    if not parts:
        return [(here, value)]
    head, rest = parts[0], parts[1:]
    if head is None:
        if not isinstance(value, list):
            return []
        return [x for i, item in enumerate(value) for x in _at(item, rest, here + (i,))]
    if isinstance(value, dict) and head in value:
        return _at(value[head], rest, here + (head,))
    return []


def _meta(row, key):
    meta = row.get("meta")
    return meta.get(key) if isinstance(meta, dict) else None


def _variant_errors(r, by_id, specs, listed, task):
    extra = r["meta"]["extra"]
    sid, seed_id = extra.get("variant"), extra.get("variant_of")
    seed = by_id.get(seed_id) if isinstance(seed_id, str) else None
    errs = []
    if seed is None:
        errs.append("meta.extra.variant_of: its seed %r is not a row of this trainset (a variant lives next to its "
                    "seed)" % (seed_id,))
    elif R.is_variant(seed):
        errs.append("meta.extra.variant_of: its seed %s is a prompt variant itself; a variant's seed is a row that is "
                    "not one" % seed_id)
    spec = specs.get(sid) if isinstance(sid, str) else None
    if not isinstance(sid, str) or sid not in listed:
        errs.append("meta.extra.variant: %r is not the id of a spec in the manifest's prompt_variants" % (sid,))
    elif spec is None:
        errs.append("meta.extra.variant: its spec %s is not valid (see the prompt_variants errors)" % sid)
    if errs:
        return errs
    errs += R.check_question(r.get("question"), task, spec)
    state, seed_state = R.state_of(r), R.state_of(seed)
    for v in verbatim_paths(task):
        parts = parse_path(v)
        if parts is not None and _at(state, parts) != _at(seed_state, parts):
            errs.append("%s: the text at this input.verbatim path differs from its seed's: a variant never changes a "
                        "row's own request" % v)
    try:
        why = None if invert_state(state, spec) == seed_state and apply_state(seed_state, spec) == state else \
            "undoing the spec's renames and tables does not give the seed's state"
    except ValueError as e:
        why = str(e)
    if why:
        errs.append("state is not a rewrite of its seed %s under spec %s: %s" % (seed_id, sid, why))
    for what, mine, theirs in (("target", r.get("target"), seed.get("target")),
                               ("meta.state_id", _meta(r, "state_id"), _meta(seed, "state_id")),
                               ("meta.cluster", _meta(r, "cluster"), _meta(seed, "cluster")),
                               ("meta.label_source", _meta(r, "label_source"), _meta(seed, "label_source"))):
        if mine != theirs:
            errs.append("%s differs from its seed %s's: a variant keeps its seed's label and groups" % (what, seed_id))
    return errs


def check_variant_rows(rows, specs, task):
    """The errors of a trainset's prompt variants: each spec of `specs` (the manifest's prompt_variants) valid
    (check_spec) and its id used once; each row with meta.extra.variant a rewrite of its seed: the seed
    (meta.extra.variant_of) a row of `rows` that is no variant, its spec listed, its question the spec's (question),
    the text at every input.verbatim path the seed's, its state the seed's under the spec (invert_state gives the
    seed's state, apply_state the row's), and its target, state_id, cluster and label_source the seed's. Every other
    row is checked elsewhere (rsi.rows.check_full_rows)."""
    errors, valid, listed = [], {}, set()
    for i, s in enumerate(specs or []):
        sid = s.get("id") if isinstance(s, dict) else None
        errs = check_spec(s, task)
        errors += ["prompt_variants[%d] (%s): %s" % (i, sid, e) for e in errs]
        if isinstance(sid, str):
            if sid in listed:
                errors.append("prompt_variants[%d]: the id %s is used twice" % (i, sid))
            elif not errs:
                valid[sid] = s
            listed.add(sid)
    by_id = {}
    for r in rows:
        if isinstance(r, dict) and isinstance(r.get("id"), str):
            by_id.setdefault(r["id"], r)
    for i, r in enumerate(rows):
        if isinstance(r, dict) and R.is_variant(r):
            errors += ["row %d (id=%s): %s" % (i, r.get("id"), e)
                       for e in _variant_errors(r, by_id, valid, listed, task)]
    return errors
