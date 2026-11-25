"""The allocation of a split rule (spec §6), and the draws of `grow` and `refresh`.

A complete rule (rsi.datasplit.filled) first gives each corpus row the set whose range holds it: by time, the closed
`ranges` of the numeric key; by group and random, a number in [0, 1) fixed by the seed and the key's value (group) or
the row's unit (random: its cluster, or its state_id with unit state_id), falling into the `fractions` taken in the
order testset, evalset, valset, trainset. The sets are then drawn in that order: a set's free rows are the rows of its
range whose id, input or cluster (with unit state_id also state_id) no set drawn before holds; the trainset's free
rows are the pool. A set takes `sizes` of its free rows (all when no size is given), stratified (as the Elliptic
example's former prepare_sets.py did): by (key value, hard label) for by time and by hard label otherwise, every
stratum its largest-remainder share, a sample with random.Random("<kind>-<seed>") from the stratum's rows sorted by
id. So the same rows and rule draw the same sets.
"""
import hashlib
import json
import random

from rsi import datasplit as DS
from rsi import rows as R
from rsi.metrics import hard_label

from . import DsetError

SET_KINDS = DS.SET_KINDS  # testset, evalset, valset, trainset: the order the sets are drawn in
# The groups the report compares state_ids between, in this order; the pool and the trainset are not compared (the
# trainset is drawn from the pool).
REPORTED = ("testset", "evalset", "valset", "pool", "trainset")


def _marks(r):
    """What a row holds that a later set must not share: its id, input fingerprint, cluster and state_id."""
    meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
    state = R.state_of(r)
    return {"id": r.get("id"), "fp": None if state is None else R.input_fingerprint(state),
            "cluster": meta.get("cluster") if isinstance(meta.get("cluster"), str) else None,
            "state_id": R.group_of(r)}


def _items(rows, task, rule):
    """The rows with their marks, hard label and value at the rule's key (None without a key)."""
    labels, key = task["decision"]["labels"], rule.get("key")
    out = []
    for r in rows:
        it = _marks(r)
        it.update(row=r, label=hard_label(r["target"], labels), key=DS.value_at(r, key)[1] if key else None)
        out.append(it)
    return out


class _Taken(object):
    """The ids, inputs, clusters and (with unit state_id) state_ids of the rows drawn so far."""

    def __init__(self, by_state_id):
        self.by_state_id = by_state_id
        self.held = {"id": set(), "fp": set(), "cluster": set(), "state_id": set()}

    def add(self, items):
        for it in items:
            for k, s in self.held.items():
                if it[k] is not None:
                    s.add(it[k])

    def holds(self, it):
        return any(it[k] in self.held[k] for k in ("id", "fp", "cluster")) or (
            self.by_state_id and it["state_id"] in self.held["state_id"])


def _draw(seed, value):
    """A number in [0, 1) fixed by the seed and the value: the first 8 bytes of sha256("<seed>\\0" + its JSON)
    over 2^64."""
    h = hashlib.sha256(("%s\0%s" % (seed, json.dumps(value, sort_keys=True))).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2 ** 64)


def _where(it, rule):
    """The set whose range holds the row; None when no range does (an unused row)."""
    if rule["by"] == "time":
        for kind in SET_KINDS:
            lo, hi = rule["ranges"][kind]
            if lo <= it["key"] <= hi:
                return kind
        return None
    value = it["key"] if rule["by"] == "group" else it["state_id" if rule["unit"] == "state_id" else "cluster"]
    u, edge = _draw(rule["seed"], value), 0.0
    for kind in SET_KINDS:
        edge += rule["fractions"][kind]
        if u < edge:
            return kind
    return None


def _sort_value(v):
    """Numbers first, by value; any other value after them, by its JSON."""
    return (0, v, "") if DS.is_number(v) else (1, 0, json.dumps(v, sort_keys=True))


def _order(items, rule):
    """Rows in the order they are written: by the key's value, then id, for by time; by id otherwise."""
    if rule.get("by") == "time":
        return sorted(items, key=lambda it: (_sort_value(it["key"]), it["id"]))
    return sorted(items, key=lambda it: it["id"])


def _sample(items, n, rule, rng):
    """n of `items` (all of them when n is None or not below their number); see the module's docstring."""
    if n is None or n >= len(items):
        return list(items)

    def by_id(it):
        return it["id"]
    if not rule.get("stratify", True):
        return rng.sample(sorted(items, key=by_id), n)
    time = rule.get("by") == "time"
    strata = {}
    for it in items:
        strata.setdefault((_sort_value(it["key"]) if time else (), it["label"]), []).append(it)
    keys, total = sorted(strata), len(items)
    quota = dict((k, n * len(strata[k]) // total) for k in keys)
    for k in sorted(keys, key=lambda k: (-(n * len(strata[k]) % total), k))[:n - sum(quota.values())]:
        quota[k] += 1
    out = []
    for k in keys:
        out += rng.sample(sorted(strata[k], key=by_id), quota[k])
    return out


def _span(items, rule):
    """[lo, hi] of the key's values over the rows when they are all numbers or all strings; None otherwise (by random,
    no rows)."""
    vals = [it["key"] for it in items]
    if not rule.get("key") or not vals:
        return None
    if all(DS.is_number(v) for v in vals) or all(isinstance(v, str) for v in vals):
        return [min(vals), max(vals)]
    return None


def _entry(items, labels, rule, range_rows, left_out):
    """A set's line of the report: rows, rows per hard label, the key's span, the rows of its range and those left out
    because a set drawn before holds their id, input, cluster (or state_id)."""
    counts = dict((label, 0) for label in labels)
    for it in items:
        counts[it["label"]] += 1
    return {"rows": len(items), "label_counts": counts, "key_span": _span(items, rule), "range_rows": range_rows,
            "left_out": left_out}


def _range_text(kind, rule):
    if rule["by"] == "time":
        return "%s %s-%s" % (rule["key"], rule["ranges"][kind][0], rule["ranges"][kind][1])
    return "fraction %s" % rule["fractions"][kind]


def _allocate(items, task, rule):
    """({kind: drawn items}, pool items, {kind: items in range}, report) of the rule on the corpus items."""
    labels = task["decision"]["labels"]
    in_range, unused = dict((k, []) for k in SET_KINDS), 0
    for it in items:
        kind = _where(it, rule)
        if kind is None:
            unused += 1
        else:
            in_range[kind].append(it)
    taken = _Taken(rule["unit"] == "state_id")
    drawn, entries, pool = {}, {}, []
    for kind in SET_KINDS:
        free = [it for it in in_range[kind] if not taken.holds(it)]
        left_out = len(in_range[kind]) - len(free)
        if kind == "trainset":
            pool = _order(free, rule)
        n = rule["sizes"].get(kind)
        held = "input, cluster or state_id" if taken.by_state_id else "input or cluster"
        if not free:
            raise DsetError("%s: its range (%s) has no free rows (%d rows in it, %d of them left out because a set drawn "
                            "before holds their %s), and dset writes no empty %s; change the rule"
                            % (kind, _range_text(kind, rule), len(in_range[kind]), left_out, held,
                               "trainset or pool" if kind == "trainset" else "set"))
        if n is not None and n > len(free):
            raise DsetError("%s: sizes.%s is %d but only %d rows are free for it: %d rows in its range (%s), %d of "
                            "them left out because a set drawn before holds their %s; lower the size or change the rule"
                            % (kind, kind, n, len(free), len(in_range[kind]), _range_text(kind, rule), left_out, held))
        drawn[kind] = _order(_sample(free, n, rule, random.Random("%s-%s" % (kind, rule["seed"]))), rule)
        taken.add(drawn[kind])
        entries[kind] = _entry(drawn[kind], labels, rule, len(in_range[kind]), left_out)
    groups = dict((k, pool if k == "pool" else drawn[k]) for k in REPORTED)
    _check_disjoint(groups, rule)
    rows = dict((k, [it["row"] for it in v]) for k, v in groups.items())
    shared = {}
    for i, a in enumerate(REPORTED):
        for b in REPORTED[i + 1:]:
            if (a, b) != ("pool", "trainset"):
                shared["%s/%s" % (a, b)] = R.shared_state_ids(rows[a], rows[b])
    pool_entry = _entry(pool, labels, rule, entries["trainset"]["range_rows"], entries["trainset"]["left_out"])
    report = {"rule": rule, "sets": entries, "pool": pool_entry, "unused_rows": unused, "shared_state_ids": shared}
    return drawn, pool, in_range, report


def _check_disjoint(groups, rule):
    """The testset, evalset, valset and pool share no id, input or cluster (nor state_id with unit state_id); the
    drawing ensures it, this makes sure."""
    marks = ("id", "fp", "cluster") + (("state_id",) if rule["unit"] == "state_id" else ())
    names = ("testset", "evalset", "valset", "pool")
    held = dict((k, dict((m, set(it[m] for it in groups[k]) - {None}) for m in marks)) for k in names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [m for m in marks if held[a][m] & held[b][m]]
            if both:
                raise DsetError("%s and %s share rows (%s): the allocation is wrong" % (a, b, ", ".join(both)))


def allocate(rows, task, rule):
    """The allocation of complete rule `rule` (filled) on the corpus rows: {"sets": {kind: rows}, "pool": rows,
    "report": {rule, sets: {kind: {rows, label_counts, key_span, range_rows, left_out}}, pool, unused_rows,
    shared_state_ids: {"<a>/<b>": n}}}; rows are the corpus rows as they are, in written order. Raises DsetError
    when a set's size exceeds its free rows, or a set or the pool would be empty."""
    drawn, pool, _, report = _allocate(_items(rows, task, rule), task, rule)
    return {"sets": dict((k, [it["row"] for it in drawn[k]]) for k in SET_KINDS),
            "pool": [it["row"] for it in pool], "report": report}


def grow(pool_rows, current_rows, n, task, rule, with_rows=()):
    """(n rows of the pool whose id and input are in neither the current trainset nor `with_rows`, the number of such
    rows): drawn as a set is, with random.Random("grow-<seed>-<rows of the current trainset>"), in written order. So
    grow does not bring back an input `check --write` cleaned out."""
    held = [_marks(r) for r in list(current_rows) + list(with_rows)]
    ids, fps = set(m["id"] for m in held), set(m["fp"] for m in held)
    free = [it for it in _items(pool_rows, task, rule) if it["id"] not in ids and it["fp"] not in fps]
    if n > len(free):
        raise DsetError("--add %d: the pool holds only %d rows whose id and input the trainset and --with rows do not "
                        "have" % (n, len(free)))
    picked = _sample(free, n, rule, random.Random("grow-%s-%d" % (rule["seed"], len(current_rows))))
    return [it["row"] for it in _order(picked, rule)], len(free)


def refresh(rows, task, rule, exclude):
    """(rows, report entry) of a new evalset: sizes.evalset rows (all when no size) of the evalset's range that hold no
    id, input or cluster (with unit state_id also state_id) of any set or the pool of the rule's allocation, nor of the
    rows in `exclude` (row lists: earlier evalsets' inputs, the current trainset's and valset's rows), drawn with
    random.Random("evalset-<seed>-refresh-<number of exclude lists>"). So the new evalset shares nothing with them."""
    drawn, pool, in_range, _ = _allocate(_items(rows, task, rule), task, rule)
    taken = _Taken(rule["unit"] == "state_id")
    for kind in SET_KINDS:
        taken.add(drawn[kind])
    taken.add(pool)
    for rs in exclude:
        taken.add([_marks(r) for r in rs])
    candidates = in_range["evalset"]
    free = [it for it in candidates if not taken.holds(it)]
    n = rule["sizes"].get("evalset")
    if n is None and not free:
        raise DsetError("evalset: no row of its range (%s) is unused: sizes.evalset is not set, so the first evalset "
                        "took its whole range, and dset writes no empty evalset; sizes.evalset must leave rows of the "
                        "range for refreshes" % _range_text("evalset", rule))
    if n is not None and n > len(free):
        raise DsetError("evalset: sizes.evalset is %d but only %d rows of its range (%s) are unused: %d in range, %d "
                        "held by a set, the pool or --exclude" % (n, len(free), _range_text("evalset", rule),
                                                                  len(candidates), len(candidates) - len(free)))
    picked = _order(_sample(free, n, rule, random.Random("evalset-%s-refresh-%d" % (rule["seed"], len(exclude)))), rule)
    entry = _entry(picked, task["decision"]["labels"], rule, len(candidates), len(candidates) - len(free))
    return [it["row"] for it in picked], entry
