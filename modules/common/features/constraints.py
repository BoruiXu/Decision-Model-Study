"""Declarative state constraints (a task's features.constraints): check and repair, standard library only.

Each item is one of:
  {"nonneg": [path, ...]}                  every number at each path is >= 0;
  {"integer": [path, ...]}                 every number at each path is a whole number;
  {"sum": path, "of": [path, ...]}         the number at `sum` equals the sum of the numbers at `of`;
  {"order": [path, path, ...]}             the numbers are non-decreasing in the order given;
  {"zero_if_zero": path, "then": [path]}   when the number at `zero_if_zero` is 0, so is each number at `then`.

A path joins keys with dots ("summary.total"); a key followed by "[]" stands for every element of the list there
("transactions[].fee"); a part of digits picks one list element, as the edit view writes it ("transactions.0.fee").
In `order` and `zero_if_zero` every "[]" of the item is bound: the item holds in each element (in each combination when
its paths go through different lists), a path without "[]" being the same in each. Two paths go through the same list
when their text up to the "[]" is the same ("t[]" in "t[].lo" and "t[].hi"). In `sum` the "[]"s of the `sum` path are
bound the same way, and an `of` path's other "[]"s are summed over: {"sum": "total", "of": ["t[].v"]} is total = the
sum of every t[].v, and {"sum": "t[].total", "of": ["t[].a", "t[].b"]} holds in each element of t.

Every leaf a constraint names must be a finite number (a bool is not one): a missing leaf or one that is not a number
is a violation that repair leaves as it is. Repair works on a deep copy, in the order of dgen's schema.repair: every
nonneg item (a negative number becomes 0), every integer item (a fraction is rounded to the nearest whole number,
halves away from zero), then the relations (sum, order, zero_if_zero) in their order, twice, so a relation sees what
a later one changed: a sum's target becomes the sum (an int when every part is one), an order's numbers are sorted
into its places, a `then` number becomes 0. A sum matches within 1e-9 (relative, or absolute near 0).
"""
import copy
import math

from . import FeatureError

KINDS = ("nonneg", "integer", "sum", "order", "zero_if_zero")
COMPANION = {"sum": "of", "zero_if_zero": "then"}  # the key a relation needs beside its kind
RELATIONS = ("sum", "order", "zero_if_zero")
SUM_TOLERANCE = 1e-9


def _is_number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def _num_text(v):
    return "null" if v is None else repr(v)


class _Path(object):
    """A parsed path: `steps` are ("key", name) and ("each", prefix) in order, `prefix` being the path's text up to and
    including that "[]" (what binds two paths' "[]" to the same list)."""

    def __init__(self, text, where):
        if not isinstance(text, str) or not text:
            raise FeatureError("%s: a path is a non-empty string (got %r)" % (where, text))
        self.text, self.steps = text, []
        done = []
        for part in text.split("."):
            key, each = part, 0
            while key.endswith("[]"):
                key, each = key[:-2], each + 1
            if not key or "[" in key or "]" in key:
                raise FeatureError("%s: %r is not a path (keys joined by dots, a key followed by [] for every "
                                   "element of its list)" % (where, text))
            done.append(key)
            self.steps.append(("key", key))
            for _ in range(each):
                done[-1] += "[]"
                self.steps.append(("each", ".".join(done)))

    def prefixes(self):
        return [p for kind, p in self.steps if kind == "each"]


def _step(node, key):
    """node[key] for a dict key or a list index as the edit view writes it; KeyError when there is none."""
    if isinstance(node, dict) and key in node:
        return key
    if isinstance(node, list) and key.isascii() and key.isdigit() and str(int(key)) == key and int(key) < len(node):
        return int(key)
    raise KeyError(key)


def _leaves(state, path, binding):
    """[(concrete path, container, key)] of `path` under `binding` (prefix -> index), an unbound "[]" over every
    element of its list; a leaf that is not there has container None and key "missing" or "not a list"."""
    out = []

    def walk(node, i, concrete):
        kind, arg = path.steps[i]
        if kind == "key":
            try:
                k = _step(node, arg)
            except KeyError:
                out.append((".".join(concrete + [arg]), None, "missing"))
                return
            keys = [(k, arg)]
        elif not isinstance(node, list):
            out.append((".".join(concrete), None, "not a list"))
            return
        else:
            keys = [(j, str(j)) for j in ([binding[arg]] if arg in binding else range(len(node)))]
        for k, part in keys:
            if i + 1 == len(path.steps):
                out.append((".".join(concrete + [part]), node, k))
            else:
                walk(node[k], i + 1, concrete + [part])

    walk(state, 0, [])
    return out


def _resolve(state, steps, binding):
    """The node at `steps`, each "[]" among them bound by `binding`; None when it is not there."""
    node = state
    for kind, arg in steps:
        if kind == "key":
            try:
                node = node[_step(node, arg)]
            except KeyError:
                return None
        elif isinstance(node, list) and arg in binding and binding[arg] < len(node):
            node = node[binding[arg]]
        else:
            return None
    return node


def _bindings(state, paths, bound):
    """Every binding (prefix -> index) of the "[]"s `bound` names, outer lists first; a list that is not there binds
    nothing, and the item's leaves there report it."""
    steps_of = {}
    for path in paths:
        for k, (kind, arg) in enumerate(path.steps):
            if kind == "each":
                steps_of.setdefault(arg, path.steps[:k])
    out = [{}]
    for p in sorted(set(bound), key=lambda p: (p.count("[]"), p)):
        grown = []
        for b in out:
            node = _resolve(state, steps_of[p], b)
            if isinstance(node, list):
                grown.extend(dict(b, **{p: j}) for j in range(len(node)))
            else:
                grown.append(b)
        out = grown
    return out


def _value(leaf):
    concrete, node, key = leaf
    return node[key] if node is not None else None


def _bad_leaves(leaves):
    """The messages of the leaves that are missing or not numbers."""
    out = []
    for concrete, node, key in leaves:
        if node is None:
            out.append("%s is %s" % (concrete, "missing" if key == "missing" else "not a list"))
        elif not _is_number(node[key]):
            out.append("%s is not a number (got %s)" % (concrete, _num_text(node[key])))
    return out


def _set(leaf, value):
    concrete, node, key = leaf
    node[key] = value


def _zero(v):
    return 0 if isinstance(v, int) else 0.0


class _Item(object):
    """One constraint item, checked when it is made: `kind`, `paths` (the kind's own), `parts` (of / then) and `rule`,
    a stable text naming it (by_rule counts and every message start with it)."""

    def __init__(self, item, where):
        if not isinstance(item, dict):
            raise FeatureError("%s: expected an object naming one of %s" % (where, ", ".join(KINDS)))
        kinds = [k for k in KINDS if k in item]
        if len(kinds) != 1:
            raise FeatureError("%s: names %s; an item names exactly one of %s" % (
                where, " and ".join(kinds) if kinds else "none", ", ".join(KINDS)))
        self.kind = kind = kinds[0]
        allowed = {kind, COMPANION.get(kind)}
        extra = sorted(k for k in item if k not in allowed)
        if extra:
            raise FeatureError("%s: %s takes %s, not %s" % (where, kind, " and ".join(
                k for k in (kind, COMPANION.get(kind)) if k), ", ".join(extra)))
        if kind in COMPANION:
            if COMPANION[kind] not in item:
                raise FeatureError("%s: %s needs %s (a list of paths)" % (where, kind, COMPANION[kind]))
            self.paths = [_Path(item[kind], "%s.%s" % (where, kind))]
            self.parts = self._list(item[COMPANION[kind]], "%s.%s" % (where, COMPANION[kind]), 1)
        else:
            self.paths = self._list(item[kind], "%s.%s" % (where, kind), 2 if kind == "order" else 1)
            self.parts = []
        if kind == "sum":
            self.rule = "sum %s = %s" % (self.paths[0].text, " + ".join(p.text for p in self.parts))
        elif kind == "order":
            self.rule = "order %s" % " <= ".join(p.text for p in self.paths)
        elif kind == "zero_if_zero":
            self.rule = "zero_if_zero %s then %s" % (self.paths[0].text, ", ".join(p.text for p in self.parts))
        else:
            self.rule = None  # nonneg and integer: one rule per path ("nonneg t[].v")

    @staticmethod
    def _list(value, where, least):
        if not isinstance(value, list) or len(value) < least:
            raise FeatureError("%s: expected a list of at least %d path%s" % (where, least, "s" if least > 1 else ""))
        return [_Path(p, "%s[%d]" % (where, i)) for i, p in enumerate(value)]

    # ---- single-path items: nonneg, integer --------------------------------------------------------------------------

    def _each_leaf(self, state):
        for path in self.paths:
            rule = "%s %s" % (self.kind, path.text)
            for leaf in _leaves(state, path, {}):
                yield rule, leaf

    def _single_violations(self, state):
        out = []
        for rule, leaf in self._each_leaf(state):
            bad = _bad_leaves([leaf])
            v = _value(leaf)
            if bad:
                out.append((rule, "%s: %s" % (rule, bad[0])))
            elif self.kind == "nonneg" and v < 0:
                out.append((rule, "%s: %s is %s, below 0" % (rule, leaf[0], _num_text(v))))
            elif self.kind == "integer" and not float(v).is_integer():
                out.append((rule, "%s: %s is %s, not a whole number" % (rule, leaf[0], _num_text(v))))
        return out

    def _single_repair(self, state):
        for _, leaf in self._each_leaf(state):
            v = _value(leaf)
            if not _is_number(v):
                continue
            if self.kind == "nonneg" and v < 0:
                _set(leaf, _zero(v))
            elif self.kind == "integer" and not float(v).is_integer():
                _set(leaf, int(math.copysign(math.floor(abs(v) + 0.5), v)))

    # ---- relations: sum, order, zero_if_zero -------------------------------------------------------------------------

    def _instances(self, state):
        """[(own leaves, part leaves)] per binding: the bound "[]"s are the sum path's for sum, every path's else."""
        bound = self.paths[0].prefixes() if self.kind == "sum" else \
            [p for path in self.paths + self.parts for p in path.prefixes()]
        out = []
        for b in _bindings(state, self.paths + self.parts, bound):
            own = [leaf for path in self.paths for leaf in _leaves(state, path, b)]
            parts = [leaf for path in self.parts for leaf in _leaves(state, path, b)]
            out.append((own, parts))
        return out

    def _relation(self, state, fix):
        """The violations of each instance; with fix, each is repaired instead (an instance with a missing leaf or one
        that is not a number is left as it is)."""
        out = []
        for own, parts in self._instances(state):
            bad = _bad_leaves(own + parts)
            if self.kind == "zero_if_zero" and not _bad_leaves(own) and _value(own[0]) != 0:
                continue  # the condition does not hold: its then numbers may be anything
            if bad:
                out.append("%s: %s" % (self.rule, "; ".join(bad)))
                continue
            if self.kind == "sum":
                values = [_value(leaf) for leaf in parts]
                total = math.fsum(values)
                if all(isinstance(v, int) for v in values):
                    total = sum(values)
                target = _value(own[0])
                if not math.isclose(target, total, rel_tol=SUM_TOLERANCE, abs_tol=SUM_TOLERANCE):
                    if fix:
                        _set(own[0], total)
                    else:
                        out.append("%s: %s is %s, the parts sum to %s" % (self.rule, own[0][0], _num_text(target),
                                                                          _num_text(total)))
            elif self.kind == "order":
                values = [_value(leaf) for leaf in own]
                if any(a > b for a, b in zip(values, values[1:])):
                    if fix:
                        for leaf, v in zip(own, sorted(values)):
                            _set(leaf, v)
                    else:
                        out.append("%s: %s" % (self.rule, ", ".join("%s is %s" % (leaf[0], _num_text(v))
                                                                    for leaf, v in zip(own, values))))
            else:
                nonzero = [leaf for leaf in parts if _value(leaf) != 0]
                for leaf in nonzero:
                    if fix:
                        _set(leaf, _zero(_value(leaf)))
                    else:
                        out.append("%s: %s is 0 but %s is %s" % (self.rule, own[0][0], leaf[0],
                                                                 _num_text(_value(leaf))))
        return out

    def violations(self, state):
        if self.kind in RELATIONS:
            return [(self.rule, m) for m in self._relation(state, False)]
        return self._single_violations(state)


class Constraints(object):
    """The items of a task's features.constraints, checked when made (FeatureError naming constraints[i] for an item
    that is not one of the forms in the module docstring)."""

    def __init__(self, constraints=None):
        if constraints is None:
            constraints = []
        if not isinstance(constraints, list):
            raise FeatureError("features.constraints: expected a list (got %s)" % type(constraints).__name__)
        self.items = [_Item(c, "features.constraints[%d]" % i) for i, c in enumerate(constraints)]

    def __len__(self):
        return len(self.items)

    def repair(self, state):
        out = copy.deepcopy(state)
        for kind in ("nonneg", "integer"):
            for item in self.items:
                if item.kind == kind:
                    item._single_repair(out)
        for _ in range(2):
            for item in self.items:
                if item.kind in RELATIONS:
                    item._relation(out, True)
        return out

    def violations(self, state):
        """[(rule, message)] of every broken constraint, in the items' order."""
        out = []
        for item in self.items:
            out.extend(item.violations(state))
        return out

    def check(self, state):
        return [m for _, m in self.violations(state)]


def _of(constraints):
    return constraints if isinstance(constraints, Constraints) else Constraints(constraints)


def repair(state, constraints):
    """A deep copy of `state` repaired by `constraints` (a list of items, or a Constraints)."""
    return _of(constraints).repair(state)


def check(state, constraints):
    """The messages of the broken constraints ([] when the state keeps them all), each starting with its rule."""
    return _of(constraints).check(state)


def violations(state, constraints):
    """[(rule, message)] of the broken constraints: the rule is the stable text of the item ("nonneg t[].v",
    "sum s.total = s.a + s.b") that by_rule counts use."""
    return _of(constraints).violations(state)
