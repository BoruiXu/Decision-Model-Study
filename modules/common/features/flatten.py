"""The generic views of a state (a row's `state`, a JSON object).

model_view: column -> float, the same columns for states of the same shape whatever their key order:
- a numeric leaf outside any list becomes "<dot path>" (a bool is 1.0 or 0.0);
- a list at path p becomes "p.count";
- its number items become "p.mean", "p.max", "p.sum";
- for dict items, each numeric leaf q (at any depth, outside nested lists) becomes "p[].q.mean", "p[].q.max",
  "p[].q.sum", over the items that have it; a nested list at q becomes "p[].q.count.mean";
- strings, nulls and non-finite numbers are skipped.

edit_view: path -> scalar for every scalar leaf, list indices as path parts ("transactions.3.v"); set_paths writes
such paths back into a deep copy of the state.

Paths join keys with dots and nothing is escaped, so a key that contains a dot or is empty is ambiguous:
{"a.b": 1} and {"a": {"b": 1}} give the same column "a.b", and set_paths reads "a.b" as the nested path only.
"""
import copy
import math

from . import FeatureError


def as_number(v):
    """v as a finite float when it is a number or a bool (1.0 or 0.0), else None."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        try:
            x = float(v)
        except OverflowError:
            return None
        return x if math.isfinite(x) else None
    return None


def _is_number(v):
    """A finite int or float (a bool is not a number here, as in rsi.rules.is_number)."""
    return not isinstance(v, bool) and isinstance(v, (int, float)) and as_number(v) is not None


def _join(path, key):
    """path.key, an empty part left out (a list item that is itself a list has the empty path)."""
    return key if not path else path if not key else path + "." + key


def _aggregate(out, name, values):
    total = math.fsum(values)
    out[name + ".mean"] = total / len(values)
    out[name + ".max"] = max(values)
    out[name + ".sum"] = total


def _item_leaves(value, path, leaves, counts):
    """The numeric leaves (path -> float) and nested list lengths (path -> count) of one list item."""
    if isinstance(value, dict):
        for k, v in value.items():
            _item_leaves(v, _join(path, str(k)), leaves, counts)
    elif isinstance(value, list):
        counts[path] = float(len(value))
    else:
        x = as_number(value)
        if x is not None:
            leaves[path] = x


def _list_columns(items, path, out):
    out[_join(path, "count")] = float(len(items))
    numbers, leaves, counts = [], {}, {}
    for item in items:
        if isinstance(item, (dict, list)):
            one_leaves, one_counts = {}, {}
            _item_leaves(item, "", one_leaves, one_counts)
            for q, x in one_leaves.items():
                leaves.setdefault(q, []).append(x)
            for q, n in one_counts.items():
                counts.setdefault(q, []).append(n)
        else:
            x = as_number(item)
            if x is not None:
                numbers.append(x)
    if numbers:
        _aggregate(out, path, numbers)
    for q, xs in leaves.items():
        _aggregate(out, _join(path + "[]", q), xs)
    for q, ns in counts.items():
        out[_join(_join(path + "[]", q), "count.mean")] = math.fsum(ns) / len(ns)


def _model_columns(value, path, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _model_columns(v, _join(path, str(k)), out)
    elif isinstance(value, list):
        _list_columns(value, path, out)
    elif path:
        x = as_number(value)
        if x is not None:
            out[path] = x


def model_view(state):
    """The state's numeric columns (column -> float), by the rules in the module docstring."""
    out = {}
    _model_columns(state, "", out)
    return out


def _scalars(value, path, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _scalars(v, _join(path, str(k)), out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _scalars(v, _join(path, str(i)), out)
    elif path:
        out[path] = value


def edit_view(state):
    """Every scalar leaf of the state (string, number, bool or null) by its path, list indices as path parts."""
    out = {}
    _scalars(state, "", out)
    return out


def _index(part):
    """A list index written as edit_view writes it (digits, no leading zero), else None."""
    if part.isascii() and part.isdigit() and str(int(part)) == part:
        return int(part)
    return None


def _locate(state, path):
    """(container, key) of the scalar leaf at `path`; KeyError when the state has no such leaf."""
    parts = path.split(".")
    node = state
    for i, part in enumerate(parts):
        if isinstance(node, dict) and part in node:
            key = part
        elif isinstance(node, list) and _index(part) is not None and _index(part) < len(node):
            key = _index(part)
        else:
            raise KeyError("%s: no such path in the state" % path)
        if i < len(parts) - 1:
            node = node[key]
    if isinstance(node[key], (dict, list)):
        raise KeyError("%s: not a scalar leaf of the state" % path)
    return node, key


def set_paths(state, changes):
    """A deep copy of the state with each path of `changes` (as edit_view writes it) set to its value.

    KeyError for a path the state does not have; FeatureError when a number would become a non-number (or a
    non-finite one), or a value is an object or a list.
    """
    out = copy.deepcopy(state)
    for path, value in changes.items():
        node, key = _locate(out, path)
        if isinstance(value, (dict, list)):
            raise FeatureError("%s: a leaf takes a scalar, not %s" % (path, type(value).__name__))
        if _is_number(node[key]) and not _is_number(value):
            raise FeatureError("%s: the number %r cannot become %r" % (path, node[key], value))
        node[key] = copy.deepcopy(value)
    return out
