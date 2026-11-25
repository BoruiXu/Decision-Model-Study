"""Data quality: the profile of a list of rows, the trainset quality gate and cleaning (check --write).

dset computes four checks of the task's quality_checks itself (spec §6): min_rows, min_positive_rows (binary tasks),
max_duplicate_rate (the share of rows whose input an earlier row has) and max_conflict_rate (the share of rows whose
input has more than one hard label). A min_ check passes at value >= threshold, a max_ check at value <= threshold. The
module agent computes any other name of quality_checks and hands it over (--checks: [{name, value, threshold,
passed}]); a name of the task's that neither computes is an error.
"""
import copy
import math

from rsi import datasplit as DS
from rsi import rows as R
from rsi.metrics import hard_label

from . import DsetError

COMPUTED = ("min_rows", "min_positive_rows", "max_duplicate_rate", "max_conflict_rate")
CHECK_KEYS = ("name", "value", "threshold", "passed")


def label_counts(rows, task):
    """Rows per hard label, in the task's label order."""
    labels = task["decision"]["labels"]
    counts = dict((label, 0) for label in labels)
    for r in rows:
        counts[hard_label(r["target"], labels)] += 1
    return counts


def _by_input(rows, task):
    """input fingerprint -> [the hard labels of its rows (a set), its rows], in first-seen order."""
    labels = task["decision"]["labels"]
    out = {}
    for r in rows:
        entry = out.setdefault(R.input_fingerprint(R.state_of(r)), [set(), []])
        entry[0].add(hard_label(r["target"], labels))
        entry[1].append(r)
    return out


def profile(rows, task):
    """{rows, label_counts, duplicate_inputs (rows whose input an earlier row has), conflicting_inputs (inputs with
    more than one hard label), conflicting_rows (the rows of those inputs)}."""
    groups = _by_input(rows, task).values()
    conflicting = [rs for labels, rs in groups if len(labels) > 1]
    return {"rows": len(rows), "label_counts": label_counts(rows, task),
            "duplicate_inputs": sum(len(rs) - 1 for _, rs in groups),
            "conflicting_inputs": len(conflicting), "conflicting_rows": sum(len(rs) for rs in conflicting)}


def _computed(rows, task):
    p = profile(rows, task)
    n = p["rows"]
    out = {"min_rows": n, "max_duplicate_rate": p["duplicate_inputs"] / float(n) if n else 0.0,
           "max_conflict_rate": p["conflicting_rows"] / float(n) if n else 0.0}
    d = task["decision"]
    if d["type"] == "binary":
        out["min_positive_rows"] = p["label_counts"][d["positive"]]
    return out


def _given(given, computed):
    """The checks the agent handed over, by name, after checking their form."""
    if given is None:
        return {}
    if not isinstance(given, list):
        raise DsetError("--checks: expected a JSON list of {name, value, threshold, passed}")
    out = {}
    for i, c in enumerate(given):
        if (not isinstance(c, dict) or sorted(c) != sorted(CHECK_KEYS) or not isinstance(c["name"], str)
                or not c["name"] or not isinstance(c["passed"], bool)
                or any(not (c[k] is None or isinstance(c[k], (int, float, str, bool)))
                       for k in ("value", "threshold"))):
            raise DsetError("--checks[%d]: expected {name, value, threshold, passed}: a name, a number, string, bool "
                            "or null as value and threshold, and passed true or false" % i)
        if c["name"] in computed:
            raise DsetError("--checks: dset computes %s itself; leave it out" % c["name"])
        if c["name"] in out:
            raise DsetError("--checks: %s is given twice" % c["name"])
        out[c["name"]] = dict((k, c[k]) for k in CHECK_KEYS)
    return out


def quality_checks(rows, task, given=None):
    """The trainset quality checks: each name of the task's quality_checks in its order, computed by dset (COMPUTED)
    or taken from `given` (the agent's list), then the agent's checks the task does not name. Raises DsetError when a
    name of the task's is neither computed nor given."""
    computed = _computed(rows, task)
    extra = _given(given, computed)
    out = []
    for name, threshold in (task.get("quality_checks") or {}).items():
        if name in computed:
            if not DS.is_number(threshold):
                raise DsetError("quality_checks.%s: the threshold must be a number (got %r)" % (name, threshold))
            value = computed[name]
            passed = value >= threshold if name.startswith("min_") else value <= threshold
            out.append({"name": name, "value": value, "threshold": threshold, "passed": passed})
        elif name in extra:
            out.append(extra.pop(name))
        else:
            raise DsetError("quality_checks.%s: dset computes only %s here; compute %s and give it with --checks "
                            "(a JSON list of {name, value, threshold, passed})" % (name, ", ".join(sorted(computed)),
                                                                                   name))
    return out + list(extra.values())


def gate(checks):
    """The quality gate: {passed: every check passed, checks}."""
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def clean(rows, task, mode):
    """check --write: drop: the rows of an input with more than one hard label all go, of any other input only its
    first row stays; merge: the rows of each input become its first row with the target their n_obs-weighted mean (the
    labels any of them names, in the task's order) and n_obs their sum. Rows keep the order of their input's first
    row."""
    labels = task["decision"]["labels"]
    out = []
    for found, rs in _by_input(rows, task).values():
        if mode == "drop":
            if len(found) == 1:
                out.append(rs[0])
            continue
        if len(rs) == 1:
            out.append(rs[0])
            continue
        row = copy.deepcopy(rs[0])
        weights = [r["meta"]["n_obs"] for r in rs]
        total = sum(weights)
        named = [label for label in labels if any(label in r["target"] for r in rs)]
        row["target"] = dict((label, math.fsum(w * r["target"].get(label, 0) for w, r in zip(weights, rs)) / total)
                             for label in named)
        row["meta"]["n_obs"] = total
        out.append(row)
    return out
