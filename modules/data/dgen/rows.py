"""The rows dgen builds (standard library only): full OneJev rows a trainset takes through `dset grow --with`.

A new row of method m from seed row s, recipe sha12 h, number i (its place among the recipe's kept rows):
- id `syn<h[:6]>_<m>_<i:06d>`; state = adapter.repair(adapter.apply(s.state, changes)), the changes made on the
  edit view (never a verbatim path: features.table.Featurizer.edit_view leaves them out), and the row is dropped
  (counted by message as dropped_by_rule) when adapter.check reports a violation; the adapter's repair and check run
  the task's features.constraints first;
- question = rsi.rows.question_of(task); target = the seed's (the label follows the seed); split "train" (as dset
  grow's resplit of the pool rows it draws);
- meta: the seed's state_id, cluster and label_source; source "dgen:<m>"; n_obs 1; extra {"seed_id",
  "partner_id" (smote, borderline), "recipe": h}.
`hard` rows are the real failure rows themselves, unchanged.
"""
import copy
import random

from features.adapter import load_adapter
from features.flatten import as_number
from rsi.rows import question_of
from rsi.rules import adapter_path, features_settings

from . import DgenError


def task_adapter(task):
    """The adapter dgen edits rows with (apply, repair, check): the task's features adapter, the generic one when the
    task names none, with the task's features.constraints run first by its repair and check. Every dgen step loads it
    here; a malformed constraint is an input error."""
    try:
        return load_adapter(adapter_path(task), features_settings(task)["constraints"])
    except ValueError as e:  # features.FeatureError
        raise DgenError("features: %s" % e)


def row_id(sha12, method, i):
    return "syn%s_%s_%06d" % (sha12[:6], method, i)


def new_row(task, seed, state, method, sha12, i, partner_id=None):
    meta = seed["meta"]
    extra = {"seed_id": seed["id"]}
    if partner_id is not None:
        extra["partner_id"] = partner_id
    extra["recipe"] = sha12
    return {"id": row_id(sha12, method, i), "state": state, "question": question_of(task),
            "target": copy.deepcopy(seed["target"]),
            "meta": {"state_id": meta.get("state_id"), "cluster": meta.get("cluster"), "source": "dgen:" + method,
                     "label_source": meta.get("label_source"), "n_obs": 1, "extra": extra},
            "split": "train"}


def edited_state(adapter, seed_state, changes):
    """(state, violations): the seed's state with `changes` applied and repaired, and adapter.check's messages."""
    state = adapter.repair(adapter.apply(copy.deepcopy(seed_state), changes))
    return state, adapter.check(state)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def interpolate(view_a, view_b, lam):
    """The changes that move the numeric leaves of edit view `view_a` a share `lam` towards `view_b` (the same keys):
    a + lam * (b - a); a leaf that is an integer in both stays an integer (rounded). Non-numeric leaves (strings,
    bools, nulls) stay the anchor's, so they are not among the changes."""
    out = {}
    for path, a in view_a.items():
        b = view_b.get(path)
        if isinstance(a, bool) or isinstance(b, bool) or as_number(a) is None or as_number(b) is None:
            continue
        v = a + lam * (b - a)
        out[path] = int(round(v)) if _is_int(a) and _is_int(b) else float(v)
    return out


def allocate(n, sizes, how="size"):
    """n split over the keys of `sizes` in proportion to their sizes (how "size") or evenly ("uniform"), by largest
    remainder (ties to the earlier key)."""
    keys = list(sizes)
    w = [float(sizes[k]) if how == "size" else 1.0 for k in keys]
    total = sum(w) or 1.0
    raw = [x * n / total for x in w]
    q = [int(x) for x in raw]
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - q[i]), i))
    for i in order[:n - sum(q)]:
        q[i] += 1
    return dict(zip(keys, q))


def stratified_sample(items, label_of, n, seed):
    """At most n of `items`, each label its share (largest remainder), seeded; in their original order."""
    by = {}
    for i, it in enumerate(items):
        by.setdefault(label_of(it), []).append(i)
    quota = allocate(min(n, len(items)), dict((k, len(v)) for k, v in by.items()))
    rng = random.Random(seed)
    chosen = []
    for k in sorted(by):
        chosen += rng.sample(by[k], min(quota[k], len(by[k])))
    return [items[i] for i in sorted(chosen)]
