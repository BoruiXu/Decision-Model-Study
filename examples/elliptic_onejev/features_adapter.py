"""Features adapter of the Elliptic++ txhist task (the task's features.adapter; standard library only).

A state is one Bitcoin address's transaction history as 01_build_data.sh builds it:
{"entity", "summary", "transactions"}. Checked on all 291,419 rows of the upstream train, val and test files:
- `transactions` lists the address's most recent transactions, at most LIST_CAP of them, each with `steps_ago`
  (time steps before the row's step), `role` (sent, received or both: the address is among the transaction's inputs,
  its outputs, or both), `tx_total_btc`, `tx_fee_btc`, `tx_num_input_addresses`, `tx_num_output_addresses`,
  `tx_smallest_output_btc` and `tx_largest_output_btc`; in every row smallest <= largest <= total;
- `summary` is derived from the whole history: `txs_listed` is the number listed; `txs_sent` counts the transactions
  with role sent or both, `txs_received` those with role received or both; `first_seen_steps_ago` is the largest
  `steps_ago`, `active_steps` the number of distinct `steps_ago`. With fewer than LIST_CAP listed, the whole history
  is listed and the summary equals what the listed transactions give; with LIST_CAP listed the history may be longer,
  and each of the four is at least what the listed ones give.
"""
import copy
import math

from features.flatten import edit_view as generic_edit_view
from features.flatten import model_view, set_paths

LIST_CAP = 32
ROLES = ("sent", "received", "both")
TX_NUMBERS = ("steps_ago", "tx_total_btc", "tx_fee_btc", "tx_num_input_addresses", "tx_num_output_addresses",
              "tx_smallest_output_btc", "tx_largest_output_btc")
SUMMARY = ("txs_sent", "txs_received", "first_seen_steps_ago", "active_steps", "txs_listed")


def _number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def _transactions(state):
    txs = state.get("transactions") if isinstance(state, dict) else None
    return [t for t in txs if isinstance(t, dict)] if isinstance(txs, list) else []


def _derived(txs):
    """The summary the listed transactions give."""
    steps = [t["steps_ago"] for t in txs if _number(t.get("steps_ago"))]
    return {"txs_sent": sum(1 for t in txs if t.get("role") in ("sent", "both")),
            "txs_received": sum(1 for t in txs if t.get("role") in ("received", "both")),
            "first_seen_steps_ago": max(steps) if steps else 0,
            "active_steps": len(set(steps)),
            "txs_listed": len(txs)}


def features(state):
    """The model view plus, per role, the number of listed transactions and the sum of each of their numbers."""
    out = model_view(state)
    txs = _transactions(state)
    for role in ROLES:
        mine = [t for t in txs if t.get("role") == role]
        out["transactions[role=%s].count" % role] = float(len(mine))
        for field in TX_NUMBERS[1:]:
            out["transactions[role=%s].%s.sum" % (role, field)] = math.fsum(
                float(t[field]) for t in mine if _number(t.get(field)))
    return out


def edit_view(state):
    """The transaction fields only: the summary is derived from them (repair recomputes it)."""
    return dict((k, v) for k, v in generic_edit_view(state).items() if k.startswith("transactions."))


def apply(state, changes):
    """set_paths restricted to the edit view: a summary path is derived, so it is refused like an unknown path."""
    view = edit_view(state)
    for path in changes:
        if path not in view:
            raise KeyError("%s: not an editable path (only the transaction fields are; the summary is derived)"
                           % path)
    return set_paths(state, changes)


def repair(state):
    """A deep copy with the summary recomputed from the transactions (with LIST_CAP listed, each count is kept when
    it is above what the listed ones give, since the history may be longer)."""
    out = copy.deepcopy(state)
    txs = _transactions(out)
    derived = _derived(txs)
    summary = out.get("summary")
    if not isinstance(summary, dict):
        summary = out["summary"] = {}
    for key in SUMMARY:
        old = summary.get(key)
        keep = len(txs) >= LIST_CAP and key != "txs_listed" and _number(old) and old > derived[key]
        summary[key] = old if keep else derived[key]
    return out


def check(state):
    """The broken rules, one message each ([] when the state is consistent)."""
    summary = state.get("summary") if isinstance(state, dict) else None
    txs = state.get("transactions") if isinstance(state, dict) else None
    if not isinstance(summary, dict):
        return ["summary: expected an object"]
    if not isinstance(txs, list) or not all(isinstance(t, dict) for t in txs):
        return ["transactions: expected a list of objects"]
    errors = []
    if len(txs) > LIST_CAP:
        errors.append("transactions: %d listed, more than the %d the history lists" % (len(txs), LIST_CAP))
    for i, t in enumerate(txs):
        if t.get("role") not in ROLES:
            errors.append("transactions.%d.role: %r is not one of %s" % (i, t.get("role"), ", ".join(ROLES)))
        bad = [f for f in TX_NUMBERS if not _number(t.get(f))]
        for f in bad:
            errors.append("transactions.%d.%s: %r is not a number" % (i, f, t.get(f)))
        for f in TX_NUMBERS:
            if f not in bad and t[f] < 0:
                errors.append("transactions.%d.%s: %r is negative" % (i, f, t[f]))
        small, large, total = "tx_smallest_output_btc", "tx_largest_output_btc", "tx_total_btc"
        if not set(bad) & {small, large, total} and not t[small] <= t[large] <= t[total]:
            errors.append("transactions.%d: smallest output %r, largest output %r and total %r are not in order "
                          "(smallest <= largest <= total)" % (i, t[small], t[large], t[total]))
    derived = _derived(txs)
    capped = len(txs) >= LIST_CAP
    for key in SUMMARY:
        v = summary.get(key)
        if not _number(v):
            errors.append("summary.%s: %r is not a number" % (key, v))
        elif key == "txs_listed" or not capped:
            if v != derived[key]:
                errors.append("summary.%s: %r, but the transactions give %r" % (key, v, derived[key]))
        elif v < derived[key]:
            errors.append("summary.%s: %r, less than the %r the listed transactions give" % (key, v, derived[key]))
    return errors


def describe():
    return (
        "The state is one Bitcoin address's transaction history. `transactions` lists its most recent transactions, "
        "at most %d. Each has `steps_ago` (how many time steps before the current one), `role` (sent, received or "
        "both: the address is among the transaction's inputs, its outputs, or both), `tx_total_btc`, `tx_fee_btc`, "
        "`tx_num_input_addresses`, `tx_num_output_addresses`, `tx_smallest_output_btc` and `tx_largest_output_btc`. "
        "Every number is 0 or more. In each transaction, tx_smallest_output_btc <= tx_largest_output_btc <= "
        "tx_total_btc. The summary is derived from the transactions: `txs_listed` is the number listed; `txs_sent` "
        "counts the transactions whose role is sent or both, `txs_received` those whose role is received or both; "
        "`first_seen_steps_ago` is the largest steps_ago and `active_steps` the number of distinct steps_ago "
        "values. With %d transactions listed the history may be longer, and each of these four is at least what "
        "the listed transactions give. Edit only the transactions; the summary is recomputed from them."
        % (LIST_CAP, LIST_CAP))
