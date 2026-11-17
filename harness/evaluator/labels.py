"""Label reading and baseline examples shared by the evaluators and eval-report. Imports without torch."""
from __future__ import annotations

from harness.config import get_field
from harness.evaluator.text import text_of


def num(row, path):
    """Numeric field value, or None when absent or not a number."""
    try:
        v = get_field(row, path)
    except KeyError:
        return None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def label_value(row, path):
    """The label at ``path``, or None when the row is unlabeled.

    Under ``target`` a key missing from the row's target dict reads as 0.0: prepare accepts a target with a
    subset of the labels and OneJev fills the rest with 0, so ``{"no": 1.0}`` is a hard negative. Anywhere
    else a missing or non-numeric value is unlabeled.
    """
    head, _, key = path.partition(".")
    target = row.get("target") if isinstance(row, dict) else None
    if head == "target" and key and "." not in key and isinstance(target, dict) and key not in target:
        return 0.0
    return num(row, path)


def baseline_examples(rows, fields) -> tuple[list[dict], list[str]]:
    """``({id, rate[, group][, text]}, skipped ids)``: rows without a label, or without a group when
    ``fields.group`` is set, are skipped."""
    out, skipped = [], []
    for row in rows:
        rate = label_value(row, fields.label)
        if rate is None:
            skipped.append(row["id"])
            continue
        ex = {"id": row["id"], "rate": rate}
        if fields.group:
            try:
                ex["group"] = get_field(row, fields.group)
            except KeyError:
                skipped.append(row["id"])
                continue
        if fields.text:
            ex["text"] = text_of(row, fields.text)
        out.append(ex)
    return out, skipped
