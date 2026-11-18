"""Evaluator plugins for tasks other than binary decisions (design section 10). Imports without torch.

A plugin is a Python file named by ``evaluator.path`` (with ``evaluator.type: plugin``). Every hook is
optional; finalize falls back to the default when one is missing:

    validate(rows) -> list[str] | None        data errors (non-empty fails the run)   default: no check
    select_rule(events) -> int                step from the harness_eval events       default: lowest select NLL
    calibrate(preds, rows) -> dict            decision thresholds                     default: none
    baselines(train_rows, eval_rows) -> dict  {name: metrics} on eval_rows            default: registered baselines
    acceptance(preds, rows) -> dict           router metrics, optional "gate"         default: NLL, accuracy, ECE

``preds`` maps row id to the rescored record ``{"id", "labels", "probs", "target"}``. ``acceptance`` and
``baselines`` run on the pooled select rows and on each eval set's select rows. A ``gate`` returned by
``acceptance`` is ``{"passed": bool, ...}`` (a bare bool is accepted).
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from harness.eval_records import nll
from harness.evaluator.binary_decision import POOLED, TRAIN_DROPPED
from harness.evaluator.labels import baseline_examples
from harness.evaluator.metrics import auc, within_group_auc


def load_plugin(path: str) -> ModuleType:
    """Import a plugin module from a file path."""
    p = Path(path).expanduser().absolute()
    if not p.is_file():
        raise FileNotFoundError(f"evaluator plugin not found: {path}")
    name = "ojh_plugin_" + hashlib.sha256(str(p).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def hook(module: ModuleType | None, name: str):
    fn = getattr(module, name, None) if module is not None else None
    return fn if callable(fn) else None


def default_acceptance(preds: dict, rows: list[dict]) -> dict:
    """NLL, accuracy (argmax of probs vs argmax of target) and 10-bin top-1 ECE."""
    recs = [preds[r["id"]] for r in rows]
    if not recs:
        return {"n": 0}
    bins = [[] for _ in range(10)]
    correct = 0
    for r in recs:
        top = max(range(len(r["probs"])), key=r["probs"].__getitem__)
        ok = top == max(range(len(r["target"])), key=r["target"].__getitem__)
        correct += ok
        conf = r["probs"][top]
        bins[min(9, int(conf * 10))].append((conf, ok))
    n = len(recs)
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins if b)
    return {"n": n, "nll": nll(recs), "accuracy": correct / n, "ece": ece}


class DefaultBaselines:
    """The registered baselines (``harness.baselines``) available for ``fields``, scored against
    ``fields.label`` (unlabeled rows are skipped; skipped train rows are kept in ``skipped``)."""

    def __init__(self, train_rows, fields):
        self.fields = fields
        self.train, self.skipped = baseline_examples(train_rows, fields)
        self._fitted = None

    def _examples(self, rows):
        return baseline_examples(rows, self.fields)[0]

    def _metrics(self, examples, scores):
        rates = [e["rate"] for e in examples]
        out = {"auc": auc(scores, rates)}
        if self.fields.group:
            out["within_group_auc"] = within_group_auc([e["group"] for e in examples], scores, rates)
        return out

    def __call__(self, train_rows, eval_rows) -> dict:
        from harness.baselines import available, predictor

        ex = self._examples(eval_rows)
        if not ex or not self.train:
            return {}
        if self._fitted is None:
            self._fitted = {b.name: b.fit(self.train) for b in available(self.fields)}
        return {name: self._metrics(ex, predictor(m)(ex)) for name, m in self._fitted.items()}


def _gate_of(value):
    if value is None:
        return None
    return {"passed": value} if isinstance(value, bool) else dict(value)


def evaluate_plugin(module: ModuleType, preds: dict, parts, train_rows: list[dict], fields, evaluator_cfg) -> dict:
    """Same top-level shape as ``binary_decision.evaluate``: sets, thresholds, gate, dropped."""
    accept = hook(module, "acceptance") or default_acceptance
    baselines = hook(module, "baselines") or DefaultBaselines(train_rows, fields)
    by_set: dict[str, list[dict]] = {}
    for row in parts.select_rows:
        by_set.setdefault(parts.select_index[row["id"]], []).append(row)
    by_set[POOLED] = list(parts.select_rows)
    sets, gate = {}, {"enabled": bool(evaluator_cfg.gate)}
    verdicts = []
    for name, rows in by_set.items():
        result = dict(accept({r["id"]: preds[r["id"]] for r in rows}, rows))
        g = _gate_of(result.pop("gate", None))
        sets[name] = {"questions": len(rows), "router": result, **(baselines(train_rows, rows) or {})}
        if g is not None:
            gate[name] = g
            verdicts.append(g.get("passed"))
    calibrate = hook(module, "calibrate")
    thresholds = (calibrate(preds, parts.select_rows) or {}) if calibrate else {}
    decided = [v for v in verdicts if v is not None]
    gate["passed"] = (all(decided) if decided else None) if evaluator_cfg.gate else None
    skipped = baselines.skipped if isinstance(baselines, DefaultBaselines) else []  # a plugin hook owns its own
    return {"sets": sets, "thresholds": thresholds, "gate": gate,
            "dropped": {TRAIN_DROPPED: skipped} if skipped else {}}
