"""Prediction rows ({"id", "probs", "decision", "uncertainty"}), the metrics of a set of them, and a baseline's
decision policy (dpolicy, as a candidate's)."""
import os

from rsi import metrics as M
from rsi.rules import policy_settings

from .inputs import read_json


class Policy(dict):
    """A decision policy (policy.json's content) with the per-pair report of its fit (`report`, written as
    policy_report.json; None when unknown)."""

    def __init__(self, policy, report=None):
        dict.__init__(self, policy)
        self.report = report


class Decided(dict):
    """label -> probability with the decision a policy took (`decision`): rows() writes that decision, not the most
    probable label."""

    def __init__(self, probs, decision):
        dict.__init__(self, probs)
        self.decision = decision


def rows(ids, probs, labels):
    """One prediction row per id: `probs` (label -> probability, in the order of `ids`), the decision (a Decided
    row's own, else the most probable label, a tie going to the earlier label in `labels`, rsi.metrics.hard_label) and
    the uncertainty (1 minus the decision's probability)."""
    if len(ids) != len(probs):
        raise ValueError("%d ids but %d probability rows" % (len(ids), len(probs)))
    out = []
    for i, p in zip(ids, probs):
        decision = p.decision if isinstance(p, Decided) else None
        p = dict((label, p[label]) for label in labels)
        if decision is None:
            decision = M.hard_label(p, labels)
        out.append({"id": i, "probs": p, "decision": decision, "uncertainty": 1.0 - p[decision]})
    return out


def applied(policy, P, labels):
    """The label -> probability rows P (None stays None) with the decision policy `policy` applied
    (dpolicy.apply.apply_policy): Decided rows of the calibrated probabilities and the policy's decision."""
    if P is None:
        return None
    from dpolicy.apply import apply_policy
    out = []
    for p in P:
        r = apply_policy(policy, {"id": None, "probs": dict((label, p[label]) for label in labels)}, labels)
        out.append(Decided(r["probs"], r["decision"]))
    return out


def with_policy(task, val_rows, P_val, P_eval):
    """(P_val, P_eval, policy). With the task's policy section (rsi.rules.policy_settings): the decision policy fitted
    as a candidate's (dpolicy.fit.fit_policy) on the valset predictions P_val (label -> probability rows in the order
    of val_rows, the valset rows with their labels), as a Policy carrying its fit report, and P_val and P_eval with it
    applied (applied). Without one: P_val and P_eval as they are, and None."""
    settings = policy_settings(task)
    if settings is None:
        return P_val, P_eval, None
    from dpolicy.fit import fit_policy
    labels = task["decision"]["labels"]
    policy, report = fit_policy(task, settings, val_rows, rows([r["id"] for r in val_rows], P_val, labels))
    return applied(policy, P_val, labels), applied(policy, P_eval, labels), Policy(policy, report)


def recorded_policy(d):
    """The decision policy (policy.json) the artifact directory `d` (a baseline or a packed candidate) lists as its
    file `policy`, as a Policy with the report it lists as `policy_report` (None without one); None when it lists no
    policy."""
    files = read_json(os.path.join(d, "manifest.json")).get("files") or {}
    if "policy" not in files:
        return None
    report = files.get("policy_report")
    return Policy(read_json(os.path.join(d, files["policy"]["path"])),
                  read_json(os.path.join(d, report["path"])) if report else None)


def metric_names(task):
    """The task's primary metric and its other metrics that rsi computes, primary first, each once. ValueError when
    the primary metric is not one rsi computes (nothing could be chosen or compared by it)."""
    m = task["metrics"]
    if m["primary"] not in M.KNOWN_METRICS:
        raise ValueError("the primary metric %r is not one rsi computes (%s)" % (m["primary"],
                                                                              ", ".join(M.KNOWN_METRICS)))
    names = []
    for n in [m["primary"]] + list(m.get("others", [])):
        if n in M.KNOWN_METRICS and n not in names:
            names.append(n)
    return names


def metric_values(task, pred_rows, label_rows):
    """metric -> value of the prediction rows against the label rows, for metric_names(task) (rsi.metrics.score, as
    `rsi score` and registration compute them)."""
    return dict((n, M.score(n, pred_rows, label_rows, task)) for n in metric_names(task))
