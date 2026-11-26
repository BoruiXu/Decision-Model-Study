"""LLM generation of dgen's llm_imitate and llm_pattern recipes, kept by an independent reviewer (standard library
only).

The calls are llm.claude.ClaudeCall objects: a generator and a reviewer, each its own `claude -p` process per call in
its own directory. The prompts are the English templates in prompts/, filled with str.format_map:
- system.md (the generator's context, put before every generator prompt): the task's decision, its labels and their
  meanings, the input description, the editable fields of the edit view (verbatim paths left out, list positions as
  "[]") with their observed type and min / median / max over the real rows, the adapter's describe() and the task's
  declarative constraints, and the verbatim paths that are never changed;
- imitate.md (M2, LLM2LLM-style): one real failure (its edit view, true label, the candidate's decision and its
  probability) -> n records the model would get wrong the same way;
- analysis.md and pattern.md (M3, PaDA-Agent-style): one analysis call per weakness type (12 members with their
  probability, the medians of correctly decided rows of the true label and of the wrongly chosen one) -> {"analysis",
  "pattern", "strategy"}; generation calls then get the pattern, the strategy and a seed of that type;
- review.md: the reviewer's whole prompt is this template, the input description and the records' states (as JSON,
  numbered from 0, ten per call); it scores realism only, 1-10 with a reason. It never sees a label, a probability,
  the generator's prompt, an analysis or a pattern.

Generation asks for {"records": [{"changes": {<edit-view path>: <value>}}]}, n_per_call (5) records per call; the
label always follows the seed (the LLM never sets one). A change set is rejected before review (rejected_pre_review,
by reason) when it names a path the seed's edit view lacks (unknown path) or a verbatim path, changes a value's type
(an int accepts an integral float, a float any finite number), changes nothing, breaks the adapter's check after
repair (also counted in dropped_by_rule by message), fails rsi.rows.check_full_rows, or repeats an earlier candidate.
A reply with output None (not JSON, not the schema, a non-zero exit, a timeout) counts in rejected_replies. The
rest is reviewed; a row is kept when its score reaches the threshold (calibrate_reviewer: the highest t such that at
least 90% of real rows score >= t), rejected_by_review when it scores below; a candidate the reviewer gave no score
(its reply failed, or skipped it) is unreviewed, never rejected. Rounds: at most 3; round r asks ceil(need * (1.3 in
round 0, 1.6 after) / n_per_call) generator calls, need being the rows still missing. The reviewer's reasons below
the threshold are fed back ("Earlier candidates were rejected by a reviewer for: ...") to the calls of the same seed
(imitate) or the same type (pattern), and a round's first calls go to those seeds (imitate) or types (pattern: the
type's next seed) before new seeds are drawn, so each rejection is retried with its reasons. A reached cap
(llm.claude.LlmBudgetExceeded) ends the generation: stopped "api budget"; the rows still missing are short. Nothing
here raises for a bad reply.
"""
import copy
import json
import math
import os
import random
import statistics

from features.table import Featurizer
from llm.claude import LlmBudgetExceeded
from rsi import rows as R
from rsi.rules import adapter_path, features_settings, verbatim_paths

from . import DgenError
from . import rows as RW

PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
N_PER_CALL, ROUNDS, BATCH = 5, 3, 10
OVERSAMPLE = (1.3, 1.6)  # round 0, later rounds
CALIBRATION_ROWS, PASS_SHARE, MIN_SCORED_SHARE = 60, 0.9, 0.5
MEMBERS, SCHEMA_ROWS, MAX_REASONS = 12, 5000, 5
EPS = 1e-9
FEEDBACK = "\n\nEarlier candidates were rejected by a reviewer for: %s. Avoid these problems."

# The answer schemas, within what the CLI's structured output handles (llm.claude.schema_problems): single types,
# closed fixed objects, and the changes map with a value schema (an anyOf of scalar types: with a bare open object the
# model answers {}); counts and score ranges are checked here.
SCALAR = {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"}]}
REVIEW_SCHEMA = {"type": "object", "required": ["scores"], "additionalProperties": False, "properties": {
    "scores": {"type": "array", "items": {
        "type": "object", "required": ["i", "score", "reason"], "additionalProperties": False,
        "properties": {"i": {"type": "integer"}, "score": {"type": "integer"}, "reason": {"type": "string"}}}}}}
RECORDS_SCHEMA = {"type": "object", "required": ["records"], "additionalProperties": False, "properties": {
    "records": {"type": "array", "items": {
        "type": "object", "required": ["changes"], "additionalProperties": False,
        "properties": {"changes": {"type": "object", "additionalProperties": SCALAR}}}}}}
ANALYSIS_SCHEMA = {"type": "object", "required": ["analysis", "pattern", "strategy"], "additionalProperties": False,
                   "properties": {"analysis": {"type": "string"}, "pattern": {"type": "string"},
                                  "strategy": {"type": "string"}}}


class Uncalibrated(DgenError):
    """The reviewer scored too few of the real rows to set a threshold."""


_TEMPLATES = {}


def fill(name, **values):
    """prompts/<name>.md filled with `values`."""
    if name not in _TEMPLATES:
        with open(os.path.join(PROMPTS, name + ".md"), encoding="utf-8") as f:
            _TEMPLATES[name] = f.read().strip()
    return _TEMPLATES[name].format_map(values)


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _g(x):
    return "%.6g" % x


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def featurizer(task):
    """The task's Featurizer: its edit_view leaves the verbatim paths out."""
    return Featurizer(adapter_path(task), verbatim_paths(task))


# ---- the generator's context -----------------------------------------------------------------------------------------

def field_of(path):
    """An edit-view path with its list positions as "[]": "tx.0.amount" -> "tx[].amount"."""
    out = []
    for part in path.split("."):
        if part.isdigit():
            if out:
                out[-1] += "[]"
            else:
                out.append("[]")
        else:
            out.append(part)
    return ".".join(out)


def _fields(views):
    """{field: [values]} over edit views, in first-seen order."""
    out = {}
    for view in views:
        for path, value in view.items():
            out.setdefault(field_of(path), []).append(value)
    return out


def describe_fields(views):
    lines = []
    for field, values in _fields(views).items():
        present = [v for v in values if v is not None]
        if not present:
            lines.append("- %s: always null" % field)
        elif all(_number(v) for v in present):
            kind = "integer" if all(isinstance(v, int) for v in present) else "number"
            lines.append("- %s: %s, min %s, median %s, max %s%s" % (
                field, kind, _g(min(present)), _g(statistics.median(present)), _g(max(present)),
                ", sometimes null" if len(present) < len(values) else ""))
        elif all(isinstance(v, bool) for v in present):
            lines.append("- %s: true or false" % field)
        elif all(isinstance(v, str) for v in present):
            distinct = sorted(set(present))
            lines.append("- %s: text, %s" % (field, "one of %s" % ", ".join(json.dumps(x, ensure_ascii=False)
                                                                            for x in distinct)
                                             if len(distinct) <= 8 else "%d distinct values" % len(distinct)))
        else:
            lines.append("- %s: mixed types" % field)
    return "\n".join(lines) or "none"


def medians(views):
    lines = ["- %s: %s" % (field, _g(statistics.median([v for v in values if _number(v)])))
             for field, values in _fields(views).items() if any(_number(v) for v in values)]
    return "\n".join(lines) or "no such records in the sample"


def system_prompt(task, adapter, real_rows, seed=0):
    """system.md for the task: the field schema from the real rows (a seeded sample of SCHEMA_ROWS when more)."""
    fz = featurizer(task)
    rows = real_rows if len(real_rows) <= SCHEMA_ROWS else random.Random(seed).sample(real_rows, SCHEMA_ROWS)
    d = task["decision"]
    constraints = [adapter.describe().strip()] if adapter.describe().strip() else []
    constraints += ["- %s" % compact(c) for c in features_settings(task)["constraints"]]
    return fill("system", decision=d["instructions"],
                labels="\n".join("- %s: %s" % (l, d["label_meanings"].get(l, "")) for l in d["labels"]),
                input_description=task["input"]["description"],
                schema="%s\n(seen in %d real records)" % (describe_fields([fz.edit_view(R.state_of(r)) for r in rows]),
                                                         len(rows)),
                constraints="\n".join(constraints) or "none",
                verbatim="\n".join("- %s" % p for p in verbatim_paths(task)) or "none")


# ---- the reviewer ----------------------------------------------------------------------------------------------------

def review(call, task, rows, batch=BATCH):
    """One (score or None, reason) per row: the reviewer's realism score (an integer 1-10) of each row's state."""
    out = []
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        records = "\n\n".join("Record %d:\n%s" % (i, compact(R.state_of(r))) for i, r in enumerate(part))
        res = call.run(fill("review", input_description=task["input"]["description"], records=records),
                       REVIEW_SCHEMA, "review")
        if res.output is None:
            out += [(None, "no review (%s)" % res.error)] * len(part)
            continue
        got = {}
        for s in res.output["scores"]:
            i, score = s["i"], s["score"]
            if 0 <= i < len(part) and i not in got and 1 <= score <= 10:
                got[i] = (score, " ".join(s["reason"].split()))
        out += [got.get(i, (None, "the reviewer gave no score")) for i in range(len(part))]
    return out


def calibrate_reviewer(call, task, real_rows, batch=BATCH):
    """The highest t in 1..10 such that at least 90% of the real rows the reviewer scored score >= t. Uncalibrated
    when it scored fewer than half of them."""
    scored = [s for s, _ in review(call, task, real_rows, batch) if s is not None]
    if not scored or len(scored) < MIN_SCORED_SHARE * len(real_rows):
        raise Uncalibrated("the reviewer scored %d of the %d real rows, too few to set its threshold"
                           % (len(scored), len(real_rows)))
    for t in range(10, 0, -1):
        if sum(1 for s in scored if s >= t) >= PASS_SHARE * len(scored) - EPS:
            return t
    return 1


# ---- the generation loop ---------------------------------------------------------------------------------------------

def _same_type(old, new):
    """(value, ok): `new` in the type of the seed's value `old`."""
    if old is None or isinstance(old, (bool, str)):
        return new, type(new) is type(old)
    if isinstance(new, bool) or not isinstance(new, (int, float)) or not math.isfinite(new):
        return None, False
    if isinstance(old, int):
        if isinstance(new, int):
            return new, True
        return (int(new), True) if new.is_integer() else (None, False)
    return float(new), True


def _candidate(task, adapter, fz, seed_row, changes, method, h, seen, stats):
    """(provisional row, None), or (None, why it is rejected before review)."""
    if not isinstance(changes, dict) or not changes:
        return None, "no changes"
    state0 = R.state_of(seed_row)
    full, editable = adapter.edit_view(copy.deepcopy(state0)), fz.edit_view(state0)
    clean = {}
    for path, value in changes.items():
        if path not in full:
            return None, "unknown path"
        if path not in editable:
            return None, "verbatim path"
        v, ok = _same_type(full[path], value)
        if not ok:
            return None, "type change"
        if v != full[path]:
            clean[path] = v
    if not clean:
        return None, "no change"
    try:
        state, bad = RW.edited_state(adapter, state0, clean)
    except (KeyError, ValueError) as e:  # a path the adapter's apply refuses (features.FeatureError is a ValueError)
        return None, "apply refused: %s" % e
    if bad:
        for msg in bad:
            stats["dropped_by_rule"][msg] = stats["dropped_by_rule"].get(msg, 0) + 1
        return None, "rule"
    row = RW.new_row(task, seed_row, state, method, h, 0)
    if R.check_full_rows([row], task):
        return None, "row check"
    key = json.dumps(state, sort_keys=True)
    if key in seen:
        return None, "duplicate"
    seen.add(key)
    return row, None


class _Picker(object):
    """The seed of each generator call: first the retries again() queued, in order and each once (a given seed, or the
    next seed of a group); then a group drawn by its size (alloc "size") or evenly ("uniform") and the next seed of
    that group's seeded shuffle."""

    def __init__(self, groups, alloc, rng):
        self.keys = sorted(groups)
        self.weights = [len(groups[k]) if alloc == "size" else 1 for k in self.keys]
        self.order = dict((k, rng.sample(groups[k], len(groups[k]))) for k in self.keys)
        self.pos = dict((k, 0) for k in self.keys)
        self.rng = rng
        self.retries = []

    def again(self, key, item=None):
        self.retries.append((key, item))

    def _take(self, k):
        item = self.order[k][self.pos[k] % len(self.order[k])]
        self.pos[k] += 1
        return item

    def next(self):
        if self.retries:
            k, item = self.retries.pop(0)
            return k, item if item is not None else self._take(k)
        k = self.rng.choices(self.keys, weights=self.weights)[0]
        return k, self._take(k)


def _feedback(reasons):
    if not reasons:
        return ""
    return FEEDBACK % "; ".join(r.rstrip(". ") for r in reasons)


def new_stats():
    return {"short": None, "stopped": None, "stop_detail": None, "rounds": 0, "rejected_replies": 0,
            "rejected_pre_review": 0, "rejected_pre_review_by": {}, "rejected_by_review": 0, "surplus": 0,
            "unreviewed": 0, "dropped_by_rule": {}, "provenance": []}


def _generate(gen, rev, task, adapter, groups, n, h, method, prompt_of, by_seed, n_per_call, rounds, threshold,
              seed, alloc, stats):
    """The kept rows; feedback and retries by seed (by_seed: imitate) or by group (pattern)."""
    fz = featurizer(task)
    rng = random.Random(seed)
    pick = _Picker(groups, alloc, rng)
    rows, seen, feedback, cands, reviewed = [], set(), {}, [], 0
    try:
        for rnd in range(rounds):
            need = n - len(rows)
            if need <= 0:
                break
            stats["rounds"] = rnd + 1
            cands, reviewed = [], 0
            for _ in range(int(math.ceil(need * OVERSAMPLE[min(rnd, 1)] / float(n_per_call)))):
                key, (seed_row, fail) = pick.next()
                fkey = seed_row["id"] if by_seed else key
                res = gen.run(prompt_of(key, seed_row, fail, n_per_call, _feedback(feedback.get(fkey))),
                              RECORDS_SCHEMA, "gen")
                if res.output is None:
                    stats["rejected_replies"] += 1
                    continue
                for rec in res.output["records"][:n_per_call]:
                    row, why = _candidate(task, adapter, fz, seed_row, rec["changes"], method, h, seen, stats)
                    if why:
                        stats["rejected_pre_review"] += 1
                        by = stats["rejected_pre_review_by"]
                        by[why] = by.get(why, 0) + 1
                        continue
                    cands.append((key, fkey, seed_row, fail, row, rnd))
            reasons, retry = {}, {}
            for start in range(0, len(cands), BATCH):
                if len(rows) >= n:
                    break
                part = cands[start:start + BATCH]
                verdicts = review(rev, task, [c[4] for c in part])
                reviewed += len(part)
                for (key, fkey, seed_row, fail, row, r), (score, reason) in zip(part, verdicts):
                    if score is None:  # no verdict (a failed reply, a skipped record): not a rejection
                        stats["unreviewed"] += 1
                    elif score >= threshold:
                        if len(rows) >= n:
                            stats["surplus"] += 1
                            continue
                        rows.append(RW.new_row(task, seed_row, row["state"], method, h, len(rows)))
                        stats["provenance"].append({
                            "id": rows[-1]["id"], "method": method, "seed_id": seed_row["id"], "type": key,
                            "kind": fail.get("kind"), "score": score, "reason": reason, "round": r})
                    else:
                        stats["rejected_by_review"] += 1
                        retry.setdefault(fkey, (key, (seed_row, fail) if by_seed else None))
                        if reason:
                            reasons.setdefault(fkey, []).append(reason)
            stats["unreviewed"] += len(cands) - reviewed
            cands, reviewed = [], 0
            for k, new in reasons.items():
                merged = [x for x in feedback.get(k, []) if x not in new] + list(dict.fromkeys(new))
                feedback[k] = merged[-MAX_REASONS:]
            for key, item in retry.values():  # the next round starts with the rejected seeds (types)
                pick.again(key, item)
    except LlmBudgetExceeded as e:
        stats["stopped"], stats["stop_detail"] = "api budget", str(e)
        stats["unreviewed"] += len(cands) - reviewed
    stats["short"] = n - len(rows)
    return rows


def imitate(gen, rev, task, adapter, seeds, n, h, system, n_per_call=N_PER_CALL, rounds=ROUNDS, threshold=8,
            seed=0, alloc="size"):
    """(rows, stats) of llm_imitate: `seeds` {weakness type: [(pool row, failure record)]}; n rows with recipe sha12
    h; `system` is system_prompt's text. stats["provenance"] has one record per row."""
    fz = featurizer(task)

    def prompt_of(key, row, fail, k, feedback):
        return "%s\n\n%s" % (system, fill("imitate", seed=compact(fz.edit_view(R.state_of(row))),
                                          true_label=fail["true"], decision=fail["decision"],
                                          prob=_g(fail["p_decision"]), n=k, feedback=feedback))

    stats = new_stats()
    rows = _generate(gen, rev, task, adapter, seeds, n, h, "llm_imitate", prompt_of, True, n_per_call, rounds,
                     threshold, seed, alloc, stats)
    return rows, stats


def kind_text(task, true, decision):
    d = task["decision"]
    text = 'the true label is "%s" but the model decided "%s"' % (true, decision)
    if d["type"] == "binary" and d.get("positive") in (true, decision):
        text += " (false %s)" % ("negatives" if true == d["positive"] else "positives")
    return text


def pattern(gen, rev, task, adapter, clusters, n, h, system, ok_rows=(), n_per_call=N_PER_CALL, rounds=ROUNDS,
            threshold=8, seed=0, alloc="size"):
    """(rows, stats) of llm_pattern: `clusters` {weakness type: [(pool row, failure record)]}; `ok_rows` [(row, its
    label)] are correctly decided rows (the diagnosis's ok sample). One analysis call per type (MEMBERS members), then
    generation as imitate's with the type's pattern and strategy; a type whose analysis failed is not generated
    (stats["clusters"][type]["error"])."""
    fz = featurizer(task)
    rng = random.Random(seed)
    stats = new_stats()
    stats["clusters"] = {}
    analyses = {}
    try:
        for key in sorted(clusters):
            members = clusters[key]
            true, decision = members[0][1]["true"], members[0][1]["decision"]
            sample = rng.sample(members, min(MEMBERS, len(members)))
            records = "\n\n".join("Record %d (the model's probability for its decision: %s):\n%s"
                                  % (i, _g(f["p_decision"]), compact(fz.edit_view(R.state_of(r))))
                                  for i, (r, f) in enumerate(sample))
            views = dict((label, [fz.edit_view(R.state_of(r)) for r, l in ok_rows if l == label])
                         for label in (true, decision))
            res = gen.run("%s\n\n%s" % (system, fill("analysis", kind=kind_text(task, true, decision), records=records,
                                                    med_ok=medians(views[true]), med_other=medians(views[decision]))),
                          ANALYSIS_SCHEMA, "analysis")
            if res.output is None:
                stats["clusters"][key] = {"size": len(members), "error": res.error}
                continue
            analyses[key] = res.output
            stats["clusters"][key] = dict(res.output, size=len(members))
    except LlmBudgetExceeded as e:
        stats["stopped"], stats["stop_detail"] = "api budget", str(e)
    if stats["stopped"] or not analyses:
        stats["stopped"] = stats["stopped"] or "no type analysis"
        stats["short"] = n
        return [], stats

    def prompt_of(key, row, fail, k, feedback):
        a = analyses[key]
        return "%s\n\n%s" % (system, fill("pattern", pattern=a["pattern"], strategy=a["strategy"],
                                          seed=compact(fz.edit_view(R.state_of(row))), true_label=fail["true"],
                                          n=k, feedback=feedback))

    groups = dict((k, clusters[k]) for k in analyses)
    rows = _generate(gen, rev, task, adapter, groups, n, h, "llm_pattern", prompt_of, False, n_per_call, rounds,
                     threshold, seed, alloc, stats)
    return rows, stats
