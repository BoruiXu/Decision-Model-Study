"""Prompt variation (standard library only): `dgen variants` and `dgen paraphrase`.

A prompt-variant spec (rsi.variants; docs/interfaces/datasets.md, "Prompt-variant rows") rewords a row's question
(the instructions and the label meanings) and lays out its state differently (renamed keys, lists of objects as
tables) without changing what either says; the labels, the target and the verbatim request text stay the seed's.
ojh's onejev_rows adapter passes each row as it is and OneJev renders each row's own question (its training's
render_example reads row["question"]), so the instructions and criteria of a variant reach training as its state does.

- make_variants (`dgen variants`): for a seeded share of the rows that are no prompt variants, one variant each
  (rsi.variants.apply_spec), the specs taken in turn; each variant right after its seed. A row whose state the spec
  cannot lay out (a table over items with other keys) is skipped and counted.
- paraphrase (`dgen paraphrase`): a generator llm.claude.ClaudeCall (prompts/paraphrase.md) writes n English
  paraphrases of the task's instructions and label meanings, with --keys also new names for the state's keys (those
  a spec may rename: no verbatim path, no key the input format needs); an independent reviewer ClaudeCall
  (prompts/paraphrase_review.md) sees only each paraphrase next to the original, part by part, and scores its
  meaning-equivalence from 1 to 10; a paraphrase is kept at KEEP_SCORE or more. Each kept one becomes a spec whose id
  is "p" and the first 10 hex digits of its content's sha256. Both calls bill api_usd lines (llm:paraphrase,
  llm:paraphrase_review) to the brief's spend file and stop at budget_left.api_usd, counted over that file, which
  every LLM call of the dispatch shares. A reply that fails (not JSON, not the schema, a non-zero exit) keeps nothing
  and is no crash.
"""
import copy
import hashlib
import json
import os
import random

from rsi import rows as R
from rsi import variants as V

from . import DgenError

KEEP_SCORE = 8  # the reviewer's meaning-equivalence score a paraphrase needs (1-10)
KEY_ROWS = 1000  # --keys: the key paths are read from at most this many real rows (a seeded sample)
BATCH = 10  # paraphrases per review call
GEN_TAG, REVIEW_TAG = "paraphrase", "paraphrase_review"


def load_specs(path, task):
    """The specs of a JSON file (a list), each valid for the task (rsi.variants.check_spec) and its id used once."""
    from .inputs import read_json
    specs = read_json(path, "--specs")
    if not isinstance(specs, list) or not specs:
        raise DgenError("--specs %s: expected a non-empty JSON list of prompt-variant specs" % path)
    errors, ids = [], set()
    for i, s in enumerate(specs):
        errors += ["specs[%d]: %s" % (i, e) for e in V.check_spec(s, task)]
        sid = s.get("id") if isinstance(s, dict) else None
        if sid in ids:
            errors.append("specs[%d]: the id %s is used twice" % (i, sid))
        ids.add(sid)
    if errors:
        raise DgenError("--specs %s: %s" % (path, "; ".join(errors[:5])))
    return specs


def make_variants(rows, specs, task, share, seed):
    """(rows with the new variant rows, each right after its seed; summary). The seeds: round(share x the rows that are
    no prompt variants) of them (at least one), drawn with random.Random(seed); the j-th seed in row order gets
    specs[j % len(specs)]."""
    pool = [i for i, r in enumerate(rows) if not R.is_variant(r)]
    k = min(len(pool), max(1, int(round(share * len(pool))))) if pool else 0
    chosen = sorted(random.Random(seed).sample(pool, k))
    taken = set(r["id"] for r in rows)
    made, by_spec, skipped, examples = {}, {}, 0, []
    for j, i in enumerate(chosen):
        spec = specs[j % len(specs)]
        try:
            v = V.apply_spec(rows[i], spec, task)
        except ValueError as e:
            skipped += 1
            examples.append("%s under %s: %s" % (rows[i]["id"], spec["id"], e))
            continue
        if v["id"] in taken:
            skipped += 1
            examples.append("%s is in the rows already" % v["id"])
            continue
        made[i] = v
        by_spec[spec["id"]] = by_spec.get(spec["id"], 0) + 1
    out = []
    for i, r in enumerate(rows):
        out.append(r)
        if i in made:
            out.append(made[i])
    return out, {"rows": len(rows), "seeds_available": len(pool), "chosen": k, "variants": len(made),
                 "by_spec": dict((s["id"], by_spec.get(s["id"], 0)) for s in specs), "skipped": skipped,
                 "skipped_examples": examples[:5], "share": share, "seed": seed}


# ---- paraphrase ------------------------------------------------------------------------------------------------------

def _walk(value, at, seen):
    if isinstance(value, dict):
        for k, v in value.items():
            seen.setdefault(at + (k,), None)
            _walk(v, at + (k,), seen)
    elif isinstance(value, list):
        for v in value:
            _walk(v, at + (None,), seen)


def key_paths(rows, task, seed=0):
    """The state key paths ("tx[].amount") of the rows (a seeded sample of KEY_ROWS when more), in first-seen order,
    that a spec may rename (rsi.variants.check_spec: no verbatim path, no key the task's input format needs)."""
    sample = rows if len(rows) <= KEY_ROWS else random.Random(seed).sample(rows, KEY_ROWS)
    seen = {}
    for r in sample:
        _walk(R.state_of(r), (), seen)
    out = []
    for parts in seen:
        path = V.path_text(parts)
        if V.parse_path(path) == parts and not V.check_spec(
                {"id": "k", "instructions": None, "criteria": None, "rename": {path: "x"}, "tables": None}, task):
            out.append(path)
    return out


def _schema(keys):
    item = {"type": "object", "required": ["instructions", "label_meanings"], "additionalProperties": False,
            "properties": {"instructions": {"type": "string"}, "label_meanings": {"type": "array", "items": {
                "type": "object", "required": ["label", "meaning"], "additionalProperties": False,
                "properties": {"label": {"type": "string"}, "meaning": {"type": "string"}}}}}}
    if keys:
        item["required"].append("keys")
        item["properties"]["keys"] = {"type": "array", "items": {
            "type": "object", "required": ["path", "name"], "additionalProperties": False,
            "properties": {"path": {"type": "string"}, "name": {"type": "string"}}}}
    return {"type": "object", "required": ["paraphrases"], "additionalProperties": False,
            "properties": {"paraphrases": {"type": "array", "items": item}}}


def _quote(text):
    return json.dumps(text, ensure_ascii=False)


def _last(path):
    return V.parse_path(path)[-1]


def generator_prompt(task, n, keys):
    from .llm_gen import fill
    d = task["decision"]
    lines = ""
    if keys:
        lines = ("\nThe names of the record's fields, by path (\"[]\" stands for every item of a list), with the name "
                 "each has now:\n%s\n" % "\n".join("- %s: %s" % (p, _quote(_last(p))) for p in keys))
    return fill("paraphrase", input_description=task["input"]["description"], instructions=d["instructions"],
                labels="\n".join("- %s: %s" % (l, d["label_meanings"][l]) for l in d["labels"]), keys=lines, n=n,
                key_rules="- with the field names: gives every field a short plain-English name for what its name "
                          "now says, adding nothing it does not say (no guessed unit or meaning), unique among the "
                          "fields next to it (the same parent); a name may stay as it is.\n" if keys else "",
                key_answer=', "keys": [{"path": "<field path>", "name": "<its new name>"}]' if keys else "",
                key_once=" and every field path once" if keys else "")


def spec_of(p, task, keys):
    """(spec, None) of a generator's paraphrase `p`, or (None, why it is no spec)."""
    d = task["decision"]
    meanings = dict((x["label"], x["meaning"].strip()) for x in p["label_meanings"])
    if len(p["label_meanings"]) != len(d["labels"]) or set(meanings) != set(d["labels"]):
        return None, "labels"
    rename = None
    if keys:
        names = dict((x["path"], x["name"].strip()) for x in p.get("keys") or [])
        if len(p.get("keys") or []) != len(keys) or set(names) != set(keys):
            return None, "keys"
        rename = dict((path, name) for path, name in names.items() if name != _last(path)) or None
    t = copy.deepcopy(task)
    t["decision"]["label_meanings"] = meanings
    criteria = R.question_of(t)["criteria"]
    instructions = p["instructions"].strip()
    spec = {"instructions": None if instructions == d["instructions"] else instructions,
            "criteria": None if criteria == R.question_of(task)["criteria"] else criteria,
            "rename": rename, "tables": None}
    spec = dict(id="p" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:10], **spec)
    errors = V.check_spec(spec, task)
    return (None, "invalid: %s" % errors[0]) if errors else (spec, None)


def review_text(task, specs):
    """The reviewer's records: each spec next to the original, part by part (nothing of the generator's prompt)."""
    q = R.question_of(task)
    out = []
    for i, s in enumerate(specs):
        vq = V.question(s, task)
        lines = ["Rewording %d:" % i, "- The decision. Original: %s Rewording: %s"
                 % (_quote(q["instructions"]), _quote(vq["instructions"]))]
        crit = q["criteria"]
        names = list(crit) if isinstance(crit, dict) else range(len(crit))
        for name in names:
            label = {"true": "yes", "false": "no"}.get(name, name) if q["type"] == "noul" else name
            lines.append("- The meaning of the label %s. Original: %s Rewording: %s"
                         % (_quote(str(label)), _quote(crit[name]), _quote(vq["criteria"][name])))
        for path, name in sorted((s["rename"] or {}).items()):
            lines.append("- The name of the field %s. Original: %s Rewording: %s" % (path, _quote(_last(path)),
                                                                                    _quote(name)))
        out.append("\n".join(lines))
    return "\n\n".join(out)


def paraphrase(inputs, n, keys_wanted, out, workdir):
    """Write the kept specs (a JSON list) to `out`; return the summary."""
    from llm.claude import ClaudeCall, LlmBudgetExceeded
    from .cli import write_json
    from .llm_gen import REVIEW_SCHEMA, fill
    task = inputs.task
    keys = key_paths(inputs.real(), task) if keys_wanted else []
    if keys_wanted and not keys:
        raise DgenError("--keys: the real rows' states have no key a prompt variant may rename")
    api_left = (inputs.brief.get("budget_left") or {}).get("api_usd")
    gen = ClaudeCall(task, os.path.join(workdir, "gen"), inputs.spend_file, None, api_left=api_left)
    rev = ClaudeCall(task, os.path.join(workdir, "review"), inputs.spend_file, None, api_left=api_left)
    gen.shared, rev.shared = rev, gen
    summary = {"requested": n, "keys": len(keys) if keys_wanted else None, "generated": 0, "invalid": {},
               "reviewed": 0, "kept": 0, "rejected_by_review": 0, "unreviewed": 0, "stopped": None,
               "stop_detail": None, "generator_error": None, "scores": [], "llm_dir": workdir}
    specs, kept = [], []
    try:
        res = gen.run(generator_prompt(task, n, keys), _schema(keys), GEN_TAG)
        if res.output is None:
            summary["generator_error"] = res.error
        for p in (res.output or {}).get("paraphrases", [])[:n]:
            summary["generated"] += 1
            spec, why = spec_of(p, task, keys)
            if spec is not None and any(s["id"] == spec["id"] for s in specs):
                spec, why = None, "duplicate"
            if why:
                summary["invalid"][why] = summary["invalid"].get(why, 0) + 1
            else:
                specs.append(spec)
        for start in range(0, len(specs), BATCH):
            part = specs[start:start + BATCH]
            r = rev.run(fill("paraphrase_review", input_description=task["input"]["description"],
                             paraphrases=review_text(task, part)), REVIEW_SCHEMA, REVIEW_TAG)
            got = {}
            for x in (r.output or {}).get("scores", []):
                if 0 <= x["i"] < len(part) and x["i"] not in got and 1 <= x["score"] <= 10:
                    got[x["i"]] = (x["score"], " ".join(x["reason"].split()))
            for i, s in enumerate(part):
                score, reason = got.get(i, (None, "no review (%s)" % r.error if r.output is None else
                                            "the reviewer gave no score"))
                summary["scores"].append({"id": s["id"], "score": score, "reason": reason})
                if score is None:
                    summary["unreviewed"] += 1
                    continue
                summary["reviewed"] += 1
                if score >= KEEP_SCORE:
                    kept.append(s)
                else:
                    summary["rejected_by_review"] += 1
    except LlmBudgetExceeded as e:
        summary["stopped"], summary["stop_detail"] = "api budget", str(e)
        summary["unreviewed"] += len(specs) - len(summary["scores"])  # the ones the stop left without a score
    write_json(out, kept)
    summary.update(kept=len(kept), ids=[s["id"] for s in kept], specs=out,
                   calls={"gen": gen.calls, "review": rev.calls},
                   usd={"gen": round(gen.spent, 6), "review": round(rev.spent, 6)})
    return summary
