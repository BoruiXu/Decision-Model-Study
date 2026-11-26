"""python -m dgen <command> ...   (PYTHONPATH=<repo>/modules/data:<repo>/modules/common:<repo>, the ojh venv's Python:
the brief's `dgen` gives the prefix)

score-pool  --brief B --out D --marker M [--max-rows N] [--seed S] [--card-wait SECONDS]
            the pool rows whose id and input the current trainset does not have (at most N, drawn with seed S) as
            D/inputs.jsonl (no labels), scored with the best version's candidate's model.predict into
            D/pool_predictions.jsonl (its output in D/score-pool.log; who scored it in D/score_pool.json). A long job
            (GPU): it writes M.pid first (its own pid and start time), runs model.predict on CUDA_VISIBLE_DEVICES when
            set, else on the idle card with the least memory in use among the brief's resources.gpu_ids (waiting up
            to --card-wait, 600 s; without any nvidia-smi it chooses none, a hung or failing one is an error), writes
            one spend line {"at", "source": "dgen:score-pool", "compute_hours"} into the brief's spend_file (also when
            model.predict fails or the job is stopped), and the marker M last (`ok`, or `failed` and the reason). It
            refuses to start when budget_left.compute_hours is 0. Start it in the background and hand back long_job
            with M.
diagnose    --brief B --scores D/pool_predictions.jsonl --out D [--seed S]
            the candidate's errors on those rows (hard label != its decision): diagnosis.json, failures.jsonl,
            ok_sample.jsonl, weakness.json, learnability.json, diagnosis.md in D
generate    --brief B --recipe R.json [--diagnosis D] --out A [--marker M]
            A/arms/<sha12>/{recipe.json, synthetic.jsonl, provenance.jsonl, stats.json} (--diagnosis: needed by the
            targeted methods); an arm that is already there, made from the same diagnosis and trainset, is not made
            again (from another one: refused); rows whose ids the trainset already holds are refused (for llm_imitate
            and llm_pattern before any LLM call). The LLM methods call `claude -p` (the task's runner.claude command)
            for a generator and an independent reviewer, keep each call in A/llm/<sha12>/{gen,review}/<n>/ with
            llm_calls.jsonl beside, and bill every call as an api_usd line of the brief's spend_file; they stop short
            (stats.json stopped "api budget") at the recipe's max_usd or the brief's budget_left.api_usd. An LLM
            recipe is a long job and needs --marker: as score-pool, it writes M.pid first and the marker M last (`ok`,
            or `failed` and the reason); stopped (SIGTERM), it ends the `claude -p` call in flight, bills it and fails.
            Start it in the background and hand back long_job with M. --marker is optional for the other methods.
filter      --brief B --arm A/arms/<sha12> [--filter rules,dedup,cllm] [--cllm-threshold 0.2]
            every check, the --filter ones removing rows: verified.jsonl, flags.jsonl, quality_summary.json,
            quality_report.md and sample.jsonl (50 kept rows, stratified by label, for the record) in the arm
neardup     --brief B --rows R [--eps-ratio 0.05] [--write OUT]
            the near-duplicate groups of the full rows R (e.g. a trainset's rows.jsonl): rows whose z-space distance
            (the space fitted on the brief's real trainset and pool rows) is at most eps-ratio times the median
            nearest-neighbour distance of R's rows, joined into groups; prompt-variant rows are left out of it.
            --write OUT (a new file): R without the near duplicates, the first row of each group kept
paraphrase  --brief B --n N --out S.json [--keys] [--marker M]
            N English paraphrases of the task's instructions and label meanings (with --keys also new names for the
            state's keys) from a generator `claude -p` call; an independent reviewer call sees each next to the
            original and scores its meaning-equivalence 1-10; those scoring 8 or more are written to S.json (a new
            file) as prompt-variant specs. Both calls are billed as api_usd lines of the brief's spend_file
            (llm:paraphrase, llm:paraphrase_review) and stop at budget_left.api_usd; they are kept in
            <S's directory>/llm/paraphrase-<S's name>/{gen,review}/. --marker M runs it as a long job (M.pid first,
            the marker M last), to start in the background.
variants    --task T --rows R --specs S.json --share F [--seed 0] --out O
            O (a new file): the rows R, and right after each of a seeded share F of R's rows that are no prompt
            variants one variant (the specs of S.json in turn; rsi.variants.apply_spec). A trainset holding them is
            then written by `dset check --rows O --variants S.json [--variants <the trainset's manifest.json>] --write
            <dir> --mode drop --round <n> --parent <the current trainset's id>`, which lists the specs in the manifest's
            prompt_variants.

A trainset is then built from verified rows: `dset grow --task <task.json> --pool <pool> --trainset <current>
--parent <its id> --with <arm>/verified.jsonl --out <dir> --round <n>`.

Every command prints one JSON summary on stdout. Exit codes: 0 done; 1 an error in the input (stdout one JSON line
{"error": <reason>}, stderr `dgen: error: <reason>`), e.g. a recipe that needs numpy or scikit-learn where they do
not import. dgen reads only the brief's inputs and overwrites no diagnosis or arm.
"""
import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time

from rsi import rows as R

from . import DGEN_VERSION, DgenError, need_ml
from .inputs import SCORES_META, Inputs, read_json, read_jsonl, sha256_file

COMMANDS = ("score-pool", "diagnose", "generate", "filter", "neardup", "paraphrase", "variants")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise DgenError("%s (see %s --help)" % (message, self.prog))


def _count(text):
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1 (got %s)" % text)
    return n


def _seconds(text):
    try:
        x = float(text)
    except ValueError:
        x = -1.0
    if not 0 <= x < float("inf"):
        raise argparse.ArgumentTypeError("expected a number of seconds >= 0 (got %s)" % text)
    return x


def _ratio(text):
    try:
        x = float(text)
    except ValueError:
        x = -1.0
    if not 0 < x < float("inf"):
        raise argparse.ArgumentTypeError("expected a number > 0 (got %s)" % text)
    return x


def _share(text):
    try:
        x = float(text)
    except ValueError:
        x = -1.0
    if not 0 < x <= 0.5:
        raise argparse.ArgumentTypeError("expected a number in (0, 0.5] (got %s)" % text)
    return x


def _fraction(text):
    try:
        x = float(text)
    except ValueError:
        x = -1.0
    if not 0 < x <= 1:
        raise argparse.ArgumentTypeError("expected a number in (0, 1] (got %s)" % text)
    return x


def build_parser():
    p = _Parser(prog="dgen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version="dgen %s" % DGEN_VERSION)
    sub = p.add_subparsers(dest="cmd", metavar="command", required=True)
    s = sub.add_parser("score-pool", help="score the pool rows outside the trainset with the best candidate")
    s.add_argument("--brief", required=True)
    s.add_argument("--out", required=True, help="where inputs.jsonl and pool_predictions.jsonl go")
    s.add_argument("--marker", required=True, help="the long job's completion marker (M.pid is written first)")
    s.add_argument("--max-rows", type=_count, help="score at most this many rows, drawn with --seed")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--card-wait", type=_seconds, default=600.0,
                   help="how long to wait for an idle card of resources.gpu_ids (without CUDA_VISIBLE_DEVICES)")
    s.set_defaults(fn=cmd_score_pool)
    d = sub.add_parser("diagnose", help="errors, weakness types, learnability and findings of the pool scores")
    d.add_argument("--brief", required=True)
    d.add_argument("--scores", required=True, help="pool_predictions.jsonl of score-pool")
    d.add_argument("--out", required=True, help="the diagnosis directory (no diagnosis.json in it yet)")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_diagnose)
    g = sub.add_parser("generate", help="the rows of one recipe (one arm)")
    g.add_argument("--brief", required=True)
    g.add_argument("--recipe", required=True, help="the recipe (JSON)")
    g.add_argument("--diagnosis", help="the diagnosis directory (for the targeted methods)")
    g.add_argument("--out", required=True, help="the arm goes to <out>/arms/<sha12>")
    g.add_argument("--marker", help="run as a long job with this completion marker (M.pid is written first); "
                                    "required by llm_imitate and llm_pattern")
    g.set_defaults(fn=cmd_generate)
    f = sub.add_parser("filter", help="the quality checks of an arm and its verified rows")
    f.add_argument("--brief", required=True)
    f.add_argument("--arm", required=True)
    f.add_argument("--filter", default="rules,dedup,cllm", help="the checks that remove rows (comma separated)")
    f.add_argument("--cllm-threshold", type=_share, default=0.2)
    f.set_defaults(fn=cmd_filter)
    n = sub.add_parser("neardup", help="near-duplicate groups of rows in the z-space; --write keeps one row a group")
    n.add_argument("--brief", required=True)
    n.add_argument("--rows", required=True, help="full rows (JSONL)")
    n.add_argument("--eps-ratio", type=_ratio, default=0.05,
                   help="near duplicates: distance <= this times the median nearest-neighbour distance")
    n.add_argument("--write", help="the rows without the near duplicates (JSONL; must not exist)")
    n.set_defaults(fn=cmd_neardup)
    q = sub.add_parser("paraphrase", help="reviewed English paraphrases of the task's prompt, as prompt-variant specs")
    q.add_argument("--brief", required=True)
    q.add_argument("--n", type=_count, required=True, help="how many paraphrases to ask for")
    q.add_argument("--out", required=True, help="the specs (JSON; must not exist)")
    q.add_argument("--keys", action="store_true", help="also new names for the state's keys")
    q.add_argument("--marker", help="run as a long job with this completion marker (M.pid is written first)")
    q.set_defaults(fn=cmd_paraphrase)
    v = sub.add_parser("variants", help="add prompt-variant rows for a seeded share of the rows")
    v.add_argument("--task", required=True, help="task.json")
    v.add_argument("--rows", required=True, help="full rows (JSONL), e.g. a trainset's rows.jsonl")
    v.add_argument("--specs", required=True, help="the prompt-variant specs (JSON list), e.g. dgen paraphrase's")
    v.add_argument("--share", type=_fraction, required=True, help="the share of the rows that get a variant")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", required=True, help="the rows with their variants (JSONL; must not exist)")
    v.set_defaults(fn=cmd_variants)
    return p


def _jsonable(o):
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError("%r is not JSON serializable" % (o,))


def dumps(obj, indent=None):
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=_jsonable)


def _replace(path, text):
    tmp = "%s.tmp-%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path, obj):
    _replace(path, dumps(obj, indent=1) + "\n")


def write_jsonl(path, rows):
    _replace(path, "".join(dumps(r) + "\n" for r in rows))


def write_file(path, obj):
    if path.endswith(".jsonl"):
        write_jsonl(path, obj)
    elif path.endswith(".json"):
        write_json(path, obj)
    else:
        _replace(path, obj)


def spend(spend_file, source, hours, note=None):
    """One spend line (docs/interfaces/spend.md), appended in one write; nothing without a spend file."""
    if not spend_file:
        return
    line = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": source,
            "compute_hours": max(0.0, float(hours))}
    if note:
        line["note"] = note
    os.makedirs(os.path.dirname(os.path.abspath(spend_file)), exist_ok=True)
    with open(spend_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        summary = args.fn(args)
    except (DgenError, OSError) as e:  # an OSError: a file that cannot be read or written
        sys.stdout.write(dumps({"error": str(e)}) + "\n")
        sys.stderr.write("dgen: error: %s\n" % e)
        return 1
    sys.stdout.write(dumps(summary, indent=2) + "\n")
    return 0


# ---- score-pool ------------------------------------------------------------------------------------------------------

def _marker(marker, status, text):
    _replace(marker, "%s\n%s\n" % (status, " ".join(str(text).split())))


class Stopped(DgenError):
    """A long job (score-pool, an LLM generate) got SIGTERM (the run was stopped): its child process (model.predict,
    the `claude -p` call in flight) is ended and what it spent so far is billed."""


_PHASE = ["starting"]  # what the long job is doing, for the stop's reason


def _stop(signum, frame):
    raise Stopped("stopped by signal %d while %s" % (signum, _PHASE[0]))


def _block_stop():
    """Hold SIGTERM back (a stop that arrives meanwhile is delivered once unblocked) so a spend line or the marker is
    written whole; the mask to restore, None where signals cannot be blocked."""
    if hasattr(signal, "pthread_sigmask"):
        return signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    return None


def _unblock(mask):
    if mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def long_job(marker, run, ok_text):
    """run(marker) as a long job: its own process group (the engine's stop, a SIGTERM to the group of <marker>.pid's
    pid, reaches the children in it), SIGTERM raising Stopped, M.pid (pid and start time) written first and the marker
    last: `ok` and ok_text(summary), or `failed` and the reason."""
    _PHASE[0] = "starting"
    marker = os.path.abspath(marker)
    if hasattr(os, "setpgrp") and os.getpgrp() != os.getpid():
        try:
            os.setpgrp()
        except OSError:
            pass
    signal.signal(signal.SIGTERM, _stop)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    if os.path.exists(marker):  # an older run's: this run's end is the marker it writes itself
        os.remove(marker)
    _replace(marker + ".pid", "%d\n%d\n" % (os.getpid(), int(time.time())))
    try:
        summary = run(marker)
    except BaseException as e:
        _block_stop()  # the marker is the job's last word: a stop now cannot cut it (the process ends right after)
        _marker(marker, "failed", "%s: %s" % (type(e).__name__, e) if not isinstance(e, DgenError) else e)
        raise
    _block_stop()
    _marker(marker, "ok", ok_text(summary))
    return summary


def cmd_score_pool(a):
    return long_job(a.marker, lambda marker: _score_pool(a, marker),
                    lambda s: "%d pool rows scored by %s in %.4g GPU hours" % (s["rows"], s["candidate"],
                                                                               s["compute_hours"]))


def _tail(path, n=3):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [x.strip() for x in f.read().splitlines() if x.strip()]
    except OSError:
        return "no output"
    return " | ".join(lines[-n:])[-500:] or "no output"


def _no_gpu_hours(inputs):
    left = (inputs.brief.get("budget_left") or {}).get("compute_hours")
    if isinstance(left, (int, float)) and not isinstance(left, bool) and left <= 0:
        raise DgenError("no GPU hours left (the brief's budget_left.compute_hours is %s): score-pool spends GPU hours, "
                        "so it does not start; ask the user (over budget) or generate without a diagnosis (smote, "
                        "borderline)" % left)


def _score_pool(a, marker):
    from .card import pick_card
    inputs = Inputs(a.brief)
    _no_gpu_hours(inputs)
    cdir, cm, entry = inputs.candidate()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    inputs_path, preds_path = os.path.join(out, "inputs.jsonl"), os.path.join(out, "pool_predictions.jsonl")
    if os.path.exists(preds_path):
        raise DgenError("%s exists: dgen overwrites no scores; score into a new --out" % preds_path)
    held = inputs.trainset_all  # the generated rows too: never scored as pool rows
    ids = set(r["id"] for r in held)
    fps = set(R.input_fingerprint(R.state_of(r)) for r in held)
    pool = inputs.pool_rows
    free = [r for r in pool if r["id"] not in ids and R.input_fingerprint(R.state_of(r)) not in fps]
    if not free:
        raise DgenError("the pool has no row whose id and input the current trainset does not have: nothing to score")
    outside = len(free)
    if a.max_rows is not None and a.max_rows < len(free):
        free = [free[i] for i in sorted(random.Random(a.seed).sample(range(len(free)), a.max_rows))]
    write_jsonl(inputs_path, [R.input_row_of(r) for r in free])
    env, card = dict(os.environ), None
    if "CUDA_VISIBLE_DEVICES" not in env:  # the caller's choice stands; else an idle card of resources.gpu_ids
        _PHASE[0] = "waiting for a card"
        card = pick_card((inputs.brief.get("resources") or {}).get("gpu_ids"), a.card_wait)
        if card is not None:
            env["CUDA_VISIBLE_DEVICES"] = card
    cuda = env.get("CUDA_VISIBLE_DEVICES")  # what model.predict runs with: the caller's, the card chosen, or none
    device = "a card chosen among resources.gpu_ids" if card else "the caller's" if cuda is not None else \
        "no card chosen: no nvidia-smi"
    tmp = preds_path + ".tmp-%d" % os.getpid()
    log = os.path.join(out, "score-pool.log")
    name = cm.get("id") or cdir
    t0, code, stopped = time.time(), None, False
    try:
        try:
            _PHASE[0] = "scoring"
            with open(log, "ab") as f:
                code = subprocess.call([entry, "--inputs", inputs_path, "--out", tmp], stdin=subprocess.DEVNULL,
                                       stdout=f, stderr=subprocess.STDOUT, env=env)
        except Stopped:
            stopped = True
            raise
        finally:  # billed however it ends: done, failed or stopped; a stop waits until the line is written
            hours = (time.time() - t0) / 3600.0
            mask = _block_stop()
            try:
                spend(inputs.spend_file, "dgen:score-pool", hours, "model.predict of %s on %d pool rows, "
                      "CUDA_VISIBLE_DEVICES=%s (%s)%s" % (
                          name, len(free), "unset" if cuda is None else cuda, device,
                          "" if code == 0 else " (stopped)" if stopped
                          else " (failed, exit %d)" % code if code is not None else " (did not run)"))
            finally:
                _unblock(mask)
        _PHASE[0] = "writing the scores"
        if code != 0:
            raise DgenError("model.predict of %s failed (exit %d): %s (log %s)" % (cdir, code, _tail(log), log))
        preds = read_jsonl(tmp, "model.predict's output")
        errors = R.check_predictions(preds, inputs.task, [r["id"] for r in free])
        if errors:
            raise DgenError("model.predict's output: %d problems: %s" % (len(errors), "; ".join(errors[:3])))
        write_json(os.path.join(out, SCORES_META), {
            "dgen_version": DGEN_VERSION, "candidate": cm.get("id"), "candidate_dir": cdir, "rows": len(free),
            "predictions_sha256": sha256_file(tmp), "inputs_sha256": sha256_file(inputs_path), "card": card,
            "cuda_visible_devices": cuda, "device": device, "compute_hours": hours,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        os.replace(tmp, preds_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"rows": len(free), "pool_rows": len(pool), "outside_trainset": outside, "candidate": cm.get("id"),
            "card": card, "cuda_visible_devices": cuda, "device": device, "compute_hours": hours,
            "inputs": inputs_path, "predictions": preds_path, "marker": marker}


# ---- diagnose --------------------------------------------------------------------------------------------------------

DIAGNOSIS_FILES = ("failures.jsonl", "ok_sample.jsonl", "weakness.json", "learnability.json", "diagnosis.md",
                   "diagnosis.json")  # diagnosis.json last: a directory that has it is complete


def cmd_diagnose(a):
    inputs = Inputs(a.brief)
    need_ml("dgen diagnose")
    from . import diagnose as DG
    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "diagnosis.json")):
        raise DgenError("%s already holds a diagnosis: dgen overwrites none; diagnose into a new --out" % out)
    files = DG.run(inputs, a.scores, a.seed)
    os.makedirs(out, exist_ok=True)
    for name in DIAGNOSIS_FILES:
        write_file(os.path.join(out, name), files[name])
    d = files["diagnosis.json"]
    return {"rows": d["rows"], "errors": dict((k, e["n"]) for k, e in d["errors"].items()),
            "types": [{"key": t["key"], "size": t["size"],
                       "signature": [s["feature"] for s in t["signature"][:3]], "cllm_hard": t["cllm_hard"]}
                      for t in d["types"]],
            "findings": [f["issue"] for f in d["findings"]], "dir": out}


def load_diagnosis(d):
    if not os.path.isfile(os.path.join(d, "diagnosis.json")):
        raise DgenError("--diagnosis %s: no diagnosis.json (run dgen diagnose into it first)" % d)
    return {"sha256": sha256_file(os.path.join(d, "diagnosis.json")),
            "failures": read_jsonl(os.path.join(d, "failures.jsonl"), "--diagnosis"),
            "weakness": read_json(os.path.join(d, "weakness.json"), "--diagnosis"),
            "learnability": read_json(os.path.join(d, "learnability.json"), "--diagnosis"),
            "ok_sample": read_jsonl(os.path.join(d, "ok_sample.jsonl"), "--diagnosis")}


# ---- generate --------------------------------------------------------------------------------------------------------

def _arm_summary(arm, recipe, stats, existing):
    return {"arm": arm, "recipe": recipe["sha12"], "name": recipe["name"], "method": recipe["method"],
            "requested": stats.get("requested"), "n": stats.get("n"), "short": stats.get("short"),
            "dropped_by_rule": sum((stats.get("dropped_by_rule") or {}).values()),
            "skipped_structure": stats.get("skipped_structure"), "stopped": stats.get("stopped"),
            "existing": existing}


def cmd_generate(a):
    if a.marker:
        return long_job(a.marker, lambda marker: _generate(a),
                        lambda s: "arm %s: %s of %s rows%s%s" % (
                            s["recipe"], s["n"], s["requested"], ", short %s" % s["short"] if s["short"] else "",
                            ", stopped: %s" % s["stopped"] if s["stopped"] else ""))
    return _generate(a)


def _generate(a):
    from . import recipes as RC
    from .rows import task_adapter
    inputs = Inputs(a.brief)
    recipe = RC.normalize(read_json(a.recipe, "--recipe"), inputs.task)
    if recipe["method"] in RC.LLM and not a.marker:
        raise DgenError("method %s is a long job (LLM calls): give --marker M, start it in the background and hand "
                        "back long_job with M" % recipe["method"])
    h = RC.sha12(recipe)
    arms = os.path.join(os.path.abspath(a.out), "arms")
    arm = os.path.join(arms, h)
    diagnosis = load_diagnosis(a.diagnosis) if a.diagnosis else None
    made_from = {"diagnosis_sha256": diagnosis["sha256"] if diagnosis and recipe["method"] in RC.TARGETED else None,
                 "trainset": (inputs.brief.get("ids") or {}).get("trainset")}
    if os.path.isdir(arm):
        old = read_json(os.path.join(arm, "recipe.json"), "the arm")
        if old.get("made_from") != made_from:
            raise DgenError("arm %s exists, made from %s; this run has %s (another diagnosis or trainset): generate "
                            "into a new --out, or change the recipe's seed" % (arm, json.dumps(old.get("made_from")),
                                                                               json.dumps(made_from)))
        return _arm_summary(arm, old, read_json(os.path.join(arm, "stats.json"), "the arm"), True)
    taken = set(r["id"] for r in inputs.trainset_all)
    if recipe["method"] in RC.LLM:  # their ids are known before any call: refuse before money is spent
        from .rows import row_id
        clash = [i for i in (row_id(h, recipe["method"], k) for k in range(recipe["n"])) if i in taken]
        if clash:
            raise DgenError("%d rows would have ids the trainset already holds, e.g. %s: an earlier round generated "
                            "this recipe; change the recipe's seed" % (len(clash), clash[0]))
    ctx = RC.Context(inputs, diagnosis, task_adapter(inputs.task),
                     os.path.join(os.path.abspath(a.out), "llm", h) if recipe["method"] in RC.LLM else None)
    _PHASE[0] = "generating %s rows" % recipe["method"]
    rows, prov, stats = RC.generate(ctx, recipe, h)
    _PHASE[0] = "writing the arm"
    clash = [r["id"] for r in rows if r["id"] in taken]
    if clash:
        raise DgenError("%d rows have ids the trainset already holds, e.g. %s: %s" % (
            len(clash), clash[0], "the diagnosis is older than the trainset: score and diagnose the pool again"
            if recipe["method"] == "hard" else "an earlier round generated this recipe; change the recipe's seed"))
    stats.update(dgen_version=DGEN_VERSION, recipe=h, name=recipe["name"])
    recipe = dict(recipe, sha12=h, source=os.path.abspath(a.recipe),
                  diagnosis=os.path.abspath(a.diagnosis) if a.diagnosis else None, made_from=made_from)
    os.makedirs(arms, exist_ok=True)
    staging = os.path.join(arms, ".%s.tmp-%d" % (h, os.getpid()))
    os.makedirs(staging)
    try:
        write_json(os.path.join(staging, "recipe.json"), recipe)
        write_jsonl(os.path.join(staging, "synthetic.jsonl"), rows)
        write_jsonl(os.path.join(staging, "provenance.jsonl"), prov)
        write_json(os.path.join(staging, "stats.json"), stats)
        os.rename(staging, arm)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _arm_summary(arm, recipe, stats, False)


# ---- filter ----------------------------------------------------------------------------------------------------------

def cmd_filter(a):
    inputs = Inputs(a.brief)
    need_ml("dgen filter")
    from . import quality as Q
    filt = [x.strip() for x in a.filter.split(",") if x.strip()]
    bad = [x for x in filt if x not in Q.FILTERABLE]
    if bad:
        raise DgenError("--filter: %s not among the checks that can remove rows (%s; discriminability is report "
                        "only)" % (", ".join(bad), ", ".join(Q.FILTERABLE)))
    from .rows import task_adapter
    from .space import Space
    arm = os.path.abspath(a.arm)
    recipe = read_json(os.path.join(arm, "recipe.json"), "--arm")
    stats = read_json(os.path.join(arm, "stats.json"), "--arm")
    rows = read_jsonl(os.path.join(arm, "synthetic.jsonl"), "--arm")
    prov = dict((p["id"], p) for p in read_jsonl(os.path.join(arm, "provenance.jsonl"), "--arm"))
    real = inputs.real()
    space = Space(inputs.task).fit(real, [inputs.label(r) for r in real])
    kept, flags, summ = Q.run_checks(inputs, space, task_adapter(inputs.task), rows, prov, filt, a.cllm_threshold)
    summ.update(recipe=recipe.get("sha12"), method=recipe.get("method"))
    seeds = dict((r["id"], r) for r in real)
    write_jsonl(os.path.join(arm, "verified.jsonl"), kept)
    write_jsonl(os.path.join(arm, "flags.jsonl"), flags)
    write_jsonl(os.path.join(arm, "sample.jsonl"), Q.sample(kept, flags, seeds, inputs.label, recipe.get("seed", 0)))
    write_file(os.path.join(arm, "quality_report.md"), Q.report(recipe, stats, summ))
    write_json(os.path.join(arm, "quality_summary.json"), summ)
    return {"arm": arm, "n": summ["n"], "kept": summ["kept"], "removed": summ["removed"],
            "removed_by": summ["removed_by"], "kept_by_label": summ["kept_by_label"], "leak_val": summ["leak_val"],
            "verified": os.path.join(arm, "verified.jsonl")}


# ---- neardup ---------------------------------------------------------------------------------------------------------

def cmd_neardup(a):
    inputs = Inputs(a.brief)
    need_ml("dgen neardup")
    from features.table import real_rows
    from . import neardup as ND
    from .space import Space
    out = os.path.abspath(a.write) if a.write else None
    if out and os.path.lexists(out):
        raise DgenError("--write %s already exists: dgen overwrites nothing; write to a new file" % a.write)
    rows = read_jsonl(a.rows, "--rows")
    errors = R.check_full_rows(rows, inputs.task, variants=[])  # a trainset's prompt variants: left out below
    if errors:
        raise DgenError("--rows %s: %d problems in the rows: %s" % (a.rows, len(errors), "; ".join(errors[:3])))
    real = real_rows(rows)  # a prompt variant is its seed's state renamed: not in the space
    space = Space(inputs.task).fit(inputs.real())
    found, median, radius = ND.groups(space.z(real), a.eps_ratio)
    groups = [[real[i]["id"] for i in g] for g in found]
    dropped = set(i for g in groups for i in g[1:])
    summary = {"rows": len(rows), "variants": len(rows) - len(real), "eps_ratio": a.eps_ratio, "median_nn": median,
               "radius": radius, "groups": groups, "rows_in_groups": sum(len(g) for g in groups),
               "dropped": len(dropped), "wrote": None}
    if out:
        write_jsonl(out, [r for r in rows if r["id"] not in dropped])
        summary["wrote"] = out
    return summary


# ---- paraphrase, variants --------------------------------------------------------------------------------------------

def _new_file(path, what):
    if os.path.lexists(path):
        raise DgenError("%s %s already exists: dgen overwrites nothing; write to a new file" % (what, path))
    return os.path.abspath(path)


def cmd_paraphrase(a):
    from . import variants as VR
    inputs = Inputs(a.brief)
    out = _new_file(a.out, "--out")
    os.makedirs(os.path.dirname(out), exist_ok=True)  # before any call is paid for
    stem = os.path.splitext(os.path.basename(out))[0]
    workdir = os.path.join(os.path.dirname(out), "llm", "paraphrase-%s" % stem)

    def run(marker=None):
        _PHASE[0] = "asking for paraphrases"
        return VR.paraphrase(inputs, a.n, a.keys, out, workdir)
    if a.marker:
        return long_job(a.marker, run, lambda s: "%d of %d paraphrases kept%s" % (
            s["kept"], s["requested"], ", stopped: %s" % s["stopped"] if s["stopped"] else ""))
    return run()


def cmd_variants(a):
    from . import variants as VR
    task = read_json(a.task, "--task")
    errors = R.check_task(task) if isinstance(task, dict) else ["not a JSON object"]
    if errors:
        raise DgenError("--task %s: %s" % (a.task, "; ".join(errors[:3])))
    out = _new_file(a.out, "--out")
    rows = read_jsonl(a.rows, "--rows")
    errors = R.check_full_rows(rows, task, variants=[])  # rows that are variants already stay as they are
    if errors:
        raise DgenError("--rows %s: %d problems in the rows: %s" % (a.rows, len(errors), "; ".join(errors[:3])))
    specs = VR.load_specs(a.specs, task)
    written, summary = VR.make_variants(rows, specs, task, a.share, a.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_jsonl(out, written)
    summary["out"] = out
    return summary
