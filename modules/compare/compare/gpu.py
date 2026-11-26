"""The GPU baselines zero_shot, few_shot and team_recipe, run as one long job in the background (never inline).

- `start_job` writes the job's items to <outdir>/gpu-<n>/job.json and starts `python -m compare gpujob` in its own
  session; it returns the marker <outdir>/gpu-<n>.done once the job has written <marker>.pid.
- The job (`run_job`) writes <marker>.pid first (its pid, its start time in Unix seconds), then one
  <outdir>/gpu-<n>/<name>/result.json per item, one spend line per item (`compare:<name>`, its GPU hours), and the
  marker last: `ok`, or `failed` and the reason. An item that fails (no card, a scoring error, a failed training)
  gets a result with status failed and the reason, which becomes its skip reason; the others go on.
- `finished` finds the result of an item in the job directories of this output directory; `write_baseline` turns a
  result into its baseline directory (cli and milestone call both).

Items: {"name", "action", "source_id", "spec", "spec_sha256", "sub", "key"}. action is `train` (a new baseline),
`repredict` (a registered one on a refreshed evalset: predictions_val copied from it, the evalset inputs predicted
again) or `test` (the milestone: the testset inputs predicted; the valset and evalset predictions copied).
zero_shot and few_shot score the base model through ojh's scorer (score_base); team_recipe is trained by the train
command (train_team) and predicts with its packed model.predict, a stored model a later refresh or the milestone runs
again without training. trsi (modules/train/trsi) is imported when used: the card (trsi.harness._idle) and
model.predict's code (trsi.pack.PREDICT_PY) are its own.
"""
import collections
import importlib
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback

from rsi import metrics as M
from rsi import space as S

from features.table import real_rows

from . import cpu
from . import preds
from . import write as W
from .inputs import read_json, read_jsonl, set_file

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODULE))
PID_WAIT_S = 60       # how long start_job waits for the job's pid file
CARD_WAIT_S = 600     # how long an item waits for an idle card (polled every CARD_POLL_S)
CARD_POLL_S = 30
LAUNCH_SLACK_S = 60   # a job writes its pid file within this long of its start (imports first)
_JOB = re.compile(r"^gpu-(\d+)(?:\.done(?:\.pid)?|\.log)?$")


# ---- the job's files ----------------------------------------------------------------------------------------

def pythonpath():
    """The PYTHONPATH of the job: compare, modules/common, modules/train (trsi) and the repository (rsi)."""
    path = [MODULE, os.path.join(REPO, "modules", "common"), os.path.join(REPO, "modules", "train"), REPO]
    old = os.environ.get("PYTHONPATH")
    return os.pathsep.join(path + ([old] if old else []))


def _numbers(outdir):
    out = set()
    for name in os.listdir(outdir):
        m = _JOB.match(name)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def job_dir(marker):
    """<outdir>/gpu-<n> of the marker <outdir>/gpu-<n>.done."""
    return marker[:-len(".done")]


def latest_marker(outdir):
    """The marker of the latest GPU job of this output directory (by its number), or None."""
    nums = _numbers(outdir)
    return os.path.join(outdir, "gpu-%d.done" % nums[-1]) if nums else None


def marker_text(marker):
    """(first line, rest) of the marker, or (None, None) when it is not there."""
    try:
        with open(marker, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None, None
    return (lines[0].strip() if lines else ""), " ".join(x.strip() for x in lines[1:]).strip()


def job_alive(marker):
    """(alive, pid): the job of `marker` counts as running unless rsi.engine.long_job_identity finds its process
    gone; a live process it cannot confirm (unconfirmed) counts as running too, so compare never starts a second job
    beside one that may still run, and never reports a running job as ended. pid None (not running) without a
    readable pid file."""
    from rsi.engine import long_job_identity, read_pid_file
    pid, started, err = read_pid_file(marker + ".pid")
    if pid is None:
        return False, None
    launched = (started - LAUNCH_SLACK_S) if started is not None else 0
    return long_job_identity(pid, started, launched)[0] != "gone", pid


def start_job(inputs, outdir, plans):
    """Start the GPU job of the items `plans` and return its marker <outdir>/gpu-<n>.done (n one more than any job of
    this output directory). Returns once the job has written <marker>.pid; RuntimeError when it does not (its log,
    <outdir>/gpu-<n>.log, says why)."""
    nums = _numbers(outdir)
    n = (nums[-1] + 1) if nums else 1
    marker = os.path.join(outdir, "gpu-%d.done" % n)
    for path in (marker, marker + ".pid"):
        if os.path.exists(path):
            os.remove(path)
    os.makedirs(job_dir(marker))
    W.write_json(os.path.join(job_dir(marker), "job.json"), {"items": list(plans)})
    log = open(job_dir(marker) + ".log", "ab")
    try:
        proc = subprocess.Popen([sys.executable, "-m", "compare", "gpujob", "--brief", inputs.brief_path,
                                 "--outdir", outdir, "--marker", marker], stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, cwd=outdir, env=dict(os.environ, PYTHONPATH=pythonpath()),
                                start_new_session=True)
    finally:
        log.close()
    end = time.time() + PID_WAIT_S
    while not os.path.exists(marker + ".pid"):
        if proc.poll() is not None or time.time() > end:
            if proc.poll() is None:   # no pid file in time: end it, so no job runs that nothing waits for
                os.killpg(proc.pid, signal.SIGKILL)
            raise RuntimeError("the GPU job did not start (exit %s; see %s)" % (proc.poll(), job_dir(marker) + ".log"))
        time.sleep(0.05)
    return marker


def long_job(inputs, outdir, items, reasons, notes):
    """The long_job hand-back (status, artifacts, notes, marker) for the GPU job of `items`: the job of an earlier
    launch of this dispatch while it still runs (never a second one), else a new one (start_job). skipped.json gets
    `reasons` now. None when the job's marker is already there (it has ended: the caller collects its results)."""
    marker = latest_marker(outdir)
    if marker is None or os.path.exists(marker) or not job_alive(marker)[0]:
        marker = start_job(inputs, outdir, items)
    if os.path.exists(marker):
        return None
    W.skipped(outdir, dict(reasons))
    return "long_job", [], ("the GPU baselines %s run as one long job (marker %s, log %s.log). Ready: %s. Skipped: %s."
                            % (", ".join(i["name"] for i in items), marker, job_dir(marker), "; ".join(notes) or "none",
                               "; ".join("%s (%s)" % kv for kv in reasons.items()) or "none")), marker


def no_result(outdir, item):
    """The skip reason of an item the ended GPU job gave no result for."""
    marker = latest_marker(outdir)
    first, rest = marker_text(marker) if marker else (None, None)
    return "the GPU job %s gave no result for %s (its marker: %s)" % (
        marker and os.path.basename(marker), item["name"], " ".join(x for x in (first, rest) if x) or "none")


def check_in(outdir, collect):
    """Mode checkin: (status, artifacts, notes, marker). While the latest GPU job runs, long_job with its marker
    again; once its marker is there, collect() (as continue); gone without its marker, failed."""
    marker = latest_marker(outdir)
    if marker is None:
        return "failed", [], "checkin: this dispatch has no GPU job", None
    if os.path.exists(marker):
        return collect()
    alive, pid = job_alive(marker)
    if alive:
        return "long_job", [], "the GPU job %d still runs (marker %s)" % (pid, marker), marker
    if pid is None:
        return "failed", [], "the GPU job of %s ended without its marker (no readable pid file)" % marker, None
    return "failed", [], "the GPU job %d ended without its marker" % pid, None


def run_job(brief, outdir, marker):
    """The job (`python -m compare gpujob`): <marker>.pid first, every item of job.json, the marker last. Returns
    the exit code: 0 when the marker says ok."""
    from .inputs import Inputs
    with open(marker + ".pid.tmp", "w") as f:
        f.write("%d\n%d\n" % (os.getpid(), int(time.time())))
    os.replace(marker + ".pid.tmp", marker + ".pid")
    try:
        inputs = Inputs.load(brief)
        for item in read_json(os.path.join(job_dir(marker), "job.json"))["items"]:
            run_item(inputs, item, os.path.join(job_dir(marker), item["name"]))
        text, code = "ok\n", 0
    except Exception as e:  # noqa: BLE001  (the marker is always written, last)
        traceback.print_exc()
        text, code = "failed\n%s: %s\n" % (type(e).__name__, str(e).replace("\n", " ")), 1
    with open(marker + ".tmp", "w") as f:
        f.write(text)
    os.replace(marker + ".tmp", marker)
    return code


class Clock(object):
    """The GPU hours one item's spend line bills: the time it holds a card (start, stop: from the card's pick to the
    end of its scoring or prediction, never the wait for a card) plus hours added whole (a failed training's
    wall-clock time; a finished training bills its own line at once, see _team_item)."""

    def __init__(self):
        self.seconds, self.since, self.added = 0.0, None, 0.0

    def start(self):
        if self.since is None:
            self.since = time.time()

    def stop(self):
        if self.since is not None:
            self.seconds += time.time() - self.since
            self.since = None

    def add(self, hours):
        self.added += float(hours)

    def hours(self):
        self.stop()
        return self.added + self.seconds / 3600.0


def run_item(inputs, item, res_dir):
    """Run one item into res_dir: its result.json (status ok with "P", "compute_hours" (charged now: the spend line),
    "cost_hours" (the baseline's cost), "config", "notes", and "model" for team_recipe; or status failed with "reason"
    and "compute_hours") and its spend line. A failure charges what its Clock counted: 0 when no card was taken."""
    shutil.rmtree(res_dir, ignore_errors=True)
    os.makedirs(res_dir)
    clock = Clock()
    try:
        res = (_team_item if item["name"] == "team_recipe" else _base_item)(inputs, item, res_dir, clock)
        res.update(key=item["key"], status="ok", compute_hours=clock.hours())
        res.setdefault("cost_hours", res["compute_hours"])
        note = None
    except Exception as e:  # noqa: BLE001  (one baseline's failure is its skip reason; the others go on)
        traceback.print_exc()
        res = {"key": item["key"], "status": "failed", "reason": "%s: %s" % (type(e).__name__, e),
               "compute_hours": clock.hours()}
        note = "failed: %s" % res["reason"][:200]
    W.spend(inputs.spend_file, "compare:%s" % item["name"], res["compute_hours"], note)
    W.write_json(os.path.join(res_dir, "result.json"), res)


def item_of(plan, inputs, sub=None):
    """The job item of a Plan (or of a milestone Plan with action test)."""
    key = {"purpose": inputs.purpose, "name": plan.name, "action": plan.action, "source_id": plan.source_id,
           "spec_sha256": plan.spec_sha256, "ids": inputs.ids}
    return {"name": plan.name, "action": plan.action, "source_id": plan.source_id, "spec": plan.spec,
            "spec_sha256": plan.spec_sha256, "sub": sub or "bl-%s" % plan.name, "key": key}


def finished(outdir, item, retry=False):
    """The result of `item` in this output directory's job directories (the latest job first), or None. With retry
    (modes first, resume and fresh) a failed result counts as none, so a new job runs the item again; continue and
    checkin take it as the item's end (its reason)."""
    for n in reversed(_numbers(outdir)):
        path = os.path.join(outdir, "gpu-%d" % n, item["name"], "result.json")
        try:
            res = read_json(path)
        except (OSError, ValueError):
            continue
        if res.get("key") == item["key"]:
            return None if retry and res.get("status") != "ok" else res
    return None


# ---- budget ------------------------------------------------------------------------------------------------

def estimate(name, settings, resources, action="train"):
    """The GPU hours GPU baseline `name` is expected to take for `action`: a team_recipe re-prediction (it only runs
    the stored model) compare.gpu_hours.predict; else compare.gpu_hours.<name>, else for team_recipe
    resources.onejev.estimate[<its base>] (its training); None when none gives a number."""
    hours = settings.get("gpu_hours") or {}
    e = hours.get("predict") if name == "team_recipe" and action == "repredict" else hours.get(name)
    if e is None and name == "team_recipe" and action != "repredict":
        base = (team_config(settings, resources) or {}).get("base")
        e = (((resources.get("onejev") or {}).get("estimate")) or {}).get(base)
    try:
        e = float(e)
    except (TypeError, ValueError):
        return None
    return e if math.isfinite(e) and e >= 0 else None


def why_not(plan, inputs, left, predict_only=False):
    """(reason, left): why GPU plan `plan` cannot run (None when it can), and the GPU hours left after it. It runs
    only with what it needs (team_recipe a config, the others a base model) and when `left` (budget_left.compute_hours,
    None: no budget) is at least its estimate (estimate: a team_recipe re-prediction, or with predict_only a team
    recipe an earlier job of the dispatch trained (trained_team), is held by the prediction estimate, not its
    training's); each plan that runs takes its estimate from what is left."""
    settings, resources = inputs.settings, inputs.resources
    if plan.name == "team_recipe" and settings.get("team_recipe") is None:
        return "compare.team_recipe is not set (the team's training config)", left
    if plan.name != "team_recipe" and not plan.spec.get("base"):
        return ("no base model: set compare.%s.base or resources.onejev.default_config.base" % plan.name), left
    if left is None:
        return None, None
    action = "repredict" if predict_only else plan.action
    e = estimate(plan.name, settings, resources, action)
    if e is None:
        where = ("compare.gpu_hours.predict" if plan.name == "team_recipe" and action == "repredict" else
                 "compare.gpu_hours.%s%s" % (plan.name, ", resources.onejev.estimate.<base>"
                                             if plan.name == "team_recipe" else ""))
        return "no GPU-hour estimate (%s), and the compute budget has %s GPU hours left" % (where, _num(left)), left
    if e > left + 1e-9:
        return "needs %s GPU hours, %s left" % (_num(e), _num(left)), left
    return None, left - e


def _num(x):
    return "%g" % x


# ---- scoring the base model ---------------------------------------------------------------------------------

def pick_card(gpu_ids):
    """The uuid of the idle card (under trsi.harness.IDLE_MIB in use) with the least memory in use among `gpu_ids`
    (resources.gpu_ids; all cards when not set), from trsi.harness._idle; waits up to CARD_WAIT_S for one."""
    from trsi import harness
    allowed = tuple(int(i) for i in gpu_ids) if isinstance(gpu_ids, list) and gpu_ids else None
    end = time.time() + CARD_WAIT_S
    while True:
        idle = harness._idle(allowed)
        if idle:
            return idle[0][2]
        if time.time() >= end:
            raise RuntimeError("no idle card among %s after %d s" % (list(allowed) if allowed else "all cards",
                                                                     CARD_WAIT_S))
        time.sleep(CARD_POLL_S)


def _load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ojh_repo(onejev):
    """The onejev-harness repository: the parent of the .venv of resources.onejev.ojh_bin (<repo>/.venv/bin/ojh)."""
    ojh_bin = onejev.get("ojh_bin")
    if not ojh_bin:
        raise RuntimeError("resources.onejev.ojh_bin is not set: the base model's catalog entry cannot be found")
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(ojh_bin))))
    venv = os.path.dirname(os.path.dirname(path))
    if os.path.basename(venv) != ".venv":
        raise RuntimeError("resources.onejev.ojh_bin %s is not <ojh repo>/.venv/bin/ojh" % ojh_bin)
    return os.path.dirname(venv)


def _machine(onejev):
    """(model_dir, env) of ojh's machine config resources.onejev.machine, as ojh reads it (harness/config.py:
    ~ and $VAR expanded, a relative model_dir taken from the file's directory); without one ojh's inferred model_dir
    ($OJH_MODEL_DIR, else ~/.cache/onejev-harness/models) and no env."""
    path = onejev.get("machine")
    if not path:
        d = os.environ.get("OJH_MODEL_DIR") or "~/.cache/onejev-harness/models"
        return os.path.abspath(os.path.expanduser(d)), {}
    path = os.path.expandvars(os.path.expanduser(str(path)))
    conf = _load_yaml(path)
    d = os.path.expandvars(os.path.expanduser(str(conf.get("model_dir") or "")))
    if not d:
        raise RuntimeError("%s has no model_dir" % path)
    d = d if os.path.isabs(d) else os.path.join(os.path.dirname(os.path.abspath(path)), d)
    env = dict((str(k), str(v)) for k, v in (conf.get("env") or {}).items())
    return d, env


def base_scoring(onejev, base):
    """{"recipe", "kwargs"}: how an untrained base `base` is scored, the same way trsi scores a candidate of that
    base trained with its catalog default recipe (trsi.pack.score_kwargs on that recipe's flags), so a zero-shot or
    few-shot baseline and a candidate of the base are scored alike. The recipe: the one-card default of
    <ojh repo>/catalog/bases/<base>.yaml (what trsi takes for recipe null, trsi.candidate.resolve_recipe), else the
    first of its default_recipes; its flags as ojh's resolve builds a run's (harness/resolve.py: the recipe's onejev
    flags, full_finetune for mode full, fsdp by ojh's harness.resolve.parallel_for when ojh is importable, else the
    recipe's fsdp, and for mode lora each field of its lora section as lora_<field> (target_modules as it is), e.g.
    lora.autocast → lora_autocast). Without a default recipe, the flags are empty (bfloat16, no autocast).
    multimodal from the catalog."""
    from pathlib import Path
    from trsi.candidate import CandidateError, resolve_recipe
    from trsi.pack import score_kwargs
    repo = ojh_repo(onejev)
    catalog = os.path.join(repo, "catalog", "bases", "%s.yaml" % base)
    if not os.path.isfile(catalog):
        raise RuntimeError("the base %s is not in ojh's catalog (%s)" % (base, catalog))
    cat = _load_yaml(catalog)
    try:
        path = str(resolve_recipe(None, base, Path(repo)))
    except CandidateError:
        first = next(iter((cat.get("default_recipes") or {}).values()), None)
        path = os.path.join(repo, first) if first else None
    recipe = _load_yaml(path) if path else {}
    flags = dict(recipe.get("onejev") or {})
    mode, gpus = recipe.get("mode"), int(recipe.get("gpus") or 1)
    if mode == "full":
        flags["full_finetune"] = True
    if mode == "lora":
        for k, v in (recipe.get("lora") or {}).items():
            flags[k if k == "target_modules" else "lora_" + k] = v
    try:
        from harness.resolve import parallel_for   # ojh's own rule (its venv)
        fsdp = mode is not None and parallel_for(mode, float(cat.get("params_b") or 0), gpus,
                                                 recipe.get("fsdp")) == "fsdp"
    except Exception:  # noqa: BLE001  (ojh not importable here, or a recipe it refuses: the recipe's own flag)
        fsdp = bool(recipe.get("fsdp"))
    if fsdp:
        flags["fsdp"] = True
    rel = os.path.relpath(path, repo) if path else None
    return {"recipe": rel, "kwargs": score_kwargs({"flags": flags, "base": {"multimodal": cat.get("multimodal")}})}


def score_base(onejev, base, rows, card, workdir=None, kwargs=None):
    """id -> label -> probability: the untrained base model `base` scored on `rows` (input rows: id, state,
    question) on the card `card` (a uuid).

    How ojh scores a checkpoint (onejev-harness harness/score.py, catalog/bases/<base>.yaml):
    - `score_checkpoint(model_dir, rows, *, base_dir=None, multimodal, device="cuda:0", dtype="bfloat16",
      autocast=True, chat_template=None, media_root=None, disable_fla_kernels=False)` returns {id: {"id", "labels",
      "probs", "target"}}. It loads through `load_scoring_bundle`: OneJev's `common.load_bundle` on the weights of
      `base_dir` when given (then `model_dir` is a LoRA adapter put on top with peft), else of `model_dir`; then
      `Readout.attach` with `ReadoutConfig.load(model_dir)` (readout.json beside the weights, and its head file when
      the readout has a head); rows are scored one at a time (max_rows 1), each row needing a target.
    - A base checkpoint as ojh downloads it (`ojh fetch-base`, into <machine model_dir>/<base>) holds readout.json
      (the catalog entry lists its sha256), so the untrained base is scored as a full checkpoint: model_dir = the
      base directory, base_dir None.
    - kwargs (default base_scoring(onejev, base)["kwargs"]): those trsi scores a candidate of the base's default
      recipe with (trsi.pack.score_kwargs, as ojh's finalize: float32 only for a full fine-tune without FSDP, else
      the recipe's dtype; autocast for lora_autocast or a full fine-tune without no_autocast); multimodal from the
      catalog entry.
    This is not a copy of trsi's scorer: a scoring directory gets predict.json (those kwargs, the machine config's
    env, and COMPARE_TEST_SCORER as test_scorer in tests only), trsi.pack.PREDICT_PY as predict.py with its
    PREDICT_SH entry, and model/ linking <model_dir>/<base>; its `predict` runs with CUDA_VISIBLE_DEVICES=<card>.
    `workdir` holds it (default a temporary directory, removed after)."""
    from trsi.pack import PREDICT_PY, PREDICT_SH
    if kwargs is None:
        kwargs = base_scoring(onejev, base)["kwargs"]
    model_dir, env = _machine(onejev)
    weights = os.path.join(model_dir, base)
    if not os.path.isdir(weights):
        raise RuntimeError("the base %s is not at %s (ojh fetch-base %s)" % (base, weights, base))
    work = tempfile.mkdtemp(prefix="score-%s-" % base, dir=workdir)
    try:
        cfg = {"kwargs": kwargs, "env": env, "base": base}
        if os.environ.get("COMPARE_TEST_SCORER"):   # tests only
            cfg["test_scorer"] = os.environ["COMPARE_TEST_SCORER"]
        W.write_json(os.path.join(work, "predict.json"), cfg)
        with open(os.path.join(work, "predict.py"), "w", encoding="utf-8") as f:
            f.write(PREDICT_PY)
        entry = os.path.join(work, "predict")
        with open(entry, "w", encoding="utf-8") as f:
            f.write(PREDICT_SH.format(python=shlex.quote(str(onejev.get("python") or sys.executable))))
        os.chmod(entry, os.stat(entry).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.symlink(weights, os.path.join(work, "model"))
        inp, out = os.path.join(work, "inputs.jsonl"), os.path.join(work, "predictions.jsonl")
        W.write_jsonl(inp, rows)
        run_predict(entry, inp, out, card)
        return dict((r["id"], r["probs"]) for r in read_jsonl(out))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_predict(entry, inputs_path, out, card):
    """`<entry> --inputs <inputs_path> --out <out>` on the card `card`. Its output is streamed into this job's log
    line by line; when it fails, RuntimeError with its last lines, starting "OOM" when it ran out of GPU memory
    (trsi.pack.OOM_RE anywhere in its output: the last lines may be another error after it)."""
    from trsi.pack import OOM_RE
    tail, oom = collections.deque(maxlen=3), False
    p = subprocess.Popen([entry, "--inputs", inputs_path, "--out", out], stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, errors="replace",
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=card))
    try:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            oom = oom or bool(OOM_RE.search(line))
            if line.strip():
                tail.append(line.strip())
    except BaseException:   # a stop (SIGTERM is the job's group's), or a log that cannot be written: end predict too
        p.kill()
        raise
    finally:
        p.stdout.close()
        code = p.wait()
    if code != 0:
        last = " | ".join(tail)[-500:] or "no output"
        if oom:
            raise RuntimeError("OOM: %s ran out of GPU memory on %s (exit %d): %s" % (entry, card, code, last))
        raise RuntimeError("%s failed (exit %d): %s" % (entry, code, last))


def _score_sets(onejev, base, sets, card, workdir, kwargs=None):
    """{set name: id -> probabilities} for `sets` (name -> rows): one scoring (one model load) when the ids of all the
    sets are distinct, else one per set."""
    ids = [r["id"] for rows in sets.values() for r in rows]
    if len(ids) == len(set(ids)):
        probs = score_base(onejev, base, [r for rows in sets.values() for r in rows], card, workdir, kwargs)
        return dict((s, dict((r["id"], probs[r["id"]]) for r in rows)) for s, rows in sets.items())
    return dict((s, score_base(onejev, base, rows, card, workdir, kwargs)) for s, rows in sets.items())


# ---- few-shot prompts ---------------------------------------------------------------------------------------

def few_shot_examples(train_rows, k, seed, task):
    """The k labelled examples of few_shot, from features.table.real_rows(train_rows), in trainset order: for a
    binary task k/2 per label (rounded up for the positive label), else stratified by label (cpu.subsample); drawn
    with random.Random(seed). Fewer when a label has fewer rows."""
    labels = task["decision"]["labels"]
    pool = real_rows(train_rows)
    y = [M.hard_label(r["target"], labels) for r in pool]
    if len(labels) != 2:
        return [pool[i] for i in cpu.subsample(y, k, seed)]
    pos = task["decision"].get("positive") or labels[0]
    quota = dict((label, (k + 1) // 2 if label == pos else k // 2) for label in labels)
    rng = random.Random(seed)
    chosen = []
    for label in labels:
        idx = [i for i, v in enumerate(y) if v == label]
        chosen += rng.sample(idx, min(quota[label], len(idx)))
    return [pool[i] for i in sorted(chosen)]


def few_shot_prefix(examples, max_chars, task):
    labels = task["decision"]["labels"]
    parts = []
    for i, r in enumerate(examples, 1):
        text = json.dumps(r["state"], sort_keys=True, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + " \u2026"
        parts.append("Example %d:\nInput: %s\nAnswer: %s" % (i, text, M.hard_label(r["target"], labels)))
    return ("Here are %d labelled examples.\n\n%s\n\nNow answer for the input below.\n\n"
            % (len(examples), "\n\n".join(parts)))


def few_shot_rows(rows, train_rows, k, max_chars, seed, task):
    """`rows` with their question.instructions prefixed by k labelled trainset examples (few_shot_examples): "Here
    are <k> labelled examples.\\n\\n" + the examples + "\\n\\nNow answer for the input below.\\n\\n", each example
    "Example <i>:\\nInput: <state JSON, sort_keys, cut to max_chars with ' …'>\\nAnswer: <label>". Everything else
    in each row is unchanged (the rows themselves are not changed)."""
    prefix = few_shot_prefix(few_shot_examples(train_rows, k, seed, task), max_chars, task)
    out = []
    for r in rows:
        x = dict(r)
        x["question"] = dict(r["question"], instructions=prefix + r["question"]["instructions"])
        out.append(x)
    return out


# ---- the team's recipe ----------------------------------------------------------------------------------------

def team_config(settings, resources):
    """The train command's config.json for compare.team_recipe (None when it is not set): its base, else
    resources.onejev.default_config.base; keys left null (the seed, a base neither gives) are left out, so the
    train command's defaults apply."""
    cfg = settings.get("team_recipe")
    if cfg is None:
        return None
    out = dict(cfg)
    if out.get("base") is None:
        out["base"] = (((resources.get("onejev") or {}).get("default_config")) or {}).get("base")
    return dict((k, v) for k, v in out.items() if v is not None or k in ("recipe", "max_steps"))


def team_task(task, cfg):
    """A copy of `task` whose search_space also allows the team recipe's config `cfg`: packing checks the candidate's
    config_request against the search_space (`rsi validate candidate`, rsi.space), which holds only the agent's tuning
    dimensions, so the team recipe (its own recipe, base, overrides) would be refused. Each path of `cfg` that is not
    null (base, recipe, seed (the train command's default 0 when cfg has none), max_steps, overrides.<group>.<name>)
    gets its value added: a key of its own, one more allowed value, or a range widened to it. Everything else stays."""
    task = json.loads(json.dumps(task))
    space = task.get("search_space")
    if space is None:
        return task
    for path, value in S.request_paths(dict({"seed": 0}, **cfg)):
        allowed = space.get(path)
        if allowed is None:
            space[path] = [value]
        elif S.allowed_values(allowed) is not None:
            if value not in S.allowed_values(allowed):
                S.allowed_values(allowed).append(value)
        else:
            allowed["min"], allowed["max"] = min(allowed["min"], value), max(allowed["max"], value)
    return task


def train_team(inputs, cfg, out):
    """Train config `cfg` with the train command (modules/train/SKILL.md, "Train command") and return the packed
    candidate's directory <out>/team/cand. It runs with every argument the contract requires (--task
    <out>/team/task.json (team_task: the task with its search_space widened to allow `cfg`), --trainset,
    --trainset-id, --valset, --valset-id, --config, --config-source default, --round, --producer train, --out
    <out>/team/cand, --marker <out>/team/train.done) and PYTHONPATH=<repo root>, and blocks until it ends; its output
    goes to this job's log. RuntimeError unless its marker says ok."""
    team = os.path.join(out, "team")
    shutil.rmtree(team, ignore_errors=True)
    os.makedirs(team)
    task_path, cfg_path, cand, marker = (os.path.join(team, x) for x in ("task.json", "config.json", "cand",
                                                                         "train.done"))
    W.write_json(task_path, team_task(inputs.task, cfg))
    W.write_json(cfg_path, cfg)
    inp, ids = inputs.brief["inputs"], inputs.ids
    argv = [inputs.train_command, "--task", task_path, "--trainset", inp["trainset"],
            "--trainset-id", ids["trainset"], "--valset", inp["valset"], "--valset-id", ids["valset"],
            "--config", cfg_path, "--config-source", "default", "--round", str(inputs.round), "--producer", "train",
            "--out", cand, "--marker", marker]
    sys.stdout.flush()
    p = subprocess.run(argv, stdin=subprocess.DEVNULL, cwd=team, env=dict(os.environ, PYTHONPATH=REPO))
    first, reason = marker_text(marker)
    if first != "ok":
        raise RuntimeError("the train command failed (exit %d): %s" % (p.returncode, reason or "no marker"))
    return cand


# ---- one item ----------------------------------------------------------------------------------------------

def _inputs_file(inputs, s):
    return inputs.brief["inputs"]["evalset_inputs" if s == "eval" else "testset_inputs"]


def _base_item(inputs, item, res_dir, clock):
    """zero_shot or few_shot: score the base model on the sets the action needs (few_shot: the rows of few_shot_rows),
    scored as base_scoring says; the clock runs from the card's pick to the end of the scoring."""
    spec, action, onejev = item["spec"], item["action"], inputs.resources.get("onejev") or {}
    base = spec.get("base")
    if not base:
        raise RuntimeError("no base model: set compare.%s.base or resources.onejev.default_config.base"
                           % item["name"])
    if action == "train":
        sets = {"val": inputs.val_inputs, "eval": inputs.eval_inputs}
    else:
        sets = {"eval": inputs.eval_inputs} if action == "repredict" else {"test": inputs.testset_inputs}
    scoring = base_scoring(onejev, base)
    config = {"base": base, "scoring": scoring}
    if item["name"] == "few_shot":
        train = inputs.baseline_rows[item["source_id"]]["train"] if action == "test" else inputs.train_rows
        args = (train, spec["k"], spec["max_example_chars"], spec["seed"], inputs.task)
        sets = dict((s, few_shot_rows(rows, *args)) for s, rows in sets.items())
        config.update(k=spec["k"], examples=[r["id"] for r in few_shot_examples(train, spec["k"], spec["seed"],
                                                                                 inputs.task)])
    card = pick_card(inputs.resources.get("gpu_ids"))
    clock.start()
    P = _score_sets(onejev, base, sets, card, res_dir, scoring["kwargs"])
    clock.stop()
    shots = " with %d labelled trainset examples in each prompt" % len(config["examples"]) if "examples" in config \
        else ""
    what = {"train": "the valset and the evalset inputs", "repredict": "the evalset inputs of %s" % inputs.ids["evalset"],
            "test": "the testset inputs of %s" % inputs.testset_id}[action]
    return {"P": P, "config": config,
            "notes": "%s: the base model %s scored as is%s on %s%s" % (
                item["name"], base, shots, what, ", from %s" % item["source_id"] if item["source_id"] else "")}


def trained_team(outdir, item):
    """The candidate directory a GPU job of this output directory trained for team item `item` (the same key; its
    train.done says ok), kept when a later step of that job failed; None when there is none."""
    for n in reversed(_numbers(outdir)):
        team = os.path.join(outdir, "gpu-%d" % n, "team_recipe", "team")
        try:
            key = read_json(os.path.join(team, "key.json"))
        except (OSError, ValueError):
            continue
        cand = os.path.join(team, "cand")
        if (key == item["key"] and marker_text(os.path.join(team, "train.done"))[0] == "ok"
                and os.path.isfile(os.path.join(cand, "manifest.json"))):
            return cand
    return None


def _team_item(inputs, item, res_dir, clock):
    """team_recipe: train the config with the train command and predict the evalset inputs with the packed model
    (train), or run a registered baseline's stored model on the new evalset inputs (repredict) or the testset
    inputs (test). A training is billed on its own spend line (its candidate's cost.compute_hours) as soon as it has
    finished, right after its key.json, so a stop later cannot leave a training that a later job uses unbilled; the
    item's own line bills only the prediction (the baseline's cost counts both). A training an earlier job of this
    dispatch finished (trained_team) is used again, never repeated nor billed again. trsi is imported before anything
    runs, so a missing library fails before hours of training."""
    importlib.import_module("trsi.harness")   # pick_card needs it (and pyyaml): fail now, not after the training
    action, P, cost = item["action"], {}, 0.0
    if action == "train":
        cfg = team_config(inputs.settings, inputs.resources)
        model_dir = trained_team(os.path.dirname(os.path.dirname(res_dir)), item)
        if model_dir is None:
            t0 = time.time()
            try:
                model_dir = train_team(inputs, cfg, res_dir)
            except Exception:
                clock.add((time.time() - t0) / 3600.0)   # a failed training: its wall-clock time on one card
                raise
            W.write_json(os.path.join(res_dir, "team", "key.json"), item["key"])
            cm = read_json(os.path.join(model_dir, "manifest.json"))
            W.spend(inputs.spend_file, "compare:team_recipe", float((cm.get("cost") or {}).get("compute_hours") or 0.0),
                    "the training (its candidate's cost.compute_hours)")
        cm = read_json(os.path.join(model_dir, "manifest.json"))
        cost = float((cm.get("cost") or {}).get("compute_hours") or 0.0)
        P["val"] = set_file(model_dir, "predictions_val")
        config = {"config": cfg, "train_command": inputs.train_command, "candidate": cm.get("notes")}
        sets = ["eval"]
    else:
        src = inputs.baselines[item["source_id"]]
        cm, model_dir, config = src["manifest"], src["dir"], src["manifest"]["config"]
        sets = ["eval"] if action == "repredict" else ["test"]
    if not (cm.get("model") or {}).get("predict"):
        raise RuntimeError("%s has no model.predict to run" % model_dir)
    entry = os.path.join(model_dir, cm["model"]["predict"])
    card = pick_card(inputs.resources.get("gpu_ids"))
    t0 = time.time()
    clock.start()
    for s in sets:
        P[s] = os.path.join(res_dir, "predictions_%s.jsonl" % s)
        run_predict(entry, _inputs_file(inputs, s), P[s], card)
    clock.stop()
    return {"P": P, "cost_hours": cost + (time.time() - t0) / 3600.0, "config": config,
            "model": {"dir": model_dir, "location": cm["model"].get("location") or "model",
                      "predict": cm["model"]["predict"]},
            "notes": "team_recipe: %s" % (
                "trained with the train command (%s), its packed model on the evalset inputs"
                % json.dumps(config["config"], sort_keys=True) if action == "train" else
                "the stored model of %s on the %s inputs of %s" % (
                    item["source_id"], "evalset" if action == "repredict" else "testset",
                    inputs.ids["evalset"] if action == "repredict" else inputs.testset_id))}


# ---- the baseline directory of a result -------------------------------------------------------------------------

def write_baseline(outdir, inputs, item, res, plan):
    """The baseline directory (item["sub"]) of the finished item `item` (its result `res` has status ok), for Plan
    `plan`. Probabilities are ordered by the rows they are on; a file is copied as it is. repredict copies the source
    baseline's predictions_val; test (the milestone) copies its predictions_val and predictions_eval and keeps its
    config and features, with testset_id and predictions_test.

    The decision policy (with the task's policy section): zero_shot and few_shot fit theirs on their valset
    predictions (preds.with_policy) when trained, and a re-prediction or the milestone applies the source baseline's
    recorded policy.json to the new scores. team_recipe never fits one: its packed candidate fitted its own, which its
    model.predict applies, and the baseline keeps that candidate's (or its source's) policy.json."""
    P, action, src_id = res["P"], item["action"], item["source_id"]
    view = inputs.of_baseline(src_id) if action == "test" else inputs
    labels = inputs.task["decision"]["labels"]

    def probs(s, rows):
        x = P.get(s)
        return x if x is None or isinstance(x, str) else [x[r["id"]] for r in rows]

    P_val, P_eval = probs("val", view.val_rows), probs("eval", view.eval_inputs or [])
    P_test = probs("test", inputs.testset_inputs)
    config, features = res.get("config"), None
    team = item["name"] == "team_recipe"
    if action == "train":
        if team:
            policy = preds.recorded_policy(res["model"]["dir"])
        else:
            P_val, P_eval, policy = preds.with_policy(inputs.task, inputs.val_rows, P_val, P_eval)
    else:
        src = inputs.baselines[src_id]
        policy = preds.recorded_policy(src["dir"])
        P_val = set_file(src["dir"], "predictions_val")
        if action == "test":
            P_eval = set_file(src["dir"], "predictions_eval")
            config, features = src["manifest"]["config"], src["manifest"]["features"]
        if policy is not None and not team:   # new scores of the base model: the recorded policy applied
            if action == "test":
                P_test = preds.applied(policy, P_test, labels)
            else:
                P_eval = preds.applied(policy, P_eval, labels)
    model = res.get("model") if action != "test" else None
    sub = W.baseline_dir(outdir, item["name"], view, plan, config, P_val, P_eval, features,
                         res.get("cost_hours", res["compute_hours"]),
                         model=model, reused_from=src_id if action != "train" else None,
                         testset_id=inputs.testset_id if action == "test" else None,
                         P_test=P_test, model_from=model and model["dir"],
                         sub=item["sub"], notes=res.get("notes"), policy=policy)
    errors = W.check(outdir, sub, "baseline", inputs.task)
    if errors:
        raise RuntimeError("%s is not a valid baseline: %s" % (sub, "; ".join(errors)))
    return sub
