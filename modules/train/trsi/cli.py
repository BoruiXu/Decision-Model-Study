"""python -m trsi <command> ...

init     init --brief <brief.json>: write <outdir>/trsi/run.yaml and exp.yaml from the train brief and its task, and
         make the start the incumbent (init <run.yaml> initialises an existing run.yaml)
check    validate the run config (and with --ojh, `ojh check` the start without using cards)
launch   train one seed of the start (--start) or of a proposal on the cards given (--gpus), through `ojh run`;
         returns at once; --force also trains again a trial whose packing keeps failing (recorded as failed, charged)
watch    watch <run.yaml> --marker <path>: in the background, wait until every trial launched so far has ended, then
         write the marker (ok); writes <marker>.pid; on SIGTERM ends the ojh trainings of every trial of the run still
         running (those launched after the watch too) and writes failed; on any other error writes failed too
adopt    adopt <run.yaml> --candidate <dir>: take a registered candidate as the start's seed it was trained with
collect  pack every finished trial as a candidate <outdir>/cand-<trial>/ (model, model.predict, predictions_val, rsi
         score) and record its numbers. Before each trial it picks the scoring card (harness.pick_scoring_card: in
         gpu_ids, no other process, enough free memory, ojh's card lock; --gpu first when it qualifies); after an out-
         of-memory error it packs once more on another card. A trial whose packing failed stays "unpacked" (not
         compared, not retrained) and is packed again by the next collect. An ojh run whose tmux session is gone
         without a DONE and unchanged for 2 minutes has failed ("ended without DONE"). It deletes the checkpoints a
         trial no longer needs, without asking (_tidy): once packed, `ojh cleanup --confirm` of its run; once its DONE
         says failed, all of them; of the runs a relaunch replaced, all of them once their DONE is there.
next     apply the rules (probe, more seeds, accept) to the candidates' numbers, update the incumbent, print what to
         do next (and why unpacked trials are not packed)
report   write <out_dir>/<run_name>/report.md; with --search <dir> --ids <registered-<k>.json ...> also the
         framework's search

launch, watch (at every poll) and collect (before and after packing each trial) rewrite progress.json (trsi.progress);
a failure to write it is printed to stderr and never fails the command.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import yaml

from . import __version__, archive, harness, pack, progress, report
from .candidate import (Candidate, CandidateError, candidate_id, dead_ends, file_sha256, load_proposal,
                        outside_space, parse_candidate, request_mapping)
from .config import ConfigError, RunConfig, load_run, write_from_brief
from .harness import HarnessError, TrialResult
from .rules import after_probe, decide


class Refused(RuntimeError):
    pass


# ---- cost ---------------------------------------------------------------------------------------------------

def cost_of(run: RunConfig):
    """Compute hours of a trial: its candidate's cost (GPU hours + scoring), or the GPU hours of a failed run."""
    def f(rec: dict) -> float:
        r = rec.get("result") or {}
        return float(r["compute_hours"] if r.get("compute_hours") is not None else r.get("gpu_hours") or 0)
    return f


def spent(run: RunConfig) -> float:
    """What this run's trials cost. Every trained trial is a candidate the framework counts, the start's too; a
    candidate adopted from an earlier round was paid for then."""
    c = cost_of(run)
    total = sum(c(rec) for rec in archive.trials(run).values() if rec["charged"])
    total += sum(e["gpu_h"] for e in archive.events(run, "retry") if e.get("charged"))
    return total


def estimate(run: RunConfig, c: Candidate) -> float | None:
    """Expected cost of one seed: mean of finished trials with the same base and recipe, else budget.estimate."""
    sha = file_sha256(c.recipe)
    cf = cost_of(run)
    seen = [cf(r) for r in archive.trials(run).values()
            if r["status"] == "done" and not r.get("adopted") and r["candidate"]["base"] == c.base
            and r["candidate"]["recipe_sha256"] == sha]
    est = sum(seen) / len(seen) if seen else run.estimate_gpu_h.get(c.base)
    if est is None:
        return None
    ep = c.overrides.get("onejev.epochs")
    if ep is not None:  # cost scales with epochs
        base_ep = float(((yaml.safe_load(c.recipe.read_text()) or {}).get("onejev") or {}).get("epochs", 1))
        est *= float(ep.value) / base_ep
    return est


# ---- helpers ------------------------------------------------------------------------------------------------

def _gpus(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _need_init(run: RunConfig) -> tuple[Candidate, str]:
    inc = archive.incumbent(run)
    if inc is None:
        raise Refused("run `python -m trsi init --brief <brief.json>` first")
    return inc


def _results(run: RunConfig) -> tuple[dict[str, dict[int, TrialResult]], dict[str, dict[int, dict]]]:
    done: dict[str, dict[int, TrialResult]] = defaultdict(dict)
    recs: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in archive.trials(run).values():
        recs[rec["candidate_id"]][rec["seed"]] = rec
        if archive.state_of(rec) == "done":   # packed: its candidate's numbers exist
            done[rec["candidate_id"]][rec["seed"]] = TrialResult.from_dict(rec["result"])
    return done, recs


def _probe_event(run: RunConfig, cid: str, against: str) -> dict | None:
    return next((e for e in reversed(archive.events(run, "probe"))
                 if e["candidate"] == cid and e["against"] == against), None)


def _candidate(run: RunConfig, a) -> tuple[Candidate, object, bool]:
    if a.start:
        return run.start, None, True
    if not a.proposal:
        raise Refused("give a proposal YAML or --start")
    c, pred = load_proposal(Path(a.proposal), run)
    return c, pred, False


def _proposed(run: RunConfig) -> set[str]:
    return {e["candidate"] for e in archive.events(run, "propose")}


def _record_proposal(run: RunConfig, c: Candidate, cid: str, pred, path: str) -> None:
    if pred is None or cid in _proposed(run):
        return
    archive.append(run, {"event": "propose", "candidate": cid, "candidate_dict": c.to_dict(), "proposal": path,
                         "prediction": {"metric": pred.metric, "value": pred.value, "p_accept": pred.p_accept}})


def _cards(run: RunConfig) -> str:
    return "all cards" if run.gpus.ids is None else str(list(run.gpus.ids))


def _progress(run: RunConfig, packing: tuple[str, str] | None = None) -> None:
    """Rewrite progress.json (trsi.progress.write); a failure is printed to stderr (the watch's log in the
    background), never raised: not even when stderr cannot be written either (a full disk)."""
    try:
        err = progress.write(run, packing)
    except Exception as e:  # write returns its errors; this guards against a bug in it
        err = f"{type(e).__name__}: {e}"
    if err:
        with contextlib.suppress(Exception):
            print(f"trsi: progress.json not written: {err}", file=sys.stderr, flush=True)


def stop_reason(run: RunConfig) -> str | None:
    """The STOP line of `trsi next` when the search must stop now (the budget cannot pay for another seed, or
    this dispatch has tried search.max_configs configurations and none is open), else None; progress.json's notes
    show it too."""
    inc = archive.incumbent(run)
    if inc is None:
        return None
    done, _ = _results(run)
    inc_missing = [s for s in run.rules.seeds if s not in done.get(inc[1], {})]
    left = run.budget_gpu_h - spent(run)
    recipes = run.space.recipes or (run.start.recipe,)
    ests = [x for x in (estimate(run, Candidate(b, r)) for b in run.space.bases for r in recipes) if x]
    verdicts = {v["candidate"] for v in archive.events(run, "verdict")}
    open_c = [e for e in archive.events(run, "propose") if e["candidate"] not in verdicts]
    mc = run.rules.max_configs
    if ests and left < min(ests):
        return "STOP: the budget cannot pay for another seed. Write the report (python -m trsi report)."
    if not inc_missing and not open_c and mc is not None and len(_proposed(run)) >= mc:
        return (f"STOP: this dispatch (returns included) has tried search.max_configs = {mc} configurations. "
                "Write the report.")
    return None


# ---- commands -----------------------------------------------------------------------------------------------

def cmd_check(a) -> int:
    run = load_run(Path(a.run))
    harness.load_exp(run)
    problems = [f"start: {p}" for p in outside_space(run.start, run.space)]
    fw = run.framework
    print(f"run       {run.run_name} -> records in {run.home}; candidates in {fw.outdir}/cand-<trial>/")
    print(f"exp       {run.exp} (positive label {harness.positive_label(run)!r})")
    print(f"sets      trainset {fw.trainset_id}, valset {fw.valset_id}; task {fw.task}")
    print(f"ojh       {run.ojh.bin}" + (f" (after sourcing {run.ojh.env_file})" if run.ojh.env_file else "")
          + (f"; machine {run.ojh.machine}" if run.ojh.machine else ""))
    print(f"start     {run.start.label()}")
    print(f"space     bases {list(run.space.bases)}; {len(run.space.recipes) + run.space.default_recipe} recipe(s); "
          "overrides " + (", ".join(f"{k} {list(s.values) or list(s.range)}" for k, s in run.space.overrides.items())
                          or "none"))
    for r in run.space.recipes + (run.start.recipe,):
        d = yaml.safe_load(r.read_text()) or {}
        print(f"recipe    {r.name}: mode {d.get('mode')}, {d.get('gpus', 1)} GPU(s)")
        if d.get("mode") not in ("full", "lora"):
            problems.append(f"recipe {r}: mode must be full or lora")
    for b in run.space.bases:
        if b not in run.estimate_gpu_h:
            print(f"warning   no budget.estimate for {b}: launch is refused until a trial of it has finished")
    print(f"gpus      {_cards(run)}" + (" (shared cards allowed)" if run.gpus.allow_shared else ""))
    print(f"rules     {run.rules}; metric {run.metric.primary}, guards {run.metric.guards}; budget "
          f"{run.budget_gpu_h} compute hours")
    for d in dead_ends(run.start, run.space):
        print(f"warning   start uses {d}")
    if a.ojh and not problems:
        n = harness.recipe_gpus(run.start.recipe)
        ids = list(run.gpus.ids[:n]) if run.gpus.ids else list(range(n))
        out = harness.check(run, run.start, run.rules.seeds[0], ids)
        print("ojh check " + json.dumps(out)[:400])
    for p in problems:
        print(f"PROBLEM   {p}")
    return 1 if problems else 0


def cmd_init(a) -> int:
    if bool(a.brief) == bool(a.run):
        raise Refused("give --brief <brief.json> (or the run.yaml it wrote)")
    path = write_from_brief(Path(a.brief)) if a.brief else Path(a.run)
    run = load_run(path)
    harness.load_exp(run)
    print(f"run.yaml  {run.path}")
    if archive.incumbent(run):
        print(f"{run.home} is already initialised; the incumbent is in incumbent.yaml")
        return 0
    cid = candidate_id(run.start, harness.exp_sha256(run))
    (run.home / "proposals").mkdir(parents=True, exist_ok=True)
    archive.append(run, {"event": "init", "run_file": str(run.path), "run_sha256": file_sha256(run.path),
                         "exp_sha256": harness.exp_sha256(run), "trsi": __version__})
    archive.set_incumbent(run, run.start, cid, "start (the configuration the search begins from)")
    print(f"initialised {run.home}; incumbent = start `{cid}` {run.start.label()}")
    print(f"next: train the start's seeds {list(run.rules.seeds)} on idle cards from {_cards(run)}:")
    for s in run.rules.seeds:
        print(f"  python -m trsi launch {run.path} --start --seed {s} --gpus <ids>")
    print(f"or, when the brief gives best_candidate (trained with this configuration): "
          f"python -m trsi adopt {run.path} --candidate <its directory>")
    return 0


def cmd_launch(a) -> int:
    run = load_run(Path(a.run))
    inc_c, inc_cid = _need_init(run)
    c, pred, is_start = _candidate(run, a)
    exp_sha = harness.exp_sha256(run)
    cid = candidate_id(c, exp_sha)
    is_start = is_start or cid == candidate_id(run.start, exp_sha)
    tid = f"{cid}-s{a.seed}"
    if not is_start:
        bad = outside_space(c, run.space)
        if bad:
            raise Refused("outside the task's search_space (widening it is a task change: ask in questions):\n  "
                          + "\n  ".join(bad))
        dead = dead_ends(c, run.space)
        if dead and not a.force:
            raise Refused("uses a dead end (add --force only with a reason the dead end does not apply):\n  "
                          + "\n  ".join(dead))
        mc = run.rules.max_configs
        if mc is not None and cid not in _proposed(run) and len(_proposed(run)) >= mc:
            raise Refused(f"this dispatch (returns included) has tried search.max_configs = {mc} configurations: "
                          "stop and report")
    if a.seed not in run.rules.seeds:
        raise Refused(f"seed {a.seed} is not in rules.seeds {list(run.rules.seeds)}")
    probe = run.rules.seeds[0]
    if not is_start and a.seed != probe and cid != inc_cid:
        pe = _probe_event(run, cid, inc_cid)
        if not (pe and pe["verdict"] == "more_seeds") and not a.force:
            raise Refused(f"seed {a.seed} comes after the probe: seed {probe} must beat the incumbent by more than "
                          f"{run.rules.probe_margin} first (run `trsi collect` and `trsi next`)")
    rec = archive.trial_record(run, tid)
    # --force on a trial whose packing keeps failing: it counts as failed (charged by the retry below), trains again
    again = bool(a.force and rec and archive.state_of(rec) == "unpacked" and not rec.get("adopted"))
    if any(e["trial"] == tid for e in archive.events(run, "use")):
        if rec and rec["status"] != "failed" and not again:
            print(f"{tid} is already part of this run ({archive.state_of(rec)}): {rec['run_dir']}")
            return 0
    if rec and rec["status"] in ("running", "done") and not again:
        _record_proposal(run, c, cid, pred, str(a.proposal))
        archive.append(run, {"event": "use", "trial": tid, "start": is_start, "charged": not rec.get("adopted"),
                             "reused": True})
        _progress(run)
        print(f"reused {tid} ({rec['status']}) from {rec['run_dir']}; no new training")
        return 0
    gpus = _gpus(a.gpus)
    if run.gpus.ids is not None and not set(gpus) <= set(run.gpus.ids):
        raise Refused(f"cards {gpus} are not in the task's resources.gpu_ids {list(run.gpus.ids)}")
    need = harness.recipe_gpus(c.recipe)
    if len(gpus) != need:
        raise Refused(f"recipe {c.recipe.name} trains on {need} card(s); got {len(gpus)}")
    est = estimate(run, c)
    if est is None and not is_start:
        raise Refused(f"no cost estimate for {c.base}: the task's resources.onejev.estimate has none")
    sp = spent(run)
    if est is not None and sp + est > run.budget_gpu_h:
        raise Refused(f"budget: {sp:.1f} spent + about {est:.1f} for this seed > {run.budget_gpu_h:.1f} compute "
                      "hours left in the brief; stop and report")
    _record_proposal(run, c, cid, pred, str(a.proposal))
    out = harness.launch(run, c, a.seed, gpus)
    # only now that ojh took it: a relaunch it refused (HarnessError) charges the failed run nothing more
    if again:
        r = rec.get("result") or {}
        r["error"] = f"not packed, trained again (launch --force): {r.get('pack_error')}"
        rec.update(status="failed", result=r)
        archive.write_trial(run, rec)
        archive.append(run, {"event": "result", "trial": tid, "status": "failed", "gpu_h": r.get("gpu_hours"),
                             "error": r["error"]})
    if rec and rec["status"] == "failed":
        archive.append(run, {"event": "retry", "trial": tid, "gpu_h": cost_of(run)(rec), "charged": True})
    new = {"trial": tid, "candidate_id": cid, "candidate": c.to_dict(), "seed": a.seed, "status": "running",
           "run_dir": out["run_dir"], "session": out.get("session"), "log": out.get("log"),
           "attempts": (rec or {}).get("attempts", 0) + 1, "result": None,
           # the runs it replaces: _tidy deletes their checkpoints once their DONE is there
           "old_run_dirs": ((rec or {}).get("old_run_dirs") or []) + ([rec["run_dir"]] if rec else [])}
    archive.write_trial(run, new)
    archive.append(run, {"event": "launch", "trial": tid, "run_dir": out["run_dir"], "gpus": gpus})
    archive.append(run, {"event": "use", "trial": tid, "start": is_start, "charged": True})
    _tidy(run, tid, new, say=lambda m: print(m, file=sys.stderr))   # the replaced run (launch --force: now)
    _progress(run)
    print(json.dumps(out, indent=2))
    print(f"launched {tid}. When every launch is done: python -m trsi watch {run.path} --marker <marker> "
          "(it goes to the background and writes <marker>.pid), then hand back long_job with that marker.")
    return 0


def cmd_adopt(a) -> int:
    run = load_run(Path(a.run))
    _need_init(run)
    fw = run.framework
    d = Path(a.candidate).resolve()
    m = json.loads((d / "manifest.json").read_text())
    req = m.get("config_request")
    if not m.get("id") or not isinstance(req, dict):
        raise Refused(f"{d} is not a registered candidate with a config_request")
    exp_sha = harness.exp_sha256(run)
    try:
        c = parse_candidate(request_mapping(req, "start", run.start.note), str(d), run.ojh.root)
    except CandidateError as e:
        raise Refused(str(e)) from None
    if candidate_id(c, exp_sha) != candidate_id(run.start, exp_sha):
        raise Refused(f"{m['id']} ({c.label()}) is not the start ({run.start.label()}); adopt only the candidate the "
                      "start came from")
    seed = int(req.get("seed") or 0)
    if seed not in run.rules.seeds:
        raise Refused(f"{m['id']} was trained with seed {seed}, not one of rules.seeds {list(run.rules.seeds)}")
    probs = [f"{k} {m.get(k)} != {v}" for k, v in (("trainset_id", fw.trainset_id), ("valset_id", fw.valset_id),
                                                     ) if m.get(k) != v]
    sha = file_sha256(fw.train_command)
    if (m.get("code_version") or {}).get("sha256") != sha:
        probs.append(f"trained by other code (code_version.sha256 {(m.get('code_version') or {}).get('sha256')}, "
                     f"now {sha})")
    if probs:
        raise Refused(f"{m['id']} cannot stand for the start here: " + "; ".join(probs)
                      + "; launch the start's seeds instead")
    hours = float((m.get("cost") or {}).get("compute_hours") or 0)
    res = TrialResult(str(d), seed, "done", pack.metrics(run, m, d), hours, compute_hours=hours, candidate_dir=str(d))
    tid = f"{candidate_id(run.start, exp_sha)}-s{seed}"
    archive.write_trial(run, {"trial": tid, "candidate_id": candidate_id(run.start, exp_sha),
                              "candidate": run.start.to_dict(), "seed": seed, "status": "done", "run_dir": str(d),
                              "adopted": True, "registered_id": m["id"], "result": res.to_dict()})
    archive.append(run, {"event": "adopt", "trial": tid, "candidate": m["id"], "dir": str(d)})
    archive.append(run, {"event": "use", "trial": tid, "start": True, "charged": False})
    print(f"adopted {m['id']} as {tid} (no training): {run.metric.primary} "
          f"{res.metrics.get(run.metric.primary):.4f}, ece {res.metrics['ece']:.4f}")
    return 0


def _tidy(run: RunConfig, tid: str, rec: dict, say=print) -> None:
    """Delete the checkpoints trial `tid`'s ojh runs no longer need; the user's standing rule, never asked.
    - Its run, packed (its candidate, checked by rsi validate, holds the model: hard links or copies): what ojh's
      deletion_candidates.json lists (unselected checkpoints, merged models), through `ojh cleanup --confirm`; ojh keeps
      the selected checkpoint. What ojh leaves (it says why) is reported once, not tried again at every collect.
    - Its run, failed by its DONE (harness.state "failed"): every checkpoint of the run; nothing in it is packed any more.
    - Runs it superseded (old_run_dirs: relaunched after a failure, a lost session or launch --force): every checkpoint
      once their DONE is there, whatever it says; they are never packed.
    Nothing while a run trains or has no DONE (a lost session: its runner may still write it), for an unpacked trial
    (the next collect packs it) or an adopted one. A failure is only said; the next collect tries again."""
    if rec.get("adopted"):
        return
    st = archive.state_of(rec)
    jobs = [(Path(d), "superseded") for d in rec.get("old_run_dirs") or []]
    if rec.get("run_dir") and st in ("done", "failed"):
        jobs.append((Path(rec["run_dir"]), st))
    cand = Path((rec.get("result") or {}).get("candidate_dir") or "")
    for run_dir, kind in jobs:
        try:
            if kind == "done":
                left = harness.deletion_left(run_dir)
                if not left or left == rec.get("cleanup_left") or not (cand / "manifest.json").is_file() \
                        or not (cand / "model").is_dir():
                    continue
                said = harness.cleanup(run, run_dir)
                rec["cleanup_left"] = after = harness.deletion_left(run_dir)
                archive.write_trial(run, rec)
                gone = [x for x in left if x not in after]
                if after:
                    say(f"{tid}  ojh cleanup left {len(after)} path(s) in {run_dir}: {said[-1000:]}")
            elif harness.state(run_dir) in (("failed",) if kind == "failed" else ("done", "failed")):
                gone = harness.delete_checkpoints(run_dir)
            else:
                continue
        except Exception as e:   # best effort: never fails a collect or a launch
            say(f"{tid}  checkpoints in {run_dir} not deleted ({type(e).__name__}: {e}); the next collect tries again")
            continue
        if gone:
            archive.append(run, {"event": "cleanup", "trial": tid, "run_dir": str(run_dir), "run": kind,
                                 "deleted": gone})
            say(f"{tid}  deleted {len(gone)} checkpoint dir(s) its candidate does not need in {run_dir}")


def _pack(run: RunConfig, a, tid: str, rec: dict, res: TrialResult, start: bool) -> dict:
    """Pack trial `tid` on the card harness.pick_scoring_card picks now (--gpu first when it qualifies; when it does
    not, say why), holding ojh's card lock meanwhile; after an out-of-memory error once more on another card.
    res.scoring_card is the uuid of the last try."""
    need, min_free, lock_dir = pack.scoring_needs(run, res)
    tried: tuple[str, ...] = ()
    while True:
        why: dict = {}
        i, uuid, lock = harness.pick_scoring_card(run.gpus.ids, need, min_free, lock_dir, tried, a.gpu, why)
        try:
            if a.gpu is not None and i != a.gpu and not tried:
                print(f"{tid}: card {a.gpu} cannot score now ({why.get(a.gpu, 'not listed by nvidia-smi')}); "
                      f"scoring on card {i} ({uuid}) instead")
            res.scoring_card = uuid
            _progress(run, packing=(tid, str(i)))
            return pack.build(run, rec, res, archive._cand_from(rec["candidate"]), start,
                              run.framework.outdir / f"cand-{tid}", uuid)
        except pack.PackOOM:
            if tried:
                raise
            print(f"{tid}: card {i} ({uuid}) ran out of memory; packing again on another card")
            tried = (uuid,)
        finally:
            harness.release(lock)


def cmd_collect(a) -> int:
    run = load_run(Path(a.run))
    _need_init(run)
    if a.gpu is not None and run.gpus.ids is not None and a.gpu not in run.gpus.ids:
        raise Refused(f"card {a.gpu} is not in the task's resources.gpu_ids {list(run.gpus.ids)}")
    m = run.metric.primary
    cost = cost_of(run)
    for tid, used in archive.trials(run).items():
        rec = archive.trial_record(run, tid)
        if rec["status"] == "running":
            if harness.state(Path(rec["run_dir"]), rec.get("session")) == "running":
                print(f"{tid}  running  {rec['run_dir']}")
                continue
            res = harness.read_result(Path(rec["run_dir"]), rec["seed"])
            rec.update(status=res.status, result=res.to_dict())
            archive.write_trial(run, rec)
            if res.status != "done":
                archive.append(run, {"event": "result", "trial": tid, "status": res.status, "gpu_h": res.gpu_hours,
                                     "error": res.error})
        if archive.state_of(rec) == "unpacked" and not rec.get("adopted"):
            res = TrialResult.from_dict(rec["result"])
            try:
                out = _pack(run, a, tid, rec, res, used["start"])
                res.metrics, res.compute_hours, res.candidate_dir = out["metrics"], out["compute_hours"], out["dir"]
                res.pack_error = None
            except (pack.PackError, HarnessError) as e:
                res.pack_error = f"packing the candidate failed (the next collect tries again): {e}"
            rec["result"] = res.to_dict()
            archive.write_trial(run, rec)
            archive.append(run, {"event": "result", "trial": tid, "status": archive.state_of(rec),
                                 m: res.metrics.get(m), "gpu_h": res.gpu_hours, "error": res.pack_error,
                                 "scoring_card": res.scoring_card})
            _progress(run)
        _tidy(run, tid, rec)
        r = rec.get("result") or {}
        val = (r.get("metrics") or {}).get(m)
        err = r.get("pack_error") if archive.state_of(rec) == "unpacked" else r.get("error")
        print(f"{tid}  {archive.state_of(rec):<8} {m} {val if val is None else round(val, 4)}  {cost(rec):.2f} h"
              + (f"  {r['candidate_dir']}" if r.get("candidate_dir") else "") + (f"  {err}" if err else ""))
    _progress(run)
    return 0


def cmd_next(a) -> int:
    run = load_run(Path(a.run))
    inc_c, inc_cid = _need_init(run)
    seeds = run.rules.seeds
    probe = seeds[0]
    m = run.metric.primary
    todo: list[str] = []
    done, recs = _results(run)
    inc_missing = [s for s in seeds if s not in done.get(inc_cid, {})]
    if inc_missing:
        st = {s: archive.state_of(recs[inc_cid][s]) for s in inc_missing if s in recs.get(inc_cid, {})}
        waiting = [s for s, x in st.items() if x == "running"]
        unpacked = [s for s, x in st.items() if x == "unpacked"]
        todo.append(f"incumbent seeds {inc_missing} not finished" + (f" (running: {waiting})" if waiting else "")
                    + (f" (not packed: {unpacked}; run collect again)" if unpacked else "")
                    + "; candidates are compared once they are")
    else:
        if not any(e["candidate"] == inc_cid for e in archive.events(run, "curve")):
            mean = sum(done[inc_cid][s].metrics[m] for s in seeds) / len(seeds)
            archive.append(run, {"event": "curve", "candidate": inc_cid, "spent": spent(run),
                                 "incumbent_mean": mean, "why": "incumbent complete"})
        verdicts = {e["candidate"] for e in archive.events(run, "verdict")}
        for e in archive.events(run, "propose"):
            cid = e["candidate"]
            inc_c, inc_cid = archive.incumbent(run)  # may change inside this loop
            if cid in verdicts or cid == inc_cid:
                continue
            c = archive._cand_from(e["candidate_dict"])
            res = done.get(cid, {})
            if probe not in res:
                pr = recs.get(cid, {}).get(probe)
                st = archive.state_of(pr) if pr else None
                todo.append(f"`{cid}` {c.label()}: probe seed {probe} {st or 'not launched'}"
                            + (" -> relaunch it or drop the candidate" if st == "failed" else "")
                            + (" -> run collect again" if st == "unpacked" else ""))
                continue
            pe = _probe_event(run, cid, inc_cid)
            if pe is None:
                v, g = after_probe(res[probe], done[inc_cid][probe], run.rules, run.metric)
                pe = {"event": "probe", "candidate": cid, "against": inc_cid, "verdict": v, "gain": g}
                archive.append(run, pe)
            if pe["verdict"] == "reject":
                detail = f"probe seed {probe} gain {pe['gain']:+.4f} <= {run.rules.probe_margin}"
                archive.append(run, {"event": "verdict", "candidate": cid, "against": inc_cid, "verdict": "rejected",
                                     "stage": "probe", "detail": detail})
                todo.append(f"`{cid}` {c.label()}: rejected ({detail})")
                continue
            missing = [s for s in seeds if s not in res]
            if missing:
                running = [s for s in missing if recs.get(cid, {}).get(s, {}).get("status") in ("running", "done")]
                launch = [s for s in missing if s not in running]
                todo.append(f"`{cid}` {c.label()}: probe passed ({pe['gain']:+.4f}); "
                            + (f"launch seeds {launch}: python -m trsi launch {run.path} <its proposal> --seed S "
                               "--gpus <ids>" if launch else f"waiting for seeds {running}"))
                continue
            ok, detail = decide(res, done[inc_cid], run.rules, run.metric)
            archive.append(run, {"event": "verdict", "candidate": cid, "against": inc_cid,
                                 "verdict": "accepted" if ok else "rejected", "stage": "decide", "detail": detail})
            todo.append(f"`{cid}` {c.label()}: {'ACCEPTED' if ok else 'rejected'} ({detail})")
            if ok:
                archive.set_incumbent(run, c, cid, detail)
                archive.append(run, {"event": "curve", "candidate": cid, "spent": spent(run),
                                     "incumbent_mean": sum(res[s].metrics[m] for s in seeds) / len(seeds),
                                     "why": f"accepted {cid}"})
    sp = spent(run)
    left = run.budget_gpu_h - sp
    inc_c, inc_cid = archive.incumbent(run)
    print(f"incumbent `{inc_cid}` {inc_c.label()}")
    print(f"budget    {sp:.1f} / {run.budget_gpu_h:.1f} compute hours spent, {left:.1f} left")
    for t in todo:
        print(f"- {t}")
    open_c = [e for e in archive.events(run, "propose")
              if e["candidate"] not in {v["candidate"] for v in archive.events(run, "verdict")}]
    for tid, rec in archive.trials(run).items():
        if archive.state_of(rec) == "unpacked" and not rec.get("adopted"):
            print(f"- {tid} not packed: {(rec.get('result') or {}).get('pack_error')} (the next collect tries again; "
                  "`launch ... --force` trains it again)")
    stop = stop_reason(run)
    if stop:
        print(stop)
    elif not inc_missing and not open_c:
        print("no open candidate: diagnose the incumbent and write 1-2 proposals")
    return 0


def read_ids(paths: list[str]) -> dict:
    """Candidate directory name -> registered id, from the engine's registered-<k>.json ([{"dir", "kind", "id"}];
    entries of other kinds skipped) or a JSON object {directory: id}; directories may be paths."""
    out = {}
    for path in paths:
        d = json.loads(Path(path).read_text())
        if isinstance(d, list) and all(isinstance(x, dict) and "dir" in x and "id" in x for x in d):
            out.update((Path(x["dir"]).name, x["id"]) for x in d if x.get("kind", "candidate") == "candidate")
        elif isinstance(d, dict):
            out.update((Path(k).name, v) for k, v in d.items())
        else:
            raise Refused(f"{path}: expected the engine's registered-<k>.json ([{{dir, kind, id}}]) or a JSON object "
                          "{candidate directory: registered id}")
    return out


def cmd_report(a) -> int:
    run = load_run(Path(a.run))
    _need_init(run)
    print(report.write(run, cost_of(run), spent(run)))
    if bool(a.search) != bool(a.ids):
        raise Refused("--search and --ids go together")
    if a.search:
        dest = report.write_search(run, cost_of(run), read_ids(a.ids), Path(a.search))
        v = pack.rsi("validate", "search", dest, "--task", run.framework.task)
        if v.returncode != 0:
            print(f"rsi validate search {dest}:\n{(v.stderr or v.stdout).strip()}", file=sys.stderr)
            return 2
        print(dest)
    return 0


# ---- watch --------------------------------------------------------------------------------------------------

class _Stopped(BaseException):
    """SIGTERM or SIGINT in the watcher; a BaseException, so no `except Exception` (progress.json's) swallows it."""


def _write_marker(marker: Path, text: str) -> None:
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, marker)


def _end_trainings(run: RunConfig, trials: list[tuple[str, dict]]) -> list[str]:
    """Kill the tmux session of every ojh training of the run still running: the watched trials and every trial of
    the run now (one launched after this watch started, by a watcher being replaced, too). Returns them as
    "<trial> (<session>)"."""
    try:
        now = archive.trials(run)
    except Exception:
        now = {}
    ended = []
    for tid, r in {**dict(trials), **now}.items():
        if not r.get("session"):
            continue
        try:
            running = harness.state(Path(r["run_dir"]), r["session"]) == "running"
        except Exception:   # an unreadable DONE: the session goes all the same
            running = True
        if running:
            harness.kill_session(r["session"])
            ended.append(f"{tid} ({r['session']})")
    return ended


# How many polls in a row may fail to read the run directories with an OSError (a flaky network filesystem) before the
# watch gives up: until then it logs the error and polls again, the trainings untouched.
WATCH_READ_ERRORS = 3


def _watch_loop(run: RunConfig, trials: list[tuple[str, dict]], marker: Path, pid_file: Path, poll: float) -> int:
    """Rewrites progress.json at every poll (the last time just before the ok marker) and on a stop; a failure to
    write it only goes to the log, so the marker is always written. A stop (SIGTERM, SIGINT) or an error ends the
    run's ojh trainings before the failed marker: once the marker is there, nothing watches them any more. An OSError
    reading the run directories is such an error only on WATCH_READ_ERRORS polls in a row."""
    def stop(signum, frame):
        raise _Stopped(signal.Signals(signum).name)

    before = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        _write_marker(pid_file, f"{os.getpid()}\n{int(time.time())}\n")
        failed = 0
        while True:
            try:
                ended = all(harness.state(Path(r["run_dir"]), r.get("session")) != "running" for _, r in trials)
            except OSError:
                failed += 1
                if failed >= WATCH_READ_ERRORS:
                    raise
                traceback.print_exc()
                print(f"watch: reading the trainings failed ({failed} of {WATCH_READ_ERRORS} polls in a row); "
                      "trying again at the next poll", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(poll)
                continue
            failed = 0
            _progress(run)
            if ended:
                _write_marker(marker, "ok\n")
                return 0
            time.sleep(poll)
    except (_Stopped, Exception) as e:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if isinstance(e, _Stopped):
            why = f"stopped by {e}"
        else:
            traceback.print_exc()
            sys.stderr.flush()
            why = "the watcher stopped on an error (%s)" % f"{type(e).__name__}: {e}".replace("\n", " ")[:1000]
        ended = _end_trainings(run, trials)
        _progress(run)
        _write_marker(marker, f"failed\n{why}; ended the ojh trainings: {', '.join(ended) or 'none'}\n")
        return 1
    finally:  # a foreground watch leaves its process as it found it
        for s, h in before.items():
            signal.signal(s, h)


def cmd_watch(a) -> int:
    run = load_run(Path(a.run))
    _need_init(run)
    marker = Path(a.marker).resolve()
    pid_file = marker.with_name(marker.name + ".pid")
    if marker.exists():
        raise Refused(f"{marker} exists: give each watch a new marker path")
    trials = [(tid, rec) for tid, rec in archive.trials(run).items() if rec["status"] == "running"]
    with contextlib.suppress(FileNotFoundError):
        pid_file.unlink()
    marker.parent.mkdir(parents=True, exist_ok=True)
    if a.foreground:
        return _watch_loop(run, trials, marker, pid_file, a.poll)
    sys.stdout.flush()
    sys.stderr.flush()
    child = os.fork()
    if child == 0:  # the watcher: its own session (process group), so a stop reaches it and nothing else
        code = 1
        try:
            os.setsid()
            log = os.open(str(marker) + ".log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            null = os.open(os.devnull, os.O_RDONLY)
            os.dup2(null, 0)
            os.dup2(log, 1)
            os.dup2(log, 2)
            code = _watch_loop(run, trials, marker, pid_file, a.poll)
        except BaseException as e:   # never end without a marker: the engine would wait for it for ever
            with contextlib.suppress(BaseException):
                traceback.print_exc()
                sys.stderr.flush()
            with contextlib.suppress(BaseException):
                if not marker.exists():
                    why = f"{type(e).__name__}: {e}".replace("\n", " ")[:1000]
                    _write_marker(marker, f"failed\nthe watcher stopped on an error ({why}); see {marker}.log\n")
        finally:
            os._exit(code)
    end = time.time() + 30
    while time.time() < end and not (pid_file.is_file() and len(pid_file.read_text().split()) == 2):
        time.sleep(0.02)
    if not pid_file.is_file():
        raise Refused(f"the watcher (pid {child}) did not write {pid_file}")
    print(f"watching {len(trials)} trial(s) in process {child}: {', '.join(t for t, _ in trials) or 'none'}")
    print(f"marker {marker} (ok when they have all ended; pid in {pid_file}; log {marker}.log)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m trsi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"trsi {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("run", nargs="?", help="an existing run.yaml (instead of --brief)")
    p.add_argument("--brief", help="the train brief (brief.json) to write run.yaml and exp.yaml from")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("check")
    p.add_argument("run")
    p.add_argument("--ojh", action="store_true", help="also run `ojh check` on the start (no cards used)")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("collect")
    p.add_argument("run")
    p.add_argument("--gpu", type=int, help="the card (from resources.gpu_ids) model.predict scores on when it "
                                           "qualifies (no other process, enough free memory, ojh's card lock free); "
                                           "else, and by default, the qualifying card with the most free memory")
    p.set_defaults(fn=cmd_collect)
    p = sub.add_parser("next")
    p.add_argument("run")
    p.set_defaults(fn=cmd_next)
    p = sub.add_parser("report")
    p.add_argument("run")
    p.add_argument("--search", help="also write the framework's search artifact into this directory")
    p.add_argument("--ids", nargs="+", help="the engine's registered-<k>.json after the first part of the hand-back "
                                            "(or a JSON object {candidate directory: registered id})")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("launch")
    p.add_argument("run")
    p.add_argument("proposal", nargs="?")
    p.add_argument("--start", action="store_true")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpus", required=True, help="idle card ids from the task's resources.gpu_ids, e.g. 0 or 2,3")
    p.add_argument("--force", action="store_true", help="allow a dead end or seeds after a probe that did not pass; "
                                                        "train again a trial whose packing keeps failing")
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("adopt")
    p.add_argument("run")
    p.add_argument("--candidate", required=True, help="a registered candidate directory (the brief's best_candidate)")
    p.set_defaults(fn=cmd_adopt)
    p = sub.add_parser("watch")
    p.add_argument("run")
    p.add_argument("--marker", required=True, help="the completion marker to write (a new path each time)")
    p.add_argument("--poll", type=float, default=60.0, help="seconds between checks (default 60)")
    p.add_argument("--foreground", action="store_true", help="do not go to the background")
    p.set_defaults(fn=cmd_watch)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (Refused, ConfigError, CandidateError, HarnessError, report.SearchError) as e:
        print(f"trsi: {type(e).__name__.lower()}: {e}", file=sys.stderr)
        return 2
