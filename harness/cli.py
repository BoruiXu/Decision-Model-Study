"""The ``ojh`` command line. Imports without torch; GPU work happens inside the modules it calls."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from harness import (cleanup, doctor, fetch_base, finalize, gpu, inspect_data, launch, make_split, onejev,
                     oneshot, prepare, registry, report_eval, resolve, runner, status)
from harness.catalog import ROOT, CatalogError, sha256_file
from harness.config import ConfigError, check_writable, find_machine, load_experiment
from harness.gpu import GpuUnavailable, GpuWaitTimeout
from harness.prepare import IdOverlap, ShaMismatch, TestSplitRefused
from harness.report_eval import ReportRefused
from harness.resolve import Infeasible

KNOWN = (FileNotFoundError, ConfigError, CatalogError, ShaMismatch, IdOverlap, TestSplitRefused, ReportRefused, GpuWaitTimeout,
         GpuUnavailable, Infeasible)
REFUSALS = (TestSplitRefused, ReportRefused, Infeasible)


def _lock_of(run_dir) -> dict:
    p = Path(run_dir) / "run.lock.yaml"
    if not p.is_file():
        raise ConfigError(f"{run_dir} has no run.lock.yaml: pass a run directory")
    return yaml.safe_load(p.read_text())


def _fetch_patch(lock: dict, dest: Path) -> Path:
    """Download serve.patch at its pinned revision into ``dest`` unless a verified copy is already there."""
    spec = lock["patch"]
    if not (dest.is_file() and sha256_file(dest) == spec["sha256"]):
        from huggingface_hub import hf_hub_download

        got = Path(hf_hub_download(repo_id=spec["hf_repo"], filename=spec["file"], revision=spec["revision"],
                                   local_dir=dest.parent))
        if got.resolve() != dest.resolve():
            dest.write_bytes(got.read_bytes())
    if sha256_file(dest) != spec["sha256"]:
        raise ConfigError(f"Unexpected SHA-256 for {dest}")
    return dest


def cmd_inspect(a) -> int:
    report = inspect_data.inspect_path(a.path, rows=a.rows)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str) if a.json else inspect_data.render(report))
    return 0


def _set_files(args) -> list[tuple[str, str]]:
    out = []
    for arg in args:
        name, sep, path = arg.partition("=")
        if sep and name and not os.path.exists(arg):
            out.append((name, path))
        else:
            out.append((Path(arg).stem, arg))
    return out


def cmd_make_split(a) -> int:
    if a.mode == "holdout":
        if len(a.files) != 1:
            raise ConfigError("--mode holdout takes exactly one train file")
        manifest = make_split.holdout(a.files[0], a.out, group=a.group, seed=a.seed, frac=a.frac,
                                      allow_test=a.allow_test)
        print(json.dumps({"out_dir": str(Path(a.out).resolve()), "rows": manifest["rows"],
                          "groups": {k: len(v) for k, v in manifest["groups"].items()},
                          "sha256": manifest["sha256"]}, indent=2))
        return 0
    obj = make_split.select_report(_set_files(a.files), group=a.group, seed=a.seed, frac=a.frac,
                                   allow_test=a.allow_test, id_field=a.id)
    path = make_split.write_split_file(obj, a.out)
    print(json.dumps({"split_file": str(path.resolve()),
                      "sets": {k: {"rows": v["rows"], "groups": {s: len(g) for s, g in v["groups"].items()}}
                               for k, v in obj.items()}}, indent=2))
    return 0


def cmd_doctor(a) -> int:
    checks = doctor.run_checks(doctor.default_probes(), network=not a.no_network, timeout=a.timeout)
    print(json.dumps(doctor.as_json(checks), indent=2) if a.json else doctor.render(checks))
    return doctor.exit_code(checks)


def cmd_setup(a) -> int:
    find_machine(a.machine)
    lock = onejev.load_lock()
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    patch = _fetch_patch(lock, vendor / "serve.patch")
    dest = onejev.ensure_onejev(vendor / "OneJev", lock, patch)
    print(f"OneJev ready at {dest} (revision {lock['revision']}, patch verified)")
    return 0


def cmd_fetch_base(a) -> int:
    m, _ = find_machine(a.machine)
    check_writable(m, Path(m.model_dir) / a.name)
    print(fetch_base.fetch_base(a.name, m.model_dir))
    return 0


def cmd_prepare(a) -> int:
    (m, _), exp = find_machine(a.machine), load_experiment(a.exp)
    out = Path(m.out_root) / exp.name / "prepared"
    check_writable(m, out)
    info = prepare.prepare(exp, out, allow_test=a.allow_test, source=os.path.abspath(a.exp), force=a.force)
    print(json.dumps(info["counts"], indent=2))
    return 0


def cmd_resolve(a) -> int:
    try:
        m, origin = find_machine(a.machine)
        path = resolve.resolve(load_experiment(a.exp), m, machine_origin=origin, allow_shared=a.allow_shared,
                               source=os.path.abspath(a.exp))
    except Infeasible as e:
        print(f"infeasible: {e}", file=sys.stderr)
        for alt in e.alternatives:
            print(f"  alternative: {alt}", file=sys.stderr)
        return 2
    print(path)
    return 0


def cmd_run(a) -> int:
    gpus = oneshot.parse_gpus(a.gpus) if a.gpus is not None else None
    out = oneshot.run(a.exp, base=a.base, gpus=gpus, recipe=a.recipe, machine=a.machine, allow_test=a.allow_test,
                      allow_shared=a.allow_shared, force=a.force)
    print(json.dumps(out, indent=2))
    return 0


def cmd_check(a) -> int:
    gpus = oneshot.parse_gpus(a.gpus) if a.gpus is not None else None
    out = oneshot.check(a.exp, base=a.base, gpus=gpus, recipe=a.recipe, machine=a.machine,
                        allow_test=a.allow_test, force=a.force)
    print(json.dumps(out, indent=2))
    return 0


def cmd_launch(a) -> int:
    print(json.dumps(launch.launch(a.lock), indent=2))
    return 0


def cmd_status(a) -> int:
    print(status.render(status.status(a.run)))
    return 0


def _on_one_gpu(run, fn, allow_shared: bool = False, wait: bool = False, retry: str | None = None):
    """Score on one card: the run's first explicit card (or one picked like the run's). A shared card is used
    when the lock allows it (runner.lock_allows_sharing) or the command got --allow-shared. A foreground
    command checks once and never blocks: without ``wait`` (--wait) a busy card is refused with each card's
    problem and the runnable ``retry`` lines."""
    lock = _lock_of(run)
    g = lock["machine"]["gpu"]
    shared = allow_shared or runner.lock_allows_sharing(lock)
    if g["source"] == "explicit":
        kw = {"explicit": list(g["explicit"])[:1], "allow_shared": shared}
    elif g["source"] == "scan":
        kw = {"allow_shared": shared}
    else:
        kw = {}
    machine = runner._machine(lock["machine"])
    try:
        lease = gpu.acquire(machine, 1, wait=wait, **kw)
    except GpuUnavailable as e:
        hint = []
        if retry and not wait:
            hint.append(f"wait for a card (every {gpu.RETRY_S} s, up to {machine.wait_timeout_s} s): `{retry} --wait`")
        if retry and not kw.get("allow_shared") and any("process" in r for r in e.reasons.values()):
            hint.append(f"only if the user agrees to share the card with those processes: `{retry} --allow-shared`")
        raise GpuUnavailable(e.reasons, e.cards, head=e.head, hint=e.hint + hint) from None
    try:
        return fn(lease)
    finally:
        lease.release()


def cmd_finalize(a) -> int:
    out = _on_one_gpu(a.run, lambda lease: finalize.finalize(Path(a.run), devices=lease.uuids),
                      allow_shared=a.allow_shared, wait=a.wait, retry=f"ojh finalize {a.run}")
    print(f"{out['run_id']}: {out['status']}")
    for f in out["failures"]:
        print(f"  {f}", file=sys.stderr)
    return 0 if out["status"] == "done" else 1


def cmd_note(a) -> int:
    lock = _lock_of(a.run)
    print(registry.note(runner._machine(lock["machine"]).registry_path, lock["run_id"], a.text))
    return 0


def cmd_eval_report(a) -> int:
    if not a.allow_report:
        raise ReportRefused("report halves are read only with --allow-report")
    retry = (f"ojh eval-report {a.run} --allow-report" + (" --allow-test" if a.allow_test else "")
             + (f" --seed {a.seed}" if a.seed is not None else ""))
    out = _on_one_gpu(a.run, lambda lease: report_eval.run(
        Path(a.run), allow_report=True, seed=a.seed, allow_test=a.allow_test, devices=lease.uuids),
        allow_shared=a.allow_shared, wait=a.wait, retry=retry)
    print(f"wrote {Path(a.run) / 'report_eval.json'} (seed {out['seed']})")
    return 0


def cmd_table(a) -> int:
    path = a.registry or find_machine(a.machine)[0].registry_path
    print(registry.table(path, include_incomplete=a.all))
    return 0


def cmd_cleanup(a) -> int:
    cands = cleanup.candidates(a.run)
    if not cands:
        print("no deletion candidates")
        return 0
    for c in cands:
        print(f"{c['bytes']:>14}  {c['path']}")
    if not a.confirm:
        print(f"{len(cands)} candidate(s); nothing deleted (add --confirm to delete)")
        return 0
    done, skipped = cleanup.delete_report(a.run, confirm=True)
    for c in done:
        print(f"deleted {c['path']}")
    for c in skipped:
        print(f"skipped {c['path']}: {c['reason']}", file=sys.stderr)
    return 0


MACHINE_HELP = "machine config (default: $OJH_MACHINE, then ~/.config/onejev-harness/machine.yaml, else inferred)"

MAKE_SPLIT_HELP = """Deterministic, group-aware splits (same inputs and seed -> same output).

--mode select-report (default): for each FILE (an eval set; its name is the file stem, or NAME=PATH),
  whole groups go to "select" or "report" so the select share is about --frac. Writes --out as the
  data.split_file format {set: {"assignment": {id: "select"|"report"}, "seed", "frac", "group", "source",
  "source_sha256", "rows", "groups", "id_field"}}. Ids come from --id (a dotted field, default "id") and are
  written as text, e.g. "0007" stays "0007"; the adapter must yield that same text as each row's id, so raw
  files (csv, chat jsonl) can be split before any conversion.
  An existing --out with different content is never overwritten.

--mode holdout: carve a validation part out of ONE train FILE by whole groups (about --frac of the rows go
  to val). Writes into the directory --out:
    train.jsonl    the remaining rows, unchanged, in source order, one JSON object per line
    val.jsonl      the held-out rows, unchanged, in source order
    manifest.json  {"mode": "holdout", "source", "source_sha256", "group", "seed", "frac",
                    "rows": {"train", "val"}, "groups": {"train": [...], "val": [...]},
                    "sha256": {"train.jsonl", "val.jsonl"}}
  Rows are copied as read; csv/tsv cells stay text: a label 1 is written as "1", a score 1.50 as "1.50", so the
  adapter compares strings (r["label"] == "1") or casts explicitly (float(r["label"])), never == 1. No ids are
  added or changed, so the adapter
  reads both files the same way and keeps its own id rule. Existing output files are never overwritten.

Test data is refused without --allow-test: rows whose split-like column (split, set, subset, partition,
fold, ...; any case, any dict depth) starts with "test", and files whose name or directory starts with "test".
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ojh", description="Training harness around OneJev")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def cmd(name, fn, help_):
        s = sub.add_parser(name, help=help_, description=help_)
        s.set_defaults(fn=fn)
        return s

    s = cmd("inspect", cmd_inspect, "read-only facts about data files (no decisions, writes nothing)")
    s.add_argument("path", help="a data file or a directory")
    s.add_argument("--json", action="store_true", help="print the full report as JSON")
    s.add_argument("--rows", type=int, default=3, help="sample rows per file (default 3)")
    s = sub.add_parser(
        "make-split", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="deterministic, group-aware select/report split file or train/val holdout",
        description=MAKE_SPLIT_HELP)
    s.set_defaults(fn=cmd_make_split)
    s.add_argument("files", nargs="+", metavar="FILE",
                   help="select-report: eval files (set name = file stem, or NAME=PATH); holdout: one train file")
    s.add_argument("--out", required=True,
                   help="select-report: the split file to write; holdout: the output directory")
    s.add_argument("--group", required=True, help="dotted group field; whole groups stay on one side")
    s.add_argument("--id", default="id", metavar="FIELD",
                   help="select-report: dotted field holding each row's id, kept as text (default: id); the "
                        "adapter must yield the same value as the row id. Not used by holdout")
    s.add_argument("--seed", type=int, default=make_split.SEED, help=f"default {make_split.SEED}")
    s.add_argument("--frac", type=float, default=0.5,
                   help="select-report: select share; holdout: val share (default 0.5)")
    s.add_argument("--mode", choices=("select-report", "holdout"), default="select-report")
    s.add_argument("--allow-test", action="store_true", help="read test* rows and files (needs approval)")
    s = cmd("doctor", cmd_doctor, "read-only environment check (python, torch/CUDA, GPUs, pins, OneJev, tmux, "
                                  "directories, HF hub); exit 1 on a blocking problem")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-network", action="store_true", help="skip the Hugging Face hub check")
    s.add_argument("--timeout", type=float, default=5.0, help="network timeout in seconds (default 5)")
    s = cmd("setup", cmd_setup, "fetch OneJev and the patch, verify sha256")
    s.add_argument("--machine", help=MACHINE_HELP)
    s = cmd("fetch-base", cmd_fetch_base, "download a catalog base model and verify it")
    s.add_argument("name")
    s.add_argument("--machine", help=MACHINE_HELP)
    s = cmd("prepare", cmd_prepare, "validate and convert experiment data")
    s.add_argument("exp")
    s.add_argument("--machine", help=MACHINE_HELP)
    s.add_argument("--allow-test", action="store_true", help="read split test* rows")
    s.add_argument("--force", action="store_true",
                   help="overwrite data prepared from another experiment file with the same name")
    s = cmd("resolve", cmd_resolve, "resolve an experiment into run.lock.yaml")
    s.add_argument("exp")
    s.add_argument("--machine", help=MACHINE_HELP)
    s.add_argument("--allow-shared", action="store_true",
                   help="use requested cards that carry other processes (ask the user first; recorded in the lock)")
    s = cmd("run", cmd_run, "prepare, resolve and launch in one call; prints a JSON summary and returns at once")
    s.add_argument("exp")
    s.add_argument("--base", help="catalog base (overrides or fills base in the experiment; never chosen for you)")
    s.add_argument("--gpus", help="card indices such as 0,1 (overrides or fills gpus in the experiment)")
    s.add_argument("--recipe", help="recipe file (default: the experiment's recipe, else the base's default_recipes)")
    s.add_argument("--machine", help=MACHINE_HELP)
    s.add_argument("--allow-test", action="store_true", help="read split test* rows (needs approval)")
    s.add_argument("--force", action="store_true",
                   help="overwrite data prepared from another experiment file with the same name")
    s.add_argument("--allow-shared", action="store_true",
                   help="use requested cards that carry other processes (ask the user first; recorded in the lock)")
    s = cmd("check", cmd_check, "validate an experiment like ojh run (prepare, config, adapter, plan-only, "
                                "fingerprint) without judging the cards; writes no lock and launches nothing")
    s.add_argument("exp")
    s.add_argument("--base", help="catalog base (as ojh run)")
    s.add_argument("--gpus", help="card indices such as 0,1 (as ojh run; only their count is used, the cards are "
                                  "not checked)")
    s.add_argument("--recipe", help="recipe file (as ojh run)")
    s.add_argument("--machine", help=MACHINE_HELP)
    s.add_argument("--allow-test", action="store_true", help="read split test* rows (needs approval)")
    s.add_argument("--force", action="store_true",
                   help="check even if another experiment file prepared data under this name (check never "
                        "overwrites prepared data: it validates in a scratch dir)")
    s = cmd("launch", cmd_launch, "start a run in tmux and return at once")
    s.add_argument("lock")
    s = cmd("status", cmd_status, "short progress report of a run")
    s.add_argument("run")
    s = cmd("finalize", cmd_finalize, "select, rescore, evaluate and register a run")
    s.add_argument("run")
    s.add_argument("--wait", action="store_true",
                   help="wait for a free card (polls up to the machine's wait_timeout_s); without it the command "
                        "checks once and never blocks")
    s.add_argument("--allow-shared", action="store_true",
                   help="score on the run's card even if other processes use it (ask the user first)")
    s = cmd("note", cmd_note, "set the conclusion of a run in the registry")
    s.add_argument("run")
    s.add_argument("text")
    s = cmd("eval-report", cmd_eval_report, "evaluate the report halves with the locked checkpoint")
    s.add_argument("run")
    s.add_argument("--wait", action="store_true",
                   help="wait for a free card (polls up to the machine's wait_timeout_s); without it the command "
                        "checks once and never blocks")
    s.add_argument("--allow-shared", action="store_true",
                   help="score on the run's card even if other processes use it (ask the user first)")
    s.add_argument("--allow-report", action="store_true", help="required: read report halves")
    s.add_argument("--allow-test", action="store_true", help="read split test* rows")
    s.add_argument("--seed", type=int, default=None, help="default: first seed of the run")
    s = cmd("table", cmd_table, "render the registry as a comparison table")
    s.add_argument("--machine", help=MACHINE_HELP)
    s.add_argument("--registry", help="registry file (default: the machine config's registry)")
    s.add_argument("--all", action="store_true", help="include incomplete records")
    s = cmd("cleanup", cmd_cleanup, "list, and with --confirm delete, unselected checkpoints")
    s.add_argument("run")
    s.add_argument("--confirm", action="store_true", help="actually delete")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args) or 0
    except KNOWN as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2 if isinstance(e, REFUSALS) else 1


if __name__ == "__main__":
    sys.exit(main())
