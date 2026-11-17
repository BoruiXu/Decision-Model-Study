"""The ``ojh`` command line. Imports without torch; GPU work happens inside the modules it calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from harness import (cleanup, fetch_base, finalize, gpu, launch, onejev, prepare, registry, report_eval,
                     resolve, runner, status)
from harness.catalog import ROOT, sha256_file
from harness.config import ConfigError, load_experiment, load_machine
from harness.gpu import GpuWaitTimeout
from harness.prepare import IdOverlap, ShaMismatch, TestSplitRefused
from harness.report_eval import ReportRefused
from harness.resolve import Infeasible

KNOWN = (ConfigError, ShaMismatch, IdOverlap, TestSplitRefused, ReportRefused, GpuWaitTimeout, Infeasible)
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


def cmd_setup(a) -> int:
    load_machine(a.machine)
    lock = onejev.load_lock()
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    patch = _fetch_patch(lock, vendor / "serve.patch")
    dest = onejev.ensure_onejev(vendor / "OneJev", lock, patch)
    print(f"OneJev ready at {dest} (revision {lock['revision']}, patch verified)")
    return 0


def cmd_fetch_base(a) -> int:
    m = load_machine(a.machine)
    print(fetch_base.fetch_base(a.name, m.model_dir))
    return 0


def cmd_prepare(a) -> int:
    m, exp = load_machine(a.machine), load_experiment(a.exp)
    info = prepare.prepare(exp, Path(m.out_root) / exp.name / "prepared", allow_test=a.allow_test)
    print(json.dumps(info["counts"], indent=2))
    return 0


def cmd_resolve(a) -> int:
    try:
        path = resolve.resolve(load_experiment(a.exp), load_machine(a.machine))
    except Infeasible as e:
        print(f"infeasible: {e}", file=sys.stderr)
        for alt in e.alternatives:
            print(f"  alternative: {alt}", file=sys.stderr)
        return 2
    print(path)
    return 0


def cmd_launch(a) -> int:
    print(json.dumps(launch.launch(a.lock), indent=2))
    return 0


def cmd_status(a) -> int:
    print(status.render(status.status(a.run)))
    return 0


def _on_one_gpu(run, fn):
    lock = _lock_of(run)
    lease = gpu.acquire(runner._machine(lock["machine"]), 1)
    try:
        return fn(lease)
    finally:
        lease.release()


def cmd_finalize(a) -> int:
    out = _on_one_gpu(a.run, lambda lease: finalize.finalize(Path(a.run), devices=lease.uuids))
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
    out = _on_one_gpu(a.run, lambda lease: report_eval.run(
        Path(a.run), allow_report=True, seed=a.seed, allow_test=a.allow_test, devices=lease.uuids))
    print(f"wrote {Path(a.run) / 'report_eval.json'} (seed {out['seed']})")
    return 0


def cmd_table(a) -> int:
    if a.registry:
        path = a.registry
    elif a.machine:
        path = load_machine(a.machine).registry_path
    else:
        raise ConfigError("table needs --machine or --registry")
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
    done = cleanup.delete(a.run, confirm=True)
    for c in done:
        print(f"deleted {c['path']}")
    skipped = len(cands) - len(done)
    if skipped:
        print(f"{skipped} candidate(s) skipped (outside the run dir or already gone)", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ojh", description="Training harness around OneJev")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def cmd(name, fn, help_):
        s = sub.add_parser(name, help=help_, description=help_)
        s.set_defaults(fn=fn)
        return s

    s = cmd("setup", cmd_setup, "fetch OneJev and the patch, verify sha256")
    s.add_argument("--machine", required=True)
    s = cmd("fetch-base", cmd_fetch_base, "download a catalog base model and verify it")
    s.add_argument("name")
    s.add_argument("--machine", required=True)
    s = cmd("prepare", cmd_prepare, "validate and convert experiment data")
    s.add_argument("exp")
    s.add_argument("--machine", required=True)
    s.add_argument("--allow-test", action="store_true", help="read split test* rows")
    s = cmd("resolve", cmd_resolve, "resolve an experiment into run.lock.yaml")
    s.add_argument("exp")
    s.add_argument("--machine", required=True)
    s = cmd("launch", cmd_launch, "start a run in tmux and return at once")
    s.add_argument("lock")
    s = cmd("status", cmd_status, "short progress report of a run")
    s.add_argument("run")
    s = cmd("finalize", cmd_finalize, "select, rescore, evaluate and register a run")
    s.add_argument("run")
    s = cmd("note", cmd_note, "set the conclusion of a run in the registry")
    s.add_argument("run")
    s.add_argument("text")
    s = cmd("eval-report", cmd_eval_report, "evaluate the report halves with the locked checkpoint")
    s.add_argument("run")
    s.add_argument("--allow-report", action="store_true", help="required: read report halves")
    s.add_argument("--allow-test", action="store_true", help="read split test* rows")
    s.add_argument("--seed", type=int, default=None, help="default: first seed of the run")
    s = cmd("table", cmd_table, "render the registry as a comparison table")
    s.add_argument("--machine")
    s.add_argument("--registry")
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
