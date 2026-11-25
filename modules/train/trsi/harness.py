"""The only module that talks to onejev-harness (ojh). Training goes through `ojh run`; what a finished run gives (its
state, GPU hours, selected checkpoint) is read from the files the harness writes in its run directory (DONE,
result.json, timing.json, gpus.json), and from whether its tmux session still exists. The metrics come from the
candidate packed from the run (trsi.pack), not from ojh's own predictions. The card a candidate is scored on is
chosen here too (pick_scoring_card). The harness itself is never modified."""
from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .candidate import Candidate, file_sha256, trial_id
from .config import RunConfig


class HarnessError(RuntimeError):
    pass


@dataclass
class TrialResult:
    run_dir: str
    seed: int
    status: str                      # done | failed
    metrics: dict = field(default_factory=dict)   # the candidate's metrics_val and the ECE of its predictions_val
    gpu_hours: float = 0.0           # as the harness measured it: training wall time x cards
    gpu_names: list = field(default_factory=list)
    selected_step: int | None = None
    last_step: int | None = None     # last evaluated step
    checkpoint: str | None = None    # relative to run_dir
    error: str | None = None
    compute_hours: float | None = None   # the candidate's cost.compute_hours: GPU hours + scoring time
    candidate_dir: str | None = None     # None while a done trial is not packed (not comparable yet)
    pack_error: str | None = None        # why the last packing failed; the next collect tries again
    scoring_card: str | None = None      # uuid of the card the last packing scored on

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialResult":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


# ---- the environment ----------------------------------------------------------------------------------------

def load_exp(run: RunConfig) -> dict:
    exp = yaml.safe_load(run.exp.read_text()) or {}
    if exp.get("overrides"):
        raise HarnessError(f"{run.exp}: the environment exp.yaml must not have overrides; candidates bring their own")
    if not (exp.get("fields") or {}).get("label"):
        raise HarnessError(f"{run.exp}: fields.label is missing")
    return exp


def exp_sha256(run: RunConfig) -> str:
    return file_sha256(run.exp)


def positive_label(run: RunConfig) -> str:
    """The positive class: the last part of fields.label, e.g. 'yes' for target.yes."""
    return str(load_exp(run)["fields"]["label"]).rsplit(".", 1)[-1]


def _abs(base_dir: Path, v):
    s = str(v)
    if s.startswith(("/", "$", "~")):
        return s
    return str((base_dir / s).resolve())


def _absolutize(exp: dict, base_dir: Path) -> None:
    """The workspace exp.yaml lives elsewhere; make the paths the harness resolves relative to the exp file absolute."""
    data = exp.get("data") or {}
    for k in ("train", "split_file"):
        if data.get(k):
            data[k] = _abs(base_dir, data[k])
    if isinstance(data.get("adapter"), str) and data["adapter"].endswith(".py"):
        data["adapter"] = _abs(base_dir, data["adapter"])
    for s in (data.get("eval") or {}).values():
        for k in ("train_labels", "reference"):
            if isinstance(s, dict) and s.get(k):
                s[k] = _abs(base_dir, s[k])
    ev = exp.get("evaluator") or {}
    if ev.get("path"):
        ev["path"] = _abs(base_dir, ev["path"])


def write_workspace(run: RunConfig, c: Candidate, seed: int) -> Path:
    """<home>/work/<trial id>/exp.yaml: the environment's exp.yaml with a unique name, one seed and the candidate's
    overrides. The environment's own file is not changed."""
    tid = trial_id(c, seed, exp_sha256(run))
    exp = copy.deepcopy(load_exp(run))
    _absolutize(exp, run.exp.parent)
    exp["name"] = f"{exp.get('name', 'exp')}-{tid}"
    exp["seeds"] = [seed]
    exp["overrides"] = {k: {"value": o.value, "reason": o.reason} for k, o in sorted(c.overrides.items())}
    if recipe_mode(c.recipe) == "lora":
        exp["merge"] = True   # the candidate carries the merged model, so model.predict needs no base weights
    for k in ("base", "gpus", "recipe"):  # given on the ojh run command line
        exp.pop(k, None)
    d = run.home / "work" / tid
    d.mkdir(parents=True, exist_ok=True)
    out = d / "exp.yaml"
    tmp = out.with_suffix(".yaml.tmp")
    tmp.write_text("# written by trsi; the environment is " + str(run.exp) + "\n" + yaml.safe_dump(exp, sort_keys=False))
    os.replace(tmp, out)
    return out


# ---- running ojh ----------------------------------------------------------------------------------------------

def ojh_command(run: RunConfig, args: list[str]) -> list[str]:
    if run.ojh.env_file:
        # non-interactive shells do not read the user's profile: source the env file first, then expand variables
        # left in the ojh path (e.g. $OJH_REPO, set by that env file)
        script = 'source "$0" && b=$1 && shift && eval "b=\\"$b\\"" && exec "$b" "$@"'
        return ["bash", "-c", script, str(run.ojh.env_file), run.ojh.bin, *args]
    return [run.ojh.bin, *args]


def _json_out(text: str) -> dict:
    i = text.find("{")
    if i < 0:
        raise HarnessError(f"ojh printed no JSON:\n{text[-2000:]}")
    return json.loads(text[i:])


def _ojh(run: RunConfig, args: list[str]) -> dict:
    p = subprocess.run(ojh_command(run, args), capture_output=True, text=True)
    if p.returncode != 0:
        raise HarnessError(f"ojh {args[0]} exited with code {p.returncode}. Its message (when the user must choose, put "
                           f"it in the hand-back's questions; the alternatives are the user's choice):\n"
                           f"{(p.stderr or p.stdout)[-4000:]}")
    return _json_out(p.stdout)


def recipe_gpus(recipe: Path) -> int:
    return int((yaml.safe_load(Path(recipe).read_text()) or {}).get("gpus", 1))


def recipe_mode(recipe: Path) -> str | None:
    return (yaml.safe_load(Path(recipe).read_text()) or {}).get("mode")


def _args(run: RunConfig, cmd: str, exp: Path, c: Candidate, gpus: list[int]) -> list[str]:
    args = [cmd, str(exp), "--base", c.base, "--recipe", str(c.recipe), "--gpus", ",".join(str(g) for g in gpus)]
    if run.ojh.machine:
        args += ["--machine", str(run.ojh.machine)]
    return args


def launch(run: RunConfig, c: Candidate, seed: int, gpus: list[int]) -> dict:
    """ojh run for one seed; returns immediately with ojh's JSON (run_dir, session, log, done, result, ...)."""
    exp = write_workspace(run, c, seed)
    args = _args(run, "run", exp, c, gpus)
    if run.gpus.allow_shared:
        args.append("--allow-shared")
    out = _ojh(run, args)
    if "run_dir" not in out:
        raise HarnessError(f"ojh run returned no run_dir: {out}")
    return out


def check(run: RunConfig, c: Candidate, seed: int, gpus: list[int]) -> dict:
    """ojh check: prepare, config, adapter, plan-only and environment checks, without the cards."""
    return _ojh(run, _args(run, "check", write_workspace(run, c, seed), c, gpus))


def cleanup(run: RunConfig, run_dir: Path) -> str:
    """`ojh cleanup <run_dir> --confirm`: deletes what ojh's finalize listed in the run's deletion_candidates.json
    (unselected checkpoints, merged models); returns what it printed."""
    p = subprocess.run(ojh_command(run, ["cleanup", str(run_dir), "--confirm"]), capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        raise HarnessError(f"ojh cleanup {run_dir} exited with code {p.returncode}:\n{(p.stderr or p.stdout)[-2000:]}")
    return (p.stdout + p.stderr).strip()   # ojh prints the entries it skipped to stderr


def deletion_left(run_dir: Path) -> list[str]:
    """The paths the run's deletion_candidates.json (written by ojh's finalize) lists that are still there."""
    try:
        cands = json.loads((Path(run_dir) / "deletion_candidates.json").read_text()).get("candidates") or []
    except (OSError, ValueError, AttributeError):
        return []
    return sorted(str(c["path"]) for c in cands
                  if isinstance(c, dict) and c.get("path") and os.path.lexists(str(c["path"])))


def _checkpoint_name(name: str) -> bool:
    """ojh's names for checkpoint directories (harness/run_files.py _ckpt_kind)."""
    return name.endswith(".partial") or name.startswith(("ckpt-step", "adapter-step", "merged-step")) \
        or name in ("final", "adapter")


def delete_checkpoints(run_dir: Path) -> list[str]:
    """Delete every checkpoint directory of a failed ojh run (for a failed run ojh's cleanup offers none): those ojh
    names ckpt-step*, adapter-step*, merged-step*, final, adapter or *.partial directly inside
    <run_dir>/runs/<experiment>-s<seed>/; a symbolic link is removed, never followed. Returns the deleted paths."""
    runs = Path(run_dir) / "runs"
    gone: list[str] = []
    if runs.is_symlink() or not runs.is_dir():
        return gone
    for seed_dir in sorted(runs.iterdir()):
        if seed_dir.is_symlink() or not seed_dir.is_dir():
            continue
        for d in sorted(seed_dir.iterdir()):
            if not _checkpoint_name(d.name):
                continue
            if d.is_symlink():
                d.unlink()
            elif d.is_dir():
                shutil.rmtree(d)
            else:
                continue
            gone.append(str(d))
    return gone


# ---- reading results ----------------------------------------------------------------------------------------

def session_alive(session: str) -> bool:
    """Whether ojh's tmux session (this exact name) still exists; True when tmux cannot be run or does not answer
    within 10 s (no way to tell)."""
    try:
        return subprocess.run(["tmux", "has-session", "-t", f"={session}"], capture_output=True,
                              stdin=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return True


LOST = "ended without DONE (its ojh training's tmux session is gone)"
LOST_GRACE_S = 120   # after a tmux kill-session ojh's runner may take ~30 s to write DONE


def _newest_mtime(d: Path) -> float:
    """The newest modification time of `d` and everything under it (links not followed); 0 when none is readable."""
    newest = 0.0
    for root, _, files in os.walk(d):
        for path in [root] + [os.path.join(root, f) for f in files]:
            with contextlib.suppress(OSError):
                newest = max(newest, os.lstat(path).st_mtime)
    return newest


def state(run_dir: Path, session: str | None = None) -> str:
    """running | done | failed | lost, from the harness's DONE marker; lost (a failure): no DONE, the training's tmux
    session `session` is gone and nothing under run_dir has changed for LOST_GRACE_S seconds (the runner writes DONE
    a while after its session is killed)."""
    done = Path(run_dir) / "DONE"
    if not done.is_file():
        if session and not session_alive(session) and not done.is_file() \
                and time.time() - _newest_mtime(Path(run_dir)) > LOST_GRACE_S:
            return "lost"
        return "running"
    try:
        return "done" if json.loads(done.read_text()).get("exit_code") == 0 else "failed"
    except json.JSONDecodeError:
        return "running"  # being written


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def _cost(run_dir: Path, result: dict) -> tuple[float, list[str]]:
    cards = (result.get("gpus") or {}).get("cards") or _read_json(run_dir / "gpus.json") or []
    names = [c.get("name", "") for c in cards if isinstance(c, dict)]
    gh = (result.get("wall_time") or {}).get("gpu_hours")
    if gh is None:  # failed runs have no result.json: training seconds x cards
        secs = sum(float(s.get("seconds") or 0) for s in (_read_json(run_dir / "timing.json").get("seeds") or {}).values())
        gh = secs * max(len(names), 1) / 3600
    return float(gh), names


def read_result(run_dir: Path, seed: int) -> TrialResult:
    """State, GPU hours, the selected step and checkpoint of a finished run (the metrics come from packing it)."""
    run_dir = Path(run_dir)
    st = state(run_dir)
    result = _read_json(run_dir / "result.json")
    gh, names = _cost(run_dir, result)
    if st != "done":
        if not (run_dir / "DONE").is_file():
            error = LOST
        else:
            done = _read_json(run_dir / "DONE")
            error = f"stage {done.get('stage')}: {done.get('error')}"
        return TrialResult(str(run_dir), seed, "failed", gpu_hours=gh, gpu_names=names, error=error)
    s = (result.get("seeds") or {}).get(str(seed)) or {}
    curve = s.get("curve") or []
    return TrialResult(str(run_dir), seed, "done", {}, gh, names, s.get("selected_step"),
                       max((e["step"] for e in curve), default=None), s.get("checkpoint"))


# ---- cards and sessions -------------------------------------------------------------------------------------

def kill_session(session: str) -> None:
    """End an ojh training: ojh runs it in a detached tmux session."""
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)


IDLE_MIB = 1024   # a card with less memory in use than this is idle


def _mib(s: str) -> float | None:
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _smi(query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              check=True, stdin=subprocess.DEVNULL, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise HarnessError(f"nvidia-smi {query} failed ({e})") from None


def gpu_table() -> list[tuple[int, str, float, float]]:
    """(index, uuid, MiB in use, MiB free) of every card, from nvidia-smi. A card whose memory it does not give as
    numbers ([N/A], e.g. MIG) counts as busy: inf in use, 0 free."""
    rows = []
    for line in _smi("--query-gpu=index,uuid,memory.used,memory.free").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit():
            used, free = _mib(parts[2]), _mib(parts[3])
            rows.append((int(parts[0]), parts[1]) + ((used, free) if used is not None and free is not None
                                                     else (math.inf, 0.0)))
    return rows


def compute_apps() -> dict[str, list[str]]:
    """Card uuid -> the pids of the compute processes nvidia-smi lists on it (anyone's)."""
    apps: dict[str, list[str]] = {}
    for line in _smi("--query-compute-apps=gpu_uuid,pid").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            apps.setdefault(parts[0], []).append(parts[1])
    return apps


def _idle(allowed: tuple[int, ...] | None) -> list[tuple[float, int, str]]:
    return sorted((used, i, u) for i, u, used, _ in gpu_table()
                  if (allowed is None or i in allowed) and used < IDLE_MIB)


def free_gpus(allowed: tuple[int, ...] | None, n: int) -> list[int]:
    """n idle cards (least memory in use first) among `allowed` (None: any); $RSI_TRAIN_GPUS ("0,1") overrides.
    Used by the train command, which has no agent to choose."""
    env = os.environ.get("RSI_TRAIN_GPUS", "").strip()
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    idle = _idle(allowed)
    if len(idle) < n:
        raise HarnessError(f"{n} idle card(s) needed, found {[i for _, i, _ in idle]} among "
                           f"{'all cards' if allowed is None else list(allowed)}; set RSI_TRAIN_GPUS")
    return [i for _, i, _ in idle[:n]]


def scoring_lock_settings(lock_gpu: dict, machine: Path | None) -> tuple[int, Path | None]:
    """(min_free_mib, lock_dir) of ojh's card checks: each from a run's run.lock.yaml machine.gpu (`lock_gpu`) when it
    has the key, else from the machine config file `machine` (its lock_dir relative to that file), else 0 and None."""
    conf: dict = {}
    if machine is not None and not ("min_free_mib" in lock_gpu and "lock_dir" in lock_gpu):
        try:
            conf = yaml.safe_load(Path(machine).read_text()) or {}
        except (OSError, yaml.YAMLError):
            conf = {}
        if not isinstance(conf, dict):
            conf = {}
    mf = lock_gpu["min_free_mib"] if "min_free_mib" in lock_gpu else conf.get("min_free_mib")
    ld = lock_gpu["lock_dir"] if "lock_dir" in lock_gpu else conf.get("lock_dir")
    if ld and "lock_dir" not in lock_gpu:
        ld = Path(os.path.expandvars(os.path.expanduser(str(ld))))
        ld = ld if ld.is_absolute() else Path(machine).parent / ld
    return int(mf or 0), (Path(ld) if ld else None)


def _try_lock(path: Path):
    """A descriptor holding an flock on `path` (created when missing; read-only, as ojh opens its card locks), or
    why there is none."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o664)
    except OSError as e:
        return f"its lock {path} cannot be opened ({e.strerror or e})"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return f"locked by another job ({path})"
    return fd


def release(lock) -> None:
    """Let go of the lock pick_scoring_card returned (None: nothing held)."""
    if lock is not None:
        with contextlib.suppress(OSError):
            os.close(lock)


NO_CARD = "can score this candidate now"   # in pick_scoring_card's error: the cards may free up


def pick_scoring_card(allowed: tuple[int, ...] | None, need_mib: float, min_free_mib: float = 0,
                      lock_dir: Path | None = None, exclude: tuple[str, ...] = (), prefer: int | None = None,
                      why: dict | None = None) -> tuple[int, str, object | None]:
    """(index, uuid, lock) of the card to score a candidate on now. A card qualifies when it is in `allowed` (None:
    any), its uuid is not in `exclude`, nvidia-smi lists no compute process on it (anyone's), it has at least
    max(min_free_mib, need_mib) MiB free, and, with `lock_dir`, <lock_dir>/<uuid>.lock (ojh's card lock) can be taken
    without waiting: the lock is returned, to hold while scoring (release()). `prefer` when it qualifies, else the
    one with the most free memory. `why` (a dict) gets {index: reason} of every card of `allowed` that did not
    qualify. No card: HarnessError with each card's reason. The uuid goes to CUDA_VISIBLE_DEVICES (CUDA may number
    the cards differently from nvidia-smi)."""
    floor = max(float(min_free_mib or 0), float(need_mib or 0))
    cards = [c for c in gpu_table() if allowed is None or c[0] in allowed]
    try:
        apps, apps_error = compute_apps(), None
    except HarnessError as e:
        apps, apps_error = {}, str(e)
    reasons = {} if why is None else why
    ok = []
    for i, u, used, free in cards:
        if u in exclude:
            reasons[i] = "left out: scoring on it ran out of memory"
        elif apps_error:
            reasons[i] = f"its compute processes cannot be listed ({apps_error})"
        elif apps.get(u):
            reasons[i] = f"another process runs on it (pid {', '.join(apps[u])})"
        elif math.isinf(used):
            reasons[i] = "nvidia-smi gives no memory figures for it ([N/A], e.g. MIG)"
        elif free < floor:
            reasons[i] = f"{free:.0f} MiB free < {floor:.0f} MiB needed"
        else:
            ok.append((i, u, free))
    for i, u, _ in sorted(ok, key=lambda c: (c[0] != prefer, -c[2], c[0])):
        lock = None if lock_dir is None else _try_lock(Path(lock_dir) / f"{u}.lock")
        if isinstance(lock, str):
            reasons[i] = lock
            continue
        return i, u, lock
    listed = "all cards" if allowed is None else str(list(allowed))
    raise HarnessError(f"no card among {listed} {NO_CARD} (it needs {floor:.0f} MiB free and no "
                       "other process on the card" + (", and ojh's card lock" if lock_dir else "") + "):\n"
                       + ("\n".join(f"  card {i}: {r}" for i, r in sorted(reasons.items()))
                          or "  nvidia-smi lists none"))
