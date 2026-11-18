"""GPU selection and per-card locks. Imports without torch.

Ported from the selection loop of the reference launcher: query nvidia-smi, filter, sort by free memory
(descending), take a non-blocking flock per card, re-check the card's free memory once locked, keep the
fds open for the lease lifetime, and retry every 30 s until the deadline.
"""
from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import time
import warnings
from dataclasses import dataclass, field

from .config import ConfigError, MachineConfig, check_writable

RETRY_S = 30
_QUERY = "--query-gpu=index,uuid,name,memory.free"
_APPS = "--query-compute-apps=gpu_uuid,pid,process_name,used_memory"


class GpuWaitTimeout(RuntimeError):
    """Not enough free GPUs became available before ``wait_timeout_s``."""


class GpuUnavailable(RuntimeError):
    """One or more explicitly requested cards cannot be used; ``reasons`` maps card index -> reason."""

    def __init__(self, reasons: dict, cards: list | None = None):
        self.reasons = dict(reasons)
        self.cards = list(cards or [])  # every card nvidia-smi listed, with its processes
        super().__init__("requested GPU(s) not usable:\n" + "\n".join(f"GPU {i}: {r}" for i, r in reasons.items()))


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    free_mib: int


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Parse ``--query-gpu=index,uuid,name,memory.free --format=csv,noheader,nounits`` output."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        # the name never contains a comma in practice, but tolerate it
        gpus.append(Gpu(int(parts[0]), parts[1], ", ".join(parts[2:-1]), int(parts[-1])))
    return gpus


def _nvidia_smi(uuid: str | None = None) -> str:
    cmd = ["nvidia-smi", _QUERY, "--format=csv,noheader,nounits"]
    if uuid is not None:
        cmd[1:1] = ["-i", uuid]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _nvidia_smi_apps() -> str:
    return subprocess.run(["nvidia-smi", _APPS, "--format=csv,noheader,nounits"], check=True,
                          capture_output=True, text=True).stdout


def parse_apps(text: str) -> list[dict]:
    """Parse ``--query-compute-apps=gpu_uuid,pid,process_name,used_memory`` csv (no header, no units)."""
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[1].isdigit():
            continue  # blank lines and "No running processes found"
        used = parts[-1]
        out.append({"uuid": parts[0], "pid": int(parts[1]), "name": ", ".join(parts[2:-1]),
                    "used_mib": int(used) if used.isdigit() else None})
    return out


def _process_user(pid: int) -> str:
    """Owner of ``pid`` (``ps``), or "?" when it is not visible here (another container, already gone)."""
    try:
        out = subprocess.run(["ps", "-o", "user=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "?"
    return out or "?"


def _procs_text(procs: list[dict] | None) -> str:
    if procs is None:
        return "unknown (the process query failed)"
    return "; ".join(f"pid {p['pid']} (user {p.get('user', '?')}, {p['name']}, {p['used_mib']} MiB)"
                     for p in procs) or "none"


def check_explicit(machine: MachineConfig, indices: list[int], need_mib: int, *, query=_nvidia_smi,
                   apps_query=_nvidia_smi_apps, user_of=_process_user, allow_shared: bool = False) -> list[dict]:
    """Validate explicitly requested cards: each exists, is allowed by gpu_include/gpu_exclude, has at
    least max(need_mib, machine.min_free_mib) MiB free and, unless ``allow_shared``, runs no compute
    process (any process found here was not started by this run: the run starts only after this check).
    Returns one dict per card (index, uuid, name, free_mib, need_mib, processes with pid, user, name and
    memory) or raises GpuUnavailable with a reason per failing card."""
    need = max(int(need_mib), machine.min_free_mib)
    known = {g.index: g for g in parse_nvidia_smi(query())}
    apps_error = None
    try:
        apps = parse_apps(apps_query())
    except Exception as e:  # unknown is not "no process": refused below unless allow_shared
        apps, apps_error = [], f"{type(e).__name__}: {e}"
    users: dict[int, str] = {}

    def procs(uuid):
        if apps_error is not None:
            return None
        out = []
        for a in apps:
            if a["uuid"] == uuid:
                if a["pid"] not in users:
                    users[a["pid"]] = user_of(a["pid"])
                out.append({"pid": a["pid"], "user": users[a["pid"]], "name": a["name"], "used_mib": a["used_mib"]})
        return out

    reasons: dict[int, str] = {}
    cards = []
    seen = set()
    for i in indices:
        g = known.get(i)
        if i in seen:
            reasons[i] = "listed twice"
            continue
        seen.add(i)
        if g is None:
            reasons[i] = f"not found (nvidia-smi lists {sorted(known)})"
            continue
        p = procs(g.uuid)
        if i in machine.gpu_exclude or (machine.gpu_include is not None and i not in machine.gpu_include):
            reasons[i] = "excluded by the machine config (gpu_include/gpu_exclude)"
        elif g.free_mib < need:
            reasons[i] = f"{g.free_mib} MiB free < {need} MiB needed; processes: {_procs_text(p)}"
        elif p is None and not allow_shared:
            reasons[i] = (f"cannot list its compute processes (nvidia-smi --query-compute-apps failed: {apps_error}); "
                          "only if the user accepts not knowing who else uses the card, run with --allow-shared")
        elif p and not allow_shared:
            reasons[i] = (f"shared with {len(p)} compute process(es) not started by this run: {_procs_text(p)}; "
                          "use it only after asking the user, with --allow-shared")
        cards.append({"index": g.index, "uuid": g.uuid, "name": g.name, "free_mib": g.free_mib,
                      "need_mib": need, "processes": p})
    if reasons:
        everything = [{"index": g.index, "uuid": g.uuid, "name": g.name, "free_mib": g.free_mib,
                       "processes": procs(g.uuid)} for g in known.values()]
        raise GpuUnavailable(reasons, everything)
    return cards


def _explicit(machine: MachineConfig, n: int, explicit, need_mib: int, query, apps_query,
              allow_shared: bool = False) -> "GpuLease":
    apps_query = apps_query or _nvidia_smi_apps
    if not explicit:
        raise ConfigError("gpu_source is 'explicit' but no cards were given: pass --gpus i,j,...")
    if len(explicit) != n:
        raise ConfigError(f"{len(explicit)} card(s) given for a run on {n} card(s)")
    cards = check_explicit(machine, list(explicit), need_mib, query=query, apps_query=apps_query,
                           allow_shared=allow_shared)
    fds: list[int] = []
    try:
        if machine.lock_dir:
            check_writable(machine, machine.lock_dir)
            _prepare_lock_dir(machine.lock_dir)
            reasons = {}
            for c in cards:
                path = os.path.join(machine.lock_dir, f"{c['uuid']}.lock")
                fd = _try_lock(path)
                if fd is None:
                    reasons[c["index"]] = f"locked by another run ({path})"
                    continue
                fds.append(fd)
                fresh = [x for x in parse_nvidia_smi(query(c["uuid"])) if x.uuid == c["uuid"]]
                if not fresh or fresh[0].free_mib < c["need_mib"]:
                    reasons[c["index"]] = (f"{fresh[0].free_mib if fresh else '?'} MiB free < {c['need_mib']} "
                                           "MiB needed once locked")
            if reasons:
                raise GpuUnavailable(reasons)
        lease = GpuLease([c["uuid"] for c in cards], [c["name"] for c in cards], fds)
        fds = []
        return lease
    finally:
        for fd in fds:
            os.close(fd)


@dataclass
class GpuLease:
    uuids: list[str]
    names: list[str]
    _fds: list[int] = field(default_factory=list, repr=False)

    def release(self) -> None:
        """Close the lock fds (dropping the flocks). Safe to call twice."""
        fds, self._fds = self._fds, []
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def _prepare_lock_dir(lock_dir: str) -> None:
    if os.path.isdir(lock_dir):
        if not os.stat(lock_dir).st_mode & stat.S_IWGRP:
            warnings.warn(
                f"lock_dir {lock_dir} is not group-writable; other users on this machine "
                "cannot be excluded from your GPUs unless they can lock there too",
                stacklevel=3,
            )
        return
    os.makedirs(lock_dir, mode=0o2775, exist_ok=True)
    os.chmod(lock_dir, 0o2775)  # makedirs mode is masked by umask


def _try_lock(path: str) -> int | None:
    """Open (creating if needed) and flock a lock file; None if the card is busy or the file is unusable.

    Opened read-only: flock works on a read-only fd, so group members need no write access to the file.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o664)
        try:
            os.chmod(path, 0o664)  # umask-proof so other group members can open it
        except OSError:
            pass
    except FileExistsError:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return None
    except PermissionError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _preassigned(n: int, query) -> GpuLease:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    if len(ids) < n:
        raise RuntimeError(
            f"gpu_source is 'preassigned' but CUDA_VISIBLE_DEVICES={raw!r} lists {len(ids)} device(s); need {n}")
    ids = ids[:n]
    known: list[Gpu] = []
    try:
        known = parse_nvidia_smi(query())
    except Exception:  # no nvidia-smi or unparsable: names stay unknown
        pass
    names = []
    for i in ids:
        match = next((g for g in known if g.uuid == i or str(g.index) == i), None)
        names.append(match.name if match else "unknown")
    return GpuLease(ids, names)


def acquire(machine: MachineConfig, n: int, *, query=_nvidia_smi, sleep=time.sleep,
            clock=time.monotonic, explicit: list[int] | None = None, need_mib: int = 0,
            apps_query=None, allow_shared: bool = False) -> GpuLease:
    """Pick and lock ``n`` GPUs according to ``machine``; hold the returned lease for the whole run.

    ``query(uuid=None)`` returns nvidia-smi csv text for all cards, or for one card when given its uuid;
    ``apps_query()`` the compute processes (default: nvidia-smi --query-compute-apps).
    ``gpu_source: scan`` polls every RETRY_S seconds up to ``wait_timeout_s`` for cards allowed by
    gpu_include/gpu_exclude with at least ``min_free_mib`` free and, unless ``allow_shared``, no other compute
    process (an unreadable process list counts as unknown, so the card waits too); the timeout names each
    card's problem at the last check. ``explicit``: the given cards are validated once with
    ``check_explicit`` against ``need_mib`` and refused at once, never waited for.
    """
    apps_query = apps_query or _nvidia_smi_apps
    if machine.order != "free_desc":
        raise ConfigError(f"unsupported gpu order: {machine.order!r} (only 'free_desc')")
    if machine.gpu_source == "explicit":
        return _explicit(machine, n, explicit, need_mib, query, apps_query, allow_shared)
    if machine.gpu_source == "preassigned":
        return _preassigned(n, query)
    if machine.gpu_source != "scan":
        raise ConfigError(f"unsupported gpu_source: {machine.gpu_source!r}")
    if machine.lock_dir:
        check_writable(machine, machine.lock_dir)
        _prepare_lock_dir(machine.lock_dir)

    deadline = clock() + machine.wait_timeout_s
    while True:
        reasons: dict[int, str] = {}  # why each candidate card is not taken now (for the timeout message)
        gpus = []
        busy: dict[str, list[dict]] = {}
        apps_error = None
        if not allow_shared:
            try:
                for a in parse_apps(apps_query()):
                    busy.setdefault(a["uuid"], []).append(a)
            except Exception as e:
                apps_error = f"{type(e).__name__}: {e}"
        for g in parse_nvidia_smi(query()):
            if (machine.gpu_include is not None and g.index not in machine.gpu_include) \
                    or g.index in machine.gpu_exclude:
                continue
            if g.free_mib < machine.min_free_mib:
                reasons[g.index] = f"{g.free_mib} MiB free < min_free_mib {machine.min_free_mib}"
            elif apps_error is not None:
                reasons[g.index] = f"process list unavailable ({apps_error})"
            elif busy.get(g.uuid):
                procs = [{"pid": a["pid"], "user": _process_user(a["pid"]), "name": a["name"],
                          "used_mib": a["used_mib"]} for a in busy[g.uuid]]
                reasons[g.index] = f"{len(procs)} other compute process(es): {_procs_text(procs)}"
            else:
                gpus.append(g)
        gpus.sort(key=lambda g: g.free_mib, reverse=True)
        got: list[Gpu] = []
        fds: list[int] = []
        try:
            for g in gpus:
                fd = None
                if machine.lock_dir:
                    fd = _try_lock(os.path.join(machine.lock_dir, f"{g.uuid}.lock"))
                    if fd is None:
                        reasons[g.index] = "locked by another run"
                        continue
                    try:
                        fresh = [x for x in parse_nvidia_smi(query(g.uuid)) if x.uuid == g.uuid]
                    except BaseException:
                        os.close(fd)
                        raise
                    if not fresh or fresh[0].free_mib < machine.min_free_mib:
                        reasons[g.index] = (f"{fresh[0].free_mib if fresh else '?'} MiB free < min_free_mib "
                                            f"{machine.min_free_mib} once locked")
                        os.close(fd)
                        continue
                    fds.append(fd)
                got.append(g)
                if len(got) == n:
                    lease = GpuLease([x.uuid for x in got], [x.name for x in got], fds)
                    fds = []  # ownership moved to the lease
                    return lease
        finally:
            for fd in fds:
                os.close(fd)
        if clock() > deadline:
            rule = ("other compute processes allowed (--allow-shared)" if allow_shared
                    else "no other compute process (use --allow-shared only after asking the user)")
            last = "; ".join(f"GPU {i}: {r}" for i, r in sorted(reasons.items())) or "no candidate card"
            raise GpuWaitTimeout(f"no {n} GPU(s) with >= {machine.min_free_mib} MiB free and {rule} within "
                                 f"{machine.wait_timeout_s} s; last check: {last}")
        sleep(RETRY_S)
