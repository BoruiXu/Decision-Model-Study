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

from .config import ConfigError, MachineConfig

RETRY_S = 30
_QUERY = "--query-gpu=index,uuid,name,memory.free"


class GpuWaitTimeout(RuntimeError):
    """Not enough free GPUs became available before ``wait_timeout_s``."""


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
            clock=time.monotonic) -> GpuLease:
    """Pick and lock ``n`` GPUs according to ``machine``; hold the returned lease for the whole run.

    ``query(uuid=None)`` returns nvidia-smi csv text for all cards, or for one card when given its uuid.
    """
    if machine.order != "free_desc":
        raise ConfigError(f"unsupported gpu order: {machine.order!r} (only 'free_desc')")
    if machine.gpu_source == "preassigned":
        return _preassigned(n, query)
    if machine.gpu_source != "scan":
        raise ConfigError(f"unsupported gpu_source: {machine.gpu_source!r}")
    if machine.lock_dir:
        _prepare_lock_dir(machine.lock_dir)

    deadline = clock() + machine.wait_timeout_s
    while True:
        gpus = [g for g in parse_nvidia_smi(query())
                if (machine.gpu_include is None or g.index in machine.gpu_include)
                and g.index not in machine.gpu_exclude
                and g.free_mib >= machine.min_free_mib]
        gpus.sort(key=lambda g: g.free_mib, reverse=True)
        got: list[Gpu] = []
        fds: list[int] = []
        try:
            for g in gpus:
                fd = None
                if machine.lock_dir:
                    fd = _try_lock(os.path.join(machine.lock_dir, f"{g.uuid}.lock"))
                    if fd is None:
                        continue
                    try:
                        fresh = [x for x in parse_nvidia_smi(query(g.uuid)) if x.uuid == g.uuid]
                    except BaseException:
                        os.close(fd)
                        raise
                    if not fresh or fresh[0].free_mib < machine.min_free_mib:
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
            raise GpuWaitTimeout(
                f"no {n} GPU(s) with >= {machine.min_free_mib} MiB free within {machine.wait_timeout_s} s")
        sleep(RETRY_S)
