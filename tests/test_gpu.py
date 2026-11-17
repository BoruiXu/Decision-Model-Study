import fcntl
import os
import stat
import warnings

import pytest

from harness.config import ConfigError, MachineConfig
from harness.gpu import Gpu, GpuLease, GpuWaitTimeout, acquire, parse_nvidia_smi


def machine(tmp_path, **kw):
    kw.setdefault("lock_dir", str(tmp_path / "locks"))
    return MachineConfig(python="py", model_dir="m", out_root="o", **kw)


class FakeSmi:
    """Injected query: returns csv text; `after_lock` maps uuid -> free MiB seen on per-card re-query."""

    def __init__(self, cards, per_card=None):
        self.cards = dict(cards)  # uuid -> (index, name, free)
        self.per_card = per_card or {}
        self.calls = []

    def __call__(self, uuid=None):
        self.calls.append(uuid)
        rows = []
        for u, (i, name, free) in self.cards.items():
            if uuid is not None and u != uuid:
                continue
            if uuid is not None and u in self.per_card:
                free = self.per_card[u]
            rows.append(f"{i}, {u}, {name}, {free}")
        return "\n".join(rows) + "\n"


CARDS = {
    "GPU-a": (0, "A100", 10000),
    "GPU-b": (1, "A100", 60000),
    "GPU-c": (2, "A100", 40000),
    "GPU-d": (3, "A100", 80000),
}


def no_sleep(_):
    raise AssertionError("should not sleep")


def test_parse_nvidia_smi():
    gpus = parse_nvidia_smi("0, GPU-a, NVIDIA A100-SXM4-80GB, 1234\n\n1, GPU-b, X, 5\n")
    assert gpus == [Gpu(0, "GPU-a", "NVIDIA A100-SXM4-80GB", 1234), Gpu(1, "GPU-b", "X", 5)]


def test_exclude_and_min_free(tmp_path):
    m = machine(tmp_path, gpu_exclude=[3], min_free_mib=30000)
    lease = acquire(m, 2, query=FakeSmi(CARDS), sleep=no_sleep)
    assert lease.uuids == ["GPU-b", "GPU-c"]
    lease.release()


def test_include(tmp_path):
    m = machine(tmp_path, gpu_include=[0, 2])
    lease = acquire(m, 2, query=FakeSmi(CARDS), sleep=no_sleep)
    assert sorted(lease.uuids) == ["GPU-a", "GPU-c"]
    lease.release()


def test_order_free_desc(tmp_path):
    lease = acquire(machine(tmp_path), 3, query=FakeSmi(CARDS), sleep=no_sleep)
    assert lease.uuids == ["GPU-d", "GPU-b", "GPU-c"]
    assert lease.names == ["A100"] * 3
    lease.release()


def test_unknown_order_raises(tmp_path):
    with pytest.raises(ConfigError):
        acquire(machine(tmp_path, order="random"), 1, query=FakeSmi(CARDS), sleep=no_sleep)


def test_locked_card_skipped(tmp_path):
    m = machine(tmp_path)
    os.makedirs(m.lock_dir)
    os.chmod(m.lock_dir, 0o2775)
    fd = os.open(os.path.join(m.lock_dir, "GPU-d.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        lease = acquire(m, 2, query=FakeSmi(CARDS), sleep=no_sleep)
        assert lease.uuids == ["GPU-b", "GPU-c"]
        lease.release()
    finally:
        os.close(fd)


def test_lease_holds_lock_until_release(tmp_path):
    m = machine(tmp_path)
    lease = acquire(m, 1, query=FakeSmi(CARDS), sleep=no_sleep)
    path = os.path.join(m.lock_dir, "GPU-d.lock")
    fd = os.open(path, os.O_RDWR)
    with pytest.raises(BlockingIOError):
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease.release()
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)
    lease.release()  # idempotent


def test_recheck_after_lock(tmp_path):
    m = machine(tmp_path, min_free_mib=30000)
    smi = FakeSmi(CARDS, per_card={"GPU-d": 100})  # freed memory vanished after we locked it
    lease = acquire(m, 2, query=smi, sleep=no_sleep)
    assert lease.uuids == ["GPU-b", "GPU-c"]
    lease.release()
    # the rejected card's lock was dropped again
    fd = os.open(os.path.join(m.lock_dir, "GPU-d.lock"), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)


def test_timeout_raises(tmp_path):
    t = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    m = machine(tmp_path, min_free_mib=999999, wait_timeout_s=100)
    with pytest.raises(GpuWaitTimeout):
        acquire(m, 1, query=FakeSmi(CARDS), sleep=sleep, clock=lambda: t[0])
    assert sleeps and all(s == 30 for s in sleeps)


def test_retry_then_success(tmp_path):
    t = [0.0]
    smi = FakeSmi(CARDS)
    state = {"n": 0}
    orig = smi.__call__

    def query(uuid=None):
        if uuid is None:
            state["n"] += 1
            if state["n"] < 3:
                return ""
        return orig(uuid)

    def sleep(s):
        t[0] += s

    lease = acquire(machine(tmp_path, wait_timeout_s=600), 1, query=query, sleep=sleep, clock=lambda: t[0])
    assert lease.uuids == ["GPU-d"] and t[0] == 60
    lease.release()


def test_no_lock_dir_means_no_locking(tmp_path):
    m = machine(tmp_path, lock_dir=None)
    lease = acquire(m, 2, query=FakeSmi(CARDS), sleep=no_sleep)
    assert lease.uuids == ["GPU-d", "GPU-b"]
    assert not (tmp_path / "locks").exists()
    lease.release()


def test_preassigned_bypasses_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-x,GPU-y")

    def boom(uuid=None):
        raise FileNotFoundError("nvidia-smi")

    m = machine(tmp_path, gpu_source="preassigned")
    lease = acquire(m, 2, query=boom, sleep=no_sleep)
    assert lease.uuids == ["GPU-x", "GPU-y"]
    assert lease.names == ["unknown", "unknown"]
    assert not (tmp_path / "locks").exists()


def test_preassigned_names_by_index(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    m = machine(tmp_path, gpu_source="preassigned")
    lease = acquire(m, 2, query=FakeSmi(CARDS), sleep=no_sleep)
    assert lease.uuids == ["1", "2"] and lease.names == ["A100", "A100"]


def test_preassigned_too_few_or_unset(tmp_path, monkeypatch):
    m = machine(tmp_path, gpu_source="preassigned")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        acquire(m, 1, query=FakeSmi(CARDS))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        acquire(m, 2, query=FakeSmi(CARDS))


def test_warns_when_lock_dir_not_group_writable(tmp_path):
    d = tmp_path / "locks"
    d.mkdir()
    d.chmod(0o755)
    with pytest.warns(UserWarning, match=str(d)) as rec:
        acquire(machine(tmp_path), 1, query=FakeSmi(CARDS), sleep=no_sleep).release()
    assert "other users" in str(rec[0].message)


def test_no_warning_when_group_writable_or_created(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        acquire(machine(tmp_path), 1, query=FakeSmi(CARDS), sleep=no_sleep).release()
        assert (tmp_path / "locks").stat().st_mode & stat.S_IWGRP
        acquire(machine(tmp_path), 1, query=FakeSmi(CARDS), sleep=no_sleep).release()


def _lockable(m, uuid):
    fd = os.open(os.path.join(m.lock_dir, f"{uuid}.lock"), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def test_partial_round_releases_locks_before_retry(tmp_path):
    m = machine(tmp_path, min_free_mib=30000)
    os.makedirs(m.lock_dir)
    os.chmod(m.lock_dir, 0o2775)
    held = os.open(os.path.join(m.lock_dir, "GPU-d.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seen = []

    class Stop(Exception):
        pass

    def sleep(_):
        seen.append((_lockable(m, "GPU-b"), _lockable(m, "GPU-c")))
        raise Stop

    try:
        with pytest.raises(Stop):
            acquire(m, 3, query=FakeSmi(CARDS), sleep=sleep)
    finally:
        os.close(held)
    assert seen == [(True, True)]


def test_query_error_mid_recheck_releases_all(tmp_path):
    m = machine(tmp_path)
    base = FakeSmi(CARDS)

    def query(uuid=None):
        if uuid == "GPU-b":
            raise RuntimeError("smi died")
        return base(uuid)

    with pytest.raises(RuntimeError, match="smi died"):
        acquire(m, 3, query=query, sleep=no_sleep)
    assert _lockable(m, "GPU-d") and _lockable(m, "GPU-b")


def test_unopenable_lock_file_is_busy(tmp_path):
    m = machine(tmp_path)
    os.makedirs(m.lock_dir)
    os.chmod(m.lock_dir, 0o2775)
    bad = os.path.join(m.lock_dir, "GPU-d.lock")
    open(bad, "w").close()
    os.chmod(bad, 0)
    try:
        lease = acquire(m, 1, query=FakeSmi(CARDS), sleep=no_sleep)
        assert lease.uuids == ["GPU-b"]
        lease.release()
    finally:
        os.chmod(bad, 0o600)


def test_new_lock_file_is_group_writable(tmp_path):
    old = os.umask(0o022)
    try:
        m = machine(tmp_path)
        acquire(m, 1, query=FakeSmi(CARDS), sleep=no_sleep).release()
    finally:
        os.umask(old)
    assert os.stat(os.path.join(m.lock_dir, "GPU-d.lock")).st_mode & stat.S_IWGRP


def test_lock_dir_under_forbidden_root_refused(tmp_path):
    m = machine(tmp_path, forbidden_write_roots=[str(tmp_path)])
    with pytest.raises(ConfigError, match="forbidden"):
        acquire(m, 1, query=FakeSmi(CARDS), sleep=no_sleep)
    assert not (tmp_path / "locks").exists()
