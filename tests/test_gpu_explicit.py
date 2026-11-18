import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from harness.config import ConfigError, MachineConfig
from harness.gpu import GpuUnavailable, acquire, check_explicit, parse_apps
from harness.resolve import Infeasible

sys.path.insert(0, str(Path(__file__).parent))
from test_resolve import fake_fp, plan_ok, setup  # noqa: E402

CARDS = "0, GPU-a, NVIDIA A100-SXM4-80GB, 80000\n1, GPU-b, NVIDIA A100-SXM4-80GB, 70000\n" \
        "2, GPU-c, NVIDIA A100-SXM4-80GB, 9000\n3, GPU-d, NVIDIA A100-SXM4-80GB, 81000\n"
APPS = "GPU-c, 4242, python3, 71000\nGPU-b, 77, /usr/bin/other, 10000\n"


def query(uuid=None):
    rows = [r for r in CARDS.splitlines() if uuid is None or f", {uuid}," in r]
    return "\n".join(rows) + "\n"


def apps():
    return APPS


def machine(**kw):
    return MachineConfig(python="py", model_dir="m", out_root="o", gpu_source="explicit", **kw)


def test_parse_apps():
    assert parse_apps(APPS) == [{"uuid": "GPU-c", "pid": 4242, "name": "python3", "used_mib": 71000},
                                {"uuid": "GPU-b", "pid": 77, "name": "/usr/bin/other", "used_mib": 10000}]
    assert parse_apps("No running processes found\n") == []


def test_check_explicit_ok_lists_processes():
    cards = check_explicit(machine(), [1, 0], need_mib=40000, query=query, apps_query=apps)
    assert [c["index"] for c in cards] == [1, 0] and [c["uuid"] for c in cards] == ["GPU-b", "GPU-a"]
    assert cards[0]["processes"] == [{"pid": 77, "name": "/usr/bin/other", "used_mib": 10000}]
    assert cards[1]["processes"] == [] and cards[0]["free_mib"] == 70000


def test_check_explicit_reasons_per_card():
    with pytest.raises(GpuUnavailable) as e:
        check_explicit(machine(gpu_exclude=[3]), [2, 7, 3, 0], need_mib=40000, query=query, apps_query=apps)
    msg = str(e.value)
    assert "GPU 2: 9000 MiB free < 40000 MiB needed" in msg and "pid 4242 python3 71000 MiB" in msg
    assert "GPU 7: not found" in msg
    assert "GPU 3: excluded" in msg
    assert "GPU 0" not in msg.split("\n", 1)[1]  # card 0 is fine
    assert set(e.value.reasons) == {2, 7, 3}


def test_min_free_mib_when_larger_than_the_estimate():
    with pytest.raises(GpuUnavailable, match="75000 MiB needed"):
        check_explicit(machine(min_free_mib=75000), [1], need_mib=1000, query=query, apps_query=apps)


def test_duplicate_index_refused():
    with pytest.raises(GpuUnavailable, match="twice"):
        check_explicit(machine(), [0, 0], need_mib=0, query=query, apps_query=apps)


def test_acquire_explicit_locks_when_lock_dir_set(tmp_path):
    m = machine(lock_dir=str(tmp_path / "locks"))
    lease = acquire(m, 2, explicit=[3, 0], need_mib=1000, query=query, apps_query=apps)
    assert lease.uuids == ["GPU-d", "GPU-a"] and len(lease._fds) == 2
    assert (tmp_path / "locks" / "GPU-d.lock").exists()
    with pytest.raises(GpuUnavailable, match="GPU 3: locked"):
        acquire(m, 1, explicit=[3], need_mib=1000, query=query, apps_query=apps)
    lease.release()
    acquire(m, 1, explicit=[3], need_mib=1000, query=query, apps_query=apps).release()


def test_acquire_explicit_without_lock_dir_takes_no_locks():
    lease = acquire(machine(), 1, explicit=[1], need_mib=1000, query=query, apps_query=apps)
    assert lease.uuids == ["GPU-b"] and lease.names == ["NVIDIA A100-SXM4-80GB"] and lease._fds == []


def test_acquire_explicit_needs_the_cards():
    with pytest.raises(ConfigError, match="--gpus"):
        acquire(machine(), 1, query=query, apps_query=apps)
    with pytest.raises(ConfigError, match="2 card"):
        acquire(machine(), 1, explicit=[0, 1], query=query, apps_query=apps)


def test_fake_nvidia_smi_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text("#!/bin/sh\ncase \"$*\" in\n"
                      f"  *query-compute-apps*) printf '{APPS}' ;;\n"
                      f"  *) printf '{CARDS}' ;;\nesac\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(GpuUnavailable, match="pid 4242"):
        check_explicit(machine(), [2], need_mib=40000)
    assert check_explicit(machine(), [3], need_mib=40000)[0]["uuid"] == "GPU-d"


def resolve_explicit(tmp_path, gpus, **kw):
    from harness.resolve import resolve
    exp, m = setup(tmp_path, recipe="joey_full_1gpu", base="joey-1")
    exp.gpus = gpus
    m.gpu_source = kw.pop("source", "scan")
    return resolve(exp, m, query=query, apps_query=apps, plan_runner=plan_ok(10, len(gpus)),
                   fingerprint=fake_fp, **kw)


def test_resolve_with_explicit_cards(tmp_path):
    lock = yaml.safe_load(resolve_explicit(tmp_path, [1, 3]).read_text())
    assert lock["n_gpus"] == 2
    assert lock["machine"]["gpu"]["source"] == "explicit" and lock["machine"]["gpu"]["explicit"] == [1, 3]
    cards = lock["feasibility"]["cards"]
    assert [c["uuid"] for c in cards] == ["GPU-b", "GPU-d"] and cards[0]["processes"][0]["pid"] == 77


def test_resolve_explicit_card_without_memory_is_infeasible(tmp_path):
    with pytest.raises(Infeasible) as e:
        resolve_explicit(tmp_path, [2])
    assert "GPU 2: 9000 MiB free" in str(e.value)
    assert any("GPU 0" in a and "GPU 3" in a for a in e.value.alternatives)  # facts, not a switch


def test_explicit_source_without_card_list_is_an_error(tmp_path):
    from harness.resolve import resolve
    exp, m = setup(tmp_path, recipe="joey_full_1gpu", base="joey-1")
    m.gpu_source = "explicit"
    with pytest.raises(ConfigError, match="--gpus"):
        resolve(exp, m, query=query, apps_query=apps, plan_runner=plan_ok(10, 1), fingerprint=fake_fp)


def test_runner_passes_explicit_cards_to_acquire(tmp_path):
    from test_runner import FakeLease, FakePopen, make_lock
    from harness.runner import run
    path, lock, run_dir = make_lock(tmp_path, seeds=(42,))
    lock["machine"]["gpu"].update(source="explicit", explicit=[1, 3])
    lock["estimate_gib"] = 32.0
    path.write_text(yaml.safe_dump(lock))
    seen = {}

    def fake_acquire(m, n, **kw):
        seen.update(source=m.gpu_source, n=n, **kw)
        return FakeLease()
    FakePopen.codes = [0]
    assert run(path, acquire=fake_acquire, popen=FakePopen, finalize=lambda rd: None, killpg=lambda *a: None) == 0
    assert seen == {"source": "explicit", "n": 2, "explicit": [1, 3], "need_mib": 32768}
