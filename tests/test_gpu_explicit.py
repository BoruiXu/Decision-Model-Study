import os
import re
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


def user_of(pid):
    return {77: "alice", 4242: "bob"}.get(pid, "?")


def test_check_explicit_ok_lists_processes():
    cards = check_explicit(machine(), [3, 0], need_mib=40000, query=query, apps_query=apps, user_of=user_of)
    assert [c["index"] for c in cards] == [3, 0] and [c["uuid"] for c in cards] == ["GPU-d", "GPU-a"]
    assert cards[0]["processes"] == [] and cards[0]["free_mib"] == 81000


def test_shared_card_refused_by_default_with_its_processes():
    with pytest.raises(GpuUnavailable) as e:
        check_explicit(machine(), [1, 0], need_mib=40000, query=query, apps_query=apps, user_of=user_of)
    msg = str(e.value)
    assert "GPU 1: shared with 1 compute process(es) not started by this run" in msg
    assert "pid 77 (user alice, /usr/bin/other, 10000 MiB)" in msg and "--allow-shared" in msg
    assert set(e.value.reasons) == {1}
    cards = check_explicit(machine(), [1, 0], need_mib=40000, query=query, apps_query=apps, user_of=user_of,
                           allow_shared=True)
    assert cards[0]["processes"] == [{"pid": 77, "user": "alice", "name": "/usr/bin/other", "used_mib": 10000}]


def test_acquire_explicit_honours_allow_shared():
    with pytest.raises(GpuUnavailable, match="shared"):
        acquire(machine(), 1, explicit=[1], need_mib=1000, query=query, apps_query=apps)
    lease = acquire(machine(), 1, explicit=[1], need_mib=1000, query=query, apps_query=apps, allow_shared=True)
    assert lease.uuids == ["GPU-b"]


def test_check_explicit_reasons_per_card():
    with pytest.raises(GpuUnavailable) as e:
        check_explicit(machine(gpu_exclude=[3]), [2, 7, 3, 0], need_mib=40000, query=query, apps_query=apps)
    msg = str(e.value)
    assert "GPU 2: 9000 MiB free < 40000 MiB needed" in msg and "pid 4242 (user ?, python3, 71000 MiB)" in msg
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
    lease = acquire(machine(), 1, explicit=[0], need_mib=1000, query=query, apps_query=apps)
    assert lease.uuids == ["GPU-a"] and lease.names == ["NVIDIA A100-SXM4-80GB"] and lease._fds == []


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
    lock = yaml.safe_load(resolve_explicit(tmp_path, [0, 3]).read_text())
    assert lock["n_gpus"] == 2
    assert lock["machine"]["gpu"]["source"] == "explicit" and lock["machine"]["gpu"]["explicit"] == [0, 3]
    assert lock["machine"]["gpu"]["allow_shared"] is False
    cards = lock["feasibility"]["cards"]
    assert [c["uuid"] for c in cards] == ["GPU-a", "GPU-d"] and cards[0]["processes"] == []


def test_resolve_refuses_a_shared_card_unless_allowed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(Infeasible) as e:
        resolve_explicit(tmp_path / "a", [1])
    assert "shared with 1 compute process" in str(e.value)
    assert any("--allow-shared" in a and "ask" in a for a in e.value.alternatives)
    lock = yaml.safe_load(resolve_explicit(tmp_path / "b", [1], allow_shared=True).read_text())
    assert lock["machine"]["gpu"]["allow_shared"] is True
    assert lock["feasibility"]["cards"][0]["processes"][0]["pid"] == 77


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
    lock["machine"]["gpu"].update(source="explicit", explicit=[1, 3], allow_shared=True)
    lock["estimate_gib"] = 32.0
    path.write_text(yaml.safe_dump(lock))
    seen = {}

    def fake_acquire(m, n, **kw):
        seen.update(source=m.gpu_source, n=n, **kw)
        return FakeLease()
    FakePopen.codes = [0]
    assert run(path, acquire=fake_acquire, popen=FakePopen, finalize=lambda rd: None, killpg=lambda *a: None) == 0
    assert seen == {"source": "explicit", "n": 2, "explicit": [1, 3], "need_mib": 32768, "allow_shared": True}


def resolve_base(tmp_path, base, recipe, gpus):
    from harness.resolve import resolve
    exp, m = setup(tmp_path, recipe=recipe, base=base)
    exp.gpus = gpus
    return resolve(exp, m, query=query, apps_query=apps, plan_runner=plan_ok(10, len(gpus)), fingerprint=fake_fp)


def test_too_small_cards_get_concrete_alternatives_and_no_useless_wait(tmp_path):
    with pytest.raises(Infeasible) as e:
        resolve_base(tmp_path, "elephant-1", "elephant_full_fsdp", [3])  # full 9B on one 80 GB card: ~144 GiB
    alts = "\n".join(e.value.alternatives)
    assert "wait" not in alts  # an 80 GB card never holds 144 GiB, however long one waits
    assert "LoRA" in alts and "examples/recipes/joey_lora_1gpu.yaml" in alts  # elephant-1 has no LoRA default
    assert "more GPUs: full FSDP over 2 cards" in alts
    assert "smaller base" in alts and "joey-1" in alts and "mammoth-1" not in alts.split("smaller base", 1)[1]
    assert "never switches" in alts or "ask the user" in alts


def test_base_lora_default_is_named(tmp_path):
    with pytest.raises(Infeasible) as e:
        resolve_base(tmp_path, "mammoth-1", "elephant_full_fsdp", [0, 3])
    alts = "\n".join(e.value.alternatives)
    assert "examples/recipes/mammoth_lora_ddp.yaml" in alts and "LoRA" in alts


def test_wait_is_suggested_only_when_the_card_would_fit(tmp_path):
    with pytest.raises(Infeasible) as e:
        resolve_explicit(tmp_path, [2])  # joey full needs 32 GiB; 9000 MiB free + 71000 MiB of pid 4242
    alts = "\n".join(e.value.alternatives)
    assert "GPU 2 would have enough memory once the listed processes end" in alts


def test_a_failed_process_query_is_not_an_empty_card():
    def broken():
        raise subprocess_error()
    with pytest.raises(GpuUnavailable) as e:
        check_explicit(machine(), [0], need_mib=1000, query=query, apps_query=broken)
    msg = str(e.value)
    assert "GPU 0: cannot list its compute processes" in msg and "--allow-shared" in msg
    cards = check_explicit(machine(), [0], need_mib=1000, query=query, apps_query=broken, allow_shared=True)
    assert cards[0]["processes"] is None  # unknown, not "none"


def subprocess_error():
    import subprocess
    return subprocess.CalledProcessError(9, ["nvidia-smi", "--query-compute-apps=..."])


def test_resolve_with_a_failed_process_query(tmp_path):
    from harness.resolve import resolve
    exp, m = setup(tmp_path, recipe="joey_full_1gpu", base="joey-1")
    exp.gpus = [2]  # also too little memory: the alternatives must cope with unknown processes

    def broken():
        raise subprocess_error()
    with pytest.raises(Infeasible) as e:
        resolve(exp, m, query=query, apps_query=broken, plan_runner=plan_ok(10, 1), fingerprint=fake_fp)
    assert "GPU 2: 9000 MiB free" in str(e.value) and "unknown" in str(e.value)
    assert not any("GPU 0 (" in a for a in e.value.alternatives)  # unknown processes: not offered as free


def _alts(tmp_path, base, recipe, gpus, **machine_kw):
    from harness.resolve import resolve
    exp, m = setup(tmp_path, recipe=recipe, base=base)
    exp.gpus = gpus
    for k, v in machine_kw.items():
        setattr(m, k, v)
    with pytest.raises(Infeasible) as e:
        resolve(exp, m, query=query, apps_query=apps, plan_runner=plan_ok(10, len(gpus)), fingerprint=fake_fp,
                source=str(tmp_path / "exp.yaml"))
    return e.value.alternatives


def test_alternatives_respect_min_free_mib(tmp_path):
    # joey LoRA needs ~5 GiB and GPU 2 has 9000 MiB, but the machine wants 75000 MiB free on a card: the old
    # message said "cards with >= 5 GiB free (the best requested card has 9 GiB)"
    alts = _alts(tmp_path, "joey-1", "joey_lora_1gpu", [2], min_free_mib=75000)
    text = "\n".join(alts)
    assert "cards with >=" not in text
    now = [a for a in alts if "once the listed processes end" not in a]  # waiting is about later, not now
    picked = {int(g) for a in now for g in re.findall(r"--gpus (\d+)", a)}
    assert picked and all(dict(((0, 80000), (1, 70000), (2, 9000), (3, 81000)))[g] >= 75000 for g in picked)
    assert any("GPU 2 would have enough memory once the listed processes end" in a for a in alts)


def test_no_fsdp_suggestion_where_fsdp_does_not_apply(tmp_path):
    # joey-1 (2B) on more cards runs DDP: each card still needs the whole model, so more cards never help
    alts = _alts(tmp_path, "joey-1", "joey_full_1gpu", [2], min_free_mib=100000)
    assert not any("FSDP" in a for a in alts)


def test_every_alternative_is_a_runnable_ojh_run_line(tmp_path):
    alts = _alts(tmp_path, "elephant-1", "elephant_full_fsdp", [3])
    runnable = [a for a in alts if "`ojh run " in a]
    assert len(runnable) >= 3
    for a in runnable:
        cmd = a.split("`ojh run ", 1)[1].split("`", 1)[0]
        assert cmd.startswith(str(tmp_path / "exp.yaml") + " --base ") and " --gpus " in cmd and " --recipe " in cmd
    fsdp = next(a for a in runnable if "FSDP" in a)
    assert "--base elephant-1 --gpus 3,0 --recipe examples/recipes/elephant_full_fsdp.yaml" in fsdp
    lora = next(a for a in runnable if "LoRA" in a)
    assert "--gpus 3 --recipe examples/recipes/joey_lora_1gpu.yaml" in lora
    smaller = next(a for a in runnable if "smaller base" in a)
    assert "--base joey-1 --gpus 3 --recipe examples/recipes/joey_full_1gpu.yaml" in smaller


@pytest.mark.parametrize("gpu_extra,expected", [({}, True), ({"allow_shared": False}, False),
                                                 ({"allow_shared": True}, True)])
def test_runner_passes_the_sharing_choice_in_scan_mode(tmp_path, gpu_extra, expected):
    from test_runner import FakeLease, FakePopen, make_lock
    from harness.runner import run
    path, lock, run_dir = make_lock(tmp_path, seeds=(42,))
    lock["machine"]["gpu"].pop("allow_shared", None)
    lock["machine"]["gpu"].update(gpu_extra)  # no key: a lock from before the rule (sharing allowed then)
    path.write_text(yaml.safe_dump(lock))
    seen = {}
    FakePopen.codes = [0]
    assert run(path, acquire=lambda m, n, **kw: seen.update(source=m.gpu_source, **kw) or FakeLease(),
               popen=FakePopen, finalize=lambda rd: None, killpg=lambda *a: None) == 0
    assert seen == {"source": "scan", "allow_shared": expected}
