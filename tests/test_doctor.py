import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from harness import cli, doctor
from harness.catalog import ROOT
from harness.onejev import verify_onejev

SMI = "0, NVIDIA A100-SXM4-80GB, 570.124.06, 81920, 80000\n1, NVIDIA A100-SXM4-80GB, 570.124.06, 81920, 1000\n"


def good(tmp_path, **over):
    pins = doctor.pins()
    versions = {"torch": pins["torch"], "transformers": pins["transformers"],
                "flash-linear-attention": pins["flash-linear-attention"], "peft": pins["peft"]}
    venv = str(ROOT / ".venv")
    env = {"HOME": str(tmp_path / "home"), "OJH_OUT_ROOT": str(tmp_path / "runs"),
           "OJH_MODEL_DIR": str(tmp_path / "models"), "TMPDIR": str(tmp_path / "tmp"),
           "HF_HOME": str(tmp_path / "hf")}
    p = doctor.Probes(
        python_version=(3, 11, 9), prefix=venv, base_prefix="/opt/py",
        version=lambda name: versions.get(name),
        torch_info=lambda: {"version": pins["torch"], "cuda_available": True, "cuda": "12.8", "devices": 2},
        nvidia_smi=lambda: SMI, which=lambda name: f"/usr/bin/{name}",
        onejev=lambda: [], disk_free=lambda path: 500 * 2**30, writable=lambda path: True,
        http=lambda url, timeout: 200, env=env, machine=None)
    return replace(p, **over)


def statuses(checks):
    return {c.name: c.status for c in checks}


def test_all_good_exits_zero(tmp_path):
    checks = doctor.run_checks(good(tmp_path))
    assert all(c.status == "ok" for c in checks), [(c.name, c.status, c.detail) for c in checks]
    names = [c.name for c in checks]
    for n in ("python", "venv", "torch", "cuda", "gpus", "transformers", "flash-linear-attention", "peft",
              "onejev", "tmux", "machine config", "dir runs (out_root)", "dir base weights (model_dir)",
              "dir uv cache (UV_CACHE_DIR)", "dir python (UV_PYTHON_INSTALL_DIR)", "dir HF cache (HF_HOME)",
              "dir temp (TMPDIR)", "hf hub"):
        assert n in names, n
    gpus = next(c for c in checks if c.name == "gpus")
    assert "A100" in gpus.detail and "2 card" in gpus.detail
    assert doctor.exit_code(checks) == 0


@pytest.mark.parametrize("over,name,hint", [
    ({"python_version": (3, 10, 12)}, "python", "setup.sh"),
    ({"torch_info": lambda: (_ for _ in ()).throw(ImportError("No module named torch"))}, "torch", "setup.sh"),
    ({"torch_info": lambda: {"version": "2.8.0+cu128", "cuda_available": False, "cuda": "12.8", "devices": 0}},
     "cuda", "driver"),
    ({"nvidia_smi": lambda: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))}, "gpus", "driver"),
    ({"version": lambda name: "0.4.0" if name == "flash-linear-attention" else doctor.pins().get(name)},
     "flash-linear-attention", "setup.sh"),
    ({"which": lambda name: None}, "tmux", "tmux"),
    ({"onejev": lambda: ["vendor/OneJev missing"]}, "onejev", "setup"),
    ({"writable": lambda path: "runs" not in str(path)}, "dir runs (out_root)", "OJH_OUT_ROOT"),
])
def test_blocking_problems_fail_with_a_hint(tmp_path, over, name, hint):
    checks = doctor.run_checks(good(tmp_path, **over))
    c = next(c for c in checks if c.name == name)
    assert c.status == "fail" and hint in c.hint, (c.detail, c.hint)
    assert doctor.exit_code(checks) == 1


def test_torch_version_with_local_suffix_matches_the_pin(tmp_path):
    pin = doctor.pins()["torch"]
    checks = doctor.run_checks(good(tmp_path, torch_info=lambda: {"version": f"{pin}+cu128", "cuda_available": True,
                                                                  "cuda": "12.8", "devices": 1}))
    assert statuses(checks)["torch"] == "ok"


def test_network_and_space_problems_only_warn(tmp_path):
    def down(url, timeout):
        raise OSError("timed out")
    checks = doctor.run_checks(good(tmp_path, http=down, disk_free=lambda path: 3 * 2**30))
    s = statuses(checks)
    assert s["hf hub"] == "warn" and s["dir base weights (model_dir)"] == "warn"
    assert doctor.exit_code(checks) == 0


def test_network_check_can_be_skipped(tmp_path):
    calls = []
    p = good(tmp_path, http=lambda url, timeout: calls.append(url) or 200)
    assert statuses(doctor.run_checks(p, network=False))["hf hub"] == "skip" and calls == []
    env = dict(p.env, HF_HUB_OFFLINE="1")
    assert statuses(doctor.run_checks(replace(p, env=env)))["hf hub"] == "skip" and calls == []


def test_not_in_the_repo_venv_warns(tmp_path):
    s = statuses(doctor.run_checks(good(tmp_path, prefix="/usr", base_prefix="/usr")))
    assert s["venv"] == "warn"


def test_bad_machine_config_fails(tmp_path):
    env = dict(good(tmp_path).env, OJH_OUT_ROOT=str(ROOT / "runs"))
    checks = doctor.run_checks(good(tmp_path, env=env))
    c = next(c for c in checks if c.name == "machine config")
    assert c.status == "fail" and "inside the repo" in c.detail


def test_render_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "default_probes", lambda: good(tmp_path, which=lambda n: None))
    assert cli.main(["doctor", "--no-network"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "tmux" in out and "hint:" in out
    assert cli.main(["doctor", "--json", "--no-network"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert any(c["name"] == "tmux" and c["status"] == "fail" for c in data["checks"])


def test_verify_onejev_is_read_only(tmp_path):
    lock = {"revision": "a" * 40, "patched_sha256": {"x.py": "0" * 64}}
    assert verify_onejev(tmp_path / "missing", lock) == [f"{tmp_path / 'missing'} is missing: run setup.sh (or ojh setup)"]
    d = tmp_path / "OneJev"
    d.mkdir()
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    problems = verify_onejev(d, lock)
    assert problems and all("expected" in p or "sha256" in p.lower() or "HEAD" in p for p in problems)
    assert sorted(p.name for p in d.iterdir()) == [".git"]


def test_default_probes_build_without_torch():
    p = doctor.default_probes()
    assert p.python_version[:2] >= (3, 11) and callable(p.torch_info)
