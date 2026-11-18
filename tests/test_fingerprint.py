import hashlib
import os
import stat
import sys
from pathlib import Path

from harness.config import MachineConfig
from harness.resolve import _fingerprint, installed_packages


def executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def pipless_python(tmp_path) -> Path:
    """A python without pip, like a venv made by `uv venv` (no --seed)."""
    return executable(tmp_path / "python", "#!/bin/sh\n"
                      'case "$*" in *"-m pip"*) echo "No module named pip" >&2; exit 1 ;; esac\n'
                      f'exec "{sys.executable}" "$@"\n')


def test_fingerprint_needs_no_pip(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "nvidia-smi", "#!/bin/sh\necho 570.124.06\necho 570.124.06\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    m = MachineConfig(python=str(pipless_python(tmp_path)), model_dir="/m", out_root="/o")
    fp = _fingerprint(m)
    pkgs = installed_packages(m.python)
    assert any(p.startswith("pyyaml==") for p in pkgs) and pkgs == sorted(pkgs)
    assert fp["packages_sha256"] == hashlib.sha256("\n".join(pkgs).encode()).hexdigest()
    assert fp["packages"] == len(pkgs) and fp["packages_source"] == "importlib.metadata"
    assert fp["driver_version"] == "570.124.06"


def test_fingerprint_without_nvidia_smi_records_none(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    m = MachineConfig(python=sys.executable, model_dir="/m", out_root="/o")
    assert _fingerprint(m)["driver_version"] is None


def test_fingerprint_carries_the_setup_stamp(tmp_path):
    import json
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = executable(venv / "bin" / "python", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / ".ojh-setup.json").write_text(json.dumps({"uv_version": "0.13.9", "python_version": "3.11.18"}))
    fp = _fingerprint(MachineConfig(python=str(py), model_dir="/m", out_root="/o"))
    assert fp["setup"] == {"uv_version": "0.13.9", "python_version": "3.11.18"}
    (venv / ".ojh-setup.json").unlink()
    assert _fingerprint(MachineConfig(python=str(py), model_dir="/m", out_root="/o"))["setup"] is None


def test_packages_do_not_depend_on_the_cwd(tmp_path, monkeypatch):
    stray = tmp_path / "work" / "straypkg-9.9.dist-info"
    stray.mkdir(parents=True)
    (stray / "METADATA").write_text("Metadata-Version: 2.1\nName: straypkg\nVersion: 9.9\n")
    monkeypatch.chdir(tmp_path)
    here = installed_packages(sys.executable)
    monkeypatch.chdir(tmp_path / "work")  # e.g. an ojh command run from a source checkout or a workspace
    assert installed_packages(sys.executable) == here
    assert not any(p.startswith("straypkg") for p in here)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "work"))
    assert installed_packages(sys.executable) == here
