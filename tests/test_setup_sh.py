import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_setup_sh_parses():
    subprocess.run(["bash", "-n", str(ROOT / "setup.sh")], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "examples" / "env.example.sh")], check=True)


def repo_copy(tmp_path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for f in ("setup.sh", "constraints.txt", "pyproject.toml"):
        shutil.copy(ROOT / f, repo / f)
    return repo


def dry_run(tmp_path, repo, extra=""):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    ojh = tmp_path / "big" / "ojh"
    script = (f'export OJH_HOME="{ojh}"; unset TMPDIR; source "{ROOT}/examples/env.example.sh"; '
              f'bash "{repo}/setup.sh" --dry-run {extra}')
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
    return proc, ojh, home


def test_dry_run_prints_the_plan_below_ojh_home(tmp_path):
    repo = repo_copy(tmp_path)
    proc, ojh, home = dry_run(tmp_path, repo)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "dry run: nothing is changed" in out
    for label, sub in (("tools (OJH_TOOLS_DIR)", "tools"), ("uv cache (UV_CACHE_DIR)", "cache/uv"),
                       ("python (UV_PYTHON_INSTALL_DIR)", "python"), ("venv (OJH_VENV)", "venv"),
                       ("HF cache (HF_HOME)", "cache/huggingface"), ("temp (TMPDIR)", "tmp"),
                       ("runs (OJH_OUT_ROOT)", "runs"), ("base weights (OJH_MODEL_DIR)", "models")):
        line = next(l for l in out.splitlines() if label in l)
        assert str(ojh / sub) in line and "GiB free" in line, line
    assert f"UV_UNMANAGED_INSTALL={ojh / 'tools' / 'uv'}" in out  # uv on PATH is hidden: it would be installed
    assert f"python install 3.11" in out and f"venv --python 3.11 {ojh / 'venv'}" in out
    assert f"ln -sfn {ojh / 'venv'} {repo / '.venv'}" in out
    assert "--index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0" in out
    assert "-c constraints.txt -e .[gpu]" in out and "pip check" in out and "ojh setup" in out
    for n in range(1, 6):
        assert f"[{n}/5]" in out
    # nothing written: only the TMPDIR that env.example.sh creates
    assert sorted(p.relative_to(ojh).as_posix() for p in ojh.rglob("*")) == ["tmp"]
    assert list(home.iterdir()) == [] and sorted(p.name for p in repo.iterdir()) == [
        "constraints.txt", "pyproject.toml", "setup.sh"]


def test_dry_run_refuses_out_root_in_the_repo(tmp_path):
    repo = repo_copy(tmp_path)
    script = (f'export OJH_OUT_ROOT="{repo}/runs"; bash "{repo}/setup.sh" --dry-run')
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 1 and "inside the repo" in proc.stderr


def test_python_opt_in_and_unknown_flag(tmp_path):
    repo = repo_copy(tmp_path)
    proc = subprocess.run(["bash", str(repo / "setup.sh"), "--bogus"], capture_output=True, text=True)
    assert proc.returncode == 2 and "unknown argument" in proc.stderr
    proc = subprocess.run(["bash", str(repo / "setup.sh"), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "--dry-run" in proc.stdout and "--python" in proc.stdout
