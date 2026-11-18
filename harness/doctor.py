"""``ojh doctor``: read-only health check of the environment. Imports without torch (torch is imported
only when probed).

Checks Python and the venv, torch and CUDA, the GPUs (nvidia-smi), the pinned transformers,
flash-linear-attention and peft, the OneJev checkout and patch hashes, tmux, the machine config, every
directory the harness and setup.sh write (writable, free space) and, optionally, the Hugging Face hub.
Every probe is injectable (``Probes``) so the checks can be tested without a GPU. A ``fail`` is blocking
(exit code 1) and carries a hint; a ``warn`` is not blocking.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness.catalog import ROOT

GPU_PACKAGES = ("transformers", "flash-linear-attention", "peft")
LOW_SPACE_GIB = {"runs (out_root)": 20, "base weights (model_dir)": 20}  # others: 5 GiB
SETUP = "run `bash setup.sh` (see README 环境配置)"


@dataclass
class Check:
    name: str
    status: str  # ok, warn, fail, skip
    detail: str
    hint: str = ""


def pins() -> dict[str, str]:
    """``name -> version`` of the ``==`` pins in pyproject's gpu extra (torch, transformers, ...)."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    out = {}
    for spec in data["project"]["optional-dependencies"]["gpu"]:
        if "==" in spec:
            name, version = spec.split("==", 1)
            out[name.strip()] = version.strip()
    return out


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_info() -> dict:
    import torch

    return {"version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
            "cuda": torch.version.cuda, "devices": torch.cuda.device_count() if torch.cuda.is_available() else 0}


def _nvidia_smi() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free",
                           "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
                          timeout=30).stdout


def _onejev() -> list[str]:
    from harness.onejev import load_lock, verify_onejev

    return verify_onejev(ROOT / "vendor" / "OneJev", load_lock())


def _existing(path) -> Path:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _disk_free(path) -> int:
    return shutil.disk_usage(_existing(path)).free


def _writable(path) -> bool:
    p = _existing(path)
    return p.is_dir() and os.access(p, os.W_OK | os.X_OK)


def _http(url: str, timeout: float) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "onejev-harness-doctor"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:  # an HTTP answer still means the hub is reachable
        return e.code


@dataclass
class Probes:
    python_version: tuple = tuple(sys.version_info[:3])
    prefix: str = sys.prefix
    base_prefix: str = sys.base_prefix
    version: Callable[[str], str | None] = _version
    torch_info: Callable[[], dict] = _torch_info
    nvidia_smi: Callable[[], str] = _nvidia_smi
    which: Callable[[str], str | None] = shutil.which
    onejev: Callable[[], list[str]] = _onejev
    disk_free: Callable[[Any], int] = _disk_free
    writable: Callable[[Any], bool] = _writable
    http: Callable[[str, float], int] = _http
    env: dict = field(default_factory=lambda: dict(os.environ))
    machine: Any = None  # (MachineConfig, origin); None: found with ``env`` like any ojh command


def default_probes() -> Probes:
    return Probes()


@contextlib.contextmanager
def _environ(env: dict):
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB free"


def _dirs(p: Probes, machine) -> list[tuple[str, str, bool]]:
    """(label, path, blocking when not writable)."""
    env = p.env
    home = env.get("HOME", str(Path.home()))
    tools = env.get("OJH_TOOLS_DIR") or str(ROOT / ".tools")
    out = []
    if machine is not None:
        out += [("runs (out_root)", machine.out_root, True), ("base weights (model_dir)", machine.model_dir, True)]
    out += [("HF cache (HF_HOME)", env.get("HF_HOME") or os.path.join(home, ".cache", "huggingface"), True),
            ("temp (TMPDIR)", env.get("TMPDIR") or "/tmp", True),
            ("tools (OJH_TOOLS_DIR)", tools, False),
            ("uv cache (UV_CACHE_DIR)", env.get("UV_CACHE_DIR") or os.path.join(tools, "cache", "uv"), False),
            ("python (UV_PYTHON_INSTALL_DIR)", env.get("UV_PYTHON_INSTALL_DIR") or os.path.join(tools, "python"),
             False),
            ("venv (OJH_VENV)", env.get("OJH_VENV") or str(ROOT / ".venv"), False)]
    return out


def run_checks(p: Probes, *, network: bool = True, timeout: float = 5.0) -> list[Check]:
    out: list[Check] = []
    add = lambda *a, **k: out.append(Check(*a, **k))  # noqa: E731
    pin = pins()

    v = p.python_version
    add("python", "ok" if v[:2] >= (3, 11) else "fail", ".".join(map(str, v)),
        "" if v[:2] >= (3, 11) else f"Python >= 3.11 is required; {SETUP}: it installs a uv-managed 3.11")
    venv = os.path.realpath(p.env.get("OJH_VENV") or ROOT / ".venv")
    in_venv = p.prefix != p.base_prefix and os.path.realpath(p.prefix) == venv
    add("venv", "ok" if in_venv else "warn", p.prefix,
        "" if in_venv else f"not running from the harness venv ({venv}); use <repo>/.venv/bin/ojh")

    try:
        t = p.torch_info()
    except Exception as e:  # ImportError, or a broken CUDA install raising on import
        t = None
        add("torch", "fail", f"not importable: {type(e).__name__}: {e}", SETUP)
    if t is not None:
        ok = str(t["version"]).split("+")[0] == pin.get("torch")
        add("torch", "ok" if ok else "fail", f"{t['version']} (pin {pin.get('torch')}), CUDA build {t.get('cuda')}",
            "" if ok else f"re-run setup.sh; it installs the pinned torch from the CUDA 12.8 wheel index")
    if t is None:
        add("cuda", "fail", "torch missing", "install torch first (setup.sh)")
    elif t.get("cuda_available"):
        add("cuda", "ok", f"available, {t.get('devices')} device(s)")
    else:
        add("cuda", "fail", "torch.cuda.is_available() is False",
            "the NVIDIA driver must support CUDA 12.8 wheels (driver R570 or newer; see nvidia-smi); "
            "check CUDA_VISIBLE_DEVICES")
    try:
        rows = [r.split(", ") for r in p.nvidia_smi().strip().splitlines() if r.strip()]
        listed = "; ".join(f"{r[0]} {r[1]} ({r[4]}/{r[3]} MiB free)" for r in rows if len(r) >= 5)
        drivers = sorted({r[2] for r in rows if len(r) >= 5})
        add("gpus", "ok" if rows else "fail", f"{len(rows)} card(s), driver {', '.join(drivers) or '?'}: {listed}",
            "" if rows else "nvidia-smi lists no GPU: check the NVIDIA driver")
    except Exception as e:
        add("gpus", "fail", f"nvidia-smi failed: {type(e).__name__}: {e}",
            "install the NVIDIA driver (R570+ for CUDA 12.8 wheels) and make sure nvidia-smi is on PATH")

    for name in GPU_PACKAGES:
        got, want = p.version(name), pin.get(name)
        ok = got is not None and got == want
        add(name, "ok" if ok else "fail", f"{got or 'not installed'} (pin {want})",
            "" if ok else f"{SETUP} again; it installs the pins from constraints.txt")

    problems = p.onejev()
    add("onejev", "fail" if problems else "ok", "; ".join(problems) or "pinned revision, patch hashes verified",
        f"{SETUP} (or <repo>/.venv/bin/ojh setup)" if problems else "")
    tmux = p.which("tmux")
    add("tmux", "ok" if tmux else "fail", tmux or "not found",
        "" if tmux else "ojh launch runs training in tmux: install tmux (apt install tmux, or ask the admin)")

    machine = None
    try:
        if p.machine is not None:
            machine, origin = p.machine
        else:
            from harness.config import find_machine
            with _environ(p.env):
                machine, origin = find_machine(None)
        add("machine config", "ok", f"{origin['source']}" + (f" ({origin['path']})" if origin.get("path") else ""))
    except Exception as e:
        add("machine config", "fail", f"{type(e).__name__}: {e}",
            "fix the machine config, or set OJH_OUT_ROOT / OJH_MODEL_DIR outside the repo")

    for label, path, blocking in _dirs(p, machine):
        free = p.disk_free(path)
        need = LOW_SPACE_GIB.get(label, 5)
        if not p.writable(path):
            var = label.split("(")[-1].rstrip(")")
            add(f"dir {label}", "fail" if blocking else "warn", f"{path}: not writable",
                f"set {'OJH_OUT_ROOT' if var == 'out_root' else 'OJH_MODEL_DIR' if var == 'model_dir' else var} "
                "to a writable directory (examples/env.example.sh puts all of them on one disk)")
        elif free < need * 2**30:
            add(f"dir {label}", "warn", f"{path}: {_gib(free)} (< {need} GiB)",
                "move it to a bigger disk (examples/env.example.sh)")
        else:
            add(f"dir {label}", "ok", f"{path}: {_gib(free)}")

    endpoint = p.env.get("HF_ENDPOINT") or "https://huggingface.co"
    if not network or p.env.get("HF_HUB_OFFLINE") in ("1", "true", "True"):
        add("hf hub", "skip", "not checked (--no-network or HF_HUB_OFFLINE)")
    else:
        try:
            code = p.http(endpoint + "/api/models/baselabs/joey-1", timeout)
            add("hf hub", "ok" if code < 500 else "warn", f"{endpoint} answered HTTP {code}")
        except Exception as e:
            add("hf hub", "warn", f"{endpoint} not reachable within {timeout:g} s: {type(e).__name__}: {e}",
                "needed for setup and fetch-base only; offline, copy the weights into model_dir and set "
                "HF_HUB_OFFLINE=1 (README 环境配置)")
    return out


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "fail" for c in checks) else 0


MARK = {"ok": " ok ", "warn": "WARN", "fail": "FAIL", "skip": "skip"}


def render(checks: list[Check]) -> str:
    lines = [f"[{MARK[c.status]}] {c.name}: {c.detail}" + (f"\n       hint: {c.hint}" if c.hint else "")
             for c in checks]
    fails = sum(c.status == "fail" for c in checks)
    lines.append(f"{fails} blocking problem(s)" if fails else "no blocking problem")
    return "\n".join(lines)


def as_json(checks: list[Check]) -> dict:
    return {"ok": exit_code(checks) == 0, "checks": [asdict(c) for c in checks]}
