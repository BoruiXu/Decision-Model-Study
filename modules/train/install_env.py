"""Helper of install_env.sh (stdlib only; pyyaml is used to validate an edit when it is importable).

Subcommands (run `install_env.py <subcommand> --help`). The ones install_env.sh reads print human lines first, then a
line `#INSTALL_ENV_VARS` and shell assignments (shlex-quoted) that install_env.sh evals.

  probe        (venv python) Python tag, platform, torch build, installed versions, a pip nvcc
  nvcc         rank the nvcc compilers on this machine against torch's CUDA version
  gpus         compute capabilities, and the card for the GPU test (no processes, then most free memory)
  imports      (venv python) import each library and report ok / missing / failed
  gpu-test     (venv python) a tiny forward + backward through fla's chunk_gated_delta_rule (joey-1's op) and
               causal_conv1d_fn on random tensors
  machine-env  plan, and with --apply make, the edit of CUDA_HOME, PATH and TILELANG_CACHE_DIR in the machine
               config's env (read with pyyaml; written line by line: other keys, comments and CRLF stay; backup first)
  run-env      run a command with the machine config's env (and --set overrides) on top of the current environment
               (as ojh does: values are not expanded)
"""
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

MARK = "#INSTALL_ENV_VARS"
DISTS = ("torch", "torchvision", "triton", "transformers", "flash-linear-attention", "fla-core", "causal-conv1d",
         "tilelang", "pyyaml", "pytest", "pip")
PIP_NVCC_DISTS = ("nvidia-cuda-nvcc", "nvidia-cuda-nvcc-cu13", "nvidia-cuda-nvcc-cu12")


def emit(**values):
    """The marker line and one shell assignment per value (None becomes an empty string)."""
    print(MARK)
    for key, value in values.items():
        text = "" if value is None else " ".join(str(value).splitlines())   # one line per assignment
        print("%s=%s" % (key, shlex.quote(text)))
    sys.stdout.flush()


def var_name(dist):
    return "V_" + re.sub(r"[^A-Z0-9]", "_", dist.upper())


# ---- probe (venv python) ----

def pip_nvcc():
    """Absolute path of the nvcc binary a pip `nvidia-cuda-nvcc` package installed in this environment, else None
    (the -cu12 packages ship only ptxas)."""
    try:
        import importlib.metadata as md
    except ImportError:
        return None
    for dist in PIP_NVCC_DISTS:
        try:
            files = md.files(dist) or []
        except md.PackageNotFoundError:
            continue
        for f in files:
            if f.name == "nvcc" and f.parent.name == "bin":
                path = os.path.realpath(str(f.locate()))
                if os.access(path, os.X_OK):
                    return path
    return None


def cmd_probe(args):
    import importlib.metadata as md
    import sysconfig
    versions = {}
    for dist in DISTS:
        try:
            versions[var_name(dist)] = md.version(dist)
        except md.PackageNotFoundError:
            versions[var_name(dist)] = ""
    torch_version = torch_cuda = abi = torch_error = ""
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda or ""
        abi = "TRUE" if bool(torch._C._GLIBCXX_USE_CXX11_ABI) else "FALSE"
    except Exception as e:  # noqa: BLE001 - any failure means "cannot pick a wheel"
        torch_error = "%s: %s" % (type(e).__name__, e)
    emit(P_PY_VERSION="%d.%d.%d" % sys.version_info[:3], P_PY_TAG="cp%d%d" % sys.version_info[:2],
         P_PLATFORM=sysconfig.get_platform(), P_TORCH=torch_version, P_TORCH_CUDA=torch_cuda, P_CXX11ABI=abi,
         P_TORCH_ERROR=torch_error, P_NVCC_PIP=pip_nvcc() or "", **versions)
    return 0


# ---- nvcc ----

def nvcc_version(home):
    """(major, minor) of <home>/bin/nvcc when it runs, else None."""
    nvcc = os.path.join(home, "bin", "nvcc")
    if not os.access(nvcc, os.X_OK):
        return None
    try:
        out = subprocess.run([nvcc, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_cuda(text):
    m = re.match(r"(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def cmd_nvcc(args):
    """Every nvcc found, in the order CUDA_HOME, CUDA_PATH, the machine config's CUDA_HOME, /usr/local/cuda*, the
    venv's pip package, PATH; the best one matches torch's CUDA major.minor, then its major, then is the newest."""
    seen, found = set(), []
    machine_home = read_env(args.machine).get("CUDA_HOME") if args.machine and os.path.isfile(args.machine) else None
    sources = [("CUDA_HOME", os.environ.get("CUDA_HOME")), ("CUDA_PATH", os.environ.get("CUDA_PATH")),
               ("machine config", machine_home)]
    sources += [("/usr/local", d) for d in sorted(glob.glob("/usr/local/cuda*"))]
    if args.pip_nvcc:
        sources.append(("pip nvidia-cuda-nvcc", os.path.dirname(os.path.dirname(args.pip_nvcc))))
    on_path = shutil.which("nvcc")
    if on_path:
        sources.append(("PATH", os.path.dirname(os.path.dirname(os.path.realpath(on_path)))))
    for source, home in sources:
        if not home or not os.path.isdir(home):
            continue
        real = os.path.realpath(home)
        if real in seen:
            continue
        seen.add(real)
        version = nvcc_version(real)
        if version:
            found.append((source, real, version))
    want = parse_cuda(args.torch_cuda)

    def rank(item):
        order, (_source, _home, version) = item
        match = 0 if want and version == want else 1 if want and version[0] == want[0] else 2
        return (match, -version[0], -version[1], order)

    ranked = [item for _, item in sorted(enumerate(found), key=rank)]
    for source, home, version in ranked:
        print("nvcc %d.%d  %s  (%s)" % (version[0], version[1], home, source))
    best = ranked[0] if ranked else None
    emit(NVCC_HOME=best[1] if best else "", NVCC_VERSION="%d.%d" % best[2] if best else "",
         NVCC_SOURCE=best[0] if best else "", NVCC_COUNT=len(ranked))
    return 0


# ---- GPUs ----

def nvidia_smi(*query):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe] + list(query), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


def gpu_exclude(machine):
    """The machine config's top-level `gpu_exclude: [i, j]` (inline list), else []."""
    if not machine or not os.path.isfile(machine):
        return []
    with open(machine, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"gpu_exclude\s*:\s*\[([^\]]*)\]", line)
            if m:
                return [int(x) for x in re.findall(r"\d+", m.group(1))]
    return []


def cmd_gpus(args):
    rows = nvidia_smi("--query-gpu=index,uuid,name,compute_cap,memory.used,memory.total",
                      "--format=csv,noheader,nounits")
    if rows is None:
        print("no nvidia-smi, or it failed")
        emit(GPU_COUNT=0, GPU_MAX_CAP="", GPU_CAPS="", GPU_PICK_UUID="", GPU_PICK_DESC="")
        return 0
    apps = nvidia_smi("--query-compute-apps=gpu_uuid", "--format=csv,noheader") or []
    busy = {r[0] for r in apps if r}
    exclude = set(gpu_exclude(args.machine))
    cards = []
    for r in rows:
        if len(r) < 6:
            continue
        index, uuid, name, cap, used, total = r[:6]
        try:
            capf, free = float(cap), int(float(total)) - int(float(used))
        except ValueError:
            continue  # [N/A]
        cards.append({"index": int(index), "uuid": uuid, "name": name, "cap": capf, "free": free,
                      "busy": uuid in busy, "excluded": int(index) in exclude})
    usable = [c for c in cards if not c["excluded"] and c["free"] >= args.min_free_mib]
    usable.sort(key=lambda c: (c["busy"], -c["cap"], -c["free"]))
    pick = usable[0] if usable else None
    for c in cards:
        print("gpu %d %s cc %.1f free %d MiB%s%s" % (c["index"], c["name"], c["cap"], c["free"],
                                                     ", has processes" if c["busy"] else "",
                                                     ", excluded" if c["excluded"] else ""))
    caps = sorted({c["cap"] for c in cards})
    emit(GPU_COUNT=len(cards), GPU_MAX_CAP="%.1f" % max(caps) if caps else "",
         GPU_CAPS=" ".join("%.1f" % c for c in caps), GPU_PICK_UUID=pick["uuid"] if pick else "",
         GPU_PICK_DESC=("%d %s (cc %.1f, %d MiB free%s)" % (pick["index"], pick["name"], pick["cap"], pick["free"],
                                                           ", has processes" if pick["busy"] else ""))
         if pick else "")
    return 0


# ---- imports (venv python) ----

def _import_torch():
    import torch
    return torch.__version__


def _import_transformers():
    import transformers
    return transformers.__version__


def _import_fla():
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401 - joey-1's op
    return getattr(fla, "__version__", "?")


def _import_causal_conv1d():
    import causal_conv1d
    from causal_conv1d import causal_conv1d_fn  # noqa: F401 - loads the CUDA extension
    return getattr(causal_conv1d, "__version__", "?")


def _import_tilelang():
    import tilelang
    return getattr(tilelang, "__version__", "?")


def _import_yaml():
    import yaml
    return yaml.__version__


IMPORTS = {"torch": _import_torch, "transformers": _import_transformers, "fla": _import_fla,
           "causal_conv1d": _import_causal_conv1d, "tilelang": _import_tilelang, "yaml": _import_yaml}


def cmd_imports(args):
    out = {}
    for name in args.names.split(","):
        up = name.upper()
        try:
            out["IMP_" + up], out["IMPV_" + up], out["IMPE_" + up] = "ok", IMPORTS[name](), ""
        except ModuleNotFoundError as e:
            out["IMP_" + up] = "missing" if e.name == name else "failed"
            out["IMPV_" + up], out["IMPE_" + up] = "", "%s: %s" % (type(e).__name__, e)
        except Exception as e:  # noqa: BLE001 - an import that raises anything else is broken
            out["IMP_" + up], out["IMPV_" + up], out["IMPE_" + up] = "failed", "", _first_line(e)
    emit(**out)
    return 0


# ---- GPU test (venv python) ----

HOPPER_ERROR = "on Hopper GPUs produces incorrect results"


def _first_line(e):
    text = str(e).strip().splitlines()
    return "%s: %s" % (type(e).__name__, text[0] if text else "")


def cmd_gpu_test(args):
    import traceback
    t0 = time.time()
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        print("torch sees no GPU (CUDA_VISIBLE_DEVICES=%s)" % os.environ.get("CUDA_VISIBLE_DEVICES"))
        emit(GT_FLA="failed", GT_FLA_ERR="torch sees no GPU", GT_CC="skip", GT_CC_ERR="")
        return 1
    dev = torch.device("cuda", 0)
    cap = torch.cuda.get_device_capability(0)
    print("card: %s (sm%d%d), torch %s, CUDA_HOME=%s" % (torch.cuda.get_device_name(0), cap[0], cap[1],
                                                       torch.__version__, os.environ.get("CUDA_HOME", "")))
    result = {"GT_FLA": "failed", "GT_FLA_ERR": "", "GT_CC": "missing", "GT_CC_ERR": ""}
    bf16 = torch.bfloat16
    torch.manual_seed(0)
    try:
        import triton
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        print("triton %s, FLA_TILELANG=%s" % (triton.__version__, os.environ.get("FLA_TILELANG", "(unset)")))
        try:
            from fla.ops.common.backends.tilelang import TileLangBackend
            print("fla tilelang backend: available %s, enabled %s" % (TileLangBackend.is_available(),
                                                                      TileLangBackend.is_enabled()))
        except Exception as e:  # noqa: BLE001 - informational only
            print("fla tilelang backend: unknown (%s)" % _first_line(e))
        # joey-1 (transformers qwen3_5): bf16 q/k/v with 128-wide heads, fp32 log-decay g, bf16 beta, l2norm in kernel
        B, T, H, K, V = 1, 256, 4, 128, 128
        q = torch.randn(B, T, H, K, device=dev, dtype=bf16).requires_grad_()
        k = torch.randn(B, T, H, K, device=dev, dtype=bf16).requires_grad_()
        v = torch.randn(B, T, H, V, device=dev, dtype=bf16).requires_grad_()
        g = F.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32)).detach().requires_grad_()
        beta = torch.rand(B, T, H, device=dev, dtype=bf16).requires_grad_()
        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                      use_qk_l2norm_in_kernel=True)
        o.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        bad = [n for n, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
               if t.grad is None or not bool(torch.isfinite(t.grad).all()) or not bool(t.grad.abs().sum() > 0)]
        if bad:
            result["GT_FLA_ERR"] = "gradients missing, zero or not finite: %s" % ", ".join(bad)
        else:
            result["GT_FLA"] = "ok"
            print("fla chunk_gated_delta_rule: forward + backward ok, gradients finite (%.1fs)" % (time.time() - t0))
    except Exception as e:  # noqa: BLE001 - reported, with the traceback on stderr
        traceback.print_exc(limit=-3)
        result["GT_FLA_ERR"] = _first_line(e)
        if HOPPER_ERROR in str(e):
            result["GT_FLA_ERR"] += " [the fla/Triton Hopper error: tilelang missing, disabled or without nvcc]"
    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError as e:
        result["GT_CC_ERR"] = _first_line(e)
    else:
        try:
            x = torch.randn(1, 256, 64, device=dev, dtype=bf16).requires_grad_()   # (batch, channels, length)
            w = torch.randn(256, 4, device=dev, dtype=bf16).requires_grad_()
            causal_conv1d_fn(x, w, None, activation="silu").float().sum().backward()
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(x.grad).all()) and bool(torch.isfinite(w.grad).all())
            result["GT_CC"] = "ok" if ok else "failed"
            result["GT_CC_ERR"] = "" if ok else "gradients not finite"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc(limit=-3)
            result["GT_CC"], result["GT_CC_ERR"] = "failed", _first_line(e)
    emit(**result)
    return 0 if result["GT_FLA"] == "ok" else 1


# ---- the machine config's env block (line based) ----

class EditError(Exception):
    pass


ENTRY_RE = re.compile(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*)\s*:(?:\s+(.*?))?\s*$")
PLAIN_RE = re.compile(r"[A-Za-z0-9_./+@%,=~-][A-Za-z0-9_./:+@%,=~-]*")
SPECIAL_RE = re.compile(r"[-+]?(\d[\d_]*(\.\d*)?([eE][-+]?\d+)?|\.\d+|\.inf|\.nan)|true|false|yes|no|on|off|y|n|"
                        r"null|~", re.I)
# A PATH written from scratch: nvcc's bin, the venv's bin, then these when they exist (never the caller's PATH: it
# reaches the training and scoring processes through the machine config).
SYSTEM_PATH = ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")


def split_value(raw):
    """(value, comment) of what follows `KEY:` on a line: the scalar as a string (None when empty or null) and the
    inline comment with its leading whitespace ('' when there is none)."""
    raw = (raw or "").strip()
    if raw.startswith("#"):
        return None, " " + raw
    if raw.startswith('"'):
        end = 1
        while end < len(raw):
            if raw[end] == "\\":
                end += 2
                continue
            if raw[end] == '"':
                return json.loads(raw[:end + 1]), raw[end + 1:]
            end += 1
        raise EditError("unterminated quoted value: %s" % raw)
    if raw.startswith("'"):
        m = re.match(r"'((?:[^']|'')*)'", raw)
        if not m:
            raise EditError("unterminated quoted value: %s" % raw)
        return m.group(1).replace("''", "'"), raw[m.end():]
    m = re.search(r"\s#", raw)
    value, comment = (raw[:m.start()], raw[m.start():]) if m else (raw, "")
    value = value.strip()
    return (None if value in ("", "~", "null") else value), comment


def format_scalar(value):
    if PLAIN_RE.fullmatch(value) and not SPECIAL_RE.fullmatch(value) and not value.endswith(":"):
        return value
    return json.dumps(value)


EMPTY_INLINE_RE = re.compile(r"env\s*:\s*\{\s*\}\s*(#.*)?$")


def env_block(lines):
    """(index of the top-level `env:` line or None, index one past its last entry,
    {KEY: (index, indent, value, comment)}). `env: {}` counts as an empty block; any other inline map raises
    EditError (the line editor cannot rewrite it)."""
    start = next((i for i, line in enumerate(lines) if re.match(r"env\s*:", line)), None)
    if start is None:
        return None, None, {}
    if EMPTY_INLINE_RE.match(lines[start]):
        return start, start + 1, {}
    rest = lines[start].split(":", 1)[1].strip()
    if rest and not rest.startswith("#"):
        raise EditError("env is written as an inline map (`env: %s`), which this line editor does not rewrite" % rest)
    entries, last, i = {}, start, start + 1
    while i < len(lines) and (not lines[i].strip() or lines[i][0] in " \t"):
        m = ENTRY_RE.match(lines[i])
        if m and not lines[i].lstrip().startswith("#"):
            value, comment = split_value(m.group(3))
            entries[m.group(2)] = (i, m.group(1), value, comment)
            last = i
        i += 1
    return start, last + 1, entries


def env_str(value):
    """A machine config env value as ojh passes it to the process (harness/config.py `_env`)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return None if value is None else str(value)


def read_env(machine):
    """The machine config's env as strings, read with pyyaml (inline maps work); the line parser when pyyaml is
    missing."""
    with open(machine, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
    except ImportError:
        try:
            _start, _end, entries = env_block(text.splitlines())
        except EditError as e:
            print("install_env.py: %s; reading it needs pyyaml" % e, file=sys.stderr)
            return {}
        return {k: v for k, (_i, _ind, v, _c) in entries.items()}
    data = yaml.safe_load(text) or {}
    env = data.get("env") if isinstance(data, dict) else None
    return {str(k): env_str(v) for k, v in env.items()} if isinstance(env, dict) else {}


def set_env(text, updates, comment):
    """`text` with the env entries in `updates` replaced (keeping their inline comments) or added; every other line
    unchanged, CRLF line endings kept. The comment line goes above the added entries, else above the first replaced
    one."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    start, end, entries = env_block(lines)
    indent = next((e[1] for e in entries.values()), "  ")
    if start is None:
        new = ([""] if lines and lines[-1].strip() else []) + ["env:", indent + "# " + comment]
        new += ["%s%s: %s" % (indent, k, format_scalar(v)) for k, v in updates.items()]
        return newline.join(lines + new) + newline
    m = EMPTY_INLINE_RE.match(lines[start])
    if m:
        lines[start] = "env:" + (" " + m.group(1) if m.group(1) else "")
    replaced = {entries[k][0]: "%s%s: %s%s" % (entries[k][1], k, format_scalar(v), entries[k][3])
                for k, v in updates.items() if k in entries}
    added = ["%s%s: %s" % (indent, k, format_scalar(v)) for k, v in updates.items() if k not in entries]
    out = []
    for i, line in enumerate(lines):
        if i == end and added:
            out += [indent + "# " + comment] + added
        if replaced and i == min(replaced) and not added:
            out.append(entries[next(k for k in entries if entries[k][0] == i)][1] + "# " + comment)
        out.append(replaced.get(i, line))
    if end >= len(lines) and added:
        out += [indent + "# " + comment] + added
    return newline.join(out) + newline


def validate_edit(old_text, new_text, updates):
    """None when pyyaml reads both files and only the updated env keys differ; else the problem. Without pyyaml the
    line parser re-reads the env block."""
    try:
        import yaml
        old, new = yaml.safe_load(old_text) or {}, yaml.safe_load(new_text) or {}
    except ImportError:
        _s, _e, entries = env_block(new_text.splitlines())
        wrong = [k for k, v in updates.items() if k not in entries or entries[k][2] != v]
        return "env re-read differs for %s" % wrong if wrong else None
    except Exception as e:  # noqa: BLE001 - a YAML error in the new file
        return "the edited file is not valid YAML: %s" % _first_line(e)
    old_env, new_env = dict(old.get("env") or {}), dict(new.get("env") or {})
    for k, v in updates.items():
        if new_env.get(k) != v:
            return "env.%s reads back as %r, not %r" % (k, new_env.get(k), v)
    for k in set(old_env) | set(new_env):
        if k not in updates and old_env.get(k) != new_env.get(k):
            return "env.%s changed" % k
    for k in set(old) | set(new):
        if k != "env" and old.get(k) != new.get(k):
            return "top-level key %s changed" % k
    return None


def is_cuda_bin(path):
    return bool(re.search(r"/cuda[^/]*/bin/?$", path)) or bool(re.search(r"/nvidia/cuda_nvcc/bin/?$", path))


def plan_cuda_home(env, best, torch_cuda):
    """The CUDA_HOME to write and a note. The config's CUDA_HOME is kept when its nvcc runs, unless its CUDA major
    differs from torch's and `best` (the ranked nvcc) matches torch's major; else `best`."""
    want = parse_cuda(torch_cuda)
    best_version = nvcc_version(best) if best else None
    current = env.get("CUDA_HOME")
    current_version = nvcc_version(current) if current else None
    if current_version:
        if (want and current_version[0] != want[0] and best_version and best_version[0] == want[0]
                and os.path.realpath(best) != os.path.realpath(current)):
            return best, ("CUDA_HOME %s has nvcc %d.%d but torch is built with CUDA %s: switching to %s (nvcc %d.%d)"
                          % ((current,) + current_version + (torch_cuda, best) + best_version))
        return current, ""
    if best_version:
        return best, ("CUDA_HOME %s has no working nvcc: replacing it with %s" % (current, best)) if current else ""
    return None, ""


def cmd_machine_env(args):
    """Plan (and with --apply write) CUDA_HOME, PATH and TILELANG_CACHE_DIR in the machine config's env. PATH is kept
    when it starts with <CUDA_HOME>/bin, else CUDA_HOME's bin goes first; TILELANG_CACHE_DIR is kept when set. The
    plan is printed and emitted (ME_CUDA_HOME, ME_PATH, ME_TILELANG) so the GPU test can run with it first."""
    machine = os.path.realpath(args.machine)   # a symlinked config is edited at its target
    env = read_env(machine)
    home, note = plan_cuda_home(env, args.cuda_home, args.torch_cuda)
    if note:
        print(note)
    if not home:
        print("no nvcc: CUDA_HOME cannot be set")
        emit(ME_STATUS="no_nvcc", ME_CHANGES="", ME_BACKUP="", ME_CUDA_HOME="", ME_PATH="", ME_TILELANG="",
             ME_MANUAL="")
        return 1
    cuda_bin = home.rstrip("/") + "/bin"
    current_path = env.get("PATH")
    if current_path and current_path.split(":")[0].rstrip("/") == cuda_bin:
        path = current_path
    else:
        rest = current_path.split(":") if current_path else [args.venv_bin] + [d for d in SYSTEM_PATH
                                                                               if os.path.isdir(d)]
        old_home = (env.get("CUDA_HOME") or "").rstrip("/")
        old_bins = {old_home + "/bin", os.path.realpath(old_home + "/bin")} if old_home else set()
        parts = [cuda_bin]
        for p in rest:
            stale = p.rstrip("/") in old_bins or os.path.realpath(p) in old_bins   # the replaced CUDA_HOME's bin
            if p and not is_cuda_bin(p) and not stale and p not in parts:
                parts.append(p)
        path = ":".join(parts)
    want = {"CUDA_HOME": home, "PATH": path, "TILELANG_CACHE_DIR": env.get("TILELANG_CACHE_DIR") or args.tilelang_cache}
    updates = {k: v for k, v in want.items() if env.get(k) != v}
    for k, v in want.items():
        if k in updates:
            print("%s %s: %s -> %s" % ("set" if args.apply else "would set", k, env.get(k), v))
        else:
            print("keep %s: %s" % (k, v))
    planned = dict(ME_CUDA_HOME=want["CUDA_HOME"], ME_PATH=want["PATH"], ME_TILELANG=want["TILELANG_CACHE_DIR"])
    manual = "; ".join("%s: %s" % (k, format_scalar(v)) for k, v in updates.items())
    expected = dict(item.partition("=")[::2] for item in args.expect or [])
    mismatch = [k for k, v in expected.items() if want.get(k) != v]
    if mismatch:
        print("refusing to write: the plan for %s differs from the values the GPU test ran with" % ", ".join(mismatch))
        emit(ME_STATUS="failed", ME_CHANGES=" ".join(updates), ME_BACKUP="", ME_MANUAL=manual, ME_CUDA_HOME=home,
             ME_PATH=path, ME_TILELANG=want["TILELANG_CACHE_DIR"])
        return 1
    status, backup = ("plan" if updates else "right"), ""
    if updates:
        with open(machine, encoding="utf-8", newline="") as f:
            old_text = f.read()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        comment = ("%s install_env.sh: tilelang's JIT (fla on Hopper) needs nvcc; ojh does not expand variables in "
                   "env, so PATH is written out in full. Backup: %s" % (stamp[:8], os.path.basename(machine)
                                                                         + ".bak-" + stamp))
        try:
            new_text = set_env(old_text, updates, comment)
        except EditError as e:
            print("cannot edit the machine config: %s. Set these in its env by hand: %s" % (e, manual))
            emit(ME_STATUS="manual", ME_CHANGES=" ".join(updates), ME_BACKUP="", ME_MANUAL=manual, **planned)
            return 1
        problem = validate_edit(old_text, new_text, updates)
        if problem:
            print("edit not written: %s" % problem)
            emit(ME_STATUS="failed", ME_CHANGES=" ".join(updates), ME_BACKUP="", ME_MANUAL=manual, **planned)
            return 1
        if args.apply:
            backup = machine + ".bak-" + stamp
            n = 1
            while os.path.exists(backup):
                backup, n = "%s.bak-%s-%d" % (machine, stamp, n), n + 1
            shutil.copy2(machine, backup)
            tmp = machine + ".install_env.tmp"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(new_text)
            shutil.copymode(machine, tmp)
            os.replace(tmp, machine)
            print("wrote %s (backup %s)" % (machine, backup))
            status = "changed"
    emit(ME_STATUS=status, ME_CHANGES=" ".join(updates), ME_BACKUP=backup, ME_MANUAL=manual, **planned)
    return 0


def cmd_run_env(args):
    """Run a command with the machine config's env (then each --set KEY=VALUE) on top of the current environment."""
    env = dict(os.environ)
    if args.machine and os.path.isfile(args.machine):
        env.update({k: v for k, v in read_env(args.machine).items() if v is not None})
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            print("run-env: --set needs KEY=VALUE, got %s" % item, file=sys.stderr)
            return 2
        env[key] = value
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("run-env: no command", file=sys.stderr)
        return 2
    os.execvpe(command[0], command, env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True
    sub.add_parser("probe").set_defaults(fn=cmd_probe)
    p = sub.add_parser("nvcc")
    p.add_argument("--torch-cuda", default="")
    p.add_argument("--pip-nvcc", default="")
    p.add_argument("--machine", default="")
    p.set_defaults(fn=cmd_nvcc)
    p = sub.add_parser("gpus")
    p.add_argument("--machine", default="")
    p.add_argument("--min-free-mib", type=int, default=2048)
    p.set_defaults(fn=cmd_gpus)
    p = sub.add_parser("imports")
    p.add_argument("--names", required=True, help="comma separated: " + ",".join(IMPORTS))
    p.set_defaults(fn=cmd_imports)
    sub.add_parser("gpu-test").set_defaults(fn=cmd_gpu_test)
    p = sub.add_parser("machine-env")
    p.add_argument("--machine", required=True)
    p.add_argument("--cuda-home", default="", help="the best nvcc's CUDA home (the nvcc subcommand)")
    p.add_argument("--torch-cuda", default="")
    p.add_argument("--venv-bin", required=True)
    p.add_argument("--tilelang-cache", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--expect", action="append", metavar="KEY=VALUE",
                   help="refuse when the plan differs (the values the GPU test ran with)")
    p.set_defaults(fn=cmd_machine_env)
    p = sub.add_parser("run-env")
    p.add_argument("--machine", default="")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override after the machine config's env")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run_env)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
