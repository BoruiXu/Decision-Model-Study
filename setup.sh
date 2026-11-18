#!/usr/bin/env bash
# Build the training environment with uv, end to end, writing only below the directories printed first.
#
# Usage: bash setup.sh [--dry-run] [--python <path>] [--machine <machine.yaml>]
#   --dry-run          print the plan (directories, steps, what is already done) and change nothing
#   --python <path>    opt in to an existing Python >= 3.11 instead of a uv-managed 3.11
#   --machine <file>   passed to `ojh setup` (default: $OJH_MACHINE, ~/.config/onejev-harness/machine.yaml,
#                      else inferred)
#
# Steps (each skipped when already done, so a re-run is cheap):
#   1. uv: use `uv` from PATH, else <tools>/uv/uv, else install it there with UV_UNMANAGED_INSTALL
#      (the installer then touches no shell profile and nothing in ~/.local/bin)
#   2. Python 3.11 managed by uv, in $UV_PYTHON_INSTALL_DIR (no python shims are linked anywhere)
#   3. the venv (<repo>/.venv; with $OJH_VENV elsewhere, <repo>/.venv is a symlink to it)
#   4. packages: torch/torchvision from the CUDA 12.8 wheel index, then the repo with the pinned GPU stack
#      (constraints.txt), then `uv pip check`
#   5. `ojh setup`: fetch OneJev at the pinned revision, apply serve.patch, verify every sha256
#
# Directories (environment variables; defaults in brackets; `source examples/env.example.sh` sets them all
# below one root):
#   OJH_TOOLS_DIR [<repo>/.tools]  UV_CACHE_DIR [<tools>/cache/uv]  UV_PYTHON_INSTALL_DIR [<tools>/python]
#   OJH_VENV [<repo>/.venv]  HF_HOME [<tools>/hf]  TMPDIR [<tools>/tmp]
#   OJH_OUT_ROOT [~/ojh_runs] and OJH_MODEL_DIR [~/.cache/onejev-harness/models] are only printed here;
#   ojh uses them later (runs and base weights).
set -euo pipefail

# ---- pins: the CUDA 12.8 wheels of torch (the one place this index URL lives) ----
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
TORCH_PKGS=(torch==2.8.0 torchvision==0.23.0)
PYTHON_VERSION="3.11"
UV_INSTALLER_URL="https://astral.sh/uv${OJH_UV_VERSION:+/$OJH_UV_VERSION}/install.sh"

DRY=0
PYTHON_OPT=""
MACHINE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --python) PYTHON_OPT="${2:?--python needs a path}"; shift ;;
    --machine) MACHINE="$(cd "$(dirname "${2:?--machine needs a file}")" && pwd)/$(basename "$2")"; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }

OJH_TOOLS_DIR="$(abspath "${OJH_TOOLS_DIR:-$REPO/.tools}")"
export UV_CACHE_DIR="$(abspath "${UV_CACHE_DIR:-$OJH_TOOLS_DIR/cache/uv}")"
export UV_PYTHON_INSTALL_DIR="$(abspath "${UV_PYTHON_INSTALL_DIR:-$OJH_TOOLS_DIR/python}")"
export UV_PYTHON_BIN_DIR="$(abspath "${UV_PYTHON_BIN_DIR:-$OJH_TOOLS_DIR/bin}")"  # in case a uv links shims
export UV_PYTHON_INSTALL_BIN=0                       # newer uv: do not link python shims at all
export UV_NO_MODIFY_PATH=1                           # the uv installer: never edit shell profiles
export HF_HOME="$(abspath "${HF_HOME:-$OJH_TOOLS_DIR/hf}")"
export TMPDIR="$(abspath "${TMPDIR:-$OJH_TOOLS_DIR/tmp}")"
VENV="$(abspath "${OJH_VENV:-$REPO/.venv}")"
OUT_ROOT="${OJH_OUT_ROOT:-$HOME/ojh_runs}"
MODEL_DIR="${OJH_MODEL_DIR:-$HOME/.cache/onejev-harness/models}"
UV_HOME="$OJH_TOOLS_DIR/uv"
STAMP="$VENV/.ojh-packages"
PKG_KEY="$(cat constraints.txt pyproject.toml | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-64) ${TORCH_INDEX_URL} ${TORCH_PKGS[*]}"

free_of() {  # free space of the nearest existing parent
  local d="$1"
  while [ ! -e "$d" ]; do d="$(dirname "$d")"; done
  df -Pk "$d" | awk 'NR==2 {printf "%.1f GiB free", $4/1048576}'
}
say() { printf '%s\n' "$*"; }
run() { if [ "$DRY" = 1 ]; then say "    would run: $*"; else "$@"; fi; }

say "onejev-harness setup ($([ "$DRY" = 1 ] && echo 'dry run: nothing is changed' || echo 'installing'))"
say "  repo                   $REPO"
for pair in "tools (OJH_TOOLS_DIR)|$OJH_TOOLS_DIR" "uv cache (UV_CACHE_DIR)|$UV_CACHE_DIR" \
            "python (UV_PYTHON_INSTALL_DIR)|$UV_PYTHON_INSTALL_DIR" "venv (OJH_VENV)|$VENV" \
            "HF cache (HF_HOME)|$HF_HOME" "temp (TMPDIR)|$TMPDIR" "runs (OJH_OUT_ROOT)|$OUT_ROOT" \
            "base weights (OJH_MODEL_DIR)|$MODEL_DIR"; do
  printf '  %-30s %s  [%s]\n' "${pair%%|*}" "${pair#*|}" "$(free_of "${pair#*|}")"
done
say "  torch                  ${TORCH_PKGS[*]} from $TORCH_INDEX_URL"
case "$OUT_ROOT/" in "$REPO"/*) say "error: OJH_OUT_ROOT is inside the repo; choose another directory" >&2; exit 1 ;; esac
case "$MODEL_DIR/" in "$REPO"/*) say "error: OJH_MODEL_DIR is inside the repo; choose another directory" >&2; exit 1 ;; esac

[ "$DRY" = 1 ] || mkdir -p "$OJH_TOOLS_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$TMPDIR"

# 1. uv
say "[1/5] uv"
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"; say "    found on PATH: $UV"
elif [ -x "$UV_HOME/uv" ]; then
  UV="$UV_HOME/uv"; say "    already installed: $UV"
else
  UV="$UV_HOME/uv"
  say "    installing into $UV_HOME (UV_UNMANAGED_INSTALL: no shell profile or ~/.local/bin changes)"
  if [ "$DRY" = 1 ]; then
    say "    would run: curl -LsSf $UV_INSTALLER_URL | UV_UNMANAGED_INSTALL=$UV_HOME sh"
  else
    command -v curl >/dev/null || { echo "error: curl is needed to install uv" >&2; exit 1; }
    curl -LsSf "$UV_INSTALLER_URL" | env UV_UNMANAGED_INSTALL="$UV_HOME" INSTALLER_NO_MODIFY_PATH=1 sh
  fi
fi
if [ -x "$UV" ]; then say "    $("$UV" --version)"; fi

# 2. Python
say "[2/5] Python $PYTHON_VERSION"
if [ -n "$PYTHON_OPT" ]; then
  PY="$PYTHON_OPT"; say "    using --python $PY (opt-in)"
  "$PY" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' \
    || { echo "error: --python $PY is not Python >= 3.11" >&2; exit 1; }
elif [ -x "$UV" ] && PY="$("$UV" python find --managed-python --system "$PYTHON_VERSION" 2>/dev/null)"; then
  say "    already installed: $PY"
else
  PY="$PYTHON_VERSION"
  run "$UV" python install "$PYTHON_VERSION"
fi

# 3. venv
say "[3/5] venv $VENV"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 11))'; then
  say "    already there: $("$VENV/bin/python" --version)"
else
  run "$UV" venv --python "$PY" "$VENV"
fi
if [ "$VENV" != "$REPO/.venv" ]; then
  if [ -L "$REPO/.venv" ] || [ ! -e "$REPO/.venv" ]; then
    run ln -sfn "$VENV" "$REPO/.venv"; say "    <repo>/.venv -> $VENV"
  else
    echo "error: $REPO/.venv exists and is not a link to OJH_VENV=$VENV; move it away first" >&2; exit 1
  fi
fi

# 4. packages
say "[4/5] packages (constraints.txt + ${TORCH_PKGS[*]})"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$PKG_KEY" ]; then
  say "    already installed for this constraints.txt / pyproject.toml / torch pin"
else
  run "$UV" pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX_URL" "${TORCH_PKGS[@]}"
  run "$UV" pip install --python "$VENV/bin/python" -c constraints.txt -e ".[gpu]"
  run "$UV" pip check --python "$VENV/bin/python"
  [ "$DRY" = 1 ] || printf '%s' "$PKG_KEY" > "$STAMP"
fi

# 5. OneJev
say "[5/5] OneJev (pinned revision + serve.patch, sha256 verified)"
run "$VENV/bin/ojh" setup ${MACHINE:+--machine "$MACHINE"}

say "done. next: $REPO/.venv/bin/ojh doctor"
