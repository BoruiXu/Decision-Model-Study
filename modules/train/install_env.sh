#!/usr/bin/env bash
# Install every library a training run needs into the onejev-harness (ojh) venv, then check the environment end to
# end. Safe to rerun: on a ready machine it changes nothing. The train agent runs it, without asking, when a run
# fails on a missing or incompatible library (SKILL.md, 常见错误).
#
# Usage: install_env.sh --ojh <onejev-harness repo> --machine <machine.yaml> [--env-file <ojh-env.sh>]
#                       [--cache-root DIR] [--check] [--no-gpu-test]
#   --ojh          the onejev-harness repo; the venv is <ojh>/.venv
#   --machine      ojh's machine config; on Hopper its env gets CUDA_HOME, PATH and TILELANG_CACHE_DIR
#   --env-file     ojh's environment file (the task's resources.onejev.env_file), sourced first
#   --cache-root   pip, uv, TMPDIR, triton and tilelang caches and the logs (default <dir of machine.yaml>/cache)
#   --check        install and edit nothing (no GPU test, no log): print what is missing and what would be done
#   --no-gpu-test  skip the GPU forward + backward test
#
# Steps (each prints ok / missing / failed lines; skip and warn are informational):
#   1. base venv: <ojh>/setup.sh when <ojh>/.venv is missing. An existing venv is never rebuilt, and torch, triton,
#      transformers and fla keep their versions: every install pins their installed versions too.
#   2. pyyaml and pytest
#   3. causal-conv1d 1.7.0: the release wheel for torch major.minor, CUDA major, C++ ABI and Python tag; a source
#      build when that fails and nvcc's CUDA major is torch's; else a warning (training then takes the slower path)
#   4. tilelang 0.1.15 when a GPU has compute capability >= 9.0, and CUDA_HOME, PATH (nvcc's bin first) and
#      TILELANG_CACHE_DIR in the machine config's env: written only after the GPU test passes with the new values,
#      or, with a warning, when the test cannot run (--no-gpu-test, no usable GPU); backup machine.yaml.bak-<time>
#      first; left alone when right
#   5. checks: imports; a tiny forward + backward through fla's chunk_gated_delta_rule (joey-1's op) and
#      causal_conv1d_fn on a free GPU, run with the machine config's env; ojh doctor --no-network
# Installs use <ojh>/constraints.txt; uv (PATH, $OJH_TOOLS_DIR/uv/uv, <ojh>/.tools/uv/uv, <dir of machine.yaml>/
# tools/uv/uv), else the venv's pip. Exit 0: every required item ok (--check: and nothing to do); 1: a required
# item failed (--check: or something would be done); 2: usage. causal-conv1d and pytest are not required.
set -uo pipefail

CAUSAL_CONV1D_VERSION=1.7.0
TILELANG_VERSION=0.1.15
CAUSAL_CONV1D_RELEASES=https://github.com/Dao-AILab/causal-conv1d/releases/download
PROTECTED=(torch torchvision triton transformers flash-linear-attention fla-core)
HOPPER_CAP=9.0
GPU_TEST_TIMEOUT_S=1800
MARK='#INSTALL_ENV_VARS'

OJH="" MACHINE="" ENV_FILE="" CACHE_ROOT="" CHECK=0 GPU_TEST=1
while [ $# -gt 0 ]; do
  case "$1" in
    --ojh) OJH="${2:?--ojh needs a directory}"; shift ;;
    --machine) MACHINE="${2:?--machine needs a file}"; shift ;;
    --env-file) ENV_FILE="${2:?--env-file needs a file}"; shift ;;
    --cache-root) CACHE_ROOT="${2:?--cache-root needs a directory}"; shift ;;
    --check) CHECK=1 ;;
    --no-gpu-test) GPU_TEST=0 ;;
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$OJH" ] && [ -n "$MACHINE" ] || { echo "usage: install_env.sh --ojh <repo> --machine <machine.yaml> (see --help)" >&2; exit 2; }
[ -d "$OJH" ] || { echo "error: --ojh $OJH is not a directory" >&2; exit 2; }
[ -f "$MACHINE" ] || { echo "error: --machine $MACHINE is not a file" >&2; exit 2; }
[ -z "$ENV_FILE" ] || [ -f "$ENV_FILE" ] || { echo "error: --env-file $ENV_FILE is not a file" >&2; exit 2; }
OJH="$(cd "$OJH" && pwd)"
MACHINE="$(cd "$(dirname "$MACHINE")" && pwd)/$(basename "$MACHINE")"
case "${CACHE_ROOT:=$(dirname "$MACHINE")/cache}" in /*) ;; *) CACHE_ROOT="$PWD/$CACHE_ROOT" ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$HERE/install_env.py"
VENV_PY="$OJH/.venv/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"

if [ -n "$ENV_FILE" ]; then set +u; source "$ENV_FILE"; set -u; fi
export PIP_CACHE_DIR="$CACHE_ROOT/pip" UV_CACHE_DIR="$CACHE_ROOT/uv" TRITON_CACHE_DIR="$CACHE_ROOT/triton" \
       TILELANG_CACHE_DIR="$CACHE_ROOT/tilelang" XDG_CACHE_HOME="$CACHE_ROOT/xdg"
LOG=""
if [ "$CHECK" = 0 ]; then
  mkdir -p "$CACHE_ROOT"/{pip,uv,tmp,triton,tilelang,xdg,logs} || { echo "error: cannot create $CACHE_ROOT" >&2; exit 1; }
  export TMPDIR="$CACHE_ROOT/tmp"
  LOG="$CACHE_ROOT/logs/install_env-$STAMP.log"
  exec > >(tee -a "$LOG") 2>&1
elif [ -d "$CACHE_ROOT/tmp" ]; then
  export TMPDIR="$CACHE_ROOT/tmp"
fi

FAILED=() WARNED=() CHANGED=() PLANNED=()
say() { printf '%s\n' "$*"; }
line() { printf '  %-8s %s\n' "$1" "$2"; }
ok() { line ok "$*"; }
skip() { line skip "$*"; }
missing() { line missing "$*"; }
would() { line would "$*"; PLANNED+=("$*"); }
failed() { line failed "$*"; FAILED+=("$*"); }
warn() { line warn "$*"; WARNED+=("$*"); }
changed() { line done "$*"; CHANGED+=("$*"); }
indent() { sed 's/^/           /'; }
upname() { printf '%s' "$1" | tr 'a-z' 'A-Z' | tr -c 'A-Z0-9' '_'; }   # torch-x.y -> TORCH_X_Y
ver() { local v; v="V_$(upname "$1")"; printf '%s' "${!v:-}"; }
cap_ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }

run_helper() {  # run_helper <command...>: print the helper's human lines indented, eval its variables
  local out rc
  out="$("$@")"; rc=$?
  printf '%s\n' "$out" | sed "/^$MARK\$/,\$d" | sed '/^$/d' | indent
  eval "$(printf '%s\n' "$out" | sed -n "/^$MARK\$/,\$p" | sed 1d | grep -E '^[A-Z_][A-Z0-9_]*=')"
  return $rc
}

finish() {
  say ""
  say "summary"
  if [ ${#CHANGED[@]} -eq 0 ]; then say "  changes: none"; else say "  changes:"; printf '    - %s\n' "${CHANGED[@]}"; fi
  if [ "$CHECK" = 1 ]; then
    if [ ${#PLANNED[@]} -eq 0 ]; then say "  would do: nothing"; else say "  would do:"; printf '    - %s\n' "${PLANNED[@]}"; fi
  fi
  [ ${#WARNED[@]} -eq 0 ] || { say "  warnings:"; printf '    - %s\n' "${WARNED[@]}"; }
  [ ${#FAILED[@]} -eq 0 ] || { say "  failed:"; printf '    - %s\n' "${FAILED[@]}"; }
  [ -z "$LOG" ] || say "  log: $LOG"
  local rc=0
  if [ ${#FAILED[@]} -gt 0 ] || { [ "$CHECK" = 1 ] && [ ${#PLANNED[@]} -gt 0 ]; }; then rc=1; fi
  if [ "$CHECK" = 1 ]; then
    say "result: $([ $rc = 0 ] && echo 'ready: nothing to do' || echo 'not ready (see would do / failed)')"
  else
    say "result: $([ $rc = 0 ] && echo ok || echo failed)"
  fi
  exit $rc
}

say "install_env.sh $(date '+%Y-%m-%d %H:%M:%S') ($([ "$CHECK" = 1 ] && echo 'check: nothing is installed or edited' || echo install))"
say "  ojh         $OJH"
say "  machine     $MACHINE"
say "  env file    ${ENV_FILE:-(none)}"
say "  cache root  $CACHE_ROOT"
say "  log         ${LOG:-(none with --check)}"

# ---- 1. base venv ----
say "[1/5] base venv ($OJH/.venv)"
if [ -x "$VENV_PY" ] && "$VENV_PY" -c 'import sys' >/dev/null 2>&1; then
  ok "venv exists ($("$VENV_PY" --version 2>&1)); never rebuilt here"
elif [ -e "$OJH/.venv" ] || [ -L "$OJH/.venv" ]; then
  failed "$OJH/.venv exists but its python does not run; not rebuilt (move it away, then rerun)"
  finish
elif [ ! -f "$OJH/setup.sh" ]; then
  failed "no venv and no $OJH/setup.sh to build it"
  finish
elif [ "$CHECK" = 1 ]; then
  missing "venv $OJH/.venv"
  would "build the venv: bash $OJH/setup.sh --machine $MACHINE (UV_CACHE_DIR, TMPDIR under $CACHE_ROOT)"
  say "  (the other steps are planned once the venv exists)"
  finish
else
  missing "venv $OJH/.venv: running setup.sh (the repo's README: uv, Python 3.11, pinned packages, OneJev)"
  say "  \$ bash $OJH/setup.sh --machine $MACHINE"
  (cd "$OJH" && bash setup.sh --machine "$MACHINE") 2>&1 | sed 's/^/    | /'
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ] || [ ! -x "$VENV_PY" ]; then failed "setup.sh exit $rc"; finish; fi
  changed "built $OJH/.venv with setup.sh"
fi

run_helper "$VENV_PY" "$HELPER" probe || true
if [ -z "${P_PY_TAG:-}" ]; then failed "cannot probe the venv ($HELPER probe)"; finish; fi
for d in "${PROTECTED[@]}"; do eval "ORIG_V_$(upname "$d")=\$(ver $d)"; done
base_ok=1
for d in torch triton transformers flash-linear-attention; do
  if [ -z "$(ver "$d")" ]; then
    base_ok=0
    failed "the venv has no $d: rerun $OJH/setup.sh by hand (an existing venv is not rebuilt here)"
  fi
done
if [ -n "${P_TORCH_ERROR:-}" ]; then base_ok=0; failed "import torch: $P_TORCH_ERROR"; fi
[ "$base_ok" = 0 ] || ok "Python $P_PY_VERSION ($P_PY_TAG, $P_PLATFORM); torch $P_TORCH (CUDA ${P_TORCH_CUDA:-none}, cxx11abi $P_CXX11ABI); triton $(ver triton); transformers $(ver transformers); flash-linear-attention $(ver flash-linear-attention)"

UV=""
for c in "$(command -v uv 2>/dev/null)" "${OJH_TOOLS_DIR:+$OJH_TOOLS_DIR/uv/uv}" "$OJH/.tools/uv/uv" \
         "$(dirname "$MACHINE")/tools/uv/uv"; do
  if [ -n "$c" ] && [ -x "$c" ]; then UV="$c"; break; fi
done
if [ -n "$UV" ]; then
  INSTALL=("$UV" pip install --python "$VENV_PY"); INSTALLER="uv $("$UV" --version 2>/dev/null | awk '{print $2}') ($UV)"
elif [ -n "$(ver pip)" ]; then
  INSTALL=("$VENV_PY" -m pip install --disable-pip-version-check --no-input); INSTALLER="pip $(ver pip) (the venv's)"
else
  INSTALL=(); INSTALLER="none: no uv found and the venv has no pip"
fi
say "  installer   $INSTALLER"
REPO_CONSTRAINTS=()
[ -f "$OJH/constraints.txt" ] && REPO_CONSTRAINTS=(-c "$OJH/constraints.txt")
PROTECT="$CACHE_ROOT/tmp/install_env-protect-$STAMP-$$.txt"
if [ "$CHECK" = 0 ]; then
  for d in "${PROTECTED[@]}"; do [ -z "$(ver "$d")" ] || printf '%s==%s\n' "$d" "$(ver "$d")"; done > "$PROTECT"
  trap 'rm -f "$PROTECT"' EXIT
fi

install() {  # install <requirement or pip option>...; NO_REPO_CONSTRAINTS=1 leaves out constraints.txt
  if [ ${#INSTALL[@]} -eq 0 ]; then say "    no installer: no uv found and the venv has no pip"; return 1; fi
  local cons=(-c "$PROTECT")
  [ "${NO_REPO_CONSTRAINTS:-0}" = 1 ] || cons+=(${REPO_CONSTRAINTS[@]+"${REPO_CONSTRAINTS[@]}"})
  say "  \$ ${INSTALL[*]} ${cons[*]} $*"
  "${INSTALL[@]}" "${cons[@]}" "$@" 2>&1 | sed 's/^/    | /'
  local rc=${PIPESTATUS[0]}
  run_helper "$VENV_PY" "$HELPER" probe || true
  local d o before after
  for d in "${PROTECTED[@]}"; do
    o="ORIG_V_$(upname "$d")"; before="${!o:-}"; after="$(ver "$d")"
    if [ "$before" != "$after" ]; then failed "$d changed from ${before:-none} to ${after:-none} by: $*"; rc=1; fi
  done
  return "$rc"
}

run_helper "$VENV_PY" "$HELPER" nvcc --torch-cuda "${P_TORCH_CUDA:-}" --pip-nvcc "${P_NVCC_PIP:-}" --machine "$MACHINE" || true
if [ -n "${NVCC_HOME:-}" ]; then
  say "  nvcc        $NVCC_HOME (CUDA $NVCC_VERSION, from $NVCC_SOURCE; torch's CUDA is ${P_TORCH_CUDA:-none})"
else
  say "  nvcc        none found (CUDA_HOME, /usr/local/cuda*, the venv's nvidia-cuda-nvcc, PATH)"
fi

# ---- 2. pyyaml, pytest ----
say "[2/5] pyyaml, pytest"
for d in pyyaml pytest; do
  why="trsi needs it"; [ "$d" = pytest ] && why="trsi's tests need it"
  if [ -n "$(ver "$d")" ]; then ok "$d $(ver "$d")"; continue; fi
  if [ "$CHECK" = 1 ]; then missing "$d ($why)"; would "install $d (with constraints.txt)"; continue; fi
  missing "$d ($why): installing"
  if install "$d" || NO_REPO_CONSTRAINTS=1 install "$d"; then
    changed "installed $d $(ver "$d")"
  elif [ "$d" = pyyaml ]; then failed "pyyaml not installed"
  else warn "pytest not installed (only trsi's tests need it)"; fi
done

# ---- 3. causal-conv1d ----
say "[3/5] causal-conv1d $CAUSAL_CONV1D_VERSION"
wheel_name() {  # the release wheel for this venv's torch, CUDA, C++ ABI, Python and platform
  local tmm cu
  [[ "${P_TORCH:-}" =~ ^([0-9]+)\.([0-9]+) ]] || return 1; tmm="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  [[ "${P_TORCH_CUDA:-}" =~ ^([0-9]+)\. ]] || return 1; cu="${BASH_REMATCH[1]}"
  [ -n "${P_CXX11ABI:-}" ] || return 1
  case "${P_PLATFORM:-}" in linux-*) ;; *) return 1 ;; esac
  printf 'causal_conv1d-%s+cu%storch%scxx11abi%s-%s-%s-%s.whl' "$CAUSAL_CONV1D_VERSION" "$cu" "$tmm" "$P_CXX11ABI" \
    "$P_PY_TAG" "$P_PY_TAG" "${P_PLATFORM//[-.]/_}"
}
cc="$(ver causal-conv1d)"
if [ -n "$cc" ]; then
  if [ "${cc%%+*}" = "$CAUSAL_CONV1D_VERSION" ]; then ok "causal-conv1d $cc"
  else warn "causal-conv1d $cc is installed, not $CAUSAL_CONV1D_VERSION: left as it is"; fi
else
  WHEEL="$(wheel_name)" || WHEEL=""
  URL="${WHEEL:+$CAUSAL_CONV1D_RELEASES/v$CAUSAL_CONV1D_VERSION/${WHEEL//+/%2B}}"
  fallback="" no_fallback="no nvcc for a source build"
  if [ -n "${NVCC_HOME:-}" ]; then
    if [ "${NVCC_VERSION%%.*}" = "${P_TORCH_CUDA%%.*}" ]; then fallback="source build with $NVCC_HOME/bin/nvcc"
    else no_fallback="no source build: nvcc $NVCC_VERSION does not match torch's CUDA ${P_TORCH_CUDA:-?} major"; fi
  fi
  if [ "$CHECK" = 1 ]; then
    missing "causal-conv1d (optional: without it training takes the slower PyTorch path)"
    if [ -n "$WHEEL" ]; then would "install causal-conv1d $CAUSAL_CONV1D_VERSION from the prebuilt wheel $WHEEL ($URL); if that fails: ${fallback:-nothing ($no_fallback)}"
    elif [ -n "$fallback" ]; then would "install causal-conv1d $CAUSAL_CONV1D_VERSION by a $fallback (no prebuilt wheel for torch ${P_TORCH:-?}, CUDA ${P_TORCH_CUDA:-?}, $P_PY_TAG, $P_PLATFORM)"
    else warn "causal-conv1d: no prebuilt wheel for this venv, and $no_fallback"; fi
  else
    missing "causal-conv1d: installing"
    cc_ok=0
    if [ -n "$WHEEL" ]; then
      say "  wheel $WHEEL (torch $P_TORCH, CUDA $P_TORCH_CUDA, cxx11abi $P_CXX11ABI, $P_PY_TAG)"
      install "causal-conv1d @ $URL" && cc_ok=1
    fi
    if [ "$cc_ok" = 0 ] && [ -n "$fallback" ]; then
      say "  falling back to a $fallback (can take 10+ minutes)"
      saved_path="$PATH" saved_home="${CUDA_HOME-}" had_home="${CUDA_HOME+x}"
      if install setuptools wheel ninja packaging; then
        export CUDA_HOME="$NVCC_HOME" PATH="$NVCC_HOME/bin:$PATH" CAUSAL_CONV1D_FORCE_BUILD=TRUE
        install --no-build-isolation "causal-conv1d==$CAUSAL_CONV1D_VERSION" && cc_ok=1
        export PATH="$saved_path"; unset CAUSAL_CONV1D_FORCE_BUILD
        if [ -n "$had_home" ]; then export CUDA_HOME="$saved_home"; else unset CUDA_HOME; fi
      fi
    fi
    if [ "$cc_ok" = 1 ] && [ -n "$(ver causal-conv1d)" ]; then changed "installed causal-conv1d $(ver causal-conv1d)"
    else warn "causal-conv1d not installed (wheel: ${WHEEL:-none}; source build: ${fallback:-$no_fallback}): training works, through the slower PyTorch path"; fi
  fi
fi

# ---- 4. tilelang on Hopper ----
say "[4/5] tilelang $TILELANG_VERSION (needed on GPUs with compute capability >= $HOPPER_CAP)"
run_helper "$VENV_PY" "$HELPER" gpus --machine "$MACHINE" || true
CVD="${CUDA_VISIBLE_DEVICES-}"
GPU_DESC="CUDA_VISIBLE_DEVICES=$CVD (set by the caller)"
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then CVD="${GPU_PICK_UUID:-}"; GPU_DESC="${GPU_PICK_DESC:-}"; fi
with_gpu() { if [ -n "$CVD" ]; then env CUDA_VISIBLE_DEVICES="$CVD" "$@"; else "$@"; fi; }
gpu_skip_reason() {
  if [ "$CHECK" = 1 ]; then echo "not run with --check"
  elif [ "$GPU_TEST" = 0 ]; then echo "--no-gpu-test"
  elif [ -z "$CVD" ]; then echo "no GPU with 2 GiB free (nvidia-smi)"; fi
}
GPU_TESTED=0
gpu_test() {  # gpu_test [--set KEY=VALUE]...: run and report the GPU test through the machine config's env
  GT_FLA="" GT_FLA_ERR="" GT_CC="" GT_CC_ERR="" GPU_TESTED=1
  local what="the machine config's env" tmo
  tmo=()
  [ $# -eq 0 ] || what="the machine config's env with the planned values"
  command -v timeout >/dev/null && tmo=(timeout "$GPU_TEST_TIMEOUT_S")
  say "  GPU test on $GPU_DESC through $what (first run compiles kernels)"
  run_helper with_gpu ${tmo[@]+"${tmo[@]}"} "$VENV_PY" "$HELPER" run-env --machine "$MACHINE" "$@" -- \
    "$VENV_PY" "$HELPER" gpu-test || true
  if [ "${GT_FLA:-}" = ok ]; then ok "GPU: fla chunk_gated_delta_rule forward + backward"
  else failed "GPU: fla chunk_gated_delta_rule forward + backward: ${GT_FLA_ERR:-no result (crash or timeout; see the log)}"; fi
  case "${GT_CC:-}" in
    ok) ok "GPU: causal_conv1d_fn forward + backward" ;;
    failed) warn "GPU: causal_conv1d_fn forward + backward: ${GT_CC_ERR:-}" ;;
    *) skip "GPU: causal_conv1d_fn (not installed)" ;;
  esac
}

NEED_TILELANG=0
if [ "${GPU_COUNT:-0}" = 0 ]; then
  skip "no GPU found by nvidia-smi: tilelang not checked"
elif cap_ge "$GPU_MAX_CAP" "$HOPPER_CAP"; then
  NEED_TILELANG=1
else
  skip "tilelang not needed: compute capability ${GPU_CAPS} (not Hopper)"
fi
if [ "$NEED_TILELANG" = 1 ]; then
  tl="$(ver tilelang)"
  if [ -n "$tl" ]; then
    if [ "$tl" = "$TILELANG_VERSION" ]; then ok "tilelang $tl"; else warn "tilelang $tl is installed, not $TILELANG_VERSION: left as it is"; fi
  elif [ "$CHECK" = 1 ]; then
    missing "tilelang (fla's backward on Hopper with Triton < 3.7.1 needs it)"
    would "install tilelang==$TILELANG_VERSION (with constraints.txt)"
  else
    missing "tilelang: installing"
    if install "tilelang==$TILELANG_VERSION"; then changed "installed tilelang $(ver tilelang)"
    else failed "tilelang $TILELANG_VERSION not installed"; fi
  fi
  machine_env() {
    run_helper "$VENV_PY" "$HELPER" machine-env --machine "$MACHINE" --cuda-home "${NVCC_HOME:-}" \
      --torch-cuda "${P_TORCH_CUDA:-}" --venv-bin "$OJH/.venv/bin" --tilelang-cache "$CACHE_ROOT/tilelang" "$@" || true
  }
  machine_env
  case "${ME_STATUS:-failed}" in
    right) ok "machine config env: CUDA_HOME ($ME_CUDA_HOME), PATH and TILELANG_CACHE_DIR already right" ;;
    plan)
      missing "machine config env: $ME_CHANGES"
      if [ "$CHECK" = 1 ]; then
        would "set $ME_CHANGES in the env of $MACHINE once a GPU test passes with the new values (backup first)"
      else
        reason="$(gpu_skip_reason)"
        if [ -z "$reason" ]; then
          gpu_test --set "CUDA_HOME=$ME_CUDA_HOME" --set "PATH=$ME_PATH" --set "TILELANG_CACHE_DIR=$ME_TILELANG"
        else
          warn "the new machine env is written without a GPU test ($reason)"
        fi
        if [ -n "$reason" ] || [ "${GT_FLA:-}" = ok ]; then
          machine_env --apply --expect "CUDA_HOME=$ME_CUDA_HOME" --expect "PATH=$ME_PATH" \
            --expect "TILELANG_CACHE_DIR=$ME_TILELANG"   # exactly the values the GPU test ran with
          if [ "${ME_STATUS:-}" = changed ]; then changed "set $ME_CHANGES in the env of $MACHINE (backup $ME_BACKUP)"
          else failed "machine config env not written (see above); set by hand: ${ME_MANUAL:-?}"; fi
        else
          say "  the GPU test failed with the planned values: $MACHINE left unchanged"
        fi
      fi ;;
    no_nvcc) failed "no nvcc for tilelang's JIT: install a CUDA toolkit (or nvidia-cuda-nvcc >= 13 into the venv), then rerun" ;;
    manual) failed "machine config env: cannot edit it in place; set by hand: ${ME_MANUAL:-?}" ;;
    *) failed "machine config env not edited (see above)" ;;
  esac
fi

# ---- 5. checks ----
say "[5/5] checks"
names=torch,transformers,fla,causal_conv1d,yaml
[ "$NEED_TILELANG" = 0 ] || names="$names,tilelang"
run_helper with_gpu "$VENV_PY" "$HELPER" run-env --machine "$MACHINE" -- "$VENV_PY" "$HELPER" imports --names "$names" || true
for n in ${names//,/ }; do
  u="$(upname "$n")"; s="IMP_$u" v="IMPV_$u" e="IMPE_$u"
  case "${!s:-failed}" in
    ok) ok "import $n ${!v:-}" ;;
    missing) if [ "$n" = causal_conv1d ]; then line missing "import causal_conv1d (optional: the slower PyTorch path is used)"
             elif [ "$CHECK" = 1 ]; then line missing "import $n"
             else failed "import $n: missing"; fi ;;
    *) if [ "$n" = causal_conv1d ]; then warn "import causal_conv1d failed: ${!e:-} (the slower PyTorch path is used)"
       else failed "import $n: ${!e:-no result}"; fi ;;
  esac
done

reason="$(gpu_skip_reason)"
if [ "$GPU_TESTED" = 1 ]; then skip "GPU forward + backward test: already run in step 4 with the planned values"
elif [ "$CHECK" = 1 ] || [ "$GPU_TEST" = 0 ]; then skip "GPU forward + backward test ($reason)"
elif [ -n "$reason" ]; then warn "GPU forward + backward test not run: $reason"
else gpu_test; fi

if [ -x "$OJH/.venv/bin/ojh" ]; then
  out="$(OJH_MACHINE="$MACHINE" "$OJH/.venv/bin/ojh" doctor --no-network 2>&1)"; rc=$?
  printf '%s\n' "$out" | grep '^\[' | grep -v '^\[ ok \]' | indent
  if [ "$rc" = 0 ]; then ok "ojh doctor: $(printf '%s\n' "$out" | tail -n 1)"
  else failed "ojh doctor exit $rc: $(printf '%s\n' "$out" | grep '^\[fail\]' | head -n 3 | tr '\n' ' ')"; fi
else
  failed "ojh doctor: $OJH/.venv/bin/ojh not found"
fi

finish
