# Put everything onejev-harness writes below one directory (a big disk), then build and use the environment:
#
#   cp examples/env.example.sh ~/ojh-env.sh      # or anywhere; edit OJH_HOME below
#   source ~/ojh-env.sh
#   bash setup.sh                                # uv, Python 3.11, .venv, pinned packages, OneJev
#   <repo>/.venv/bin/ojh doctor                  # read-only health check
#
# Source it in every shell that runs ojh (or add the `source` line to your own shell profile yourself).
# Already-set variables are kept, so you can override any one of them before sourcing.

export OJH_HOME="${OJH_HOME:-/big/disk/ojh}"                      # the one root: change this line

export OJH_TOOLS_DIR="${OJH_TOOLS_DIR:-$OJH_HOME/tools}"          # uv itself (setup.sh installs it here if missing)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$OJH_HOME/cache/uv}"         # uv's download and wheel cache (several GB)
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$OJH_HOME/python}"  # uv-managed Python 3.11
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-$OJH_HOME/tools/bin}"       # where uv may link python shims (not ~/.local/bin)
export OJH_VENV="${OJH_VENV:-$OJH_HOME/venv}"                     # the venv; setup.sh links <repo>/.venv to it
export HF_HOME="${HF_HOME:-$OJH_HOME/cache/huggingface}"          # Hugging Face hub cache and token
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OJH_HOME/cache/triton}"      # kernels compiled by flash-linear-attention
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OJH_HOME/cache}"        # other tools' caches (instead of ~/.cache)
export TMPDIR="${TMPDIR:-$OJH_HOME/tmp}"                          # temporary files of uv, pip builds and Python
export OJH_OUT_ROOT="${OJH_OUT_ROOT:-$OJH_HOME/runs}"             # ojh run outputs and the registry (used when there is no machine config)
export OJH_MODEL_DIR="${OJH_MODEL_DIR:-$OJH_HOME/models}"         # base weights from `ojh fetch-base` (elephant-1 ~18 GB, joey-1 ~5 GB)

mkdir -p "$TMPDIR"                                                # tools fail when TMPDIR does not exist
