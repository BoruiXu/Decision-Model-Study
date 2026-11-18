#!/usr/bin/env bash
# Usage: ./setup.sh [<machine.yaml>]
# Creates .venv from the machine config's `python` (without a machine file: $OJH_SETUP_PYTHON, else python3.11,
# else python3), installs the pinned GPU stack, then runs `ojh setup`.
set -euo pipefail
[ $# -le 1 ] || { echo "usage: $0 [<machine.yaml>]" >&2; exit 2; }
MACHINE="${1:+$(cd "$(dirname "$1")" && pwd)/$(basename "$1")}"
cd "$(dirname "$0")"
if [ -n "$MACHINE" ]; then
  PYTHON_BIN="$(sed -n 's/^python:[[:space:]]*//p' "$MACHINE" | head -n1 | tr -d "\"'")"
  [ -n "$PYTHON_BIN" ] || { echo "no 'python:' entry in $MACHINE" >&2; exit 1; }
else
  PYTHON_BIN="${OJH_SETUP_PYTHON:-$(command -v python3.11 || command -v python3)}"
fi
"$PYTHON_BIN" -m venv .venv
# torch comes from the cu128 index, which pyproject cannot express.
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -c constraints.txt -e ".[gpu]"
.venv/bin/python -m pip check
if [ -n "$MACHINE" ]; then
  .venv/bin/ojh setup --machine "$MACHINE"
else
  .venv/bin/ojh setup   # machine config: $OJH_MACHINE, ~/.config/onejev-harness/machine.yaml, else inferred
fi
