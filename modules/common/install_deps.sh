#!/usr/bin/env bash
# Install the libraries of requirements.txt (numpy, scikit-learn, xgboost, pyyaml: what modules/compare and
# modules/data/dgen import; pyyaml is checked as the module yaml) into the environment of a Python, in practice the
# onejev-harness (ojh) venv, and check that each imports at the version requirements.txt asks for. Safe to rerun: on
# a ready environment it changes nothing.
# modules/train/install_env.sh runs it with the same venv.
#
# Usage: install_deps.sh --python <venv python> [--check] [--constraint <file>]...
#   --python      the interpreter whose environment gets the libraries (e.g. <ojh>/.venv/bin/python)
#   --check       install nothing: print what is missing and what would be done
#   --constraint  a constraints file for the install (-c), repeatable (install_env.sh passes the versions of torch
#                 and the other packages it never moves, and ojh's constraints.txt)
#
# Each requirement prints one line: "ok <name> <version>", "missing <name> (<why>)" or "wrong <name> <version>
# (<what requirements.txt asks>)". The install is `<python> -m pip install -r requirements.txt`; a venv without pip
# (uv builds them so) gets `uv pip install --python <python> -r requirements.txt` instead, with the uv of $UV, PATH
# or $OJH_TOOLS_DIR/uv/uv ($UV, when set, is used even when pip is there). Exit 0: every requirement ok (--check: and
# nothing to do); 1: something missing, wrong or failed (--check: an install would run); 2: usage.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$HERE/requirements.txt"
PY="" CHECK=0 CONS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="${2:?--python needs an interpreter}"; shift ;;
    --check) CHECK=1 ;;
    --constraint) CONS+=("${2:?--constraint needs a file}"); shift ;;
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$PY" ] || { echo "usage: install_deps.sh --python <venv python> [--check] [--constraint <file>]... (see --help)" >&2; exit 2; }
command -v "$PY" >/dev/null 2>&1 || { echo "error: --python $PY is not an executable" >&2; exit 2; }
for c in ${CONS[@]+"${CONS[@]}"}; do [ -f "$c" ] || { echo "error: --constraint $c is not a file" >&2; exit 2; }; done
[ -f "$REQ" ] || { echo "error: $REQ not found" >&2; exit 2; }

# The probe runs in the target interpreter (Python 3.8+, standard library only): one tab-separated line per
# requirement, "ok name version", "missing name why" or "wrong name version why".
read -r -d '' PROBE <<'EOF'
import importlib, re, sys
try:
    from importlib import metadata
except ImportError:
    metadata = None
IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
OPS = {"==": lambda c: c == 0, "!=": lambda c: c != 0, ">=": lambda c: c >= 0, "<=": lambda c: c <= 0,
       ">": lambda c: c > 0, "<": lambda c: c < 0}


def release(v):
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", v)
    return [int(x) for x in m.group(1).split(".")] if m else []


def compare(a, b):
    n = max(len(a), len(b))
    a, b = a + [0] * (n - len(a)), b + [0] * (n - len(b))
    return (a > b) - (a < b)


def satisfied(version, spec):
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        m = re.match(r"(==|!=|>=|<=|>|<)\s*(\S+)$", part)
        if m is None or not OPS[m.group(1)](compare(release(version), release(m.group(2)))):
            return False
    return True


for line in open(sys.argv[1]):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)$", line)
    name, spec = m.group(1), m.group(2).strip()
    mod = IMPORT_NAMES.get(name.lower(), name.lower().replace("-", "_"))
    try:
        module = importlib.import_module(mod)
    except Exception as e:
        why = (str(e).splitlines() or [type(e).__name__])[0]
        print("missing\t%s\timport %s: %s" % (name, mod, why))
        continue
    version = None
    if metadata is not None:
        try:
            version = metadata.version(name)
        except Exception:
            version = None
    version = version or getattr(module, "__version__", None)
    if version is None:
        print("wrong\t%s\t?\tno version found; requirements.txt asks %s" % (name, spec or "any"))
    elif spec and not satisfied(str(version), spec):
        print("wrong\t%s\t%s\trequirements.txt asks %s" % (name, version, spec))
    else:
        print("ok\t%s\t%s" % (name, version))
EOF

NEED=0
probe() {  # print one line per requirement; NEED = how many are not ok; status 1 when the probe itself failed
  local out rc st name a b
  NEED=0
  out="$("$PY" - "$REQ" <<<"$PROBE" 2>&1)"; rc=$?
  if [ "$rc" != 0 ]; then
    printf '%s\n' "$out" | sed 's/^/    | /'
    echo "  failed probing $PY (exit $rc)"
    NEED=1
    return 1
  fi
  while IFS=$'\t' read -r st name a b; do
    case "$st" in
      ok) echo "  ok $name $a" ;;
      missing) echo "  missing $name ($a)"; NEED=$((NEED + 1)) ;;
      wrong) echo "  wrong $name $a ($b)"; NEED=$((NEED + 1)) ;;
      *) [ -z "$st" ] || echo "    | $st $name $a $b" ;;
    esac
  done <<<"$out"
  return 0
}

echo "install_deps.sh: $PY ($("$PY" --version 2>&1)), $REQ ($([ "$CHECK" = 1 ] && echo 'check: nothing is installed' || echo install))"
probe
if [ "$NEED" = 0 ]; then echo "result: ready: nothing to do"; exit 0; fi
if [ "$CHECK" = 1 ]; then
  echo "  would install $REQ into $PY"
  echo "result: not ready (see missing / wrong)"
  exit 1
fi

if [ -n "${UV:-}" ] && [ -x "$UV" ]; then
  INSTALL=("$UV" pip install --python "$PY")
elif "$PY" -m pip --version >/dev/null 2>&1; then
  INSTALL=("$PY" -m pip install --disable-pip-version-check --no-input)
else
  UV=""
  for c in "$(command -v uv 2>/dev/null)" "${OJH_TOOLS_DIR:+$OJH_TOOLS_DIR/uv/uv}"; do
    if [ -n "$c" ] && [ -x "$c" ]; then UV="$c"; break; fi
  done
  if [ -z "$UV" ]; then
    echo "  failed: $PY has no pip and no uv was found (set UV to a uv executable)"
    echo "result: failed"
    exit 1
  fi
  INSTALL=("$UV" pip install --python "$PY")
fi
CARGS=()
for c in ${CONS[@]+"${CONS[@]}"}; do CARGS+=(-c "$c"); done
echo "  \$ ${INSTALL[*]} ${CARGS[*]+${CARGS[*]}} -r $REQ"
"${INSTALL[@]}" ${CARGS[@]+"${CARGS[@]}"} -r "$REQ" 2>&1 | sed 's/^/    | /'
rc=${PIPESTATUS[0]}
echo "after the install (exit $rc):"
probe
if [ "$rc" = 0 ] && [ "$NEED" = 0 ]; then echo "result: ok"; exit 0; fi
echo "result: failed"
exit 1
