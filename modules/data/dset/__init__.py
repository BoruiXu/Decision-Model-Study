"""dset: the data module's split tool (standard library only, Python 3.9+).

Splits the labelled rows of a corpus into the four sets and the training pool by the task's split rule (data.split),
grows a trainset from the pool, checks and cleans rows, and draws a fresh evalset from rows no set has used. Run as
`PYTHONPATH=<repo root>/modules/data python3 -m dset <command>`; see dset.cli and docs/superpowers/specs (§6).
"""
import os
import sys

DSET_VERSION = "0.1.1"

# The repository root (three levels above this package's directory): dset uses the framework's rsi.rows and
# rsi.datasplit, so they are found ahead of anything else on the path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if sys.path[:1] != [REPO_ROOT]:
    sys.path.insert(0, REPO_ROOT)


class DsetError(Exception):
    """An error in the input or the rule: dset prints `dset: error: <reason>` and exits 1."""
