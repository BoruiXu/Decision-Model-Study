"""features: a row's state as numbers, for the compare baselines and for data generation (standard library only,
Python 3.9+).

- flatten: the generic model view (column -> float, list aggregates) and edit view (path -> scalar), and set_paths,
  which writes an edit view back into a state;
- adapter: a task's adapter file (the task's features.adapter), whose functions replace the generic ones;
- constraints: the task's declarative constraints (features.constraints: nonneg, integer, sum, order, zero_if_zero),
  repaired and checked by the adapter before its own repair and check;
- table: the Featurizer (a deep copy of each row's state only, never the row; fixed columns; the label-leak check)
  and real_rows (the rows that are not prompt variants).

Import it with `<repo root>/modules/common` on sys.path; it uses the framework's rsi.rows, so the repository root is
put on the path too.
"""
import os
import sys

FEATURES_VERSION = "1.0.0"

# The repository root (three levels above this package's directory): table.py uses rsi.rows.state_of.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class FeatureError(ValueError):
    """A state, an adapter or a feature table that cannot be used: the message says which and why."""
