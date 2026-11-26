"""The feature table of a list of rows: the Featurizer and real_rows.

The Featurizer gives the adapter only a deep copy of each row's state (rsi.rows.state_of), never the row: no id, no
target, no meta. Its columns are the sorted union of the fit rows' columns; a column a row lacks is 0.0.
"""
import copy

from rsi.rows import state_of

from . import FEATURES_VERSION, FeatureError
from .adapter import load_adapter
from .flatten import as_number


def real_rows(rows):
    """The rows that are not prompt variants (no meta.extra.variant): a variant row duplicates its seed with renamed
    keys and tables, so every fit, refit, example draw and data generation step takes only these."""
    out = []
    for r in rows:
        meta = r.get("meta")
        extra = meta.get("extra") if isinstance(meta, dict) else None
        if not (isinstance(extra, dict) and "variant" in extra):
            out.append(r)
    return out


def _pattern(path):
    """A verbatim path as parts, None standing for any list index: "messages[].content" -> ["messages", None,
    "content"]."""
    parts = []
    for seg in path.split("."):
        indices = 0
        while seg.endswith("[]"):
            seg, indices = seg[:-2], indices + 1
        if seg:
            parts.append(seg)
        parts.extend([None] * indices)
    return parts


def _under(key, pattern):
    """Whether the edit-view path `key` is at or under the verbatim pattern."""
    parts = key.split(".")
    if len(parts) < len(pattern):
        return False
    for part, want in zip(parts, pattern):
        if want is None and not (part.isascii() and part.isdigit()):
            return False
        if want is not None and part != want:
            return False
    return True


def _g(x):
    return "%.6g" % x


def _leaks(columns, feats, labels):
    """The columns that determine a label: one value exactly on the rows of that label (at least two of them) and
    other values on every other row."""
    rows_of = {}
    for i, y in enumerate(labels):
        rows_of.setdefault(y, []).append(i)
    if len(rows_of) < 2:
        return []
    out = []
    for c in columns:
        values = [f.get(c, 0.0) for f in feats]
        for y, idx in rows_of.items():
            if len(idx) < 2 or len(set(values[i] for i in idx)) != 1:
                continue
            v = values[idx[0]]
            if all(values[i] != v for i in range(len(values)) if labels[i] != y):
                out.append("column %s is %s exactly on the rows labelled %s" % (c, _g(v), y))
                break
    return out


class Featurizer(object):
    """Rows -> a numeric table through the task's adapter (the generic one when adapter_path is None). Callers pass
    rsi.rules.adapter_path(task) (resolved against the repository root), never the task's features.adapter string.

    `verbatim` lists the state paths whose text is the row's own request ("messages[].content", "[]" for list
    items): edit_view never offers them."""

    def __init__(self, adapter_path=None, verbatim=None):
        self.adapter = load_adapter(adapter_path)
        self.verbatim = list(verbatim or [])
        self._patterns = [_pattern(p) for p in self.verbatim]
        self.columns = None

    def _features(self, row):
        feats = self.adapter.features(copy.deepcopy(state_of(row)))
        if not isinstance(feats, dict):
            raise FeatureError("row %s: the adapter's features gave %s, not an object of column -> number"
                               % (row.get("id"), type(feats).__name__))
        out = {}
        for k, v in feats.items():
            x = as_number(v)
            if not isinstance(k, str) or x is None:
                raise FeatureError("row %s: column %r is %r, not a finite number" % (row.get("id"), k, v))
            out[k] = x
        return out

    def fit(self, rows, labels=None):
        """Fix the columns on `rows` and return them. `labels` (optional) is each row's hard label, in row order:
        FeatureError names every column that determines a label (see _leaks)."""
        feats = [self._features(r) for r in rows]
        columns = sorted(set(k for f in feats for k in f))
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(feats):
                raise FeatureError("fit: %d labels for %d rows" % (len(labels), len(feats)))
            leaks = _leaks(columns, feats, labels)
            if leaks:
                raise FeatureError("label leak: %s; such a column determines the label, so the features may not "
                                   "use it" % "; ".join(leaks))
        self.columns = columns
        return list(columns)

    def transform(self, rows):
        """(ids, matrix): one row of floats per row, in the fit's column order (0.0 for a column a row lacks)."""
        if self.columns is None:
            raise FeatureError("transform before fit: the columns are not known yet")
        ids, matrix = [], []
        for r in rows:
            f = self._features(r)
            ids.append(r["id"])
            matrix.append([f.get(c, 0.0) for c in self.columns])
        return ids, matrix

    def edit_view(self, state):
        """The adapter's edit view of the state without the paths at or under a verbatim path."""
        view = self.adapter.edit_view(copy.deepcopy(state))
        return dict((k, v) for k, v in view.items() if not any(_under(k, p) for p in self._patterns))

    def signature(self):
        """What the features depend on: this code's version, the adapter file and its sha256, the column count."""
        return {"version": FEATURES_VERSION, "adapter": self.adapter.path, "adapter_sha256": self.adapter.sha256,
                "columns": len(self.columns) if self.columns is not None else 0}
