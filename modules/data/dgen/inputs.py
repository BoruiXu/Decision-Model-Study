"""What dgen reads: only the paths of its brief's `inputs` (the data module's trainset purpose). The task, the current
trainset's rows, the pool's rows, the valset's inputs.jsonl (never its labels) and the best version's candidate;
nothing of the evalset or the testset, which the brief does not give.

Real rows are the observed ones: neither prompt variants (features.table.real_rows) nor rows dgen generated in an
earlier round (meta.source "dgen:<method>", which a later trainset holds). Only real rows are seeds, SMOTE anchors and
partners, the feature space's scaler, the CLLM reference and the distance reference; the generated rows of the
trainset still count for the dedup and copy checks and for what score-pool leaves out."""
import hashlib
import json
import os

from features.table import real_rows
from rsi import rows as R
from rsi.metrics import hard_label

from . import DgenError


def read_json(path, what=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise DgenError("%s: cannot read %s (%s)" % (what or path, path, e))


def read_jsonl(path, what=None):
    try:
        return R.read_jsonl(path)
    except (OSError, ValueError) as e:
        raise DgenError("%s: %s" % (what or path, e))


def set_file(d, logical, what):
    """The path of logical file `logical` (rows, inputs) of the set directory `d`, from its manifest."""
    m = read_json(os.path.join(d, "manifest.json"), what)
    spec = (m.get("files") or {}).get(logical) if isinstance(m, dict) else None
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise DgenError("%s %s: its manifest.json lists no files.%s" % (what, d, logical))
    return os.path.join(d, spec["path"])


SCORES_META = "score_pool.json"  # beside pool_predictions.jsonl: who scored it (score-pool writes, diagnose checks)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def union(*row_lists):
    """The rows of the lists, each id once (the first row of an id; the trainset is drawn from the pool)."""
    seen, out = set(), []
    for rows in row_lists:
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
    return out


def generated(row):
    """Whether dgen generated the row (meta.source "dgen:<method>")."""
    source = (row.get("meta") or {}).get("source") if isinstance(row.get("meta"), dict) else None
    return isinstance(source, str) and source.startswith("dgen:")


class Inputs(object):
    """A data brief's inputs, read when first used. trainset_rows and pool_rows are real rows (no prompt variants, no
    generated rows); trainset_all is every trainset row but the prompt variants (generated ones included); val_inputs
    are the valset's input rows (no labels)."""

    def __init__(self, brief_path):
        self.brief_path = os.path.abspath(brief_path)
        self.brief = b = read_json(brief_path, "--brief")
        if not isinstance(b, dict) or not isinstance(b.get("inputs"), dict) or "task" not in b["inputs"]:
            raise DgenError("--brief %s: not a brief (no inputs.task)" % brief_path)
        self.inputs = b["inputs"]
        self.task = read_json(self.inputs["task"], "the brief's task")
        errors = R.check_task(self.task) if isinstance(self.task, dict) else ["not a JSON object"]
        if errors:
            raise DgenError("the brief's task %s: %s" % (self.inputs["task"], "; ".join(errors[:3])))
        self.labels = self.task["decision"]["labels"]
        self.spend_file = b.get("spend_file")
        self._cache = {}

    def _need(self, key, why):
        if key not in self.inputs:
            raise DgenError("the brief gives no %s, which %s needs" % (key, why))
        return self.inputs[key]

    def _rows(self, key, logical, why):
        if (key, logical) not in self._cache:
            d = self._need(key, why)
            rows = read_jsonl(set_file(d, logical, "the brief's %s" % key), "the brief's %s" % key)
            self._cache[(key, logical)] = real_rows(rows) if logical == "rows" else rows
        return self._cache[(key, logical)]

    @property
    def trainset_all(self):
        return self._rows("trainset", "rows", "dgen")

    @property
    def trainset_rows(self):
        return [r for r in self.trainset_all if not generated(r)]

    @property
    def pool_rows(self):
        return [r for r in self._rows("pool", "rows", "dgen") if not generated(r)]

    @property
    def val_inputs(self):
        return self._rows("valset", "inputs", "the leak_val check")

    def real(self):
        """The real rows of the trainset and the pool, each id once: what the space, CLLM and the distance check fit
        on."""
        return union(self.trainset_rows, self.pool_rows)

    def known(self):
        """Every row of the trainset (generated ones included) and the pool, each id once: what the copy check
        compares with."""
        return union(self.trainset_all, self.pool_rows)

    def label(self, row):
        return hard_label(row["target"], self.labels)

    def candidate_manifest(self):
        """The best version's candidate's manifest, {} when the brief gives none."""
        d = self.inputs.get("best_candidate")
        return read_json(os.path.join(d, "manifest.json"), "the brief's best_candidate") if d else {}

    def candidate(self):
        """(directory, manifest, absolute model.predict) of the best version's candidate."""
        d = self._need("best_candidate", "score-pool (the best version's candidate; a round without a best version "
                                         "has none)")
        m = read_json(os.path.join(d, "manifest.json"), "the brief's best_candidate")
        entry = (m.get("model") or {}).get("predict") if isinstance(m, dict) else None
        if not isinstance(entry, str) or not entry:
            raise DgenError("best_candidate %s: its manifest has no model.predict" % d)
        path = os.path.join(d, entry)
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise DgenError("best_candidate %s: model.predict %s is not an executable file" % (d, entry))
        return d, m, path
