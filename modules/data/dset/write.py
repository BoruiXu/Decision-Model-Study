"""What dset writes: set directories and their manifests (docs/interfaces/datasets.md), the corpus manifest and split
report, and the task version with the filled rule. The same input writes the same bytes: the only time stamp is a
manifest's created_at. Everything a command writes goes through publishing(): all of it lands, or none of it."""
import contextlib
import copy
import datetime
import json
import os
import shutil
import tempfile

from rsi import artifacts as A
from rsi import datasplit as DS
from rsi import rows as R

from . import DSET_VERSION

SPLIT_NAMES = {"testset": "test", "evalset": "eval", "valset": "val", "pool": "train", "trainset": "train"}
STAGING = A.DSET_STAGING  # ".dset-": the name prefix of what publishing() writes before putting it in place


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(obj))


def write_jsonl(path, rows):
    """Write the rows, one JSON object a line; return the manifest's files entry {path, rows}."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": os.path.basename(path), "rows": len(rows)}


def resplit(rows, name):
    """The rows as they are but for `split`, which becomes `name`."""
    out = []
    for r in rows:
        r = dict(r)
        r["split"] = name
        out.append(r)
    return out


def label_row(row):
    return {"id": row["id"], "target": row["target"], "meta": {"label_source": row["meta"]["label_source"]}}


def _plain(v):
    return "%s" % v


# The sealed sets: their manifest.json is denied to no module, so their notes carry no label counts (split_report.json
# in the corpus keeps them).
SEALED = ("evalset", "testset")


def notes_line(command, entry, key=None, kind=None):
    """`dset <command>: rows <n>; labels <label> <rows>, ...; <key> <lo>-<hi>; range rows <r>; left out <l>`, the parts
    the entry has (the key's span only when there is one); no label counts for a set of kind SEALED."""
    parts = ["rows %d" % entry["rows"]]
    if "label_counts" in entry and kind not in SEALED:
        parts.append("labels " + ", ".join("%s %d" % kv for kv in entry["label_counts"].items()))
    if key and entry.get("key_span") is not None:
        parts.append("%s %s-%s" % (key, _plain(entry["key_span"][0]), _plain(entry["key_span"][1])))
    parts += ["%s %d" % (name.replace("_", " "), entry[name]) for name in ("range_rows", "left_out") if name in entry]
    return "dset %s: %s" % (command, "; ".join(parts))


def train_view(report):
    """The split report as train may read it (a trainset's quality_report.json): no label counts of the evalset and the
    testset."""
    out = copy.deepcopy(report)
    for kind in ("evalset", "testset"):
        out["sets"][kind].pop("label_counts", None)
    return out


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest(kind, task, rnd, files, notes, parents=(), **extra):
    m = {"kind": kind, "id": None, "task_id": task["task_id"], "task_version": task["version"], "round": rnd,
         "producer": "data", "created_at": _now(), "parents": list(parents), "files": files, "notes": notes}
    m.update(extra)
    return m


def write_set(d, kind, rows, task, rnd, notes, parents=(), quality_report=None, **extra):
    """Write set `kind` (testset, evalset, valset, pool or trainset) of the full rows `rows` into the new directory d:
    rows.jsonl for the pool, trainset and valset; inputs.jsonl (rsi.rows.input_row_of) for the valset, evalset and
    testset; labels.jsonl for the evalset and testset; quality_report.json for the trainset. `extra`: the kind's own
    manifest fields."""
    os.makedirs(d)
    files = {}
    if kind in ("pool", "trainset", "valset"):
        files["rows"] = write_jsonl(os.path.join(d, "rows.jsonl"), rows)
    if kind in ("valset", "evalset", "testset"):
        files["inputs"] = write_jsonl(os.path.join(d, "inputs.jsonl"), [R.input_row_of(r) for r in rows])
    if kind in ("evalset", "testset"):
        files["labels"] = write_jsonl(os.path.join(d, "labels.jsonl"), [label_row(r) for r in rows])
    if kind == "trainset":
        write_json(os.path.join(d, "quality_report.json"), quality_report)
        files["quality_report"] = {"path": "quality_report.json"}
    write_json(os.path.join(d, "manifest.json"), manifest(kind, task, rnd, files, notes, parents, **extra))
    return d


def write_task(d, task, current_md):
    """The task version with the filled rule: task.json, and TASK.md: rsi.datasplit.fill_task_md of the current
    TASK.md's bytes (read in binary, so CRLF stays) and the rule, the bytes registration accepts as a fill."""
    os.makedirs(d)
    write_json(os.path.join(d, "task.json"), task)
    with open(os.path.join(d, "TASK.md"), "wb") as f:
        f.write(DS.fill_task_md(current_md, task["data"]["split"]))
    return d


CORPUS_FILES = ("split_report.json", "manifest.json")  # what sets adds to a corpus without a manifest, in this order


def corpus_files(task, rnd, rows_file, n_rows, script, command, label_method, report, notes):
    """{name: bytes} of the files sets adds to a corpus without a manifest (CORPUS_FILES): split_report.json, the whole
    report (every set's label counts too), and manifest.json."""
    files = {"rows": {"path": rows_file, "rows": n_rows}, "script": {"path": script},
             "split_report": {"path": "split_report.json"}}
    m = manifest("corpus", task, rnd, files, notes, command=command, sources=DS.source_paths(task),
                 label_method=label_method)
    return {"split_report.json": dumps({"dset_version": DSET_VERSION, "split": report}).encode("utf-8"),
            "manifest.json": dumps(m).encode("utf-8")}


class _Publish(object):
    """New directories, each written inside a temporary directory next to its place (dir()), and new files (file()),
    put in place by commit(): the directories by rename in the order asked, then the files, each written to a temporary
    name and renamed. abort() removes all of it, and the parent directories made for it."""

    def __init__(self):
        self.dirs, self.files, self.temps, self.placed, self.made = [], [], [], [], []

    def _parent(self, final):
        parent = os.path.dirname(os.path.abspath(final))
        missing = []
        while not os.path.isdir(parent):
            missing.append(parent)
            parent = os.path.dirname(parent)
        for d in reversed(missing):
            os.mkdir(d)
            self.made.append(d)
        return os.path.dirname(os.path.abspath(final))

    def dir(self, final):
        """The path to write the new directory `final` at (the writer makes it); it is renamed into place at commit."""
        tmp = tempfile.mkdtemp(prefix=STAGING, dir=self._parent(final))
        self.temps.append(tmp)
        self.dirs.append((os.path.join(tmp, os.path.basename(os.path.abspath(final))), final))
        return self.dirs[-1][0]

    def file(self, final, data):
        self.files.append((final, data))

    def commit(self):
        for staged, final in self.dirs:
            if os.path.lexists(final):
                raise OSError("%s exists now: dset overwrites nothing" % final)
            os.rename(staged, final)
            self.placed.append(final)
        for tmp in self.temps:
            os.rmdir(tmp)
        self.temps = []
        for final, data in self.files:
            if os.path.lexists(final):
                raise OSError("%s exists now: dset overwrites nothing" % final)
            tmp = os.path.join(self._parent(final), "%s%s.tmp" % (STAGING, os.path.basename(final)))
            self.temps.append(tmp)
            with open(tmp, "wb") as f:
                f.write(data)
            os.rename(tmp, final)
            self.temps.remove(tmp)
            self.placed.append(final)

    def abort(self):
        for p in reversed(self.temps + self.placed):
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.lexists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        for d in reversed(self.made):
            try:
                os.rmdir(d)
            except OSError:
                pass


@contextlib.contextmanager
def publishing():
    """Write through the _Publish given; at the end everything is put in place, and on any error nothing stays."""
    pub = _Publish()
    try:
        yield pub
        pub.commit()
    except BaseException:
        pub.abort()
        raise
