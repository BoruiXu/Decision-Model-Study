"""Workspace: STATE.json, the append-only registry and registered artifact copies.

Layout: <ws>/STATE.json, <ws>/registry.jsonl, <ws>/artifacts/<id>/.
Registration copies a source directory into a staging area, checks the copy
(content hash, artifact validation, workspace rules), then moves it to
artifacts/<id>/ and records it. An evalset, testset or corpus is handed over:
once every check has passed its source directory is removed (MOVED_KINDS),
so only the workspace holds the labels; a refused registration changes
nothing (but the compute of a refused candidate, below). Per-kind workspace
rules live in REGISTER_CHECKS (refusals) and
REGISTER_EFFECTS (STATE updates). Among the refusals, the numbers a
candidate or an evaluation reports (metrics, baselines, train gate, deltas
and intervals) are recomputed with rsi.metrics, so the promotion rule and the
train gate rest on numbers the code has checked.

Compute used (STATE budget_used.compute_hours) is the cost.compute_hours of
the registered candidates plus that of refused ones: a candidate refused after
its source checks (not because the run stopped) is noted in
budget_used.refused {source real path: hours} (absent in older workspaces:
empty) and counted once per source and cost; registering that source later
removes it from refused and adds its cost only when the cost changed. The
unregistered candidates of a hand-back a stop closes before it is processed
(held for over_budget, or handed back) count the same way
(count_stopped_handback).
budget_overrun tells the engine, before it registers a hand-back, whether its
candidates would go over budget.compute_hours without the over_budget
approval (the engine then holds the hand-back and asks).

promote, approve, next_round and stop_check carry out the top-level decisions;
the decisions themselves come from the pure rules in rsi.rules.

Every change to STATE.json is a read-modify-write inside Workspace.transaction(): it holds a short
exclusive lock on <ws>/STATE.lock (fcntl.flock), reloads STATE.json from disk, applies the change and
saves once. So `rsi approve`, `rsi answer` and `rsi stop` can write while `rsi run` works on the same
workspace (its Workspace object loaded earlier) without either losing the other's update. Readers need no
lock: STATE.json is replaced atomically.
"""
import contextlib
import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile

try:
    import fcntl
except ImportError:  # no fcntl (not POSIX, e.g. Windows): STATE writes are not locked (still atomic)
    fcntl = None

from rsi import datasplit as DS
from rsi import metrics as M
from rsi import rows as R
from rsi.artifacts import EXTERNAL_PREFIX, KIND_CHECKS, MOVED_KINDS, link_errors, load_manifest, validate_artifact
from rsi.rules import (APPROVAL_ITEMS, EPS, check_stop, decide_promotion, is_number, max_decision_mismatches,
                       reproduction_tolerance, stop_settings)

KINDS = ("task",) + tuple(KIND_CHECKS)
DATASETS = ("trainset", "valset", "evalset", "testset")
# A corpus (every labelled row and the script that wrote them) and a pool (the rows of the training range) are current
# too; STATE of a workspace from before them has no such keys (current reads None).
CURRENT_KINDS = ("task",) + DATASETS + ("corpus", "pool")
# Logical file whose rows are compared when checking data sets for overlap.
OVERLAP_FILE = {"trainset": "rows", "valset": "rows", "evalset": "inputs", "testset": "inputs", "pool": "rows"}
TASK_FILES = ("task.json", "TASK.md")
# Approvals that guard fixed rules: always required, and each one covers a single
# registration (the registration that relies on it records `used_by`).
SINGLE_USE_APPROVALS = ("task_change", "refresh_sets", "use_testset")
# Still allowed after the run has stopped: cleanup approvals and final experiment records.
AFTER_STOP_APPROVALS = ("delete",)
AFTER_STOP_KINDS = ("record",)
# Engine records (see the engine records section of Workspace).
DISPATCH_STATUSES = ("pending", "running", "handed_back", "long_job", "failed", "done")
WAITING_KINDS = ("approval", "question")
STATE_LOCK = "STATE.lock"
_HELD = {}  # lock path -> how often this process holds it (state_lock is reentrant within a process)


class RsiError(Exception):
    """A refusal. `errors` holds one message per problem."""

    def __init__(self, errors):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = list(errors)
        super(RsiError, self).__init__("\n".join(self.errors))


class StoppedError(RsiError):
    """A refusal because the run has stopped (STATE stopped is set). `rsi run` meets it when the user stopped the
    run from another process while a step was under way: the step is dropped and the run goes on to its report."""


def _decided(d):
    """How a round was decided, for refusals: `promote V1 from evaluation-0001`, or for a round
    closed by `rsi close-round` its keep and reason (`keep: closed without evaluation: ...`)."""
    if d.get("evaluation_id") is None:
        return "%s: %s" % (d.get("decision"), d.get("reason"))
    return "%s%s from %s" % (d.get("decision"), " " + d["version"] if d.get("version") else "",
                             d["evaluation_id"])


def _close(state, rnd, why):
    """Close round `rnd` of `state` (close_round, and stop_check_if_closed on a copy): a keep without evaluation,
    one more round without promotion."""
    state["no_promotion_streak"] += 1
    rnd["decision"] = {"decision": "keep", "version": None, "evaluation_id": None, "reason": why, "at": now_iso()}


def _same_hours(a, b):
    """True when compute hours `a` (None: none recorded) and `b` are the same number."""
    return is_number(a) and is_number(b) and abs(a - b) <= EPS


def _added_hours(ws, src, cost):
    """The compute hours registering a candidate from `src` (a real path) costing `cost` adds to the hours used:
    none when its refusal already counted that cost (budget_used.refused), else `cost`."""
    return 0 if _same_hours((ws.state["budget_used"].get("refused") or {}).get(src), cost) else cost


def _budget_excess(ws, added):
    """(used, budget) when `added` more compute hours take budget_used.compute_hours over budget.compute_hours
    and the task lists over_budget among its approvals (going over then needs that approval: _check_budget,
    Workspace.budget_overrun); None otherwise. Whether the round has the approval is the caller's to check."""
    task = ws.task()
    budget = task["budget"]["compute_hours"]
    if "over_budget" not in task["approvals"] or budget is None:
        return None
    used = ws.state["budget_used"]["compute_hours"]
    if used + added <= budget + EPS:
        return None
    return used, budget


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(ts):
    """A time written by now_iso (2026-11-22T01:02:03Z) as an aware datetime, or None."""
    try:
        return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError):
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(path):
    """sha256 over the sorted (relative path, bytes) of every regular file under `path`."""
    found = []
    for dirpath, _dirs, names in os.walk(path):
        for name in names:
            fp = os.path.join(dirpath, name)
            if os.path.isfile(fp):
                found.append((os.path.relpath(fp, path).replace(os.sep, "/"), fp))
    h = hashlib.sha256()
    for rel, fp in sorted(found):
        h.update(("%s\0%d\0" % (rel, os.path.getsize(fp))).encode("utf-8"))
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def check_task_dir(path):
    """Return (task, errors) for a task directory holding task.json and TASK.md."""
    if not os.path.isdir(path):
        return None, ["%s is not a directory" % path]
    errors = []
    if not os.path.isfile(os.path.join(path, "TASK.md")):
        errors.append("TASK.md: not found in %s" % path)
    try:
        with open(os.path.join(path, "task.json"), encoding="utf-8") as f:
            task = json.load(f)
    except (OSError, ValueError) as e:
        return None, errors + ["cannot load task.json in %s: %s" % (path, e)]
    return task, errors + R.check_task(task)


def _inside(child, parent):
    return os.path.commonpath([child, parent]) == parent


@contextlib.contextmanager
def state_lock(root):
    """Hold the exclusive lock <root>/STATE.lock (fcntl.flock) for a read-modify-write of STATE.json; waits
    while another process holds it. Reentrant within a process (an inner hold is a no-op). Without fcntl it
    only counts the holds: the write is unlocked."""
    path = os.path.realpath(os.path.join(root, STATE_LOCK))
    if path in _HELD:
        _HELD[path] += 1
        try:
            yield
        finally:
            _HELD[path] -= 1
        return
    fd = None
    if fcntl is not None:
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            raise RsiError("cannot open the STATE lock %s: %s (STATE.json is only changed under this lock; the "
                           "workspace directory must be writable)" % (path, e))
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as e:
            os.close(fd)
            raise RsiError("cannot lock %s: %s" % (path, e))
        except BaseException:
            os.close(fd)
            raise
    _HELD[path] = 1
    try:
        yield
    finally:
        del _HELD[path]
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _initial_state(task):
    return {
        "task_id": task["task_id"],
        "task_version": task["version"],
        "round": 1,
        "current": dict((k, None) for k in CURRENT_KINDS),
        "versions": [],
        "best": None,
        "eval_use_count": 0,
        "approvals": [],
        "rounds": {"1": {"entry": "data", "artifacts": [], "decision": None, "plan_id": None,
                        "opened_at": now_iso()}},
        "no_promotion_streak": 0,
        "stopped": None,
        "budget_used": {"compute_hours": 0},
        # Written by the engine, read by rsi.flow (see the engine records section of Workspace).
        "dispatches": [],
        "decisions": [],
        "waiting": None,
    }


class Workspace(object):
    _depth = 0  # transactions this object has open (only the outermost one locks, reloads and saves)
    # The real path of the source a registration in progress copies, once it passed the source and duplicate checks
    # (a candidate refused after that counts its compute: _count_refused); None otherwise.
    _source_now = None

    def __init__(self, ws):
        self.root = os.path.abspath(ws)
        sp = self._state_path()
        if not os.path.isfile(sp):
            raise RsiError("%s is not an rsi workspace (no STATE.json); run `rsi init` first" % self.root)
        try:
            with open(sp, encoding="utf-8") as f:
                self._state = json.load(f)
        except (OSError, ValueError) as e:
            raise RsiError("cannot read %s: %s" % (sp, e))
        self._entries = self._load_registry()
        self._by_id = dict((e["id"], e) for e in self._entries)
        self._claims = []
        self.warnings = []  # notes on the last registration for the user (`rsi register` prints them)

    @classmethod
    def init(cls, ws, task_dir):
        """Create a workspace at `ws` (artifacts/, an empty registry and work/, the run's own files: run.jsonl,
        the dispatches' output directories, the operator's notes) and register `task_dir` as task-0001."""
        root = os.path.abspath(ws)
        if os.path.isfile(os.path.join(root, "STATE.json")):
            raise RsiError("%s is already an rsi workspace" % root)
        if os.path.exists(root) and (not os.path.isdir(root) or os.listdir(root)):
            raise RsiError("%s exists and is not an empty directory" % root)
        src = os.path.realpath(task_dir)
        task, errors = check_task_dir(src)
        if errors:
            raise RsiError(errors)
        if _inside(os.path.realpath(root), src):
            raise RsiError("workspace %s must not be inside the task directory %s" % (root, src))
        os.makedirs(os.path.join(root, "artifacts"))
        os.makedirs(os.path.join(root, "work"))
        open(os.path.join(root, "registry.jsonl"), "w").close()
        self = cls.__new__(cls)
        self.root = root
        self._state = _initial_state(task)
        self._entries = []
        self._by_id = {}
        self._claims = []
        self.warnings = []
        self._register("task", src, ())
        return self

    # ---- state and registry -------------------------------------------------

    @property
    def state(self):
        return self._state

    def _state_path(self):
        return os.path.join(self.root, "STATE.json")

    def _registry_path(self):
        return os.path.join(self.root, "registry.jsonl")

    def _reload_state(self):
        """Read STATE.json again (another process may have saved it); kept as it is while the file does not
        exist yet (`Workspace.init`)."""
        sp = self._state_path()
        if not os.path.isfile(sp):
            return
        try:
            with open(sp, encoding="utf-8") as f:
                self._state = json.load(f)
        except (OSError, ValueError) as e:
            raise RsiError("cannot read %s: %s" % (sp, e))

    @contextlib.contextmanager
    def transaction(self):
        """A read-modify-write of STATE: hold STATE.lock, reload STATE.json, let the body change self.state,
        then save once. When the body raises (a refusal included) nothing is saved and self.state is as it was
        reloaded. Nested transactions (a writer calling another) join the outer one: no second lock, reload
        or save."""
        if self._depth:
            self._depth += 1
            try:
                yield self._state
            finally:
                self._depth -= 1
            return
        with state_lock(self.root):
            self._reload_state()
            before = copy.deepcopy(self._state)
            self._depth = 1
            try:
                yield self._state
                self.save_state()
            except BaseException:
                self._state = before
                raise
            finally:
                self._depth = 0

    def save_state(self):
        """Write STATE.json atomically (temp file in the same directory, then os.replace). Callers changing
        STATE use transaction(), which calls this once at its end."""
        fd, tmp = tempfile.mkstemp(prefix=".STATE.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._state_path())
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def _load_registry(self):
        out = []
        try:
            with open(self._registry_path(), encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            out.append(json.loads(line))
                        except ValueError as e:
                            raise RsiError("registry.jsonl line %d: invalid JSON (%s)" % (n, e))
        except OSError as e:
            raise RsiError("cannot read %s: %s" % (self._registry_path(), e))
        return out

    def _reload_registry(self):
        """Read registry.jsonl again (another process may have registered since this object loaded it); writers call
        it inside a transaction, where no registration can commit meanwhile."""
        self._entries = self._load_registry()
        self._by_id = dict((e["id"], e) for e in self._entries)

    def _append_registry(self, entry):
        with open(self._registry_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry

    # ---- lookups ------------------------------------------------------------

    def entries(self, kind=None):
        return [e for e in self._entries if kind is None or e["kind"] == kind]

    def entry(self, aid):
        if aid not in self._by_id:
            raise RsiError("unknown artifact id '%s'" % aid)
        return self._by_id[aid]

    def path(self, aid):
        self.entry(aid)
        return os.path.join(self.root, "artifacts", aid)

    def manifest(self, aid):
        """manifest.json of a registered artifact (task.json for a task)."""
        if self.entry(aid)["kind"] == "task":
            with open(os.path.join(self.path(aid), "task.json"), encoding="utf-8") as f:
                return json.load(f)
        return load_manifest(self.path(aid))

    def rows(self, aid, logical):
        """Rows of logical file `logical` of a registered artifact."""
        spec = self.manifest(aid)["files"][logical]
        return R.read_jsonl(os.path.join(self.path(aid), spec["path"]))

    def current(self, kind):
        if kind not in CURRENT_KINDS:
            raise RsiError("kind '%s' has no current artifact (only %s)" % (kind, ", ".join(CURRENT_KINDS)))
        return self._state["current"].get(kind)

    def task(self):
        """The current task definition."""
        return self.manifest(self.current("task"))

    def version(self, name):
        for v in self._state["versions"]:
            if v["version"] == name:
                return v
        return None

    def _open_approvals(self, what, ref=None):
        """Unconsumed approvals of `what` in the current round (matching `ref` if given)."""
        rnd = self._state["round"]
        return [a for a in self._state["approvals"]
                if a.get("what") == what and a.get("round") == rnd and not a.get("used_by")
                and (ref is None or a.get("ref") == ref)]

    def has_approval(self, what, ref=None):
        """True when STATE has an unconsumed approval of `what` in the current round."""
        return bool(self._open_approvals(what, ref))

    def split_fill_errors(self, task, task_md=None):
        """Why registering task definition `task` would not be a fill of the split rule ([] when it is one): a fill
        keeps the task_id, raises the version by one and only completes the current version's incomplete data.split,
        keeping every value it writes (rsi.datasplit.fill_errors against the current task), before any evalset or
        testset is registered; given `task_md` (its TASK.md, text or bytes), that must be, byte for byte,
        rsi.datasplit.fill_task_md(the current TASK.md's bytes, the new data.split): a writer builds it with that
        function from the bytes it read, not from a text-mode read. A fill registers without a task_change approval
        (docs/interfaces/task.md, data); registration always compares the TASK.md."""
        return _fill_errors(self, self.current("task"), task, task_md, self._entries)

    @staticmethod
    def is_attempt(kind, producer):
        """True for an artifact that counts as an attempt: a search. One dispatch of the training-tuning module
        runs one search; its trial candidates are not counted (also those of an old workspace: train candidates,
        tune's trials), the search they are compared in is the attempt."""
        return kind == "search"

    def attempts(self, rnd=None):
        """Ids of the attempts registered in round `rnd` (default: the current round), in order."""
        r = self._round(rnd)
        entries = [self._by_id[a] for a in r["artifacts"]]
        return [e["id"] for e in entries if self.is_attempt(e["kind"], e["producer"])]

    def attempt_limit(self):
        """stop.max_attempts_per_round of the current task (default 3)."""
        return stop_settings(self.task())["max_attempts_per_round"]

    def passed_candidates(self, rnd=None):
        """Candidates registered in round `rnd` (default: current) whose train gate passed."""
        r = self._round(rnd)
        return [a for a in r["artifacts"] if self._by_id[a]["kind"] == "candidate"
                and self.manifest(a)["train_gate"]["passed"] is True]

    def best_delivery_on_current_evalset(self):
        """The best version's predictions on the current evalset: its own delivery when that is on the
        current evalset, else the first non-milestone delivery of its candidate on it; None when there
        is no best version or no such delivery (the best version must be re-delivered first)."""
        best = self._state["best"]
        v = self.version(best) if best is not None else None
        cur = self.current("evalset")
        if v is None or cur is None:
            return None
        if self.manifest(v["delivery_id"])["evalset_id"] == cur:
            return v["delivery_id"]
        for e in self.entries("delivery"):
            d = self.manifest(e["id"])
            if d["candidate_id"] == v["candidate_id"] and d["evalset_id"] == cur \
                    and "predictions_test" not in d["files"]:
                return e["id"]
        return None

    def summary(self, aid):
        """One line on a registered artifact's module gate or role (SUMMARIES), or None."""
        fn = SUMMARIES.get(self.entry(aid)["kind"])
        return fn(self, self.manifest(aid)) if fn else None

    def package_location(self, delivery_id):
        """Where a registered delivery's package is: its path under artifacts/<id>/, or for
        `external:<name>` the external entry's location and sha256."""
        m = self.manifest(delivery_id)
        loc = m["package"]["location"]
        if loc.startswith(EXTERNAL_PREFIX):
            ext = m.get("external", {}).get(loc[len(EXTERNAL_PREFIX):], {})
            return "%s（%s，sha256 %s）" % (ext.get("location"), loc, ext.get("sha256"))
        return os.path.join(self.path(delivery_id), loc)

    def decision_mismatches(self, aid):
        """For a registered delivery: rows of its predictions_val whose decision differs from its candidate's
        while every probability is within the reproduction tolerance (the flips the task's
        delivery.max_decision_mismatch_rate allowed at registration); 0 for other kinds."""
        if self.entry(aid)["kind"] != "delivery":
            return 0
        m, task = self.manifest(aid), self.task()
        want = dict((r["id"], r) for r in self.rows(m["candidate_id"], "predictions_val"))
        flips, _off = _reproduction_diff(self.rows(aid, "predictions_val"), want, task["decision"]["labels"],
                                         reproduction_tolerance(task))
        return len(flips)

    def unverified(self, aid):
        """Metric and baseline names a registered artifact reports that rsi does not compute,
        so registration could not verify them (see unverified_names)."""
        return unverified_names(self.entry(aid)["kind"], self.manifest(aid), self.task())

    def is_kind(self, aid, kind):
        """True when `aid` is a registered artifact of `kind`."""
        e = self._by_id.get(aid)
        return e is not None and e["kind"] == kind

    # ---- engine records: dispatches, decisions, the pending question ---------
    # STATE["dispatches"]: [{"id": "d-0001", "round", "module", "purpose" (what it was dispatched for:
    #   rsi.flow.PURPOSES), "outdir" (relative to the workspace), "brief", "runner", "session", "status"
    #   (DISPATCH_STATUSES: pending until its agent is launched, then running), "returns" (times sent back to its
    #   session), "started_at", "ended_at", "marker"
    #   (absolute, or relative to the workspace)}];
    # STATE["decisions"]: [{"round", "point", "options", "choice", "by", "reason", "at"}];
    # STATE["waiting"]: null or {"id" (q-NNNN), "kind" (approval | question), "what", "ref", "question", "options",
    #   "since"}; STATE["waiting_seq"]: the number of the last waiting id.
    # A workspace made before these fields existed has none: the readers treat a missing field as empty.
    # The writers below change one field each in a transaction (one save, atomically); on a refusal or a
    # failed save neither STATE.json nor the in-memory STATE changes.

    def add_dispatches(self, records):
        """Append dispatch records in one save (several dispatched together are never half written) and
        return their ids. Each record gets the next id d-NNNN (an id in the record is
        replaced); absent fields default to round: the current round, status: running, returns: 0,
        started_at: now, and null brief, runner, session, ended_at, marker and purpose. module and outdir
        are required, status must be one of DISPATCH_STATUSES."""
        with self.transaction():
            current = list(self._state.get("dispatches") or [])
            numbers = [int(d["id"][2:]) for d in current if re.match(r"^d-\d+$", str(d.get("id")))]
            n = max(numbers + [0])
            new, errors = [], []
            for i, r in enumerate(records):
                d = {"round": self._state["round"], "module": None, "purpose": None, "outdir": None, "brief": None,
                     "runner": None, "session": None, "status": "running", "returns": 0, "started_at": now_iso(),
                     "ended_at": None, "marker": None}
                d.update(r)
                for field in ("module", "outdir"):
                    if not d.get(field):
                        errors.append("dispatch record %d: %s is required" % (i, field))
                if d["status"] not in DISPATCH_STATUSES:
                    errors.append("dispatch record %d: status %r is not one of %s"
                                  % (i, d["status"], ", ".join(DISPATCH_STATUSES)))
                n += 1
                d["id"] = "d-%04d" % n
                new.append(d)
            if errors:
                raise RsiError(errors)
            self._state["dispatches"] = current + new
        return [d["id"] for d in new]

    def update_dispatch(self, did, **fields):
        """Change fields of dispatch `did` in one save and return the updated record; the id cannot
        change and a status must be one of DISPATCH_STATUSES."""
        if "id" in fields:
            raise RsiError("the id of dispatch %s cannot change" % did)
        if "status" in fields and fields["status"] not in DISPATCH_STATUSES:
            raise RsiError("status %r is not one of %s" % (fields["status"], ", ".join(DISPATCH_STATUSES)))
        with self.transaction():
            self.dispatch(did)  # refuses an unknown id
            out = []
            for d in self._state.get("dispatches") or []:
                if d.get("id") == did:
                    d = dict(d)
                    d.update(fields)
                    changed = d
                out.append(d)
            self._state["dispatches"] = out
        return changed

    def add_decision(self, record):
        """Append a decision record in one save and return it. point and choice are required (choice one
        of options when options are given); absent fields default to round: the current round, options:
        [], by: "code", reason: "", at: now."""
        with self.transaction():
            d = {"round": self._state["round"], "point": None, "options": [], "choice": None, "by": "code",
                 "reason": "", "at": now_iso()}
            d.update(record)
            errors = ["decision: %s is required" % f for f in ("point", "choice") if not d.get(f)]
            if d["options"] and d["choice"] not in d["options"]:
                errors.append("decision: choice %r is not one of the options %s"
                              % (d["choice"], ", ".join(d["options"])))
            if errors:
                raise RsiError(errors)
            self._state["decisions"] = list(self._state.get("decisions") or []) + [d]
        return d

    def set_waiting(self, record):
        """Set the question the run waits on, in one save, and return it. kind (one of WAITING_KINDS),
        question and options (a non-empty list) are required; what and ref default to null, since to now.
        A record without an id gets the next q-NNNN (STATE waiting_seq counts them), so an answer can name the
        question it answers (`rsi answer --ref`)."""
        w = {"kind": None, "what": None, "ref": None, "question": None, "options": None, "since": now_iso()}
        w.update(record)
        errors = []
        if w["kind"] not in WAITING_KINDS:
            errors.append("waiting: kind %r is not one of %s" % (w["kind"], ", ".join(WAITING_KINDS)))
        if not w["question"]:
            errors.append("waiting: question is required")
        if not isinstance(w["options"], list) or not w["options"]:
            errors.append("waiting: options must be a non-empty list")
        if errors:
            raise RsiError(errors)
        with self.transaction():
            if not w.get("id"):
                n = int(self._state.get("waiting_seq") or 0) + 1
                self._state["waiting_seq"] = n
                w["id"] = "q-%04d" % n
            self._state["waiting"] = w
        return w

    def mark_delivered(self, question_ids, to):
        """Note on the answers (decisions) of the questions `question_ids` that they were given to `to` (a
        dispatch's launch or brief), in one save: an answer reaches a module once (`to` is noted once)."""
        ids = set(question_ids)
        with self.transaction():
            out = []
            for d in self._state.get("decisions") or []:
                if d.get("question_id") in ids and d.get("point") == "handback" \
                        and to not in (d.get("delivered_to") or []):
                    d = dict(d)
                    d["delivered_to"] = list(d.get("delivered_to") or []) + [to]
                out.append(d)
            self._state["decisions"] = out

    def clear_waiting(self):
        """Clear the question the run waits on (one save)."""
        with self.transaction():
            self._state["waiting"] = None

    def dispatches(self, rnd=None, module=None):
        """The dispatch records in order; only those of round `rnd` and of `module` when given."""
        return [d for d in self._state.get("dispatches") or []
                if (rnd is None or d.get("round") == rnd) and (module is None or d.get("module") == module)]

    def dispatch(self, did):
        """The dispatch record with id `did`."""
        for d in self._state.get("dispatches") or []:
            if d.get("id") == did:
                return d
        raise RsiError("unknown dispatch id '%s'" % did)

    def decisions(self, rnd=None, point=None):
        """The decision records in order; only those of round `rnd` and decision point `point` when given."""
        return [d for d in self._state.get("decisions") or []
                if (rnd is None or d.get("round") == rnd) and (point is None or d.get("point") == point)]

    def waiting(self):
        """The question the run waits on (STATE waiting), or None."""
        return self._state.get("waiting") or None

    # ---- registration -------------------------------------------------------

    def register(self, kind, src, fill_only=False):
        """Check `src` against the artifact contract and workspace rules; copy it in (move it in for
        an evalset, testset or corpus); return its id. A candidate refused after its source checks (not because the
        run stopped) still cost its training: _count_refused adds it to the compute used, in a save of its own.
        fill_only (a task): register it only as a fill of the split rule, checked on the staged copy
        (split_fill_errors with its TASK.md bytes) before any approval is weighed, so it never uses a task_change
        approval: the engine registers a module's task version this way, and the registry names its producer data
        (rsi-data wrote the rule; other task versions are the orchestrator's)."""
        if kind not in AFTER_STOP_KINDS:
            self._refuse_if_stopped("registering a %s" % kind)
        if kind not in KINDS:
            raise RsiError("unknown kind '%s' (expected one of %s)" % (kind, ", ".join(KINDS)))
        self._source_now = None
        try:
            return self._register(kind, src, REGISTER_CHECKS.get(kind, ()), fill_only and kind == "task")
        except StoppedError:
            raise
        except RsiError:
            if kind == "candidate" and self._source_now:
                self._count_refused(self._source_now)
            raise
        finally:
            self._source_now = None

    def _count_refused(self, src):
        """A candidate from `src` (a real path) was refused: its manifest's cost.compute_hours (a number, kind
        candidate; else nothing) counts toward budget_used.compute_hours and is noted in budget_used.refused
        {source: hours}, once per source and cost (R2): the same source at the same cost again is the same
        training; at another cost it is a new one, counted again (refused keeps the latest cost). Its own save,
        so no rollback of the refused registration takes it back."""
        try:
            m = load_manifest(src)
        except (OSError, ValueError):
            return
        cost = m.get("cost") if isinstance(m, dict) else None
        cost = cost.get("compute_hours") if isinstance(cost, dict) else None
        if not isinstance(m, dict) or m.get("kind") != "candidate" or not is_number(cost) or cost < 0:
            return
        if _same_hours((self._state["budget_used"].get("refused") or {}).get(src), cost):
            return  # counted already (as STATE was loaded; checked again under the lock)
        with self.transaction():
            self._reload_registry()
            if any(e["source"] == src for e in self._entries):
                return  # registered meanwhile (another process): its registration counted its cost
            used = self._state["budget_used"]
            refused = dict(used.get("refused") or {})
            if not _same_hours(refused.get(src), cost):
                refused[src] = cost
                used["refused"] = refused
                used["compute_hours"] = used["compute_hours"] + cost

    def handback_sources(self, d):
        """Real paths of the subdirectories dispatch `d`'s current hand-back (<outdir>/handback.json, when it is
        valid) lists that are not registered."""
        from rsi.briefs import read_handback  # rsi.briefs imports this module
        if not d.get("outdir"):
            return []
        out = os.path.join(self.root, d["outdir"])
        hb = read_handback(out)
        if not hb["ok"]:
            return []
        known = set(e["source"] for e in self._entries)
        return [src for src in (os.path.realpath(os.path.join(out, n)) for n in hb["artifacts"]) if src not in known]

    def count_stopped_handback(self, d):
        """A stop closes dispatch `d` with its hand-back not registered (held for over_budget, waiting for the
        answer to its questions, or handed back and not processed yet): the candidates it lists were trained, so
        each one not registered counts its compute as a refused candidate's (_count_refused: once per source and
        cost, never a registered source). Workspace.stop does it for the hand-backs STATE shows; the engine, settling
        after the stop, for those it finds then; the keying keeps the two from counting a source twice."""
        for src in self.handback_sources(d):
            self._count_refused(src)

    def refused_compute_hours(self):
        """The compute hours of refused candidates in budget_used.compute_hours (the sum of budget_used.refused;
        0 for a workspace made before it)."""
        return sum(v for v in (self._state["budget_used"].get("refused") or {}).values() if is_number(v))

    def budget_overrun(self, sources):
        """Whether registering the candidates among `sources` (real paths of a hand-back's subdirectories not
        registered yet; those whose manifest is a candidate with a number for cost.compute_hours count) would take
        the compute used over budget.compute_hours without the over_budget approval registration asks for (the
        check of _check_budget: over_budget among the task's approvals, a budget, no over_budget approval in this
        round). Returns {"dirs": those candidates' sources, "cost": the hours they add (a source whose refusal
        counted the same hours adds none), "used", "budget"} when it would, else None. Reads only."""
        if self.has_approval("over_budget"):
            return None
        dirs, added = [], 0
        for src in sources:
            try:
                m = load_manifest(src)
            except (OSError, ValueError):
                continue
            cost = m.get("cost") if isinstance(m, dict) else None
            cost = cost.get("compute_hours") if isinstance(cost, dict) else None
            if isinstance(m, dict) and m.get("kind") == "candidate" and is_number(cost):
                dirs.append(src)
                added += _added_hours(self, src, cost)
        over = _budget_excess(self, added) if dirs else None
        if over is None:
            return None
        return {"dirs": dirs, "cost": added, "used": over[0], "budget": over[1]}

    def _register(self, kind, src, checks, fill_only=False):
        self._claims = []  # approvals this registration relies on; consumed only on commit
        self.warnings = []
        src = self._source(src)
        # The staged copy follows links, so links are checked on the source (R43, R45).
        links = link_errors(src, kind in MOVED_KINDS)
        if links:
            raise RsiError(["source %s: %s" % (src, e) for e in links])
        arts = os.path.join(self.root, "artifacts")
        tmp = tempfile.mkdtemp(prefix=".staging-", dir=arts)
        try:
            staged = os.path.join(tmp, "artifact")
            try:
                shutil.copytree(src, staged)
            except OSError as e:  # e.g. a link loop or a dangling link inside a linked directory
                raise RsiError("cannot copy the source %s into the workspace: %s" % (src, e))
            digest = content_sha256(staged)
            for e in self._entries:
                if e["content_sha256"] == digest:
                    raise RsiError("source %s has the same content as %s (registered from %s)"
                                   % (src, e["id"], e["source"]))
            self._source_now = src  # a new source, its content new: a refusal from here on is its own
            manifest, errors = self._validate(kind, staged)
            if fill_only and manifest is not None:
                fill = self.split_fill_errors(manifest, _read_bytes(os.path.join(staged, "TASK.md")))
                if fill:
                    raise RsiError(["not a fill of the split rule: %s" % e for e in fill])
            if manifest is not None:
                for check in checks:
                    if errors and check in COSTLY_CHECKS:
                        continue  # refused already: skip the checks that read a large file (the corpus)
                    errors += check(self, manifest, staged)
            if errors:
                raise RsiError([e.replace(staged, src) for e in errors])
            # Every check passed. A sealed set is handed over (R37): its source is set aside before the
            # commit (put back if the commit fails) and removed after it.
            aside = self._set_aside(kind, src) if kind in MOVED_KINDS else None
            try:
                aid = self._commit(kind, src, digest, manifest, staged, "data" if fill_only else "orchestrator")
            except BaseException:
                if aside is not None:
                    os.rename(aside, src)
                raise
        finally:
            self._claims = []
            shutil.rmtree(tmp, ignore_errors=True)
        if aside is not None:
            try:
                shutil.rmtree(aside)
            except OSError as e:  # registered all the same: report the leftover, do not fail (R43)
                self.warnings.append("%s is registered, but removing its source %s (moved aside to %s) failed: %s; "
                                     "that directory still holds the labels: delete it by hand (after a `delete` "
                                     "approval when the task requires one)" % (aid, src, aside, e))
        return aid

    @staticmethod
    def _set_aside(kind, src):
        """Rename the source directory to a hidden sibling and return that path; refuse (nothing changed)
        when the source cannot be moved, since a registered evalset or testset leaves no copy behind."""
        parent, name = os.path.split(src)
        aside = os.path.join(parent, ".%s.rsi-handed-over-%s" % (name, os.urandom(4).hex()))
        try:
            os.rename(src, aside)
        except OSError as e:
            raise RsiError("cannot move the source %s into the workspace (%s): a registered %s is handed over and "
                           "its source directory removed, so it must be movable; nothing was registered"
                           % (src, e, kind))
        return aside

    def _source(self, src):
        path = os.path.realpath(src)
        if not os.path.isdir(path):
            raise RsiError("source %s is not a directory" % src)
        arts = os.path.realpath(os.path.join(self.root, "artifacts"))
        if _inside(path, arts):
            raise RsiError("source %s is a registered artifact copy inside the workspace; "
                           "write a new version to a new directory" % path)
        if _inside(os.path.realpath(self.root), path):
            raise RsiError("source %s contains the workspace %s" % (path, self.root))
        for e in self._entries:
            if e["source"] == path:
                raise RsiError("source %s is already registered as %s; write a new version to a new directory"
                               % (path, e["id"]))
        return path

    def _validate(self, kind, path):
        """Return (manifest or None, errors): the artifact contract plus common workspace checks."""
        if kind == "task":
            task, errors = check_task_dir(path)
            return (None if errors else task), errors
        errors = validate_artifact(kind, path, self.task())
        if errors:
            return None, errors
        manifest = load_manifest(path)
        return manifest, self._common_errors(manifest)

    def _common_errors(self, m):
        s = self._state
        errors = []
        if m["id"] is not None:
            errors.append("id must be null before registration (got %r)" % m["id"])
        if m["task_id"] != s["task_id"]:
            errors.append("task_id is %r but the workspace task is %r" % (m["task_id"], s["task_id"]))
        if m["task_version"] != s["task_version"]:
            errors.append("task_version is %d but the workspace task is %s v%d"
                          % (m["task_version"], s["task_id"], s["task_version"]))
        if m["round"] != s["round"]:
            errors.append("round is %d but the workspace is in round %d" % (m["round"], s["round"]))
        for p in m["parents"]:
            if p not in self._by_id:
                errors.append("parents: %s is not registered" % p)
        return errors

    def _commit(self, kind, src, digest, manifest, staged, task_producer="orchestrator"):
        """Move the checked copy in, append it to the registry and apply its STATE effects, in one
        transaction: on STATE (and the registry, for the next id) as saved now, since another process may
        have written since the checks ran. A stop recorded meanwhile refuses the registration, and the
        approvals it claimed must still be unused. A task version's producer is `task_producer` (data for a fill of
        the split rule handed back by rsi-data), another kind's its manifest's."""
        with self.transaction():
            self._entries = self._load_registry()
            self._by_id = dict((e["id"], e) for e in self._entries)
            if kind not in AFTER_STOP_KINDS:
                self._refuse_if_stopped("registering a %s" % kind)
            claims = self._reclaim()
            aid = "%s-%04d" % (kind, len(self.entries(kind)) + 1)
            dst = os.path.join(self.root, "artifacts", aid)
            if os.path.exists(dst):
                raise RsiError("%s exists but is not in registry.jsonl; repair the workspace first" % dst)
            if kind == "task":
                files = dict((n, file_sha256(os.path.join(staged, n))) for n in TASK_FILES)
                producer, parents = task_producer, []
            else:
                manifest["id"] = aid
                _write_json(os.path.join(staged, "manifest.json"), manifest)
                files = dict((n, file_sha256(os.path.join(staged, spec["path"])))
                             for n, spec in manifest["files"].items())
                producer, parents = manifest["producer"], list(manifest["parents"])
            os.rename(staged, dst)
            s = self._state
            self._append_registry({
                "id": aid, "kind": kind, "round": s["round"], "producer": producer,
                "source": src, "content_sha256": digest, "path": "artifacts/" + aid,
                "files": files, "parents": parents, "registered_at": now_iso(),
            })
            s["rounds"][str(s["round"])]["artifacts"].append(aid)
            effect = REGISTER_EFFECTS.get(kind)
            if effect:
                effect(self, manifest, aid)
            for a in claims:
                a["used_by"] = aid
        return aid

    def unlisted(self):
        """Registry entries that STATE's rounds do not list (a registration cut off between appending
        registry.jsonl and saving STATE)."""
        listed = set(a for r in self._state["rounds"].values() for a in r.get("artifacts", []))
        return [e for e in self._entries if e["id"] not in listed]

    def repair_registry(self):
        """Finish registrations cut off between registry.jsonl and STATE (the registry is appended first): list
        each such entry in its round and, in the current round, apply the registration's STATE effects (current
        artifact, budget, evalset uses, plan) and mark the single-use approval it relied on used. Returns the
        repaired ids ([] and no save when there are none)."""
        if not self.unlisted():
            return []
        done = []
        with self.transaction():
            self._entries = self._load_registry()
            self._by_id = dict((e["id"], e) for e in self._entries)
            for e in self.unlisted():
                rnd = self._state["rounds"].get(str(e["round"]))
                if rnd is None:
                    continue
                aid, kind = e["id"], e["kind"]
                if e["round"] == self._state["round"]:
                    m = self.manifest(aid)
                    for what in _single_use_needed(self, kind, m, aid):
                        opened = self._open_approvals(what)
                        if opened:
                            opened[0]["used_by"] = aid
                    effect = REGISTER_EFFECTS.get(kind)
                    if effect:
                        effect(self, m, aid)
                rnd["artifacts"].append(aid)
                done.append(aid)
        return done

    def _reclaim(self):
        """The approvals this registration claimed (self._claims, taken from STATE when its checks ran), found
        again in STATE as reloaded for the commit; refused when one was used by another registration meanwhile."""
        out = []
        for c in self._claims:
            found = None
            for a in self._state["approvals"]:
                if not a.get("used_by") and not any(a is o for o in out) and \
                        all(a.get(k) == c.get(k) for k in ("what", "ref", "by", "quote", "at", "round")):
                    found = a
                    break
            if found is None:
                raise RsiError("the %s approval this registration relies on was used by another registration "
                               "meanwhile; nothing was registered" % c.get("what"))
            out.append(found)
        return out

    # ---- top-level decisions: promote, approve, next, stop check -------------

    def _refuse_if_stopped(self, action):
        st = self._state["stopped"]
        if st:
            raise StoppedError("the run has stopped (%s, at %s): %s is refused; only `rsi approve <ws> delete` "
                           "and `rsi register <ws> record <dir>` remain allowed (`rsi report <ws>` prints the "
                           "run report)" % (st.get("reason"), st.get("at"), action))

    def _round(self, rnd=None):
        """STATE's entry for round `rnd` (default: the current round)."""
        return self._state["rounds"][str(self._state["round"] if rnd is None else rnd)]

    def _is_milestone(self, aid):
        """True for a registered evaluation with uses_testset (a milestone evaluation)."""
        e = self._by_id.get(aid)
        return e is not None and e["kind"] == "evaluation" and self.manifest(aid)["uses_testset"] is True

    def evaluations(self, rnd=None):
        """Registered non-milestone evaluations of round `rnd` (default: current), in order."""
        return [a for a in self._round(rnd)["artifacts"]
                if self._by_id[a]["kind"] == "evaluation" and not self._is_milestone(a)]

    def best_primary(self):
        """Primary metric of the best version, from its evaluation's metrics.candidate (or None)."""
        best = self._state["best"]
        if best is None:
            return None
        v = self.version(best)
        if v is None:
            raise RsiError("STATE best %s is missing from versions" % best)
        value = self.manifest(v["evaluation_id"])["metrics"]["candidate"].get(self.task()["metrics"]["primary"])
        return value if is_number(value) else None

    def stop_check(self):
        """(stop, reason) from rsi.rules.check_stop; changes nothing."""
        return check_stop(self.task(), self._state, self.best_primary())

    def stop_check_if_closed(self):
        """(stop, reason): the stop check as it would be once the current round is closed (close_round: its decision
        a keep without evaluation, no_promotion_streak one more), so a question can say whether closing the round
        stops the run. Works on a copy of STATE; changes nothing."""
        s = copy.deepcopy(self._state)
        _close(s, s["rounds"][str(s["round"])], "closed without evaluation")
        return check_stop(self.task(), s, self.best_primary())

    def promote(self, evaluation_id):
        """Apply the promotion rule to a registered evaluation of this round.

        Returns (decision, version or None, reason). "promote" adds version
        V<n> and makes it best and resets no_promotion_streak; "keep" adds 1 to
        the streak. Either way the round's decision is set; once per round.
        A milestone evaluation (uses_testset) is refused: it decides nothing.
        """
        with self.transaction():
            self._refuse_if_stopped("`rsi promote`")
            s = self._state
            rnd, cur = s["round"], self._round()
            if self._is_milestone(evaluation_id):
                raise RsiError("%s is a milestone evaluation (uses_testset): milestone evaluations are not promotion "
                               "decisions; `rsi promote` takes this round's evalset evaluation" % evaluation_id)
            if cur["decision"] is not None:
                raise RsiError("round %d is already decided (%s): promote runs once per round"
                               % (rnd, _decided(cur["decision"])))
            e = self._by_id.get(evaluation_id)
            if e is None:
                here = [a for a in cur["artifacts"] if self._by_id[a]["kind"] == "evaluation"]
                known = ("this round's evaluations: %s" % ", ".join(here)) if here else \
                    "round %d has no registered evaluation" % rnd
                raise RsiError("%s is not registered (%s): register the judge's evaluation with "
                               "`rsi register <ws> evaluation <dir>` before `rsi promote`" % (evaluation_id, known))
            if e["kind"] != "evaluation":
                raise RsiError("%s is a %s, not an evaluation: `rsi promote` takes a registered evaluation id"
                               % (evaluation_id, e["kind"]))
            if e["round"] != rnd:
                raise RsiError("%s was registered in round %d, not the current round %d: "
                               "evaluate this round's candidate and register that evaluation"
                               % (evaluation_id, e["round"], rnd))
            m = self.manifest(evaluation_id)
            if m["evalset_id"] != self.current("evalset"):
                raise RsiError("%s was scored on %s but the current evalset is %s: evaluate again on the "
                               "current evalset" % (evaluation_id, m["evalset_id"], self.current("evalset")))
            gate = self.manifest(m["candidate_id"])["train_gate"]["passed"]
            decision, reason = decide_promotion(self.task(), m, s["best"] is not None, gate)
            version = None
            if decision == "promote":
                version = "V%d" % (len(s["versions"]) + 1)
                s["versions"].append({"version": version, "candidate_id": m["candidate_id"],
                                      "delivery_id": m["delivery_id"], "evaluation_id": evaluation_id,
                                      "round": rnd})
                s["best"] = version
                s["no_promotion_streak"] = 0
            else:
                s["no_promotion_streak"] += 1
            cur["decision"] = {"decision": decision, "version": version, "evaluation_id": evaluation_id,
                               "reason": reason, "at": now_iso()}
            return decision, version, reason

    def close_round(self, reason):
        """Close a round that has no evaluation to decide on: a keep without a version.

        For a round in which no candidate could be evaluated (e.g. none passed
        the train gate). Refused when the round is already decided, when it has
        a registered evaluation (milestone evaluations aside: that round is
        decided with `rsi promote`), or after a stop. The decision records null
        version and evaluation ids, and no_promotion_streak goes up by one, so
        the round counts toward max_rounds and patience. Returns
        ("keep", None, reason) like promote.
        """
        with self.transaction():
            self._refuse_if_stopped("`rsi close-round`")
            s = self._state
            rnd, cur = s["round"], self._round()
            errors = []
            if not reason or not reason.strip():
                errors.append("--reason: why the round closes without an evaluation is required")
            if cur["decision"] is not None:
                errors.append("round %d is already decided (%s): a round is decided once"
                              % (rnd, _decided(cur["decision"])))
            else:
                evaluations = self.evaluations()
                if evaluations:
                    errors.append("round %d has registered evaluations (%s): decide the round with "
                                  "`rsi promote <ws> %s`, not `rsi close-round`"
                                  % (rnd, ", ".join(evaluations), evaluations[-1]))
            if errors:
                raise RsiError(errors)
            why = "closed without evaluation: %s" % reason.strip()
            _close(s, cur, why)
            return "keep", None, why

    def stop(self, by, quote):
        """Record a user-requested stop at any time; return its reason.

        `quote` is the user's own words. Refused when the run has already
        stopped. Afterwards the after-stop rules apply as for any stop. A
        hand-back not processed yet (held for an over_budget approval, or
        handed back) ends unregistered: its candidates were trained, so their
        compute counts in the same save, as a refused candidate's
        (count_stopped_handback).
        """
        with self.transaction():
            self._refuse_if_stopped("`rsi stop`")
            errors = []
            if not by or not by.strip():
                errors.append("--by: the name of the person who asked to stop is required")
            if not quote or not quote.strip():
                errors.append("--quote: the user's own words are required")
            if errors:
                raise RsiError(errors)
            # The registry as it is now, under the lock: a held candidate the engine registered after this object
            # was loaded (approved, its dispatch not yet done) is no held compute.
            self._reload_registry()
            for d in self._state.get("dispatches") or []:
                if d.get("status") in ("handed_back", "long_job"):
                    self.count_stopped_handback(d)
            reason = "user requested: %s" % quote
            self._state["stopped"] = {"reason": reason, "by": by, "at": now_iso(), "round": self._state["round"]}
            return reason

    def closed(self, rnd=None):
        """True when round `rnd` (default: current) was decided by `rsi close-round`."""
        d = self._round(rnd)["decision"]
        return d is not None and d.get("evaluation_id") is None

    def approve(self, what, by, quote, ref=None):
        """Record a user approval in the current round and return it.

        `quote` is the user's own words. A plan approval needs `ref`: a plan
        registered in this round. A STATE waiting that asks for this approval
        (kind approval, the same what, and the same ref when it names one) is
        cleared in the same save: the approval answers it.
        """
        with self.transaction():
            if what not in AFTER_STOP_APPROVALS:
                self._refuse_if_stopped("approving %s" % what)
            s = self._state
            errors = []
            if what not in APPROVAL_ITEMS:
                errors.append("unknown approval item '%s' (expected one of %s)" % (what, ", ".join(APPROVAL_ITEMS)))
            if not by or not by.strip():
                errors.append("--by: the name of the person who approved is required")
            if not quote or not quote.strip():
                errors.append("--quote: the user's own words are required")
            if what == "plan":
                if not ref:
                    errors.append("approving a plan needs --ref <plan id> (a plan registered in round %d)"
                                  % s["round"])
                else:
                    e = self._by_id.get(ref)
                    if e is None or e["kind"] != "plan":
                        errors.append("--ref %s is not a registered plan" % ref)
                    elif e["round"] != s["round"]:
                        errors.append("--ref %s was registered in round %d, not the current round %d"
                                      % (ref, e["round"], s["round"]))
            if errors:
                raise RsiError(errors)
            a = {"what": what, "ref": ref, "by": by, "quote": quote, "at": now_iso(), "round": s["round"]}
            s["approvals"].append(a)
            w = s.get("waiting")
            if w and w.get("kind") == "approval" and w.get("what") == what and (not w.get("ref") or w.get("ref") == ref):
                s["waiting"] = None
            return a

    def next_round(self):
        """Run the stop check, then open the next round.

        A round holding a registered evaluation but no decision is refused
        first (no state change), so a stop cannot skip judging it; milestone
        evaluations (uses_testset) decide nothing and do not count. Then the
        stop check: (None, reason) when a stop condition holds (STATE.stopped
        is written). Otherwise (n, entry module of the round's plan); refuses,
        naming each missing step, when the round has no decision (from
        `rsi promote` or `rsi close-round`), no plan, or (when the task lists
        "plan" in approvals) no approval of its plan.
        """
        with self.transaction():
            self._refuse_if_stopped("`rsi next`")
            s = self._state
            rnd, cur = s["round"], self._round()
            undecided = self.evaluations()
            if undecided and cur["decision"] is None:
                raise RsiError("`rsi next` refused in round %d: the round has registered evaluations (%s) but no "
                               "decision; run `rsi promote <ws> <evaluation id>` (e.g. `rsi promote <ws> %s`) first"
                               % (rnd, ", ".join(undecided), undecided[-1]))
            stop, reason = self.stop_check()
            if stop:
                s["stopped"] = {"reason": reason, "at": now_iso(), "round": s["round"]}
                return None, reason
            errors = []
            if cur["decision"] is None:
                errors.append("round %d has no decision: run `rsi promote <ws> <evaluation id>` after a registered "
                              "evaluation, or `rsi close-round <ws> --reason <text>` when no candidate can be "
                              "evaluated" % rnd)
            plan = cur["plan_id"]
            if plan is None:
                errors.append("round %d has no plan: register the judge's plan with `rsi register <ws> plan <dir>` "
                              "first" % rnd)
            elif "plan" in self.task()["approvals"] and not self.has_approval("plan", plan):
                errors.append("plan %s is not approved: record the user's approval with "
                              "`rsi approve <ws> plan --ref %s --by <name> --quote <their words>` first" % (plan, plan))
            if errors:
                raise RsiError(["`rsi next` refused in round %d:" % rnd] + errors)
            entry = self.manifest(plan)["entry_module"]
            n = rnd + 1
            s["round"] = n
            s["rounds"][str(n)] = {"entry": entry, "artifacts": [], "decision": None, "plan_id": None,
                                   "opened_at": now_iso()}
            return n, entry

    # ---- helpers for per-kind checks ----------------------------------------

    def ref_errors(self, field, aid, kind=None):
        """Errors when `aid` is not a registered artifact (of `kind`, when given)."""
        e = self._by_id.get(aid)
        if e is None:
            return ["%s: %s is not registered" % (field, aid)]
        if kind is not None and e["kind"] != kind:
            return ["%s: %s is a %s, not a %s" % (field, aid, e["kind"], kind)]
        return []

    def current_errors(self, field, aid, kind):
        """Errors when `aid` is not the current artifact of `kind`."""
        cur = self.current(kind)
        if cur is None:
            return ["%s: %s, but there is no current %s yet (none registered, or none passed its gate)"
                    % (field, aid, kind)]
        if aid != cur:
            return ["%s: %s is not the current %s %s" % (field, aid, kind, cur)]
        return []

    def approval_errors(self, what, needed_for):
        """Errors when the current round has no usable approval of `what`.

        A single-use approval is claimed for the registration in progress and
        consumed (`used_by`) only when that registration commits.
        """
        if what not in SINGLE_USE_APPROVALS:
            return [] if self.has_approval(what) else [
                "%s needs a %s approval in round %d" % (needed_for, what, self._state["round"])]
        for a in self._open_approvals(what):
            if not any(a is c for c in self._claims):
                self._claims.append(a)
                return []
        rnd = self._state["round"]
        used = [a["used_by"] for a in self._state["approvals"]
                if a.get("what") == what and a.get("round") == rnd and a.get("used_by")]
        note = " (approvals are single-use; this round's were used by %s)" % ", ".join(used) if used else ""
        return ["%s needs a %s approval in round %d%s" % (needed_for, what, rnd, note)]


# ---- per-kind registration checks: fn(ws, manifest, path) -> [errors] -------

def _src_rows(path, manifest, logical):
    return R.read_jsonl(os.path.join(path, manifest["files"][logical]["path"]))


def _coverage_errors(ws, path, manifest, logical, set_kind):
    """Predictions in `logical` must cover exactly the ids of the current `set_kind`."""
    cur = ws.current(set_kind)
    if cur is None or logical not in manifest["files"]:
        return []
    expected = [r["id"] for r in ws.rows(cur, OVERLAP_FILE[set_kind])]
    errors = R.check_predictions(_src_rows(path, manifest, logical), ws.task(), expected)
    return ["%s: %s (against %s %s)" % (logical, e, set_kind, cur) for e in errors]


def _single_use_needed(ws, kind, m, aid):
    """The single-use approvals a registration of `kind` relied on (as its checks claim them), for a repair."""
    if kind == "task":
        return ["task_change"] if ws.current("task") not in (None, aid) and not _was_split_fill(ws, m, aid) else []
    if kind in ("valset", "evalset", "testset", "corpus", "pool"):
        return ["refresh_sets"] if ws.current(kind) not in (None, aid) else []
    if kind == "delivery" and "predictions_test" in (m.get("files") or {}):
        return ["use_testset"]
    if kind == "evaluation" and m.get("uses_testset") is True:
        return ["use_testset"]
    return []


def _sealed_ids(entries):
    """The evalsets and testsets among registry `entries`: once one is registered the split rule is fixed."""
    return [e["id"] for e in entries if e["kind"] in ("evalset", "testset")]


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _fill_errors(ws, prev, task, task_md, entries):
    """Why task definition `task` (TASK.md `task_md`, text or bytes; None: not compared) is no fill of the split rule
    of the registered task version `prev`, with the registry `entries` as they were then
    (Workspace.split_fill_errors)."""
    errors = DS.fill_errors(ws.manifest(prev), task)
    if task_md is not None:
        if isinstance(task_md, str):
            task_md = task_md.encode("utf-8")
        current = _read_bytes(os.path.join(ws.path(prev), "TASK.md"))
        split = DS.task_split(task)
        if not task_md.startswith(current):
            errors.append("TASK.md: it does not start with the TASK.md of %s byte for byte; a fill keeps that text and "
                          "appends only the split rule section (rsi.datasplit.fill_task_md)" % prev)
        elif not isinstance(split, dict) or task_md != DS.fill_task_md(current, split):
            errors.append("TASK.md: what it appends to the TASK.md of %s is not the split rule section of its "
                          "data.split (rsi.datasplit.split_section_md); a fill appends exactly that" % prev)
    sealed = _sealed_ids(entries)
    if sealed:
        errors.append("%s registered: the split rule is fixed once an evalset or testset is registered"
                      % ", ".join(sealed))
    return errors


def _was_split_fill(ws, task, aid):
    """Whether task version `aid` (registered) was a fill of the split rule when it registered: against the task
    registered before it and its TASK.md, with the registry as it was then (Workspace.split_fill_errors then)."""
    before = []
    for e in ws.entries():
        if e["id"] == aid:
            break
        before.append(e)
    tasks = [e["id"] for e in before if e["kind"] == "task"]
    if not tasks:
        return False
    return not _fill_errors(ws, tasks[-1], task, _read_bytes(os.path.join(ws.path(aid), "TASK.md")), before)


def _split_changed(old, new):
    """Whether data.split differs between two task versions: complete rules compared with their defaults written out
    (rsi.datasplit.filled), others as written."""
    a, b = DS.task_split(old), DS.task_split(new)
    if DS.is_complete(a) and DS.is_complete(b):
        return DS.filled(a) != DS.filled(b)
    return a != b


def _check_task_change(ws, task, path):
    """A new task version keeps the task_id and raises the version by one, with a task_change approval; a fill of the
    split rule (Workspace.split_fill_errors) needs none. Once an evalset or testset is registered, data.split is
    fixed: a version that changes it is refused, approval or not."""
    s = ws.state
    errors = []
    if task["task_id"] != s["task_id"]:
        errors.append("task_id is %r but the workspace task is %r; a different task needs a new workspace"
                      % (task["task_id"], s["task_id"]))
    if task["version"] != s["task_version"] + 1:
        errors.append("version is %d, expected %d (current version %d + 1)"
                      % (task["version"], s["task_version"] + 1, s["task_version"]))
    fill = ws.split_fill_errors(task, _read_bytes(os.path.join(path, "TASK.md")))
    if not fill:
        return errors
    sealed = _sealed_ids(ws.entries())
    if sealed and _split_changed(ws.task(), task):
        errors.append("data.split is fixed once an evalset or testset is registered (%s registered); "
                      "a new task version must keep it" % ", ".join(sealed))
    approval = ws.approval_errors("task_change", "a new task version")
    if approval and DS.task_split(ws.task()) != DS.task_split(task) and not DS.changes_outside_split(ws.task(), task):
        approval = ["%s (its task.json changes only data.split, but it is no fill of the split rule, which needs "
                    "none: %s)" % (approval[0], "; ".join(fill))]
    return errors + approval


def _check_disjoint(ws, m, path):
    """The current trainset, valset, evalset, testset and pool share no id, input or cluster, but for the trainset and
    the pool (the trainset is drawn from the pool); the corpus holds every row and is not compared. With the split
    rule's unit state_id (data.split.unit) they share no state_id either; otherwise sharing one is allowed."""
    kind = m["kind"]
    by_state = DS.task_split(ws.task()).get("unit") == "state_id"
    mine = _src_rows(path, m, OVERLAP_FILE[kind])
    errors = []
    for other in DATASETS + ("pool",):
        cur = ws.current(other)
        if other == kind or cur is None or {kind, other} == {"trainset", "pool"}:
            continue
        theirs = ws.rows(cur, OVERLAP_FILE[other])
        ov = R.overlap(mine, theirs)
        shared = R.shared_state_ids(mine, theirs) if by_state else 0
        if any(ov.values()) or shared:
            errors.append("%s overlaps the current %s %s (ids %d, inputs %d, clusters %d%s)%s"
                          % (kind, other, cur, ov["ids"], ov["inputs"], ov["clusters"],
                             ", state_ids %d" % shared if by_state else "",
                             ": the split rule's unit is state_id, so no state_id is in two sets" if shared else ""))
    return errors


def _corpus_index(ws, cid, ids):
    """{id: (input fingerprint, target, state_id, cluster)} of the rows of corpus `cid` whose id is in `ids`. The corpus
    file is read line by line and only those rows are kept: a corpus holds every labelled row (Elliptic: some 290k rows,
    over 600 MB), the set being registered a few thousand."""
    rel = ws.manifest(cid)["files"]["rows"]["path"]
    out = {}
    for r in R.iter_jsonl(os.path.join(ws.path(cid), rel)):
        rid = r.get("id")
        if isinstance(rid, str) and rid in ids:
            meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            out[rid] = (R.input_fingerprint(R.state_of(r)), r.get("target"), R.group_of(r), meta.get("cluster"))
    return out


def _check_corpus_registered(ws, m, path):
    """A task that names its raw data (paths in data.sources, or data.rows) has its sets drawn from a corpus: a valset,
    evalset, testset or pool registers only once a corpus is current, so _check_from_corpus checks it. A task naming
    neither (sets made by hand, a task without a data section, an old workspace's string sources) needs none."""
    task = ws.task()
    if ws.current("corpus") is not None or not (DS.source_paths(task) or (task.get("data") or {}).get("rows")):
        return []
    return ["no corpus is current, but this task's data comes from its raw data (data.sources, data.rows): register "
            "the corpus (the preprocessing script and all labelled rows) before the %s, as the data module's hand-back "
            "does" % m["kind"]]


def _check_from_corpus(ws, m, path):
    """With a current corpus, every row of a valset, evalset, testset or pool is the corpus row of its id: the same
    input (input_fingerprint of the state), the same state_id and cluster (which _check_disjoint compares the sets by) and,
    for the rows holding a label (a valset's or pool's rows, an evalset's or testset's labels), the same target. So the
    sets trace back to the registered preprocessing script. The corpus vouches for the sets registered after it
    (_check_corpus_first); without one (sets registered by hand, an old workspace) nothing is checked. A costly check
    (COSTLY_CHECKS): it runs only when every check before it passed."""
    cid = ws.current("corpus")
    if cid is None:
        return []
    kind, labels = m["kind"], ws.task()["decision"]["labels"]
    rows = _src_rows(path, m, OVERLAP_FILE[kind])
    labelled = _src_rows(path, m, "labels") if kind in ("evalset", "testset") else rows
    targets = dict((r["id"], r["target"]) for r in labelled)
    try:
        corpus = _corpus_index(ws, cid, set(r["id"] for r in rows))
    except (OSError, ValueError) as e:
        return ["cannot read the current corpus %s: %s" % (cid, e)]
    absent, inputs, fields, other = [], [], [], []
    for r in rows:
        rid = r["id"]
        if rid not in corpus:
            absent.append(rid)
            continue
        fp, target, state_id, cluster = corpus[rid]
        if R.input_fingerprint(R.state_of(r)) != fp:
            inputs.append(rid)
        if R.group_of(r) != state_id or r["meta"].get("cluster") != cluster:
            fields.append(rid)
        t = targets.get(rid)
        if t is not None and any(t.get(k, 0) != target.get(k, 0) for k in labels):
            other.append(rid)
    errors = []
    for ids, what in ((absent, "are not rows of"), (inputs, "have another input than the row of the same id in"),
                      (fields, "have another state_id or cluster than the row of the same id in"),
                      (other, "have another target than the row of the same id in")):
        if ids:
            errors.append("%d rows of the %s %s the current corpus %s, e.g. %s: every set is drawn from the "
                          "corpus" % (len(ids), kind, what, cid, ids[0]))
    return errors


def _check_corpus_first(ws, m, path):
    """The first corpus comes before any valset, evalset, testset or pool: a corpus vouches only for the sets registered
    after it (_check_from_corpus), so one registered after them would vouch for rows nobody checked against it. A
    replacement corpus (a corpus is current: refresh_sets) is not compared with the sets already current."""
    if ws.current("corpus") is not None:
        return []
    cur = [(k, ws.current(k)) for k in ("testset", "evalset", "valset", "pool") if ws.current(k)]
    if not cur:
        return []
    return ["%s %s current, but a corpus vouches only for the valset, evalset, testset and pool registered after it: "
            "register the corpus first (before the sets), as the data module's hand-back does"
            % (", ".join("%s %s" % kc for kc in cur), "is" if len(cur) == 1 else "are")]


def _check_refresh(ws, m, path):
    cur = ws.current(m["kind"])
    if cur is None:
        return []
    return ws.approval_errors("refresh_sets", "%s %s is already current: registering a new %s"
                              % (m["kind"], cur, m["kind"]))


def _check_new_evalset_rows(ws, m, path):
    """A new evalset shares no ids or inputs with any evalset registered before."""
    mine = _src_rows(path, m, "inputs")
    errors = []
    for e in ws.entries("evalset"):
        ov = R.overlap(mine, ws.rows(e["id"], "inputs"))
        if ov["ids"] or ov["inputs"]:
            errors.append("evalset overlaps earlier evalset %s (ids %d, inputs %d): "
                          "a refreshed evalset needs new rows" % (e["id"], ov["ids"], ov["inputs"]))
    return errors


def _check_parent_trainset(ws, m, path):
    p = m["parent_trainset"]
    return [] if p is None else ws.ref_errors("parent_trainset", p, "trainset")


def _check_candidate(ws, m, path):
    """A candidate is trained on the current trainset and valset. The data module does not train (whether a new
    trainset helps is judged by the round's train and judge dispatches): a candidate whose producer is data (an old
    workspace's A/B draft) is refused; registered ones stay readable."""
    errors = []
    if m.get("producer") == "data":
        errors.append("producer data: the data module does not train; a new trainset is tested by the round's "
                      "training-tuning (train) and judge dispatches, so the data module registers no candidate")
    errors += ws.current_errors("trainset_id", m["trainset_id"], "trainset")
    errors += ws.current_errors("valset_id", m["valset_id"], "valset")
    return errors + _coverage_errors(ws, path, m, "predictions_val", "valset")


def _check_budget(ws, m, path):
    """With "over_budget" in the task's approvals, going over budget.compute_hours needs approval. A candidate
    whose refusal already counted its cost (budget_used.refused, same source and cost) adds nothing."""
    added = _added_hours(ws, ws._source_now, m["cost"]["compute_hours"])
    over = _budget_excess(ws, added)
    if over is None:
        return []
    used, budget = over
    return ws.approval_errors("over_budget", "a candidate costing %s compute hours (total %s, above "
                              "budget.compute_hours %s)" % (m["cost"]["compute_hours"], used + added, budget))


def _check_attempts(ws, m, path):
    """A round allows stop.max_attempts_per_round attempts (searches)."""
    if not ws.is_attempt(m["kind"], m["producer"]):
        return []
    done, limit, rnd = ws.attempts(), ws.attempt_limit(), ws.state["round"]
    if len(done) < limit:
        return []
    return ["%d of %d attempts used in round %d (%s): stop.max_attempts_per_round allows %d attempts "
            "(searches) per round; deliver and evaluate a candidate that passed the "
            "train gate, or close the round with `rsi close-round <ws> --reason <text>`"
            % (len(done), limit, rnd, ", ".join(done), limit)]


def _check_search(ws, m, path):
    """Trial candidates registered; each ok trial's metric_val is its candidate's metrics_val[primary]
    (R40); the ok trials' candidates, the default included, share one code_version.sha256 (Task 14)."""
    errors = ws.ref_errors("trainset_id", m["trainset_id"], "trainset")
    errors += ws.ref_errors("valset_id", m["valset_id"], "valset")
    ok = []
    for i, t in enumerate(m["trials"]):
        if t["status"] != "ok":  # other trials may have no candidate (M3) and are not compared
            continue
        missing = ws.ref_errors("trials.candidate_id", t["candidate_id"], "candidate")
        errors += missing
        if not missing:
            ok.append((i, t, ws.manifest(t["candidate_id"])))
    return errors + _trial_metric_errors(ws, ok) + _search_code_version_errors(ok)


def _trial_metric_errors(ws, ok):
    primary = ws.task()["metrics"]["primary"]
    errors = []
    for i, t, c in ok:
        want = c["metrics_val"].get(primary)
        if is_number(want) and abs(t["metric_val"] - want) > EPS:
            errors.append("trials[%d].metric_val is %r but its candidate %s has metrics_val.%s %r (checked when it "
                          "was registered): copy each trial's metric_val from its candidate's metrics_val"
                          % (i, t["metric_val"], t["candidate_id"], primary, want))
    return errors


def _search_code_version_errors(ok):
    """All ok trials' candidates share one code_version.sha256 (hex case ignored); null shows nothing."""
    groups = []  # [sha256 or None, [candidate ids]] in order of first appearance
    for _i, t, c in ok:
        sha = (c.get("code_version") or {}).get("sha256")
        key = sha.lower() if isinstance(sha, str) else None
        for g in groups:
            if g[0] == key:
                g[1].append(t["candidate_id"])
                break
        else:
            groups.append([key, [t["candidate_id"]]])
    errors = []
    unknown = [ids for key, ids in groups if key is None]
    if unknown:
        errors.append("code_version.sha256 is null for trial candidates %s: a search compares configurations of "
                      "one version of the training code, which only the sha256 of code_version.entry shows; "
                      "train the trials with a training command that records its sha256" % ", ".join(unknown[0]))
    known = [(key, ids) for key, ids in groups if key is not None]
    if len(known) > 1:
        errors.append("the ok trials ran different code versions (code_version.sha256 %s): a search compares "
                      "configurations of one version of the training code, the default included; changes to the "
                      "train command or its defaults land between rounds, so rerun the trials with one version"
                      % "; ".join("%s (%s)" % (key, ", ".join(ids)) for key, ids in known))
    return errors


def _check_delivery(ws, m, path):
    errors = ws.ref_errors("candidate_id", m["candidate_id"], "candidate")
    valset = ws.ref_errors("valset_id", m["valset_id"], "valset")
    if not errors:
        gate = ws.manifest(m["candidate_id"])["train_gate"]
        if not gate["passed"]:
            errors.append("candidate_id: %s failed its train gate (train_gate.passed is false: %s); only a "
                          "candidate that passed the train gate is delivered" % (m["candidate_id"], gate["detail"]))
        if not valset:
            errors += _reproduction_errors(ws, m, path)
    errors += valset
    errors += ws.current_errors("evalset_id", m["evalset_id"], "evalset")
    errors += _coverage_errors(ws, path, m, "predictions_eval", "evalset")
    if "testset_id" in m:
        errors += ws.current_errors("testset_id", m["testset_id"], "testset")
    if "predictions_test" in m["files"]:
        errors += ws.approval_errors("use_testset", "predictions_test")
        errors += _coverage_errors(ws, path, m, "predictions_test", "testset")
    return errors


def _tol_text(x):
    """A tolerance as the docs write it: 1e-06 -> 1e-6, 0.001 stays 0.001."""
    m = re.match(r"^(.*)e([+-])0*(\d+)$", "%g" % x)
    return "%g" % x if m is None else "%se%s%s" % (m.group(1), "-" if m.group(2) == "-" else "", m.group(3))


def _reproduction_errors(ws, m, path):
    """predictions_val is the package run on the inputs.jsonl of the candidate's valset and reproduces the
    candidate's predictions_val: the same ids, the same decision and every probability within the task's
    reproduction tolerance (rsi.rules.reproduction_tolerance). This is what runtime.consistent_with_offline
    means; the flag alone is not trusted."""
    cid, vid = m["candidate_id"], m["valset_id"]
    validated_on = ws.manifest(cid)["valset_id"]
    if vid != validated_on:
        return ["valset_id: %s, but candidate %s was validated on %s: run the package on %s's inputs.jsonl, the "
                "rows of the candidate's predictions_val" % (vid, cid, validated_on, validated_on)]
    task = ws.task()
    preds = _src_rows(path, m, "predictions_val")
    coverage = R.check_predictions(preds, task, [r["id"] for r in ws.rows(vid, "rows")])
    if coverage:
        return ["predictions_val: %s (against valset %s)" % (e, vid) for e in coverage]
    want = dict((r["id"], r) for r in ws.rows(cid, "predictions_val"))
    tol = reproduction_tolerance(task)
    flips, off = _reproduction_diff(preds, want, task["decision"]["labels"], tol)
    errors = []
    if off:
        errors.append("predictions_val: %d of %d rows do not reproduce candidate %s's predictions_val (every "
                      "probability within %s required; the task's delivery.reproduction_tolerance, default 1e-6), "
                      "e.g. %s: the package's prediction interface must give the candidate's predictions on %s's "
                      "inputs.jsonl, whatever runtime.consistent_with_offline says"
                      % (len(off), len(preds), cid, _tol_text(tol), "; ".join(off[:3]), vid))
    cap = max_decision_mismatches(task, len(preds))
    if len(flips) > cap:
        errors.append("predictions_val: %d rows have another decision than candidate %s's predictions_val with "
                      "every probability within %s, e.g. %s; the task allows %d such rows "
                      "(delivery.max_decision_mismatch_rate %s of %d valset rows, rounded down; default 0)"
                      % (len(flips), cid, _tol_text(tol), "; ".join(flips[:3]), cap,
                         task.get("delivery", {}).get("max_decision_mismatch_rate", 0), len(preds)))
    return errors


def _reproduction_diff(preds, want, labels, tol):
    """Compare a delivery's predictions_val rows with the candidate's (`want`, id -> row): returns (flips,
    off). A flip is a row whose decision differs while every probability is within `tol` (a near-boundary
    row); off lists the rows that do not reproduce otherwise. Both are descriptions starting with the id."""
    flips, off = [], []
    for r in preds:
        w = want.get(r["id"])
        if w is None:
            off.append("%s (not in the candidate's predictions_val)" % r["id"])
            continue
        far = None
        for label in labels:
            if abs(r["probs"][label] - w["probs"].get(label, 0)) > tol:
                far = "probs.%s %r, candidate %r" % (label, r["probs"][label], w["probs"].get(label))
                break
        if r["decision"] != w["decision"]:
            flip = "%s (decision %s, candidate %s" % (r["id"], r["decision"], w["decision"])
            if far is None:
                flips.append(flip + ")")
            else:
                off.append("%s; %s)" % (flip, far))
        elif far is not None:
            off.append("%s (%s)" % (r["id"], far))
    return flips, off


def _check_feedback(ws, m, path):
    errors = []
    if "delivery_id" in m:
        errors += ws.ref_errors("delivery_id", m["delivery_id"], "delivery")
    if "candidate_id" in m:
        errors += ws.ref_errors("candidate_id", m["candidate_id"], "candidate")
    return errors


def _best_before_errors(ws, bb):
    best = ws.state["best"]
    if best is None:
        return [] if bb is None else ["best_before must be null: the workspace has no best version yet"]
    if bb is None:
        return ["best_before is null but the current best version is %s" % best]
    if bb["version"] != best:
        return ["best_before.version is %s but the current best version is %s" % (bb["version"], best)]
    v = ws.version(best)
    if v is None:
        return ["STATE best %s is missing from versions" % best]
    if bb["evaluation_id"] != v["evaluation_id"]:
        return ["best_before.evaluation_id is %s but the best version %s was evaluated in %s"
                % (bb["evaluation_id"], best, v["evaluation_id"])]
    return []


def _check_evaluation(ws, m, path):
    errors = ws.ref_errors("delivery_id", m["delivery_id"], "delivery")
    if not errors:
        d = ws.manifest(m["delivery_id"])
        for field in ("candidate_id", "evalset_id"):
            if d[field] != m[field]:
                errors.append("%s is %s but delivery %s has %s"
                              % (field, m[field], m["delivery_id"], d[field]))
        if m["uses_testset"] and "predictions_test" not in d["files"]:
            errors.append("uses_testset is true but delivery %s has no predictions_test" % m["delivery_id"])
    errors += ws.current_errors("evalset_id", m["evalset_id"], "evalset")
    errors += _best_before_errors(ws, m["best_before"])
    if m["uses_testset"]:
        return errors + _milestone_errors(ws, m) + ws.approval_errors("use_testset", "uses_testset")
    used, limit = ws.state["eval_use_count"], ws.task()["eval_use_limit"]
    if used >= limit:
        errors.append("eval_use_count %d has reached eval_use_limit %d for evalset %s: refresh the evalset "
                      "(needs a refresh_sets approval) before another evaluation"
                      % (used, limit, ws.current("evalset")))
    return errors


def _milestone_errors(ws, m):
    """A milestone evaluation (uses_testset) scores the best version and decides nothing (R27).

    It neither checks nor uses the evalset's eval_use_limit.
    """
    best = ws.state["best"]
    if best is None:
        return ["uses_testset is true but there is no best version yet: a milestone evaluation scores "
                "the best version"]
    errors = []
    v = ws.version(best)
    if v is not None and m["candidate_id"] != v["candidate_id"]:
        errors.append("uses_testset is true but candidate_id %s is not the candidate of the best version %s "
                      "(%s): a milestone evaluation scores the best version"
                      % (m["candidate_id"], best, v["candidate_id"]))
    if m["recommendation"]["decision"] != "keep":
        errors.append("recommendation.decision is %r but a milestone evaluation (uses_testset) must say "
                      "'keep': milestone evaluations are not promotion decisions"
                      % m["recommendation"]["decision"])
    return errors


def _check_recommendation(ws, m, path):
    """recommendation.decision must be what the promotion rule gives (milestones: see _milestone_errors)."""
    if m["uses_testset"]:
        return []
    errors = ws.ref_errors("candidate_id", m["candidate_id"], "candidate")
    if errors:
        return errors
    gate = ws.manifest(m["candidate_id"])["train_gate"]["passed"]
    decision, reason = decide_promotion(ws.task(), m, ws.state["best"] is not None, gate)
    rec = m["recommendation"]["decision"]
    if rec != decision:
        return ["recommendation.decision is %r but the promotion rule gives %r: %s" % (rec, decision, reason)]
    return []


def _check_plan(ws, m, path):
    """A plan cites a registered evaluation; evaluation_id null only after `rsi close-round`."""
    if m["evaluation_id"] is not None:
        return ws.ref_errors("evaluation_id", m["evaluation_id"], "evaluation")
    if ws.closed():
        return []
    rnd = ws.state["round"]
    d = ws.state["rounds"][str(rnd)]["decision"]
    how = "it was decided by `rsi promote` (%s)" % d["evaluation_id"] if d else "it has no decision"
    return ["evaluation_id is null, but round %d was not closed with `rsi close-round` (%s): a plan cites "
            "a registered evaluation, and only a round closed without evaluation takes a plan with "
            "evaluation_id null" % (rnd, how)]


# ---- reported numbers recomputed with rsi.metrics (R36) -----------------------

def score_form(labels, predictions, train=None, clusters=None, compare=None):
    """The `rsi score` command that computes a reported number, quoted in refusals; arguments are
    the files as `<artifact id>/<file>` (e.g. `<valset-0001>/rows.jsonl`)."""
    parts = ["rsi score --task <task.json> --labels %s --predictions %s" % (labels, predictions)]
    for option, value in (("--train", train), ("--clusters", clusters), ("--compare", compare)):
        if value is not None:
            parts.append("%s %s" % (option, value))
    return "`%s`" % " ".join(parts)


def unverified_names(kind, m, task):
    """Names in a candidate or evaluation manifest that rsi does not compute, in order of appearance.

    Registration skips them: unknown metrics and baselines, and for a candidate also
    the task's unknown baselines (the train gate is checked against the known ones).
    """
    names = []

    def add(name, known):
        if name not in known and name not in names:
            names.append(name)

    def baselines(reported):
        for b, values in reported.items():
            add(b, M.KNOWN_BASELINES)
            if b in M.KNOWN_BASELINES and isinstance(values, dict):
                for n in values:
                    add(n, M.KNOWN_METRICS)

    if kind == "candidate":
        for n in m["metrics_val"]:
            add(n, M.KNOWN_METRICS)
        baselines(m["baselines_val"])
        for b in task["metrics"]["baselines"]:
            add(b, M.KNOWN_BASELINES)
    elif kind == "evaluation" and m["uses_testset"]:
        for n in m.get("metrics_test", {}):
            add(n, M.KNOWN_METRICS)
    elif kind == "evaluation":
        for n in list(m["metrics"]["candidate"]) + list(m["metrics"]["best"] or {}):
            add(n, M.KNOWN_METRICS)
        baselines(m["metrics"]["baselines"])
        if m["best_before"] is not None:  # a delta for the primary metric is expected (and checked when known)
            add(task["metrics"]["primary"], M.KNOWN_METRICS)
    return names


def _same_ids(preds, labels):
    ids = [r.get("id") for r in preds]
    return len(set(ids)) == len(ids) and set(ids) == set(r["id"] for r in labels)


def _mismatch(field, reported, value, source, form):
    if not is_number(reported):
        return ["%s: reported %r is not a number" % (field, reported)]
    if abs(reported - value) <= EPS:
        return []
    return ["%s: reported %r, recomputed %r from %s; compute the reported numbers with %s"
            % (field, reported, value, source, form)]


def _verify_metrics(field, reported, preds, labels, task, source, form):
    """Errors for the known metrics in `reported` (name -> number) that rsi.metrics does not reproduce."""
    errors = []
    for name, value in reported.items():
        if name not in M.KNOWN_METRICS:
            continue
        try:
            want = M.score(name, preds, labels, task)
        except ValueError as e:
            errors.append("%s.%s: cannot be recomputed from %s (%s)" % (field, name, source, e))
            continue
        errors += _mismatch("%s.%s" % (field, name), value, want, source, form)
    return errors


def _verify_baselines(field, reported, train, labels, task, groups, source, form):
    errors = []
    for b, values in reported.items():
        if b not in M.KNOWN_BASELINES or not isinstance(values, dict):
            continue
        try:
            preds = M.baseline_predictions(b, train, labels, task, groups)
        except ValueError as e:
            errors.append("%s.%s: cannot be recomputed from %s (%s)" % (field, b, source, e))
            continue
        errors += _verify_metrics("%s.%s" % (field, b), values, preds, labels, task, source, form)
    return errors


def _gate_errors(task, passed, preds, train, labels, form):
    """train_gate.passed must say whether the primary metric beats every baseline listed in the
    task, strictly in its direction. With a listed baseline rsi does not compute, only a pass
    that a known baseline contradicts is refused."""
    primary, higher = task["metrics"]["primary"], task["metrics"]["higher_is_better"]
    listed = task["metrics"]["baselines"]
    known = [b for b in listed if b in M.KNOWN_BASELINES]
    if primary not in M.KNOWN_METRICS or not known:
        return []
    try:
        value = M.score(primary, preds, labels, task)
        base = [(b, M.score(primary, M.baseline_predictions(b, train, labels, task), labels, task))
                for b in known]
    except ValueError:
        return []  # reported by the metric checks
    behind = [(b, v) for b, v in base if not (value > v if higher else value < v)]
    if passed and behind:
        return ["train_gate.passed is true but val %s %r does not beat baseline %s (%s %r): the train gate "
                "passes only when the primary metric beats every baseline listed in the task, strictly (%s is "
                "better); compute the numbers with %s"
                % (primary, value, behind[0][0], primary, behind[0][1], "higher" if higher else "lower", form)]
    if not passed and not behind and len(known) == len(listed):
        return ["train_gate.passed is false but val %s %r beats every baseline listed in the task (%s): the "
                "train gate passed; compute the numbers with %s"
                % (primary, value, ", ".join("%s %r" % bv for bv in base), form)]
    return []


def _check_candidate_numbers(ws, m, path):
    """metrics_val and baselines_val as rsi.metrics computes them on the current valset (baselines
    from the candidate's trainset), and train_gate.passed as those numbers give it."""
    val = ws.current("valset")
    if val is None or m["valset_id"] != val:
        return []  # refused by _check_candidate
    labels = ws.rows(val, "rows")
    preds = _src_rows(path, m, "predictions_val")
    if not _same_ids(preds, labels):
        return []  # coverage: refused by _check_candidate
    task, train_id = ws.task(), m["trainset_id"]
    form = score_form("<%s>/rows.jsonl" % val, "<candidate>/predictions_val.jsonl", "<%s>/rows.jsonl" % train_id)
    errors = _verify_metrics("metrics_val", m["metrics_val"], preds, labels, task,
                             "predictions_val against %s" % val, form)
    if not ws.is_kind(train_id, "trainset"):
        return errors
    train = ws.rows(train_id, "rows")
    errors += _verify_baselines("baselines_val", m["baselines_val"], train, labels, task, None,
                                "%s scored on %s" % (train_id, val), form)
    return errors + _gate_errors(task, m["train_gate"]["passed"], preds, train, labels, form)


def _best_delivery(ws, m, v):
    """The best version's predictions on this evaluation's evalset: its own delivery when that
    is on the same evalset, else a delivery in parents of its candidate on the current evalset."""
    if ws.manifest(v["delivery_id"])["evalset_id"] == m["evalset_id"]:
        return v["delivery_id"]
    for p in m["parents"]:
        if ws.is_kind(p, "delivery"):
            d = ws.manifest(p)
            if d["candidate_id"] == v["candidate_id"] and d["evalset_id"] == ws.current("evalset"):
                return p
    return None


def _check_evaluation_numbers(ws, m, path):
    """metrics.candidate, metrics.baselines, metrics.best and deltas[primary] as rsi.metrics computes
    them on the evalset labels; a milestone evaluation's metrics_test on the testset labels."""
    if m["uses_testset"]:
        return _milestone_numbers(ws, m)
    did, eid = m["delivery_id"], m["evalset_id"]
    if not (ws.is_kind(did, "delivery") and ws.is_kind(eid, "evalset")) or ws.manifest(did)["evalset_id"] != eid:
        return []  # refused by _check_evaluation
    labels, inputs = ws.rows(eid, "labels"), ws.rows(eid, "inputs")
    preds = ws.rows(did, "predictions_eval")
    if not _same_ids(preds, labels):
        return []
    task, cid = ws.task(), m["candidate_id"]
    tid = ws.manifest(cid)["trainset_id"] if ws.is_kind(cid, "candidate") else None
    best = ws.state["best"]
    v = ws.version(best) if best is not None and m["best_before"] is not None else None
    bid = _best_delivery(ws, m, v) if v is not None else None

    def form(predictions=did, compare=bid):
        return score_form("<%s>/labels.jsonl" % eid, "<%s>/predictions_eval.jsonl" % predictions,
                          "<%s>/rows.jsonl" % (tid or "trainset"), "<%s>/inputs.jsonl" % eid,
                          compare and "<%s>/predictions_eval.jsonl" % compare)

    errors = _verify_metrics("metrics.candidate", m["metrics"]["candidate"], preds, labels, task,
                             "%s predictions_eval against %s labels" % (did, eid), form())
    if ws.is_kind(tid, "trainset"):
        errors += _verify_baselines("metrics.baselines", m["metrics"]["baselines"], ws.rows(tid, "rows"),
                                    labels, task, M.groups_of(inputs), "%s scored on %s" % (tid, eid), form())
    if v is None:
        return errors  # no best version: best_before is checked against STATE by _best_before_errors
    if bid is None:
        return errors + ["metrics.best: the best version %s's delivery %s is on %s, not %s, and parents list no "
                         "delivery of its candidate %s on %s: deliver the best version on the current evalset "
                         "first (package %s on the current evalset inputs), then list that delivery in parents"
                         % (best, v["delivery_id"], ws.manifest(v["delivery_id"])["evalset_id"], eid,
                            v["candidate_id"], eid, v["candidate_id"])]
    best_preds = ws.rows(bid, "predictions_eval")
    if not _same_ids(best_preds, labels):
        return errors
    if isinstance(m["metrics"]["best"], dict):
        errors += _verify_metrics("metrics.best", m["metrics"]["best"], best_preds, labels, task,
                                  "%s predictions_eval (best version %s) against %s labels" % (bid, best, eid),
                                  form(bid, None))
    primary = task["metrics"]["primary"]
    reported = m["deltas"].get(primary)
    if primary not in M.KNOWN_METRICS or not isinstance(reported, dict):
        return errors
    ci = M.ci_settings(task)
    source = ("%s minus %s (best version %s) on %s, paired cluster bootstrap with resamples %d, seed %d, level %r, "
              "clusters from the %s inputs" % (did, bid, best, eid, ci["resamples"], ci["seed"], ci["level"], eid))
    try:
        want = M.paired_delta(primary, preds, best_preds, labels, task, M.clusters_of(inputs), **ci)
    except ValueError as e:
        return errors + ["deltas.%s: cannot be recomputed from %s (%s)" % (primary, source, e)]
    for key in ("delta", "ci_low", "ci_high"):
        errors += _mismatch("deltas.%s.%s" % (primary, key), reported[key], want[key], source, form())
    return errors


def _milestone_numbers(ws, m):
    """A milestone evaluation's metrics_test against the testset labels."""
    did = m["delivery_id"]
    if "metrics_test" not in m or not ws.is_kind(did, "delivery"):
        return []
    d = ws.manifest(did)
    tid = d.get("testset_id")
    if "predictions_test" not in d["files"] or not ws.is_kind(tid, "testset"):
        return []  # refused by _check_evaluation / _check_delivery
    labels, preds = ws.rows(tid, "labels"), ws.rows(did, "predictions_test")
    if not _same_ids(preds, labels):
        return []
    return _verify_metrics("metrics_test", m["metrics_test"], preds, labels, ws.task(),
                           "%s predictions_test against %s labels" % (did, tid),
                           score_form("<%s>/labels.jsonl" % tid, "<%s>/predictions_test.jsonl" % did))


def _check_record(ws, m, path):
    errors = []
    for aid in m["artifacts"]:
        errors += ws.ref_errors("artifacts", aid)
    return errors


# Workspace rules checked at registration, on top of validate_artifact and the
# common checks. Each kind maps to a list so later rules can be appended.
REGISTER_CHECKS = {
    "task": [_check_task_change],
    "trainset": [_check_disjoint, _check_parent_trainset],
    "valset": [_check_refresh, _check_disjoint, _check_corpus_registered, _check_from_corpus],
    "evalset": [_check_refresh, _check_disjoint, _check_new_evalset_rows, _check_corpus_registered, _check_from_corpus],
    "testset": [_check_refresh, _check_disjoint, _check_corpus_registered, _check_from_corpus],
    "corpus": [_check_refresh, _check_corpus_first],
    "pool": [_check_refresh, _check_disjoint, _check_corpus_registered, _check_from_corpus],
    "candidate": [_check_attempts, _check_candidate, _check_budget, _check_candidate_numbers],
    "search": [_check_attempts, _check_search],
    "delivery": [_check_delivery],
    "feedback": [_check_feedback],
    "evaluation": [_check_evaluation, _check_evaluation_numbers, _check_recommendation],
    "plan": [_check_plan],
    "record": [_check_record],
}
# Checks that read a large file (the whole corpus): each comes last in its kind's list and runs only when every check
# before it passed, so a registration refused on cheaper grounds does not read the corpus.
COSTLY_CHECKS = (_check_from_corpus,)


# ---- one-line summaries for `rsi register` and `rsi status`: fn(ws, manifest) ----

def _num(x):
    return "%.6g" % x if is_number(x) else "none"


def _summary_trainset(ws, m):
    q = m["quality"]
    if q["passed"]:
        return "quality gate: passed"
    failed = [c["name"] for c in q["checks"] if not c["passed"]]
    return "quality gate: failed (%s); never current" % ", ".join(failed)


def _summary_candidate(ws, m):
    primary = ws.task()["metrics"]["primary"]
    g = m["train_gate"]
    val = "val %s %s" % (primary, _num(m["metrics_val"].get(primary)))
    if g["passed"]:
        return "train gate: passed; %s" % val
    return "train gate: failed (%s); %s; cannot be delivered" % (g["detail"], val)


def _summary_search(ws, m):
    g = m["gain_vs_default"]
    return ("tune gate: passed; gain vs default %s %s (best %s, default %s)"
            % (g["metric"], "%+.6g" % g["delta"], m["best_candidate"], m["default_candidate"]))


def _summary_evaluation(ws, m):
    return "milestone evaluation on the testset; not a promotion decision" if m["uses_testset"] else None


SUMMARIES = {
    "trainset": _summary_trainset,
    "candidate": _summary_candidate,
    "search": _summary_search,
    "evaluation": _summary_evaluation,
}


# ---- STATE updates after a successful registration: fn(ws, manifest, id) ---

def _effect_task(ws, task, aid):
    s = ws.state
    s["task_id"], s["task_version"] = task["task_id"], task["version"]
    s["current"]["task"] = aid


def _effect_set_current(ws, m, aid):
    ws.state["current"][m["kind"]] = aid


def _effect_trainset(ws, m, aid):
    # A trainset that failed its quality gate is kept as evidence, never current (R28).
    if m["quality"]["passed"]:
        _effect_set_current(ws, m, aid)


def _effect_evalset(ws, m, aid):
    _effect_set_current(ws, m, aid)
    ws.state["eval_use_count"] = 0  # the use limit counts uses of the current evalset


def _effect_candidate(ws, m, aid):
    # A source whose refusal counted it (budget_used.refused) leaves refused; its cost is added unless that refusal
    # counted the same cost (the same training, fixed in place).
    used = ws.state["budget_used"]
    src = ws.entry(aid)["source"]
    used["compute_hours"] = used["compute_hours"] + _added_hours(ws, src, m["cost"]["compute_hours"])
    refused = used.get("refused") or {}
    if src in refused:
        used["refused"] = dict((k, v) for k, v in refused.items() if k != src)


def _effect_evaluation(ws, m, aid):
    if not m["uses_testset"]:  # a milestone evaluation does not count toward eval_use_limit
        ws.state["eval_use_count"] += 1


def _effect_plan(ws, m, aid):
    s = ws.state
    s["rounds"][str(s["round"])]["plan_id"] = aid


REGISTER_EFFECTS = {
    "task": _effect_task,
    "trainset": _effect_trainset,
    "valset": _effect_set_current,
    "evalset": _effect_evalset,
    "testset": _effect_set_current,
    "corpus": _effect_set_current,
    "pool": _effect_set_current,
    "candidate": _effect_candidate,
    "evaluation": _effect_evaluation,
    "plan": _effect_plan,
}
