"""Artifact directory validation (stdlib only)."""
import hashlib
import json
import os
import re

from rsi import datasplit as DS
from rsi import rows as R
from rsi import space as S
from rsi.rules import EPS, check_attribution, is_number
from rsi.schema import load_schema, validate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # a relative train command is under it

ARTIFACT_FILES = {
    "trainset": ["rows", "quality_report"],
    "valset": ["rows", "inputs"],
    "evalset": ["inputs", "labels"],
    "testset": ["inputs", "labels"],
    "corpus": ["rows", "script"],
    "pool": ["rows"],
    "candidate": ["predictions_val"],
    "search": [],
    "delivery": ["predictions_eval", "predictions_val"],
    "feedback": ["trace"],
    "evaluation": ["report"],
}


def load_manifest(path):
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def merged_schema(kind):
    """Kind-specific schema with `common` properties and required merged in."""
    common = load_schema("common")
    sch = load_schema(kind)
    props = dict(common["properties"])
    props.update(sch.get("properties", {}))
    required = list(common.get("required", []))
    required += [r for r in sch.get("required", []) if r not in required]
    return {"type": "object", "required": required, "properties": props,
            "additionalProperties": False}


def _count_rows(fp):
    with open(fp, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _is_remote(location):
    return re.match(r"^[^/:]+:/", location) is not None


def _sha256(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Kinds whose source directory registration moves into the workspace instead of copying (R37): the
# sealed sets' labels, and the corpus (every labelled row), then exist only under artifacts/, not in the
# producing module's output directory.
MOVED_KINDS = ("evalset", "testset", "corpus")


def _confined(path, rel, field):
    """(path joined to the artifact directory, None) when `rel` is a relative path that does not climb out
    of the artifact directory `path`, else (None, error naming `field`).

    The check is on the path as written. Symbolic links are a separate rule (link_errors): registration
    copies what they point to, and only a kind that is moved in (MOVED_KINDS) refuses links that leave the directory."""
    if os.path.isabs(rel):
        return None, "%s: path %s must be relative" % (field, rel)
    norm = os.path.normpath(rel)
    if norm == os.pardir or norm.startswith(os.pardir + os.sep):
        return None, "%s: path %s escapes the artifact directory" % (field, rel)
    return os.path.join(path, rel), None


def link_errors(path, moved):
    """Errors for links under directory `path` (walked without following links).

    Any kind: a symbolic link to nothing. Registration copies other kinds following their links, so the
    registered copy holds what they point to. An evalset, testset or corpus (`moved`, MOVED_KINDS) is handed
    over instead: its directory is moved, so a symbolic link leaving the directory, or a file with a second
    name (hard link), would leave the content, the labels, where it was; both are refused.
    """
    root = os.path.realpath(path)
    errors = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in sorted(dirnames + filenames):
            fp = os.path.join(dirpath, name)
            rel = os.path.relpath(fp, path)
            if os.path.islink(fp):
                target = os.path.realpath(fp)
                if not os.path.exists(fp):
                    errors.append("%s is a symbolic link to %s, which does not exist" % (rel, target))
                elif moved and os.path.commonpath([root, target]) != root:
                    errors.append("%s is a symbolic link to %s, outside the directory: an evalset, testset or corpus "
                                  "is moved into the workspace at registration and the linked file would stay "
                                  "behind; put the file itself in the directory" % (rel, target))
            elif moved and name in filenames:
                n = os.lstat(fp).st_nlink
                if n > 1:
                    errors.append("%s has %d hard links: an evalset, testset or corpus is moved into the workspace "
                                  "at registration and the other name would keep the content; write the file without "
                                  "a second name" % (rel, n))
    return errors


def _str_ids(rows):
    """The string ids of `rows`; a row whose id is not a string is reported by the row checks."""
    return set(r.get("id") for r in rows if isinstance(r.get("id"), str))


def _common_checks(path, manifest):
    errors = link_errors(path, manifest.get("kind") in MOVED_KINDS)
    if errors:
        return errors
    files = manifest.get("files", {})
    for name, spec in files.items():
        fp, err = _confined(path, spec["path"], "files.%s" % name)
        if err:
            errors.append(err)
            continue
        if not os.path.isfile(fp):
            errors.append("files.%s: file %s not found" % (name, spec["path"]))
            continue
        if "rows" in spec:
            try:
                n = _count_rows(fp)
            except (OSError, UnicodeDecodeError) as e:
                errors.append("files.%s: cannot read %s (%s)" % (name, spec["path"], e))
                continue
            if n != spec["rows"]:
                errors.append("files.%s: manifest says %d rows, %s has %d"
                              % (name, spec["rows"], spec["path"], n))
    for name, spec in manifest.get("external", {}).items():
        loc = spec["location"]
        if _is_remote(loc):
            continue
        if not os.path.isfile(loc):
            errors.append("external.%s: local file %s does not exist" % (name, loc))
        elif _sha256(loc).lower() != spec["sha256"].lower():
            errors.append("external.%s: sha256 mismatch for %s" % (name, loc))
    return errors


def _read(path, manifest, logical):
    return R.read_jsonl(os.path.join(path, manifest["files"][logical]["path"]))


def _required_files(kind, manifest):
    return ["files: missing required logical file '%s'" % n
            for n in ARTIFACT_FILES.get(kind, []) if n not in manifest.get("files", {})]


def _check_full(kind):
    def check(path, manifest, task):
        errors = _required_files(kind, manifest)
        if errors:
            return errors
        try:
            full = _read(path, manifest, "rows")
            if not full:
                return [_empty_error(kind, manifest, "rows")]
            errors = R.check_full_rows(full, task) + _test_split_errors(full)
            if kind == "valset":
                errors += _inputs_match_rows(_read(path, manifest, "inputs"), full, task)
        except (ValueError, OSError) as e:
            return [str(e)]
        if kind == "trainset":
            q = manifest["quality"]
            expected = all(c["passed"] for c in q["checks"])
            if q["passed"] != expected:
                errors.append("quality.passed is %s but checks give %s" % (q["passed"], expected))
        return errors
    return check


def _test_split_errors(full):
    """A trainset's, valset's or pool's rows whose split names a test split: onejev-harness refuses to train or validate
    on them (its rule: the split, stripped and in lower case, starts with test), short of an approved --allow-test."""
    bad = [r.get("id") for r in full
           if isinstance(r, dict) and isinstance(r.get("split"), str) and r["split"].strip().lower().startswith("test")]
    if not bad:
        return []
    return ["split: %d rows have a split starting with test, e.g. %s: onejev-harness does not train or validate on "
            "such rows; name the split e.g. train or val" % (len(bad), ", ".join(str(i) for i in bad[:3]))]


def _inputs_match_rows(inputs, full, task):
    """A valset's inputs.jsonl: valid input rows, each the input row of its rows.jsonl row (R.input_row_of: only the
    top-level keys R.STATE_INPUT_KEYS, so no target, and meta reduced to R.INPUT_META)."""
    errors = ["inputs: " + e for e in R.check_input_rows(inputs, task)]
    by_id = dict((r["id"], r) for r in full if isinstance(r.get("id"), str))
    ids = _str_ids(inputs)
    if ids != set(by_id):
        return errors + ["inputs: ids differ from rows.jsonl (%d only in inputs, %d only in rows)"
                         % (len(ids - set(by_id)), len(set(by_id) - ids))]
    differ = [r["id"] for r in inputs if isinstance(r.get("id"), str) and r != R.input_row_of(by_id[r["id"]])]
    if differ:
        errors.append("inputs: %d rows are not the input row of their rows.jsonl row (only the top-level keys %s, "
                      "meta only %s), e.g. %s" % (len(differ), ", ".join(R.STATE_INPUT_KEYS), ", ".join(R.INPUT_META),
                                                  ", ".join(str(i) for i in differ[:3])))
    return errors


# The name prefix of what modules/data/dset writes before putting it in place (dset.write.STAGING): in a corpus
# directory such an entry is what a dset run that stopped part way left, and must not be registered with it.
DSET_STAGING = ".dset-"


def _check_corpus(path, manifest, task):
    """A corpus: the data module's preprocessing script (files.script) and every labelled row it wrote (files.rows).
    The rows are valid full rows whatever their split (a corpus is not trained on), fit the split rule and
    data.rows.expect (rsi.datasplit.corpus_errors); `sources` names at least one path when the task's data.sources
    names any, and every path in it is, or lies inside, a path of the task's data.sources (the raw data). The script is
    not run; its sha256 is recorded at registration like every file's. The corpus holds at least one row, and no
    DSET_STAGING entry."""
    errors = _required_files("corpus", manifest)
    if errors:
        return errors
    left = sorted(n for n in os.listdir(path) if n.startswith(DSET_STAGING))
    if left:
        return ["%s: left by a dset run that stopped part way; delete it (and run dset sets again)" % ", ".join(left)]
    try:
        full = _read(path, manifest, "rows")
    except (ValueError, OSError) as e:
        return [str(e)]
    if not full:
        return [_empty_error("corpus", manifest, "rows")]
    errors = R.check_full_rows(full, task) + DS.corpus_errors(full, task)
    allowed = [os.path.normpath(p) for p in DS.source_paths(task)]
    if allowed and not manifest["sources"]:
        errors.append("sources: empty, but the task's data.sources names the raw data (%s): list the paths the "
                      "preprocessing read" % ", ".join(allowed))
    for i, src in enumerate(manifest["sources"]):
        p = os.path.normpath(src)
        if not any(p == a or p.startswith(a.rstrip(os.sep) + os.sep) for a in allowed):
            errors.append("sources[%d]: %s is not a path of the task's data.sources, nor inside one (%s): the "
                          "preprocessing reads only the raw data the task names"
                          % (i, src, ", ".join(allowed) or "none"))
    return errors


def _empty_error(kind, manifest, logical):
    """A data set (trainset, valset, evalset, testset, pool, corpus) holds at least one row: an empty one would
    register and then score, train or judge on nothing."""
    return ("files.%s: %s holds no rows; %s %s needs at least one"
            % (logical, manifest["files"][logical]["path"], "an" if kind[0] in "aeiou" else "a", kind))


def _check_split(path, manifest, task):
    errors = _required_files("evalset", manifest)
    if errors:
        return errors
    try:
        inputs = _read(path, manifest, "inputs")
        labels = _read(path, manifest, "labels")
    except (ValueError, OSError) as e:
        return [str(e)]
    if not inputs:
        return [_empty_error(manifest["kind"], manifest, "inputs")]
    errors = ["inputs: " + e for e in R.check_input_rows(inputs, task)]
    errors += ["labels: " + e for e in R.check_label_rows(labels, task)]
    a, b = _str_ids(inputs), _str_ids(labels)
    if a != b:
        errors.append("ids differ between inputs and labels (%d only in inputs, %d only in labels)"
                      % (len(a - b), len(b - a)))
    return errors


def _pred_errors(path, manifest, task, logical):
    try:
        return ["%s: %s" % (logical, e)
                for e in R.check_predictions(_read(path, manifest, logical), task)]
    except (ValueError, OSError) as e:
        return [str(e)]


def _trace_errors(path, manifest, task, logical="trace"):
    try:
        return ["%s: %s" % (logical, e)
                for e in R.check_feedback_rows(_read(path, manifest, logical), task)]
    except (ValueError, OSError) as e:
        return [str(e)]


def _predict_errors(path, manifest):
    """model.predict: an executable file inside the candidate directory, called as
    `<candidate dir>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>`."""
    rel = manifest["model"]["predict"]
    fp, err = _confined(path, rel, "model.predict")
    if err:
        return [err + " (the prediction entry lives in the candidate directory and loads files elsewhere "
                      "only through `external` entries)"]
    if not os.path.isfile(fp):
        return ["model.predict: %s not found in the candidate directory" % rel]
    if not os.access(fp, os.X_OK):
        return ["model.predict: %s is not executable (callers run `<candidate dir>/%s --inputs <inputs.jsonl> "
                "--out <predictions.jsonl>`; chmod +x it)" % (rel, rel)]
    return []


def _code_version_errors(manifest):
    """code_version.sha256 must be the sha256 of code_version.entry when the entry is an absolute path
    to a file on this machine; other entries (host:/path, names, paths not here) are not checked."""
    cv = manifest["code_version"]
    entry = cv["entry"]
    if not os.path.isabs(entry) or not os.path.isfile(entry):
        return []
    actual = _sha256(entry)
    if cv["sha256"] is None:
        return ["code_version.sha256 is null but code_version.entry %s is a local file (sha256 %s): write the "
                "sha256 of the training code that produced the candidate" % (entry, actual)]
    if cv["sha256"].lower() != actual:
        return ["code_version.sha256 %s does not match code_version.entry %s (sha256 %s): the file changed after "
                "training, or the candidate was trained with other code" % (cv["sha256"], entry, actual)]
    return []


def _command_path(p):
    """A train command path as one file: relative to the repository root when relative, links resolved."""
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(REPO_ROOT, p))


def _train_command_errors(manifest, task):
    """With the task's resources.train_command set, the candidate was trained by it: code_version.entry names the
    same file (the same text, or the same file once a relative path is taken from the repository root)."""
    command = (task.get("resources") or {}).get("train_command")
    entry = manifest["code_version"]["entry"]
    if not command or entry == command or _command_path(entry) == _command_path(command):
        return []
    return ["code_version.entry %s is not the task's resources.train_command %s: a candidate is trained by the "
            "task's train command" % (entry, command)]


def _config_request_errors(manifest, task):
    """With a search_space in the task, the candidate's config_request lies in it, a path the space does not name
    keeping the default configuration's value (rsi.space)."""
    space = task.get("search_space")
    if space is None:
        return []
    if "config_request" not in manifest:
        return ["config_request: missing: the task has a search_space, so a candidate says which configuration of "
                "it it was trained with ({\"base\", \"recipe\", \"overrides\", \"seed\", \"max_steps\"})"]
    return S.request_errors(manifest["config_request"], space, S.default_request(task))


def _check_candidate(path, manifest, task):
    errors = _required_files("candidate", manifest)
    if errors:
        return errors
    primary = task["metrics"]["primary"]
    if primary not in manifest["metrics_val"]:
        errors.append("metrics_val: missing primary metric '%s'" % primary)
    errors += _predict_errors(path, manifest) + _code_version_errors(manifest)
    errors += _train_command_errors(manifest, task) + _config_request_errors(manifest, task)
    return errors + _pred_errors(path, manifest, task, "predictions_val")


def _check_search(path, manifest, task):
    """Trials, budget and the tune gate: the best trial is not worse than the default (R28).

    A trial with status "ok" has a candidate and a number metric_val (the
    primary metric on the valset); other trials may have neither (M3). The
    best and default candidates are ok trials.
    """
    errors = []
    trials = manifest["trials"]
    ok = {}
    for i, t in enumerate(trials):
        if t["status"] != "ok":
            continue
        if t["candidate_id"] is None:
            errors.append("trials[%d]: an ok trial needs a candidate_id" % i)
        if not is_number(t["metric_val"]):
            errors.append("trials[%d]: an ok trial needs a number metric_val" % i)
        if t["candidate_id"] is not None:
            ok[t["candidate_id"]] = t
    ids = [t["candidate_id"] for t in trials if t["candidate_id"] is not None]
    if len(set(ids)) != len(ids):
        errors.append("trials: duplicate candidate_id")
    for key in ("default_candidate", "best_candidate"):
        if manifest[key] not in ids:
            errors.append("%s: %r is not in trials" % (key, manifest[key]))
        elif manifest[key] not in ok:
            errors.append("%s: %r is not the candidate of an ok trial" % (key, manifest[key]))
    used = sum(t["compute_hours"] for t in trials)
    if used > manifest["budget"]["compute_hours"] + EPS:
        errors.append("budget: trials use %s compute_hours, over budget %s"
                      % (used, manifest["budget"]["compute_hours"]))
    if errors:
        return errors
    return _tune_gate_errors(manifest, task, ok)


def _tune_gate_errors(manifest, task, ok):
    primary = task["metrics"]["primary"]
    higher = task["metrics"]["higher_is_better"]
    bid, did = manifest["best_candidate"], manifest["default_candidate"]
    best, default = ok[bid]["metric_val"], ok[did]["metric_val"]
    gain = manifest["gain_vs_default"]
    errors = []
    if (best < default - EPS) if higher else (best > default + EPS):
        errors.append("best_candidate %s (metric_val %s) is worse than default_candidate %s (metric_val %s) on "
                      "%s (%s is better): the tune gate needs the best not worse than the default; hand back "
                      "the default as best_candidate"
                      % (bid, best, did, default, primary, "higher" if higher else "lower"))
    if gain["metric"] != primary:
        errors.append("gain_vs_default.metric is %r but the primary metric is %r" % (gain["metric"], primary))
    if abs(gain["delta"] - (best - default)) > EPS:
        errors.append("gain_vs_default.delta is %s but best - default metric_val is %s (%s - %s)"
                      % (gain["delta"], best - default, best, default))
    return errors


EXTERNAL_PREFIX = "external:"


def _package_errors(path, manifest):
    """package.location: a path inside the delivery directory, or `external:<name>` naming an
    `external` entry (which carries the sha256); never an absolute path elsewhere."""
    loc = manifest["package"]["location"]
    if loc.startswith(EXTERNAL_PREFIX):
        name = loc[len(EXTERNAL_PREFIX):]
        if name not in manifest.get("external", {}):
            return ["package.location %s: no external entry '%s'; list the package under `external` with its "
                    "location and sha256" % (loc, name)]
        return []
    fp, err = _confined(path, loc, "package.location")
    if err:
        return [err + " (the package goes in the delivery directory, or in an `external` entry named by "
                      "`external:<name>`)"]
    if not os.path.exists(fp):
        return ["package.location: %s not found in the delivery directory" % loc]
    return []


def _check_delivery(path, manifest, task):
    errors = _required_files("delivery", manifest)
    if errors:
        return errors
    files = manifest["files"]
    if not manifest["runtime"]["consistent_with_offline"]:
        errors.append("runtime.consistent_with_offline: false: the package's predictions_val do not reproduce the "
                      "candidate's; fix the package (registration checks the reproduction itself)")
    if "predictions_test" in files and "testset_id" not in manifest:
        errors.append("testset_id: required when predictions_test is present")
    errors += _package_errors(path, manifest)
    errors += _pred_errors(path, manifest, task, "predictions_eval")
    errors += _pred_errors(path, manifest, task, "predictions_val")
    if "predictions_test" in files:
        errors += _pred_errors(path, manifest, task, "predictions_test")
    if "trace" in files:
        errors += _trace_errors(path, manifest, task)
    return errors


def _check_feedback(path, manifest, task):
    errors = _required_files("feedback", manifest)
    if "delivery_id" not in manifest and "candidate_id" not in manifest:
        errors.append("delivery_id or candidate_id: at least one is required")
    if errors:
        return errors
    return _trace_errors(path, manifest, task)


def _check_evaluation(path, manifest, task):
    errors = _required_files("evaluation", manifest)
    if errors:
        return errors
    primary = task["metrics"]["primary"]
    metrics = manifest["metrics"]
    if primary not in metrics["candidate"]:
        errors.append("metrics.candidate: missing primary metric '%s'" % primary)
    if manifest["best_before"] is not None:
        if metrics["best"] is None:
            errors.append("metrics.best: required when best_before is set")
        elif primary not in metrics["best"]:
            errors.append("metrics.best: missing primary metric '%s' (required when best_before is set)" % primary)
        if primary not in manifest["deltas"]:
            errors.append("deltas: missing primary metric '%s'" % primary)
    # A milestone evaluation (uses_testset) also reports the testset metrics.
    if manifest["uses_testset"]:
        if "metrics_test" not in manifest:
            errors.append("metrics_test: required when uses_testset is true")
        elif primary not in manifest["metrics_test"]:
            errors.append("metrics_test: missing primary metric '%s'" % primary)
    elif "metrics_test" in manifest:
        errors.append("metrics_test: only allowed when uses_testset is true")
    for i, f in enumerate(manifest["failures"]):
        errors += ["failures[%d]: %s" % (i, e) for e in check_attribution(f["attribution"], f["module"])]
    return errors


def _check_plan(path, manifest, task):
    errors = []
    for i, it in enumerate(manifest["issues"]):
        errors += ["issues[%d]: %s" % (i, e) for e in check_attribution(it["attribution"], it["module"])]
    if manifest["entry_module"] not in set(it["module"] for it in manifest["issues"]):
        errors.append("entry_module: '%s' is not a module of any issue" % manifest["entry_module"])
    return errors


KIND_CHECKS = {
    "trainset": _check_full("trainset"),
    "valset": _check_full("valset"),
    "evalset": _check_split,
    "testset": _check_split,
    "corpus": _check_corpus,
    "pool": _check_full("pool"),
    "candidate": _check_candidate,
    "search": _check_search,
    "delivery": _check_delivery,
    "feedback": _check_feedback,
    "evaluation": _check_evaluation,
    "plan": _check_plan,
    "record": lambda path, manifest, task: [],
}


def validate_artifact(kind, path, task):
    """Return a list of error strings (empty when the artifact is valid)."""
    if kind not in KIND_CHECKS:
        return ["unknown kind '%s'" % kind]
    try:
        manifest = load_manifest(path)
    except (OSError, ValueError) as e:
        return ["cannot load manifest.json in %s: %s" % (path, e)]
    errors = validate(manifest, merged_schema(kind))
    if isinstance(manifest, dict) and manifest.get("kind") != kind:
        errors.append("manifest kind is %r, expected %r" % (manifest.get("kind"), kind))
    if errors:
        return errors
    errors = _common_checks(path, manifest)
    if errors:
        return errors
    return KIND_CHECKS[kind](path, manifest, task)
