"""python -m dset <command> ...   (PYTHONPATH=<repo root>/modules/data)

sets     split the corpus rows by the task's data.split, with --split (the agent's rule) merged under it: write the
         testset, evalset, valset, pool and trainset under --out; the corpus manifest and split report when the corpus
         has no manifest yet; the task version with the filled rule (--out/task/) when the task's rule was incomplete
grow     a new trainset: the current one (--trainset), --add N rows drawn from the pool, the rows of --with
check    report the rows, duplicate ids, duplicate inputs, inputs with more than one hard label and the overlap with
         each --against file; --write a cleaned trainset (--mode drop or merge)
refresh  a new evalset of rows in the evalset's range that no set, the pool or an --exclude file holds

Every command prints one JSON summary on stdout. Exit codes: 0 done; 1 an error in the input or the rule (stderr
`dset: error: <reason>`, nothing written); 2 the trainset quality gate failed (sets only: nothing written). An output
directory must not exist yet: dset overwrites nothing. --dry-run prints the summary and writes nothing.
"""
import argparse
import copy
import json
import os
import sys

from rsi import datasplit as DS
from rsi import rows as R

from . import DSET_VERSION, DsetError
from . import quality as Q
from . import split as S
from . import write as W

SETS = S.REPORTED  # the directories sets writes, in this order: testset, evalset, valset, pool, trainset
MODES = {"drop": "dset check --mode drop: the rows of inputs with more than one hard label dropped, of any other input "
                 "only the first row kept",
         "merge": "dset check --mode merge: the rows of each input merged into one, target their n_obs-weighted mean, "
                  "n_obs their sum"}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise DsetError("%s (see %s --help)" % (message, self.prog))


def _round(text):
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1 (got %s)" % text)
    return n


def _count(text):
    try:
        n = int(text)
    except ValueError:
        n = -1
    if n < 0:
        raise argparse.ArgumentTypeError("expected an integer >= 0 (got %s)" % text)
    return n


def build_parser():
    p = _Parser(prog="dset", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version="dset %s" % DSET_VERSION)
    sub = p.add_subparsers(dest="cmd", metavar="command", required=True)

    s = sub.add_parser("sets", help="split the corpus into the four sets and the pool by the rule")
    s.add_argument("--task", required=True, help="task.json (its TASK.md next to it)")
    s.add_argument("--corpus", required=True, help="the corpus directory: rows.jsonl and the script, or a manifest")
    s.add_argument("--out", required=True, help="where task/ and the set directories go (they must not exist)")
    s.add_argument("--round", type=_round, required=True, help="the manifests' round")
    s.add_argument("--split", help="the agent's rule (JSON), merged under the task's data.split")
    s.add_argument("--script", help="the preprocessing script, a file in --corpus (corpus without a manifest)")
    s.add_argument("--command", help="the command that runs the script (corpus without a manifest)")
    s.add_argument("--label-method", help="how the labels were made (corpus without a manifest)")
    s.add_argument("--checks", help="the quality checks dset does not compute: JSON [{name, value, threshold, passed}]")
    s.add_argument("--dry-run", action="store_true", help="print the summary only")
    s.set_defaults(fn=cmd_sets)

    g = sub.add_parser("grow", help="a new trainset: the current one plus rows from the pool and --with")
    g.add_argument("--task", required=True)
    g.add_argument("--pool", required=True, help="the pool directory")
    g.add_argument("--out", required=True, help="the new trainset's directory (must not exist)")
    g.add_argument("--round", type=_round, required=True)
    g.add_argument("--trainset", help="the current trainset's directory (with --parent)")
    g.add_argument("--parent", help="the current trainset's registered id (with --trainset)")
    g.add_argument("--add", type=_count, help="rows to draw from the pool rows not in the trainset")
    g.add_argument("--with", dest="with_rows", help="full rows to add (JSONL), e.g. rows written from feedback")
    g.add_argument("--label-method", help="default: the current trainset's")
    g.add_argument("--checks")
    g.add_argument("--dry-run", action="store_true")
    g.set_defaults(fn=cmd_grow)

    c = sub.add_parser("check", help="duplicates, conflicts and overlaps of a rows file; --write a cleaned trainset")
    c.add_argument("--task", required=True)
    c.add_argument("--rows", required=True, help="full rows (JSONL)")
    c.add_argument("--against", action="append", help="a rows or inputs file to count overlaps with (repeatable)")
    c.add_argument("--write", help="write the cleaned rows as a trainset into this directory (must not exist)")
    c.add_argument("--mode", choices=sorted(MODES), help="with --write: drop or merge")
    c.add_argument("--round", type=_round, help="with --write")
    c.add_argument("--parent", help="with --write: the registered trainset the rows come from")
    c.add_argument("--checks", help="with --write")
    c.set_defaults(fn=cmd_check)

    r = sub.add_parser("refresh", help="a new evalset from rows no set, the pool or --exclude holds")
    r.add_argument("--task", required=True, help="task.json with the complete rule the sets were drawn by")
    r.add_argument("--corpus", required=True)
    r.add_argument("--out", required=True, help="the new evalset's directory (must not exist)")
    r.add_argument("--round", type=_round, required=True)
    r.add_argument("--exclude", action="append", help="rows or inputs whose rows the new evalset must not share "
                                                      "(earlier evalsets' inputs, the current trainset and valset)")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_refresh)
    return p


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        return args.fn(args)
    except (DsetError, OSError) as e:  # an OSError while writing: publishing() has removed what was written
        sys.stderr.write("dset: error: %s\n" % e)
        return 1


# ---- reading ---------------------------------------------------------------------------------------------------------

def _some(errors, n=3):
    more = " (and %d more)" % (len(errors) - n) if len(errors) > n else ""
    return "; ".join(errors[:n]) + more


def _load_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise DsetError("%s: cannot read %s (%s)" % (what, path, e))


def _read_rows(path, what):
    try:
        return R.read_jsonl(path)
    except (OSError, ValueError) as e:
        raise DsetError("%s: %s" % (what, e))


def _load_task(path):
    task = _load_json(path, "--task")
    errors = R.check_task(task) if isinstance(task, dict) else ["expected a JSON object"]
    if errors:
        raise DsetError("--task %s: %s" % (path, _some(errors)))
    return task


def _manifest(d, what):
    """The manifest.json of directory d, None when it has none."""
    if not os.path.isdir(d):
        raise DsetError("%s %s: not a directory" % (what, d))
    path = os.path.join(d, "manifest.json")
    if not os.path.isfile(path):
        return None
    m = _load_json(path, what)
    if not isinstance(m, dict):
        raise DsetError("%s: %s is not a JSON object" % (what, path))
    return m


def _rows_file(m, what):
    """The rows file of a set or corpus directory: its manifest's files.rows, rows.jsonl without a manifest."""
    if m is None:
        return "rows.jsonl"
    spec = (m.get("files") or {}).get("rows") if isinstance(m.get("files"), dict) else None
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise DsetError("%s: its manifest.json lists no files.rows" % what)
    return spec["path"]


def _dir_rows(d, what):
    """(rows, manifest or None) of a set or corpus directory."""
    m = _manifest(d, what)
    return _read_rows(os.path.join(d, _rows_file(m, what)), what), m


def _check_rows(rows, task, what, train=False):
    """Full rows of the task (rsi.rows.check_full_rows); `train`: also no split starting with test."""
    errors = R.check_full_rows(rows, task)
    if train:
        bad = [r.get("id") for r in rows if isinstance(r.get("split"), str)
               and r["split"].strip().lower().startswith("test")]
        if bad:
            errors.append("split: %d rows have a split starting with test, e.g. %s: a trainset does not train on them"
                          % (len(bad), bad[0]))
    if errors:
        raise DsetError("%s: %d problems in the rows: %s" % (what, len(errors), _some(errors)))


def _new_paths(paths):
    """Refuse when a path dset would write exists, or the directory it would go into holds .dset-* entries: a run that
    stopped part way leaves its staging (.dset-*), the directories it had put in place and the corpus's
    split_report.json."""
    there = [p for p in paths if os.path.lexists(p)]
    parents = []
    for p in paths:
        d = os.path.dirname(p) or "."
        if d not in parents:
            parents.append(d)
    there += [os.path.join(d, n) for d in parents if os.path.isdir(d) for n in sorted(os.listdir(d))
              if n.startswith(W.STAGING)]
    if there:
        raise DsetError("already exists: %s. dset overwrites nothing: write to a new --out, or, when these are what a "
                        "run that stopped part way left (the new directories under --out, the corpus's "
                        "split_report.json and %s* entries), delete them and run again"
                        % (", ".join(there), W.STAGING))


def _given_checks(path):
    return None if path is None else _load_json(path, "--checks")


def _emit(summary):
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


# ---- sets ------------------------------------------------------------------------------------------------------------

def _rule(task, split_file):
    """(complete rule with its defaults written, the task version it belongs to, whether that is a new version): the
    agent's rule merged under the task's data.split. A complete rule of the task's stays as it is; an incomplete one is
    filled in, a new task version (rsi.datasplit.fill_errors)."""
    given = DS.task_split(task)
    chosen = {}
    if split_file:
        chosen = _load_json(split_file, "--split")
        if not isinstance(chosen, dict):
            raise DsetError("--split %s: expected a JSON object (a split rule)" % split_file)
        errors = DS.split_errors(chosen, partial=True)
        if errors:
            raise DsetError("--split %s: %s" % (split_file, _some(errors)))
    merged, errors = DS.merge_split(given, chosen)
    if errors:
        raise DsetError(_some(errors))
    errors = DS.split_errors(merged)
    if errors:
        raise DsetError("the rule (the task's data.split%s): %s" % (" with --split" if split_file else "",
                                                                    _some(errors)))
    rule = DS.filled(merged)
    if DS.is_complete(given):
        if DS.filled(given) != rule:
            raise DsetError("the task's data.split is complete, so the rule stays as it is (defaults written out: %s); "
                            "--split may repeat it, not change it (it gives %s)"
                            % (json.dumps(DS.filled(given)), json.dumps(rule)))
        return rule, task, False
    new = copy.deepcopy(task)
    new["version"] = task["version"] + 1
    new["data"]["split"] = rule
    errors = DS.fill_errors(task, new)
    if errors:
        raise DsetError("the task version with the rule is not a fill of the task's: %s" % _some(errors))
    return rule, new, True


def _corpus_source(a):
    """(rows file, script, label method, manifest or None) of the corpus: from its manifest when it has one (dset
    then writes nothing into it), else rows.jsonl and the --script, --command and --label-method to write one from."""
    m = _manifest(a.corpus, "--corpus")
    if m is not None:
        label_method = m.get("label_method")
        if not isinstance(label_method, str) or not label_method:
            raise DsetError("--corpus %s: its manifest.json has no label_method" % a.corpus)
        return _rows_file(m, "--corpus"), None, label_method, m
    lacking = [o for o, v in (("--script", a.script), ("--command", a.command), ("--label-method", a.label_method))
               if not v]
    if lacking:
        raise DsetError("--corpus %s has no manifest.json, so dset writes one: give %s"
                        % (a.corpus, ", ".join(lacking)))
    norm = os.path.normpath(a.script)
    if os.path.isabs(a.script) or norm == os.pardir or norm.startswith(os.pardir + os.sep):
        raise DsetError("--script %s: the name of the script inside --corpus" % a.script)
    if not os.path.isfile(os.path.join(a.corpus, a.script)):
        raise DsetError("--script %s: no such file in %s" % (a.script, a.corpus))
    return "rows.jsonl", a.script, a.label_method, None


def _read_bytes(path, what):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        raise DsetError("%s: %s" % (what, e))


def cmd_sets(a):
    task = _load_task(a.task)
    rule, task_v, new_version = _rule(task, a.split)
    current_md = _read_bytes(os.path.join(os.path.dirname(a.task), "TASK.md"), "--task") if new_version else None
    rows_file, script, label_method, corpus_m = _corpus_source(a)
    rows = _read_rows(os.path.join(a.corpus, rows_file), "--corpus")
    _check_rows(rows, task_v, "--corpus %s" % a.corpus)
    errors = DS.corpus_errors(rows, task_v)
    if errors:
        raise DsetError("--corpus %s: %s" % (a.corpus, _some(errors)))
    outs = dict((k, os.path.join(a.out, k)) for k in ("task",) + SETS)
    if not a.dry_run:  # the directories under --out, and the corpus files when dset writes them
        _new_paths([outs[k] for k in ("task",) + SETS]
                   + ([os.path.join(a.corpus, n) for n in W.CORPUS_FILES] if corpus_m is None else []))
    alloc = S.allocate(rows, task_v, rule)
    report, trainset = alloc["report"], alloc["sets"]["trainset"]
    checks = Q.quality_checks(trainset, task_v, _given_checks(a.checks))
    gate = Q.gate(checks)
    wrote = []
    summary = dict(report, gate=gate, wrote=wrote)
    if not gate["passed"] or a.dry_run:
        _emit(summary)
        return 0 if gate["passed"] else 2
    with W.publishing() as pub:  # all of it or nothing; the corpus manifest last
        if new_version:
            W.write_task(pub.dir(outs["task"]), task_v, current_md)
            wrote.append(outs["task"])
        for kind in SETS:
            entry = report["pool"] if kind == "pool" else report["sets"][kind]
            set_rows = W.resplit(alloc["pool"] if kind == "pool" else alloc["sets"][kind], W.SPLIT_NAMES[kind])
            notes = W.notes_line("sets", entry, rule.get("key"), kind)
            extra = {}
            if kind == "trainset":
                extra = {"quality_report": {"dset_version": DSET_VERSION, "profile": Q.profile(trainset, task_v),
                                            "checks": checks, "split": W.train_view(report)},
                         "label_method": label_method, "parent_trainset": None,
                         "changes": "first version: %d rows drawn by dset sets from the pool of %d rows"
                                    % (len(trainset), len(alloc["pool"])),
                         "quality": gate}
            W.write_set(pub.dir(outs[kind]), kind, set_rows, task_v, a.round, notes, **extra)
            wrote.append(outs[kind])
        if corpus_m is None:
            notes = "dset sets: rows %d; labels %s; unused rows %d" % (
                len(rows), ", ".join("%s %d" % kv for kv in Q.label_counts(rows, task_v).items()),
                report["unused_rows"])
            for name, data in W.corpus_files(task_v, a.round, rows_file, len(rows), script, a.command, label_method,
                                             report, notes).items():
                pub.file(os.path.join(a.corpus, name), data)
            wrote.append(a.corpus)
    _emit(summary)
    return 0


# ---- grow, check, refresh --------------------------------------------------------------------------------------------

def cmd_grow(a):
    task = _load_task(a.task)
    rule = DS.filled(DS.task_split(task))
    if (a.trainset is None) != (a.parent is None):
        raise DsetError("--trainset and --parent go together: the current trainset's directory and its registered id")
    if a.add is None and a.with_rows is None:
        raise DsetError("nothing to add: give --add N, --with FILE or both")
    if not a.dry_run:
        _new_paths([a.out])
    pool, _ = _dir_rows(a.pool, "--pool")
    _check_rows(pool, task, "--pool %s" % a.pool, train=True)
    current, current_m = _dir_rows(a.trainset, "--trainset") if a.trainset else ([], None)
    extra = []
    if a.with_rows:
        extra = _read_rows(a.with_rows, "--with")
        _check_rows(extra, task, "--with %s" % a.with_rows, train=True)
    drawn, free = S.grow(pool, current, a.add or 0, task, rule, extra)
    rows = current + W.resplit(drawn, "train") + extra
    _check_rows(rows, task, "the new trainset (the current one, the rows drawn and the --with rows)", train=True)
    label_method = a.label_method or (current_m or {}).get("label_method")
    if not isinstance(label_method, str) or not label_method:
        raise DsetError("--label-method: needed (no --trainset whose manifest has one)")
    checks = Q.quality_checks(rows, task, _given_checks(a.checks))
    gate = Q.gate(checks)
    info = {"parent_trainset": a.parent, "kept": len(current), "added": len(drawn), "with": len(extra),
            "pool_rows": len(pool), "pool_left": free - len(drawn)}
    summary = dict(info, profile=Q.profile(rows, task), gate=gate, wrote=[])
    if not a.dry_run:
        changes = "dset grow: %d rows of %s kept, %d drawn from the pool, %d from %s" % (
            len(current), a.parent or "no trainset", len(drawn), len(extra), a.with_rows or "no --with file")
        notes = W.notes_line("grow", {"rows": len(rows), "label_counts": summary["profile"]["label_counts"]})
        notes += "; kept %d; added %d; with %d" % (len(current), len(drawn), len(extra))
        with W.publishing() as pub:
            W.write_set(pub.dir(a.out), "trainset", rows, task, a.round, notes, [a.parent] if a.parent else [],
                        quality_report={"dset_version": DSET_VERSION, "profile": summary["profile"],
                                        "checks": checks, "grow": info},
                        label_method=label_method, parent_trainset=a.parent, changes=changes, quality=gate)
        summary["wrote"].append(a.out)
    _emit(summary)
    return 0


def cmd_check(a):
    task = _load_task(a.task)
    rows = _read_rows(a.rows, "--rows")
    seen, repeated = set(), []  # (row, id) of each row whose id an earlier row has
    for i, r in enumerate(rows):
        rid = r.get("id")
        if isinstance(rid, str) and rid in seen:
            repeated.append((i, rid))
        elif isinstance(rid, str):
            seen.add(rid)
    # Every row is checked. A repeated id is reported (and refused by --write), so only the error it raises itself,
    # rsi.rows' "row <i> (id=<id>): duplicate id" of exactly those rows, is left out.
    repeat_errors = set("row %d (id=%s): duplicate id" % ir for ir in repeated)
    errors = [e for e in R.check_full_rows(rows, task) if e not in repeat_errors]
    if errors:
        raise DsetError("--rows %s: %d problems in the rows: %s" % (a.rows, len(errors), _some(errors)))
    summary = Q.profile(rows, task)
    summary["duplicate_ids"] = len(repeated)
    summary["against"] = {}
    for f in a.against or []:
        other = _read_rows(f, "--against")
        ov = R.overlap(rows, other)
        ov["state_ids"] = R.shared_state_ids(rows, other)
        summary["against"][f] = ov
    if a.write is None:
        given = [o for o, v in (("--mode", a.mode), ("--round", a.round), ("--parent", a.parent),
                                ("--checks", a.checks)) if v is not None]
        if given:
            raise DsetError("%s: only with --write" % ", ".join(given))
        _emit(summary)
        return 0
    if a.mode is None or a.round is None:
        raise DsetError("--write needs --mode (drop or merge) and --round")
    if repeated:
        raise DsetError("--rows %s: %d rows repeat an id, e.g. %s: give each row its own id before --write"
                        % (a.rows, len(repeated), repeated[0][1]))
    _new_paths([a.write])
    cleaned = Q.clean(rows, task, a.mode)
    _check_rows(cleaned, task, "--rows %s" % a.rows, train=True)
    checks = Q.quality_checks(cleaned, task, _given_checks(a.checks))
    gate = Q.gate(checks)
    written = Q.profile(cleaned, task)
    m = _manifest(os.path.dirname(a.rows) or ".", "--rows")
    source = m.get("label_method") if m and isinstance(m.get("label_method"), str) else None
    label_method = "%s; %s" % (source or "the labels of %s as given" % os.path.basename(a.rows), MODES[a.mode])
    notes = W.notes_line("check", written) + "; %d rows before --mode %s" % (len(rows), a.mode)
    with W.publishing() as pub:
        W.write_set(pub.dir(a.write), "trainset", cleaned, task, a.round, notes, [a.parent] if a.parent else [],
                    quality_report={"dset_version": DSET_VERSION, "profile": written, "checks": checks,
                                    "check": {"mode": a.mode, "rows": a.rows, "before": summary}},
                    label_method=label_method, parent_trainset=a.parent,
                    changes="%s: %d rows of %s became %d" % (MODES[a.mode], len(rows), a.rows, len(cleaned)),
                    quality=gate)
    summary.update(written=written, gate=gate, wrote=[a.write])
    _emit(summary)
    return 0


def cmd_refresh(a):
    task = _load_task(a.task)
    split = DS.task_split(task)
    if not DS.is_complete(split):
        raise DsetError("refresh draws by the rule the sets were drawn by, and the task's data.split is not complete "
                        "(use the task version the first round's sets wrote)")
    rule = DS.filled(split)
    if not a.dry_run:
        _new_paths([a.out])
    rows, _ = _dir_rows(a.corpus, "--corpus")
    _check_rows(rows, task, "--corpus %s" % a.corpus)
    errors = DS.corpus_errors(rows, task)
    if errors:
        raise DsetError("--corpus %s: %s" % (a.corpus, _some(errors)))
    exclude = [(f, _read_rows(f, "--exclude")) for f in a.exclude or []]
    drawn, entry = S.refresh(rows, task, rule, [rs for _, rs in exclude])
    summary = {"rule": rule, "evalset": entry, "exclude": dict((f, len(rs)) for f, rs in exclude), "wrote": []}
    if not a.dry_run:
        with W.publishing() as pub:
            W.write_set(pub.dir(a.out), "evalset", W.resplit(drawn, "eval"), task, a.round,
                        W.notes_line("refresh", entry, rule.get("key"), "evalset"))
        summary["wrote"].append(a.out)
    _emit(summary)
    return 0
