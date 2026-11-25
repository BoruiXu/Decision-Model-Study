"""Shared code of the example mock modules: deterministic stand-ins for the five module agents (stdlib only).

`rsi run <ws> --runner mock --mock-dir examples/mock_modules` runs <module>.py from this directory as the agent
of a dispatch (rsi.runners.MockRunner):

    <module>.py --brief <outdir>/brief.json --outdir <outdir> --session <id> [--feedback <file>]

A mock module works the way the module skills (modules/<module>/SKILL.md) tell an agent to: it reads only its
brief.json (the paths under `inputs`, `ids`, `round`, `purpose`, `task_json`, `train_command`, `rsi`, `best`),
writes each artifact as a subdirectory of the output directory, checks each with `rsi validate`, computes every
number it reports with `rsi score` (both through the brief's `rsi` line, the only rsi commands it runs), and
ends by writing handback.json. Sent back to its session (feedback-<k>.md), continued after a long job, or taken
over by a fresh session (history-<k>.md), it reads that file and acts on it. A module keeps its own notes in
<outdir>/.mock-state.json and logs each call to <outdir>/mock-calls.jsonl ({"event": "start" | "end", "call",
"session", "feedback", "purpose", "round", ...}), so a test can see which session got which feedback.

The fictional task is spam-check (tests/fixtures.binary_task): an e-mail (one user message) -> yes (spam) / no.
Rows come from fixed templates; a row's text carries a number unique to its set (namespace), so sets never share
inputs. bin/train is the train command (the contract in modules/train/SKILL.md): a self-contained file whose
candidates carry an executable model.predict. Point the task's resources.train_command at it. Its model learns
spam words from the trainset and follows them with probability q (the config's capacity and lr set q), so a
config change moves the metrics in a known direction.

Knobs: RSI_MOCK_SCENARIO names a JSON file {module: {knob: {round: value}}} (rounds as strings); without it every
module does its plain job. "First call" below is the first launch of a dispatch made in that round. Per module:

  data     near_dups {round: rate}    the first call's trainset holds this share of near-duplicate rows (its
                                      quality gate fails above quality_checks.max_near_dup_rate); a return whose
                                      feedback names the failed gate deduplicates
  train    trials {round: [config]}   the trial configs of the round's first search (the first is the search's
                                      default); default, and every later search: [{}, {"lr": 0.1}, {"lr": 0.3}]
           config {round: {k: v}}     config overrides of every trial of every search in the round, on top of the
                                      best version's candidate config (or the train command's defaults)
           long_job {round: {"release": path} | {"seconds": s} | {"checkins": n}}
                                      the first call starts its trials in the background (waiting for the
                                      release file, or s seconds, first), writes the pid file
                                      <marker>.pid and hands back long_job (marker search.marker); a
                                      check-in hands back long_job again; with checkins the job waits for
                                      the n-th check-in, which lets it run and waits for its marker; the
                                      continued session collects the trials
  deliver  malformed {round: true}    the first call's delivery has a malformed manifest and is handed back
                                      without validating it, so registration refuses it; the return fixes it
  judge    entry {round: module}      the plan's entry module (default train); a revise answer that names a
                                      module (data, train, deliver) moves the entry there
  any      questions {round: [text]}  the hand-back questions of the first call (the engine registers nothing it
                                      handed back until the user answers; the answer comes back to the same
                                      session: data then drops the rows of unknown source it added when the
                                      answer says 删, any other module hands back the same subdirectories again)
           hang {round: {"started": path, "release": path}}
                                      the first call writes `started`, then waits for `release`
           kill_parent {round: "after_handback"}
                                      once (a flag file beside the scenario file), kill the process that ran the
                                      script (`rsi run`) right after writing handback.json

`python3 mocklib.py --after <file> --wait-s <s> --marker <marker> --batch <argvs.json>` is the background starter
of a long job: it waits until <file> exists (or <s> seconds pass, without a file), runs the command lines of the
batch file one after the other, then writes `ok` to the marker.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

LABELS = ("yes", "no")
SPAM = {
    "S1": "Congratulations, your lottery ticket {n} has won! Claim the lottery payout now.",
    "S2": "You are selected for a prize {n}. Reply to receive your prize today.",
    "S3": "Double your crypto in 24 hours, wallet code {n}, guaranteed crypto returns.",
    "S4": "Refinance now at {n} basis points lower, the refinance offer expires soon.",
    "S5": "Dear winner, reference {n}, send your bank details to the winner desk.",
    "S6": "Online pharmacy discount {n} percent, no prescription pharmacy deals.",
    "S7": "Free casino spins {n}, join the casino tonight.",
    "S8": "Unclaimed inheritance of {n} dollars awaits, the inheritance lawyer is waiting.",
}
HAM = {
    "H1": "Can we move the meeting {n} to Thursday? The meeting room is booked.",
    "H2": "Attached is invoice {n} for last month, the invoice is due in 30 days.",
    "H3": "Your order {n} has shipped, shipping tracking is available tomorrow.",
    "H4": "This week's newsletter issue {n}: team updates and the newsletter archive.",
    "H5": "Your password reset code is {n}; ignore it if you did not request a password change.",
    "H6": "Lunch at noon? Table {n} at the usual lunch place.",
    "H7": "The quarterly report draft {n} is ready for review, report comments welcome.",
    "H8": "Updated schedule {n} for the release, schedule changes highlighted.",
}
GREET = ("Hi,", "Hello,", "Dear customer,", "Hey there,", "Good morning,", "Greetings,")
SIGN = ("Best regards.", "Thanks.", "Cheers.", "Sent from my phone.", "Kind regards.", "See you.")
GROUPS = ("web", "mobile", "api")
STATE_FILE = ".mock-state.json"
CALLS_FILE = "mock-calls.jsonl"
ATTRIBUTION = {"data": "data_coverage", "train": "model_capacity", "deliver": "runtime"}


# ---- files ---------------------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def h01(*parts):
    """A deterministic number in [0, 1) from strings."""
    d = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(d[:12], 16) / float(16 ** 12)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_text(path):
    """The file's text, or "" when it does not exist."""
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def rows_of(art_dir, logical="rows"):
    """The rows of a logical file of an artifact directory (its manifest's `files` entry)."""
    m = read_json(os.path.join(art_dir, "manifest.json"))
    return read_jsonl(os.path.join(art_dir, m["files"][logical]["path"]))


def art_id(path):
    """The registered id of an input path the brief gives: its artifact directory's name (a file input, such as
    an evalset's inputs.jsonl or the task's task.json, lies directly in that directory)."""
    return os.path.basename(os.path.dirname(path) if os.path.isfile(path) else path.rstrip(os.sep))


# ---- rows ----------------------------------------------------------------------------------------

def target(label):
    return {"yes": 1.0 if label == "yes" else 0.0, "no": 0.0 if label == "yes" else 1.0}


def hard(t):
    return "yes" if t.get("yes", 0) >= t.get("no", 0) else "no"


def question_of(task):
    """The OneJev question of the task's decision (docs/interfaces/README.md): spam-check is a yes/no binary
    decision, so a noul question whose criteria "true" / "false" are the meanings of yes / no."""
    d = task["decision"]
    return {"type": "noul", "instructions": d["instructions"],
            "criteria": {"true": d["label_meanings"]["yes"], "false": d["label_meanings"]["no"]}}


def gen_rows(question, split, ns, prefix, n_clusters, per_cluster=2):
    """Full rows (OneJev rows asking `question`, in split `split`) of namespace `ns` (a small int whose digits
    lead every number in the texts, so two namespaces never share an input): cluster c is spam when c % 5 < 2,
    its template goes round the templates, each cluster has `per_cluster` greeting/sign variants of one text;
    meta.state_id is the channel the mail came in by."""
    spam_keys, ham_keys = sorted(SPAM), sorted(HAM)
    rows, i = [], 0
    for c in range(n_clusters):
        is_spam = c % 5 < 2
        keys = spam_keys if is_spam else ham_keys
        key = keys[(c // 5) % len(keys)]
        body = (SPAM if is_spam else HAM)[key].format(n=ns * 100000 + c)
        cluster = "%s-c%04d" % (prefix, c)
        for v in range(per_cluster):
            text = "%s %s %s" % (GREET[(c + v) % len(GREET)], body, SIGN[(c * 7 + v) % len(SIGN)])
            rows.append({"id": "%s-%05d" % (prefix, i), "state": {"messages": [{"role": "user", "content": text}]},
                         "question": question, "target": target("yes" if is_spam else "no"),
                         "meta": {"source": "synthetic-mail", "state_id": GROUPS[c % len(GROUPS)],
                                  "cluster": cluster, "label_source": "rule", "n_obs": 1, "extra": {"template": key}},
                         "split": split})
            i += 1
    return rows


def near_duplicates(rows, rate, prefix):
    """int(len(rows) * rate) extra rows: case and whitespace variants of existing rows (same cluster)."""
    out = []
    for j in range(int(len(rows) * rate)):
        src = rows[(j * 13) % len(rows)]
        dup = json.loads(json.dumps(src))
        dup["id"] = "%s-%05d" % (prefix, j)
        dup["state"]["messages"][0]["content"] = "  " + src["state"]["messages"][0]["content"].upper() + "  "
        out.append(dup)
    return out


INPUT_META = ("state_id", "cluster", "source", "n_obs")  # the only meta keys of an input row


def input_row(r):
    """The inputs.jsonl row: the full row without target, its meta only INPUT_META (other meta fields, such as
    label_source and extra, may follow the label and stay out of what deliver gets)."""
    out = dict((k, v) for k, v in r.items() if k != "target")
    out["meta"] = dict((k, v) for k, v in r["meta"].items() if k in INPUT_META)
    return out


def label_row(r):
    """The labels.jsonl row: id, target and meta.label_source (which input rows leave out)."""
    return {"id": r["id"], "target": r["target"], "meta": {"label_source": r["meta"]["label_source"]}}


def _norm(v):
    if isinstance(v, str):
        return " ".join(v.split()).lower()
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return dict((k, _norm(x)) for k, x in v.items())
    return v


def fingerprint(inp):
    """Near-duplicate key: every string lower-cased with whitespace collapsed."""
    return json.dumps(_norm(inp), sort_keys=True)


def overlap(a, b):
    """(ids, inputs, clusters) shared by two row lists."""
    ids = len(set(r["id"] for r in a) & set(r["id"] for r in b))
    fps = len(set(fingerprint(r["state"]) for r in a) & set(fingerprint(r["state"]) for r in b))
    cl = len(set(r["meta"]["cluster"] for r in a) & set(r["meta"]["cluster"] for r in b))
    return ids, fps, cl


# ---- the two rsi commands a module may run --------------------------------------------------------

def rsi_parts(brief):
    """(environment additions, argv) of the brief's `rsi` line, e.g. PYTHONPATH=<repo root> python3 -m rsi."""
    env, argv = {}, []
    for tok in shlex.split(brief["rsi"]):
        if not argv and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            k, v = tok.split("=", 1)
            env[k] = v
        else:
            argv.append(tok)
    return env, argv


def rsi_env(brief):
    """The environment the brief's rsi line and the train command run in (PYTHONPATH=<repo root>)."""
    env = dict(os.environ)
    env.update(rsi_parts(brief)[0])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def rsi(brief, *args):
    """Run `rsi validate` or `rsi score` through the brief's rsi line; (exit code, output)."""
    assert args[0] in ("validate", "score"), "a module runs only rsi validate and rsi score"
    p = subprocess.run(rsi_parts(brief)[1] + list(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True, env=rsi_env(brief))
    return p.returncode, p.stdout + p.stderr if p.returncode else p.stdout


def score(brief, labels, predictions, train=None, clusters=None, compare=None):
    """What `rsi score` prints for these files, parsed (the numbers a manifest reports)."""
    args = ["score", "--task", brief["task_json"], "--labels", labels, "--predictions", predictions]
    for opt, val in (("--train", train), ("--clusters", clusters), ("--compare", compare)):
        if val:
            args += [opt, val]
    code, out = rsi(brief, *args)
    if code != 0:
        raise SystemExit("rsi score failed:\n%s" % out)
    return json.loads(out)


def validate(brief, kind, path):
    """(passed, output) of `rsi validate <kind> <dir> --task <task.json>`."""
    code, out = rsi(brief, "validate", kind, path, "--task", brief["task_json"])
    return code == 0, out.strip()


def train_argv(brief, trainset, trainset_id, valset, valset_id, config, source, producer, out, marker):
    """The train command line of the contract in modules/train/SKILL.md."""
    return [brief["train_command"], "--task", brief["task_json"], "--trainset", trainset, "--trainset-id",
            trainset_id, "--valset", valset, "--valset-id", valset_id, "--config", config, "--config-source", source,
            "--round", str(brief["round"]), "--producer", producer, "--out", out, "--marker", marker]


def run_argv(brief, argv):
    """Run a train command line (train_argv) to its end; (marker's first line, the marker's text, its output)."""
    marker = argv[argv.index("--marker") + 1]
    p = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, env=rsi_env(brief))
    text = read_text(marker)
    return (text.splitlines() or [""])[0].strip(), text, p.stdout


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---- manifests -----------------------------------------------------------------------------------

def base_manifest(brief, kind, producer, parents=(), notes=""):
    return {"kind": kind, "id": None, "task_id": brief["task_id"], "task_version": brief["task_version"],
            "round": brief["round"], "producer": producer, "created_at": now_iso(), "parents": list(parents),
            "files": {}, "notes": notes}


def write_record(c, name, failure, hypothesis, action, metric, before, after, conclusion, knowledge, artifacts=(),
                 automation="scripted"):
    d = c.new_dir(name)
    m = base_manifest(c.brief, "record", c.module, notes="第 %d 次启动（会话 %s）" % (c.n, c.session))
    m.update({"module": c.module, "owner": "mock-%s" % c.module, "failure": failure, "hypothesis": hypothesis,
              "action": action, "result": {"metric": metric, "before": before, "after": after,
                                           "conclusion": conclusion},
              "knowledge": knowledge, "automation": automation, "artifacts": list(artifacts)})
    write_json(os.path.join(d, "manifest.json"), m)
    return d


# ---- one call of a module ------------------------------------------------------------------------

class Call(object):
    """One run of a mock module: the brief, the feedback file, the scenario knobs, the call log."""

    def __init__(self, module):
        p = argparse.ArgumentParser(prog=module)
        for opt in ("--brief", "--outdir", "--session"):
            p.add_argument(opt, required=True)
        p.add_argument("--feedback")
        a = p.parse_args()
        self.module, self.outdir, self.session = module, os.path.abspath(a.outdir), a.session
        self.brief = read_json(a.brief)
        self.task = read_json(self.brief["task_json"])
        self.round, self.purpose = self.brief["round"], self.brief["purpose"]
        self.inputs = self.brief["inputs"]
        self.feedback_file = os.path.basename(a.feedback) if a.feedback else None
        self.feedback = read_text(a.feedback) if a.feedback else ""
        self.scenario_path = os.environ.get("RSI_MOCK_SCENARIO") or ""
        self.scenario = read_json(self.scenario_path) if os.path.isfile(self.scenario_path) else {}
        self.knobs = self.scenario.get(module, {})
        calls = os.path.join(self.outdir, CALLS_FILE)
        starts = [x for x in read_jsonl(calls) if x.get("event") == "start"] if os.path.exists(calls) else []
        self.n = len(starts) + 1  # this call's number in the output directory (= the dispatch's launch)
        self.first = self.n == 1
        self.log("start", feedback=self.feedback, pid=os.getpid())
        hang = self.knob("hang") if self.first else None
        if hang:
            with open(hang["started"], "w") as f:
                f.write("%d\n" % os.getpid())
            if not wait_for(hang["release"], 300, within=self.outdir):
                sys.exit(1)

    def knob(self, name):
        """The scenario's value of knob `name` for this round, or None."""
        return (self.knobs.get(name) or {}).get(str(self.round))

    def log(self, event, **fields):
        rec = {"event": event, "call": self.n, "session": self.session, "feedback_file": self.feedback_file,
               "purpose": self.purpose, "round": self.round, "at": now_iso()}
        rec.update(fields)
        with open(os.path.join(self.outdir, CALLS_FILE), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def reason(self, default=""):
        """The reason line of the feedback (原因：… / 决策点给出的原因：…), or `default`."""
        m = re.search(r"^(?:原因|决策点给出的原因)：(.*)$", self.feedback, re.M)
        return m.group(1) if m else default

    def state(self):
        path = os.path.join(self.outdir, STATE_FILE)
        return read_json(path) if os.path.exists(path) else {}

    def save_state(self, st):
        write_json(os.path.join(self.outdir, STATE_FILE), st)

    def answers(self):
        """The user's answers to this dispatch's questions given in the feedback: [(question, answer)]."""
        return re.findall(r"^- (.*)\n  用户（[^）]*）的回答：(.*)$", self.feedback, re.M)

    def previous_artifacts(self):
        """The subdirectories the dispatch's previous hand-back listed (kept by the engine as handback-<n>.json)."""
        backs = sorted(glob.glob(os.path.join(self.outdir, "handback-*.json")), key=os.path.getmtime)
        return list(read_json(backs[-1]).get("artifacts") or []) if backs else []

    def rehand_if_answered(self):
        """A launch carrying the user's answers to the last hand-back's questions (nothing of it was registered):
        hand the same subdirectories back; True when it did."""
        if not (self.answers() and self.handed_back_before()):
            return False
        names = self.previous_artifacts()
        self.handback([os.path.join(self.outdir, n) for n in names],
                      notes="用户回答了问题（%s），交回的产物不变" % self.answers()[-1][1])
        return True

    def handed_back_before(self):
        """True when an earlier call of this dispatch wrote a hand-back (the engine keeps it as handback-<n>.json)."""
        return bool(glob.glob(os.path.join(self.outdir, "handback-*.json"))) or (
            not self.first and os.path.exists(os.path.join(self.outdir, "handback.json")))

    def new_dir(self, name):
        """A new artifact subdirectory: `name`, or name-<call> (name-<call>-<k>) when that is taken."""
        rel, k = name, 1
        while os.path.exists(os.path.join(self.outdir, rel)):
            rel = "%s-%d" % (name, self.n) if k == 1 else "%s-%d-%d" % (name, self.n, k)
            k += 1
        path = os.path.join(self.outdir, rel)
        os.makedirs(path)
        return path

    def check(self, kind, path):
        ok, out = validate(self.brief, kind, path)
        if not ok:
            self.fail("%s %s 没通过 rsi validate：%s" % (kind, os.path.basename(path), out))

    def handback(self, artifacts, status="ok", marker=None, notes=""):
        qs = self.knob("questions") if self.first else None
        h = {"status": status, "artifacts": [os.path.basename(a) for a in artifacts], "marker": marker,
             "notes": notes, "questions": list(qs or [])}
        write_json(os.path.join(self.outdir, "handback.json"), h)
        self.log("end", status=status, artifacts=h["artifacts"])
        kill = self.knob("kill_parent")
        if kill == "after_handback" and once(self.scenario_path, "kill-%s-%s" % (self.module, self.round)):
            os.kill(os.getppid(), signal.SIGKILL)

    def fail(self, notes):
        self.handback([], status="failed", notes=notes)
        sys.exit(0)


def once(scenario_path, name):
    """True the first time `name` is asked for this scenario (a flag file beside the scenario file)."""
    flags = (scenario_path or os.path.join(os.getcwd(), "scenario")) + ".flags"
    os.makedirs(flags, exist_ok=True)
    try:
        os.close(os.open(os.path.join(flags, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def wait_for(path, timeout_s, within=None):
    """Wait until `path` exists (True); False after timeout_s seconds, or once the directory `within` is gone
    (the workspace was removed: nobody will release the wait)."""
    end = time.time() + timeout_s
    while not os.path.exists(path):
        if time.time() > end or (within and not os.path.isdir(within)):
            return False
        time.sleep(0.05)
    return True


def start_background(argvs, marker, log, release=None, wait_s=0):
    """Start the train command lines `argvs` one after the other, detached from this process (their own session),
    after the release file appears or wait_s seconds; when all have ended (each writes its own marker) the job
    writes its completion marker `marker` (ok: the hand-back that collects them reports a trial that failed).
    Returns its pid. Its pid file <marker>.pid (the pid, then the start time in Unix seconds) is written here,
    before the long_job hand-back, so a stop of the run can terminate the job (rsi.engine)."""
    batch = marker + ".batch.json"
    write_json(batch, argvs)
    cmd = [sys.executable, os.path.abspath(__file__), "--after", release or "", "--wait-s", str(wait_s),
           "--marker", marker, "--batch", batch]
    with open(log, "ab") as f:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
    with open(marker + ".pid", "w") as f:
        f.write("%d\n%d\n" % (p.pid, int(time.time())))
    return p.pid


def _background_main(argv):
    p = argparse.ArgumentParser(prog="mocklib")
    p.add_argument("--after", default="")
    p.add_argument("--wait-s", type=float, default=0)
    p.add_argument("--marker", required=True)
    p.add_argument("--batch", required=True)
    a = p.parse_args(argv)
    if a.after:
        if not wait_for(a.after, 600, within=os.path.dirname(a.marker)):
            if os.path.isdir(os.path.dirname(a.marker)):
                with open(a.marker, "w") as f:
                    f.write("failed\nthe release file %s never appeared\n" % a.after)
            return 1
    elif a.wait_s:
        time.sleep(a.wait_s)
    for cmd in read_json(a.batch):
        subprocess.call(cmd)
    with open(a.marker, "w") as f:
        f.write("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(_background_main(sys.argv[1:]))
