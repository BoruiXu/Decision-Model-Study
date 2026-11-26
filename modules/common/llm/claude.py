"""ClaudeCall: one LLM call that generates or reviews data, as one fresh `claude -p` process.

The command line is the task's agent command (rsi.runners.claude.ClaudeCodeRunner.from_task: `runner.claude.command`,
by default `claude -p --dangerously-skip-permissions --output-format json`, then `runner.claude.args`, which hold
`--model claude-opus-5-5 --effort xhigh` unless the task names its own), with the `--model` / `--effort` values
replaced in place when the call is given a model or an effort, every `--mcp-config` of the task's command and
arguments removed (with all its values, also as `--mcp-config=...`), `--setting-sources project` given once (in place
of any sources the task's runner.claude names: its args or setting_sources), `--strict-mcp-config` added (once), then
`--json-schema <schema> --tools "" --session-id <a new uuid>`:
- the CLI checks the answer against the schema and reports it as "structured_output". The schema must stay within
  what the CLI handles (schema_problems): no `type` lists, `properties` only with `"type": "object"`, local `$ref`s
  only, and every object either closed (`additionalProperties: false`) or a map whose value schema constrains its
  values (`additionalProperties: {...}` with a type, an anyOf, ..., e.g. an anyOf of scalar types): with a bare open
  object `{"type": "object"}` the model answers `{}` (checked on Claude Code 2.1.331);
- data calls never get MCP servers, user settings or ambient context, whatever the task's agent command says:
  `--tools ""` (no built-in tool: "Built-in tools to make available: "" for none"), `--strict-mcp-config` and no
  `--mcp-config` (no MCP server), `--setting-sources project` (no user or local settings) and a working directory that
  is a new empty temporary directory outside the workspace (tempfile.mkdtemp, removed after the call; no project
  settings or CLAUDE.md there);
- a new session per call, never `--resume`, so the `total_cost_usd` of the result object is this call's own cost.
The prompt goes on stdin. The process runs in its own session and process group; a call that runs past timeout_s or
whose caller is stopped gets SIGTERM to its group, SIGKILL after STOP_GRACE_S, and then a final SIGKILL to the group,
so a member that ignores SIGTERM does not outlive it. Its output (stdout and stderr) is kept in
`<workdir>/<n>/out.json` and the prompt in `<workdir>/<n>/prompt.md` (n = 1, 2, ...).

Every call, whatever its outcome:
- appends {"at", "tag", "session", "cost_usd" (reported, else null), "billed_usd", "ok", "error", "prompt_sha256",
  "call"} to `<workdir>/llm_calls.jsonl`;
- appends the spend line {"at", "source": "llm:<tag>", "api_usd": <billed>} to the spend file (docs/interfaces/
  spend.md). The billed amount is the reported cost. Without one: a call that was killed (it ran past timeout_s, the
  caller was stopped, or a signal ended it) may have spent anything, so it is billed at the highest api_usd already
  recorded for its source in the spend file, else KILLED_USD, with a note; a call that could not start, or ended by
  itself without a result object (the CLI prints one whenever the model ran), is billed 0 with a note.
The two lines are written with SIGTERM held back, so a stop cannot cut them.

run() never raises for a bad reply: a process that cannot start, runs past timeout_s, exits non-zero or reports
is_error, a reply without a JSON answer and an answer that does not fit the schema (rsi.schema) all give an
LlmResult whose output is None and whose error says why, with the call billed. It raises:
- LlmBudgetExceeded, before a call, when a cap is reached:
  - usd_cap (a recipe's max_usd): this object's billed spend plus that of its `shared` partner (a recipe's generator
    and reviewer are each other's partner, so the recipe's two accounts add up) is at least the cap;
  - api_left (the brief's budget_left.api_usd): the `api_usd` amounts already in the spend file, counted as the
    engine counts them (rsi.workspace.spend_line_problems), are at least it. Every LLM call of a dispatch appends to
    that one file, so all its recipes, reviewers and paraphrases share this account.
  None means no limit;
- the exception that stopped a call while it ran (e.g. the caller's SIGTERM handler's, as dgen's long jobs raise),
  after the call's process group is ended (SIGTERM, SIGKILL after STOP_GRACE_S) and the call is logged and billed;
- ValueError for a schema with schema_problems (a programming error), before anything is spent.
"""
import collections
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid

from rsi.runners.base import stop_process
from rsi.runners.claude import ClaudeCodeRunner, parse_output, structured_output
from rsi.schema import validate
from rsi.workspace import spend_line_problems

EPS = 1e-9
TIMEOUT_S = 1800
TAIL_CHARS = 300
KILLED_USD = 1.0  # the bill of a killed call when its source has no recorded cost yet
STOP_GRACE_S = 2.0  # a stopped caller ends its call within 2 x this: inside the engine's 5 s before SIGKILL
SETTING_SOURCES = "project"  # never the user's or local settings
CONSTRAINING = ("type", "anyOf", "oneOf", "allOf", "$ref", "enum", "const")  # a value schema needs one of these

LlmResult = collections.namedtuple("LlmResult", "output cost_usd error log session")
_Launch = collections.namedtuple("_Launch", "out code error killed stop mask")


class LlmBudgetExceeded(Exception):
    """A cap is reached: the call was not started. The message says which cap and how much is spent."""


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _block_stop():
    """Hold SIGTERM back (one that arrives meanwhile is delivered once unblocked); the mask to restore, or None."""
    if hasattr(signal, "pthread_sigmask"):
        return signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    return None


def _unblock(mask):
    if mask is not None:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def _drain(proc):
    """What a stopped call printed, its pipes closed (b"" when that cannot be read within STOP_GRACE_S)."""
    try:
        out, _ = proc.communicate(timeout=STOP_GRACE_S)
        return out or b""
    except Exception:
        for f in (proc.stdin, proc.stdout):
            try:
                if f is not None:
                    f.close()
            except OSError:
                pass
        return b""


def _kill_group(pid):
    """A final SIGKILL to a stopped call's process group: a member that ignores SIGTERM does not outlive the call."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # ESRCH: nothing left in the group; EPERM: not ours to signal
        pass


def _append(path, obj):
    """One JSON line appended in one write."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _cost(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _spend_objects(spend_file):
    try:
        with open(spend_file, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except (OSError, TypeError):
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def api_spent(spend_file):
    """The sum of the `api_usd` amounts in the spend file (0 without one), counted as the engine counts them."""
    return sum(spend_line_problems(obj)[0].get("api_usd", 0.0) for obj in _spend_objects(spend_file))


def highest_cost(spend_file, source):
    """The highest `api_usd` amount of the spend file's lines of `source`, None without one."""
    amounts = [spend_line_problems(obj)[0].get("api_usd") for obj in _spend_objects(spend_file)
               if isinstance(obj, dict) and obj.get("source") == source]
    amounts = [a for a in amounts if a is not None]
    return max(amounts) if amounts else None


def _resolve(root, ref):
    """The schema a local `$ref` ("#/$defs/name", a JSON pointer into the root schema) names; None when it is not local
    or does not resolve."""
    if not isinstance(ref, str) or not ref.startswith("#"):
        return None
    node = root
    for part in [p for p in ref[1:].split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def schema_problems(schema, path="$", root=None):
    """What in a schema the CLI's structured output cannot use, one text per problem:
    - a `type` list (write an anyOf);
    - `properties` without a `type` (give "type": "object");
    - an open object: one of type object that is neither closed (`additionalProperties: false`) nor a map with a value
      schema (`additionalProperties: {...}`), and a value schema that constrains nothing (`additionalProperties: {}`,
      an open map in disguise: it needs one of CONSTRAINING);
    - a `$ref` that is not local or does not resolve.
    It walks properties, patternProperties, additionalProperties, items (a schema or a list), prefixItems,
    anyOf/allOf/oneOf and $defs/definitions (so every schema a `$ref` names is checked where it is defined)."""
    if not isinstance(schema, dict):
        return []
    root = schema if root is None else root
    problems = []
    kind, ap = schema.get("type"), schema.get("additionalProperties")
    if isinstance(kind, list):
        problems.append("%s: a type list (use anyOf)" % path)
    if "properties" in schema and "type" not in schema:
        problems.append("%s: properties without a type (give \"type\": \"object\")" % path)
    if isinstance(ap, dict) and not any(k in ap for k in CONSTRAINING):
        problems.append("%s: an open map (additionalProperties constrains nothing: give its values a type or an anyOf)"
                        % path)
    elif kind == "object" and not (ap is False or isinstance(ap, dict)):
        problems.append("%s: an open object (give additionalProperties false or a value schema)" % path)
    if "$ref" in schema and _resolve(root, schema["$ref"]) is None:
        problems.append("%s: $ref %s is not a local definition" % (path, json.dumps(schema["$ref"])))
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        for name, sub in sorted((schema.get(key) or {}).items()):
            problems += schema_problems(sub, "%s.%s.%s" % (path, key, name) if key != "properties"
                                        else "%s.%s" % (path, name), root)
    items = schema.get("items")
    if isinstance(items, list):
        for i, sub in enumerate(items):
            problems += schema_problems(sub, "%s.items[%d]" % (path, i), root)
    else:
        problems += schema_problems(items, "%s.items" % path, root)
    problems += schema_problems(ap, "%s.additionalProperties" % path, root)
    for key in ("prefixItems", "anyOf", "allOf", "oneOf"):
        for i, sub in enumerate(schema.get(key) or []):
            problems += schema_problems(sub, "%s.%s[%d]" % (path, key, i), root)
    return problems


def _has(args, option):
    return any(a == option or a.startswith(option + "=") for a in args)


def without_option(args, option, variadic=False):
    """`args` without any `option` ("--opt v" or "--opt=v") and its value; `variadic`: every value up to the next
    argument starting with "-" (e.g. `--mcp-config a.json b.json`)."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        i += 1
        if a.startswith(option + "="):
            continue
        if a == option:
            if variadic:
                while i < len(args) and not args[i].startswith("-"):
                    i += 1
            else:
                i += 1
            continue
        out.append(a)
    return out


def set_option(args, option, value):
    """`args` with `option` given exactly once, set to `value`: in place of its first occurrence, else appended."""
    for i, a in enumerate(args):
        if a == option or a.startswith(option + "="):
            return list(args[:i]) + [option, value] + without_option(args[i:], option)
    return list(args) + [option, value]


def with_option(args, option, value):
    """`args` with the value of `option` (as "--opt v" or "--opt=v") replaced in place; appended when absent."""
    if value is None:
        return list(args)
    out = list(args)
    for i, a in enumerate(out):
        if a == option and i + 1 < len(out):
            out[i + 1] = value
            return out
        if a.startswith(option + "="):
            out[i] = "%s=%s" % (option, value)
            return out
    return out + [option, value]


class ClaudeCall(object):
    """LLM calls of one role (a recipe's generator or its reviewer) logged in `workdir`, billed to `spend_file`."""

    def __init__(self, task, workdir, spend_file, usd_cap, model=None, effort=None, command=None,
                 timeout_s=TIMEOUT_S, api_left=None, shared=None):
        r = ClaudeCodeRunner.from_task(task)
        base = (list(command) if command is not None else list(r.command)) + \
            with_option(with_option(r.args, "--model", model), "--effort", effort)
        base = set_option(without_option(base, "--mcp-config", variadic=True), "--setting-sources", SETTING_SOURCES)
        if not _has(base, "--strict-mcp-config"):
            base.append("--strict-mcp-config")
        self.base = base
        self.workdir, self.spend_file = os.path.abspath(str(workdir)), spend_file
        self.usd_cap, self.api_left, self.shared = usd_cap, api_left, shared
        self.timeout_s = timeout_s
        self.calls = 0  # the calls this object started
        self._spent = 0.0

    @property
    def spent(self):
        """The dollars billed for this object's calls."""
        return self._spent

    def argv(self, schema, session):
        return self.base + ["--json-schema", json.dumps(schema, ensure_ascii=False), "--tools", "",
                            "--session-id", session]

    def check_caps(self):
        """LlmBudgetExceeded when a cap is reached (see the module docstring)."""
        if self.usd_cap is not None:
            spent = self.spent + (self.shared.spent if self.shared is not None and self.shared is not self else 0.0)
            if spent >= self.usd_cap - EPS:
                raise LlmBudgetExceeded("the recipe's cap max_usd $%.2f is reached: its generator and reviewer "
                                        "spent $%.4f" % (self.usd_cap, spent))
        if self.api_left is not None:
            used = api_spent(self.spend_file)
            if used >= self.api_left - EPS:
                raise LlmBudgetExceeded("the data-LLM budget is reached: the dispatch's spend file holds $%.4f of "
                                        "api_usd lines and budget_left.api_usd is $%.2f" % (used, self.api_left))

    def _new_dir(self):
        os.makedirs(self.workdir, exist_ok=True)
        n = 1 + max([int(x) for x in os.listdir(self.workdir) if x.isdigit()] or [0])
        while True:
            d = os.path.join(self.workdir, str(n))
            try:
                os.mkdir(d)
                return n, d
            except FileExistsError:
                n += 1

    def _launch(self, argv, cwd, prompt):
        """The call's _Launch. An exception while it runs (a stop) ends its process group and is returned as `stop`,
        with SIGTERM held back (`mask` restores it) until the call is billed."""
        try:
            proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            return _Launch(b"", None, "could not start %s: %s" % (argv[0], e), False, None, _block_stop())
        try:
            out, _ = proc.communicate(prompt.encode("utf-8"), timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            mask = _block_stop()
            stop_process(proc.pid, proc, grace=STOP_GRACE_S)
            _kill_group(proc.pid)
            return _Launch(_drain(proc), proc.returncode, "the call ran past its time limit of %g s and was stopped"
                           % self.timeout_s, True, None, mask)
        except BaseException as e:  # the caller is stopped (its SIGTERM handler raised) or interrupted
            mask = _block_stop()
            stop_process(proc.pid, proc, grace=STOP_GRACE_S)
            _kill_group(proc.pid)
            return _Launch(_drain(proc), proc.returncode, "the call was stopped (%s: %s)" % (type(e).__name__, e),
                           True, e, mask)
        mask = _block_stop()
        code = proc.returncode
        return _Launch(out or b"", code, None, code is not None and code < 0, None, mask)

    def _bill(self, cost, killed, tag, error):
        """(billed dollars, note or None) of a call."""
        if cost is not None:
            return cost, None
        if not killed:
            return 0.0, "cost not reported (%s)" % error
        known = highest_cost(self.spend_file, "llm:%s" % tag)
        if known is not None:
            return known, "cost not reported (%s); billed at the highest llm:%s call cost recorded" % (error, tag)
        return KILLED_USD, "cost not reported (%s); billed %.2f USD, no llm:%s call cost recorded yet" % (
            error, KILLED_USD, tag)

    def run(self, prompt, schema, tag):
        """One call: LlmResult(output, cost_usd, error, log, session); see the module docstring."""
        problems = schema_problems(schema)
        if problems:
            raise ValueError("a schema the CLI cannot use: %s" % "; ".join(problems))
        self.check_caps()
        self.calls += 1
        n, d = self._new_dir()
        session = str(uuid.uuid4())
        cwd = tempfile.mkdtemp(prefix="rsi-llm-")
        try:
            launch = self._launch(self.argv(schema, session), cwd, prompt)
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        try:
            out, code, error = launch.out, launch.code, launch.error
            log = os.path.join(d, "out.json")
            with open(log, "wb") as f:
                f.write(out)
            with open(os.path.join(d, "prompt.md"), "w", encoding="utf-8") as f:
                f.write(prompt)
            _reported, info = parse_output(log)
            cost = _cost(info.get("total_cost_usd"))
            output = structured_output(log)
            if error is None and code != 0:
                tail = " ".join(out.decode("utf-8", errors="replace").split())[-TAIL_CHARS:]
                error = "exit code %s%s" % (code, ": %s" % tail if tail else "")
            if error is None and info.get("is_error"):
                error = "the CLI reported an error: %s" % str(info.get("result") or info.get("subtype"))[:TAIL_CHARS]
            if error is None and output is None:
                error = "the reply holds no JSON answer"
            if error is None:
                problems = validate(output, schema)
                if problems:
                    error = "the answer does not fit the schema: %s" % "; ".join(problems[:3])
            if error is not None:
                output = None
            billed, note = self._bill(cost, launch.killed, tag, error or "no result object")
            self._spent += billed
            _append(os.path.join(self.workdir, "llm_calls.jsonl"), {
                "at": _now(), "tag": tag, "session": session, "cost_usd": cost, "billed_usd": billed,
                "ok": error is None, "error": error,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "call": n})
            if self.spend_file:
                line = {"at": _now(), "source": "llm:%s" % tag, "api_usd": billed}
                if note:
                    line["note"] = note
                _append(self.spend_file, line)
        finally:
            _unblock(launch.mask)
        if launch.stop is not None:
            raise launch.stop
        return LlmResult(output, cost, error, log, session)
