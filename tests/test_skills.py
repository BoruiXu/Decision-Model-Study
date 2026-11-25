"""Checks that keep the skill files consistent with the rsi code.

skill_sections(path) returns the second-level headings of a skill file and
skill_section(path, title) the text under one of them; the module-skill
tests reuse both.
"""
import contextlib
import importlib
import io
import os
import re
import shlex
import sys
import types
import unittest
from unittest import mock

from rsi import briefs, cli, engine
from rsi.rules import APPROVAL_ITEMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS = os.path.join(ROOT, "skills")
ORCH_DIR = os.path.join(SKILLS, "rsi-orchestrator")
ORCH = os.path.join(ORCH_DIR, "SKILL.md")
# Each module keeps its skill, code and references in modules/<module>/ (D8); the engine hands the agent
# modules/<module>/SKILL.md. The operator skill is not a module and stays in skills/.
MODULES = os.path.join(ROOT, "modules")
MODULE_SKILLS = ("data", "train", "deliver", "judge")
TEMPLATE_NAME = "_template"
TRAIN_COMMAND = "<repo root>/modules/train/train"
ORCH_SECTIONS = ("概述", "入口", "任务接入", "运行", "回答问题", "进度", "审批点", "异常处理", "常见错误")
# The old step-by-step skill had these sections; the engine does that work now.
OLD_ORCH_SECTIONS = ("主流程", "派 subagent", "交回之后", "闭环", "长任务")
CLI_COMMANDS = tuple(name for name, _help in cli.COMMANDS)
# The engine runs these; the operator skill never tells anyone to (rsi.engine / rsi.decisions do).
ENGINE_COMMANDS = ("promote", "next", "close-round")
# Vocabulary of the brief and handback.json that module skills spell out in backticks.
PROTOCOL_TOKENS = set(briefs.HANDBACK_KEYS) | set(briefs.HANDBACK_STATUSES) - {"long_job"} | {
    "module", "purpose", "round", "ids", "inputs", "outdir", "goal", "sources", "dset"}
# `rsi <subcommand>`; the lookbehind keeps skill names such as rsi-orchestrator out (ruling R5).
COMMAND_RE = re.compile(r"(?<![\w-])rsi\s+([a-z][a-z-]*)")
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
# Placeholders that stand in a position with fixed choices get a valid value before parsing.
CHOICE_PLACEHOLDERS = {"<kind>": "trainset", "<item>": "plan"}
MODULE_CALL = "python3 -m rsi "
BRIEF = os.path.join(ORCH_DIR, "templates", "brief.md")
KINDS = ("task", "trainset", "valset", "evalset", "testset", "corpus", "pool", "candidate", "search",
         "delivery", "feedback", "evaluation", "plan", "record")
MODULE_SECTIONS = ("输入", "输出", "门", "单独运行", "步骤和策略子 skill", "模块指标",
                   "实验记录", "长任务", "负责人和版本", "常见错误")
MODULES_DOC = os.path.join(ROOT, "docs", "interfaces", "modules.md")
DATASETS_DOC = os.path.join(ROOT, "docs", "interfaces", "datasets.md")
INTERFACES_README = os.path.join(ROOT, "docs", "interfaces", "README.md")
TEMPLATE = os.path.join(MODULES, TEMPLATE_NAME, "SKILL.md")
ENGINE_DOC = os.path.join(ROOT, "docs", "interfaces", "engine.md")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _split_fences(text):
    """Yield (line, in_fence) for each line; fence marker lines count as in the fence."""
    fence = None
    for line in text.splitlines():
        marker = line.lstrip()[:3]
        if marker in ("```", "~~~"):
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            yield line, True
            continue
        yield line, fence is not None


def skill_sections(path):
    """Second-level headings of a Markdown file, without the leading "## ", in order.

    Headings inside fenced code blocks are ignored.
    """
    return [line[3:].strip() for line, fenced in _split_fences(read(path))
            if not fenced and line.startswith("## ")]


def skill_section(path, title):
    """Text under the second-level heading `title`, up to the next heading of level 1 or 2."""
    out, inside = [], False
    for line, fenced in _split_fences(read(path)):
        if not fenced and (line.startswith("## ") or line.startswith("# ")):
            if inside:
                break
            inside = line.startswith("## ") and line[3:].strip() == title
            continue
        if inside:
            out.append(line)
    if not inside:
        raise AssertionError("%s has no section '## %s'" % (path, title))
    return "\n".join(out)


def frontmatter(path):
    """key -> value of the YAML frontmatter (simple `key: value` lines only)."""
    lines = read(path).splitlines()
    if not lines or lines[0].strip() != "---":
        raise AssertionError("%s does not start with a --- frontmatter block" % path)
    out = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith(" "):
            out[key.strip()] = value.strip().strip("'\"")
    raise AssertionError("%s: frontmatter is not closed by ---" % path)


def _as_command(snippet):
    """`rsi ...` from a snippet that is `rsi ...` or contains `python3 -m rsi ...`, else None."""
    snippet = snippet.strip()
    if snippet.startswith("rsi "):
        return snippet
    head, sep, rest = snippet.partition(MODULE_CALL)
    return "rsi " + rest if sep else None


def command_lines(text):
    """`rsi ...` commands written in inline code spans or on lines of fenced code blocks.

    The full invocation `PYTHONPATH=<repo root> python3 -m rsi <subcommand> ...`
    counts as `rsi <subcommand> ...`.
    """
    found = []
    for line, fenced in _split_fences(text):
        snippets = [line] if fenced else CODE_SPAN_RE.findall(line)
        found += [c for c in map(_as_command, snippets) if c]
    return found


# Shell words that end the rsi part of a command line (a redirection, a pipe, the next command).
SHELL_OPERATORS = (">", ">>", "<", "|", "||", "&&", "&", ";", "2>", "2>&1")


def parse_error(command):
    """None when `command` parses with the real rsi argument parser, else the parser's message. A shell operator
    (SHELL_OPERATORS, or a word ending in ";") ends the rsi part: what follows is the shell's, not rsi's."""
    try:
        words = shlex.split(command)[1:]
    except ValueError as e:
        return str(e)
    tokens = []
    for t in words:
        if t in SHELL_OPERATORS:
            break
        if t.endswith(";"):
            tokens.append(t[:-1])
            break
        tokens.append(t)
    tokens = [CHOICE_PLACEHOLDERS.get(t, t) for t in tokens]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            cli.build_parser().parse_args(tokens)
    except SystemExit:
        return err.getvalue().strip() or "argument error"
    return None


TRAIN_DIR = os.path.join(ROOT, "modules", "train")
# trsi's subcommands (modules/train/trsi/cli.py); the train skill's flow uses every one of them.
TRSI_COMMANDS = ("init", "check", "adopt", "launch", "watch", "collect", "next", "report")
# trsi options that take a number: a placeholder after one of them gets 0 before parsing.
TRSI_INT_OPTIONS = ("--seed", "--gpu", "--poll")
_TRSI = []


def trsi_cli():
    """trsi's command line module, imported from modules/train without running anything. trsi needs pyyaml to
    work, not to parse its command line: when this interpreter has no pyyaml (the system python3), an empty
    stand-in module named yaml is in sys.modules for the import only."""
    if not _TRSI:
        stub = None
        try:
            import yaml  # noqa: F401
        except ImportError:
            stub = sys.modules["yaml"] = types.ModuleType("yaml")
        sys.path.insert(0, TRAIN_DIR)
        try:
            _TRSI.append(importlib.import_module("trsi.cli"))
        finally:
            sys.path.remove(TRAIN_DIR)
            if stub is not None and sys.modules.get("yaml") is stub:
                del sys.modules["yaml"]
    return _TRSI[0]


def tool_lines(text, tool):
    """The words after `<tool>` of every command of a module's command line tool (trsi, dset) in inline code spans
    or on lines of fenced code blocks: `<tool> <command> ...`, or a line that runs `... -m <tool> <command> ...`. A
    shell operator (SHELL_OPERATORS, a word starting with >, | or &, or ending in ;) ends the tool's part."""
    prefix, call = tool + " ", "-m %s " % tool
    found = []
    for line, fenced in _split_fences(text):
        for snippet in ([line] if fenced else CODE_SPAN_RE.findall(line)):
            snippet = snippet.strip()
            if snippet.startswith(prefix):
                rest = snippet[len(prefix):]
            elif call in snippet:
                rest = snippet.split(call, 1)[1]
            else:
                continue
            words = []
            for w in rest.split():
                if w in SHELL_OPERATORS or w.startswith((">", "|", "&")):
                    break
                if w.endswith(";"):
                    words.append(w[:-1])
                    break
                words.append(w)
            found.append(words)
    return found


def trsi_lines(text):
    """The words after `trsi` of every trsi command (tool_lines)."""
    return tool_lines(text, "trsi")


def trsi_parse_error(words):
    """None when `trsi <words>` parses with trsi's real argument parser (its command handlers replaced, so nothing
    runs), else the parser's message."""
    cli_mod = trsi_cli()
    try:
        words = shlex.split(" ".join(words))
    except ValueError as e:
        return str(e)
    words = ["0" if i and words[i - 1] in TRSI_INT_OPTIONS and w.startswith("<") else w for i, w in enumerate(words)]
    ran, err = [], io.StringIO()
    with contextlib.ExitStack() as stack:
        for name in [n for n in dir(cli_mod) if n.startswith("cmd_")]:
            stack.enter_context(mock.patch.object(cli_mod, name, lambda a, name=name: ran.append(name) or 0))
        stack.enter_context(contextlib.redirect_stderr(err))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        try:
            cli_mod.main(words)
        except SystemExit:
            return err.getvalue().strip() or "argument error"
    return None if ran else "no trsi command ran"


DATA_DIR = os.path.join(ROOT, "modules", "data")
# dset options that take a number: a placeholder after one of them gets 1 before parsing.
DSET_INT_OPTIONS = ("--round", "--add")
_DSET = []


def dset_cli():
    """dset's command line module (modules/data/dset/cli.py), imported without running anything."""
    if not _DSET:
        sys.path.insert(0, DATA_DIR)
        try:
            _DSET.append(importlib.import_module("dset.cli"))
        finally:
            sys.path.remove(DATA_DIR)
    return _DSET[0]


def dset_lines(text):
    """The words after `dset` of every dset command (tool_lines). The form `dset <命令> …` (a placeholder or an
    option where the command goes) names no command and is left out."""
    return [w for w in tool_lines(text, "dset") if w and not w[0].startswith(("<", "-"))]


def dset_parse_error(parser, words):
    """None when `dset <words>` parses with dset's own parser (parse_args only: no command runs), else its message.
    dset's parser raises DsetError where argparse would exit. A bare `dset <command>` names the command, it is no
    call: it only has to be one of dset's commands (its --help exits 0)."""
    try:
        words = shlex.split(" ".join(words))
    except ValueError as e:
        return str(e)
    bare = len(words) == 1
    words = ["1" if i and words[i - 1] in DSET_INT_OPTIONS and w.startswith("<") else w for i, w in enumerate(words)]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            parser.parse_args(words + ["--help"] if bare else words)
    except SystemExit as e:
        return None if bare and e.code == 0 else (err.getvalue().strip() or "argument error")
    except dset_cli().DsetError as e:
        return str(e)
    return None


def orchestrator_files():
    """SKILL.md and every Markdown template of the orchestrator skill."""
    paths = [ORCH]
    tdir = os.path.join(ORCH_DIR, "templates")
    if os.path.isdir(tdir):
        paths += [os.path.join(tdir, n) for n in sorted(os.listdir(tdir)) if n.endswith(".md")]
    return paths


class TestOrchestratorSkill(unittest.TestCase):
    def test_orchestrator_sections_and_no_flow_table(self):
        sections = skill_sections(ORCH)
        for title in ORCH_SECTIONS:
            self.assertIn(title, sections, "rsi-orchestrator lacks '## %s'" % title)
        # The engine owns the flow: no step table, no dispatch instructions, no per-step registration.
        text = read(ORCH)
        for title in OLD_ORCH_SECTIONS:
            self.assertNotIn(title, sections, "the operator skill still has '## %s'" % title)
        for old in ("| 步骤 |", "① 数据", "templates/brief.md", "rows.jsonl", "labels.jsonl", "原目录"):
            self.assertNotIn(old, text)
        self.assertLess(len(text.splitlines()), 130)
        self.assertIn("不决定流程", text)

    def test_orchestrator_uses_dashboard(self):
        text = read(ORCH)
        self.assertIn("<ws>/dashboard.html", text)
        self.assertIn("rsi dashboard <ws> --no-open", text)
        self.assertIn("rsi dashboard <ws>", skill_section(ORCH, "入口"))
        self.assertIn("现在在等什么", skill_section(ORCH, "进度"))
        self.assertIn("● 运行中", skill_section(ORCH, "进度"))

    def test_orchestrator_commands_exist(self):
        used = set()
        for path in orchestrator_files():
            for name in COMMAND_RE.findall(read(path)):
                self.assertIn(name, CLI_COMMANDS, "%s: `rsi %s` is not an rsi subcommand"
                              % (os.path.relpath(path, ROOT), name))
                used.add(name)
        # The operator uses every command that is not the engine's (templates add score and brief).
        self.assertEqual(sorted(set(CLI_COMMANDS) - used), sorted(ENGINE_COMMANDS),
                         "rsi subcommands the operator skill never calls (should be the engine's only)")
        for name in ("init", "validate", "run", "answer", "approve", "status", "stop", "report", "dashboard"):
            self.assertIn(name, COMMAND_RE.findall(read(ORCH)), name)
        # The operator never runs the engine's commands.
        for command in command_lines(read(ORCH)):
            self.assertNotIn(command.split()[1], ENGINE_COMMANDS, command)
        for path in orchestrator_files():
            self.assertEqual([n for n in COMMAND_RE.findall(read(path)) if n in ENGINE_COMMANDS], [], path)

    def test_orchestrator_command_forms_parse(self):
        checked = 0
        for path in orchestrator_files():
            for command in command_lines(read(path)):
                if len(command.split()) <= 2:
                    continue  # a bare `rsi <subcommand>` names the command, it is not a call
                error = parse_error(command)
                self.assertIsNone(error, "%s: `%s` does not match the CLI: %s"
                                  % (os.path.relpath(path, ROOT), command, error))
                checked += 1
        self.assertGreater(checked, 0)

    def test_exit_codes_match_the_engine(self):
        section = skill_section(ORCH, "运行")
        rows = {}
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("|") and cells[0].isdigit():
                rows[int(cells[0])] = cells[1]
        self.assertEqual(sorted(rows), sorted(set(engine.STATUS_EXIT_CODES.values()) | {1}))
        for status, code in engine.STATUS_EXIT_CODES.items():
            self.assertIn(status, section)  # the --json statuses
        self.assertIn("rsi run", section)

    def test_background_run_recipe_and_notes(self):
        text = skill_section(ORCH, "运行")
        self.assertIn("PYTHONPATH=<repo root> python3 -m rsi run <ws> --json", text)
        # P11 (sim2): one output file per run and its exit code beside it, so a run the lock refuses never
        # overwrites the output of the run before it.
        self.assertIn("<ws>/work/run-out-", text)
        self.assertIn('echo $? > "$out.exit"', text)
        self.assertNotIn("<ws>/work/run-out.json", text)
        # I1 (final review): the background command itself prints the file name first (shell variables do not
        # outlive a Bash call and the run's own output goes to the file), and the operator takes the name from there.
        recipe = [line for line, fenced in _split_fences(text) if fenced and "python3 -m rsi run" in line]
        self.assertEqual(len(recipe), 1, recipe)
        line = recipe[0]
        self.assertLess(line.index('echo "$out"'), line.index("python3 -m rsi run"))
        self.assertIn('--json > "$out.json" 2> "$out.err"', line)  # the status lines go to <name>.err
        self.assertTrue(line.startswith("out=<ws>/work/run-out-"), line)
        for sec in ("运行", "进度"):
            self.assertIn("tail -n 3 <文件名>.err", skill_section(ORCH, sec), sec)
        for words in ("先打印", "文件名", "不要猜"):
            self.assertIn(words, text)
        self.assertIn("运行锁", read(ORCH))  # a stop during a run: wait for the run, then run once more
        self.assertIn("## 待用户回答", skill_section(ORCH, "进度"))

    def test_plan_and_code_defaults_are_told_right(self):
        # P12 (sim2): a plan has no cost estimate; the registered plan's place; approvals, the milestone and a
        # module's question have no code default.
        text = read(ORCH)
        self.assertNotIn("预计花费", text)
        answer = skill_section(ORCH, "回答问题")
        self.assertIn("<ws>/artifacts/<plan_id>/manifest.json", answer)
        self.assertIn("没有代码默认", answer)
        for kind in ("批准类", "里程碑", "模块交回时的提问"):
            self.assertIn(kind, answer)

    def test_no_old_wording_of_the_top_level_doing_the_engines_work(self):
        # P3 (sim2): the engine (`rsi run`) registers, decides the round, writes the report; nothing says the
        # orchestrator ("顶层") does.
        from rsi import briefs as B, dashboard, dashboard_pages
        old = ("由顶层", "等顶层", "顶层登记", "顶层按此顺序登记", "登记只由顶层做", "等顶层告诉你", "主流程表为准")
        paths = [module_path(n) for n in MODULE_SKILLS + (TEMPLATE_NAME,)] + orchestrator_files() + [
            B.__file__, dashboard.__file__, dashboard_pages.__file__, os.path.join(ROOT, "rsi", "report.py"),
            ENGINE_DOC, os.path.join(ROOT, "docs", "interfaces", "task.md"), os.path.join(ROOT, "README.md")]
        for p in paths:
            text = read(p)
            for phrase in old:
                self.assertNotIn(phrase, text, "%s still says %s" % (os.path.relpath(p, ROOT), phrase))

    def test_answer_section_matches_the_cli(self):
        section = skill_section(ORCH, "回答问题")
        self.assertIn("rsi answer <ws> <option> --by <name> --quote", section)
        for option in ("approve", "revise", "stop", "resume", "new_session", "close_round", "train", "data",
                       "yes", "no", "已回答"):
            self.assertIn("`%s`" % option, section, option)
        self.assertIn("没有 `resume`", section)  # R11: the train gate question has no resume option
        self.assertIn("rsi approve", section)

    def test_over_budget_is_answered_with_commands_that_parse(self):
        # B1 / S1 (third simulation): the over_budget question (before a training, or a hand-back held over the budget)
        # is relayed as it is and answered approve (rsi answer, or rsi approve <ws> over_budget) or stop; approved, the
        # round runs to its decision, the held hand-back is registered without retraining, and the budget stops the run.
        forms = ('rsi answer <ws> approve --ref <qid> --by <name> --quote "<用户原话>"',
                 'rsi approve <ws> over_budget --by <name> --quote "<用户原话>"',
                 'rsi answer <ws> stop --ref <qid> --by <name> --quote "<用户原话>"')
        answer = skill_section(ORCH, "回答问题")
        item = [line for line in answer.splitlines() if "`over_budget`" in line]
        self.assertEqual(len(item), 1, item)
        found = command_lines(item[0])
        for form in forms:
            self.assertIn(form, found)
            self.assertIsNone(parse_error(form), form)
        for words in ("原文", "交回先不登记", "不重新训练", "判定后运行因算力达到预算停止", "候选不登记"):
            self.assertIn(words, item[0])
        row = [line for line in skill_section(ORCH, "审批点").splitlines() if line.startswith("| `over_budget` |")]
        self.assertEqual(len(row), 1, row)
        for form in forms[:2]:
            form = form.replace(" --ref <qid>", "")
            self.assertIn(form, command_lines(row[0]))
            self.assertIsNone(parse_error(form), form)
        self.assertNotIn("派活前预计超出", row[0])  # the old wording: the question also comes after a hand-back
        self.assertIn("运行因预算停止", row[0])

    def test_orchestrator_lists_all_approval_items(self):
        section = skill_section(ORCH, "审批点")
        for item in APPROVAL_ITEMS:
            self.assertIn("`%s`" % item, section, "审批点 does not list %s" % item)

    def test_brief_gives_modules_no_workspace(self):
        # R21: modules get input paths and an output directory, never the workspace root,
        # and only the orchestrator registers. R23: the brief carries the exact invocations.
        # R36: besides `rsi validate`, modules run `rsi score` for the numbers they report.
        text = read(BRIEF)
        self.assertNotIn("<ws>", text)
        calls = [c for c in command_lines(text) if len(c.split()) > 2]
        self.assertEqual(sorted(set(c.split()[1] for c in calls)), ["score", "validate"])
        for c in calls:
            self.assertIsNone(parse_error(c), c)
        self.assertIn("PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>", text)
        self.assertIn("PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels ", text)

    def test_orchestrator_templates_exist(self):
        refs = set(re.findall(r"templates/[\w.-]+", read(ORCH)))
        self.assertIn("templates/TASK.md", refs)
        self.assertIn("templates/task.example.json", refs)
        for ref in sorted(refs):
            self.assertTrue(os.path.isfile(os.path.join(ORCH_DIR, ref)), "%s does not exist" % ref)
        for name in MODULE_SKILLS:  # the code-built brief is documented in templates/brief.md
            self.assertIn("templates/brief.md", read(module_path(name)), name)
        self.assertTrue(os.path.isfile(BRIEF))


def module_path(name):
    return os.path.join(MODULES, name, "SKILL.md")


def lower_tokens(text):
    """Backticked tokens made only of lowercase letters (ruling R6), without the vocabulary of the brief and
    handback.json (PROTOCOL_TOKENS), which module skills spell out in the same sections."""
    return [t for t in CODE_SPAN_RE.findall(text) if re.fullmatch(r"[a-z]+", t) and t not in PROTOCOL_TOKENS]


class TestModuleSkills(unittest.TestCase):
    def test_module_skills_have_required_sections(self):
        for path in [TEMPLATE] + [module_path(n) for n in MODULE_SKILLS]:
            sections = skill_sections(path)
            for title in MODULE_SECTIONS:
                self.assertIn(title, sections, "%s lacks '## %s'" % (os.path.relpath(path, ROOT), title))

    def test_module_io_kinds_exist(self):
        for name in MODULE_SKILLS:
            for title in ("输入", "输出"):
                tokens = lower_tokens(skill_section(module_path(name), title))
                self.assertTrue(tokens, "%s: ## %s names no artifact kind" % (name, title))
                for token in tokens:
                    self.assertIn(token, KINDS, "%s: ## %s: `%s` is not an artifact kind" % (name, title, token))

    def test_module_skills_describe_brief_and_handback(self):
        # The brief arrives as brief.md/brief.json, the module answers with handback.json (format of
        # rsi.briefs), a redo is a resumed session with feedback-<k>.md, a long job is a long_job hand-back.
        for name in MODULE_SKILLS + (TEMPLATE_NAME,):
            path = module_path(name)
            text = read(path)
            self.assertIn("brief.md", skill_section(path, "输入"), name)
            self.assertIn("brief.json", skill_section(path, "输入"), name)
            self.assertIn("feedback-<k>.md", skill_section(path, "输入"), name)
            out = skill_section(path, "输出")
            self.assertIn("handback.json", out, name)
            for key in briefs.HANDBACK_KEYS:
                self.assertIn("`%s`" % key, out, "%s: handback key %s" % (name, key))
            for status in briefs.HANDBACK_STATUSES:
                self.assertIn("`%s`" % status, out, "%s: handback status %s" % (name, status))
            long_job = skill_section(path, "长任务")
            self.assertIn("long_job", long_job, name)
            self.assertIn("marker", long_job, name)
            self.assertIn("handback.json", long_job, name)
            self.assertNotIn("<ws>", text, name)

    def test_module_skills_state_the_question_and_long_job_contracts(self):
        # P1/P2 (sim2): a hand-back with questions registers nothing until the user's answer comes back to the same
        # session; a long job writes <marker>.pid naming the job process itself and is terminated on a stop; the
        # second part of a two-part hand-back comes as a followup, no return. The wordings are the engine's.
        source = read(engine.__file__)
        for head in ("# 用户回答了你的问题", "# 继续：交回的第二部分", "# 长任务已结束"):
            self.assertIn(head, source)
        for name in MODULE_SKILLS + (TEMPLATE_NAME,):
            path = module_path(name)
            text = read(path)
            out = skill_section(path, "输出")
            self.assertIn("一个都不登记", out, name)
            self.assertIn("# 用户回答了你的问题", out, name)
            self.assertIn("同一个会话", out, name)
            inp = skill_section(path, "输入")
            self.assertIn("# 继续：交回的第二部分", inp, name)
            self.assertIn("不算退回", inp, name)
            long_job = skill_section(path, "长任务")
            for token in ("<marker>.pid", "date +%s", "$$", "os.getpid()", "终止", "rm -f <marker> <marker>.pid"):
                self.assertIn(token, long_job, "%s: %s" % (name, token))
            self.assertNotIn("$!", text, name)  # in a shell it may name a wrapper, not the job
            self.assertNotIn("setsid", text, name)  # no such command on macOS
        # M2 (final review): a long job's questions are asked once its marker is there; without artifacts listed the
        # answer comes with "# 长任务已结束".
        for name in MODULE_SKILLS + (TEMPLATE_NAME,):
            out = skill_section(module_path(name), "输出")
            self.assertIn("完成标记出现后才问", out, name)
            self.assertIn("「# 长任务已结束」", out, name)
        for where, text in (("briefs.HANDBACK_FORMAT", briefs.HANDBACK_FORMAT), ("engine.md", read(ENGINE_DOC)),
                            ("templates/brief.md", read(BRIEF))):
            self.assertIn("完成标记出现后才问", text, where)
        # M5: the README's rsi answer row says a module's question holds its hand-back until the answer.
        row = [line for line in read(os.path.join(ROOT, "README.md")).splitlines() if line.startswith("| `rsi answer` |")]
        self.assertEqual(len(row), 1)
        self.assertIn("不登记", row[0])
        self.assertIn("同一个会话", row[0])
        self.assertNotIn("下次续跑时转给它", row[0])
        for where, text in (("briefs.HANDBACK_FORMAT", briefs.HANDBACK_FORMAT), ("engine.md", read(ENGINE_DOC))):
            self.assertIn(".pid", text, where)
            self.assertIn("不登记", text, where)
            self.assertNotIn("$!", text, where)
            self.assertNotIn("setsid", text, where)

    def test_deliver_never_gets_labels_and_judge_never_reads_valset(self):
        self.assertNotIn("labels.jsonl", skill_section(module_path("deliver"), "输入"))
        self.assertIn("inputs.jsonl", skill_section(module_path("deliver"), "输入"))
        self.assertIn("labels.jsonl", skill_section(module_path("deliver"), "门"))
        # R25: valset is for data, train and tune only (also in the interface overview table, I1).
        self.assertNotIn("valset", lower_tokens(skill_section(module_path("judge"), "输入")))
        for path in (MODULES_DOC, DATASETS_DOC, INTERFACES_README):
            self.assertTrue(any(line.startswith("| `valset` |") for line in read(path).splitlines()),
                            "%s has no valset row" % path)
        for line in read(MODULES_DOC).splitlines() + read(DATASETS_DOC).splitlines() + \
                read(INTERFACES_README).splitlines():
            if line.startswith("| `valset` |") and "不读" in line:
                self.assertEqual(line.split("|")[-2].strip(), "不读", line)  # judge column
            if line.startswith("| `valset` |") and "judge" in line:
                self.fail("valset visible to judge: " + line)

    def test_train_command_contract(self):
        # R26: fixed path, one producer (tune was merged into train; the data module does not train), marker outside
        # the candidate directory.
        for p in (module_path("train"), MODULES_DOC):
            self.assertIn(TRAIN_COMMAND, read(p), p)
        train = read(module_path("train"))
        for token in ("--marker", "--producer train", "unregistered"):
            self.assertIn(token, train)
        self.assertNotIn("输出目录下的 `DONE`", train)
        # 2026-11-25: the data module does not train (no A/B of trainsets); ② and ④ judge its trainset.
        data = read(module_path("data")).split("## 负责人和版本")[0]  # the change log keeps the history
        for token in (TRAIN_COMMAND, "--producer data", "ab-<k>-old", "unregistered"):
            self.assertNotIn(token, data)
        self.assertIn("不训练", skill_section(module_path("data"), "模块指标"))
        # S5: every place that names the train command also names the override and the default.
        docs = os.path.join(ROOT, "docs", "interfaces")
        paths = [module_path("train")] + [
            MODULES_DOC, os.path.join(docs, "models.md"), os.path.join(docs, "task.md"),
            os.path.join(ROOT, "README.md"), os.path.join(ORCH_DIR, "templates", "TASK.md"), ORCH]
        for p in paths:
            text = read(p)
            self.assertIn("resources.train_command", text, p)
            self.assertIn(TRAIN_COMMAND, text, p)
        self.assertIn("PYTHONPATH=<repo root>", skill_section(module_path("train"), "输出"))
        self.assertRegex(read(BRIEF), r"训练命令：")
        self.assertEqual(os.path.join(ROOT, briefs.DEFAULT_TRAIN_COMMAND),
                         TRAIN_COMMAND.replace("<repo root>", ROOT).replace("/", os.sep))
        self.assertTrue(os.path.isfile(os.path.join(ROOT, briefs.DEFAULT_TRAIN_COMMAND)))

    def test_no_old_module_paths(self):
        # D8: the module skills moved from skills/rsi-<module>/ to modules/<module>/; skills/rsi-tune is gone.
        # Historical plans (docs/superpowers/) keep the old names.
        old = re.compile(r"skills/(rsi-(data|train|tune|deliver|judge|<)|_module_template)|rsi-tune/")
        roots = [os.path.join(ROOT, d) for d in ("rsi", "skills", "modules", "examples")]
        files = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "docs", "plan.md"),
                 os.path.join(ROOT, "docs", "design.md")]
        files += [os.path.join(ROOT, "docs", "interfaces", n) for n in os.listdir(os.path.join(ROOT, "docs", "interfaces"))]
        for top in roots:
            for d, _dirs, names in os.walk(top):
                files += [os.path.join(d, n) for n in names if n.endswith((".md", ".py", ".json")) or n == "train"]
        checked = 0
        for p in files:
            if "__pycache__" in p:
                continue
            for i, line in enumerate(read(p).splitlines(), 1):
                self.assertIsNone(old.search(line), "%s:%d still names an old module path: %s"
                                  % (os.path.relpath(p, ROOT), i, line.strip()[:120]))
            checked += 1
        self.assertGreater(checked, 20)

    def test_no_separate_tune_stage(self):
        # Live status §2: the stages are ① 数据 → ② 训练调优 → ③ 交付 → ④ 评测与决策; tuning is no module, stage, skill
        # or option of its own (an old workspace's tune is read only, under ②).
        docs = os.path.join(ROOT, "docs", "interfaces")
        paths = [ORCH, BRIEF, os.path.join(ROOT, "README.md")] + [
            os.path.join(MODULES, n, "SKILL.md") for n in sorted(os.listdir(MODULES))
            if os.path.isfile(os.path.join(MODULES, n, "SKILL.md"))] + [
            os.path.join(docs, n) for n in sorted(os.listdir(docs)) if n.endswith(".md")]
        for p in paths:
            text = read(p)
            for stale in ("rsi-tune", "③ 调优", "④ 交付", "⑤"):
                self.assertNotIn(stale, text, "%s still says %s" % (os.path.relpath(p, ROOT), stale))
        for p in (ORCH, BRIEF):
            self.assertNotIn("`tune`", read(p), p)
        for p in (os.path.join(ROOT, "README.md"), MODULES_DOC):
            self.assertIn("① 数据 → ② 训练调优 → ③ 交付 → ④ 评测与决策", read(p), p)

    def test_modules_use_only_validate_and_score(self):
        # R21/R36: modules run `rsi validate` and `rsi score`, nothing else; every form parses.
        for name in MODULE_SKILLS + (TEMPLATE_NAME,):
            calls = [c for c in command_lines(read(module_path(name))) if len(c.split()) > 2]
            self.assertTrue(calls, name)
            for c in calls:
                self.assertIn(c.split()[1], ("validate", "score"), "%s: %s" % (name, c))
                self.assertIsNone(parse_error(c), "%s: %s" % (name, c))
            self.assertNotIn("<ws>", read(module_path(name)))
        # R36: train and judge give the `rsi score` forms for what they report.
        def score_calls(name):
            return [c for c in command_lines(read(module_path(name)))
                    if len(c.split()) > 2 and c.split()[1] == "score"]
        train = score_calls("train")
        self.assertTrue(any("--train" in c for c in train), train)
        judge = score_calls("judge")
        self.assertTrue(any("--compare" in c and "--clusters" in c and "--train" in c for c in judge), judge)
        self.assertTrue(any("testset" in c for c in judge), judge)  # milestone metrics_test
        self.assertIn("rsi score", skill_section(module_path("train"), "门"))  # the search's numbers too
        # The judge gets the candidate's trainset for the baselines, as the orchestrator table says.
        self.assertIn("trainset", lower_tokens(skill_section(module_path("judge"), "输入")))

    def test_runnable_candidates_and_reproduction_in_skills(self):
        # Task 14: deliver reproduces the candidate on the valset inputs (never labels); train writes the
        # prediction entry and code version, and runs one code version a search; scoring is not owner-defined.
        deliver_in = skill_section(module_path("deliver"), "输入")
        self.assertIn("valset", lower_tokens(deliver_in))
        self.assertNotIn("rows.jsonl", deliver_in)
        deliver_gate = skill_section(module_path("deliver"), "门")
        for text in ("predictions_val", "1e-6", "package.location", "external:<名字>", "cost_unit"):
            self.assertIn(text, deliver_gate)
        train = read(module_path("train"))
        for text in ("model.predict", "--inputs <inputs.jsonl> --out <predictions.jsonl>", "code_version"):
            self.assertIn(text, train)
        self.assertIn("code_version.sha256", skill_section(module_path("train"), "门"))  # one code version a search
        self.assertIn("inputs.jsonl", skill_section(module_path("data"), "输出"))
        judge_owner = skill_section(module_path("judge"), "步骤和策略子 skill")
        self.assertNotIn("由模块负责人填写：打分", judge_owner)
        self.assertIn("rsi score", judge_owner)
        self.assertIn("登记后路径", skill_section(ORCH, "运行"))


class TestTrainSkill(unittest.TestCase):
    """modules/train/SKILL.md: CD's Training RSI skill rewritten as the framework's training-tuning module (D17)."""

    def test_trsi_command_lines_parse(self):
        # Every trsi command line the skill quotes parses with trsi's own parser; the flow uses every trsi command.
        names = set(n[len("cmd_"):] for n in dir(trsi_cli()) if n.startswith("cmd_"))
        self.assertEqual(names, set(TRSI_COMMANDS))
        used, checked = set(), 0
        for words in trsi_lines(read(module_path("train"))):
            if not words or words[0].startswith("<") or words in (["--help"], ["--version"]):
                continue  # `trsi <命令>` stands for any command; --help names none
            self.assertIn(words[0], names, words)
            used.add(words[0])
            if len(words) == 1:
                continue  # a bare `trsi collect` names the command, it is not a call
            error = trsi_parse_error(words)
            self.assertIsNone(error, "modules/train/SKILL.md: `trsi %s` does not match trsi's command line: %s"
                              % (" ".join(words), error))
            checked += 1
        self.assertEqual(used, set(TRSI_COMMANDS))
        self.assertGreater(checked, len(TRSI_COMMANDS))
        # Fix round 1 (I1): the search reads every registered-<n>.json of the dispatch (a return's second search
        # still names the first search's trials), and that glob form parses.
        reports = [w for w in trsi_lines(read(module_path("train"))) if w[:1] == ["report"] and "--ids" in w]
        self.assertTrue(reports)
        for w in reports:
            self.assertEqual(w[w.index("--ids") + 1], "<outdir>/registered-*.json", w)
            self.assertIsNone(trsi_parse_error(w), w)

    def test_flow_rules_and_references(self):
        text = read(module_path("train"))
        source = read(engine.__file__)
        # the engine's feedback heads
        for head in ("# 定时检查", "# 长任务已结束", "# 继续：交回的第二部分", "# 用户回答了你的问题"):
            self.assertIn(head, source)
            self.assertIn(head, text)
        for words in ("nvidia-smi", "ojh doctor", "`resources.gpu_ids`", "`search_space`", "`search.max_configs`",
                      "`note`", "`prediction`", "SIGKILL", "STOP", "unpacked", "`questions`", "ojh cleanup",
                      "诊断 → 改动 → 预测 → 结果", "<marker>.pid", "registered-<n>.json"):
            self.assertIn(words, text)
        # Fix round 1: the check-in heading in the flow and the long-job section; a restarted trial is relaunched only
        # once collect (given the whole Bash timeout) shows it failed; a background job runs in its own session (a stop
        # kills its process group); YAML exponents keep a dot.
        for title in ("步骤和策略子 skill", "长任务"):
            self.assertIn("「# 定时检查」", skill_section(module_path("train"), title), title)
        for words in ("600000", "failed 后再", "start_new_session=True", "2.0e-5", "registered-*.json"):
            self.assertIn(words, text)
        refs = set(re.findall(r"reference/[\w.-]+\.md", text))
        self.assertEqual(refs, {"reference/search_space.md", "reference/protocol.md", "reference/lessons.md"})
        for ref in refs:
            self.assertTrue(os.path.isfile(os.path.join(TRAIN_DIR, ref)), ref)
        self.assertNotIn("A100", read(os.path.join(TRAIN_DIR, "reference", "search_space.md")))  # T18: no machine


class TestDataSkill(unittest.TestCase):
    """modules/data/SKILL.md: the data module's steps with dset (modules/data/dset)."""

    def test_dset_command_lines_parse(self):
        # Every `dset <command> …` line of the data skill parses with dset's own parser; the skill uses all four.
        parser = dset_cli().build_parser()
        used = set()
        for words in dset_lines(read(module_path("data"))):       # the words after `dset`
            used.add(words[0])
            self.assertIsNone(dset_parse_error(parser, words), words)
        self.assertEqual(used, {"sets", "grow", "check", "refresh"})
        steps = skill_section(module_path("data"), "步骤和策略子 skill")
        for token in ("question_of", "split.json", "--dry-run", "record"):
            self.assertIn(token, steps)


class TestSkillFiles(unittest.TestCase):
    def test_frontmatter(self):
        # skills/ holds the operator skill only; every module's skill is modules/<module>/SKILL.md, named rsi-<module>.
        names = sorted(n for n in os.listdir(SKILLS)
                       if os.path.isdir(os.path.join(SKILLS, n)) and not n.startswith((".", "_")))
        self.assertEqual(names, ["rsi-orchestrator"])
        fm = frontmatter(ORCH)
        self.assertEqual(fm.get("name"), "rsi-orchestrator")
        self.assertTrue(fm.get("description"))
        folders = sorted(n for n in os.listdir(MODULES)
                         if os.path.isdir(os.path.join(MODULES, n)) and not n.startswith((".", "_")))
        self.assertEqual(folders, sorted(briefs.MODULE_PURPOSES))
        self.assertEqual(sorted(MODULE_SKILLS), folders)
        for name in folders:
            fm = frontmatter(module_path(name))
            self.assertEqual(fm.get("name"), "rsi-" + name, "%s: frontmatter name" % name)
            self.assertEqual(fm.get("name"), briefs.SKILLS[name])
            self.assertTrue(fm.get("description"), "%s: frontmatter description is empty" % name)
        self.assertEqual(frontmatter(TEMPLATE).get("name"), "rsi-<模块名>")

    def test_engine_gives_each_module_its_folders_skill(self):
        # R11: the engine hands each module agent modules/<module>/SKILL.md.
        for module in briefs.MODULE_PURPOSES:
            path = briefs.skill_path(module)
            self.assertEqual(path, module_path(module))
            self.assertTrue(os.path.isfile(path), path)
        source = read(engine.__file__)
        self.assertIn("B.skill_path(", source)
        self.assertNotIn('"skills", B.SKILLS', source)


class TestHelpers(unittest.TestCase):
    """The parsing helpers the checks above rely on (so that a check cannot pass vacuously)."""

    def test_helpers(self):
        self.assertEqual(COMMAND_RE.findall("load rsi-orchestrator, then run rsi status <ws>"), ["status"])
        self.assertEqual(COMMAND_RE.findall("`rsi check-stop <ws>` and rsi-judge"), ["check-stop"])
        # A shell redirection or a following command ends the rsi part, which is still parsed in full.
        self.assertIsNone(parse_error('rsi run <ws> --json > "$out.json"; echo $? > "$out.exit"'))
        self.assertIsNotNone(parse_error('rsi run <ws> --bogus > "$out.json"'))
        self.assertIsNotNone(parse_error("rsi run --json; echo x"))
        self.assertIsNone(parse_error("rsi register <ws> <kind> <dir>"))
        self.assertIsNone(parse_error('rsi approve <ws> plan --ref <plan_id> --by <name> --quote "a b"'))
        self.assertIsNotNone(parse_error("rsi register <ws> model <dir>"))
        self.assertIsNotNone(parse_error("rsi approve <ws> plan"))
        text = ("run `rsi status <ws>` then `PYTHONPATH=<repo root> python3 -m rsi validate task <d>`"
                " and `rsi-judge`\n```\nrsi next <ws>\n```\n")
        self.assertEqual(command_lines(text), ["rsi status <ws>", "rsi validate task <d>", "rsi next <ws>"])
        # trsi command lines: the trsi part ends at a shell operator; trsi's own parser judges it.
        self.assertEqual(trsi_lines("`trsi next <run.yaml>` and\n```\nsh -c 'x; PYTHONPATH=<d> <py> -m trsi collect "
                                    "<run.yaml> --gpu <卡号> > <log> 2>&1' &\n```\n"),
                         [["next", "<run.yaml>"], ["collect", "<run.yaml>", "--gpu", "<卡号>"]])
        self.assertIsNone(trsi_parse_error(["collect", "<run.yaml>", "--gpu", "<卡号>"]))
        self.assertIsNone(trsi_parse_error(["launch", "<run.yaml>", "--start", "--seed", "0", "--gpus", "<卡>"]))
        self.assertIsNotNone(trsi_parse_error(["launch", "<run.yaml>", "--seed", "0"]))  # --gpus is required
        self.assertIsNotNone(trsi_parse_error(["report", "--search", "<dir>"]))  # the run.yaml is required
        self.assertIsNotNone(trsi_parse_error(["stop", "<run.yaml>"]))
        # dset command lines: the form `dset <命令> …` is left out; dset's own parser judges the rest.
        self.assertEqual(dset_lines("`dset <命令> …` and `dset check --task <t> --rows <r> > <log>`"),
                         [["check", "--task", "<t>", "--rows", "<r>"]])
        parser = dset_cli().build_parser()
        self.assertIsNone(dset_parse_error(parser, ["grow", "--task", "<t>", "--pool", "<p>", "--add", "<N>",
                                                    "--round", "<round>", "--out", "<o>"]))
        self.assertIsNotNone(dset_parse_error(parser, ["refresh", "--task", "<t>", "--corpus", "<c>", "--round", "1"]))
        self.assertIsNotNone(dset_parse_error(parser, ["check", "--task", "<t>", "--rows", "<r>", "--mode", "keep"]))
        self.assertIsNone(dset_parse_error(parser, ["sets"]))          # a bare name: a dset command
        self.assertIsNotNone(dset_parse_error(parser, ["split"]))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "SKILL.md")
            with open(p, "w", encoding="utf-8") as f:
                f.write("---\nname: x\ndescription: y\n---\n# T\n## A\ntext\n```\n## not\n```\n## B\nmore\n")
            self.assertEqual(skill_sections(p), ["A", "B"])
            self.assertEqual(skill_section(p, "A").strip(), "text\n```\n## not\n```")
            self.assertEqual(frontmatter(p), {"name": "x", "description": "y"})


if __name__ == "__main__":
    unittest.main()
