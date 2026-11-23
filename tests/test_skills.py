"""Checks that keep the skill files consistent with the rsi code.

skill_sections(path) returns the second-level headings of a skill file and
skill_section(path, title) the text under one of them; the module-skill
tests reuse both.
"""
import contextlib
import io
import os
import re
import shlex
import unittest

from rsi import cli
from rsi.rules import APPROVAL_ITEMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS = os.path.join(ROOT, "skills")
ORCH_DIR = os.path.join(SKILLS, "rsi-orchestrator")
ORCH = os.path.join(ORCH_DIR, "SKILL.md")

MODULE_SKILLS = ("rsi-data", "rsi-train", "rsi-tune", "rsi-deliver", "rsi-judge")
ORCH_SECTIONS = ("概述", "入口", "任务接入", "主流程", "派 subagent", "交回之后", "闭环",
                 "审批点", "长任务", "异常处理", "常见错误")
CLI_COMMANDS = tuple(name for name, _help in cli.COMMANDS)
# `rsi <subcommand>`; the lookbehind keeps skill names such as rsi-orchestrator out (ruling R5).
COMMAND_RE = re.compile(r"(?<![\w-])rsi\s+([a-z][a-z-]*)")
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
# Placeholders that stand in a position with fixed choices get a valid value before parsing.
CHOICE_PLACEHOLDERS = {"<kind>": "trainset", "<item>": "plan"}
MODULE_CALL = "python3 -m rsi "
BRIEF = os.path.join(ORCH_DIR, "templates", "brief.md")
KINDS = ("task", "trainset", "valset", "evalset", "testset", "candidate", "search",
         "delivery", "feedback", "evaluation", "plan", "record")
MODULE_SECTIONS = ("输入", "输出", "门", "单独运行", "步骤和策略子 skill", "模块指标",
                   "实验记录", "长任务", "负责人和版本", "常见错误")
MODULES_DOC = os.path.join(ROOT, "docs", "interfaces", "modules.md")
DATASETS_DOC = os.path.join(ROOT, "docs", "interfaces", "datasets.md")
INTERFACES_README = os.path.join(ROOT, "docs", "interfaces", "README.md")
TEMPLATE = os.path.join(SKILLS, "_module_template", "SKILL.md")
# ① to ⑤ rows of the orchestrator's 主流程 table, in module order.
ROW_MARKS = {"rsi-data": "① 数据", "rsi-train": "② 训练", "rsi-tune": "③ 调优",
             "rsi-deliver": "④ 交付", "rsi-judge": "⑤ 评测与决策"}


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


def parse_error(command):
    """None when `command` parses with the real rsi argument parser, else the parser's message."""
    try:
        tokens = shlex.split(command)[1:]
    except ValueError as e:
        return str(e)
    tokens = [CHOICE_PLACEHOLDERS.get(t, t) for t in tokens]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            cli.build_parser().parse_args(tokens)
    except SystemExit:
        return err.getvalue().strip() or "argument error"
    return None


def orchestrator_files():
    """SKILL.md and every Markdown template of the orchestrator skill."""
    paths = [ORCH]
    tdir = os.path.join(ORCH_DIR, "templates")
    if os.path.isdir(tdir):
        paths += [os.path.join(tdir, n) for n in sorted(os.listdir(tdir)) if n.endswith(".md")]
    return paths


class TestOrchestratorSkill(unittest.TestCase):
    def test_orchestrator_has_required_sections(self):
        sections = skill_sections(ORCH)
        for title in ORCH_SECTIONS:
            self.assertIn(title, sections, "rsi-orchestrator lacks '## %s'" % title)

    def test_orchestrator_uses_dashboard(self):
        text = read(ORCH)
        self.assertIn("<ws>/dashboard.html", text)
        self.assertIn("rsi dashboard <ws> --no-open", text)
        self.assertIn("rsi dashboard <ws>", skill_section(ORCH, "入口"))

    def test_orchestrator_records_questions_in_notes(self):
        text = read(ORCH)
        self.assertIn("## 待用户回答", text)
        para = [l for l in text.splitlines() if l.startswith("**待用户回答**")]
        self.assertEqual(len(para), 1)
        self.assertEqual(para[0].count("rsi dashboard <ws> --no-open"), 2)

    def test_orchestrator_commands_exist(self):
        used = set()
        for path in orchestrator_files():
            for name in COMMAND_RE.findall(read(path)):
                self.assertIn(name, CLI_COMMANDS, "%s: `rsi %s` is not an rsi subcommand"
                              % (os.path.relpath(path, ROOT), name))
                used.add(name)
        # Every mechanical step has a command, and the top level calls each of them.
        self.assertEqual(sorted(set(CLI_COMMANDS) - used), [],
                         "rsi subcommands the orchestrator skill never calls")

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

    def test_orchestrator_mentions_all_modules(self):
        text = read(ORCH)
        for name in MODULE_SKILLS:
            self.assertIn(name, text)

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
        self.assertIn("templates/brief.md", refs)
        self.assertIn("templates/TASK.md", refs)
        for ref in sorted(refs):
            self.assertTrue(os.path.isfile(os.path.join(ORCH_DIR, ref)), "%s does not exist" % ref)


def module_path(name):
    return os.path.join(SKILLS, name, "SKILL.md")


def lower_tokens(text):
    """Backticked tokens made only of lowercase letters (ruling R6)."""
    return [t for t in CODE_SPAN_RE.findall(text) if re.fullmatch(r"[a-z]+", t)]


def orchestrator_row(mark):
    """(input cell, output cell) of the 主流程 row whose first cell starts with `mark`."""
    for line in skill_section(ORCH, "主流程").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0].startswith(mark):
            return cells[2], cells[3]
    raise AssertionError("no 主流程 row for %s" % mark)


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

    def test_module_io_matches_orchestrator_table(self):
        for name, mark in ROW_MARKS.items():
            cell_in, cell_out = orchestrator_row(mark)
            for title, cell in (("输入", cell_in), ("输出", cell_out)):
                want = {t for t in lower_tokens(cell) if t in KINDS}
                got = set(lower_tokens(skill_section(module_path(name), title)))
                self.assertEqual(got, want, "%s: ## %s differs from the 主流程 row" % (name, title))

    def test_deliver_never_gets_labels(self):
        self.assertNotIn("labels.jsonl", skill_section(module_path("rsi-deliver"), "输入"))
        self.assertIn("inputs.jsonl", skill_section(module_path("rsi-deliver"), "输入"))
        self.assertIn("labels.jsonl", skill_section(module_path("rsi-deliver"), "门"))

    def test_train_command_contract(self):
        # R26: fixed path, three producers, marker outside the candidate directory.
        path = "<repo root>/skills/rsi-train/bin/train"
        for p in [module_path(n) for n in ("rsi-train", "rsi-tune", "rsi-data")] + [MODULES_DOC]:
            self.assertIn(path, read(p), p)
        train = read(module_path("rsi-train"))
        for token in ("--marker", "--producer <train、tune 或 data>", "unregistered"):
            self.assertIn(token, train)
        self.assertNotIn("输出目录下的 `DONE`", train)
        self.assertIn("unregistered", read(module_path("rsi-data")))
        self.assertNotIn("unregistered", read(module_path("rsi-tune")))

    def test_train_command_location_consistent(self):
        # S5: every place that names the train command also names the override and the default.
        default = "<repo root>/skills/rsi-train/bin/train"
        docs = os.path.join(ROOT, "docs", "interfaces")
        paths = [module_path(n) for n in ("rsi-train", "rsi-tune", "rsi-data")] + [
            MODULES_DOC, os.path.join(docs, "models.md"), os.path.join(docs, "task.md"),
            os.path.join(ROOT, "README.md"), os.path.join(ORCH_DIR, "templates", "TASK.md"), ORCH]
        for p in paths:
            text = read(p)
            self.assertIn("resources.train_command", text, p)
            self.assertIn(default, text, p)
        self.assertIn("PYTHONPATH=<repo root>", skill_section(module_path("rsi-train"), "输出"))
        self.assertRegex(read(BRIEF), r"训练命令：")

    def test_refused_outputs_guidance(self):
        # S2: a refused output may be fixed in place and registered again; hiding is not deleting.
        text = read(ORCH)
        self.assertIn("原目录", text)
        self.assertIn("再登记", text)
        self.assertNotIn(".refused", text)
        self.assertIn("旧目录原样留着（不要改名、不要删", text)

    def test_judge_does_not_read_valset(self):
        # R25: valset is for data, train and tune only (also in the interface overview table, I1).
        self.assertNotIn("valset", lower_tokens(skill_section(module_path("rsi-judge"), "输入")))
        for path in (MODULES_DOC, DATASETS_DOC, INTERFACES_README):
            self.assertTrue(any(line.startswith("| `valset` |") for line in read(path).splitlines()),
                            "%s has no valset row" % path)
        for line in read(MODULES_DOC).splitlines() + read(DATASETS_DOC).splitlines() + \
                read(INTERFACES_README).splitlines():
            if line.startswith("| `valset` |") and "不读" in line:
                self.assertEqual(line.split("|")[-2].strip(), "不读", line)  # judge column
            if line.startswith("| `valset` |") and "judge" in line:
                self.fail("valset visible to judge: " + line)

    def test_modules_use_only_validate_and_score(self):
        # R21/R36: modules run `rsi validate` and `rsi score`, nothing else; every form parses.
        for name in MODULE_SKILLS + ("_module_template",):
            calls = [c for c in command_lines(read(module_path(name))) if len(c.split()) > 2]
            self.assertTrue(calls, name)
            for c in calls:
                self.assertIn(c.split()[1], ("validate", "score"), "%s: %s" % (name, c))
                self.assertIsNone(parse_error(c), "%s: %s" % (name, c))
            self.assertNotIn("<ws>", read(module_path(name)))


    def test_reported_numbers_come_from_score(self):
        # R36: train and judge give the `rsi score` forms for what they report; tune copies them.
        def score_calls(name):
            return [c for c in command_lines(read(module_path(name)))
                    if len(c.split()) > 2 and c.split()[1] == "score"]
        train = score_calls("rsi-train")
        self.assertTrue(any("--train" in c for c in train), train)
        judge = score_calls("rsi-judge")
        self.assertTrue(any("--compare" in c and "--clusters" in c and "--train" in c for c in judge), judge)
        self.assertTrue(any("testset" in c for c in judge), judge)  # milestone metrics_test
        self.assertIn("rsi score", skill_section(module_path("rsi-tune"), "门"))
        # The judge gets the candidate's trainset for the baselines, as the orchestrator table says.
        self.assertIn("trainset", lower_tokens(skill_section(module_path("rsi-judge"), "输入")))

    def test_runnable_candidates_and_reproduction_in_skills(self):
        # Task 14: deliver reproduces the candidate on the valset inputs (never labels); train writes the
        # prediction entry and code version; tune runs one code version; scoring is not owner-defined.
        deliver_in = skill_section(module_path("rsi-deliver"), "输入")
        self.assertIn("valset", lower_tokens(deliver_in))
        self.assertNotIn("rows.jsonl", deliver_in)
        deliver_gate = skill_section(module_path("rsi-deliver"), "门")
        for text in ("predictions_val", "1e-6", "package.location", "external:<名字>", "cost_unit"):
            self.assertIn(text, deliver_gate)
        train = read(module_path("rsi-train"))
        for text in ("model.predict", "--inputs <inputs.jsonl> --out <predictions.jsonl>", "code_version"):
            self.assertIn(text, train)
        self.assertIn("code_version.sha256", skill_section(module_path("rsi-tune"), "门"))
        self.assertIn("inputs.jsonl", skill_section(module_path("rsi-data"), "输出"))
        judge_owner = skill_section(module_path("rsi-judge"), "步骤和策略子 skill")
        self.assertNotIn("由模块负责人填写：打分", judge_owner)
        self.assertIn("rsi score", judge_owner)
        orch = read(ORCH)
        self.assertIn("本轮之内训练命令和默认配置都不能改", orch)
        self.assertIn("登记后路径", skill_section(ORCH, "闭环"))


class TestSkillFiles(unittest.TestCase):
    def test_frontmatter(self):
        names = sorted(n for n in os.listdir(SKILLS)
                       if os.path.isdir(os.path.join(SKILLS, n)) and not n.startswith((".", "_")))
        self.assertIn("rsi-orchestrator", names)
        for name in names:
            path = os.path.join(SKILLS, name, "SKILL.md")
            self.assertTrue(os.path.isfile(path), "%s has no SKILL.md" % name)
            fm = frontmatter(path)
            self.assertEqual(fm.get("name"), name, "%s: frontmatter name" % name)
            self.assertTrue(fm.get("description"), "%s: frontmatter description is empty" % name)


class TestHelpers(unittest.TestCase):
    def test_command_regex_ignores_skill_names(self):
        self.assertEqual(COMMAND_RE.findall("load rsi-orchestrator, then run rsi status <ws>"), ["status"])
        self.assertEqual(COMMAND_RE.findall("`rsi check-stop <ws>` and rsi-judge"), ["check-stop"])

    def test_parse_error(self):
        self.assertIsNone(parse_error("rsi register <ws> <kind> <dir>"))
        self.assertIsNone(parse_error('rsi approve <ws> plan --ref <plan_id> --by <name> --quote "a b"'))
        self.assertIsNotNone(parse_error("rsi register <ws> model <dir>"))
        self.assertIsNotNone(parse_error("rsi approve <ws> plan"))

    def test_command_lines(self):
        text = ("run `rsi status <ws>` then `PYTHONPATH=<repo root> python3 -m rsi validate task <d>`"
                " and `rsi-judge`\n```\nrsi next <ws>\n```\n")
        self.assertEqual(command_lines(text), ["rsi status <ws>", "rsi validate task <d>", "rsi next <ws>"])

    def test_sections_skip_fences(self):
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
