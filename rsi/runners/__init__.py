"""Agent runners: how rsi.engine starts a module agent (see rsi.runners.base for the interface).

claude  ClaudeCodeRunner, a headless Claude Code session (the default of `rsi run`)
mock    MockRunner, a script per module (tests and flow simulations)
manual  ManualRunner, a person fills the output directory
"""
import os

from rsi.runners.base import Runner
from rsi.runners.claude import ClaudeCodeRunner
from rsi.runners.manual import ManualRunner
from rsi.runners.mock import MockRunner
from rsi.workspace import RsiError

RUNNERS = ("claude", "mock", "manual")


def make_runner(name, ws, mock_dir=None):
    """The runner `rsi run --runner <name>` uses for workspace `ws` (claude takes the task's runner.claude)."""
    if name == "claude":
        return ClaudeCodeRunner.from_task(ws.task())
    if name == "manual":
        return ManualRunner()
    if name == "mock":
        if not mock_dir:
            raise RsiError("--runner mock needs --mock-dir DIR (the directory of the mock module scripts <module>.py)")
        if not os.path.isdir(mock_dir):
            raise RsiError("--mock-dir %s is not a directory" % mock_dir)
        return MockRunner(os.path.abspath(mock_dir))
    raise RsiError("unknown runner '%s' (one of %s)" % (name, ", ".join(RUNNERS)))


def attach_runner(name, ws):
    """A runner that can find the agent of a dispatch record made with runner `name` (it launches nothing)."""
    if name == "claude":
        return ClaudeCodeRunner.from_task(ws.task())
    if name == "manual":
        return ManualRunner()
    if name == "mock":
        return MockRunner(None)
    return Runner()


__all__ = ["Runner", "ClaudeCodeRunner", "ManualRunner", "MockRunner", "RUNNERS", "make_runner", "attach_runner"]
