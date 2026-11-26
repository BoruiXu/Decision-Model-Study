"""llm: one-shot `claude -p` calls with structured output, for the LLM calls that generate or review data (standard
library only, Python 3.9+).

- claude: ClaudeCall, one fresh `claude -p` process per call (the task's runner.claude command and arguments, the
  default model and effort of rsi.runners.claude, tools disabled, a JSON schema for the answer); every call is logged
  and billed as an `api_usd` line of the dispatch's spend file, and the caps are checked before each call.

Import it with `<repo root>/modules/common` on sys.path; it uses the framework's rsi (the runner's command and its
output parser, the schema validator, the spend-line rules), so the repository root is put on the path too.
"""
import os
import sys

LLM_VERSION = "1.0.0"

# The repository root (three levels above this package's directory): claude.py uses rsi.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
