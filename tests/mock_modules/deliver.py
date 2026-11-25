"""Mock rsi-deliver module for the engine tests (see _mock.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _mock  # noqa: E402

_mock.main("deliver")
