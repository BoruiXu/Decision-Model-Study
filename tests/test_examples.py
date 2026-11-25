"""examples/elliptic_onejev: the task template starts the run from the data module (the data agent preprocesses the
raw rows and fills the split rule; only the debug sizes are pinned)."""
import json
import os
import re
import unittest

from rsi import datasplit as DS
from rsi import rows as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "examples", "elliptic_onejev")
TEMPLATE_JSON = os.path.join(EXAMPLE, "task.template.json")
TEMPLATE_MD = os.path.join(EXAMPLE, "TASK.template.md")
# The placeholders the operator fills, the same in both files (README: one sed for both).
PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestEllipticExample(unittest.TestCase):
    def test_template_is_a_valid_task_once_filled(self):
        text = read(TEMPLATE_JSON)
        for k, v in {"<ROWS_DIR>": "/data/addr_txhist_noul", "<REPO_ROOT>": "/repo", "<OJH_PYTHON>": "/o/python",
                     "<OJH_BIN>": "/o/ojh", "<OJH_MACHINE>": "/o/machine.yaml", "<OJH_ENV_FILE>": "/o/env.sh",
                     "<CLAUDE_BIN>": "/c/claude", '"<GPU_IDS>"': "[0, 1]", '"<JOEY1_GPU_HOURS>"': "0.3"}.items():
            text = text.replace(k, v)
        task = json.loads(text)
        self.assertEqual(R.check_task(task), [])
        self.assertEqual(DS.source_paths(task), ["/data/addr_txhist_noul"])
        self.assertFalse(DS.is_complete(DS.task_split(task)))                 # the agent fills the rule
        self.assertEqual(DS.task_split(task), {"sizes": {"trainset": 1000, "valset": 500, "evalset": 500,
                                                         "testset": 1000}})   # only the debug sizes are pinned
        self.assertEqual(task["data"]["rows"]["expect"], {"rows": 291419, "label_counts": {"yes": 14720, "no": 276699}})
        self.assertEqual(set(PLACEHOLDER.findall(read(TEMPLATE_JSON))), set(PLACEHOLDER.findall(read(TEMPLATE_MD))))
        self.assertFalse(os.path.exists(os.path.join(EXAMPLE, "prepare_sets.py")))


if __name__ == "__main__":
    unittest.main()
