"""examples/elliptic_onejev: the task template starts the run from the data module (the data agent preprocesses the
raw rows and fills the split rule; only the debug sizes are pinned); its features adapter on a txhist state."""
import copy
import json
import os
import re
import sys
import unittest

from rsi import datasplit as DS
from rsi import rows as R
from rsi.rules import adapter_path, compare_settings, policy_settings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "modules", "common"))
from features.adapter import load_adapter  # noqa: E402

EXAMPLE = os.path.join(REPO, "examples", "elliptic_onejev")
TEMPLATE_JSON = os.path.join(EXAMPLE, "task.template.json")
TEMPLATE_MD = os.path.join(EXAMPLE, "TASK.template.md")
# The placeholders the operator fills, the same in both files (README: one sed for both).
PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*>")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def small_txhist_state():
    """A txhist state in the shape 01_build_data.sh writes, with 3 transactions and the summary they give."""
    def tx(steps_ago, role, total, fee, n_in, n_out, smallest, largest):
        return {"steps_ago": steps_ago, "role": role, "tx_total_btc": total, "tx_fee_btc": fee,
                "tx_num_input_addresses": n_in, "tx_num_output_addresses": n_out,
                "tx_smallest_output_btc": smallest, "tx_largest_output_btc": largest}
    return {"entity": "Bitcoin address transaction history",
            "summary": {"txs_sent": 2, "txs_received": 2, "first_seen_steps_ago": 4, "active_steps": 2,
                        "txs_listed": 3},
            "transactions": [tx(0, "received", 0.13562, 0.00161, 20, 4, 0.0125271, 0.0688),
                             tx(4, "sent", 10.01, 0.00724789, 327, 2, 0.0100001, 10),
                             tx(4, "both", 0.999815, 0.000185, 1, 4, 0.0243433, 0.363883)]}


class TestEllipticExample(unittest.TestCase):
    def test_template_is_a_valid_task_once_filled(self):
        text = read(TEMPLATE_JSON)
        for k, v in {"<ROWS_DIR>": "/data/addr_txhist_noul", "<REPO_ROOT>": "/repo", "<OJH_PYTHON>": "/o/python",
                     "<OJH_BIN>": "/o/ojh", "<OJH_MACHINE>": "/o/machine.yaml", "<OJH_ENV_FILE>": "/o/env.sh",
                     "<CLAUDE_BIN>": "/c/claude", '"<GPU_IDS>"': "[0, 1]", '"<JOEY1_GPU_HOURS>"': "0.3",
                     '"<ELEPHANT1_GPU_HOURS>"': "0.5", '"<API_USD_BUDGET>"': "20",
                     '"<AGENT_USD_BUDGET>"': "50"}.items():
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
        # Every new section is on: ③ Compare's six baselines (the team recipe on elephant-1), the decision policy,
        # the features adapter with its constraints, the generated-row cap, both dollar budgets, three rounds.
        self.assertEqual(compare_settings(task)["baselines"], ["lr", "gbdt", "mlp", "zero_shot", "few_shot",
                                                               "team_recipe"])
        self.assertEqual(compare_settings(task)["team_recipe"]["base"], "elephant-1")
        self.assertEqual(policy_settings(task)["objective"], "f1")
        self.assertTrue(os.path.isfile(adapter_path(task)))
        self.assertEqual(task["quality_checks"]["max_synthetic_share"], 0.3)
        self.assertEqual((task["budget"]["api_cost"], task["budget"]["agent_cost"], task["stop"]["max_rounds"]),
                         (20, 50, 3))
        a = load_adapter(adapter_path(task), task["features"]["constraints"])
        st = small_txhist_state()
        self.assertEqual(a.check(st), [])
        bad = copy.deepcopy(st); bad["transactions"][0]["steps_ago"] = 0.6; bad["transactions"][1]["tx_fee_btc"] = -1
        self.assertEqual(sorted(m.split()[0] for m in a.check(bad) if m.startswith(("nonneg", "integer"))),
                         ["integer", "nonneg"])                       # the constraints' rules, then the adapter's
        self.assertEqual(a.check(a.repair(bad)), [])

    def test_elliptic_adapter_on_a_txhist_state(self):
        a = load_adapter(os.path.join(REPO, "examples/elliptic_onejev/features_adapter.py"))
        st = small_txhist_state()
        self.assertIn("summary.txs_sent", a.features(st))
        self.assertEqual(a.check(st), [])
        bad = copy.deepcopy(st); bad["summary"]["txs_sent"] += 5
        self.assertTrue(a.check(bad)); self.assertEqual(a.check(a.repair(bad)), [])


if __name__ == "__main__":
    unittest.main()
