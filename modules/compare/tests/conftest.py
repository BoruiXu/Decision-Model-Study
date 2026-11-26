"""A workspace-like tree for the compare module: a binary task whose state has one informative feature (`score`), a
200-row trainset, a 100-row valset and a 100-row evalset (only its inputs.jsonl is given), the registered candidate-0001
(its predictions_val follow `score`), and compare briefs over them, as rsi.briefs.build_brief writes them."""
import json
import math
import os
import random
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
REPO = MODULE.parent.parent
for _p in (str(MODULE), str(REPO / "modules" / "common"), str(REPO / "modules" / "train"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rsi.artifacts import validate_artifact  # noqa: E402  (rsi is stdlib only)
from rsi.rows import question_of  # noqa: E402
from rsi.rules import compare_settings  # noqa: E402

RUN = MODULE / "run"

TASK = {
    "kind": "task", "task_id": "risk", "version": 1,
    "decision": {"type": "binary", "instructions": "Is this account risky?", "labels": ["yes", "no"],
                 "label_meanings": {"yes": "the account is risky", "no": "the account is not risky"},
                 "positive": "yes"},
    "input": {"format": "object", "description": "An account and its transactions."},
    "data": {"sources": [], "label_sources": ["outcome"]},
    "metrics": {"primary": "auc", "higher_is_better": True, "others": ["log_loss"], "baselines": ["majority"]},
    "promotion_rule": {"min_delta": 0.01, "ci_lower_min": None, "guards": []},
    "stop": {"target": None, "max_rounds": 3, "patience": 2},
    "budget": {"compute_hours": None, "annotation_items": None, "api_cost": None},
    "resources": {},
    "eval_use_limit": 3,
    "approvals": ["task_change", "over_budget", "use_testset"],
    "compare": {"baselines": ["lr", "gbdt", "mlp"], "python": sys.executable},
}
SIZES = (("trainset", 200), ("valset", 100), ("evalset", 100))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def manifest(d):
    return json.loads((Path(d) / "manifest.json").read_text())


def handback(outdir):
    return json.loads((Path(outdir) / "handback.json").read_text())


def rsi_validate(kind, d, task):
    return validate_artifact(kind, str(d), task)


def rsi_score_compare(brief, predictions, compare):
    """The `delta` that `rsi score --compare` prints for `predictions` minus `compare` on the brief's valset rows, the
    clusters from its inputs.jsonl."""
    b = json.loads(Path(brief).read_text())
    valset = b["inputs"]["valset"]
    out = subprocess.run([sys.executable, "-m", "rsi", "score", "--task", b["task_json"], "--labels",
                          os.path.join(valset, "rows.jsonl"), "--predictions", str(predictions), "--clusters",
                          os.path.join(valset, "inputs.jsonl"), "--compare", str(compare)],
                         env=dict(os.environ, PYTHONPATH=str(REPO)), stdout=subprocess.PIPE, check=True,
                         universal_newlines=True).stdout
    return json.loads(out)["delta"]


def write_registered(outdir, n, dirs):
    """What the engine leaves for a followup: registered-<n>.json ([{"dir", "kind", "id"}] for `dirs`, subdirectory ->
    id) beside feedback-<n>.md."""
    write_json(Path(outdir) / ("registered-%d.json" % n), [{"dir": d, "kind": "baseline", "id": i}
                                                           for d, i in dirs.items()])
    (Path(outdir) / ("feedback-%d.md" % n)).write_text("# Continue: the second part of the hand-back\n")


def make_rows(task, kind, n, rng, prefix=None):
    """Rows <prefix or kind>-<i> whose label is yes when score + noise > 0.7 (about a quarter); `junk` and the
    transactions carry no signal."""
    rows = []
    for i in range(n):
        score = rng.gauss(0, 1)
        y = "yes" if score + rng.gauss(0, 0.3) > 0.7 else "no"
        state = {"score": round(score, 4), "junk": round(rng.gauss(0, 1), 4),
                 "transactions": [{"amount": round(abs(rng.gauss(0, 1)) * 100, 2)} for _ in range(rng.randint(1, 4))]}
        rows.append({"id": "%s-%d" % (prefix or kind, i), "state": state, "question": question_of(task),
                     "target": {"yes": float(y == "yes"), "no": float(y == "no")},
                     "meta": {"state_id": "s%d" % (i % 13), "cluster": "c%d" % i, "source": "test",
                              "label_source": "outcome", "n_obs": 1},
                     "split": kind})
    return rows


def set_dir(root, kind, sid):
    """The directory of set `sid`: root/<kind> for <kind>-0001, else root/<sid>."""
    return root / (kind if sid == "%s-0001" % kind else sid)


def make_set(root, task, kind, n, rng, sid=None):
    """A registered-looking set `sid` (default <kind>-0001): trainset rows; valset rows and inputs; evalset and
    testset inputs and labels."""
    sid = sid or "%s-0001" % kind
    d = set_dir(root, kind, sid)
    d.mkdir()
    rows = make_rows(task, kind, n, rng, None if sid == "%s-0001" % kind else sid)
    inputs = [dict((k, v) for k, v in r.items() if k != "target") for r in rows]
    files = {}
    if kind in ("trainset", "valset"):
        write_jsonl(d / "rows.jsonl", rows)
        files["rows"] = {"path": "rows.jsonl", "rows": n}
    if kind in ("valset", "evalset", "testset"):
        write_jsonl(d / "inputs.jsonl", inputs)
        files["inputs"] = {"path": "inputs.jsonl", "rows": n}
    if kind in ("evalset", "testset"):
        write_jsonl(d / "labels.jsonl", [{"id": r["id"], "target": r["target"]} for r in rows])
        files["labels"] = {"path": "labels.jsonl", "rows": n}
    write_json(d / "manifest.json", {"kind": kind, "id": sid, "files": files})
    return d


def make_candidate(root):
    """root/candidate: the registered candidate-0001 on trainset-0001 and valset-0001, whose predictions_val give each
    valset row P(yes) = sigmoid(3 * score - 2) (a decision of the most probable label)."""
    d = root / "candidate"
    d.mkdir()
    rows = [json.loads(x) for x in (root / "valset" / "rows.jsonl").read_text().splitlines() if x.strip()]
    preds = []
    for r in rows:
        p = 1.0 / (1.0 + math.exp(2.0 - 3.0 * r["state"]["score"]))
        decision = "yes" if p >= 0.5 else "no"
        preds.append({"id": r["id"], "probs": {"yes": p, "no": 1.0 - p}, "decision": decision,
                      "uncertainty": 1.0 - max(p, 1.0 - p)})
    write_jsonl(d / "predictions_val.jsonl", preds)
    write_json(d / "manifest.json", {"kind": "candidate", "id": "candidate-0001", "trainset_id": "trainset-0001",
                                     "valset_id": "valset-0001",
                                     "files": {"predictions_val": {"path": "predictions_val.jsonl"}}})
    return d


def make_brief(root, task, baselines=None, ids=None, name="brief.json", **fields):
    """root/<name> over root/{task,candidate,trainset,valset,evalset} (other set ids: `ids`, kind -> id) and the
    registered baselines `baselines` (id -> directory); `fields` replace brief fields."""
    (root / "task").mkdir(exist_ok=True)
    write_json(root / "task" / "task.json", task)
    ids = dict(dict((kind, "%s-0001" % kind) for kind, _ in SIZES), **(ids or {}))
    inputs = {"task": str(root / "task" / "task.json"), "trainset": str(set_dir(root, "trainset", ids["trainset"])),
              "valset": str(set_dir(root, "valset", ids["valset"])),
              "evalset_inputs": str(set_dir(root, "evalset", ids["evalset"]) / "inputs.jsonl")}
    if (root / "candidate").is_dir():
        inputs["candidate"] = str(root / "candidate")
    for bid, d in sorted((baselines or {}).items()):
        inputs[bid] = str(d)
    out = root / "out"
    brief = {"module": "compare", "skill": "rsi-compare", "purpose": "compare", "round": 1, "dispatch_id": "d-0003",
             "task_id": task["task_id"], "task_version": task["version"], "task_json": inputs["task"], "ids": ids,
             "inputs": inputs, "input_notes": dict((k, k) for k in inputs), "outdir": str(out),
             "progress_file": str(out / "progress.json"), "spend_file": str(out / "spend.jsonl"),
             "rsi": "python3 -m rsi", "attempts": "1 / 3", "goal": "compare", "handback": "",
             "budget_left": {"compute_hours": None, "api_usd": None, "agent_usd": None, "annotation_items": None},
             "compare": compare_settings(task), "policy": task.get("policy"),
             "features": {"adapter": None, "constraints": []}, "resources": {}, "train_command": "train"}
    brief.update(fields)
    path = root / name
    write_json(path, brief)
    return str(path)


def workspace(root, task, baselines=None, **fields):
    rng = random.Random(7)
    for kind, n in SIZES:
        make_set(root, task, kind, n, rng)
    make_candidate(root)
    return make_brief(root, task, baselines, **fields)


@pytest.fixture
def task():
    return json.loads(json.dumps(TASK))


@pytest.fixture
def brief(tmp_path_factory, task):
    return workspace(tmp_path_factory.mktemp("ws"), task)


@pytest.fixture
def brief_with_registered_lr(tmp_path_factory, task):
    """A brief whose task asks for lr only, with an lr baseline registered as baseline-0001 (trained on the same
    sets by the module itself, then copied into the workspace as registration does)."""
    from compare.cli import main

    task["compare"]["baselines"] = ["lr"]
    root = tmp_path_factory.mktemp("ws")
    first = workspace(root, task)
    scratch = tmp_path_factory.mktemp("first")
    assert main(["run", "--brief", first, "--outdir", str(scratch), "--mode", "first"]) == 0
    reg = root / "baseline-0001"
    shutil.copytree(str(scratch / "bl-lr"), str(reg))
    m = manifest(reg)
    m["id"] = "baseline-0001"
    write_json(reg / "manifest.json", m)
    return make_brief(root, task, {"baseline-0001": reg})
