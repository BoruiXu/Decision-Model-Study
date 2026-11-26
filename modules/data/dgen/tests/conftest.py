"""A data-module tree for dgen: a binary task whose state is {"x", "junk", "tx": [{"amount"}] (1 or 2 items)}, labelled
yes exactly when x > 0.5 (the yes rows have x in (0.5, 0.8] or in [3, 6]); a 300-row pool, a 150-row trainset (the
first half of the pool, as dset draws it), a 60-row valset (rows and inputs.jsonl); the best version's candidate,
whose model.predict says yes only for 0.5 < x <= 0.8, so it errs on every yes row with x > 0.8; and a data brief
(trainset purpose) over them, as rsi.briefs.build_brief writes one. The trainset also holds DGEN_ID, a row generated in
an earlier round (meta.source dgen:smote) by OLD_RECIPE: never a seed, an anchor or a reference, still a known row."""
import atexit
import json
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent
REPO = DATA.parent.parent
for _p in (str(DATA), str(REPO / "modules" / "common"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dgen.recipes import normalize, sha12  # noqa: E402  (the recipe format is stdlib only)
from rsi.artifacts import validate_artifact  # noqa: E402  (rsi is stdlib only)
from rsi.rows import input_row_of, question_of  # noqa: E402

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
}
OLD_RECIPE = {"name": "old", "method": "smote", "n": 5, "seed": 9}
DGEN_ID = "syn%s_smote_000000" % sha12(normalize(OLD_RECIPE, TASK))[:6]
PREDICT = '''#!%s
"""model.predict of the fake candidate: P(yes) 0.9 for 0.5 < x <= 0.8, else 0.1 (x <= 0.5) or 0.2 (x > 0.8). It
prints its CUDA_VISIBLE_DEVICES, then sleeps $FAKE_PREDICT_SLEEP seconds (default 0)."""
import json, os, sys, time
print("CUDA_VISIBLE_DEVICES=%%s" %% os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
time.sleep(float(os.environ.get("FAKE_PREDICT_SLEEP", "0")))
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
with open(args["--inputs"]) as f, open(args["--out"], "w") as out:
    for line in f:
        r = json.loads(line)
        x = r["state"]["x"]
        p = 0.9 if 0.5 < x <= 0.8 else (0.1 if x <= 0.5 else 0.2)
        out.write(json.dumps({"id": r["id"], "probs": {"yes": p, "no": 1 - p},
                              "decision": "yes" if p >= 0.5 else "no", "uncertainty": None}) + "\\n")
''' % sys.executable


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def make_rows(kind, n, rng):
    rows = []
    for i in range(n):
        u = rng.random()
        x = rng.uniform(0, 0.5) if u < 0.5 else (rng.uniform(0.5001, 0.8) if u < 0.8 else rng.uniform(3, 6))
        y = "yes" if x > 0.5 else "no"
        state = {"x": round(x, 6), "junk": round(rng.gauss(0, 1), 6),
                 "tx": [{"amount": round(rng.uniform(1, 100), 2)} for _ in range(rng.randint(1, 2))]}
        rows.append({"id": "%s-%d" % (kind, i), "state": state, "question": question_of(TASK),
                     "target": {"yes": float(y == "yes"), "no": float(y == "no")},
                     "meta": {"state_id": "%s-s%d" % (kind, i), "cluster": "%s-c%d" % (kind, i % 40),
                              "source": "test", "label_source": "outcome", "n_obs": 1},
                     "split": "val" if kind == "valset" else "train"})
    return rows


def make_set(root, kind, rows):
    d = root / kind
    d.mkdir()
    write_jsonl(d / "rows.jsonl", rows)
    files = {"rows": {"path": "rows.jsonl", "rows": len(rows)}}
    if kind == "valset":
        write_jsonl(d / "inputs.jsonl", [input_row_of(r) for r in rows])
        files["inputs"] = {"path": "inputs.jsonl", "rows": len(rows)}
    m = {"kind": kind, "id": "%s-0001" % kind, "files": files}
    if kind == "trainset":
        m["label_method"] = "outcome"
    write_json(d / "manifest.json", m)
    return d


def make_tree(root):
    """root/{task, trainset, valset, pool, candidate, out}, root/out/brief.json; returns the brief's path."""
    rng = random.Random(7)
    (root / "task").mkdir()
    write_json(root / "task" / "task.json", TASK)
    pool = make_rows("pool", 300, rng)
    old = json.loads(json.dumps([r for r in pool[:150] if 0.5 < r["state"]["x"] <= 0.8][0]))
    old["state"]["x"] = round(old["state"]["x"] - 0.01, 6)
    old["meta"].update(source="dgen:smote", extra={"seed_id": old["id"], "recipe": DGEN_ID[3:9]})
    old["id"] = DGEN_ID
    sets = {"pool": make_set(root, "pool", pool), "trainset": make_set(root, "trainset", pool[:150] + [old]),
            "valset": make_set(root, "valset", make_rows("valset", 60, rng))}
    cand = root / "candidate"
    cand.mkdir()
    (cand / "predict").write_text(PREDICT)
    os.chmod(cand / "predict", os.stat(cand / "predict").st_mode | stat.S_IXUSR)
    write_json(cand / "manifest.json", {"kind": "candidate", "id": "candidate-0001", "model": {"predict": "predict"}})
    out = root / "out"
    out.mkdir()
    inputs = {"task": str(root / "task" / "task.json"), "trainset": str(sets["trainset"]),
              "valset": str(sets["valset"]), "pool": str(sets["pool"]), "best_candidate": str(cand)}
    brief = {"module": "data", "purpose": "trainset", "round": 2, "task_json": inputs["task"], "inputs": inputs,
             "ids": {"trainset": "trainset-0001", "valset": "valset-0001"}, "outdir": str(out),
             "spend_file": str(out / "spend.jsonl"), "budget_left": {"compute_hours": None, "api_usd": None},
             "features": {"adapter": None, "constraints": []}, "resources": {"gpu_ids": [0, 1]}}
    write_json(out / "brief.json", brief)
    return out / "brief.json"


def fake_smi(d, table):
    """A directory holding an executable nvidia-smi that prints `table` (index, uuid, MiB used lines), put first on
    the host's PATH: the PATH to run with."""
    d.mkdir(exist_ok=True)
    (d / "nvidia-smi").write_text("#!/bin/sh\nprintf '%s'\n" % "".join(line + "\\n" for line in table))
    os.chmod(d / "nvidia-smi", 0o755)
    return str(d) + os.pathsep + os.environ.get("PATH", "")


_IDLE = Path(tempfile.mkdtemp(prefix="dgen-tests-smi-"))
atexit.register(shutil.rmtree, str(_IDLE), True)
IDLE_PATH = fake_smi(_IDLE, ["0, GPU-test-idle, 0"])   # one idle card, never the host's nvidia-smi


def dgen_env(**extra):
    """The environment of a dgen run, hermetic on a GPU host: no CUDA_VISIBLE_DEVICES and a fake nvidia-smi with one
    idle card (GPU-test-idle) first on PATH, so score-pool never asks the host's cards; `extra` overrides (PATH too)."""
    env = dict((k, v) for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES")
    env["PATH"] = IDLE_PATH
    env.update(extra, PYTHONPATH=os.pathsep.join([str(DATA), str(REPO / "modules" / "common"), str(REPO)]))
    return env


def run_dgen(*args, ok=True, env=None):
    """`python -m dgen <args>`: its JSON summary (stdout); with ok, exit code 0 is asserted."""
    p = subprocess.run([sys.executable, "-m", "dgen"] + [str(a) for a in args], env=env or dgen_env(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if ok:
        assert p.returncode == 0, p.stderr
    assert len(p.stdout.strip().splitlines()) == 1 or p.returncode == 0, p.stdout
    return json.loads(p.stdout)


def run_dset(*args):
    env = dict(os.environ, PYTHONPATH=str(DATA))
    return subprocess.run([sys.executable, "-m", "dset"] + [str(a) for a in args], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)


def rsi_validate(kind, d, task):
    return validate_artifact(kind, str(d), task)


def generate(diagnosed, recipe, out):
    path = Path(out) / ("%s.json" % recipe["name"])
    write_json(path, recipe)
    s = run_dgen("generate", "--brief", diagnosed["brief"], "--recipe", path, "--diagnosis", diagnosed["dir"],
                 "--out", out)
    return Path(s["arm"])


@pytest.fixture
def brief(tmp_path):
    return make_tree(tmp_path)


@pytest.fixture(scope="module")
def diagnosed(tmp_path_factory):
    """A tree whose pool is scored and diagnosed: {"brief", "dir" (the diagnosis), "root", "rows" (id -> row of the
    trainset, the valset and the pool)}."""
    root = tmp_path_factory.mktemp("diagnosed")
    b = make_tree(root)
    d = root / "out" / "dgen"
    assert run_dgen("score-pool", "--brief", b, "--out", d, "--marker", root / "out" / "score.done")["card"] == \
        "GPU-test-idle"                                         # the fake idle card: never the host's
    run_dgen("diagnose", "--brief", b, "--scores", d / "pool_predictions.jsonl", "--out", d)
    rows = dict((r["id"], r) for kind in ("trainset", "valset", "pool") for r in read_jsonl(root / kind / "rows.jsonl"))
    return {"brief": b, "dir": d, "root": root, "rows": rows}
