"""The test environment: a framework task with search_space and search, a registered-looking trainset and valset
of OneJev rows, a train brief, and a fake ojh repository (catalog, recipes, the fake `ojh` command). Scoring a
checkpoint goes through tests/fake_scorer.py (TRSI_TEST_SCORER), so no GPU and no real ojh are needed. The fake
`nvidia-smi` prints $FAKE_GPUS (index, uuid, MiB used, MiB free per line) and $FAKE_GPU_APPS (gpu_uuid, pid per line;
with $FAKE_GPU_BUSY_ONCE, a file path, every card also runs a process until one query has seen it);
the fake `tmux` logs every command but has-session to <tmp>/tmux.log, and has-session fails when FAKE_TMUX_GONE is 1."""
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from trsi.cli import main

HERE = Path(__file__).parent
MODULE = HERE.parent
REPO = MODULE.parent.parent
sys.path.insert(0, str(REPO))
from rsi.rows import question_of  # noqa: E402  (rsi is stdlib only)

RECIPE = {"mode": "lora", "gpus": 1, "onejev": {"lr": 1.0e-4, "epochs": 1, "brier_weight": 1},
          "lora": {"r": 32, "alpha": 64, "dropout": 0.05}}
JOEY = {"mode": "full", "gpus": 1, "onejev": {"lr": 1.0e-5, "epochs": 1, "accum_rows": 32}}
EL, MM = "examples/recipes/el_lora_1g.yaml", "examples/recipes/mm_lora_1g.yaml"

TASK = {
    "kind": "task", "task_id": "fraud", "version": 1,
    "decision": {"type": "binary", "instructions": "Is this address illicit?", "labels": ["yes", "no"],
                 "label_meanings": {"yes": "illicit", "no": "licit"}, "positive": "yes"},
    "input": {"format": "messages", "description": "The address's transaction history."},
    "data": {"sources": [], "label_sources": ["outcome"]},
    "metrics": {"primary": "auprc", "higher_is_better": True, "others": ["brier"], "baselines": ["majority"]},
    "promotion_rule": {"min_delta": 0.01, "ci_lower_min": None,
                       "guards": [{"metric": "brier", "higher_is_better": False, "max_regression": 0.01}]},
    "stop": {"target": None, "max_rounds": 2, "patience": 2},
    "budget": {"compute_hours": 100, "annotation_items": None, "api_cost": None},
    "resources": {
        "onejev": {"default_config": {"base": "elephant-1", "recipe": EL, "overrides": {}, "seed": 0,
                                      "max_steps": None},
                   "estimate": {"elephant-1": 8, "mammoth-1": 23}},
        "gpu_ids": [0, 1, 2, 3],
    },
    "eval_use_limit": 3,
    "approvals": ["task_change", "over_budget", "use_testset"],
    "search_space": {
        "base": ["elephant-1", "mammoth-1"],
        "recipe": [EL, MM],
        "overrides.onejev.lr": [5.0e-5, 1.0e-4, 2.0e-4],
        "overrides.onejev.brier_weight": {"values": [0, 1, 3], "dead": {"3": "Elliptic: 3 seeds mean -0.082"}},
        "overrides.lora.r": {"min": 8, "max": 128},
        "seed": [0, 1, 2],
    },
    "search": {"max_configs": 3, "seeds": [0, 1, 2], "probe_margin": 0.035, "accept_margin": 0.035,
               "guards": {"ece": 0.02, "brier": 0.005}},
}


FAKE_NVIDIA_SMI = """import os, sys
cols = {"--query-gpu": ("index", "uuid", "memory.used", "memory.free"), "--query-compute-apps": ("gpu_uuid", "pid")}
kind, fields = next(a for a in sys.argv[1:] if a.startswith("--query-")).split("=", 1)
text = os.environ["FAKE_GPUS" if kind == "--query-gpu" else "FAKE_GPU_APPS"]
once = os.environ.get("FAKE_GPU_BUSY_ONCE")
if kind == "--query-compute-apps" and once and not os.path.exists(once):
    open(once, "w").close()
    text += "".join(x.split(",")[1].strip() + ", 999\\n" for x in os.environ["FAKE_GPUS"].splitlines() if x.strip())
for line in text.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if line.strip():
        print(", ".join(parts[cols[kind].index(f)] for f in fields.split(",")))
"""


def row(i, split, task=TASK):
    """OneJev row a<i>: illicit (yes) when i % 10 == 0; the fake scorer reads the label from the text."""
    y = "yes" if i % 10 == 0 else "no"
    return {"id": f"a{i}", "state": {"messages": [{"role": "user", "content": f"address {i} label={y}"}]},
            "question": question_of(task), "target": {"yes": float(y == "yes"), "no": float(y == "no")},
            "meta": {"state_id": f"s{i % 7}", "cluster": f"c{i}", "source": "test", "label_source": "outcome",
                     "n_obs": 1},
            "split": split}


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_set(d: Path, kind: str, rows: list, with_inputs: bool):
    d.mkdir(parents=True)
    write_jsonl(d / "rows.jsonl", rows)
    files = {"rows": {"path": "rows.jsonl", "rows": len(rows)}}
    if with_inputs:
        write_jsonl(d / "inputs.jsonl", [{k: v for k, v in r.items() if k != "target"} for r in rows])
        files["inputs"] = {"path": "inputs.jsonl", "rows": len(rows)}
    (d / "manifest.json").write_text(json.dumps({"kind": kind, "id": f"{kind}-0001", "files": files}))


def age(d: Path, seconds: float) -> None:
    """Set the modification time of `d` and everything under it `seconds` into the past."""
    t = time.time() - seconds
    for root, _, files in os.walk(d):
        for p in [root] + [os.path.join(root, f) for f in files]:
            os.utime(p, (t, t))


def wait_for(cond, timeout=20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def executable(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Env:
    def __init__(self, root: Path, capsys):
        self.root = root
        self.capsys = capsys
        self.ojh_root = root / "ojh_root"
        self.out = root / "work" / "round-1" / "train"
        self.brief = self.out / "brief.json"
        self.task_json = root / "task" / "task.json"
        self.trainset, self.valset = root / "sets" / "trainset", root / "sets" / "valset"
        self.calls = root / "calls.txt"
        self.run = self.out / "trsi" / "run.yaml"
        self.tmux_log = root / "tmux.log"

    def write_task(self, **changes):
        t = json.loads(json.dumps(TASK))
        t["resources"]["onejev"].update(python=sys.executable, ojh_bin=str(self.ojh_root / ".venv" / "bin" / "ojh"),
                                        machine=str(self.root / "machine.yaml"))
        for k, v in changes.items():
            t[k] = v
        self.task_json.parent.mkdir(parents=True, exist_ok=True)
        self.task_json.write_text(json.dumps(t, indent=2))
        return t

    def write_brief(self, **changes):
        b = {"module": "train", "purpose": "train", "round": 1, "task_id": "fraud", "task_version": 1,
             "task_json": str(self.task_json), "ids": {"trainset": "trainset-0001", "valset": "valset-0001"},
             "inputs": {"task": str(self.task_json), "trainset": str(self.trainset), "valset": str(self.valset)},
             "outdir": str(self.out), "budget_left": {"compute_hours": 100, "api_usd": None, "agent_usd": None,
                                                       "annotation_items": None},
             "dispatch_id": "d-0002"}
        b.update(changes)
        self.out.mkdir(parents=True, exist_ok=True)
        self.brief.write_text(json.dumps(b, indent=2))
        return b

    def init(self):
        code, out, err = self.trsi("init", "--brief", self.brief)
        assert code == 0, err
        return out

    def write_run(self, **changes):
        """Change top-level keys of the generated run.yaml."""
        d = yaml.safe_load(self.run.read_text())
        d.update(changes)
        self.run.write_text(yaml.safe_dump(d))

    def trsi(self, *args) -> tuple[int, str, str]:
        self.capsys.readouterr()
        code = main([str(a) for a in args])
        out = self.capsys.readouterr()
        return code, out.out, out.err

    def proposal(self, name, base="elephant-1", recipe=EL, overrides=None, note="a reason", prediction=None):
        d = {"base": base, "recipe": recipe, "note": note,
             "overrides": {k: {"value": v, "reason": f"try {k}"} for k, v in (overrides or {}).items()},
             "prediction": prediction if prediction is not None else {"auprc": 0.9, "p_accept": 0.3}}
        p = self.out / "trsi" / "d-0002" / "proposals" / f"{name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(d))
        return p

    def n_calls(self) -> int:
        return len(self.calls.read_text().splitlines()) if self.calls.is_file() else 0

    def train_argv(self, config: Path, out: Path, marker: Path, producer: str = "train") -> list[str]:
        """The train command on this environment's task and sets."""
        return [str(MODULE / "train"), "--task", str(self.task_json), "--trainset", str(self.trainset),
                "--trainset-id", "trainset-0001", "--valset", str(self.valset), "--valset-id", "valset-0001",
                "--config", str(config), "--config-source", "default", "--round", "1", "--producer", producer,
                "--out", str(out), "--marker", str(marker)]

    def rsi(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "rsi", *map(str, args)], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(REPO)})


@pytest.fixture
def env(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path, capsys)
    for rel, d in ((EL, RECIPE), (MM, {**RECIPE, "onejev": {**RECIPE["onejev"], "lr": 5.0e-5}}),
                   ("examples/recipes/joey_full_1gpu.yaml", JOEY)):
        (e.ojh_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (e.ojh_root / rel).write_text(yaml.safe_dump(d))
    (e.ojh_root / "catalog" / "bases").mkdir(parents=True)
    (e.ojh_root / "catalog" / "bases" / "joey-1.yaml").write_text(
        yaml.safe_dump({"name": "joey-1", "multimodal": True,
                        "default_recipes": {1: "examples/recipes/joey_full_1gpu.yaml"}}))
    executable(e.ojh_root / ".venv" / "bin" / "ojh", f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_ojh.py'} \"$@\"\n")
    (tmp_path / "machine.yaml").write_text("out_root: /nowhere\n")
    (tmp_path / "fake_nvidia_smi.py").write_text(FAKE_NVIDIA_SMI)
    executable(tmp_path / "bin" / "nvidia-smi",
               f"#!/bin/sh\nexec {sys.executable} {tmp_path / 'fake_nvidia_smi.py'} \"$@\"\n")
    # cards: 0 busy (another user's process), 1 and 2 idle (2 the most free), 3 above the idle threshold, 4 idle but
    # not in gpu_ids
    monkeypatch.setenv("FAKE_GPUS", "0, GPU-fake-0, 30000, 51920\n1, GPU-fake-1, 5, 81915\n2, GPU-fake-2, 3, 81917\n"
                                    "3, GPU-fake-3, 2000, 79920\n4, GPU-fake-4, 0, 81920\n")
    monkeypatch.setenv("FAKE_GPU_APPS", "GPU-fake-0, 4242\n")
    executable(tmp_path / "bin" / "tmux", '#!/bin/sh\ncase "$1" in has-session) [ "$FAKE_TMUX_GONE" = 1 ] && exit 1; '
               f'exit 0;; esac\necho "$@" >> {e.tmux_log}\n')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    write_set(e.trainset, "trainset", [row(i, "train") for i in range(60)], with_inputs=False)
    write_set(e.valset, "valset", [row(i, "val") for i in range(100, 200)], with_inputs=True)
    monkeypatch.setenv("FAKE_OJH_OUT", str(tmp_path / "ojh_runs"))
    monkeypatch.setenv("FAKE_OJH_CALLS", str(e.calls))
    monkeypatch.delenv("FAKE_OJH_PENDING", raising=False)
    monkeypatch.setenv("FAKE_OJH_QUALITY", "{}")
    monkeypatch.setenv("TRSI_TEST_SCORER", str(HERE / "fake_scorer.py"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("FAKE_SCORER_FAIL", raising=False)
    monkeypatch.delenv("FAKE_SCORER_CALLS", raising=False)
    for k in ("FAKE_SCORER_OOM_CARD", "FAKE_OJH_SLEEP", "FAKE_TMUX_GONE", "FAKE_GPU_BUSY_ONCE", "FAKE_OJH_CLEANUP_SKIP"):
        monkeypatch.delenv(k, raising=False)
    e.write_task()
    e.write_brief()
    e.init()
    return e
