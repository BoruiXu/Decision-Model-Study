"""End to end on CPU: inspect -> make-split -> (agent writes adapter + exp.yaml) -> ojh run with a fake trainer
-> runner -> finalize -> RESULT.md and ojh status."""
import functools
import json
import random
import shlex
import sys
from pathlib import Path

import yaml

from harness import cli, gpu, oneshot, runner
from harness.eval_records import nll
from harness.finalize import finalize
from harness.inspect_data import inspect_path
from harness.make_split import select_report, write_split_file

sys.path.insert(0, str(Path(__file__).parent))
from test_finalize import FakeScorer  # noqa: E402

CARDS = "0, GPU-a, NVIDIA A100-SXM4-80GB, 80000\n"
TOPICS = ["math", "code", "chat", "travel", "food", "news"]

ADAPTER = '''"""chat + label jsonl -> OneJev rows (written by the agent after `ojh inspect`)."""
import json


def adapt(path, split="train"):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            yes = 1.0 if r["label"] == "good" else 0.0
            yield {"id": r["id"], "split": split, "state": {"messages": r["conversation"]},
                   "question": {"type": "noul", "instructions": "Is this reply good?",
                                "criteria": {"true": "good", "false": "bad"}},
                   "target": {"yes": yes, "no": 1.0 - yes}, "meta": {"topic": r["topic"]}}
'''


def raw_rows(prefix, n, rng):
    out = []
    for i in range(n):
        good = rng.random() < 0.5
        words = rng.choices(["fine", "great", "ok"] if good else ["bad", "wrong", "meh"], k=5)
        out.append({"id": f"{prefix}{i}", "label": "good" if good else "bad", "topic": rng.choice(TOPICS),
                    "conversation": [{"role": "user", "content": "question " + " ".join(rng.choices(TOPICS, k=4))},
                                     {"role": "assistant", "content": " ".join(words)}]})
    return out


class FakeTrainer:
    """Popen stand-in for harness.train_entry: writes the log, per-step select predictions and checkpoints."""

    def __init__(self, cmd, **kw):
        arg = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
        seed_dir = Path(arg("--runs-dir")) / arg("--name")
        select = Path(arg("--eval").split("=", 1)[1])
        every = int(arg("--eval-every"))
        rows = [json.loads(line) for line in select.read_text().splitlines()]
        rng = random.Random(int(arg("--seed")))
        (seed_dir / "preds").mkdir(parents=True)
        log = []
        for step in range(0, 10 + 1, every):
            noise = 0.4 if step == 0 else 0.4 / (1 + abs(step - 6))  # best around step 6
            recs = []
            for r in rows:
                t = r["target"].get("yes", 0.0)
                p = min(0.98, max(0.02, 0.2 + 0.6 * t + rng.uniform(-noise, noise)))
                recs.append({"id": r["id"], "labels": ["yes", "no"], "probs": [p, 1 - p], "target": [t, 1 - t]})
            path = seed_dir / "preds" / f"select-step{step}.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in recs))
            if step:
                log.append({"step": step, "loss": 0.4, "sec_per_step": 0.1})
                ck = seed_dir / f"ckpt-step{step}"
                ck.mkdir()
                (ck / "model.safetensors").write_bytes(b"w" * step)
            log.append({"event": "harness_eval", "step": step, "set": "select", "n": len(recs), "nll": nll(recs),
                        "path": str(path)})
        (seed_dir / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))

    pid = 1

    def wait(self, timeout=None):
        return 0


def test_inspect_run_finalize_result(tmp_path, capsys):
    rng = random.Random(0)
    data = tmp_path / "data"
    data.mkdir()
    for name, n in (("train", 120), ("val", 48)):
        (data / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw_rows(name[0], n, rng)))

    # 1. facts only
    facts = {Path(f["path"]).name: f for f in inspect_path(data)["files"]}
    assert facts["train.jsonl"]["name_hint"] == "train" and facts["val.jsonl"]["name_hint"] == "val"
    assert facts["train.jsonl"]["onejev"]["passes"] is False
    assert {v["value"] for v in facts["train.jsonl"]["fields"]["label"]["values"]} == {"good", "bad"}
    assert facts["train.jsonl"]["fields"]["conversation[].content"]["text_like"] is True

    # 2. the agent's work: a workspace outside the repo with an adapter, a split file and exp.yaml
    ws = tmp_path / "work"
    ws.mkdir()
    (ws / "chat_adapter.py").write_text(ADAPTER)
    write_split_file(select_report([("val", str(data / "val.jsonl"))], group="topic"), ws / "splits.json")
    (ws / "exp.yaml").write_text(yaml.safe_dump({
        "name": "chat_router",
        "data": {"adapter": "chat_adapter.py", "train": str(data / "train.jsonl"),
                 "eval": {"val": {"train_labels": str(data / "val.jsonl"), "adapter_args": {"split": "val"}}},
                 "split_file": "splits.json"},
        "fields": {"label": "target.yes", "group": "meta.topic", "text": "messages"},
        "evaluator": {"bootstrap_reps": 50}}))
    (ws / "machine.yaml").write_text(yaml.safe_dump({"python": sys.executable, "model_dir": str(tmp_path / "models"),
                                                     "out_root": str(tmp_path / "runs")}))

    # 3. ojh run: the fake tmux runs the real runner at once, with a fake trainer and scorer
    def tmux(argv):
        inner = shlex.split(argv[-1])
        lock_path = inner[inner.index("harness.runner") + 1]
        acquire = functools.partial(gpu.acquire, query=lambda uuid=None: CARDS, apps_query=lambda: "")
        rc = runner.run(lock_path, acquire=acquire, popen=FakeTrainer,
                        finalize=lambda rd: finalize(rd, scorer=FakeScorer(), devices=["GPU-a"]),
                        killpg=lambda *a: None)
        assert rc == 0

    out = oneshot.run(ws / "exp.yaml", base="joey-1", gpus=[0], machine=str(ws / "machine.yaml"),
                      query=lambda uuid=None: CARDS, apps_query=lambda: "",
                      plan_runner=lambda lock: "PLAN_OK\n10 steps for 1 ranks\n",
                      fingerprint=lambda: {"pip_freeze_sha256": "ab" * 32, "driver_version": "570"}, tmux=tmux)
    run_dir = Path(out["run_dir"])
    assert out["recipe"]["source"] == "catalog default" and out["gpus"][0]["uuid"] == "GPU-a"

    # 4. what the user gets back
    done = json.loads((run_dir / "DONE").read_text())
    assert done["exit_code"] == 0 and done["stage"] == "finalize"
    res = json.loads((run_dir / "result.json").read_text())
    md = (run_dir / "RESULT.md").read_text()
    assert res["run_id"] == run_dir.name and res["data"]["rows"]["train"] == 120
    assert res["data"]["rows"]["select"] + sum(res["data"]["rows"]["report"].values()) == 48
    assert res["data"]["adapter"]["name"].endswith("chat_adapter.py")
    s = res["seeds"]["42"]
    assert [p["step"] for p in s["curve"]] == [0, 2, 4, 6, 8, 10] and s["selected_step"] == 6
    assert {"model", "tfidf", "group_prior", "length"} <= set(s["metrics"]["methods"])
    assert s["metrics"]["methods"]["model"]["auc_ci95"] is not None
    assert "6 | " in md and "← selected" in md and "## Gate" in md
    assert (run_dir / "adapter" / "chat_adapter.py").read_text() == ADAPTER
    capsys.readouterr()
    assert cli.main(["status", str(run_dir)]) == 0
    assert f"result: {run_dir / 'RESULT.md'}" in capsys.readouterr().out
