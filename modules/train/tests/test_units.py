"""ECE, rules, the search space, the workspace exp.yaml, the scoring card and model.predict's scorer."""
import fcntl
import json
import os
import subprocess
import sys

import pytest
import yaml

from conftest import EL, HERE, MM, row, write_jsonl
from trsi import harness, pack
from trsi.progress import train_step
from trsi.candidate import Candidate, Override, dead_ends, outside_space
from trsi.config import ConfigError, MetricSpec, Rules, load_run
from trsi.harness import TrialResult
from trsi.metrics import ece
from trsi.rules import after_probe, decide


# ---- ECE (the other metrics come from `rsi score`) ----

def test_ece_hand_example():
    y, p = [1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1]
    assert ece(y, p) == pytest.approx((0.1 + 0.8 + 0.6 + 0.1) / 4)    # one row per bin
    assert ece([1, 0], [1.0, 0.0]) == 0.0                              # p = 1 falls in the last bin


# ---- rules ----

def _r(seed, auprc_, ece_=0.02, brier_=0.02):
    return TrialResult("rd", seed, "done", {"auprc": auprc_, "ece": ece_, "brier": brier_})


RULES, METRIC = Rules(), MetricSpec()


def test_probe_margin():
    assert after_probe(_r(0, 0.913), _r(0, 0.883), RULES, METRIC)[0] == "reject"       # +0.030
    v, g = after_probe(_r(0, 0.923), _r(0, 0.883), RULES, METRIC)                       # +0.040
    assert v == "more_seeds" and g == pytest.approx(0.04)
    with pytest.raises(ValueError):
        after_probe(_r(1, 0.95), _r(0, 0.883), RULES, METRIC)


def test_decide_guards_and_accept():
    inc = {0: _r(0, 0.883), 1: _r(1, 0.905), 2: _r(2, 0.918)}
    better = {0: _r(0, 0.923), 1: _r(1, 0.945), 2: _r(2, 0.958)}
    ok, why = decide(better, inc, RULES, METRIC)
    assert ok and "3/3 higher" in why
    worse_ece = {s: _r(s, r.metrics["auprc"], ece_=0.05) for s, r in better.items()}     # ECE +0.03 > 0.02
    ok, why = decide(worse_ece, inc, RULES, METRIC)
    assert not ok and "ece +0.0300 (beyond 0.02)" in why
    small = {s: _r(s, r.metrics["auprc"] + 0.03) for s, r in inc.items()}
    assert not decide(small, inc, RULES, METRIC)[0]
    with pytest.raises(ValueError):
        decide({0: better[0]}, inc, RULES, METRIC)


# ---- the search space (from the task's search_space) ----

def test_outside_space_and_dead_ends(env):
    run = load_run(env.run)
    el = env.ojh_root / EL
    ok = Candidate("elephant-1", el, {"onejev.lr": Override(5e-5, "r"), "lora.r": Override(64, "r")})
    assert outside_space(ok, run.space) == []
    bad = Candidate("joey-1", env.run.parent / "exp.yaml", {"onejev.lr": Override(3e-4, "r"),
                                                            "onejev.epochs": Override(2, "r"),
                                                            "lora.r": Override(256, "r")})
    msgs = " | ".join(outside_space(bad, run.space))
    for part in ("base joey-1", "recipe", "onejev.lr = 0.0003", "onejev.epochs is not in", "lora.r = 256"):
        assert part in msgs
    dead = Candidate("elephant-1", el, {"onejev.brier_weight": Override(3.0, "r")})
    assert outside_space(dead, run.space) == [] and "dead end" in dead_ends(dead, run.space)[0]


def test_config_refuses_mode_and_derived_overrides(env):
    d = yaml.safe_load(env.run.read_text())
    d["space"]["overrides"]["mode"] = {"values": ["full"]}
    env.run.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="through recipes"):
        load_run(env.run)
    for key in ("onejev.attention", "onejev.device"):   # T6: every flag ojh reserves
        d["space"]["overrides"] = {key: {"values": ["x"]}}
        env.run.write_text(yaml.safe_dump(d))
        with pytest.raises(ConfigError, match="derives"):
            load_run(env.run)


def test_python_path_is_expanded(env, monkeypatch):
    # T10: resources.onejev.python may use ~ and set variables; a variable left unset is refused
    monkeypatch.setenv("HOME", str(env.root))
    monkeypatch.delenv("NOPE", raising=False)
    for i, (py, want) in enumerate((("~/py", str(env.root / "py")), ("$NOPE/py", None))):
        t = env.write_task()
        t["resources"]["onejev"]["python"] = py
        env.task_json.write_text(json.dumps(t))
        env.write_brief(outdir=str(env.root / f"o{i}"))
        code, _, err = env.trsi("init", "--brief", env.brief)
        run = env.root / f"o{i}" / "trsi" / "run.yaml"
        if want:
            assert code == 0 and yaml.safe_load(run.read_text())["framework"]["python"] == want, err
        else:
            assert code == 2 and "$NOPE" in err and not run.exists()


def test_launch_refusals(env):
    cases = [
        (env.proposal("a", base="joey-1"), "outside the task's search_space"),
        (env.proposal("b", overrides={"onejev.brier_weight": 3}), "dead end"),
        (env.proposal("c", prediction={"p_accept": 0.5}), "prediction must be"),
        (env.proposal("d", prediction={"auprc": 0.9, "p_accept": 1.5}), "p_accept must be in"),
        (env.proposal("e", note=""), "write a note"),
    ]
    for p, msg in cases:
        code, out, err = env.trsi("launch", env.run, p, "--seed", 0, "--gpus", 0)
        assert code == 2 and msg in err, (p.name, err)
    d = yaml.safe_load(env.proposal("f").read_text())
    d["overrides"] = {"onejev.lr": {"value": 5e-5}}
    (env.root / "f2.yaml").write_text(yaml.safe_dump(d))
    code, _, err = env.trsi("launch", env.run, env.root / "f2.yaml", "--seed", 0, "--gpus", 0)
    assert code == 2 and "needs a reason" in err
    code, _, err = env.trsi("launch", env.run, env.proposal("g"), "--seed", 0, "--gpus", 7)
    assert code == 2 and "not in the task's resources.gpu_ids" in err
    assert env.n_calls() == 0


# ---- the scoring card and model.predict's scorer ----

def test_pick_scoring_card_rules(env, monkeypatch, tmp_path):
    # (a): no other process, enough free memory (min_free_mib and need), the ojh lock free (held while scoring),
    # prefer when it qualifies, else the most free; with none left the error gives each card's reason
    monkeypatch.setenv("FAKE_GPUS", "0, GPU-fake-0, 10, 70000\n1, GPU-fake-1, 10, 30000\n2, GPU-fake-2, 10, 50000\n"
                                    "3, GPU-fake-3, 10, 60000\n5, GPU-fake-5, [N/A], [N/A]\n")
    assert harness.gpu_table()[-1][0] == 5                                  # T14: [N/A] (MIG) does not raise
    locks = tmp_path / "locks"
    locks.mkdir()
    held = os.open(locks / "GPU-fake-3.lock", os.O_RDONLY | os.O_CREAT)    # another job holds card 3
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for kw, want in (({}, 3),                                   # card 0 runs another process; 5 is [N/A]
                     ({"lock_dir": locks}, 2), ({"prefer": 1}, 1), ({"prefer": 0}, 3), ({"min_free_mib": 55000}, 3),
                     ({"need_mib": 40000, "exclude": ("GPU-fake-3",)}, 2), ({"allowed": (0, 1)}, 1)):
        i, uuid, lock = harness.pick_scoring_card(**{"allowed": None, "need_mib": 0, **kw})
        assert (i, uuid) == (want, f"GPU-fake-{want}"), kw
        harness.release(lock)
    i, _, lock = harness.pick_scoring_card(None, 0, lock_dir=locks)
    assert i == 2 and harness.pick_scoring_card(None, 0, lock_dir=locks)[0] == 1     # card 2's lock is held
    with pytest.raises(harness.HarnessError) as e:
        harness.pick_scoring_card((0, 1, 3, 5), 40000, lock_dir=locks)
    for part in ("card 0: another process", "card 1: 30000 MiB free < 40000", "card 3: locked", "card 5: "):
        assert part in str(e.value)


def test_predict_ignores_env_scorer(tmp_path):
    # T8: model.predict uses a stand-in scorer only when packing wrote it into predict.json, never from the environment
    (tmp_path / "predict.py").write_text(pack.PREDICT_PY)
    (tmp_path / "predict.json").write_text(json.dumps({"kwargs": {}, "env": {}}))
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "fake_model.json").write_text(json.dumps({"quality": 1.0, "key": "k", "seed": 0}))
    write_jsonl(tmp_path / "in.jsonl", [row(1, "val")])
    p = subprocess.run([sys.executable, str(tmp_path / "predict.py"), "--inputs", str(tmp_path / "in.jsonl"), "--out",
                        str(tmp_path / "out.jsonl")], capture_output=True, text=True,
                       env={**os.environ, "TRSI_TEST_SCORER": str(HERE / "fake_scorer.py"), "CUDA_VISIBLE_DEVICES": "0"})
    # the stand-in scorer would write out.jsonl and exit 0; ojh's scorer fails here (no harness, or no checkpoint kwargs)
    assert p.returncode != 0 and not (tmp_path / "out.jsonl").exists() and "fake" not in p.stderr


# ---- the workspace exp.yaml of one trial ----

def test_workspace(env):
    run = load_run(env.run)
    exp = env.run.parent / "exp.yaml"
    before = exp.read_text()
    c = Candidate("mammoth-1", env.ojh_root / MM, {"onejev.lr": Override(5e-5, "27B needs less")})
    p = harness.write_workspace(run, c, 1)
    w = yaml.safe_load(p.read_text())
    assert exp.read_text() == before
    assert w["name"].startswith("fraud-d-0002-") and w["name"].endswith("-s1") and w["seeds"] == [1]
    assert w["overrides"] == {"onejev.lr": {"value": 5e-5, "reason": "27B needs less"}}
    assert w["data"]["train"] == str(env.trainset / "rows.jsonl")
    assert w["data"]["eval"]["val"]["train_labels"] == str(env.valset / "rows.jsonl")
    assert w["fields"] == {"label": "target.yes", "group": "meta.state_id"} and w["merge"] is True  # LoRA: merged
    p0 = harness.write_workspace(run, c, 0)
    assert yaml.safe_load(p0.read_text())["name"] != w["name"]


def test_environment_with_overrides_is_refused(env):
    exp = env.run.parent / "exp.yaml"
    d = yaml.safe_load(exp.read_text())
    d["overrides"] = {"onejev.lr": {"value": 1, "reason": "x"}}
    exp.write_text(yaml.safe_dump(d))
    code, _, err = env.trsi("check", env.run)
    assert code == 2 and "must not have overrides" in err


def test_train_step(tmp_path):
    log = tmp_path / "train-s0.log"
    assert train_step(log) is None                                     # no log yet
    for text, want in (
        ("prepare ok\n", None),
        ("train [##..] 5/32 (15.6%) loss 0.1 eta 0.2h\n", (5, 32, 0.2)),
        # ANSI colours; a later update half written (no line end yet) is not read
        ("\x1b[32mtrain [##..] 6/32 (18.8%) loss 0.1 eta 0.2h\x1b[0m\rtrain [###.] 7/32 (21.", (6, 32, 0.2)),
        # a real ojh line: s/step (label first) beats the coarse eta: (32 - 26) x 8.3 s, not the tok/s number
        ("train [########################......] 26/32 (81.2%) loss 0.0935 ce 0.0935 brier 0.0569 slot_acc 0.969 "
         "gnorm 5.883 lr 8.96e-07 rows/s 3.87 tok/s 1219 s/step 8.3 mem 33.1G eta 0.0h\n", (26, 32, 6 * 8.3 / 3600)),
    ):
        log.write_text(text)
        assert train_step(log) == (pytest.approx(want) if want else None), text
