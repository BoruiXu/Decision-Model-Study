import json

import yaml

from harness.status import render, status


def setup(tmp_path, records, seeds=(42, 7), max_steps=0):
    run_dir = tmp_path / "plan"
    d = run_dir / "runs" / f"exp-s{seeds[-1]}"
    d.mkdir(parents=True)
    lock = {"experiment": "exp", "run_dir": str(run_dir), "seeds": list(seeds), "n_gpus": 2,
            "flags": {"epochs": 2, "max_steps": max_steps}, "plan": {"steps": 50}}
    (run_dir / "run.lock.yaml").write_text(yaml.safe_dump(lock))
    (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir


RECS = [{"step": 10, "sec_per_step": 2.0}, {"event": "harness_eval", "step": 10, "set": "val", "n": 5,
                                              "nll": 0.5, "path": "p"},
        {"step": 20, "sec_per_step": 1.0}]


def test_status_running_and_failed(tmp_path):
    run_dir = setup(tmp_path, RECS)
    s = status(run_dir)
    assert s["state"] == "running" and s["seed"] == "2/2" and s["step"] == 20
    assert s["total_steps"] == 100 and s["eta_s"] == 80
    assert s["last_eval"]["nll"] == 0.5
    assert len(render(s).splitlines()) <= 6
    (run_dir / "status").write_text("running\n")
    assert status(run_dir)["state"] == "running"
    (run_dir / "DONE").write_text(json.dumps({"exit_code": 3, "end": "x", "stage": "train"}))
    assert status(run_dir)["state"] == "failed"
    (run_dir / "DONE").write_text(json.dumps({"exit_code": 0, "end": "x", "stage": "finalize"}))
    (run_dir / "status").write_text("complete\n")
    assert status(run_dir)["state"] == "complete"


def test_status_max_steps_cap_and_no_log(tmp_path):
    run_dir = setup(tmp_path, [])
    assert status(run_dir)["step"] is None and status(run_dir)["eta_s"] is None
    run_dir = setup(tmp_path / "b", RECS, max_steps=60)
    assert status(run_dir)["total_steps"] == 60


def test_status_epochs_default(tmp_path):
    run_dir = setup(tmp_path, RECS)
    lock = yaml.safe_load((run_dir / "run.lock.yaml").read_text())
    del lock["flags"]["epochs"]
    (run_dir / "run.lock.yaml").write_text(yaml.safe_dump(lock))
    assert status(run_dir)["total_steps"] == 50
