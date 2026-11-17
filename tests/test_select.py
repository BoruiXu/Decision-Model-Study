import pytest

from harness.select import checkpoint_dir, select_step


def ev(step, nll, set_name="select"):
    return {"event": "harness_eval", "step": step, "set": set_name, "n": 3, "nll": nll, "path": "p"}


def test_select_step_rules():
    events = [ev(0, 0.1), ev(4, 0.5), ev(8, 0.3), ev(12, 0.3), ev(16, 0.4),
              ev(20, 0.01, "other"), {"step": 21, "loss": 0.0}, {"event": "done", "steps": 24}]
    # step 0 excluded even though lowest; other sets and non-eval events ignored; tie -> earlier step
    assert select_step(events) == 8
    assert select_step(events, set_name="other") == 20
    with pytest.raises(ValueError):
        select_step([ev(0, 0.1)])
    with pytest.raises(ValueError):
        select_step([])


def test_checkpoint_dir_naming_and_final_fallback(tmp_path):
    for d in ("ckpt-step4", "final", "adapter-step4", "adapter"):
        (tmp_path / d).mkdir()
    assert checkpoint_dir(tmp_path, 4, "full", last_step=10) == tmp_path / "ckpt-step4"
    assert checkpoint_dir(tmp_path, 4, "lora", last_step=10) == tmp_path / "adapter-step4"
    assert checkpoint_dir(tmp_path, 10, "full", last_step=10) == tmp_path / "final"
    assert checkpoint_dir(tmp_path, 10, "lora", last_step=10) == tmp_path / "adapter"
    assert checkpoint_dir(tmp_path, 8, "full", last_step=10) is None
