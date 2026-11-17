"""Checkpoint selection from the in-training ``harness_eval`` events. Imports without torch."""
from __future__ import annotations

from pathlib import Path


def eval_events(events, set_name: str = "select") -> list[dict]:
    return [e for e in events if e.get("event") == "harness_eval" and e.get("set") == set_name]


def select_step(events, set_name: str = "select") -> int:
    """Step with the lowest NLL on ``set_name``; ties go to the earlier step; step 0 is never selected."""
    cands = [e for e in eval_events(events, set_name) if e["step"] != 0]
    if not cands:
        raise ValueError(f"no harness_eval events for set {set_name!r} after step 0")
    return min(cands, key=lambda e: (e["nll"], e["step"]))["step"]


def checkpoint_dir(seed_dir, step: int, mode: str, last_step: int | None) -> Path | None:
    """OneJev's save dir for ``step``: ``ckpt-step<N>`` (full) or ``adapter-step<N>`` (LoRA).

    The end of training is saved as ``final/`` (full) or ``adapter/`` (LoRA) when the last step is not
    a multiple of the save interval. Returns None when no directory holds the step.
    """
    seed_dir = Path(seed_dir)
    full = mode == "full"
    path = seed_dir / (f"ckpt-step{step}" if full else f"adapter-step{step}")
    if path.is_dir():
        return path
    if step == last_step:
        end = seed_dir / ("final" if full else "adapter")
        if end.is_dir():
            return end
    return None
