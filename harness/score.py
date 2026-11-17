"""Reload a saved checkpoint and score rows one at a time with OneJev's own scoring path (GPU).

Loading follows OneJev's ``train/eval_mm.py``: ``common.load_bundle`` and ``Readout.attach`` with the readout
settings saved next to the weights. A LoRA adapter is loaded on top of the pinned base with
``peft.PeftModel.from_pretrained`` (after the readout is attached to the base, as training does). Rows are
scored with ``max_rows=1`` under the same autocast setting as the in-training eval, so the result is
comparable with the ``harness_eval`` NLL. torch, peft and OneJev are imported inside functions so this
module imports on a CPU-only machine.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

from harness.catalog import ROOT
from harness.eval_records import merge_sorted, records

ONEJEV_DIR = ROOT / "vendor" / "OneJev"
# Never skip a row here: the in-training eval already refused rows over its own max_len, and every row
# must be scored for the comparison with the training NLL.
MAX_LEN = 1 << 22


def ensure_onejev_path() -> None:
    """Make the pinned OneJev checkout importable in this process (the runner only sets it for the child)."""
    path = str(ONEJEV_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_scoring_bundle(model_dir, *, base_dir=None, multimodal: bool, device: str = "cuda:0",
                        chat_template: str | None = None):
    """(bundle, adapter) with the checkpoint's readout attached; ``adapter`` is the PeftModel or None."""
    ensure_onejev_path()
    from qev.readout import HEAD_FILE, Readout, ReadoutConfig
    from train import common

    weights = Path(base_dir) if base_dir is not None else Path(model_dir)
    bundle = common.load_bundle(str(weights), device=device, dtype="bfloat16", multimodal=multimodal,
                                chat_template=chat_template)
    rcfg = ReadoutConfig.load(model_dir)  # LoRA: the adapter's readout.json carries the attention setting
    head = Path(model_dir) / HEAD_FILE
    bundle.readout = Readout.attach(bundle.model, rcfg, slot_ids=bundle.slot_ids,
                                    head_path=head if rcfg.head else None)
    adapter = None
    if base_dir is not None:
        import peft

        # injects the LoRA layers into bundle.model in place, as get_peft_model does during training
        adapter = peft.PeftModel.from_pretrained(bundle.model, str(model_dir))
        adapter.eval()
    bundle.model.eval()
    return bundle, adapter


def score_checkpoint(model_dir, rows, *, base_dir=None, multimodal: bool, device: str = "cuda:0",
                     autocast: bool = True, chat_template: str | None = None,
                     media_root: str | None = None) -> dict[str, dict]:
    """Score ``rows`` with the checkpoint in ``model_dir`` (an adapter dir when ``base_dir`` is given).

    Returns ``{row id: {"id", "labels", "probs", "target"}}`` (``eval_records.records``).
    """
    ensure_onejev_path()
    import torch
    from train import common, sft

    if media_root:
        from train import mm

        mm.set_root(media_root)
    bundle, adapter = load_scoring_bundle(model_dir, base_dir=base_dir, multimodal=multimodal, device=device,
                                          chat_template=chat_template)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            if bundle.multimodal:
                scored = sft.collect_mm(bundle, rows, MAX_LEN, MAX_LEN, 1, device)
            else:
                scored = common.collect_slot_logits(bundle, rows, max_len=MAX_LEN, batch_tokens=MAX_LEN,
                                                    max_rows=1, device=device, quiet=True)
        recs = merge_sorted([records(scored, rows)], rows, "select")
    finally:
        del bundle, adapter
        release_gpu_memory()
    return {r["id"]: r for r in recs}
