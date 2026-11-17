"""Merge a LoRA adapter into its pinned base and save a full model directory (GPU).

The merged directory is loadable like a full OneJev checkpoint: weights, tokenizer/processor, and the readout
settings (``readout.json`` plus ``modeling_overrides`` in ``config.json``, and the readout head when there is
one) taken from the adapter dir, or from the base when the adapter has none. torch, peft and OneJev are
imported inside functions so this module imports on a CPU-only machine.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from harness.score import ensure_onejev_path, release_gpu_memory


def merge_lora(base_dir, adapter_dir, out_dir, *, multimodal: bool = False, device: str = "cuda:0",
               chat_template: str | None = None) -> Path:
    ensure_onejev_path()
    import peft
    from qev.readout import CONFIG_FILE, HEAD_FILE, ReadoutConfig
    from train import common

    base_dir, adapter_dir, out = Path(base_dir), Path(adapter_dir), Path(out_dir)
    partial = out.with_name(out.name + ".partial")
    for p in (out, partial):  # never delete here: a leftover is listed as a deletion candidate
        if p.exists():
            raise FileExistsError(f"merge target already exists: {p}")
    bundle = common.load_bundle(str(base_dir), device=device, dtype="bfloat16", multimodal=multimodal,
                                chat_template=chat_template)
    merged = None
    try:
        merged = peft.PeftModel.from_pretrained(bundle.model, str(adapter_dir)).merge_and_unload()
        merged.save_pretrained(str(partial), safe_serialization=True)
        (bundle.processor or bundle.tokenizer).save_pretrained(str(partial))
        src = adapter_dir if (adapter_dir / CONFIG_FILE).exists() else base_dir
        ReadoutConfig.load(src).save(partial)  # readout.json + config.json modeling_overrides
        if (src / HEAD_FILE).exists():
            shutil.copy2(src / HEAD_FILE, partial / HEAD_FILE)
    finally:
        del bundle, merged
        release_gpu_memory()
    os.replace(partial, out)
    return out
