"""score.py and merge.py wiring against stand-in torch/peft/OneJev modules (CPU)."""
import contextlib
import sys
import types
from pathlib import Path

import pytest

from harness import merge, score


@pytest.fixture
def fake_onejev(monkeypatch):
    ev = []
    monkeypatch.setattr(sys, "path", list(sys.path))
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"

    def autocast(device, dtype=None, enabled=True):
        ev.append(("autocast", enabled))
        return contextlib.nullcontext()
    torch.autocast = autocast
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

    class Model:
        def eval(self):
            return self

        def save_pretrained(self, out, **kw):
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "model.safetensors").write_text("w")
            (Path(out) / "config.json").write_text("{}")

    class Tok:
        def save_pretrained(self, out):
            (Path(out) / "tokenizer.json").write_text("{}")

    class Bundle:
        multimodal = False

        def __init__(self):
            self.model, self.tokenizer, self.processor, self.slot_ids, self.readout = Model(), Tok(), None, [1, 2], None

    common = types.ModuleType("train.common")
    common.block_fla_kernels = lambda: ev.append(("block",))

    def load_bundle(path, device="cuda:0", dtype="bfloat16", multimodal=False, chat_template=None):
        ev.append(("load", str(path), dtype))
        return Bundle()
    common.load_bundle = load_bundle

    def collect(bundle, rows, max_len=0, batch_tokens=0, max_rows=64, device="", quiet=False, sync_batches=None):
        ev.append(("collect", max_rows))
        return [types.SimpleNamespace(idx=i, labels=["yes", "no"], logits=[0.0, 0.0], target=[1.0, 0.0])
                for i in range(len(rows))]
    common.collect_slot_logits = collect
    sft = types.ModuleType("train.sft")
    sft.collect_mm = lambda *a, **k: pytest.fail("text bundle must not use collect_mm")
    train = types.ModuleType("train")
    train.common, train.sft = common, sft

    readout = types.ModuleType("qev.readout")
    readout.HEAD_FILE, readout.CONFIG_FILE = "readout_head.pt", "readout.json"

    class ReadoutConfig:
        head = False

        @classmethod
        def load(cls, d):
            ev.append(("readout_load", str(d)))
            return cls()

        def save(self, d):
            (Path(d) / "readout.json").write_text('{"readout": "slot_lm"}')

    class Readout:
        @classmethod
        def attach(cls, model, cfg, slot_ids=None, head_path=None, device=None):
            ev.append(("attach",))
            return cls()
    readout.ReadoutConfig, readout.Readout = ReadoutConfig, Readout
    qev = types.ModuleType("qev")
    qev.readout = readout

    peft = types.ModuleType("peft")

    class PeftModel:
        @classmethod
        def from_pretrained(cls, model, path):
            ev.append(("peft", str(path)))
            obj = cls()
            obj.model = model
            return obj

        def eval(self):
            return self

        def merge_and_unload(self):
            ev.append(("merge",))
            return self.model
    peft.PeftModel = PeftModel
    mods = {"torch": torch, "train": train, "train.common": common, "train.sft": sft, "qev": qev,
            "qev.readout": readout, "peft": peft}
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return ev


def names(ev):
    return [e[0] for e in ev]


ROWS = [{"id": "x"}, {"id": "y"}]


def test_score_full_follows_dtype_and_blocks_fla_first(fake_onejev, tmp_path):
    out = score.score_checkpoint(tmp_path / "ckpt-step8", ROWS, multimodal=False, dtype="float32",
                                 disable_fla_kernels=True)
    assert set(out) == {"x", "y"} and out["x"]["probs"] == [0.5, 0.5] and out["x"]["id"] == "x"
    n = names(fake_onejev)
    assert n.index("block") < n.index("load") and "peft" not in n
    assert ("load", str(tmp_path / "ckpt-step8"), "float32") in fake_onejev and ("collect", 1) in fake_onejev
    assert ("autocast", True) in fake_onejev


def test_score_lora_loads_base_then_adapter(fake_onejev, tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter-step8"
    score.score_checkpoint(adapter, ROWS, base_dir=base, multimodal=False, autocast=False)
    n = names(fake_onejev)
    assert "block" not in n and ("load", str(base), "bfloat16") in fake_onejev
    assert ("readout_load", str(adapter)) in fake_onejev and ("peft", str(adapter)) in fake_onejev
    assert n.index("attach") < n.index("peft") and ("autocast", False) in fake_onejev


def test_merge_lora_writes_full_dir(fake_onejev, tmp_path):
    adapter = tmp_path / "adapter-step8"
    adapter.mkdir()
    (adapter / "readout.json").write_text('{"attention": "noncausal"}')
    out = merge.merge_lora(tmp_path / "base", adapter, tmp_path / "merged-step8", dtype="bfloat16",
                           disable_fla_kernels=True)
    assert out == tmp_path / "merged-step8" and not (tmp_path / "merged-step8.partial").exists()
    for f in ("model.safetensors", "config.json", "tokenizer.json", "readout.json"):
        assert (out / f).is_file(), f
    n = names(fake_onejev)
    assert n.index("block") < n.index("load") and ("load", str(tmp_path / "base"), "bfloat16") in fake_onejev
    assert ("readout_load", str(adapter)) in fake_onejev and n.index("peft") < n.index("merge")
    with pytest.raises(FileExistsError):
        merge.merge_lora(tmp_path / "base", adapter, out)


def test_merge_lora_replaces_a_stale_partial(fake_onejev, tmp_path):
    adapter = tmp_path / "adapter-step8"
    adapter.mkdir()
    stale = tmp_path / "merged-step8.partial"  # left by a merge that died
    stale.mkdir()
    (stale / "half.safetensors").write_text("junk")
    out = merge.merge_lora(tmp_path / "base", adapter, tmp_path / "merged-step8")
    assert (out / "model.safetensors").is_file() and not (out / "half.safetensors").exists()
    assert not stale.exists()
