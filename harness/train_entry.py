"""Training entry: runs OneJev's trainer with its eval hook replaced.

Launched as `python -m harness.train_entry <train.sft flags>` (torchrun when multi-GPU). Only
`train.sft.run_evals` is replaced, so each in-training eval scores rows one at a time and writes per-row
probabilities plus a `harness_eval` event to log.jsonl. torch and OneJev are imported inside functions so
this module imports on a CPU-only machine.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from harness.eval_records import merge_sorted, nll, records


def harness_run_evals(model, bundle, eval_sets, args, rank, world, step, logf, is_main):
    import torch
    import torch.distributed as dist
    from train import common, sft

    was_training = model.training
    model.eval()
    max_len = args.eval_max_len or args.max_len
    sync = None
    if world > 1:
        def sync(n):
            count = torch.tensor([n], dtype=torch.long, device=args.device)
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            return int(count.item())
    for name, erows in eval_sets:
        mine = erows[rank::world]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=sft.use_autocast(args)):
            if bundle.multimodal:
                scored = sft.collect_mm(bundle, mine, max_len, args.batch_tokens, 1, args.device,
                                        sync_batches=sync)
            else:
                scored = common.collect_slot_logits(bundle, mine, max_len=max_len, batch_tokens=args.batch_tokens,
                                                    max_rows=1, device=args.device, quiet=True,
                                                    sync_batches=sync)
        recs = records(scored, mine)
        if world > 1:
            parts = [None] * world
            dist.all_gather_object(parts, recs)
        else:
            parts = [recs]
        if is_main:
            result = merge_sorted(parts, erows, name)
            path = Path(args.runs_dir) / args.name / "preds" / f"{name}-step{step}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in result))
            event = {"event": "harness_eval", "step": step, "set": name, "n": len(result),
                     "nll": nll(result), "path": str(path)}
            logf.write(json.dumps(event) + "\n")
            logf.flush()
            print("HARNESS_EVAL " + json.dumps(event), flush=True)
        torch.cuda.empty_cache()
    if was_training:
        model.train()


def install_audit(sft, args):
    """Port of the v1 frozen-weight / step-2 update audit (full fine-tune under --fsdp only)."""
    import torch
    import torch.distributed as dist
    from train import fsdp

    out = Path(args.runs_dir) / args.name / "audits"
    out.mkdir(parents=True, exist_ok=True)
    state = {"params": [], "steps": 0}
    original_shard, original_step, original_save = fsdp.shard_model, torch.optim.AdamW.step, sft.save_weights

    def local(t):
        return t.to_local() if hasattr(t, "to_local") else t

    def digest(t):
        t = local(t).detach().contiguous().cpu()
        return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()

    def shard_model(model, world, bf16_compute=True):
        before = []
        for name, p in model.named_parameters():
            visual = ".visual." in f".{name}." or name.startswith("visual")
            assert p.requires_grad != visual, name
            assert "lora_" not in name
            before.append(dict(name=name, shape=list(p.shape), elements=p.numel(), trainable=p.requires_grad))
        assert any(r["name"].endswith("lm_head.weight") and r["trainable"] for r in before)
        (out / f"modules_rank{dist.get_rank()}.json").write_text(json.dumps(before, indent=2) + "\n")
        result = original_shard(model, world, bf16_compute)
        state["params"] = list(model.named_parameters())
        frozen = {n: digest(p) for n, p in state["params"] if not p.requires_grad}
        (out / f"frozen_before_rank{dist.get_rank()}.json").write_text(json.dumps(frozen, indent=2) + "\n")
        return result

    def optimizer_step(self, closure=None):
        state["steps"] += 1
        n = state["steps"]
        # The first optimizer step has zero LR during warmup; updates are verified at step 2.
        audit = n in (1, 2)
        records_, snapshots = [], {}
        if audit:
            for name, p in state["params"]:
                if not p.requires_grad:
                    assert p.grad is None, name
                    continue
                value = local(p).detach()
                g = None if p.grad is None else local(p.grad).detach()
                assert g is not None, f"missing gradient: {name}"
                assert bool(torch.isfinite(g).all()), name
                nonzero = torch.count_nonzero(g).item()
                records_.append(dict(name=name, local_elements=g.numel(), nonzero=nonzero,
                                     gradient_norm=float(torch.linalg.vector_norm(g.float())), finite=True))
                if n == 2 and nonzero:
                    idx = torch.nonzero(g.flatten()[:65536], as_tuple=False).flatten()[:8]
                    snapshots[name] = (idx, value.flatten()[idx].clone())
        result = original_step(self, closure)
        if audit:
            changed = []
            by_name = dict(state["params"])
            for name, (idx, before) in snapshots.items():
                after = local(by_name[name]).detach().flatten()[idx]
                assert bool(torch.isfinite(after).all()), name
                if bool(torch.ne(before, after).any()):
                    changed.append(name)
            report = dict(step=n, rank=dist.get_rank(), lr=self.param_groups[0]["lr"],
                          gradient_tensors=len(records_), finite=True, changed_tensors=changed, tensors=records_)
            if n == 2:
                assert changed, "No parameter update at the first positive learning rate"
            (out / f"gradient_step{n}_rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n")
        return result

    def save_weights(model, bundle, out_dir, full, sharded=False, is_main=True):
        assert full and sharded
        frozen = {n: digest(p) for n, p in state["params"] if not p.requires_grad}
        before = json.loads((out / f"frozen_before_rank{dist.get_rank()}.json").read_text())
        assert frozen == before, "Frozen vision weights changed"
        (out / f"frozen_verified_rank{dist.get_rank()}.json").write_text(
            json.dumps(dict(passed=True, tensors=len(frozen))) + "\n")
        original_save(model, bundle, out_dir, full, sharded, is_main)

    fsdp.shard_model = shard_model
    torch.optim.AdamW.step = optimizer_step
    sft.save_weights = save_weights


def main():
    from train import sft

    sft.run_evals = harness_run_evals
    if os.environ.get("HARNESS_AUDIT") == "1":
        args = sft.parse_args(sys.argv[1:])
        if args.full_finetune and args.fsdp:
            install_audit(sft, args)
        else:
            print("HARNESS_AUDIT=1 skipped: the audit applies only to full fine-tune with --fsdp", flush=True)
    raise SystemExit(sft.main())


if __name__ == "__main__":
    main()
