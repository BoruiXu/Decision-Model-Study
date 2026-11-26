# What can change: base, recipe, overrides

A candidate = base + recipe + overrides. Below are the common options, which class each belongs to, its prior and its cost. What may really be searched is set by the task definition's `search_space`: the user approved it with the task definition, and widening it is changing the task definition, to be written into the hand-back's `questions`. The keys of `search_space` carry the `overrides.` prefix (`overrides.onejev.lr`); candidate YAMLs do not (`onejev.lr`).

## Three classes of change

| Class | What changes | How to write it | Prior |
|---|---|---|---|
| Model choice | The base | `base` (harness `catalog/bases/`: joey-1 2B, elephant-1 9B, mammoth-1 27B) | Structural, high; with a larger model, look at the learning rate too |
| Training mode | Full or LoRA, GPU count, parallelism | `recipe` (`mode`, `gpus`, `fsdp` in the recipe file) | Structural; full training is expensive, and unstable on Elliptic (see lessons) |
| Hyperparameters | Learning rate, epochs, loss, batch size, LoRA settings and so on | `overrides` (`onejev.<flag>`, `lora.<field>`, each with a reason) | Fitting; most are worth only small changes |

The training mode can change only by switching the recipe, never through an override: the recipe's other settings (e.g. `fsdp: true` in the elephant full recipe) do not work with LoRA, and the harness skill requires the same. Flags the harness derives itself (`attention`, `readout`, `eval_every`, `save_every`, `seed`, `model` and so on) cannot be overrides.

## Common overrides

| Key | Meaning | Common values | Notes |
|---|---|---|---|
| `onejev.lr` | Learning rate | 0.25×, 0.5×, 2× the recipe value | The literature and Elliptic both show: the larger the model, the smaller the right learning rate |
| `onejev.epochs` | Training epochs | 1, 2 | The cost multiplies with the epochs (`trsi` multiplies it in when estimating the budget) |
| `onejev.brier_weight` | Weight of the Brier term in the loss | 0, 1, 3 | Affects calibration; watch the ECE and Brier guards |
| `onejev.warmup_frac` | Warmup fraction | 0.03, 0.1 | |
| `onejev.weight_decay` | Weight decay | 0, 0.01, 0.1 | |
| `onejev.max_grad_norm` | Gradient clipping | 0.3, 1 | |
| `onejev.accum_rows` | Rows accumulated per step (the effective batch size) | 32, 64, 128 | The rows actually packed per step vary with row length; check the actual value in the harness log |
| `onejev.no_augment` | Turns off input augmentation | true | |
| `lora.r`, `lora.alpha` | LoRA rank and scaling | r 8, 32, 128 (alpha usually 2×r) | Changes how much the model can learn; structural |
| `lora.target_modules` | Which layers LoRA is added to | `"q_proj,k_proj,v_proj,o_proj"` (attention only), `"gate_proj,up_proj,down_proj"` (MLP only) | Structural |

Keys that affect only GPU memory, not the training result (`onejev.batch_tokens`, `onejev.max_rows`), are not candidates and should not be in the search space. When GPU memory runs short, handle it by the harness's troubleshooting table: that is fixing the environment, not a candidate.

## Order of choosing candidates

1. First see where the diagnosis points: the curve peaks early → learning rate, regularization; still improving at the end → epochs; seeds differ a lot → try big changes first, small changes cannot be told apart.
2. Structural before fitting: another base, which layers to train, the LoRA rank.
3. When switching the base, also prepare a "smaller learning rate" candidate.
4. With similar evidence, choose the cheaper: 1 GPU over several, LoRA over full, a small model over a large one.
5. Choose no dead end, unless you can show that the premise it failed on no longer holds.
