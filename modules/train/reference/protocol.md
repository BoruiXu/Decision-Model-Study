# How to compare training configs, and why

## Rules

1. **Probe**: a candidate first runs only the probe seed (seed 0); the primary metric in its candidate's `metrics_val` (at packing, the checkpoint the harness chose runs `model.predict` on the valset, and `rsi score` computes the metric from those predictions) is compared with the incumbent's seed 0. Only when it beats it by more than `probe_margin` are the other seeds added; otherwise it is not accepted.
2. **Accept**: once all seeds have run, it is accepted only when its mean beats the incumbent's (on the same seeds) by more than `accept_margin` and the mean worsening of ECE and Brier is within the guards.
3. **Budget**: each trial is booked at its candidate's `cost.compute_hours` (the GPU hours the harness records, i.e. training wall-clock × GPUs, plus scoring time); a failed trial is booked at the GPU hours the harness recorded all the same. The start's seeds are booked too (every trained trial is a candidate the framework registers); a registered candidate taken over with `trsi adopt` is not booked again, as it was booked in its own round. The budget is the brief's `budget_left.compute_hours`; when the spent hours plus the next seed's expected cost would exceed it, nothing more is launched.
4. **No reruns**: a trial with the same base, the same recipe content, the same overrides, the same seed and the same environment exp.yaml trains only once; another run that needs it reuses it directly, but still books it.

The margins come from the task's `search` (`probe_margin`, `accept_margin`), 0.035 when not set. The default 0.035 comes from Elliptic: over 3 seeds of the same recipe, the standard deviation of the validation AUPRC is 0.0175, and 0.035 is twice that. On another task, once the start's 3 seeds have run, check the margins against the actual standard deviation; when they do not fit, write the suggested values in the `record` and in the hand-back's `questions`: changing the margins is changing the task definition, which the user approves.

## Why not other methods (all measured on Elliptic)

- **Short-run screening**: screening the same options twice with two seeds gave rankings correlated at only 0.08. The 4 options a screening put 0.04–0.13 above the team recipe came out at −0.083 to +0.002 after full training; not one won. Short trainings favour changes that "learn fast early".
- **A small-data setting**: with only 1/4 of the training data, the ranking of 6 recipes correlated with full training at only 0.09–0.37, and the seed-to-seed variation was 2.2 times that of full training.
- **One seed + row bootstrap**: about nine tenths of the difference between seeds comes from the training itself (initialization, data order, which checkpoint is chosen), and only about a tenth from the limited number of valset rows. Resampling rows cannot see the former, and takes a seed's lucky draw for an improvement.
- **3 seeds for small improvements**: to see +0.01 with 80% power takes about 38 seeds per side. So 3 seeds only confirm big improvements; small improvements cannot be told apart within this budget and are simply not accepted.

Stand-ins (short runs, small data, small models) can be measured again on a later task: they are worth using only when their ranking correlates with full training at ≥ 0.8 over at least 5 recipes.

## Comparing two versions of the Skill

- With the same budget (e.g. 100 GPU hours), look at the best config found when the budget runs out (the "Best scheme against compute hours" table of `report.md`).
- The two versions share one trial record (the same `<outdir>/trsi/`); trials already run are reused directly, but booked at their price.
- The judge must be a new session that sees only the evidence of the time, never choosing with the results already known.
- Change the Skill on one task and only test it on another; the method counts as better only when the improvement also holds on the task that took no part in the change.
