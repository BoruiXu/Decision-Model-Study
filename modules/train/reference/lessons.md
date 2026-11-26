# What has been learned (Elliptic++ address task, transaction history input)

**Note**: the numbers below come from runs made directly with OneJev (81ce62f + serve.patch) on 2026-11-21 to 23, not yet through the harness. The metric is the AUPRC on the full valset (steps 30–34, 24,225 addresses, 1,838 illicit), with the checkpoint chosen by the AUPRC on a validation subset. The harness chooses the checkpoint by NLL by default, so the numbers will differ once rerun through the harness.

## Start

| Config | Seed 0 / 1 / 2 | Mean |
|---|---|---|
| elephant-1 LoRA, team recipe (`recipes/elephant_lora_1g.yaml`) | 0.883 / 0.905 / 0.918 | 0.902 (standard deviation 0.0175) |
| Baseline: XGBoost, same transaction history input | — | 0.885 |
| Baseline: the untrained base | — | 0.05–0.06 (below random, 0.076) |

On this task the untrained decision model ranks backwards (AUROC 0.28–0.39); it must be fine-tuned.

## Changes tried (all against the team recipe)

| Change | Seed 0 / 1 / 2 | Mean delta | Conclusion |
|---|---|---|---|
| Switch to mammoth-1 (same learning rate) | 0.895 / 0.929 / 0.889 | +0.002 | No win; the validation score fell all the way after step 1200 |
| mammoth-1 + half the learning rate (5e-5) | 0.907 / — / — | +0.024 (seed 0) | The most promising, but did not pass 0.035; best at step 800, down to 0.804 at the last step |
| Train 2 epochs | 0.876 / 0.869 / 0.919 | −0.014 | Dead end |
| Half the batch size (accum_rows 32) | 0.876 / 0.902 / — | −0.005 | Dead end |
| Brier weight 3 | 0.831 / 0.721 / 0.907 | −0.082 | Dead end |
| No input augmentation | 0.782 / 0.886 / 0.811 | −0.076 | Dead end |
| elephant-1 full (2 H200) | 0.616 / — / — | −0.27 | Checkpoints jump between 0.30 and 0.68, ECE 0.31; not a candidate until the cause is found |

## Four lessons

1. **Short-run screening cannot be trusted**: the rankings of two screenings correlated at only 0.08; of the 4 changes a screening favoured, not one won after full training.
2. **3 seeds cannot tell small improvements apart**: the standard deviation across seeds is 0.0175, so 3 seeds can confirm only improvements above 0.035.
3. **With a larger model, tune the learning rate too**: 27B on 9B's learning rate peaked early and then fell; with half the learning rate, seed 0 went from 0.895 to 0.907.
4. **Check the premise of a borrowed change first**: for a change that worked elsewhere, first write down why it worked, then see whether that holds here; when it does not, spend no GPU on it.

Also, the small setting with 1/4 of the data does not rank like full training (correlation 0.09–0.37) and cannot be used to compare configs.

## Cost reference

| Model / mode | Peak GPU memory | GPUs | 1 epoch (169k rows) | Storage |
|---|---|---|---|---|
| elephant-1 LoRA | 30 GB | 1 | H200 3–4 h, H100 about 5 h, A100 7–7.5 h | 337 MB per checkpoint, 1.7 GB per run |
| elephant-1 full (FSDP) | 86 GB each | 2 H200 (2 80 GB GPUs are not enough) | 1.6 h | 15 GB per checkpoint, 71 GB per run |
| mammoth-1 LoRA | 75 GB | 1 H200 | 9.5–18 h | 0.9 GB per checkpoint, 4.4 GB per run |
| mammoth-1 full | Does not fit even on 4 H200 | — | — | — |

In A100 GPU hours: about 8 per seed for elephant-1 LoRA, about 23 for mammoth-1 LoRA (an H200 is about 1.7 times as fast as an A100).

## Environment pitfalls met

| Symptom | Cause | Fix |
|---|---|---|
| Training 2.5 times slower | Another job stacked on the same GPU competes for compute | Put the important trainings on GPUs with no one else's jobs |
| GPU memory overflow when saving a checkpoint | Saving needs extra GPU memory, and a neighbour's use was below its peak at that moment | Estimate GPU memory at the peak plus a margin for saving |
| A batch size change actually sees more or fewer samples (accum_rows 32 sees 27% more, 128 sees 12% fewer) | The rows actually packed per step vary with row length | When comparing, look at the actual samples per step in the log |
| The per-step loss jumps between 0.01 and 0.7–1.0 | Rows of similar length are batched together, and length correlates strongly with the label | Smooth the curve before reading it; do not conclude from this that training is broken |
| On H100 and H200, training through the harness fails at the first backward step, asking for tilelang | flash-linear-attention 0.5.2 does not use Triton 3.4's backward kernel on Hopper GPUs and uses tilelang instead; the harness's `setup.sh` installs neither tilelang nor causal-conv1d. tilelang's just-in-time compilation needs CUDA 12.8's nvcc; the system's 12.0 does not work with gcc 13 | Install tilelang 0.1.15 and causal-conv1d 1.7.0 into the harness's environment (with the harness's `-c constraints.txt`, `--no-build-isolation` for causal-conv1d, `CUDA_HOME=/usr/local/cuda-12.8` during the install); in the machine config's `env`, set `CUDA_HOME` and put `/usr/local/cuda-12.8/bin` first in `PATH`. The versions of torch, triton and flash-linear-attention do not move. `install_env.sh` does all this (see "Installing the environment" in `README.md`) |
| The training finished, and the harness wrap-up was killed by signal 15 (xtrah200) | The machine's GPU watchdog: a process that used its GPU less than 20% of the time over 10 minutes is warned, then killed 2 minutes later; a user whose total GPU use is below 10% over 20 minutes is logged out by force. The wrap-up first scores again on the valset (using the GPU, about 4.5 minutes), and when exp.yaml has `fields.text` it then trains TF-IDF and length baselines on the CPU while the process still holds the GPU | Do not set `fields.text` in exp.yaml (the exp.yaml `trsi init` writes does not set it); never let a process hold a GPU idle |
