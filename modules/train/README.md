# Train module (`modules/train/`)

Searches the training config within the task definition's search space (`search_space`): base, recipe, `onejev.<flag>`, `lora.<field>`, seed, step count. Training always goes through onejev-harness (`ojh run`). The agent does the judging (choosing free GPUs, diagnosing, proposing candidates, check-ins), and `trsi` does the mechanical work: building the ojh workspace, launching, waiting, packing every finished training as a framework candidate, comparing by rule, writing the `search`. The agent's flow is in `SKILL.md`.

## Files

| Path | Content |
|---|---|
| `SKILL.md` | The module agent's flow |
| `trsi/` | Command line: `init`, `check`, `launch`, `watch`, `adopt`, `collect`, `next`, `report` |
| `train` | The framework's train command: trains one config, packs it as a candidate, writes the completion marker |
| `install_env.sh`, `install_env.py` | Environment install: puts the libraries training needs into ojh's `.venv` and checks the whole (see "Installing the environment"); the `.py` is its helper (probing, machine config edits, GPU test) |
| `recipes/` | Recipes (ojh format) |
| `reference/` | What can be searched, how to compare, what has been learned |
| `tests/` | pytest: a fake `ojh` (`fake_ojh.py`) and a fake scoring function (`fake_scorer.py`) |

## Installing the environment

`install_env.sh` installs the libraries a training needs into ojh's `.venv`, then checks the whole. It is safe to rerun: when the machine is ready it changes nothing. The module agent runs it itself when a library is missing or incompatible (without asking the user, see "Common mistakes" in `SKILL.md`).

```bash
bash <repo>/modules/train/install_env.sh --ojh <ojh_repo> --machine <machine.yaml> [--env-file <ojh-env.sh>] \
  [--cache-root <dir>] [--check] [--no-gpu-test]
```

- In the task: `--ojh` is `resources.onejev.python` without the trailing `/.venv/bin/python`, `--machine` is `resources.onejev.machine`, `--env-file` is `resources.onejev.env_file` (sourced first).
- Six steps: ① When there is no `<ojh>/.venv`, run `setup.sh --machine <machine.yaml>` as ojh's README says; an existing venv is not rebuilt, and the versions of torch, triton, transformers and fla do not move (every install adds their current versions as constraints, and checks them afterwards). ② pyyaml, pytest. ③ numpy, scikit-learn and xgboost, which the compare module and data generation import: `modules/common/install_deps.sh --python <ojh>/.venv/bin/python` installs `modules/common/requirements.txt` and checks each one, with the versions of ① and ojh's `constraints.txt` as constraints (again without `constraints.txt` when that fails) and the same uv; not required: training does not need them and ③ Compare is a report, so a failure only warns (③ then hands back failed and is skipped, and the data generation steps that need them, dgen diagnose, smote, borderline and filter, refuse with an input error, so the data agent chooses another recipe), but an install that moved a protected version (torch, triton, transformers, fla) fails. ④ causal-conv1d 1.7.0: chooses a prebuilt wheel from the GitHub release by torch's major.minor, the CUDA major version, the C++ ABI (`torch._C._GLIBCXX_USE_CXX11_ABI`) and Python; when that does not work and the nvcc found has the same CUDA major version as torch, builds it from source; when neither works, only warns (training goes on with the slow PyTorch implementation). ⑤ With a GPU of compute capability ≥ 9.0 (Hopper), installs tilelang 0.1.15, finds nvcc (`CUDA_HOME`, `/usr/local/cuda*`, `nvidia-cuda-nvcc` in the venv; preferring one with the same CUDA version as torch), and writes `CUDA_HOME`, the full `PATH` and `TILELANG_CACHE_DIR` into the machine config's `env`. An existing `CUDA_HOME` whose nvcc runs is kept, unless its CUDA major version differs from torch's and another nvcc's matches (then it is replaced, and the output says so). An existing `PATH` keeps its other entries and gets nvcc's bin put first; without one, it writes nvcc's bin, the venv's bin and the system directories that exist (`/usr/local/sbin` to `/bin`), not the caller's `PATH` (it would go into the training and scoring processes; ojh does not expand variables). It first runs step ⑥'s GPU test with these new values and writes them only when the test passes, and what it writes are exactly the values tested; when the test fails it writes nothing and reports failed; when the test cannot run (`--no-gpu-test`, no usable GPU) it writes them anyway, with a warning. How it writes: it reads with pyyaml and edits line by line (other keys, comments, the end-of-line comments of replaced keys and CRLF are kept), after a backup as `machine.yaml.bak-<time>`; when it is already right, nothing changes; when `env` is written inline (except `env: {}`) and cannot be edited, it reports failed and lists the values to set by hand. ⑥ Check: imports torch, transformers, fla, causal_conv1d and yaml (plus tilelang on Hopper); on a GPU with no other process and with the machine config's `env`, runs fla's `chunk_gated_delta_rule` (the operator joey-1 uses) forward and backward on random tensors, and `causal_conv1d_fn`; `ojh doctor --no-network`.
- The GPU for the GPU test: when the caller sets `CUDA_VISIBLE_DEVICES`, only that one is used (the module agent passes the UUID of a GPU in `resources.gpu_ids` with no compute process, see `SKILL.md`); when it is not set, the script picks among all GPUs (except those the machine config excludes and those without enough free memory), preferring ones with no other process, and when all have some it uses one with processes too; it does not look at `resources.gpu_ids` or at ojh's GPU locks.
- Every install carries ojh's `-c constraints.txt`; it uses uv (PATH, `$OJH_TOOLS_DIR/uv/uv`, `<ojh>/.tools/uv/uv`, `<machine config directory>/tools/uv/uv`), or the venv's pip when there is no uv.
- Caches and temporary files (pip, uv, `TMPDIR`, triton, tilelang) go under `--cache-root`, by default `<machine config directory>/cache`; the log goes to `<cache-root>/logs/install_env-<time>.log`.
- Each item prints ok, missing or failed (skip and warn are only notes), and at the end a summary (what changed, warnings, failures). When a required item fails, the exit code is 1; causal-conv1d, pytest and step ③'s libraries are not required, and failing to install them only warns.
- `--check`: installs and changes nothing, runs no GPU test, writes no log, and only prints what is missing and what it would do; exit code 0 when nothing needs doing, else 1.
- To verify that the Hopper check works: `FLA_TILELANG=0 bash install_env.sh …` turns off fla's tilelang backend (no uninstall needed); the GPU test should then report fla's Triton/Hopper error, with exit code 1.
- The first GPU test compiles kernels (about one or two minutes on an H100); building causal-conv1d from source takes ten minutes or more.

## Running

`trsi` and the train command run in ojh's `.venv` (they need pyyaml; a candidate's `model.predict` needs torch and ojh); the framework itself does not depend on them. Below, `<py>` is the task definition's `resources.onejev.python` (ojh's `.venv/bin/python`), and `<repo>` is the framework repository root.

```bash
export PYTHONPATH=<repo>/modules/train
<py> -m trsi --help
<py> -m trsi init --brief <outdir>/brief.json             # writes <outdir>/trsi/run.yaml and exp.yaml; the start becomes the incumbent
<py> -m trsi adopt <run.yaml> --candidate <best_candidate>  # from round 2: the current best version's candidate as one of the start's seeds, not retrained
<py> -m trsi launch <run.yaml> --start --seed 0 --gpus 3    # or launch <run.yaml> <candidate.yaml> --seed 0 --gpus 3; returns at once
<py> -m trsi watch <run.yaml> --marker <marker_path>        # moves to the background, writes <marker>.pid; writes ok once all trials have ended
<py> -m trsi collect <run.yaml> [--gpu 3]                   # packs every finished trial as <outdir>/cand-<trial>/
<py> -m trsi next <run.yaml>                                # compares by rule, prints the next step
<py> -m trsi report <run.yaml> --search <outdir>/search --ids <outdir>/registered-<k>.json
```

- `run.yaml` is generated by `init --brief` from the brief and the task definition; where each key comes from is at the top of `trsi/config.py`; never write it by hand.
- Candidate directory: `model/` (the chosen checkpoint; for LoRA, the model ojh merged; hard links, or copies across disks, so the candidate stays complete when `ojh cleanup` deletes the run's files), `predict` (`model.predict`, runs `predict.py` with the `<py>` written at training time), `predict.json` (scoring settings), `predictions_val.jsonl` (with the task's `policy` section: the rows with the decision policy applied, beside `predictions_val_raw.jsonl`, `policy.json` and `policy_apply.py`, which `predict.py` applies; models.md), `manifest.json` (the metrics of `rsi score`, `config`, `config_request`, `cost`, `code_version`). Checked with `rsi validate candidate` after packing.
- The scoring GPU: `collect` chooses it before packing each trial (`harness.pick_scoring_card`): in `resources.gpu_ids`, with not a single compute process listed by nvidia-smi (anyone's counts), free GPU memory no less than max(the machine config's `min_free_mib`, the model directory's size × 1.5), and, when `lock_dir` is configured, ojh's GPU lock `<lock_dir>/<uuid>.lock` available at once (held while scoring); `min_free_mib` and `lock_dir` come from `machine.gpu` of the trial's `run.lock.yaml`, else from run.yaml's `ojh.machine`. `--gpu <gpu>` (must be in `gpu_ids`) is used when it qualifies; otherwise collect prints why and takes the qualifying GPU with the most free memory; a GPU nvidia-smi shows as `[N/A]` does not qualify. `CUDA_VISIBLE_DEVICES` is set by uuid. When `model.predict` runs out of GPU memory (OOM), it scores once more on another qualifying GPU. The GPU used is recorded in the trial result and in the `result` event's `scoring_card`.
- A trial whose packing failed (a prediction, scoring or validation error) is recorded as unpacked: not compared and not retrained; its checkpoint stays in ojh's run directory and the next `collect` packs it again; `next` prints the packing error; `launch ... --force` records it as failed (GPU hours still count) and retrains it.
- When ojh's tmux session is gone, there has been no `DONE` for over 2 minutes (`harness.LOST_GRACE_S`) and no file in the run directory has changed, the trial failed (ended without DONE); `watch`, `collect` and the train command all count it so; after kill-session ojh writes `DONE` only within about 30 seconds, and during that time the trial still counts as running.
- The candidate manifest's `config` is ojh's `run.lock.yaml`, but the paths that point to labelled data (`data`'s `train`, `select`, `select_index`, `prepared_dir`, `source`; `eval_sets.*`'s `train_labels`, `reference`; `split_file`) are written as null; their sha256 are kept.
- Checkpoints no longer needed are deleted by `collect` by itself (without asking the user): once a trial is packed and its candidate has passed `rsi validate`, it runs `ojh cleanup --confirm` on the trial's ojh run (deleting the unchosen checkpoints and merged models that `deletion_candidates.json` lists; ojh keeps the chosen one); for a trial whose `DONE` says failed, it deletes all checkpoint directories in the run (`ckpt-step*`, `adapter-step*`, `merged-step*`, `final`, `adapter`, `*.partial`; ojh lists none for a failed run). A rerun trial keeps its replaced runs in its record's `old_run_dirs`, and once their `DONE` appears they are deleted whole the same way (`launch --force` deletes right after the launch succeeds). Unpacked trials, running ones and those without `DONE` are not touched. Deletions are recorded as `cleanup` events; the paths ojh left are recorded in the trial record's `cleanup_left` and reported once; what could not be deleted is only printed, and the next `collect` deletes it.
- `--ids`: the `registered-<k>.json` the engine writes after registering the first part of the hand-back (`[{"dir", "kind", "id"}]`, only candidates are taken); several may be given; JSON `{"cand-<trial>": "candidate-NNNN"}` is also accepted.
- On SIGTERM (`rsi stop`), `watch` reads this run's trials again, ends every ojh training still running with `tmux kill-session` (those launched after watch started included), and writes the marker as failed; a background watch that hits any other error also writes a failed marker, with the traceback in `<marker>.log`.
- Progress file: after `launch`, at every poll of `watch`, and before and after `collect` packs each trial, run.yaml's `framework.progress_file` is rewritten (the brief's `progress_file`, by default `<outdir>/progress.json`; format in the framework's `docs/interfaces/progress.md`): the trainings running and their GPUs, `configs`, `GPU hours`, `incumbent`, and one row per trial (the step count of its state read from `train-s<seed>.log` in ojh's run directory). A write failure only goes to stderr (the background watch's log) and never makes a command fail.

Train command (the contract is in `SKILL.md`):

```bash
PYTHONPATH=<repo> <repo>/modules/train/train --task <task.json> --trainset <dir> --trainset-id <id> \
  --valset <dir> --valset-id <id> --config <config.json> --config-source default --round 1 \
  --producer train --out <candidate_dir> --marker <marker_path>
```

- `config.json`: `{"base", "recipe", "overrides", "seed", "max_steps"}`, all optional, defaulting to joey-1, the directory's default recipe (`recipe: null`), no changes and seed 0.
- It can be started with any `python3`: it reruns itself in `resources.onejev.python`.
- GPUs: set with `RSI_TRAIN_GPUS=0` (or `0,1`); when not set, the GPUs in `resources.gpu_ids` that nvidia-smi shows as free are taken.
- Machine config: `resources.onejev.machine`, else `RSI_ONEJEV_MACHINE`.
- `RSI_TRAIN_POLL`: the interval in seconds for checking whether the ojh training has ended, default 60.
- `--producer` accepts only `train` (the data module does not train; `data` in old candidates still reads as before). The ojh experiment name is `train-<the first 8 characters of the sha256 of --out's real path>`, so the same config trained to different `--out`s does not clash.
- After training it first waits for ojh's tmux session to exit (at most 2 minutes: ojh still holds the GPU when it writes `DONE`), then packs; scoring prefers the GPU it trained on, checked as `collect --gpu` does, and switches GPUs when that one does not qualify; when no GPU qualifies, it tries again every 30 seconds (or every `RSI_TRAIN_POLL` when that is shorter), for at most 10 minutes. On SIGTERM or SIGINT it ends its own ojh training (`tmux kill-session`) and writes the marker as failed; a signal received while ojh is starting the training is handled once the tmux session is recorded.
- At the start, at every check of the ojh training, at the end and on failure it rewrites `<the parent directory of --out>/progress.json`; when two train commands run in the same directory at once, the later write overwrites the earlier.
- Every candidate's `code_version.sha256` is the sha256 of this `train` file. When `trsi/` changes, change `TRSI_VERSION` in `train` and `__version__` in `trsi/__init__.py` together (the train command refuses to run when they differ); when ojh or OneJev changes version, change `OJH_COMMIT` and `ONEJEV_COMMIT`.

## Tests

CPU only; no ojh and no GPU needed:

```bash
python3 -m venv /tmp/trsi-venv && /tmp/trsi-venv/bin/pip install pyyaml pytest
cd <repo>/modules/train && /tmp/trsi-venv/bin/python -m pytest -q
```

On the A100, with ojh's `.venv` (which has pyyaml; after `pip install pytest`): `cd <repo>/modules/train && <py> -m pytest -q`.

The tests use `tests/fake_ojh.py` in place of `ojh`, `tests/fake_scorer.py` in place of `harness.score.score_checkpoint` (packing writes it into `predict.json` only when `TRSI_TEST_SCORER` is set, and `model.predict` looks only at `predict.json`), and the fake `nvidia-smi` and `tmux` that `tests/conftest.py` writes; the `rsi score` and `rsi validate` that packing uses are the framework's real commands. The framework's own tests (`<repo>/tests`) do not include these.
