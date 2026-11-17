# OneJev Training Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `onejev-harness`, a standalone repo (OneJev pinned + agent skills + a thin compatibility layer) with which an agent can turn a dataset into a selected checkpoint, per-row predictions, locked thresholds, a baseline comparison and a complete run record, on any single node.

**Architecture:** Approach C from the design: `prepare` and `resolve` deterministically turn data + machine config + base catalog + recipe into `run.lock.yaml`; `launch` executes only that lock in tmux and returns at once; `finalize` selects, rescores, evaluates against baselines and registers. OneJev itself is never edited; the harness wraps it with its own training entry that only replaces `train.sft.run_evals`.

**Tech Stack:** Python 3.11, PyYAML, numpy/scipy/scikit-learn (CPU parts); torch 2.8.0+cu128, transformers 5.19.0, peft 0.21.2, flash-linear-attention 0.5.2 and patched OneJev (GPU parts); tmux; pytest.

**Spec:** `ONEJEV_HARNESS_DESIGN.md` (moved to `docs/design.md` in Task 1). Section numbers below (“design §N”) refer to it.

## Global Constraints

- Repo root: `/Users/xtra_mac/research/DynamicServe/onejev-harness/`, its own git repo; package `harness`, CLI `ojh` (entry point `harness.cli:main`).
- OneJev: `https://github.com/OmniJev/OneJev.git` @ `81ce62f1597c91e46767d4d02d6ac2e18534fe94` + `serve.patch` from HF `baselabs/joey-1` @ `9efbe485efbed9598dfca5294cbbe8da00e21908`. The patch sha256 and the patched-file sha256s are copied from `dynamicserve-agreement-router/configs/models.json` (`joey.sha256["serve.patch"]`, `onejev.patched_sha256`).
- GPU env pins are copied verbatim from `dynamicserve-agreement-router/training/{setup.sh,constraints.txt}` (torch==2.8.0, torchvision==0.23.0 from the cu128 index; transformers 5.19.0; peft 0.21.2; flash-linear-attention 0.5.2). CPU extras: `pyyaml`, `numpy`, `scipy`, `scikit-learn==1.9.1` (the version that produced the v1 baselines).
- Every module under `harness/` except `train_entry.py`, `score.py`, `merge.py` must import without torch, so the CPU test suite runs on a Mac.
- Fixed locations (pre-flight ruling): OneJev is checked out at `<repo>/vendor/OneJev` and the patch is stored at `<repo>/vendor/serve.patch`; `vendor/` is git-ignored. `prepare` writes to `<out_root>/<exp.name>/prepared/`, which `resolve` reads; base weights live at `<model_dir>/<base name>`.
- No personal path, account name or machine value (`/Users/`, `/home/`, `xbr`, `/shared/ssd`, `gpu_locks`, a hard-coded GPU index) anywhere in `harness/`, `adapters/`, `skills/`, `catalog/`. They may appear only in `examples/` (as placeholders) and `docs/`.
- Rows whose split is `test*` are never read without `--allow-test`; report halves are never read without `--allow-report`.
- Nothing is deleted without `ojh cleanup --confirm`. Training mode (`full` | `lora`) comes only from the recipe or a recorded override; the harness never switches it.
- `ojh launch` returns immediately (session, log path, done marker).
- Recipes must state `mode` explicitly (OneJev's own CLI default is LoRA).
- Code and comments in English; README and skills in Chinese.
- Tests that need the v1 data read the directory from env var `HARNESS_V1_DIR` (the local `regen_20261111/`) and are skipped when it is unset.
- Run tests with `.venv/bin/python -m pytest -q` from the repo root.

## Review Focus

1. A lock directory that is not group-writable on a multi-user machine silently gives no mutual exclusion → `gpu.acquire` warns (Task 6 test `test_warns_when_lock_dir_not_group_writable`).
2. Eval rows with no usable label (truncated / unlabeled questions, `target` missing) → dropped and counted, never scored as 0 (Task 5 test `test_unlabeled_rows_dropped_and_counted`).
3. A training process that dies (OOM, killed tmux) → done marker still written with the non-zero exit code and `status` reports `failed` (Task 9 test `test_runner_writes_done_on_child_failure`).
4. The same id in train and an eval set, or in two eval sets → `prepare` refuses (Task 3 test `test_rejects_id_overlap`).
5. Many tied scores → thresholds follow the v1 tie rules deterministically (Task 4 test `test_fit_threshold_tie_rule`, `test_max_share_ties_interpolated`).

---

### Task 1: Repo skeleton and config loading

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md` (stub), `docs/design.md` (copy of `ONEJEV_HARNESS_DESIGN.md`), `docs/plan.md` (copy of this file)
- Create: `harness/__init__.py`, `harness/config.py`
- Test: `tests/test_config.py`, `tests/test_no_hardcoded_values.py`

**Interfaces:**
- Produces:
  - `get_field(row: dict, dotted: str) -> Any` (raises `KeyError`).
  - `MachineConfig` dataclass: `python: str`, `model_dir: str`, `out_root: str`, `registry: str | None`, `gpu_include: list[int] | None`, `gpu_exclude: list[int]`, `min_free_mib: int = 0`, `lock_dir: str | None = None`, `order: str = "free_desc"`, `wait_timeout_s: int = 3600`, `gpu_source: str = "scan"`, `forbidden_write_roots: list[str]`, `env: dict[str, str]`. Property `registry_path` defaults to `<out_root>/registry.jsonl`.
  - `ExperimentConfig` dataclass mirroring design §7: `name`, `base`, `recipe`, `data` (`adapter`, `train`, `train_sha256`, `eval: dict[str, EvalSetSpec]`, `split_file`), `subset: SubsetSpec | None`, `fields` (`label`, `group`, `consistency`, `text`), `evaluator` (`type` default `binary_decision`, `budgets`, `gate` default true, `path` for plugins), `seeds: list[int]`, `gpus: int | None`, `merge: bool = False`, `overrides: dict`. `EvalSetSpec(train_labels: str, reference: str | None)`.
  - `load_machine(path) -> MachineConfig`, `load_experiment(path) -> ExperimentConfig` (raise `ConfigError` naming the missing key).

- [ ] **Step 1:** `git init` the repo; create `pyproject.toml` (requires-python >=3.11; deps pyyaml, numpy, scipy, scikit-learn==1.9.1; extra `dev` = pytest; extra `gpu` = the pinned GPU packages; script `ojh = harness.cli:main`). Create `.venv` with Python ≥3.11 and `pip install -e '.[dev]'`. Copy the design doc and this plan into `docs/`.
- [ ] **Step 2: Write failing tests.** `test_get_field_dotted` (`get_field({"meta": {"f": .5}}, "meta.f") == .5`; missing → `KeyError`); `test_machine_defaults` (minimal yaml → `gpu_source == "scan"`, `lock_dir is None`, `wait_timeout_s == 3600`, `registry_path` ends with `registry.jsonl`); `test_experiment_eval_sets` (`reference` defaults to `None`, `budgets == [0.10, 0.15]` parsed); `test_missing_key_named` (`ConfigError` message contains `data.train`); `test_no_hardcoded_values` (scan the four directories in Global Constraints for the forbidden substrings; must find none).
- [ ] **Step 3:** Run `.venv/bin/python -m pytest tests/test_config.py tests/test_no_hardcoded_values.py -q` → fails (module missing).
- [ ] **Step 4:** Implement `harness/config.py` with `yaml.safe_load` and dataclasses.
- [ ] **Step 5:** Run the same command → PASS.
- [ ] **Step 6:** Commit `feat: repo skeleton and config loading`.

### Task 2: OneJev pin, setup and base catalog

**Files:**
- Create: `onejev.lock`, `setup.sh`, `constraints.txt` (copied from `dynamicserve-agreement-router/training/constraints.txt`), `harness/onejev.py`, `harness/catalog.py`, `harness/fetch_base.py`
- Create: `catalog/bases/{joey-1,elephant-1,mammoth-1,onejev-qwen3.5-0.8b,onejev-qwen3.5-4b,onejev-qwen3.5-9b,onejev-27b}.yaml`
- Test: `tests/test_onejev.py`, `tests/test_catalog.py`

**Interfaces:**
- Produces:
  - `onejev.lock` (yaml): `repo`, `revision`, `patch: {hf_repo, revision, file, sha256}`, `patched_sha256: {relpath: sha256}`.
  - `ensure_onejev(dest: Path, lock: dict, patch_path: Path) -> Path`: clone if absent, checkout detached revision, apply patch only if `git apply --reverse --check` fails, then verify every `patched_sha256` entry; raises `OneJevMismatch`. Ported from `dynamicserve-agreement-router/training/fetch.py`.
  - Catalog entry fields: `name`, `hf_repo`, `revision` (40-hex), `params_b: float`, `multimodal: bool`, `attention: causal|noncausal`, `readout: slot_lm|head`, `sha256: {file: sha}`, `published_recipe: str` (free text).
  - `load_base(name) -> dict`; `fetch_base(name, model_dir: Path) -> Path` (`huggingface_hub.snapshot_download` with `allow_patterns=list(sha256)`, then verify).
- Values: joey-1 and elephant-1 repo/revision/sha256 from `configs/models.json`; joey-1 `params_b: 2`, elephant-1 `9`, mammoth-1 `27`, all `multimodal: true`, `attention: noncausal`, `readout: slot_lm`. mammoth-1 revision and weight sha256s: look up at implementation time from `https://huggingface.co/api/models/baselabs/mammoth-1` and its LFS metadata. OneJev Qwen entries: model ids from OneJev `train/configs/*.yaml`, `attention: causal`, revision looked up the same way.

- [ ] **Step 1: Failing tests.** `test_patch_applied_once` (temp git repo + small patch: two calls leave the patch applied once and verify passes); `test_patched_sha_mismatch_raises`; `test_catalog_entries_valid` (every yaml loads, revision matches `^[0-9a-f]{40}$`, required keys present).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `harness/onejev.py`, `harness/catalog.py`, `harness/fetch_base.py`; write `onejev.lock` and catalog files; write `setup.sh` (create venv from the machine config's `python`, install `.[gpu]` with `-c constraints.txt`, then `ojh setup`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Additionally verify on the Mac (network, read-only): download `serve.patch` from elephant-1 and mammoth-1 and compare sha256 with joey-1's. Record the result in `onejev.lock` as a comment; if they differ, add per-base `patch` overrides to the catalog and say so in the commit message.
- [ ] **Step 6:** Commit `feat: pinned OneJev setup and base catalog`.

### Task 3: Data preparation

**Files:**
- Create: `adapters/__init__.py`, `adapters/onejev_rows.py`, `adapters/template.py`, `harness/prepare.py`
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `get_field` (Task 1).
- Produces:
  - `adapt(path: Path) -> Iterator[dict]` in each adapter module (OneJev row: `id`, `state`, `question{type,instructions,criteria}`, `target`, `meta`, `split`).
  - `validate_rows(rows: list[dict]) -> list[str]` (errors: duplicate id, target keys not ⊆ criteria, target not summing to 1 ± 1e-6, missing state).
  - `apply_subset(rows, spec: SubsetSpec) -> list[dict]`: equality filter on dotted fields, then `random.Random(seed).sample` of `n` over rows sorted by id.
  - `EvalParts` dataclass: `select_rows: list[dict]` (train-label version, merged), `select_index: dict[id, set_name]`, `reference: dict[id, dict]`, `report_ids: dict[set_name, list[str]]`, `dropped: dict[str, list[str]]`.
  - `build_eval_parts(exp, *, allow_test=False) -> EvalParts`: join `train_labels` and `reference` by id (intersection; the rest go to `dropped`); `split_file` assignment `{set: {"assignment": {id: "select"|"report"}}}`; ids missing from the assignment go to `dropped["unassigned"]`; rows with `split` starting with `test` raise `TestSplitRefused` unless `allow_test`.
  - `prepare(exp, out_dir: Path, *, allow_test=False) -> dict` writing `train.jsonl`, `select.jsonl`, `select_index.json`, `prepared.json` (`train_sha256_source`, `train_sha256_prepared`, `subset`, counts, `dropped`). Raises `ShaMismatch` when the source train sha256 differs from `data.train_sha256`; raises `IdOverlap` when an id is in train and an eval set, or in two eval sets.

- [ ] **Step 1: Failing tests.** `test_validate_rows_errors`; `test_subset_deterministic` (same seed → same ids; different seed → different); `test_split_assignment` (select/report partition); `test_test_split_refused`; `test_rejects_id_overlap`; `test_sha_mismatch_stops`; `test_v1_select_matches` (skip without `HARNESS_V1_DIR`): building from `full/export_majority75/binary/val_{id,catood}.jsonl` + `train/splits.json` yields exactly the 380 ids of `train/data/select.jsonl`, and the source train sha256 starts with `fe83846076cb6c72`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run (with `HARNESS_V1_DIR=/Users/xtra_mac/research/DynamicServe/regen_20261111`) → PASS.
- [ ] **Step 5:** Commit `feat: data preparation, subsets and eval parts`.

### Task 4: Metrics and baselines (ported from v1)

**Files:**
- Create: `harness/evaluator/__init__.py`, `harness/evaluator/metrics.py`, `harness/evaluator/text.py`, `harness/baselines/__init__.py`, `harness/baselines/tfidf.py`
- Test: `tests/test_metrics.py`, `tests/test_tfidf.py`

**Interfaces:**
- Produces (each ported verbatim from the named v1 source, with a provenance comment):
  - `auc(scores, rates) -> float | None` and `within_group_auc(groups, scores, rates) -> float | None` ← `data_pipeline_v2/sanity_check/common.py` (`auc`, `within_category_auc`, re-keyed to take a group list).
  - `aggregate(probabilities, agreements, threshold) -> dict` and `fit_threshold(probabilities, agreements) -> dict` ← `evaluation/binary_metrics.py`.
  - `max_share_at_budget(scores, incons, budget) -> float`, `budget_threshold(scores, incons, budget) -> dict`, `at_threshold(scores, labels, incons, t) -> dict` ← `regen_20261111/train/gate.py`.
  - `ece(probs, rates, bins=10) -> float` (equal-width bins on predicted probability, weighted |mean p − mean rate|).
  - `text_of(row, mode: str) -> str`: `"messages"` = `"\n".join(f"{m['role']}: {m['content']}")` over `state.messages` (same as v1 `request_text`); `"state_json"` = `json.dumps(state, ensure_ascii=False, sort_keys=True)`.
  - `TextBaseline(train: list[dict])` with `.strength` and `.predict(rows) -> list[float]`; `group_prior(train, rows) -> list[float]` ← `data_pipeline_v2/sanity_check/step1_data/cheap_baseline.py` (`Features`, `logistic`, `choose_strength` with `STRENGTHS = (1., 10., 100.)`, `SEED = 0`, `< 60` examples → `10.`). Examples are dicts `{id, text, rate, group}`.

- [ ] **Step 1: Failing tests.** `test_auc_soft_rates` (hand-computed 3-item case); `test_within_group_prior_is_half` (constant score per group → 0.5); `test_fit_threshold_tie_rule` (equal Jaccard → nearest to 0.5, then larger); `test_max_share_ties_interpolated` (tied group straddling the budget gives a fractional share); `test_budget_threshold_none_feasible` (returns `threshold == 1.1`, `small_share == 0.0`); `test_text_of_messages`; `test_tfidf_small_train_uses_10`; `test_tfidf_matches_v1` (skip without `HARNESS_V1_DIR`: fit on `full/export_majority75/binary/train.jsonl`, `strength == 1.0`, predictions on each select half match the `tfidf` column of `full/check_majority75_preds_{val_id,val_catood}.jsonl` within 1e-6).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Port the functions.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: v1 metrics and TF-IDF baseline`.

### Task 5: `binary_decision` evaluator

**Files:**
- Create: `harness/evaluator/binary_decision.py`
- Test: `tests/test_binary_decision.py`

**Interfaces:**
- Consumes: `EvalParts` (Task 3), metrics and baselines (Task 4), `get_field`.
- Produces:
  - `evaluate(preds: dict[str, float], parts: EvalParts, train_rows: list[dict], fields, evaluator_cfg) -> dict` returning:
    - `sets`: `{set_name | "pooled_select": {questions, positive_rate, groups, unseen_groups, router, tfidf, group_prior, all_positive, oracle}}`. Each entry is a summary dict with `auc`, `within_group_auc`, `jaccard_at_own_best_threshold`, `ece` (router only), and `share_at_budget_<pct>` per budget.
    - `thresholds`: `best_jaccard` plus `budget_<pct>` per budget, fitted on pooled select with router scores, each with `per_set` from `at_threshold`.
    - `gate`: `{set: {auc: bool, within_group_auc: bool}}` and `passed: bool`.
    - `tfidf_strength`, `dropped`.
  - Labels come from the `reference` rows (`fields.label`); `incons = 1 − fields.consistency`. Group-dependent entries are omitted when `fields.group` is unset; budget entries and `oracle` are omitted when `fields.consistency` is unset. Rows without a usable label are dropped and counted.
  - Key names may differ from v1 `gate_elephant.json`; the test maps them.

- [ ] **Step 1: Failing tests.**
  - `test_no_consistency_no_budgets`; `test_no_group_no_prior`; `test_gate_fails_when_router_worse`; `test_unlabeled_rows_dropped_and_counted`.
  - `test_matches_v1_gate` (skip without `HARNESS_V1_DIR`): with preds from `train/predictions_select_elephant_{val_id,val_catood}.jsonl` (`p_agree`), every router, tfidf, category_prior and oracle_f number in `train/gate_elephant.json` matches within 1e-3.
  - Same test, thresholds: within 1e-5 of 10%: 0.75491, 15%: 0.43782, best Jaccard: 0.33459 (values recorded in `loop/STATE.json`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: binary decision evaluator with baselines and gate`.

### Task 6: GPU selection and locks

**Files:**
- Create: `harness/gpu.py`
- Test: `tests/test_gpu.py`

**Interfaces:**
- Produces:
  - `Gpu(index: int, uuid: str, name: str, free_mib: int)`; `parse_nvidia_smi(text: str) -> list[Gpu]` for `--query-gpu=index,uuid,name,memory.free --format=csv,noheader,nounits`.
  - `GpuLease`: `.uuids: list[str]`, `.names: list[str]`, `.release()`; holds one open `fcntl.flock` fd per card for its lifetime.
  - `acquire(machine: MachineConfig, n: int, *, query=_nvidia_smi, sleep=time.sleep, clock=time.monotonic) -> GpuLease`:
    - filter by include/exclude and `min_free_mib`, order by free memory descending;
    - `flock(LOCK_EX | LOCK_NB)` on `<lock_dir>/<uuid>.lock`, then re-query that card and keep it only if still ≥ `min_free_mib`;
    - retry every 30 s until `wait_timeout_s` → `GpuWaitTimeout`;
    - `gpu_source == "preassigned"` → read `CUDA_VISIBLE_DEVICES`, no scan, no lock.
  - Warns (`warnings.warn`) when `lock_dir` is set but not group-writable.
- Port the selection order and recheck from `regen_20261111/train/command_elephant.sh`.

- [ ] **Step 1: Failing tests.** `test_exclude_and_min_free`; `test_order_free_desc`; `test_locked_card_skipped` (hold a flock on one uuid in the test; acquire returns the others); `test_recheck_after_lock`; `test_timeout_raises` (fake clock); `test_preassigned_bypasses_scan`; `test_warns_when_lock_dir_not_group_writable`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: GPU selection with configurable locks`.

### Task 7: Resolve → `run.lock.yaml`

**Files:**
- Create: `harness/resolve.py`, `examples/recipes/{elephant_full_fsdp,joey_full_1gpu,joey_lora_1gpu,mammoth_lora_ddp}.yaml`
- Test: `tests/test_resolve.py`

**Interfaces:**
- Consumes: configs (Task 1), catalog (Task 2), `prepare` output (Task 3), `parse_nvidia_smi`/`Gpu` (Task 6).
- Produces:
  - Recipe yaml: `mode: full|lora` (required), `gpus: int`, `onejev: {flag_name: value}` (snake_case OneJev flags), `lora: {r, alpha, dropout, target_modules, autocast}` (lora only).
  - `estimate_gib(params_b, mode, n_gpus, fsdp) -> float`:
    - full: `params_b × 16` GB, divided by `n_gpus` under FSDP;
    - lora: `params_b × 2 × 1.15` GB per card.
    - The constants are module-level `FULL_BYTES_PER_PARAM = 16`, `LORA_BYTES_PER_PARAM = 2`, `LORA_OVERHEAD = 1.15`, marked initial.
  - `parallel_for(mode, params_b, n_gpus, requested_fsdp: bool | None) -> str` in `{"single", "ddp", "fsdp"}`:
    - full and `params_b >= 9` → `fsdp`;
    - lora with FSDP requested → raises `Infeasible`.
  - `parse_plan_only(stdout: str) -> int` (regex `(\d+) steps for (\d+) ranks`; requires `PLAN_OK`); `eval_interval(steps) = ceil(steps / 8)`.
  - `build_command(lock: dict, seed: int) -> list[str]`:
    - `python -m torch.distributed.run --standalone --nproc_per_node N -m harness.train_entry ...` when N > 1, otherwise `python -m harness.train_entry ...`;
    - flags: recipe flags + `--full-finetune` for full + `--fsdp` when parallel is fsdp + LoRA flags + `--attention/--readout` from the catalog + `--multimodal` when the base is multimodal + `--eval select=<prepared>/select.jsonl --eval-at-start --eval-every K --save-every K --seed S --name <exp>-s<S> --runs-dir <run_dir>/runs --train <prepared>/train.jsonl --model <model_dir>/<base>`.
  - `resolve(exp, machine, *, query=..., plan_runner=...) -> Path`:
    - writes `<out_root>/<exp>/<run_id>/run.lock.yaml` with every field of design §8 “lock 文件”;
    - applies `exp.overrides` last and records each under `overrides` with its reason (an override without `reason` → `ConfigError`);
    - raises `Infeasible(alternatives: list[str])` when the estimate exceeds the free memory of the best `n` candidate cards.
    - `run_id` = UTC timestamp + short hash of the lock body.
- Example recipe values: design §8 table (elephant: 4 GPUs FSDP, lr 3.5e-6, 1 epoch, brier 1, warmup .03, wd 0, max-grad-norm 1, max-len 16384, batch-tokens 16384, max-rows 8, accum-rows 32, bf16, plus `split: train` and `log_every: 1` as in the v1 command; joey full: 1 GPU, lr 1e-5, brier 0, max-rows 1, accum-rows 32; LoRA: r 32, alpha 64, dropout 0.05; mammoth LoRA adds autocast).

- [ ] **Step 1: Failing tests.**
  - `test_elephant_command_matches_v1` (skip without `HARNESS_V1_DIR`): the flag set equals the training flags of `train/command_elephant.sh` (entry module, paths and the eval file name excluded) given `plan_runner` → 176 steps, so `--eval-every 22`.
  - `test_mode_required`; `test_lora_fsdp_infeasible`; `test_mammoth_full_infeasible_lists_lora` (4 cards × 80 GB free → `Infeasible`, alternatives mention LoRA); `test_override_needs_reason`; `test_lock_has_all_fields`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement and write the four example recipes.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: resolve experiments into run locks`.

### Task 8: Training entry (eval hook)

**Files:**
- Create: `harness/train_entry.py`, `harness/eval_records.py`
- Test: `tests/test_eval_records.py`

**Interfaces:**
- Consumes: patched OneJev on `PYTHONPATH`: `train.sft.run_evals(model, bundle, eval_sets, args, rank, world, step, logf, is_main)`; `train.sft.collect_mm(bundle, rows, max_len, batch_tokens, max_rows, device, sync_batches=None)`; `train.common.collect_slot_logits(bundle, rows, max_len=, batch_tokens=, max_rows=, device=, quiet=, sync_batches=)`. Scored items have `.idx`, `.labels`, `.logits`, `.target`.
- Produces:
  - `eval_records.records(scored, rows) -> list[dict]` with `{"id", "labels", "probs", "target"}` (softmax of logits in float64).
  - `eval_records.nll(records) -> float` = mean of −Σ tᵢ·log max(pᵢ, 1e-12).
  - `train_entry.main()`: replaces `sft.run_evals` with `harness_run_evals`, then calls `sft.main()`. `harness_run_evals` does the following:
    - scores each eval set with `max_rows=1`, using `collect_mm` when `bundle.multimodal`, otherwise `collect_slot_logits`, with an all-reduce-MAX `sync_batches` when `world > 1`;
    - gathers records to rank 0;
    - writes `<runs_dir>/<name>/preds/<set>-step<step>.jsonl` (sorted by id);
    - appends `{"event": "harness_eval", "step", "set", "n", "nll", "path"}` to `logf`.
  - When `HARNESS_AUDIT=1` and full fine-tune: port the frozen-weight and step-2 update checks from `dynamicserve-agreement-router/training/publisher_native_train.py` (`shard_model`, `optimizer_step`, `save_weights` patches), writing to `<runs_dir>/<name>/audits/`.
- For yes/no targets `nll` must equal the wrapper's binary formula.

- [ ] **Step 1: Failing tests (CPU).** `test_records_softmax` (fake scored objects); `test_nll_matches_v1_binary` (yes/no case equals `−[y·log p + (1−y)·log(1−p)]` averaged); `test_records_sorted_and_complete`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `eval_records.py` and `train_entry.py` (torch and OneJev imported only inside `main`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: training entry with per-row eval dumps`.

### Task 9: Launch, runner and status

**Files:**
- Create: `harness/launch.py`, `harness/runner.py`, `harness/status.py`
- Test: `tests/test_runner.py`, `tests/test_status.py`

**Interfaces:**
- Consumes: lock (Task 7), `acquire` (Task 6), `build_command` (Task 7).
- Produces:
  - `launch(lock_path) -> dict(session, log, done, run_dir)`: session name `ojh-<user>-<exp>-<run_id[:8]>` with `getpass.getuser()`; starts `tmux new-session -d -s <session> '<python> -m harness.runner <lock_path> >> <run_dir>/runner.log 2>&1'`; returns without waiting. The tmux call is injectable for tests.
  - `runner.run(lock_path, *, acquire=..., popen=..., finalize=...) -> int`:
    - acquire the cards;
    - export `CUDA_VISIBLE_DEVICES` and the machine `env`, plus `PYTHONPATH=<onejev_dir>:<repo>`, `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, `WANDB_MODE=disabled`, `OMP_NUM_THREADS=4`;
    - for each seed run `build_command(lock, seed)` and record start/end into `timing.json`;
    - write `gpus.json` (uuid, name);
    - call `finalize` if every seed exits 0;
    - always write `<run_dir>/DONE` = `{"exit_code", "end", "stage": "train"|"finalize"}` in a `finally` block;
    - release the lease.
  - `status(run_dir) -> dict(state, seed, step, total_steps, last_eval, eta_s, gpus)` from `log.jsonl` (training records carry `step`, `sec_per_step`; evals are `harness_eval`), `DONE` and the lock; `render(status) -> str` at most 6 lines.

- [ ] **Step 1: Failing tests.** `test_launch_returns_immediately` (fake tmux records the command; session contains the user name); `test_runner_writes_done_on_child_failure` (fake popen exit 3 → `DONE.exit_code == 3`, finalize not called); `test_runner_runs_seeds_in_order`; `test_status_running_and_failed` (synthetic logs).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: tmux launch, runner and status`.

### Task 10: Finalize, scoring, LoRA merge and registry

**Files:**
- Create: `harness/select.py`, `harness/score.py`, `harness/merge.py`, `harness/finalize.py`, `harness/registry.py`, `harness/plugins.py`
- Test: `tests/test_select.py`, `tests/test_registry.py`, `tests/test_finalize.py`

**Interfaces:**
- Consumes: `harness_eval` events (Task 8), `evaluate` (Task 5), `EvalParts` (Task 3).
- Produces:
  - `select_step(events, set_name="select") -> int`: minimum `nll`, tie → earlier step, step 0 excluded.
  - Checkpoint dir: `runs/<name>/ckpt-step<N>` for full, `runs/<name>/adapter-step<N>` for LoRA (OneJev naming).
  - `score.score_checkpoint(model_dir, rows, *, base_dir=None, multimodal, device="cuda:0") -> dict[id, dict]`:
    - loads with `common.load_bundle(path, device, "bfloat16", multimodal=...)` and attaches `qev.readout.Readout` as `train/eval_mm.py` does;
    - LoRA: loads the base, then `peft.PeftModel.from_pretrained`;
    - scores with `max_rows=1`;
    - returns `eval_records.records`.
  - `merge.merge_lora(base_dir, adapter_dir, out_dir) -> Path`: `merge_and_unload`, `save_pretrained`, copy tokenizer/processor and `readout.json` / `config.json` `modeling_overrides` from the adapter dir or base.
  - `finalize(run_dir, *, scorer=score.score_checkpoint, merger=merge.merge_lora) -> dict`, per seed:
    - select the step;
    - rescore select rows; if |NLL − training NLL| > 1e-3 → status `failed`;
    - if `merge`: merge, rescore, and compare per-row probabilities within 1e-3;
    - positive-class probability for `evaluate` = `probs[k]`, where `k` is the last dotted component of `fields.label` (`target.yes` → `yes`);
    - write `predictions_select.jsonl`, `thresholds.json`, `comparison.json`, `timing.json` (adds `gpu_hours = Σ train seconds × n_gpus / 3600`), `checkpoint.sha256` (every file in the selected dir; LoRA also records the base revision), `deletion_candidates.json` (unselected ckpt dirs and merged dirs with sizes), `RUN_CARD.md`;
    - append one registry record; with more than one seed, append a summary record with mean and std of each metric.
  - `registry.REQUIRED = ("exp", "run_id", "seed", "train_label_version", "reference_label_version", "data_sha256", "base", "base_revision", "recipe", "mode", "onejev", "gpu_name", "n_gpus", "gpu_hours", "selected_step", "checkpoint_sha256", "metrics", "baselines", "gate", "overrides", "conclusion")`.
  - `append(path, record)` holds `fcntl` on `<path>.lock` and sets `record["complete"]`; `table(path, include_incomplete=False) -> str` renders markdown. `conclusion` starts empty; the skill tells the agent to fill it with `ojh note RUN "<text>"`, which rewrites that run's record.
  - `plugins.load_plugin(path: str) -> ModuleType` (import by file path). When `evaluator.type == "plugin"`, `evaluator.path` names the plugin.
    - `finalize` calls the plugin's hooks where they exist (`validate`, `select_rule`, `calibrate`, `baselines`, `acceptance`; signatures as in design §10) and falls back to the defaults otherwise.
    - When the type is not `binary_decision`, `evaluate` is not called.

- [ ] **Step 1: Failing tests (CPU, fakes injected).** `test_select_step_rules`; `test_registry_incomplete_hidden`; `test_registry_concurrent_append` (two processes, no lost lines); `test_finalize_writes_all_files` (synthetic run dir, fake scorer returns the training-time probs); `test_finalize_nll_mismatch_fails`; `test_finalize_lora_merge_compared` (fake merger plus a scorer that diverges by 0.01 → failed); `test_multi_seed_summary`; `test_finalize_uses_plugin_hooks` (a tiny plugin file whose `select_rule` picks the last step and whose `acceptance` returns a marker, both reflected in the outputs).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: finalize, scoring, merge and registry`.

### Task 11: CLI, eval-report and cleanup

**Files:**
- Create: `harness/cli.py`, `harness/report_eval.py`, `harness/cleanup.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `ojh` subcommands `setup`, `fetch-base`, `prepare`, `resolve`, `launch`, `status`, `finalize`, `note`, `eval-report`, `table`, `cleanup`, with the flags of design §7 (`--machine`, `--allow-report`, `--allow-test`, `--confirm`). `prepare` also takes `--machine` (it needs `out_root`).
  - `report_eval.run(run_dir, *, allow_report: bool, scorer=...) -> dict` scores report ids with the selected checkpoint and the locked thresholds (no refit) and writes `report_eval.json`. Raises `ReportRefused` unless `allow_report`.
  - `cleanup.candidates(run_dir) -> list[dict]`; `cleanup.delete(run_dir, confirm: bool)` deletes only when `confirm`.

- [ ] **Step 1: Failing tests.** `test_report_refused_without_flag`; `test_cleanup_lists_but_keeps_without_confirm`; `test_cli_help_lists_all_subcommands`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement with `argparse` subparsers.
- [ ] **Step 4:** Run the full suite `.venv/bin/python -m pytest -q` → PASS.
- [ ] **Step 5:** Commit `feat: ojh command line`.

### Task 12: Skills, README and examples

**Files:**
- Create: `skills/training-with-onejev/SKILL.md`, `skills/adapting-data-for-onejev/SKILL.md`, `.claude/skills/` symlinks, `AGENTS.md`, `CLAUDE.md`, `README.md`, `plugins/README.md` (hook signatures and a minimal example plugin), `examples/machine.example.yaml`, `examples/dynamicserve_v1_repro.yaml`
- Test: `tests/test_examples.py`

**Interfaces:**
- Consumes: CLI (Task 11), recipes (Task 7).
- Content (Chinese):
  - Main skill: `description` only says when to use it (training a OneJev decision model from a dataset; LoRA or full; any GPU node). Body:
    - flow prepare → resolve → launch → status → finalize;
    - the three judgment points;
    - the ssh usage (`ssh <host> ojh …`);
    - approval rules (cleanup, report/test, pushing);
    - hand-back after launch (no polling);
    - full vs LoRA rule (switch only by override with reason; compare like with like);
    - failure table (OOM → lower `batch_tokens`, then `max_rows`; FLA error → `disable_fla_kernels` override; NCCL timeout; plan-only failure → adapter/data; GPU wait timeout; rescore NLL mismatch);
    - filling `conclusion` with `ojh note`.
  - Data skill: the row format, writing `adapters/<name>.py` from `template.py`, and self-testing with `ojh prepare`.
  - `AGENTS.md` / `CLAUDE.md`: one paragraph each pointing to `skills/`.
  - `examples/machine.example.yaml`: every key commented, placeholder paths.

- [ ] **Step 1: Failing test.** `test_examples_resolve` (machine example + v1 repro example + each recipe pass `resolve` with fake GPUs and fake plan output); `test_skill_symlinks_resolve`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Write the files.
- [ ] **Step 4:** Run → PASS; run `test_no_hardcoded_values` again.
- [ ] **Step 5:** Commit `docs: skills, README and examples`.

### Task 13: Acceptance on a GPU node (A1–A8)

**Ask the user before this task**: it uses the shared A100 and copies the repo there (rsync; pushing to GitHub needs separate approval). The machine config with the DynamicServe values (lock dir, GPU 3 excluded, ≥ 48 GB free, out_root on `/shared/ssd`) is written outside the repo, e.g. in `~/ojh/machine.yaml` on the A100.

**Files:**
- Create: `docs/acceptance.md` (results table A1–A8 with numbers, run ids and dates)

- [ ] **Step 1:** rsync the repo; run `ojh setup --machine M`; confirm the patched OneJev verifies.
- [ ] **Step 2 (A2):** Score the existing v1 `ckpt-step154` on the 380 select rows with `score.score_checkpoint`; the probabilities match `predictions_select_elephant_{val_id,val_catood}.jsonl` `p_agree` within 1e-4.
- [ ] **Step 3 (A3):** On the Mac, `HARNESS_V1_DIR=… pytest tests/test_binary_decision.py tests/test_tfidf.py -q` → PASS (already part of the suite; record it).
- [ ] **Step 4 (A1):** `ojh prepare` → `ojh resolve` → `ojh launch` with the v1 repro example and `elephant_full_fsdp`, seed 42; hand back. After DONE:
  - the selected step is 154 or 176;
  - select NLL 0.457 ± 0.01;
  - pooled AUC 0.856 ± 0.01;
  - within-group AUC 0.816 ± 0.01.
- [ ] **Step 5 (A4, A7):** `joey_full_1gpu` and `joey_lora_1gpu` (with `merge: true`) short runs with an override `max_steps: 40` (reason: acceptance smoke). Both records are complete; the A7 merge check passes.
- [ ] **Step 6 (A8):** `ojh resolve` with `mammoth_lora_ddp` on the cards available. It reports feasible/infeasible with the estimate, and plan-only runs when feasible.
- [ ] **Step 7 (A5):** Repeat A4 with a second machine config (different `out_root`, `registry`, `lock_dir`); `test_no_hardcoded_values` passes.
- [ ] **Step 8 (A6):** Dispatch a fresh subagent with only the repo path, the machine and data locations, and a one-line task naming the recipe. Record where it hesitated or erred; fix the skill and rerun once.
- [ ] **Step 9:** Fill `docs/acceptance.md`; if A1 fails, add “尚未与 v1 对齐” to the README and to every record via `ojh note`, then compare `train_entry` with `publisher_native_train.py`.
- [ ] **Step 10:** Commit `docs: acceptance results`.
