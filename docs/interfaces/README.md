# Artifact interfaces: overview

This directory describes the format of each artifact kind in the framework. Files that machines read and write are always JSON / JSONL (UTF-8); validation is done by the `rsi` commands, and the documents only describe the meaning.

## Artifact kinds

| Kind | Producer | Users | Description |
|---|---|---|---|
| `task` | orchestrator (with the user) | the top level and all modules | Task definition, see task.md |
| `trainset` | data | train; data builds the next version on the previous one; judge uses it only for the baselines | Trainset, full rows, see datasets.md |
| `valset` | data (round 1, or an approved refresh) | data, train; deliver gets only `inputs.jsonl` | Valset, full rows plus input rows without labels; choosing checkpoints, setting thresholds, comparing configs; data uses it for the dedup check; deliver reproduces the candidate's predictions on its inputs |
| `evalset` | produced by data, handed to judge to keep | judge (with labels); deliver gets only `inputs.jsonl` | Evalset, inputs and labels stored apart; used only for promotion decisions |
| `testset` | produced by data, handed to judge to keep | judge (with labels); deliver gets only `inputs.jsonl`, and so does compare at the milestone (to score the baselines); every use needs a `use_testset` approval | Sealed testset, inputs and labels stored apart; used only for milestone releases |
| `corpus` | data (round 1's `sets` dispatch) | data only: the `sets` dispatch that produced it and `refresh` dispatches | The preprocessing script and all the labelled rows it writes, handed over into the workspace at registration; the four sets and the pool all come from it, see datasets.md |
| `pool` | data (round 1) | data: when it builds the trainsets of later rounds | Training pool: all full rows of the training period, from which the trainsets are drawn, see datasets.md |
| `candidate` | train (an old workspace also has some produced by tune, read only) | train, deliver, judge | Candidate model, see models.md |
| `search` | train (an old workspace also has some produced by tune, read only) | train (search history), judge, orchestrator | Config search record |
| `baseline` | compare | compare (reused or re-predicted in later comparisons), judge (its `predictions_eval`; at the milestone its `predictions_test`) | A baseline model trained on the trainset (or on none) with its predictions on the valset and the evalset inputs, and on the testset inputs at the milestone (a `use_testset` approval); its valset metrics recomputed at registration, see compare.md |
| `comparison` | compare | judge, orchestrator, the report and pages | The chosen candidate against each baseline on the valset: metrics, paired deltas and the strongest baseline, recomputed at registration; a report, never a gate, see compare.md |
| `delivery` | deliver | judge, the user | Model package, prediction interface, predictions on the evalset, latency and cost, see delivery.md |
| `feedback` | deliver, or the interface once connected to a real system | data, judge | Execution feedback |
| `evaluation` | judge | orchestrator | Evaluation report and promotion recommendation, see evaluation.md |
| `plan` | judge | orchestrator, then dispatched to the module the plan names; judge checks it against the next round's evaluation to see whether it worked | Next round's plan |
| `record` | every module (the top level may write some too) | judge, orchestrator; each module reads its own in later rounds | Each module's experiment records, the material of Meta-RSI, see record.md |

## Common manifest fields

Except for `task`, every artifact directory holds a `manifest.json` with the common fields below. The directory of a `task` holds `task.json` and `TASK.md`, and `task.json` itself is its manifest.

| Field | Type | Description |
|---|---|---|
| `kind` | enum | Artifact kind |
| `id` | string or null | null before registration; at registration it becomes `<kind>-<four-digit number>`, e.g. `trainset-0003` |
| `task_id`, `task_version` | string, int >= 1 | The task definition it follows |
| `round` | int >= 1 | The round it was produced in; at registration it must equal STATE's current round |
| `producer` | enum | `orchestrator`, `data`, `train`, `compare`, `deliver`, `judge` (an old workspace also has `tune`, read only, under ② Train & tune) |
| `created_at` | string | ISO 8601 time |
| `parents` | list of strings | Ids of registered artifacts it depends on; may be empty |
| `files` | object | Logical name to `{"path": relative path, "rows": int, optional}` |
| `external` | object, optional | Logical name to `{"location": "host:/absolute/path" or "/absolute/path", "sha256": string}`, for large files kept elsewhere |
| `notes` | string, optional | Notes |

Relative paths in `files` and other fields must not, as written, leave the artifact directory (absolute paths and paths that leave it through `..` are refused). An artifact directory may contain symlinks: registration copies what a link points to, and the registered copy holds the actual file. The exceptions are evalset, testset and corpus, whose whole directory is moved at registration: a symlink pointing outside the directory or a file with a second name (link count above 1) would leave that content (the labels) where it was, so validation and registration both refuse them. A link to a file that does not exist is refused for every kind.

Values of type number in a manifest or task definition must be finite: `NaN` and `Infinity`, though JSON can read them in, are always refused by validation.

## Workspace layout

- `<ws>/STATE.json`: the current state.
- `<ws>/registry.jsonl`: the registry, append only.
- `<ws>/dashboard.html`: the overview of the progress pages, generated by `rsi` for people to read only; it is no artifact and takes no part in any check or decision.
- `<ws>/dashboard/`: the secondary progress pages `rounds.html` (rounds and artifacts), `events.html` (the event log) and `report.html` (the run report), generated with the overview and of the same nature.
- `<ws>/STATE.lock`: the lock file held while reading, changing and writing STATE.json (an empty file, `flock`), see engine.md.
- `<ws>/work/run.jsonl`: the engine's run log, one event per line (actions, dispatches, returns, decisions, answers, and so on), append only; `<ws>/work/run.lock` is the lock of a running `rsi run` (it holds the process id).
- `<ws>/work/round-<n>/<module>[-k]/`: the output directory of a dispatch: `brief.md`, `brief.json`, `prompt-<n>.md`, `agent-<n>.log`, `feedback-<n>.md`, `history-<n>.md`, `handback.json`, and the artifact subdirectories the module writes; `<ws>/work/decisions/<n>-<decision point>/` holds a decision agent's files. They are no artifacts; registration copies the artifact subdirectories into `artifacts/`. The engine's records (dispatches, decisions, the waiting question) are in STATE's `dispatches`, `decisions` and `waiting`, see engine.md.
- `<ws>/work/round-<n>/<module>[-k]/progress.json`: the optional progress file a module writes (the brief's `progress_file`); the progress pages and the status line of `rsi run` show it as it is; format in progress.md. No artifact; it takes no part in checks or decisions.
- `<ws>/work/round-<n>/<module>[-k]/spend.jsonl`: the spend file a module appends to (the brief's `spend_file`): GPU hours, data LLM dollars and annotation items it caused outside its own agent session, which the engine counts in the spend ledger (STATE's `budget_used`); format in spend.md. No artifact.
- `<ws>/artifacts/<id>/`: artifacts copied from their source directories at registration; evalset, testset and corpus are moved here, and their source directories no longer exist after registration (see "Hand-over" in datasets.md).

Once the run has stopped (STATE's `stopped` is written, by `rsi next` hitting a stop condition or by `rsi stop` at the user's request), `rsi promote`, `rsi close-round`, `rsi next`, `rsi stop`, registrations other than `record` and approvals other than `delete` are all refused; the `delete` approval (cleanup) and `record` registrations (last experiment records) are still allowed. `rsi report` changes no state (it rewrites the progress pages) and can produce the run report before and after the stop.

## Decision rows (JSONL)

Dataset rows are OneJev rows (onejev-harness's `onejev_rows` adapter reads them directly, and so does the candidate's prediction entry), with no format conversion. Decision types are `binary`, `choice` and `score`. Labels are always strings; the labels of `score` are `"0"` to `"k-1"`.

| Row type | Used in | Fields |
|---|---|---|
| Full row | `rows.jsonl` of trainset, valset, pool and corpus | `id`; `state` (formatted by the task definition's `input.format`: `messages`, `fields` or `object`); `question`; `target`; `meta.state_id`, `meta.cluster`, `meta.source`, `meta.label_source` (`rule`, `model_judge`, `human`, `outcome`), `meta.n_obs` (int >= 1), `meta.extra` (optional object), and `meta` may hold other fields; `split` (non-empty string) |
| Input row | `inputs.jsonl` of evalset, testset and valset | A full row with only the top-level keys `id`, `state`, `question`, `meta`, `split` (so no `target`), and `meta` with only `state_id`, `cluster`, `source`, `n_obs` (`input_row_of` in `rsi/rows.py`); any other key at the top level or in `meta` is refused at registration (input rows go to deliver and to the candidate's prediction entry, without labels) |
| Label row | `labels.jsonl` of evalset and testset | `id`; `target`; `meta.label_source`, the only key of `meta` |
| Prediction row | A candidate's output on input rows | `id`; `probs` (label to probability, covering all labels, summing to 1 within `abs(sum - 1) <= 1e-4`, each probability in [0, 1]); `decision` (one label); `uncertainty` (number or null) |
| Execution feedback row | feedback | `id`; `state` (this call's input, i.e. the input row's `state`; an old workspace's trace calls it `input`, which is still read); `output` (`probs`, `decision`); `model` (version or candidate id); `latency_ms`; `timestamp`; `outcome` (a label or null) |

`state` is the model's input, formatted as the task definition's `input` section says: with `format: messages` it is `{"messages": [{"role": str, "content": str}, ...]}`, non-empty; with `format: fields` it is `{"fields": {...}}`, whose keys must cover the task definition's `input.fields`; with `format: object` it is any non-empty JSON object (e.g. `{"entity": ..., "summary": {...}, "transactions": [...]}`), whose inner structure is not checked and which goes to the model as it is (OneJev accepts any non-empty state object).

`question` is the OneJev question `{"type", "instructions", "criteria"}`; on every row it must equal exactly the question derived from the task definition's `decision` (`question_of` in `rsi/rows.py`), checked row by row at registration and refused on a mismatch:

| The task's `decision` | `question.type` | `question.criteria` |
|---|---|---|
| `binary` with labels `yes` and `no` | `noul` | `{"true": meaning of yes, "false": meaning of no}` (`label_meanings`; OneJev's noul labels are always `yes` and `no`) |
| `choice`, or a `binary` whose labels are not `yes` and `no` | `choice` | `label_meanings`: label to meaning |
| `score` | `score` | `[ meaning of "0", meaning of "1", ... ]`, in label order |

`question.instructions` equals `decision.instructions`. So the question the model sees is the decision the task definition writes, and data modules cannot each reword it. The one exception is a trainset's prompt-variant rows (`meta.extra.variant`): each carries the question of a spec the trainset's manifest lists, a checked paraphrase of the instructions and the label meanings with the same type and labels, next to its seed row (datasets.md, "Prompt-variant rows"); every other row and every other set keeps the task's question.

`target` is always a probability map, checked by OneJev's rules (onejev-harness `validate_rows`): a non-empty object whose keys are all the question's labels (i.e. the task's labels), a label left out having probability 0; every value in [0, 1]; summing to 1 within `abs(sum - 1) <= 1e-6`. A hard label is written as one label with 1 and the rest 0, e.g. `{"yes": 1.0, "no": 0.0}` for a binary task; a soft label from several observations or several annotators gives each label's share. `target` is never a bare string (`"yes"` is refused). The "hard label" the metrics use is the label with the highest probability in `target` (see "Metric definitions" in task.md).

Two fields of `meta` are set by the data module for the task, for the modules and the registration checks to use:

- `meta.state_id`: the group the row belongs to (OneJev's state id, by which onejev-harness groups); the `group_prior` baseline groups by it, and judge reports per-group results by it.
- `meta.cluster`: the id of the near-duplicate cluster. Samples from the same template, rewrites and near duplicates belong to one cluster; a whole cluster may fall into one set only, and the same cluster cannot appear in two sets.

**Input rows carry no labels.** An input row has only the five top-level keys `id`, `state`, `question`, `meta`, `split`, and `meta` only the four keys `state_id`, `cluster`, `source`, `n_obs`: the extra top-level keys of a full row and the other fields of `meta` may derive from the labels (e.g. the agreement rate `f`, `mode_score`, `mode_idx` of a OneJev export, from whose `f` the `target` is computed; `label_source` counts too, since how a label was obtained may correlate with the label itself, e.g. illicit from human and licit from rule), so they do not go into input rows (`input_row_of` keeps only these keys), `label_source` is only in full rows and label rows, and registration refuses an input row with any other top-level key or `meta` key, naming the key in the error. A label row likewise has only the three top-level keys `id`, `target`, `meta`. No field of an input row (`state`, `question`, `split` and these four `meta` fields) may carry a value derived from the labels. When adapting a OneJev export, drop fields such as `f`, `mode_score` and `mode_idx` from the input rows; full rows may keep them (train sees the labels anyway).

`split` is OneJev's split name, written by the data module (e.g. `train`, `val`, `eval`, `test`). Note that onejev-harness refuses to read training and validation rows whose `split` starts with `test` (unless `--allow-test` is given), so do not use such names for the rows of trainset, valset and pool: registration refuses such rows (the same rule as onejev-harness: after stripping surrounding whitespace and lowering the case it starts with `test`, see datasets.md).

**Rows of old workspaces.** Rows registered before the change to OneJev rows use `input` and `meta.group`, which are read as `state` and `meta.state_id` (`state_of` and `group_of` in `rsi/rows.py`), so metrics, baselines and overlap checks of old workspaces work as before; newly registered rows must have the format above.

Row `id`s are unique across trainset, valset, evalset, testset and pool: the same id cannot appear in two sets (except trainset and pool: a trainset is drawn from the pool). When a trainset, valset, evalset, testset or pool is registered, it is refused if it shares any id, any input (by the fingerprint of `state`, ignoring differences in case and whitespace) or any cluster with the current sets of the other kinds; when the split rule says `unit: state_id`, `state_id` is compared too; a new evalset also may not share ids or inputs with any evalset registered before; every row of a valset, evalset, testset or pool registered after a corpus must be a row of the corpus with the same id, the same input, the same `state_id` and cluster and the same label (so the first corpus must be registered before them). Details in "Overlap checks" in datasets.md.

The label rows of evalset and testset go only to judge. deliver gets only `inputs.jsonl` (input rows: the evalset's, the valset's of the candidate sent to evaluation, and the testset's when approved) and never touches labels; train gets only the trainset and the valset, never the evalset or the testset. The check functions are in `rsi/rows.py`.
