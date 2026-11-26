# Data artifacts: trainset, valset, evalset, testset, corpus, pool

All six kinds of data artifacts are produced by the data module, and their directories hold a `manifest.json` with the common fields of README.md. Validation is done by `validate_artifact` in `rsi/artifacts.py`: first the manifest is checked against the common fields plus the kind's extra fields (no fields that are not listed), then that each file in `files` exists, that `rows` matches the file's count of non-empty lines and the sha256 of a local `external`, and last the kind's checks inside the directory. A trainset, valset, evalset, testset, pool or corpus must have at least one row (empty rows or inputs are refused): an empty set, once registered, can be neither trained on nor evaluated. The row formats are in "Decision rows" of README.md: every row is a OneJev row (`id`, `state`, `question`, `target`, `meta`, `split`), `question` must equal the question derived from the task's `decision` (a trainset's prompt-variant rows: their spec's question, see "Prompt-variant rows"), and `target` is checked by OneJev's rules; the row checks are in `rsi/rows.py`.

## Who sees the labels

Quoting section 7 of the design document:

| Set | Produced by | Who sees the labels | Use | Changes |
|---|---|---|---|---|
| `trainset` | data | train; compare (its baselines are fitted on it); judge uses it only for the baselines | Training | May have a new version every round |
| `valset` | data, round 1 | data, train, compare | Choosing checkpoints, setting thresholds, comparing configs, dedup checks | Fixed after round 1; a refresh needs the user's approval |
| `evalset` | data, round 1, handed to judge once produced | judge only | Promotion decisions | Fixed; the number of uses is capped, and a refresh after the cap needs the user's approval |
| `testset` | data, round 1, handed to judge once produced | judge only, and every use needs the user's approval (`use_testset`) | Final check at a milestone release | Never changes |
| `corpus` | data, round 1's `sets` dispatch | data only: the `sets` dispatch that produced it (the brief gives it again when continued) and `refresh` dispatches | All the labelled rows and the preprocessing script that writes them; the four sets and the pool are cut from it, and refresh draws new evalsets from it | Registered once in round 1; registering again needs the user's approval |
| `pool` | data, round 1 | data (`sets` and `trainset` dispatches; dgen scores, diagnoses and fits its filters on it) | Training pool: all rows of the training period, from which later rounds' trainsets are drawn | Registered once in round 1; registering again needs the user's approval |

The evalset and the testset keep inputs and labels in two files: deliver gets only `inputs.jsonl` (the testset's only when approved), and the label file goes only to judge; the valset has an extra `inputs.jsonl` without labels, with which deliver reproduces the candidate's valset predictions; train never gets the evalset or the testset. ③ Compare gets the evalset's `inputs.jsonl` only (its baselines predict it for the judge) and, at the milestone (`milestone_compare`, after the user's yes), the testset's `inputs.jsonl` only. judge gets the `rows.jsonl` of the trainset of the candidate sent to evaluation, only to compute the baselines on the evalset (the baselines come from the trainset's prior, see "Metric definitions" in task.md).

## `trainset`

Trainset, full rows. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"trainset"` | Artifact kind |
| `label_method` | string | How the labels were obtained, e.g. rule, judge, human |
| `parent_trainset` | string or null | The id of the previous trainset; null for the first version |
| `changes` | string | What changed from the previous version |
| `quality` | object | Results of the data quality checks |
| `prompt_variants` | list of specs, optional | The prompt-variant specs of the trainset's variant rows (see "Prompt-variant rows"); absent or empty when it has none |

The sub-fields of `quality`: `passed` (bool); `checks` (a list, each item `name`, `value`, `threshold`, `passed`). Checks inside the directory: every row is a valid full row, and `split` does not start with `test` (onejev-harness's rule: a split that starts with `test` after stripping surrounding whitespace and lowering the case is not used for training and validation); every prompt-variant row is a rewrite of its seed under a spec of `prompt_variants`, and every spec is valid (`check_variant_rows` in `rsi/variants.py`); `quality.passed` must equal the AND of all `checks[].passed`.

A trainset with `quality.passed` false is registered as usual and stays in the registry as evidence, but never becomes the current trainset (STATE's `current.trainset` is unchanged), so no candidate can refer to it. `rsi register` prints the quality gate's result after the id, e.g. `trainset-0004 (quality gate: failed (dup_rate); never current)`. Registration itself adds the failed check `generated_matches_heldout` when generated or prompt-variant rows match held-out inputs (see "Generated rows and the held-out sets" under "Overlap checks").

Required files: `rows` (full rows, JSONL, with the `rows` count written) and `quality_report` (the quality report, in a format the data module chooses).

```
trainset-0003/
  manifest.json
  rows.jsonl
  quality_report.json
```

**Generated rows.** A trainset of a later round may hold rows the data module generated with dgen (`modules/data/dgen`; built into the trainset by `dset grow --with <arm>/verified.jsonl`, one `--with` per arm). They are full rows like any other, so the checks above apply to them as they are:

| Field | A generated row (`smote`, `borderline`, and the LLM methods) |
|---|---|
| `id` | `syn<sha6>_<method>_<n>`: the first 6 hex digits of the recipe's sha12, the method, the row's number (six digits) |
| `state` | its seed's state edited through the task's features adapter (`apply`, then `repair`; a row that `check` refuses is dropped; `repair` and `check` run the task's `features.constraints` first); the state paths of the task's `input.verbatim` (a row's own request text) are never changed |
| `question` | the task's question (`question_of`) |
| `target` | its seed's: the label follows the seed |
| `meta` | `state_id`, `cluster` and `label_source` of the seed; `source` `"dgen:<method>"`; `n_obs` 1; `extra` `{"seed_id", "partner_id" (smote and borderline: the real row it was interpolated with), "recipe": <sha12>}` |
| `split` | `"train"` |

Seeds are always real rows: never prompt variants, nor rows generated in an earlier round (`meta.source` starting with `dgen:`, which a later trainset holds). Those generated rows are never seeds, SMOTE anchors or partners, nor part of the feature space's scaler, the CLLM reference or the distance reference; they still count for the dedup and copy checks, and pool scoring leaves them out. The `hard` method adds real failure rows of the pool themselves, unchanged (their own id, `meta` and `source`). The LLM methods (`llm_imitate`, `llm_pattern`: `meta.source` `"dgen:llm_imitate"` or `"dgen:llm_pattern"`) seed every call with a real failure of the pool, never let the LLM set a label, and keep a row only when an independent reviewer, which sees nothing but the input description and the records' states, scores its realism at least the threshold calibrated on 60 real pool rows; each of their calls is a `claude -p` process with the task's agent command, logged in `<out>/llm/<sha12>/` (`gen/` and `review/`, beside the arms: `<out>` is `dgen generate`'s `--out`) and billed as an `api_usd` line of the dispatch's spend file (spend.md). A data call never gets MCP servers, user settings or ambient context, whatever the task's agent command says: `--tools ""` (no built-in tool), `--strict-mcp-config` with every `--mcp-config` of the task's `runner.claude` command and args removed (no MCP server), `--setting-sources project` in place of any sources the task's `runner.claude` names (no user or local settings), and a new empty temporary directory outside the workspace as its working directory, removed after the call (no project settings or CLAUDE.md); a call that is stopped or runs past its time limit has its whole process group killed. An LLM recipe runs as a long job (`dgen generate --marker`). Each arm keeps `provenance.jsonl` (seed, partner, weakness type of every row; for the LLM methods also the reviewer's score and reason and the round), `flags.jsonl` (every check per row) and `quality_summary.json` beside `verified.jsonl`. dgen reads only its brief's inputs: the trainset, the pool, the valset's `inputs.jsonl` (for the `leak_val` check: never its labels) and the best version's candidate; its diagnosis takes each error from the candidate's own decision on pool rows, and its CLLM reference and filters are fitted on the trainset and the pool only.

**Prompt-variant rows.** A trainset may also hold prompt variants: rows that reword the prompt of a seed row of the same trainset and lay out its state differently, without changing what either says, so the model learns the decision rather than one phrasing of it (the data module's lever; decision-model inputs stay in English). Only a trainset holds them: a valset, evalset, testset, pool or corpus row with `meta.extra.variant` is refused. A spec (`rsi/variants.py`; the items of the manifest's `prompt_variants`) is

```json
{"id": "p1a2b3c4d5e", "instructions": "Does this address take part in illicit activity?",
 "criteria": null, "rename": {"summary.txs_sent": "transactions sent"}, "tables": ["transactions"]}
```

| Key | Value |
|---|---|
| `id` | a string matching `^[a-z0-9][a-z0-9-]*$`, unique in the manifest |
| `instructions` | a paraphrase that replaces the question's `instructions` (the task's `decision.instructions`), or null |
| `criteria` | the question's criteria with paraphrased meanings, or null: the same keys (`true`/`false` of a noul question, the labels of a choice question) or the same length (a score question); the labels never change |
| `rename` | `{<path>: <new key>}` or null: each path names one object key of the state in the seed's naming, keys joined by dots and `[]` for every item of a list (`transactions[].amount`); the new names of keys under one parent are unique |
| `tables` | `[<path>]` or null: each named list of objects with the same keys becomes `{"columns": [key, ...], "rows": [[value, ...], ...]}` (columns in the first item's key order, after the renames inside the items) |

A spec is refused when a renamed or tabled path is, lies under or holds a path of the task's `input.verbatim` (a row's own request text stays where it is, as it is), or moves a key the input format needs (`messages`, `messages[].role`, `messages[].content`; `fields` and each task field), and when it changes nothing. A variant row (`apply_spec`) is its seed with the id `<seed id>~<spec id>`, the spec's question (`question(spec, task)`: the task's question with the spec's instructions and criteria), the state rewritten by the spec, and `meta.extra` with `variant` (the spec's id) and `variant_of` (the seed's id) added; its target, `meta` and `split` are the seed's. Every transform is invertible (`invert_state`). Registration checks each variant row, naming the row:

- its seed `variant_of` is a row of the same trainset that is no prompt variant;
- its spec is in `prompt_variants`;
- its `question` equals the spec's question;
- inverting its state gives the seed's state exactly (and the spec applied to the seed's state gives its state);
- its `target`, `meta.state_id`, `meta.cluster` and `meta.label_source` equal the seed's;
- the text at every `input.verbatim` path (with `[]` for every item) is byte-identical to the seed's.

Every other row keeps the rule that `question` equals the task's question. onejev-harness's `onejev_rows` adapter passes each row as it is, and OneJev renders each row's own `question`, so a variant's instructions and criteria reach training together with its state. The data module writes specs with `dgen paraphrase` (reviewed paraphrases) or by hand, adds variant rows for a seeded share of a trainset's rows with `dgen variants`, and writes the trainset with `dset check --write --variants` (`dset grow` keeps the current trainset's variants and their specs). ③ Compare, few-shot examples and every dgen step use only the rows that are no prompt variants (`features.table.real_rows`), and dset's quality checks are computed on those rows too: a variant adds no data.

## `valset`

Valset, with no extra fields (`kind` always `"valset"`). Required files: `rows` (full rows) and `inputs` (input rows, without labels), both with the `rows` count written. The data module writes `inputs.jsonl` by keeping only the top-level keys `id`, `state`, `question`, `meta`, `split` of each row of `rows.jsonl` (so no `target`), and only `state_id`, `cluster`, `source`, `n_obs` of `meta` (`input_row_of` in `rsi/rows.py`), and it goes to deliver: deliver runs the delivery package on it once, and registration compares the result with the candidate's `predictions_val` row by row (see delivery.md); deliver never sees the valset's labels. Checks inside the directory: valid full rows, `split` not starting with `test` (as for trainset); valid input rows (no top-level key other than these five, not even `target`; no `meta` key other than these four); `inputs` and `rows` have the same set of ids, and each input row equals exactly the row of `rows.jsonl` with the same id processed the same way (only the five top-level keys, only the four `meta` keys), so the rows of `rows.jsonl` and their `meta` may hold other fields too.

```
valset-0001/
  manifest.json
  rows.jsonl
  inputs.jsonl
```

## `evalset` and `testset`

The evalset and the sealed testset, with the same format and no extra fields (`kind` always `"evalset"` and `"testset"`). Required files: `inputs` (input rows: full rows keeping only `id`, `state`, `question`, `meta`, `split`, with only `state_id`, `cluster`, `source`, `n_obs` of `meta`) and `labels` (label rows: `{"id", "target", "meta": {"label_source"}}`, `meta` with `label_source` as its only key), both with the `rows` count written. Checks inside the directory: the input rows and the label rows are each valid (an input row has no top-level key other than these five, not even `target`, and no `meta` key other than these four; a label row has only `id`, `target`, `meta` at the top level, must have `meta.label_source`, and no other `meta` key); both sides have the same set of ids.

**Input rows carry no labels.** Input rows go to deliver and to the candidate's prediction entry, and their `meta` keeps only the four keys above, because the other fields of `meta` may derive from the labels (e.g. the agreement rate `f`, `mode_score`, `mode_idx` of a OneJev export; `label_source` too, since how a label was obtained may correlate with the label, so it is only in full rows and label rows); registration refuses any other key. The fields kept may not carry values derived from the labels either. When adapting a OneJev export, drop such fields from the input rows.

```
evalset-0002/
  manifest.json
  inputs.jsonl
  labels.jsonl
```

**Hand-over: moved into the workspace at registration.** Artifacts of the other kinds are copied at registration, and their source directories stay where they are; evalset, testset and corpus (see below) are handed over at registration: once every check passes, the source directory is moved into the workspace (`<ws>/artifacts/<id>/`), the original directory no longer exists, and so the output directory of the module that produced it no longer holds these labels. When registration refuses, nothing is moved: the source directory stays as it is, to be fixed and registered again. When the source directory cannot be moved (e.g. no write permission on its parent directory) registration refuses, and again nothing changes. The registry's `source` still records the original path, and a directory created again later at the same path cannot be registered (one source path registers only once). A symlink in the directory pointing outside it, or a file with a link count above 1, would leave the labels where they were, so validation and registration both refuse them; write the files themselves into the directory. When registration is killed midway (after the source directory was moved aside and before the registry was written), the source directory is in the hidden sibling directory `.<name>.rsi-handed-over-*`: rename it back by hand and register again.

So the data module produces the evalset, the sealed testset and the corpus directly in the output directory the brief gives, keeps no copies elsewhere and keeps no labels in intermediate files; after handing back it no longer reads them. Known limitation: the code can only move the source directory away; processes on the same machine can still read the label files in the workspace, and the isolation between modules rests on convention (the brief gives modules no workspace path); the code does not prevent it.

## `corpus`

The preprocessing script the data agent writes in round 1's `sets` dispatch, and all the labelled full rows it writes from the raw data (the task's `data.sources`, see task.md). The four sets and the pool are all cut from these rows. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"corpus"` | Artifact kind |
| `command` | string, non-empty | The command that runs the script, e.g. `python3 preprocess.py`; not run at registration; a rerun audit is done by the operator with it |
| `sources` | list of strings | The raw data paths the script read; may not be empty when the task's `data.sources` gives paths, and each item must equal one of their `path`s or lie under one |
| `label_method` | string | How the labels were obtained |

Required files: `rows` (all full rows, JSONL, with the `rows` count written) and `script` (the preprocessing script); it may also have `split_report` (the split summary `dset` writes). Checks inside the directory: no entry starting with `.dset-` (left by a `dset` that stopped midway; delete it and run again); every row is a valid full row (no repeated ids, `question` equal to the task's question), `split` is not checked (the corpus is not trained on, and rows may keep the split names of the raw data); when the task's split rule is complete (`data.split` in task.md), every row has the rule's `key`, and with `by: time` its value is a number (not a string, bool or null); when the task writes `data.rows.expect`, the row count and the row count of each of its hard labels are equal one by one (hard labels in "Metric definitions" of task.md); `sources` as above. One message per kind of problem, giving the row count and the first id (`corpus_errors` in `rsi/datasplit.py`). Like the other files, the script's sha256 is recorded in the registry's `files` at registration.

Like evalset and testset, the corpus's whole directory is **handed over** into the workspace (see "Hand-over" above), and later dispatches no longer read it (except refresh: new evalsets can only be drawn from it). Once registered it is the current corpus (STATE's `current.corpus`); registering another while there is a current corpus needs a `refresh_sets` approval. A corpus vouches only for the valset, evalset, testset and pool registered after it, which must all come from it (see "Overlap checks"), so the first corpus must be registered before these sets: if any of these kinds already has a current one, it is refused, the error saying to register the corpus first (the data module's hand-back is already in this order). A replacement corpus (when there is a current corpus) is not compared with the sets already current. The corpus takes no part in the overlap comparisons between sets (it is the whole). When the corpus itself is registered it is read into memory whole (the row checks); later, registering a set reads it row by row and keeps only the rows that set uses.

```
corpus-0001/
  manifest.json
  rows.jsonl
  preprocess.py
  split_report.json
```

## `pool`

Training pool: all labelled full rows of the training period (the rule's trainset range), registered once in round 1. Later rounds' trainsets are drawn from it, and data's `trainset` dispatches get the pool, not the raw data or the corpus. No extra fields (`kind` always `"pool"`). Required files: `rows` (full rows, with the `rows` count written). Checks inside the directory as for trainset: every row is a valid full row, and `split` does not start with `test` (it feeds training). Registration copies it; once registered it is the current pool (STATE's `current.pool`), and registering another while there is a current pool needs a `refresh_sets` approval.

```
pool-0001/
  manifest.json
  rows.jsonl
```

An old workspace's STATE `current` has no `corpus` and `pool` keys; they read as none (null), and registration works as usual.

## Overlap checks

`meta.state_id` and `meta.cluster` of the rows are set by the data module for the task:

- `meta.state_id`: the group the row belongs to (OneJev's state id); the `group_prior` baseline groups by it, and judge reports per-group results by it; the rows of old workspaces use `meta.group`, which is read as `meta.state_id`.
- `meta.cluster`: the id of the near-duplicate cluster. Samples from the same template, rewrites and near duplicates belong to one cluster; the split puts a whole cluster into one set, and the same cluster cannot appear in two sets.

Row `id`s are unique across trainset, valset, evalset, testset and pool (except trainset and pool: a trainset is drawn from the pool). These sets may not repeat each other, which is checked at registration: when a trainset, valset, evalset, testset or pool is registered, it is compared with the current artifacts of each of the other kinds (trainset and pool are not compared with each other; the corpus is the whole and takes no part), and it is refused if it shares any id, any input or any cluster, the error giving the three overlap counts. A new evalset is also compared with every evalset registered before, and may not share ids or inputs with them (a refreshed evalset must be new samples).

**The same `state_id`.** The same `state_id` (e.g. rows of the same address at different time steps) may appear in several sets, and registration does not refuse it (`dset` reports how many each pair of sets shares). Only when the task's split rule says `unit: state_id` (`data.split` in task.md) does each comparison above also compare `state_id`, refusing any shared one, the error also giving `state_ids N`.

**From the corpus.** Every row of a valset, evalset, testset or pool registered after a corpus (with a current corpus at registration) must be the corpus's row with the same id: the same input fingerprint; the same `meta.state_id` and `meta.cluster` (the overlap comparisons above rest on these two fields); for labelled rows (`rows.jsonl` of valset and pool, `labels.jsonl` of evalset and testset) also the same `target` (a label not written counts as 0). A mismatch is refused, one message per kind of problem, giving the row count, the first id and the corpus's id. So the four sets and the pool can all be traced back to a registered preprocessing script; sets registered before a corpus cannot, so the first corpus must be registered before them (see the `corpus` section above). This check reads the corpus, so it comes last, only after the checks before it have passed. Trainsets are not checked (later rounds' trainsets may add rows written from feedback). When the task writes raw data (`data.sources` gives paths, or `data.rows` is written), the sets must come from a corpus: without a current corpus, registering a valset, evalset, testset or pool is always refused, the error saying to register the corpus (the preprocessing script and all the labelled rows) first; `rsi register` and the engine are the same. A task that writes no raw data (sets registered by hand, no `data` section, an old workspace's string `sources`) needs no corpus, and without a corpus none of this is checked.

**Generated rows and the held-out sets.** The data module never sees the evalset's or the testset's inputs, so the framework checks a new trainset's generated rows against them: every row whose `meta.source` starts with `dgen:` and every prompt-variant row (taken as the state it inverts to under its spec) whose state has the coarse fingerprint of an input of the current evalset or testset. The coarse fingerprint (`coarse_fingerprint` in `rsi/rows.py`) is the input fingerprint of the state with every number rounded to 4 significant digits (and written as a float). Registration runs this check on the staged copy, before its content hash is taken (`Workspace._generated_heldout_matches`); when any row matches, the registered copy's `quality.checks` get `{"name": "generated_matches_heldout", "value": <the number of rows>, "threshold": 0, "passed": false}` and `quality.passed` is false, so the trainset is registered as evidence and never becomes current, and `rsi register` prints `(quality gate: failed (generated_matches_heldout); never current)`. The `quality_gate_failed` feedback to the data module names at most 20 of those trainset row ids and never the held-out inputs. Real rows keep the rules above: an identical input is refused by the overlap check, a shared `state_id` is only reported, and a near-identical real row passes.

`overlap(rows_a, rows_b)` in `rsi/rows.py` returns the three overlap counts `ids`, `inputs`, `clusters`. `inputs` counts by input fingerprint (the row's `state`; `input` for rows of old workspaces), ignoring differences in case and whitespace, so the same input with a new id, changed case or changed spaces is still found. `shared_state_ids(rows_a, rows_b)` returns how many `state_id`s two groups of rows share (by `meta.group` for rows of old workspaces).

## Large files kept elsewhere

`external` records large files not put in the directory. A `location` of `/absolute/path` counts as local: the file must exist and its sha256 must match; `host:/absolute/path` counts as remote and is not checked.
