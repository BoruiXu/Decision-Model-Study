# Model artifacts: candidate, search

Both kinds are produced by the train module (`train`) (one dispatch is one search: it first hands back the candidates of its trials, and once they are registered hands back, in the same session, the search that compares them; an old workspace's candidates and searches with `producer` `tune` are read as usual), and their directories hold a `manifest.json` with the common fields of README.md. Validation is done by `validate_artifact` in `rsi/artifacts.py`: the manifest is checked against the common fields plus the kind's extra fields (no fields that are not listed). The format of prediction rows is in "Decision rows" of README.md. Ids here are only checked to be strings; whether they are registered and whether the predictions cover the valset are checked at registration.

## `candidate`

One candidate model from a training. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"candidate"` | Artifact kind |
| `trainset_id` | string | The trainset trained on |
| `valset_id` | string | The valset used for validation |
| `config` | object | The fully resolved training config: every key the training code actually used is written out, including keys left at their defaults, not only the changed ones |
| `config_source` | string | `"default"` (the train command's default config) or the id of a registered candidate or search (whose config it started from); an old workspace's `"tune"` means a tuning trial; not checked at registration |
| `config_request` | object, optional | The config this training asked for, by config path: `{"base", "recipe", "overrides", "seed", "max_steps"}`, all five keys written, null for what was not set; `overrides` is `{"onejev": {<flag>: value}, "lora": {<field>: value}}` (flat keys such as `{"onejev.lr": value}` work too). Required when the task has a `search_space`, and registration checks that it lies in the config space (rules in `search_space` of task.md) |
| `code_version` | object | The version of the code that trained this candidate, see below |
| `model` | object | The model's location and prediction entry, see below |
| `metrics_val` | object | metric name → number, the results on the valset; must hold the task's primary metric |
| `baselines_val` | object | baseline name → {metric name → number}, the baselines' results on the valset |
| `train_gate` | object | The train gate's result, see below |
| `cost` | object | Cost, see below |

The sub-fields of `model`: `location` (where the model files are, for people); `sha256` (string or null); `load` (string, loading instructions for people); `predict` (string, the prediction entry, see below). The sub-fields of `train_gate`: `passed` (bool); `detail` (string). The sub-fields of `cost`: `compute_hours` (number).

The sub-fields of `code_version`: `entry` (string, the name of the training code, e.g. the train command's path: `<repo root>/modules/train/train` by default, or the task definition's `resources.train_command` when given, which registration checks: the same string, or the same file once a relative path is resolved against the repository root and symlinks are resolved); `sha256` (string or null, the sha256 of the file `entry`). When `entry` is the absolute path of a file that exists on the machine running validation or registration, its sha256 is computed and compared with `sha256` (hex, case ignored), and a mismatch or a null `sha256` is refused; a `host:/path`, a relative path, a name or a path that does not exist on this machine is not checked. When training runs on another host, write `entry` as `host:/path`: otherwise, if the registering machine has another copy of the code (another checkout) at the same path, an honest candidate is refused too.

Only the one file `entry` has its sha256 computed; other code it imports or calls, and default config files it reads, are not included. So whenever the training code or its defaults change, the `entry` file itself must change too: either the train command is a self-contained file (code and defaults written in it), or the file embeds a version string that changes with the code (e.g. a git commit or a hash of the other code files), updated by the owner together with the code. Given that, the same `code_version.sha256` means the same training code; and with `config` being the fully resolved config, these two items tell whether the "default config" is the same between candidates.

**The prediction entry `model.predict`.** A candidate must run directly, so that downstream (delivery) never reimplements the scoring from a description in words. `model.predict` is the relative path of an executable file in the candidate directory, which as written may not leave the candidate directory (it may be a symlink pointing elsewhere; registration copies the file it points to), and the file exists and is executable. It is always called as:

```
<candidate dir>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>
```

- `--inputs`: input rows, JSONL, i.e. rows like those of `inputs.jsonl` (OneJev rows without `target`, `meta` keeping only `state_id`, `cluster`, `source`, `n_obs`: `{"id", "state", "question", "meta", "split"}`, see "Decision rows" in README.md); the entry reads only `id`, `state` and `question` and relies on no other field; the rows have no `target`.
- `--out`: writes prediction rows, JSONL, one `{"id", "probs", "decision", "uncertainty"}` per input row, in the same format as `predictions_val`.
- Exit code 0 on success, non-zero on failure.
- The model files the entry reads are in the candidate directory; large files that do not fit can only be referred to through entries of the common field `external` (with sha256), never through other paths on the training machine.

The candidate's own `predictions_val` should be exactly what this entry produces on the valset's `inputs.jsonl`; at delivery, deliver runs it again through the packaged interface, and registration checks that the two agree (see delivery.md).

**The decision policy.** When the task has a `policy` section (see task.md), the train command fits a decision policy on the candidate's valset predictions and the candidate carries it; model.predict applies it, so `predictions_val`, a delivery's reproduction and the evaluation need nothing else. The candidate directory then holds:

- `predictions_val_raw.jsonl` (logical file `predictions_val_raw`, optional): model.predict's rows before the policy;
- `policy.json` (logical file `policy`, optional): `{"version": 1, "objective", "accepted", "reason", "labels", "calibration": {"method", "params"}, "decision": {"rule", "threshold", "positive"}, "select": {"metric", "delta", "ci_low", "ci_high", "n_fit", "n_select"}, "settings_sha256"}`. `labels` are the task's labels in order (an argmax tie goes to the earlier); `threshold` is null for "never positive" (and outside a binary task, where `positive` is null too); `select` holds the chosen pair's delta and interval against the identity policy on the select part (null numbers for the identity policy); `settings_sha256` is the sha256 of the canonical JSON of the effective settings (`policy_settings`). The identity policy (`none` + `argmax`, `accepted` false) gives the model's own most probable label, unchanged probabilities;
- `policy_report.json` (logical file `policy_report`, optional): the fit's report: the split (`split`, `n_fit`, `n_select`), the identity policy's primary metric on the select part (`primary`), and per pair of a calibration and a rule (`pairs`) its parameters, threshold, delta and interval, improvement and its lower bound, primary metric, `accepted`, and `rejected` (why) when refused;
- `policy_apply.py`: a copy of `modules/common/dpolicy/apply.py` (standard library only, no import of the repository), which `predict.py` loads by path to apply `policy.json` to each row;
- `predictions_val.jsonl`: the rows with the policy applied, byte for byte what model.predict writes.

`metrics_val` and the train gate are computed on `predictions_val`, the rows with the policy applied. Without a policy section none of these files is there and the candidate is as before.

Required file: `predictions_val`. Checks inside the directory: valid prediction rows (fields, repeated ids, labels, probabilities); `metrics_val` holds the primary metric; `model.predict` is an executable file in the candidate directory; the sha256 of `code_version`'s local file matches; `code_version.entry` is the task's `resources.train_command` when the task gives one; `config_request` lies in the space when the task has a `search_space` (these two are also checked by `rsi validate candidate --task`, the error naming the key). The id coverage is checked at registration.

**Recomputed at registration.** The definitions of metrics and baselines are in "Metric definitions" of task.md. train computes these numbers with `rsi score --task <task.json> --labels <valset>/rows.jsonl --predictions <candidate>/predictions_val.jsonl --train <trainset>/rows.jsonl` and writes them into the manifest as they are. At registration the code recomputes the three items below by the same definitions and refuses a difference over 1e-9, the error giving the field, the reported value, the recomputed value and the `rsi score` command to use:

- Each known metric in `metrics_val`: `predictions_val` against the rows of the current valset.
- Each known metric of each known baseline in `baselines_val`: the baseline computed with the rows of the trainset named by the candidate's `trainset_id`, scored on the current valset.
- `train_gate.passed`: equals "the primary metric is strictly better than every baseline the task lists" (direction by `metrics.higher_is_better`; equal is not better), judged on the recomputed numbers, whatever baselines `baselines_val` writes. When the task's baselines include ones `rsi` does not know, it is checked against the known baselines only: `passed: true` when a known baseline was not beaten is refused, while `passed: false` is accepted (the unknown baseline may not have been beaten). Not checked when the primary metric is unknown or there is no known baseline.

`train_gate.detail` is a description for people and is not checked. Unknown metric and baseline names are skipped, and `rsi register` appends `(not verified: <names>)` to its output line.

A candidate with `train_gate.passed` false can be registered but cannot be sent to evaluation: registering a `delivery` that refers to it is refused. `rsi register` prints the train gate's result and the valset primary metric after the id, e.g. `candidate-0003 (train gate: passed; val auc 0.81)`.

## `search`

One config search: the result of one dispatch (or one return) of the train module, and one attempt of this round (the per-round cap is in task.md). No required files. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"search"` | Artifact kind |
| `trainset_id` | string | The trainset searched on |
| `valset_id` | string | The valset searched on |
| `strategy` | string | The search strategy |
| `budget` | object | Budget, see below |
| `trials` | list | Each trial, see below |
| `default_candidate` | string | The id of the candidate of the default config |
| `best_candidate` | string | The id of the best candidate |
| `gain_vs_default` | object | The gain over the default config, see below |

The sub-fields of `budget`: `compute_hours`. Each item of `trials`: `candidate_id` (string, or null), `config`, `metric_val` (the primary metric on the valset, number or null; taken from the trial candidate's `metrics_val`, i.e. the number the train command computed with `rsi score`), `compute_hours`, `status`. `status` `ok` means the trial succeeded, and then `candidate_id` and `metric_val` must both be given; trials with other statuses (e.g. `failed`, `timeout`) may have null `candidate_id` and `metric_val`. The sub-fields of `gain_vs_default`: `metric`, `delta`.

Checks inside the directory:

- `best_candidate` and `default_candidate` are both the `candidate_id` of a trial with `status` `ok`; non-null `candidate_id`s do not repeat; the sum of the trials' `compute_hours` is not more than `budget.compute_hours`.
- Tune gate: the `metric_val` of `best_candidate` is not worse than that of `default_candidate` (direction by the task definition's `metrics.higher_is_better`, tolerance 1e-9); when that cannot be met, hand back the default config's candidate as `best_candidate`.
- `gain_vs_default.metric` equals the primary metric, and `gain_vs_default.delta` equals the best minus the default `metric_val` (tolerance 1e-9); when lower is better, a negative value means better.

Registration also checks (trials whose `status` is not `ok` are not checked):

- `trainset_id` and `valset_id` are registered, and the `candidate_id` of every trial with `status` `ok` is registered as a candidate.
- Each ok trial's `metric_val` equals its candidate's `metrics_val[primary metric]` at registration (tolerance 1e-9; when the primary metric is a known metric, that number was recomputed and checked when the candidate was registered), so the tune gate rests on checked numbers.
- The candidates of all ok trials (the default candidate included) have the same `code_version.sha256` (hex, case ignored), i.e. one search compares only different configs under the same version of the training code. Different sha256s are refused, the error listing the candidates by sha256; a candidate with a null sha256 is refused too, since null cannot show it is the same code. Changes to the train command or the default config go between rounds, not in the middle of a search.

`rsi register` prints the gain after the id, e.g. `search-0001 (tune gate: passed; gain vs default auc +0.05 (best candidate-0002, default candidate-0001))`.
