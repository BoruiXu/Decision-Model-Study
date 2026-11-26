# Compare artifacts: baseline, comparison

Both kinds are produced by the compare module (`compare`, stage ③ Compare), and their directories hold a `manifest.json` with the common fields of README.md. Validation is done by `validate_artifact` in `rsi/artifacts.py`: the manifest is checked against the common fields plus the kind's extra fields (no fields that are not listed). The format of prediction rows is in "Decision rows" of README.md. A comparison is a report: no gate and no promotion decision rests on it. Within one hand-back the engine registers baselines and comparisons after the search and before the delivery (see engine.md).

## `baseline`

One baseline model and its predictions. A baseline is trained on the trainset (or on none, such as a zero-shot base model), scored on the valset and the evalset inputs, and at the milestone also on the testset inputs. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"baseline"` | Artifact kind |
| `name` | string matching `^[a-z][a-z0-9_]*$` | The baseline's name, e.g. `lr`, `gbdt`, `mlp`, `zero_shot`, `few_shot`, `team_recipe`; a comparison lists it under this name |
| `trainset_id` | string or null | The trainset it was trained on; null for a baseline that needs no trainset |
| `valset_id` | string | The valset of its `predictions_val` |
| `evalset_id` | string | The evalset whose `inputs.jsonl` its `predictions_eval` are on |
| `testset_id` | string or null | The testset whose `inputs.jsonl` its `predictions_test` are on; set exactly when the file `predictions_test` is there (a milestone baseline), else null |
| `spec` | object | The resolved specification of the baseline (model family, grid, settings) |
| `spec_sha256` | string | The sha256 the compare module computes over the canonical JSON of the name, the resolved spec (for a CPU baseline also the library that fits it and its version), the features key (`features_key`: the features code's `FEATURES_VERSION` and the sha256 of the task's adapter file, null for the generic views; null for a baseline that reads the rows as text), its version (`COMPARE_VERSION`) and the policy key (the task's `policy` section and `DPOLICY_VERSION`, null without the section); a registered baseline is reused when it and the sets match (see "Reuse" in modules/compare/SKILL.md) |
| `config` | object | The configuration chosen (e.g. the grid point picked on the valset) |
| `features` | object or null | The features the baseline was fitted on (e.g. the featurizer's signature); null for a baseline that reads the rows as text |
| `metrics_val` | object | metric name → number, the results of `predictions_val` on the valset; must hold the task's primary metric |
| `cost` | object | `{"compute_hours": number >= 0}`: the GPU hours it used (0 for a CPU baseline); the compare module also writes them to its spend file (`compare:<name>`, spend.md), from which the ledger counts them (a baseline's `cost` is not counted) |
| `model` | object, optional | `{"location", "predict"}` of a stored model that can predict again (the team recipe's packed model, run again on a refreshed evalset and at the milestone instead of training again; a milestone baseline has none, its `reused_from` has it): `location` (string, where the model files are, for people) and `predict` (string, the prediction entry) |
| `reused_from` | string or null | The id of the registered baseline this one was made from (re-predicted on a refreshed evalset, or scored on the testset at the milestone); null for a baseline trained here |

Files: `predictions_val` and `predictions_eval` are required; `policy` (`policy.json`, the decision policy fitted on the valset predictions, in the format of a candidate's: see models.md) and `predictions_test` are optional.

With the task's `policy` section every baseline carries a policy (with its fit report `policy_report`, `policy_report.json`, when known), and its prediction files hold the rows with it applied. `lr`, `gbdt`, `mlp`, `zero_shot` and `few_shot` fit theirs on their valset predictions exactly as a candidate does (`compare.preds.with_policy`); `team_recipe` never fits one: its packed candidate fitted its own, which its `model.predict` applies, and the baseline keeps that candidate's `policy.json`. A re-predicted or milestone baseline keeps its source's policy and applies it to its new scores (a CPU refit must reproduce the registered `predictions_val`, probabilities and decisions).

`model.predict`, when given, follows the rule of a candidate's prediction entry (see models.md): the relative path of an executable file in the baseline directory, which as written may not leave it, called as `<baseline dir>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>`.

Checks inside the directory: valid prediction rows in each predictions file (fields, repeated ids, labels, probabilities); `metrics_val` holds the primary metric; `model.predict`, when given, is an executable file in the directory; `testset_id` is set exactly when `predictions_test` is there.

Registration also checks:

- `valset_id` names a registered valset and `evalset_id` a registered evalset (not necessarily the current ones), and `trainset_id`, when not null, a registered trainset.
- `reused_from`, when not null, names a registered baseline with the same `name`.
- The ids of `predictions_val` are exactly those of the valset's `rows.jsonl`, and the ids of `predictions_eval` exactly those of the evalset's `inputs.jsonl`.
- With `predictions_test`: `testset_id` is the current testset, the ids of `predictions_test` are exactly those of its `inputs.jsonl`, and the registration needs a `use_testset` approval in the current round, single-use like a milestone delivery's (the engine writes it right before the registration, quoting the user's milestone consent).
- **Recomputed:** each known metric in `metrics_val`, from `predictions_val` against the valset's rows, by the definitions in "Metric definitions" of task.md; a difference over 1e-9 is refused. Compute it with `rsi score --task <task.json> --labels <valset>/rows.jsonl --predictions <baseline>/predictions_val.jsonl`.

## `comparison`

The chosen candidate against each baseline on the valset. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"comparison"` | Artifact kind |
| `candidate_id` | string | The candidate compared |
| `trainset_id` | string | The candidate's trainset |
| `valset_id` | string | The candidate's valset, on which every number is computed |
| `evalset_id` | string | The evalset every baseline's `predictions_eval` are on (the judge scores them there) |
| `baselines` | object | baseline name → id of a registered `baseline`; may be empty; no baseline is named `candidate` |
| `metrics_val` | object | `"candidate"` and each name in `baselines` → {metric name → number} on the valset; `metrics_val.candidate` must hold the primary metric |
| `deltas_val` | object | baseline name → `{"metric", "delta", "ci_low", "ci_high"}`: the primary metric of the candidate minus that of the baseline, with its interval; each name is in `baselines`, and `metric` is the primary metric |
| `strongest` | string or null | The name in `baselines` with the best primary metric on the valset, in the metric's direction; a tie goes to the alphabetically first name; null exactly when `baselines` is empty |
| `skipped` | object | baseline name → reason, for the baselines the task asks for that were not compared (e.g. an unknown name, or a GPU baseline the compute budget could not cover); no name is both here and in `baselines` |

Required file: `report` (the comparison for people to read). The compare module writes it as `report.md`: the candidate against its strongest baseline (`Candidate vs strongest baseline (<name>): Δ<primary> <delta> [<lo>, <hi>] on valset`), a table of model | primary metric | each other metric | Δ of the primary metric (candidate − model) [interval] (`rsi.report.comparison_table`, the table REPORT.md shows for the round), then a line for each skipped baseline. The comparison's `notes` is that first line.

The compare module writes the comparison itself (`compare/report.py`, fixed code, every number with `rsi.metrics` from the registered prediction files): the candidate is the brief's (its `predictions_val` carry its decision policy, like the baselines'), and the baselines are those registered for the round (the followup's `registered-<n>.json`) and those reused (their ids from the brief). A baseline on other sets than the candidate's goes under `skipped` with why, since registration would refuse it. The judge then reports the same baselines on the evalset in the evaluation's `baselines_eval`, and at the milestone the testset ones in `baselines_test` (evaluation.md); `rsi report`, `rsi status` and the overview's ③ node show the candidate against the strongest baseline. All of it is a report: nothing gates on it.

Checks inside the directory: `metrics_val` holds `candidate` (with the primary metric) and each name in `baselines`, and no other key; every name in `deltas_val` is in `baselines`, and each `metric` is the primary metric; no name is both in `baselines` and in `skipped`; `strongest` is a name in `baselines`, null exactly when `baselines` is empty.

Registration also checks:

- `candidate_id` is a registered candidate whose `trainset_id` and `valset_id` equal the comparison's; `evalset_id` is a registered evalset.
- Each id in `baselines` is a registered `baseline` listed under its own `name` and only once, so the numbers are never shown under another baseline's name.
- Each baseline's `valset_id` and `evalset_id` equal the comparison's, and its `trainset_id` is null or the comparison's. A baseline on an earlier evalset is refused: after an evalset refresh it is re-predicted on the new evalset as a new baseline with `reused_from`.
- **Recomputed** on the rows of the valset, by the definitions in task.md, refusing a difference over 1e-9. The error names the field (e.g. `deltas_val.lr.ci_low`), the reported and the recomputed value, and the `rsi score` command that gives the number:
  - `metrics_val.candidate`: each known metric of the candidate's registered `predictions_val`; `rsi score --task <task.json> --labels <valset>/rows.jsonl --predictions <candidate>/predictions_val.jsonl`.
  - `metrics_val.<name>`: each known metric of that baseline's registered `predictions_val`; the same command with `--predictions <baseline>/predictions_val.jsonl`.
  - `deltas_val.<name>`: `delta`, `ci_low` and `ci_high` of the paired cluster bootstrap of the primary metric, the candidate's `predictions_val` minus the baseline's, clusters from the valset's `inputs.jsonl`, with the task's `metrics.ci` settings; `rsi score --task <task.json> --labels <valset>/rows.jsonl --predictions <candidate>/predictions_val.jsonl --clusters <valset>/inputs.jsonl --compare <baseline>/predictions_val.jsonl` prints them as `delta`.
  - `strongest`: from the recomputed primary metric of each baseline.

  The deltas and `strongest` are not checked when the primary metric is one `rsi` does not compute; unknown metric names in `metrics_val` (of a baseline or a comparison) are skipped, and `rsi register` appends `(not verified: <names>)` to its output line.
