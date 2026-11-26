# Evaluation and plan artifacts: evaluation, plan

Both kinds are produced by judge, and their directories hold a `manifest.json` with the common fields of README.md. Validation is done by `validate_artifact` in `rsi/artifacts.py`: the manifest is checked against the common fields plus the kind's extra fields (no fields that are not listed). Ids are only checked to be strings; whether they are registered, whether they agree with STATE, and the evalset's use count are checked at registration. "Primary metric" means `metrics.primary` in the task definition.

## Attributions and modules

Every item in `failures` and `issues` gives one attribution and one module, which must match by the table below (`MODULE_OF` in `rsi/rules.py`, checked by `check_attribution`). An attribution not in the table is an error too.

| Attribution | Module | Meaning |
|---|---|---|
| `data_coverage` | `data` | The training data does not cover the inputs that go wrong |
| `label_noise` | `data` | The labels themselves are noisy |
| `label_definition` | `data` | The label definition is unsuitable or ambiguous |
| `model_capacity` | `train` | The model lacks capacity |
| `calibration` | `train` | The probabilities are not calibrated |
| `config` | `train` | Hyperparameters or the training config are unsuitable |
| `runtime` | `deliver` | Problems of the delivery package at runtime (latency, cost, disagreement with offline) |

## `evaluation`

Evaluates one delivery and recommends whether to promote. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"evaluation"` | Artifact kind |
| `delivery_id` | string | The delivery evaluated |
| `candidate_id` | string | The candidate evaluated |
| `evalset_id` | string | The evalset used |
| `best_before` | null or object | The current best version, `{"version", "evaluation_id"}`; null while there is no best version |
| `metrics` | object | `candidate` (metric → value), `best` (null or metric → value), `baselines` (baseline name → metrics) |
| `deltas` | object | metric → `{"delta", "ci_low", "ci_high"}`, the candidate's difference from the current best and its confidence interval |
| `recommendation` | object | `decision` (`promote` or `keep`), `reason` |
| `failures` | list | Each item `issue`, `evidence`, `attribution`, `module` |
| `uses_testset` | bool | Whether this evaluation used the sealed testset; true means a milestone evaluation, see below |
| `metrics_test` | object, milestone evaluations only | metric name → number, the results on the sealed testset; required and holding the primary metric when `uses_testset` is true, not allowed when it is false |
| `baselines_eval` | object, optional; not in a milestone evaluation | ③ Compare's baselines on the evalset (report only, never read by the promotion rule): baseline id → `{"name", "metrics", "delta"}`, `name` the baseline's name, `metrics` (metric name → number, holding the primary metric) its `predictions_eval` against the evalset labels, `delta` `{"metric", "delta", "ci_low", "ci_high"}` the primary metric of the delivery's `predictions_eval` minus the baseline's, with its interval; required to cover the round's comparison, see below |
| `baselines_test` | object, milestone evaluations only, optional | The same shape on the sealed testset: the milestone baselines' `predictions_test` against the testset labels, and the milestone delivery's `predictions_test` minus each one's |

Required file: `report` (the evaluation report for people). Checks inside the directory:

- `metrics.candidate` holds the primary metric.
- When `best_before` is not null (there is a current best version), `metrics.best` is not null and holds the primary metric, and `deltas` holds the primary metric; missing ones are refused, so the promotion rule never quietly arrives at keep for want of numbers.
- When `uses_testset` is true, there is a `metrics_test` holding the primary metric; when it is false, there is no `metrics_test`.
- `baselines_eval` only when `uses_testset` is false, `baselines_test` only when it is true; in each entry `metrics` holds the primary metric and `delta.metric` is the primary metric.
- Each failure's `attribution` matches its `module`.

**Recomputed at registration.** The definitions of metrics, baselines, deltas and intervals are in "Metric definitions" of task.md. judge computes these numbers with `rsi score` and writes them into the manifest as they are:

- `metrics.candidate`, `metrics.baselines`, `deltas[primary metric]`: `rsi score --task <task.json> --labels <evalset>/labels.jsonl --predictions <delivery>/predictions_eval.jsonl --train <trainset>/rows.jsonl --clusters <evalset>/inputs.jsonl --compare <best-delivery>/predictions_eval.jsonl`, whose printed `metrics`, `baselines` and `delta` go into these three places (`<trainset>` is the trainset named by the evaluated candidate's `trainset_id`; without a best version, leave out `--compare`).
- `metrics.best`: the same command with `--predictions` replaced by the best version's `predictions_eval.jsonl`.

At registration the code recomputes the items below by the same definitions and refuses a difference over 1e-9, the error giving the field, the reported value, the recomputed value and the `rsi score` command to use:

- Each known metric in `metrics.candidate`: the `predictions_eval` of `delivery_id` against the labels of `evalset_id`.
- Each known metric of each known baseline in `metrics.baselines`: the baseline computed with the rows of the trainset named by the evaluated candidate's `trainset_id`, the groups taken from the evalset's `inputs.jsonl`.
- When `best_before` is not null, each known metric in `metrics.best`, and the `delta`, `ci_low` and `ci_high` of `deltas[primary metric]` (with the task's `metrics.ci` settings, the clusters taken from the evalset's `inputs.jsonl`). The predictions on the best version's side are: the best version's own delivery, when its `evalset_id` is the same as this evaluation's; otherwise the delivery listed in this evaluation's `parents` whose candidate is the best version's candidate and whose evalset is the current evalset. Without either, registration refuses, and the best version must first be packaged and delivered on the current evalset (see below).

**③ Compare's baselines on the evalset** (`baselines_eval`). When this round holds a `comparison` of the evaluated candidate whose `evalset_id` equals the evaluation's (the latest one, when there are several: `rsi.workspace.latest_comparison`), each baseline it lists must be a key of `baselines_eval`; a missing one is refused with `baselines_eval: missing <id> (<name>)`, and a key that is not a baseline of that comparison is refused (without such a comparison, any key is). The judge's brief gives each of them as `baseline_eval_<id>` (the baseline directory), and the comparison as `comparison`. The judge does not assemble the numbers: `rsi score --task <task.json> --labels <evalset>/labels.jsonl --predictions <delivery>/predictions_eval.jsonl --clusters <evalset>/inputs.jsonl --baselines-on evalset --baseline <baseline dir>` (one `--baseline` per baseline) prints the whole `baselines_eval` object, which goes into the evaluation as it is. Registration recomputes each entry by the same definitions (the `rsi.metrics.score` and `rsi.metrics.baseline_delta` that `rsi.metrics.baselines_scored` assembles for `rsi score`): `name` must be the baseline's; each known metric in `metrics` from the baseline's `predictions_eval` against the evalset labels; `delta`'s `delta`, `ci_low` and `ci_high` the paired cluster bootstrap of the primary metric, the delivery's `predictions_eval` minus the baseline's, clusters from the evalset's `inputs.jsonl`, the task's `metrics.ci` settings (`rsi.metrics.baseline_delta`, as a comparison's `deltas_val`). A mismatch is refused with that command. These numbers are a report: no gate and no promotion decision reads them. When the primary metric is one `rsi` does not compute, `rsi score --baselines-on` refuses and the judge computes the entries itself (said in the report; registration checks the known metrics only); when it is undefined on the set (e.g. `auc` on labels of one class), neither the candidate's metrics nor any delta can be computed, so the whole evaluation fails as it would without ③, the judge hands back `failed` naming the cause, and the evalset needs a refresh.

Items of `deltas` other than the primary metric, `failures` and the report file are not checked. Unknown metric and baseline names are skipped, and `rsi register` appends `(not verified: <names>)` to its output line; when the primary metric is an unknown metric, `deltas` is not checked.

Registration also checks: `recommendation.decision` must equal what the promotion rule (`decide_promotion` in `rsi/rules.py`, see `promotion_rule` in task.md) concludes for this evaluation, and a mismatch is refused, the error giving the rule's conclusion and reason. The numbers the rule reads are those checked above, so promotion decisions rest on numbers the code has checked. Each registration (except a milestone evaluation) adds 1 to the current evalset's use count `eval_use_count`; at `eval_use_limit` registration refuses, and the evalset must be refreshed first.

After an evalset refresh, the best version's original evaluation and delivery are both on the old evalset. When evaluating on the new evalset, `best_before` still gives the best version and its original evaluation; `metrics.best` and `deltas` are computed with a new delivery of the best version's candidate on the current evalset, whose id goes into the common field `parents`, by which registration checks them (without such a delivery, registration refuses). Known limitation: the stop condition's `target` is still judged by the primary metric of the best version's original evaluation (its result on the old evalset); recording results per evalset is left for later.

### Milestone evaluation

An evaluation with `uses_testset` true is a milestone evaluation: as the run is about to stop, it checks the current best version on the sealed testset, and it makes no promotion decision. Registration checks:

- There is a current best version, `candidate_id` is the current best version's candidate, and `best_before` gives the current best version.
- The delivery `delivery_id` names has `predictions_test`; this round has a `use_testset` approval not yet used (the approval of deliver's run on the testset was used by that delivery, so another one is needed here).
- `recommendation.decision` must be `keep`, and is not checked against the promotion rule.
- Each known metric in `metrics_test` is recomputed by "Metric definitions" in task.md: this delivery's `predictions_test` against the labels of its `testset_id`, computed with `rsi score --task <task.json> --labels <testset>/labels.jsonl --predictions <delivery>/predictions_test.jsonl`; a mismatch is refused. The `metrics` and `deltas` of a milestone evaluation are not checked.
- ③ Compare's baselines on the testset (`baselines_test`): each baseline with `predictions_test` registered in this round (`rsi.workspace.milestone_baselines`: the milestone's compare step scored the baselines of the best version's comparison on the testset inputs) must be a key (`baselines_test: missing <id> (<name>)`), and no other key is allowed. The milestone judge's brief gives each as `baseline_test_<id>`, and `rsi score --task <task.json> --labels <testset>/labels.jsonl --predictions <delivery>/predictions_test.jsonl --clusters <testset>/inputs.jsonl --baselines-on testset --baseline <baseline dir>` (one `--baseline` per baseline) prints the whole `baselines_test` object; registration recomputes it the same way on the testset (each baseline's `predictions_test` against the labels of the delivery's `testset_id`, the milestone delivery's `predictions_test` minus it, clusters from the testset's `inputs.jsonl`). A milestone evaluation has no `baselines_eval` (the round's evaluation reports them on the evalset). Report only.
- The evalset's use count is neither checked nor increased.

`metrics`, `deltas` and `recommendation` are still written on the evalset (the candidate is the best version, so the delta is usually 0), and the results on the testset go only into `metrics_test`. `rsi promote` refuses a milestone evaluation (milestone evaluations are not promotion decisions); `rsi next` does not count it when checking whether this round has an evaluation without a decision. judge writes no `plan` for it.

## `plan`

The next round's plan, from the evaluation's results. Extra fields:

| Field | Type | Description |
|---|---|---|
| `kind` | always `"plan"` | Artifact kind |
| `evaluation_id` | string or null | The evaluation the plan rests on; may be null only when this round was closed by `rsi close-round` |
| `issues` | list | Each item `issue`, `evidence`, `attribution`, `module`, `change`, `expected` |
| `entry_module` | enum `data`, `train`, `deliver` (an old plan's `tune` is read as usual, the same as `train`) | The module the next round enters at |

The sub-fields of `expected`: `metric` (the metric expected to improve), `direction` (`up` or `down`), `size` (the expected size, a number).

No required files. Checks inside the directory: each issue's attribution matches its module; `entry_module` must be one of the modules that appear in `issues`.

Registration also checks `evaluation_id`: when not null, it must be a registered evaluation; when null, this round must have been closed by `rsi close-round` (no candidate to evaluate, the round's decision having a null `evaluation_id`), otherwise it is refused. A closed round has no evaluation, and judge writes the plan from this round's candidates (e.g. why they failed the train gate), searches and experiment records, in the same format.
