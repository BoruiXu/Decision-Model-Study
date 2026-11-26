# task: the task definition

`task.json` is the task definition and also the manifest of a `task` artifact; `TASK.md` is the same content for people to read (template in `skills/rsi-orchestrator/templates/TASK.md`). Validation is done by `rsi`'s task check; the rules are in `rsi/schemas/task.json` and `rsi/rows.py`. No level allows fields that are not listed.

## Top-level fields

- `kind`: always `"task"`.
- `task_id`: the task's identifier, matching `[a-z0-9][a-z0-9-]*`.
- `version`: the task definition's version, an integer >= 1. Changing the task definition raises the version and needs a `task_change` approval (except a new version that fills in the split rule, only before the evalset and the sealed testset are registered, see `data`).
- `decision`: the decision definition.
- `input`: the input format, i.e. the format of `state` in each dataset row.
- `features`: optional, how a row's `state` becomes numbers (the features adapter), see below.
- `data`: the data sources.
- `metrics`: the metrics.
- `promotion_rule`: the rule for promoting a candidate.
- `stop`: the stop conditions.
- `budget`: the budget.
- `resources`: the resources available, an object that the orchestrator and the modules read as they need, see below.
- `search_space`: optional, the config space the train module (`rsi-train`) may search, see below.
- `search`: optional, the train module's search settings, see below.
- `compare`: optional, turns on ③ Compare and sets its baselines, see below.
- `policy`: optional, turns on the decision policy that every candidate and every compare baseline fits on its valset predictions, see below.
- `eval_use_limit`: how many times one evalset may be used at most, an integer >= 1.
- `quality_checks`: optional, the checks and thresholds of the data quality gate, any object, see below.
- `delivery`: optional, the delivery settings, see below.
- `flow`: optional, the main flow's settings, see below.
- `runner`: optional, the settings for how `rsi run` calls module agents and runs code modules, see below.
- `approvals`: the items that need the user's approval, from `task_change`, `over_budget`, `use_testset`, `refresh_sets`, `plan`, `delete`. Of these, `task_change`, `refresh_sets` and `use_testset` guard fixed rules (the task definition, the frozen valset and evalset, the sealed testset) and always need approval whether listed or not (except a new version that fills in the split rule, only before the evalset and the sealed testset are registered, see `data`), and each approval holds for one registration only; this list decides only whether `plan`, `over_budget` and `delete` need approval. `delete` covers deletions the orchestrator and the user make by hand; rsi-train's `trsi collect` deletes unusable checkpoints automatically (save points not selected, checkpoints of failed or replaced trainings) without approval, whether `delete` is here or not (the user's decision).

## `resources`

Any object; the items below have a fixed meaning (the modules read the rest as they need):

- `resources.train_command`: optional, a string, the location of the train command (an absolute path, or a path relative to the repository root). When not given, `<repo root>/modules/train/train` is used; the caller runs it with `PYTHONPATH=<repo root>` in the environment. When given, registering a candidate requires its `code_version.entry` to be this command (the same string, or the same file once a relative path is resolved against the repository root and symlinks are resolved), see models.md.
- `resources.onejev`: optional, an object, the environment in which the train module runs onejev-harness (`python`, `ojh_bin`, `machine`, `env_file`, `default_config`, `estimate` and so on, interpreted by the module); the schema checks only that it is an object. Its `default_config` (shaped like a candidate's `config_request`: `{"base", "recipe", "overrides", "seed", "max_steps"}`) is the default config, against which registration checks the keys `search_space` does not list (see below).
- `resources.gpu_ids`: optional, a list of non-negative integers, the GPU numbers the train module may use (all GPUs when not written); at each launch the module picks free ones from it.

## `search_space` (optional)

The train module searches only inside it, and a user approving the task definition approves this space. The keys are config paths: `base`, `recipe`, `seed`, `max_steps`, `overrides.onejev.<flag>`, `overrides.lora.<field>`; each key's value is exactly one of these three:

- a non-empty list of allowed values;
- `{"min": <number>, "max": <number>}` (both ends allowed, `min` not greater than `max`); `base`, `recipe` and `seed` cannot be written as ranges, only as lists or as the `{"values", "dead"}` below (the train module tries those values one by one), and the task check refuses them written as ranges;
- `{"values": [<allowed values>, …], "dead": {<value, written as text>: <why it is a dead end>}}`, with `dead` optional: known dead ends for the module to see (trsi judges dead ends); at registration the values in `dead` still count as inside the space.

`dead` can only be written inside a key; there is no top-level `dead`.

When the task has a `search_space`, every candidate registered must have a `config_request` (see models.md), checked by these rules (`rsi/space.py`):

- A key whose value is null is not checked (it means the recipe's or train command's default).
- A key in `search_space`: the value is in the list (or in `values`, the `dead` ones included; the same JSON value, so `true` is not `1`), or is a number inside the range.
- A key not in `search_space`: the value must equal the value at the same path of `resources.onejev.default_config`; refused when the task gives no default config or the default config lacks the key. A task without a default config pins a setting with a one-value list (e.g. `"base": ["joey-1"]`).
- A refusal names the key, the value and the allowed range.

## `search` (optional)

The train module's search settings, interpreted by the module (the framework checks only the shape): `max_configs` (an integer >= 1, how many configs one search tries at most), `seeds` (a list of distinct integers, the first being the probe's seed), `probe_margin`, `accept_margin` (numbers >= 0), `guards` (metric name to the largest regression allowed, a number >= 0).

## `decision`

- `type`: `binary`, `choice` or `score`.
- `instructions`: the decision's instructions, for people and the model.
- `labels`: the list of labels, at least 2 strings. The labels of `score` must be `"0"` to `"k-1"` in order.
- `label_meanings`: label to meaning; its key set must equal `labels`.
- `positive`: the positive label. Required for `binary` and must be in `labels`; not allowed for the other types.

Every dataset row carries the OneJev question `question` derived from `decision` (`instructions` is the `instructions` here, `criteria` comes from `label_meanings`; a yes/no `binary` is a `noul` question, `score` a `score` question, the rest `choice` questions), checked row by row at registration, see "Decision rows" in README.md.

## `input`

This section describes the model's input, i.e. the `state` of each dataset row (called `input` in the rows of old workspaces, see "Decision rows" in README.md).

- `format`: `messages`, `fields` or `object`. With `messages`, `state` is `{"messages": [{"role", "content"}, ...]}` (non-empty); with `fields`, `state` is `{"fields": {...}}`; with `object`, `state` is any non-empty JSON object (e.g. a structured `{"entity", "summary", "transactions"}`), whose inner structure is not checked and which goes to the model as it is.
- `description`: a description of the input.
- `fields`: the list of field names, which the keys of `state.fields` must cover. Required and non-empty when `format` is `fields`; not allowed for `messages` or `object`.
- `verbatim`: optional, a list of state paths whose text is a row's own request, e.g. `messages[].content` in a routing task (dots between keys, `[]` for every item of a list); `[]` when not written. Neither data generation nor prompt variation ever changes that text, and the features' edit view never offers these paths (nor any path under them). The default is filled in only in one place, `verbatim_paths` in `rsi/rules.py`.

## `features` (optional)

How a row's `state` becomes numbers for the compare module's baselines and for data generation (`modules/common/features`, standard library only).

```json
"features": {"adapter": "examples/elliptic_onejev/features_adapter.py", "constraints": []}
```

- `adapter`: optional, a Python file (standard library only), an absolute path or a path relative to the repository root. The task check refuses a path that is not an existing file, naming the path it resolved. The file may define any of `features(state)` (column → number), `edit_view(state)` (path → scalar), `apply(state, changes)`, `repair(state)`, `check(state)` (a list of the broken rules) and `describe()` (the rules as text for the LLM prompts); every function it leaves out is the generic one. When not written, every function is the generic one.
- `constraints`: optional, a list, `[]` when not written: declarative constraints on the state's numbers (`modules/common/features/constraints.py`). Each item is one of:
  - `{"nonneg": [path, ...]}`: every number at each path is 0 or more;
  - `{"integer": [path, ...]}`: every number at each path is a whole number;
  - `{"sum": path, "of": [path, ...]}`: the number at `sum` equals the sum of the numbers at `of` (within 1e-9, relative, or absolute near 0);
  - `{"order": [path, path, ...]}`: the numbers are non-decreasing in the order given (e.g. min, median, max);
  - `{"zero_if_zero": path, "then": [path, ...]}`: when the number at `zero_if_zero` is 0, so is each number at `then`.

  A path joins keys with dots; a key followed by `[]` stands for every element of the list there (`transactions[].fee`), and a part of digits picks one element (`transactions.0.fee`). In `order` and `zero_if_zero` every `[]` of the item is bound: the item holds in each element (each combination when its paths go through different lists), and a path without `[]` is the same in each; two paths go through the same list when their text up to the `[]` is the same. In `sum` the `[]`s of the `sum` path are bound the same way and an `of` path's other `[]`s are summed over: `{"sum": "total", "of": ["t[].v"]}` is total = the sum of every `t[].v`, and `{"sum": "t[].total", "of": ["t[].a", "t[].b"]}` holds in each element of `t`. Every leaf an item names must be a number (a bool is not one); a leaf that is missing or not a number is a violation that repair leaves as it is.

  The adapter's `repair` and `check` run the constraints first, then the adapter's own functions. Repair works on a copy, in this order: every `nonneg` item (a negative number becomes 0), every `integer` item (rounded to the nearest whole number, halves away from zero), then the relations (`sum`, `order`, `zero_if_zero`) in their order, twice: a sum's number becomes the sum, an order's numbers are sorted into its places, a `then` number becomes 0. Check gives one message per broken item and place, starting with the item's rule (`nonneg transactions[].fee`, `sum total = t[].v`, `order s.min <= s.med <= s.max`). dgen edits rows through this adapter, and `dset repair` repairs rows with it. The task check refuses an item with another key or a value of the wrong type (`order` needs at least two paths); the features component refuses an item that does not name exactly one kind with the key it needs (`of` for `sum`, `then` for `zero_if_zero`) when it loads the adapter, so dgen and `dset repair` stop with that error.

The generic functions:

- `features`, the model view: a number outside any list becomes a column named by its dot path (`summary.txs_sent`; a bool is 1 or 0); a list at path `p` becomes `p.count`, its number items `p.mean`, `p.max` and `p.sum`, and each number `q` inside its object items (at any depth, outside nested lists) `p[].q.mean`, `p[].q.max` and `p[].q.sum` over the items that have it; a nested list becomes `p[].q.count.mean`. Strings and non-finite numbers are skipped.
- `edit_view`: every scalar leaf by its path, list indices as path parts (`transactions.3.tx_total_btc`); `apply` writes such paths back into a copy of the state, refusing a path the state does not have and a number that would become a non-number.
- `repair` returns the state unchanged, `check` finds nothing, `describe` is empty.

Paths join keys with dots and nothing is escaped, so a state key that contains a dot or is empty is ambiguous: `{"a.b": 1}` and `{"a": {"b": 1}}` give the same column `a.b`, and `apply` reads `a.b` as the nested path only. Adapters for states with such keys define their own `features`, `edit_view` and `apply`.

The features are computed from a copy of each row's `state` only, never from the row's id, target or meta. Their columns are fixed on the rows they are fitted on (a column a row lacks is 0); fitting with the rows' hard labels refuses a column that takes one value exactly on the rows of one label (at least two of them) and other values on every other row, since it determines the label. Every user of the features fits them only on rows that are not prompt variants (`real_rows`: the rows without `meta.extra.variant`).

The defaults are filled in only in one place, `features_settings` in `rsi/rules.py`; `adapter_path` there resolves the adapter's path.

## `compare` (optional)

③ Compare runs only when this section is there; without it (an old workspace, or a task that wants none) the flow goes from ② Train & tune to ④ Deliver and the pages show `③ Compare (off)`. The compare module (a code module, no agent) trains the baselines once per trainset, reports the round's candidate against each of them on the valset (a `comparison`) and has each predict the evalset inputs for the judge; at the milestone it scores them on the testset inputs. It is a report only, never a gate (see engine.md).

```json
"compare": {"baselines": ["lr", "gbdt", "mlp", "zero_shot", "few_shot", "team_recipe"], "seed": 0,
            "python": null, "max_train_rows": null, "grids": {},
            "zero_shot": {"base": null}, "few_shot": {"base": null, "k": 4, "max_example_chars": 2000},
            "team_recipe": {"base": "elephant-1", "recipe": "recipes/elephant_lora_1g.yaml", "overrides": {},
                            "seed": 0, "max_steps": null},
            "gpu_hours": {"zero_shot": 0.1, "few_shot": 0.1, "predict": 0.1}}
```

- `baselines`: required, a list of at least one name, each matching `^[a-z][a-z0-9_]*$` and listed once. The compare module knows `lr`, `gbdt`, `mlp`, `zero_shot`, `few_shot` and `team_recipe`; a name it does not know is not trained and the comparison records it under `skipped`.
- `seed`: an integer, 0 when not written: the fixed seed of every baseline.
- `python`: a string or null (default): the interpreter that runs the compare module (it needs numpy, scikit-learn, xgboost or lightgbm, and pyyaml: `modules/common/install_deps.sh --python <it>` installs them); null takes `resources.onejev.python`. With neither, every compare dispatch hands back `failed`.
- `max_train_rows`: an integer >= 1 or null (default): the most trainset rows a baseline is fitted on; null fits on all of them.
- `grids`: an object, `{}` when not written: `<name>` → the grid (an object: parameter → list of values) that replaces that baseline's default grid (the defaults are in "Baselines" of `modules/compare/SKILL.md`).
- `zero_shot`: `{"base"}`, the base model scored with no training (a string, or null: `resources.onejev.default_config.base`).
- `few_shot`: `{"base", "k", "max_example_chars"}`: the same base model with `k` (an integer >= 1, default 4) labelled trainset examples in each row's instructions, each example's state cut at `max_example_chars` (an integer >= 1, default 2000).
- `team_recipe`: a config of the same shape as a candidate's `config_request` (`{"base", "recipe", "overrides", "seed", "max_steps"}`, all five required), which the train command trains as the team's existing recipe (a null `base` takes `resources.onejev.default_config.base`, a null `seed` the train command's default; a relative `recipe` is looked up under `modules/train`, then the ojh repository, as for every config); null when not written.
- `gpu_hours`: an object of numbers >= 0, `{"zero_shot": 0.1, "few_shot": 0.1, "predict": 0.1}` when not written: the estimated GPU hours of each GPU baseline (`team_recipe`'s training, when not written here, is `resources.onejev.estimate` of its base), and under `predict` of one prediction of a stored model on the evalset inputs (a team-recipe baseline re-predicted after a refresh, or one whose training an earlier launch of the dispatch finished: it is never trained again, so its training estimate does not hold it). A GPU baseline whose estimate is more than the GPU hours left (`budget_left.compute_hours`) is skipped, with the reason under the comparison's `skipped`. Milestone scoring (the baselines of the best version's comparison on the testset inputs) only predicts and is not held by the budget: the user's yes to the milestone approves it, and the milestone question says it may use GPU hours beyond the budget.

The defaults are filled in only in one place, `compare_settings` in `rsi/rules.py` (null without the section): an object default is merged with what the task writes, one level deep (`few_shot: {"k": 2}` keeps the default `base` and `max_example_chars`). The compare brief carries these settings as `compare`.

## `policy` (optional)

A decision policy is a calibration of a model's probabilities followed by a decision rule (`modules/common/dpolicy`, standard library only). With this section, the train command fits one for every candidate it packs, on the candidate's valset predictions (valset labels only), and stores it in the candidate (`policy.json`, see models.md); the candidate's `model.predict` applies it, so its `predictions_val`, deliveries and evaluations carry the policy's probabilities and decisions. The compare module fits one the same way for each baseline (see compare.md). Without the section (an old workspace, or a task that wants none) nothing is fitted and every candidate is as before. Version 1 is a calibration and a threshold only: no abstain band, no second stage and no refit on later labels.

```json
"policy": {"objective": "f1", "calibrations": ["none", "platt", "temperature", "isotonic"],
           "rules": [{"rule": "max_f1"}], "split": {"by": "time", "key": "meta.extra.step", "select_fraction": 0.4},
           "min_lo": 0}
```

Every field is optional (`"policy": {}` takes all the defaults):

- `objective`: a known metric (see "Metric definitions"), by which a policy is chosen; `f1` for a `binary` task, `log_loss` for `choice` and `score` when not written. The task check refuses a metric `rsi` does not compute, a binary-only metric in another task, and `expected_cost` without `metrics.costs`.
- `calibrations`: a non-empty list of `none`, `platt`, `temperature` and `isotonic`; all four for a `binary` task, `["none", "temperature"]` otherwise when not written (the task check refuses `platt` and `isotonic` outside a `binary` task).
  - `none` keeps the probabilities.
  - `platt`: Newton's method on (a, b) of sigmoid(a · logit p + b), L2 1e-6, p clipped to [1e-6, 1 − 1e-6], at most 100 steps.
  - `temperature`: golden-section search over log T in [−5, 5] (100 iterations) minimizing the log loss; sigmoid(logit p / T) for a binary task, softmax(log p / T) over the labels otherwise.
  - `isotonic`: pool adjacent violators on p, applied as linear interpolation, clipped at both ends.

  In a binary task p is the probability of `decision.positive`, and the other label gets 1 − the calibrated value.
- `rules`: a non-empty list of decision rules `{"rule", …its parameters}`; when not written, `[{"rule": "max_f1"}]` for the objective `f1`, `[{"rule": "cost"}]` for `expected_cost` and `[{"rule": "argmax"}]` otherwise (always `argmax` outside a `binary` task). A binary rule's threshold is chosen on the calibrated scores of the fit part: a row is positive when its calibrated probability is at least the threshold, the candidate thresholds are the distinct scores, and a tie in the rule's objective goes to the highest threshold.
  - `max_f1`: the highest F1 of the positive label;
  - `cost` (`cost_fp`, `cost_fn` >= 0, optional): the lowest `cost_fp` × false positives + `cost_fn` × false negatives, predicting none being a candidate too; without its own costs, those of `metrics.costs` (cost_fp = costs[negative][positive] − costs[negative][negative], cost_fn likewise); the task check refuses a cost rule with neither;
  - `precision_at` (`target` in (0, 1], required): of the thresholds with at least that precision, the one with the highest recall; none reaching it means never positive;
  - `recall_at` (`target` in (0, 1], required): of the thresholds with at least that recall, the one with the highest precision;
  - `top_frac` (`frac` in (0, 1], required): the highest threshold at which at least that share of the rows is positive;
  - `fixed` (`value`, required): that threshold;
  - `argmax`: the most probable label, a tie going to the earlier label in `decision.labels` (threshold 0.5).

  Every rule but `argmax` is a threshold rule, for `binary` tasks only: the task check refuses it elsewhere.
- `split`: `{"by", "key", "select_fraction"}`, how the valset is split into a fit part and a select part. When not written: by `time` on `data.split.key` when `data.split.by` is `time`, else by `group` on `meta.state_id`, `select_fraction` 0.4; a split written in part keeps the defaults of the fields it leaves out (the key of a `group` split is `meta.state_id`). By `time`, the latest max(1, round(n × select_fraction)) of the n distinct values at `key` (a number in every row) go to select; by `group`, the groups (the value at `key`; a row without one is a group of its own) ordered by the sha256 of the value, the last ceil(n × select_fraction) of them. `select_fraction` is strictly between 0 and 1. A `time` split needs a key: the task check refuses `"by": "time"` without one when `data.split` is not by time.
- `min_lo`: a number, 0 when not written. Every pair of a calibration and a rule (in the order written) is fitted on the fit part, applied to the select part and compared there with the identity policy (`none` + `argmax`, the model's own most probable label) by the paired cluster bootstrap of "Deltas and intervals" below (`rsi.metrics.paired_delta`, the task's `metrics.ci`, clusters from `meta.cluster`). The improvement is the delta in the objective's direction and its lower bound `ci_low` (higher is better) or −`ci_high` (lower is better). A pair is accepted when that lower bound is above `min_lo` **and** the task's primary metric (`metrics.primary`, computed by `rsi.metrics` in its direction) on the select part is not worse with the pair applied than with the identity policy (tolerance 1e-12): a policy never trades the primary metric for its objective (e.g. `isotonic`'s step function ties scores and can lower `auprc` for an equal `f1`; such a pair is refused with the reason). A primary metric `rsi` does not compute is not checked. Of the accepted pairs the one with the largest improvement wins (a tie: the earlier pair), its parameters those fitted on the fit part.

Every pair's numbers (delta, interval, primary metric, accepted or the reason it was refused) are kept beside the policy as `policy_report.json`. When no pair is accepted, when the fit or the select part lacks a label of the task (rare positives: e.g. `the select split lacks label yes`), or when the valset cannot be split, the policy is the identity policy, not accepted, with the reason. The defaults are filled in only in one place, `policy_settings` in `rsi/rules.py` (null without the section); the compare brief carries the task's section as `policy`.

## `data`

```json
"data": {
  "sources": [{"path": "/abs/path/raw", "about": "raw data: what each file is"}],
  "label_sources": ["human"],
  "rows": {"about": "one row per (address, time step); state as it is; the time step goes into meta.extra.step",
           "expect": {"rows": 291419, "label_counts": {"yes": 14720, "no": 276699}}},
  "split": {"by": "time", "key": "meta.extra.step",
            "ranges": {"trainset": [1, 29], "valset": [30, 32], "evalset": [33, 34], "testset": [35, 49]},
            "sizes": {"trainset": 1000, "valset": 500, "evalset": 500, "testset": 1000},
            "stratify": true, "unit": "cluster", "seed": 0}
}
```

- `sources`: the raw data sources, a list, possibly empty. Each item is a string (the old form: only a description, not a path, not on the read-deny list) or `{"path", "about"}` (`about` optional). `path` is an absolute path on this machine (file or directory), may not contain `(`, `)`, `*`, `?`, `[` and may not start with `!`: a read-deny rule cannot express such a path, so the task check refuses it rather than waiting for the dispatch. These paths are "the raw data".
- `label_sources`: the allowed label sources, from `rule`, `model_judge`, `human`, `outcome`.
- `rows` (optional): a description of all the labelled rows. `about` (required) is the row description for the data agent: what a row is and where each field comes from. `expect` (optional) is the exact count of all rows `rows` and the row count of each hard label `label_counts` (whose keys must be the task's labels), checked one by one when the `corpus` is registered and refused on a mismatch: the labels go through the preprocessing script the agent writes, and they pass only when the counts match.
- `split` (optional, may be written in part): the split rule, which puts all rows into the four sets.
  - `by`: `time` (the `key`'s value is a number, split by the closed intervals of `ranges`), `group` (whole groups by the `key`'s value, split at random by `fractions`) or `random` (split at random by unit, by `fractions`, no `key`).
  - `key`: a dotted path into the row, e.g. `meta.extra.step`, `meta.state_id`.
  - `ranges`: one `[lo, hi]` (lo ≤ hi) for each of `trainset`, `valset`, `evalset`, `testset`, not overlapping; rows outside all intervals are not used. Only `time` uses it.
  - `fractions`: a number in (0, 1] for each of the four sets, summing to at most 1; the remaining rows are not used. Only `group` and `random` use it.
  - `sizes`: each set's row count, integers >= 1; left out or null means all rows in range are taken. The training pool is not limited by it.
  - `stratify`: true when not written; `time` stratifies by (the `key`'s value, hard label), the others by hard label.
  - `unit`: `cluster` (the value when not written) or `state_id`. The same `state_id` may appear in several sets, and only the counts are reported; only with `state_id` written are sets separated by it, and registration checks it too.
  - `seed`: an integer, 0 when not written.

  A rule is "complete" when it has `by`; `time` and `group` have a `key`; `time` has four `ranges`, and `group` and `random` have four `fractions`. The other keys take their defaults. The task check only checks that what is written is consistent (`ranges` only with `time`, `fractions` only with `group` and `random`, no `key` with `random`, intervals not overlapping, fractions summing to at most 1); the rule is checked by `rsi/datasplit.py`.

**Filling in the split rule.** When the rule is incomplete, the data agent fills it in from the data in round 1's `sets` dispatch (time, group or random, ranges, sizes, stratification, unit, seed), without the user approving or being asked. The keys the task wrote are used as they are (`ranges`, `fractions` and `sizes` looked at set by set), and a conflict with the agent's choice is an error; the filled rule (defaults written out too) is recorded as a new version of the task definition: `task.json` with version + 1 and only the filled `data.split` added, and `TASK.md` being the current version's text followed directly by the output of `rsi.datasplit.split_section_md(rule)`, byte for byte; write the file with `rsi.datasplit.fill_task_md(<the current TASK.md's bytes>, rule)`, built from the text read as bytes, never read and written in text mode (a CRLF line ending would be turned into LF, and the fill would no longer match). The section opens with one newline (when the text ends with a newline, there is exactly one blank line before the heading), then the heading "## Split rule (`data.split`)", the line "Filled in by rsi-data from the data; the reasons are in that dispatch's record." and the rule as JSON with the defaults written out (a ```json code block, one top-level key per line). Registering this new version needs no `task_change`, as long as it is a **fill**: the same `task_id`, version + 1; `task.json` identical to the current version except for `data.split`; the current `data.split` incomplete, the new one complete, and every key value the current one wrote kept; `TASK.md` byte for byte equal to `fill_task_md(the current version's TASK.md, the new data.split)`; and the workspace never registered an evalset or sealed testset. A new version that is no fill needs `task_change` as usual; when `task.json` changed only `data.split` but is not a fill, the refusal gives the reasons it is no fill. When the author wrote a complete rule, there is no new version.

**The rule is fixed.** Once the evalset or the sealed testset is registered, the split rule is fixed for this run (refresh keeps it): any new version that changes `data.split` is refused, even with `task_change` (when both are complete, the rules are compared with the defaults written out).

## `metrics`

- `primary`: the primary metric's name.
- `higher_is_better`: whether higher is better for the primary metric. When the primary metric is one of the known metrics in "Metric definitions" below, it must agree with that metric's direction (false for `log_loss`, `brier` and `expected_cost`, true for the rest), and so must the known metrics in the guards, or the task check reports an error.
- `others`: the other metrics to report.
- `auc`, `auprc`, `brier` and `f1` are defined only for `binary` tasks: when the decision type is not `binary` and the primary metric, `others`, the guards or `policy.objective` name them, the task check reports an error (saying it is for binary tasks only).
- `baselines`: the baselines that must be compared, at least 1.
- `costs`: optional, the cost of each decision for each true label, `{<true label>: {<decision>: number}}`, both levels keyed by exactly the task's labels (the task check reports other keys), e.g. `{"yes": {"yes": 0, "no": 10}, "no": {"yes": 1, "no": 0}}`. `expected_cost` needs it: the task check refuses `expected_cost` as the primary metric, in `others`, in a guard or as `policy.objective` without it.
- `ci`: optional, the settings of the delta's confidence interval `{"resamples", "seed", "level"}`, each of the three optional: `resamples` an integer >= 100, 1000 when not written; `seed` an integer, 0 when not written; `level` between 0 and 1 (both excluded), 0.95 when not written. The defaults are filled in only in one place, `ci_settings` in `rsi/metrics.py`.

## Metric definitions

The known metrics, the known baselines and the interval's algorithm are fixed code (`rsi/metrics.py`), not decided by each module. The modules compute the numbers they report with `rsi score` (see the command list in the README), and at registration `rsi register` recomputes them by the same definitions and refuses a difference over 1e-9, the error giving the field, the reported value, the recomputed value and the `rsi score` command to use. So promotion decisions and the train gate both rest on numbers the code has checked.

Below, the "hard label" is the label with the highest probability in `target`, on a tie the one that comes first in `decision.labels`. Prediction rows and label rows are matched by `id`, and the predictions must cover exactly the label rows' ids. Sums use `math.fsum`, so the result does not depend on the order of the rows or the Python version.

| Known metric | Direction | Definition |
|---|---|---|
| `accuracy` | higher is better | The share of rows whose prediction's `decision` equals the hard label |
| `log_loss` | lower is better | The mean over rows of −ln p, where p is the probability the prediction gives the hard label, clipped to [1e-12, 1] |
| `macro_f1` | higher is better | The F1 of each label of `decision.labels` (`decision` against the hard label), averaged; a label with neither true rows nor predictions gets F1 0 and still takes part in the average |
| `auc` | higher is better | `binary` only: the ROC AUC with the probability the prediction gives `decision.positive` as the score and whether the hard label is the positive class as the truth, a positive-negative pair with equal scores counting 0.5; undefined when the label rows have no positives or no negatives, and registration and `rsi score` report an error |
| `auprc` | higher is better | `binary` only: the average precision on the same scores and truth (the step sum of the area under the PR curve): with scores from high to low, the sum at each threshold of "the increase in recall × the precision at that threshold"; rows with equal scores are one threshold (the same as scikit-learn's `average_precision_score`); undefined when the label rows have no positives, which reports an error |
| `brier` | lower is better | `binary` only: the mean over rows of (p − y)², where p is the probability the prediction gives `decision.positive` and y is 1 when the hard label is the positive class, 0 otherwise |
| `f1` | higher is better | `binary` only: the F1 of `decision.positive`, 2tp / (2tp + fp + fn) with the prediction's `decision` against the hard label; 0 when the denominator is 0 |
| `expected_cost` | lower is better | The mean over rows of `metrics.costs[hard label][decision]`; an error without `metrics.costs` |

| Known baseline | Definition |
|---|---|
| `majority` | Predicts the trainset prior for every row, i.e. the mean of the training rows' `target`; `decision` is the label with the highest probability in the prior (ties by label order) |
| `group_prior` | Predicts for each row the prior of its group (`meta.state_id`; `meta.group` in the rows of old workspaces) in the training rows; the overall prior when the training rows lack that group. The evalset's label rows have no `meta.state_id`, so the groups come from the same set's `inputs.jsonl` |

Baselines use the same definitions as the model: for each known baseline, the same known metrics are computed on its prediction rows. A candidate's baselines are computed with the trainset its `trainset_id` names, and the baselines in an evaluation with the trainset of the candidate evaluated.

**Deltas and intervals.** `deltas[primary metric]` is "candidate − current best": `delta` is the difference of the primary metric of the two predictions on the same label rows; the interval comes from a paired cluster bootstrap: the rows are grouped into clusters by `meta.cluster` (a row without a cluster is a cluster of its own), the clusters sorted by name; `resamples` times, `random.Random(seed).randrange` draws with replacement as many clusters as there are, and the difference of the two predictions' metrics is recomputed on the rows drawn (a draw on which a metric is undefined is skipped, e.g. `auc` drawing rows of only one class, `auprc` drawing rows without positives); `ci_low` and `ci_high` are the (1 − level)/2 and (1 + level)/2 quantiles of these differences, a quantile being interpolated linearly between the two neighbouring sorted values (position (n − 1) × q). The evalset's clusters come from its `inputs.jsonl`. The same inputs give the same results under Python 3.9 and 3.14.

**Unknown names.** A task may list metrics or baselines not in the tables above (e.g. `net_cost`, a custom baseline); `rsi` neither computes nor checks them, and the modules compute them and explain the method in their reports. Registration skips them, and `rsi register` appends `(not verified: <names>)` to its output line. When the primary metric is an unknown metric, the train gate, the deltas and the intervals are not checked; when the listed baselines include unknown ones, the train gate is checked against the known baselines only (see models.md).

## `promotion_rule`

- `min_delta`: how much a candidate must improve the primary metric over the current best; the gain must be greater than it, and exactly equal (a tie within a tolerance of 1e-9) does not count.
- `ci_lower_min`: the minimum lower bound of the gain's confidence interval; null when not required.
- `guards`: the list of guards. Each item has `metric` (must be the primary metric or one in `others`), `higher_is_better`, `max_regression` (>= 0, the largest regression allowed).

The decision is made by `decide_promotion` in `rsi/rules.py`, and the numbers it reads were checked at registration by "Metric definitions". Without a current best version, a candidate whose `train_gate.passed` is true is promoted. With a current best version, let `s` be 1 when higher is better for the primary metric and -1 when lower is better: the gain `s × deltas[primary metric].delta` must be greater than `min_delta`; when `ci_lower_min` is not null, the lower bound of the gain's interval (`ci_low` for `s = 1`, `-ci_high` for `s = -1`) is not less than it; each guard's regression (`best − candidate` when higher is better, `candidate − best` when lower is better, from the evaluation's `metrics`) is not more than `max_regression`. When a number needed is missing, keep (`keep`).

## `quality_checks` (optional)

The checks and thresholds of the data module's quality gate, e.g. a minimum sample size, a maximum near-duplicate rate, a minimum label reliability, a maximum share of generated rows, written as the task sees fit (e.g. `{"min_rows": 2000, "max_near_dup_rate": 0.01, "max_synthetic_share": 0.3}`). `dset` computes `min_rows`, `min_positive_rows`, `max_duplicate_rate`, `max_conflict_rate` and `max_synthetic_share` (the share of rows whose `meta.source` starts with `dgen:`; it passes at a share no greater than the threshold) itself. The data module reads it, computes the value of each check and writes them with their thresholds into the trainset's `quality.checks`. The framework checks only that it is an object and does not interpret what is in it. When not written, the data module sets its own checks for the task and explains them in its report.

## `delivery` (optional)

- `reproduction_tolerance`: a number greater than 0, 1e-6 when not written. When a delivery is registered, the delivery package's predictions on the valset inputs must reproduce the candidate's `predictions_val`: each probability differing by at most this value (see delivery.md). When the model runs on a GPU in low precision such as bf16 with varying batch sizes, probabilities may differ by more than 1e-6; then raise it to the real precision, e.g. 1e-4.
- `max_decision_mismatch_rate`: a number at least 0 and less than 1, 0 when not written. How many rows may have probabilities within the tolerance but a different decision: at most floor(this value × the valset's row count). For rows near the decision boundary, a tiny move in probability flips the decision, which can happen whatever the tolerance; this item loosens the check for them; `rsi` does not know the model's decision rule (threshold, how ties are taken) and cannot tell from the probabilities whether a decision should flip, so it gives only a capped row count. Rows with a different decision and a probability outside the tolerance, and flips over the cap, are refused as usual. When the loosening applies, `rsi register` appends `(decision mismatches: <rows>)` to its output line.

The defaults of both are filled in only in one place, `reproduction_tolerance` and `max_decision_mismatches` in `rsi/rules.py`.

## `flow` (optional)

The main flow is decided by code (`next_action` in `rsi/flow.py`); only the items below can be set here. The `tune` setting from before training and tuning were merged into one module (the train module, `rsi-train`) (`every_round` or `on_demand`) still passes validation, but no longer has any effect.

- `max_returns`: an integer >= 0, 2 when not written. How many times one module may be sent back at most in a round. The count used is the sum of the `returns` of the module's dispatches this round (a field of the dispatch record, plus 1 each time the dispatch is sent back to its session), plus the number of new dispatches after the round's first, so dispatching again in a new session does not reset the count. While it is not used up, the code's default at a decision point is to send back or dispatch again: an artifact the hand-back gave refused at registration (`handback_refused`, decided on the spot while the hand-back is processed) goes back to the session with the registration's error text, a step still not done after the hand-back (`handback_failed`, including a hand-back that registered nothing) goes back to the session, a trainset that failed the quality gate (`quality_gate_failed`) goes back to data, and a failed train gate (`train_gate_failed`) goes back to this round's train session to search again; once it is used up the default is to ask the user (`ask`): the question's options are the ways out that can change the state (`resume` back to the session, `new_session` a new session, `close_round` close this round, `stop` stop the run, plus `train` and `data` after a failed train gate), and the user chooses one with `rsi answer`; the user's choice is not bound by the limit. For a compare dispatch (③ Compare, report only) the default once it is used up is `skip` instead: the dispatch is closed as skipped and the flow goes on to ④ Deliver; `skip` is also among the options of a question about a compare dispatch. Without dispatch records (a workspace run by hand), the default after a failed quality gate is to ask the user too. The second part of a two-part hand-back (train writing the search after its trial candidates are registered, compare writing the comparison after its baselines are registered, judge writing the plan after this round's decision) is a continuation that is no return the first time; asking again after that is a return to the session, and once the count is used up it turns into `handback_failed`, which asks the user by default (skips, for compare).

- `max_continues`: an integer >= 0, 10 when not written. When a long job's completion marker has `ok` on its first line but the hand-back lists no artifacts yet, the engine carries on the session to have it collect the results (the dispatch record's `continued` plus 1, not a return); one dispatch is carried on automatically this many times at most, and at the limit it is recorded as failed (`capped`), and the default of the decision point `handback_failed` becomes asking the user (for a compare dispatch, `skip`: ③ Compare never stops the run).

- `checkin_minutes`: a number > 0 or null; when not written (or null) there are no check-ins. While a dispatch is in the long-job state (it handed back `long_job`) and its completion marker has not appeared, every that many minutes (from the dispatch's latest hand-back) the engine has the same session do a check-in in the continuation mode `checkin` (the feedback starts with "# Check-in", the dispatch record's `checkins` plus 1, not a return); the agent checks the environment, handles anything wrong first, and while the long job still runs hands back `long_job`: the same completion marker when it did not restart trials, or a new completion marker when it restarted trials and started a new trsi watch. With check-ins on, `rsi run` does not exit but waits in the process for the completion marker or the next check-in.
- `max_checkins`: an integer >= 0, 48 when not written. The most automatic check-ins of one dispatch; at the limit, when another check-in is due, the user is asked (question `checkin`, options `continue` for that many check-ins more, `wait` for no more check-ins and only waiting for the completion marker, `stop` to stop the run). Once the completion marker appears the question is dropped.
- `page_refresh_seconds`: a number > 0, 60 when not written. While `rsi run` waits (for an agent, or for a completion marker with check-ins on), once this many seconds have passed since the progress pages were last written, it takes a GPU snapshot first (`<ws>/work/gpus.json`) and then rewrites the pages. See "While waiting" in engine.md.
- `status_minutes`: a number > 0, 5 when not written. While `rsi run` waits, when it has printed no status line on stderr for this many minutes it prints one more (a heartbeat); a change of state prints one at any time. See "While waiting" in engine.md.

The defaults of all items are filled in only in one place, `flow_settings` in `rsi/rules.py`.

## `decisions` (optional)

At the decision points (`rsi/decisions.py`) code chooses by rule by default. Here, per decision point, an agent can judge instead: the keys are the decision points `train_gate_failed`, `quality_gate_failed`, `handback_refused`, `handback_failed`, and the values `code` (default) or `agent`. With `agent`, the engine has the runner launch a decision agent (`claude -p` plus `--json-schema`) and gives it the situation, the options the limits allow, the code's default and the relevant facts; it must choose one of these options and give a reason (`{"choice", "reason"}`); the code checks it and carries it out, and when the answer is invalid or the runner has no decision agent (`manual`) the code's default is used. When the limits allow only `ask`, no agent is called. Every decision is recorded in STATE's `decisions` and in `<ws>/work/run.jsonl` (`by` is `code`, `agent` or the answering user).

## `runner` (optional)

`rsi run <ws> --runner claude` (the default) runs modules in headless Claude Code sessions; the command is `claude -p --dangerously-skip-permissions --output-format json`, followed by the extra arguments here, then `--session-id <uuid>` for a new session (the engine makes it first and records it in the dispatch record) or `--resume <session id>` when carrying on; the prompt is not put on the command line but written into `prompt-<n>.md` in the output directory and given on standard input; the working directory is this dispatch's output directory. There are two items, `claude` and `code`. The four fields of `claude` are all optional:

- `command`: a list of strings that replaces the default command above (the part before the extra arguments), e.g. to drop `--dangerously-skip-permissions`.
- `args`: a list of strings, extra arguments after the command, e.g. `["--model", "sonnet"]`. Without `--model`, `--model claude-opus-5-5` is added, and without `--effort`, `--effort xhigh`: module agents and decision agents both use Opus 5.5 with effort xhigh by default.
- `timeout_s`: a number >= 1, the longest one agent call may run, in seconds from its start, 21600 (6 hours) when not written; an agent that times out is terminated and this dispatch is recorded as failed.
- `setting_sources`: a list of strings from `user`, `project`, `local`, at least one; when given, `--setting-sources <comma-separated>` is passed and module agents read only the Claude Code settings files of these sources. When not written, the agent reads all settings as you do day to day (your hooks, permissions and MCP servers included); `CLAUDE.md` and memory are not affected by it (see "Runners" in engine.md).

The defaults are filled in only in one place, `rsi/runners/claude.py`.

`code` sets how a code module (`compare`: a fixed command, no agent) is run. Its dispatches always go through the code runner, whatever `--runner` says, except `--runner mock` (see "Runners" in engine.md). Its two fields are optional:

- `commands`: an object whose keys are code modules (`compare`) and whose values are argv lists (at least one non-empty string), the command that replaces the default `<repo root>/modules/<module>/run`, e.g. `{"compare": ["/opt/venv/bin/python", "-m", "my_compare"]}`. A relative path as the first word is taken from the repository root (the command runs in the dispatch's output directory, so give the other words as absolute paths); a bare name is looked up on PATH. The engine appends `--brief <outdir>/brief.json --outdir <outdir> --mode <launch mode>` and, whenever the dispatch has a feedback file for the launch (from the second launch on, and at the first launch of a dispatch a decision made), `--feedback <outdir>/<feedback file>`; the contract of the arguments is in `modules/<module>/SKILL.md`.
- `timeout_s`: a number >= 1, the longest one launch of the command may run, in seconds from its start, 21600 (6 hours) when not written; a command that times out is terminated and the dispatch is recorded as failed.

The defaults are filled in only in one place, `rsi/runners/code.py`. `rsi run` reads `runner.code` once, when it first runs a code module's command, so a change takes effect at the next `rsi run`.

## `stop`

- `target`: the primary metric's target value; null when not set.
- `max_rounds`: the maximum number of rounds, an integer >= 1.
- `patience`: optional, stop after this many rounds in a row without promotion, an integer >= 1; 3 when not written.
- `max_attempts_per_round`: optional, the most attempts per round, an integer >= 1; 3 when not written.

The defaults of the two optional fields are filled in only in one place, `stop_settings` in `rsi/rules.py`, from which the stop check, the registration checks and `rsi status` all take them; the stop settings line of `rsi status` prints the values in effect, marking defaults `(default)`, e.g. `patience 3 (default)`.

The stop check is done by `check_stop` in `rsi/rules.py` (called by `rsi next` and `rsi check-stop`; when this round has a registered evaluation without a promotion decision, `rsi next` first refuses and asks for `rsi promote`, without a stop check): it stops when any one holds: the current best version's primary metric reaches `target` (not higher than it when lower is better), the completed rounds (rounds with a decision, including those closed by `rsi close-round`) reach `max_rounds`, the rounds in a row without promotion reach `patience`, or the use of any resource with a budget reaches it (`budget.compute_hours`, `api_cost`, `agent_cost`, `annotation_items`; see `budget` below and spend.md).

**Attempts per round.** An attempt is a search registered in this round: one dispatch (or one return) of the train module is one search, and the trial candidates it compares do not count separately (nor do an old workspace's train candidates and tune trials). When this round already has `max_attempts_per_round` attempts, registering another search is refused, the error giving the counts (e.g. `3 of 3 attempts used in round 2`) and pointing to `rsi close-round`. The attempts line of `rsi status` shows how many this round used; when they are used up and no candidate of this round passed the train gate, it adds a line suggesting `rsi close-round`.

**Closing a round.** When this round has no candidate to evaluate (e.g. every candidate failed the train gate), `rsi close-round <ws> --reason <reason>` closes it: the round's decision is recorded as `{"decision": "keep", "version": null, "evaluation_id": null, "reason": "closed without evaluation: <reason>", "at": ...}`, and the rounds in a row without promotion go up by 1, so this round counts in `max_rounds` and `patience` too. It is refused when this round already has a decision, has a registered evaluation (except a milestone evaluation; then use `rsi promote`), or the run has stopped. The round's dispatches that failed and whose long jobs may still run (e.g. a check-in failed) are closed first in the same save, and their long jobs are terminated after the save (`long_job_stop`'s `context` is `close`, see engine.md). After that, the plan judge writes has `evaluation_id` null (see `plan` in evaluation.md), and `rsi next` does the stop check, checks the plan and the approval as usual.

**User stop.** Besides the stop conditions above, the user may ask to end the run at any time: `rsi stop <ws> --by <name> --quote <user's own words>` writes STATE's `stopped` (`reason` `user requested: <words>`, plus `by`, `at` and the round), after which, as with any other stop, only `delete` approvals and `record` registrations are allowed. After every stop, `rsi report` produces the run report.

## `budget`

- `compute_hours`: the cap on GPU hours.
- `annotation_items`: the cap on human annotation items.
- `api_cost`: the cap on the dollars of the LLM calls that generate or review data (the data generator and its reviewer, prompt-variant rewrites).
- `agent_cost` (optional): the cap on the dollars of the agent sessions (module and decision agents, `claude -p`).

Each is a number or null; null (or, for `agent_cost`, leaving it out) means no cap. The two dollar caps are separate: `api_cost` never covers agent sessions, and `agent_cost` never covers data LLM calls.

The code counts all four in the spend ledger (STATE's `budget_used`; see `docs/interfaces/spend.md`): GPU hours are the `cost.compute_hours` of registered candidates and of candidates refused at registration (refused ones once per source directory, see "Compute of refused candidates" in `docs/interfaces/engine.md`) plus the `compute_hours` lines of the dispatches' spend files; data LLM dollars and annotation items come from the spend files only; agent dollars are what each agent session's runner reports, counted by the engine. Once the use of any resource with a cap reaches it, `rsi next` stops the run (after the round's decision). When `approvals` includes `over_budget`, a reached cap asks this round's `over_budget` approval before ① a new trainset, ② and ③, and registering a candidate that would take the GPU hours over `compute_hours` needs it too (`rsi run` holds such a hand-back before registering it and asks you). The brief gives each module what is left of each cap (`budget_left`); a module stops its own spending there and asks the user (written into `questions`) before going over.
