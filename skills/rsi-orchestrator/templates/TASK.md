# Task: <task_id>

This file is the version of `task.json` for people; its sections match the fields one to one. After any change, update `task.json` to match and raise `version`.

## Decision (`decision`)

- Type (`type`): binary / choice / score
- Decision instructions (`instructions`): what question does this decision answer?
- Labels (`labels`) and their meanings (`label_meanings`):
  - `<label>`: <meaning>
- Positive class (`positive`, binary only):

## Input (`input`)

The format of `state` (the model's input) in each row of the datasets; each row's `question` derives from the decision above, see "Decision rows" in `docs/interfaces/README.md`.

- Format (`format`): messages / fields / object
- Description (`description`):
- Fields (`fields`, fields format only):
- Verbatim paths (`verbatim`, optional, `[]` when not written): the state paths whose text is a row's own request, e.g. `messages[].content` (dots between keys, `[]` for every item of a list); data generation and prompt variants never change that text. All instructions, label meanings and state keys are in English.

## Features (`features`, optional)

How a row's `state` becomes numbers for ③ Compare's CPU baselines and for the data module's generation; without the section the generic views are used.

- Adapter (`adapter`, optional): a Python file (standard library only, a path relative to the repository root or absolute) that may define `features`, `edit_view`, `apply`, `repair`, `check` and `describe`; what it leaves out is generic:
- Constraints (`constraints`, optional, `[]` when not written): declarative rules on the state's numbers, each one of `nonneg`, `integer`, `sum` + `of`, `order`, `zero_if_zero` + `then` (grammar in "features" of `docs/interfaces/task.md`); `dset repair` and data generation repair and check rows by them:

## Data (`data`)

- Raw data sources (`sources`):
- Allowed label sources (`label_sources`): rule / model_judge / human / outcome
- Row description (`rows`, optional): what a row is and where each field comes from; the expected total rows and rows per label (`expect`, optional):
- Split rule (`split`, optional, may be written in part): by (time / group / random), key, ranges or fractions, sizes, stratify, unit, seed; what is not written the data module fills in from the data (before the evalset and the sealed testset are registered) and records as a new version of the task definition (a section "Split rule (`data.split`)" is added at the end of this file, written by `rsi.datasplit.fill_task_md`):

## Data quality checks (`quality_checks`, optional)

The checks and thresholds of the data module's quality gate (e.g. a minimum sample size, a maximum near-duplicate rate, a minimum label reliability, the largest share of generated rows `max_synthetic_share`); when not written, the data module sets its own:

## Metrics (`metrics`)

The metrics `rsi` computes by fixed definitions and checks at registration: `accuracy`, `log_loss` (lower is better), `macro_f1`, `auc` (binary only), `auprc` (binary only), `brier` (lower is better, binary only), `f1` (binary only), `expected_cost` (lower is better, needs `metrics.costs`); baselines: `majority` (the trainset prior), `group_prior` (the trainset prior grouped by `meta.state_id`). Definitions in "Metric definitions" of `docs/interfaces/task.md`. Other names may be written too, but `rsi` neither computes nor checks them; the modules compute them and explain the method.

- Primary metric (`primary`), and whether higher is better (`higher_is_better`; for a known metric it must match the metric's direction):
- Other metrics (`others`):
- Baselines that must be compared (`baselines`):
- Confidence interval of the delta (`ci`, optional): resamples `resamples` (1000 when not written), random seed `seed` (0 when not written), confidence level `level` (0.95 when not written)

## Promotion rule (`promotion_rule`)

- Minimum gain (`min_delta`):
- Required lower bound of the confidence interval (`ci_lower_min`):
- Guards (`guards`): metric, direction, largest regression allowed (`max_regression`)

## Decision policy (`policy`, optional)

With this section every candidate and every ③ Compare baseline fits a calibration and a decision rule on its valset predictions (valset labels only), and its `model.predict` applies them; without it nothing is fitted. Every field is optional (`{}` takes all the defaults; details in "policy" of `docs/interfaces/task.md`):

- Objective (`objective`): the metric the policy is chosen by (`f1` for binary, `log_loss` otherwise when not written; `expected_cost` needs `metrics.costs`):
- Calibrations (`calibrations`), decision rules (`rules`), how the valset is split to fit and select (`split`), the smallest accepted improvement (`min_lo`):

## ③ Compare (`compare`, optional)

With this section ③ Compare runs between ② Train & tune and ④ Deliver: it trains baselines once per trainset, reports the candidate against each on the valset and has them predict the evalset for the judge; a report only, never a gate. Without it ③ is off. Details in "compare" of `docs/interfaces/task.md`.

- Baselines (`baselines`, required): from `lr`, `gbdt`, `mlp` (CPU, on the features), `zero_shot`, `few_shot` (the base model without training, and with labelled examples in its prompt) and `team_recipe` (the team's existing training recipe):
- The team's recipe (`team_recipe`, with `team_recipe` in the baselines): `{"base", "recipe", "overrides", "seed", "max_steps"}`, like a candidate's config:
- Other settings (optional): `seed`, `python`, `max_train_rows`, `grids`, `zero_shot`, `few_shot`, `gpu_hours`:

## Stop conditions (`stop`)

- Target (`target`):
- Maximum rounds (`max_rounds`):
- Rounds in a row without promotion (`patience`, optional, 3 when not written):
- Maximum attempts per round (`max_attempts_per_round`, optional, 3 when not written; one search counts as one attempt, and the trial candidates in a search do not count separately):

The user can ask to end the run at any time. Every stop delivers the run report; when no version was promoted, what is delivered is a failure report stating each round's conclusions.

## Budget (`budget`)

- GPU hours (`compute_hours`):
- Annotation items (`annotation_items`):
- Data LLM cost in dollars (`api_cost`: the LLM calls that generate or review data and paraphrase prompts):
- Agent cost in dollars (`agent_cost`, optional: the module and decision agents' sessions):

Note: `rsi` keeps a spend ledger of all four (`docs/interfaces/spend.md`): GPU hours of candidates and of the work modules report in their spend files, data LLM dollars and annotation items from the spend files, agent dollars as each agent session reports them; the two dollar budgets are separate. Once any budget is reached the run stops after the round's decision, and with the `over_budget` approval going on needs your approval first.

## Resources (`resources`)

The machines, models, quotas and so on available. The location of the train command goes into `train_command` (optional, an absolute path); when not written, `<repo root>/modules/train/train` is used; that is, `resources.train_command`:

## Evalset use limit (`eval_use_limit`)

How many times one evalset may be used at most:

## Delivery (`delivery`, optional)

- Reproduction tolerance (`reproduction_tolerance`, 1e-6 when not written): how far the delivery package's probabilities on the valset inputs may be from the candidate's at most; raise it, e.g. to 1e-4, when the model runs on a GPU in low precision (e.g. bf16) and the batch size varies:
- Allowed share of flipped decisions (`max_decision_mismatch_rate`, from 0 up to but not including 1, 0 when not written): the largest share of valset rows whose probabilities are within the tolerance but whose decision flipped near the decision boundary (a cap of the row count times this share, rounded down):

## Items needing approval (`approvals`)

`task_change`, `refresh_sets` and `use_testset` guard fixed rules: they always need approval, each approval is valid for one registration only, and they need not be checked here.

This list decides only whether the three items below need approval; check: plan / over_budget / delete
