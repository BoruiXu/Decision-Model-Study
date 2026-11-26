# Task: elliptic-addr-txhist

This file is the version of `task.json` (version 1, template `task.template.json`) for people; its sections match the fields one to one. The placeholders (upper-case names in angle brackets; the GPU list, the GPU-hour estimates and the two dollar budgets are replaced together with the quotes around them) are the same in both files; fill them in with one command for both (see `examples/elliptic_onejev/README.md`). After any change, update `task.json` to match and raise `version`.

## Decision (`decision`)

- Type (`type`): binary
- Decision instructions (`instructions`): Is this Bitcoin address involved in illicit activity?
- Labels (`labels`) and their meanings (`label_meanings`):
  - `yes`: illicit: the address is involved in illicit activity
  - `no`: licit: the address is not involved in illicit activity
- Positive class (`positive`): `yes`

## Input (`input`)

- Format (`format`): object
- Description (`description`): The transaction history of one Elliptic++ bitcoin address up to its time step: the OneJev txhist state as 01_build_data.sh builds it, kept as built. An object with three top-level keys: entity (what the object describes), summary (an overview of the history) and transactions (the address's transactions).
- Verbatim paths (`verbatim`): none (`[]`): the state holds no request text; data generation and prompt variants may change any transaction field

## Features (`features`)

How a state becomes numbers for ③ Compare's CPU baselines and for data generation.

- Adapter (`adapter`): `examples/elliptic_onejev/features_adapter.py`: the generic model view plus per-role counts and sums of the transactions; only the transaction fields are editable; `repair` recomputes the summary from the transactions and `check` holds the history's consistency rules
- Constraints (`constraints`), run before the adapter's own repair and check:
  - every transaction number is 0 or more (`nonneg`): `steps_ago`, `tx_total_btc`, `tx_fee_btc`, `tx_num_input_addresses`, `tx_num_output_addresses`, `tx_smallest_output_btc`, `tx_largest_output_btc` of each `transactions[]`;
  - whole numbers (`integer`): `steps_ago`, `tx_num_input_addresses`, `tx_num_output_addresses`;
  - in order (`order`): `tx_smallest_output_btc` <= `tx_largest_output_btc` <= `tx_total_btc`

```json
{
  "adapter": "examples/elliptic_onejev/features_adapter.py",
  "constraints": [
    {"nonneg": ["transactions[].steps_ago", "transactions[].tx_total_btc", "transactions[].tx_fee_btc",
                "transactions[].tx_num_input_addresses", "transactions[].tx_num_output_addresses",
                "transactions[].tx_smallest_output_btc", "transactions[].tx_largest_output_btc"]},
    {"integer": ["transactions[].steps_ago", "transactions[].tx_num_input_addresses",
                 "transactions[].tx_num_output_addresses"]},
    {"order": ["transactions[].tx_smallest_output_btc", "transactions[].tx_largest_output_btc",
               "transactions[].tx_total_btc"]}
  ]
}
```

## Data (`data`)

- Raw data sources (`sources`): `<ROWS_DIR>`: the Elliptic++ address-level OneJev txhist rows 01_build_data.sh writes: train.jsonl (time steps 1–29), val.jsonl (30–34), test.jsonl (35–49); val_sub4000.jsonl is not used
- Allowed label sources (`label_sources`): human
- Row description (`rows`): one row per (address, time step): each row of train, val and test.jsonl becomes one row, keeping the upstream id (addr_<time step>_<address>); state as it is (OneJev txhist: entity, summary, transactions); question is the task's question; target as it is (yes is illicit), the upstream meta.y used only to check it; meta.state_id is the address; the time step goes into meta.extra.step (the upstream meta.step, or taken from the id when missing); the upstream meta's y, post43, seen_in_train are not written out; label_source is human, n_obs is 1
  - Expected (`expect`): 291419 rows, of which 14720 `yes` and 276699 `no`
- Split rule (`split`): only each set's row count is set (`sizes`): trainset 1000, valset 500, evalset 500, testset 1000 (debug scale). The rest (by, key, ranges or fractions, stratify, unit, seed) the data module fills in from the data in round 1, recorded as a new version of the task definition (a section "Split rule" is added at the end of this file)

## Data quality checks (`quality_checks`)

- Minimum rows (`min_rows`): 1000
- Minimum positive rows (`min_positive_rows`): 20
- Largest share of generated rows (`max_synthetic_share`): 0.3 (rows the data module generates with dgen, `meta.source` starting with `dgen:`)

## Metrics (`metrics`)

- Primary metric (`primary`): auprc, higher is better (`higher_is_better`: true)
- Other metrics (`others`): brier, log_loss, auc, accuracy, macro_f1
- Baselines that must be compared (`baselines`): majority
- Confidence interval of the delta (`ci`): resamples `resamples` 1000, random seed `seed` 0, confidence level `level` 0.95

## Promotion rule (`promotion_rule`)

- Minimum gain (`min_delta`): 0.01
- Required lower bound of the confidence interval (`ci_lower_min`): null (not required)
- Guards (`guards`): brier, lower is better, regression at most 0.005; log_loss, lower is better, regression at most 0.02

## Stop conditions (`stop`)

- Target (`target`): null (not set)
- Maximum rounds (`max_rounds`): 3
- Rounds in a row without promotion (`patience`): not written, 3
- Maximum attempts per round (`max_attempts_per_round`): not written, 3

## Budget (`budget`)

- Compute hours (`compute_hours`): 8 (training, ③ Compare's GPU baselines and the data module's pool scoring over three rounds)
- Annotation items (`annotation_items`): null
- Data LLM cost in dollars (`api_cost`: the LLM calls that generate or review data and paraphrase prompts): "<API_USD_BUDGET>"
- Agent cost in dollars (`agent_cost`: the module and decision agents' sessions): "<AGENT_USD_BUDGET>"

## Resources (`resources`)

The train command `train_command` is `<REPO_ROOT>/modules/train/train`; onejev-harness's python, ojh, machine config and environment file, the starting config when there is no best version (joey-1's single-GPU default recipe), and the GPU hours one joey-1 training and one elephant-1 training (③ Compare's team recipe) are estimated to take; the GPU numbers allowed:

```json
{
  "train_command": "<REPO_ROOT>/modules/train/train",
  "onejev": {
    "python": "<OJH_PYTHON>",
    "ojh_bin": "<OJH_BIN>",
    "machine": "<OJH_MACHINE>",
    "env_file": "<OJH_ENV_FILE>",
    "default_config": {"base": "joey-1", "recipe": null, "overrides": {}, "seed": 0, "max_steps": null},
    "estimate": {"joey-1": "<JOEY1_GPU_HOURS>", "elephant-1": "<ELEPHANT1_GPU_HOURS>"}
  },
  "gpu_ids": "<GPU_IDS>"
}
```

## Config space (`search_space`) and search settings (`search`)

- Learning rate (`overrides.onejev.lr`): 5e-06, 1e-05, 2e-05
- Warmup fraction (`overrides.onejev.warmup_frac`): 0.03, 0.1
- Seed (`seed`): 0, 1, 2
- Search settings: at most 3 candidates (`max_configs`), probe margin `probe_margin` 0.01, accept margin `accept_margin` 0.01

## Decision policy (`policy`)

Every candidate and every ③ Compare baseline fits a calibration and a decision threshold on its valset predictions, chosen by F1 (`objective`: f1); the other settings take their defaults (`policy_settings`: calibrations none, platt, temperature, isotonic; the rule max_f1; the valset split by time when the data split is by time, else by address).

```json
{"objective": "f1"}
```

## ③ Compare (`compare`)

③ Compare runs between ② Train & tune and ④ Deliver and is a report only, never a gate. Its baselines (`baselines`): logistic regression (`lr`), gradient-boosted trees (`gbdt`) and an MLP (`mlp`) on the features above, on the CPU; joey-1 scored with no training (`zero_shot`) and with 4 labelled trainset examples in the prompt (`few_shot`); and the team's existing recipe (`team_recipe`): elephant-1 with the LoRA recipe `recipes/elephant_lora_1g.yaml` (1 GPU), no changes, seed 0, no step limit, trained once per trainset by the train command. The module skips a GPU baseline when `budget_left.compute_hours` is less than its estimate (`resources.onejev.estimate.elephant-1` for the team recipe). The other settings take their defaults (`compare_settings`):

```json
{
  "baselines": ["lr", "gbdt", "mlp", "zero_shot", "few_shot", "team_recipe"],
  "team_recipe": {"base": "elephant-1", "recipe": "recipes/elephant_lora_1g.yaml", "overrides": {}, "seed": 0,
                  "max_steps": null}
}
```

## Evalset use limit (`eval_use_limit`)

3

## Delivery (`delivery`)

- Reproduction tolerance (`reproduction_tolerance`): 0.0001
- Allowed share of flipped decisions (`max_decision_mismatch_rate`): 0.001

## Flow (`flow`)

- Check-in interval (`checkin_minutes`): 30 minutes
- Most automatic continuations to collect results after a long job ends (`max_continues`): 30

## Calling module agents (`runner`)

```json
{"claude": {"command": ["<CLAUDE_BIN>", "-p", "--dangerously-skip-permissions", "--output-format", "json"]}}
```

## Items needing approval (`approvals`)

task_change, over_budget, use_testset, refresh_sets, plan, delete. `task_change`, `refresh_sets` and `use_testset` always need approval whether listed or not (except the new version that fills in the split rule in round 1).
