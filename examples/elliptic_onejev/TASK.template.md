# 任务说明：elliptic-addr-txhist

本文件是 `task.json`（版本 1，模板 `task.template.json`）的人读版本，章节与字段一一对应。占位符（尖括号里的大写名字；卡号列表和 GPU 小时估计这两个连两边的引号一起换）在两个文件里相同，填的时候用同一条命令一起改（见 `examples/elliptic_onejev/README.md`）。改动后必须同步 `task.json`，并升 `version`。

## 决策（`decision`）

- 类型（`type`）：binary
- 决策说明（`instructions`）：Is this Bitcoin address involved in illicit activity?
- 标签（`labels`）与含义（`label_meanings`）：
  - `yes`：illicit: the address is involved in illicit activity
  - `no`：licit: the address is not involved in illicit activity
- 正类（`positive`）：`yes`

## 输入（`input`）

- 格式（`format`）：object
- 说明（`description`）：The transaction history of one Elliptic++ bitcoin address up to its time step: the OneJev txhist state as 01_build_data.sh builds it, kept as built. An object with three top-level keys: entity (what the object describes), summary (an overview of the history) and transactions (the address's transactions).

## 数据（`data`）

- 原始数据来源（`sources`）：`<ROWS_DIR>`：01_build_data.sh 写的 Elliptic++ 地址级 OneJev txhist 行：train.jsonl（时间步 1–29）、val.jsonl（30–34）、test.jsonl（35–49）；val_sub4000.jsonl 不用
- 允许的标签来源（`label_sources`）：human
- 行说明（`rows`）：一行一个（地址，时间步）：train、val、test.jsonl 的每一行写成一行，id 沿用上游的（addr_<时间步>_<地址>）；state 原样（OneJev txhist：entity、summary、transactions）；question 用任务的问题；target 原样（yes 是 illicit），上游 meta.y 只用来核对它；meta.state_id 是地址；时间步写进 meta.extra.step（上游 meta.step，没有时从 id 取）；上游 meta 的 y、post43、seen_in_train 不写出；label_source 是 human，n_obs 是 1
  - 预期（`expect`）：291419 行，其中 `yes` 14720 行、`no` 276699 行
- 切分规则（`split`）：只定了各集合的行数（`sizes`）：trainset 1000、valset 500、evalset 500、testset 1000（调试规模）。其余（by、key、ranges 或 fractions、stratify、unit、seed）由数据模块在第一轮按数据补全，记成任务定义的新版本（本文件末尾会多一节「切分规则」）

## 数据质量检查（`quality_checks`）

- 最少行数（`min_rows`）：1000
- 最少正例行数（`min_positive_rows`）：20

## 指标（`metrics`）

- 主指标（`primary`）：auprc，越高越好（`higher_is_better`: true）
- 其他指标（`others`）：brier、log_loss、auc、accuracy、macro_f1
- 必须对比的基线（`baselines`）：majority
- 差值置信区间（`ci`）：抽样次数 `resamples` 1000、随机种子 `seed` 0、置信水平 `level` 0.95

## 晋升规则（`promotion_rule`）

- 最小提升（`min_delta`）：0.01
- 置信区间下界要求（`ci_lower_min`）：null（不要求）
- 护栏（`guards`）：brier 越低越好，最多退步 0.005；log_loss 越低越好，最多退步 0.02

## 停止条件（`stop`）

- 目标值（`target`）：null（不设）
- 最大轮数（`max_rounds`）：1
- 连续无晋升轮数（`patience`）：不写，为 3
- 每轮最多尝试次数（`max_attempts_per_round`）：不写，为 3

## 预算（`budget`）

- 算力小时（`compute_hours`）：8
- 标注条数（`annotation_items`）：null
- API 费用（`api_cost`）：null

## 资源（`resources`）

训练命令 `train_command` 是 `<REPO_ROOT>/modules/train/train`；onejev-harness 的 python、ojh、机器配置和环境文件，没有最佳版本时的起点配置（joey-1 单卡默认配方），joey-1 一次训练的 GPU 小时估计；允许使用的卡号：

```json
{
  "train_command": "<REPO_ROOT>/modules/train/train",
  "onejev": {
    "python": "<OJH_PYTHON>",
    "ojh_bin": "<OJH_BIN>",
    "machine": "<OJH_MACHINE>",
    "env_file": "<OJH_ENV_FILE>",
    "default_config": {"base": "joey-1", "recipe": null, "overrides": {}, "seed": 0, "max_steps": null},
    "estimate": {"joey-1": "<JOEY1_GPU_HOURS>"}
  },
  "gpu_ids": "<GPU_IDS>"
}
```

## 配置空间（`search_space`）和搜索设置（`search`）

- 学习率（`overrides.onejev.lr`）：5e-06、1e-05、2e-05
- 预热比例（`overrides.onejev.warmup_frac`）：0.03、0.1
- 种子（`seed`）：0、1、2
- 搜索设置：最多 3 个候选（`max_configs`），探针门槛 `probe_margin` 0.01，接受门槛 `accept_margin` 0.01

## 评测集使用上限（`eval_use_limit`）

3

## 交付（`delivery`）

- 复现容差（`reproduction_tolerance`）：0.0001
- 允许的决策翻转比例（`max_decision_mismatch_rate`）：0.001

## 流程（`flow`）

- 定时检查间隔（`checkin_minutes`）：30 分钟
- 长任务结束后自动接着收集结果的次数上限（`max_continues`）：30

## 模块 agent 的调用（`runner`）

```json
{"claude": {"command": ["<CLAUDE_BIN>", "-p", "--dangerously-skip-permissions", "--output-format", "json"]}}
```

## 需要审批的事项（`approvals`）

task_change、over_budget、use_testset、refresh_sets、plan、delete。`task_change`、`refresh_sets`、`use_testset` 无论是否列出都要审批（第一轮补全切分规则的新版本除外）。
