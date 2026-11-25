# 任务说明：<task_id>

本文件是 `task.json` 的人读版本，章节与字段一一对应。改动后必须同步 `task.json`，并升 `version`。

## 决策（`decision`）

- 类型（`type`）：binary / choice / score
- 决策说明（`instructions`）：这个决策要回答什么问题？
- 标签（`labels`）与含义（`label_meanings`）：
  - `<标签>`：<含义>
- 正类（`positive`，仅 binary）：

## 输入（`input`）

数据集每一行的 `state`（模型的输入）的格式；每一行的 `question` 由上面的决策得出，见 `docs/interfaces/README.md` 的「决策样本行」。

- 格式（`format`）：messages / fields / object
- 说明（`description`）：
- 字段（`fields`，仅 fields 格式）：

## 数据（`data`）

- 原始数据来源（`sources`）：
- 允许的标签来源（`label_sources`）：rule / model_judge / human / outcome
- 行说明（`rows`，可选）：一行是什么、各字段从哪来；预期的总行数和各标签行数（`expect`，可选）：
- 切分规则（`split`，可选，可以只写一部分）：by（time / group / random）、key、ranges 或 fractions、sizes、stratify、unit、seed；没写的由数据模块按数据补全（评测集、封存 test 登记之前），记成任务定义的新版本（本文件末尾加一节「切分规则」，由 `rsi.datasplit.fill_task_md` 生成）：

## 数据质量检查（`quality_checks`，可选）

数据模块质量门的检查项和阈值（例如最少样本量、近似重复率上限、标签信度下限），不写时由数据模块自定：

## 指标（`metrics`）

`rsi` 按固定定义计算并在登记时核对的指标：`accuracy`、`log_loss`（越低越好）、`macro_f1`、`auc`（仅 binary）、`auprc`（仅 binary）、`brier`（越低越好，仅 binary）；基线：`majority`（训练集先验）、`group_prior`（按 `meta.state_id` 分组的训练集先验）。定义见 `docs/interfaces/task.md` 的「指标的定义」。别的名字也可以写，但 `rsi` 不计算、不核对，由模块自己算并说明算法。

- 主指标（`primary`），是否越高越好（`higher_is_better`，已知指标必须与它的方向一致）：
- 其他指标（`others`）：
- 必须对比的基线（`baselines`）：
- 差值置信区间（`ci`，可选）：抽样次数 `resamples`（不写时 1000）、随机种子 `seed`（不写时 0）、置信水平 `level`（不写时 0.95）

## 晋升规则（`promotion_rule`）

- 最小提升（`min_delta`）：
- 置信区间下界要求（`ci_lower_min`）：
- 护栏（`guards`）：指标、方向、允许的最大退步（`max_regression`）

## 停止条件（`stop`）

- 目标值（`target`）：
- 最大轮数（`max_rounds`）：
- 连续无晋升轮数（`patience`，可选，不写时为 3）：
- 每轮最多尝试次数（`max_attempts_per_round`，可选，不写时为 3；一次搜索（search）算一次，搜索里的试验候选不单独算）：

用户随时可以要求结束运行。每次停止都交付运行报告；一个版本都没有升级时，交付的是说明各轮结论的失败报告。

## 预算（`budget`）

- 算力小时（`compute_hours`）：
- 标注条数（`annotation_items`）：
- API 费用（`api_cost`）：

注意：`rsi` 只把已登记候选模型的 `cost.compute_hours` 计入已用算力，并据此做超预算审批和停止检查；标注条数和 API 费用代码不统计也不强制，模块在交回的报告里写用量，要超出时停下问你。

## 资源（`resources`）

可用的机器、模型、额度等。训练命令的位置写 `train_command`（可选，绝对路径），不写时用 `<repo root>/modules/train/train`；即 `resources.train_command`：

## 评测集使用上限（`eval_use_limit`）

同一评测集最多使用几次：

## 交付（`delivery`，可选）

- 复现容差（`reproduction_tolerance`，不写时为 1e-6）：交付包在验证集输入上的概率与候选的概率最多差多少；模型用低精度（例如 bf16）在 GPU 上跑、batch 大小会变时调大，例如 1e-4：
- 允许的决策翻转比例（`max_decision_mismatch_rate`，0 到 1 之间不含 1，不写时为 0）：概率在容差之内、但靠近决策边界而翻了决策的验证集行，最多占多少（按行数向下取整封顶）：

## 需要审批的事项（`approvals`）

`task_change`、`refresh_sets`、`use_testset` 守护固定规则，总是需要审批，每次审批只对一次登记有效，不必勾选。

本列表只决定以下三项是否需要审批，勾选：plan / over_budget / delete
