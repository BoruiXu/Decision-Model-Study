# task：任务定义

`task.json` 是任务定义，也是 `task` 类产物的 manifest；`TASK.md` 是同一份内容的人读版本（模板见 `skills/rsi-orchestrator/templates/TASK.md`）。校验由 `rsi` 的任务检查完成，规则在 `rsi/schemas/task.json` 和 `rsi/rows.py`。所有层级都不允许出现未列出的字段。

## 顶层字段

- `kind`：固定为 `"task"`。
- `task_id`：任务标识，匹配 `[a-z0-9][a-z0-9-]*`。
- `version`：任务定义版本，整数 >= 1。改任务定义要升版本，且需要 `task_change` 审批。
- `decision`：决策定义。
- `input`：输入格式。
- `data`：数据来源。
- `metrics`：指标。
- `promotion_rule`：候选晋升规则。
- `stop`：停止条件。
- `budget`：预算。
- `resources`：可用资源，任意 object，由 orchestrator 和各模块按需读取。其中 `resources.train_command`（可选，字符串，绝对路径）是训练命令的位置，没给时用 `<repo root>/skills/rsi-train/bin/train`；调用方运行它时环境里带 `PYTHONPATH=<repo root>`。
- `eval_use_limit`：同一评测集最多使用的次数，整数 >= 1。
- `quality_checks`：可选，数据质量门的检查项和阈值，任意 object，见下。
- `delivery`：可选，交付的设置，见下。
- `approvals`：需要用户审批的事项，取值来自 `task_change`、`over_budget`、`use_testset`、`refresh_sets`、`plan`、`delete`。其中 `task_change`、`refresh_sets`、`use_testset` 守护固定规则（任务定义、冻结的验证集和评测集、封存 test），无论是否列出都必须审批，且每次审批只对一次登记有效；本列表只决定 `plan`、`over_budget`、`delete` 是否需要审批。

## `decision`

- `type`：`binary`、`choice` 或 `score`。
- `instructions`：给人和模型看的决策说明。
- `labels`：标签列表，至少 2 个字符串。`score` 的标签必须是 `"0"` 到 `"k-1"` 依次排列。
- `label_meanings`：标签到含义的映射，键集合必须等于 `labels`。
- `positive`：正类标签。`binary` 必填且必须在 `labels` 中；其他类型不允许出现。

## `input`

- `format`：`messages` 或 `fields`。
- `description`：对输入的说明。
- `fields`：字段名列表。`format` 为 `fields` 时必填且非空；为 `messages` 时不允许。

## `data`

- `sources`：可用的原始数据来源描述，字符串列表，可以为空。
- `label_sources`：允许的标签来源，取值 `rule`、`model_judge`、`human`、`outcome`。

## `metrics`

- `primary`：主指标名。
- `higher_is_better`：主指标是否越高越好。主指标是下面「指标的定义」里的已知指标时，必须与它的方向一致（`log_loss` 为 false，其余为 true），护栏里的已知指标同样如此，否则任务检查报错。
- `others`：其他要报告的指标。
- `auc` 只对 `binary` 任务有定义：决策类型不是 `binary` 时，主指标、`others` 或护栏里写了 `auc`，任务检查报错（错误信息写明 auc 只用于二分类任务）。
- `baselines`：必须对比的基线（对照组），至少 1 个。
- `ci`：可选，差值置信区间的设置 `{"resamples", "seed", "level"}`，三项都可省略：`resamples` 为整数 >= 100，不写时为 1000；`seed` 为整数，不写时为 0；`level` 在 0 和 1 之间（不含两端），不写时为 0.95。默认值只在 `rsi/metrics.py` 的 `ci_settings` 一处补上。

## 指标的定义

已知指标、已知基线和区间的算法是固定代码（`rsi/metrics.py`），不由模块各自决定。模块报告的数用 `rsi score` 算（见 README 的命令一览），登记时 `rsi register` 用同样的定义重算，差超过 1e-9 就拒绝登记，错误信息给出字段、报告值、重算值和该用的 `rsi score` 命令。升级判定和训练门因此都建立在代码核对过的数上。

下面的「硬标签」指 `target` 里概率最大的标签，并列时取 `decision.labels` 里靠前的那个。预测行和标签行按 `id` 对应，预测必须恰好覆盖标签行的 id。求和用 `math.fsum`，结果与行的顺序和 Python 版本无关。

| 已知指标 | 方向 | 定义 |
|---|---|---|
| `accuracy` | 越高越好 | 预测行的 `decision` 等于硬标签的行所占比例 |
| `log_loss` | 越低越好 | 每行取 −ln p 的平均，p 是预测给硬标签的概率，截到 [1e-12, 1] |
| `macro_f1` | 越高越好 | 对 `decision.labels` 的每个标签算 F1（`decision` 对硬标签），取平均；某个标签既没有真实行也没有被预测时它的 F1 记 0，仍参与平均 |
| `auc` | 越高越好 | 只用于 `binary`：以预测给 `decision.positive` 的概率为分数、硬标签是否为正类为真值的 ROC AUC，分数相同的正负对记 0.5；标签行里没有正类或没有负类时无定义，登记和 `rsi score` 会报错 |

| 已知基线 | 定义 |
|---|---|
| `majority` | 对每一行都预测训练集的先验，即训练行 `target` 的平均；`decision` 取先验里概率最大的标签（并列按标签顺序） |
| `group_prior` | 对每一行预测它的 `meta.group` 在训练行里的先验；训练行里没有这个分组时用整体先验。评测集的标签行没有 `meta.group`，分组取自同一集合的 `inputs.jsonl` |

基线的指标与模型用同一套定义：对每个已知基线，用它的预测行算同样的已知指标。候选的基线用它的 `trainset_id` 指的训练集算，评测里的基线用被评测候选的训练集算。

**差值和区间。** `deltas[主指标]` 是「候选 − 当前最佳」：`delta` 是两份预测在同一批标签行上的主指标之差；区间用配对簇自助法：按 `meta.cluster` 把行分簇（没有 cluster 的行自成一簇），簇按名字排序；共抽 `resamples` 次，每次用 `random.Random(seed).randrange` 有放回地抽出与簇数相同个数的簇，在抽中的行上重算两份预测的指标之差（某次抽样上指标无定义时跳过这次，例如 `auc` 抽到的行只有一类）；`ci_low`、`ci_high` 是这些差值的 (1 − level)/2 和 (1 + level)/2 分位数，分位数在相邻的两个排序值之间线性插值（位置 (n − 1) × q）。评测集的簇取自评测集的 `inputs.jsonl`。同样的输入在 Python 3.9 和 3.14 下得到同样的结果。

**未知名字。** 任务可以列出不在上表里的指标或基线（例如 `expected_cost`、某个自定义对照组），`rsi` 不计算、不核对它们，由模块自己算、在报告里说明算法。登记时它们被跳过，`rsi register` 在输出行末尾加 `(not verified: <名字>)`。主指标是未知指标时，训练门、差值和区间都不核对；列出的基线里有未知基线时，训练门只按已知基线核对（见 models.md）。

## `promotion_rule`

- `min_delta`：候选在主指标上相对当前最佳要提升的量，提升必须大于它；恰好等于它（容差 1e-9 内的平局）不算。
- `ci_lower_min`：提升置信区间下界的最小值；不要求时为 null。
- `guards`：护栏列表。每项含 `metric`（必须是主指标或 `others` 中的指标）、`higher_is_better`、`max_regression`（>= 0，允许的最大退步）。

判定由 `rsi/rules.py` 的 `decide_promotion` 完成，它读的数在登记时已按「指标的定义」核对过。没有当前最佳版本时，候选的 `train_gate.passed` 为真就升级。有当前最佳版本时，设 `s` 在主指标越大越好时为 1、越小越好时为 -1：改进量 `s × deltas[主指标].delta` 必须大于 `min_delta`；`ci_lower_min` 非 null 时，改进区间下界（`s = 1` 取 `ci_low`，`s = -1` 取 `-ci_high`）不小于它；每个护栏的退步量（越大越好时 `best − candidate`，越小越好时 `candidate − best`，取自评测的 `metrics`）不超过 `max_regression`。所需数值缺失时保留（`keep`）。

## `quality_checks`（可选）

数据模块质量门的检查项和阈值，例如最少样本量、近似重复率上限、标签信度下限，写法由任务自定（例如 `{"min_rows": 2000, "max_near_dup_rate": 0.01}`）。data 模块读取它，逐项算出数值，和阈值一起写进 trainset 的 `quality.checks`。框架只检查它是 object，不解释其中的内容。不写时由 data 模块按任务自定检查项，并在报告里说明。

## `delivery`（可选）

- `reproduction_tolerance`：number，大于 0，不写时为 1e-6。登记 delivery 时，交付包在验证集输入上的预测要复现候选的 `predictions_val`：每个概率相差不超过这个值（见 delivery.md）。模型在 GPU 上用 bf16 等低精度、以不同的 batch 大小运行时，概率可能差出 1e-6，这时按实际精度调大，例如 1e-4。
- `max_decision_mismatch_rate`：number，至少 0 且小于 1，不写时为 0。概率在容差之内、决策却不同的行最多允许多少：上限是 floor(这个值 × 验证集行数)。靠近决策边界的行，概率只要动一点决策就会翻，容差调到多大都可能出现，这一项就是对它们的放宽；`rsi` 不知道模型的决策规则（阈值、并列时怎么取），无法从概率推出决策该不该翻，所以只给一个封顶的行数。决策不同且有概率超出容差的行、超过上限的翻转都照常拒绝。放宽生效时 `rsi register` 在输出行末尾加 `(decision mismatches: <行数>)`。

两项的默认值只在 `rsi/rules.py` 的 `reproduction_tolerance` 和 `max_decision_mismatches` 一处补上。

## `stop`

- `target`：主指标的目标值；不设时为 null。
- `max_rounds`：最大轮数，整数 >= 1。
- `patience`：可选，连续多少轮没有晋升就停止，整数 >= 1；不写时为 3。
- `max_attempts_per_round`：可选，每轮最多尝试几次，整数 >= 1；不写时为 3。

两个可选字段的默认值只在 `rsi/rules.py` 的 `stop_settings` 一处补上，停止检查、登记检查和 `rsi status` 都从它取值；`rsi status` 的 stop settings 行打印生效的值，用了默认值的标 `(default)`，例如 `patience 3 (default)`。

停止检查由 `rsi/rules.py` 的 `check_stop` 完成（`rsi next` 和 `rsi check-stop` 调用；本轮已登记评测但还没有升级判定时，`rsi next` 先拒绝并要求 `rsi promote`，不做停止检查）：当前最佳版本的主指标达到 `target`（越小越好时为不高于）、已完成的轮数（已有判定的轮次，包括 `rsi close-round` 关闭的轮次）达到 `max_rounds`、连续未升级轮数达到 `patience`、或已用算力小时数达到 `budget.compute_hours`，任一成立即停止。

**每轮的尝试次数。** 一次尝试是本轮登记的一个 `producer` 为 `train` 的 candidate，或一个 search；tune 的试验候选（`producer` 为 `tune`）不单独算，它们的 search 算一次。本轮已有 `max_attempts_per_round` 次尝试时，再登记这样的 candidate 或 search 会被拒绝，错误信息给出次数（例如 `3 of 3 attempts used in round 2`）并指向 `rsi close-round`。`rsi status` 的 attempts 行显示本轮用了几次；次数用完而本轮没有候选通过训练门时，它另起一行提示运行 `rsi close-round`。

**关闭一轮。** 本轮没有可评测的候选时（例如所有候选都没过训练门），用 `rsi close-round <ws> --reason <理由>` 关闭本轮：本轮的判定记为 `{"decision": "keep", "version": null, "evaluation_id": null, "reason": "closed without evaluation: <理由>", "at": ...}`，连续未升级轮数加 1，所以这一轮同样计入 `max_rounds` 和 `patience`。本轮已有判定、已登记了评测（里程碑评测除外，这时要用 `rsi promote`）、或运行已停止时拒绝。之后 judge 写的计划 `evaluation_id` 为 null（见 evaluation.md 的 `plan`），`rsi next` 照常做停止检查、核对计划和审批。

**用户停止。** 除上面的停止条件外，用户随时可以要求结束运行：`rsi stop <ws> --by <name> --quote <用户原话>` 写入 STATE 的 `stopped`（`reason` 为 `user requested: <原话>`，另记 `by`、`at` 和轮次），之后与其他停止一样，只允许 `delete` 审批和登记 `record`。每次停止后用 `rsi report` 生成运行报告。

## `budget`

- `compute_hours`：算力小时数上限。
- `annotation_items`：人工标注条数上限。
- `api_cost`：API 费用上限。

三项都是数字或 null；null 表示不设限。

代码只统计算力：已用算力小时数（STATE 的 `budget_used.compute_hours`，`rsi status` 的 compute hours 行）是已登记 candidate 的 `cost.compute_hours` 之和，别的花费（例如数据生成、打标、推理服务占用的算力）不计入。`approvals` 含 `over_budget` 时，登记一个 candidate 会使它超过 `compute_hours` 时，需要本轮的 `over_budget` 审批；它达到 `compute_hours` 时 `rsi next` 停止运行。`annotation_items` 和 `api_cost` 由顶层按各模块报告的用量累计、在派活前比较，代码不统计也不强制，超出时同样要问用户、记 `over_budget` 审批。
