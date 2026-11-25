# task：任务定义

`task.json` 是任务定义，也是 `task` 类产物的 manifest；`TASK.md` 是同一份内容的人读版本（模板见 `skills/rsi-orchestrator/templates/TASK.md`）。校验由 `rsi` 的任务检查完成，规则在 `rsi/schemas/task.json` 和 `rsi/rows.py`。所有层级都不允许出现未列出的字段。

## 顶层字段

- `kind`：固定为 `"task"`。
- `task_id`：任务标识，匹配 `[a-z0-9][a-z0-9-]*`。
- `version`：任务定义版本，整数 >= 1。改任务定义要升版本，且需要 `task_change` 审批（补全切分规则的新版本除外，只在评测集、封存 test 登记之前，见 `data`）。
- `decision`：决策定义。
- `input`：输入格式，即数据集每一行 `state` 的格式。
- `data`：数据来源。
- `metrics`：指标。
- `promotion_rule`：候选晋升规则。
- `stop`：停止条件。
- `budget`：预算。
- `resources`：可用资源，object，由 orchestrator 和各模块按需读取，见下。
- `search_space`：可选，训练调优模块（`rsi-train`）可以搜索的配置空间，见下。
- `search`：可选，训练调优模块的搜索设置，见下。
- `eval_use_limit`：同一评测集最多使用的次数，整数 >= 1。
- `quality_checks`：可选，数据质量门的检查项和阈值，任意 object，见下。
- `delivery`：可选，交付的设置，见下。
- `flow`：可选，主流程的设置，见下。
- `runner`：可选，`rsi run` 调用模块 agent 的设置，见下。
- `approvals`：需要用户审批的事项，取值来自 `task_change`、`over_budget`、`use_testset`、`refresh_sets`、`plan`、`delete`。其中 `task_change`、`refresh_sets`、`use_testset` 守护固定规则（任务定义、冻结的验证集和评测集、封存 test），无论是否列出都必须审批（补全切分规则的新版本除外，只在评测集、封存 test 登记之前，见 `data`），且每次审批只对一次登记有效；本列表只决定 `plan`、`over_budget`、`delete` 是否需要审批。`delete` 管 orchestrator 和用户手动做的删除；rsi-train 的 `trsi collect` 自动删用不上的 checkpoint（没选中的保存点、失败或被换下的训练的 checkpoint），不管这里有没有 `delete`，都不经审批（用户的决定）。

## `resources`

任意 object；下面几项有固定含义（其余由模块按需读取）：

- `resources.train_command`：可选，字符串，训练命令的位置（绝对路径，或相对仓库根目录的路径）。没给时用 `<repo root>/modules/train/train`；调用方运行它时环境里带 `PYTHONPATH=<repo root>`。给了时，登记 candidate 要求它的 `code_version.entry` 就是这个命令（同一字符串，或相对路径按仓库根目录解析、符号链接解开后是同一个文件），见 models.md。
- `resources.onejev`：可选，object，训练调优模块跑 onejev-harness 用的环境（`python`、`ojh_bin`、`machine`、`env_file`、`default_config`、`estimate` 等，由模块解释）；schema 只检查它是 object。其中 `default_config`（与 candidate 的 `config_request` 同形：`{"base", "recipe", "overrides", "seed", "max_steps"}`）是默认配置，登记时用它核对 `search_space` 没有列出的键（见下）。
- `resources.gpu_ids`：可选，非负整数列表，训练调优模块可用的 GPU 编号（没写就是全部卡），模块每次启动时从中选空闲的。

## `search_space`（可选）

训练调优模块只在这里面搜，用户批准任务定义即批准了这个空间。键是配置路径：`base`、`recipe`、`seed`、`max_steps`、`overrides.onejev.<flag>`、`overrides.lora.<field>`；每个键的值恰好是下面三种之一：

- 允许取值的非空列表；
- `{"min": <数>, "max": <数>}`（两端都允许，`min` 不大于 `max`）；`base`、`recipe`、`seed` 不能写成区间，只能是列表或下面的 `{"values", "dead"}`（训练调优模块逐个试这些取值），写成区间时任务校验就拒绝；
- `{"values": [<允许的取值>, …], "dead": {<取值，按文本写>: <为什么是死路>}}`，`dead` 可选：给模块看的已知死路（死路由 trsi 判断），登记时 `dead` 里的取值仍算在空间里。

`dead` 只能写在某个键里，没有顶层的 `dead`。

任务有 `search_space` 时，登记的每个 candidate 都要有 `config_request`（见 models.md），按下面的规则核对（`rsi/space.py`）：

- 值为 null 的键不核对（等于用配方或训练命令的默认值）。
- `search_space` 里有的键：取值在列表里（或 `values` 里，`dead` 的也算；同一个 JSON 值，`true` 不等于 `1`），或是区间内的数。
- `search_space` 里没有的键：取值必须等于 `resources.onejev.default_config` 在同一路径上的值；任务没给默认配置或默认配置不含这个键时拒绝。没有默认配置的任务用单值列表固定一个设置（例如 `"base": ["joey-1"]`）。
- 拒绝时错误信息写明键、取值和允许的范围。

## `search`（可选）

训练调优模块的搜索设置，由模块解释（框架只检查形状）：`max_configs`（整数 >= 1，一次搜索最多试几个配置）、`seeds`（不重复的整数列表，第一个是试探用的 seed）、`probe_margin`、`accept_margin`（数 >= 0）、`guards`（指标名到允许的最大退化，数 >= 0）。

## `decision`

- `type`：`binary`、`choice` 或 `score`。
- `instructions`：给人和模型看的决策说明。
- `labels`：标签列表，至少 2 个字符串。`score` 的标签必须是 `"0"` 到 `"k-1"` 依次排列。
- `label_meanings`：标签到含义的映射，键集合必须等于 `labels`。
- `positive`：正类标签。`binary` 必填且必须在 `labels` 中；其他类型不允许出现。

数据集的每一行都带着由 `decision` 得出的 OneJev 问题 `question`（`instructions` 就是这里的 `instructions`，`criteria` 来自 `label_meanings`；yes/no 的 `binary` 是 `noul` 问题，`score` 是 `score` 问题，其余是 `choice` 问题），登记时逐行核对，见 README.md 的「决策样本行」。

## `input`

这一节描述模型的输入，即数据集每一行的 `state`（旧工作区的行叫 `input`，见 README.md 的「决策样本行」）。

- `format`：`messages`、`fields` 或 `object`。`messages` 时 `state` 是 `{"messages": [{"role", "content"}, ...]}`（非空）；`fields` 时 `state` 是 `{"fields": {...}}`；`object` 时 `state` 是任意非空 JSON object（例如结构化的 `{"entity", "summary", "transactions"}`），不检查内部结构，原样交给模型。
- `description`：对输入的说明。
- `fields`：字段名列表，`state.fields` 的键必须覆盖它。`format` 为 `fields` 时必填且非空；为 `messages` 或 `object` 时不允许。

## `data`

```json
"data": {
  "sources": [{"path": "/abs/path/raw", "about": "原始数据：每个文件是什么"}],
  "label_sources": ["human"],
  "rows": {"about": "每个（地址，时间步）一行；state 照原样；时间步写进 meta.extra.step",
           "expect": {"rows": 291419, "label_counts": {"yes": 14720, "no": 276699}}},
  "split": {"by": "time", "key": "meta.extra.step",
            "ranges": {"trainset": [1, 29], "valset": [30, 32], "evalset": [33, 34], "testset": [35, 49]},
            "sizes": {"trainset": 1000, "valset": 500, "evalset": 500, "testset": 1000},
            "stratify": true, "unit": "cluster", "seed": 0}
}
```

- `sources`：原始数据来源，列表，可以为空。每项是字符串（旧写法：只是说明，不是路径，不进禁读名单）或 `{"path", "about"}`（`about` 可选）。`path` 是本机绝对路径（文件或目录），不能含 `(`、`)`、`*`、`?`、`[`，不能以 `!` 开头：禁读规则写不出这样的路径，所以任务校验时就拒绝，不等到派出时。这些路径就是「原始数据」。
- `label_sources`：允许的标签来源，取值 `rule`、`model_judge`、`human`、`outcome`。
- `rows`（可选）：全部带标签行的说明。`about`（必填）是给 data agent 的行说明：一行是什么、各字段从哪来。`expect`（可选）是全部行的确切行数 `rows` 和各硬标签的行数 `label_counts`（键必须是任务的标签），`corpus` 登记时逐个核对，对不上就拒绝：标签经过 agent 写的预处理脚本，数目对得上才放行。
- `split`（可选，可以只写一部分）：切分规则，把全部行分到四个集合。
  - `by`：`time`（`key` 的值是数，按 `ranges` 的闭区间分）、`group`（按 `key` 的值整组随机分，按 `fractions`）或 `random`（按单位随机分，按 `fractions`，不写 `key`）。
  - `key`：行里的点路径，例如 `meta.extra.step`、`meta.state_id`。
  - `ranges`：`trainset`、`valset`、`evalset`、`testset` 各一个 `[lo, hi]`（lo ≤ hi），互不重叠；落在所有区间外的行不用。只有 `time` 用它。
  - `fractions`：四个集合各一个 (0, 1] 的数，和不超过 1，剩下的行不用。只有 `group`、`random` 用它。
  - `sizes`：各集合的行数，整数 >= 1；不写或 null 表示范围里的行取完。训练池不受它限制。
  - `stratify`：不写时为 true，`time` 按（`key` 的值，硬标签）分层，其余按硬标签。
  - `unit`：`cluster`（不写时的值）或 `state_id`。同一 `state_id` 可以出现在多个集合里，只报告数目；只有写 `state_id` 时才按它分开，登记时也按它查。
  - `seed`：整数，不写时为 0。

  规则「完整」指：有 `by`；`time`、`group` 有 `key`；`time` 有四个 `ranges`，`group`、`random` 有四个 `fractions`。其余键用默认值。任务校验只查写了的部分是否自洽（`ranges` 只配 `time`，`fractions` 只配 `group`、`random`，`random` 不写 `key`，区间不重叠，比例之和不超过 1），规则由 `rsi/datasplit.py` 检查。

**补全切分规则。** 规则不完整时，data agent 在第一轮 `sets` 派出里按数据补全（time、group 还是 random，范围、大小、分层、单位、seed），用户不审批、也不被问。任务写了的键原样使用（`ranges`、`fractions`、`sizes` 按集合逐个看），和 agent 的选择冲突就报错；补好的规则（默认值也写明）记成任务定义的新版本：`task.json` 版本 +1、只多了补全的 `data.split`，`TASK.md` 是当前版本原文末尾直接接上 `rsi.datasplit.split_section_md(规则)` 的输出，一字不差；写文件用 `rsi.datasplit.fill_task_md(当前 TASK.md 的字节, 规则)`，从按字节读到的原文拼出，不要按文本模式读写（换行符 CRLF 会被改成 LF，补全就对不上）。这一节以一个换行开头（原文以换行结尾时，标题前正好空一行），然后是标题「## 切分规则（`data.split`）」、一句「由 rsi-data 按数据填写，理由见那次派出的 record。」和写明默认值的规则 JSON（```json 代码块，每个顶层键一行）。登记这个新版本不需要 `task_change`，只要它是**补全**：同一 `task_id`、版本 +1；`task.json` 除 `data.split` 外和当前版本完全相同；当前的 `data.split` 不完整，新的完整，当前写了的键值都保留；`TASK.md` 逐字节等于 `fill_task_md(当前版本的 TASK.md, 新的 data.split)`；工作区从没登记过评测集或封存 test。不是补全的新版本照常要 `task_change`；`task.json` 只改了 `data.split` 却不是补全时，拒绝信息写出不算补全的原因。作者写了完整规则时不会有新版本。

**规则固定。** 评测集或封存 test 登记之后，切分规则在本次运行里固定（refresh 沿用）：任何改了 `data.split` 的新版本都拒绝，有 `task_change` 也不行（两边都完整时按写明默认值之后的规则比较）。

## `metrics`

- `primary`：主指标名。
- `higher_is_better`：主指标是否越高越好。主指标是下面「指标的定义」里的已知指标时，必须与它的方向一致（`log_loss`、`brier` 为 false，其余为 true），护栏里的已知指标同样如此，否则任务检查报错。
- `others`：其他要报告的指标。
- `auc`、`auprc`、`brier` 只对 `binary` 任务有定义：决策类型不是 `binary` 时，主指标、`others` 或护栏里写了它们，任务检查报错（错误信息写明它只用于二分类任务）。
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
| `auprc` | 越高越好 | 只用于 `binary`：同样的分数和真值上的 average precision（PR 曲线下面积的阶梯和）：分数从高到低，每个阈值处「召回率的增量 × 该阈值的精确率」之和；分数相同的行是同一个阈值（与 scikit-learn 的 `average_precision_score` 相同）；标签行里没有正类时无定义，会报错 |
| `brier` | 越低越好 | 只用于 `binary`：每行取 (p − y)² 的平均，p 是预测给 `decision.positive` 的概率，y 在硬标签是正类时为 1，否则为 0 |

| 已知基线 | 定义 |
|---|---|
| `majority` | 对每一行都预测训练集的先验，即训练行 `target` 的平均；`decision` 取先验里概率最大的标签（并列按标签顺序） |
| `group_prior` | 对每一行预测它的分组（`meta.state_id`；旧工作区的行是 `meta.group`）在训练行里的先验；训练行里没有这个分组时用整体先验。评测集的标签行没有 `meta.state_id`，分组取自同一集合的 `inputs.jsonl` |

基线的指标与模型用同一套定义：对每个已知基线，用它的预测行算同样的已知指标。候选的基线用它的 `trainset_id` 指的训练集算，评测里的基线用被评测候选的训练集算。

**差值和区间。** `deltas[主指标]` 是「候选 − 当前最佳」：`delta` 是两份预测在同一批标签行上的主指标之差；区间用配对簇自助法：按 `meta.cluster` 把行分簇（没有 cluster 的行自成一簇），簇按名字排序；共抽 `resamples` 次，每次用 `random.Random(seed).randrange` 有放回地抽出与簇数相同个数的簇，在抽中的行上重算两份预测的指标之差（某次抽样上指标无定义时跳过这次，例如 `auc` 抽到的行只有一类、`auprc` 抽到的行没有正类）；`ci_low`、`ci_high` 是这些差值的 (1 − level)/2 和 (1 + level)/2 分位数，分位数在相邻的两个排序值之间线性插值（位置 (n − 1) × q）。评测集的簇取自评测集的 `inputs.jsonl`。同样的输入在 Python 3.9 和 3.14 下得到同样的结果。

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

## `flow`（可选）

主流程由代码决定（`rsi/flow.py` 的 `next_action`），这里只有下面几项可调。训练与调优合并成一个模块（训练调优，`rsi-train`）之前的 `tune`（`every_round` 或 `on_demand`）仍然能通过校验，但不再起作用。

- `max_returns`：整数 >= 0，不写时为 2。同一模块在一轮里最多退回几次。已用的次数是该模块本轮各次派出的 `returns`（派出记录里的字段，每退回原会话一次加 1）之和，加上本轮第一次派出之后又新派的次数，所以换一个新会话重派不会把次数清零。没用完时，决策点的代码默认是退回或重派：交回的产物登记被拒（`handback_refused`，处理交回时当场决定）带着登记错误原文退回原会话，交回后本步仍没完成（`handback_failed`，包括交回了却没有登记任何产物）退回原会话，训练集没过质量门（`quality_gate_failed`）退回 data，训练门没过（`train_gate_failed`）退回本轮的 train 会话再搜一次；用完后默认问用户（`ask`）：问题的选项是能改变状态的出口（`resume` 退回原会话、`new_session` 新开会话、`close_round` 关闭本轮、`stop` 停止运行，训练门没过时还有 `train`、`data`），用户用 `rsi answer` 选一个；用户的选择不受上限限制。没有派出记录（手工运行的工作区）时，质量门没过的默认也是问用户。两段式交回的后半段（train 的试验候选登记后写 search，judge 在本轮判定后写计划）第一次是不算退回的续接；之后再要就是退回原会话，次数用完时转为 `handback_failed`，默认问用户。

- `max_continues`：整数 >= 0，不写时为 10。长任务的完成标记第一行是 `ok`、交回里却还没有列出产物时，引擎接着原会话让它收集结果（派出记录的 `continued` 加 1，不算退回）；同一次派出最多自动接着这么多次，到了上限这次派出记为失败（`capped`），决策点 `handback_failed` 的默认改为问用户。

- `checkin_minutes`：数 > 0 或 null，不写（或 null）时不做定时检查。派出处于长任务状态（交回 `long_job`）、完成标记还没出现时，引擎每隔这么多分钟（从这次派出最近一次交回算起）用续接模式 `checkin` 让同一个会话做一次定时检查（反馈开头「# 定时检查」，派出记录的 `checkins` 加 1，不算退回）；agent 检查环境、有异常先处理，长任务还在运行就以 `long_job` 交回：如果没重启试验，就交回同一个完成标记；如果重启了试验、开了新的 trsi watch，就交回新的完成标记。开了定时检查时 `rsi run` 不退出，在进程里等完成标记或下一次检查。
- `max_checkins`：整数 >= 0，不写时为 48。同一次派出自动定时检查的次数上限；到了上限、检查又到期时问用户（问题 `checkin`，选项 `continue` 再检查这么多次、`wait` 不再检查只等完成标记、`stop` 停止运行）。完成标记出现后这个问题作废。
- `page_refresh_seconds`：数 > 0，不写时为 60。`rsi run` 在等（等 agent，或开了定时检查时等完成标记）期间，距上一次写进度页面满这么多秒，就先采一次 GPU 快照（`<ws>/work/gpus.json`），再重写页面。见 engine.md「等待期间」。
- `status_minutes`：数 > 0，不写时为 5。`rsi run` 在等期间，这么多分钟没往 stderr 打过状态行就再打一行（心跳）；状态变化时随时打。见 engine.md「等待期间」。

各项的默认值只在 `rsi/rules.py` 的 `flow_settings` 一处补上。

## `decisions`（可选）

决策点（`rsi/decisions.py`）上默认由代码按规则选择。这里可以按决策点改成让 agent 判断：键是决策点 `train_gate_failed`、`quality_gate_failed`、`handback_refused`、`handback_failed`，值是 `code`（默认）或 `agent`。`agent` 时引擎用调用器启动一个决策 agent（`claude -p` 加 `--json-schema`），给它情况、上限允许的选项、代码的默认和相关事实，它必须在这些选项里选一个并写理由（`{"choice", "reason"}`）；代码校验后执行，答案不合格或调用器没有决策 agent（`manual`）时用代码的默认。上限只允许 `ask` 时不调用 agent。每次决策都记进 STATE 的 `decisions` 和 `<ws>/work/run.jsonl`（`by` 是 `code`、`agent` 或回答的用户）。

## `runner`（可选）

`rsi run <ws> --runner claude`（默认）用无界面的 Claude Code 会话执行模块，命令是 `claude -p --dangerously-skip-permissions --output-format json`，再接这里的额外参数，然后是新会话的 `--session-id <uuid>`（引擎先生成并记进派出记录）或续改时的 `--resume <会话 id>`；提示词不放在命令行上，而是写进输出目录的 `prompt-<n>.md` 从标准输入给出；工作目录是这次派出的输出目录。只有一项 `claude`，其中四项都可选：

- `command`：字符串列表，替换上面的默认命令（额外参数之前的部分），例如去掉 `--dangerously-skip-permissions`。
- `args`：字符串列表，接在命令后面的额外参数，例如 `["--model", "sonnet"]`。没写 `--model` 时加 `--model claude-opus-5-5`，没写 `--effort` 时加 `--effort xhigh`：模块 agent 和决策 agent 默认都用 Opus 5.5、effort xhigh。
- `timeout_s`：number >= 1，一次 agent 调用最长运行多少秒，从启动算起，不写时为 21600（6 小时）；超时的 agent 被终止，这次派出记为失败。
- `setting_sources`：字符串列表，取值 `user`、`project`、`local`，至少一个；给了就传 `--setting-sources <逗号分隔>`，模块 agent 只读这些来源的 Claude Code 设置文件。不写时 agent 和你平时一样读全部设置（包括你的 hooks、权限、MCP 服务器）；`CLAUDE.md` 和记忆不受它影响（见 engine.md「调用器」）。

默认值只在 `rsi/runners/claude.py` 一处补上。

## `stop`

- `target`：主指标的目标值；不设时为 null。
- `max_rounds`：最大轮数，整数 >= 1。
- `patience`：可选，连续多少轮没有晋升就停止，整数 >= 1；不写时为 3。
- `max_attempts_per_round`：可选，每轮最多尝试几次，整数 >= 1；不写时为 3。

两个可选字段的默认值只在 `rsi/rules.py` 的 `stop_settings` 一处补上，停止检查、登记检查和 `rsi status` 都从它取值；`rsi status` 的 stop settings 行打印生效的值，用了默认值的标 `(default)`，例如 `patience 3 (default)`。

停止检查由 `rsi/rules.py` 的 `check_stop` 完成（`rsi next` 和 `rsi check-stop` 调用；本轮已登记评测但还没有升级判定时，`rsi next` 先拒绝并要求 `rsi promote`，不做停止检查）：当前最佳版本的主指标达到 `target`（越小越好时为不高于）、已完成的轮数（已有判定的轮次，包括 `rsi close-round` 关闭的轮次）达到 `max_rounds`、连续未升级轮数达到 `patience`、或已用算力小时数达到 `budget.compute_hours`，任一成立即停止。

**每轮的尝试次数。** 一次尝试是本轮登记的一个 search：训练调优模块的一次派出（或一次退回）是一次搜索，它比较的试验候选不单独算（旧工作区里 train 的候选、tune 的试验也不算）。本轮已有 `max_attempts_per_round` 次尝试时，再登记 search 会被拒绝，错误信息给出次数（例如 `3 of 3 attempts used in round 2`）并指向 `rsi close-round`。`rsi status` 的 attempts 行显示本轮用了几次；次数用完而本轮没有候选通过训练门时，它另起一行提示运行 `rsi close-round`。

**关闭一轮。** 本轮没有可评测的候选时（例如所有候选都没过训练门），用 `rsi close-round <ws> --reason <理由>` 关闭本轮：本轮的判定记为 `{"decision": "keep", "version": null, "evaluation_id": null, "reason": "closed without evaluation: <理由>", "at": ...}`，连续未升级轮数加 1，所以这一轮同样计入 `max_rounds` 和 `patience`。本轮已有判定、已登记了评测（里程碑评测除外，这时要用 `rsi promote`）、或运行已停止时拒绝。本轮失败了、长任务可能还在跑的派出（例如定时检查失败了）在同一次保存里先关闭，保存后终止它们的长任务（`long_job_stop` 的 `context` 是 `close`，见 engine.md）。之后 judge 写的计划 `evaluation_id` 为 null（见 evaluation.md 的 `plan`），`rsi next` 照常做停止检查、核对计划和审批。

**用户停止。** 除上面的停止条件外，用户随时可以要求结束运行：`rsi stop <ws> --by <name> --quote <用户原话>` 写入 STATE 的 `stopped`（`reason` 为 `user requested: <原话>`，另记 `by`、`at` 和轮次），之后与其他停止一样，只允许 `delete` 审批和登记 `record`。每次停止后用 `rsi report` 生成运行报告。

## `budget`

- `compute_hours`：算力小时数上限。
- `annotation_items`：人工标注条数上限。
- `api_cost`：API 费用上限。

三项都是数字或 null；null 表示不设限。

代码只统计算力：已用算力小时数（STATE 的 `budget_used.compute_hours`，`rsi status` 的 compute hours 行）是已登记 candidate 和登记被拒的 candidate 的 `cost.compute_hours` 之和（被拒的按来源目录计一次，见 `docs/interfaces/engine.md` 的「被拒候选的算力」），别的花费（例如数据生成、打标、推理服务占用的算力）不计入。`approvals` 含 `over_budget` 时，登记一个 candidate 会使它超过 `compute_hours` 时，需要本轮的 `over_budget` 审批（`rsi run` 在登记这样的交回之前先暂缓、问你）；它达到 `compute_hours` 时 `rsi next` 停止运行。`annotation_items` 和 `api_cost` 代码不统计也不强制：简报把它们列在剩余预算里，模块在交回的 `notes` 里报告用量，要超出时停下问用户（写进 `questions`）。
