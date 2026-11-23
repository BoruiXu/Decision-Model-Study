# 模块接口汇总

顶层和五个模块的输入、输出、门汇总。权威来源是 `skills/rsi-orchestrator/SKILL.md` 的主流程表和各模块 `SKILL.md` 的「输入」「输出」「门」三节；产物字段见 README.md、datasets.md、models.md、delivery.md、evaluation.md、record.md，本文不重复字段表。每个模块 skill 另有「单独运行」一节，说明负责人如何在自己的沙盒里用已登记产物的副本运行模块。模块模板见 `skills/_module_template/SKILL.md`。

## 输入、输出、门

模块都是由顶层派出的 subagent：只读简报里列出的产物路径，把产物写进简报给的输出目录（manifest 的 `id` 为 null、`round` 按简报），交回前用 `rsi validate` 校验，报告里的指标、对照组和差值用 `rsi score` 算（定义固定在代码里，见 task.md 的「指标的定义」），不运行其他 `rsi` 命令，不读工作区。登记只由顶层做。每个模块每做一次改进实验，另交一份 `record`。

| 部分 | 输入（产物类别） | 输出（产物类别） | 门 |
|---|---|---|---|
| 顶层 | 用户需求；各模块交回的产物 | `task`；版本登记；停止时的运行报告和交付 | 校验、抽查、登记、升级判定、关闭轮次、审批、停止和运行报告，全部由 `rsi` 命令执行 |
| `rsi-data` | `task`；可选 `trainset`、`plan`、`feedback`；之后各轮还有 `valset` | `trainset`；第一轮或获批刷新时另出 `testset`、`evalset`、`valset`（含不带标签的 `inputs.jsonl`）；`record` | 任务定义 `quality_checks` 规定的数据质量检查（没过的 `trainset` 登记后只作证据，不成为当前训练集）；各集合间无重复；`evalset`、`testset` 登记时移入工作区，不留副本 |
| `rsi-train` | `task`、`trainset`、`valset`；训练配置（默认，或某个 `search` 的最佳配置） | `candidate`、`record` | 验证集上主指标严格优于任务定义列出的对照组（没过的 `candidate` 不能送评，登记会拒绝它的 `delivery`）；指标、对照组用 `rsi score` 算，登记时重算核对；候选带可直接运行的预测入口 `model.predict`、训练代码版本 `code_version` 和完整解析后的 `config` |
| `rsi-tune` | `task`、`trainset`、`valset`、历史 `candidate` 和 `search`、预算 | `search`、若干 `candidate`（`config_source` 为 "tune"）、`record` | 最佳配置在验证集上不差于默认配置，做不到则交回默认配置的结果（`rsi validate` 核对最佳不差于默认、增益等于两者之差）；试验的 `metric_val` 取自训练命令用 `rsi score` 算出的候选主指标（登记时核对）；所有试验候选出自同一版训练代码（`code_version.sha256` 相同，登记时核对） |
| `rsi-deliver` | `task`、`candidate`、`evalset` 的 `inputs.jsonl`、候选所用 `valset` 的 `inputs.jsonl`；获批时加 `testset` 的 `inputs.jsonl` | `delivery`、可选 `feedback`、`record` | 模型包能加载；交付包在验证集输入上的预测复现候选的 `predictions_val`（登记时逐行核对）；交付包在交付目录里或是带 sha256 的 `external` 条目；不接触标签 |
| `rsi-judge` | `task`、`candidate`、`delivery`、有 `search` 时的 `search`、`evalset`（含标签）、送评候选所用的 `trainset`（只用来算对照组）、当前最佳版本的 `evaluation` 和 `delivery`、`feedback`、全部 `record`；获批时加 `testset`。本轮由 `rsi close-round` 关闭时没有 `delivery`，给本轮全部 `candidate`（含训练门没过的）、`search` 和 `record` | `evaluation`、`plan`、`record`；本轮关闭时只出 `plan` 和 `record`，`plan` 的 `evaluation_id` 为 null，问题和证据来自本轮候选的训练门结果和实验记录 | 指标、对照组、差值和区间用 `rsi score` 算，登记时重算核对；建议与升级规则一致；归因只取规定的值；计划的每个问题有证据和唯一归因 |

`candidate` 的 `config_source` 取 "default"、"tune"（调优试验）或某个已登记 `search` 的 id（见 models.md）。

## 数据可见性

与 datasets.md 的「谁能看到标签」一致，这里按模块列：

| 集合 | data | train | tune | deliver | judge |
|---|---|---|---|---|---|
| `trainset` | 产出 | 读（含标签） | 读（含标签） | 不读 | 读送评候选所用的（只算对照组） |
| `valset` | 第一轮产出，之后读 | 读（含标签） | 读（含标签） | 只读 `inputs.jsonl` | 不读 |
| `evalset` | 第一轮产出，交出后不再保留 | 不读 | 不读 | 只读 `inputs.jsonl` | 读（含标签） |
| `testset` | 第一轮产出，交出后不再保留 | 不读 | 不读 | 获批时只读 `inputs.jsonl` | 获批时读（含标签） |

## 训练命令

`rsi-train` 提供一个不需要 agent 参与的训练命令。位置：任务定义的 `resources.train_command`（可选，绝对路径）给了就用它，没给就是 `<repo root>/skills/rsi-train/bin/train`（`<repo root>` 来自简报，简报的「训练命令」一行写明实际路径）。调用方运行它时环境里带 `PYTHONPATH=<repo root>`，它据此运行 `python3 -m rsi score`。它由 `rsi-train` 的负责人实现，供 `rsi-tune` 的每次试验和 `rsi-data` 的新旧训练集比较调用。契约写在 `skills/rsi-train/SKILL.md` 的「输出」一节：

- 输入：任务定义、训练集目录、验证集目录、配置文件，另有产物 id、`config_source`、轮次、`--producer`（`train`、`tune` 或 `data`）、候选目录和 `--marker` 参数。
- 输出：候选目录本身是一个 `candidate` 产物目录（`id` 为 null），通过 `rsi validate candidate`；带可执行的预测入口 `model.predict`（调用方式见 models.md）、`code_version`（训练命令的路径和 sha256）和完整解析后的 `config`。
- 完成标记：`--marker` 给的文件，在候选目录之外，第一行 ok 或 failed；退出码 0 当且仅当 ok。
- 可无人值守地在后台运行；只读训练集和验证集。
- 数据模块的新旧比较用 `--producer data` 和 `--trainset-id unregistered`，候选是草稿，放在 data 自己输出目录的 `ab/` 下，不交回登记；比较结果写进 data 的 `record`。

配置文件的格式由 `rsi-train` 的负责人定。
