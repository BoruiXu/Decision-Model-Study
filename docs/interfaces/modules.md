# 模块接口汇总

顶层和四个模块的输入、输出、门汇总。主流程是 ① 数据 → ② 训练调优 → ③ 交付 → ④ 评测与决策 → 顶层核对 → 停止与交付；训练与调优合并成一个模块（训练调优，模块名 `train`，skill `rsi-train`），旧工作区里的 `tune` 派出和产物照常读，但不再派出。主流程由代码决定（`rsi/flow.py`，见 engine.md），每个模块拿到的输入由代码生成的简报给出（`rsi/briefs.py`），权威来源是各模块 `SKILL.md` 的「输入」「输出」「门」三节；产物字段见 README.md、datasets.md、models.md、delivery.md、evaluation.md、record.md，本文不重复字段表。每个模块 skill 另有「单独运行」一节，说明负责人如何在自己的沙盒里用已登记产物的副本运行模块。每个模块的 skill、代码、参考资料和测试都在自己的文件夹 `modules/<模块>/` 里，引擎派出时把 `modules/<模块>/SKILL.md` 的全文给模块 agent；模块模板见 `modules/_template/SKILL.md`。

## 输入、输出、门

模块都是由引擎派出的 agent（`rsi run`，每次一个会话）：只读简报里列出的产物路径，把产物写进简报给的输出目录（manifest 的 `id` 为 null、`round` 按简报），交回前用 `rsi validate` 校验，报告里的指标、对照组和差值用 `rsi score` 算（定义固定在代码里，见 task.md 的「指标的定义」），不运行其他 `rsi` 命令，不读工作区。登记由引擎做。每个模块每做一次改进实验，另交一份 `record`。模块还可以随时在输出目录写 `progress.json`（可选，简报的 `progress_file`），进度页面和 `rsi run` 的状态行原样显示它，格式见 progress.md。

| 部分 | 输入（产物类别） | 输出（产物类别） | 门 |
|---|---|---|---|
| 顶层 | 用户需求；各模块交回的产物 | `task`；版本登记；停止时的运行报告和交付 | 校验、抽查、登记、升级判定、关闭轮次、审批、停止和运行报告，全部由 `rsi` 命令执行 |
| `rsi-data` | `task`；第一轮 sets 派出的原始数据（简报的 `sources`，只有这次派出读）和续做时已登记的 `corpus`、`pool`；可选 `trainset`、`plan`、`feedback`；之后各轮还有 `valset`、`pool` 和本模块以前各轮的 `record`；获批刷新评测集时有 `corpus` 和以前各 `evalset` 的 `inputs.jsonl` | `trainset`；第一轮另出 `task`（只在补全了切分规则时）、`corpus`、`pool`，第一轮或获批刷新时另出 `testset`、`evalset`、`valset`（含不带标签的 `inputs.jsonl`）；`record`；引擎按 `task`、`corpus`、`testset`、`evalset`、`valset`、`pool`、`trainset` 的顺序登记 | 任务定义 `quality_checks` 规定的数据质量检查（没过的 `trainset` 登记后只作证据，不成为当前训练集）；各集合间无重复；`evalset`、`testset` 登记时移入工作区，不留副本 |
| `rsi-train`（训练调优） | `task`、`trainset`、`valset`；已登记的 `candidate` 和 `search`（按 id）、当前最佳版本的候选（`best_candidate`）、开启本轮的 `plan`；任务的 `search_space`、`search`、`resources.onejev`、`resources.gpu_ids`；本模块以前各轮的 `record`（含旧工作区 tune 的）；剩余预算 | 一次派出是一次搜索，两段交回：先交回本次搜索所有成功试验的 `candidate`（带 `config_request`）和 `record`，登记后同一会话收到「# 继续：交回的第二部分」，再交回 `search` | 训练门：验证集上主指标严格优于任务定义列出的对照组（没过的 `candidate` 不能送评，登记会拒绝它的 `delivery`）；指标、对照组用 `rsi score` 算，登记时重算核对；候选带可直接运行的预测入口 `model.predict`、训练代码版本 `code_version`（`entry` 是任务的训练命令）和完整解析后的 `config`，`config_request` 在任务的 `search_space` 里。调优门：最佳配置在验证集上不差于默认配置，做不到则交回默认配置的结果（`rsi validate` 核对最佳不差于默认、增益等于两者之差）；试验的 `metric_val` 取自候选的主指标（登记时核对）；所有试验候选出自同一版训练代码（`code_version.sha256` 相同，登记时核对） |
| `rsi-deliver` | `task`、`candidate`、`evalset` 的 `inputs.jsonl`、候选所用 `valset` 的 `inputs.jsonl`、本模块以前各轮的 `record`；获批时加 `testset` 的 `inputs.jsonl` | `delivery`、可选 `feedback`、`record` | 模型包能加载；交付包在验证集输入上的预测复现候选的 `predictions_val`（登记时逐行核对）；交付包在交付目录里或是带 sha256 的 `external` 条目；不接触标签 |
| `rsi-judge` | `task`、`candidate`、`delivery`、有 `search` 时的 `search`、`evalset`（含标签）、送评候选所用的 `trainset`（只用来算对照组）、当前最佳版本的 `evaluation` 和 `delivery`、`feedback`、全部 `record`，评测时另有开启本轮的 `plan`（判断它是否奏效）；获批时加 `testset`。本轮由 `rsi close-round` 关闭时没有 `delivery`，给本轮全部 `candidate`（含训练门没过的）、`search` 和 `record` | `evaluation`、`plan`、`record`；本轮关闭时只出 `plan` 和 `record`，`plan` 的 `evaluation_id` 为 null，问题和证据来自本轮候选的训练门结果和实验记录 | 指标、对照组、差值和区间用 `rsi score` 算，登记时重算核对；建议与升级规则一致；归因只取规定的值；计划的每个问题有证据和唯一归因 |

`candidate` 的 `config_source` 取 "default" 或某个已登记 `candidate`、`search` 的 id（旧工作区还有 "tune"，见 models.md）。

**训练调优的一轮。** 本轮的 ② 完成 = 本轮登记了 `search`，且至少一个候选过了训练门；尝试次数只按 `search` 计（`stop.max_attempts_per_round`）。一次搜索的试验都没过训练门时，决策点 `train_gate_failed` 的选项是 `train`（退回同一个 train 会话再搜一次，算一次退回）、`data`（派 rsi-data 出新训练集）和 `ask`；尝试用完时关闭本轮。训练在后台跑时以 `long_job` 交回；任务的 `flow.checkin_minutes` 开了定时检查时，引擎按时让同一会话检查环境（反馈开头「# 定时检查」，不算退回，见 task.md 的 `flow`）。

## 数据可见性

与 datasets.md 的「谁能看到标签」一致，这里按模块列：

| 集合 | data | train | deliver | judge |
|---|---|---|---|---|
| `trainset` | 产出 | 读（含标签） | 不读 | 读送评候选所用的（只算对照组） |
| `valset` | 第一轮产出，之后读 | 读（含标签） | 只读 `inputs.jsonl` | 不读 |
| `evalset` | 第一轮或刷新时产出，交出后不再保留；刷新时只读以前各评测集的 `inputs.jsonl` | 不读 | 只读 `inputs.jsonl` | 读（含标签） |
| `testset` | 第一轮产出，交出后不再保留 | 不读 | 获批时只读 `inputs.jsonl` | 获批时读（含标签） |

## 训练命令

`rsi-train` 提供一个不需要 agent 参与的训练命令。位置：任务定义的 `resources.train_command`（可选，绝对路径或相对仓库根目录的路径）给了就用它，没给就是 `<repo root>/modules/train/train`（`<repo root>` 来自简报，简报的「训练命令」一行写明实际路径）。调用方运行它时环境里带 `PYTHONPATH=<repo root>`，它据此运行 `python3 -m rsi score`。它由 `rsi-train` 的负责人实现，用 trsi 的同一套代码训练一个配置：`rsi-train` 的候选的 `code_version.entry` 都写它，负责人手动单次训练时调用它。`rsi-data` 不训练，不调用它：新训练集好不好，由本轮的 `rsi-train` 在它上面训练、`rsi-judge` 在评测集上和当前最佳版本比较来判断。契约写在 `modules/train/SKILL.md` 的「训练命令（接口契约）」一节：

- 输入：任务定义、训练集目录、验证集目录、配置文件，另有产物 id、`config_source`、轮次、`--producer`（只接受 `train`；旧工作区的候选里还有 `data`，照常能读，登记会拒绝新的）、候选目录和 `--marker` 参数。
- 输出：候选目录本身是一个 `candidate` 产物目录（`id` 为 null），通过 `rsi validate candidate`；带可执行的预测入口 `model.predict`（调用方式见 models.md）、`code_version`（训练命令的路径和 sha256）、完整解析后的 `config` 和所要求的 `config_request`。
- 完成标记：`--marker` 给的文件，在候选目录之外，第一行 ok 或 failed；退出码 0 当且仅当 ok。
- 可无人值守地在后台运行；只读训练集和验证集。
- 进度：写进 `--out` 所在目录的 `progress.json`（见 progress.md）。

配置文件的格式由 `rsi-train` 的负责人定。
