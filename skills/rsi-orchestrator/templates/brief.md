# 派活说明模板

顶层每次派 subagent 时复制下面分隔线以下的部分，替换尖括号里的内容后整份发出。只填该模块输入列出的产物路径和一个输出目录，不写工作区根目录。

---

你负责 RSI 运行里的一个模块。先加载 skill `<skill 名>`，按它的「输入」「输出」「门」工作。

- skill：`<rsi-data、rsi-train、rsi-tune、rsi-deliver 或 rsi-judge 之一>`
- 任务定义：`<task.json 的路径>`，人读版是同一目录的 `TASK.md`
- 本轮轮次：`<n>`。每个产物 manifest 的 `round` 写这个数，`task_id` 写 `<task_id>`，`task_version` 写 `<version>`，`id` 写 null。
- 当前登记 id：trainset `<id 或 无>`，valset `<id 或 无>`，evalset `<id 或 无>`，testset `<id 或 无>`。manifest 里引用这些集合时用这些 id。
- 当前最佳版本（只在派 rsi-judge 时写，其他模块删去这一行）：版本名 `<V 加序号，或 无>`，evaluation `<id>`，delivery `<id>`。`best_before` 按这些值写。评测集刷新过时另填最佳版本在当前评测集上的 delivery `<id 或 无>`：用它算最佳版本一侧的指标，并把它的 id 写进 evaluation 的 `parents`。
- 训练命令：`<resources.train_command 的绝对路径；任务定义没给时写 <repo root>/skills/rsi-train/bin/train>`（只在派 rsi-data、rsi-train、rsi-tune 时写，其他模块删去这一行）。调用它时环境里带 `PYTHONPATH=<repo root>`。
- 本次要做的事：<第一轮按任务定义；之后写计划里分给本模块的问题、要改什么、预期提升>
- 可用预算：<算力小时、标注条数、调用费用>
- 输入产物（只读这些，没列出的产物和目录不要找、不要读）：

  | id | 路径 | 说明 |
  |---|---|---|
  | `<id>` | `<路径>` | <用途> |

- 输出目录：`<输出目录>`，每个产物一个子目录。
- 要报告的门数字：<该模块 skill「门」一节列出的各项>，逐项给出数值、阈值和是否通过，另报告实际花费。
- 报告的指标、对照组、差值和区间：用 `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <标签行文件> --predictions <预测文件>` 算（需要对照组时加 `--train <训练集 rows.jsonl>`；需要和另一份预测比时加 `--compare <另一份预测>`，同时加 `--clusters <inputs.jsonl>` 给出区间用的簇；标签行没有 `meta.group`（例如评测集的 `labels.jsonl`）而要算 `group_prior` 时也要加 `--clusters <inputs.jsonl>`；完整写法见模块 skill），把它打印的数原样写进 manifest。这些数的定义固定在代码里，登记时会按同样的定义重算，不一致就被拒。
- 交回前：对每个产物运行 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>`，通过再交回。你只能运行 `rsi validate` 和 `rsi score` 这两个 `rsi` 命令；不要运行 `rsi register` 或其他 `rsi` 命令，登记只由顶层做。
- 交回内容：每个产物的类别和目录、门数字、花费、`record` 的目录、遇到的问题。一个产物要引用本次另一个产物的登记 id 时（例如 `search` 引用 `candidate`，`plan` 引用 `evaluation`），先交回被引用的那个，等顶层告诉你 id 再补上。
- 遇到需要用户决定的事就停下报告，例如要超出预算、要改任务定义、要用封存 test、要刷新验证集或评测集、要删除产物、输入有问题。不要自己决定，也不要绕过。
- 长任务交回规则：预计超过 <N> 分钟的任务在后台启动（例如 `nohup`、`tmux`），启动后立刻交回完成标记路径、日志路径和预计耗时。任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`。不要等待或轮询，顶层看到标记后会请你收尾。
- 不要派别的 subagent；需要训练时调用 `rsi-train` 提供的训练命令。
