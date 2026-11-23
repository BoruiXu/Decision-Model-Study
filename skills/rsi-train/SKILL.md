---
name: rsi-train
description: 由 rsi-orchestrator 派发训练模块时使用：用训练集和验证集训出候选模型，并提供不需要 agent 参与的训练命令。Use when the orchestrator dispatches the RSI train module; not for general model-training requests.
---

# rsi-train：训练模块

训练决策模型，并提供不需要 agent 参与的训练命令。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`、`trainset`、`valset`。
- 训练配置：默认配置，或某个已登记 `search` 的最佳配置。用后者时简报给出该 `search` 的 id 和路径，候选的 `config_source` 写这个 id；用默认配置时写 "default"。以某个 `search` 的最佳配置为基础再改动时，同样写该 `search` 的 id，并在 `record` 里写明改了什么。
- 不拿评测集和封存 test。

## 输出

- `candidate`：模型、可直接运行的预测入口（model.predict）、完整解析后的训练配置、训练代码的版本（code_version）、所用训练集和验证集的 id、验证集预测和指标、对照组、训练门结果、成本。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [models.md](../../docs/interfaces/models.md)。 `record` 见 [record.md](../../docs/interfaces/record.md)。

### 训练命令（接口契约）

训练命令的位置：任务定义的 `resources.train_command`（可选，绝对路径）给了就用它，没给就是 `<repo root>/skills/rsi-train/bin/train`（`<repo root>` 是简报给的框架仓库根目录）；简报的「训练命令」一行写明实际路径。它由负责人实现。调用方运行它时环境里带 `PYTHONPATH=<repo root>`，训练命令据此运行 `python3 -m rsi score` 算指标，不依赖自己所在的路径。调优的每次试验和数据模块的新旧比较都调用它，所以它是本模块接口的一部分：

```
<训练命令> --task <task.json> --trainset <trainset 目录> --trainset-id <id> --valset <valset 目录> --valset-id <id> --config <配置文件> --config-source <default 或 tune 或 search id> --round <n> --producer <train、tune 或 data> --out <候选目录> --marker <完成标记路径>
```

- 输入：任务定义、训练集目录、验证集目录、配置文件（JSON）；其余参数写进 manifest。
- 输出：候选目录本身就是一个 `candidate` 产物目录，id 为 null，用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 能通过。其中：
  - 预测入口：候选目录里的一个可执行文件，相对路径写在 `model.predict`，调用方式固定为 `<候选目录>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>`（读输入行，写预测行，成功退出码 0，见 [models.md](../../docs/interfaces/models.md)）。模型文件放在候选目录里，放不下的大文件只能通过 manifest 的 external 条目引用。predictions_val 就用这个入口在验证集的 inputs.jsonl 上跑出来，这样交付才能复现它。
  - 训练代码版本：`code_version.entry` 写训练命令的路径，`code_version.sha256` 写训练命令文件的 sha256（例如由命令对自己算出）；entry 是运行校验或登记的那台机器上存在的绝对路径时会核对 sha256；训练在别的主机上跑时 entry 写 `主机:/路径`（不核对），免得登记机器同一路径上的另一份代码把诚实的候选拒掉。只有这一个文件被算 sha256，所以训练代码或默认值一改，这个文件必须跟着变：把代码和默认值写在这一个文件里，或在文件里嵌入随代码变化的版本串（例如 git commit），改代码时一起更新。
  - 配置：manifest 的 config 写完整解析后的配置，训练代码用到的每一个键都写，包括取默认值的键，不只写改动的部分。
- 完成标记：`--marker` 给的文件，必须在候选目录之外（不进产物目录，也不出现在 manifest 的 files 里），最后写，第一行 ok 或 failed，failed 时第二行写原因。退出码 0 当且仅当写了 ok。
- 数据模块的新旧比较也调用它，`--producer data`；训练集还没登记，`--trainset-id` 写 unregistered。这样的候选是草稿，登记会拒绝它，这是有意的，不交回登记。
- 不需要 agent 参与，可在后台无人值守运行；只读两个数据集目录和配置，不读评测集和封存 test。
- 配置文件的格式由负责人写在「步骤和策略子 skill」一节。

## 门

- 训练门：验证集上主指标严格优于任务定义列出的每个对照组（方向按任务定义，相等不算），结果写进 `train_gate`（`passed`、`detail`）。
- 指标和对照组一律用 `rsi score` 算，不自己实现（定义见 [task.md](../../docs/interfaces/task.md) 的「指标的定义」）：`PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <valset>/rows.jsonl --predictions <candidate>/predictions_val.jsonl --train <trainset>/rows.jsonl`，输出的 `metrics` 写进 `metrics_val`，`baselines` 写进 `baselines_val`，`passed` 按这些数判断。训练命令同样调用它。登记时代码按同样的定义重算 `metrics_val`、`baselines_val` 和 `train_gate.passed`，对不上就被拒；输出的 `not_computed` 里的名字由本模块自己算，在报告里写明算法。
- 候选的预测覆盖整个验证集，`metrics_val` 含任务的主指标；`model.predict` 是候选目录内的可执行文件，`code_version` 和 `config` 按上面的契约写全。
- 门没过：能在本轮修就修；否则如实写 `passed: false` 交回，由顶层决定。没过门的候选不会被送评，登记会拒绝它的 delivery。
- 训练命令的候选目录用 validate 检查通过才算交出。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 `trainset`、`valset` 和配置文件 从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`。不需要工作区，也不读登记表。
2. 输出写到沙盒下自己建的输出目录，每个产物一个子目录，manifest 的 `id` 写 null，`round` 自拟。
3. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验，通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
4. 训练命令同样可以在沙盒里直接调用，候选目录用 validate 检查，完成标记放在候选目录之外。

## 步骤和策略子 skill

由模块负责人填写：训练方法、架构、校准、阈值选择、checkpoint 选择、训练命令和预测入口的实现。训练命令和默认配置的改动在两轮之间做：同一轮里 tune 的试验和本模块的候选要出自同一版训练代码。把可重复的步骤写成脚本，skill 只写需要判断的部分。

## 模块指标

由模块负责人填写：训练集固定时，相对上一种训练方法的提升，及其测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），写明改了什么、前后数值、结论，`module` 写 `train`。

## 长任务

由模块负责人填写：训练（建议通过训练命令后台运行，用 `--marker` 的完成标记，见契约）。预计超过简报里规定的时长时，在后台启动，写完成标记（第一行 ok 或 failed），立刻交回标记路径、日志路径和预计耗时，不轮询、不等待。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。已知的一条：用评测集挑 checkpoint，或训练命令需要 agent 才能跑完。
