# 简报

每次引擎派出一个模块 agent，代码（`rsi/briefs.py`）都按工作区的状态生成这次的简报，写进这次派出的输出目录：`brief.md`（人读，下面是它的结构）和 `brief.json`（同样的内容，结构化）。模块 agent 的提示词是模块 skill 的全文加 `brief.md`。操作会话不写简报，也不派活；想看某个模块现在会拿到什么简报时，运行 `rsi brief`（只打印，不需要派出记录）。

被退回时，同一个会话收到的是 `feedback-<k>.md`（`<k>` 是第几次启动）：为什么退回（错误原文、没过的门和数值）、要它做什么，以及用户对它上次提的问题的回答。不算退回的续接也用 `feedback-<k>.md`：两段式交回的后半段（「# 继续：交回的第二部分」）、用户回答了它的问题（「# 用户回答了你的问题」）、长任务结束后收尾（「# 长任务已结束」）。会话续不上时，新会话收到 `history-<k>.md`（上次的简报、交回和反馈）加重建的简报。

## `brief.json` 的字段

| 字段 | 含义 |
|---|---|
| `module`、`skill`、`purpose` | 模块、它的 skill 名、这次派出的目的（`sets`、`trainset`、`refresh`、`train`、`deliver`、`judge`、`plan`、`milestone_deliver`、`milestone_judge`） |
| `dispatch_id` | 这次派出的 id（d-NNNN；不经派出生成的简报是 null） |
| `round`、`task_id`、`task_version`、`task_json` | 轮次、任务 id 和版本、任务定义的路径 |
| `ids` | 当前登记的 trainset、valset 的 id（评测集、封存 test 的 id 只在需要它们的模块和目的里出现），manifest 里引用它们时用这些 id |
| `inputs`、`input_notes` | 输入名到绝对路径，和每项的一句说明；只有这些路径可读，没有工作区根目录。data 产出训练集时拿训练池 `pool`；刷新评测集时拿 `corpus` 和以前各评测集的 `inputs.jsonl`（`evalset_inputs_<id>`，不带标签）；第一轮续做时拿已登记的 `corpus` 和 `pool`。除了各模块自己的输入，data、train、deliver 还拿本模块以前各轮的 `record`（只给自己的；train 的含旧工作区里 tune 的），judge 的评测另拿开启本轮的 `plan` |
| `outdir` | 输出目录，每个产物一个子目录 |
| `progress_file` | `<outdir>/progress.json` 的绝对路径：模块可选写的进度文件，进度页面和 `rsi run` 的状态行原样显示它，格式见 `docs/interfaces/progress.md` |
| `rsi` | 模块能运行的 `rsi` 命令前缀（只有 `validate` 和 `score`） |
| `budget_left`、`attempts` | 剩余预算（算力小时、标注条数、调用费用）和本轮尝试次数 |
| `goal` | 本次要做的事：第一轮按任务定义，之后来自计划里分给本模块的问题 |
| `handback` | 交回格式（见下） |
| `train_command` | 只有 train 有：训练命令（data 不训练，没有它） |
| `sources` | 只有 data 的第一轮 sets 有：原始数据（任务定义 `data.sources` 的路径），只在这次派出读，之后的派出都禁读；不在 `inputs` 里 |
| `dset` | 只有 data 有：切分工具的命令前缀 `PYTHONPATH=<repo root>/modules/data python3 -m dset`，用法见 rsi-data 的 skill |
| `search_space`、`search`、`resources`、`checkin_minutes` | 只有 train 有：任务的配置空间、搜索设置、`resources.onejev` 和 `resources.gpu_ids`，以及定时检查的间隔（分钟，null 表示不做定时检查） |
| `best` | 只有 judge 有：当前最佳版本的版本名、evaluation 和 delivery（`best_before` 按它写） |

## `brief.md` 的结构

以下是 `brief.md` 的各条，尖括号处由代码填：

- skill：`<rsi-data、rsi-train、rsi-deliver 或 rsi-judge 之一>`（在 `modules/<模块>/SKILL.md`）
- 任务定义：`<task.json 的路径>`，人读版是同一目录的 `TASK.md`
- 本轮轮次：`<n>`。每个产物 manifest 的 `round` 写这个数，`task_id` 写 `<task_id>`，`task_version` 写 `<version>`，`id` 写 null。
- 当前登记 id：trainset `<id 或 无>`，valset `<id 或 无>`，需要时还有 evalset、testset。
- 当前最佳版本（只有 rsi-judge 有）：版本名、evaluation、delivery；评测集刷新过时另有最佳版本在当前评测集上的 delivery。
- 训练命令：`<resources.train_command 的绝对路径；任务定义没给时是 <repo root>/modules/train/train>`（只有 rsi-train 有）。调用它时环境里带 `PYTHONPATH=<repo root>`。
- 原始数据（任务的 `data.sources`，只在本次派出读，之后的派出都禁读）：`<路径>`（只有 rsi-data 第一轮 sets、任务写了原始数据路径时有）。
- 切分工具：`PYTHONPATH=<repo root>/modules/data python3 -m dset`，用法见本模块 skill（只有 rsi-data 有）。
- 本次要做的事：`<goal>`
- 可用预算（剩余）和本轮尝试次数。
- 输入产物（只读这些，没列出的产物和目录不要找、不要读）：名字、路径、说明的表。
- 输出目录：`<输出目录>`，每个产物一个子目录。
- 进度文件（可选）：`<输出目录>/progress.json`。想让进度页面和 `rsi run` 的状态行显示做到哪一步，就随时整个重写它（先写临时文件再 `os.replace`），格式见 `<repo root>/docs/interfaces/progress.md`；不写也可以。
- 要报告的门数字：该模块 skill「门」一节列出的各项，逐项给出数值、阈值和是否通过，另报告实际花费。
- 报告的指标、对照组、差值和区间：用 `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <标签行文件> --predictions <预测文件>` 算（需要对照组时加 `--train <训练集 rows.jsonl>`；需要和另一份预测比时加 `--compare <另一份预测>`，同时加 `--clusters <inputs.jsonl>` 给出区间用的簇；标签行没有 `meta.state_id` 而要算 `group_prior` 时也加 `--clusters <inputs.jsonl>`；完整写法见模块 skill），把它打印的数原样写进 manifest。这些数的定义固定在代码里，登记时会按同样的定义重算，不一致就被拒。
- 交回前：对每个产物运行 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>`，通过再交回。你只能运行 `rsi validate` 和 `rsi score` 这两个 `rsi` 命令；登记由引擎做。
- 一个产物要引用本次另一个产物的登记 id 时（例如 `search` 引用 `candidate`，`plan` 引用 `evaluation`），先交回被引用的那个；引擎登记后让同一个会话接着做（`feedback-<k>.md` 开头是「# 继续：交回的第二部分」，不算退回），登记 id 在里面。
- 遇到需要用户决定的事就停下报告（写进 `handback.json` 的 `questions`），例如要超出预算、要改任务定义、要用封存 test、要刷新验证集或评测集、要删除产物、输入有问题。不要自己决定，也不要绕过。
- 交回格式：在输出目录写 `handback.json`：`status`（`ok`、`failed` 或 `long_job`）、`artifacts`（输出目录下的子目录名，每个产物一个）、`marker`（长任务完成标记的绝对路径，`long_job` 时必填，其他写 null）、`notes`（门数字、花费、遇到的问题；`failed` 时写原因）、`questions`（需要用户决定的事，没有写 `[]`）。`questions` 不为空时，这次交回的产物一个都不登记：用户回答后，回答送回同一个会话（反馈开头是「# 用户回答了你的问题」，不算退回），按回答改好或确认后重新交回要登记的全部子目录；`long_job` 交回里的问题等完成标记出现后才问，没列产物时回答随「# 长任务已结束」的反馈送回。长任务（预计超过 10 分钟）在后台启动后立刻以 `long_job` 交回，任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`；不要等待或轮询。要交回 `long_job` 时完成标记已经在了（长任务已经结束）就不要以 `long_job` 交回：早于这次启动的标记引擎永远不认，直接读结果接着做完再交回。交回 `long_job` 之前在完成标记旁写 `<marker>.pid`：第一行是长任务进程自己的进程号（由长任务自己写，例如 shell 里的 `$$` 或 Python 的 `os.getpid()`），第二行是它的启动时间（Unix 秒，`date +%s`）；运行被停止时引擎按它终止长任务。每个长任务用新的标记路径（例如 `<名>-<k>.done`，`<k>` 每次加一），启动前 `rm -f <marker> <marker>.pid`：比这次启动早的标记文件不算这个长任务结束。
- 不要派别的 subagent。rsi-train 的简报接着写：试验一律按本模块 skill 经 `trsi launch` 训练，训练命令只供参考（候选的 `code_version.entry` 写它，负责人手动单次训练时调用它）；rsi-data 的简报接着写：本模块不训练，新训练集的好坏由本轮的训练调优训练、评测与决策和当前最佳版本比较来判断。
