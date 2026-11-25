---
name: rsi-train
description: 由 rsi-orchestrator 派发训练调优模块时使用：在任务的配置空间里搜训练方案（底座、配方、超参、种子），训练经 onejev-harness（ojh），每个跑完的试验打包成候选；先跑种子 0 当探针，过了门槛才补种子，按规则判接不接受，最后复盘。Use when the orchestrator dispatches the RSI train (training-tuning) module; not for general model-training requests.
---

# rsi-train：训练调优模块（Skill 2.0，`trsi` 1.4.3）

训练由 onejev-harness 完成（`ojh run`）。本模块决定**怎么训练**：在任务定义的配置空间（`search_space`）里选底座、配方（全参或 LoRA）、带原因的超参改动和种子，在简报给的算力预算内比较新旧方案。一次派出是一次搜索。只做本模块的事，不派别的 subagent。

`trsi`（本目录的 `trsi/`）做机械的事：按简报生成 ojh 工作区，检查范围和预算，启动，等待，把每个跑完的试验打包成框架的候选，按规则判，记账，写报告和 `search`。你做判断：检查环境、选卡、读数据和训练曲线、诊断、提候选、写预测、训练期间的定时检查、复盘。

流程一览（细节见「步骤和策略子 skill」）：

1. 读简报；检查卡、磁盘和 ojh，从 `resources.gpu_ids` 里选空闲的卡。
2. `trsi init` 写 run.yaml，起点成为当前方案；起点接过当前最佳版本的候选（`trsi adopt`），或训练起点的种子。
3. 诊断，写候选（带 `note`、`prediction`），先跑种子 0（探针）。
4. `trsi watch` 在后台等这一批试验结束，你以 `long_job` 交回；训练期间收到「# 定时检查」就检查环境。
5. 收到「# 长任务已结束」：`trsi collect` 打包，`trsi next` 判，照它补种子、提新候选或停。
6. 第一段交回所有候选和一份复盘 `record`；收到「# 继续：交回的第二部分」后写 `search` 交回。

不做：不改 onejev-harness、OneJev 和本模块的代码；不做评测这一层（阈值、决策策略、评测集和 test 上的评测是交付、评测模块的事）。

## 输入

简报由代码生成，放在输出目录里：`brief.md`（人读）和 `brief.json`（同样的内容，结构化：`module`、`purpose`、`round`、`task_json`、`ids`、`inputs`、`outdir`、`budget_left`、`goal` 等；字段说明见框架仓库根目录下的 `skills/rsi-orchestrator/templates/brief.md`）。输入以简报的 `inputs` 为准，只读那里列出的路径。被退回时，同一个会话会收到 `feedback-<k>.md`（`<k>` 是第几次启动），写明为什么退回和要改什么：按它改好，再重新写 `handback.json`。同一个会话还会收到两种不算退回的 `feedback-<k>.md`：开头是「# 继续：交回的第二部分」的（两段式交回的后半段，例如 train 先交回试验候选、judge 先交回评测，登记后在这里拿到登记 id，再写 search 或计划），和开头是「# 用户回答了你的问题」的（见「输出」）；训练期间还有开头是「# 定时检查」的（见「长任务」）。

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`（任务定义 task.json）、`trainset`、`valset`：都含标签，trsi 用它们训练和打分。
- `best_candidate`：当前最佳版本的候选（有最佳版本时才有），本次搜索的起点。
- 已登记的 `candidate` 和 `search`（按 id 列出）：以前各次试验的配置、结果和比较。
- `plan`：开启本轮的计划（有的话），看它分给本模块的问题。
- 本模块以前各轮的 `record`（只给自己的）：以前试过什么、结果如何。
- 简报的其他字段：`search_space`（配置空间，只在这里面搜）；`search`（任务的搜索设置：`search.max_configs`、`search.seeds`、`search.probe_margin`、`search.accept_margin`、`search.guards`）；`resources.onejev`（`resources.onejev.python` 是 ojh 的 `.venv` 里的 python；`resources.onejev.ojh_bin`、`resources.onejev.env_file`、`resources.onejev.machine` 是 ojh 的命令、环境变量文件和机器配置；`resources.onejev.default_config` 是没有最佳版本时的起点；`resources.onejev.estimate` 是一个种子预计的算力小时）；`resources.gpu_ids`（能用的卡，没写就是全部卡）；`checkin_minutes`（定时检查的间隔，null 是不检查）；`budget_left`（剩余预算，`budget_left.compute_hours` 是算力小时）；`dispatch_id`（这次派出，也是 trsi 的 run 名）；`train_command`（训练命令的位置，候选的 `code_version.entry` 必须是它）。
- 本模块目录（本 skill 所在的目录）里的 `reference/`、`recipes/`，以及 trsi 和 ojh 为本次搜索写的文件（输出目录下的 `trsi/`、ojh 的运行目录和日志）也可以读。
- 不拿评测集和封存 test：它们的文件，以及任务定义 `data.sources` 写的原始数据，一律不找、不读。

## 输出

- 若干 `candidate`：本次搜索每个打包好的试验一个，`trsi collect` 写在 `<outdir>/cand-<试验>/`：模型（选中的 checkpoint）、可直接运行的预测入口（model.predict）、完整解析后的训练配置和所要求的配置（`config_request`）、训练代码的版本（`code_version`）、所用训练集和验证集的 id、验证集预测和指标、对照组、训练门结果、成本。起点的种子也是候选；从已登记候选接过来的不再交回。
- `record`：本次搜索的复盘，一份，写法见「实验记录」。
- `search`：第二段交回。本次搜索的所有试验（配置、种子、验证集主指标、花费、状态）、默认候选（起点）、最佳候选、相对起点的增益，由 `trsi report` 写。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [models.md](../../docs/interfaces/models.md)。 `record` 见 [record.md](../../docs/interfaces/record.md)。

两段式交回：第一段以 `ok` 交回本次打包的所有 `cand-*` 子目录（已经登记过的不再列）和 `record`；引擎登记后让同一个会话接着做（`feedback-<k>.md` 开头是「# 继续：交回的第二部分」，不算退回），里面列出每个子目录的登记 id，同样的对应关系也在输出目录的 `registered-<n>.json`（每次续接一个文件，文件名在反馈里）；再写好 `search` 交回。一个候选都打包不出来时（试验都失败了），以 `failed` 交回，`notes` 写原因。

进度文件：输出目录里的 `progress.json`（简报的 `progress_file`，格式见 [progress.md](../../docs/interfaces/progress.md)）由 trsi 写（launch 之后、watch 每次轮询、collect 打包每个试验前后），进度页面和 `rsi run` 的状态行显示它；不用手写，也不要改。

交回：全部产物写好、校验通过后，在输出目录写 `handback.json`：`status`（`ok`、`failed` 或 `long_job`）、`artifacts`（输出目录下的子目录名，每个产物一个）、`marker`（长任务完成标记的绝对路径，`long_job` 时必填，其他写 null）、`notes`（门数字、花费、遇到的问题；`failed` 时写原因）、`questions`（需要用户决定的事，没有写 `[]`）。引擎读到它才算交回，缺了或格式不对按交回失败处理，重新写好即可。读取很严格：不能有上面没列出的字段，`artifacts` 只能是子目录的名字（不含路径分隔符，不重复），`questions` 的每一项必须是非空字符串，`marker` 必须是绝对路径或 null，`failed` 时 `notes` 不能为空。

交回带 `questions` 时，这次交回的产物一个都不登记，引擎先问用户；用户回答后，回答送回同一个会话（`feedback-<k>.md` 开头是「# 用户回答了你的问题」，不算退回）。按回答修改或确认产物，再重新写 `handback.json`，列出要登记的全部子目录（上次列过的也要再列）。长任务（`long_job`）交回里的 `questions` 等完成标记出现后才问；它没列产物时，回答写在开头是「# 长任务已结束」的反馈里，和长任务的结果一起送回。会话是无界面的，没有人在看：要问用户的事只能写进 `questions`。

### 训练命令（接口契约）

训练命令的位置：任务定义的 `resources.train_command`（可选，绝对路径）给了就用它，没给就是 `<repo root>/modules/train/train`（`<repo root>` 是简报给的框架仓库根目录）；简报的「训练命令」一行写明实际路径。它用 trsi 的同一套代码训练一个配置、阻塞到结束、打包成候选。调用方运行它时环境里带 `PYTHONPATH=<repo root>`；它会换到任务的 `resources.onejev.python` 里运行，打包时运行 `python3 -m rsi score` 算指标，不依赖自己所在的路径。负责人的手动单次训练调用它，候选的 `code_version.entry` 都写它，所以它是本模块接口的一部分；本模块的搜索不调用它，试验一律经 `trsi launch`。数据模块不训练、不调用它：新训练集好不好，由本模块在它上面训练出候选、再由评测和当前最佳版本比较来判断。

```
<训练命令> --task <task.json> --trainset <trainset 目录> --trainset-id <id> --valset <valset 目录> --valset-id <id> --config <配置文件> --config-source <default 或已登记 candidate、search 的 id> --round <n> --producer train --out <候选目录> --marker <完成标记路径>
```

- 输入：任务定义、训练集目录、验证集目录、配置文件；其余参数写进 manifest。配置文件是 JSON：`{"base", "recipe", "overrides", "seed", "max_steps"}`，都可省，默认 joey-1、底座的单卡默认配方（recipe 写 null）、不改、种子 0；overrides 的键是 `onejev.<flag>`、`lora.<field>`。卡用 `RSI_TRAIN_GPUS` 指定，不设时取 `resources.gpu_ids` 里 nvidia-smi 显示空闲的卡。
- 输出：候选目录本身就是一个 `candidate` 产物目录，id 为 null，用 `PYTHONPATH=<repo root> python3 -m rsi validate candidate <候选目录> --task <task.json>` 能通过。其中：
  - 预测入口：候选目录里的一个可执行文件，相对路径写在 `model.predict`，调用方式固定为 `<候选目录>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>`（读输入行，写预测行，成功退出码 0，见 [models.md](../../docs/interfaces/models.md)）。模型文件放在候选目录里。predictions_val 就用这个入口在验证集的 inputs.jsonl 上跑出来，这样交付才能复现它。
  - 训练代码版本：`code_version.entry` 写训练命令的路径，`code_version.sha256` 写训练命令文件的 sha256。只有这一个文件被算 sha256，所以训练代码一改，这个文件必须跟着变：改了 `trsi/` 就一起改训练命令里的 `TRSI_VERSION` 和 `trsi/__init__.py` 的版本号（不一致时训练命令拒绝运行），ojh 或 OneJev 换版本时改训练命令里的 `OJH_COMMIT`、`ONEJEV_COMMIT`。
  - 配置：manifest 的 config 写完整解析后的配置（ojh 的 `run.lock.yaml`），`config_request` 写所要求的配置，五个键都写。
- 完成标记：`--marker` 给的文件，必须在候选目录之外（不进产物目录，也不出现在 manifest 的 files 里），最后写，第一行 ok 或 failed，failed 时第二行写原因。退出码 0 当且仅当写了 ok。收到 SIGTERM 或 SIGINT 时结束自己的 ojh 训练，写 failed。
- `--producer` 只接受 train，别的值（包括以前数据模块用的 data）报错、退出码 2、不写标记。旧工作区里 producer 是 data 的候选照常能读，登记会拒绝新的。`--trainset-id` 写训练集的登记 id；手动单次训练用还没登记的训练集时写 unregistered，这样的候选登记会拒绝。
- 进度：它把进度写进 `--out` 所在目录的 `progress.json`（`<dirname(--out)>/progress.json`，格式见 [progress.md](../../docs/interfaces/progress.md)），开始时、训练中每次检查、结束和失败时都重写。调用方想让进度页面显示它，就把候选目录直接放在自己的输出目录下（例如 `<outdir>/cand-1`）。
- 不需要 agent 参与，可在后台无人值守运行；只读两个数据集目录和配置，不读评测集和封存 test。

## 门

- 训练门（每个候选）：验证集上主指标严格优于任务定义列出的每个对照组（方向按任务定义，相等不算），结果写进 `train_gate`（`passed`、`detail`）。`trsi collect` 打包时已经算好。
- 指标和对照组一律用 `rsi score` 算，不自己实现（定义见 [task.md](../../docs/interfaces/task.md) 的「指标的定义」）：`PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <valset>/rows.jsonl --predictions <candidate>/predictions_val.jsonl --train <trainset>/rows.jsonl`，输出的 `metrics` 写进 `metrics_val`，`baselines` 写进 `baselines_val`，`passed` 按这些数判断。`trsi collect` 和训练命令都调用它。登记时代码按同样的定义重算，对不上就被拒。
- 候选完整：预测覆盖整个验证集，`metrics_val` 含主指标；`model.predict` 是候选目录内的可执行文件；`code_version` 的 entry 是简报的训练命令、sha256 对得上；`config_request` 落在 `search_space` 里。`trsi collect` 打包后已经用 `rsi validate` 检查过；要再查一个就运行 `PYTHONPATH=<repo root> python3 -m rsi validate candidate <outdir>/cand-<试验> --task <task.json>`。
- `search` 的门（第二段）：`best_candidate` 的验证集主指标（`metric_val`）不差于 `default_candidate`（起点）的；做不到就把起点的候选作为最佳交回，`gain_vs_default` 如实记录，不包装成提升。每个试验的 `metric_val` 是它的候选 `metrics_val` 里的主指标，原样抄过来（登记时核对，容差 1e-9），比较候选也只用这些数。一次搜索只用一版训练代码：所有 ok 试验的候选（起点的也在内）的 `code_version.sha256` 必须相同。试验花费之和不超过 `budget.compute_hours`。`trsi report` 按这些规则写，写完用 `PYTHONPATH=<repo root> python3 -m rsi validate search <outdir>/search --task <task.json>` 检查。
- 门没过：能在本次搜索里修就修（换候选、补种子）；否则如实交回（`passed: false`），由引擎按决策点处理（默认回到本会话再搜一次，或派数据模块）。没过训练门的候选不会被送评。

## 步骤和策略子 skill

### 命令和路径

- `<py>`：简报的 `resources.onejev.python`（ojh 的 `.venv/bin/python`，带 pyyaml）。`<模块目录>`：本 skill 所在的目录（`<repo root>/modules/train`）。`<outdir>`：简报的输出目录。
- 下文的 `trsi <命令> ...` 都指 `PYTHONPATH=<模块目录> <py> -m trsi <命令> ...`，在哪个目录下运行都可以。`trsi --help` 列出全部命令，每个命令加 `--help` 看参数。`trsi` 自己找到框架的 `rsi`（打包时运行 `rsi score`、`rsi validate`）。
- `<run.yaml>`：`trsi init --brief <outdir>/brief.json` 写的 `<outdir>/trsi/run.yaml`。`<记录目录>`：`<outdir>/trsi/<dispatch_id>`（`trsi init` 打印出来），trsi 的记录（`archive.jsonl`、`incumbent.yaml`）在这里，候选 YAML 写在它下面的 `proposals/`，`trsi report <run.yaml>` 写它下面的 `report.md`。每个试验的记录在 `<outdir>/trsi/trials/<试验>.json`：ojh 的运行目录（`run_dir`）、tmux 会话（`session`）和日志。
- ojh：简报的 `resources.onejev.ojh_bin`；给了 `env_file` 时先在同一个 shell 里 `source` 它，机器配置用 `OJH_MACHINE=<machine>` 传，例如 `bash -c 'source <env_file> && OJH_MACHINE=<machine> <ojh_bin> doctor --no-network'`。`ojh status <run_dir>` 打印一个训练的简短进度。

### 流程

1. **读简报**：`brief.json` 的 `task_json`、`search_space`、`search`、`resources`、`checkin_minutes`、`inputs`、`budget_left`、`dispatch_id`、`outdir`。读计划里分给本模块的问题、本模块以前的 `record` 和已登记的 `search`（以前哪些配置试过、结论是什么），看训练集的数据事实（正例比例、输入长度和标签的关系、重复行）。只读 `inputs` 列出的路径；评测集、测试集的文件和任务 `data.sources` 写的原始数据一律不读。
2. **检查环境**，选卡由你负责：
   - `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`：只从 `resources.gpu_ids`（没写就是全部卡）里挑空闲的卡（占用不到 1 GiB）。一个试验占配方要求的卡数（单卡配方一张），不和别的试验或别人的进程共用一张卡。
   - `df -h <outdir>`，以及 ojh 运行目录所在的盘（`ojh doctor` 列出的 runs (out_root)）：每个候选自带选中的 checkpoint（joey-1 全参约 4 GB），ojh 的运行目录里还有训练中的保存点。不够就先不启动，写进 `questions`。
   - `ojh doctor --no-network`（调用方式见上）：有 fail 就停，原话写进 `questions`。
3. **初始化**：`trsi init --brief <outdir>/brief.json`（写 `<outdir>/trsi/run.yaml` 和 `exp.yaml`，起点成为当前方案；已经初始化过就保留原样），然后 `trsi check <run.yaml> --ojh`（核对 run.yaml，对起点做一次 `ojh check`，不占卡）。种子是 `search.seeds`（没写就是 `search_space` 的 seed，再没有就是 0、1、2），第一个是探针种子。起点：
   - 简报有 `best_candidate`：`trsi adopt <run.yaml> --candidate <best_candidate>`，把它当作起点训练时那个种子的结果接过来，不重训；起点其余的种子照常训练。
   - 没有：每个种子一条 `trsi launch <run.yaml> --start --seed 0 --gpus <卡>`（`--seed 1`、`--seed 2` 同样，各用一张空闲卡）。
   起点的种子都跑完、打包好之前，`trsi next` 不比较任何候选。
4. **提候选**：按「诊断」「提候选」两节写好候选 YAML，每个先跑探针：`trsi launch <run.yaml> <记录目录>/proposals/<名字>.yaml --seed 0 --gpus <卡>`，立即返回。卡不够时先启动放得下的，其余等下一批。第一轮还什么都没训练、证据不够写出带证据的诊断时，可以先只跑起点，收集后再提候选（第 7 步回到这里）。
5. **以长任务交回**：这一批都启动后，`trsi watch <run.yaml> --marker <outdir>/watch-1.done`（每次用新的路径：watch-2.done、watch-3.done……）。它自己转到后台，写好 `<marker>.pid`，等到目前启动的所有试验都结束后写 ok。看到它打印的 marker 和 pid 文件后写 `handback.json`：`status` 是 `long_job`，`marker` 是这个标记的绝对路径，`artifacts` 是 `[]`，`notes` 写启动了哪些试验、用哪些卡、预计多久。然后结束会话，不等待、不轮询。
6. **收到「# 定时检查」**（每 `checkin_minutes` 分钟一次，不算退回）：
   - 检查：这些试验的卡上有没有别人的进程（`nvidia-smi`，`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv`）；日志在不在前进、有没有报错（ojh 运行目录里的 `train-s<种子>.log`、`runner.log`，`ojh status <run_dir>`；OOM、NaN、Traceback、长时间没有新的步数）；磁盘够不够；监视进程还在不在（`<marker>.pid` 第一行的进程号）。
   - 一切正常：以 `long_job` 交回**同一个**标记（反馈里写着它的路径）。
   - 某个试验明显坏了（损失 NaN、报错后不动、远超预计时间没有进展）：结束它。`trsi --help` 没有结束单个试验的命令；ojh 的训练在 tmux 会话里，`tmux kill-session -t <session>`（会话名在 `<outdir>/trsi/trials/<试验>.json`）结束它，ojh 在约 30 秒内写 `DONE`，下一次 `trsi collect` 把它记为 failed（会话不在了、又超过 2 分钟没有 `DONE` 也没有动静时也一样，原因是 ended without DONE），卡时照记。监视进程不用动，交回同一个标记。不要因为中途的验证分数不好看就结束试验：中途和短跑的排名不可信（见 `reference/protocol.md`）。
   - 要换卡重跑（卡被别人占了、显存被挤爆、训练慢得反常）：按下面的顺序做。旧的监视进程留到新的监视起来之后再结束：这期间运行被停止时，引擎只知道旧的监视进程，它收到 SIGTERM 会结束本次运行所有在跑的训练（包括刚重跑的）。
     1. `tmux kill-session -t <session>` 结束要换卡的试验，等它的运行目录里出现 `DONE`（ojh 在 kill-session 之后约 30 秒内写；等了约 2 分钟还没有，也照样往下做，collect 会记为 ended without DONE），再 `trsi collect <run.yaml>` 把它记为 failed。collect 先把已经跑完的试验打包（每个要几分钟），所以 Bash 的超时设成 600000 毫秒；等 collect 打印这个试验是 failed 后再 launch，还显示 running 就再 collect 一次。
     2. 在 `resources.gpu_ids` 里另一张空闲卡上重跑：`trsi launch <run.yaml> <记录目录>/proposals/<名字>.yaml --seed 0 --gpus <新卡>`（起点用 `--start`，种子按原来的）。失败那次的卡时照记。
     3. 开新的监视：`trsi watch <run.yaml> --marker <outdir>/watch-2.done`（新路径），它等所有还在跑的试验。
     4. 用 SIGKILL 结束旧的监视进程：先确认 `<旧标记的 .pid 第一行的进程号>` 还是它：`ps -p <pid> -o args=` 打印的是 `... -m trsi watch ... --marker <旧标记>`；是才 `kill -9 <pid>`。什么都没打印（它已经自己退出了）或是别的进程（进程号被复用了）就不要 kill。不要用 SIGTERM（或不带参数的 `kill`）：`trsi watch` 收到 SIGTERM 会结束所有试验的训练。
     5. 立刻以 `long_job` 交回**新的**标记：引擎只在交回时才知道标记换了，旧标记不再算数。
   - 监视进程不在了（没有标记就退出了）：开新的监视，交回新的标记。监视进程自己出错时会写 failed 标记（第二行 the watcher stopped on an error，详情在 `<marker>.log`；读运行目录时的 OSError 要连续 3 次轮询才算出错，之前只记进日志），并像收到 SIGTERM 一样先结束本次运行所有在跑的训练（标记一写，就没有谁再看着它们）；派出记为失败，退回时看 `<marker>.log`，`trsi collect` 把被结束的试验记为 failed，再照常重跑（卡时照记）。
7. **收到「# 长任务已结束」**：
   - `trsi collect <run.yaml> --gpu <卡号>`（`--gpu` 是首选的打分卡，在 `resources.gpu_ids` 里；可以不给）。它把每个跑完的试验打包成 `<outdir>/cand-<试验>/` 并逐个打印状态：done（打包好）、failed（训练失败，或 ojh 的会话不在了、超过 2 分钟没有 `DONE`）、unpacked（打包失败，原因在同一行）、running。
   - 打分卡由 collect 在打包每个试验前重新选，不用你选：在 `resources.gpu_ids` 里、卡上没有任何计算进程（谁的都算）、空闲显存不少于机器配置的 `min_free_mib` 和模型目录大小的 1.5 倍中较大的一个、ojh 的卡锁（机器配置的 `lock_dir`）没被别人占着（打分期间由它拿着）；`--gpu` 的卡合格就用它，不合格就打印原因，换空闲显存最多的合格卡；打分时显存不够（OOM），换一张合格的卡再打一次。用的卡记在试验结果的 `scoring_card` 里。
   - 用不上的 checkpoint 由 collect 自动删，不问用户：试验打包好（候选通过了 `rsi validate`，模型在候选目录里，是硬链接或拷贝）后，它对这个试验的 ojh 运行做 `ojh cleanup --confirm`（删 ojh 的 `deletion_candidates.json` 列出的没选中的保存点和合并出的模型，ojh 留着选中的那个）；失败的试验（`DONE` 说失败）记为 failed 时，删掉它运行里所有的 checkpoint；重跑一个试验后，它被换下的旧运行在 `DONE` 出现后整个删掉 checkpoint（`trsi launch` 加 `--force` 重训 unpacked 的试验时，launch 成功后马上删）。unpacked 的试验（下一次 collect 还要打包它）、还在跑的、没有 `DONE` 的都不删。删了什么 collect 逐行打印，写进 `notes`；ojh 留下没删的（原因它会说）只报一次；没删成的下一次 collect 再删。
   - 打包每个试验要几分钟（拷 checkpoint，在验证集上加载模型跑一遍 `model.predict`），而 Bash 工具一次最多 10 分钟。要打包的只有一两个时在前台运行，超时设成 600000 毫秒；更多时按长任务在后台运行它（写 `<marker>.pid`，结束时写标记），以 `long_job` 交回，等下一次「# 长任务已结束」再往下走。每次后台 collect 用新的标记（`<k>` 每次加一：collect-1.done、collect-2.done……），启动前 `rm -f <outdir>/collect-<k>.done <outdir>/collect-<k>.done.pid`：引擎只认修改时间不早于这次启动的标记。它要在自己的会话里运行（Python 的 `start_new_session=True`，各系统都能用）：这样 sh 的 `$$` 就是它的进程组号，运行被停止时引擎终止整个进程组，collect 不会留下来：

     ```
     <py> -c 'import subprocess, sys; subprocess.Popen(["sh", "-c", sys.argv[1]], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)' 'printf "%s\n%s\n" $$ "$(date +%s)" > <outdir>/collect-<k>.done.pid; PYTHONPATH=<模块目录> <py> -m trsi collect <run.yaml> --gpu <卡号> > <outdir>/collect-<k>.log 2>&1 && echo ok > <outdir>/collect-<k>.done || printf "failed\ntrsi collect 出错，见 collect-<k>.log\n" > <outdir>/collect-<k>.done'
     ```

     被打断的 `collect` 不要紧：没打包完的试验下一次 `collect` 重新打包。
   - unpacked 的试验由下一次 `collect` 再打包，不重训（`trsi launch` 也不会重训它）；`trsi next` 也打印它的打包错误。原因修不好、一直打包不成时，同一条 `trsi launch` 命令加 `--force`：把它记为 failed（卡时照记）后重训。failed 的试验看运行目录的 `DONE`、`runner.log`、`train-s<种子>.log`，修好后用同一条 `trsi launch` 命令重跑（卡时照记）。
   - 然后 `trsi next <run.yaml>`，照它的输出走：
     - 起点的种子没跑完（incumbent seeds … not finished）：补起点的种子，或再 `collect`；
     - 探针过了（probe passed … launch seeds …）：用同一个候选 YAML 补这些种子（`--seed 1`、`--seed 2`），回到第 5 步；
     - 探针失败（failed -> relaunch it or drop the candidate）：重跑或放弃这个候选；
     - rejected 或 ACCEPTED：判定已记下；接受时它更新当前方案，之后的候选和新的当前方案比；
     - no open candidate：没有待判的候选，回到第 4 步提新候选（不超过 `search.max_configs`）；
     - STOP：到第 8 步。
     每次启动前都重新看一遍卡。
8. **停**：这次派出（含退回后再搜）提的候选数到了 `search.max_configs`；预算不够再跑一个种子（`trsi launch` 说 budget）；`trsi next` 打印 STOP；或剩下的问题都是数据或评测的问题（写进 `record`，交给对应模块）。还在跑的试验等它们结束、收集完再停，不留占卡的训练。
9. **第一段交回**：`trsi report <run.yaml>`（写 `<记录目录>/report.md`：每个试验、预测和结果、当前方案随花费的变化），照它写 `record`（见「实验记录」，放在 `<outdir>/record/`），用 `rsi validate` 检查。然后写 `handback.json`：`status` 是 `ok`，`artifacts` 是本次打包的所有 `cand-*` 子目录（已经登记过的不再列）加 `record`，`notes` 写每个候选的门、总花费和遇到的问题。
10. **第二段交回**（收到「# 继续：交回的第二部分」）：`trsi report <run.yaml> --search <outdir>/search --ids <outdir>/registered-*.json`。输出目录里所有的 `registered-<n>.json` 都要给：退回后再搜一次时，前一次搜索的试验的登记 id 在较早的文件里，只给最新的一个会让它们成了 unregistered（第 1 轮时 `trsi report` 因此找不到起点的已登记试验而报错）；内容是 `[]` 的文件也没关系。它写 `search` 并用 `rsi validate` 检查；然后以 `ok` 交回 `search`。

**退回后再搜一次**（训练门没过时，引擎默认让同一个会话再搜）：trsi 的 run 还是同一个（run 名是派出号），当前方案、跑过的试验和提过的候选都接着用；`search.max_configs` 按整个派出算，不按每次搜索重新计。`trsi next` 已经打印 STOP（候选数用完或预算不够）时，不再提新候选，只交回一份 `record`（用还没登记过的子目录名，例如 `record-2`），在 `record` 和 `notes` 里写明这次没有试新的配置、为什么；引擎照常按决策点往下走。还能提时照第 4–10 步做，只交回新打包的 `cand-*`，`record`、`search` 用 `record-2`、`search-2`。

### 诊断

用 3–5 句话写当前方案的问题，每条带证据（写进候选的 `note` 和最后的 `record`）：

- 训练曲线：每个试验的 ojh 运行目录里的 `RESULT.md`（`report.md` 里有选中的步数）。最好的步很早、后面一路变差 → 学习率偏大或过拟合；到最后还在变好 → 训练不够。
- 种子之间差多少：差得大时，小改进分不出来，先试大改动。
- 数据事实：正例比例、输入长度和标签的关系、重复行。它们会让曲线出现和“训练好坏”无关的现象。
- 计划分给本模块的问题、以前的 `record` 和 `search`。
- 对照 `reference/lessons.md`（别的任务上已经学到的）。

### 提候选

- 在 `<记录目录>/proposals/` 下写 YAML，一个候选一个文件，一次只改一处想法。一次派出最多 `search.max_configs` 个候选（起点不算；退回后同一会话再搜时，前一次提的也算在内；超了 `trsi launch` 拒绝）。
- YAML 里的数字带指数时写小数点和带符号的指数（`2.0e-5`、`1.0e+4`，不写 `2e-5`、`1.0e4`）：PyYAML 按 YAML 1.1 把后两种读成字符串，`trsi launch` 会说它不在 `search_space` 里。
- 只在 `search_space` 里：底座取 `search_space` 的 base 列出的（没写就只能用起点的），配方取它的 recipe 列出的，override 的键是去掉 `overrides.` 前缀的配置路径（`search_space` 写 `overrides.onejev.lr`，候选写 `onejev.lr`），取值在允许的列表或范围里；步数是 `onejev.max_steps`；种子不写在候选里，由 `--seed` 给。超出范围 `trsi launch` 拒绝。能改什么、先验高低和代价见 `reference/search_space.md`。
- `note`：针对哪条诊断、为什么可能有效。
- `prediction`：主指标在探针种子上的预期值，以及被接受的概率 `p_accept`。复盘时和实际对照。
- 先验高的先试：结构类（换底座、全参或 LoRA、训练哪些层）优先于拟合类微调；换大模型时连学习率一起调小；从别处借来的改法，先写下它在原处为什么有效、这里满不满足；证据差不多时选便宜的。
- 死路（`search_space` 里标了 dead 的取值）默认拒绝；只有能说明前提已经变了才加 `--force`，原因写进 `note`。探针没过的候选不补种子。

```yaml
base: joey-1
recipe: null          # null：底座的单卡默认配方；换配方写 search_space 的 recipe 里列出的名字
overrides:
  onejev.lr: {value: 5.0e-6, reason: "起点的验证分数在第 20 步见顶后一路下降：学习率偏大"}
note: 针对诊断 1（曲线早早见顶）：学习率减半
prediction: {auprc: 0.62, p_accept: 0.3}   # 键是任务的主指标；数值只是示意
```

### 判输赢

细节和依据见 `reference/protocol.md`。比较只用候选的数：`metrics_val`（打包时用 `model.predict` 在验证集上跑出、`rsi score` 算的），ECE 由 `predictions_val` 算（15 个等宽箱）。规则由 `trsi next` 执行，不要自己另算、另判。

- 探针：候选的探针种子比当前方案的同一种子高过 `search.probe_margin`（默认 0.035），才补其余种子；否则不接受。
- 接受：所有种子平均比当前方案高过 `search.accept_margin`（默认 0.035），且护栏（`search.guards`，默认 ECE 0.02、Brier 0.005）的平均变差不超过阈值。
- 为什么不用“一个种子 + 按行 bootstrap”：在 Elliptic 上，同一配方换种子的标准差是 0.0175 AUPRC，种子之间约九成的差来自训练本身，按行重抽看不到这部分。
- 拿不准就不接受：留着当前方案没有成本；靠噪声赢的方案会让后面每一步都建立在错的起点上。

## 规则

- 只在 `search_space` 里搜。配置空间是任务定义的一部分，用户批准任务定义时已经批准；要扩大它就是改任务定义，写进 `questions`。
- 搜索期间训练代码不变：不改 `trsi/`、训练命令、`recipes/`、onejev-harness 和 OneJev。发现要改的地方写进 `record` 和 `notes`，由负责人在两轮之间改。
- 试验只经 `trsi launch` 训练：不直接 `ojh run`，不调用训练命令，否则记账、复用、比较和打包都会漏掉它。不手写、手改 `run.yaml`、`exp.yaml`（由 `trsi init --brief <outdir>/brief.json` 生成）、`archive.jsonl`、`incumbent.yaml`、`trials/`、`run.lock.yaml`。
- 卡只用 `resources.gpu_ids` 里的空闲卡，由你选；不加 `--allow-shared`。ojh 说卡忙时换一张空闲卡；不可行、放不下或要用户选替代方案时，把 ojh 的原话写进 `questions`，不自己换配方。
- 预算以简报为准：`trsi init` 把 `budget_left.compute_hours` 写进 run.yaml，`trsi launch` 在已花加下一个种子的预计超过它时拒绝；失败的试验照记。
- 不读评测集、测试集的文件，也不读任务 `data.sources` 写的原始数据；不给 ojh 传 `--allow-test`、`--allow-report`。
- 不删候选目录和 ojh 的运行目录。用不上的 checkpoint 由 `trsi collect` 自动删（第 7 步），不用问用户；不自己运行 `ojh cleanup`，不手工删。
- 要问用户的事写进交回的 `questions`，不直接问用户。
- 交回 `long_job` 后结束会话：不等待、不轮询日志。

## 实验记录

每次搜索交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），按复盘的顺序写：诊断 → 改动 → 预测 → 结果，以及预测准不准。

- `module` 写 `train`，`owner` 写 `rsi-train`，`automation` 写 `automated`。
- `failure`：诊断（3–5 句，每条带证据）。
- `hypothesis`：每个候选的 `note` 和 `prediction`。
- `action`：每个候选相对当前方案改了什么，跑了哪些种子，`trsi next` 怎么判（探针没过、接受、不接受），候选的子目录名。
- `result`：`metric` 写 `val <主指标>`；`before` 是搜索开始时当前方案各种子的平均，`after` 是搜索结束时当前方案各种子的平均（没有接受任何候选时等于 `before`）；`conclusion`：接受了候选写 `success`，都没接受写 `failure`，因预算或故障提前停、没比出结果写 `inconclusive`。
- `knowledge`：逐条对照预测和实际，预测偏在哪；学到了什么；剩下的若是数据或评测的问题，写明该交给哪个模块；训练代码要改的地方。
- `artifacts`：相关的已登记 id（例如 `best_candidate` 的 id）；本次的候选还没登记，写在 `action` 里。

数以 `<记录目录>/report.md` 为准。

## 模块指标

同样算力预算下，搜到的最佳方案相对起点的增益（`search` 的 `gain_vs_default`），以及预测的准确度（`prediction` 和实际的差）。测量方法见 `reference/protocol.md` 的「比较两个版本的 Skill」。

## 长任务

训练是长任务：`trsi watch` 就是它（见流程第 5 步），自己写 `<marker>.pid`。在后台运行的 `trsi collect` 也是长任务（流程第 7 步）。

长任务的交回：预计超过简报规定的时长时，在后台启动，立刻写 `handback.json`，`status` 写 `long_job`，`marker` 写完成标记的绝对路径，然后结束会话，不轮询、不等待。要交回 `long_job` 时完成标记已经在了（长任务在你交回之前就结束了，例如上一个会话启动的）：不要以 `long_job` 交回。引擎只认这次启动之后写的标记，更早的标记永远不算出现，运行会一直等；直接读结果，接着做完，再交回。任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`。标记是 `ok` 时，引擎让同一个会话继续（`feedback-<k>.md` 开头是「# 长任务已结束」），由它读结果、写好产物、再交回；标记不是 `ok` 时这次派出记为失败。`trsi watch` 在试验都结束后写 ok（有试验失败也写 ok，由 `trsi collect` 报告）；它自己出错时写 failed，并先结束所有在跑的训练（见流程第 6 步）。每个长任务用新的标记路径（`watch-<k>.done`、`collect-<k>.done`、`install-<k>.done`，`<k>` 每次加一），自己启动的先 `rm -f <marker> <marker>.pid`：引擎只认修改时间不早于这次启动的标记，同一路径上以前的标记不算这个长任务结束。

交回 `long_job` 之前，在完成标记旁写 `<marker>.pid`：第一行是长任务进程自己的进程号，第二行是它的启动时间（Unix 秒，`date +%s`）。`trsi watch` 自己写（它在自己的会话里运行）。自己在后台启动的任务（例如 `trsi collect`）要在自己的会话里运行，进程号由它自己写：照第 7 步的写法，Python 的 `subprocess.Popen(["sh", "-c", <脚本>], start_new_session=True, ...)` 起一个 sh，脚本先写 `$$` 和 `date +%s`，再运行命令、写标记（sh 的 `$$` 就是进程组号）；Python 里的任务一开始写 `os.getpid()` 和 `int(time.time())`。不要用 shell 给出的后台进程号代替，它可能是包装进程而不是长任务本身。`<marker>.pid` 写好后再写 `handback.json`。运行被停止时，引擎按这个文件确认进程是这个长任务后终止它（整个进程组）：`trsi watch` 收到 SIGTERM 时先结束所有试验的 ojh 训练，再把标记写成 failed。没有这个文件就终止不了，只能列为可能仍在运行、请用户手动结束。

定时检查：任务设了 `flow.checkin_minutes` 时，长任务期间引擎每隔这么多分钟让同一个会话做一次定时检查（`feedback-<k>.md` 开头是「# 定时检查」，不算退回），做法见流程第 6 步：正常就交回同一个标记，换了监视进程就交回新的标记。到了 `flow.max_checkins` 次，引擎转去问用户。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 `trainset`、`valset`（要接过来的候选也一样）从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`，要有 `resources.onejev`、`search_space` 和 `search`。不需要工作区，也不读登记表。
2. 手写一份最小的 `brief.json` 放进沙盒的输出目录：`{"task_json", "outdir", "round", "inputs": {"trainset", "valset"}, "ids": {"trainset", "valset"}, "budget_left": {"compute_hours"}}`（接候选时 `inputs` 再加 `best_candidate`），然后按「步骤和策略子 skill」的流程走；没有引擎时，等完成标记和定时检查都由你自己做。
3. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验，通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
4. 训练命令同样可以在沙盒里直接调用，候选目录用 validate 检查，完成标记放在候选目录之外。

## 负责人和版本

- 负责人：待指定。本模块在框架仓库的 `modules/train/` 里独立维护（skill、`trsi/`、训练命令 `train`、`recipes/`、`reference/`、`tests/`，见 `README.md`）。
- 版本：Skill 2.0，`trsi` 1.4.3（`trsi/__init__.py`；训练命令的 `TRSI_VERSION` 和它一致）。
- 变更：`trsi` 1.4.3（2026-11-25）训练命令只接受 `--producer train`：数据模块不再自己训练验证新训练集，好坏交给本模块和评测（`trsi/` 的代码不变）。`trsi` 1.4.2（2026-11-25）`trsi watch` 读运行目录遇到 OSError 时下一次轮询再读，连续 3 次才算出错。`trsi` 1.4.1（2026-11-25）`trsi launch` 被 ojh 拒绝的重跑不再记卡时（launch 成功后才记重跑和失败）；`trsi watch` 出错时也先结束所有在跑的训练再写 failed；训练命令结束训练出错时仍写标记；后台 collect、install 每次用新的标记（`collect-<k>.done`），换监视进程前先用 `ps` 确认进程号；install_env.sh 由系统的 `python3` 在后台启动，GPU 测试用 `resources.gpu_ids` 里一张空闲卡（`CUDA_VISIBLE_DEVICES`）；训练命令的契约写明 `progress.json`。2026-11-25 `install_env.sh` 装环境并检查，训练缺库或库不兼容时不问用户直接运行（`trsi` 不变）。`trsi` 1.4.0（2026-11-24）collect 给每个试验重新选打分卡（没有别的进程、空闲显存够、ojh 的卡锁），OOM 换卡再打一次，结果记 `scoring_card`；会话不在又没有 `DONE` 的训练算失败；`search.max_configs` 的文字写明按派出算；训练命令只接受 `--producer train`、`data`；`launch --force` 重训一直打包不成的试验；collect 打包好或记为失败后自动删用不上的 checkpoint（不问用户），候选的模型改为硬链接；换监视进程时先起新的再 `kill -9` 旧的。`trsi` 1.3.0（2026-11-24）launch、watch、collect 和训练命令写进度文件 `progress.json`。2.0（2026-11-24）改造成框架的训练调优模块：配置空间来自任务定义，不再单独批准；run.yaml 由 `trsi init` 按简报生成；起点接当前最佳版本的候选；每个试验打包成候选；训练期间以长任务交回、定时检查；两段式交回 `search`；GPU 由 agent 从 `resources.gpu_ids` 里选。1.0（2026-11-23）是独立使用的 Training RSI skill（`trsi` 1.0.0）。
- 已知限制：`trsi` 没有结束单个试验的命令（用 `tmux kill-session`）；`trsi collect` 一次打包所有跑完的试验，不能只打包一个；默认门槛 0.035 只在 Elliptic 上量过（见 `reference/protocol.md`）。

## 常见错误

| 现象或错误 | 处理 |
|---|---|
| `trsi launch` 说 outside the task's search_space | 候选超出任务的配置空间。换候选；要扩大配置空间是改任务定义，写进 `questions` |
| `trsi launch` 说 uses a dead end | 先看死路的原因；前提没变就换候选 |
| `trsi launch` 说 budget | 简报给的预算不够再跑一个种子：停，复盘后交回 |
| `trsi launch` 说 no cost estimate | 任务的 `resources.onejev.estimate` 没有这个底座：不跑它，写进 `questions` |
| `trsi launch` 说 comes after the probe | 先 `trsi collect`、`trsi next`，探针过了才补种子 |
| `trsi launch` 说 this dispatch (returns included) has tried search.max_configs | 这次派出提的候选数到了上限（退回后再搜也算在内）：不再提新候选，跑完已提的就停 |
| `trsi launch` 说 not in the task's resources.gpu_ids，或 recipe trains on N card(s) | 换 `resources.gpu_ids` 里的空闲卡，卡数和配方一致 |
| `trsi launch` 说 already part of this run 或 reused | 同一配置同一种子已经训练过（或还在跑），不会重训，这是预期。但刚结束的试验显示 (running)，说明 collect 还没把它记成 failed：再 `trsi collect`，看到 failed 再 launch |
| `ojh run` 非零退出（卡忙、放不下、不可行、没有默认配方） | 卡忙就换另一张空闲卡；其余把 ojh 的原话和它列的替代方案写进 `questions`，由用户选 |
| `trsi collect` 显示 failed | 看运行目录的 `DONE`、`runner.log`、`train-s<种子>.log`，按 onejev-harness 的故障表处理。卡时照记。修好后用同一条 `trsi launch` 命令重跑 |
| `trsi collect` 显示 unpacked | 打包（`model.predict`、`rsi score`、`rsi validate`）出错，原因在同一行（`trsi next` 也打印）：不重训，下一次 `trsi collect` 再打包；一直不成就用同一条 `trsi launch` 加 `--force` 重训（卡时照记），或在 `record` 和 `notes` 里写明，这个试验不交回 |
| `trsi collect` 说 no card … can score this candidate now | 逐卡写了原因（别人的进程、空闲显存不够、卡锁被占）：等卡空出来再 `collect`；一直没有就写进 `questions` |
| 训练因为缺库或库不兼容失败：`ImportError`、`ModuleNotFoundError`；H100、H200 上第 1 步反向就失败，日志是 `Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results ... install tilelang`（`DONE` 的 stage 是 train）；或日志警告 `causal_conv1d_fn is falling back to its reference PyTorch implementation`（能训练，但慢） | 用户已经授权自动安装，不用问。`<ojh 仓库>` 是 `resources.onejev.python` 去掉末尾的 `/.venv/bin/python`，`<machine>`、`<env_file>` 是 `resources.onejev.machine`、`resources.onejev.env_file`（没有 env_file 就不写 `--env-file`）。先在前台只读检查：`bash <模块目录>/install_env.sh --ojh <ojh 仓库> --machine <machine> --env-file <env_file> --check`（不到一分钟，看缺什么、会做什么，写进 `notes`）。然后一律按表下的写法在后台运行完整的脚本，以 `long_job` 交回：即使 `--check` 说什么都不用做，GPU 测试在内核缓存是空的时候也可能超过 Bash 的 10 分钟（setup.sh、从源码编译更慢）。收到「# 长任务已结束」后读标记第二行的 `exit=<退出码>` 和 `install-<k>.log` 末尾的 summary。退出码 0：把 summary（装了什么、改了机器配置的哪些键、备份和日志的路径）写进 `notes` 和 `record`，再用同一条 `trsi launch` 命令重跑（卡时照记）。退出码非 0：把 failed 行原话写进 `questions`，不重跑。用法见 `README.md`「装环境」 |
| 训练完成，收尾被 signal 15 杀掉（`DONE` 的 stage 是 finalize） | 机器上的显卡看门狗杀了占着卡空等的进程（见 `reference/lessons.md`）。写进 `questions`；collect 记为 failed 时已删了它的 checkpoint，只能重跑 |
| `trsi next` 说 incumbent seeds … not finished | 起点的种子还没跑完、没接过来或没打包好：补 `trsi launch <run.yaml> --start --seed 1 --gpus <卡>` 这样的种子，或再 `trsi collect` |
| `trsi watch` 说 marker exists | 每次 watch 用新的标记路径 |
| `trsi adopt` 拒绝（不是起点的配置、种子不在 seeds 里、训练集或验证集不同、训练代码不同） | 改为训练起点的种子：`trsi launch <run.yaml> --start --seed 0 --gpus <卡>` |
| `trsi init` 说任务没有 `resources.onejev.default_config`，或 `best_candidate` 没有 `config_request` | 任务定义或候选缺字段：以 `failed` 交回，`notes` 写原因 |
| 用 SIGTERM 结束监视进程 | 它会结束所有试验的训练。换卡重跑时用 SIGKILL（`kill -9`） |
| 定时检查里换了监视进程，却交回旧标记 | 旧标记永远不会写出，引擎一直等。交回新的标记 |
| 交回 `long_job` 后还在等、反复看日志 | 浪费额度：交回后结束会话，引擎到时会让你检查 |
| 直接 `ojh run`、用训练命令跑试验，或手改 run.yaml、exp.yaml、`archive.jsonl`、`incumbent.yaml`、`trials/` | 记账、复用、比较和打包都会漏掉它：试验只经 `trsi launch` |
| 自己运行 `ojh cleanup`，或手工删 checkpoint、候选目录 | 用不上的 checkpoint 由 `trsi collect` 自动删（第 7 步）；候选目录和 ojh 运行目录不删 |
| `trsi collect` 说 checkpoints not deleted | 下一次 collect 再删；磁盘因此不够时写进 `questions` |
| 读评测集、测试集或任务 `data.sources` 写的原始数据；用评测集挑配置或 checkpoint | 违反隔离：只用简报给的训练集和验证集 |

后台运行 install_env.sh（`<k>` 每次加一，启动前 `rm -f <outdir>/install-<k>.done <outdir>/install-<k>.done.pid`；照第 7 步后台 collect 的写法，标记旁的 `<marker>.pid` 由 sh 自己写；标记第一行总是 ok，这样它失败时派出不会被记为失败；第二行是 `exit=<install_env.sh 的退出码>`）。起后台的是系统的 `python3`，不是 `<py>`：要装的正是这个 venv，它可能缺东西或坏了，那样的话 `<py>` 起不来，什么都不会运行，也不会写标记。GPU 测试只用 `resources.gpu_ids` 里的一张空闲卡：用 `nvidia-smi --query-gpu=index,uuid,memory.used --format=csv` 和 `nvidia-smi --query-compute-apps=gpu_uuid --format=csv` 挑一张 `resources.gpu_ids`（没写就是全部卡）里没有计算进程的卡，把它的 UUID 作为 `CUDA_VISIBLE_DEVICES` 传给脚本（不传时脚本会在所有卡里挑，可能用到别人的卡）：

```
python3 -c 'import subprocess, sys; subprocess.Popen(["sh", "-c", sys.argv[1]], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)' 'printf "%s\n%s\n" $$ "$(date +%s)" > <outdir>/install-<k>.done.pid; CUDA_VISIBLE_DEVICES=<卡的 UUID> bash <模块目录>/install_env.sh --ojh <ojh 仓库> --machine <machine> --env-file <env_file> > <outdir>/install-<k>.log 2>&1; printf "ok\nexit=%s\n" $? > <outdir>/install-<k>.done'
```

然后写 `handback.json`：`status` 是 `long_job`，`marker` 是 `<outdir>/install-<k>.done` 的绝对路径，`artifacts` 是 `[]`。
