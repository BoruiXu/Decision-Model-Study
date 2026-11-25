---
name: rsi-orchestrator
description: 在用户想从一句需求做出一个决策模型、继续一个已有的 RSI 运行、回答运行在等的问题、或询问 RSI 运行进度时使用。Use when the user wants a decision model built from a plain request, wants to continue an RSI run, answers a question the run waits on, or asks how a run is going.
---

# RSI 操作 skill

## 概述

主流程由代码写好：`rsi run` 里的流程引擎按工作区状态决定下一步，自己派模块 agent（data、train（训练调优）、deliver、judge 四个模块，各是一个无界面的 Claude Code 会话），校验并登记它们的交回，做升级判定、关闭本轮、推进轮次、检查停止、生成运行报告。你不决定流程，也不派模块，只做四件事：

1. 和用户对话，起草任务定义，经用户确认后建立工作区（「任务接入」）。
2. 运行 `rsi run`，读它的退出码和那一行原因（「运行」）。
3. 把运行在等的问题和选项转述给用户，把用户的原话用 `rsi answer` 记下来，再运行 `rsi run`（「回答问题」）。
4. 用户问进度时，读 `rsi status` 和进度页面转述（「进度」）。

规则由代码判定：任务定义、验证集、评测集、封存 test、版本登记、升级、停止都在 `rsi` 里。不自己判断该不该升级、该不该停、下一步该派谁；不手动运行 promote、next、close-round 和登记产物的命令（任务定义的新版本除外），它们由引擎运行；不手改 `STATE.json`、`registry.jsonl` 或 `artifacts/` 下的文件；不做模块的活（不造数据、不训练、不打包、不打分、不写计划）。`rsi` 拒绝时，照错误信息处理。

下文的 `rsi <子命令>` 都指 `PYTHONPATH=<repo root> python3 -m rsi <子命令>`，`<repo root>` 是框架仓库根目录，即含 `rsi/cli.py` 的目录，`<ws>` 是工作区目录。第一次用时先确定 `<repo root>`：从本文件解开符号链接后的真实路径往上找含 `rsi/cli.py` 的目录，找不到就问用户，再用 `PYTHONPATH=<repo root> python3 -m rsi --help` 确认能运行。工作区建立后，把 `<repo root>` 记进工作笔记 `<ws>/work/notes.md`，换会话后从那里取。

## 入口

- 新任务：从「任务接入」开始。
- 继续已有工作区：读 `<ws>/work/notes.md` 拿到 `<repo root>`，运行 `rsi dashboard <ws>` 重新打开进度页面，运行 `rsi status <ws>` 看现在的状态，然后从「运行」开始。`rsi run` 可续跑：进程退出、机器重启、会话中断后再运行它就接着走，不重复派出、不重复登记。
- 用户回答了运行在等的问题：「回答问题」。
- 用户问进度：「进度」。

## 任务接入

1. 对照 `templates/TASK.md` 的各节，找出用户没给的信息，一次问完，不分多轮追问。常缺的是标签及含义、模型能看到的输入、数据来源、数据质量检查的阈值、主指标和基线、目标、升级规则、停止条件、预算和资源。`templates/task.example.json` 是一个填好的样例。数据来源写成 `data.sources` 的 `{"path", "about"}`：本机原始数据的绝对路径和一句说明（数据模块第一轮读它、写预处理脚本，之后的派出都禁读）。同时告诉用户：corpus 登记后你会把这些路径设成读不了（`chmod 000`，见「运行」），之后要复跑预处理审计或重做数据时临时恢复权限，做完再锁回；这期间运行里的 agent 读不到原始数据。
2. 替用户定默认值时说明理由。`approvals` 默认勾选 `plan`、`over_budget`、`delete`（每轮计划都要用户确认），用户明确不要才去掉。训练命令默认在 `<repo root>/modules/train/train`，任务定义的 `resources.train_command`（绝对路径）可以换掉它。可选的 `flow`（`max_returns`、`max_continues`）、`decisions`（哪些决策点让 agent 判断，默认都由代码按规则选）和 `runner.claude`（命令、额外参数、超时；模块 agent 默认用 Opus 5.5、effort xhigh，`args` 里写 `--model` / `--effort` 可以换）不写就用默认，字段见 `docs/interfaces/task.md`。
3. 在工作区之外新建任务目录，写 `task.json` 和 `TASK.md`，两者逐项一致，`version` 为 1。切分规则 `data.split` 可以不写或只写一部分（例如只写各集合的行数 `sizes`）：rsi-data 在第一轮按数据补全，记成任务定义的新版本，不问用户。运行 `rsi validate task <dir>`，不通过就改到通过。
4. 把 `TASK.md` 给用户看。用户明确确认后运行 `rsi init <ws> --task <dir>`，`rsi init` 之后进度页面 `<ws>/dashboard.html` 已经生成并在浏览器里打开，每 5 秒刷新，之后每个改变工作区的 `rsi` 命令和引擎的每一步都会更新它。把这个路径告诉用户（没有图形界面时用户自己打开它）。再建 `<ws>/work/notes.md` 记下 `<repo root>`。此后任务定义固定，再改要走 `task_change`（见「审批点」）。

## 运行

运行 `rsi run <ws>`。默认 `--runner claude`：每个模块是一个无界面的 Claude Code 会话，有全部权限；`--runner manual` 只写简报，等人或别的会话把输出目录填好；`--runner mock` 加 `--mock-dir <目录>` 只用于测试。它会跑很久，模块 agent 在跑时它一直等着，所以用后台方式运行，别让会话卡在里面：在后台运行下面这一行（`PYTHONPATH=<repo root> python3 -m rsi run <ws> --json`，`<ws>`、`<repo root>` 换成实际路径）。它先打印这次运行的输出文件名（不带后缀；时间加 shell 的进程号，两次运行不会同名），再运行，把结果写进 `<文件名>.json`，状态行（stderr）写进 `<文件名>.err`，退出码写进 `<文件名>.exit`：

```
out=<ws>/work/run-out-$(date +%Y%m%d-%H%M%S)-$$; echo "$out"; PYTHONPATH=<repo root> python3 -m rsi run <ws> --json > "$out.json" 2> "$out.err"; echo $? > "$out.exit"
```

任务有 `data.sources`、corpus 还没登记时，第一次启动 `rsi run` 之后再在后台运行下面这一行：第一轮数据派出的 corpus 一登记，就把原始数据设成读不了（禁读名单只挡 agent 自己的文件工具，挡不住它启动的子进程）。先用 `ls -ld <路径>` 把每个路径原来的权限记进 `<ws>/work/notes.md`，锁了以后也记一笔：

```
until grep -q '"kind": "corpus"' <ws>/registry.jsonl; do sleep 60; done; chmod 000 <data.sources 的每个路径>
```

之后要复跑预处理（审计 corpus 的 sha256）或重做数据时，`chmod` 回原来的权限，做完立即再 `chmod 000`，并删掉复跑写出的带标签的行。

后台任务的输出就是这个文件名：记下它（换会话前写进 `<ws>/work/notes.md`），shell 变量 `$out` 到下一次调用就没了。每次运行一个文件，被运行锁拒绝的第二次运行写进它自己的文件，不会覆盖前一次的结果；所以不要猜最新的 `run-out-*.json`，那可能正是被拒的那次。它结束时会通知你；这时读 `<文件名>.json`（`{"status", "reason"}`）和 `<文件名>.exit`（退出码），按下表决定下一步。运行期间用 `rsi status` 或进度页面看进度，`tail -n 3 <文件名>.err` 看最近的状态行（`rsi run` 在状态变化时打一行，等待期间每 `flow.status_minutes` 分钟（默认 5）没打过就再打一行），不轮询日志，也不派 subagent 去看。

它停下时打印一行原因，退出码说明停在哪里：

| 退出码 | 状态 | 含义和下一步 |
|---|---|---|
| 0 | 运行已结束 | 已停止，运行报告在 `<ws>/REPORT.md`。把停止理由、当前最佳版本和主指标、报告里列出的模型包和评测报告路径交给用户（一律用报告里的登记后路径，不用模块输出目录里的）。一个版本都没有升级时报告开头写着「没有产生可发布的版本」，如实告诉用户卡在哪里，不把任何候选说成可用的版本 |
| 10 | 在等用户回答 | 那一行原因里有问题和选项：`在等你回答：…（可选：a / b）`。照「回答问题」做，再运行 `rsi run` |
| 11 | 在等长任务 | 某个模块的长任务（训练、搜索、大批量推理）在后台跑，原因里有完成标记的路径。标记写出后再运行 `rsi run`，引擎会让同一个会话收尾。用后台命令等标记，例如 `until [ -e <标记> ]; do sleep 300; done`；标记在远端主机上时用 `until ssh <host> test -e <标记>; do sleep 300; done`，`<host>` 和标记路径一起记进 `<ws>/work/notes.md`。超过预计耗时两倍还没有标记，看一次日志末尾和进程是否还在，再决定继续等还是问用户 |
| 12 | 在等手动交回 | 只有 `--runner manual` 会这样：原因里有输出目录和简报路径，简报 `brief.md` 在输出目录里。输出目录填好并写了 `handback.json` 后再运行 `rsi run` |
| 13 | 用完了 `--max-steps` | 不是故障：再运行 `rsi run` 接着走 |
| 1 | 被拒或出错（`--json` 时 status 是 `error`） | 照错误信息处理；另一个 `rsi run` 还在这个工作区上跑时，等它停下 |

`--json` 时打印 `{"status": ..., "reason": ...}`，status 是 `done`、`waiting_user`、`waiting_long_job`、`waiting_manual`、`max_steps` 或 `error`。每个动作、决策和回答都记在 `<ws>/work/run.jsonl`，进度页面的「事件记录」页把它们和登记合在一起显示。

## 回答问题

`rsi run` 退出码 10 时，问题在那一行原因里，也在进度页面概览的「现在在等什么」里，页面还列着可选项和要运行的命令。

1. 把问题和选项转述给用户：问的是什么、每个选项的意思（问题文字里已经写了）。决策点的问题文字写了代码为什么转来问用户（例如到了退回上限）；批准类问题（计划、刷新评测集、超预算）、里程碑和模块交回时的提问本来就要用户定，没有代码默认，不要说成有。不替用户选；用户问你的意见时可以说，但回答要用户自己给。
2. 用户给出回答后运行 `rsi answer <ws> <option> --by <name> --quote "<用户原话>"`：`<option>` 是问题给出的选项之一，`--by` 是回答的人，`--quote` 是用户的原话，不转述、不缩写。再加上 `--ref <qid>`：转述给用户的那个问题的 id（`q-NNNN`，`rsi run` 停下时和页面上都有），问题已经换了时 `rsi answer` 拒绝，回答不会落到别的问题上。选项不在问题给出的范围内、问题已经换了或已经没有在等的问题时，`rsi answer` 拒绝并说明原因。
3. 批准类的问题（计划、刷新评测集、超预算）也用 `rsi answer`：`approve` 记审批，`stop` 停止运行，计划的 `revise` 把用户的话发回 rsi-judge 改计划，judge 交回新计划后会再问一次。批准计划前把计划讲给用户：从哪个模块进入、每个问题的证据、要改什么、预期提升。登记后的计划在 `<ws>/artifacts/<plan_id>/manifest.json`（`<plan_id>` 在问题文字里），计划里没有花费估计。也可以用 `rsi approve <ws> plan --ref <plan_id> --by <name> --quote "<用户原话>"` 批准计划，效果相同，运行正在等这项审批时它同时算作回答。
4. 超预算（`over_budget`）的问题有两种：已用算力达到预算、还要派训练时问一次；模块交回的候选会让已用算力超过预算时，那份交回先不登记，当场问。把问题原文转述给用户（交回暂缓的那种写了哪次派出、哪些候选、要多少算力、已用多少、预算多少；两种都写了选了之后会怎样），不替用户估算、不改说法。用户批准时运行 `rsi answer <ws> approve --ref <qid> --by <name> --quote "<用户原话>"`，也可以用 `rsi approve <ws> over_budget --by <name> --quote "<用户原话>"`，效果相同（它同时算作回答）；用户不批准时运行 `rsi answer <ws> stop --ref <qid> --by <name> --quote "<用户原话>"`，运行停止，那份交回的候选不登记（它们训练过，算力仍计入已用算力）。告诉用户：批准后本轮接着做到判定为止（交回暂缓的，下一次 `rsi run` 登记那份已交回的候选，不重新训练），判定后运行因算力达到预算停止。
5. 里程碑发布（停止条件满足、有最佳版本时问一次，要用封存 test，用两次）：选项 `yes` 或 `no`。用户的原话会被引擎引用进两次 `use_testset` 审批，所以 `--quote` 一定是用户的原话。
6. 模块交回时提的问题（选项是 `已回答`）：这次交回的产物在回答前都不登记。`--quote` 写用户对这些问题的回答，下一次 `rsi run` 把它送回提问模块的同一个会话，模块按回答修改后重新交回，那时才登记。
7. 记下回答后运行 `rsi run`。`close_round` 和 `stop` 在 `rsi answer` 时就生效，其余的回答由下一次 `rsi run` 执行。

决策点的问题（训练门没过、质量门没过、交回被拒或没完成）的选项是：`resume`（回到原来的会话，带着原因再做一次）、`new_session`（新开会话）、`close_round`（关闭本轮，本轮还能关闭时才有）、`stop`（停止运行），训练门没过时还有 `train`、`data`（回到哪个模块）。默认由代码选，只有上限（`flow.max_returns`、`flow.max_continues`）用完、没有可回的会话，或任务定义设了让 agent 判断而它选了问你时才轮到用户。训练门没过的问题没有 `resume`，这是设计：选 `train` 就是回到本轮 train 的会话再搜一次。

## 进度

你可以把自己要问用户的事一行一行记在 `<ws>/work/notes.md` 的 `## 待用户回答` 小节里（问完删掉），进度页面会把这些行显示成等用户回答的问题；运行在等的问题不用记，页面直接从 STATE 读。

- `rsi status <ws>`：第一行 `now:` 是现在在做什么（和 `rsi run` 的状态行同一行），然后是第几轮、当前最佳版本和主指标、本轮尝试次数、评测集使用次数、已登记的产物及门的结果。`rsi check-stop <ws>` 只打印会不会停，不改状态。
- 后台 `rsi run` 的状态行：`tail -n 3 <文件名>.err`（`<文件名>` 是「运行」里记下的输出文件名）看最近三行，例如 `20:23 第1轮 ② 训练调优 d-0001 · 第 3 批：3 个训练在跑（卡 0、4、6） · 配置 3/3 · GPU 小时 1.3/8 · 当前方案 c4339e55d6 · 下次检查 20:46`：本地时间、第几轮，每个打开的派出在做什么（模块写的进度，或它的状态）和下次定时检查的时间。等待期间引擎每 `flow.page_refresh_seconds` 秒（默认 60）重写一次进度页面。
- 进度页面 `<ws>/dashboard.html`：概览的「现在在等什么」显示正在运行的 agent（模块、会话、开始时间、调用器）、在等的问题和要运行的命令，工作流图里运行中的模块标「● 运行中」；「现在在做什么」是每个打开的派出和模块写的 `progress.json`（试验表、更新时间），「agent 的判断」是最近的交回 notes 和问题，「GPU」是 `rsi run` 等待时采的显卡占用，「最近事件」是事件记录的最近 10 条；顶栏只在有 `rsi run` 在跑时显示「● 运行中」，否则显示「等待 rsi run」或在等什么；`<ws>/dashboard/` 下是轮次与产物（每轮派出了哪些 agent、各退回几次）、事件记录和运行报告。`rsi dashboard <ws> --no-open` 重写页面而不打开浏览器。
- 回答用户：用几句话转述第几轮、当前最佳版本和主指标、正在等什么、离停止条件还差多少。要完整的进度时，`rsi report <ws>` 打印到目前为止的运行报告。

## 审批点

引擎在这些地方停下问用户，用「回答问题」里的 `rsi answer` 记下回答。`--by` 写批准人，`--quote` 写用户的原话，不转述，不替用户批准。

| 审批项 | 谁来问 | 怎么记 |
|---|---|---|
| 确认任务定义 | 你，在 `rsi init` 之前 | 用户确认后 `rsi init <ws> --task <dir>`（工作区还没建立） |
| `task_change` | 你，用户要改已固定的任务定义时 | `rsi approve <ws> task_change --by <name> --quote "<用户原话>"`，再登记新版本（见「异常处理」） |
| `plan` | 引擎，judge 写好计划、任务要求批准时 | `rsi answer <ws> approve --by <name> --quote "<用户原话>"` |
| `over_budget` | 引擎，已用算力达到预算还要派训练时，或模块交回的候选会让已用算力超过预算时（那份交回先不登记） | `rsi answer <ws> approve --by <name> --quote "<用户原话>"`，或 `rsi approve <ws> over_budget --by <name> --quote "<用户原话>"`；批准后本轮做到判定，然后运行因预算停止 |
| `use_testset` | 引擎，里程碑发布；deliver 跑 test 预测、judge 用 test 打分各要一次，都由里程碑的 `yes` 引出 | 回答里程碑问题，两次审批由引擎写 |
| `refresh_sets` | 引擎，评测集使用次数到上限要换新评测集时 | `rsi answer <ws> approve --by <name> --quote "<用户原话>"` |
| `delete` | 你，删除模型文件、中间结果等任何产物之前；当前最佳版本的模型和 `<ws>/artifacts/` 不删 | `rsi approve <ws> delete --by <name> --quote "<用户原话>"`，然后再删。删除不经过 `rsi`，所以任务要求 `delete` 审批时先问用户、记下审批再删。例外（用户的决定）：rsi-train 的 `trsi collect` 自动删用不上的 checkpoint（打包好的试验没选中的保存点、失败或被换下的训练的全部 checkpoint），不管 `approvals` 里有没有 `delete`，不经审批 |

哪些审批必需、一次批准能用几次、在哪一轮有效，都由代码判定；缺审批时相关的步骤会被拒绝并写明缺哪一项。

## 异常处理

- 模块失败、交回被拒、门没过：引擎按决策点处理，默认带着原因回到同一个会话再做，用完上限才问用户。不要自己重派，不要改模块的输出。
- 用户中途改需求：先判断是否改到任务定义，即 `task.json` 的任何字段。
  - 不改任务定义，例如只是希望下一轮侧重某类错误：这类意见在引擎问计划时用 `revise` 说，或写进对模块提问的回答里；不要自己改简报。
  - 改任务定义：从当前版本开始改，不从你最初写的任务目录：`rsi status <ws>` 的 `task: <task_id> vN (task-000N)` 一行给出当前版本，把 `<ws>/artifacts/task-000N/` 里的 `task.json` 和 `TASK.md` 复制到一个新目录再改。rsi-data 补全过切分规则时，当前版本里有完整的 `data.split` 和 `TASK.md` 末尾的「切分规则」一节：原样保留（评测集或封存 test 登记之后，改了 `data.split` 的版本一律被拒）。新目录里的 `task.json` 和 `TASK.md` 写好后，`version` 加 1，运行 `rsi validate task <dir>`，通过后给用户确认，记 `task_change` 审批，再 `rsi register <ws> task <dir>`。之后的产物按新版本产出。改了标签或输入时，几个数据集合都可能不再适用，和用户逐个确定是否按新定义重做；换集合要 `refresh_sets` 审批，由引擎来问。
  - 换成另一个任务（`task_id` 不同）：用新的工作区，从任务接入开始。
- 评测集刷新后，停止条件的 `target` 仍按最佳版本原来那份评测的主指标判断，即旧评测集上的成绩，可能偏高或偏低。批准刷新时把这一点告诉用户。
- 用户要结束运行：先确认是结束整个运行，而不是暂停（暂停时什么都不用做，之后按「入口」继续）。然后运行 `rsi stop <ws> --by <name> --quote "<用户原话>"`，它打印 `stop: user requested: <原话>`。后台的 `rsi run` 正在等 agent 时，它在一两秒内发现停止，终止正在运行的模块 agent（和决策 agent），关闭它们的派出记录，生成运行报告并以退出码 0 退出；没有 `rsi run` 在跑（或它被杀过）时，再运行一次 `rsi run`，它会终止留下的 agent、关闭记录、生成报告（还有 `rsi run` 在跑时第二个会被运行锁拒绝，等前一个退出即可）。模块在后台的长任务也会被终止（模块在完成标记旁写了 `<marker>.pid` 时）；没能终止的列在 `rsi run` 的原因和 `REPORT.md` 里，把进程号和完成标记告诉用户，由用户决定怎么结束它。交付按「运行」里退出码 0 的做法。停止后只能补登记 `record` 和做 `delete` 审批后的清理。
- 运行报告写好后，可以在 `<ws>/REPORT.md` 末尾加一节「## 解读」：这次运行学到了什么、问题主要归因在哪、如果继续该先试什么。只转述已有的依据（judge 的评测报告和计划、各模块 `record` 的结论），每条注明出处，不自己做新的分析；加完后运行 `rsi dashboard <ws> --no-open`，页面的「运行报告」页才显示解读。
- 会话中断：重新加载本 skill，按「入口」接着走。

## 常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 自己判断升级、停止或下一步派谁 | 绕开代码里的规则，版本登记不可信 | 只运行 `rsi run`，照它打印的原因和问题做 |
| 替用户回答，或 `--quote` 写转述、缩写 | 伪造审批；引擎会把这句话引用进封存 test 的审批和模块的反馈 | 把问题转述给用户，用户的原话原样放进 `--quote` |
| 手改 `STATE.json`、`registry.jsonl` 或 `artifacts/` 下的文件 | 登记和规则检查失效 | 状态变化一律通过 `rsi` 命令 |
| 自己做模块的活（改数据、调参、补写 evaluation 或 plan），或手动登记模块的产物 | 模块负责人无法改进自己的模块，产物无人负责；绕开引擎的校验和抽查 | 交给引擎，由它回到对应模块的会话 |
| 两个 `rsi run` 同时跑在一个工作区上 | 第二个被拒绝，错误里写着另一个的进程号 | 等第一个停下；被杀的 `rsi run` 不留下锁（`<ws>/work/run.lock` 是进程持有的 `flock`），不要手删 |
| 让会话守着长任务，或反复看日志 | 浪费额度 | `rsi run` 在退出码 11 停下，标记出现后再运行 |
| 用户说结束，只是不再运行 | 工作区没有停止记录，下次会话会接着跑；用户拿不到报告 | `rsi stop` 记下用户原话，再 `rsi run` 出报告 |
| 把模块输出目录里的模型包路径交给用户 | 输出目录会被清理或改写，交出去的不是登记过的那份 | 用 `rsi report` 列出的登记后路径 |
