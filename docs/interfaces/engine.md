# 流程引擎接口

`rsi run <ws>` 运行流程引擎：主流程是代码（`rsi/flow.py` 的 `next_action`），具体模块由 agent 实现。本文说明引擎和模块 agent、操作者之间的接口：动作类型、STATE 里引擎的记录、决策点和默认选择、调用器、同一会话的退回、等待期间的页面刷新和状态行、退出码和锁。产物本身的格式见 README.md 及各产物文档，任务定义里和引擎有关的设置（`flow`、`decisions`、`runner`）见 task.md。

## 循环

`rsi run` 反复做三件事：`next_action(ws)` 算出下一个动作（纯函数，只读工作区）；把动作和结果记进 `<ws>/work/run.jsonl`，执行动作；重写进度页面，状态变了就往 stderr 打一行状态（见「等待期间」）。遇到要问用户、等长任务、等手动交回、运行结束或用完 `--max-steps`（默认 200）就停下，打印停下的原因。所有状态都在工作区里，进程被杀、机器重启后再运行 `rsi run` 接着走：已经启动的 agent 会被重新找到（调用器按记录的进程号接上），不会重复派出，也不会重复登记。进度页面的"现在在等什么"直接来自同一个 `next_action`，不会和引擎说法不一致。

## 动作类型

`next_action` 返回一个字典，`type` 是下列之一；每个动作还带 `step`（主流程里的位置，供页面和日志用）。

| `type` | 字段 | 引擎怎么做 |
|---|---|---|
| `dispatch` | `module`、`purpose`、`reason`；deliver 另有 `candidate_id` | 写派出记录，生成简报，启动 agent |
| `resume` | `dispatch_id`、`feedback`、`mode` | 把这次派出送回它的会话。`mode`：`resume`（退回，计入 `flow.max_returns`）；`followup`（两段式交回的后半段：train 的 search、judge 的计划，是正常的继续，不算退回）；`answer`（把用户对它提问的回答送回去，它那次交回的产物还没登记，不算退回）；`checkin`（长任务的完成标记还没出现、定时检查到期：让会话检查一次环境后再交回，不算退回，见下） |
| `wait_agent` | `dispatch_ids` | 轮询在跑的 agent（和还没启动的 `pending` 记录：先启动它），直到有一个结束（手动调用器：直接停下） |
| `wait_long_job` | `dispatch_id`、`marker`、`checkin_at` | 长任务的完成标记还没出现。没开定时检查（`checkin_at` 为 null）：停下，状态 `waiting_long_job`；开了（任务的 `flow.checkin_minutes`）：不退出，在进程里等到完成标记出现、`checkin_at`（下一次检查的时间，UTC，向上取整到秒）到了或运行被停止，再走下一步 |
| `process_handback` | `dispatch_id` | 交回带问题时先问用户、什么都不登记；交回的候选会让已用算力超过预算而本轮还没有 `over_budget` 批准时，也什么都不登记，先问 `over_budget`；否则校验并登记交回的产物，处理被拒、失败 |
| `decide` | `point`、`options`、`default`、`dispatch_id`、`reason` | 在决策点上选择并执行（见下） |
| `ask` | `kind`（`approval` 或 `question`）、`what`、`ref`、`question`、`options` | 写 STATE 的 `waiting`，停下，状态 `waiting_user` |
| `promote` | `evaluation_id` | 升级判定（`rsi promote` 的逻辑） |
| `close_round` | `reason` | 关闭本轮（尝试次数用完仍没有候选通过训练门） |
| `next` | `reason` | 检查停止条件，开下一轮或停止（`rsi next` 的逻辑） |
| `report` | — | 写 `<ws>/REPORT.md` |
| `done` | — | 已停止且报告已写：状态 `done` |

顺序（第一个成立的）：已停止 → 报告或结束；STATE 里有待答问题 → 再问一次；用户已回答提问的交回 → 带着回答送回它的会话（`resume`，`mode` 为 `answer`）；有要处理的交回、失败的派出（决策点）、在跑或待启动的 agent、长任务 → 对应动作；本轮未判定 → 评测的 `promote`、尝试次数用完的 `close_round`、主流程里第一个没完成的步骤（① 数据、② 训练调优、③ 交付、④ 评测与决策，评测集用到上限和超预算先问用户）；本轮已判定 → 停止时的里程碑发布和 `next`，否则 judge 写计划、计划审批、`next`。

## STATE 里引擎的记录

STATE.json 增加三项，页面、`rsi status` 和引擎都读它们；写入都在 `STATE.lock` 保护的事务里。

**`dispatches`**：派出记录，按 id（`d-0001`…）排列。

| 字段 | 含义 |
|---|---|
| `id`、`round`、`module`、`purpose` | 编号、所在轮次、模块（`data`、`train`、`deliver`、`judge`；旧工作区还有 `tune`，照常读、不再派出：升级框架之前，先让打开的 tune 派出结束，或 `rsi stop`，因为续它（退回、续跑、定时检查、换会话）要重建简报，会报 `unknown module 'tune'`）、目的（`sets`、`trainset`、`refresh`、`train`、`deliver`、`judge`、`plan`、`milestone_deliver`、`milestone_judge`；旧工作区的 tune 派出是 `tune`） |
| `outdir` | 输出目录，相对工作区：`work/round-<n>/<模块>[-k]/`（同一轮里同一模块再派时加后缀）；里面有 `brief.md`、`brief.json`（顶层字段 `dispatch_id` 是这次派出的 id）、`prompt-<n>.md`、`agent-<n>.log`、`feedback-<n>.md`、`registered-<n>.json`（两段式交回的后半段：前半段登记了什么，见「同一会话的退回」）、`history-<n>.md`、`handback.json`（当前的）和 `handback-<n>.json`（第 n 次启动写的，下次启动前改名保留） |
| `denied` | 最近一次启动按当时简报算出的禁止读取名单（绝对路径，目录或单个文件）；每次启动（含退回、续跑、换会话，简报重建之后）重算，同时写进 run.jsonl 的 `launch` 事件；不写进简报。只有 `claude` 调用器强制，`mock`、`manual` 只记录 |
| `runner`、`session`、`pid` | 调用器名、会话 id（`claude` 是引擎生成并先记下的 uuid，`--session-id`）、进程号 |
| `status` | `pending`（记录已写、agent 还没启动）、`running`、`handed_back`（交回了、还没登记；交回带问题时一直是它，直到用户回答；因超预算暂缓登记时一直是它，直到批准后登记或运行停止）、`long_job`、`failed`、`done` |
| `launches`、`launch_mode` | 启动次数（含退回、续跑、继续、带回答、定时检查、换会话；没启动就关闭的那次不算）；最近一次启动的方式：`first`、`resume`、`continue`、`followup`、`answer`、`checkin`、`fresh` |
| `returns`、`continued`、`followups`、`answered`、`checkins`、`capped` | 退回原会话的次数（只有它计入 `flow.max_returns`）；长任务收尾续跑的次数；两段式交回的后半段（继续）的次数；把用户的回答送回会话的次数；长任务定时检查的次数（上限 `flow.max_checkins`）；因 `flow.max_continues` 记为失败时写 `max_continues` |
| `started_at`、`launched_at`、`ended_at`、`exit_code` | 时间和 agent 的退出码；`launched_at` 和 `status: running` 在调用器被调用前一起写，`pending` 的记录没有 `launched_at` |
| `marker`、`errors`、`feedback`、`candidate_id` | 长任务完成标记、登记被拒等错误原文、最近一次启动用的反馈文件、deliver 派出时要交付的候选 |
| `long_job_since`、`long_job_launch` | 长任务开始的那次启动的 `launched_at` 和启动序号（它的戳 `.launch-<n>`，见「完成标记出现」）：第一次以这个完成标记交回 `long_job` 时记下，之后以同一个标记交回（定时检查）不改，交回新的标记（或已经见过结束、记在 `marker_seen` 的同一个标记：那是新的长任务）时换成那次启动的；停止时用它确认长任务的进程（见「中途停止」） |
| `marker_seen` | 处理过一份长任务交回、读到它的完成标记（第一行是 `ok` 或别的）时记下的标记（同 `marker` 的写法）：这个长任务已经结束，标记文件之后被删掉也不再算还在运行的长任务，停止或关闭派出时不再找它。之后再有以 `long_job` 交回的（哪怕用同一个标记路径）就清掉：那是新的长任务 |
| `registered`、`refused` | 登记了的产物（`dir`、`kind`、`id`、`launch`）和登记被拒的（`dir`、`kind`、`error`、`launch`）；`launch` 是交回它的那次启动 |
| `asked` | 交回带问题时：`launch`（那次启动）、`question_id`（STATE `waiting` 的 id）、`questions`；这次交回的产物在用户回答前都不登记 |
| `held_budget` | 交回因超预算暂缓登记时：`launch`（那次启动）、`question_id`（STATE `waiting` 的 id）、`dirs`（会超预算的候选，交回里的子目录名）、`cost`（登记它们要加的算力小时；登记被拒时已计过同样花费的来源不再加）、`used`（当时的已用算力）、`budget`（`budget.compute_hours`）。批准后照常登记，这一项保留作记录：它的 `launch` 是派出现在的 `launches`、派出还是 `handed_back`（或长任务的 `long_job`）、本轮还没有 `over_budget` 批准、运行没有停止时，交回仍在暂缓（`rsi.flow.budget_held`） |
| `long_job_stop` | 运行停止或派出被关闭时长任务还没结束：完成标记的 pid 文件的 `pid`、`pid_file`、`result`（`terminated` 已终止；`gone` 进程已不在；`unconfirmed`、`no_pid_file`、`bad_pid_file`、`error`（终止时出错）没能终止，可能仍在运行；`pending`：派出已经关闭、还没来得及终止，这时只有 `result`、`context`、`at`）、`why`；`others`：输出目录里其他标记不存在的 pid 文件（还没交回的新长任务），每个也有 `pid`、`pid_file`、`result`、`why`（没有时是 `[]`）；`at`；`context`：`stop`（运行停止，或用户选了 `stop`）或 `close`（`new_session`、`close_round`、`rsi close-round` 关闭了派出）；旧记录没有 `others`、`context`，按 `stop` 读（见「中途停止」） |
| `closed` | 不再续的派出为什么关闭（用户选了 `new_session`、`close_round`、`stop`，或运行在这一步中途被停止）；关闭时状态是 `done`，`capped` 清掉 |

**`decisions`**：决策记录。字段：`point`（决策点，或 `plan`、`milestone`、`handback`、审批项名）、`options`、`choice`、`by`（`code`、`agent` 或回答的用户名）、`reason`、`quote`（用户的原话，用户回答时有）、`round`、`at`、`dispatch_id`（回答超预算暂缓的问题时是暂缓的派出）、`key`（同一个决策的识别键：轮次、决策点、派出、启动次数、派出数、本轮产物数；中断后重跑和用户的回答靠它复用同一条决策）。

**`waiting`**：引擎在等的问题，没有时是 `null`。字段：`id`（`q-NNNN`，每个新问题一个，STATE 的 `waiting_seq` 计数；同一个问题再问时不变）、`kind`（`approval` 或 `question`）、`what`（决策点或审批项）、`ref`（派出 id 或计划 id）、`dispatch_id`（可选：超预算暂缓登记时问的 `over_budget`，是暂缓的派出；这时 `ref` 仍是 `null`，所以 `rsi approve <ws> over_budget` 不带 `--ref` 也能回答它）、`question`、`options`、`since`。`rsi answer` 和 `rsi approve` 回答后清掉它。问题过时后流程不再等它，引擎清掉它：审批已经记下；带 `dispatch_id` 的审批问题，它的派出已关闭或不认识；决策点的问题，流程撇开 `waiting` 重新推一遍时已经不再给出同一个决策点、同一个派出的 `decide`（派出已续上或被换掉、这一步已经完成、本轮已判定，例如手工 `rsi close-round` 之后），或它所指的派出已关闭、不认识、属于别的轮次；里程碑问题已经有了答复（包括手工流程里判定之后记的 `use_testset` 审批）；定时检查的问题，它的派出已关闭、不再是长任务、完成标记已出现，或用户已经回答过。到了退回上限、派出状态是 `done` 但这一步没完成时（例如 train 交回了试验候选还没写 search），问题不过时，回答照常执行。

## 决策点和默认选择

代码在这些地方做选择，每个都有一个由代码算出的默认；任务定义的 `decisions.<决策点>: "agent"` 可以改成让决策 agent 判断（它必须在上限允许的选项里选一个并写理由，答案不合格就用代码的默认）。`ask` 永远在选项里；上限允许的只有 `ask` 时不调用 agent。上限是每个模块每轮的 `flow.max_returns`（默认 2）。

| 决策点 | 选项 | 代码的默认 | 变成问用户的时候 |
|---|---|---|---|
| `train_gate_failed`（本轮的 search 已登记，没有候选通过训练门，还有尝试次数） | `train`、`data`、`ask` | `train`：回到本轮的 train 会话在配置空间里再搜一次（下一次尝试） | train 已到退回上限；或这次 train 派出最近一次启动只登记了实验记录（`record`，没有新的试验候选或 search，例如 trsi 已说 STOP：再退回也只会这样） |
| `quality_gate_failed`（本轮的新训练集没过质量门） | `resume_data`、`ask` | `resume_data`：回到 data 的会话 | 没有派出记录可回（手工运行的工作区）；data 已到上限 |
| `handback_refused`（模块交回的产物登记被拒） | `resume`、`ask` | `resume`：带着登记的错误原文回到同一个会话 | 该模块已到上限 |
| `handback_failed`（交回后本步仍没完成） | `resume`、`ask` | `resume` | 该模块已到上限；或这次派出因 `flow.max_continues` 记为失败（`capped`） |

两处和代码实际行为有关的说明：

- **训练门没过的默认没有"质量相关时选 `data`"这一条。** 设计里写过"候选报告里有质量相关的失败标记就派 data"，但候选的 manifest 里没有这样的结构化标记，代码不能算，所以默认是 `train`；`data` 仍然是可选项，可以由决策 agent 或用户选。
- **训练门没过的决策记在本轮最新的 train 派出上**（`new_session` 因此新开的是 rsi-train 会话）；本轮没有 train 派出时（候选是手工登记的），是最后一个候选所在的派出，没有就是 null。
- **训练门没过时问用户的问题没有 `resume` 选项。** 选 `train` 或 `data` 就是回到那个模块在本轮的会话（没有则新派一次，带着 `feedback-1.md`），再提供 `resume` 就和其中一个重复。其他决策点的问题有 `resume`。
- **一次搜索的试验都没过训练门时，先登记 search 再决策。** 训练调优模块一次派出是一次搜索：它交回试验候选后，流程先让同一个会话写 search（后半段），search 登记了（算一次尝试）才到 `train_gate_failed`；尝试用完时关闭本轮（`close_round`），不会无休止地退回。

选择的执行：执行前先重新读 STATE，运行已被停止就什么都不启动（见「中途停止」）。`resume`、`resume_data` 把派出送回它的会话（返回次数加 1）；`train`、`data` 回到那个模块；`new_session` 关闭出问题的派出，用同一模块（和 deliver 的候选）新开一次派出，目的是这一步当前的部分：本轮已判定时 judge 的派出换成写计划（`plan`），第一轮另三个集合已登记时 data 换成训练集（`trainset`），其余沿用原派出的目的（train 写 search 仍是 `train`，简报里有已登记的试验候选）；反馈里写原因、旧派出的错误和没过的门；`close_round`、`stop` 就是 `rsi close-round`、`rsi stop` 的效果；`stop`、`new_session`、`close_round` 关闭失败的派出时（`rsi.decisions.close_dispatch`），它的长任务若还没写完成标记（例如定时检查失败了），关闭派出的那次保存（`rsi answer` 的整个事务；`stop`、`close_round` 和停止运行、关闭本轮在同一次保存里）只记下 `long_job_stop: {"result": "pending", "context", "at"}`，保存成功后再按「中途停止」第 4 步终止长任务，另存 `long_job_stop`（`rsi.decisions.terminate_after_commit`；`context`：`stop` 时是 `stop`，`new_session`、`close_round` 时是 `close`）；保存失败（例如关闭本轮被拒）时什么都没改，长任务也不动。保存之后、终止之前中断了的，记录停在 `pending`：`rsi run` 每一步开始时（和修复登记表同一处）、下一次关闭派出（`terminate_after_commit` 顺带处理所有 `pending` 的）或运行停止后的报告这一步（`settle_after_stop`）终止它，不管派出是什么状态；按启动时间确认进程，重复终止不会误伤。终止时出错记为 `error`（没能终止），其余派出照常处理。`close_round` 还和 `rsi close-round` 一样，先关闭本轮其他失败了、长任务可能还在跑的派出。运行结束时的原因照样列出这些长任务，`REPORT.md` 把 `close` 的列在「关闭派出时的长任务」下，`stop` 的列在「停止时的长任务」下（进度页面的事件记录同样写「关闭派出时…」或「停止时…」）；`ask` 写 STATE 的 `waiting` 并停下。问用户时的选项是这些出口（`resume`、`new_session`、`close_round`（本轮还能关闭时）、`stop`，训练门没过时还有 `train`、`data`），问题文字里写了每个选项的意思。

审批类的问题（`plan`：`approve`、`revise`；`refresh_sets`：`approve`、`close_round`、`stop`，批准后先派 rsi-data 刷新评测集再交付和评测；`over_budget`：`approve`、`stop`，批准后本轮接着做到判定，判定后停止检查因已用算力达到预算而停止运行）和里程碑（`yes`、`no`）也是 `ask`；问题和选项的文字只写选了之后真会发生的事：`close_round` 会使停止条件成立时（`Workspace.stop_check_if_closed`），选项写的是关闭本轮后运行停止和停止的理由，而不是 judge 写计划；模块交回时提的问题（`handback.json` 的 `questions`）的选项是 `已回答`：交回里有问题时先问用户，**这次交回的产物一个都不登记**（派出状态保持 `handed_back`，记录的 `asked` 写明是哪次启动的问题）。用户回答后，下一次 `rsi run` 把回答送回**同一个会话**（`resume`，`mode` 为 `answer`，`answered` 加 1，不算退回；反馈文件写明那次交回的产物还没登记、用户的问题和原话），模块按回答修改或确认后重新交回，这次交回才登记（被拒等照常处理）。没有产物、只有问题的交回也一样：回答送回后由它接着做。长任务（`long_job`）交回里的问题等完成标记出现后才问；它没列产物时，回答写进长任务收尾的反馈（「# 长任务已结束」，`continue`）一起送回，列了产物时同样以 `answer` 送回。用户的原话只送达一次：写进这次派出下一次启动的反馈；只有这次派出在回答送回之前就关闭了，才写进同一模块、同一目的（`purpose`）之后任何一轮新派出的简报（`brief.md` 的「用户对本模块之前提问的回答」，`brief.json` 的 `answers`）；送达后这条回答的决策记录写上 `delivered_to`，以后不再重复。运行停止时还在等回答的交回：只登记其中的实验记录（`record`），派出关闭，问题作废。judge 写计划时的简报（`plan` 派出，或本轮判定后续上的 judge 会话）列出本轮所有模块的提问和回答（「本轮用户对各模块提问的回答」，`brief.json` 的 `round_answers`）。

`rsi answer <ws> <选项> [--ref <问题 id>] --by <name> --quote <用户原话>`：选项必须是 `waiting.options` 之一，`--by` 和 `--quote` 必填；给了 `--ref` 而它不是现在在等的问题时拒绝（问题已经换了），问题已过时也拒绝；它把回答记成一条决策，清掉 `waiting`，在同一个事务里完成（`run.jsonl` 的 `answer` 行在保存前写下，保存失败时再记一行 `answer_failed`；`rsi approve` 的 `approve` 行同样）：审批 `approve`、`stop`、`close_round` 当场生效，需要 agent 去做的（回到会话、新开会话、`revise`）由下一次 `rsi run` 执行——它会遇到同一个决策点，复用这条回答。回答只对当时的状态有效：状态变了（例如又有新的派出）就重新问。

## 调用器

调用器决定怎么启动一个模块 agent（`rsi run --runner <名字>`；任务定义的 `runner.claude` 配置 `claude`）。`claude` 默认带 `--model claude-opus-5-5 --effort xhigh`，`runner.claude.args` 里写了 `--model` 或 `--effort` 就用写的。

| 调用器 | 做法 |
|---|---|
| `claude`（默认） | 无界面的 Claude Code 会话。命令 `claude -p --dangerously-skip-permissions --output-format json`（模块 agent 拿全部权限），接任务里的额外参数，然后（有禁止名单时）`--disallowedTools <规则…>`，最后新会话是 `--session-id <uuid>`，退回时是 `--resume <会话 id>`。提示词（模块 skill `modules/<模块>/SKILL.md` 的全文加 `brief.md` 加交回格式，退回时是反馈文件）写进输出目录的 `prompt-<n>.md`，从标准输入给出；工作目录是输出目录。agent 在自己的进程组里，`rsi run` 被杀后它还活着；一个小 shell 先把 pid 写进 `agent-<n>.pid` 再换成 `claude`；输出在 `agent-<n>.log` |
| `mock` | 以子进程（自己的进程组）启动 `--mock-dir` 里的 `<模块>.py` 后立即返回，引擎轮询它，和 `claude` 一样：停止时能终止它；`rsi run` 被杀后，下一次按记录的进程号找回还在跑的脚本（命令行里有这次派出的 `--outdir`、启动时间对得上）。测试和流程模拟用，不调用任何模型 |
| `manual` | 只写 `prompt-<n>.md`，不启动；输出目录里出现 `handback.json` 就算 agent 结束。状态 `waiting_manual` |

决策 agent 用同一个命令加 `--json-schema <schema>`，在自己的进程组里启动，每 0.5 秒查一次：运行被停止时立即终止它，最长 `min(timeout_s, 1800)` 秒；文件在 `work/decisions/<n>-<决策点>/`；`manual` 没有决策 agent，用代码的默认。

**禁止读取规则（只防 agent 自己的工具误读，不是隔离）。** `claude` 调用器按这次启动的简报，把读取不该拿的东西的规则传给 `--disallowedTools`（Claude Code 文档写明禁止规则在 `--dangerously-skip-permissions` 下仍然生效）。名单由 `rsi/briefs.py` 的 `denied_paths(ws, brief)` 算：工作区 `artifacts/` 下每个已登记的产物（任何 kind，简报是唯一合法输入），简报 `inputs` 里没有任何路径落在其目录中的，整个目录；trainset、valset、evalset、testset、corpus、pool 中简报只给了个别文件的（例如 deliver 只拿 `inputs.jsonl`），manifest `files` 里其余的数据文件逐个；简报给了整个目录的、或其他 kind 的产物里给了文件的（任务的 `task.json`）不列。因此 train 读不到评测集、测试集、不是当前的训练/验证集，也读不到交付的预测、评测、反馈和计划等简报没给的产物；简报给 data、train、deliver 本模块以前各轮的实验记录（`record`，只给自己的；train 另有旧工作区 tune 的），给 judge 的评测开启本轮的计划（上一轮的 `plan`），这些不在名单里，别的模块的记录、本轮的记录和别的计划仍在；deliver 读不到评测集的 `labels.jsonl`；judge 读不到测试集（里程碑 judge 除外，简报给了它测试集的标签）。任务的原始数据（每个已登记任务版本的 `data.sources` 路径）除了 data 第一轮的 sets 派出（简报的 `sources`，只给当前版本的），每次派出都在名单里；corpus 只给 data 的 refresh 和续做的 sets（corpus 已登记），pool 只给 data 产出训练集的派出和续做的 sets（pool 已登记），其余派出按上面的规则禁读。data 各次派出的输出目录（工作区 `work/` 下，可能留着没登记的带标签集合：被拒的交回、没删的 `redo/`）也在名单里，只有这次派出自己的除外。决策 agent（`decide`）的名单是所有数据集目录（含 corpus、pool）、所有 delivery、evaluation、feedback 产物、data 各次派出的输出目录和原始数据路径。`deny_rules(paths)` 给每个目录 `Read(//<绝对路径>/**)`、`Edit(//<绝对路径>/**)`，每个文件 `Read(//<绝对路径>)`、`Edit(//<绝对路径>)`（开头单斜杠是相对会话目录，所以要两个斜杠），判断不了类型的路径（不存在，或在被 `chmod 000` 的目录下）两种都给；工作区根是符号链接（如 macOS 的 `/tmp`、`/var`）时，真实路径和写法路径两种都给。名单里的路径含 `(`、`)`、`*`、`?`、`[`（或以 `!` 开头）时规则解析器会误读，`claude` 调用器拒绝启动（RsiError，派出记录不写启动），需要把工作区挪到别的路径。每条规则是 argv 里的一个独立参数；`--disallowedTools` 是可变参数（`<tools...>`，吃到下一个以 `-` 开头的参数为止），所以固定放在命令行末尾、`--session-id` / `--resume`（决策 agent 是 `--json-schema <schema>`）之后或之前紧邻，规则吞不到别的选项；提示词走标准输入。**已知范围：这是防 agent 自己的工具（Read、Edit 以及点名文件的命令如 `cat`）误读的护栏，不是隔离。** agent 启动的子进程（`python` 里 `open()`、`grep -r`、`find`、`cp` 等）不受限制。更强的限制要靠系统沙盒或独立用户。

**模块 agent 带着你的 Claude Code 配置。** `claude -p` 和你平时用的是同一个程序：它照常读你的用户设置（`~/.claude/settings.json`，包括 hooks、权限、MCP 服务器）、输出目录往上各级目录里的项目设置和 `CLAUDE.md`、以及你的记忆。模块 agent 因此可能看到和你自己的工作有关的说明。要限制它读哪些设置文件，在任务定义的 `runner.claude.setting_sources` 里列出要读的来源（`user`、`project`、`local` 中的一个或几个，传给 CLI 的 `--setting-sources`）；它只管设置文件，`CLAUDE.md` 和记忆照常读取。需要更强的隔离时，用单独的系统用户或容器运行。

模块 agent 交回时在输出目录写 `handback.json`（格式见 `rsi/briefs.py` 的 `HANDBACK_FORMAT` 和 `read_handback`，模块 skill 的「输出」节也写了）：

| 字段 | 含义 |
|---|---|
| `status` | `ok`、`failed`、`long_job` |
| `artifacts` | 输出目录下的子目录名，每个产物一个；引擎按依赖顺序登记（task、corpus、testset、evalset、valset、pool、trainset、candidate、search、delivery、feedback、evaluation、plan、record），已登记的跳过。有 `task.json`、没有 `manifest.json` 的子目录是任务新版本，只有 rsi-data 能交回，且只在它是补全切分规则时登记（在暂存的副本上按 `Workspace.split_fill_errors` 逐字节核对 TASK.md，不用任何 `task_change` 审批），登记表里它的 `producer` 是 `data`；否则拒绝，错误写「rsi-data 只能交回补全切分规则的任务新版本」和原因。交回里的 task、corpus、某个集合或 pool 被拒（或有判断不了类别的子目录，它排在最前）时，排在它后面的这些都不登记，记为被拒（「本次交回的 … 被拒，后面依赖它的集合不登记」，决策点 `handback_refused` 的说明里只写一次，后面列出没登记的类别）；`record` 照常登记 |
| `marker` | 长任务完成标记的绝对路径，`long_job` 时必填，其他写 `null` |
| `notes` | 报告（门数字、花费、问题）；`failed` 时必须写原因 |
| `questions` | 需要用户决定的事，没有写 `[]` |

处理：格式不对、缺了或模块的进程退出了却没有 `handback.json`，都是交回失败（决策点 `handback_failed`）。`questions` 不为空时先问用户，回答前不登记（见上）。登记被拒的产物留在派出记录里，决策点 `handback_refused` 当场决定。

处理一份交回的顺序：问题（`questions`）→ 长任务收尾续跑 → 超预算暂缓 → 登记。

**超预算暂缓登记。** 任务的 `approvals` 含 `over_budget`、`budget.compute_hours` 不为 `null`、本轮还没有 `over_budget` 批准时，引擎在登记一份交回之前先算：交回里还没登记的候选（`manifest` 的 `kind` 是 `candidate`、`cost.compute_hours` 是数）加起来，会不会让已用算力超过预算（`Workspace.budget_overrun`，和登记时的检查是同一个判断）。会的话这份交回**一个都不登记**（实验记录也不登记），也不算登记被拒：派出状态保持 `handed_back`，记录写 `held_budget`，STATE `waiting` 问 `over_budget`（`kind` 为 `approval`，`ref` 为 `null`，`dispatch_id` 是这次派出，选项 `approve`、`stop`），`rsi run` 以退出码 10 停下。问题文字写明哪次派出、哪些候选、要多少算力、已用多少、预算多少，以及：`approve` 批准超预算，登记这次交回的产物，本轮接着做到判定为止，判定后因算力超过预算停止运行；`stop` 现在停止运行（同 `rsi stop`），这次交回的候选不登记。用 `rsi answer <ws> approve` 或 `rsi approve <ws> over_budget` 批准后，下一次 `rsi run` 登记**同一份交回**，不重新训练（日志 `budget_released`）；同一轮里的批准对后面的交回同样有效。交回同时有问题时先问问题，回答送回会话、模块重新交回后再按预算判断。暂缓后 `rsi run` 被中断再运行，仍是同一个问题（同一个 `q` 号），不重复提问，也不重派。选 `stop`（或批准后、登记前 `rsi stop`）时，暂缓结束、交回不登记：暂缓的候选训练过，停止的同一次保存里按被拒候选的计法计入已用算力（见下）；运行停止的收尾照常：交回里的实验记录登记，派出关闭。

**被拒候选的算力。** 登记被拒的候选也训练过：它 `manifest` 的 `cost.compute_hours`（是数时）计入已用算力 `budget_used.compute_hours`，并记在 STATE 的 `budget_used.refused`（`{候选来源目录的真实路径: 小时数}`，旧工作区没有这一项时按空处理；`Workspace.refused_compute_hours()` 是它的和）。按来源目录计一次：同一目录、同样的花费再被拒，是同一次训练，不再加；花费变了算新的一次训练，再加。之后这个目录登记成功时，从 `refused` 里删掉，花费和被拒时一样就不再加（原地修好、没重训），不一样就照常加。登记因运行已停止被拒的（`StoppedError`）、`manifest` 读不了或花费不是数的、来源已登记过或内容和已登记的产物相同的，不计。停止关闭一个交回还没处理的派出时（超预算暂缓的、等回答的、交回了还没处理的），交回里列出、还没登记的候选照同样的方式计入（按来源目录和花费计一次，已登记的来源不计）：`rsi stop`（`rsi answer … stop` 也是它）计 STATE 里已是交回状态的，停止后 `rsi run` 收尾时才发现交回的（停止时 STATE 里它还在运行），由收尾计入；两处按同一个键，不会重复。这一笔用单独的一次保存记下，不随登记失败回滚。`long_job`：标记写出且第一行是 `ok` 时，列了产物就登记，没列就让同一个会话继续去收集（`continued` 加 1，最多 `flow.max_continues` 次，到了上限记为失败）；标记不是 `ok` 记为失败。里程碑发布用到封存 test 的登记（交付里带 `predictions_test`、评测的 `uses_testset` 为 true）前，引擎引用用户对里程碑问题的原话写 `use_testset` 审批。

**完成标记出现。** 标记文件在，而且修改时间不早于长任务开始的那次启动才算出现（`rsi.flow.marker_done`）。那次启动的时间取它的戳：引擎每次启动 agent 之前在输出目录写 `.launch-<n>`（`<n>` 是记录的 `launches`），和标记在同一个文件系统上，修改时间出自同一个时钟（工作区在 NFS 上时是服务器的钟，可能和本机差几秒），比的是两个文件的时间，不留余量（标记的修改时间不早于戳的就算；文件时间只到整秒的文件系统上，和戳同一秒写的标记也算）；派出记录的 `long_job_launch` 记着是哪一次启动。没有这个戳（旧记录、旧工作区）时才用本机时间：`long_job_since`，没有时用 `launched_at`，允许早 2 秒。更早的文件是以前的长任务留下的（模块重用了标记路径而没有先删掉）：不处理交回，停止时照样终止这个长任务。所以每个长任务用新的标记路径，启动前 `rm -f <marker> <marker>.pid`。

**长任务的进程号文件。** 模块在后台启动长任务时，在完成标记旁写 `<marker>.pid`（交回 `long_job` 之前写好）：第一行是长任务进程的进程号，第二行是它的启动时间（Unix 秒，例如 `date +%s`；也接受 `2026-11-23T20:15:00Z` 这样的 UTC 时间）。第二行可以不写，但没有它时确认只靠"进程还活着、启动不早于这次派出"，进程号被复用就可能认错，所以要写。进程号必须是长任务进程本身，由任务自己写：例如 `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <命令>' > <日志> 2>&1 &`（`exec` 让命令沿用这个进程号），或 Python 任务一开始写 `os.getpid()` 和 `int(time.time())`；不要用 shell 给出的后台进程号代替，在交互 shell 里它可能是包装进程。最好让长任务在自己的进程组里运行（例如 `subprocess.Popen(..., start_new_session=True)`），停止时整组终止；它不是进程组组长时只终止这一个进程。运行停止时引擎用它终止还没结束的长任务（见「中途停止」）。

## 中途停止

用户随时可以在另一个进程里停止运行（`rsi stop`，或回答 `stop`），哪怕 `rsi run` 正在等 agent 或决策 agent。停止会停下 agent：

1. 引擎在启动或续上任何 agent 之前、做和执行决策之前、当场处理登记被拒之前，都重新读 STATE：运行已停止就不再启动，原本要启动或续上的派出关闭（状态 `done`，`closed` 写明原因），这一步结束。登记因为运行已停止被拒不算产物被拒（不是 `handback_refused`）：这次派出关闭，没登记的子目录写在 `closed` 里，它的长任务若还没写完成标记（例如定时检查交回了一部分产物），按第 4 步终止；交回里的实验记录（`record`，停止后仍允许登记）照样登记。
2. `rsi run` 在等 agent 时每次查询都重新读 STATE，两次查询之间每秒看一次：运行已停止就不再等，立即进入生成报告这一步。决策 agent 每 0.5 秒查一次，同样被终止。
3. 生成报告这一步先处理所有还是 `pending` 或 `running` 的派出：`pending` 的（还没启动）直接关闭，那次启动不计入 `launches`；`running` 的，本进程启动的用自己的进程句柄，其他的（例如 `rsi run` 被杀后留下的）通过它的调用器重新找到进程。agent 已经结束的先记下结果；还在运行、且确认是这次启动的那个进程（自己的子进程，或调用器的进程识别：`claude` 是程序对、启动时间对得上，`mock` 是命令行里有这次派出的输出目录、启动时间对得上）的，终止它的整个进程组（先 SIGTERM，几秒后 SIGKILL），记录关闭为「agent 已终止」；手动完成的或找不到进程的，只关闭记录。然后其他还开着的派出（交回未登记、失败、长任务）都关闭，交回里的 `record` 先登记。
4. 完成标记还没出现的长任务（记录里有 `marker`、标记还没出现（见「完成标记出现」）、也不是 `marker_seen` 的派出，不管状态：`long_job`，正在运行、失败了或还没启动的定时检查，长任务派出失败了的退回）：读 `<marker>.pid`；进程还活着（已结束、还没被父进程回收的僵尸进程不算活着；Linux 上僵尸进程的其他线程还在跑（主线程先退出的多线程进程），或它是进程组长而组里还有不是僵尸的进程时，算活着）、启动时间不早于长任务开始的那次启动（派出记录的 `long_job_since`，旧记录没有时用 `launched_at`；定时检查重新启动会话不改变它；按整秒比，允许早 2 秒：Linux 的 `ps` 从开机时间推算启动时间，可能早一秒）、不晚于 pid 文件的时间（文件第二行写的启动时间，没写时用文件的修改时间：长任务启动后才写它；晚了 2 秒以上就是复用了进程号的别的进程；早于它是长任务晚写了 pid 文件，仍是它），才确认是这个长任务，终止它的进程组（结果 `terminated`）；进程已不在记为 `gone`；没有 pid 文件（`no_pid_file`）、文件读不懂（`bad_pid_file`）或确认不了（`unconfirmed`，例如进程号被复用）就不动它，列为"没能终止、可能仍在运行"。同时，派出输出目录里其他 `<名>.pid`，只要 `<名>` 不存在或比 pid 文件早 2 秒以上（以前的长任务留下的标记，路径被重用了；不是记录的 `marker`、`marker_seen` 的，也不是 `claude` 调用器自己的 `agent-<n>.pid`），也按同样的方法确认后终止（启动时间不早于这次派出建立的 `started_at`）：定时检查重启了 watch、还没交回新的完成标记时，引擎只知道旧标记，新 watch 就是这样找到的。结果写进派出记录的 `long_job_stop`（标记的 pid 文件在最上层，其他的在 `others` 里），`run.jsonl` 对每个 pid 文件记 `long_job_stopped`、`long_job_gone` 或 `long_job_left`（带 `context`，`others` 里的带 `orphan: true`）；`rsi run` 结束时的原因（`--json` 的 `reason`）写出终止了哪些长任务、哪些没能终止（附原因，请用户手动结束），`REPORT.md` 的「停止状态」一节逐个列出（进程号、完成标记；`others` 里进程已不在的不列，进度页面的事件记录也不列）。完成标记已经出现的长任务不动。
5. 运行结束（`done`）之前再检查一次，任何派出都不会以 `pending` 或 `running` 留下，页面上也不再显示有 agent 在运行。

所以停止后再运行一次 `rsi run`（`rsi run` 被杀过也一样），它会停下留下的 agent、关闭记录、写报告，退出码 0。

## 同一会话的退回

门没过、登记被拒、交回失败，引擎默认让**原来的会话**接着做：用 `--resume <会话 id>` 启动，输入是反馈文件 `feedback-<n>.md`（原因、简报的变化、没过的门和数值、错误原文、上次交回的 notes、用户对它上次提问的回答、要做什么），派出记录的 `returns` 加 1，日志里记一条 `resume`。

同一个会话里还有四种不算退回的继续，都不计入 `flow.max_returns`：长任务结束后收尾（`continue`，`continued` 加 1，日志 `resume`，`mode` 为 `continue`）；两段式交回的后半段（`followup`：train 的试验候选登记后写 search，judge 的评测判定后写计划；`followups` 加 1，日志 `followup`；反馈文件第一行是「继续：交回的第二部分」，并列出前半段登记的每个产物：交回里的子目录 → 登记 id（例如 `` `cand-t3` → candidate-0007（candidate） ``），同样的内容写在反馈文件旁的 `registered-<n>.json`：`[{"dir", "kind", "id"}, …]`，n 是这次启动的编号；列的是这次派出在本轮最近一次 search 之后登记的全部产物，各次启动合在一起，所以前半段分两次启动才登记完时也都在；退回（`resume`）时同样列出并写这个文件）；把用户对它提问的回答送回去（`answer`，`answered` 加 1，日志 `resume_with_answer`）；长任务的定时检查（`checkin`，`checkins` 加 1，日志 `checkin`；反馈文件第一行是「定时检查」，写明完成标记和第几次检查，要做的是检查环境、有异常先处理，长任务还在运行就以 `long_job` 交回：如果没重启试验，就交回同一个完成标记；如果重启了试验、开了新的 trsi watch，就交回新的完成标记，引擎照收新的标记）。定时检查只在任务的 `flow.checkin_minutes` 开了时做：派出处于 `long_job`、完成标记还没出现时，从它最近一次交回起每隔这么多分钟一次；到了 `flow.max_checkins` 次、检查又到期时问用户（`ask`，`what` 为 `checkin`、`ref` 为派出 id，选项 `continue` 再检查这么多次、`wait` 不再检查只等完成标记、`stop` 停止运行），完成标记出现后问题作废、不再检查；`rsi status` 对每个这样的长任务打印一行（检查了几次、上限、下一次的时间）。后半段只在登记了前半段的那次启动之后免费给一次（记录的 `registered` 写着每个产物是哪次启动交回的）；后半段交回后仍然缺（例如还没有 search），再要就是退回。用户要求修改计划（`revise`）是退回。会话续不上（调用器报告会话已不存在或不能加载）或没有会话 id 时，新开一个会话，输入是 `history-<n>.md`（上次的简报、每次交回的 `handback-<k>.json` 和这次的反馈）加重建的简报，日志里记 `fresh_session`。`new_session`（决策的选项）是另一回事：关闭这次派出，新开一次派出。

**每次续上（退回、长任务收尾、继续、带回答、定时检查、新会话接手）都按现在的工作区重写简报。** 期间工作区可能变了（例如派了 data 出新训练集、刷新了评测集、train 有了新的试验候选）：`brief.md` 和 `brief.json` 重新生成，上一份留作 `brief-<k>.md` / `brief-<k>.json`（k 是它对应的那次启动），反馈文件的「简报有变化」一节列出变了什么（`ids`、输入、已用尝试、剩余预算），新产物要引用新的 id。

**中断后的恢复。** 状态是 `pending` 的记录还没启动，下一次 `rsi run` 启动它。记录写了 `launched_at`、调用器却找不到任何启动的痕迹（`claude`：没有 pid、pid 文件和日志；`mock`：没有日志；`manual`：没有 `prompt-<n>.md`）时，说明在启动前就被打断：引擎重新启动这一次，不算退回。登记写进了 `registry.jsonl`、STATE 却没来得及保存时，下一步开始前引擎把它补进本轮的产物列表，补上这次登记对 STATE 的影响（当前产物、预算、评测集使用次数、计划、它用掉的一次性审批），日志里记 `repair`。决策派出的 `feedback-1.md` 若在写之前被打断，启动时按记录里保存的原因重写。

## 等待期间：刷新页面、GPU 快照和状态行

**刷新。** 引擎在等（`wait_agent` 轮询 agent；开了定时检查时 `wait_long_job` 等完成标记）期间，每次轮询之后看一次：距上一次写进度页面满 `flow.page_refresh_seconds`（默认 60）秒，就先采一次 GPU 快照，再重写页面。每一步之后照旧写页面（不采 GPU）。页面仍是静态 HTML，可以被 rsync 同步到别的机器上看。没开定时检查时 `rsi run` 照旧以退出码 11 停下，之后页面只在下一个 `rsi` 命令或 `rsi dashboard` 时更新。

**页面。** 概览页在工作流图和三张卡片之后有四块随刷新更新的面板：「现在在做什么」（每个打开的派出：阶段、派出号、skill、状态、第几次启动及方式、退回次数、定时检查 c / 上限和下次检查时间、长任务的完成标记，以及输出目录里的 `progress.json`：摘要、要点、表格、notes、更新时间，过时（派出还活着而 5 分钟以上没有更新）和格式问题都标出）、「agent 的判断」（各派出输出目录里的 `handback.json` 和 `handback-<n>.json`，按文件时间最近 6 份：notes 前 300 字，全文可展开，以及 questions）、「GPU」（下面的快照：任务的 `resources.gpu_ids`，没写时全部卡；快照比 3 × `page_refresh_seconds` 旧时标「已过时」）、「最近事件」（事件记录的最近 10 条）。工作流图的当前阶段多一行：该阶段打开的派出的第一个 fact，没有时长任务写「长任务 · 检查 c 次」。「当前最佳」加本轮验证集上最好的候选，「离停止还有多远」的尝试一行在本轮有打开的训练调优派出时加「（N 次进行中）」。渲染只读这些文件，不调用外部命令、不写文件。

**GPU 快照** `<ws>/work/gpus.json`（`rsi/gpus.py`，原子写）：调两次 `nvidia-smi`（`--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu` 和 `--query-compute-apps=gpu_uuid,pid,used_memory`，都带 `--format=csv,noheader,nounits`，各自 5 秒超时；超时的进程被杀掉，杀掉后 1 秒还不退出（例如卡在内核里）就不再等它，所以等待从不会卡在 `nvidia-smi` 上；这样留下的 `nvidia-smi` 还在时不再启动新的，快照直接是 `error`「上一次 nvidia-smi 仍未结束」），写全部卡：

```json
{"at": "2026-11-24T12:46:23Z", "status": "ok", "error": null, "uid": 1001,
 "gpus": [{"index": 0, "uuid": "GPU-…", "name": "NVIDIA A100-SXM4-80GB", "memory_used_mib": 37000,
           "memory_total_mib": 81920, "utilization_pct": 97,
           "processes": [{"pid": 4242, "memory_mib": 36000, "owner": "ours", "user": "alice"}]}]}
```

`status` 是 `ok`、`missing`（没有 `nvidia-smi`）或 `error`（出错或超时，`error` 写原因，出错前读到的卡保留）。进程的 `owner`：`ours`（属主 uid 等于 `uid`，即采集它的 `rsi run` 的用户）、`other`（`user` 是属主的用户名）、`unknown`（查不到属主，例如另一个 pid 命名空间的进程）；属主从 `/proc/<pid>` 读，没有 `/proc` 时用 `ps -o uid= -p <pid>`。`[N/A]` 之类的值（MIG）记为 null。只有这里的定时刷新调用 `nvidia-smi`；`rsi dashboard`、其他命令和页面渲染只读最近一次快照。框架只显示，不据此选卡、不动任何进程。

**状态行。** `rsi run` 往 stderr 打状态行（`rsi/live.py` 的 `now_line`，页面用同一份数据），stdout 不变，`--json` 时仍只有那一个 JSON 对象：

```
20:23 第1轮 ② 训练调优 d-0001 · 第 3 批：3 个训练在跑（卡 0、4、6） · 配置 3/3 · GPU 小时 1.3/8 · 当前方案 c4339e55d6 · 下次检查 20:46
```

- 开头是本地时间 HH:MM 和轮次；每个打开的派出（`pending`、`running`、`long_job`、`handed_back`）一段：「阶段 派出号 · 在做什么 · 前 3 个 fact · 下次检查 HH:MM」，几段之间用「 ｜ 」。
- 「在做什么」：STATE 在等一个和这次派出有关的问题（`waiting` 的 `ref` 或 `dispatch_id` 是它的派出号，不论哪种：交回时的提问、超预算暂缓、定时检查到上限、关于它的决策）时是 `等你回答（q-NNNN）`，排在进度之前，这时不写「下次检查」（裁定 R5：盯着 `.err` 的用户要看得出运行在等自己）；否则派出输出目录里的 `progress.json`（见 progress.md）有 summary 就用它，过时（派出还活着而 5 分钟以上没有更新）加「（进度 N 分钟没有更新）」；否则按状态：`待启动`、`rsi-<模块> 运行中（第 k 次启动）`（续接时加启动方式，例如「 · 定时检查」）、`长任务在跑，等完成标记 <文件名>`、`长任务已完成，等 rsi run 收尾`（标记已出现）、`已交回，等登记`。fact 是 `progress.json` 的前 3 个；「下次检查」只在长任务开了定时检查时有。已作废的问题（`rsi.flow.stale_waiting`）不算在等。
- 运行已停止而还有打开的派出（停止还没终止的 agent 或长任务）：这些段之前先是概览页的停止标题（「已停止，还有 N 个 agent 和 M 个长任务在运行」），再用「 ｜ 」接各段。
- 没有打开的派出：STATE 在等问题时是 `等你回答（q-NNNN）`；页面列着别的问题（还没问出的里程碑、`work/notes.md` 里记的）时是「等你在 Claude 里回答 N 个问题（进度页面上列着）」；否则是概览页「现在在等什么」的标题（例如「下一步：③ 交付」「已停止：…」）。

**什么时候打。** 每一步执行完后算 `now_key`：轮次、是否已停止、在等的问题 id、`next_action` 的类型、step 和派出号、每个打开派出的（派出号、状态、启动次数）；和上一次打出的不同、并且这一行的文字也和上一行不同，就打一行。这覆盖派出、启动或续接（含定时检查）、agent 结束（交回、长任务、失败）、登记或暂缓一份交回、决策、提问、升级判定、关闭本轮、进入下一轮、报告和结束、停止；只有进度数字变了不算。每次 `rsi run` 的第一步总会打一行（第一步就是等待时在等之前打）。等待期间 `flow.status_minutes`（默认 5）分钟没打过就再打一行（心跳），进度数字由它带出。

**出错。** 写页面、采 GPU（含 `status` 为 `error` 的快照；没有 `nvidia-smi` 不算）、打状态行出错都只在 stderr 打 `warning: <原因>`（`dashboard not updated: …`、`GPU snapshot failed: …`、`status line failed: …`），从不打断等待。同一种错误一次运行只警告一次：按哪一处出错、异常的类型和 errno 判断，不按文字（文字里可能有每次不同的临时文件名）。状态行写不出去（例如 stderr 的管道已断）时警告一次，之后不再打状态行；打警告本身出错也不影响运行。

`rsi status` 的第一行是 `now: <同样一行>`，其余行不变。

## 运行日志

`<ws>/work/run.jsonl`：每行 `{"at", "event", ...}`。事件有 `action`、`result`、`launch`、`recover`、`agent_end`、`register`、`resume`（退回，或 `mode` 为 `continue` 的长任务收尾）、`followup`（两段式交回的后半段，页面显示为"继续"）、`resume_with_answer`（把用户的回答送回会话）、`checkin`（长任务的定时检查）、`asked`（交回带问题，`held` 是暂不登记的子目录）、`budget_held`（交回因超预算暂缓登记：`dispatch_id`、`launch`、`question_id`、`dirs`、`cost`、`used`、`budget`）、`budget_released`（批准后登记暂缓的交回：`dispatch_id`、`launch`、`by` 为 `approve`）、`resume_failed`、`fresh_session`、`session_mismatch`、`clear_waiting`、`decision`（每个决策，含 `by`、`reason`、`quote`）、`decision_reused`、`decision_agent`、`decision_agent_failed`、`approval`、`continue_capped`、`closed`、`agent_stopped`、`long_job_stopped`、`long_job_gone`、`long_job_left`、`exit`；`rsi answer` 和 `rsi approve` 追加 `answer` 和 `approve`。进度页面的事件记录页把这些和登记、审批合在一起显示。

## 退出码

`rsi run` 打印一行原因（`--json` 时是 `{"status", "reason"}`），退出码表示停在哪里：

| 退出码 | `status` | 含义 |
|---|---|---|
| 0 | `done` | 已停止，报告已写 |
| 10 | `waiting_user` | 在等用户回答（原因里有问题和选项，下一步 `rsi answer`） |
| 11 | `waiting_long_job` | 在等长任务的完成标记 |
| 12 | `waiting_manual` | 在等手动交回（`--runner manual`） |
| 13 | `max_steps` | 用完了 `--max-steps`，再运行接着走 |
| 1 | `error`（`--json`） | 被拒或出错，状态不变 |

## 锁

- `<ws>/work/run.lock`：`rsi run` 运行期间对它持有独占的非阻塞 `flock`（文件里写着进程号，用于提示），第二个 `rsi run` 被拒。进度页面读这个进程号：只有它指向一个还活着的进程时，顶栏才显示「● 运行中」，否则显示「等待 rsi run」或在等什么（等你回答、等长任务完成、等手动交回）；`rsi run` 退出、放开锁之后再重写一次页面。进程不论怎样结束，内核都会释放这把锁，所以被杀的 `rsi run` 留下的文件不算锁，也就不存在两个进程同时接管的竞争。没有 `fcntl` 的平台用 `O_CREAT|O_EXCL` 建文件，进程号已不存在时用原子改名把旧文件挪开再重建。
- `<ws>/STATE.lock`：对 STATE.json 的每次读改写（登记、审批、`rsi answer`、`rsi stop`、引擎的派出记录和决策记录）都在它的独占 `flock` 下完成：重新读盘、改、一次保存。所以 `rsi approve`、`rsi answer`、`rsi stop` 可以在 `rsi run` 运行时写，互不丢失更新。读的一方不需要锁（STATE.json 是原子替换的）。没有 `fcntl` 的平台（非 POSIX）不加锁，但写入仍是原子的。
