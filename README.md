# Decision Model RSI 框架

这是一个通用的 RSI harness，用来从一句需求做出一个决策模型，并让它不断变好。用户用 prompt 描述决策任务（例如“给定一笔交易，判断放行、拦截还是转人工”），操作会话把需求整理成任务定义；之后**主流程由代码写好**：`rsi run` 里的流程引擎按工作区状态决定下一步，按 ① 数据 → ② 训练调优 → ③ 交付 → ④ 评测与决策 派四个模块（各是一个 agent），校验并登记它们的交回，做升级判定、推进轮次、检查停止。每转一圈产出一个候选版本，评测与决策判断它是否比当前最佳版本好。LLM 只出现在两处：模块内部，和代码规定好的少数决策点（默认由代码按规则选）。改动计划和设计见 [docs/plan.md](docs/plan.md)，引擎的接口见 [docs/interfaces/engine.md](docs/interfaces/engine.md)，最初的分层设计见 [docs/design.md](docs/design.md)。

框架只用 Python 标准库，不针对任何具体任务、机器或模型后端；这些由每次运行的任务定义给出。

## 目录

| 路径 | 说明 |
|---|---|
| `rsi/` | 命令行入口与机械步骤的代码（校验、登记、升级判定、轮次推进、运行报告等） |
| `rsi/cli.py` | `rsi` 子命令的注册与分发 |
| `rsi/flow.py` | 主流程状态机：`next_action(ws)` 是只读的纯函数，返回下一个动作（页面和引擎共用） |
| `rsi/engine.py`、`rsi/decisions.py` | `rsi run` 的主循环（派出、登记交回、退回、长任务、运行日志）；决策点、默认选择、问用户和 `rsi answer` |
| `rsi/briefs.py` | 代码生成模块的简报（`rsi brief`）和 `handback.json` 的格式与校验 |
| `rsi/runners/` | 调用器：`claude`（无界面 Claude Code 会话）、`mock`（测试用脚本）、`manual`（人填输出目录） |
| `rsi/__main__.py` | 支持 `python3 -m rsi` 调用 |
| `rsi/schemas/` | 各类产物和任务定义的 JSON schema |
| `skills/rsi-orchestrator/` | 很薄的“操作” skill：和用户对话起草任务定义、运行 `rsi run`、转述待答问题并用 `rsi answer` 记下回答、转述进度；不决定流程。`templates/` 里有任务说明模板、填好的任务定义样例和简报结构说明 |
| `modules/data/`、`modules/train/`、`modules/deliver/`、`modules/judge/` | 四个模块各一个文件夹，由模块负责人各自维护：入口 skill `SKILL.md`（引擎派出时把它的全文给模块 agent；固定输入、输出、门和单独运行方式），以及模块自己的代码、参考资料和测试。`modules/train/` 是训练调优模块：`trsi` 命令行、训练命令 `train`、`reference/`，见它的 `README.md` |
| `modules/_template/SKILL.md` | 模块 skill 的模板 |
| `docs/design.md` | 顶层设计（中文） |
| `docs/interfaces/` | 各类产物的接口说明（中文）：总览、任务定义、数据、模型、交付、评测与计划、实验记录、模块输入输出，以及流程引擎（`engine.md`：动作、STATE 的派出和决策记录、决策点、调用器、退出码、锁） |
| `docs/superpowers/plans/` | 实施计划 |
| `tests/` | 标准库 unittest 测试 |

## 安装

不需要 pip 安装，只要 Python 3.9 以上。让 Claude Code 找到 skill：把顶层 skill 和四个模块 skill 用符号链接放进 Claude Code 的 skill 目录，所有项目都用就放 `~/.claude/skills/`，只在某个项目里用就放那个项目的 `.claude/skills/`：

```bash
REPO=/path/to/RSI_code_driven_harness   # 本仓库的绝对路径
mkdir -p ~/.claude/skills
ln -sfn "$REPO/skills/rsi-orchestrator" ~/.claude/skills/rsi-orchestrator
for m in data train deliver judge; do
  ln -sfn "$REPO/modules/$m" ~/.claude/skills/rsi-$m
done
```

用符号链接，不要复制：操作 skill 从自己 SKILL.md 解开符号链接后的真实路径往上找含 `rsi/cli.py` 的目录，作为仓库根目录 `<repo root>`，之后用 `PYTHONPATH=<repo root> python3 -m rsi ...` 调用命令（找不到时会问用户）；训练命令默认在 `<repo root>/modules/train/train`，任务定义的 `resources.train_command`（绝对路径）可以换掉它；调用时环境里带 `PYTHONPATH=<repo root>`。模块 agent 由引擎派出，引擎直接把 `modules/<模块>/SKILL.md` 的全文给它，不靠这些链接；链接方便模块负责人在自己的会话里单独运行模块。`modules/_template/` 是写模块 skill 的模板，不用链接。

## 启动方式

**在 Claude Code 里**：装好 skill 后直接说明需求，或显式调用 `/rsi-orchestrator`。操作 skill 和你对话起草任务定义（`task.json` 和 `TASK.md`），你确认后它运行 `rsi init`，再用后台方式运行 `rsi run`；运行停在要问你的地方时，它把问题和选项转述给你，把你的原话用 `rsi answer` 记下来，再继续 `rsi run`。它不决定流程。

**直接用命令行**（在仓库根目录，`python3 -m rsi --help` 看全部命令）：

```bash
python3 -m rsi validate task <任务目录>          # 任务目录里有 task.json 和 TASK.md
python3 -m rsi init <ws> --task <任务目录>        # 建工作区，生成并打开进度页面
python3 -m rsi run <ws>                            # 默认 --runner claude：每个模块是一个无界面的 Claude Code 会话
# 退出码 10：在等你回答。问题和选项在打印的那一行里，也在进度页面的“现在在等什么”里
python3 -m rsi answer <ws> <选项> --by <你的名字> --quote "<你的原话>"
python3 -m rsi run <ws>                            # 接着走，直到结束（退出码 0）
```

`rsi run` 会跑很久（模块 agent 在跑时它一直等着），可随时打断，再运行就接着走。`rsi brief <ws> <模块>` 打印一个模块现在会拿到的简报（代码按工作区状态生成，不需要派出记录），用来检查或手动派活：`rsi run --runner manual` 只写简报，等输出目录里出现 `handback.json`。注意模块 agent 默认拿全部权限，见下面的已知限制。

`rsi init` 建好工作区后会生成进度页面并在默认浏览器里打开概览页 `<ws>/dashboard.html`：最上面是等你在 Claude 里回答的问题（有时才显示），然后是常驻的工作流图（这四个阶段，加顶层核对、停止与交付；当前步骤高亮，并多一行模块进度，例如「配置 3/3」）和三张卡片：当前最佳（加本轮验证集上最好的候选）、离停止还有多远、现在在等什么；再往下是「现在在做什么」（每个打开的派出：第几次启动、定时检查、完成标记，以及模块写的 `progress.json`：摘要、要点、试验表、更新时间，过时会标出）、「agent 的判断」（最近 6 份交回的 notes 和问题）、「GPU」（任务的 `resources.gpu_ids` 或全部卡：显存、利用率、本用户和其他用户的进程，来自 `rsi run` 等待时采的 `<ws>/work/gpus.json`）和「最近事件」（事件记录的最近 10 条）。`<ws>/dashboard/` 下是三个次级页面：`rounds.html`（轮次与产物：每轮一块，最新一轮展开）、`events.html`（事件记录，新的在前）、`report.html`（运行报告：`REPORT.md` 按 Markdown 显示）。顶栏显示运行状态和运行总用时（和运行报告一样从第一次 `rsi run` 算起，没有运行过时从 `rsi init` 算起；运行中按浏览器时间实时计算），每轮标题显示该轮用时；状态只在有 `rsi run` 在跑（持有 `<ws>/work/run.lock`）时是「● 运行中」，否则是「等待 rsi run」或在等什么（等你回答、等长任务完成、等手动交回、已停止）。四个页面每 5 秒自动刷新；之后 `rsi init`、`register`、`promote`、`close-round`、`approve`、`next`、`stop`、`report` 成功后都会一起重写（写失败只在 stderr 打 `warning: dashboard not updated: <原因>`，不影响命令）。设环境变量 `RSI_NO_BROWSER=1` 或给 `rsi init`、`rsi dashboard` 加 `--no-open` 就不打开浏览器；没有图形界面时自己打开这个文件。

## 命令一览

| 命令 | 作用 |
|---|---|
| `rsi init` | 建立工作区并写入任务定义 |
| `rsi validate` | 按产物类别的 schema 校验产物目录 |
| `rsi register` | 把产物登记进工作区的注册表，打印登记 id；delivery 有任务允许的决策翻转行时行末加 `(decision mismatches: <行数>)`；产物目录复制进工作区，evalset、testset 和 corpus 则是移进去（登记后源目录不再存在，产出模块不留标签）；trainset、candidate、search 的 id 后面括号里是模块门的结果。每轮登记的 search 不超过 `stop.max_attempts_per_round`（默认 3）：一次搜索算一次尝试，它的试验候选不单独算。candidate 和 evaluation 报告的指标、对照组、训练门、差值和区间按 `rsi score` 的定义重算，对不上就拒绝；`rsi` 不认识的指标名和基线名不核对，行末加 `(not verified: <名字>)` |
| `rsi status` | 第一行 `now: <状态行>`：现在在做什么，和 `rsi run` 打在 stderr 的状态行是同一行；然后显示工作区状态和当前轮次、生效的停止设置（用默认值的标 `(default)`）、本轮尝试次数（`attempts: <已用> of <上限>`，用完且没有候选通过训练门时提示 `rsi close-round`），然后是 `best delivery on current evalset:` 一行（当前最佳版本在当前评测集上的交付，刷新评测集后没有时写明要先重新交付），最后列出本轮已登记的产物及其门的结果（带 `predictions_test` 的交付标注里程碑） |
| `rsi promote` | 按任务定义的升级规则判定本轮评测：打印 `promote V<n>: 理由` 或 `keep: 理由`；每轮一次 |
| `rsi approve` | 记录一次用户审批（`--by`、`--quote` 用户原话；批准计划时 `--ref <plan id>`）；运行正在等这项审批时，同时算作回答（清掉待答问题，记一条决策） |
| `rsi answer` | 回答运行在等的问题：`rsi answer <ws> <选项> [--ref <问题 id>] --by <name> --quote <用户原话>`，选项必须是问题给出的之一（`rsi run` 停下时打印，连同问题 id `q-NNNN`；`--ref` 不是现在在等的问题时拒绝）。审批类问题：`approve` 记审批，`stop` 停止运行，`revise`（计划）把原话发回 rsi-judge 改计划；交回的候选会超出算力预算时，那份交回暂不登记、先问 `over_budget`，`approve` 后下一次 `rsi run` 登记同一份交回（不重训），本轮做到判定后运行因预算停止；决策点的问题：`resume`、`new_session`、`close_round`、`stop`（训练门没过时还有 `train`、`data`），`close_round` 和 `stop` 立即生效，其余由下一次 `rsi run` 执行；里程碑 `yes` / `no`；模块交回时提的问题回答 `已回答`：这次交回的产物在回答前都不登记，下一次 `rsi run` 把原话送回提问模块的同一个会话，它重新交回后才登记。回答记进 STATE 的 `decisions` 和 `<ws>/work/run.jsonl`；打印 `answered <问题>: <选项> (by <name>; <接下来>)` |
| `rsi next` | 本轮已登记评测但还没有升级判定时先拒绝（要求 `rsi promote`）；然后检查停止条件，命中则打印 `stop: 理由` 并停止；否则开启下一轮，打印 `round <n>: <模块>`。本轮没有判定时拒绝，指向 `rsi promote` 或 `rsi close-round` |
| `rsi check-stop` | 只打印 `stop: 理由` 或 `continue: 理由`，不改状态 |
| `rsi close-round` | 本轮没有可评测的候选（例如尝试次数用完仍没有候选通过训练门）时关闭本轮：`--reason` 必填，记为保留（`keep`），计入已完成轮数和连续未升级轮数；之后的计划 `evaluation_id` 为 null；本轮失败了、长任务可能还在跑的派出先关闭，保存后终止它们的长任务（见 docs/interfaces/engine.md「中途停止」） |
| `rsi stop` | 用户要求结束运行时记下停止（`--by`、`--quote` 用户原话），任何时候都可以，已停止时拒绝；打印 `stop: user requested: <原话>` |
| `rsi report` | 打印运行报告（Markdown，中文标题），`--out <文件>` 时写进文件；不改状态，但每次运行都会重写进度页面。没有升级出版本时开头写明「没有产生可发布的版本」；运行没停止时是进度报告；含总用时（从第一次 `rsi run` 算起，没有运行过时从 `rsi init` 算起，并写明工作区建立的时间）和各轮用时 |
| `rsi dashboard` | 重写进度页面：概览 `<ws>/dashboard.html` 和 `<ws>/dashboard/` 下的 `rounds.html`、`events.html`、`report.html`（读 `rsi` 自己的状态和登记表，以及 `REPORT.md`、`work/notes.md` 的「待用户回答」一节、`work/run.jsonl`、`work/run.lock` 里的进程号和 `work/` 的目录列表），打印概览页的绝对路径并在浏览器里打开；`--no-open` 只重写不打开 |
| `rsi brief` | 打印一个模块的简报（Markdown）：输入路径、输出目录、预算、命令、交回格式，全部由代码按工作区状态生成；`--purpose` 指定派出的目的。引擎派出时把同样的简报写进输出目录的 `brief.md` 和 `brief.json` |
| `rsi run` | 由引擎自动走主流程：按 `rsi/flow.py` 的 `next_action` 派模块 agent（`--runner claude` 默认，无界面的 Claude Code 会话；`mock` 用 `--mock-dir` 里的模拟脚本；`manual` 只写简报，等输出目录里出现 `handback.json`）、登记交回、按代码默认做决策，直到要问用户、等长任务、等手动交回、运行结束或用完 `--max-steps`（默认 200），打印停下的原因（一行中文；`--json` 时打印 `{"status": ..., "reason": ...}`）。每个动作和结果记进 `<ws>/work/run.jsonl`，每步后重写进度页面；等 agent 期间，以及开了定时检查（`flow.checkin_minutes`）时等长任务期间，每 `flow.page_refresh_seconds`（默认 60）秒采一次 GPU 快照（`<ws>/work/gpus.json`）并重写页面；没开定时检查时，等长任务的 `rsi run` 以退出码 11 停下，页面到下一个 `rsi` 命令（或 `rsi dashboard`）才更新。状态变化时和等待期间每 `flow.status_minutes`（默认 5）分钟往 stderr 打一行状态，例如 `20:23 第1轮 ② 训练调优 d-0001 · 第 3 批：3 个训练在跑（卡 0、4、6） · 配置 3/3 · 下次检查 20:46`（stdout 不变，`--json` 时仍只有那一个 JSON 对象）。随时可以打断，再运行就接着走，不重复派出、不重复登记。退出码见下 |
| `rsi score` | 不需要工作区：`--task <task.json> --labels <标签行> --predictions <预测>` 按代码里固定的定义算任务列出的已知指标，加 `--train <训练集 rows.jsonl>` 算对照组，加 `--compare <另一份预测>`（和 `--clusters <inputs.jsonl>`）算主指标的差值和配对簇自助区间，打印一个 JSON 对象。模块用它算报告的数，登记时 `rsi register` 用同样的定义重算、不一致就拒绝（见 docs/interfaces/task.md 的「指标的定义」） |

升级、停止、能否进入下一轮都由这些命令按规则判定（规则在 `rsi/rules.py`），编排 agent 不自行决定。退出码：成功为 0（`rsi register` 登记成功但有需要人处理的事时，例如 evalset 或 testset 的源目录没能删掉，id 照常打印在 stdout，另在 stderr 打印以 `warning:` 开头的说明，退出码仍为 0）；前提不满足时命令拒绝执行，退出码 1，错误信息写明缺哪一步，状态不变；命令行参数写错（缺参数、缺 `--reason`、`--by`、`--quote` 这类必填选项、未知子命令或取值不在可选范围内）时 argparse 打印用法并以退出码 2 退出。`rsi run` 的退出码表示它停在哪里：0 运行已结束（已停止并写好报告），10 在等用户回答，11 在等长任务的完成标记，12 在等手动交回，13 用完了 `--max-steps`；被拒或出错为 1（`--json` 时 status 为 `done`、`waiting_user`、`waiting_long_job`、`waiting_manual`、`max_steps` 或 `error`）。

运行停止后（`rsi next` 命中停止条件，或 `rsi stop`），只允许 `rsi approve <ws> delete`（清理）和 `rsi register <ws> record <dir>`（最后的实验记录）；其他审批、登记、`rsi promote`、`rsi close-round`、`rsi next`、`rsi stop` 都被拒绝。只读的 `rsi status`、`rsi check-stop` 和不改状态的 `rsi report`（会重写进度页面）随时可用。

## 已知限制

模块 agent 有全部权限（`claude -p --dangerously-skip-permissions`），和工作区在同一台机器上，所以技术上读得到工作区里评测集和测试集的标签。隔离靠简报只给必要路径、模块 skill 的约定，以及 `claude` 调用器按简报自动传的禁止读取规则（`--disallowedTools`，简报没给的已登记产物整个目录、只给了部分文件的数据集其余文件、data 别的派出的输出目录，Read 和 Edit 都禁；决策 agent 也有名单；名单记在派出记录的 `denied`）；登记时的指标重算保证报上来的数字是真的，但挡不住“偷看标签来调模型”。以后需要更强隔离时，可以加一个在容器或独立用户下运行模块的调用器。禁止规则只是防 agent 自己的工具（Read、Edit 和点名文件的 `cat` 之类命令）误读的护栏，不是隔离：它启动的子进程（`python` 里 `open()`、`grep -r`、`find`、`cp` 等）不受限制；工作区路径含 `( ) * ? [` 时拒绝启动；`mock` 和 `manual` 调用器只记名单不强制。

原始数据和带标签的副本：数据模块第一轮之后，原始数据（任务的 `data.sources`）只靠操作者在 corpus 登记后的 `chmod 000` 挡住（orchestrator skill「运行」里在后台等 corpus 登记再锁的那一行）；不锁的话，之后的 agent 用子进程仍读得到它。同一台机器上别处的带标签副本（原始数据的拷贝、下载目录、以前运行的工作区）不在禁读名单里，也不会被锁。获批刷新评测集的 data 派出拿整个 corpus，其中有封存 test 和以前各评测集的标签：它只通过 `dset refresh` 用 corpus，不自己打开，这一点靠约定。

模块 agent 还带着你的 Claude Code 配置：`claude -p` 照常读你的用户设置（hooks、权限、MCP 服务器）、输出目录往上各级的项目设置和 `CLAUDE.md`、以及你的记忆。任务定义的 `runner.claude.setting_sources`（`user`、`project`、`local` 选几个，传给 `--setting-sources`）能限制读哪些设置文件；`CLAUDE.md` 和记忆不受它影响（见 docs/interfaces/engine.md「调用器」）。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## 演练

`tests/test_e2e.py` 用虚构的二分类任务 `spam-check` 和 `tests/fixtures.py` 生成的合成产物，通过真实的 `python -m rsi` 命令走完两轮、一次里程碑发布和一次三轮都没有升级的运行，不运行任何真实模块。合成的候选和评测报告里的数都由 `rsi/metrics.py` 从合成预测算出（例如让某几行预测错），登记时照常重算核对：

```bash
python3 -m unittest discover -s tests -p test_e2e.py -v
```

它展示的流程：

1. 第一轮：`init`，依次登记 testset、evalset、valset、trainset（各集合行 id 互不重叠），再登记 candidate、delivery、evaluation；`rsi promote` 打印 `promote V1`；登记 plan（入口模块 `train`）后，未批准时 `rsi next` 被拒，`rsi approve ... plan` 之后打印 `round 2: train`。
2. 第二轮：登记新一轮的 candidate、delivery、evaluation（主指标变差）；`rsi promote` 打印 `keep`；登记并批准 plan 后，因任务定义里 `stop.patience: 1`，`rsi next` 打印 `stop:`，`rsi status` 显示已停止。
3. 里程碑发布：第一轮升级出 V1 后达到 `stop.target`，`rsi check-stop` 打印 `stop:`；记一次 `use_testset` 审批后登记 V1 候选在 evalset 和 testset 输入上的 delivery，再记一次审批后登记里程碑评测（`uses_testset` 为 true，带 `metrics_test`，建议 keep）；`rsi promote` 拒绝它，`rsi next` 打印 `stop:`。
4. 三轮都没有升级：任务定义不写 `patience`（默认 3）。每轮登记三次搜索（每次两个没过训练门的试验候选和它们的 search），第四次的 search 被拒（`3 of 3 attempts used`），`rsi status` 提示 `rsi close-round`，`rsi next` 被拒并指向 `rsi promote` 或 `rsi close-round`；`rsi close-round` 关闭本轮。前两轮登记 `evaluation_id` 为 null 的计划、批准后进入下一轮（入口 `data`，登记新的训练集）；第三轮关闭后 `rsi check-stop` 和 `rsi next` 都打印 `stop:`（patience 3），`rsi report --out` 写出以「没有产生可发布的版本」开头的失败报告。
5. 三个乱序用例：没有 evaluation 时 `promote` 被拒；evaluation 的建议与规则判定不一致时登记被拒；trainset 与评测集重叠时登记被拒。
