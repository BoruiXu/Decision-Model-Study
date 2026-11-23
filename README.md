# Decision Model RSI 框架

这是一个通用的、分层的 skill 框架，用来从一句需求做出一个决策模型，并让它不断变好。用户用 prompt 描述决策任务（例如“给定一笔交易，判断放行、拦截还是转人工”），顶层把需求整理成任务定义，再调度数据、训练、调优、交付、判定五个模块，每转一圈产出一个候选版本，评测与决策判断它是否比当前最佳版本好。设计细节见 [docs/design.md](docs/design.md)。

框架只用 Python 标准库，不针对任何具体任务、机器或模型后端；这些由每次运行的任务定义给出。

## 目录

| 路径 | 说明 |
|---|---|
| `rsi/` | 命令行入口与机械步骤的代码（校验、登记、升级判定、轮次推进、运行报告等） |
| `rsi/cli.py` | `rsi` 子命令的注册与分发 |
| `rsi/__main__.py` | 支持 `python3 -m rsi` 调用 |
| `rsi/schemas/` | 各类产物和任务定义的 JSON schema |
| `skills/rsi-orchestrator/` | 顶层 workflow skill，`templates/` 里有任务说明模板、填好的任务定义样例和派活说明模板 |
| `skills/rsi-data/`、`skills/rsi-train/`、`skills/rsi-tune/`、`skills/rsi-deliver/`、`skills/rsi-judge/` | 五个模块的入口 skill，只固定输入、输出、门和单独运行方式，其余由模块负责人填写 |
| `skills/_module_template/SKILL.md` | 模块 skill 的模板 |
| `docs/design.md` | 顶层设计（中文） |
| `docs/interfaces/` | 各类产物的接口说明（中文）：总览、任务定义、数据、模型、交付、评测与计划、实验记录、模块输入输出 |
| `tests/` | 标准库 unittest 测试 |

## 安装

不需要 pip 安装，只要 Python 3.9 以上。让 Claude Code 找到 skill：把顶层 skill 和五个模块 skill 用符号链接放进 Claude Code 的 skill 目录，所有项目都用就放 `~/.claude/skills/`，只在某个项目里用就放那个项目的 `.claude/skills/`：

```bash
REPO=/path/to/RSI_decision_model_harness   # 本仓库的绝对路径
mkdir -p ~/.claude/skills
for s in rsi-orchestrator rsi-data rsi-train rsi-tune rsi-deliver rsi-judge; do
  ln -sfn "$REPO/skills/$s" ~/.claude/skills/$s
done
```

用符号链接，不要复制：顶层 skill 从自己 SKILL.md 解开符号链接后的真实路径往上找含 `rsi/cli.py` 的目录，作为仓库根目录 `<repo root>`，之后用 `PYTHONPATH=<repo root> python3 -m rsi ...` 调用命令（找不到时会问用户）；训练命令默认在 `<repo root>/skills/rsi-train/bin/train`，任务定义的 `resources.train_command`（绝对路径）可以换掉它；调用时环境里带 `PYTHONPATH=<repo root>`。`skills/_module_template/` 是写模块 skill 的模板，不用链接。

## 启动方式

装好后在 Claude Code 里直接说明需求即可；也可以显式调用 `/rsi-orchestrator`。顶层 skill 会调用下面的 `rsi` 命令完成机械步骤，模块 skill 只负责需要判断的部分。

命令行直接运行（在仓库根目录）：

```bash
python3 -m rsi --help
```

`rsi init` 建好工作区后会生成进度页面并在默认浏览器里打开概览页 `<ws>/dashboard.html`：最上面是等你在 Claude 里回答的问题（有时才显示），然后是常驻的工作流图（当前步骤高亮）和三张卡片：当前最佳、离停止还有多远、现在在等什么。`<ws>/dashboard/` 下是三个次级页面：`rounds.html`（轮次与产物：每轮一块，最新一轮展开）、`events.html`（事件记录，新的在前）、`report.html`（运行报告：`REPORT.md` 按 Markdown 显示）。顶栏显示运行总用时（运行中按浏览器时间实时计算），每轮标题显示该轮用时。四个页面每 5 秒自动刷新；之后 `rsi init`、`register`、`promote`、`close-round`、`approve`、`next`、`stop`、`report` 成功后都会一起重写（写失败只在 stderr 打 `warning: dashboard not updated: <原因>`，不影响命令）。设环境变量 `RSI_NO_BROWSER=1` 或给 `rsi init`、`rsi dashboard` 加 `--no-open` 就不打开浏览器；没有图形界面时自己打开这个文件。

## 命令一览

| 命令 | 作用 |
|---|---|
| `rsi init` | 建立工作区并写入任务定义 |
| `rsi validate` | 按产物类别的 schema 校验产物目录 |
| `rsi register` | 把产物登记进工作区的注册表，打印登记 id；delivery 有任务允许的决策翻转行时行末加 `(decision mismatches: <行数>)`；产物目录复制进工作区，evalset 和 testset 则是移进去（登记后源目录不再存在，产出模块不留标签）；trainset、candidate、search 的 id 后面括号里是模块门的结果。每轮登记的训练候选和 search 合计不超过 `stop.max_attempts_per_round`（默认 3）。candidate 和 evaluation 报告的指标、对照组、训练门、差值和区间按 `rsi score` 的定义重算，对不上就拒绝；`rsi` 不认识的指标名和基线名不核对，行末加 `(not verified: <名字>)` |
| `rsi status` | 显示工作区状态和当前轮次、生效的停止设置（用默认值的标 `(default)`）、本轮尝试次数（`attempts: <已用> of <上限>`，用完且没有候选通过训练门时提示 `rsi close-round`），然后是 `best delivery on current evalset:` 一行（当前最佳版本在当前评测集上的交付，刷新评测集后没有时写明要先重新交付），最后列出本轮已登记的产物及其门的结果（带 `predictions_test` 的交付标注里程碑） |
| `rsi promote` | 按任务定义的升级规则判定本轮评测：打印 `promote V<n>: 理由` 或 `keep: 理由`；每轮一次 |
| `rsi approve` | 记录一次用户审批（`--by`、`--quote` 用户原话；批准计划时 `--ref <plan id>`） |
| `rsi next` | 本轮已登记评测但还没有升级判定时先拒绝（要求 `rsi promote`）；然后检查停止条件，命中则打印 `stop: 理由` 并停止；否则开启下一轮，打印 `round <n>: <模块>`。本轮没有判定时拒绝，指向 `rsi promote` 或 `rsi close-round` |
| `rsi check-stop` | 只打印 `stop: 理由` 或 `continue: 理由`，不改状态 |
| `rsi close-round` | 本轮没有可评测的候选（例如尝试次数用完仍没有候选通过训练门）时关闭本轮：`--reason` 必填，记为保留（`keep`），计入已完成轮数和连续未升级轮数；之后的计划 `evaluation_id` 为 null |
| `rsi stop` | 用户要求结束运行时记下停止（`--by`、`--quote` 用户原话），任何时候都可以，已停止时拒绝；打印 `stop: user requested: <原话>` |
| `rsi report` | 打印运行报告（Markdown，中文标题），`--out <文件>` 时写进文件；不改状态，但每次运行都会重写进度页面。没有升级出版本时开头写明「没有产生可发布的版本」；运行没停止时是进度报告；含总用时和各轮用时 |
| `rsi dashboard` | 重写进度页面：概览 `<ws>/dashboard.html` 和 `<ws>/dashboard/` 下的 `rounds.html`、`events.html`、`report.html`（读 `rsi` 自己的状态和登记表，以及 `REPORT.md`、`work/notes.md` 的「待用户回答」一节和 `work/` 的目录列表），打印概览页的绝对路径并在浏览器里打开；`--no-open` 只重写不打开 |
| `rsi score` | 不需要工作区：`--task <task.json> --labels <标签行> --predictions <预测>` 按代码里固定的定义算任务列出的已知指标，加 `--train <训练集 rows.jsonl>` 算对照组，加 `--compare <另一份预测>`（和 `--clusters <inputs.jsonl>`）算主指标的差值和配对簇自助区间，打印一个 JSON 对象。模块用它算报告的数，登记时 `rsi register` 用同样的定义重算、不一致就拒绝（见 docs/interfaces/task.md 的「指标的定义」） |

升级、停止、能否进入下一轮都由这些命令按规则判定（规则在 `rsi/rules.py`），编排 agent 不自行决定。退出码：成功为 0（`rsi register` 登记成功但有需要人处理的事时，例如 evalset 或 testset 的源目录没能删掉，id 照常打印在 stdout，另在 stderr 打印以 `warning:` 开头的说明，退出码仍为 0）；前提不满足时命令拒绝执行，退出码 1，错误信息写明缺哪一步，状态不变；命令行参数写错（缺参数、缺 `--reason`、`--by`、`--quote` 这类必填选项、未知子命令或取值不在可选范围内）时 argparse 打印用法并以退出码 2 退出。

运行停止后（`rsi next` 命中停止条件，或 `rsi stop`），只允许 `rsi approve <ws> delete`（清理）和 `rsi register <ws> record <dir>`（最后的实验记录）；其他审批、登记、`rsi promote`、`rsi close-round`、`rsi next`、`rsi stop` 都被拒绝。只读的 `rsi status`、`rsi check-stop` 和不改状态的 `rsi report`（会重写进度页面）随时可用。

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
4. 三轮都没有升级：任务定义不写 `patience`（默认 3）。每轮登记三个没过训练门的候选，第四个被拒（`3 of 3 attempts used`），`rsi status` 提示 `rsi close-round`，`rsi next` 被拒并指向 `rsi promote` 或 `rsi close-round`；`rsi close-round` 关闭本轮。前两轮登记 `evaluation_id` 为 null 的计划、批准后进入下一轮（入口 `data`，登记新的训练集）；第三轮关闭后 `rsi check-stop` 和 `rsi next` 都打印 `stop:`（patience 3），`rsi report --out` 写出以「没有产生可发布的版本」开头的失败报告。
5. 三个乱序用例：没有 evaluation 时 `promote` 被拒；evaluation 的建议与规则判定不一致时登记被拒；trainset 与评测集重叠时登记被拒。
