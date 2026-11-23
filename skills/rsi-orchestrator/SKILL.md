---
name: rsi-orchestrator
description: 在用户想从一句需求做出一个决策模型、继续一个已有的 RSI 运行、或询问 RSI 运行进度时使用。Use when the user wants a decision model built from a plain request, wants to continue an RSI run, or asks how a run is going.
---

# RSI 顶层编排

## 概述

加载本 skill 的主会话就是顶层。只有主会话能派 subagent，subagent 不能再派，所以五个模块各由主会话派出的一个 subagent 执行：`rsi-data`、`rsi-train`、`rsi-tune`、`rsi-deliver`、`rsi-judge`。

顶层只做三类事：

- 协调：把需求整理成任务定义，按主流程派模块，交接时先校验再往下传，在审批点停下问用户。
- 守规则：任务定义、验证集、评测集、封存 test、版本登记由顶层固定，模块不能改。
- 执行：按 `rsi` 的判定升级或保留、推进轮次，停止时把运行报告和结果交给用户。

机械的步骤一律调用 `rsi`：校验、登记、升级判定、关闭轮次、记录审批、推进轮次、检查停止、记录用户要求的停止、生成运行报告。顶层不做模块的活（不造数据、不训练、不打包、不打分、不写计划），也不自己判断该不该升级。`rsi` 拒绝时，照错误信息补上缺的那一步。

下文的 `rsi <子命令>` 都指 `PYTHONPATH=<repo root> python3 -m rsi <子命令>`，`<repo root>` 是框架仓库根目录，即含 `rsi/cli.py` 的目录，`<ws>` 是工作区目录。第一次用时先确定 `<repo root>`：从本文件解开符号链接后的真实路径往上找含 `rsi/cli.py` 的目录，找不到就问用户，再用 `PYTHONPATH=<repo root> python3 -m rsi --help` 确认能运行。工作区建立后，把 `<repo root>` 记进工作笔记 `<ws>/work/notes.md`。

## 入口

- 新任务：从「任务接入」开始。
- 继续已有工作区：先读 `<ws>/work/notes.md`，拿到 `<repo root>` 和在等的长任务。运行 `rsi dashboard <ws>`，重新打开进度页面。再运行 `rsi status <ws>`：最后一节列出本轮已登记的产物（id 和类别，候选的训练门和验证集主指标，训练集的质量门，search 的增益，里程碑评测），对照主流程表找到第一个没完成的步骤，从那里继续。`<ws>/work/round-<n>/` 里有交回了但没登记的输出时，先处理它们。`rsi promote` 和 `rsi next` 拒绝时会写明缺哪一步。
- 用户问进度：运行 `rsi status <ws>` 和 `rsi check-stop <ws>`，用几句话转述：第几轮、当前最佳版本和主指标、正在等什么、离停止条件还差多少。用户要完整的进度时，`rsi report <ws>` 打印到目前为止的运行报告。

## 任务接入

1. 对照 `templates/TASK.md` 的各节，找出用户没给的信息，一次问完，不分多轮追问。常缺的是标签及含义、模型能看到的输入、数据来源、数据质量检查的阈值、主指标和基线、目标、升级规则、停止条件、预算和资源。`templates/task.example.json` 是一个填好的样例。
2. 替用户定默认值时说明理由。`approvals` 默认勾选 `plan`、`over_budget`、`delete`（第一版每轮计划都要用户确认），用户明确不要才去掉。
3. 在工作区之外新建任务目录，写 `task.json` 和 `TASK.md`，两者逐项一致，`version` 为 1。运行 `rsi validate task <dir>`，不通过就改到通过。
4. 把 `TASK.md` 给用户看。用户明确确认后运行 `rsi init <ws> --task <dir>`，`rsi init` 之后进度页面 `<ws>/dashboard.html` 已经生成并在浏览器里打开，每 5 秒刷新，之后每个改变工作区的 `rsi` 命令都会更新它。把这个路径告诉用户（没有图形界面时用户自己打开它）。再建 `<ws>/work/notes.md` 记下 `<repo root>`。此后任务定义固定，再改要走 `task_change`（见异常处理）。

## 主流程

每轮按下表从上往下走。第一轮从 ① 开始；之后从 `rsi next` 打印的入口模块开始，走到 ⑤ 为止，不必派全五个模块。入口是 `train` 时可以不派 ③，入口是 `tune` 时可以不派 ②，按计划判断。② 和 ③ 只依赖训练集和验证集，可以同时派；同时派时，本轮之内训练命令和默认配置都不能改，train 的候选和 tune 的试验出自同一版训练代码（search 登记时核对各试验候选的 `code_version`），对训练代码或默认配置的改进放到两轮之间。有多个候选要打包时，④ 可以同时派多个。其余步骤等上一步交回并登记后再派。

| 步骤 | 派哪个 skill | 输入产物 | 输出产物 | 交回后运行 | 通过 / 不通过 |
|---|---|---|---|---|---|
| ① 数据 | `rsi-data` | `task`；之后各轮可加当前训练集（最后一个通过质量门的 `trainset`）、`valset`、`plan`、带真实结果的 `feedback` | 第一轮 `testset`、`evalset`、`valset`、`trainset`；之后新的 `trainset`；`record` | 逐个 `rsi validate`、抽查、`rsi register`；第一轮按 testset、evalset、valset、trainset 的顺序登记 | 通过：派 ② 和 ③。数据质量门没过：照常登记（`rsi register` 打印 `quality gate: failed`，它只作为证据，不会成为当前训练集），交回 `rsi-data` 改，或问用户。没过质量门的训练集不算尝试次数，`rsi status` 不会提示关闭本轮：反复改仍过不了、用户认为不值得再试时，运行 `rsi close-round <ws> --reason "<理由>"` 关闭本轮，按「闭环」让 judge 依据本轮记录写计划 |
| ② 训练 | `rsi-train` | `task`、`trainset`、`valset`；训练配置用默认的，或用某个已登记 `search` 的最佳配置（给出它的 id 和路径，候选的 `config_source` 写这个 id；以它的最佳配置为基础再改动时同样写这个 id，并在 `record` 里写明改了什么） | `candidate`、`record` | `rsi register <ws> candidate <dir>` | 通过：同时派了 ③ 就等它也交回，再选候选送 ④。训练门没过（`rsi register` 打印 `train gate: failed`）：这个候选不能送评，它的 delivery 会被拒绝；在本轮重派 ②、③，或派 ① 出新的训练集，直到有候选通过或尝试次数用完（见表下） |
| ③ 调优 | `rsi-tune` | `task`、`trainset`、`valset`、已登记的 `candidate` 和 `search`、剩余预算 | 若干 `candidate`（`config_source` 写 `tune`）、`search`、`record` | 先逐个 `rsi register <ws> candidate <dir>`，把 id 交回 tune 写进 search，再 `rsi register <ws> search <dir>` | 通过：选候选送 ④。没有优于默认配置：tune 把默认配置的候选作为最佳交回，照常继续（最佳比默认差的 search 登记不了） |
| ④ 交付 | `rsi-deliver` | `task`；送评的 `candidate`；`evalset` 的 `inputs.jsonl`；送评候选的 `valset_id` 所指 `valset` 的 `inputs.jsonl`（用来复现候选的验证集预测）；里程碑发布时加 `testset` 的 `inputs.jsonl` | `delivery`；可选 `feedback`；`record` | `rsi register <ws> delivery <dir>` | 通过：派 ⑤。模型包不能加载，或交付包在验证集输入上复现不了候选的 `predictions_val`（登记被拒，错误信息列出不一致的行）：交回 `rsi-deliver` |
| ⑤ 评测与决策 | `rsi-judge` | `task`；送评的 `candidate`；`evalset`（含 `labels.jsonl`）；送评候选所用 `trainset` 的 `rows.jsonl`（只用来算对照组）；本轮的 `delivery`，本轮有 `search` 时也给；当前最佳版本的版本名、`evaluation` 和 `delivery`；`feedback`；全部 `record`；里程碑发布时加 `testset`（含 `labels.jsonl`） | `evaluation`、`plan`、`record` | 见「闭环」 | 登记被拒：错误原文交回 `rsi-judge` |
| 顶层核对 | 不派 | 本轮 `evaluation`、`plan` | 版本登记 | `rsi promote <ws> <evaluation_id>`，`rsi check-stop <ws>`，审批计划，`rsi next <ws>` | `round <n>: <模块>`：从该模块开始下一轮。`stop: <理由>`：交付给用户 |

送评哪个候选按验证集选：通过训练门、验证集主指标最好的那个，有 search 时通常就是它的 `best_candidate`。不要在评测集上先比几个候选再挑，评测集只用来做升级判定，每登记一次 evaluation（里程碑评测除外）就用掉一次评测集的使用次数。没通过训练门的候选不能送评（`rsi register` 拒绝它的 delivery）。第一轮还没有最佳版本，送评的候选由 `rsi promote` 按训练门升为第一个版本，评测照常做。

每轮的尝试次数有上限，即任务定义的 `stop.max_attempts_per_round`，不写时为 3。登记一个 train 产出的候选或一个 search 算一次尝试，tune 的试验候选不单独算，它们的 search 算一次。`rsi status <ws>` 的 attempts 行显示本轮用了几次，用完后再登记会被拒绝。训练门没过时，先看候选的报告判断问题在哪：配置不好就派 ③，模型或训练方法的问题就重派 ②，数据的问题就派 ① 出新训练集，在说明里写清上次为什么没过。用完了仍没有候选通过训练门时，`rsi status` 会提示关闭本轮：运行 `rsi close-round <ws> --reason "<理由>"`，再按「闭环」走。用户认为不值得再试时，也可以提前关闭。

派 ④ 之前看 `rsi status <ws>` 的 eval uses。已经到上限时，下一个 evaluation 会登记不了，先按异常处理刷新评测集，再派 ④；只剩一次时告诉用户，这是当前评测集的最后一次。

## 派 subagent

- 每次派活都按 `templates/brief.md` 写说明，派一个 subagent，让它先加载对应的模块 skill。说明里写清本次要做什么（第一轮按任务定义，之后按计划里分给该模块的问题）和可用预算。
- 说明里只写该模块输入列出的产物路径（登记后的 `<ws>/artifacts/<id>/`，或其中的文件）和一个输出目录，不给工作区根目录，不给源目录，也不给别的产物。登记只由顶层做，模块不运行 `rsi register`。
- 模块写 manifest 要用的登记 id 由顶层从 `rsi status <ws>` 查好写进说明：任务的 `task_id` 和 `version`、本轮轮次，当前 trainset、valset、evalset、testset 的 id。派 `rsi-judge` 时还要写当前最佳版本的版本名、evaluation id 和 delivery id（best 行给出版本名和 evaluation id，delivery id 在那份 evaluation 的 `delivery_id`），供它写 `best_before`。评测集刷新过、那份 delivery 不在当前评测集上时，另写最佳版本在当前评测集上的 delivery id：读 `rsi status <ws>` 的 `best delivery on current evalset: <id>` 一行（写 `none ...` 时先按异常处理的评测集刷新重新打包），judge 用它算最佳版本一侧的指标，并把它写进 evaluation 的 `parents`。
- 每份说明都写明模块可以运行的两个 `rsi` 命令的完整形式：校验用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>`，算指标、对照组和差值用 `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <标签行文件> --predictions <预测文件>`（参数见 `templates/brief.md`），把 `<repo root>` 和 `<task.json>` 换成实际路径。模块不运行别的 `rsi` 命令。
- `rsi-deliver` 只拿输入文件的路径：evalset 的 `inputs.jsonl`，送评候选的 `valset_id` 所指验证集的 `inputs.jsonl`（里程碑发布时再加 testset 的 `inputs.jsonl`）。不给这些集合的目录，绝不给 `labels.jsonl` 或验证集的 `rows.jsonl`。
- 只有 `rsi-judge` 拿到 evalset 和 testset 的 `labels.jsonl`；它另拿送评候选的 `trainset_id` 所指训练集的 `rows.jsonl`，只用来算评测集上的对照组。`rsi-data` 交出 evalset 和 testset 之后，再派 data 时只给它输入里列出的 trainset、valset 等产物，不再给 evalset 和 testset。`rsi-train`、`rsi-tune` 拿到的数据集只有 trainset 和 valset。
- 输出目录用 `<ws>/work/round-<n>/<模块>/`，同一模块本轮再派时加序号，每个产物一个子目录。派模块前先建好它的输出目录，再运行 `rsi dashboard <ws> --no-open`，页面就显示该模块正在运行（空的输出目录表示已派出）；模块不运行 `rsi dashboard`。
- 模块之间不互相派 agent，也不要让 subagent 再派 subagent。tune 的每次试验调用 `rsi-train` 提供的、不需要 agent 参与的训练命令。
- 派 `rsi-data`、`rsi-train`、`rsi-tune` 时说明里还写训练命令的路径（简报模板的「训练命令」一行）：任务定义 `resources.train_command` 给了（绝对路径）就用它，没给就是 `<repo root>/skills/rsi-train/bin/train`。调用方运行训练命令时环境里带 `PYTHONPATH=<repo root>`，训练命令靠它运行 `python3 -m rsi score`。
- 派 `rsi-train`、`rsi-tune` 之前看 `rsi status <ws>` 的 compute hours，把剩余预算写进说明。

## 交回之后

模块交回后、登记之前，先运行一次 `rsi dashboard <ws> --no-open`，页面就不再显示它在运行。把报告里需要用户决定的事问用户（问之前先记待用户回答，见审批点），再对每个产物依次做三件事：

1. `rsi validate <kind> <dir> --task <task.json>`，`<task.json>` 是 `<ws>/artifacts/<task id>/task.json`，task id 是 `rsi status <ws>` 的 task 行括号里的登记 id。不通过就把错误原文交回产出它的模块改，不自己改产物。
2. 自己抽查一项 validate 管不到的东西，例如报告里的门数字与 manifest 一致，候选的 `config` 是完整的配置而不只是改动的几项，随手看几行预测不是全选同一个标签，随手核对质量报告里的一个数（例如行数）和文件对得上。对不上就交回模块说明。候选和评测报告的指标、对照组、训练门、差值和区间，候选预测入口和交付包的位置，交付对候选验证集预测的复现，都不用自己核对，校验和登记时代码会查。
3. `rsi register <ws> <kind> <dir>`，记下打印的 id（行首），之后的说明里用它。evalset 和 testset 登记时整个目录被移进工作区，登记后原目录不再存在，这是有意的（产出它的 data 模块不再留标签）。stderr 出现 `warning: <id> is registered, but removing its source ... failed` 时，产物已经登记，照常用打印的 id；警告里写的残留目录还存着标签，任务要求 `delete` 审批时先问用户、记 `delete` 审批，再手工删掉它。evalset 或 testset 的目录里有指向目录外的符号链接或硬链接的文件、任何产物里有指向不存在文件的链接时登记被拒，交回模块把文件本身写进目录。delivery 的 id 后面出现 `(decision mismatches: <行数>)` 时，交付包在这几行靠近决策边界的验证集样本上给出了与候选不同的决策（在任务定义 `delivery.max_decision_mismatch_rate` 允许的范围内），把行数告诉用户。trainset、candidate、search 的 id 后面括号里是模块门的结果（质量门、训练门、相对默认配置的增益），门没过时按主流程表的「不通过」处理。数对不上时登记被拒，错误信息写明字段、报告值、重算值和该用的 `rsi score` 命令，原文交回模块。行末出现 `(not verified: <名字>)` 时，这些是 `rsi` 不认识的指标或基线，代码没有核对：把名字告诉用户，并在模块报告里确认它们的算法。

同一个模块交回的产物之间有引用时（tune 的 `search` 引用它的 `candidate`，judge 的 `plan` 引用它的 `evaluation`），先登记被引用的，再把 id 告诉同一个 subagent（继续它；不行就带着 id 重新派），让它补好引用后交回。模块交回的 `record` 同样校验、登记。

任何 `rsi` 命令报错都不能绕过：不手改 `STATE.json`、`registry.jsonl` 和 `artifacts/` 下的文件，不替模块改产物去迎合检查。登记被拒的理由照原文转给模块或用户。被拒的产物从未登记过：让模块在原目录里改好，改好后可以直接再登记。重新派了模块、输出到新目录时，旧目录原样留着（不要改名、不要删，进度页面在替代的产物登记后把它标为已被取代），所以不需要 `delete` 审批。

## 闭环

本轮有送评的候选时从第 1 步开始。本轮关闭了（尝试次数用完仍没有候选通过训练门，见主流程）时，`rsi close-round` 打印 `keep: closed without evaluation: <理由>`，把这一行告诉用户；本轮记为保留，计入已完成轮数和连续未升级轮数。然后从第 3 步接着走，第 4 步派 judge 写计划时没有 evaluation：给它本轮全部候选（含训练门没过的）、本轮的 search 和全部 `record`，说明本轮已关闭，`plan` 的 `evaluation_id` 写 null，问题和证据来自这些候选的训练门结果和记录。只有关闭的轮次能登记 `evaluation_id` 为 null 的计划。

1. judge 先交回 `evaluation`：校验、抽查、`rsi register <ws> evaluation <dir>`。
2. `rsi promote <ws> <evaluation_id>`，把打印的 `promote V<n>: <理由>` 或 `keep: <理由>` 原样告诉用户。升级与否只看这一行。
3. 马上运行 `rsi check-stop <ws>`，它只打印会不会停，不改状态。
4. 打印 `continue: <理由>` 时：
   1. 把 evaluation 的 id 和判定告诉 judge，它写好 `plan` 和 `record` 后交回：校验、抽查、`rsi register <ws> plan <dir>`。
   2. 把计划讲给用户：从哪个模块进入、每个问题的证据、要改什么、预期提升、预计花费。任务的 `approvals` 含 `plan` 时，等用户明确同意，再运行 `rsi approve <ws> plan --ref <plan_id> --by <name> --quote "<用户原话>"`。用户要改计划，就把意见交回 judge 出新计划，重新登记、重新审批。
   3. `rsi next <ws>` 打印 `round <n>: <模块>`，按主流程从该模块开始下一轮。
5. 打印 `stop: <理由>` 时，运行将要停止，不再需要计划和计划审批。`rsi status <ws>` 的 best 行是 `best: none`（一个版本都没有升级）时没有可验证的版本，不提里程碑发布，直接走第 3 步。否则先问用户要不要做里程碑发布，即用封存 test 验证当前最佳版本，并说明这要用两次 test（deliver 跑预测、judge 打分）。只问用户一次；用户同意后记两次 `use_testset` 审批，两次引用同一句原话。用户同意时：
   1. `rsi approve <ws> use_testset --by <name> --quote "<用户原话>"`，派 `rsi-deliver` 给当前最佳版本的候选跑 evalset 和 testset 的 `inputs.jsonl`（另给候选所用验证集的 `inputs.jsonl`，用来复现），登记它交回的 `delivery`。
   2. 再记一次 `use_testset` 审批，派 `rsi-judge` 用 testset 的标签给这份 delivery 打分，登记它交回的里程碑评测：`evaluation` 的 `uses_testset` 为 true，`candidate_id` 是当前最佳版本的候选，`best_before` 写当前最佳版本，test 上的成绩写在 `metrics_test`，建议写 keep。它不占评测集使用次数，不做 `rsi promote`（会被拒绝），judge 也不为它写 `plan`。格式和登记时的核对见 `docs/interfaces/evaluation.md` 的「里程碑评测」。
   3. 然后运行 `rsi next <ws>`，它打印 `stop: <理由>`，运行结束。
6. 交付：每次停止（`rsi next` 打印 `stop:`，或用户要求后 `rsi stop` 打印 `stop:`）都先运行 `rsi report <ws> --out <ws>/REPORT.md`。报告由代码生成，列出任务、停止理由、各轮判定、各版本、当前最佳版本的模型包和评测报告路径、里程碑评测、按模块分组的实验记录、尝试次数和预算。
   1. 在 `<ws>/REPORT.md` 末尾加一节「## 解读」，写代码给不出的部分：这次运行学到了什么（哪些改动有效、哪些无效）、问题主要归因在哪、如果继续该先试什么。只转述已有的依据，即 judge 的评测报告和计划、各模块 `record` 的结论和经验，每条注明出处，不自己做新的分析。加完后运行 `rsi dashboard <ws> --no-open`，页面的「运行报告」页才显示解读。
   2. 有当前最佳版本时，把报告连同它的模型包、预测接口和报告里列出的文件（交付目录、评测报告、里程碑评测）交给用户，说明停止理由和主指标。模型包一律用报告里列出的登记后路径（`<ws>/artifacts/<delivery id>/` 下的路径，或 `external` 条目的位置和 sha256），不用模块输出目录里的路径。
   3. 一个版本都没有升级时，报告开头写着「没有产生可发布的版本」并列出各轮的结论。把它作为失败报告交给用户，如实说明卡在哪里（例如每轮的候选都没过训练门）和解读里的建议，不把任何候选说成可用的版本。
7. `rsi next` 被拒时，照错误信息补上缺的步骤。

停止后只能补登记 `record`（`rsi register <ws> record <dir>`）和做 `delete` 审批后的清理，不能再批准使用 test。补登记了 `record` 后重新生成报告，再补上解读，最后运行 `rsi dashboard <ws> --no-open`。

## 审批点

以下事项必须停下问用户。用户明确同意后，在要用到这项批准的那一步之前用 `rsi approve` 记下：`--by` 写批准人，`--quote` 写用户的原话，不转述，不替用户批准。

| 审批项 | 什么时候问 | 记录 |
|---|---|---|
| 确认任务定义 | 任务接入后、`rsi init` 之前 | 工作区还没建立，用户确认后直接 `rsi init <ws> --task <dir>` |
| `task_change` | 用户要改已固定的任务定义 | `rsi approve <ws> task_change --by <name> --quote "<用户原话>"`，再登记新版本（见异常处理） |
| `plan` | judge 的计划登记后、`rsi next` 之前；`rsi check-stop` 打印 stop 时不问 | `rsi approve <ws> plan --ref <plan_id> --by <name> --quote "<用户原话>"` |
| `over_budget` | 派活前估计用量会超过剩余预算 | `rsi approve <ws> over_budget --by <name> --quote "<用户原话>"` |
| `use_testset` | 里程碑发布要用封存 test（见闭环）；deliver 跑 test 预测、judge 用 test 打分各要一次 | `rsi approve <ws> use_testset --by <name> --quote "<用户原话>"` |
| `refresh_sets` | 要换新的 valset、evalset 或 testset，每个集合各要一次；评测集使用次数到上限后、派 ④ 之前（见主流程） | `rsi approve <ws> refresh_sets --by <name> --quote "<用户原话>"` |
| `delete` | 删除模型文件、中间结果等任何产物之前；当前最佳版本的模型和 `<ws>/artifacts/` 不删 | `rsi approve <ws> delete --by <name> --quote "<用户原话>"` |

哪些审批必需、一次批准能用几次、在哪一轮有效，都由代码判定；缺审批时，相关的 `rsi register` 或 `rsi next` 会拒绝并写明缺哪一项。删除文件不经过 `rsi`，所以任务要求 `delete` 审批时，顶层先问用户、记下审批再删。

**待用户回答**：每次停下来问用户之前（审批、模块报告里要用户定的事、质量门没过时问怎么办等），先把问题一行写进 `<ws>/work/notes.md` 的 `## 待用户回答` 小节，再运行 `rsi dashboard <ws> --no-open`，页面最上面就显示"⏸ 等你在 Claude 里回答"和这一行；用户回答后删掉这一行，再运行一次 `rsi dashboard <ws> --no-open`（页面只在 `rsi` 命令之后重写，不会自己发现笔记变了）。确认任务定义那一步还没有工作区，不在此列。批准计划、里程碑发布、超预算、刷新评测集这几种页面会自己从状态推出来，写不写笔记都可以。模块不读也不写这一节，仍然只运行 `rsi validate` 和 `rsi score`。

## 长任务

训练、搜索、大批量推理或打标这类长任务，不让 subagent 守着：

1. subagent 在后台启动任务（例如 `nohup`、`tmux`）后立刻交回完成标记路径、日志路径和预计耗时。任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`。
2. 顶层把标记路径和日志路径记进 `<ws>/work/notes.md`，换会话后据此接着等。
3. 主会话用后台 shell 等标记出现，例如以后台方式运行 `until [ -e <标记> ]; do sleep 300; done`，它结束时会通知主会话。标记在远端主机上时（任务在别的机器上跑），改用 `until ssh <host> test -e <标记>; do sleep 300; done`，`<host>` 和标记路径一起记进笔记。期间不反复看日志，也不派 subagent 去查。
4. 等待时可以处理不依赖它的步骤，例如 ② 还在跑时先登记 ③ 交回的产物。
5. 标记出现后派同一个模块收尾（继续原来的 subagent，或按 `templates/brief.md` 重新派并附上标记和日志路径），由它读结果、写 manifest、交回产物。标记是 `failed` 时按异常处理。
6. 超过预计耗时两倍还没有标记，看一次日志末尾和进程是否还在，再决定继续等、重启还是问用户。

## 异常处理

- 模块失败（报错、标记是 `failed`、交不出产物）：先读报告和日志末尾。输入给错了（路径、轮次、产物 id）就改说明重派；模块内部的问题把错误原文交回同一模块重试。同一步连续失败两次就停下告诉用户，不自己动手做模块的活。
- 校验不过：`rsi validate` 或 `rsi register` 的错误原文交回产出它的模块。已登记的产物要出新版本时写到新目录。
- 预算超出：派活前让模块估计用量，与剩余预算比较（算力看 `rsi status <ws>`，标注条数和调用费用按模块报告累计）。会超就问用户，同意则记 `over_budget` 审批，不同意就缩小这一步或结束运行。算力用完时 `rsi next` 会停止运行。
- 用户中途改需求：先判断是否改到任务定义，即 `task.json` 的任何字段。
  - 不改任务定义，例如只是希望下一轮侧重某类错误：写进派活说明，或交给 judge 写进计划。
  - 改任务定义：在新目录写新的 `task.json` 和 `TASK.md`，`version` 加 1，运行 `rsi validate task <dir>`，通过后给用户确认，记 `task_change` 审批，再 `rsi register <ws> task <dir>`。之后的产物按新版本产出。改了标签或输入时，testset、evalset、valset、trainset 四个集合都可能不再适用，逐个和用户确定是否按新定义重做；换 testset、evalset、valset 各要一次 `refresh_sets` 审批。
  - 换成另一个任务（`task_id` 不同）：用新的工作区，从任务接入开始。
- 评测集用到上限（派 ④ 之前在 eval uses 里看到）：问用户是否刷新评测集。批准后记 `refresh_sets` 审批，派 `rsi-data` 产出新的 evalset 并登记。刷新时 data 拿不到测试集和旧评测集；新评测集和它们重叠时登记被拒，把错误里的重叠数交给 data 重新生成，这个往返是预期的。用户不同意刷新时，之后的 evaluation 都登记不了，也就不会再有升级，把这一点告诉用户：本轮用 `rsi close-round <ws> --reason "<理由>"` 关闭，或者用户同意结束时用 `rsi stop <ws> --by <name> --quote "<用户原话>"` 结束运行（这时通常更合适），再按闭环第 6 步交付。
  - 刷新之后，当前最佳版本的候选在当前评测集上还没有已登记的 delivery 时，每轮派 ④ 都让 `rsi-deliver` 把它也在当前评测集的 `inputs.jsonl` 上打包跑一遍（同样给它的 `valset_id` 所指验证集的 `inputs.jsonl`，用来复现），和送评候选的 delivery 分开交回、分开登记。登记后，之后各轮直接复用它（`rsi status <ws>` 的 `best delivery on current evalset` 行给出它的 id），不再重复打包。
  - 派 ⑤ 时两份 delivery 都给 judge：judge 用最佳版本那份算最佳版本在当前评测集上的指标和差值，把它的 delivery id 写进 evaluation 的 `parents`；`best_before` 仍写最佳版本和它原来的 evaluation。新版本升级后，它的 delivery 本来就在当前评测集上，不再需要额外打包。
  - 已知限制：停止条件的 `target` 仍按最佳版本原来那份 evaluation 的主指标判断，即旧评测集上的成绩，刷新后可能偏高或偏低。刷新时把这一点告诉用户。
- 用户要结束运行：先确认是结束整个运行，而不是暂停（暂停时什么都不用记，之后按「入口」继续）。本轮已有登记但未判定的 evaluation 时，先 `rsi promote`，让这一轮的结果算数；用户想用封存 test 验证当前最佳版本时，在停止之前按闭环第 5 步的 1、2 做（停止后不能再批准使用 test），第 3 步的 `rsi next` 换成下面的 `rsi stop`。然后运行 `rsi stop <ws> --by <name> --quote "<用户原话>"`，它打印 `stop: user requested: <原话>`，再按闭环第 6 步交付。
- 会话中断：重新加载本 skill，按「入口」接着走。

## 常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 把标签给了 deliver（给了 evalset 目录或 `labels.jsonl`） | 裁判和选手不再分开，本轮评测不可信 | 只给 `inputs.jsonl` 的文件路径，标签只给 judge |
| 把工作区根目录给模块，或让模块自己 `rsi register` | 模块能读到不该看的产物和标签，登记绕开了顶层的校验和抽查 | 说明里只给输入产物路径和输出目录，登记只由顶层做 |
| 跳过 validate 直接 register | 登记表只追加，登记错了撤不回；问题到登记时才暴露，也漏了抽查 | 先 `rsi validate`，再抽查，再 `rsi register` |
| 自己判断升级（看数字觉得好就宣布升级，或改 judge 的建议） | 绕开升级规则，版本登记不可信 | 只用 `rsi promote` 打印的结论 |
| 没批准就 next，或替用户批准、`--quote` 写转述 | `rsi next` 拒绝；替用户写的批准等于伪造审批 | 先把计划讲给用户，用户同意后 `rsi approve`，再 `rsi next` |
| 顶层自己做模块的活（改数据、调参、补写 evaluation 或 plan） | 模块负责人无法改进自己的模块，产物无人负责 | 交回对应模块 |
| 手改 `STATE.json`、`registry.jsonl` 或 `artifacts/` 下的文件 | 登记和规则检查失效 | 状态变化一律通过 `rsi` 命令 |
| 在评测集上比较几个候选再挑一个送评 | 评测集变成调参集，使用次数也被耗掉 | 按验证集选送评的候选 |
| 让模块自己写算指标、对照组或区间的代码，或替模块改数字迎合登记 | 各模块算法不一致，数字不能比较；登记被拒 | 简报里写明 `rsi score` 的完整命令，模块原样写入它打印的数 |
| 一次派出全部模块 | 下游拿不到输入，白花预算 | 按主流程顺序，只有 ② 和 ③、多个 ④ 可以并行 |
| ② 和 ③ 并行时改了训练命令或默认配置 | 「默认配置」在候选之间不再是同一个，search 登记被拒（试验候选的 `code_version` 不同） | 本轮之内训练代码和默认配置不动，改进放到两轮之间 |
| 把模块输出目录里的模型包路径交给用户 | 输出目录会被清理或改写，交出去的不是登记过的那份 | 用 `rsi report` 列出的登记后路径 |
| 让 subagent 守着长任务，或主会话反复看日志 | 浪费额度 | subagent 启动后交回，主会话后台等完成标记 |
| 改完已登记过的源目录再登记 | `rsi register` 拒绝 | 已登记产物要出新版本时写到新目录；被拒、没登记过的产物可在原目录改好再登记 |
| 本轮尝试次数用完还接着派 ②、③，或候选都没过训练门时一直等 | 登记被拒，运行卡在这一轮 | 看 `rsi status` 的 attempts 行，用完就 `rsi close-round`，让 judge 按本轮记录写计划 |
| 用户说结束，只是不再派活 | 工作区没有停止记录，下次会话会接着跑；用户拿不到报告 | `rsi stop` 记下用户原话，再交付报告 |
| 停止时只口头总结，不生成报告，或没升级版本就说没有结果 | 交付不全，失败的运行留不下依据 | 每次停止都 `rsi report <ws> --out <ws>/REPORT.md`，补上解读后交给用户；没有版本时交失败报告 |
