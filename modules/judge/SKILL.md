---
name: rsi-judge
description: 由 rsi-orchestrator 派发评测与决策模块时使用：用评测集标签给交付打分、给出升级建议、失败归因和下一轮计划。Use when the orchestrator dispatches the RSI judge module; not for general evaluation requests.
---

# rsi-judge：评测与决策模块

保管评测集的标签，打分、比较，给出升级建议、失败分析和下一轮计划。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

简报由代码生成，放在输出目录里：`brief.md`（人读）和 `brief.json`（同样的内容，结构化：`module`、`purpose`、`round`、`task_json`、`ids`、`inputs`、`outdir`、`budget_left`、`goal` 等；字段说明见框架仓库根目录下的 `skills/rsi-orchestrator/templates/brief.md`）。输入以简报的 `inputs` 为准，只读那里列出的路径。被退回时，同一个会话会收到 `feedback-<k>.md`（`<k>` 是第几次启动），写明为什么退回和要改什么：按它改好，再重新写 `handback.json`。同一个会话还会收到两种不算退回的 `feedback-<k>.md`：开头是「# 继续：交回的第二部分」的（两段式交回的后半段，例如 train 先交回试验候选、judge 先交回评测，登记后在这里拿到登记 id，再写 search 或计划），和开头是「# 用户回答了你的问题」的（见「输出」）。

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`（指标、对照组、升级规则）。
- 送评的 `candidate`，本轮的 `delivery`，本轮有 `search` 时也给。
- `evalset`，含 labels.jsonl；只有本模块拿标签。
- 送评候选所用 `trainset` 的 rows.jsonl，只用来算对照组，不做别的分析。
- 当前最佳版本的版本名，以及它的 `evaluation` 和 `delivery`；简报里写的 `best_before` 的值照用。评测集刷新过时，另拿最佳版本在当前评测集上的 `delivery`：用它算最佳版本一侧的指标和差值，并把它的 id 写进 evaluation 的 parents。
- `feedback`，全部 `record`。
- 开启本轮的 `plan`（上一轮写的，只在评测时给）：对照本轮的评测，判断这个计划是否奏效，写进评测报告。
- 里程碑发布获批时，另拿 `testset`（含标签）。

## 输出

- `evaluation`：各指标、与当前最佳版本和对照组的比较及区间、升级或保留的建议、失败项及归因、`uses_testset`；报告文件按分组和切片给出结果。
- `plan`：下一轮从哪个模块进入、每个问题的证据和改动、预期提升。送评候选用的是本轮数据模块新出的训练集时，在证据里写明它训出的候选相对当前最佳版本的结论（升级或保留、主指标差值）：数据模块不训练、自己不验证新训练集，也拿不到 `evaluation`，下一轮从 `plan` 得知新训练集的效果。关闭的轮次：写明新训练集上的候选没过训练门。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [evaluation.md](../../docs/interfaces/evaluation.md)。 `record` 见 [record.md](../../docs/interfaces/record.md)。

`plan` 引用 `evaluation` 的登记 id：先以 `ok` 交回 `evaluation`；引擎登记并判定后让同一个会话接着做（`feedback-<k>.md` 开头是「# 继续：交回的第二部分」，不算退回），`evaluation` 的登记 id 和升级判定在里面，再写 `plan` 交回。

简报说本轮已关闭（没有候选通过训练门，本轮没有 `evaluation`）时，只写 `plan` 和 `record`：plan 的 evaluation_id 写 null，问题和证据取自本轮候选的训练门结果和实验记录。

里程碑评测（简报说是里程碑发布时）：给当前最佳版本的 delivery 打分，`uses_testset` 写 true，`candidate_id` 是当前最佳版本的候选，`best_before` 写当前最佳版本；指标、差值仍按评测集算，test 上的成绩写进 `metrics_test`（含主指标）；建议写 keep，它不是升级判定，不占评测集使用次数。不写 `plan`。格式见 [evaluation.md](../../docs/interfaces/evaluation.md) 的「里程碑评测」。

交回：全部产物写好、校验通过后，在输出目录写 `handback.json`：`status`（`ok`、`failed` 或 `long_job`）、`artifacts`（输出目录下的子目录名，每个产物一个）、`marker`（长任务完成标记的绝对路径，`long_job` 时必填，其他写 null）、`notes`（门数字、花费、遇到的问题；`failed` 时写原因）、`questions`（需要用户决定的事，没有写 `[]`）。引擎读到它才算交回，缺了或格式不对按交回失败处理，重新写好即可。读取很严格：不能有上面没列出的字段，`artifacts` 只能是子目录的名字（不含路径分隔符，不重复），`questions` 的每一项必须是非空字符串，`marker` 必须是绝对路径或 null，`failed` 时 `notes` 不能为空。

交回带 `questions` 时，这次交回的产物一个都不登记，引擎先问用户；用户回答后，回答送回同一个会话（`feedback-<k>.md` 开头是「# 用户回答了你的问题」，不算退回）。按回答修改或确认产物，再重新写 `handback.json`，列出要登记的全部子目录（上次列过的也要再列）。长任务（`long_job`）交回里的 `questions` 等完成标记出现后才问；它没列产物时，回答写在开头是「# 长任务已结束」的反馈里，和长任务的结果一起送回。

## 门

- 指标、对照组、差值和区间一律用 `rsi score` 算，打印的数原样写进 evaluation，不自己实现（定义见 [task.md](../../docs/interfaces/task.md) 的「指标的定义」）。登记时代码按同样的定义重算，差超过 1e-9 就被拒：
  - `metrics.candidate`、`metrics.baselines`、`deltas` 的主指标：`PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <evalset>/labels.jsonl --predictions <delivery>/predictions_eval.jsonl --train <trainset>/rows.jsonl --clusters <evalset>/inputs.jsonl --compare <best-delivery>/predictions_eval.jsonl`，输出的 `metrics`、`baselines`、`delta` 分别写进这三处；还没有最佳版本时去掉 `--compare`，`metrics.best` 写 null，`deltas` 留空。
  - `metrics.best`：同一条命令把 `--predictions` 换成最佳版本的 `predictions_eval.jsonl`，取输出的 `metrics`。评测集刷新过时，最佳版本一侧用它在当前评测集上的 delivery（简报给出），并把这份 delivery 的 id 写进 `parents`，否则登记被拒。
  - 里程碑评测的 `metrics_test`：`PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <testset>/labels.jsonl --predictions <delivery>/predictions_test.jsonl`。
  - 输出的 `not_computed` 列出 `rsi` 不认识的指标和基线：这些由本模块自己算，在报告里写明算法，登记时不核对。
  - 切片、分组结果等报告里的其他数字由本模块自定，不核对。
- 建议必须与升级规则一致：按任务定义的 `promotion_rule` 和 [task.md](../../docs/interfaces/task.md) 描述的判定方法，用 `rsi score` 算出、写进 evaluation 的差值、区间和护栏指标（登记时会重算核对）得出 `promote` 或 `keep`（提升必须大于 `min_delta`，恰好等于它的平局，容差 1e-9 内，不算提升，写 `keep`），`reason` 引用规则和对应数字。引擎登记并判定升级时会用代码核对，不一致就被拒。
- 归因只能取 [evaluation.md](../../docs/interfaces/evaluation.md) 规定的值，每个失败项有证据和唯一的归因，模块与归因对应。
- `plan` 的 `entry_module` 必须是某个 issue 的模块，每个 issue 都有证据、改动和预期提升。
- 评测集使用次数有限，每份登记的 evaluation（里程碑评测除外）用一次；用封存 test 只在获批后，且 `uses_testset` 写 true。
- 评测集和封存 test 的标签不得写进给别的模块看的 `plan`、`record` 和报告里的逐条样本。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 `candidate`、`delivery`、`evalset`（含标签）、送评候选所用 `trainset` 的 rows.jsonl（只用来算对照组）、历史 `evaluation` 和 `record` 从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`。不需要工作区，也不读登记表。
2. 输出写到沙盒下自己建的输出目录，每个产物一个子目录，manifest 的 `id` 写 null，`round` 自拟。
3. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验，通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
4. `plan` 引用 `evaluation` 的 id 时，沙盒里手填一个即可，登记时由引擎核对。

## 步骤和策略子 skill

由模块负责人填写：切片分析、失败归因、计划生成、预期提升的校准。打分、对照组和置信区间是固定代码（`rsi score`，见「门」），不在这里另定。把可重复的步骤写成脚本，skill 只写需要判断的部分。

## 模块指标

由模块负责人填写：计划的命中率：计划里预期的提升有多少真的实现了，及其测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），写明改了什么、前后数值、结论，`module` 写 `judge`。

## 长任务

由模块负责人填写：大量重采样或逐条分析（后台运行，写完成标记）。预计超过简报里规定的时长时，在后台启动，写完成标记（第一行 ok 或 failed），立刻交回标记路径、日志路径和预计耗时，不轮询、不等待。

长任务的交回：预计超过简报规定的时长时，在后台启动，立刻写 `handback.json`，`status` 写 `long_job`，`marker` 写完成标记的绝对路径，然后结束会话，不轮询、不等待。要交回 `long_job` 时完成标记已经在了（长任务在你交回之前就结束了，例如上一个会话启动的）：不要以 `long_job` 交回。引擎只认这次启动之后写的标记，更早的标记永远不算出现，运行会一直等；直接读结果，接着做完，再交回。任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`。标记是 `ok` 时，引擎让同一个会话继续（`feedback-<k>.md` 开头是「# 长任务已结束」），由它读结果、写好产物、再以 `ok` 交回；标记不是 `ok` 时这次派出记为失败。

交回 `long_job` 之前，在完成标记旁写 `<marker>.pid`：第一行是长任务进程自己的进程号，第二行是它的启动时间（Unix 秒，`date +%s`）。进程号要由长任务自己写，例如 shell 里 `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <命令>' > <日志> 2>&1 &`（`exec` 让命令沿用这个进程号），Python 里任务一开始写 `os.getpid()` 和 `int(time.time())`；不要用 shell 给出的后台进程号代替，它可能是包装进程而不是长任务本身。最好让长任务在自己的进程组里运行（例如 Python 的 `subprocess.Popen(..., start_new_session=True)`）。`<marker>.pid` 写好后再写 `handback.json`。运行被停止时，引擎按这个文件确认进程是这个长任务后终止它（整个进程组）；没有这个文件就终止不了，只能列为可能仍在运行、请用户手动结束。

每个长任务用新的完成标记路径（例如 `<名>-<k>.done`，`<k>` 每次加一），启动前 `rm -f <marker> <marker>.pid`：引擎只认修改时间不早于这次启动的标记，同一路径上以前的标记不算这个长任务结束。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。已知的一条：建议与升级规则不一致，或归因不取规定的值。
