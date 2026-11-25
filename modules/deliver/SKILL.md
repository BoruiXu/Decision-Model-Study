---
name: rsi-deliver
description: 由 rsi-orchestrator 派发交付模块时使用：把候选模型打包成预测接口，只拿输入跑预测，测延迟和成本。Use when the orchestrator dispatches the RSI deliver module; not for general packaging requests.
---

# rsi-deliver：交付模块

打包模型和预测接口，在评测集和验证集上只拿输入跑预测，测延迟和成本，收集执行反馈。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

简报由代码生成，放在输出目录里：`brief.md`（人读）和 `brief.json`（同样的内容，结构化：`module`、`purpose`、`round`、`task_json`、`ids`、`inputs`、`outdir`、`budget_left`、`goal` 等；字段说明见框架仓库根目录下的 `skills/rsi-orchestrator/templates/brief.md`）。输入以简报的 `inputs` 为准，只读那里列出的路径。被退回时，同一个会话会收到 `feedback-<k>.md`（`<k>` 是第几次启动），写明为什么退回和要改什么：按它改好，再重新写 `handback.json`。同一个会话还会收到两种不算退回的 `feedback-<k>.md`：开头是「# 继续：交回的第二部分」的（两段式交回的后半段，例如 train 先交回试验候选、judge 先交回评测，登记后在这里拿到登记 id，再写 search 或计划），和开头是「# 用户回答了你的问题」的（见「输出」）。

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`。
- 送评的 `candidate`，含可直接运行的预测入口（manifest 的 model.predict）。评测集刷新过时，简报可能另给当前最佳版本的 `candidate`：同样打包跑预测，单独出一份交付产物。
- `inputs.jsonl`：`evalset` 的输入文件；送评候选所用 `valset` 的输入文件（不带标签），用来复现候选的验证集预测。都只给文件路径。里程碑发布获批时，再加 `testset` 的 `inputs.jsonl`。
- 本模块以前各轮的 `record`（只给自己的）：以前试过什么、结果如何。

## 输出

- `delivery`：交付包（放在交付目录里，或带 sha256 的 external 条目）、调用入口、运行环境、运行指标，评测集和验证集上的预测；获批用 test 时另出 test 预测。
- 可选 `feedback`：试运行收集的执行反馈。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [delivery.md](../../docs/interfaces/delivery.md)。 `record` 见 [record.md](../../docs/interfaces/record.md)。

预测接口的输入输出格式（输入一个样本的 state，输出 probs、decision、uncertainty，每次调用写一条执行反馈）固定在 [delivery.md](../../docs/interfaces/delivery.md) 的「预测接口」一节，接口形式（命令行、函数、HTTP）由负责人定，写在 `package.entry`。

交回：全部产物写好、校验通过后，在输出目录写 `handback.json`：`status`（`ok`、`failed` 或 `long_job`）、`artifacts`（输出目录下的子目录名，每个产物一个）、`marker`（长任务完成标记的绝对路径，`long_job` 时必填，其他写 null）、`notes`（门数字、花费、遇到的问题；`failed` 时写原因）、`questions`（需要用户决定的事，没有写 `[]`）。引擎读到它才算交回，缺了或格式不对按交回失败处理，重新写好即可。读取很严格：不能有上面没列出的字段，`artifacts` 只能是子目录的名字（不含路径分隔符，不重复），`questions` 的每一项必须是非空字符串，`marker` 必须是绝对路径或 null，`failed` 时 `notes` 不能为空。

交回带 `questions` 时，这次交回的产物一个都不登记，引擎先问用户；用户回答后，回答送回同一个会话（`feedback-<k>.md` 开头是「# 用户回答了你的问题」，不算退回）。按回答修改或确认产物，再重新写 `handback.json`，列出要登记的全部子目录（上次列过的也要再列）。长任务（`long_job`）交回里的 `questions` 等完成标记出现后才问；它没列产物时，回答写在开头是「# 长任务已结束」的反馈里，和长任务的结果一起送回。

## 门

- 门：交付包能加载；交付包复现候选。用交付包的预测接口在验证集的 inputs.jsonl 上跑出 `predictions_val`（`valset_id` 写候选的 `valset_id`），与候选的 `predictions_val` 逐行一致：`decision` 相同，每个概率相差不超过任务定义的 `delivery.reproduction_tolerance`（不写时 1e-6）；概率在容差内而决策翻了的靠近边界的行，最多放过 floor(`delivery.max_decision_mismatch_rate` × 验证集行数) 行（不写时 0 行），在报告里写明是哪几行。登记时代码逐行核对，不一致就拒绝，无论 `consistent_with_offline` 写什么；这个标志只在复现成立时写 true。
- 交付包调用候选的预测入口（`<候选目录>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>`，见 [models.md](../../docs/interfaces/models.md)），或加载同一份模型得出同样的结果；不按文字说明另写一套打分。
- 交付包自包含：`package.location` 写交付目录里的相对路径（登记时随目录进入工作区），或 `external:<名字>`，在 manifest 的 `external` 里给出位置和 sha256（本机条目必须是单个文件，目录先打成 tar 包）；不写绝对路径，也不指向自己的工作目录；交付目录里的符号链接登记时按指向的内容复制（例如交付包里的 venv），指向不存在文件的链接会被拒绝。
- 绝不读取、也不索要 labels.jsonl 或任何标签。拿不到标签不是问题，也不要要求标签；误拿到了就停下报告。
- 没有获批不碰封存 test 的输入；`delivery` 里有 test 预测时必须写 `testset_id`。
- 延迟和成本照实测，写进 `runtime`：`latency_ms_p50`、`latency_ms_p95` 是预测接口每次调用（一个样本）的墙钟毫秒数在评测集输入上的分位数，不含加载模型的时间；`cost_per_1k` 是 1000 次预测的成本，单位写在 `cost_unit`（例如 usd、gpu_seconds），不知道时写 null。执行反馈行的 `id` 由本模块自定，一份 trace 里不重复即可。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 `candidate`、评测集的 `inputs.jsonl` 和候选所用验证集的 `inputs.jsonl` 从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`。不需要工作区，也不读登记表。
2. 输出写到沙盒下自己建的输出目录，每个产物一个子目录，manifest 的 `id` 写 null，`round` 自拟。
3. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验，通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
4. 沙盒里同样只放输入文件，不放标签。

## 步骤和策略子 skill

由模块负责人填写：打包方式、接口实现、延迟和成本测量、执行反馈收集。把可重复的步骤写成脚本，skill 只写需要判断的部分。

## 模块指标

由模块负责人填写：延迟、成本、打包是否一次成功，及其测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），写明改了什么、前后数值、结论，`module` 写 `deliver`。

## 长任务

由模块负责人填写：大批量推理（后台运行，写完成标记）。预计超过简报里规定的时长时，在后台启动，写完成标记（第一行 ok 或 failed），立刻交回标记路径、日志路径和预计耗时，不轮询、不等待。

长任务的交回：预计超过简报规定的时长时，在后台启动，立刻写 `handback.json`，`status` 写 `long_job`，`marker` 写完成标记的绝对路径，然后结束会话，不轮询、不等待。要交回 `long_job` 时完成标记已经在了（长任务在你交回之前就结束了，例如上一个会话启动的）：不要以 `long_job` 交回。引擎只认这次启动之后写的标记，更早的标记永远不算出现，运行会一直等；直接读结果，接着做完，再交回。任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`。标记是 `ok` 时，引擎让同一个会话继续（`feedback-<k>.md` 开头是「# 长任务已结束」），由它读结果、写好产物、再以 `ok` 交回；标记不是 `ok` 时这次派出记为失败。

交回 `long_job` 之前，在完成标记旁写 `<marker>.pid`：第一行是长任务进程自己的进程号，第二行是它的启动时间（Unix 秒，`date +%s`）。进程号要由长任务自己写，例如 shell 里 `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <命令>' > <日志> 2>&1 &`（`exec` 让命令沿用这个进程号），Python 里任务一开始写 `os.getpid()` 和 `int(time.time())`；不要用 shell 给出的后台进程号代替，它可能是包装进程而不是长任务本身。最好让长任务在自己的进程组里运行（例如 Python 的 `subprocess.Popen(..., start_new_session=True)`）。`<marker>.pid` 写好后再写 `handback.json`。运行被停止时，引擎按这个文件确认进程是这个长任务后终止它（整个进程组）；没有这个文件就终止不了，只能列为可能仍在运行、请用户手动结束。

每个长任务用新的完成标记路径（例如 `<名>-<k>.done`，`<k>` 每次加一），启动前 `rm -f <marker> <marker>.pid`：引擎只认修改时间不早于这次启动的标记，同一路径上以前的标记不算这个长任务结束。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。已知的一条：让评测集的标签进入交付模块的任何环节，例如拿到整个评测集目录。
