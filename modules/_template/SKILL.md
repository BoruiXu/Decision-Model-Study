---
name: rsi-<模块名>
description: 由 rsi-orchestrator 派发<模块名>模块时使用：<一句话说明本模块做什么>。Use when the orchestrator dispatches the RSI <模块名> module; not for general <领域> requests.
---

# rsi-<模块名>：<模块中文名>

这是模块 skill 的模板，复制到 `modules/<模块名>/SKILL.md`（模块自己的文件夹，代码、参考资料和测试也放在这里）后替换尖括号里的内容。前四节由框架写好接口，其余六节由模块负责人填写。模块内部怎么做由负责人定，接口（输入、输出、门、单独运行）以代码生成的简报（`rsi/briefs.py`）和 `docs/interfaces/modules.md` 为准。

## 输入

简报由代码生成，放在输出目录里：`brief.md`（人读）和 `brief.json`（同样的内容，结构化：`module`、`purpose`、`round`、`task_json`、`ids`、`inputs`、`outdir`、`budget_left`、`goal` 等；字段说明见框架仓库根目录下的 `skills/rsi-orchestrator/templates/brief.md`）。输入以简报的 `inputs` 为准，只读那里列出的路径。被退回时，同一个会话会收到 `feedback-<k>.md`（`<k>` 是第几次启动），写明为什么退回和要改什么：按它改好，再重新写 `handback.json`。同一个会话还会收到两种不算退回的 `feedback-<k>.md`：开头是「# 继续：交回的第二部分」的（两段式交回的后半段，例如 train 先交回试验候选、judge 先交回评测，登记后在这里拿到登记 id，再写 search 或计划），和开头是「# 用户回答了你的问题」的（见「输出」）。

列出产物类别（用反引号）和每个的用途；只读这里列出的产物，不读工作区和登记表。简报另给本模块以前各轮的 `record`（只给自己的）。

## 输出

列出产物类别和每个的字段文档链接；每个产物写成输出目录下的一个子目录，manifest 的 `id` 写 null，`round` 按简报写。每个改进实验另交一份 `record`。

可选：随时在输出目录整个重写进度文件 `progress.json`（简报的 `progress_file`），进度页面和 `rsi run` 的状态行原样显示它，格式见 [progress.md](../../docs/interfaces/progress.md)；不写也可以。

交回：全部产物写好、校验通过后，在输出目录写 `handback.json`：`status`（`ok`、`failed` 或 `long_job`）、`artifacts`（输出目录下的子目录名，每个产物一个）、`marker`（长任务完成标记的绝对路径，`long_job` 时必填，其他写 null）、`notes`（门数字、花费、遇到的问题；`failed` 时写原因）、`questions`（需要用户决定的事，没有写 `[]`）。引擎读到它才算交回，缺了或格式不对按交回失败处理，重新写好即可。读取很严格：不能有上面没列出的字段，`artifacts` 只能是子目录的名字（不含路径分隔符，不重复），`questions` 的每一项必须是非空字符串，`marker` 必须是绝对路径或 null，`failed` 时 `notes` 不能为空。

交回带 `questions` 时，这次交回的产物一个都不登记，引擎先问用户；用户回答后，回答送回同一个会话（`feedback-<k>.md` 开头是「# 用户回答了你的问题」，不算退回）。按回答修改或确认产物，再重新写 `handback.json`，列出要登记的全部子目录（上次列过的也要再列）。长任务（`long_job`）交回里的 `questions` 等完成标记出现后才问；它没列产物时，回答写在开头是「# 长任务已结束」的反馈里，和长任务的结果一起送回。

## 门

本模块交回前必须满足的检查，写明数值、阈值和谁来核对。每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验；报告里的指标、对照组和差值用 `PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <标签行文件> --predictions <预测文件>` 算。模块只能运行这两个 `rsi` 命令。

## 单独运行

负责人在自己的沙盒里，用已登记产物的副本（路径手工指定）运行本模块，输出用 validate 检查，不需要顶层。

## 步骤和策略子 skill

由模块负责人填写：本模块的步骤和策略子 skill 清单，哪些步骤是脚本、哪些需要判断。

## 模块指标

由模块负责人填写：本模块自己的指标及测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)）。

## 长任务

由模块负责人填写：哪些步骤是长任务；在后台启动，写完成标记（第一行 ok 或 failed），交回标记路径后不轮询。

长任务的交回：预计超过简报规定的时长时，在后台启动，立刻写 `handback.json`，`status` 写 `long_job`，`marker` 写完成标记的绝对路径，然后结束会话，不轮询、不等待。要交回 `long_job` 时完成标记已经在了（长任务在你交回之前就结束了，例如上一个会话启动的）：不要以 `long_job` 交回。引擎只认这次启动之后写的标记，更早的标记永远不算出现，运行会一直等；直接读结果，接着做完，再交回。任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`。标记是 `ok` 时，引擎让同一个会话继续（`feedback-<k>.md` 开头是「# 长任务已结束」），由它读结果、写好产物、再以 `ok` 交回；标记不是 `ok` 时这次派出记为失败。

交回 `long_job` 之前，在完成标记旁写 `<marker>.pid`：第一行是长任务进程自己的进程号，第二行是它的启动时间（Unix 秒，`date +%s`）。进程号要由长任务自己写，例如 shell 里 `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <命令>' > <日志> 2>&1 &`（`exec` 让命令沿用这个进程号），Python 里任务一开始写 `os.getpid()` 和 `int(time.time())`；不要用 shell 给出的后台进程号代替，它可能是包装进程而不是长任务本身。最好让长任务在自己的进程组里运行（例如 Python 的 `subprocess.Popen(..., start_new_session=True)`）。`<marker>.pid` 写好后再写 `handback.json`。运行被停止时，引擎按这个文件确认进程是这个长任务后终止它（整个进程组）；没有这个文件就终止不了，只能列为可能仍在运行、请用户手动结束。

每个长任务用新的完成标记路径（例如 `<名>-<k>.done`，`<k>` 每次加一），启动前 `rm -f <marker> <marker>.pid`：引擎只认修改时间不早于这次启动的标记，同一路径上以前的标记不算这个长任务结束。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。
