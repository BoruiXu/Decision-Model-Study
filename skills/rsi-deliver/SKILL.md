---
name: rsi-deliver
description: 由 rsi-orchestrator 派发交付模块时使用：把候选模型打包成预测接口，只拿输入跑预测，测延迟和成本。Use when the orchestrator dispatches the RSI deliver module; not for general packaging requests.
---

# rsi-deliver：交付模块

打包模型和预测接口，在评测集和验证集上只拿输入跑预测，测延迟和成本，收集执行反馈。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`。
- 送评的 `candidate`，含可直接运行的预测入口（manifest 的 model.predict）。评测集刷新过时，简报可能另给当前最佳版本的 `candidate`：同样打包跑预测，单独出一份交付产物。
- `inputs.jsonl`：`evalset` 的输入文件；送评候选所用 `valset` 的输入文件（不带标签），用来复现候选的验证集预测。都只给文件路径。里程碑发布获批时，再加 `testset` 的 `inputs.jsonl`。

## 输出

- `delivery`：交付包（放在交付目录里，或带 sha256 的 external 条目）、调用入口、运行环境、运行指标，评测集和验证集上的预测；获批用 test 时另出 test 预测。
- 可选 `feedback`：试运行收集的执行反馈。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [delivery.md](../../docs/interfaces/delivery.md)。 `record` 见 [record.md](../../docs/interfaces/record.md)。

预测接口的输入输出格式（输入一个样本的 input，输出 probs、decision、uncertainty，每次调用写一条执行反馈）固定在 [delivery.md](../../docs/interfaces/delivery.md) 的「预测接口」一节，接口形式（命令行、函数、HTTP）由负责人定，写在 `package.entry`。

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

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。已知的一条：让评测集的标签进入交付模块的任何环节，例如拿到整个评测集目录。
