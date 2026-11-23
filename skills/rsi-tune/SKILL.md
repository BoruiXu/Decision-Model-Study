---
name: rsi-tune
description: 由 rsi-orchestrator 派发调优模块时使用：在预算内搜索比默认更好的训练配置。Use when the orchestrator dispatches the RSI tune module; not for general hyperparameter requests.
---

# rsi-tune：调优模块

利用历史实验搜索更好的配置和训练策略。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`、`trainset`、`valset`。
- 已登记的历史 `candidate` 和 `search`。
- 预算：简报给出的剩余算力等，超出要先报告。
- 不拿评测集和封存 test。

## 输出

- `search`：每次试验的配置、结果、花费，默认候选和最佳候选，相对默认配置的增益。
- 若干 `candidate`：每次试验一个，`config_source` 写 "tune"。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [models.md](../../docs/interfaces/models.md)。 `record` 见 [record.md](../../docs/interfaces/record.md)。

`search` 引用 `candidate` 的登记 id：先交回所有 `candidate`，等顶层登记并告知 id，再写好 `search` 交回。

## 门

- 每次试验调用训练命令（位置见简报的「训练命令」一行：任务定义的 `resources.train_command`，没给就是 `<repo root>/skills/rsi-train/bin/train`；运行时环境里带 `PYTHONPATH=<repo root>`；契约见 rsi-train，`--producer tune`，`--config-source tune`，每个候选目录配一个目录外的 `--marker`），不自己另写训练流程。
- 门：最佳配置在验证集上不差于默认配置。做不到就把默认配置的候选作为 `best_candidate` 交回，`gain_vs_default` 如实记录，不包装成提升。validate 会核对：`best_candidate` 的 `metric_val`（验证集主指标）不差于 `default_candidate` 的（方向按任务定义的 `higher_is_better`），`gain_vs_default.metric` 是主指标，`gain_vs_default.delta` 等于两者 `metric_val` 之差（最佳减默认）。
- 试验的 `metric_val` 是该试验候选的 `metrics_val` 里的主指标，即训练命令用 `rsi score` 算出的数，原样抄过来，不另外算。比较候选时也只用这些数。登记时代码核对每个 ok 试验的 `metric_val` 等于它的候选登记的数（容差 1e-9），不等就拒绝。
- 一次搜索只用一版训练代码：所有 ok 试验的候选（默认候选也在内）的 `code_version.sha256` 必须相同，有不同或为 null 就登记不了。搜索期间不改训练命令和默认配置；发现训练代码要改，记进 `record` 和报告，放到下一轮。
- 每个试验的 `status`：成功写 ok，并给出 `candidate_id` 和 `metric_val`；失败或超时的试验写别的状态，`candidate_id` 和 `metric_val` 可以为 null。`best_candidate` 和 `default_candidate` 必须是 ok 的试验。
- 试验花费之和不超过 `budget`，要超就先停下报告。
- `search` 用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 检查通过才算交出。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 `trainset`、`valset`、历史 `candidate` 和 `search` 从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`。不需要工作区，也不读登记表。
2. 输出写到沙盒下自己建的输出目录，每个产物一个子目录，manifest 的 `id` 写 null，`round` 自拟。
3. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验，通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
4. 试验调用训练命令（简报给路径，默认 `<repo root>/skills/rsi-train/bin/train`，任务定义的 `resources.train_command` 优先），候选目录和完成标记都放在沙盒。

## 步骤和策略子 skill

由模块负责人填写：搜索策略、利用历史实验的方法、预算分配、早停。把可重复的步骤写成脚本，skill 只写需要判断的部分。

## 模块指标

由模块负责人填写：同样预算下，相对默认配置的增益，及其测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），写明改了什么、前后数值、结论，`module` 写 `tune`。

## 长任务

由模块负责人填写：搜索（逐个试验后台运行，完成标记用训练命令的 `--marker`，见契约）。预计超过简报里规定的时长时，在后台启动，写完成标记（第一行 ok 或 failed），立刻交回标记路径、日志路径和预计耗时，不轮询、不等待。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。已知的一条：把试验在验证集上的偏差当成真实提升，或用评测集比较配置。
