---
name: rsi-judge
description: 由 rsi-orchestrator 派发评测与决策模块时使用：用评测集标签给交付打分、给出升级建议、失败归因和下一轮计划。Use when the orchestrator dispatches the RSI judge module; not for general evaluation requests.
---

# rsi-judge：评测与决策模块

保管评测集的标签，打分、比较，给出升级建议、失败分析和下一轮计划。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`（指标、对照组、升级规则）。
- 送评的 `candidate`，本轮的 `delivery`，本轮有 `search` 时也给。
- `evalset`，含 labels.jsonl；只有本模块拿标签。
- 送评候选所用 `trainset` 的 rows.jsonl，只用来算对照组，不做别的分析。
- 当前最佳版本的版本名，以及它的 `evaluation` 和 `delivery`；简报里写的 `best_before` 的值照用。评测集刷新过时，另拿最佳版本在当前评测集上的 `delivery`：用它算最佳版本一侧的指标和差值，并把它的 id 写进 evaluation 的 parents。
- `feedback`，全部 `record`。
- 里程碑发布获批时，另拿 `testset`（含标签）。

## 输出

- `evaluation`：各指标、与当前最佳版本和对照组的比较及区间、升级或保留的建议、失败项及归因、`uses_testset`；报告文件按分组和切片给出结果。
- `plan`：下一轮从哪个模块进入、每个问题的证据和改动、预期提升。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [evaluation.md](../../docs/interfaces/evaluation.md)。 `record` 见 [record.md](../../docs/interfaces/record.md)。

`plan` 引用 `evaluation` 的登记 id：先交回 `evaluation`，等顶层登记并告知 id 和升级判定，再写 `plan` 交回。

简报说本轮已关闭（没有候选通过训练门，本轮没有 `evaluation`）时，只写 `plan` 和 `record`：plan 的 evaluation_id 写 null，问题和证据取自本轮候选的训练门结果和实验记录。

里程碑评测（简报说是里程碑发布时）：给当前最佳版本的 delivery 打分，`uses_testset` 写 true，`candidate_id` 是当前最佳版本的候选，`best_before` 写当前最佳版本；指标、差值仍按评测集算，test 上的成绩写进 `metrics_test`（含主指标）；建议写 keep，它不是升级判定，不占评测集使用次数。不写 `plan`。格式见 [evaluation.md](../../docs/interfaces/evaluation.md) 的「里程碑评测」。

## 门

- 指标、对照组、差值和区间一律用 `rsi score` 算，打印的数原样写进 evaluation，不自己实现（定义见 [task.md](../../docs/interfaces/task.md) 的「指标的定义」）。登记时代码按同样的定义重算，差超过 1e-9 就被拒：
  - `metrics.candidate`、`metrics.baselines`、`deltas` 的主指标：`PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <evalset>/labels.jsonl --predictions <delivery>/predictions_eval.jsonl --train <trainset>/rows.jsonl --clusters <evalset>/inputs.jsonl --compare <best-delivery>/predictions_eval.jsonl`，输出的 `metrics`、`baselines`、`delta` 分别写进这三处；还没有最佳版本时去掉 `--compare`，`metrics.best` 写 null，`deltas` 留空。
  - `metrics.best`：同一条命令把 `--predictions` 换成最佳版本的 `predictions_eval.jsonl`，取输出的 `metrics`。评测集刷新过时，最佳版本一侧用它在当前评测集上的 delivery（简报给出），并把这份 delivery 的 id 写进 `parents`，否则登记被拒。
  - 里程碑评测的 `metrics_test`：`PYTHONPATH=<repo root> python3 -m rsi score --task <task.json> --labels <testset>/labels.jsonl --predictions <delivery>/predictions_test.jsonl`。
  - 输出的 `not_computed` 列出 `rsi` 不认识的指标和基线：这些由本模块自己算，在报告里写明算法，登记时不核对。
  - 切片、分组结果等报告里的其他数字由本模块自定，不核对。
- 建议必须与升级规则一致：按任务定义的 `promotion_rule` 和 [task.md](../../docs/interfaces/task.md) 描述的判定方法，用 `rsi score` 算出、写进 evaluation 的差值、区间和护栏指标（登记时会重算核对）得出 `promote` 或 `keep`（提升必须大于 `min_delta`，恰好等于它的平局，容差 1e-9 内，不算提升，写 `keep`），`reason` 引用规则和对应数字。顶层登记并判定升级时会用代码核对，不一致就被拒。
- 归因只能取 [evaluation.md](../../docs/interfaces/evaluation.md) 规定的值，每个失败项有证据和唯一的归因，模块与归因对应。
- `plan` 的 `entry_module` 必须是某个 issue 的模块，每个 issue 都有证据、改动和预期提升。
- 评测集使用次数有限，每份登记的 evaluation（里程碑评测除外）用一次；用封存 test 只在获批后，且 `uses_testset` 写 true。
- 评测集和封存 test 的标签不得写进给别的模块看的 `plan`、`record` 和报告里的逐条样本。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 `candidate`、`delivery`、`evalset`（含标签）、送评候选所用 `trainset` 的 rows.jsonl（只用来算对照组）、历史 `evaluation` 和 `record` 从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`。不需要工作区，也不读登记表。
2. 输出写到沙盒下自己建的输出目录，每个产物一个子目录，manifest 的 `id` 写 null，`round` 自拟。
3. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验，通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
4. `plan` 引用 `evaluation` 的 id 时，沙盒里手填一个即可，登记时由顶层核对。

## 步骤和策略子 skill

由模块负责人填写：切片分析、失败归因、计划生成、预期提升的校准。打分、对照组和置信区间是固定代码（`rsi score`，见「门」），不在这里另定。把可重复的步骤写成脚本，skill 只写需要判断的部分。

## 模块指标

由模块负责人填写：计划的命中率：计划里预期的提升有多少真的实现了，及其测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），写明改了什么、前后数值、结论，`module` 写 `judge`。

## 长任务

由模块负责人填写：大量重采样或逐条分析（后台运行，写完成标记）。预计超过简报里规定的时长时，在后台启动，写完成标记（第一行 ok 或 failed），立刻交回标记路径、日志路径和预计耗时，不轮询、不等待。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。已知的一条：建议与升级规则不一致，或归因不取规定的值。
