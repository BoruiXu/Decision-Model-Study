---
name: rsi-data
description: 由 rsi-orchestrator 派发数据模块时使用：产出训练集，第一轮或获批刷新时另出验证集、评测集和封存 test。Use when the orchestrator dispatches the RSI data module; not for general data requests.
---

# rsi-data：数据模块

获取或生成数据，检查并提高数据质量。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`：任务定义（必有）。
- 可选：上一版 `trainset`；`plan`（其中归因到数据的问题）；带真实结果的 `feedback`。
- 第一轮之后的各轮还拿 `valset`，用于新旧训练集的 A/B 比较和去重检查。不再给评测集和封存 test。

## 输出

- `trainset`：每次都出新版本，需带数据质量报告。
- 第一轮或获批刷新时另出 `testset`、`evalset`、`valset`。验证集除了完整行的 rows.jsonl，还要写一份不带标签的 inputs.jsonl（同样的 id 和输入，加上 meta.group、meta.cluster，不写 target），deliver 用它复现候选的验证集预测。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [datasets.md](../../docs/interfaces/datasets.md)。 数据产物的行格式和目录结构见该文档，`record` 见 [record.md](../../docs/interfaces/record.md)。

先交回被引用的产物：`testset`、`evalset`、`valset`、`trainset` 按这个顺序列在交回报告里，顶层按此顺序登记（trainset 之后才引用它们的 id）。

## 门

- 数据质量门：任务定义 `quality_checks` 规定的检查和阈值（样本量、标签分布、标签信度、近似重复等；任务定义没写时自定检查项，并在报告里说明）逐项算出数值、阈值、是否通过，写进 `trainset` 的 `quality`，`quality.passed` 等于各项相与。没过就如实写 false 并在报告里说明，由顶层决定是改还是问用户；这样的训练集登记后只作为证据，不会成为当前训练集，train 和 tune 不会用它。
- 训练集与验证集、评测集、封存 test 之间没有重复（近似重复也算），第一轮产出这些集合时由 data 自己查。
- 评测集和封存 test 交出之后不保留访问：直接在输出目录里生成，不在别处留副本或带标签的中间文件，也不用符号链接或硬链接引用别处的文件（校验和登记会拒绝）；顶层登记时会把这两个目录整个移进工作区，输出目录里就不再有它们。交回后不再读取，之后的派活也不会再给；标签只在产出时经手。已知限制：代码只移走目录，同一台机器上仍读得到工作区里的标签，隔离靠约定。
- 新旧训练集的比较调用训练命令（位置见简报的「训练命令」一行：任务定义的 `resources.train_command`，没给就是 `<repo root>/skills/rsi-train/bin/train`；运行时环境里带 `PYTHONPATH=<repo root>`；契约见 rsi-train），`--producer data`，用同一配置、同一验证集；新训练集还没登记，`--trainset-id` 写 unregistered。这些 A/B 候选只是草稿，放在自己输出目录下的 ab/，不交回登记（登记会拒绝，这是有意的），比较结果写进 `record`。
- 需要刷新验证集、评测集，或要超预算时，停下报告，不自己决定。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 上一版 `trainset`、`valset`、`feedback` 等输入 从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`。不需要工作区，也不读登记表。
2. 输出写到沙盒下自己建的输出目录，每个产物一个子目录，manifest 的 `id` 写 null，`round` 自拟。
3. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验，通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
4. 比较新旧训练集时调用训练命令（简报给路径，默认 `<repo root>/skills/rsi-train/bin/train`，`resources.train_command` 优先；`--producer data`、`--trainset-id unregistered`），草稿候选放在沙盒的 ab/，完成标记放在候选目录之外。

## 步骤和策略子 skill

由模块负责人填写：数据来源获取或生成、标注、质量检查、去重、切分、新旧训练集 A/B、从反馈补样本。把可重复的步骤写成脚本，skill 只写需要判断的部分。

## 模块指标

由模块负责人填写：用同一套训练配置在新旧训练集上各训一次，验证集上主指标的差，及其测量方法。

## 实验记录

由模块负责人填写：每次改进实验交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），写明改了什么、前后数值、结论，`module` 写 `data`。

## 长任务

由模块负责人填写：大批量生成、标注（例如调用 judge 模型）。预计超过简报里规定的时长时，在后台启动，写完成标记（第一行 ok 或 failed），立刻交回标记路径、日志路径和预计耗时，不轮询、不等待。

## 负责人和版本

由模块负责人填写：负责人、当前版本号、变更记录、已知限制。

## 常见错误

由模块负责人填写：本模块出现过的错误、后果和正确做法（表格）。已知的一条：把评测集或封存 test 在交出之后继续留着用，或在训练集里混入它们的样本。
