# 交付类产物：delivery、feedback

两类产物由 deliver 产出，目录里有 `manifest.json`，公共字段见 README.md。校验由 `rsi/artifacts.py` 的 `validate_artifact` 完成（额外字段之外不允许出现未列出的字段）。预测行和执行反馈行的格式见 README.md 的「决策样本行」。id 只检查是字符串，是否已登记在登记时核对。

## 预测接口

交付包对外提供同一个预测接口：

- 输入：一个样本的 `state`（输入行的 `state`，格式由任务定义的 `input.format` 决定）。
- 输出：`{"probs", "decision", "uncertainty"}`，其中 `probs` 是标签到概率的映射，`decision` 是一个标签，`uncertainty` 是 number 或 null。
- 每次调用写一条执行反馈行，追加到 `trace`：字段 `id`、`state`（这次调用的输入，即输入行的 `state`；旧工作区的 trace 里叫 `input`，登记照样读）、`output`（`probs`、`decision`）、`model`、`latency_ms`、`timestamp`、`outcome`。`outcome` 在真实结果还不知道时为 null，知道后是一个标签。执行反馈行的 `id` 是这次调用的 id，格式由 deliver 自定（不必等于输入行的 id），在一份 `trace` 里不重复即可。

接口形式（命令行、函数、HTTP）由交付模块决定，写在 `package.entry`。交付包通常包着候选的预测入口 `model.predict`（见 models.md），但对外以 `package.entry` 为准。

## `delivery`

把一个候选打包成可调用的服务。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"delivery"` | 产物类别 |
| `candidate_id` | string | 被交付的候选 |
| `evalset_id` | string | `predictions_eval` 所在的评测集，必须是当前评测集 |
| `valset_id` | string | `predictions_val` 所在的验证集，必须是候选的 `valset_id`（见「复现候选的验证集预测」） |
| `testset_id` | string，可选 | 用到 testset 时必填 |
| `package` | object | 交付包，见下 |
| `runtime` | object | 运行指标，见下 |

`package` 的子字段：`location`（交付包位置，见下）；`entry`（调用入口，写明接口形式和调用方法）；`environment`（运行环境说明）。

`runtime` 的子字段：

- `latency_ms_p50`、`latency_ms_p95`：预测接口每次调用的墙钟耗时（毫秒）在评测集输入上的中位数和 95 分位数，一次调用处理一个样本；加载模型的时间不算在内。
- `cost_unit`：string，可选，成本的单位，例如 `usd`、`gpu_seconds`。
- `cost_per_1k`：number 或 null，做 1000 次预测的成本，单位是 `cost_unit`；不知道时为 null。
- `consistent_with_offline`：bool，含义固定为下面的复现检查：交付包在验证集输入上的预测与候选的 `predictions_val` 一致。

必备文件：`predictions_eval`（交付包在评测集 `inputs.jsonl` 上的预测）、`predictions_val`（交付包在验证集 `inputs.jsonl` 上的预测）；可选 `predictions_test`、`trace`，只有出现在 `files` 里才会校验。目录内检查：预测行合法；`runtime.consistent_with_offline` 为 false 时报错；`package.location` 合法（见下）；有 `predictions_test` 就必须有 `testset_id`；有 `trace` 时执行反馈行合法。

**交付包自包含。** `package.location` 只能是两种形式之一：

- 相对交付目录的路径（文件或目录），写法上不能跳出交付目录（不以 `..` 走出去），并且存在。登记时整个交付目录被复制进工作区，交付包随之进入 `<ws>/artifacts/<delivery id>/`，所以登记后的交付包不再依赖模块的工作目录。
- `external:<名字>`：指公共字段 `external` 里的同名条目，它带有位置和 sha256，用于放不进目录的大包。本机上的条目（`location` 为 `/绝对路径`）必须是单个文件，例如打好的 tar 包，因为校验和登记要算它的 sha256；目录先打包再登记。远端条目（`主机:/路径`）不核对。

绝对路径一律拒绝。交付目录里可以有符号链接（例如交付包里的 `python3 -m venv` 环境链接到系统的解释器）：登记时复制的是链接指向的内容，登记后的交付包里是实际的文件，不再依赖链接的目标。指向不存在的文件的链接一律拒绝。`rsi report` 列出的模型包是登记后的路径 `<ws>/artifacts/<id>/<location>`，或 `external` 条目的位置和 sha256。

**复现候选的验证集预测。** deliver 拿到候选所用验证集的 `inputs.jsonl`（不带标签），用交付包的预测接口跑出 `predictions_val`，`valset_id` 写这个验证集。登记时核对：

- `valset_id` 已登记，并且等于候选的 `valset_id`（候选的 `predictions_val` 就在那个验证集上）。
- `predictions_val` 恰好覆盖该验证集的 id。
- 每一行与候选 `predictions_val` 的同 id 行一致：每个标签的概率相差不超过任务定义的 `delivery.reproduction_tolerance`（不写时 1e-6，见 task.md；`uncertainty` 不比），`decision` 相同。
- 概率都在容差之内、只有 `decision` 不同的行（靠近决策边界的行）最多 floor(`delivery.max_decision_mismatch_rate` × 验证集行数) 行，不写时为 0 行；有这样的行并被放过时，`rsi register` 在输出行末尾加 `(decision mismatches: <行数>)`。决策不同而且有概率超出容差的行不算在放宽里，照常拒绝。

`runtime.consistent_with_offline` 指的就是这项检查。复现不成立时，即使标志写 true 也拒绝登记，错误信息给出不一致的行数和前几行的差别。这样交付包确实是那个候选，评测集上的成绩才能算在候选头上。

登记时另外核对：`candidate_id` 已登记且通过了训练门（`train_gate.passed` 为 true，否则拒绝并写明训练门没过）；`evalset_id` 是当前评测集，`predictions_eval` 恰好覆盖它的 id；有 `predictions_test` 时 `testset_id` 是当前 testset、预测恰好覆盖它的 id，并且本轮有一次未用过的 `use_testset` 审批。

## `feedback`

线上或试运行收集的执行反馈。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"feedback"` | 产物类别 |
| `delivery_id` | string | 反馈来自哪个 delivery |
| `candidate_id` | string | 反馈来自哪个候选 |

`delivery_id` 和 `candidate_id` 至少要有一个。必备文件：`trace`。目录内检查：每行是合法的执行反馈行（`outcome` 必须是标签或 null）。
