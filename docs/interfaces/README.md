# 产物接口总览

本目录说明框架中每类产物的格式。机器读写的文件一律是 JSON / JSONL（UTF-8）；校验由 `rsi` 命令完成，文档只描述含义。

## 产物类别

| 类别 | 产出者 | 使用者 | 说明 |
|---|---|---|---|
| `task` | orchestrator（与用户一起） | 顶层和所有模块 | 任务定义，见 task.md |
| `trainset` | data | train；data 出下一版时以上一版为基础；judge 只用来算对照组 | 训练集，完整行，见 datasets.md |
| `valset` | data（第一轮，或获批刷新） | data、train；deliver 只拿 `inputs.jsonl` | 验证集，完整行，另有不带标签的输入行；选 checkpoint、定阈值、比较配置；data 用它做去重检查；deliver 在输入上复现候选的预测 |
| `evalset` | data 产出，交给 judge 保管 | judge（含标签）；deliver 只拿 `inputs.jsonl` | 评测集，输入与标签分开存放；只用于升级判定 |
| `testset` | data 产出，交给 judge 保管 | judge（含标签）；deliver 只拿 `inputs.jsonl`；每次使用都要 `use_testset` 审批 | 封存测试集，输入与标签分开存放；只用于里程碑发布 |
| `corpus` | data（第一轮 `sets` 派出） | 只有 data：产出它的 `sets` 派出和 `refresh` 派出 | 预处理脚本和它写出的全部带标签行，登记时移交进工作区；四个集合和 pool 都出自它，见 datasets.md |
| `pool` | data（第一轮） | data：之后各轮出训练集时 | 训练池：训练时段的全部完整行，训练集从它抽，见 datasets.md |
| `candidate` | train（旧工作区还有 tune 产出的，只读） | train、deliver、judge | 候选模型，见 models.md |
| `search` | train（旧工作区还有 tune 产出的，只读） | train（历史搜索）、judge、orchestrator | 配置搜索记录 |
| `delivery` | deliver | judge、用户 | 模型包、预测接口、评测集上的预测、延迟和成本，见 delivery.md |
| `feedback` | deliver，或接入真实系统后的接口 | data、judge | 执行反馈 |
| `evaluation` | judge | orchestrator | 评测报告和升级建议，见 evaluation.md |
| `plan` | judge | orchestrator，再派给计划指定的模块；judge 下一轮评测时对照它判断是否奏效 | 下一轮计划 |
| `record` | 每个模块（顶层也可以写） | judge、orchestrator；各模块之后几轮读自己的 | 各模块的实验记录，Meta-RSI 的材料，见 record.md |

## 公共 manifest 字段

除 `task` 外，每个产物目录里都有 `manifest.json`，含以下公共字段。`task` 的目录里是 `task.json` 和 `TASK.md`，`task.json` 本身就是它的 manifest。

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 枚举 | 产物类别 |
| `id` | string 或 null | 登记前为 null；登记时写入 `<kind>-<四位序号>`，如 `trainset-0003` |
| `task_id`、`task_version` | string、int >= 1 | 所依据的任务定义 |
| `round` | int >= 1 | 产出时的轮次；登记时必须等于 STATE 的当前轮次 |
| `producer` | 枚举 | `orchestrator`、`data`、`train`、`deliver`、`judge`（旧工作区还有 `tune`，只读，归 ② 训练调优） |
| `created_at` | string | ISO 8601 时间 |
| `parents` | string 列表 | 依赖的已登记产物 id，可以为空 |
| `files` | object | 逻辑名到 `{"path": 相对路径, "rows": int 可选}` |
| `external` | object，可选 | 逻辑名到 `{"location": "主机:/绝对路径" 或 "/绝对路径", "sha256": string}`，用于留在别处的大文件 |
| `notes` | string，可选 | 备注 |

`files` 等字段里的相对路径写法上不能跳出产物目录（绝对路径、以 `..` 走出去的路径都拒绝）。产物目录里可以有符号链接：登记时复制的是链接指向的内容，登记后的副本里是实际的文件。例外是 evalset、testset 和 corpus，它们登记时整个目录被移走，指向目录外的符号链接、有第二个名字的文件（硬链接数大于 1）都会让那份内容（标签）留在原处，所以校验和登记都拒绝。指向不存在的文件的链接，任何类别都拒绝。

manifest 和任务定义里类型为 number 的值必须是有限数：`NaN`、`Infinity` 虽然能被 JSON 读入，校验时一律拒绝。

## 工作区目录

- `<ws>/STATE.json`：当前状态。
- `<ws>/registry.jsonl`：登记表，只追加。
- `<ws>/dashboard.html`：进度页面的概览页，由 `rsi` 生成，只供人看，不是产物，不参与任何校验和判定。
- `<ws>/dashboard/`：进度页面的次级页面 `rounds.html`（轮次与产物）、`events.html`（事件记录）、`report.html`（运行报告），与概览页一起生成，性质相同。
- `<ws>/STATE.lock`：对 STATE.json 读改写时持有的锁文件（空文件，`flock`），见 engine.md。
- `<ws>/work/run.jsonl`：引擎的运行日志，每行一个事件（动作、派出、退回、决策、回答等），只追加；`<ws>/work/run.lock` 是 `rsi run` 运行时的锁（写入进程号）。
- `<ws>/work/round-<n>/<模块>[-k]/`：一次派出的输出目录：`brief.md`、`brief.json`、`prompt-<n>.md`、`agent-<n>.log`、`feedback-<n>.md`、`history-<n>.md`、`handback.json`，以及模块写的产物子目录；`<ws>/work/decisions/<n>-<决策点>/` 是决策 agent 的文件。它们不是产物，登记时才把产物子目录复制进 `artifacts/`。引擎的记录（派出、决策、待答问题）在 STATE 的 `dispatches`、`decisions`、`waiting`，见 engine.md。
- `<ws>/work/round-<n>/<模块>[-k]/progress.json`：模块可选写的进度文件（简报的 `progress_file`），进度页面和 `rsi run` 的状态行原样显示它，格式见 progress.md。不是产物，不参与校验和判定。
- `<ws>/artifacts/<id>/`：登记时从源目录复制过来的产物；evalset、testset 和 corpus 是移过来的，登记后源目录不再存在（见 datasets.md 的「交接」）。

运行停止（STATE 的 `stopped` 已写入，由 `rsi next` 命中停止条件或用户要求的 `rsi stop` 写入）后，`rsi promote`、`rsi close-round`、`rsi next`、`rsi stop`、除 `record` 以外的登记、除 `delete` 以外的审批都被拒绝；`delete` 审批（清理）和 `record` 登记（最后的实验记录）仍然允许。`rsi report` 不改状态（会重写进度页面），停止前后都能生成运行报告。

## 决策样本行（JSONL）

数据集的行就是 OneJev 行（onejev-harness 的 `onejev_rows` 适配器直接读，候选的预测入口也直接读），不做格式转换。决策类型为 `binary`、`choice`、`score`。标签一律是字符串，`score` 的标签是 `"0"` 到 `"k-1"`。

| 行类型 | 使用处 | 字段 |
|---|---|---|
| 完整行 | trainset、valset、pool、corpus 的 `rows.jsonl` | `id`；`state`（格式按任务定义的 `input.format`：`messages`、`fields` 或 `object`）；`question`；`target`；`meta.state_id`、`meta.cluster`、`meta.source`、`meta.label_source`（`rule`、`model_judge`、`human`、`outcome`）、`meta.n_obs`（int >= 1）、`meta.extra`（可选 object），`meta` 里可以有别的字段；`split`（非空字符串） |
| 输入行 | evalset、testset、valset 的 `inputs.jsonl` | 完整行只留顶层键 `id`、`state`、`question`、`meta`、`split`（所以没有 `target`），`meta` 只留 `state_id`、`cluster`、`source`、`n_obs`（`rsi/rows.py` 的 `input_row_of`）；顶层或 `meta` 里有别的键就拒绝登记（输入行是给 deliver 和候选预测入口的，不带标签） |
| 标签行 | evalset、testset 的 `labels.jsonl` | `id`；`target`；`meta.label_source`，`meta` 只有这一个键 |
| 预测行 | candidate 对输入行的输出 | `id`；`probs`（标签到概率，覆盖全部标签，和为 1，容差 `abs(sum - 1) <= 1e-4`，单个概率在 [0, 1]）；`decision`（一个标签）；`uncertainty`（number 或 null） |
| 执行反馈行 | feedback | `id`；`state`（这次调用的输入，即输入行的 `state`；旧工作区的 trace 里叫 `input`，照样读）；`output`（`probs`、`decision`）；`model`（版本或候选 id）；`latency_ms`；`timestamp`；`outcome`（标签或 null） |

`state` 是模型的输入，格式由任务定义的 `input` 一节规定：`format: messages` 时是 `{"messages": [{"role": str, "content": str}, ...]}`，非空；`format: fields` 时是 `{"fields": {...}}`，键必须覆盖任务定义的 `input.fields`；`format: object` 时是任意非空 JSON object（例如 `{"entity": ..., "summary": {...}, "transactions": [...]}`），不再检查内部结构，原样交给模型（OneJev 接受任何非空的 state object）。

`question` 是 OneJev 的问题 `{"type", "instructions", "criteria"}`，每一行都必须恰好等于由任务定义的 `decision` 得出的问题（`rsi/rows.py` 的 `question_of`），登记时逐行核对，不一致就拒绝：

| 任务的 `decision` | `question.type` | `question.criteria` |
|---|---|---|
| `binary`，标签是 `yes` 和 `no` | `noul` | `{"true": yes 的含义, "false": no 的含义}`（`label_meanings`；OneJev 的 noul 标签固定是 `yes`、`no`） |
| `choice`，或标签不是 `yes`、`no` 的 `binary` | `choice` | `label_meanings`：标签到含义 |
| `score` | `score` | `[ "0" 的含义, "1" 的含义, ... ]`，按标签顺序 |

`question.instructions` 等于 `decision.instructions`。所以模型看到的问题就是任务定义写的决策，数据模块不能各自改写。

`target` 一律是概率映射，按 OneJev 的规则（onejev-harness `validate_rows`）检查：非空 object，键都是问题的标签（即任务的标签），没写的标签概率为 0；每个值在 [0, 1]；和为 1，容差 `abs(sum - 1) <= 1e-6`。硬标签写成一个标签为 1、其余为 0，例如二分类 `{"yes": 1.0, "no": 0.0}`；多次观察或多个标注者给出的软标签写各标签的比例。`target` 从来不是一个裸字符串（`"yes"` 会被拒绝）。指标用的「硬标签」是 `target` 里概率最大的标签（见 task.md 的「指标的定义」）。

`meta` 里两个字段由 data 模块按任务定，供各模块和登记检查使用：

- `meta.state_id`：行所属的分组（OneJev 的状态 id，onejev-harness 按它分组），`group_prior` 对照组按它分组，judge 按它给出分组结果。
- `meta.cluster`：近似重复簇的 id。同一模板、改写或近似重复的样本属于同一个簇，整个簇只能落在一个集合里，同一个 cluster 不能出现在两个集合中。

**输入行不带标签。** 输入行的顶层只有 `id`、`state`、`question`、`meta`、`split` 五个键，`meta` 只有 `state_id`、`cluster`、`source`、`n_obs` 四个键：完整行多出的顶层键和 `meta` 的其他字段可能由标签推出（例如 OneJev 导出里的一致率 `f`、`mode_score`、`mode_idx`，`target` 就是由 `f` 算出来的；`label_source` 也算，标签怎么来的可能和标签本身相关，例如 illicit 来自 human、licit 来自 rule），所以不进输入行（`input_row_of` 只留这些键），`label_source` 只在完整行和标签行里，登记时输入行出现别的顶层键或 `meta` 键就拒绝，错误信息写出键名。标签行的顶层同样只有 `id`、`target`、`meta` 三个键。输入行的任何字段（`state`、`question`、`split` 和这四个 `meta` 字段）都不能带由标签推出的值。改编 OneJev 导出时，写输入行要去掉 `f`、`mode_score`、`mode_idx` 这类字段；完整行里可以留着（train 本来就看标签）。

`split` 是 OneJev 的切分名，由 data 模块写（例如 `train`、`val`、`eval`、`test`）。注意 onejev-harness 拒绝读 `split` 以 `test` 开头的训练行和验证行（除非加 `--allow-test`），trainset、valset 和 pool 的行不要用这样的名字：登记时这样的行会被拒绝（规则同 onejev-harness：去掉首尾空白、转成小写后以 `test` 开头，见 datasets.md）。

**旧工作区的行。** 改成 OneJev 行之前登记的行用 `input` 和 `meta.group`，读的时候当作 `state` 和 `meta.state_id`（`rsi/rows.py` 的 `state_of`、`group_of`），所以旧工作区的指标、对照组、重叠检查照常；新登记的行必须是上面的格式。

行 `id` 在 trainset、valset、evalset、testset、pool 之间唯一，同一个 id 不能出现在两个集合里（trainset 和 pool 除外：trainset 从 pool 抽）。登记 trainset、valset、evalset、testset 或 pool 时，它与其余几类的当前集合之间只要有相同的 id、相同的输入（按 `state` 的指纹，忽略大小写和空白差异）或相同的 cluster，就拒绝登记；切分规则写 `unit: state_id` 时另比 `state_id`；新的 evalset 还不能和以前登记过的任何 evalset 共享 id 或输入；在 corpus 之后登记的 valset、evalset、testset、pool，每一行都必须是 corpus 里同 id、同输入、同 `state_id` 和 cluster、同标签的行（所以第一份 corpus 要先于它们登记）。详见 datasets.md 的「重叠检查」。

evalset 和 testset 的标签行只给 judge。deliver 只拿 `inputs.jsonl`（输入行：evalset 的、送评候选所用 valset 的，获批时 testset 的），不碰标签；train 只拿 trainset 和 valset，从不拿 evalset 和 testset。校验函数在 `rsi/rows.py`。
