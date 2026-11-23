# 产物接口总览

本目录说明框架中每类产物的格式。机器读写的文件一律是 JSON / JSONL（UTF-8）；校验由 `rsi` 命令完成，文档只描述含义。

## 产物类别

| 类别 | 产出者 | 使用者 | 说明 |
|---|---|---|---|
| `task` | orchestrator（与用户一起） | 顶层和所有模块 | 任务定义，见 task.md |
| `trainset` | data | train、tune；data 出下一版时以上一版为基础；judge 只用来算对照组 | 训练集，完整行，见 datasets.md |
| `valset` | data（第一轮，或获批刷新） | data、train、tune；deliver 只拿 `inputs.jsonl` | 验证集，完整行，另有不带标签的输入行；选 checkpoint、定阈值、比较配置、比较新旧训练集；deliver 在输入上复现候选的预测 |
| `evalset` | data 产出，交给 judge 保管 | judge（含标签）；deliver 只拿 `inputs.jsonl` | 评测集，输入与标签分开存放；只用于升级判定 |
| `testset` | data 产出，交给 judge 保管 | judge（含标签）；deliver 只拿 `inputs.jsonl`；每次使用都要 `use_testset` 审批 | 封存测试集，输入与标签分开存放；只用于里程碑发布 |
| `candidate` | train（tune 通过训练命令产出） | tune、deliver、judge | 候选模型，见 models.md |
| `search` | tune | train（用最佳配置）、tune（历史搜索）、judge、orchestrator | 配置搜索记录 |
| `delivery` | deliver | judge、用户 | 模型包、预测接口、评测集上的预测、延迟和成本，见 delivery.md |
| `feedback` | deliver，或接入真实系统后的接口 | data、judge | 执行反馈 |
| `evaluation` | judge | orchestrator | 评测报告和升级建议，见 evaluation.md |
| `plan` | judge | orchestrator，再派给计划指定的模块 | 下一轮计划 |
| `record` | 每个模块（顶层也可以写） | judge、orchestrator | 各模块的实验记录，Meta-RSI 的材料，见 record.md |

## 公共 manifest 字段

除 `task` 外，每个产物目录里都有 `manifest.json`，含以下公共字段。`task` 的目录里是 `task.json` 和 `TASK.md`，`task.json` 本身就是它的 manifest。

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 枚举 | 产物类别 |
| `id` | string 或 null | 登记前为 null；登记时写入 `<kind>-<四位序号>`，如 `trainset-0003` |
| `task_id`、`task_version` | string、int >= 1 | 所依据的任务定义 |
| `round` | int >= 1 | 产出时的轮次；登记时必须等于 STATE 的当前轮次 |
| `producer` | 枚举 | `orchestrator`、`data`、`train`、`tune`、`deliver`、`judge` |
| `created_at` | string | ISO 8601 时间 |
| `parents` | string 列表 | 依赖的已登记产物 id，可以为空 |
| `files` | object | 逻辑名到 `{"path": 相对路径, "rows": int 可选}` |
| `external` | object，可选 | 逻辑名到 `{"location": "主机:/绝对路径" 或 "/绝对路径", "sha256": string}`，用于留在别处的大文件 |
| `notes` | string，可选 | 备注 |

`files` 等字段里的相对路径写法上不能跳出产物目录（绝对路径、以 `..` 走出去的路径都拒绝）。产物目录里可以有符号链接：登记时复制的是链接指向的内容，登记后的副本里是实际的文件。例外是 evalset 和 testset，它们登记时整个目录被移走，指向目录外的符号链接、有第二个名字的文件（硬链接数大于 1）都会让那份内容（标签）留在原处，所以校验和登记都拒绝。指向不存在的文件的链接，任何类别都拒绝。

manifest 和任务定义里类型为 number 的值必须是有限数：`NaN`、`Infinity` 虽然能被 JSON 读入，校验时一律拒绝。

## 工作区目录

- `<ws>/STATE.json`：当前状态。
- `<ws>/registry.jsonl`：登记表，只追加。
- `<ws>/dashboard.html`：进度页面的概览页，由 `rsi` 生成，只供人看，不是产物，不参与任何校验和判定。
- `<ws>/dashboard/`：进度页面的次级页面 `rounds.html`（轮次与产物）、`events.html`（事件记录）、`report.html`（运行报告），与概览页一起生成，性质相同。
- `<ws>/artifacts/<id>/`：登记时从源目录复制过来的产物；evalset 和 testset 是移过来的，登记后源目录不再存在（见 datasets.md 的「交接」）。

运行停止（STATE 的 `stopped` 已写入，由 `rsi next` 命中停止条件或用户要求的 `rsi stop` 写入）后，`rsi promote`、`rsi close-round`、`rsi next`、`rsi stop`、除 `record` 以外的登记、除 `delete` 以外的审批都被拒绝；`delete` 审批（清理）和 `record` 登记（最后的实验记录）仍然允许。`rsi report` 不改状态（会重写进度页面），停止前后都能生成运行报告。

## 决策样本行（JSONL）

决策类型为 `binary`、`choice`、`score`。标签一律是字符串，`score` 的标签是 `"0"` 到 `"k-1"`。概率和为 1，容差 `abs(sum - 1) <= 1e-4`，单个概率在 [0, 1]。

| 行类型 | 使用处 | 字段 |
|---|---|---|
| 完整行 | trainset、valset | `id`；`input`；`target`；`meta.source`、`meta.group`、`meta.cluster`、`meta.label_source`（`rule`、`model_judge`、`human`、`outcome`）、`meta.n_obs`（int >= 1）、`meta.extra`（可选 object） |
| 输入行 | evalset、testset、valset 的 `inputs.jsonl` | `id`；`input`；`meta.group`、`meta.cluster`；不得有 `target`（输入行是给 deliver 的，不带标签） |
| 标签行 | evalset、testset 的 `labels.jsonl` | `id`；`target`；`meta.source`、`meta.label_source`、`meta.n_obs`、`meta.extra`（可选） |
| 预测行 | candidate 对输入行的输出 | `id`；`probs`（标签到概率）；`decision`（一个标签）；`uncertainty`（number 或 null） |
| 执行反馈行 | feedback | `id`；`input`；`output`（`probs`、`decision`）；`model`（版本或候选 id）；`latency_ms`；`timestamp`；`outcome`（标签或 null） |

`input` 在 `format: messages` 时是 `{"messages": [{"role": str, "content": str}, ...]}`，非空；在 `format: fields` 时是 `{"fields": {...}}`，键必须覆盖任务定义的 `input.fields`。

`target` 一律是概率映射：标签到概率，覆盖全部标签，和为 1。硬标签写成一个标签为 1、其余为 0，例如二分类 `{"yes": 1.0, "no": 0.0}`；多次观察或多个标注者给出的软标签写各标签的比例。`target` 从来不是一个裸字符串（`"yes"` 会被拒绝）。指标用的「硬标签」是 `target` 里概率最大的标签（见 task.md 的「指标的定义」）。

`meta` 里两个字段由 data 模块按任务定，供各模块和登记检查使用：

- `meta.group`：报告用的切片名（例如来源渠道、输入长度段），judge 按它分组给出结果。
- `meta.cluster`：近似重复簇的 id。同一模板、改写或近似重复的样本属于同一个簇，整个簇只能落在一个集合里，同一个 cluster 不能出现在两个集合中。

行 `id` 在 trainset、valset、evalset、testset 之间唯一，同一个 id 不能出现在两个集合里。登记 trainset、valset、evalset 或 testset 时，它与其余三类的当前集合之间只要有相同的 id、相同的输入（按输入指纹，忽略大小写和空白差异）或相同的 cluster，就拒绝登记；新的 evalset 还不能和以前登记过的任何 evalset 共享 id 或输入。详见 datasets.md 的「重叠检查」。

evalset 和 testset 的标签行只给 judge。deliver 只拿 `inputs.jsonl`（输入行：evalset 的、送评候选所用 valset 的，获批时 testset 的），不碰标签；train 和 tune 只拿 trainset 和 valset，从不拿 evalset 和 testset。校验函数在 `rsi/rows.py`。
