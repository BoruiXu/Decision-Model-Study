# 数据类产物：trainset、valset、evalset、testset

四类数据产物都由 data 模块产出，目录里有 `manifest.json`，公共字段见 README.md。校验由 `rsi/artifacts.py` 的 `validate_artifact` 完成：先按公共字段加该类的额外字段校验 manifest（不允许出现未列出的字段），再检查 `files` 里每个文件存在、`rows` 与文件里非空行数一致、本地 `external` 的 sha256，最后做该类的目录内检查。行格式见 README.md 的「决策样本行」，行校验在 `rsi/rows.py`。

## 谁能看到标签

引设计文档第 7 节：

| 集合 | 由谁产出 | 谁能看到标签 | 用途 | 是否变化 |
|---|---|---|---|---|
| `trainset` | data | train、tune；judge 只用来算对照组 | 训练 | 每轮可以出新版本 |
| `valset` | data，第一轮 | data、train、tune | 选 checkpoint、定阈值、比较配置、比较新旧数据 | 第一轮后固定，刷新要用户批准 |
| `evalset` | data，第一轮，产出后交给 judge | 只有 judge | 升级判定 | 固定；使用次数有上限，到上限后刷新要用户批准 |
| `testset` | data，第一轮，产出后交给 judge | 只有 judge，且每次使用要用户批准（`use_testset`） | 里程碑发布时的最终验证 | 不变 |

evalset 和 testset 把输入和标签分成两个文件：deliver 只拿 `inputs.jsonl`（testset 的要获批），标签文件只给 judge；valset 另有一份不带标签的 `inputs.jsonl`，deliver 拿它来复现候选的验证集预测；train 和 tune 从不拿 evalset 和 testset。judge 拿送评候选所用训练集的 `rows.jsonl`，只用来算评测集上的对照组（对照组由训练集的先验得出，见 task.md 的「指标的定义」）。

## `trainset`

训练集，完整行。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"trainset"` | 产物类别 |
| `label_method` | string | 标签是怎么得到的，例如规则、judge、人工 |
| `parent_trainset` | string 或 null | 上一版训练集的 id；第一版为 null |
| `changes` | string | 相对上一版改了什么 |
| `quality` | object | 数据质量检查结果 |

`quality` 的子字段：`passed`（bool）；`checks`（列表，每项 `name`、`value`、`threshold`、`passed`）。目录内检查：每行是合法的完整行；`quality.passed` 必须等于全部 `checks[].passed` 的与。

`quality.passed` 为 false 的训练集照常登记，作为证据留在登记表里，但不会成为当前训练集（STATE 的 `current.trainset` 不变），candidate 也就不能引用它。`rsi register` 在 id 后面打印质量门结果，例如 `trainset-0004 (quality gate: failed (dup_rate); never current)`。

必备文件：`rows`（完整行 JSONL，需写 `rows` 行数）、`quality_report`（质量报告，格式由 data 模块自定）。

```
trainset-0003/
  manifest.json
  rows.jsonl
  quality_report.json
```

## `valset`

验证集，没有额外字段（`kind` 固定为 `"valset"`）。必备文件：`rows`（完整行）和 `inputs`（输入行，不带标签），两个都需写 `rows` 行数。`inputs.jsonl` 由 data 模块从 `rows.jsonl` 去掉 `target` 等标签字段写出，交给 deliver：deliver 用交付包在它上面跑一遍，登记时与候选的 `predictions_val` 逐行比较（见 delivery.md），deliver 看不到验证集的标签。目录内检查：完整行合法；输入行合法（不得有 `target`）；`inputs` 与 `rows` 的 id 集合相同，同一 id 的 `input` 完全相同。

```
valset-0001/
  manifest.json
  rows.jsonl
  inputs.jsonl
```

## `evalset` 与 `testset`

评测集和封存测试集，格式相同，没有额外字段（`kind` 分别固定为 `"evalset"`、`"testset"`）。必备文件：`inputs`（输入行）、`labels`（标签行），两个都需写 `rows` 行数。目录内检查：输入行、标签行各自合法（输入行不得有 `target`）；两边 id 集合相同。

```
evalset-0002/
  manifest.json
  inputs.jsonl
  labels.jsonl
```

**交接：登记时移入工作区。** 其他类别的产物登记时是复制，源目录留在原处；evalset 和 testset 登记时是移交：所有检查通过后，源目录被移进工作区（`<ws>/artifacts/<id>/`），原来的目录不再存在，产出它的模块的输出目录里也就不再有这份标签。登记被拒时什么都不动，源目录原样留着，改好后再登记。源目录移不走（例如所在目录没有写权限）时拒绝登记，同样什么都不改。登记表的 `source` 仍记原路径，之后在同一路径重新建目录也登记不了（同一源路径只登记一次）。目录里指向目录外的符号链接、硬链接数大于 1 的文件会让标签留在原处，校验和登记都拒绝，要把文件本身写进目录。登记中途被终止（在把源目录挪开之后、写入登记表之前）时，源目录在同级的隐藏目录 `.<名字>.rsi-handed-over-*` 里：手工把它改回原名，再登记一次。

所以 data 模块直接在简报给的输出目录里生成评测集和封存 test，不在别处留副本，也不保留中间文件里的标签；交回之后不再读取。已知限制：代码只能把源目录移走，同一台机器上的进程仍然读得到工作区里的标签文件，模块之间的隔离靠约定（简报不给模块工作区路径），代码不阻止。

## 重叠检查

行里的 `meta.group` 和 `meta.cluster` 由 data 模块按任务定：

- `meta.group`：报告用的切片名（例如来源渠道、输入长度段），judge 按它分组给出结果。
- `meta.cluster`：近似重复簇的 id。同一模板、改写或近似重复的样本属于同一个簇；切分时整个簇放进同一个集合，同一个 cluster 不能出现在两个集合中。

行 `id` 在 trainset、valset、evalset、testset 之间唯一。四个集合之间不能有重复，这个检查在登记时做：登记一个 trainset、valset、evalset 或 testset 时，它与其余三类的当前集合逐一比较，只要有相同的 id、相同的输入或相同的 cluster 就拒绝登记，错误信息给出三个重叠数。新的 evalset 还要和以前登记过的每个 evalset 比较，不能共享 id 或输入（刷新的评测集必须是新样本）。

`rsi/rows.py` 的 `overlap(rows_a, rows_b)` 返回 `ids`、`inputs`、`clusters` 三个重叠数。`inputs` 按输入指纹计，忽略大小写和空白差异，所以换 id、改大小写、改空格的同一输入仍会被发现。

## 留在别处的大文件

`external` 记录不放进目录的大文件。`location` 为 `/绝对路径` 时视为本地：文件必须存在且 sha256 一致；为 `主机:/绝对路径` 时视为远端，不检查。
