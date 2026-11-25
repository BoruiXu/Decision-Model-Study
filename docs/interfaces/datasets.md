# 数据类产物：trainset、valset、evalset、testset、corpus、pool

这六类数据产物都由 data 模块产出，目录里有 `manifest.json`，公共字段见 README.md。校验由 `rsi/artifacts.py` 的 `validate_artifact` 完成：先按公共字段加该类的额外字段校验 manifest（不允许出现未列出的字段），再检查 `files` 里每个文件存在、`rows` 与文件里非空行数一致、本地 `external` 的 sha256，最后做该类的目录内检查。trainset、valset、evalset、testset、pool、corpus 至少要有一行（rows 或 inputs 为空的拒绝）：空集合登记了也无从训练、评测。行格式见 README.md 的「决策样本行」：每一行是 OneJev 行（`id`、`state`、`question`、`target`、`meta`、`split`），`question` 必须等于任务 `decision` 得出的问题，`target` 按 OneJev 的规则检查；行校验在 `rsi/rows.py`。

## 谁能看到标签

引设计文档第 7 节：

| 集合 | 由谁产出 | 谁能看到标签 | 用途 | 是否变化 |
|---|---|---|---|---|
| `trainset` | data | train；judge 只用来算对照组 | 训练 | 每轮可以出新版本 |
| `valset` | data，第一轮 | data、train | 选 checkpoint、定阈值、比较配置、去重检查 | 第一轮后固定，刷新要用户批准 |
| `evalset` | data，第一轮，产出后交给 judge | 只有 judge | 升级判定 | 固定；使用次数有上限，到上限后刷新要用户批准 |
| `testset` | data，第一轮，产出后交给 judge | 只有 judge，且每次使用要用户批准（`use_testset`） | 里程碑发布时的最终验证 | 不变 |
| `corpus` | data，第一轮 `sets` 派出 | 只有 data：产出它的 `sets` 派出（续做时简报再给）和 `refresh` 派出 | 全部带标签行和写出它们的预处理脚本；四个集合和 pool 从它切出，refresh 从它抽新评测集 | 第一轮登记一次；再登记要用户批准 |
| `pool` | data，第一轮 | data（`sets`、`trainset` 派出） | 训练池：训练时段的全部行，之后各轮的训练集从它抽 | 第一轮登记一次；再登记要用户批准 |

evalset 和 testset 把输入和标签分成两个文件：deliver 只拿 `inputs.jsonl`（testset 的要获批），标签文件只给 judge；valset 另有一份不带标签的 `inputs.jsonl`，deliver 拿它来复现候选的验证集预测；train 从不拿 evalset 和 testset。judge 拿送评候选所用训练集的 `rows.jsonl`，只用来算评测集上的对照组（对照组由训练集的先验得出，见 task.md 的「指标的定义」）。

## `trainset`

训练集，完整行。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"trainset"` | 产物类别 |
| `label_method` | string | 标签是怎么得到的，例如规则、judge、人工 |
| `parent_trainset` | string 或 null | 上一版训练集的 id；第一版为 null |
| `changes` | string | 相对上一版改了什么 |
| `quality` | object | 数据质量检查结果 |

`quality` 的子字段：`passed`（bool）；`checks`（列表，每项 `name`、`value`、`threshold`、`passed`）。目录内检查：每行是合法的完整行，`split` 不以 `test` 开头（onejev-harness 的规则：去掉首尾空白、转成小写后以 `test` 开头的切分，它不拿来训练和验证）；`quality.passed` 必须等于全部 `checks[].passed` 的与。

`quality.passed` 为 false 的训练集照常登记，作为证据留在登记表里，但不会成为当前训练集（STATE 的 `current.trainset` 不变），candidate 也就不能引用它。`rsi register` 在 id 后面打印质量门结果，例如 `trainset-0004 (quality gate: failed (dup_rate); never current)`。

必备文件：`rows`（完整行 JSONL，需写 `rows` 行数）、`quality_report`（质量报告，格式由 data 模块自定）。

```
trainset-0003/
  manifest.json
  rows.jsonl
  quality_report.json
```

## `valset`

验证集，没有额外字段（`kind` 固定为 `"valset"`）。必备文件：`rows`（完整行）和 `inputs`（输入行，不带标签），两个都需写 `rows` 行数。`inputs.jsonl` 由 data 模块把 `rows.jsonl` 的每一行只留顶层键 `id`、`state`、`question`、`meta`、`split`（所以没有 `target`），`meta` 只留 `state_id`、`cluster`、`source`、`n_obs` 写出（`rsi/rows.py` 的 `input_row_of`），交给 deliver：deliver 用交付包在它上面跑一遍，登记时与候选的 `predictions_val` 逐行比较（见 delivery.md），deliver 看不到验证集的标签。目录内检查：完整行合法，`split` 不以 `test` 开头（同 trainset）；输入行合法（顶层不得有这五个以外的键，`target` 也不行；`meta` 里不得有这四个以外的键）；`inputs` 与 `rows` 的 id 集合相同，每个输入行恰好等于 `rows.jsonl` 里同一 id 的行经同样处理（只留五个顶层键，`meta` 只留四个键）后的结果，所以 `rows.jsonl` 的行和 `meta` 都可以多出别的字段。

```
valset-0001/
  manifest.json
  rows.jsonl
  inputs.jsonl
```

## `evalset` 与 `testset`

评测集和封存测试集，格式相同，没有额外字段（`kind` 分别固定为 `"evalset"`、`"testset"`）。必备文件：`inputs`（输入行：完整行只留 `id`、`state`、`question`、`meta`、`split`，`meta` 只留 `state_id`、`cluster`、`source`、`n_obs`）、`labels`（标签行：`{"id", "target", "meta": {"label_source"}}`，`meta` 只有 `label_source` 一个键），两个都需写 `rows` 行数。目录内检查：输入行、标签行各自合法（输入行顶层不得有这五个以外的键，`target` 也不行，`meta` 里不得有这四个以外的键；标签行顶层只有 `id`、`target`、`meta`，必须有 `meta.label_source`，`meta` 里不得有别的键）；两边 id 集合相同。

**输入行不带标签。** 输入行给 deliver 和候选的预测入口，`meta` 只留上面四个键，因为 `meta` 的其他字段可能由标签推出（例如 OneJev 导出里的一致率 `f`、`mode_score`、`mode_idx`；`label_source` 也是，标签怎么来的可能和标签相关，所以它只放在完整行和标签行里），登记时有别的键就拒绝。留下的字段同样不能带由标签推出的值。改编 OneJev 导出时，写输入行要去掉这类字段。

```
evalset-0002/
  manifest.json
  inputs.jsonl
  labels.jsonl
```

**交接：登记时移入工作区。** 其他类别的产物登记时是复制，源目录留在原处；evalset、testset 和 corpus（见下）登记时是移交：所有检查通过后，源目录被移进工作区（`<ws>/artifacts/<id>/`），原来的目录不再存在，产出它的模块的输出目录里也就不再有这份标签。登记被拒时什么都不动，源目录原样留着，改好后再登记。源目录移不走（例如所在目录没有写权限）时拒绝登记，同样什么都不改。登记表的 `source` 仍记原路径，之后在同一路径重新建目录也登记不了（同一源路径只登记一次）。目录里指向目录外的符号链接、硬链接数大于 1 的文件会让标签留在原处，校验和登记都拒绝，要把文件本身写进目录。登记中途被终止（在把源目录挪开之后、写入登记表之前）时，源目录在同级的隐藏目录 `.<名字>.rsi-handed-over-*` 里：手工把它改回原名，再登记一次。

所以 data 模块直接在简报给的输出目录里生成评测集、封存 test 和 corpus，不在别处留副本，也不保留中间文件里的标签；交回之后不再读取。已知限制：代码只能把源目录移走，同一台机器上的进程仍然读得到工作区里的标签文件，模块之间的隔离靠约定（简报不给模块工作区路径），代码不阻止。

## `corpus`

data agent 在第一轮 `sets` 派出里写的预处理脚本，和它从原始数据（任务的 `data.sources`，见 task.md）写出的全部带标签完整行。四个集合和 pool 都从这些行切出。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"corpus"` | 产物类别 |
| `command` | string，非空 | 运行脚本的命令，例如 `python3 preprocess.py`；登记时不运行，复跑审计由操作者按它做 |
| `sources` | string 列表 | 脚本读过的原始数据路径；任务的 `data.sources` 写了路径时不能为空，每一项必须等于其中某个 `path`，或在它之下 |
| `label_method` | string | 标签是怎么得到的 |

必备文件：`rows`（全部完整行 JSONL，需写 `rows` 行数）、`script`（预处理脚本）；可以另有 `split_report`（`dset` 写的切分汇总）。目录内检查：目录里不能有 `.dset-` 开头的条目（`dset` 中途停下留下的，删掉再重跑）；每行是合法的完整行（id 不重复，`question` 等于任务的问题），`split` 不查（corpus 不拿来训练，行可以保留原始数据里的切分名）；任务的切分规则完整时（task.md 的 `data.split`），每行都有规则的 `key`，`by: time` 时它的值是数（字符串、bool、null 都不行）；任务写了 `data.rows.expect` 时，行数和其中每个硬标签的行数逐个相等（硬标签见 task.md 的「指标的定义」）；`sources` 如上。每类问题报一条，写出行数和第一个 id（`rsi/datasplit.py` 的 `corpus_errors`）。脚本的 sha256 和其他文件一样，登记时记进登记表的 `files`。

corpus 和 evalset、testset 一样整个目录**移交**进工作区（见上面的「交接」），之后的派出不再读它（refresh 例外：新评测集只能从它抽）。登记后它是当前 corpus（STATE 的 `current.corpus`）；已有当前 corpus 时再登记一份要 `refresh_sets` 审批。corpus 只为在它之后登记的 valset、evalset、testset、pool 担保，它们都必须出自它（见「重叠检查」），所以第一份 corpus 要在这些集合之前登记：其中任何一类已有当前的就拒绝，错误信息让先登记 corpus（data 模块的交回本来就是这个顺序）。替换的 corpus（已有当前 corpus 时）不和已经是当前的集合比较。corpus 不参与集合之间的重叠比较（它是全集）。corpus 自己登记时整份读进内存（行检查）；之后登记集合时逐行读它，只留那个集合用到的行。

```
corpus-0001/
  manifest.json
  rows.jsonl
  preprocess.py
  split_report.json
```

## `pool`

训练池：训练时段（规则里 trainset 的范围）的全部带标签完整行，第一轮登记一次。之后各轮的训练集从它抽，data 的 `trainset` 派出拿 pool，不拿原始数据和 corpus。没有额外字段（`kind` 固定为 `"pool"`）。必备文件：`rows`（完整行，需写 `rows` 行数）。目录内检查同 trainset：每行是合法的完整行，`split` 不以 `test` 开头（它喂训练）。登记是复制；登记后它是当前 pool（STATE 的 `current.pool`），已有当前 pool 时再登记一份要 `refresh_sets` 审批。

```
pool-0001/
  manifest.json
  rows.jsonl
```

旧工作区的 STATE `current` 没有 `corpus`、`pool` 两个键，读成没有（null），登记照常。

## 重叠检查

行里的 `meta.state_id` 和 `meta.cluster` 由 data 模块按任务定：

- `meta.state_id`：行所属的分组（OneJev 的状态 id），`group_prior` 对照组按它分组，judge 按它给出分组结果；旧工作区的行用 `meta.group`，读的时候当作 `meta.state_id`。
- `meta.cluster`：近似重复簇的 id。同一模板、改写或近似重复的样本属于同一个簇；切分时整个簇放进同一个集合，同一个 cluster 不能出现在两个集合中。

行 `id` 在 trainset、valset、evalset、testset、pool 之间唯一（trainset 和 pool 除外：trainset 从 pool 抽）。这些集合之间不能有重复，这个检查在登记时做：登记一个 trainset、valset、evalset、testset 或 pool 时，它与其余几类的当前产物逐一比较（trainset 和 pool 之间不比；corpus 是全集，不参与），只要有相同的 id、相同的输入或相同的 cluster 就拒绝登记，错误信息给出三个重叠数。新的 evalset 还要和以前登记过的每个 evalset 比较，不能共享 id 或输入（刷新的评测集必须是新样本）。

**同一 `state_id`。** 同一个 `state_id`（例如同一个地址在不同时间步的行）可以出现在多个集合里，登记不拒绝（`dset` 报告各对集合共享的数目）。只有任务的切分规则写 `unit: state_id`（task.md 的 `data.split`）时，上面的每一次比较另比 `state_id`，有共享就拒绝，错误信息另给 `state_ids N`。

**出自 corpus。** 在 corpus 之后登记的 valset、evalset、testset、pool（登记时有当前 corpus），每一行都必须是 corpus 里同一 id 的行：输入指纹相同；`meta.state_id` 和 `meta.cluster` 相同（上面的重叠比较靠这两个字段）；带标签的行（valset、pool 的 `rows.jsonl`，evalset、testset 的 `labels.jsonl`）`target` 也相同（没写的标签按 0 算）。对不上就拒绝，每类问题一条，写行数、第一个 id 和 corpus 的 id。这样四个集合和 pool 都能追溯到登记过的预处理脚本；corpus 之前登记的集合不能，所以第一份 corpus 必须先于它们登记（见上面的 `corpus` 一节）。这项检查要读 corpus，放在最后，前面的检查都通过了才做。trainset 不查（之后各轮的训练集可以加入 feedback 写成的行）。任务写了原始数据（`data.sources` 有路径，或写了 `data.rows`）时，集合必须出自 corpus：还没有当前 corpus 时登记 valset、evalset、testset、pool 一律拒绝，错误信息让先登记 corpus（预处理脚本和全部带标签行）；`rsi register` 和引擎都一样。任务没写原始数据的（手工登记集合的、没有 `data` 一节的、旧工作区的字符串 `sources`）不要求 corpus，没有 corpus 时都不查。

`rsi/rows.py` 的 `overlap(rows_a, rows_b)` 返回 `ids`、`inputs`、`clusters` 三个重叠数。`inputs` 按输入指纹计（行的 `state`，旧工作区的行是 `input`），忽略大小写和空白差异，所以换 id、改大小写、改空格的同一输入仍会被发现。`shared_state_ids(rows_a, rows_b)` 返回两组行共享的 `state_id` 数（旧工作区的行按 `meta.group`）。

## 留在别处的大文件

`external` 记录不放进目录的大文件。`location` 为 `/绝对路径` 时视为本地：文件必须存在且 sha256 一致；为 `主机:/绝对路径` 时视为远端，不检查。
