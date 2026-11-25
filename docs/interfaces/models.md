# 模型类产物：candidate、search

两类产物都由训练调优模块 train 产出（一次派出是一次搜索：先交回各次试验的 candidate，登记后在同一会话交回比较它们的 search；旧工作区里 `producer` 为 `tune` 的候选和 search 照常读），目录里有 `manifest.json`，公共字段见 README.md。校验由 `rsi/artifacts.py` 的 `validate_artifact` 完成（额外字段之外不允许出现未列出的字段）。预测行格式见 README.md 的「决策样本行」。这里的 id 只检查是字符串；它们是否已登记、预测是否覆盖 valset，在登记时核对。

## `candidate`

训练得到的一个候选模型。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"candidate"` | 产物类别 |
| `trainset_id` | string | 训练用的 trainset |
| `valset_id` | string | 评估用的 valset |
| `config` | object | 完整解析后的训练配置：训练代码实际用到的每一个键都写出来，包括没改、取默认值的键，不只写改动的部分 |
| `config_source` | string | `"default"`（训练命令的默认配置）、或某个已登记 candidate 或 search 的 id（以它的配置为起点）；旧工作区的 `"tune"` 表示调优试验；登记时不核对 |
| `config_request` | object，可选 | 这次训练所要求的配置，按配置路径写：`{"base", "recipe", "overrides", "seed", "max_steps"}`，五个键都要写，没指定的写 null；`overrides` 是 `{"onejev": {<flag>: 值}, "lora": {<field>: 值}}`（也可以写成 `{"onejev.lr": 值}` 这样的平铺键）。任务有 `search_space` 时必填，登记时核对它落在配置空间里（规则见 task.md 的 `search_space`） |
| `code_version` | object | 训出这个候选的代码版本，见下 |
| `model` | object | 模型位置和预测入口，见下 |
| `metrics_val` | object | 指标名 → number，验证集上的成绩，必须包含任务的主指标 |
| `baselines_val` | object | 对照组名 → {指标名 → number}，对照组在验证集上的成绩 |
| `train_gate` | object | 训练门结果，见下 |
| `cost` | object | 成本，见下 |

`model` 的子字段：`location`（模型文件所在位置，给人看）；`sha256`（string 或 null）；`load`（string，给人看的加载说明）；`predict`（string，预测入口，见下）。`train_gate` 的子字段：`passed`（bool）；`detail`（string）。`cost` 的子字段：`compute_hours`（number）。

`code_version` 的子字段：`entry`（string，训练代码的名字，例如训练命令的路径：默认 `<repo root>/modules/train/train`，任务定义的 `resources.train_command` 给了就是它，登记时核对：同一字符串，或相对路径按仓库根目录解析、符号链接解开后是同一个文件）；`sha256`（string 或 null，`entry` 这个文件的 sha256）。`entry` 是运行校验或登记的那台机器上存在的文件的绝对路径时，会算它的 sha256 并与 `sha256` 比较（十六进制不分大小写），不一致或 `sha256` 为 null 都拒绝；`主机:/路径`、相对路径、名字或本机上不存在的路径不核对。训练在别的主机上跑时，`entry` 写成 `主机:/路径`：否则登记机器的同一路径上若有另一份代码（另一个 checkout），诚实的候选也会被拒。

只有 `entry` 这一个文件被算 sha256，它 import 或调用的其他代码、读取的默认配置文件都不在内。所以训练代码或它的默认值一改，`entry` 文件本身就必须跟着变：要么训练命令是一个自包含的文件（代码和默认值都写在里面），要么在这个文件里嵌入一个随代码变化的版本串（例如 git commit 或其他代码文件的哈希），由负责人在改代码时一起更新。做到这一点，`code_version.sha256` 相同才说明训练代码相同；再加上 `config` 是完整解析后的配置，「默认配置」在不同候选之间是否是同一个，看这两项即可。

**预测入口 `model.predict`。** 候选必须能直接运行，下游（交付）不用按文字说明重新实现打分。`model.predict` 是候选目录下一个可执行文件的相对路径，写法上不能跳出候选目录（可以是指向别处的符号链接，登记时复制它指向的文件），文件存在且有可执行权限。调用方式固定：

```
<候选目录>/<model.predict> --inputs <inputs.jsonl> --out <predictions.jsonl>
```

- `--inputs`：输入行 JSONL，即 `inputs.jsonl` 那样的行（OneJev 行去掉 `target`、`meta` 只留 `state_id`、`cluster`、`source`、`n_obs`：`{"id", "state", "question", "meta", "split"}`，见 README.md 的「决策样本行」），入口只读 `id`、`state` 和 `question`，不靠别的字段；行里没有 `target`。
- `--out`：写预测行 JSONL，每个输入行一行 `{"id", "probs", "decision", "uncertainty"}`，与 `predictions_val` 的格式相同。
- 成功时退出码 0，失败时非 0。
- 入口要读的模型文件放在候选目录里；放不下的大文件只能通过公共字段 `external` 的条目引用（带 sha256），不能依赖训练机器上别的路径。

候选自己的 `predictions_val` 就应当是用这个入口在验证集的 `inputs.jsonl` 上跑出来的结果；交付时 deliver 用打包后的接口再跑一遍，登记时核对两者一致（见 delivery.md）。

必备文件：`predictions_val`。目录内检查：预测行合法（字段、重复 id、标签、概率）；`metrics_val` 含主指标；`model.predict` 是候选目录内的可执行文件；`code_version` 的本地文件 sha256 一致；任务给了 `resources.train_command` 时 `code_version.entry` 是它；任务有 `search_space` 时 `config_request` 在空间里（这两项 `rsi validate candidate --task` 也查，错误信息写明键）。id 覆盖在登记时核对。

**登记时重算。** 指标和对照组的定义见 task.md 的「指标的定义」。train 用 `rsi score --task <task.json> --labels <valset>/rows.jsonl --predictions <candidate>/predictions_val.jsonl --train <trainset>/rows.jsonl` 算出这些数，原样写进 manifest。登记时代码按同样的定义重算下面三项，差超过 1e-9 就拒绝，错误信息给出字段、报告值、重算值和该用的 `rsi score` 命令：

- `metrics_val` 里每个已知指标：`predictions_val` 对当前验证集的行。
- `baselines_val` 里每个已知基线的每个已知指标：基线用候选的 `trainset_id` 所指训练集的行算，在当前验证集上打分。
- `train_gate.passed`：等于「主指标严格优于任务列出的每个基线」（方向按 `metrics.higher_is_better`，相等不算优于），用重算的数判断，与 `baselines_val` 里写了哪些基线无关。任务列出的基线里有 `rsi` 不认识的时，只按已知基线核对：已知基线没被超过却写 `passed: true` 会被拒绝，`passed: false` 则接受（可能是没超过那个未知基线）。主指标是未知指标或没有已知基线时不核对。

`train_gate.detail` 是给人看的说明，不核对。不认识的指标名和基线名跳过，`rsi register` 在输出行末尾加 `(not verified: <名字>)`。

`train_gate.passed` 为 false 的候选可以登记，但不能送评：登记引用它的 `delivery` 会被拒绝。`rsi register` 在 id 后面打印训练门结果和验证集主指标，例如 `candidate-0003 (train gate: passed; val auc 0.81)`。

## `search`

一次配置搜索：训练调优模块一次派出（或一次退回）的结果，也是这一轮的一次尝试（每轮的次数上限见 task.md）。无必备文件。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"search"` | 产物类别 |
| `trainset_id` | string | 搜索用的 trainset |
| `valset_id` | string | 搜索用的 valset |
| `strategy` | string | 搜索策略 |
| `budget` | object | 预算，见下 |
| `trials` | 列表 | 每次试验，见下 |
| `default_candidate` | string | 默认配置对应的候选 id |
| `best_candidate` | string | 最优候选 id |
| `gain_vs_default` | object | 相对默认配置的提升，见下 |

`budget` 的子字段：`compute_hours`。`trials` 每项：`candidate_id`（string，或 null）、`config`、`metric_val`（验证集上的主指标，number 或 null；取自该试验候选的 `metrics_val`，即训练命令用 `rsi score` 算出的数）、`compute_hours`、`status`。`status` 为 `ok` 表示试验成功，此时 `candidate_id` 和 `metric_val` 都必须给出；其他状态（例如 `failed`、`timeout`）的试验 `candidate_id` 和 `metric_val` 可以为 null。`gain_vs_default` 的子字段：`metric`、`delta`。

目录内检查：

- `best_candidate` 和 `default_candidate` 都是 `status` 为 `ok` 的试验的 `candidate_id`；非 null 的 `candidate_id` 不重复；`trials` 的 `compute_hours` 之和不超过 `budget.compute_hours`。
- 调优门：`best_candidate` 的 `metric_val` 不差于 `default_candidate` 的（方向按任务定义的 `metrics.higher_is_better`，容差 1e-9）；做不到就把默认配置的候选作为 `best_candidate` 交回。
- `gain_vs_default.metric` 等于主指标，`gain_vs_default.delta` 等于最佳减默认的 `metric_val`（容差 1e-9），越小越好时为负数表示变好。

登记时另外核对（`status` 不是 `ok` 的试验都不核对）：

- `trainset_id`、`valset_id` 已登记，`status` 为 `ok` 的试验的 `candidate_id` 都已登记为 candidate。
- 每个 ok 试验的 `metric_val` 等于它的候选登记时的 `metrics_val[主指标]`（容差 1e-9；主指标是已知指标时，那个数在候选登记时已重算核对过），所以调优门建立在核对过的数上。
- 所有 ok 试验的候选（默认候选也在内）的 `code_version.sha256` 相同（十六进制不分大小写），即一次搜索只比较同一版训练代码下的不同配置。有不同的 sha256 时拒绝，错误信息按 sha256 列出各自的候选；有候选的 sha256 为 null 时也拒绝，因为 null 说明不了是同一版代码。训练命令或默认配置要改，放到两轮之间改，不在搜索进行中改。

`rsi register` 在 id 后面打印增益，例如 `search-0001 (tune gate: passed; gain vs default auc +0.05 (best candidate-0002, default candidate-0001))`。
