# 评测与计划类产物：evaluation、plan

两类产物由 judge 产出，目录里有 `manifest.json`，公共字段见 README.md。校验由 `rsi/artifacts.py` 的 `validate_artifact` 完成（额外字段之外不允许出现未列出的字段）。id 只检查是字符串，是否已登记、是否与 STATE 一致、评测集使用次数，在登记时核对。「主指标」指任务定义里的 `metrics.primary`。

## 归因与模块对应

`failures` 和 `issues` 里每一项都写一个归因和一个模块，两者必须按下表对应（`rsi/rules.py` 的 `MODULE_OF`，由 `check_attribution` 检查）。归因不在表中也报错。

| 归因 | 模块 | 含义 |
|---|---|---|
| `data_coverage` | `data` | 训练数据没覆盖到出错的输入 |
| `label_noise` | `data` | 标签本身有噪声 |
| `label_definition` | `data` | 标签定义不合适或有歧义 |
| `model_capacity` | `train` | 模型能力不足 |
| `calibration` | `train` | 概率没有校准 |
| `config` | `train` | 超参数或训练配置不合适 |
| `runtime` | `deliver` | 交付包运行时的问题（延迟、成本、与离线不一致） |

## `evaluation`

对一个交付做评测，并给出是否升级的建议。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"evaluation"` | 产物类别 |
| `delivery_id` | string | 被评测的 delivery |
| `candidate_id` | string | 被评测的候选 |
| `evalset_id` | string | 使用的评测集 |
| `best_before` | null 或 object | 当前最佳版本，`{"version", "evaluation_id"}`；还没有最佳版本时为 null |
| `metrics` | object | `candidate`（指标 → 数值）、`best`（null 或指标 → 数值）、`baselines`（基线名 → 指标） |
| `deltas` | object | 指标 → `{"delta", "ci_low", "ci_high"}`，候选相对当前最佳的差值和置信区间 |
| `recommendation` | object | `decision`（`promote` 或 `keep`）、`reason` |
| `failures` | 列表 | 每项 `issue`、`evidence`、`attribution`、`module` |
| `uses_testset` | bool | 这次评测是否用了封存 test；为 true 时是里程碑评测，见下 |
| `metrics_test` | object，仅里程碑评测 | 指标名 → number，封存 test 上的成绩；`uses_testset` 为 true 时必填且含主指标，为 false 时不允许出现 |

必备文件：`report`（人读的评测报告）。目录内检查：

- `metrics.candidate` 含主指标。
- `best_before` 非 null 时（有当前最佳版本），`metrics.best` 非 null 且含主指标，`deltas` 含主指标；缺了就拒绝，升级规则不会因为缺数而悄悄得出 keep。
- `uses_testset` 为 true 时有 `metrics_test` 且含主指标；为 false 时没有 `metrics_test`。
- 每个 failure 的 `attribution` 与 `module` 一致。

**登记时重算。** 指标、对照组、差值和区间的定义见 task.md 的「指标的定义」。judge 用 `rsi score` 算出这些数，原样写进 manifest：

- `metrics.candidate`、`metrics.baselines`、`deltas[主指标]`：`rsi score --task <task.json> --labels <evalset>/labels.jsonl --predictions <delivery>/predictions_eval.jsonl --train <trainset>/rows.jsonl --clusters <evalset>/inputs.jsonl --compare <best-delivery>/predictions_eval.jsonl`，打印的 `metrics`、`baselines`、`delta` 分别对应这三处（`<trainset>` 是被评测候选的 `trainset_id` 所指的训练集；没有最佳版本时不加 `--compare`）。
- `metrics.best`：同样的命令，`--predictions` 换成最佳版本的 `predictions_eval.jsonl`。

登记时代码按同样的定义重算下面几项，差超过 1e-9 就拒绝，错误信息给出字段、报告值、重算值和该用的 `rsi score` 命令：

- `metrics.candidate` 里每个已知指标：`delivery_id` 的 `predictions_eval` 对 `evalset_id` 的标签。
- `metrics.baselines` 里每个已知基线的每个已知指标：基线用被评测候选的 `trainset_id` 所指训练集的行算，分组取自评测集的 `inputs.jsonl`。
- `best_before` 非 null 时，`metrics.best` 里每个已知指标，以及 `deltas[主指标]` 的 `delta`、`ci_low`、`ci_high`（任务的 `metrics.ci` 设置，簇取自评测集的 `inputs.jsonl`）。最佳版本一侧的预测是：最佳版本自己的 delivery，当它的 `evalset_id` 与这份评测相同；否则是这份评测 `parents` 里列出的、候选是最佳版本的候选、评测集是当前评测集的 delivery。两者都没有时拒绝登记，要先把最佳版本在当前评测集上打包交付（见下）。

`deltas` 里主指标以外的项、`failures`、报告文件不核对。不认识的指标名和基线名跳过，`rsi register` 在输出行末尾加 `(not verified: <名字>)`；主指标是未知指标时 `deltas` 不核对。

登记时另外核对：`recommendation.decision` 必须等于升级规则（`rsi/rules.py` 的 `decide_promotion`，见 task.md 的 `promotion_rule`）对这份评测得出的结论，不一致就拒绝登记，错误信息给出规则的结论和理由。规则读的数就是上面核对过的数，所以升级判定建立在代码核对过的数上。每登记一份（里程碑评测除外），当前评测集的使用次数 `eval_use_count` 加 1；已到 `eval_use_limit` 时拒绝登记，要先刷新评测集。

评测集刷新后，最佳版本原来那份 evaluation 和 delivery 都在旧评测集上。在新评测集上评测时，`best_before` 仍写最佳版本和它原来的 evaluation；`metrics.best` 和 `deltas` 用最佳版本的候选在当前评测集上的新 delivery 算，并把这份 delivery 的 id 写进公共字段 `parents`，登记时据此核对（没有这份 delivery 就拒绝登记）。已知限制：停止条件的 `target` 仍按最佳版本原来那份 evaluation 的主指标判断（旧评测集上的成绩）；按评测集分别记录成绩留待以后。

### 里程碑评测

`uses_testset` 为 true 的 evaluation 是里程碑评测：运行将要停止时，用封存 test 验证当前最佳版本，它不做升级判定。登记时核对：

- 当前已有最佳版本，`candidate_id` 是当前最佳版本的候选，`best_before` 写当前最佳版本。
- `delivery_id` 所指的 delivery 有 `predictions_test`；本轮有一次未用过的 `use_testset` 审批（deliver 跑 test 预测那次审批已被那份 delivery 用掉，这里要另一次）。
- `recommendation.decision` 必须是 `keep`，不按升级规则核对。
- `metrics_test` 里每个已知指标按 task.md 的「指标的定义」重算：这份 delivery 的 `predictions_test` 对它的 `testset_id` 的标签，用 `rsi score --task <task.json> --labels <testset>/labels.jsonl --predictions <delivery>/predictions_test.jsonl` 算；不一致就拒绝登记。里程碑评测的 `metrics`、`deltas` 不核对。
- 不检查、也不增加评测集使用次数。

`metrics`、`deltas`、`recommendation` 仍按评测集写（候选就是最佳版本，差值通常为 0），test 上的成绩只写在 `metrics_test`。`rsi promote` 拒绝里程碑评测（milestone evaluations are not promotion decisions）；`rsi next` 检查本轮有没有未判定的评测时不算它。judge 不为它写 `plan`。

## `plan`

根据评测结果给出下一轮计划。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"plan"` | 产物类别 |
| `evaluation_id` | string 或 null | 计划所依据的 evaluation；只有本轮由 `rsi close-round` 关闭时可以为 null |
| `issues` | 列表 | 每项 `issue`、`evidence`、`attribution`、`module`、`change`、`expected` |
| `entry_module` | 枚举 `data`、`train`、`deliver`（旧计划里的 `tune` 照常读，等同 `train`） | 下一轮从哪个模块进入 |

`expected` 的子字段：`metric`（预期改善的指标）、`direction`（`up` 或 `down`）、`size`（预期幅度，number）。

无必备文件。目录内检查：每个 issue 的归因与模块一致；`entry_module` 必须是 `issues` 中出现的某个模块。

登记时另外核对 `evaluation_id`：非 null 时必须是已登记的 evaluation；为 null 时，本轮必须是由 `rsi close-round` 关闭的（没有可评测的候选，本轮判定的 `evaluation_id` 为 null），否则拒绝。关闭的轮次没有 evaluation，judge 根据本轮的候选（例如训练门没过的原因）、search 和实验记录写计划，格式不变。
