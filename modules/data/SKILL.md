---
name: rsi-data
description: 由 rsi-orchestrator 派发数据模块时使用：产出训练集，第一轮或获批刷新时另出验证集、评测集和封存 test。Use when the orchestrator dispatches the RSI data module; not for general data requests.
---

# rsi-data：数据模块

获取或生成数据，检查并提高数据质量。只做本模块的事，不派别的 subagent；模块内部怎么做由负责人定，本 skill 只固定接口。

## 输入

简报由代码生成，放在输出目录里：`brief.md`（人读）和 `brief.json`（同样的内容，结构化：`module`、`purpose`、`round`、`task_json`、`ids`、`inputs`、`outdir`、`budget_left`、`goal` 等；字段说明见框架仓库根目录下的 `skills/rsi-orchestrator/templates/brief.md`）。输入以简报的 `inputs` 为准，只读那里列出的路径。被退回时，同一个会话会收到 `feedback-<k>.md`（`<k>` 是第几次启动），写明为什么退回和要改什么：按它改好，再重新写 `handback.json`。同一个会话还会收到两种不算退回的 `feedback-<k>.md`：开头是「# 继续：交回的第二部分」的（两段式交回的后半段，例如 train 先交回试验候选、judge 先交回评测，登记后在这里拿到登记 id，再写 search 或计划），和开头是「# 用户回答了你的问题」的（见「输出」）。

只读下面列出的产物，没列出的产物和目录不要找、不要读；不读工作区，不读登记表。

- `task`：任务定义（必有）。
- 第一轮产出各集合的派出（sets）：简报的 `sources` 列出原始数据（任务定义 `data.sources` 的路径，不在 `inputs` 里），只在这次派出读，之后的派出都禁读；续做时（corpus 已登记）还拿已登记的 `corpus` 和 `pool`，不改它们。
- 可选：上一版 `trainset`；`plan`（其中归因到数据的问题）；带真实结果的 `feedback`。
- 之后各轮产出训练集时拿训练池 `pool`（训练时段的全部带标签行），新训练集从它抽；不拿 corpus，也不读原始数据。
- 第一轮之后的各轮还拿 `valset`，用于去重检查（新训练集不能和它重叠）。
- 获批刷新评测集时（refresh）拿 `corpus`（新评测集从它抽）和以前各个 `evalset` 的 `inputs.jsonl`（不带标签，输入名 `evalset_inputs_<id>`），新评测集不能与它们重叠；不拿 `pool`、封存 test 和评测集的标签，也不读原始数据。
- 简报的 `dset`（各目的都有）：切分工具的命令前缀，用法见下面的步骤。
- 本模块以前各轮的 `record`（只给自己的）：以前试过什么、为什么这样改；改得好不好不在里面，见「模块指标」。

## 输出

- `trainset`：每次都出新版本，需带数据质量报告。
- 第一轮产出各集合时另出 `corpus` 和 `pool`：`corpus` 是预处理脚本和它写出的全部带标签完整行（manifest 写运行脚本的命令、读过的原始数据路径和标签来源），登记时整个目录移进工作区；`pool` 是训练时段的全部完整行，训练集从它抽。字段见 [datasets.md](../../docs/interfaces/datasets.md)。
- `task`：只在补全了切分规则时交回：`dset sets` 在输出目录写的任务新版本（task.json 的版本 +1、只多了补全的 data.split；TASK.md 末尾多一节「切分规则」）。引擎只登记补全切分规则的任务新版本，改了别的（哪怕有一份还没用掉的 task_change 审批）一律拒绝；它登记后，本次其他产物的 manifest 写新版本号（dset 已写好）。
- 第一轮或获批刷新时另出 `testset`、`evalset`、`valset`。验证集除了完整行的 rows.jsonl，还要写一份不带标签的 inputs.jsonl（rows.jsonl 的每一行只留 id、state、question、meta、split 五个顶层键，所以没有 target；meta 只留 state_id、cluster、source、n_obs），deliver 用它复现候选的验证集预测。评测集和封存 test 的 inputs.jsonl 同样是这样处理过的完整行，labels.jsonl 每行写 `{"id", "target", "meta": {"label_source"}}`（meta 只有 label_source）。输入行有别的顶层键、meta 有别的键都会被拒绝登记，标签行也只能有 id、target、meta 三个顶层键：多出的顶层键和 meta 的其他字段可能由标签推出（label_source 也算：标签怎么来的可能和标签相关，所以它只放在完整行和标签行），输入行里留下的任何字段也都不能带由标签推出的值。
- 每一行都是 OneJev 行：id、state（模型的输入，格式按任务定义的 input 一节）、question、target、meta（state_id、cluster、source、label_source、n_obs，可以有别的字段）、split。question 必须恰好等于由任务 decision 得出的问题（yes/no 的 binary 是 noul 问题，criteria 是 `{"true": yes 的含义, "false": no 的含义}`；choice 的 criteria 是 label_meanings；score 的 criteria 是各级含义的列表；instructions 照抄 decision.instructions），target 的键是任务的标签、和为 1（容差 1e-6）。外来的 OneJev 数据补齐 meta 字段、核对 question 后原样使用，不做别的转换，只是写输入行时去掉由标签推出的 meta 字段（例如 f、mode_score、mode_idx）和 id、state、question、meta、split 以外的顶层键；trainset、valset 的 split 不要以 test 开头（onejev-harness 不读这样的训练行和验证行，登记时就拒绝）。格式和对照表见 [README.md](../../docs/interfaces/README.md) 的「决策样本行」。
- `record`：每个改进实验一份。

每个产物写成输出目录下的一个子目录，manifest 的 id 写 null，round、task_id、task_version 按简报写，字段见 [datasets.md](../../docs/interfaces/datasets.md)。 数据产物的行格式和目录结构见该文档，`record` 见 [record.md](../../docs/interfaces/record.md)。

交回顺序：第一轮的 `artifacts` 依次写 `task`（只在补全了切分规则时）、`corpus`、`testset`、`evalset`、`valset`、`pool`、`trainset`、`record`，引擎也按这个顺序登记：先登记任务新版本，后面的 manifest 都写新版本号；corpus 先于各集合，各集合和 pool 登记时核对每一行都出自 corpus；evalset、testset、corpus 登记时整个目录移走，输出目录里不再留带标签的副本。其中一个被拒时，排在它后面的都不登记（记为被拒），按反馈改好后把没登记的一起重新交回。corpus 已登记（续做）时不再交 corpus，`dset sets` 用简报给的那份，不改它。

交回：全部产物写好、校验通过后，在输出目录写 `handback.json`：`status`（`ok`、`failed` 或 `long_job`）、`artifacts`（输出目录下的子目录名，每个产物一个）、`marker`（长任务完成标记的绝对路径，`long_job` 时必填，其他写 null）、`notes`（门数字、花费、遇到的问题；`failed` 时写原因）、`questions`（需要用户决定的事，没有写 `[]`）。引擎读到它才算交回，缺了或格式不对按交回失败处理，重新写好即可。读取很严格：不能有上面没列出的字段，`artifacts` 只能是子目录的名字（不含路径分隔符，不重复），`questions` 的每一项必须是非空字符串，`marker` 必须是绝对路径或 null，`failed` 时 `notes` 不能为空。

交回带 `questions` 时，这次交回的产物一个都不登记，引擎先问用户；用户回答后，回答送回同一个会话（`feedback-<k>.md` 开头是「# 用户回答了你的问题」，不算退回）。按回答修改或确认产物，再重新写 `handback.json`，列出要登记的全部子目录（上次列过的也要再列）。长任务（`long_job`）交回里的 `questions` 等完成标记出现后才问；它没列产物时，回答写在开头是「# 长任务已结束」的反馈里，和长任务的结果一起送回。

## 门

- 数据质量门：任务定义 `quality_checks` 规定的检查和阈值（样本量、标签分布、标签信度、近似重复等）逐项算出数值、阈值、是否通过，写进 `trainset` 的 `quality`，`quality.passed` 等于各项相与。`dset` 自己算其中四项：`min_rows`、`min_positive_rows`（binary）、`max_duplicate_rate`（输入和前面某行相同的行占比）、`max_conflict_rate`（所在输入有两个以上硬标签的行占比），连同数据概况写进 manifest 的 `quality` 和 `quality_report.json`。任务里别的检查项由你算好，写成 JSON 文件 `[{"name", "value", "threshold", "passed"}]`，用 `--checks` 交给 `dset` 一起写（任务列了而两边都没算的，`dset` 报错）；任务定义没写 `quality_checks` 时自定检查项，同样用 `--checks` 交，并在 `record` 里说明。第一轮切分时 `dset` 先算门再写，没过就什么都不写（见步骤第 6 步）；之后各轮的 grow、check 门没过也照写：如实交回，由引擎按决策点处理（默认带着原因回到你的会话重做）；这样的训练集登记后只作为证据，不会成为当前训练集，train 不会用它。
- 集合之间没有重复：`dset` 按 testset、evalset、valset、trainset 的顺序抽，前面集合抽中的行占用的 id、输入、cluster 不进后面的集合；pool 是 trainset 范围里剩下的全部行，同样不和这三个集合共享 id、输入、cluster。登记时引擎再查一遍，各集合和 pool 的每一行还必须出自登记的 corpus。同一 `state_id`（同一个对象，例如同一个地址在不同时间的行）出现在多个集合里是允许的：`dset` 不拒绝，只报告数目（汇总和训练集 `quality_report.json` 里的 `shared_state_ids`），你把数目写进 `record`；只有规则写 `unit: state_id` 时才按 `state_id` 分开，登记时也按它查。之后各轮的新训练集同样不能和验证集、评测集、封存 test 共享 id、输入、cluster（登记时拒绝）。
- 切分规则：任务定义 `data.split` 写了的键原样用；没写的由你在第一轮 sets 派出里按数据定（`by`、`key`、`ranges` 或 `fractions`、`sizes`、`stratify`、`unit`、`seed`），经 `dset` 写成任务定义的新版本（补全），不需要审批，也不问用户；为什么这样定写进 `record`。评测集或封存 test 登记之后，规则在本次运行里固定：refresh 沿用它，之后的派出改不了它（改了 `data.split` 的任务版本一律被拒）。
- 带标签的集合交出之后不保留访问：评测集、封存 test 和 corpus 直接在输出目录里生成，不在别处留副本或带标签的中间文件，也不用符号链接或硬链接引用别处的文件（校验和登记会拒绝）；登记时这三个目录整个移进工作区，输出目录里就不再有它们。交回之后，你不留 corpus 的副本，也不留评测集、封存 test 标签的副本，不再读它们。之后的派出不再给评测集和封存 test 的标签：获批刷新评测集的派出（refresh）拿 corpus（新评测集只能从它抽）和以前各评测集的 `inputs.jsonl`（不带标签），用来让新评测集避开用过的行。corpus 里是全部行和它们的标签，封存 test 和以前各评测集的行也在内：refresh 只通过 `dset refresh` 用它，不自己打开、不看封存 test 和以前评测集的行；标签只在产出它们的 sets、refresh 派出里经手。已知限制：代码只移走目录，同一台机器上仍读得到工作区里的标签，隔离靠约定。
- 原始数据：只有第一轮的 sets 派出读简报 `sources` 列出的原始数据（任务的 `data.sources`）；之后的派出（trainset、refresh）和其他模块一律禁读：产出训练集用 `pool`，刷新评测集用 `corpus`。corpus 登记后操作者会把原始数据设成读不了（`chmod 000`）。
- 本模块只产出数据，不训练，也不自己验证新训练集的好坏：新训练集登记后，本轮的 ② 训练调优在它上面训练、产出候选，④ 评测与决策在评测集上把送评候选和当前最佳版本（旧训练集训出的）比较，升级还是保留就是这个判断（见「模块指标」）。本模块的门是上面的质量门和去重检查。
- 需要刷新验证集、评测集，或要超预算时，停下报告，不自己决定。

## 单独运行

负责人在自己的沙盒目录里运行，不需要顶层：

1. 把 上一版 `trainset`、`valset`、`pool`、`feedback` 等输入 从已登记产物复制到沙盒，路径手工指定；任务定义用 `<task.json>`。第一轮的切分读任务 `data.sources` 写的原始数据。不需要工作区，也不读登记表。
2. 输出写到沙盒下自己建的输出目录，每个产物一个子目录，manifest 的 `id` 写 null，`round` 自拟。
3. `dset` 照「步骤和策略子 skill」运行，前缀是 `PYTHONPATH=<repo root>/modules/data python3 -m dset`，`<outdir>` 是上面的输出目录。
4. 每个产物用 `PYTHONPATH=<repo root> python3 -m rsi validate <kind> <dir> --task <task.json>` 校验（切分写了 `task/` 时用新的 `task.json`），通过才算交出。可以运行的 `rsi` 命令只有它和 `rsi score`（算报告里的指标、对照组和差值），不运行 `rsi register` 等其他命令。
5. 不训练。想知道新训练集好不好（由负责人做，不是本模块的步骤），把它交给训练调优模块训练（它的「单独运行」），再按评测模块的做法和当前最佳版本比较。

## 步骤和策略子 skill

### 命令和路径

- `<outdir>`：简报的输出目录。`<task.json>`：简报的 `task_json`。`<round>`：简报的 `round`。`<trainset>`、`<valset>`、`<pool>`、`<corpus>`：简报 `inputs` 里同名的已登记产物目录；`<trainset_id>`、`<valset_id>`：简报 `ids` 里的 id。
- 下文的 `dset <命令> …` 都指简报的 `dset` 前缀（`PYTHONPATH=<repo root>/modules/data python3 -m dset`）加上命令，在哪个目录下运行都可以。`dset --help` 列出四个命令（sets、grow、check、refresh），每个命令加 `--help` 看参数。每个命令在 stdout 打一个 JSON 汇总；退出码 0 是成功，1 是输入或规则有错（stderr 一行 `dset: error: …`，什么都不写），2 是训练集质量门没过（只有 sets，什么都不写）。要写的目录必须还不存在，`dset` 不覆盖任何东西；加 `--dry-run` 只打印汇总，什么都不写。
- 分工：原始数据到 OneJev 行的预处理由你写脚本；切分和抽样一律由 `dset` 按规则做，不自己写切分代码。同样的行和规则切出同样的集合，汇总可以事后核对，refresh 也靠它重算第一轮的分配。

### 第一轮：sets

1. **读任务**：任务定义的 `decision`、`input`、`data`（`sources`、`label_sources`、`rows`、`split`）和 `quality_checks`。`data.rows.about` 说一行是什么、各字段从哪来；`data.rows.expect` 是全部行的确切行数和各硬标签的行数，corpus 登记时逐个核对。
2. **看原始数据**：只读简报 `sources` 列出的路径（不在 `inputs` 里），看格式、行数、标签分布、能做切分的字段（时间、分组）。别处的数据不读。
3. **写预处理脚本**：在 `<outdir>/corpus/` 写一个脚本，例如 `preprocess.py`（单个文件，只用标准库）。它只读 `sources`（绝对路径写在脚本里），在 corpus 目录里运行，输出写相对路径 `rows.jsonl`：每一行是完整的 OneJev 行（格式见「输出」），`question` 是 `rsi.rows.question_of(task)` 的值（先用 `PYTHONPATH=<repo root> python3 -c 'import json, sys; from rsi.rows import question_of; print(json.dumps(question_of(json.load(open(sys.argv[1])))))' <task.json>` 算出，作为常量写进脚本，复跑时不依赖任务文件和框架），`meta` 有五个框架键（`state_id`、`cluster`、`source`、`label_source`、`n_obs`；相同或近似重复的输入写同一个 `cluster`，`dset` 和登记都按 cluster 防止跨集合重复），切分要用的字段（例如时间步）放进 `meta.extra`，由标签推出的上游字段不写出；照 `data.rows.about` 写。输出必须确定：同样的输入写出同样的字节（按 id 排序，不写时间，不用没固定种子的随机数），操作者会按 manifest 的 `command` 复跑、核对 sha256。运行 `cd <outdir>/corpus && python3 preprocess.py`；预计超过 10 分钟就按长任务运行（见「长任务」）。corpus 目录里只放脚本和 `rows.jsonl`（`dset` 再加 `manifest.json` 和 `split_report.json`），日志写在它外面。
4. **核对行**：`dset check --task <task.json> --rows <outdir>/corpus/rows.jsonl`，看行数、各标签行数、重复输入和冲突（同一输入有两个以上硬标签）。行数和各标签行数必须等于 `data.rows.expect`（对不上 corpus 登记会被拒）：对不上就改脚本重跑，不改任务。
5. **定切分规则**：任务 `data.split` 写了的键照用（`ranges`、`fractions`、`sizes` 按集合逐个看，冲突时 `dset` 报错）；没写的由你按数据定，不问用户：
   - `by`：有时间字段、模型要用在以后的数据上时用 `time`：`key` 是时间字段，`ranges` 按时间顺序 trainset 最早，然后 valset、evalset、testset（各一个闭区间，互不重叠，区间外的行不用）。有自然分组、同一组不该跨集合时用 `group`：`key` 是分组字段，按 `fractions` 整组随机分。都没有时用 `random`：按 `fractions` 以 cluster 为单位随机分。
   - `sizes`：没写的按数据定。每个集合范围里去掉被前面集合占用的行之后要够它的大小；评测集的范围要给以后的 refresh 留出没用过的行（`sizes.evalset` 不写就把范围取完，以后刷新不了）；每个集合都要有足够的正例（binary），主指标才算得稳。
   - `stratify`（默认 true：time 按（key 值，硬标签）分层，其余按硬标签）；`unit`（默认 `cluster`。同一 `state_id` 跨集合是允许的，只报告；要模拟完全没见过的对象时才用 `state_id`）；`seed`（默认 0）。
   把你定的部分写进 `<outdir>/split.json`（可以只写任务没写的键；写了任务的键就要和任务的一样），先加 `--dry-run` 试：

   ```
   dset sets --task <task.json> --corpus <outdir>/corpus --script preprocess.py --command "python3 preprocess.py" --label-method "<标签怎么来的>" --split <outdir>/split.json --round <round> --out <outdir> --dry-run
   ```

   看汇总再定：各集合的 `rows`、`label_counts`、`key_span`、`range_rows`（范围里的行数）、`left_out`（因前面集合占用而排除的行数），`pool`、`unused_rows`、`shared_state_ids`（两两共享的 `state_id` 数）和 `gate`（质量门）。不满意就改 `split.json` 再试。
6. **切分**：去掉 `--dry-run` 再运行同一条命令。它写 `<outdir>` 下的 `testset`、`evalset`、`valset`、`pool`、`trainset`，在 corpus 目录里写 `manifest.json`（`command`、`sources`、`label_method` 和脚本、行文件）和 `split_report.json`（完整汇总，含各集合标签分布）。任务的规则不完整时还写 `<outdir>/task/`：任务定义的新版本（`task.json` 版本 +1、只多了补全的 `data.split`；`TASK.md` 末尾多一节「切分规则」），这次写出的所有 manifest 都写新版本号。这些文件不手改。
   - 退出码 2（质量门没过，什么都没写）：调整你定的部分（范围、大小、分层）再试；是任务作者写死的部分（例如 `sizes`、`quality_checks` 的阈值）造成的，交回 `questions`，不改它们。
   - 退出码 1：照 `dset: error:` 后面的原因改。说某个集合的 sizes 比可用的行多：减小你定的大小或放宽范围；是作者写的大小，交回 `questions`。
   - 任务 `quality_checks` 里 `dset` 不算的检查：算好写进 `<outdir>/checks.json`，命令加 `--checks <outdir>/checks.json`。
   - 交回后有产物被拒、被退回，要接着做：见下面的「续做」。
7. **校验**：先校验 `task/`（有的话），其余产物用新的 `task.json` 校验（没有 `task/` 时用 `<task.json>`）：

   ```
   PYTHONPATH=<repo root> python3 -m rsi validate task <outdir>/task
   PYTHONPATH=<repo root> python3 -m rsi validate <kind> <outdir>/<kind> --task <outdir>/task/task.json
   ```

   `<kind>` 依次是 corpus、testset、evalset、valset、pool、trainset。续做时只校验要交回的产物：corpus 已登记就不再校验它。
8. **写 record**（见「实验记录」）：脚本做什么、行数核对、规则和为什么这样切、`dset` 汇总。
9. **交回**：删掉临时文件（试跑留下的输出、带标签的中间文件：corpus 和各集合目录以外不留带标签的行），写 `handback.json`（`notes` 不写评测集和封存 test 的标签分布）：`artifacts` 依次是 `task`（只在写了 `task/` 时）、`corpus`、`testset`、`evalset`、`valset`、`pool`、`trainset`、`record`，引擎也按这个顺序登记。corpus、evalset、testset 登记时整个目录移进工作区；交回之后不留它们的副本，也不再读。

**续做**（交回有产物被拒、被退回，或回答问题后，同一个会话接着做）：反馈和 `registered-<n>.json` 写着哪些产物已经登记；重新生成的简报里 `task_json` 是当前的任务版本（补全的规则已登记时就是新版本）。`dset` 不覆盖任何目录，所以按 corpus 登记了没有分两种情况：
- corpus 还没登记（它之后的产物也都没登记）：按反馈改好脚本或行（改了就重跑脚本，再做第 4 步），删掉 `<outdir>` 下的 `task/`、五个集合目录，以及 corpus 目录里 `dset` 写的 `manifest.json` 和 `split_report.json`（脚本和 `rows.jsonl` 留着），再从第 6 步重新切（`--task` 用简报现在的 `task_json`）。
- corpus 已登记（简报 `inputs` 里有 `corpus`；pool 也登记了时还有 `pool`）：已登记的 corpus 一律不碰。在一个新目录里重切：`dset sets --task <task.json> --corpus <corpus> --split <outdir>/split.json --round <round> --out <outdir>/redo`（`redo/` 已存在就先删掉；`dset` 只读 corpus 的 manifest，`--script`、`--command`、`--label-method` 不用给）。同样的 corpus 和规则切出同样的集合。对每个还没登记的产物（testset、evalset、valset、pool、trainset，和 `redo/` 里有而还没登记的 `task`），删掉 `<outdir>` 下的旧目录，把 `redo/` 里同名的目录移上来；然后 `rm -rf <outdir>/redo`：里面有重抽的评测集、封存 test 的带标签副本。只交回还没登记的产物：corpus 不再交；只登记了 corpus 时，pool 和各集合都要交；pool 也登记了时 pool 不再交。
然后照第 7–9 步校验（只校验要交回的产物）、写 record、交回（只列还没登记的产物，顺序不变）。第一次交回的 `record` 不管别的产物是否被拒都已登记（复制，输出目录里的还在）：新的 record 写进一个还没登记过的子目录名，例如 `<outdir>/record-2`，`artifacts` 里列 `record-2`；改写 `<outdir>/record` 不会再登记。

### 之后各轮：trainset

按计划里归因到数据的问题选做法（可以组合），新训练集写到 `<outdir>/trainset`：

- 覆盖不足（某类样本太少、训练集太小）：从训练池再抽，`dset grow --task <task.json> --pool <pool> --trainset <trainset> --parent <trainset_id> --add <N> --round <round> --out <outdir>/trainset`。抽的是 pool 里 id 和输入都不在当前训练集里的行，按规则分层。
- 标签噪声（同一输入几个标签、重复行）：先看 `dset check --task <task.json> --rows <trainset>/rows.jsonl --against <valset>/rows.jsonl`（重复、冲突和与验证集的重叠），再写清理后的训练集 `dset check --task <task.json> --rows <trainset>/rows.jsonl --write <outdir>/trainset --mode merge --round <round> --parent <trainset_id>`：merge 把同一输入的行合成一行，target 按 `n_obs` 加权平均；`--mode drop` 删掉冲突输入的全部行、重复输入只留第一行。
- 有带真实结果的 `feedback`：把结果写成完整行（`meta.label_source` 是 `outcome`，任务的 `data.label_sources` 要允许它；`question` 同第一轮；id 不和已有的重复），放进 `<outdir>/feedback_rows.jsonl`，`dset grow --task <task.json> --pool <pool> --trainset <trainset> --parent <trainset_id> --with <outdir>/feedback_rows.jsonl --round <round> --out <outdir>/trainset`。

`dset grow` 和带 `--write` 的 `dset check` 写新训练集时算好质量门，写进 manifest 的 `quality` 和 `quality_report.json`；任务里 `dset` 不算的检查项算好写进 `<outdir>/checks.json`，命令加 `--checks <outdir>/checks.json`。门没过也照写、如实交回（见「门」）。然后用 `PYTHONPATH=<repo root> python3 -m rsi validate trainset <outdir>/trainset --task <task.json>` 校验，写 `record`（见「实验记录」），交回 `trainset`、`record`。不训练：新训练集的好坏由本轮的训练调优和评测判断（见「模块指标」）。

### 刷新评测集：refresh

获批刷新时，新评测集从 corpus 里抽，避开用过的行。corpus 里有全部行的标签（封存 test 的也在内）：只通过下面这条命令用它，不自己打开、不看封存 test 和以前评测集的行。`dset refresh --task <task.json> --corpus <corpus> --out <outdir>/evalset --round <round> --exclude <evalset_inputs> --exclude <trainset>/rows.jsonl --exclude <valset>/rows.jsonl`。`--exclude` 给简报 `inputs` 里每个 `evalset_inputs_<id>`（以前各评测集的 `inputs.jsonl`，不带标签）各一次，再加当前训练集和验证集的行。`dset` 按任务里固定的规则重算第一轮的分配，从评测集的范围里取没被任何集合、pool 和这些文件占用的行，抽 `sizes.evalset` 行。剩的行不够时它报错：规则固定、改不了，停下，交回 `questions`。用 `PYTHONPATH=<repo root> python3 -m rsi validate evalset <outdir>/evalset --task <task.json>` 校验，交回 `evalset`、`record`；交回后不留 corpus 和新评测集标签的副本。`dset` 只刷新评测集；验证集、封存 test 要刷新时按「输出」的格式自己做。

## 模块指标

本轮的评测和当前最佳版本的比较：新训练集登记后，本轮的 ② 训练调优在它上面训练出候选，④ 评测与决策在评测集上把送评候选和当前最佳版本（旧训练集训出的）比较，主指标的差值、区间和升级或保留的判定写进本轮的 `evaluation`，由 judge 记录。本模块不训练、不量这个指标，交回时 `record` 的 `result` 不填数值（见「实验记录」）。下一次派出时这样得知上一版训练集的效果：本模块以前各轮的 `record` 写着当时改了什么、为什么；读开启本轮的 `plan`（judge 按上一轮评测写了新训练集的结论）。第一轮没有旧训练集，没有这个指标：第一轮交的是可复现的切分（脚本、规则、`dset` 汇总）和质量门。

## 实验记录

每次派出交一份 `record`（字段见 [record.md](../../docs/interfaces/record.md)），放在 `<outdir>/record/`，用 `PYTHONPATH=<repo root> python3 -m rsi validate record <outdir>/record --task <task.json>` 校验。`module` 写 `data`，`owner` 写 `rsi-data`，`automation` 写 `automated`。

- 第一轮 sets：
  - `failure`：原始数据是什么、任务要什么（还没有集合）。
  - `hypothesis`：切分规则和**为什么这样切**：你定的每个键（`by`、`key`、`ranges` 或 `fractions`、`sizes`、`stratify`、`unit`、`seed`）各写一句数据上的依据（例如时间跨度、各段的行数和正例数、分组大小）；任务作者写了的键注明照用。
  - `action`：预处理脚本做什么（读哪些文件、怎么建行、`meta.extra` 放了什么、丢了哪些字段）、运行命令、行数和各标签行数与 `data.rows.expect` 的核对、运行过的 `dset` 命令。
  - `result`：`metric` 写 `val <主指标>`，`before`、`after` 写 null（没有比较），`conclusion` 写 `inconclusive`。
  - `knowledge`：数据的事实（正例比例、随时间的变化、重复和冲突、跨集合共享的 `state_id` 数）和对以后各轮的提示。
  - manifest 的 `notes`：`dset` 汇总：各集合的行数、`key` 跨度、范围内行数、因重叠排除的行数，pool 和未用的行数，`shared_state_ids`，质量门。评测集和封存 test 的标签分布（各标签的行数）不写进 record，也不写进 `handback.json` 的 `notes`：`dset` 的汇总和 corpus 的 `split_report.json` 里有，抄的时候去掉（它们 manifest 的 `notes` 里 `dset` 已经不写）。
- 之后各轮 trainset：`failure` 是计划里归因到数据的问题；`hypothesis` 是为什么这样改有效；`action` 是 `dset` 命令和改动（加了多少行、删了多少行）；`result` 写 `{"metric": "eval <主指标>", "before": null, "after": null, "conclusion": "inconclusive"}`：好坏由本轮 ④ 在评测集上和当前最佳版本比较来判断，本模块没有量，所以数值写 null、结论写 `inconclusive`（`result` 不能有别的字段）；manifest 的 `notes` 写明「由本轮 ② 训练调优和 ④ 评测检验」，再写 `dset` 汇总和质量门。
- refresh：`action` 写排除了哪些文件、范围里还剩多少没用过的行、抽了多少；`notes` 写 `dset` 汇总，不写新评测集的标签分布（`handback.json` 的 `notes` 也不写）。
- `artifacts`：相关的已登记 id（简报的 `ids`）；本次的产物还没登记，写在 `action` 里。

## 长任务

本模块的长任务：数据量大的预处理（预计超过 10 分钟）、大批量标注（例如调用 judge 模型）。本模块不训练，没有训练的长任务。长任务期间可以写简报的进度文件 `<outdir>/progress.json`（格式见 [progress.md](../../docs/interfaces/progress.md)），进度页面显示它；没有长任务就不写。

长任务的交回：预计超过简报规定的时长时，在后台启动，立刻写 `handback.json`，`status` 写 `long_job`，`marker` 写完成标记的绝对路径，然后结束会话，不轮询、不等待。要交回 `long_job` 时完成标记已经在了（长任务在你交回之前就结束了，例如上一个会话启动的）：不要以 `long_job` 交回。引擎只认这次启动之后写的标记，更早的标记永远不算出现，运行会一直等；直接读结果，接着做完，再交回。任务结束时无论成败都写完成标记，第一行是 `ok` 或 `failed`。标记是 `ok` 时，引擎让同一个会话继续（`feedback-<k>.md` 开头是「# 长任务已结束」），由它读结果、写好产物、再以 `ok` 交回；标记不是 `ok` 时这次派出记为失败。

交回 `long_job` 之前，在完成标记旁写 `<marker>.pid`：第一行是长任务进程自己的进程号，第二行是它的启动时间（Unix 秒，`date +%s`）。进程号要由长任务自己写，例如 shell 里 `sh -c 'printf "%s\n%s\n" $$ "$(date +%s)" > <marker>.pid; exec <命令>' > <日志> 2>&1 &`（`exec` 让命令沿用这个进程号），Python 里任务一开始写 `os.getpid()` 和 `int(time.time())`；不要用 shell 给出的后台进程号代替，它可能是包装进程而不是长任务本身。最好让长任务在自己的进程组里运行（例如 Python 的 `subprocess.Popen(..., start_new_session=True)`）。`<marker>.pid` 写好后再写 `handback.json`。运行被停止时，引擎按这个文件确认进程是这个长任务后终止它（整个进程组）；没有这个文件就终止不了，只能列为可能仍在运行、请用户手动结束。

每个长任务用新的完成标记：`<k>` 是这一种长任务在本次派出里的第几次（第一次 1，重做时 2、3……），预处理是 `<outdir>/prep-<k>.done`，别的长任务照样用 `<outdir>/<名>-<k>.done`。启动前先 `rm -f <marker> <marker>.pid`，标记和 pid 文件都从干净的开始：引擎只认修改时间不早于这次启动的标记，同一路径上以前留下的标记不算这个长任务结束。

例如在后台跑预处理（sh 在自己的会话里，`$$` 就是进程组号）：

```
rm -f <outdir>/prep-<k>.done <outdir>/prep-<k>.done.pid
python3 -c 'import subprocess, sys; subprocess.Popen(["sh", "-c", sys.argv[1]], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)' 'printf "%s\n%s\n" $$ "$(date +%s)" > <outdir>/prep-<k>.done.pid; cd <outdir>/corpus && python3 preprocess.py > <outdir>/prep-<k>.log 2>&1 && echo ok > <outdir>/prep-<k>.done || printf "failed\n预处理出错，见 prep-<k>.log\n" > <outdir>/prep-<k>.done'
```

等 `<outdir>/prep-<k>.done.pid` 写出来（例如 `until [ -s <outdir>/prep-<k>.done.pid ]; do sleep 1; done`）再写 `handback.json`（`status` 是 `long_job`，`marker` 是 `<outdir>/prep-<k>.done`）。

定时检查：任务设了 `flow.checkin_minutes` 时，长任务期间引擎每隔这么多分钟把开头是「# 定时检查」的反馈送回同一个会话（不算退回，次数有上限 `flow.max_checkins`）。看长任务还在不在、有没有进展：进程（`<marker>.pid` 第一行的进程号，`ps -p <pid>`）、日志（例如 `<outdir>/prep-<k>.log`）、写了的话 `<outdir>/progress.json`、磁盘。正常就以 `long_job` 交回**同一个**标记，不等待。进程已经不在却没写标记，或者出错了要重跑：先结束还在跑的部分（`kill -- -<pid>`：整个进程组），再按上面的做法用下一个 `<k>` 重新启动，交回新的标记。

## 负责人和版本

- 负责人：框架维护者。本模块在框架仓库的 `modules/data/` 里维护：本 skill 和切分工具 `dset/`。
- 版本：Skill 1.2；dset 0.1.1（`modules/data/dset/__init__.py` 的 `DSET_VERSION`）。
- 变更：1.2（2026-11-25）不再自己训练验证新训练集；好坏交给训练调优和评测：去掉新旧训练集的 A/B 训练（`ab-<k>` 草稿候选和标记），简报不再给训练命令；之后各轮 trainset 的 `record` 的 `result` 写 null 和 `inconclusive`，由本轮 ② 和 ④ 检验；模块指标改为本轮评测和当前最佳版本的比较。1.1（2026-11-25）长任务每次用新的完成标记（`prep-<k>.done`、`ab-<k>.done`），启动前删掉旧的；A/B 候选放在 `<outdir>/ab-<k>-old`、`ab-<k>-new`（进度写进 `<outdir>/progress.json`）；定时检查怎么做；`dset` 不再把评测集和封存 test 的标签分布写进它们 manifest 的 `notes`，交回的 `notes` 也不写。1.0（2026-11-25）流程从数据模块开始：第一轮 sets 里由你写预处理脚本（登记成 corpus）、按数据定切分规则（记成任务定义的新版本，不需要审批），`dset` 切出四个集合和训练池 pool；之后各轮从 pool 增减训练集，refresh 从 corpus 抽新评测集。
- 已知限制：`dset` 只刷新评测集；corpus、pool 登记和切分时整份读进内存；原始数据只支持本机路径，不对原始数据算 sha256；登记时不运行预处理脚本（复跑审计由操作者按 manifest 的 `command` 做）；禁读名单只挡 agent 自己的文件工具，隔离靠约定和操作者的 `chmod 000`；没有完整规则的工作区不能用 refresh。

## 常见错误

| 错误 | 后果 | 正确做法 |
|---|---|---|
| 交出后再读评测集、封存 test 或 corpus，或留着它们的标签副本 | 评测集、封存 test 的标签混进之后的判断，评测不再独立 | 交回之后不留副本、不再读；refresh 只用简报给的 corpus 和不带标签的 `inputs.jsonl` |
| 第一轮 sets 之后（第二轮起的 trainset、refresh 派出）读 `data.sources` | 原始数据在这些派出的禁读名单里；读到的标签可能混进新集合，corpus 登记后它也已被设成读不了 | 新训练集从 `pool` 抽，新评测集从 `corpus` 抽 |
| 自己写切分、抽样代码，或手改 `dset` 写出的集合 | 集合不能按规则复现，没有汇总可以核对，refresh 也重算不出第一轮的分配 | 切分只用 `dset`：规则写进 `split.json`，由 sets 执行 |
| 预处理脚本的输出不确定（写了时间、没固定种子、没排序、依赖文件列出的顺序） | 复跑审计的 sha256 对不上，集合追溯不到登记的脚本 | 同样的输入写出同样的字节：按 id 排序、不写时间、固定种子；交回前跑两次比较 sha256 |
| 在输出目录外（例如 /tmp）留带标签的中间文件，或在 corpus 和集合目录之外留带标签的行 | 交回之后标签还在，之后的派出读得到 | 带标签的行只写进 corpus 和集合目录；临时文件交回前删掉 |
| 改任务作者写了的切分键（在 `split.json` 里换掉，或手改 `task/task.json`） | `dset` 报 `data.split.<键>` 冲突；手改的任务版本引擎不登记 | 作者写的键照用；行不通（行数不够、门过不了）时交回 `questions` |
| 脚本不在 corpus 目录里运行，或输出写成绝对路径 | 按 manifest 的 `command` 复跑写不出同一个文件 | 在 corpus 目录里运行，输出写相对路径 `rows.jsonl` |
| 行数或各标签行数和 `data.rows.expect` 对不上 | corpus 登记被拒：预处理丢了行或标签错了 | 改脚本重跑，不改 `expect` |
| 把同一 `state_id` 跨集合当成错误，删掉这些行 | 白白丢行；这是允许的，只报告数目 | 写进 `record`；要模拟完全没见过的对象时才定 `unit: state_id` |
| `sizes.evalset` 不写或等于评测集范围的行数 | 以后的 refresh 没有没用过的行 | 给评测集的范围留出余量 |
| `dset` 说 already exists | 上一次的输出还在，`dset` 不覆盖 | 重做一次 corpus 还没登记的切分：删掉 `<outdir>` 下的 `task/`、五个集合目录，以及 corpus 目录里的 `manifest.json` 和 `split_report.json`（脚本和 `rows.jsonl` 留着）再跑。corpus 已登记时照「续做」在 `redo/` 里重切，已登记的 corpus 一律不碰。中途停下的那次（corpus 的 `manifest.json` 还没写出）只留下已放好的新目录、corpus 里的 `split_report.json` 和 `.dset-` 开头的条目：删掉它们再跑 |
| grow、check 的 `--write`、refresh 说 already exists | 那个输出目录已经有了 | 它还没登记：删掉再跑；已经登记过（复制留在输出目录里的）：换一个还没登记过的子目录名，例如 `<outdir>/trainset-2`、`<outdir>/evalset-2`，交回时列这个名字 |
| 把评测集或封存 test 在交出之后继续留着用，或在训练集里混入它们的样本 | 评测结果不再可信 | 训练集只从 `pool` 和 feedback 写成的行来；登记时也会因重叠拒绝 |
| 为了验证新训练集自己训练（调用训练命令、trsi 或别的训练代码） | 数据模块成了自己数据的裁判；训练不记进预算，和 ② 的搜索重复；产出的候选登记会拒绝 | 不训练：`record` 的 `result` 写 null 和 `inconclusive`，好坏由本轮 ② 训练调优和 ④ 评测判断 |
