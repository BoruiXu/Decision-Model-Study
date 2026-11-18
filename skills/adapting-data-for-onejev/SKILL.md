---
name: adapting-data-for-onejev
description: Use when a user's dataset (jsonl, csv, chat logs, a directory of split files, soft or hard labels, OneJev rows or not) must become a OneJev training experiment — before the first ojh run on new data, when it is unclear which field is the label, group, consistency or split, when there is no validation set, or when ojh prepare / plan-only rejects the rows. 认数据、写适配器、写 exp.yaml、切 select/report。
---

# 把数据变成 OneJev 实验

## 概览

认数据、写适配器、填 `exp.yaml` 都在这个 skill 里做，不在代码里。`ojh inspect` 只给事实，判断由你做；**标签含义和判定问题由用户确认**。产出放在实验工作区（位置见 `training-with-onejev`，不在本 repo 里）：

```
<工作区>/exp.yaml      # 必有
<工作区>/adapter.py    # 数据不是 OneJev 行时
<工作区>/splits.json   # 用 ojh make-split 切分时
```

## 步骤

1. **看事实**：`ojh inspect <数据位置> --json`。每个文件给出：行数、字段路径的类型和填充率（`fields`）、低基数字段的取值计数（`values`）、[0,1] 数值（`unit_interval`）、文本字段（`text_like`）、样例（`samples`）、split 列（`split_columns`）、文件名提示（`name_hint`）、split 文件结构（`split_file`）、是否已是 OneJev 行（`onejev.passes`）。test 文件和 test 行它只报行数，不给字段和样例。
2. **按下面的判定表定格式**；细节和小例子见同目录 `formats.md`。
3. **问用户**（推不出来的，一次问完，见“必须问用户”）。
4. **写适配器**（已是 OneJev 行就不用写）。
5. **需要时切分**：`ojh make-split`。
6. **填 `exp.yaml`**：从 `<repo>/examples/exp.template.yaml` 复制到工作区再改。
7. **自测**：小样本 `ojh prepare` → 全量 `ojh prepare` → 回 `training-with-onejev` 的 `ojh run`。

## 判定表

| inspect 看到的 | 判断 | 做法 |
|---|---|---|
| `onejev.passes: true` | 已是 OneJev 行 | `data.adapter: onejev_rows`，不写代码 |
| 有 `messages`（role/content 列表）+ 一个标签字段 | 对话 + 标签 | `state.messages` 直接用 `messages` |
| 一个文本字段（prompt/text/question…）+ 标签字段，jsonl 或 csv | 文本 + 标签 | `state.messages = [{"role": "user", "content": 文本}]` |
| 一个文件，`split_columns` 非空（train/val/test） | 单文件带 split 列 | 同一文件既作 `data.train` 又作评测集，用 `adapter_args` 告诉适配器取哪个 split，见 `formats.md` |
| 多个文件，`name_hint` 是 train/validation/test | 按文件名分 | train → `data.train`，val/dev → `data.eval`；**test 文件不写进配置** |
| 只有一份数据，没有 split 列也没有验证集 | 缺验证集 | `ojh make-split --mode holdout` 切出 `train.jsonl`/`val.jsonl` 两个文件；没有 test 就不造 test |
| 有验证集但没有 select/report 切分 | 缺 report | 验证集够大（约 ≥200 行）时 `--mode select-report`；太小就不切，全部算 select，并告诉用户 |
| 标签是 [0,1] 的小数 | 软标签 | `target = {"yes": p, "no": 1 - p}` |
| 标签是 0/1、true/false、两种字符串 | 硬标签 | `target = {"yes": 1.0, "no": 0.0}` 或反过来，按用户确认的正类 |
| 标签多于两类 | 非二分类 | `choice`/`score` 题型，并需要评测插件（`plugins/README.md`）；先和用户确认任务 |

**分组字段**（`fields.group`）：类别、来源、主题、模板一类的低基数字段（inspect 的取值计数里找）。用于分组先验基线和类内 AUC。

**切分用的组**（`make-split --group`）：同一题、同一模板、同一对话的行必须在同一边。有最细的“同源”字段就用它；只有粗类别（如几到几十个取值的 `source`）时，它照样做 `fields.group`，但切分用哪个组要问用户：按类别整组切（测类别外泛化），还是用 id 当组（逐行随机切）。

**一致比例字段**（`fields.consistency`）：每题“判正类是对的”的比例，例如多次采样中一致的比例 `meta.f`。来源：一个 [0,1] 数值字段，含义由用户确认；软标签本身就是这个比例时，用同一个值。硬标签训练、另有比例字段时（如 majority 标签 + `meta.f`），比例放 `meta` 里填进 `fields.consistency`。没有就不填，只是少了预算下的承接率和 oracle。

## 必须问用户（推不出来时，一次问完）

拿下面的模板发给用户，等确认后再写适配器：

```
数据：<文件和行数>；标签字段 <label>，取值 <值: 条数, ...>
1. 正类（yes）是 <label = ?>，含义是：<……>。对吗？
2. 模型看到的判定问题（instructions）拟为：
   “<根据上面的内容判断：……？>”
   yes 表示：<……>；no 表示：<……>
   可以直接用，或请改。
3. 分组字段用 <source>（<k> 个取值）——用于分组基线和类内 AUC。对吗？
4. 一致比例字段：<无 / meta.f（含义……）>
5. 切分：<已有 val / 从全部数据按 <组字段> 切 <比例，建议 0.1–0.2> 做 val / val 再对半分 select、report>
6. 工作区：<路径>
```

只在第 3–5 项确实有歧义时才问它们；第 1、2 项除非数据里已写明（例如已是 OneJev 行），否则都要问。用户的回答写成 `exp.yaml` 里的注释。

## 写适配器

- 复制 `<repo>/adapters/template.py` 到 `<工作区>/adapter.py`，实现 `adapt(path, **args) -> Iterator[dict]`：读一个源文件，逐行 yield OneJev 行。`exp.yaml` 里写 `data.adapter: ./adapter.py`（相对 `exp.yaml`）。适配器文件的 sha256 和参数都记进 lock；prepare 之后再改适配器，resolve 会报错，要重新 prepare。
- 参数：`data.adapter_args`（所有文件）、`data.train_adapter_args`（训练文件）、`data.eval.<集名>.adapter_args`（该评测集），后两者合并在前者之上，以 `adapt(path, **args)` 传入。用它区分同一文件的不同角色（如 `{split: val}`），不要靠文件名或符号链接。
- OneJev 行：`id`、`state`、`question`、`target`、`meta`、`split`（字段表见 `formats.md`）。
- `question` 对每行都相同：`{"type": "noul", "instructions": <用户确认的问题>, "criteria": {"yes": ..., "no": ...}}`。
- `split`：训练文件的行写 `"train"`（示例配方用 `--split train` 取训练行），评测文件的行写 `"val"`。
- `id`：数据里有唯一的 id 列（`id`、`qid`、`meta.uid` 等）就用它的**文本**作 id（`str(...)`；`"0007"` 保持 `"0007"`），切分时把同一列传给 `ojh make-split --id <字段>`；没有就派生稳定的 id（如 `<文件名>:<行号>`，或文本的哈希），这时不能用 make-split 按原始文件切。train 和评测集之间不能重复。
- 分组字段、一致比例字段放进 `meta`。
- **确定性**：不用随机数、时间、无序的 set/dict 遍历；同一文件每次产出同样的行和顺序。
- **不重新打标签**：只按用户确认的映射换写法。缺标签的行跳过，并把条数告诉用户；不要补、不要猜。
- **不读 test**：不 yield split 以 `test` 开头的行，不打开 test 文件（inspect 已经替你挡住了 test 的字段和样例，不要绕过它去看）。

## 切分：`ojh make-split`

```bash
# 没有验证集：从训练文件按组切出 val，--out 是目录
ojh make-split --mode holdout --out <工作区>/holdout --group <组字段> --frac <val 比例> <训练文件>
# 有验证集：按组分 select / report，--out 是切分文件
ojh make-split --mode select-report --out <工作区>/splits.json --group <组字段> --id <id 字段> <集名>=<val 文件>
```

- 两种模式都按组切（同一组永远在同一边），默认 `--seed 20261115 --frac 0.5`，确定性。不要另写切分脚本，不要换 seed 挑结果。已有的输出不会被覆盖：参数变了就换个输出位置。
- `--group` 是**原始行**上的点分字段（如 `source`），不是 `meta.source`；缺这个字段的行会报错。
- **holdout**：`--frac` 是 val 的比例，默认 0.5 偏大，按用户在问题 5 里确认的比例传。`--out` 是目录（写成 `.json` 会被拒绝），里面是 `train.jsonl`、`val.jsonl`（原样复制的行，csv 的单元格转成 JSON 值，id 不加不改）和 `manifest.json`。`data.train` 指 `<out>/train.jsonl`，`data.eval.val.train_labels` 指 `<out>/val.jsonl`，适配器用同一套规则读这两个 jsonl（源文件是 csv 时也要按 jsonl 读）。val 全部算 select（没有 report），汇报时说明。
- **select-report**：每个文件是一个评测集，集名取文件名主干或 `名字=路径`，要和 `data.eval` 的键一致；输出填进 `data.split_file`。`--id <点分字段>`（默认 `id`）指出原始行里的 id 列，make-split 按文本写入，适配器必须产出同样的文本作行 id；原始的 csv、chat jsonl 都可以直接切，不要先转成 OneJev 行（那样会绕过适配器文件、`adapter_args` 和 lock 里的适配器 sha256）。原始数据没有任何 id 列时：不切，全部算 select，并告诉用户。

## 填 exp.yaml

从 `examples/exp.template.yaml` 复制，按注释填：`name`、`data.adapter`（及需要的 `adapter_args`）、`data.train`、`data.eval.<集名>.train_labels`（可选 `reference`）、`data.split_file`、`fields.label: target.yes`、`fields.group`、`fields.consistency`、`fields.text: messages`。`base`、`gpus`、`recipe` 不填，由 `ojh run --base --gpus` 给。不认识的键 prepare 会报错。

## 自测

1. 小样本：每个源文件取前 200 行放到 `<工作区>/sample/`（csv 用 `head -n 201` 保留表头），复制一份 `exp.sample.yaml` 指向它们，`name` 加 `-sample`。运行 `ojh prepare <工作区>/exp.sample.yaml`。
2. 按报错修适配器（会列前 20 条：schema、id 重复、target 不合法、train 与评测集 id 重叠）。看 counts（train / select / report）和 `prepared.json` 的 `dropped` 是否合理。
3. 全量：`ojh prepare <工作区>/exp.yaml`。把打印的源文件 sha256 写进 `data.train_sha256`（完整值）。
4. 回 `training-with-onejev` 第 3 步。

## 常见错误

| 错误 | 正确做法 |
|---|---|
| 看标签取值就自己定了哪个是正类 | 问用户；正类决定所有指标的方向 |
| 自己编了判定问题直接用 | 用模板给用户看，确认后再写 |
| 把高基数字段（id、prompt）当 `fields.group` | 用低基数的类别/来源字段 |
| 有同源字段却按行随机切，同一题落在两边 | `--group` 用同源字段 |
| 把 test 文件写进 `data.eval`，或用 `head`/`cat` 看 test 内容 | test 不进配置、不看 |
| holdout 用默认 `--frac 0.5`，一半数据进了 val | 用问题 5 里用户确认的比例 |
| select-report 的 `--id` 和适配器的 id 不是同一列，或适配器把 id 转成了数字 | `--id` 指适配器取 id 的那一列，适配器用 `str(...)` |
| 用符号链接或文件名区分同一文件的 train/val | 用 `train_adapter_args` / `eval.<集名>.adapter_args` |
| 缺标签的行填 0 或按多数类补 | 跳过并报条数 |
| 适配器里 `random.shuffle` 或遍历 set | 保持源文件顺序 |
| 把 `adapter.py`、`exp.yaml` 写进 repo 的 `adapters/` | 写在工作区 |
| 软标签当硬标签（四舍五入） | 原样写成分布，除非用户要求硬标签 |
| 跳过小样本自测直接全量 `ojh run` | 先 `ojh prepare` 小样本 |
| `train_sha256` 不写 | 全量 prepare 后写完整值 |
