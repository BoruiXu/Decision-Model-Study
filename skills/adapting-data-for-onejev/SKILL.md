---
name: adapting-data-for-onejev
description: Use when a dataset is not yet in OneJev row format and needs an adapter (adapters/<name>.py) before ojh prepare — checking row format, writing and self-testing an adapter. 数据转成 OneJev 行、写适配器。
---

# 把数据适配成 OneJev 行

训练前，`ojh prepare` 通过适配器读取数据。数据已经是 OneJev 行（每行一个 JSON）时，用内置的 `onejev_rows` 适配器即可，不用写代码。否则写一个适配器。训练流程见 `training-with-onejev` skill。

## OneJev 行格式

每一行是一个 dict：

| 字段 | 内容 |
|---|---|
| `id` | 唯一字符串；train 与各评测集之间、评测集之间都不能重复 |
| `state` | `{"messages": [{"role": ..., "content": ...}, ...]}`，模型输入；不能为空 |
| `question` | `{"type": ..., "instructions": ..., "criteria": ...}` |
| `target` | `{标签: 概率}`，是在 `criteria` 上的分布；键必须是该题型的标签，值在 [0, 1]，总和为 1 |
| `meta` | 自由字典。实验配置里 `fields.group`、`fields.consistency`、子集的 `where` 引用的字段放在这里（用点号路径，例如 `meta.category`） |
| `split` | 切分名；以 `test` 开头的行，没有 `--allow-test` 一律拒绝读取 |

题型决定合法的标签（`prepare` 按这个校验）：

- `noul`：标签是 `yes`、`no`；
- `choice`：标签是 `criteria` 的键；
- `score`：标签是 `"0"`、`"1"`、… ，个数等于 `criteria` 的长度；
- 其他题型会被拒绝（unknown question type）。

`fields.label` 写正类在 `target` 里的键（例如 `target.yes`）。

## 写适配器

1. 复制 `adapters/template.py` 为 `adapters/<name>.py`，实现 `adapt(path) -> Iterator[dict]`：读一个源文件，逐行 yield 上面格式的 dict。只做格式转换，不要在适配器里切分评测集、重新打标签或读 test 以外不该读的文件。
2. 在实验配置里写 `data.adapter: <name>`（模块名，不带 `.py`）。
3. 适配器必须是确定性的：同一个源文件每次产生同样的行和顺序，否则 sha256 和 lock 对不上。

## 自测

1. 先用一个很小的源文件（几十行）和一份只含它的实验配置，运行 `ojh prepare <exp.yaml> --machine <machine.yaml>`。
2. 它会检查：schema、id 唯一、target 分布、split 字段、train 与评测集 id 不重叠、`data.train_sha256`（没写会警告并记录算出的值，写完整值以便以后核对）。有错会列出前 20 条，按提示修适配器。
3. 看输出的 counts（train、select、report 条数）和 `<out_root>/<name>/prepared/prepared.json` 里的 `dropped`（参照连接、未分配的 id）是否合理。
4. 通过后换成完整数据再跑一次，然后回到 `training-with-onejev` 做 `ojh resolve`。

## 评测集的声明

评测集由实验配置显式列出，没列的文件一律不读：

```yaml
data:
  train: <DATA>/train.jsonl
  eval:
    <集名>:
      train_labels: <DATA>/<集名>.jsonl   # 选模算 NLL 用，必须和训练用的标签版本相同
      reference: <DATA>/<集名>_ref.jsonl  # 可选：定阈值、AUC、门用的参照标签；不写等同 train_labels
  split_file: <DATA>/splits.json         # 可选：{集名: {"assignment": {id: "select" | "report"}}}
```

没有 `split_file` 时，评测集里的行不会被分到 select 或 report，所以需要选模和定阈值时必须提供它。`train_labels` 和 `reference` 按 id 关联，只用两边都有的 id，被去掉的 id 记进记录卡。
