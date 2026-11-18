# OneJev 行格式和常见数据格式的例子

## OneJev 行

| 字段 | 内容 |
|---|---|
| `id` | 唯一字符串；train 与各评测集之间、评测集之间都不能重复 |
| `state` | `{"messages": [{"role": ..., "content": ...}, ...]}`，模型输入；不能为空 |
| `question` | `{"type": ..., "instructions": ..., "criteria": ...}` |
| `target` | `{标签: 概率}`，键必须是该题型的标签，值在 [0, 1]，总和为 1 |
| `meta` | 自由字典；`fields.group`、`fields.consistency`、`subset.where` 引用这里的字段（如 `meta.category`） |
| `split` | 训练行 `"train"`，评测行 `"val"`；以 `test` 开头的行没有 `--allow-test` 一律拒绝 |

题型和合法标签：`noul` → `yes`/`no`（`criteria` 是 `{"yes": 说明, "no": 说明}`）；`choice` → `criteria` 的键；`score` → `"0"`、`"1"`…，个数等于 `criteria` 列表长度。`fields.label` 写正类的键，二分类就是 `target.yes`。

下面各例只写 `adapt` 的主体；`Q` 是用户确认过的 `question` 字典，`POS` 是用户确认的正类取值，`SPLIT` 来自 `adapter_args`（训练文件 `{split: train}`，评测集 `{split: val}`，见例 6）。

## 1. 已是 OneJev 行

`data.adapter: onejev_rows`，不写代码。只核对 `question` 文本是用户要的、`fields.label` 指向正类。

## 2. 对话 + 标签（jsonl）

```python
for i, line in enumerate(open(path, encoding="utf-8")):
    r = json.loads(line)
    if r.get("label") is None:
        continue                                   # 缺标签：跳过，计数告诉用户
    yes = 1.0 if r["label"] == POS else 0.0
    yield {"id": r.get("id") or f"{path.name}:{i}", "state": {"messages": r["messages"]},
           "question": Q, "target": {"yes": yes, "no": 1.0 - yes},
           "meta": {"category": r.get("category")}, "split": SPLIT}
```

## 3. 文本 + 标签（csv）

```python
for i, r in enumerate(csv.DictReader(open(path, encoding="utf-8", newline=""))):
    yes = 1.0 if r["label"] == POS else 0.0          # csv 里都是字符串，POS 也写成字符串
    yield {"id": f"{path.name}:{i}", "state": {"messages": [{"role": "user", "content": r["prompt"]}]},
           "question": Q, "target": {"yes": yes, "no": 1.0 - yes},
           "meta": {"source": r["source"]}, "split": SPLIT}
```
`exp.yaml`：`fields.group: meta.source`。

## 4. 软标签 + 一致比例

标签 `p_agree` 在 [0,1]，本身就是一致比例：

```python
p = float(r["p_agree"])
row["target"] = {"yes": p, "no": 1.0 - p}
row["meta"]["f"] = p                               # fields.consistency: meta.f
```

硬标签另有比例字段时：`target` 用硬标签，`meta["f"] = float(r["agree_rate"])`。

## 5. 单文件带 split 列

同一个文件既作训练又作评测，用 `adapter_args` 告诉适配器取哪一部分：

```yaml
data:
  adapter: ./adapter.py
  train: <数据>/data.csv
  train_adapter_args: {split: train}
  eval:
    val: {train_labels: <数据>/data.csv, adapter_args: {split: val}}
```
```python
def adapt(path, split):
    for i, r in enumerate(csv.DictReader(open(path, encoding="utf-8", newline=""))):
        if r["split"] != split:                    # test 行在这里被跳过，不看内容
            continue
        ...  # 同例 3；"split": "train" if split == "train" else "val"；id 用数据里的 id，没有就 f"{path.name}:{i}"
```

## 6. 多个文件按名字

`train.jsonl`、`dev.jsonl`、`test.jsonl`：

```yaml
data:
  train: <数据>/train.jsonl
  eval:
    dev: {train_labels: <数据>/dev.jsonl}
  # test.jsonl 不写；只有用户批准 --allow-test 才另行处理
```
再加 `train_adapter_args: {split: train}` 和 `dev` 的 `adapter_args: {split: val}`，适配器把它写进行的 `split` 字段。

## 7. 没有验证集（holdout）

```bash
ojh make-split --mode holdout --out <工作区>/holdout --group source --frac 0.2 <数据>/data.csv
# 写出 <工作区>/holdout/{train.jsonl, val.jsonl, manifest.json}
```
```yaml
data:
  adapter: ./adapter.py
  train: holdout/train.jsonl                       # 相对 exp.yaml
  train_adapter_args: {split: train}
  eval:
    val: {train_labels: holdout/val.jsonl, adapter_args: {split: val}}
```
两个文件都是 jsonl（csv 源的单元格已转成 JSON 值，`label` 可能变成数字 `1` 而不是 `"1"`），适配器用 `json.loads` 读，`split` 参数只用来写行的 `split` 字段。原始行有 `id` 就原样用；没有就派生（如 `f"{path.stem}:{i}"`，两个文件名不同，不会重复）。
