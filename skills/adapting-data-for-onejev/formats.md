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

下面各例只写 `adapt` 的主体；`Q` 是用户确认过的 `question` 字典，`POS` 是用户确认的正类取值。

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

适配器每次只拿到一个路径。给同一个文件起两个名字，让适配器按文件名取行：

```bash
ln -s <数据>/data.csv <工作区>/train.csv
ln -s <数据>/data.csv <工作区>/val.csv
```
```python
want = path.stem                                   # "train" 或 "val"
for i, r in enumerate(csv.DictReader(open(path, encoding="utf-8", newline=""))):
    if r["split"] != want:                         # test 行在这里被跳过，不看内容
        continue
    ...  # 同例 3，"split": want，id 用 f"{r['split']}:{i}" 或数据里的 id
```
`exp.yaml`：`data.train: train.csv`，`data.eval.val.train_labels: val.csv`。sha256 按链接指向的源文件算。

## 6. 多个文件按名字

`train.jsonl`、`dev.jsonl`、`test.jsonl`：

```yaml
data:
  train: <数据>/train.jsonl
  eval:
    dev: {train_labels: <数据>/dev.jsonl}
  # test.jsonl 不写；只有用户批准 --allow-test 才另行处理
```
适配器按 `path.stem` 设 `split`：`"train"` 或 `"val"`。

## 7. 没有验证集（holdout）

```bash
ln -s <数据>/train.jsonl <工作区>/train.jsonl
ln -s <数据>/train.jsonl <工作区>/val.jsonl
ojh make-split --out <工作区>/splits.json --group <组字段> --mode holdout <数据>/train.jsonl
```
适配器读 `splits.json`（路径相对适配器文件），按 `path.stem` 只 yield 分到这一边的行。按 `ojh make-split --help` 核对文件结构和 id 的写法，让适配器产出的 `id` 和它一致。
