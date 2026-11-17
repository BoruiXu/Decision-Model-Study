# 评测插件

二分类以外的任务（例如多选一、多档位），由使用方写一个 Python 文件作为评测插件，在实验配置里这样指定：

```yaml
evaluator:
  type: plugin
  path: <PLUGIN>/my_task.py
```

repo 只提供接口，不提供具体任务的插件。插件里每个钩子都是可选的，不提供时用默认行为。

## 钩子

| 钩子 | 作用 | 不提供时 |
|---|---|---|
| `validate(rows) -> list[str] \| None` | 任务特有的数据检查；返回非空的错误列表会让 run 失败 | 不检查 |
| `select_rule(events) -> int` | 选模规则：从训练日志里的 `harness_eval` 事件中返回要选的 step | NLL 最低 |
| `calibrate(preds, rows) -> dict` | 决策阈值，返回的字典写进记录 | 不定阈值 |
| `baselines(train_rows, eval_rows) -> dict` | 对照组，返回 `{名字: 指标字典}` | 只有 TF-IDF 和分组先验 |
| `acceptance(preds, rows) -> dict` | 路由器的指标，可带 `"gate"`（`{"passed": bool, ...}`，也可以直接是 bool） | 只报 NLL、准确率、ECE |

- `preds` 是 `{行 id: {"id", "labels", "probs", "target"}}`，即复评后的逐题记录。
- `acceptance` 和 `baselines` 会在 select 合并集上、以及每个评测集的 select 部分上各调用一次。
- 门是否生效仍由实验配置的 `evaluator.gate` 决定；所有给出了 `gate` 的集合都通过，run 才算通过。

## 最小示例

```python
"""my_task.py：只检查数据，其余用默认。"""


def validate(rows):
    errors = []
    for r in rows:
        if len(r["target"]) < 2:
            errors.append(f"{r['id']}: target needs at least 2 labels")
    return errors


def acceptance(preds, rows):
    # 例：top-1 准确率，并要求不低于 0.5 才算过门
    ok = 0
    for r in rows:
        p = preds[r["id"]]
        top = max(range(len(p["probs"])), key=p["probs"].__getitem__)
        ok += p["target"][top] == max(p["target"])
    acc = ok / len(rows) if rows else 0.0
    return {"n": len(rows), "accuracy": acc, "gate": {"passed": acc >= 0.5}}
```
