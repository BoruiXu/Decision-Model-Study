# 实验记录：record

每个模块在完成一次改动后产出一份 `record`，是 Meta-RSI 的材料（设计文档第 9 节）。目录里有 `manifest.json`，公共字段见 README.md。校验由 `rsi/artifacts.py` 的 `validate_artifact` 完成，只检查字段的类型和取值，没有额外的目录内检查，也不要求必备文件。

每份记录回答：发现了什么问题，为什么认为改动有效，改了什么，结果如何，学到了什么，这一步现在是人工、脚本还是自动完成。额外字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | 固定 `"record"` | 产物类别 |
| `module` | 枚举 `data`、`train`、`tune`、`deliver`、`judge`、`orchestrator` | 记录属于哪个模块 |
| `owner` | string | 负责人 |
| `failure` | string | 发现的问题 |
| `hypothesis` | string | 为什么认为改动有效 |
| `action` | string | 改了什么 |
| `result` | object | 结果，见下 |
| `knowledge` | string | 学到了什么 |
| `automation` | 枚举 `manual`、`scripted`、`automated` | 这一步现在的完成方式 |
| `artifacts` | string 列表 | 相关产物的 id |

`result` 的子字段：

- `metric`：指标名，并写明在哪个集合上量的，例如 `val accuracy`、`val auc`；只写 `accuracy` 说不清是验证集还是评测集。
- `before`、`after`：改动前后的数值（number 或 null），必须在同一个集合上、用同一个指标量出来，两者才能比。`before` 是这次实验的起点：之前的最佳结果、默认配置的结果，或旧数据训出的结果，按实验比的是什么选；`after` 是改动后的结果。量不出来时写 null。
- `conclusion`：`success`、`failure` 或 `inconclusive`。
