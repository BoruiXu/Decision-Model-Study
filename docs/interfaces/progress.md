# 模块进度文件 `progress.json`

框架不读任何训练或 trsi 专用的文件。模块想让页面和命令行显示进度，就在**本次派出的输出目录**里写 `progress.json`；框架原样显示它给的内容。写不写由模块决定，不写只是看不到进度，不影响交回和登记。

```json
{
  "version": 1,
  "updated_at": "2026-11-24T12:46:23Z",
  "summary": "第 3 批：3 个训练在跑（卡 0、4、6）",
  "facts": [{"label": "配置", "value": "3/3"}, {"label": "GPU 小时", "value": "1.3/8"},
            {"label": "当前方案", "value": "c4339e55d6（lr 5e-6）"}],
  "table": {"title": "本次搜索的试验",
            "columns": ["试验", "相对起点的改动", "seed", "状态", "卡", "用时", "验证集 auprc"],
            "rows": [["c4339e55d6-s1", "lr 5e-6", 1, "训练 22/32 步 · 剩约 3 分", "0", "6 分", null]]},
  "notes": ["next：lr 5e-6 试探过门槛，补 seed 1、2"]
}
```

| 键 | 必需 | 规则（读的一方：`rsi/progress.py` 的 `read_progress`） |
|---|---|---|
| `version` | 否 | 不写或 1；其他值：整个文件不用，记一条问题 |
| `updated_at` | 是 | UTC ISO 8601（`Z` 或带时区）；缺或解析不了：照常显示，记一条问题，不判断是否过时 |
| `summary` | 否 | 字符串，一句话；超过 200 字截断加「…」 |
| `facts` | 否 | `[{label, value}]`，label 字符串 ≤ 40 字，value 字符串或数 ≤ 80 字；最多用前 12 个；坏的项跳过并记问题。**顺序有意义**：命令行状态行只用前 3 个 |
| `table` | 否 | `{title?, columns, rows}`：columns 1–12 个字符串；rows 最多用前 100 行，每行是列表（按列数补 `null` 或截断），格子是字符串、数、布尔或 null，每格 ≤ 200 字 |
| `notes` | 否 | 字符串列表，最多用前 20 条，每条 ≤ 500 字 |

- 文件是 UTF-8 JSON 对象，不超过 256 KiB（超过不读，记问题）；其他键忽略；写的一方用临时文件加 `os.replace` 原子替换。读不了、不是对象或 JSON 坏了：不显示进度，只显示一条问题，**从不让页面或引擎出错**。所有文字在页面上转义。
- **过时**：派出还活着（`running`，或 `long_job` 且完成标记还没出现）而 `now − updated_at > PROGRESS_STALE_SECONDS = 300` 秒时，页面和状态行标「进度 N 分钟没有更新」。
- 简报：`brief.json` 的 `progress_file` 是 `<输出目录>/progress.json` 的绝对路径，`brief.md` 用一行说明它是可选的、格式见本文。

## 读的一方怎么显示

`rsi.progress.read_progress(path, now)` 文件不存在时返回 None；否则返回 `path`、`updated_at`（带时区的时间或 null）、`age_s`（距 `updated_at` 的秒数或 null）、`summary`、`facts`（(label, value) 列表）、`table`（`title`、`columns`、`rows`，或 null）、`notes` 和 `problems`（每条违反的规则一句）。

- 超过上限的文字截断，末尾加「…」，不算问题；`facts`、`rows`、`notes` 超过条数的部分不用，也不算问题。
- 格子和 fact 的值：字符串原样，数按 `%.6g` 写，布尔写「是」「否」，`null` 显示为空。
- `table.columns` 超过 12 个时只用前 12 个，记一条问题；不是列表、为空或有不是字符串的列名时整个表格不用，记一条问题。不是列表的行跳过，类型不对的格子显示为空，各记一条问题。
- `rsi.progress.dispatch_progress(ws, d)` 读派出 `d` 输出目录里的文件，另加 `stale`（上面「过时」的判断）。

## 例：训练调优模块

`trsi init --brief` 把简报的 `progress_file` 记进 run.yaml（`framework.progress_file`；简报没有时是 `<输出目录>/progress.json`）；`trsi launch` 之后、`trsi watch` 每次轮询（后台，训练期间一直在写）、`trsi collect` 每个试验打包前后各写一次。训练命令 `modules/train/train` 开始、训练中每次轮询、结束时写 `<--out 的父目录>/progress.json`（同一个目录里同时跑两个训练命令时后写的覆盖先写的）。写失败只记进日志，**从不**让 watch、collect 或训练命令失败。内容：

- `summary`：在跑几个训练和它们的卡、几个在排队、几个在打包；
- `facts`（依此顺序）：`配置` 已提出的配置数 / `search.max_configs`（每次派出计）、`GPU 小时` 已花 / 简报的预算、`当前方案` 当前方案的候选 id 和相对起点的改动；
- `table` 每个试验一行：试验、相对起点的配置改动（和起点一样写「起点」）、seed、状态（`等待开始` / `训练 k/n 步 · 剩约 m 分`，从 ojh 的训练日志读 / `打包中（卡 i）` / `完成，未打包` / `完成` / `失败：<原因前 80 字>`）、卡、用时、验证集主指标；
- `notes`：STOP 的原因（`trsi next` 打印的同一句）和未打包试验的打包错误。

训练日志是 `<ojh 运行目录>/train-s<seed>.log`，步数的行形如 `train [###...] 22/32 (68.8%) loss … eta 0.1h`；读最后 64 KiB 里最后一个完整的这样的行（去掉 ANSI 颜色码；最后一个换行之后还没写完的部分不读），读不到就写「等待开始」。剩余时间：行里有 `s/step` 时按 (n − k) × s/step 算，否则用 `eta`。用时从最后一次启动算到 ojh 写出 `DONE` 的时间（还在跑时到现在）。
