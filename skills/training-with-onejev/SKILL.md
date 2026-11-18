---
name: training-with-onejev
description: Use when a user wants a OneJev decision model, router or predictor trained on their data on a GPU node — they give a data location, GPU ids or a base (joey-1, elephant-1, mammoth-1, onejev-qwen3.5-*) — or asks about a run's progress, RESULT.md, OOM, an infeasible resolve, a DONE error, report/test evaluation or checkpoint cleanup. 训练 OneJev 决策模型、全参或 LoRA、查进度、看结果、清理 checkpoint。
---

# 用 OneJev 训练决策模型

## 概览

用户只给三样：**数据位置、GPU、基座**。agent 认数据、写实验配置（必要时写适配器），`ojh` 脚本做机械的事（转数据、解析、启动、选模、评测、登记），结束后 agent 读 `RESULT.md` 汇报。

**基座、卡、训练方式、标签含义都由用户决定。** agent 可以列选项、给建议，不能替用户选，也不能悄悄换。

## 在哪里运行

- `ojh` 在 GPU 机器上运行，写全路径 `<repo>/.venv/bin/ojh`（下文简写 `ojh`）。不在 GPU 机器上就 `ssh <host> <repo>/.venv/bin/ojh …`。
- 数据要在 GPU 机器上读得到；用户给的路径在哪台机器上不清楚时，在第 1 步一起问。
- 第一次用：按 `README.md` 的“环境配置”节（`source` 环境变量文件 → `bash setup.sh` → `ojh doctor` → `ojh fetch-base <基座>`，下载权重不需要审批）。`/home` 小的机器先把 `OJH_HOME` 指到大盘。
- 环境出问题（导入错误、CUDA 不可用、版本不对、目录写不了、没有 tmux）先跑 `ojh doctor`，按它的 hint 和 README 的故障排查表处理；要装系统软件或升级驱动时告诉用户，不要自己改环境。
- 机器配置可选，按优先级取：`--machine` > `$OJH_MACHINE` > `~/.config/onejev-harness/machine.yaml` > 自动推断。

## 主流程

1. **确认三样输入**，缺什么问什么，一次问完：
   - 数据位置（文件或目录）。
   - GPU 卡号（如 `0` 或 `2,3`）。没给就问，不要自己挑卡。
   - 基座。没给就列出 `<repo>/catalog/bases/` 里的基座和参数量（`params_b`），请用户选；**不要替用户选**。
   - 输出位置：run 写到机器配置的 `out_root`；没有机器配置时是 `$OJH_OUT_ROOT`，不设就是 `~/ojh_runs`。用户要放到别处（大盘、共享盘），设 `OJH_OUT_ROOT=<目录>` 或传 `--machine <machine.yaml>`。不能在本 repo 里（`ojh` 会拒绝）。
   - 同时告诉用户实验工作区的位置：默认 `~/onejev-work/<实验名>/`，`<实验名>` = 数据名-基座（数据是文件取文件名去后缀，是目录取目录名；如 `/data/qa.csv` + joey-1 → `qa-joey-1`），用户可以改。工作区**不在本 repo 里**；机器配置的 `forbidden_write_roots` 覆盖了它时，换个位置并告诉用户。
2. **认数据，写 `exp.yaml`**。**REQUIRED SUB-SKILL:** `adapting-data-for-onejev`。产出是工作区里的 `exp.yaml`，必要时还有 `adapter.py`、`splits.json`。用户对标签含义和判定问题的确认在这一步拿到。
3. **启动**：
   ```bash
   ojh run <工作区>/exp.yaml --base <基座> --gpus <i,j,...>
   ```
   - 它依次 prepare → resolve → launch，**立即返回**一段 JSON：`run_dir`、`session`、`log`、`done`（DONE 路径）、`result`（`RESULT.md` 将出现的位置），以及实际用的 `base`、`recipe`、`gpus`、`machine`。
   - 配方按 `--recipe` > `exp.yaml` 的 `recipe` > 基座 catalog 的 `default_recipes`（按卡数）取，agent 不挑。现有默认：joey-1 1 卡全参；elephant-1 2–8 卡全参 FSDP；mammoth-1 2–8 卡 **LoRA**（DDP）。其余基座（Qwen3.5 系列等）和不在范围内的卡数没有默认，`ojh run` 会报错并列出示例配方：把它们给用户，用户选定后才加 `--recipe`。
   - 选了 mammoth-1 时告诉用户默认是 LoRA；它的结果不能和全参的放在一起比。
   - 非零退出：把报错原因和列出的替代方案原样告诉用户，**问用户选哪个**。不要自己换基座、配方、卡或训练方式再跑。
4. **交回，不轮询**（见下节）。
5. **结束后读 `<run_dir>/RESULT.md`**（机器可读的在 `result.json`），按下面的汇报格式回报。
6. 用户认可结论后，`ojh note <run_dir> "<一句话结论和依据>"` 写进登记表。没有结论的记录 `ojh table` 默认不显示。

## 启动之后：交回，不轮询

- **被编排的子 agent**：把 `ojh run` 的 JSON 原样交回调用方，立即结束本轮。
- **独立运行**：把 `session`、`log`、`done`、`result` 路径告诉用户，然后停下。之后用户问起才 `ojh status <run_dir>` 看一次。
- 用户明确要你盯着时，也只偶尔看：两次 `ojh status` 之间至少隔 30 分钟，按它给的剩余时间定下次。
- 任何情况下都不写 `sleep`/`while`/`watch` 循环，不隔几秒查一次。
- `DONE` 存在就结束了：`exit_code` 为 0 读 `RESULT.md`；否则看 `stage` 和 `error`，查故障表。

## 汇报格式（读 RESULT.md 后）

1. 选中的 step 和 checkpoint；复评 NLL 是否对上。
2. 指标对比表：路由器 vs 基线（TF-IDF、分组先验、长度）和 oracle，每项带 bootstrap 置信区间，按评测集分列。置信区间大面积重叠时说“没有显著差别”，不说“更好”。
3. 阈值：各预算下的阈值、实际不一致率和承接率；最佳 Jaccard 阈值。
4. 门的结论（通过/未通过）和逐项理由：每个评测集的门条目带 `best`，即每项指标上最强的那条基线；说清模型比的是哪条、差多少。
5. `RESULT.md` 里的所有警告，原样列出。
6. 下一步建议（例如是否值得跑 report）。跑 report/test 要先问用户。

## 审批点（先问，得到明确的“可以”再做）

| 动作 | 说明 |
|---|---|
| `ojh eval-report <RUN> --allow-report` | 读 report 部分 |
| 任何带 `--allow-test` 的命令 | 读 test；会记进 lock、`RESULT.md` 和登记表 |
| `ojh run … --allow-shared` | 和卡上已有的进程（常是别人的）共用卡；会记进 lock。先把进程列表（pid、用户、显存）给用户看 |
| `ojh cleanup <RUN> --confirm` | 先不带 `--confirm` 跑，把候选和大小给用户看 |
| 任何 override（含换卡数、改 batch、全参↔LoRA） | 见下节 |
| 换基座、换配方（`--recipe`）、换卡 | 用户给的输入，只有用户能改 |
| `git push` 本 repo | |

## 改设置：只能写 override，带原因

```yaml
overrides:
  onejev.batch_tokens: {value: 8192, reason: "OOM at step 3, see runner.log"}
```

- 键只能是 `mode`、`gpus`、`fsdp`、`onejev.<flag>`、`lora.<字段>`；harness 自己推导的标志（`full_finetune`、`fsdp`、`attention`、`readout`、`eval_every`、`save_every`、`seed`、`model` 等）不能写在 `onejev.` 下。
- 冒烟短跑：`onejev.max_steps`（评测间隔仍按完整 epoch 算，中途只评一次）。
- 写好后重新 `ojh run`，得到新 lock。**不要改 `run.lock.yaml`**，执行时只读它。
- 全参放不下时，替代方案里会有 LoRA；向用户提议，同意后才写 `mode` override。全参和 LoRA 的结果分开比，不混在一张对比里。

## 故障表

先看 `DONE`（`exit_code`、`stage`、`error`）、`<run_dir>/runner.log`、`ojh status <run_dir>`。

| 现象 | 处理 |
|---|---|
| `ojh run` 报 GPU 校验失败（空闲显存不够，或卡上已有计算进程：默认拒绝，列出 pid、用户、显存） | 把原因和进程列表原样告诉用户，问：换哪几张卡、等一等、减卡，还是共用这张卡。只有用户明确同意共用，才加 `--allow-shared` 重跑。不碰别人的进程 |
| `ojh run` 报不可行（Infeasible），列出替代方案 | 列给用户选（等卡、加卡、换小基座、LoRA）。不自己选 |
| 训练中 OOM，或另一位用户中途占了卡（`nvidia-smi` 里有别人的进程） | 被占卡：告诉用户，问换卡还是等。真 OOM：提议 override `onejev.batch_tokens` 减半（再不行 `onejev.max_rows`），用户同意后重跑 |
| plan-only 失败 | 多半是数据或适配器的问题：回 `adapting-data-for-onejev` 自测，修好后重跑 |
| prepare 报 `data.train_sha256` 不一致 | 数据文件变了。告诉用户，问是否是新数据版本；确认后才更新 `train_sha256` |
| `DONE` 的 `stage: verify`，prepared 的 sha256 和 lock 对不上 | 启动后又有人 prepare 了同名实验。重新 `ojh run`，不要改 lock |
| `DONE` 的 `stage: acquire`（卡被别人锁住、复查显存不够或等卡超时） | 告诉用户原因，问等一等还是换卡 |
| `ojh run` 报没有默认配方 | 把列出的示例配方给用户选，选定后 `--recipe` |
| `DONE` 的 `stage: train` | 看 `runner.log` 末尾：OOM / FLA kernel（提议 override `onejev.disable_fla_kernels: true`）/ NCCL 超时（查别人的进程，或提议减卡） |
| `DONE` 的 `stage: finalize`；复评 NLL 对不上 | 不手改任何数字，如实报告；可以 `ojh finalize <run_dir>` 重跑收尾 |
| 门没过 | 记录照常写。如实说路由器没超过哪条基线（门条目的 `best`）；不改评测配置去凑 |
| 基座权重不存在 | `ojh fetch-base <基座>` 后重跑 |
| `ojh` 没有 `run`/`inspect`/`make-split` 子命令，或找不到 `examples/exp.template.yaml` | GPU 机器上的 repo 版本旧。告诉用户更新并重装；不要自己补代码，也不要退回分步命令拼流程 |

## 常见错误

| 错误 | 正确做法 |
|---|---|
| 用户没说基座，agent 挑了一个“合适的” | 列出 catalog，请用户选 |
| GPU 校验失败后自己换了别的卡或减了卡数 | 报原因，问用户 |
| 不可行时自己切到 LoRA | 提议，用户同意后写 `mode` override |
| `ojh run` 之后 `sleep 600` 再查，或循环查 `status` | 交回 JSON 或告诉用户路径，停下 |
| 只报路由器 AUC，不报基线和置信区间 | 按汇报格式：对比基线、oracle，带 CI |
| OOM 后直接改 `run.lock.yaml` | 写带原因的 override，问用户，重跑 |
| 为了看效果顺手跑了 `eval-report` 或 `--allow-test` | 先问 |
| 把工作区、`exp.yaml`、适配器放进本 repo | 放在工作区（repo 之外） |
| 清理磁盘时直接 `cleanup --confirm` | 先列候选，问用户 |

## 进阶：分步命令

`ojh run` 包了 `prepare`、`resolve`、`launch`；要单独重做某一步时用它们（`ojh <cmd> --help`）。`finalize` 可单独重跑；`ojh table` 渲染登记表对比。
