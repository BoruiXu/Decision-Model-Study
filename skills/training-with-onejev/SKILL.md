---
name: training-with-onejev
description: Use when training a OneJev decision model (full fine-tune or LoRA) from a dataset on any single GPU node — preparing data, resolving a run lock, launching, monitoring, finalizing (checkpoint selection, thresholds, baselines, gate) and recording the result. 训练 OneJev 决策模型、全参或 LoRA、任意单机 GPU 节点。
---

# 用 OneJev 训练决策模型

本 repo 把一份数据训练成：选好的 checkpoint、逐题预测、阈值、和基线的对比、一条登记记录。
所有参数来自三处：**机器配置**（每人每台机器一份，模板见 `examples/machine.example.yaml`）、**实验配置**（示例见 `examples/dynamicserve_v1_repro.yaml`）、**lock**（`run.lock.yaml`，执行时只读它）。skill 只写流程和判断，不写任何取值。下面的 `<machine.yaml>`、`<exp.yaml>`、`<RUN>`、`<host>` 都由使用方给出。

## 在哪里运行

`ojh` 装在有 GPU 的机器上，在那里运行。你不在 GPU 机器上时，用 ssh 调用，例如 `ssh <host> ojh status <RUN>`。`status` 输出很短，通过 ssh 看很便宜。记录和登记表都留在 GPU 机器的 `out_root` 下。第一次使用先在 GPU 机器上执行 `bash setup.sh <machine.yaml>`（建环境，并执行 `ojh setup`：拉 OneJev、打补丁、校验 sha256），再 `ojh fetch-base <基座名> --machine <machine.yaml>`（基座在 `catalog/bases/`）。

## 流程

1. **prepare**：`ojh prepare <exp.yaml> --machine <machine.yaml>`。适配器把数据转成 OneJev 行，校验 schema、id 唯一、target 分布和 split 字段，按配置派生训练子集，并把 sha256 和实验配置里的 `data.train_sha256` 比对，不一致就停。输出写到 `<out_root>/<name>/prepared/`。数据格式不是 OneJev 行时，先用 `adapting-data-for-onejev` skill。
2. **resolve**：`ojh resolve <exp.yaml> --machine <machine.yaml>`。检测 GPU，结合数据、机器、基座、配方解析出 `<out_root>/<name>/<run_id>/run.lock.yaml`（命令会打印路径），并在目标机器上跑 plan-only，按每 epoch 步数的 1/8 设评测和保存间隔。放不下时退出码是 2，并在 stderr 列出替代方案（`alternative: …`）；harness 不会自动换方式。
3. **launch**：`ojh launch <RUN>/run.lock.yaml`。在 tmux 里依次取卡加锁、按 seed 训练、自动 finalize、写 `DONE` 完成标记（含退出码）。命令**立即返回**会话名、日志路径和完成标记路径。
4. **status**：`ojh status <RUN>`，输出当前 step / 总 step、最近一次评测、预计剩余时间和占用的卡。
5. **finalize**：launch 结束时自动调用；失败或想重跑时 `ojh finalize <RUN>`。它选 checkpoint、重载后逐题打分、跑训练基线（TF-IDF、分组先验、全选正类、oracle）、定阈值、出对比表并判门，写 `RUN_CARD.md` 并登记。
6. **写结论**：读 `<RUN>/RUN_CARD.md` 后，用 `ojh note <RUN> "<一句话结论和依据>"` 填登记表里的 `conclusion`。没有 `conclusion` 的记录算未完成，`ojh table` 默认不显示（加 `--all` 才显示）。`ojh table --machine <machine.yaml>` 把登记表渲染成对比表。

## 长任务规则（不要轮询）

`ojh launch` 返回后，**立即向用户回报会话名、日志路径和完成标记路径，然后停下**。不要写等待循环，不要隔几秒查一次。用户回来问或之后再需要时，才用 `ojh status <RUN>` 看一次；`DONE` 存在就看 `RUN_CARD.md`。

## 三个判断点（只有这三处需要你判断，其余都是确定性的）

1. **改 override**。需要偏离配方时，在实验配置的 `overrides` 里写，**每项都必须有原因**：
   ```yaml
   overrides:
     onejev.batch_tokens: {value: 8192, reason: "OOM at step 3, see runner.log"}
   ```
   键只能是 `mode`、`gpus`、`fsdp`、`onejev.<flag>`、`lora.<字段>`。`onejev.` 下不能写 harness 自己推导的标志（`full_finetune`、`fsdp`、`attention`、`readout`、`multimodal`、`eval_every`、`save_every`、`seed`、`name`、`train`、`eval`、`model`、`runs_dir` 等），要用对应的键。改完重新 `ojh resolve`，得到新的 lock；lock 里会记录每个 override 和原因。不要直接改 lock 里的参数，也不要在 lock 之外改任何东西。
2. **写适配器**：数据不是 OneJev 行时，见 `adapting-data-for-onejev`。
3. **按故障表处理**（见下）。

## 全参 vs LoRA

- 训练方式（`full` | `lora`）只来自配方（配方必须显式写 `mode`）或一条带原因的 override。harness 从不自动切换。
- 全参放不下时，`resolve` 会在 `Infeasible` 的替代方案里列出 LoRA，你可以向用户提议，但**切换前要先问用户**，且只能写成 `mode` override 并写明原因。LoRA 多卡只能用 DDP（每张卡放一份完整 bf16 基座），OneJev 的 FSDP 只支持全参。
- 同一组对比实验里，全参和 LoRA 的结果**分开看**，不能混在一起比；对比只在训练方式相同、数据和评测集相同的 run 之间做。

## 审批规则（先问用户，得到明确的“可以”再做）

- `ojh cleanup <RUN> --confirm`：先不带 `--confirm` 运行，列出候选和大小给用户看；只有用户同意才加 `--confirm`。没有 `--confirm` 不删任何东西。
- `ojh eval-report <RUN> --allow-report`，以及任何带 `--allow-test` 的命令（`prepare`、`eval-report`）：report 和 test 部分默认不读。用 `--allow-test` 要用户批准，并写进记录。
- 推送本 repo（`git push`）。
- 全参和 LoRA 之间的切换（见上）。

## 故障表

先看 `<RUN>/runner.log` 和 `ojh status <RUN>`，再对照：

| 现象 | 处理 |
|---|---|
| OOM（CUDA out of memory） | 先写 override `onejev.batch_tokens` 调低（例如减半），每项写原因；还不行再调低 `onejev.max_rows`。重新 `ojh resolve`，再 `ojh launch` |
| FLA kernel 报错 | 写 override `onejev.disable_fla_kernels: true`，写明原因；不能悄悄加标志 |
| NCCL 超时 | 先查这些卡上是否有别人的进程（`nvidia-smi`）；或用 `gpus` override 减少卡数后重新 resolve，写明原因 |
| plan-only 失败（resolve 阶段） | 多半是数据或适配器的问题：检查 `prepared/` 里的 `train.jsonl`、适配器输出，重新 `ojh prepare` |
| 等卡超时（`GpuWaitTimeout`） | 继续等，或调整机器配置里的 GPU 规则（`gpu_include` / `gpu_exclude` / `min_free_mib` / `wait_timeout_s`），再重新 launch |
| 复评 NLL 对不上（rescore 与训练时的 NLL 差超阈值） | **不要手改任何数字**。run 的状态是 `failed`，原因写在 `RUN_CARD.md`，如实向用户报告 |
| resolve 报不可行（`Infeasible`，退出码 2） | 按替代方案选：等卡、减卡、换更小的基座，或（问过用户后）写 `mode` override 换训练方式 |
| finalize 的门没过（状态 `failed`） | 记录照常写。如实报告：路由器没有超过 TF-IDF 或分组先验。不要为了过门改评测配置 |

## run 目录里有什么

`run.lock.yaml`（全部设置）、`runner.log`、`DONE`、`timing.json`、`checkpoint.sha256`、`predictions_select.jsonl`、`deletion_candidates.json`、`dropped.json`、`RUN_CARD.md`（一页摘要：选中的 checkpoint、复评 NLL、对照表、阈值、门、overrides、删除候选）。每个 seed 的 checkpoint 在 `runs/<实验名>-s<seed>/` 下。

## 不要做的事

- 在 lock 之外改参数而不记录；
- 读 test 或 report 部分而没有用户批准；
- 自动删 checkpoint；
- 放不下时自己切到 LoRA（或反过来）；
- 阻塞等训练结束或轮询。
