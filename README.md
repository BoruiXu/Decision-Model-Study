# onejev-harness

一个通用的训练 repo：固定版本的 OneJev + agent skill + 一层兼容代码。人或 agent（Claude Code、Codex）只用这个 repo，就能在任意单机 GPU 节点上把一份数据训练成 OneJev 决策模型，得到选好的 checkpoint、逐题预测、阈值、和基线的对比，以及一条完整的登记记录。设计见 `docs/design.md`。

支持的差异：数据格式（适配器）、GPU（卡型、卡数、显存、是否共享）、基座（`catalog/bases/`：joey-1、elephant-1、mammoth-1 和 OneJev 自带的 Qwen3.5 系列）、训练方式（全参或 LoRA）、每个人自己的路径和环境（机器配置）。

不做：多机训练、云平台、调度系统对接、数据生成和打标签、超参搜索、修改 OneJev、推理服务。

## 安装（在有 GPU 的机器上）

```bash
cp examples/machine.example.yaml <machine.yaml>   # 改成本机的路径和 GPU 规则
bash setup.sh <machine.yaml>        # 用其中的 python 建 .venv，装固定版本的依赖（见 constraints.txt），再执行 ojh setup（拉 OneJev、打补丁、校验 sha256）
# 建好后把 <machine.yaml> 的 python 指到 <repo>/.venv/bin/python（训练用的环境）
<repo>/.venv/bin/ojh fetch-base elephant-1 --machine <machine.yaml>   # 下载固定 revision 的基座权重并校验
```

`setup.sh` 不会激活 `.venv`，所以命令要写全路径 `<repo>/.venv/bin/ojh`（或先 `source <repo>/.venv/bin/activate`）；下文的 `ojh` 都是它的简写。agent 不在 GPU 机器上时，用 `ssh <host> <repo>/.venv/bin/ojh …` 调用；记录和登记表都留在 GPU 机器的 `out_root` 下。

## 使用流程

```bash
ojh prepare <exp.yaml> --machine <machine.yaml>   # 转数据、校验、算 sha256，写到 <out_root>/<name>/prepared/
ojh resolve <exp.yaml> --machine <machine.yaml>   # 生成 <out_root>/<name>/<run_id>/run.lock.yaml
ojh launch <RUN>/run.lock.yaml                    # 在 tmux 里训练，立即返回会话名、日志、完成标记
ojh status <RUN>                                  # 进度、最近一次评测、剩余时间、占用的卡
ojh finalize <RUN>                                # 选 checkpoint、复评、基线、阈值、判门、登记（launch 结束时自动调用）
ojh note <RUN> "<一句话结论和依据>"                  # 填登记表里的 conclusion
ojh table --machine <machine.yaml>                # 把登记表渲染成对比表
```

`<RUN>` 是 run 目录（含 `run.lock.yaml`）。训练结束（成功或失败）时 runner 写 `<RUN>/DONE`：`exit_code`、`stage`（`verify`、`acquire`、`train`、`finalize`），失败时还有一行 `error`；每个 seed 实际执行的命令在 `<RUN>/command-s<seed>.json`。其他命令：

- `ojh eval-report <RUN> --allow-report`：用锁定的 checkpoint 和阈值评 report 部分，必须显式加参数；带 `--allow-test` 才读 test（要用户批准）。
- `ojh cleanup <RUN>`：列出未选中的 checkpoint 和大小；只有加 `--confirm` 才删。

## 配置

- **机器配置**：每人每台机器一份，模板 `examples/machine.example.yaml`（每个键都有说明）。`forbidden_write_roots` 里的目录不会被写：`prepare`（prepared 目录）、`resolve`（run 目录、登记表、`lock_dir`）、`fetch-base`（基座目录）和取卡加锁前都会检查，要写的路径在其下（含经符号链接）就报 `ConfigError`。
- **路径**：配置里的路径可以用 `~` 和 `$VAR`。相对路径按所在配置文件的目录解析：机器配置里的相对机器配置文件，实验配置里的（数据、评测集、`split_file`、配方、评测插件）相对实验配置文件；配方在那里找不到、但在 repo 根目录下存在时（如 `examples/recipes/…`），取 repo 根目录下的。解析后都是绝对路径，和在哪个目录下运行 `ojh` 无关。
- **实验配置**：数据、字段、评测集、seed、卡数、`overrides`。v1 复现的例子是 `examples/dynamicserve_v1_repro.yaml`。`data.split_file`（可选）把评测集的行分成 select 和 report：不写则所有行都算 select；写了则每个评测集都必须在其中，缺失会报错，集内未分配的 id 被丢弃并记录。每个评测集可以给 `train_labels`（选模算 NLL）和 `reference`（定阈值、AUC、门）两份标签。
- **配方**：`examples/recipes/*.yaml`（`elephant_full_fsdp`、`joey_full_1gpu`、`joey_lora_1gpu`、`mammoth_lora_ddp`），必须显式写 `mode: full` 或 `mode: lora`。它们是示例，不是默认值；LoRA 配方的超参没有实测。
- **lock**：`run.lock.yaml` 写下全部设置（训练方式、完整命令行参数、数据 sha256、评测标签和 split 文件的 sha256、基座和 OneJev 的 revision、环境指纹、评测器配置、`overrides` 及原因、是否读了 test），执行时只读它。runner 取卡前、finalize 评测前都会核对 prepared 文件的 sha256；`resolve` 之后又跑过 `prepare` 时 run 直接失败（`DONE` 的 `stage` 是 `verify`），要重新 `resolve`。偏离配方只能写 `overrides: {键: {value: ..., reason: ...}}`，每项必须写原因。

## 原则

- 训练方式只由配方或带原因的 override 决定；放不下时 `resolve` 报不可行并列出替代方案，不会自动切换。
- test 部分和 report 部分默认不读（`--allow-test`、`--allow-report`）。train 文件里 split 以 `test` 开头的行同样要 `--allow-test` 才读。`prepare --allow-test` 会记进 `prepared.json` 和 lock，finalize 和 `eval-report` 沿用它，`RUN_CARD.md` 和登记记录里都写明读过 test。
- 不会自动删除 checkpoint；`ojh cleanup --confirm` 之前要先问用户。
- `ojh launch` 立即返回，不阻塞；用 `ojh status` 之后再查。

## 其他任务（非二分类）

用 `evaluator.type: plugin` 和 `evaluator.path` 指定评测插件，接口见 `plugins/README.md`。

## Agent 使用

`skills/training-with-onejev`（训练全流程、判断点、审批规则、故障表）和 `skills/adapting-data-for-onejev`（写适配器）。`AGENTS.md` 和 `CLAUDE.md` 指向它们；`.claude/skills/` 里是符号链接。

## 开发

```bash
.venv/bin/python -m pytest -q        # CPU 测试，不需要 torch
```
