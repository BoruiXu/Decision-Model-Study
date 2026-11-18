# onejev-harness

一个通用的训练 repo：固定版本的 OneJev + agent skill + 一层兼容代码。人或 agent（Claude Code、Codex）只用这个 repo，就能在任意单机 GPU 节点上把一份数据训练成 OneJev 决策模型，得到选好的 checkpoint、逐题预测、阈值、和基线的对比，以及一条完整的登记记录。设计见 `docs/design.md`。

支持的差异：数据格式（适配器）、GPU（卡型、卡数、显存、是否共享）、基座（`catalog/bases/`：joey-1、elephant-1、mammoth-1 和 OneJev 自带的 Qwen3.5 系列）、训练方式（全参或 LoRA）、每个人自己的路径和环境（机器配置）。

不做：多机训练、云平台、调度系统对接、数据生成和打标签、超参搜索、修改 OneJev、推理服务。

## 安装（在有 GPU 的机器上）

```bash
bash setup.sh [<machine.yaml>]      # 建 .venv，装固定版本的依赖（见 constraints.txt），再执行 ojh setup（拉 OneJev、打补丁、校验 sha256）
<repo>/.venv/bin/ojh fetch-base joey-1   # 下载固定 revision 的基座权重并校验（放到机器配置的 model_dir）
```

机器配置可选（见下文“配置”）：不给时 `ojh` 用 `<repo>/.venv/bin/python`、输出目录 `$OJH_OUT_ROOT`（不设则 `~/ojh_runs`）、`$OJH_MODEL_DIR`（不设则 `~/.cache/onejev-harness/models`），卡由 `--gpus` 指定。输出目录和模型目录不能在 repo 里（会报 `ConfigError`）；要放到别处，设 `OJH_OUT_ROOT`/`OJH_MODEL_DIR` 或传 `--machine`。共享机器上要加锁、排除卡、禁止写某些目录时，从 `examples/machine.example.yaml` 复制一份；`setup.sh <machine.yaml>` 用其中的 python 建 `.venv`，建好后把它的 python 改成 `<repo>/.venv/bin/python`（训练用的环境）。

`setup.sh` 不会激活 `.venv`，所以命令要写全路径 `<repo>/.venv/bin/ojh`（或先 `source <repo>/.venv/bin/activate`）；下文的 `ojh` 都是它的简写。agent 不在 GPU 机器上时，用 `ssh <host> <repo>/.venv/bin/ojh …` 调用；记录和登记表都留在 GPU 机器的 `out_root` 下。

## 使用流程（一次性）

用户给三样：**数据位置、GPU、基座**。认数据和写配置是 agent（skill）的事，脚本只做机械的事；代码不自动识别格式，也不自动选基座。

```bash
ojh inspect <数据位置> [--json]           # 1. 只读的事实：文件、行数、字段类型和填充率、取值、split 列、样例、是否已是 OneJev 行
# 2. agent 判断格式和字段，需要时写适配器（adapters/template.py 或一个 .py 文件），
#    从 examples/exp.template.yaml 复制出 <工作区>/exp.yaml（工作区放在 repo 之外）；
#    需要切分时：ojh make-split --out <工作区>/splits.json --group <组字段> <val 文件>
ojh run <工作区>/exp.yaml --base joey-1 --gpus 0   # 3. prepare → resolve → launch，立即返回 JSON（run_dir、tmux 会话、日志、DONE、基座、配方、卡、数据 sha256）
ojh status <RUN>                          # 4. 进度；结束后打印 RESULT.md 的路径
# 5. 读 <RUN>/RESULT.md（机器可读：result.json）：数据、设置、训练曲线、选中的 checkpoint、带 95% 置信区间的指标表、阈值、门和理由、警告、下一步命令
```

- `--base`、`--gpus` 覆盖或补上 `exp.yaml` 里的 `base`、`gpus`；没有基座就报错并列出 catalog。配方取 `--recipe`，否则 `exp.yaml` 的 `recipe`，否则基座的 `default_recipes`（joey-1 1 卡、elephant-1 2–8 卡、mammoth-1 2–8 卡）；没有对应卡数的默认值时报错并列出选项。
- `--gpus` 的每张卡都会检查：存在、空闲显存够（resolve 的每卡估算和 `min_free_mib` 取大）、列出卡上的进程；不满足就逐卡说明原因。resolve 判定不可行时退出码 2 并列出替代方案，不会自己换基座或配方。
- `ojh make-split`：`--mode select-report`（默认）把每个评测集按组分成 select/report，写成 `data.split_file`；`--mode holdout` 从一个训练文件按组切出 val，写到目录 `--out` 下的 `train.jsonl`、`val.jsonl`、`manifest.json`。确定性（默认 `--seed 20261115`），`--frac` 是 select（或 val）的比例；格式见 `ojh make-split --help` 和 `docs/design.md` 第 15 节。

`<RUN>` 是 run 目录（含 `run.lock.yaml`）。训练结束（成功或失败）时 runner 写 `<RUN>/DONE`：`exit_code`、`stage`（`verify`、`acquire`、`train`、`finalize`），失败时还有一行 `error`；每个 seed 实际执行的命令在 `<RUN>/command-s<seed>.json`。之后：

- `ojh note <RUN> "<一句话结论和依据>"`：填登记表里的 conclusion；`ojh table`：把登记表渲染成对比表。
- `ojh eval-report <RUN> --allow-report`：用锁定的 checkpoint 和阈值评 report 部分，必须显式加参数；带 `--allow-test` 才读 test（都要用户批准）。
- `ojh cleanup <RUN>`：列出未选中的 checkpoint 和大小；只有加 `--confirm` 才删（要用户批准）。

### 分步命令

`ojh run` 就是下面三步；需要在中间改 lock 的 `overrides` 或单独排查时分开跑：

```bash
ojh prepare <exp.yaml> [--machine <machine.yaml>]   # 转数据、校验、算 sha256，写到 <out_root>/<name>/prepared/
ojh resolve <exp.yaml> [--machine <machine.yaml>]   # 生成 <out_root>/<name>/<run_id>/run.lock.yaml（exp.yaml 要有 base、recipe、gpus）
ojh launch <RUN>/run.lock.yaml                      # 在 tmux 里训练，立即返回会话名、日志、完成标记
ojh finalize <RUN>                                  # 选 checkpoint、复评、基线、阈值、判门、登记、写 RESULT.md（launch 结束时自动调用）
```

## 配置

- **机器配置**（可选）：按顺序取 `--machine`、`$OJH_MACHINE`、`~/.config/onejev-harness/machine.yaml`，都没有就推断（见上）。模板 `examples/machine.example.yaml`（每个键都有说明）。`gpu_source`：`scan`（扫描加锁）、`preassigned`（用 `CUDA_VISIBLE_DEVICES`）、`explicit`（用 `--gpus`；推断时的默认值）。实际用的设置和来源记进 lock。`forbidden_write_roots` 里的目录不会被写：`prepare`（prepared 目录）、`resolve`（run 目录、登记表、`lock_dir`）、`fetch-base`（基座目录）和取卡加锁前都会检查，要写的路径在其下（含经符号链接）就报 `ConfigError`。
- **路径**：配置里的路径可以用 `~` 和 `$VAR`。相对路径按所在配置文件的目录解析：机器配置里的相对机器配置文件，实验配置里的（数据、评测集、`split_file`、适配器文件、配方、评测插件）相对实验配置文件；配方在那里找不到、但在 repo 根目录下存在时（如 `examples/recipes/…`），取 repo 根目录下的。解析后都是绝对路径，和在哪个目录下运行 `ojh` 无关。
- **实验配置**：模板 `examples/exp.template.yaml`（每个键都有注释），v1 复现的例子是 `examples/dynamicserve_v1_repro.yaml`。`data.split_file`（可选）把评测集的行分成 select 和 report：不写则所有行都算 select；写了则每个评测集都必须在其中，缺失会报错，集内未分配的 id 被丢弃并记录。每个评测集可以给 `train_labels`（选模算 NLL）和 `reference`（定阈值、AUC、门）两份标签。`evaluator.bootstrap_reps`（默认 1000，0 关闭）和 `bootstrap_seed` 控制置信区间。
- **适配器**：`data.adapter` 是 `adapters/` 里的模块名，或一个 `.py` 文件路径（相对实验配置）。`data.adapter_args`、`data.train_adapter_args`、`data.eval.<集>.adapter_args` 以 `adapt(path, **args)` 传入，一个带 split 列的文件可以同时作训练和评测来源。适配器文件的 sha256 记进 lock 和结果页，文件复制到 `<RUN>/adapter/`。
- **配方**：`examples/recipes/*.yaml`（`elephant_full_fsdp`、`joey_full_1gpu`、`joey_lora_1gpu`、`mammoth_lora_ddp`），必须显式写 `mode: full` 或 `mode: lora`。基座的 `default_recipes` 指向其中几份；LoRA 配方的超参没有实测。
- **基线**：`harness/baselines` 的注册表里有 `tfidf`、`group_prior`、`length`（只看文本长度的逻辑回归），缺字段的自动跳过；门要求模型在每个评测集上胜过可用基线中最好的一个。
- **lock**：`run.lock.yaml` 写下全部设置（训练方式、完整命令行参数、配方路径和 sha256、数据 sha256、适配器 sha256 和参数、评测标签和 split 文件的 sha256、基座和 OneJev 的 revision、环境指纹、机器设置及来源、卡、评测器配置、`overrides` 及原因、是否读了 test），执行时只读它。runner 取卡前、finalize 评测前都会核对 prepared 文件的 sha256；`resolve` 之后又跑过 `prepare` 时 run 直接失败（`DONE` 的 `stage` 是 `verify`），要重新 `resolve`。偏离配方只能写 `overrides: {键: {value: ..., reason: ...}}`，每项必须写原因。

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
