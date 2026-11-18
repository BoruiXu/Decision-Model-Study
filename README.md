# onejev-harness

一个通用的训练 repo：固定版本的 OneJev + agent skill + 一层兼容代码。人或 agent（Claude Code、Codex）只用这个 repo，就能在任意单机 GPU 节点上把一份数据训练成 OneJev 决策模型，得到选好的 checkpoint、逐题预测、阈值、和基线的对比，以及一条完整的登记记录。设计见 `docs/design.md`。

支持的差异：数据格式（适配器）、GPU（卡型、卡数、显存、是否共享）、基座（`catalog/bases/`：joey-1、elephant-1、mammoth-1 和 OneJev 自带的 Qwen3.5 系列）、训练方式（全参或 LoRA）、每个人自己的路径和环境（机器配置）。

不做：多机训练、云平台、调度系统对接、数据生成和打标签、超参搜索、修改 OneJev、推理服务。

## 环境配置

环境由 `setup.sh` 用 [uv](https://docs.astral.sh/uv/) 一次建好：uv 本身、Python 3.11、`.venv`、固定版本的依赖、OneJev。机器上不需要预先有 Python 3.11、conda 或 uv，也不需要 root。

### 前提

| 项目 | 要求 |
|---|---|
| 系统 | Linux x86_64（在 Ubuntu 22.04 上验证过） |
| GPU 驱动 | NVIDIA 驱动支持 CUDA 12.8 的 wheel（R570 或更新；`nvidia-smi` 右上角的 CUDA Version ≥ 12.8）。不需要单独装 CUDA toolkit |
| 工具 | `git`、`tmux`（训练在 tmux 里跑）、`curl`（第一次装 uv） |
| 网络 | 能访问 astral.sh 和 GitHub（uv、Python、OneJev）、PyPI 和 download.pytorch.org（依赖）、huggingface.co（补丁和基座权重） |
| 磁盘 | 环境 5.5–11 GB：依赖包（torch 和 CUDA 库等）约 5 GB，加 Python 80 MB、OneJev 100 MB。这 5 GB 在 uv 缓存和 venv 之间怎么分看文件系统：两者在同一文件系统时 uv 用硬链接，只占一份，`du` 把它算在先统计的目录里（A100 实测：缓存 5.2 GB、venv 50 MB）；不在同一个文件系统，或缓存被清理过时，venv 自己约 5 GB，缓存从几十 MB（另一台机器实测 95 MB）到 5 GB 不等。按最坏情况预留约 11 GB；缓存和 venv 不共用时，`uv cache clean`（用同样的 `UV_CACHE_DIR`）可以收回缓存，不影响 venv。基座权重 elephant-1 约 18 GB、joey-1 约 5 GB；每个 run 的 checkpoint 另算（全参每个 checkpoint 约为基座大小，LoRA 小得多） |

### 步骤

```bash
export OJH_HOME=/big/disk/ojh              # 1. 一块大盘上的目录（不要放在 /home）
mkdir -p "$OJH_HOME" && cp examples/env.example.sh "$OJH_HOME/ojh-env.sh"
                                            #    把文件里 OJH_HOME 那一行改成同一个目录
source /big/disk/ojh/ojh-env.sh             #    之后每个要跑 ojh 的 shell 都用完整路径 source 一次
bash setup.sh --dry-run                     # 2. 先看计划：每个目录放在哪、剩余空间、要做哪些步骤（什么都不改）
bash setup.sh                               #    再执行；重跑会跳过已完成的步骤
<repo>/.venv/bin/ojh doctor                 # 3. 只读检查：Python、torch/CUDA、GPU、依赖版本、OneJev、tmux、各目录、HF hub
<repo>/.venv/bin/ojh fetch-base joey-1      # 4. 下载固定 revision 的基座权重并校验（到 OJH_MODEL_DIR）
```

在 A100 节点（Ubuntu 22.04，系统 Python 3.10，无 conda）上从零跑完约 4 分钟，重跑约 1 秒；`/home` 下不写任何文件。新开的 shell 里还没有 `OJH_HOME`：它由 `ojh-env.sh` 本身设置 `OJH_HOME`（以及其余变量），所以 source 时写文件的完整路径（上例的 `/big/disk/ojh/ojh-env.sh`），不要写 `$OJH_HOME/…`。文件和数据都在大盘上，不占 home。

`setup.sh` 的五步：① uv，版本固定（`setup.sh` 里的 `UV_VERSION`，现在是 0.13.9）：PATH 上的 uv 正好是这个版本才用，否则用 `UV_UNMANAGED_INSTALL` 把固定版本装到 `$OJH_TOOLS_DIR/uv`（不改 shell 配置文件，不往 `~/.local/bin` 放东西）；② `uv python install 3.11` 到 `$UV_PYTHON_INSTALL_DIR`；③ `uv venv` 建 `.venv`（`OJH_VENV` 在别处时，`<repo>/.venv` 是指向它的符号链接）；④ 先从 CUDA 12.8 的 wheel 源装 `torch==2.8.0`、`torchvision==0.23.0`，再按 `constraints.txt` 装其余固定版本，最后 `uv pip check`；⑤ `ojh setup`：拉 OneJev 到固定 revision、打 `serve.patch`、逐项校验 sha256；最后把 uv 版本、Python 版本和 torch 源写进 venv 的 `.ojh-setup.json`，`ojh resolve` 把它记进 lock 的 `fingerprint.setup`，`ojh doctor` 核对 uv 版本。已有的 Python ≥3.11 可以用 `--python <路径>` 显式指定（opt-in）；`--machine <machine.yaml>` 传给 `ojh setup`。

### 放到大盘上

`examples/env.example.sh` 把所有会写的位置都放在 `OJH_HOME` 下（每行都有注释，已设置的变量不覆盖）：

| 变量 | 内容 | 不设时 |
|---|---|---|
| `OJH_TOOLS_DIR` | uv 本身 | `<repo>/.tools` |
| `UV_CACHE_DIR` | uv 的下载和 wheel 缓存 | `$OJH_TOOLS_DIR/cache/uv` |
| `UV_PYTHON_INSTALL_DIR` | uv 管理的 Python 3.11 | `$OJH_TOOLS_DIR/python` |
| `OJH_VENV` | 虚拟环境 | `<repo>/.venv` |
| `HF_HOME` | Hugging Face 缓存 | setup 时 `$OJH_TOOLS_DIR/hf`，之后 `~/.cache/huggingface` |
| `TMPDIR` | 临时文件 | setup 时 `$OJH_TOOLS_DIR/tmp`，之后系统默认 |
| `OJH_OUT_ROOT` | run 的输出和登记表（没有机器配置时） | `~/ojh_runs` |
| `OJH_MODEL_DIR` | 基座权重（没有机器配置时） | `~/.cache/onejev-harness/models` |
| `OJH_WORK_ROOT` | 实验工作区（每个实验一个目录：`exp.yaml`、适配器、`splits.json`），skill 默认放这里 | `$OJH_HOME/work`，没有 `OJH_HOME` 时 `~/onejev-work` |

`/home` 快满时一定先 `source` 这份文件再跑 `setup.sh`：表里“不设时”在 `~` 下的几项会写到 home。`OJH_OUT_ROOT`、`OJH_MODEL_DIR` 不能在 repo 里。

### 离线或网络慢

- `setup.sh` 要联网；重跑会用 `UV_CACHE_DIR` 里的缓存，已完成的步骤直接跳过，所以中断后重跑即可。
- 网络慢时可以先在能联网的机器上用同样的 `OJH_HOME` 布局跑一次，再把 `OJH_HOME`（含 uv 缓存）和 repo 一起拷过去重跑 `setup.sh`。
- 训练本身不需要网络：runner 设 `HF_HUB_OFFLINE=1`。基座权重可以在别处 `ojh fetch-base` 后拷到 `OJH_MODEL_DIR/<基座名>`；`ojh doctor --no-network` 跳过网络检查。

### 故障排查

| 现象 | 处理 |
|---|---|
| 没有 Python 3.11（系统只有 3.10、没有 conda） | 不用管：`setup.sh` 用 uv 装 Python 3.11，不需要系统 Python |
| `/home` 满了 | `source examples/env.example.sh`（`OJH_HOME` 指到大盘）后重跑 `setup.sh`；`ojh doctor` 列出每个目录的剩余空间。之前没设变量时留下的 `~/.cache/uv` 可以删 |
| CUDA 和驱动不匹配（`ojh doctor` 的 cuda 一项 FAIL，`torch.cuda.is_available()` 为 False） | 看 `nvidia-smi` 的驱动版本：CUDA 12.8 的 wheel 要 R570 或更新。驱动太旧要请管理员升级；也检查 `CUDA_VISIBLE_DEVICES` 有没有设成空 |
| flash-linear-attention 导入或编译出错 | `ojh doctor` 核对版本是否是固定的 0.5.2；不是就重跑 `setup.sh`。训练时 FLA kernel 报错按 skill 的故障表加 `--disable-fla-kernels` override（要问用户）。Triton 编译缓存放在 `TRITON_CACHE_DIR` |
| 依赖解析冲突（`uv pip install` 或 `uv pip check` 报错） | 不要手动改版本：确认 `constraints.txt`、`pyproject.toml` 没被改过，删掉 `$OJH_VENV`（或 `<repo>/.venv`）后重跑 `setup.sh` |
| 没有 tmux | `ojh launch`/`ojh run` 需要它：`apt install tmux`，或请管理员安装 |
| uv 装到了 `~/.local/bin` 或改了 `.bashrc` | 那是直接跑 uv 官方安装脚本的结果；`setup.sh` 用 `UV_UNMANAGED_INSTALL` 不会这样。可以删掉 `~/.local/bin/uv`、`uvx` 和配置文件里加的那行 |

### 用法约定

`setup.sh` 不会激活 `.venv`，所以命令要写全路径 `<repo>/.venv/bin/ojh`（或先 `source <repo>/.venv/bin/activate`）；下文的 `ojh` 都是它的简写。agent 不在 GPU 机器上时，用 `ssh <host> 'source /big/disk/ojh/ojh-env.sh && <repo>/.venv/bin/ojh …'` 调用：非交互的 ssh 不读你的 shell 配置，环境变量文件要在同一条命令里 source；记录和登记表都留在 GPU 机器的 `out_root` 下。很多服务器会拒绝短时间内反复新建的 ssh 连接，所以复用一条：`ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=2h <host> 'source /big/disk/ojh/ojh-env.sh && <repo>/.venv/bin/ojh …'`（或把这几个选项写进 `~/.ssh/config`）。

机器配置可选（见下文“配置”）：不给时 `ojh` 用 `<repo>/.venv/bin/python`、输出目录 `$OJH_OUT_ROOT`（不设则 `~/ojh_runs`）、`$OJH_MODEL_DIR`（不设则 `~/.cache/onejev-harness/models`），卡由 `--gpus` 指定。输出目录和模型目录不能在 repo 里（会报 `ConfigError`）；要放到别处，设 `OJH_OUT_ROOT`/`OJH_MODEL_DIR` 或传 `--machine`。共享机器上要加锁、排除卡、禁止写某些目录时，从 `examples/machine.example.yaml` 复制一份（其中的 `python` 写 `<repo>/.venv/bin/python`）。

## 使用流程（一次性）

用户给三样：**数据位置、GPU、基座**。认数据和写配置是 agent（skill）的事，脚本只做机械的事；代码不自动识别格式，也不自动选基座。

```bash
ojh inspect <数据位置> [--json]           # 1. 只读的事实：文件、行数、字段类型和填充率、取值、split 列、样例、是否已是 OneJev 行
# 2. agent 判断格式和字段，需要时写适配器（adapters/template.py 或一个 .py 文件），
#    从 examples/exp.template.yaml 复制出 <工作区>/exp.yaml（工作区放在 repo 之外）；
#    需要切分时：ojh make-split --out <工作区>/splits.json --group <组字段> <val 文件>
ojh check <工作区>/exp.yaml --base joey-1 --gpus 0 # 可选：不看卡的全部检查（prepare、配置、适配器、plan-only、环境指纹），卡忙时也能先查实验
ojh run <工作区>/exp.yaml --base joey-1 --gpus 0   # 3. prepare → resolve → launch，立即返回 JSON（run_dir、tmux 会话、日志、DONE、基座、配方、卡、数据 sha256）
ojh status <RUN>                          # 4. 进度；结束后打印 RESULT.md 的路径
# 5. 读 <RUN>/RESULT.md（机器可读：result.json）：数据、设置、训练曲线、选中的 checkpoint、带 95% 置信区间的指标表、阈值、门和理由、警告、下一步命令
```

- `--base`、`--gpus` 覆盖或补上 `exp.yaml` 里的 `base`、`gpus`；没有基座就报错并列出 catalog。配方取 `--recipe`，否则 `exp.yaml` 的 `recipe`，否则基座的 `default_recipes`（joey-1 1 卡、elephant-1 2–8 卡、mammoth-1 2–8 卡）；没有对应卡数的默认值时报错并列出选项。
- `--gpus` 的每张卡都会检查：存在、空闲显存够（resolve 的每卡估算和 `min_free_mib` 取大）、卡上没有其他计算进程；不满足就逐卡说明原因，并列出卡上进程的 pid、用户和显存。卡上已有进程（常是别人的）时默认拒绝；确认可以共用后加 `--allow-shared`，它会记进 lock。`ojh finalize`、`ojh eval-report` 沿用 lock 的选择（更早的 lock 没有这个键，照旧允许共用）；run 之后卡被别人占了时，用户同意后这两个命令也可以加 `--allow-shared`。这两个命令在前台运行，只检查一次卡、从不阻塞：卡不满足时立刻报出每张卡的问题和可以直接运行的下一条命令；要等卡就显式加 `--wait`（每 30 秒查一次，最长 `wait_timeout_s`）。只有 tmux 里的 runner（包括它自动调用的 finalize）会等卡。resolve 判定不可行时退出码 2 并列出替代方案，不会自己换基座或配方。
- `ojh make-split`：`--mode select-report`（默认）把每个评测集按组分成 select/report，写成 `data.split_file`；`--mode holdout` 从一个训练文件按组切出 val，写到目录 `--out` 下的 `train.jsonl`、`val.jsonl`、`manifest.json`。确定性（默认 `--seed 20261115`），`--frac` 是 select（或 val）的比例；select-report 用 `--id <点分字段>`（默认 `id`）读每行的 id，按文本写入，适配器要产出同样的文本作行 id，所以原始 csv/jsonl 可以直接切；格式见 `ojh make-split --help` 和 `docs/design.md` 第 15 节。

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

- **机器配置**（可选）：按顺序取 `--machine`、`$OJH_MACHINE`、`~/.config/onejev-harness/machine.yaml`，都没有就推断（见上）。模板 `examples/machine.example.yaml`（每个键都有说明）。`gpu_source`：`scan`（扫描加锁，等到卡满足 `min_free_mib` 且没有别人的进程，最长 `wait_timeout_s`；共用卡要 `--allow-shared`）、`preassigned`（用 `CUDA_VISIBLE_DEVICES`）、`explicit`（用 `--gpus`；推断时的默认值）。实际用的设置和来源记进 lock。`forbidden_write_roots` 里的目录不会被写：`prepare`（prepared 目录）、`resolve`（run 目录、登记表、`lock_dir`）、`fetch-base`（基座目录）和取卡加锁前都会检查，要写的路径在其下（含经符号链接）就报 `ConfigError`。
- **路径**：配置里的路径可以用 `~` 和 `$VAR`。相对路径按所在配置文件的目录解析：机器配置里的相对机器配置文件，实验配置里的（数据、评测集、`split_file`、适配器文件、配方、评测插件）相对实验配置文件；配方在那里找不到、但在 repo 根目录下存在时（如 `examples/recipes/…`），取 repo 根目录下的。解析后都是绝对路径，和在哪个目录下运行 `ojh` 无关。
- **实验名**：`name` 决定 `<out_root>/<name>/prepared/`。另一份实验文件用过同一个名字时，prepare（以及 `ojh run`、`ojh check`）拒绝覆盖它的 prepared 数据（`prepared.json` 记下实验文件路径），换个名字，或确认后加 `--force`。
- **实验配置**：模板 `examples/exp.template.yaml`（每个键都有注释）；复现 DynamicServe v1 elephant 的完整例子是 `examples/dynamicserve_v1_repro.yaml`（路径用占位符，`docs/design.md` 第 7 节逐键说明）。`data.split_file`（可选）把评测集的行分成 select 和 report：不写则所有行都算 select；写了则每个评测集都必须在其中，缺失会报错，集内未分配的 id 被丢弃并记录。每个评测集可以给 `train_labels`（选模算 NLL）和 `reference`（定阈值、AUC、门）两份标签。`evaluator.bootstrap_reps`（默认 1000，0 关闭）和 `bootstrap_seed` 控制置信区间。
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
