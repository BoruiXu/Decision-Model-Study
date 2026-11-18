# OneJev 训练 Harness 设计

> - 版本：v1 设计（2026-11-17）；第 15 节一次性流程（2026-11-18）。
> - 本文件描述 harness 的设计；实现见 `harness/`，验收结果见 `docs/acceptance.md`。

## 1. 为什么做

决策模型的训练目前由 OneJev 完成。外面的流程是：每做一个实验，agent 就现写一套 shell 脚本。v1 一共写了 `command_*.sh` 4 份、`score_select*.sh` 4 份，再加上 `gate.py`、`make_splits.py`。这些脚本的路径都写死在个人账号下，于是：

- 每次实验的做法都可能不一样；
- 只有一个账号能跑；
- 实验记录要靠人或 agent 记得去写；
- 只跑过一个 seed，GPU 时间没有记录。

训练是决策模型闭环里最固定的一段，适合做成通用组件：任何人用自己的 agent（Claude Code 或 Codex）打开这个 repo，给它一份数据，就能得到可复现的训练结果和完整记录。以后其他决策模型（Agent Safety、金融、Kernel 等）也用这个 repo 训练。

## 2. 目标、非目标、反目标

**目标**
1. 一个独立的通用 repo，由三部分组成：OneJev（固定版本）、agent skill，以及一层兼容代码。agent 只用这个 repo，就能把一份数据训练成：
   - 选好的 checkpoint；
   - 逐题预测；
   - 阈值；
   - 和基线的对比结果；
   - 一条完整的记录。
2. 兼容以下几方面的差异：
   - 数据：格式和字段不同；
   - GPU：卡型、卡数、显存不同，是否与他人共享；
   - 基座：joey-1、elephant-1、mammoth-1，以及 OneJev 自带的 Qwen3.5 系列；
   - 训练方式：全参或 LoRA；
   - 人：每人有自己的路径和环境。
3. 同一份锁定配置重跑，得到同样的设置和可比的结果。每个 run 自动生成一条完整的登记记录。
4. 在训练侧完成选模、定阈值、和基线对比并判门，不依赖其他工具。
5. 复现 v1 elephant 的结果（第 13 节 A1），证明和旧流程等价。

**非目标（v1 不做）**
- 多机训练；Modal 等云平台；对接 Slurm、K8s 等调度系统；
- 数据生成、打标签、重新切分评测集（只读已有的数据版本，可以按配置派生训练子集）；
- 超参搜索；只支持“一份实验配置展开成多个 seed”；
- 修改 OneJev 本身；推理服务。

**反目标（即使能跑也算失败）**
- agent 在 lock 之外改了参数，而记录里看不出来；
- 默认会读 test 或 report 部分；
- 自动删除 checkpoint；
- 放不下全参时自动改成 LoRA，或者反过来；训练方式只能由配方或带原因的 override 决定；
- 代码里写死某个人的路径、某台机器的规则或某个项目的取值；
- 阻塞等训练结束，让 agent 守着白耗 token。

## 3. 约束

- **OneJev 版本**：`github.com/OmniJev/OneJev@81ce62f1597c91e46767d4d02d6ac2e18534fe94`，再打上 baselabs 发布的 `serve.patch`（第 5 节）。
- **环境**：Python 3.11、torch 2.8.0+cu128、transformers 5.19.0、peft 0.21.2、flash-linear-attention 0.5.2。来源是现有的 `training/{setup.sh,constraints.txt,recorded_environment.json}`。
- **机器规则一律可配置，repo 里不写具体取值**。可配置的项目包括：
  - 可用 GPU 列表或排除列表；
  - 最小空闲显存；
  - 锁目录（可选）；
  - 取卡顺序和等卡超时；
  - 输出根目录；
  - 禁止写入的目录；
  - Python 环境和模型目录。

  以 DynamicServe 的共享 A100 为例，取值是：锁 `~/DynamicServe/gpu_locks/<UUID>.lock`，排除 GPU 3，空闲显存 ≥ 48 GB，输出写 `/shared/ssd`，禁止写 `/home`。这份配置由使用方自己保存，调用时传入。
- **任何机器上都不变的规则**（写进代码）：
  - 只用自己拿到锁或分配到的卡；
  - 不碰其他用户的进程。
- **代码位置**：只在新 repo 里写代码，不改动现有项目的任何文件。

## 4. 实现方式

| | 做法 | 取舍 |
|---|---|---|
| A. 脚本包办 | 一个 driver 从头跑到尾，skill 只做导航 | 复现性最强；但碰到新卡型、新数据就要改 driver 代码，agent 无法适配；前期代码最多 |
| B. skill 为主 | skill 写决策过程，脚本只提供原子工具，由 agent 组合 | 最灵活、最快能做出来；但每次运行的做法可能不同，难以复现和审计 |
| **C. 先解析再执行（已选）** | 脚本把数据、机器、基座、配方确定性地解析成 `run.lock.yaml`；agent 只能在固定的几处介入（见下）；执行时只读 lock | 兼顾复现和适配：lock 既是复现记录，也是 agent 唯一能改参数的地方；代价是要先写好画像表 |

C 中 agent 可以介入的三处：
- 改 lock 里的 `overrides`，并写明原因；
- 为新数据格式写适配器；
- 按故障表处理报错。

## 5. OneJev 版本

**补丁做了什么**：`serve.patch` 改了 6 个文件，新增 486 行、删除 47 行，加了三样东西：
- `--attention noncausal`：在全注意力层里，前缀块内部、问题块内部各自双向注意。前缀块看不到问题块，所以前缀仍然可以共享缓存。GatedDeltaNet 层保持因果。
- `--readout head`：可选的 slot + pointer 读出头。
- 读出和注意力设置随权重一起保存（`readout.json`，以及 `config.json` 里的 `modeling_overrides`），推理时按同样的设置运行。

不加这两个参数时（默认 `causal` + `slot_lm`），行为和原版一样。

**为什么必须用这个版本**：
- joey-1、elephant-1、mammoth-1 的模型卡都写明用了块内双向的注意力掩码；`readout.json` 都是 `slot_lm`，都没用读出头。
- 三个模型 repo 里的 `setup.sh` 都固定了 “OneJev @81ce62f + serve.patch”，原注释是 "pinned as in our runs"。公开页面都没有写训练用的代码版本，这是能拿到的、和这几个模型一致的版本。
- 上游 main 截至 2026-11-16（`874a8a5`）仍然没有这些功能。

**做法**：
- repo 里只放一个 `onejev.lock`，写明 OneJev 的 repo 和 revision、补丁来源（HF `baselabs/joey-1` 的固定 revision 和 `serve.patch` 的 sha256），以及打完补丁后各文件的 sha256。
- `ojh setup` 负责 clone、checkout、打补丁、逐项校验，做法沿用现有的 `training/fetch.py`。
- OneJev 代码不提交进 repo，也不跟随上游更新。

**实现时要核实**：elephant-1 和 mammoth-1 仓库里的 `serve.patch` 是否与 joey-1 的完全相同（文件大小相同，内容还没比对）。如果不同，以所训基座对应的那份为准，并记进 lock。

## 6. repo 结构

repo 是一个独立的 git 仓库，不依赖 DynamicServe 的代码。

```
onejev-harness/
  README.md  AGENTS.md  CLAUDE.md        # 后两个指向 skills/，Codex 和 Claude Code 都能找到
  skills/
    training-with-onejev/SKILL.md        # 主 skill：流程、判断点、故障表、审批规则
    adapting-data-for-onejev/SKILL.md    # 新数据格式怎么写适配器
  .claude/skills/ -> ../skills/*         # 符号链接
  onejev.lock                            # 第 5 节
  setup.sh  constraints.txt
  catalog/bases/                         # 公开模型的事实信息：repo、revision、sha256、尺寸、注意力和读出设置
    joey-1.yaml  elephant-1.yaml  mammoth-1.yaml  onejev-qwen3.5-{0.8b,4b,9b,27b}.yaml
  examples/
    machine.example.yaml                 # 机器配置模板，每个键都有注释
    recipes/elephant_full_fsdp.yaml      # 示例配方，不是默认值
    recipes/joey_full_1gpu.yaml
    recipes/joey_lora_1gpu.yaml
    recipes/mammoth_lora_ddp.yaml
    dynamicserve_v1_repro.yaml           # 复现 v1 的实验配置，路径用占位符
  harness/                               # Python 包，命令行入口 ojh
    cli  prepare  resolve  gpu  launch  train_entry  status  finalize  score
    evaluator/binary_decision  baselines/  registry  cleanup
  adapters/  onejev_rows.py（直通加校验）  template.py
  plugins/README.md                      # 其他任务的插件接口
  tests/
  docs/design.md
```

repo 里不放任何项目的机器配置、数据路径或任务规则，这些都由使用方提供。

## 7. 运行位置、命令和实验配置

**在哪里运行**：
- `ojh` 装在有 GPU 的机器上，在那里运行。
- agent 不在 GPU 机器上时，用 `ssh <主机> ojh …` 调用，skill 里写明这种用法。v1 不做远程模式。
- `status` 的输出很短，通过 ssh 查看的成本很低。
- 记录和登记表都留在 GPU 机器的 `out_root` 下。登记表路径可以配置，默认是 `<out_root>/registry.jsonl`；多人共用一份时，把它指到大家都能写的目录。

**命令**

| 命令 | 做什么 | 由谁决定 |
|---|---|---|
| `ojh inspect PATH` | 只读：列出数据文件的事实（第 15 节） | 确定性，不做决定 |
| `ojh make-split …` | 按组确定性地切 select/report，或从 train 切出 val（第 15 节） | 确定性 |
| `ojh run EXP --base B --gpus i,j` | prepare → resolve → launch 一次完成，立即返回（第 15 节） | 确定性；基座和卡由用户给 |
| `ojh setup --machine M` | 建环境，拉 OneJev 并打补丁，逐项校验 sha256 | 确定性 |
| `ojh fetch-base NAME --machine M` | 按 catalog 下载固定 revision 的基座权重，并校验 sha256 | 确定性 |
| `ojh prepare EXP` | 用适配器把数据转成 OneJev 行；校验 schema、id 唯一、target 分布、split 字段；按配置派生训练子集；计算 sha256，和 EXP 里写的值比对，不一致就停 | 确定性；新格式由 agent 写适配器 |
| `ojh resolve EXP --machine M` | 检测 GPU；结合数据、机器、基座、配方解析出 `run.lock.yaml`（第 8 节）；在目标机器上跑 plan-only 得到每 epoch 步数，按 1/8 epoch 设评测和保存间隔；不可行就报错并给出建议 | 确定性 |
| （agent） | 需要时改 lock 里的 `overrides`，每项都写明原因 | **agent 判断点**，会被记录 |
| `ojh launch LOCK` | 在 tmux 里依次执行：取卡加锁 → 按 seed 依次训练 → 自动接着 finalize → 写完成标记（含退出码）。**启动后立即返回**会话名、日志路径和完成标记 | 确定性 |
| `ojh status RUN` | 读日志，报告当前 step / 总 step、最近一次评测、预计剩余时间和占用的卡，输出要短 | 确定性 |
| `ojh finalize RUN` | 选 checkpoint → 重载后逐题打分 → 训练基线 → 定阈值 → 出对比表并判门 → 写记录卡并登记（第 9–11 节）。launch 结束时自动调用，也可以单独重跑 | 确定性 |
| `ojh eval-report RUN --allow-report` | 用锁定的 checkpoint 和阈值在 report 部分评测，**必须显式加参数** | 确定性 |
| `ojh table` | 把登记表渲染成实验对比表 | 确定性 |
| `ojh cleanup RUN` | 列出未选中的 checkpoint 及大小；只有加 `--confirm` 才删，skill 要求 agent 先问用户 | 用户批准 |

**实验配置示例**（v1 复现，对应 `examples/dynamicserve_v1_repro.yaml`，路径由使用方填写）：
```yaml
name: elephant_majority75_repro
base: elephant-1
recipe: examples/recipes/elephant_full_fsdp.yaml   # 或使用方自己的配方文件
data:
  adapter: onejev_rows
  train: <DATA>/export_majority75/binary/train.jsonl
  train_sha256: fe83846076cb6c72…                    # 写完整值
  eval:                                              # 评测集由使用方显式列出，没列的文件一律不读
    val_id:
      train_labels: <DATA>/export_majority75/binary/val_id.jsonl
    val_catood:
      train_labels: <DATA>/export_majority75/binary/val_catood.jsonl
  split_file: <RUN>/train/splits.json                # 可选；固定读取，不重新切分
fields:
  label: target.yes                                  # 正类在 target 里的键
  group: meta.category                               # 可选：分组（类别）
  consistency: meta.f                                # 可选：逐题一致比例，用于预算下的承接率
  text: messages                                     # 基线用的文本：state.messages 拼成 "role: content"
evaluator:
  type: binary_decision
  budgets: [0.10, 0.15]
  gate: true
seeds: [42]
gpus: 4
overrides: {}
```
调用时加 `--machine <使用方的机器配置>`。

**训练子集**：做消融（例如只用部分题型，或按条数抽样做规模对比）时，常要从同一份数据版本里取子集。`prepare` 支持在实验配置里声明：
```yaml
subset:
  where: {meta.pair_design: full}   # 按字段取值过滤
  n: 1000                           # 可选：抽样条数
  seed: 7
```
派生出的子集文件、规则和 sha256 都写进 lock，这样数据来源不会断。

**训练标签和参照标签分开**：v1 的软标签消融用软标签训练，并按软标签 NLL 选模；但阈值和门是按 majority75 硬标签和 `meta.f` 算的。因此每个评测集可以给两份标签：
- `train_labels`：选模时算 NLL 用，必须和训练用的标签版本相同；
- `reference`：定阈值、算 AUC、判门、取一致比例用；不写就和 `train_labels` 相同。

两份按 id 关联，只用两边都有的 id；被去掉的 id 记进记录卡。登记表里同时写训练标签版本和参照标签版本。

## 8. 画像和解析规则

**机器配置**由使用方提供（每人每台机器一份）。repo 只带一份 `examples/machine.example.yaml` 作为模板。可配置的键：
- Python 环境路径、基座模型目录、`out_root`、登记表路径；
- GPU 规则：
  - 可用卡列表或排除列表；
  - 最小空闲显存；
  - 锁目录（可选，不配就不加锁）；
  - 取卡顺序（默认按空闲显存从多到少）；
  - 等卡超时（默认 1 小时）；
  - `gpu_source`：`scan`（默认）或 `preassigned`。
- 禁止写入的目录；环境变量（例如 `HF_HUB_OFFLINE=1`）。

代码里不出现任何个人路径或机器取值。

**GPU 来源**：
- `scan`：扫描 `nvidia-smi`，按配置筛卡、加锁。多人共用一台机器时，锁目录必须配成所有人都能写的共享路径；放在个人 home 下互斥不了。
- `preassigned`：启动前已设好 `CUDA_VISIBLE_DEVICES`（例如调度系统分配的卡），直接用这些卡，跳过扫描和加锁。
- `explicit`：用 `ojh run --gpus i,j` 指定的卡（第 15 节）。没有机器配置时默认就是它；`--gpus` 给了卡号时也按它处理。

**基座 catalog**：

| 基座 | 来源 | 结构 | 注意力 / 读出 | 发布时的配方 |
|---|---|---|---|---|
| joey-1 | `baselabs/joey-1@9efbe48` | Qwen3.5-2B（HF 元数据显示约 3B） | noncausal / slot_lm | lr 1e-5，1 epoch，16×H100 |
| elephant-1 | `baselabs/elephant-1@b3f8a93` | Qwen3.5-9B | noncausal / slot_lm | lr 3.5e-6，1 epoch，128×H100，FSDP |
| mammoth-1 | `baselabs/mammoth-1`（revision 实现时固定） | Qwen3.5-27B | noncausal / slot_lm | lr 2.7e-6，1 epoch，72×GB300，FSDP |

每个条目还要写清：
- 权重的 sha256；
- 是否多模态，多模态的加 `--multimodal`；
- 视觉塔冻结，即加 `--full-finetune`、不加 `--train-vision`。

**示例配方**（取自 DynamicServe v1，只作示例，不是默认值）：

| 示例配方 | 设置 |
|---|---|
| `elephant_full_fsdp` | 4 卡 FSDP，lr 3.5e-6，1 epoch，brier 1，warmup 0.03，weight decay 0，max-grad-norm 1，max-len 16384，batch-tokens 16384，max-rows 8，accum-rows 32，bf16，noncausal，slot_lm |
| `joey_full_1gpu` | 1 卡，lr 1e-5，brier 0，max-rows 1，accum-rows 32，其余同上；只训 1 epoch（实测 1 epoch 后过拟合） |
| `joey_lora_1gpu` | 1 卡 LoRA，r 32，alpha 64，dropout 0.05，目标模块用 OneJev 默认（所有线性投影），其余同 `joey_full_1gpu`；学习率等超参没有实测，是起点而不是推荐值 |
| `mammoth_lora_ddp` | mammoth-1 多卡 LoRA（DDP），LoRA 参数同上，开 `--lora-autocast`；超参同样没有实测 |

**解析规则**（初始值，用 plan-only 和复现实验校准）：
- **训练方式：全参或 LoRA，由配方指定**。OneJev 命令行默认是 LoRA，加 `--full-finetune` 才是全参；所以每份配方都必须显式写 `mode: full` 或 `mode: lora`，harness 不依赖 OneJev 的默认值。
  - **全参**：
    - 加 `--full-finetune`，不加 `--train-vision`，视觉塔冻结；
    - fp32 主权重加 bf16 autocast 前向；
    - 只有全参才能用 FSDP。
  - **LoRA**：
    - 参数包括 r、alpha、dropout、目标模块，以及可选的 `--lora-autocast`；
    - OneJev 的 `--fsdp` 只支持全参（代码里直接报错 "--fsdp shards a full fine-tune"），所以多卡 LoRA 只能用 DDP，每张卡都要放一份完整的 bf16 基座权重。
- **卡数和并行方式**：
  - 卡数以配方或实验配置为准；拿不到足够的卡时报错，并列出替代方案（等卡、减卡、换配方）。
  - 全参：基座 ≥ 9B 时用 FSDP；单卡用单进程；多卡且不用 FSDP 时用 DDP。
  - LoRA：单卡用单进程，多卡用 DDP。
- **放不放得下**：按下面的口径粗算，再和候选卡的实际空闲显存比较。实际以 plan-only 和试跑为准；粗算的系数在 A1、A4、A7 跑完后校准。
  - 全参约需 16 字节/参数。FSDP 时按卡数平摊；不用 FSDP 时每张卡都要放下全部。
  - LoRA 约需 2 字节/参数的 bf16 基座，加上 adapter 和激活，每张卡都要放下。
- **放不下怎么办**：`resolve` 报“不可行”，列出可行的替代方案（加卡、换更小的基座、换另一种训练方式），**不自动切换**。要换训练方式，由 agent 写 override 并写明原因。
  - 例：mammoth-1 全参约需 430 GB，4×A100-80GB 放不下；改 LoRA 时每卡约需 54 GB 权重再加激活，必须用空闲显存足够的卡。在 DynamicServe 的共享 A100 上，“空闲 ≥ 48 GB”的筛卡门槛不够，需要配置更高的门槛或者用独占的卡。
- **评测和保存间隔**：每 epoch 步数取 plan-only 的实际结果（设 `PLAN_WORLD=卡数`），间隔取 ceil(步数 ÷ 8)。
- **FLA kernel 报错**：故障表给出 `--disable-fla-kernels`，但要以 override 的形式写进 lock，不能悄悄加。

**lock 文件**写下全部内容，执行时只读它：
- 训练方式（全参或 LoRA）及其参数；
- OneJev 的完整命令行、卡数、seed 列表；
- 数据路径和 sha256（包括派生子集的规则）；
- 基座的 revision 和 sha256；OneJev 的 revision 和补丁的 sha256；
- 环境指纹：pip freeze 的哈希、驱动版本；
- 评测器的配置；
- `overrides` 及每项的原因。

## 9. 训练、选模和打分

- **训练入口**：harness 有自己的训练入口 `train_entry`。它只替换 OneJev 的 `sft.run_evals`；训练循环、loss、优化器和保存都不动，做法和 v1 的 `publisher_native_train.py` 一样。替换后的评测：
  - 每次评测按每批 1 行逐题打分，把每个评测集每道题的选项概率写成 `<集合>-step<N>.jsonl`；
  - 在 `log.jsonl` 里写一条评测事件，内容是 `train_labels` 上的 NLL。
  
  这样做的原因：
  - v1 elephant 的 select NLL 0.457 就是这样算出来的；OneJev 自带的评测按 `max_rows` 成批，bf16 下数值会略有差别。
  - 每一步都有逐题预测，选模和后续分析都不用重新打分。
- **可选审计**：确认冻结的视觉塔权重在训练前后没有变化，并确认第一个正学习率的 step 确实更新了参数（沿用 v1 wrapper 的检查）。
- **训练中评测用的数据**：只用每个评测集的 select 部分（第 10 节）。
- **选模**：select 合并集上 NLL 最低的 checkpoint，同分取更早的一步，不含 step 0。插件可以换成别的规则。
- **选中之后**：保存、重载，再用 OneJev 的打分路径（`train.common.batch_scores`，按 checkpoint 保存的读出设置，每批 1 行）复评一次。复评的 NLL 应与训练时一致，不一致就把 run 标成 `failed`。
- **LoRA 的 checkpoint**：OneJev 每次保存的是 adapter。处理方式如下：
  - 重载复评时加载“固定 revision 的基座 + 选中的 adapter”；
  - 实验配置里写了 `merge: true` 时，finalize 再把选中的 adapter 合并进基座，另存一个完整模型目录（带上读出设置），用合并后的模型再复评一次，结果应与“基座 + adapter”一致（初定容差 1e-3，bf16 合并会有小误差）；
  - 默认不合并。合并后的目录很大，同样只列为删除候选，不自动删除。
- **问题文本**：直接读每行自带的 `question` 字段，不另外传问题文件。DynamicServe 的导出行里就是 v1 打分用的 elephant_r8 问题文本；验收 A2 会核实两者的打分结果一致。

## 10. 评测器和基线（都在训练侧）

**通用输入约定**：约定的是格式，不是路径。
- 训练文件和评测文件：在实验配置里显式列出，没列出的文件一律不读。
- 标签字段 `fields.label`：正类在 `target` 里的键。
- 可选的分组字段 `fields.group`。
- 可选的切分文件 `data.split_file`：结构是 `{评测集名: {"assignment": {id: "select" | "report"}}}`；DynamicServe 的 `splits.json` 正好是这个结构。没有切分文件时，整个评测集都算 select。
- 可选的一致比例字段 `fields.consistency`：给了才计算“预算下的承接率”。
- 基线文本 `fields.text`：
  - `messages`：把 `state.messages` 拼成每行一条 "role: content"，和 v1 的基线一致；
  - `state_json`：把 state 序列化成 JSON；
  - 也可以由适配器指定一个函数。

**内置评测器 `binary_decision`**：
- **select 合并集**：所有评测集的 select 部分合起来，用于选模和定阈值。v1 复现时，它必须和现有的 `train/data/select.jsonl`（380 行）完全一致。
- **阈值**：在 select 合并集上锁定：
  - 一个最佳 Jaccard 阈值；
  - 给了一致比例字段时，再为每个预算各锁定一个阈值，满足“不一致率 ≤ 预算”。

  不一致率 = 所有判为正类的题的 (1 − 一致比例) 的平均值，不用 0/1 标签计算。每个阈值都按评测集分别报告实际不一致率；在合并集上满足预算的阈值，在单个评测集上可能超出。
- **指标**：在每个评测集的 select 部分、以及 select 合并集上计算：
  - AUC；
  - 组内 AUC（需要分组字段）；
  - 各预算下的正类承接率（需要一致比例字段）；
  - 各自最佳阈值下的 Jaccard；
  - ECE。
- **对照组**（finalize 时自动计算，用同一份训练数据、同一批 select 题）：
  - **TF-IDF**：完整照搬 v1 的 `TextBaseline`（`data_pipeline_v2/sanity_check/step1_data/cheap_baseline.py`），代码复制进 `harness/baselines/` 并注明出处：
    - 特征：词 1–2 gram 和字符（`char_wb`）2–5 gram 的 TF-IDF（`min_df=2`，`sublinear_tf`，各最多 20 万维），再加一维 log1p(文本长度) ÷ 10；
    - 模型：逻辑回归，`max_iter=2000`。软目标的做法是每题复制成一正一负两条，权重分别是 y 和 1 − y（各加 1e-9）；
    - 正则强度在 {1, 10, 100} 中选：在训练题上做 3 折交叉验证（`KFold`，打乱，种子 0），取留出 AUC 最高的。训练题少于 60 道时固定用 10。
  - **分组先验**：给了分组字段时计算，取训练集每组的平均标签；未见过的组用总体平均。
  - **全选正类**：承接率 100%，报告此时的不一致率。
  - **oracle**：给了一致比例字段时计算，直接用真实的一致比例排序。
- **门**（可以在实验配置里关掉）：在每个评测集的 select 部分上，路由器的 AUC 和组内 AUC 都要高于 TF-IDF 和分组先验（第 15 节起：高于注册基线中可用的最好一个，含 length）。不过就把 run 标成 `failed`，但记录照常写。
- **报告**：对比表把路由器和所有对照组放在一起，按评测集分别列出。
- **report 和 test**：
  - `ojh eval-report` 只有加 `--allow-report` 才评 report 部分，用的是锁定的 checkpoint 和阈值，不重新调参；
  - split 是 test 的行一律拒绝读取，除非显式加 `--allow-test`。这种情况要用户批准，并写进记录。

**其他任务的插件**：二分类以外的任务（例如多选一、多档位），由使用方按路径指定插件。repo 只放接口说明：

| 钩子 | 作用 | 不提供时 |
|---|---|---|
| `validate(rows)` | 任务特有的数据检查 | 不检查 |
| `select_rule(events)` | 选模规则 | NLL 最低 |
| `calibrate(preds, rows)` | 决策阈值 | 不定阈值 |
| `baselines(train_rows, eval_rows)` | 对照组 | 只有 TF-IDF 和分组先验 |
| `acceptance(preds, rows)` | 指标和门 | 只报 NLL、准确率、ECE |

## 11. 记录和登记

每个 run 一个目录 `<out_root>/<exp>/<run_id>/`，里面有：
- `run.lock.yaml`，以及实际执行的命令；
- 环境指纹、所用 GPU 的 UUID 和型号；
- OneJev 的 `args.json` 和 `log.jsonl`；每一步的逐题预测；checkpoint；
- 选中 checkpoint 重载后的逐题预测 `predictions_select.jsonl`；
- `thresholds.json`、`comparison.json`（路由器加对照组的指标和门）；
- `timing.json`：墙钟时间，以及 GPU 时间 = 训练耗时 × 卡数；
- `RUN_CARD.md`：一页摘要；
- `status`：`running`、`done`、`failed` 或 `incomplete`。

finalize 完成后，往登记表追加一行，包含：
- 实验名、run id、seed；
- 训练标签版本、参照标签版本、数据 sha256；
- 基座、配方、OneJev 版本；
- GPU 型号和卡数、GPU 时间；
- 训练方式；选中的 step 和 checkpoint 的 sha256（LoRA 记 adapter 的 sha256 和基座 revision，合并了就再记合并后模型的 sha256）；
- 主要指标、对照组指标、门的结果；
- overrides；
- 一句话结论（由 agent 填写，写明依据）。

缺任何一项就标 `incomplete`，`ojh table` 默认不显示这类 run。

**多 seed**：一份配置里的多个 seed 依次运行（v1 不并行）。finalize 在每个 seed 完成后各写一条记录，全部结束后再写一条汇总，给出各指标的均值和标准差。

## 12. Agent skill

**主 skill `training-with-onejev`**：
- 什么时候用；
- 流程：prepare → resolve → launch → status → finalize；
- 三个判断点：改 override、写适配器、按故障表处理；
- 全参放不下时，可以列出 LoRA 方案，但切换必须写成 override 并说明原因；同一组对比实验里，全参和 LoRA 的结果要分开看，不能混在一起比；
- 审批规则：删 checkpoint、用 report 或 test、推送 repo，都要先问用户；
- 长任务规则：launch 返回后立即回报会话名、日志和完成标记，不轮询；
- 故障表：
  - OOM：按表依次降低 `batch_tokens` 和 `max_rows`；
  - FLA kernel 报错；
  - NCCL 超时；
  - plan-only 失败（多半是数据或适配器的问题）；
  - 等卡超时；
  - 复评 NLL 对不上。

**数据 skill `adapting-data-for-onejev`**：
- OneJev 行格式：`id`、`state`、`question{type,instructions,criteria}`、`target`（在 criteria 上的分布）、`meta`、`split`；
- 怎么写适配器、怎么自测。

skill 只描述流程和判断，所有参数都来自机器配置、实验配置和 lock。

## 13. 验收

| # | 检查 | 通过标准 |
|---|---|---|
| A1 | 复现 v1 elephant：数据 sha256 `fe838460…`（5,695 行），同一份 `splits.json`，示例配方 `elephant_full_fsdp`，seed 42 | 选中 step154 或相邻的 step176（两者 select NLL 只差 0.0003）；select NLL ≈ 0.457，合并 AUC ≈ 0.856，组内 AUC ≈ 0.816；初定容差 NLL ±0.01、AUC ±0.01，GPU 不确定性造成的偏差写进报告 |
| A2 | 用 harness 对现有的 `ckpt-step154` 重新打分 | 与 `predictions_select_elephant_{val_id,val_catood}.jsonl` 逐题一致（容差 1e-4） |
| A3 | 内置评测器在 v1 数据上算出的 TF-IDF、分组先验、oracle 和三套阈值 | 与 `gate_elephant.json`、`thresholds_elephant.json` 一致（容差 1e-3） |
| A4 | joey 示例配方在 1 卡上短跑（`max_steps` 设小值） | 跑完，生成完整的登记记录 |
| A5 | 换一份机器配置和 `out_root`，模拟另一个人 | 能跑通；代码里搜不到个人路径或机器取值 |
| A6 | 一个只拿到这个 repo 的新 agent，按 skill 完成 A4 | 不需要额外提示就能完成；发现的 skill 问题都修掉 |
| A7 | joey LoRA 示例配方在 1 卡上短跑，开 `merge: true` | 跑完并生成完整记录；“基座 + adapter”复评的 NLL 与训练时一致；合并后的模型复评与“基座 + adapter”一致（容差 1e-3）；记录里有 adapter 的 sha256 和基座 revision |
| A8 | 用 `mammoth_lora_ddp` 只跑 `resolve`，加上 plan-only | 在给定的卡上给出可行或不可行的判断，以及显存估算；不可行时列出替代方案。不要求真正训练 |

A1 是“harness 与旧流程等价”的判据。不通过时，harness 仍可用于新实验，但 README 和每张记录卡都要写明“尚未与 v1 对齐”；排查时先比对训练入口和 v1 wrapper 在评测和审计上的差异。

## 14. 留给实现者

- 命令行框架、包的组织方式、配置解析库；
- 测试粒度：解析规则、评测器、基线和数据校验要有单元测试；GPU 部分靠验收测试覆盖；
- `status` 的输出格式（要短）、`RUN_CARD.md` 的版式、`ojh table` 的列；
- tmux 会话的命名规则（要带用户名，避免和别人冲突）；
- OOM 降参表的具体档位；
- LoRA 合并用 OneJev 自带的 `--merge` 逻辑还是 peft 直接合并（要求合并后保留读出设置，并通过 A7 的一致性检查）；
- `char_wb` 等基线细节以复制过来的 v1 代码为准，A3 负责核对。

## 15. 一次性流程（one-shot）

### 分工

repo = skill + 固定版本的 OneJev + 脚本。用户只给三样：**数据位置、要用的 GPU、基座**。

- **认数据、改数据是 skill 的事**：agent 用 `ojh inspect` 看事实，判断格式和字段（标签、分组、一致比例、文本、split），需要时写适配器，填实验配置。
- **脚本只做机械的事**：只读的事实、确定性的切分、校验、解析、启动、finalize、结果页。
- **代码里不做自动格式识别，也不自动选基座**。基座、卡、训练方式由用户定；不可行时报原因和替代方案，不自动切换。

主路径：`ojh inspect` → agent 写适配器和 `exp.yaml` → `ojh run --base --gpus` → `ojh status` → 读 `RESULT.md`。

### `ojh inspect <path> [--json] [--rows N]`

只读，不写任何文件，不做判断。`<path>` 是文件或目录（目录递归，跳过隐藏项）。支持 jsonl、json、csv、tsv；装了 pyarrow 时也读 parquet，否则列进 `skipped`。每个文件（`--json` 的键）：

- `type`、`size_bytes`、`rows`；`name_hint`：文件名里的 train/training/val/valid/validation/dev/test/eval；
- `fields`：每个点分路径（列表元素记作 `[]`，例如 `conversation[].role`）的 `types`（类型计数）、`fill`（填充率）；低基数（≤20 个不同值且不是每行一个）时 `values`（取值和计数）；数值的 `min`/`max`/`mean`/`unit_interval`（是否都在 [0,1]）；字符串的 `avg_len`/`text_like`（平均长度 ≥20）；列表的 `avg_items`。csv/tsv 单元格能解析成数就按数统计，空单元格算缺失；
- `split_columns`：名字像 split（split、set、fold、partition 等）或取值都像 split 名的低基数字段，及各取值的行数；
- `split_file`：JSON 形如 `{集: {"assignment": {id: "select"|"report"}}}` 时，各集的 select/report 数；
- `onejev`：`passes`、`errors`、`first_errors`，用 `prepare.validate_rows` 检查是否已是 OneJev 行；
- `samples`：前 N 行（默认 3），字符串截到 200 字符，列表截到 5 项。

文件名像 test 的文件只数行数（`note` 说明），不读字段和样例；样例从不取 test 开头的 split 的行（`test_rows` 给出这种行的数量）。

### `ojh make-split`

`ojh make-split --out <out> --group <点分字段> [--seed 20261115] [--frac 0.5] [--mode select-report|holdout] [--allow-test] <file>...`

确定性、按组：组键排序后用 `random.Random("<seed>:<集名>")` 打乱，逐组贪心：加入第一部分能让它的行数更接近 `frac × 总行数` 就加入，否则进第二部分。同一组永远在同一边。缺组字段的行直接报错。

- **select-report**（默认）：每个文件是一个评测集，集名取文件名主干，或写成 `名字=路径`。`--frac` 是 select 的比例。`--out` 是切分文件，格式就是 `data.split_file`：`{集: {"assignment": {id: "select"|"report"}, "seed", "frac", "group", "source", "source_sha256", "rows": {"select", "report"}, "groups": {"select": [...], "report": [...]}}}`。每行要有 `id`，即适配器产出的 id。已有的 `--out` 内容不同就拒绝覆盖。
- **holdout**：只接受一个训练文件，`--frac` 是 val 的比例，`--out` 是**目录**（像 `.json` 文件名的 `--out` 会报错）。写出：
  - `<out>/train.jsonl`、`<out>/val.jsonl`：原样复制的行，保持源文件顺序，每行一个 JSON 对象（csv/tsv 的单元格转成 JSON 值）；不加也不改 id，适配器用同一套规则读这两个文件；
  - `<out>/manifest.json`：`{"mode": "holdout", "source", "source_sha256", "group", "seed", "frac", "rows": {"train", "val"}, "groups": {"train": [...], "val": [...]}, "sha256": {"train.jsonl", "val.jsonl"}}`。

  实验配置里 `data.train` 指 `<out>/train.jsonl`，`data.eval.val.train_labels` 指 `<out>/val.jsonl`；不写 `split_file` 时 val 全部算 select。已存在的输出文件不覆盖。
- split 以 test 开头的行、文件名像 test 的文件，都要 `--allow-test` 才读。

### `ojh run <exp.yaml> --base <名字> --gpus <i,j,...> [--recipe <路径>] [--machine <m.yaml>] [--allow-test]`

一次完成 prepare → resolve → launch，立即返回，打印 JSON：`run_dir`、`session`（tmux 会话）、`log`、`done`（DONE 路径）、`base`、`recipe`（`path`、`source`、`mode`）、`gpus`（每张卡的编号、UUID、型号、空闲显存、需要的显存、上面的进程）、`data_sha256`、`machine`（机器配置的来源）、`result`（`RESULT.md` 将出现的位置）。

- `--base`、`--gpus` 覆盖或补上 `exp.yaml` 里的 `base`、`gpus`（`gpus` 可以写卡号列表）。基座从不自动选：两处都没有就报错并列出 catalog 里的基座。
- 配方：`--recipe` > `exp.yaml` 的 `recipe` > 基座的 `default_recipes`。
- resolve 判定不可行时退出码 2，打印替代方案；不换基座、不换配方、不启动。

### 指定的卡（`gpu_source: explicit`）

`--gpus` 列出的每张卡都要：在 nvidia-smi 里存在；不被 `gpu_include`/`gpu_exclude` 排除；空闲显存 ≥ max(resolve 的每卡估算, `min_free_mib`)。失败时逐卡给出原因，并列出卡上的进程（pid、进程名、显存，来自 `--query-compute-apps`）；resolve 把其他空闲显存够的卡作为事实列在替代方案里。lock 记下 `machine.gpu.source: explicit` 和卡号，`feasibility.cards` 记下检查结果。runner 不扫描也不等卡：按 lock 里的卡号重新检查，配了 `lock_dir` 时同样加锁（被别人锁住就失败，`DONE` 的 `stage` 是 `acquire`），然后把这些卡作为预先分配的卡交给训练。`scan` 和 `preassigned` 不变。

### 基座的默认配方

`catalog/bases/<基座>.yaml` 的 `default_recipes`：键是卡数 `N`、范围 `A-B` 或 `A+`，值是相对 repo 根目录的配方路径。现有的：joey-1 1 卡 → `joey_full_1gpu`；elephant-1 2–8 卡 → `elephant_full_fsdp`；mammoth-1 2–8 卡 → `mammoth_lora_ddp`。其余基座没有合适的示例配方，不写。没有对应卡数的默认值时报错，列出该基座的默认值、所有示例配方和 `--recipe`。

### 机器配置可选

按顺序取第一个：`--machine`；`$OJH_MACHINE`（指向的文件必须存在）；`~/.config/onejev-harness/machine.yaml`；推断的默认值：python = `<repo>/.venv/bin/python`，`out_root` = `$OJH_OUT_ROOT` 或 `./ojh_runs`（转成绝对路径），`model_dir` = `$OJH_MODEL_DIR` 或 `~/.cache/onejev-harness/models`，登记表在 `out_root` 下，不加锁，`gpu_source: explicit`。`setup`、`fetch-base`、`prepare`、`resolve`、`run` 都按这个顺序取。lock 的 `machine` 记下实际用的全部设置和 `origin`（来源和文件路径）；已有的机器配置文件照常可用。

### 适配器

`data.adapter` 可以是 `adapters/` 里的模块名，也可以是 `.py` 文件路径（相对实验配置文件）。参数：`data.adapter_args`（所有文件）、`data.train_adapter_args`（训练文件）、`data.eval.<集>.adapter_args`（该评测集的文件），后两者合并在前者之上，以 `adapt(path, **args)` 传入；这样一个带 split 列的文件可以同时作训练和评测来源。适配器给训练行设 `split: "train"`（配方用 `--split train`）。prepare 在 `prepared.json` 记下适配器文件的 sha256 和参数；resolve 发现适配器在 prepare 之后改过就报错，把 sha256 和参数写进 lock，并把文件复制到 `<run>/adapter/`；`RUN_CARD.md` 和 `RESULT.md` 写明适配器和 sha256。确定性检查不变。

### 实验配置模板

`examples/exp.template.yaml`：每个键都有注释（`data.train`、带 `train_labels`/`reference`/`adapter_args` 的评测集、`split_file`、`fields.label/group/consistency/text`、`subset`、`evaluator`、`seeds`、`merge`、`overrides`）。`base`、`gpus`、`recipe` 可以不写，由 `ojh run` 提供。

### 结果页：`RESULT.md` 和 `result.json`

finalize 最后写进 run 目录（`result.json` 是机器可读的同一份内容）：

- 数据：各部分行数（train、select 及各评测集、report）、训练和参照标签版本、sha256、适配器；
- 基座、配方（lock 记下配方路径和 sha256）、卡和训练墙钟时间、GPU 时间；
- 训练曲线（每次评测的 select NLL），选中的 step、checkpoint 和 sha256、复评 NLL；
- select 合并集上的指标表：模型、每个基线、全选正类、oracle 的 AUC、组内 AUC、各预算下的承接率（带 95% bootstrap 置信区间）和各自最佳阈值下的 Jaccard；
- 锁定的阈值：合并集和各评测集上的承接率、Jaccard、不一致率；
- 门的结论和逐项理由（模型对最好基线的比较）；失败项、警告（被去掉的 id、未见过的组、select 太少、按题 bootstrap、读了 test、override、门关闭、推断的机器配置）；
- 下一步命令：`ojh note`、`ojh eval-report --allow-report`、`ojh cleanup`、`ojh cleanup --confirm`，读 report 和删除的两条标明要用户批准。

`ojh status <RUN>` 在 `RESULT.md` 存在时打印它的路径。

### Bootstrap 置信区间

评测器对 AUC、组内 AUC 和各预算下的承接率给出 95% 置信区间（2.5 和 97.5 百分位）：每次重抽先有放回地抽与组数相同个组，再在每个抽到的组里有放回地抽与组大小相同条题；同一组被抽到两次算两个簇（组内 AUC 不跨簇配对）。同一评测集的所有方法用同一组重抽，区间可以成对比较。默认 1000 次、种子 20261115，`evaluator.bootstrap_reps`、`evaluator.bootstrap_seed` 可改，0 关闭。没有分组字段时按题重抽，并在结果里写明。计算用向量化实现，测试核对它与第 10 节移植的 v1 函数一致。

### 基线插件

`harness/baselines` 里有一个小注册表：`Baseline(name, needs, fit, description)`，`needs` 是需要的字段（`text`、`group`），缺了就跳过；`fit(训练样例)` 返回带 `predict(样例) -> [0,1] 概率` 的对象。内置 `tfidf`、`group_prior` 和 `length`（对 log(1+字符长度) 做逻辑回归，软目标同 TF-IDF）；`register()` 可以加。门对每个评测集比较模型和可用基线中最好的一个（每项指标分别取最好），并记下是哪个；没有任何基线的集照旧判不通过。插件评测器的默认基线也用这个注册表。
