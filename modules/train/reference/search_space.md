# 能改什么：基座、配方、overrides

一个候选 = 基座 + 配方 + overrides。下面是常用的选项、它们属于哪一类、先验高低和代价。真正允许搜什么，以任务定义的 `search_space` 为准：用户批准任务定义时已经批准了它，要扩大就是改任务定义，写进交回的 `questions`。`search_space` 的键带 `overrides.` 前缀（`overrides.onejev.lr`），候选 YAML 里不带（`onejev.lr`）。

## 三类改动

| 类 | 改什么 | 怎么写 | 先验 |
|---|---|---|---|
| 模型选择 | 基座 | `base`（harness `catalog/bases/`：joey-1 2B、elephant-1 9B、mammoth-1 27B） | 结构类，高；换大模型时学习率要一起看 |
| 训练方式 | 全参或 LoRA、卡数、并行方式 | `recipe`（配方文件里的 `mode`、`gpus`、`fsdp`） | 结构类；全参贵，而且在 Elliptic 上不稳（见 lessons） |
| 超参 | 学习率、轮数、损失、批大小、LoRA 设置等 | `overrides`（`onejev.<flag>`、`lora.<字段>`，每项写原因） | 拟合类，多数只值很小的变化 |

训练方式只能通过换配方来改，不能写成 override：配方里的其他设置（例如 elephant 全参配方的 `fsdp: true`）和 LoRA 不兼容，harness 的 skill 也这样要求。harness 自己推导的标志（`attention`、`readout`、`eval_every`、`save_every`、`seed`、`model` 等）不能当 override。

## 常用 override

| 键 | 含义 | 常见取值 | 备注 |
|---|---|---|---|
| `onejev.lr` | 学习率 | 配方值的 0.25×、0.5×、2× | 文献和 Elliptic 都显示：模型越大，合适的学习率越小 |
| `onejev.epochs` | 训练轮数 | 1、2 | 代价按轮数成倍增加（`trsi` 估算预算时会乘上去） |
| `onejev.brier_weight` | 损失里 Brier 项的权重 | 0、1、3 | 影响校准；看 ECE、Brier 护栏 |
| `onejev.warmup_frac` | 预热比例 | 0.03、0.1 | |
| `onejev.weight_decay` | 权重衰减 | 0、0.01、0.1 | |
| `onejev.max_grad_norm` | 梯度裁剪 | 0.3、1 | |
| `onejev.accum_rows` | 每步累计的行数（等效批大小） | 32、64、128 | 实际每步装进的行数随行长变化，看 harness 日志里的实际值 |
| `onejev.no_augment` | 关掉输入增强 | true | |
| `lora.r`、`lora.alpha` | LoRA 秩和缩放 | r 8、32、128（alpha 一般 2×r） | 改模型能学多少，属于结构类 |
| `lora.target_modules` | LoRA 加在哪些层 | `"q_proj,k_proj,v_proj,o_proj"`（只注意力）、`"gate_proj,up_proj,down_proj"`（只 MLP） | 结构类 |

只影响显存、不影响训练结果的键（`onejev.batch_tokens`、`onejev.max_rows`）不是候选，不该出现在配置空间里。显存不够时按 harness 的故障表处理，那是修环境，不是候选。

## 选候选的顺序

1. 先看诊断指向哪里：曲线早早见顶 → 学习率、正则；到最后还在变好 → 轮数；种子之间差很大 → 先试大改动，小改动分不出来。
2. 结构类优先于拟合类：换基座、训练哪些层、LoRA 秩。
3. 换基座时，同时准备一个"小学习率"的候选。
4. 证据差不多时选便宜的：1 张卡优于多张，LoRA 优于全参，小模型优于大模型。
5. 不选死路，除非能说明它失败的前提已经不成立。
