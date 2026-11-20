# Elephant-1（OneJev）

这个目录放 Elephant-1 在 Elliptic++ 上的实验代码，用 OneJev 跑，包括零样本、静态微调（LoRA、全参数）和滚动增量训练。

## 模型和框架

| 项目 | 内容 |
|---|---|
| 模型 | `baselabs/elephant-1`，revision `b3f8a934`，9B，MIT 许可。它在 Qwen3.5-9B 上做了全参数决策后训练，损失是交叉熵加 Brier |
| 注意力 | 全注意力层用分块双向掩码：前缀是 system 和 state，后缀是问题和选项。线性注意力层保持因果。这个设置写在模型的 `config.json` 里 |
| 读出 | `slot_lm`：在答案位置，对选项字母的 LM logits 做 softmax，温度为 1 |
| OneJev | `OmniJev/OneJev`，commit `81ce62f`，再打上 Elephant-1 模型仓库自带的 `serve.patch`。补丁加了 `--attention noncausal` 和 `--readout slot_lm`，并新增 `qev/readout.py`。OneJev 源码不在这个仓库里 |
| 环境 | `requirements_onejev_freeze.txt`。这是实际跑通的完整依赖，主要版本是 torch 2.8.0+cu128、transformers 5.19.0 |

```bash
git clone https://github.com/OmniJev/OneJev.git && git -C OneJev checkout 81ce62f
git -C OneJev apply <elephant-1 模型目录>/serve.patch
export ONEJEV_ROOT=$PWD/OneJev ELEPHANT_MODEL=<elephant-1 模型目录>
```

## 数据转成 OneJev 格式

先按上层 README 生成 `data/processed/elliptic_pp_addr_sqo` 和 `elliptic_pp_tx_sqo{,_named}`，然后转成 OneJev 格式。命令在 `crypto-fraud-detection/` 目录下运行：

```bash
for S in json text; do for Q in choice noul; do
  python elephant/scripts/export_onejev_data.py --data data/processed/elliptic_pp_addr_sqo --out data/onejev/addr_${S}_${Q} --state $S --question $Q
done; done
python elephant/scripts/export_onejev_data.py --data data/processed/elliptic_pp_addr_sqo --out data/onejev/addr_json_choice_swap \
    --state json --question choice --swap --splits val
for V in "" _named; do
  python elephant/scripts/export_onejev_data.py --data data/processed/elliptic_pp_tx_sqo$V --out data/onejev/tx${V}_json_choice --state json --question choice
done
python scripts/build_addr_txhist.py      # 交易历史 state，直接写成 OneJev 格式：data/onejev/addr_txhist_noul
```

每行的格式是 `{"id", "split", "state", "question", "target", "meta"}`，下面是几个选项的含义。

| 选项 | 含义 |
|---|---|
| `--state json` | state 是特征字典，OneJev 把它渲染成 JSON，这是 Elephant 最常见的输入形式 |
| `--state text` | state 是 "- 名称: 数值" 格式的文本，和 Qwen / Kev 实验的输入相同 |
| `--question choice` | 选项是 Illicit / Licit，P(非法) 取 Illicit 那一项 |
| `--question noul` | 改成是 / 否问题，P(非法) = P(yes) |
| `--swap` | 选项顺序改成 Licit、Illicit，用来检查位置偏差 |
| `--ids` | 用标签预算文件筛选训练集 |

每个输出目录里还会生成 `val_sub4000.jsonl`，是从 val 中随机抽的 4,000 条（seed 0），给训练中途评估用。

state 里不会出现地址字符串、txId 和时间步，脚本里有 assert 检查。

## 零样本实验

**静态零样本 EL0。** 运行 `elephant/scripts/run_elephant_zs.sh <gpu>`，单卡，需要先设置 `ONEJEV_PYTHON`、`JEV_PYTHON`、`ELEPHANT_MODEL`、`ONEJEV_ROOT`。它依次做四件事：
1. 在地址 val 上比较 4 种输入形式（json / text × choice / noul），外加一次选项交换检查；
2. 按 val AUPRC 选出最好的形式，写到 `results/elephant/best_form.txt`，我们跑出来是 `addr_json_noul`；
3. 只用这一种形式跑地址 test；
4. 跑交易级的 val 和 test。

交易历史输入单独打分：

```bash
for S in val test; do
  $ONEJEV_PYTHON elephant/scripts/onejev_score.py --onejev $ONEJEV_ROOT --model $ELEPHANT_MODEL \
      --rows data/onejev/addr_txhist_noul/$S.jsonl --name EL0_addr_txhist_noul --split $S --out results/elephant
done
```

`onejev_score.py` 和 OneJev 官方的 `train.eval_mm` 走同一条代码路径：加载模型、按 checkpoint 配置设置注意力和读出、用生产环境的固定渲染、bf16 批量取 slot logits。在 200 条样本上，两者的准确率、NLL、ECE 完全一致。

它和 `train.eval_mm` 有一个区别：它保存原始 logits，不像官方 `--dump-preds` 那样四舍五入到 5 位小数，所以 AUPRC 不会出现人为的并列。运行时屏幕上应该打印 `attention noncausal, readout slot_lm`。

**零样本加滚动重校准。** 只用 CPU，在 jev 环境里运行：

```bash
python elephant/scripts/rolling_recalib.py --results results/elephant --name EL0_addr_json_noul --window 5
python elephant/scripts/rolling_recalib.py --results results/elephant --name EL0_addr_txhist_noul --window 5
```

模型权重不变。给第 t 步打分之前，只用最近 5 个已有标签的步（第 t−5 到 t−1 步）做两件事：拟合一个 2 参数的 Platt 映射 `p' = sigmoid(a · logit(p) + b)`，并重新选 F1 阈值。

斜率 a 不加限制。a < 0 表示最近的标签显示模型的排序是反的，映射会把排序翻过来。

## 微调和滚动训练

和零样本一样在 `crypto-fraud-detection/` 下运行，需要 `ONEJEV_PYTHON`、`ELEPHANT_MODEL`、`ONEJEV_ROOT`。checkpoint 写到 `ckpt/`，日志写到 `logs/`，预测写到 `results/elephant/`。

两个入口脚本，参数都是 `<state> <方式> <GPU>`：

```bash
elephant/scripts/run_elephant_ft.sh      <json|txhist> <full|lora> <gpus>   # 静态：训练 → 选 checkpoint → val/test 打分
elephant/scripts/run_elephant_rolling.sh <json|txhist> <full|lora> <gpus>   # 滚动：从对应的静态模型出发，逐步更新并打分
```

| 参数 | 取值 |
|---|---|
| state | `json`：统计 state（`data/onejev/addr_json_noul`）；`txhist`：交易历史（`data/onejev/addr_txhist_noul`） |
| 方式 | `full`：全参数，FSDP，至少 2 张卡；`lora`：LoRA，1 张卡 |
| GPU | 例如 `0` 或 `0,1` |

两种 state 都用 noul 问题，都用全部训练数据（第 1–29 步，169,438 条），seed 0。方法名是 `EL_<方式>_<state>_all_s0`，滚动版是 `EL_<方式>_<state>_all_roll_d1`。滚动要先跑对应的静态实验。

```bash
elephant/scripts/run_elephant_ft.sh txhist full 0,1        # 静态全参数 → EL_full_txhist_all_s0
elephant/scripts/run_elephant_ft.sh txhist lora 0          # 静态 LoRA → EL_lora_txhist_all_s0（滚动 LoRA 的起点）
elephant/scripts/run_elephant_rolling.sh txhist lora 0     # 滚动 LoRA → EL_lora_txhist_all_roll_d1
```

`configs/` 里是 `EL_full_txhist_all_s0` 和 `EL_full_txhist_all_roll_d1` 的完整参数：`args.json` 是 `train.sft`（或 `onejev_rolling.py`）保存的全部参数，`cmd.txt` 是启动命令。

### 静态微调

| 项目 | 全参数（`full`） | LoRA（`lora`） |
|---|---|---|
| 入口 | `python -m torch.distributed.run --standalone --nproc-per-node <N> -m train.sft --config train/configs/onejev_9b_full.yaml`（FSDP） | `python -m train.sft`，`--merge` |
| 训练的参数 | 全部参数 | LoRA r 32，alpha 64，dropout 0.05（`train.sft` 默认） |
| lr | 5e-6 | 1e-4 |
| batch | 每个微批最多 16,384 token、64 行；每次更新至少 64 行（`onejev_9b_full.yaml`） | 同左（`--max-len 16384 --accum-rows 64`） |
| `pad_options` | 0.5（来自 `onejev_9b_full.yaml`） | 0 |
| 共同 | 1 个 epoch，3% warmup + cosine，损失 CE + Brier（权重 1），weight decay 0，梯度裁剪 1.0，数据增强开（训练时随机打乱选项顺序，评估用固定顺序），每 250 步在 `val_sub4000` 上评估，每 400 步保存，`--eval-at-start` | 同左 |

训练在 `$ONEJEV_ROOT` 下运行，所以脚本传的都是绝对路径。脚本开头会检查两种 state 的 train 和 val_sub4000 的 id 和顺序完全一致。

**训练时必须加 `--attention noncausal --readout slot_lm`。** `train.sft` 的默认是因果注意力，不加的话训练用的注意力和 Elephant 本身不一致。

**选 checkpoint**（`scripts/select_and_score.sh`，训练完自动调用）：
1. 每个保存的 checkpoint（全参数：`ckpt-step*` 和 `final`；LoRA：`adapter-step*` 和 `adapter`）都在 `val_sub4000` 上打分。`train.sft` 自己的评估只记录 acc、NLL、ECE，没有 AUPRC。
2. 按 AUPRC（4 位小数）取最高；并列时取更早的一步。
3. 用选中的 checkpoint 在完整的 val 和 test 上打分。LoRA 的 adapter 在打分时合并进模型。

中间结果写到 `results/elephant/ckpt_select/`：每个 checkpoint 一个 `<方法名>_<ckpt>_valsub.parquet`，另有 `<方法名>_selection.json`，记录选择规则和各 checkpoint 的 AUPRC。

### 滚动增量训练

`scripts/onejev_rolling.py`，由 `run_elephant_rolling.sh` 调用。

| 项目 | 全参数（`full`） | LoRA（`lora`） |
|---|---|---|
| 起点 | 静态实验选中的 checkpoint，从 `ckpt_select/<方法名>_selection.json` 读取 | 静态实验的 `merged/`（`train.sft --merge` 合并的最终 adapter）；脚本会把 `adapter/readout.json` 拷进去 |
| 每步更新 | 全参数，`onejev_9b_full.yaml`，lr 1e-6，至少 2 张卡 | 新建一个 LoRA adapter（r 32，alpha 64，`--accum-rows 64`），lr 1e-5，训完合并，1 张卡 |
| F1 阈值 | 起点模型在 val 上的打分：拷贝 `<静态方法名>_val.parquet` | 起点模型在 val 上的打分：用 `merged/` 给 val 打分 |

两种方式共同的协议：
- **标签延迟 1 步。** 给第 t 步（35–49）打分前，新数据是上一次更新后到第 t−1 步之间新到的有标签数据；第 35 步的新数据是第 30–34 步（`--init_trained_through 29`，表示起点已用第 1–29 步训练过）。
- **回放。** 从更早的数据中无放回均匀抽取 0.5 倍新数据量，混合后打乱。抽样只由 `--seed` 和 t 决定，中断后重跑会从最后完成的那一步接着跑。
- **每步训练。** 调用一次 `train.sft`，训 1 个 epoch，用 `train.sft` 自带的 3% warmup + cosine 调度，`--attention noncausal --readout slot_lm`。
- **输出。** 每一步的打分写到 `ckpt/<方法名>_roll_d1/steps/step_<t>.parquet`，最后合并成 `results/elephant/<方法名>_roll_d1_test.parquet`，同时写好 `<方法名>_roll_d1_val.parquet`，供 `eval_decisions.py` 选阈值。上一步的模型会删掉；起点模型不在工作目录里，不会被删。

## 输出

预测写到 `results/elephant/<名字>_{val,test}.parquet`，列为 `id, y, p, post43, seen_in_train, step, logit_pos, logit_neg, latency_ms`。重校准版本另有 `p_raw` 和 `thr`。`latency_ms` 是整个打分循环的总时间 ÷ 行数，是吞吐量，不是单条请求延迟。

所有作业都不覆盖已有文件：同名的 run 目录或结果文件已存在时，作业直接退出。

格式和上层的基线一致，可以直接用 `scripts/eval_decisions.py results/elephant` 和 `scripts/summarize_results.py` 评估。

## 目录

```
README.md                         本文件
requirements_onejev_freeze.txt    onejev 环境的完整依赖
scripts/export_onejev_data.py     数据集 → OneJev 行格式（state 两种 × 问题两种，选项交换，标签预算，val 子集）
scripts/onejev_score.py           用 Elephant-1 或微调后的 checkpoint 打分，保存原始 logits，输出 parquet
scripts/rolling_recalib.py        零样本模型的滚动 Platt 重校准和逐步阈值
scripts/run_elephant_zs.sh        零样本一键流程：选输入形式 → 地址 test → 交易 val/test
scripts/run_elephant_ft.sh        静态微调：<json|txhist> × <full|lora>，train.sft 训练 → 选 checkpoint → val/test 打分
scripts/select_and_score.sh       选 checkpoint（val_sub4000 AUPRC，并列取更早）并在完整 val/test 上打分
scripts/run_elephant_rolling.sh   滚动：<json|txhist> × <full|lora>，从对应的静态模型出发，第 35–49 步逐步更新并打分
scripts/onejev_rolling.py         滚动增量训练：每个测试步调用一次 train.sft，再调用 onejev_score.py
configs/                          EL_full_txhist_all_s0 和 EL_full_txhist_all_roll_d1 的完整参数（args.json）和启动命令（cmd.txt）
```
