# Elephant-1（OneJev）

这个目录放 Elephant-1 在 Elliptic++ 上的实验代码，用 OneJev 跑。现在只有零样本部分（不训练）。微调和滚动训练的部分待补，会从另一台机器推送。

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

## 输出

预测写到 `results/elephant/<名字>_{val,test}.parquet`，列为 `id, y, p, post43, seen_in_train, step, logit_pos, logit_neg, latency_ms`。重校准版本另有 `p_raw` 和 `thr`。

格式和上层的基线一致，可以直接用 `scripts/eval_decisions.py results/elephant` 和 `scripts/summarize_results.py` 评估。

## 目录

```
README.md                         本文件
requirements_onejev_freeze.txt    onejev 环境的完整依赖
scripts/export_onejev_data.py     数据集 → OneJev 行格式（state 两种 × 问题两种，选项交换，标签预算，val 子集）
scripts/onejev_score.py           用 Elephant-1 或微调后的 checkpoint 打分，保存原始 logits，输出 parquet
scripts/rolling_recalib.py        零样本模型的滚动 Platt 重校准和逐步阈值
scripts/run_elephant_zs.sh        零样本一键流程：选输入形式 → 地址 test → 交易 val/test
```

微调（静态 LoRA、全参数）和滚动训练的代码待补。
