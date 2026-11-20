# Elephant-1（OneJev）

这个目录放 Elephant-1（`baselabs/elephant-1`，9B）在 Elliptic++ 上的实验代码，用 OneJev 训练和打分。实验分三部分：零样本、静态微调（全参数和 LoRA）、滚动增量训练。

每个实验的完整超参数，看对应脚本的开头，或者 `configs/` 里保存的参数文件。

## 准备

OneJev 不在这个仓库里，需要自己 clone，切到 commit `81ce62f`，再打上 Elephant-1 模型仓库自带的 `serve.patch`。依赖见 `requirements_onejev_freeze.txt`。

```bash
git clone https://github.com/OmniJev/OneJev.git && git -C OneJev checkout 81ce62f
git -C OneJev apply <elephant-1 模型目录>/serve.patch
export ONEJEV_ROOT=$PWD/OneJev ELEPHANT_MODEL=<elephant-1 模型目录>
export ONEJEV_PYTHON=<onejev 环境的 python> JEV_PYTHON=<jev 环境的 python>
```

下面的命令都在 `crypto-fraud-detection/` 目录下运行。

## 数据

先按上层 README 生成处理后的数据集，再转成 OneJev 的行格式。

state 有三种：
- `json`：55 个地址统计特征，写成 JSON；
- `text`：同样的特征，写成文本；
- `txhist`：交易历史。

问题也有两种：
- `choice`：选 Illicit 还是 Licit；
- `noul`：回答 yes 或 no。

```bash
for S in json text; do for Q in choice noul; do
  python elephant/scripts/export_onejev_data.py --data data/processed/elliptic_pp_addr_sqo --out data/onejev/addr_${S}_${Q} --state $S --question $Q
done; done
python scripts/build_addr_txhist.py      # 交易历史 state → data/onejev/addr_txhist_noul
```

## 实验

| 实验 | 命令 | 说明 |
|---|---|---|
| 零样本 | `elephant/scripts/run_elephant_zs.sh <gpu>` | 先在 val 上比较 4 种输入形式，选出最好的一种，再用它跑 test |
| 零样本 + 滚动重校准 | `python elephant/scripts/rolling_recalib.py --results results/elephant --name <名字>` | 模型权重不动，每一步用最近的标签重新校准概率。只用 CPU |
| 静态微调 | `elephant/scripts/run_elephant_ft.sh <json\|txhist> <full\|lora> <gpus>` | 用第 1–29 步训练。训练完在验证子集上选 checkpoint，再在 val 和 test 上打分 |
| 滚动训练 | `elephant/scripts/run_elephant_rolling.sh <json\|txhist> <full\|lora> <gpus>` | 从静态微调的模型出发，第 35–49 步逐步更新：标签延迟 1 步，同时回放旧数据 |

训练时必须加 `--attention noncausal --readout slot_lm`，脚本里已经加好了。

## 输出

预测文件写在 `results/elephant/<名字>_{val,test}.parquet`，格式和非 LLM 基线一样，可以直接用 `scripts/eval_decisions.py` 和 `scripts/summarize_results.py` 评估。checkpoint 写在 `ckpt/`，日志写在 `logs/`。

## 目录

```
scripts/export_onejev_data.py     数据集 → OneJev 行格式
scripts/onejev_score.py           打分，保存原始 logits
scripts/rolling_recalib.py        零样本的滚动重校准
scripts/run_elephant_zs.sh        零样本流程
scripts/run_elephant_ft.sh        静态微调流程
scripts/select_and_score.sh       选 checkpoint，然后在 val / test 上打分
scripts/run_elephant_rolling.sh   滚动训练流程
scripts/onejev_rolling.py         滚动训练：每一步调用一次 train.sft
configs/                          EL_full_txhist_all 静态版和滚动版的参数、启动命令
requirements_onejev_freeze.txt    onejev 环境依赖
```
