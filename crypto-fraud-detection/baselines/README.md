# 非 LLM 基线

这个目录放非 LLM 基线的代码，包括树模型、加图特征的树模型和 GNN。数据构建和评估用的是上层 `scripts/` 里的共用脚本。

| 方法 | 输入 | 脚本 | 设置 |
|---|---|---|---|
| LR、RF、XGB | 55 个地址特征 | `baseline_addr.py` | XGB：500 棵树，深度 6，学习率 0.05，subsample 0.8，colsample 0.5<br>RF：200 棵树，max_features sqrt<br>LR：中位数填补后标准化<br>均不做类别重加权 |
| XGB 滚动 | 同上 | `rolling_xgb_addr.py` | 第 t 步打分前，用第 t−1 步及以前的所有标签从头重训 |
| XGB-Graph（GADBench） | 165 维图聚合特征 | `xgb_graph_addr.py` | 树参数同 XGB；`--roll refit` 为滚动版 |
| GraphSAGE、GAT | 每步图快照 | `gnn_addr.py` | hidden 128，dropout 0.3，lr 1e-3，最多 200 epoch，按验证集 AUPRC 选 epoch<br>`--roll refit`：每步从头重训<br>`--roll inc`：增量更新，新数据加 0.5 倍旧数据回放 |
| XGB txhist | 交易历史展平成表格 | `xgb_txhist.py` | 树参数同 XGB；是 Elephant txhist 输入的对照 |
| XGB txhist 滚动 | 同上 | `rolling_xgb_txhist.py` | 同 XGB 滚动 |

**滚动协议。** 模拟线上部署，标签延迟 1 步：给第 t 步（35–49）打分之前，只能用到第 t−1 步及以前的标签。

**评估。**
- `scripts/eval_decisions.py`：阈值在验证集上选，在测试集上报告 AUPRC、AUROC、F1、Brier、ECE，以及按时间步分层的 bootstrap 置信区间。分全部、第 43 步前、第 43 步后、新地址几个子集。
- `scripts/summarize_results.py`：把所有预测文件汇总成一张 AUPRC / AUROC 表。

每个方法都把预测写成 `results/<任务>/<方法>_{val,test}.parquet`，列为 `id, y, p, post43, seen_in_train`。Elephant 的结果也用同样的格式，可以直接一起评估。

**运行顺序。** 在 `crypto-fraud-detection/` 目录下运行：

```bash
pip install -r requirements_jev.txt
python scripts/build_addr_sqo.py data/raw/elliptic_pp data/processed/elliptic_pp_addr_sqo
python scripts/build_addr_txhist.py                  # 输出 data/onejev/addr_txhist_noul
python baselines/baseline_addr.py && python baselines/rolling_xgb_addr.py
python baselines/gnn_addr.py --model GAT [--roll refit|inc]   # 第一次运行会缓存每步图快照，XGB-Graph 要用
python baselines/xgb_graph_addr.py [--roll refit]
python baselines/xgb_txhist.py && python baselines/rolling_xgb_txhist.py
python scripts/eval_decisions.py results/addr
python scripts/summarize_results.py results/summary_all.csv
```
