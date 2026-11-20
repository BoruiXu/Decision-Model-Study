# 非 LLM 基线

这个目录放非 LLM 基线的代码，包括树模型、加图特征的树模型和 GNN。数据构建和评估用上层 `scripts/` 里的共用脚本。完整超参数写在各个脚本的开头。

## 方法

| 方法 | 输入 | 脚本 |
|---|---|---|
| LR、RF、XGB | 55 个地址特征 | `baseline_addr.py` |
| XGB 滚动 | 同上 | `rolling_xgb_addr.py` |
| XGB-Graph（GADBench） | 自身特征，加 1 跳和 2 跳邻居的均值，共 165 维 | `xgb_graph_addr.py` |
| GraphSAGE、GAT | 每个时间步一张地址图 | `gnn_addr.py` |
| XGB txhist | 交易历史展平成表格，是 Elephant txhist 的对照 | `xgb_txhist.py` |
| XGB txhist 滚动 | 同上 | `rolling_xgb_txhist.py` |

**滚动。** 给第 t 步打分之前，只用第 t−1 步及以前的标签，也就是标签延迟 1 步。滚动有两种做法：
- `refit`：每一步从头重训；
- `inc`：在上一步的模型上增量更新，只有 GNN 有这个选项。

## 运行

在 `crypto-fraud-detection/` 目录下运行：

```bash
pip install -r requirements_jev.txt
python scripts/build_addr_sqo.py data/raw/elliptic_pp data/processed/elliptic_pp_addr_sqo
python scripts/build_addr_txhist.py
python baselines/baseline_addr.py && python baselines/rolling_xgb_addr.py
python baselines/gnn_addr.py --model GAT [--roll refit|inc]   # 第一次运行会缓存图快照，XGB-Graph 要用
python baselines/xgb_graph_addr.py [--roll refit]
python baselines/xgb_txhist.py && python baselines/rolling_xgb_txhist.py
python scripts/eval_decisions.py results/addr
python scripts/summarize_results.py results/summary_all.csv
```

预测文件写在 `results/<任务>/<方法>_{val,test}.parquet`，列为 `id, y, p, post43, seen_in_train`。Elephant 的预测也是这个格式。

`eval_decisions.py` 在 val 上选阈值，在 test 上报告 AUPRC、AUROC、F1、Brier、ECE，并分第 43 步前后分别统计。
