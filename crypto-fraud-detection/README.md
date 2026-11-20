# 加密货币非法检测：Elephant-1 与非 LLM 基线（Elliptic++）

本目录包含三部分：
- 数据：Elliptic++ 的来源、处理和划分；
- 非 LLM 基线的实验代码（见 `baselines/`）；
- Elephant-1 用 OneJev 跑的实验代码（见 `elephant/`），包括零样本、静态微调和滚动训练。

所有方法共用同一套数据切分和同一个评估脚本。

## 数据

### 下载

我们用的是 Elliptic++（Elmougy & Liu, KDD 2023）。
- GitHub：https://github.com/git-disl/EllipticPlusPlus
- 数据文件在 README 链接的 Google Drive 文件夹里（ID `1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l`）。

```bash
gdown --folder https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l -O data/raw/elliptic_pp
md5sum -c data/splits_ref/raw_elliptic_pp_md5.txt   # 核对下载的 9 个 csv 和我们用的是否一致
```

我们用到 7 个 csv，合计约 2.1 GB：
- `wallets_features.csv`：地址 × 时间步，55 个地址特征；
- `wallets_classes.csv`：地址标签；
- `txs_features.csv`：交易特征，共 182 个；
- `txs_classes.csv`：交易标签；
- `txs_edgelist.csv`：交易之间的边；
- `AddrTx_edgelist.csv`：输入地址 → 交易；
- `TxAddr_edgelist.csv`：交易 → 输出地址。

标签有三种：1 是非法，2 是合法，3 是未知。

### 清洗

完整步骤见 `data/DATA_PIPELINE.md`。用 `scripts/build_addr_sqo.py` 生成地址级数据，`scripts/build_tx_sqo.py` 生成交易级数据，`scripts/build_addr_txhist.py` 生成交易历史 state。

- **去重。** `wallets_features.csv` 里有些（地址, 时间步）行被原样重复，最多重复 568 次。我们只保留一份，共删掉 347,569 行，剩下 920,691 行。脚本会检查这些重复行是否完全相同。
- **未知标签。** 未知标签的行全部删除，不当作合法，也不参与训练。在图模型里，这些地址只作为节点提供特征，它们的标签从不使用。
- **审计。** 有两条检查，都通过了：
  - 每个地址的标签不随时间变化；
  - 一个地址是非法的，当且仅当它出现在至少一笔非法交易里。
- **记录的问题（只记录，不修改）。**
  - 55 个地址特征是全生命周期统计。同一个地址在不同时间步，这些特征完全相同。
  - 10.6% 的行里，`last_block_appeared_in` 超出了当前时间步，也就是用到了未来信息。
  - 这是我们后来做 txhist 输入的原因。
- **交易级。** 有 3 个特征和时间步几乎完全相关（|Spearman| ≥ 0.99），会泄露时间，我们把它们删掉了：Aggregate_feature_7、8、10。

### 特征构建（各模型的输入）

| 输入 | 用于 | 怎么构建 |
|---|---|---|
| 地址统计 state | Elephant（统计 state）、LR、RF、XGB | 55 个地址特征，按原始列顺序写成"- 特征名: 值"，一行一个。数值最多保留 6 位有效数字，缺失值写 `NA`。问题是 "Is this Bitcoin address involved in illicit activity?"，选项是 A. Illicit / B. Licit。Prompt 里不出现地址字符串、txId 和时间步，脚本里有检查。树模型直接用这 55 个数值 |
| 交易历史 state（`addr_txhist`） | Elephant txhist、XGB txhist | 用"截至当前步的交易历史"**替换**上面的生命周期统计，这样不含未来信息（`build_addr_txhist.py`）。内容分两部分：<br>• 汇总：发送和接收交易数、最早出现在几步之前、活跃了几步。<br>• 最近 32 笔交易，从新到旧：几步之前、地址的角色（发送 / 接收 / 两者）、交易总额、手续费、输入和输出地址数、最小和最大输出。<br>XGB txhist 把同样的信息展平成表格：汇总字段、最近 8 笔交易、各交易字段的最小 / 最大 / 平均值（`xgb_txhist.py`） |
| 每步的图快照 | GraphSAGE、GAT | 每个时间步一张图。节点是这一步出现的所有地址，包括未知标签的地址。边是这一步交易里"输入地址 → 输出地址"，转成无向，去掉自环。节点特征是 55 个地址特征做 signed log1p，再用训练集的均值和方差标准化 |
| 图聚合特征（165 维） | XGB-Graph | 按 GADBench 的做法，用无参数的均值聚合：h₀ 是自身特征，h₁ 是 1 跳邻居 h₀ 的均值，h₂ 是 1 跳邻居 h₁ 的均值。拼成 [h₀, h₁, h₂]，共 55×3 = 165 维。聚合只在同一时间步的图内进行，不跨时间步，也不用标签 |
| 交易 state | 交易级任务 | 两个版本：179 个特征的完整版，以及只用 17 个有名字特征的版本 |

### 划分

| 切分 | 时间步 | 地址级行数（非法） | 交易级行数（非法） |
|---|---|---|---|
| train | 1–29 | 169,438（7,859，4.6%） | 26,381（2,871） |
| val | 30–34 | 24,225（1,838，7.6%） | 3,513（591） |
| test | 35–49 | 97,756（5,023，5.1%） | 16,670（1,083） |

- **按时间切分，不做随机切分。** 阈值、checkpoint 和超参数只在验证集上选。
- **第 43 步前后分开报告。** 第 43 步前后暗网市场关停，分布突变，测试集按 `post43 = step ≥ 43` 拆成两段：地址级第 43 步前有 2,848 个非法，之后有 2,175 个。
- **地址重叠。** 测试集只有 2.7% 的行，其地址在训练期出现过。
- **交易级的标签预算**（1%、5%、10%）按（时间步, 标签）分层抽样，5 个 seed。同一个 seed 下 1% ⊂ 5% ⊂ 10%。预算文件在 `data/splits_ref/elliptic_pp_tx_sqo/budgets/`，直接用这些文件就行，不用重新抽。

重建后，用 `data/splits_ref/` 里的 md5 和各切分的 id 列表逐行核对，方法见 `data/DATA_PIPELINE.md` 第 5 节。

## 非 LLM 基线

见 `baselines/README.md`，包括 LR、RF、XGB（静态和滚动）、XGB-Graph、GraphSAGE、GAT、XGB txhist。

## Elephant-1（OneJev）

见 `elephant/README.md`，包括零样本、零样本加滚动重校准、静态微调（全参数和 LoRA）和滚动增量训练。

## 目录

```
README.md               本文件
requirements_jev.txt    数据处理、基线和评估的依赖
data/DATA_PIPELINE.md   数据下载、清洗、划分的完整步骤，以及核对方法
data/splits_ref/        原始文件和处理后文件的 md5、各切分的 id 列表、交易级标签预算
scripts/                共用脚本：数据构建、评估
baselines/              非 LLM 基线
elephant/               Elephant-1 / OneJev 代码
```
