# 数据下载、清洗与划分（Elliptic++，地址级和交易级）

本文的目标是：在新机器上重新下载原始数据、运行两个构建脚本之后，得到的数据和本机**逐字节一致**。核对方法见第 5 节。

---

## 1. 下载原始数据

- **来源：** GitHub `git-disl/EllipticPlusPlus`（Elmougy & Liu, KDD 2023）。README 里链接的 Google Drive 文件夹 ID 是 `1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l`。

```bash
conda activate jev                     # 依赖见 env/requirements_jev_freeze.txt，至少需要 pandas、numpy、gdown、transformers
cd ~/Jev && mkdir -p data/raw/elliptic_pp
gdown --folder https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l -O data/raw/elliptic_pp
# 如果 csv 被放进了子目录，把 9 个 csv 移到 data/raw/elliptic_pp/ 根目录下（目录里可能还有别的文件，用不到）
md5sum -c data/splits_ref/raw_elliptic_pp_md5.txt   # 和本机的原始文件比对（在 ~/Jev 目录下运行）
```

**9 个文件的行数（不含表头），总大小约 2.1 GB：**

| 文件 | 行数 | 内容 |
|---|---|---|
| txs_features.csv | 203,769 | txId、Time step，以及 182 个特征：93 个 Local、72 个 Aggregate、17 个有名字的 |
| txs_classes.csv | 203,769 | txId, class（1 = 非法，2 = 合法，3 = 未知） |
| txs_edgelist.csv | 234,355 | 交易 → 交易 |
| wallets_features.csv | 1,268,260 | address、Time step，以及 55 个地址特征 |
| wallets_features_classes_combined.csv | 1,268,260 | 上面的文件加上 class，**我们没有用** |
| wallets_classes.csv | 822,942 | address, class |
| AddrTx_edgelist.csv | 477,117 | 输入地址 → 交易 |
| TxAddr_edgelist.csv | 837,124 | 交易 → 输出地址 |
| AddrAddr_edgelist.csv | 2,868,964 | 地址 → 地址，**目前没用** |

如果 md5 和本机不一致（比如官方更新了数据），**先核对上表的行数**，再按第 5 节比对处理后的样本 id。

---

## 2. 共同的清洗规则（两个数据集都适用）

1. **标签：** 1 → y = 1（非法，选项 A），2 → y = 0（合法，选项 B），**3（未知）的行直接删除**。
   - 未知标签不用于训练，也不当作合法。
   - 未知标签只在两个地方用到：GNN 的图里作为节点（只用特征，不用标签），以及 `seen_in_train` 的判定。
2. **时间切分：**
   - 第 1–29 步 → `train`；第 30–34 步 → `val`；第 35–49 步 → `test`。
   - 代码：`split_of(step) = np.select([step <= 29, step <= 34], ["train", "val"], "test")`。
   - val 只用来选 checkpoint 和阈值。
3. **`post43 = step >= 43`：** 第 43 步前后暗网市场关停，分布发生突变，评估时分开报告。
4. **数值格式化 `fmt(v)`：**
   - 缺失值写成 `NA`；
   - 整数值写成整数；
   - 其余用 `f"{v:.6g}"`，即最多 6 位有效数字。
5. **State 文本：** 每个特征一行，格式是 `- <特征名>: <值>`，按原始列的顺序排列。
6. **Prompt 中不出现地址字符串、txId 和时间步。** 构建脚本里有 assert 检查。
7. **问题和选项：**
   - 地址级的问题是 "Is this Bitcoin address involved in illicit activity?"；
   - 交易级的问题是 "Is this Bitcoin transaction involved in illicit activity?"；
   - 选项都是 `{"A": "Illicit", "B": "Licit"}`。
8. **排序和 id：** 记录按 (step, address) 或 (step, txId) 排序。id 的格式是 `addr_<step>_<address>` 或 `tx_<step>_<txId>`。

---

## 3. 地址级数据集 `elliptic_pp_addr_sqo`（T1）

```bash
python scripts/build_addr_sqo.py data/raw/elliptic_pp data/processed/elliptic_pp_addr_sqo
```

**处理步骤：**
1. 读取 `wallets_features.csv`。特征是除 address 和 Time step 以外的 55 列，列名里的空格换成下划线，比如 `num_txs_as receiver` 改成 `num_txs_as_receiver`。
2. **去重：** 原始文件里有些 (address, step) 行被原样重复了多次，最多重复 568 次。只保留一份，共删掉 **347,569** 行，剩下 920,691 行。
   - 脚本会 assert 这些重复行完全相同。
3. 和 `wallets_classes.csv` 做多对一 join，要求每一行都能拿到标签。
4. **审计（assert）：**
   - 每个地址的标签不随时间变化；
   - "地址非法 ⇔ 该地址出现在至少一笔非法交易的输入或输出中"，这条规则对所有地址都成立。
5. **量化未来信息（只记录，不修改）：**
   - 多时间步出现的地址，其 55 个特征 100% 在各行相同，说明这些是全生命周期统计；
   - 10.6% 的行，`last_block_appeared_in` 超出了当前时间步的区块范围；
   - 17.3% 的行，`num_timesteps_appeared_in` 大于 1。
6. 删除未知标签的行，添加以下字段：
   - `seen_in_train`：该地址在第 1–29 步出现过（任何行，包括未知标签的行），并且当前行不在训练集。
   - `post43`；
   - `id`。
7. 写出 `{train,val,test}.jsonl`、`CHECKS.md` 和 `manifest.json`。

**记录字段：** `id, address, step, split, y, label, state（特征名到 float 或 None 的字典）, state_text, question, options, seen_in_train, post43`。

| 切分 | 行数 | 非法 | 非法率 |
|---|---|---|---|
| train（1–29） | 169,438 | 7,859 | 4.64% |
| val（30–34） | 24,225 | 1,838 | 7.59% |
| test（35–49） | 97,756 | 5,023 | 5.14%（第 43 步前 2,848，第 43 步后 2,175） |

**其他统计：**
- 实体重叠：val 中 15.7% 的行、test 中 2.7% 的行，地址在训练期出现过。
- Token 长度（Qwen3 tokenizer，state 加 question）：平均 714，最长 817。

---

## 4. 交易级数据集 `elliptic_pp_tx_sqo` 和 `_named`（T2，迁移任务的目标）

```bash
python scripts/build_tx_sqo.py data/raw/elliptic_pp data/processed/elliptic_pp_tx_sqo
# 一次生成三样：elliptic_pp_tx_sqo/、elliptic_pp_tx_sqo_named/、elliptic_pp_tx_sqo/budgets/
```

**处理步骤：**
1. 读取 `txs_features.csv`。共 182 个非时间特征：93 个匿名 Local、72 个匿名 Aggregate、17 个有名字的。
   - 论文里说的 183 个特征包含了 Time step，时间步不进 prompt。
   - txId 无重复。
   - 17 个有名字的特征有缺失值，最多占 1.11%，写成 `NA`。
2. **去掉时间代理特征。** 规则：和 Time step 的 |Spearman| ≥ 0.99，**并且**同一时间步内的标准差小于总标准差的 0.2。
   - 符合规则的有 Aggregate_feature_8（ρ = 0.998）、Aggregate_feature_10（0.997）、Aggregate_feature_7（0.995），共 3 个，删掉。
   - Aggregate_feature_43 的 ρ 也有 0.995，但它在同一时间步内的标准差比例是 0.34，不符合规则，所以保留。
3. **两种输入变体，行和 id 完全相同：**
   - `elliptic_pp_tx_sqo/`：179 个特征（182 − 3），prompt 平均约 2,980 个 token，最长 3,048。
   - `elliptic_pp_tx_sqo_named/`：只用 17 个有名字的特征，平均约 238 个 token。
   - 有名字的特征是：in_txs_degree, out_txs_degree, total_BTC, fees, size, num_input_addresses, num_output_addresses, in_BTC_min/max/mean/median/total, out_BTC_min/max/mean/median/total。
4. 删除未知标签的交易。额外添加 `header = "Bitcoin transaction statistics:"`；`seen_in_train` 一律为 False。另外加了两个和地址关联的字段：
   - `addr_link`：这笔交易碰到了某个在地址训练集里有标签行的地址；
   - `addr_link_illicit`：碰到的地址在地址训练集里是非法的。
5. **标签预算** `budgets/{0.01,0.05,0.10}_s{0..4}.txt`：每个文件是训练集 id 的列表，一行一个。
   - 抽样时按 (step, y) 分层。对每个 seed：
     - 用 `rng = np.random.default_rng(seed)` 给每一行抽一个均匀随机数，按它算出该行在所在层内的排名；
     - 再给每一层抽一个偏移量 v ∈ [0, 1)；
     - 预算 b 下，每层取 `floor(b · n_层 + v)` 行，即排名最靠前的那些行。
   - 这样做保证同一个 seed 下 **1% ⊂ 5% ⊂ 10%**，而且每个预算都覆盖全部 29 个训练时间步。
   - 两种输入变体、所有方法，都用同一批预算样本。
   - 100% 预算就是全部 26,381 行训练集。

| 切分 | 交易数 | 非法 |
|---|---|---|
| train | 26,381 | 2,871 |
| val | 3,513 | 591 |
| test | 16,670 | 1,083（非法率：第 43 步前 9.2%，第 43 步后 2.5%） |

| 预算 | 平均行数（最少到最多） | 平均非法数（最少到最多） |
|---|---|---|
| 1% | 265（262–269） | 28（26–33） |
| 5% | 1,321（1,319–1,323） | 145（143–148） |
| 10% | 2,638（2,636–2,642） | 287（285–289） |

**地址和交易的关联（解释迁移实验时要用到）：**
- 地址标签是交易标签的函数。地址训练集的非法行里，只有 0.6% 是完全靠第 30 步以后的交易来判定的。
- 20.7% 的测试集交易碰到了在地址训练集里有标签行的地址。这部分交易的非法率是 0.3%，其余交易是 8.1%。
- 只有 1.1% 的测试集非法交易碰到了训练集里的非法地址。

---

## 5. 核对重建结果是否和本机一致

```bash
python - <<'EOF'
import hashlib, json, pathlib, pandas as pd
ref = json.load(open("data/splits_ref/processed_jsonl_md5.json"))
for k, v in ref.items():
    f = pathlib.Path("data/processed") / k
    md5 = hashlib.md5(f.read_bytes()).hexdigest()
    print(k, "OK" if md5 == v["md5"] else "MD5 DIFF", v["rows"])
    if md5 != v["md5"]:   # md5 不一致时，比较 id、step、y 是否一致
        d, s = k.split("/"); s = s.replace(".jsonl", "")
        a = pd.read_json(f, lines=True)[["id", "step", "y"]]
        b = pd.read_csv(f"data/splits_ref/{d}/{s}_ids.csv.gz")[["id", "step", "y"]]
        print("   same ids/labels:", a.sort_values("id").reset_index(drop=True).equals(b.sort_values("id").reset_index(drop=True)))
EOF
diff -r data/processed/elliptic_pp_tx_sqo/budgets data/splits_ref/elliptic_pp_tx_sqo/budgets && echo "budgets identical"
```

- 如果 md5 不一致，但 id、step、y 都一致，可能是 pandas 版本不同，导致浮点数格式或 json 序列化有细微差别。这种情况可以接受，但要记录下来。
- 如果 id 不一致，**不要继续做实验**，先检查原始数据。
- 为了避免因为 numpy 版本不同导致预算抽样不同，建议**直接使用包里的 `data/splits_ref/elliptic_pp_tx_sqo/budgets`（`cp -r` 覆盖过去）**，覆盖重新生成的那份。

---

## 6. 其他派生数据

| 内容 | 生成命令 | 说明 |
|---|---|---|
| GNN 图快照 | `baselines/gnn_addr.py` 第一次运行时自动缓存到 `.../graph_snapshots.pt` | 每个时间步一张图：节点是该步的全部 (地址, 步) 行（包括未知标签）；边是同一步内交易的输入地址到输出地址，转成无向，去掉自环；特征做 signed log1p，再用训练集的统计量标准化 |
