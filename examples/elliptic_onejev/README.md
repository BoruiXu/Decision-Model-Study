# Elliptic++ 示例：从数据模块开始的一轮运行

任务：判断一个比特币地址（Elliptic++ 地址级数据）是不是非法地址。模型的输入是这个地址到当前时间步为止的交易历史（OneJev txhist 行），输出 `yes`（非法）或 `no`（合法）的概率。训练调优用 joey-1 单卡全参数默认配方，只跑**一轮**（`stop.max_rounds` 1）：① 数据 → ② 训练调优 → ③ 交付 → ④ 评测与决策，然后停止。

流程从数据模块开始：第一轮的数据 agent 读原始数据（任务的 `data.sources`），自己写预处理脚本把它写成 OneJev 行（脚本和全部带标签的行登记成 `corpus`），按数据定切分规则（任务只定了各集合的行数），由切分工具 `dset` 切出 testset、evalset、valset、训练池 pool 和 trainset。集合不手工做，也不手工登记。

| 文件 | 内容 |
|---|---|
| `task.template.json` | 任务定义模板：原始数据目录和机器相关的值是占位符 `<…>`；切分规则只写了调试规模的 `sizes` |
| `TASK.template.md` | 它的人读版，占位符相同 |
| `README.md` | 本文件 |

测试：`python3 -m unittest discover -s tests -p test_examples.py`（仓库根目录）。

## 原始数据

`01_build_data.sh`（DM-RSI-Framework 仓库的 `pipelines/elliptic_elephant/`）把数据写到 `$DMRSI_DATA/data/onejev/addr_txhist_noul/`：`train.jsonl`（169,438 行，非法 7,859）、`val.jsonl`（24,225 行，非法 1,838）、`test.jsonl`（97,756 行，非法 5,023）；`val_sub4000.jsonl` 不用。三个文件合计 291,419 行、非法 14,720 行，就是任务 `data.rows.expect` 的数。每行形如：

```
{"id": "addr_30_<地址>", "split": "val",
 "state": {"entity": "Bitcoin address transaction history", "summary": {...}, "transactions": [...]},
 "question": {"type": "noul", "instructions": "Is this Bitcoin address involved in illicit activity?"},
 "target": {"yes": 0.0, "no": 1.0},
 "meta": {"state_id": "<地址>", "step": 30, "y": 0, "post43": false, "seen_in_train": false}}
```

任务定义的 `data` 一节告诉数据 agent 要什么，怎么做由它定：

- `sources`：这个目录（占位符 `<ROWS_DIR>`）。只有第一轮产出集合的派出读它；之后的派出和其他模块都禁读，corpus 登记后你把它设成读不了（第 5 步）。
- `rows`：`about` 是行的约定：一行一个（地址，时间步），state 原样，`question` 用任务的问题（yes = illicit，no = licit），`meta.state_id` 是地址，时间步写进 `meta.extra.step`，上游 `meta` 的 `y`、`post43`、`seen_in_train` 不写出。`expect` 是全部行的行数和各标签行数，corpus 登记时逐个核对，对不上就拒绝：标签经过 agent 写的脚本，数目对得上才放行。
- `split`：只有调试规模的 `sizes`（trainset 1000、valset 500、evalset 500、testset 1000）。按哪个字段、怎么分段、分层、单位和 seed 由 agent 按数据定，理由写进它的 record；补好的规则记成任务定义的新版本 `task-0002`（只多了 `data.split`），不问你，也不需要审批。同一个地址出现在几个集合里是允许的，`dset` 只报告数目（agent 定 `unit: state_id` 时才分开）。要放大时只改 `sizes`，用新的任务目录和新工作区。

`data.sources` 也可以指向 Elliptic++ 的原始 CSV（`01_build_data.sh` 下载的那些），由 agent 的脚本自己建 txhist 行；这时 `rows.about` 要写清 txhist 行怎么建（例如照 `01_build_data.sh` 调用的 `build_addr_txhist.py` 的做法），`rows.expect` 照样核对。

## 步骤

下面的 `<REPO_ROOT>` 是部署好的框架仓库，`<W>` 是这次运行的工作目录，`<ROWS_DIR>` 是放上面三个文件的目录（绝对路径）。

### 1. 准备原始数据

在运行的机器上把三个文件放进 `<ROWS_DIR>`（在这台机器上跑 `01_build_data.sh`，或从别处复制过来并核对 sha256），核对行数：

```bash
wc -l <ROWS_DIR>/train.jsonl <ROWS_DIR>/val.jsonl <ROWS_DIR>/test.jsonl   # 169438、24225、97756
```

模块 agent 和你用同一个用户运行。这台机器上别的带标签的副本（`01_build_data.sh` 的中间结果 `$DMRSI_DATA/data/processed/elliptic_pp_addr_sqo/`、原始 CSV `$DMRSI_DATA/data/raw/elliptic_pp/`、别处的 `addr_txhist_noul`）不在禁读名单里：`rsi run` 之前把它们 `chmod 000`，或者不放在这台机器上。`<ROWS_DIR>` 本身在第 5 步才锁。

### 2. 任务定义

```bash
mkdir -p <W>/task
cp <REPO_ROOT>/examples/elliptic_onejev/task.template.json <W>/task/task.json
cp <REPO_ROOT>/examples/elliptic_onejev/TASK.template.md <W>/task/TASK.md
```

两个文件里的占位符用同一条命令一起改，两个文件保持一致：

| 占位符 | 填什么 |
|---|---|
| `<ROWS_DIR>` | 原始数据目录的绝对路径（`data.sources` 的 `path`） |
| `<REPO_ROOT>` | 框架仓库的绝对路径（`resources.train_command` 由它得出 `<REPO_ROOT>/modules/train/train`） |
| `<OJH_PYTHON>` | onejev-harness 的 `.venv/bin/python` |
| `<OJH_BIN>` | onejev-harness 的 `.venv/bin/ojh` |
| `<OJH_MACHINE>` | ojh 的 `machine.yaml` |
| `<OJH_ENV_FILE>` | ojh 的环境文件（`ojh-env.sh`） |
| `<CLAUDE_BIN>` | `claude` 的绝对路径（`runner.claude.command` 的第一项） |
| `"<GPU_IDS>"`（连引号） | 允许训练调优使用的卡号列表，例如 `[0, 1]`；`nvidia-smi --query-gpu=index,memory.used --format=csv` 看到有人占用的卡不写 |
| `"<JOEY1_GPU_HOURS>"`（连引号） | joey-1 一次训练的 GPU 小时估计（数字）。不做冒烟训练（开跑前还没有训练集），1000 行训练集先填 0.3（A100 上实测 0.18，略放宽） |

```bash
cd <W>/task
sed -i -e 's#<ROWS_DIR>#/abs/path/addr_txhist_noul#g' -e 's#<REPO_ROOT>#/abs/path/rsi#g' \
       -e 's#<OJH_PYTHON>#/abs/path/.venv/bin/python#g' -e 's#<OJH_BIN>#/abs/path/.venv/bin/ojh#g' \
       -e 's#<OJH_MACHINE>#/abs/path/machine.yaml#g' -e 's#<OJH_ENV_FILE>#/abs/path/ojh-env.sh#g' \
       -e 's#<CLAUDE_BIN>#/abs/path/claude#g' -e 's#"<GPU_IDS>"#[0, 1]#g' -e 's#"<JOEY1_GPU_HOURS>"#0.3#g' task.json TASK.md
grep -n '<[A-Z][A-Z0-9_]*>' task.json TASK.md    # 应当什么都不打印
PYTHONPATH=<REPO_ROOT> python3 -m rsi validate task <W>/task
```

`rsi init` 之后任务定义就固定了，再改 `resources.onejev.estimate` 或 `delivery.*` 要走 `task_change` 审批和新版本，所以这里先定好：`reproduction_tolerance` 用模板的 1e-4（A100 上跨卡完全复现），`max_decision_mismatch_rate` 用 0.001。数据 agent 补的切分规则是唯一不需要审批的新版本。

### 3. 建工作区

```bash
export PYTHONPATH=<REPO_ROOT> RSI_NO_BROWSER=1
bash <REPO_ROOT>/modules/train/install_env.sh --ojh <ojh 仓库> --machine <OJH_MACHINE> --env-file <OJH_ENV_FILE>   # 训练要的库装齐并检查；退出码 0 才往下走
python3 -m rsi init <W>/ws --task <W>/task --no-open
printf '# 工作笔记\n\nrepo root: %s\n' <REPO_ROOT> > <W>/ws/work/notes.md
python3 -m rsi status <W>/ws
<CLAUDE_BIN> -p 'reply ok' --output-format json      # 结果里有 ok：模块 agent 能启动
```

`<ojh 仓库>` 是 `<OJH_PYTHON>` 去掉末尾的 `/.venv/bin/python`；`install_env.sh` 重跑安全（机器就绪时什么都不改），用法见 `modules/train/README.md`「装环境」。

不手工登记任何集合：四个集合、corpus 和 pool 都由运行里的数据派出产生并登记。`<ROWS_DIR>` 保持可读，第一轮的数据派出要读它。

### 4. 在 tmux 里运行

按 `skills/rsi-orchestrator/SKILL.md` 的「运行」：每次运行一个输出文件，文件名在启动的 shell 里定好，记进 `<W>/ws/work/notes.md`。

```bash
cat > <W>/run.sh <<'EOF'
#!/bin/bash
# One `rsi run` on workspace $1 with the framework at $2; result in $3.json, status lines in $3.err, exit code in $3.exit.
PYTHONPATH=$2 python3 -m rsi run "$1" --max-steps 1000 --json > "$3.json" 2> "$3.err"; echo $? > "$3.exit"
EOF
out=<W>/ws/work/run-out-$(date +%Y%m%d-%H%M%S)-$$; echo "run output: $out" >> <W>/ws/work/notes.md
tmux new-session -d -s rsi-elliptic "bash <W>/run.sh <W>/ws <REPO_ROOT> $out"
until [ -e $out.exit ]; do sleep 300; done; cat $out.exit $out.json
```

退出码按 skill 的表处理，处理完再运行一次（重新执行最后三行，`out` 是新的）：

- `0`：运行结束，报告在 `<W>/ws/REPORT.md`。
- `10`：在等用户回答（计划、里程碑等）。把问题和选项转给用户，用 `rsi answer <W>/ws <选项> --ref <q-NNNN> --by <名字> --quote "<用户原话>"` 记下。切分规则不会来问；V1 之后会问里程碑。
- `11`：在等长任务（训练、大的预处理）。原因里有完成标记，`until [ -e <标记> ]; do sleep 300; done` 之后再运行。
- `13`：用完了 `--max-steps`（这里是 1000；每次定时检查约占 3 步，默认的 200 不够），不是故障：重新执行同一条命令（最后三行）。
- `1`：被拒或出错，照错误信息处理。

进度用 `python3 -m rsi status <W>/ws` 看（第一行 `now:` 是现在在做什么），`tail -n 3 $out.err` 看 `rsi run` 最近的状态行，或打开 `<W>/ws/dashboard.html`（等待期间每 `flow.page_refresh_seconds` 秒重写一次，默认 60）。

### 5. corpus 登记后锁住原始数据

第一轮数据派出的交回登记后（`corpus-0001` 在登记表里），之后的派出都不读原始数据，运行照常往下走。禁读名单只挡 agent 自己的文件工具，所以这时把它设成读不了：

```bash
until grep -q '"kind": "corpus"' <W>/ws/registry.jsonl; do sleep 60; done
chmod 000 <ROWS_DIR>
head -c1 <ROWS_DIR>/test.jsonl                       # 应当是 Permission denied
```

（第一行可以在第 4 步启动运行后就放到后台等。）以后要复跑预处理或重做数据，先 `chmod 755 <ROWS_DIR>`，做完再锁回。

### 6. 检查第一轮的数据

- 登记顺序：`task-0002`、`corpus-0001`、`testset-0001`、`evalset-0001`、`valset-0001`、`pool-0001`、`trainset-0001`（`python3 -m rsi status <W>/ws` 里 trainset 的质量门是 passed），然后是数据派出的 record；STATE 的 `approvals` 里没有 `task_change`。
- `task-0002` 和 `task-0001` 只差 `version` 和 `data.split`；`data.split` 完整，`sizes` 是 1000、500、500、1000。四个集合的行数也是这些。
- corpus 的 manifest 有 `rows`、`script`、`split_report`；trainset 的 `quality_report.json` 有 `split.shared_state_ids`（几个集合共享的地址数）；数据派出的 record 写了规则和为什么这样切（`hypothesis`）。
- 复跑审计（可选）：临时 `chmod 755 <ROWS_DIR>`，把 `<W>/ws/artifacts/corpus-0001/` 里登记的脚本复制到一个新目录 `<W>/audit/`，在那里按 corpus manifest 的 `command` 运行，新 `rows.jsonl` 的 sha256 应当等于 `<W>/ws/registry.jsonl` 里 `corpus-0001` 那行的 `files.rows`；做完立即 `chmod 000 <ROWS_DIR>`，删掉 `<W>/audit`（里面是带标签的行）。

只在验收运行里，结果存好、运行停止后，在运行之外删掉模型权重，只删这些路径：每个试验的 ojh 运行目录（`<W>/ws/work/round-*/train/trsi/trials/<试验>.json` 的 `run_dir`）下的 `runs/*/ckpt-step*`、`runs/*/final`，以及 `<W>/ws/artifacts/candidate-*/model/` 和 `<W>/ws/work/round-*/train/cand-*/model/`。正式运行不这样删：当前最佳版本的模型和 `<W>/ws/artifacts/` 一律不删，其余手动删除先记 `delete` 审批（见 `skills/rsi-orchestrator/SKILL.md`「审批点」）；用不上的 checkpoint 由 `trsi collect` 自动删。
