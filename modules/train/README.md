# 训练调优模块（`modules/train/`）

在任务定义的配置空间（`search_space`）里搜训练配置：底座、配方、`onejev.<flag>`、`lora.<field>`、种子、步数。训练一律通过 onejev-harness（`ojh run`）。agent 做判断（选空闲卡、诊断、提候选、定时检查），`trsi` 做机械的事：生成 ojh 工作区、启动、等待、把每个跑完的训练打包成框架的候选、按规则比较、写 `search`。agent 的流程在 `SKILL.md`。

## 文件

| 位置 | 内容 |
|---|---|
| `SKILL.md` | 模块 agent 的流程 |
| `trsi/` | 命令行：`init`、`check`、`launch`、`watch`、`adopt`、`collect`、`next`、`report` |
| `train` | 框架的训练命令：训一个配置、打包成候选、写完成标记 |
| `install_env.sh`、`install_env.py` | 装环境：训练要的库装进 ojh 的 `.venv` 并整体检查（见「装环境」）；`.py` 是它的助手（探测、改机器配置、GPU 测试） |
| `recipes/` | 配方（ojh 格式） |
| `reference/` | 能搜什么、怎么比较、已经学到的 |
| `tests/` | pytest：假的 `ojh`（`fake_ojh.py`）和假的打分函数（`fake_scorer.py`） |

## 装环境

`install_env.sh` 把一次训练要的库装进 ojh 的 `.venv`，再整体检查一遍。重跑安全：机器已经就绪时什么都不改。模块 agent 遇到缺库或库不兼容时自己运行它（不问用户，见 `SKILL.md` 的「常见错误」）。

```bash
bash <repo>/modules/train/install_env.sh --ojh <ojh 仓库> --machine <machine.yaml> [--env-file <ojh-env.sh>] \
  [--cache-root <目录>] [--check] [--no-gpu-test]
```

- 任务里的对应：`--ojh` 是 `resources.onejev.python` 去掉末尾的 `/.venv/bin/python`，`--machine` 是 `resources.onejev.machine`，`--env-file` 是 `resources.onejev.env_file`（先 source 它）。
- 五步：① 没有 `<ojh>/.venv` 时按 ojh 的 README 运行 `setup.sh --machine <machine.yaml>`；已有的 venv 不重建，torch、triton、transformers、fla 的版本不动（每次安装都把它们当前的版本加作约束，装完再核对）。② pyyaml、pytest。③ causal-conv1d 1.7.0：按 torch 的 major.minor、CUDA 大版本、C++ ABI（`torch._C._GLIBCXX_USE_CXX11_ABI`）和 Python 选 GitHub release 的预编译 wheel；不行、且找到的 nvcc 和 torch 的 CUDA 大版本相同时从源码编译；都不行只警告（训练照常，走慢的 PyTorch 实现）。④ 有 compute capability ≥ 9.0 的卡（Hopper）时装 tilelang 0.1.15，找 nvcc（`CUDA_HOME`、`/usr/local/cuda*`、venv 里的 `nvidia-cuda-nvcc`，优先和 torch 的 CUDA 版本相同的），在机器配置的 `env` 里写 `CUDA_HOME`、完整的 `PATH` 和 `TILELANG_CACHE_DIR`。已有的 `CUDA_HOME` 的 nvcc 能运行就保留，除非它的 CUDA 大版本和 torch 的不同、而另有一个相同的（这时换掉并说明）。`PATH` 已有时保留其余各项、把 nvcc 的 bin 放到最前；没有时写 nvcc 的 bin、venv 的 bin 和存在的系统目录（`/usr/local/sbin` 到 `/bin`），不用调用者的 `PATH`（它会进训练和打分的进程；ojh 不展开变量）。先用这些新值跑一次第⑤步的 GPU 测试，通过了才写，写的正是测试用的值；测试失败就不写，报 failed；测试跑不了时（`--no-gpu-test`、没有可用的卡）照写，并给出警告。写法：用 pyyaml 读，逐行改（其他键、注释、被替换的键的行尾注释、CRLF 都保留），先备份成 `machine.yaml.bak-<时间>`；已经对了就不动；`env` 是内联写法（`env: {}` 除外）改不了时报 failed，并列出要手工设的值。⑤ 检查：import torch、transformers、fla、causal_conv1d、yaml（Hopper 上加 tilelang）；在一张没有别的进程的卡上、用机器配置的 `env`，对随机张量跑一次 fla 的 `chunk_gated_delta_rule`（joey-1 用的算子）前向加反向，以及 `causal_conv1d_fn`；`ojh doctor --no-network`。
- GPU 测试的卡：调用方设了 `CUDA_VISIBLE_DEVICES` 就只用它（模块 agent 传 `resources.gpu_ids` 里一张没有计算进程的卡的 UUID，见 `SKILL.md`）；没设时脚本在所有卡里挑（机器配置排除的卡和空闲显存不够的除外），优先没有别的进程的，都有进程时也会用有进程的卡，也不看 `resources.gpu_ids` 和 ojh 的卡锁。
- 安装都带 ojh 的 `-c constraints.txt`；用 uv（PATH、`$OJH_TOOLS_DIR/uv/uv`、`<ojh>/.tools/uv/uv`、`<机器配置所在目录>/tools/uv/uv`），没有时用 venv 里的 pip。
- 缓存和临时文件（pip、uv、`TMPDIR`、triton、tilelang）放在 `--cache-root`，缺省是 `<机器配置所在目录>/cache`；日志写到 `<cache-root>/logs/install_env-<时间>.log`。
- 每项打印 ok、missing 或 failed（skip、warn 只是说明），最后是 summary（改了什么、警告、失败）。有必需的一项 failed 时退出码 1；causal-conv1d 和 pytest 不是必需的，装不上只警告。
- `--check`：不装不改，不跑 GPU 测试，不写日志，只打印缺什么、会做什么；什么都不用做时退出码 0，否则 1。
- Hopper 检查有效的验证：`FLA_TILELANG=0 bash install_env.sh …` 关掉 fla 的 tilelang 后端（不用卸载），GPU 测试应当报 fla 的 Triton/Hopper 错误，退出码 1。
- 第一次跑 GPU 测试要编译内核（H100 上约一两分钟）；从源码编译 causal-conv1d 要十分钟以上。

## 运行

`trsi` 和训练命令在 ojh 的 `.venv` 里运行（要 pyyaml；候选的 `model.predict` 要 torch 和 ojh），框架本身不依赖它们。下面 `<py>` 是任务定义的 `resources.onejev.python`（ojh 的 `.venv/bin/python`），`<repo>` 是框架仓库根目录。

```bash
export PYTHONPATH=<repo>/modules/train
<py> -m trsi --help
<py> -m trsi init --brief <输出目录>/brief.json          # 写 <输出目录>/trsi/run.yaml、exp.yaml，起点成为当前方案
<py> -m trsi adopt <run.yaml> --candidate <best_candidate>  # 第 2 轮起：当前最佳版本的候选当作起点的一个种子，不重训
<py> -m trsi launch <run.yaml> --start --seed 0 --gpus 3    # 或 launch <run.yaml> <候选.yaml> --seed 0 --gpus 3；立即返回
<py> -m trsi watch <run.yaml> --marker <标记路径>            # 转到后台，写 <标记>.pid；试验都结束后写 ok
<py> -m trsi collect <run.yaml> [--gpu 3]                   # 每个跑完的试验打包成 <输出目录>/cand-<试验>/
<py> -m trsi next <run.yaml>                                # 按规则比较，打印下一步
<py> -m trsi report <run.yaml> --search <输出目录>/search --ids <输出目录>/registered-<k>.json
```

- `run.yaml` 由 `init --brief` 按简报和任务定义生成，各键的来源见 `trsi/config.py` 开头，不手写。
- 候选目录：`model/`（选中的 checkpoint；LoRA 是 ojh 合并后的模型；硬链接，跨盘时拷贝，所以 `ojh cleanup` 删了运行里的文件，候选照样完整）、`predict`（`model.predict`，用训练时写入的 `<py>` 运行 `predict.py`）、`predict.json`（打分设置）、`predictions_val.jsonl`、`manifest.json`（`rsi score` 的指标、`config`、`config_request`、`cost`、`code_version`）。打包后用 `rsi validate candidate` 检查。
- 打分用的卡：`collect` 在打包每个试验前选（`harness.pick_scoring_card`）：在 `resources.gpu_ids` 里，nvidia-smi 列出的计算进程一个都没有（谁的都算），空闲显存不少于 max(机器配置的 `min_free_mib`, 模型目录大小 × 1.5)，配了 `lock_dir` 时 ojh 的卡锁 `<lock_dir>/<uuid>.lock` 能立即拿到（打分期间拿着）；`min_free_mib`、`lock_dir` 取试验 `run.lock.yaml` 的 `machine.gpu`，没有时取 run.yaml 的 `ojh.machine`。`--gpu <卡号>`（必须在 `gpu_ids` 里）合格就用它，否则打印原因、取空闲显存最多的合格卡；nvidia-smi 显示 `[N/A]` 的卡不合格。按 uuid 设 `CUDA_VISIBLE_DEVICES`。`model.predict` 显存不够（OOM）时换一张合格卡再打一次。用的卡记进试验结果和 `result` 事件的 `scoring_card`。
- 打包失败（预测、打分或校验出错）的试验记为 unpacked：不比较、不重训，checkpoint 留在 ojh 的运行目录，下一次 `collect` 再打包；`next` 打印打包错误；`launch ... --force` 把它记为失败（卡时照记）后重训。
- ojh 的 tmux 会话不在了、又超过 2 分钟（`harness.LOST_GRACE_S`）没有 `DONE`、运行目录里也没有文件变动：试验失败（ended without DONE），`watch`、`collect` 和训练命令都这样算；kill-session 之后 ojh 约 30 秒内才写 `DONE`，这段时间仍算在跑。
- 候选 manifest 的 `config` 是 ojh 的 `run.lock.yaml`，但指向带标签数据的路径（`data` 的 `train`、`select`、`select_index`、`prepared_dir`、`source`，`eval_sets.*` 的 `train_labels`、`reference`，`split_file`）写成 null，sha256 照留。
- 用不上的 checkpoint 由 `collect` 自动删（不问用户）：试验打包好、候选通过 `rsi validate` 之后，对它的 ojh 运行做 `ojh cleanup --confirm`（删 `deletion_candidates.json` 列出的没选中的保存点和合并模型；ojh 留着选中的）；`DONE` 说失败的试验，删掉运行里所有 checkpoint 目录（`ckpt-step*`、`adapter-step*`、`merged-step*`、`final`、`adapter`、`*.partial`，失败的运行 ojh 不列）。重跑的试验在记录的 `old_run_dirs` 里留着被换下的运行，它们的 `DONE` 出现后同样整个删掉（`launch --force` 在 launch 成功后马上删）。unpacked、还在跑、没有 `DONE` 的不删。删了记 `cleanup` 事件；ojh 留下的路径记在试验记录的 `cleanup_left`，只报一次；没删成只打印，下一次 `collect` 再删。
- `--ids`：引擎在第一段交回登记后写的 `registered-<k>.json`（`[{"dir", "kind", "id"}]`，只取 candidate），可以给多个；也接受 JSON `{"cand-<试验>": "candidate-NNNN"}`。
- `watch` 收到 SIGTERM（`rsi stop`）时重新读本次运行的试验，`tmux kill-session` 结束所有还在跑的 ojh 训练（包括 watch 启动之后才 launch 的），标记写 failed；后台 watch 出了别的错也写 failed 标记，traceback 在 `<标记>.log`。
- 进度文件：`launch` 之后、`watch` 每次轮询、`collect` 打包每个试验前后，重写 run.yaml 的 `framework.progress_file`（简报的 `progress_file`，缺省 `<输出目录>/progress.json`；格式见框架的 `docs/interfaces/progress.md`）：在跑的训练和卡、`配置`、`GPU 小时`、`当前方案`，每个试验一行（状态从 ojh 运行目录的 `train-s<种子>.log` 读步数）。写失败只打到 stderr（后台 watch 的日志），从不让命令失败。

训练命令（契约见 `SKILL.md`）：

```bash
PYTHONPATH=<repo> <repo>/modules/train/train --task <task.json> --trainset <dir> --trainset-id <id> \
  --valset <dir> --valset-id <id> --config <config.json> --config-source default --round 1 \
  --producer train --out <候选目录> --marker <完成标记>
```

- `config.json`：`{"base", "recipe", "overrides", "seed", "max_steps"}`，都可省，默认 joey-1、目录里的默认配方（`recipe: null`）、不改、种子 0。
- 用任何 `python3` 启动都可以：它会换到 `resources.onejev.python` 里重新运行。
- 卡：`RSI_TRAIN_GPUS=0`（或 `0,1`）指定；不设时从 `resources.gpu_ids` 里取 nvidia-smi 显示空闲的卡。
- 机器配置：`resources.onejev.machine`，没写时用 `RSI_ONEJEV_MACHINE`。
- `RSI_TRAIN_POLL`：检查 ojh 训练是否结束的间隔秒数，默认 60。
- `--producer` 只接受 `train`（数据模块不训练；旧候选里的 `data` 照常能读）。ojh 的实验名用 `train-<--out 实路径的 sha256 前 8 位>`，同一配置训到不同的 `--out` 不撞名。
- 训练结束后先等 ojh 的 tmux 会话退出（最多 2 分钟：ojh 写 `DONE` 时还占着卡），再打包；打分首选它训练用的卡，和 `collect --gpu` 一样核对，不合格就换卡；一张合格的卡都没有时每 30 秒（`RSI_TRAIN_POLL` 更短时用它）再试，最多 10 分钟。收到 SIGTERM 或 SIGINT 时结束自己的 ojh 训练（`tmux kill-session`），标记写 failed；ojh 正在启动训练时收到的，等记下 tmux 会话再处理。
- 开始、每次检查 ojh 训练、结束和失败时重写 `<--out 的父目录>/progress.json`；同一个目录里同时跑两个训练命令时后写的覆盖先写的。
- 所有候选的 `code_version.sha256` 都是 `train` 这个文件的 sha256。改了 `trsi/` 就把 `train` 里的 `TRSI_VERSION` 和 `trsi/__init__.py` 的 `__version__` 一起改（不一致时训练命令拒绝运行）；ojh 或 OneJev 换版本时改 `OJH_COMMIT`、`ONEJEV_COMMIT`。

## 测试

只要 CPU，不需要 ojh 和 GPU：

```bash
python3 -m venv /tmp/trsi-venv && /tmp/trsi-venv/bin/pip install pyyaml pytest
cd <repo>/modules/train && /tmp/trsi-venv/bin/python -m pytest -q
```

在 A100 上用 ojh 的 `.venv`（已有 pyyaml，`pip install pytest` 后）：`cd <repo>/modules/train && <py> -m pytest -q`。

测试用 `tests/fake_ojh.py` 代替 `ojh`，用 `tests/fake_scorer.py` 代替 `harness.score.score_checkpoint`（打包时设了 `TRSI_TEST_SCORER` 才把它写进 `predict.json`，`model.predict` 只看 `predict.json`），用 `tests/conftest.py` 写的假 `nvidia-smi`、`tmux`；打包用到的 `rsi score`、`rsi validate` 是框架的真命令。框架自己的测试（`<repo>/tests`）不包含这里。
