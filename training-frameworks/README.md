# Decision Model 训练框架

## 专用训练实现

| 项目 | 可以学习什么 | 项目地址 |
| --- | --- | --- |
| Baseten Joey-1 SFT | 从 Joey-1 继续全量微调。`prepare_data.py` 定义输入和目标，`run.sh` 用 `torchrun` 启动 OneJev，`tied_embeddings.patch` 处理 FSDP 下共享的输入 embedding 和 LM head | [官方示例目录](https://github.com/basetenlabs/ml-cookbook/tree/main/training-jobs/examples/joey-1-sft/training) |
| OneJev | 多模态决策 SFT、选项分布 loss、全量微调、LoRA 与 FSDP。主仓库提供四个尺寸的配置 | [源码](https://github.com/OmniJev/OneJev) · [训练入口](https://github.com/OmniJev/OneJev/blob/main/train/sft.py) · [训练配置](https://github.com/OmniJev/OneJev/tree/main/train/configs) |
| Perplexity Decider v1.1 | 查看发布检查点对应的配置、数据清单、Slurm 启动脚本，以及决策头实现 | [源码目录](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/main/source) · [训练目录](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/main/training) |
| Kev | 比较 pointer head、LoRA 和概率校准 | [项目仓库](https://github.com/jaredpalmer/kev) |
| Mapika Decider | 阅读多阶段 SFT、公开数据混合和拒答样本构造 | [项目仓库](https://github.com/Mapika/decider) · [训练说明](https://github.com/Mapika/decider/blob/main/MODEL_CARD.md#training) |
| OpenJev | 对照 SFT 和 REINFORCE-Analysis 两种训练路径 | [项目仓库](https://github.com/ejhshen/OpenJev) · [训练文档](https://github.com/ejhshen/OpenJev/blob/main/docs/TRAINING.md) |
| train-jev | 中文说明、Qwen 骨干上的决策头、LoRA 和 KL 蒸馏 | [项目仓库](https://github.com/shibing624/train-jev) |
| JevTrainer | 用 YAML 切换 marker、slot 和 pointer 读出，对比 LoRA 和全量微调 | [公开代码](https://github.com/hongliang-wei/jevtrainer) |

## 基础库、数据和评测

| 资源 | 用途 | 项目地址 |
| --- | --- | --- |
| PyTorch | 反向传播、AdamW、`torchrun` 和 FSDP | [源码](https://github.com/pytorch/pytorch) · [FSDP2 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) |
| Transformers | 加载模型、tokenizer 和 processor，适配模型前向 | [源码](https://github.com/huggingface/transformers) |
| PEFT | LoRA 的配置、加载与合并 | [源码](https://github.com/huggingface/peft) |
| OneJev-Data | 公开的 94,707 条 OneJev 训练记录，提供 100 条的入门样本 | [数据仓库](https://huggingface.co/datasets/OmniJev/OneJev-Data) |
| BANKING77 | Joey-1 官方 SFT 示例的数据 | [数据仓库](https://huggingface.co/datasets/mteb/banking77) · [官方转换脚本](https://github.com/basetenlabs/ml-cookbook/blob/main/training-jobs/examples/joey-1-sft/training/prepare_data.py) |
| Decision Index | 加载决策模型并执行评测 | [评测代码](https://github.com/apolinario/decision-index) · [榜单](https://huggingface.co/spaces/multimodalart/jev-decision-index) |

## 用 Joey-1 做行为一致性预测

1. **读懂数据。** 按官方示例构造 `state`、`question` 和 `target`。`state` 放当前请求、已有对话和工具状态。`question` 用 `noul` 询问固定的大小模型对是否会表现一致。
2. **离线生成标签。** 让两个生成模型读取同一状态。按约定比较最终答案、工具调用或程序行为。一致为 `yes`，有行为差异为 `no`。原始输出和标签证据单独存放，router 输入只含决策前可用的信息。
3. **跟踪原生读出。** 从 Joey 权重中的 `setup.sh` 和 `serve.patch` 找到配套的 OneJev 版本。训练走 `train.sft` 和 `batch_scores`，保留 `noncausal` 完整注意力和 `slot_lm`。在答案位置按渲染后的选项顺序取两个分数，对 `s_yes - s_no` 计算 BCE，等价于对两个原生选项做交叉熵。[Joey 模型文件](https://huggingface.co/baselabs/joey-1/tree/main) · [OneJev 训练补丁](https://huggingface.co/baselabs/joey-1/blob/main/serve.patch)
4. **看阈值是否有效。** 在校准集选阈值，再在独立测试集报告小模型承接比例、承接请求中的不一致率和整体行为保持率。把阈值结果与全选大、全选小以及未微调的 Joey 比较。

官方 Joey 示例的交叉熵会加 Brier 项。复用它做纯 BCE 实验时，把 `brier_weight` 显式设为 0。对于 soft target，`target` 可以保存同一状态多次配对的一致比例。使用 FSDP 时，直接参考官方示例的共享权重补丁和启动配置。[官方 `run.sh`](https://github.com/basetenlabs/ml-cookbook/blob/main/training-jobs/examples/joey-1-sft/training/run.sh) · [共享权重补丁](https://github.com/basetenlabs/ml-cookbook/blob/main/training-jobs/examples/joey-1-sft/training/tied_embeddings.patch)
