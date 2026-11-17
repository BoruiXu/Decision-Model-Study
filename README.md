# Decision Model Training

决策模型接收上下文和带候选项的问题，输出候选概率。这个仓库整理公开权重、训练入口、数据格式和评测工具，重点关注能读源码、能换数据继续训练的项目。

| 目录 | 内容 |
| --- | --- |
| [open-source-models](open-source-models/README.md) | 高分决策模型、轻量基线、官方权重与评测入口 |
| [training-frameworks](training-frameworks/README.md) | Joey-1 微调配方、OneJev、其他决策训练实现和公开数据 |
| [books](books/llm-finetuning-best-practices-pitfalls-debugging.pdf) | 《LLM Fine-Tuning: Best Practices, Common Pitfalls, and Systematic Debugging》教材 PDF（中文，417 页；数据准备、SFT、LoRA/QLoRA、DPO、超参数、51 个故障条目与完整演练） |

## 在线学习资料

- [Decision Blocks（决策模型架构模块）](https://yilechen.net/DM-k7p9x2/architecture-blocks/#/)
- [Decision Landscape（决策智能产业全景）](https://yilechen.net/DM-k7p9x2/industrial-landscape/)
- **TandemBench** 用双人厨房协作题评测决策模型能否从候选团队方案中选出最优方案，以及能否判断候选项之外是否还有更好的方案。[论文](https://arxiv.org/abs/2611.05827) · [GitHub](https://github.com/lililibaohang/TandemBench)
- **Artificial Analysis Decision Index v2.1** 按六项评测的加权总分排名。五项采用平均 Decision Score，单次决策得分为 `50 × (1 + c − (p − c)²)`，其中 `c` 表示是否答对（0 或 1），`p` 是模型给所选答案的概率。答对时高置信度得分更高，答错时高置信度扣分更多。第六项 AA-Symmetry 衡量等价改写前后的选项和置信度是否一致。[榜单与分项](https://artificialanalysis.ai/evaluations/artificial-analysis-decision-index) · [评分方法](https://artificialanalysis.ai/methodology/decision-index)

  单项 **Accuracy** 通常只看选中的答案是否正确。页面 **Accuracy → Headline Metrics** 汇总各项的主指标，还包括文档全对率、非幻觉率和改写一致率。[GLiDE v1.2 权重（Qwen3.8-27B 底座）](https://huggingface.co/fastino/GLiDE-27B/tree/glide-v1.2) · [Mammoth 27B 权重](https://huggingface.co/baselabs/mammoth-1)

## 阅读顺序

1. 从 Joey-1 的官方 SFT 示例开始。先看 `prepare_data.py` 如何把业务标签转成决策问题，再看 `run.sh` 如何启动 OneJev 训练。[Joey-1 SFT](https://github.com/basetenlabs/ml-cookbook/tree/main/training-jobs/examples/joey-1-sft/training)
2. 读 OneJev 的训练代码，跟踪 `state`、问题和候选项如何变成 token，选项分数如何进入 loss。[OneJev 训练入口](https://github.com/OmniJev/OneJev#training)
3. 用 Clef、Perplexity Decider 和 Kev 对照不同的决策读出。再用同一张工单对照 Mammoth 和 GLiDE 的选项读出与多题前缀缓存。[模型目录](open-source-models/README.md) · [Mammoth 27B 与 GLiDE v1.2](open-source-models/README.md#mammoth-27b-与-glide-v12)
4. 把决策任务换成固定大小模型对的行为一致性预测。对同一请求离线生成两份回答，按预先定义的规则生成一致标签。训练时输入请求和历史，监督信号用 `yes/no`，评测同时看小模型承接比例和承接请求中的不一致率。具体的 Joey-1 读出和 loss 见[训练目录](training-frameworks/README.md#用-joey-1-做行为一致性预测)。
