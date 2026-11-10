# Decision Model Training

决策模型接收上下文和带候选项的问题，输出候选概率。这个仓库整理公开权重、训练入口、数据格式和评测工具，重点关注能读源码、能换数据继续训练的项目。

| 目录 | 内容 |
| --- | --- |
| [open-source-models](open-source-models/README.md) | 高分决策模型、轻量基线、官方权重与评测入口 |
| [training-frameworks](training-frameworks/README.md) | Joey-1 微调配方、OneJev、其他决策训练实现和公开数据 |
| [books](books/llm-finetuning-best-practices-pitfalls-debugging.pdf) | 《LLM Fine-Tuning: Best Practices, Common Pitfalls, and Systematic Debugging》教材 PDF（中文，417 页；数据准备、SFT、LoRA/QLoRA、DPO、超参数、51 个故障条目与完整演练） |

## 阅读顺序

1. 从 Joey-1 的官方 SFT 示例开始。先看 `prepare_data.py` 如何把业务标签转成决策问题，再看 `run.sh` 如何启动 OneJev 训练。[Joey-1 SFT](https://github.com/basetenlabs/ml-cookbook/tree/main/training-jobs/examples/joey-1-sft/training)
2. 读 OneJev 的训练代码，跟踪 `state`、问题和候选项如何变成 token，选项分数如何进入 loss。[OneJev 训练入口](https://github.com/OmniJev/OneJev#training)
3. 用 Clef、Perplexity Decider 和 Kev 对照不同的决策读出。模型权重与对应实现见[模型目录](open-source-models/README.md)。
4. 把决策任务换成固定大小模型对的行为一致性预测。对同一请求离线生成两份回答，按预先定义的规则生成一致标签。训练时输入请求和历史，监督信号用 `yes/no`，评测同时看小模型承接比例和承接请求中的不一致率。具体的 Joey-1 读出和 loss 见[训练目录](training-frameworks/README.md#用-joey-1-做行为一致性预测)。
