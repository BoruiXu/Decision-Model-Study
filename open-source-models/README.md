# 开源与开权重 Decision Models

先看高分模型的权重和原生读出，再按参数规模补充本地基线。表中规模沿用作者的型号口径。评测按对应模型版本和测试分组查看原始结果。[Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) · [Artificial Analysis Decision Index](https://artificialanalysis.ai/evaluations/artificial-analysis-decision-index)

## 优先阅读的模型

| 模型 | 规模 | 许可 | 学习重点 | 项目地址 |
| --- | --- | --- | --- | --- |
| Joey-1 | 2B 级 | MIT | 原生选项 logits、块双向注意力、可继续做全量 SFT | [模型与补丁](https://huggingface.co/baselabs/joey-1) · [官方微调代码](https://github.com/basetenlabs/ml-cookbook/tree/main/training-jobs/examples/joey-1-sft/training) |
| Elephant-1 | 9B | MIT | 与 Joey 同系列的较大模型，适合对照相同决策接口下的能力差异 | [模型仓库](https://huggingface.co/baselabs/elephant-1) · [微调代码](https://github.com/basetenlabs/ml-cookbook/tree/main/training-jobs/examples/elephant-1-sft/training) |
| Mammoth-1 | 27B | MIT | Base Labs 大模型基线，可对照共享前缀与选项读出 | [模型仓库](https://huggingface.co/baselabs/mammoth-1) |
| OneJev | 0.8B、4B、9B、27B | Apache-2.0 | 同时提供训练代码、数据和多个尺寸的权重，适合顺着代码学习整条流程 | [模型系列](https://huggingface.co/collections/OmniJev/onejev) · [27B 权重](https://huggingface.co/OmniJev/OneJev-27B) · [源码](https://github.com/OmniJev/OneJev) |
| Perplexity Decider v1.1 | 27B 级 | Apache-2.0 | 独立决策头、非因果完整注意力，以及公开的训练配置和数据清单 | [模型仓库](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) · [训练目录](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/main/training) |
| Cloudflare Clef | 27B 级 | Apache-2.0 | 联合决策头，可和 Joey 的 LM head 槽位读出对照 | [权重、决策头与加载代码](https://huggingface.co/Cloudflare/clef) |
| H2O-Lightning-4B | 4B 级 | Apache-2.0 | 多模态决策，模型卡列有 JevBench 和 ImageJevBench 结果 | [模型仓库](https://huggingface.co/h2oai/h2o-lightning-4b) |

## Mammoth 27B 与 GLiDE v1.2

| 对比项 | Mammoth 27B | GLiDE v1.2 |
| --- | --- | --- |
| 底座 | Qwen3.5-27B | Qwen3.8-27B |
| 注意力 | 48 层 Gated DeltaNet 保持因果递推。16 层全注意力在 `state` 和本题各自的块内双向，问题块也能看 `state`。 | 48 层 Gated DeltaNet 保持因果递推。16 层全注意力在整条题目输入内双向，`state` 和问题可以互相看见。 |
| 选项读出 | 用 LM head 在答案位置读取选项标签对应的词表 logits。 | 用独立的 255 维线性头在答案位置给各候选位置打分。 |

[Mammoth 架构与读出](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models) · [GLiDE v1.2 训练说明](https://fastino.ai/blog/training-glide-sota-decision-model) · [GLiDE v1.2 推理代码](https://huggingface.co/fastino/GLiDE-27B/blob/d199b6d7e04523d87986b8b68cd62272ed1cefc0/glide/model.py)

### 从工单读出选项概率

```text
state：客户说「我的银行卡被重复扣款了。」
问题：这张工单应该分给哪个团队？
A. 账单    B. 技术支持    C. 账号管理
```

两个模型都在题目末尾加上答案提示，并从 prefill 最后一个输入 token 的隐藏状态读出分数。Mammoth 取 LM head 中 `A`、`B`、`C` 对应的 logits。GLiDE 把隐藏状态送进专用线性头，取前 3 维，分别对应 `A`、`B`、`C` 在候选列表中的位置。

假设取出的 logits 为 `[2.0, 1.0, 0.0]`，按温度 1 在这三项上做 softmax，得到约 `[66.5%, 24.5%, 9.0%]`。服务端选概率最大的「账单」，并返回每个选项的概率。这里的数值用于演示计算，两种实现都可以在 prefill 后直接返回决策。[Mammoth 读出公式](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models) · [GLiDE softmax 与选项映射](https://huggingface.co/fastino/GLiDE-27B/blob/d199b6d7e04523d87986b8b68cd62272ed1cefc0/glide/model.py#L161)

### 多题时复用 state 的 prefill 缓存

给同一张工单再加一道「是否紧急？」。在 Mammoth 中，前缀（系统提示词和 `state`）内的 token 能看到整个前缀，但看不到问题块。换题时前缀的计算结果相同，Mammoth 先 prefill 一次，保存全注意力层的 K/V 和 Gated DeltaNet 层的卷积状态与递推状态。

```text
前缀 prefill 一次 → 前缀缓存
                     ├─ 分给哪个团队？ + A/B/C → 团队概率
                     └─ 是否紧急？     + Yes/No → P(Yes)
```

每道题都从原始前缀缓存继续，只输入自己的后缀（问题、候选项和答案提示）。公开实现串行时为每题复制缓存，批处理时把缓存复制到各题所在的 batch row，因此各题可以独立更新缓存。问题仍会读取前缀的 K/V，前缀本身只编码一次。[Mammoth 分支推理补丁](https://huggingface.co/baselabs/mammoth-1/blob/61cec44c281c693992e3d6ebeb44dd3402dfa186/serve.patch#L88) · [OneJev 缓存复制](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/engine.py#L241)

GLiDE v1.2 的公开实现把每道题和完整的 `state` 拼成各自的序列。由于全注意力层中的 `state` 表示会受本题影响，换题时会重新编码 `state`。这些序列可以放在同一个 batch 里执行。[GLiDE v1.2 推理代码](https://huggingface.co/fastino/GLiDE-27B/blob/d199b6d7e04523d87986b8b68cd62272ed1cefc0/glide/model.py#L333)

## 轻量模型与实现对照

| 模型 | 公开产物 | 项目地址 |
| --- | --- | --- |
| Intern-Decision | 0.8B、4B 权重和决策实现，适合学习 marker 读出 | [0.8B 权重](https://huggingface.co/internlm/Intern-Decision-0.8B) · [4B 权重](https://huggingface.co/internlm/Intern-Decision-4B) · [源码](https://github.com/InternLM/Intern-Decision) |
| Kev-4B | LoRA 适配器及决策实现，按模型卡加载对应骨干 | [适配器](https://huggingface.co/jaredpalmer/kev-4b) · [训练和推理源码](https://github.com/jaredpalmer/kev) |
| Mapika Decider-4B | 4B 权重和公开训练代码，可阅读 SFT 数据混合方案 | [模型仓库](https://huggingface.co/Mapika/decider-4b) · [源码](https://github.com/Mapika/decider) |
| IOTA Decide 4B v0.4 | Apache-2.0 权重，适合对照轻量决策头 | [模型与代码](https://huggingface.co/macrocosm-os/iota-decide-4b-v0.4) |
| Laya | 约 421M 的编码器决策模型，适合做低成本基线 | [权重](https://huggingface.co/convaiinnovations/laya) · [源码](https://github.com/NandhaKishorM/laya) |

## 使用自定义许可的公开权重

| 模型 | 权重许可 | 项目地址 |
| --- | --- | --- |
| GLiDE v1.2 | Fastino Community License | [v1.2 权重和推理代码](https://huggingface.co/fastino/GLiDE-27B/tree/glide-v1.2) |
| Drex Max v1 / Drex 2.0 | Nace.AI Open RAIL-M | [Drex Max v1](https://huggingface.co/nace-ai/drex-max-v1) · [Drex 2.0](https://huggingface.co/nace-ai/drex-v2.0) |
| Liquid d1-3B / d1-omni-600M | LFM Open License | [d1-3B](https://huggingface.co/LiquidAI/d1-3B) · [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) |

Base Labs 的技术报告可以配合三组权重阅读，重点看输入序列化、分块注意力、选项交叉熵和 Brier 项。[技术报告](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)
