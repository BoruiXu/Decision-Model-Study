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
| Drex Max v1 / Drex 2.0 | Nace.AI Open RAIL-M | [Drex Max v1](https://huggingface.co/nace-ai/drex-max-v1) · [Drex 2.0](https://huggingface.co/nace-ai/drex-v2.0) |
| Liquid d1-3B / d1-omni-600M | LFM Open License | [d1-3B](https://huggingface.co/LiquidAI/d1-3B) · [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) |

Base Labs 的技术报告可以配合三组权重阅读，重点看输入序列化、分块注意力、选项交叉熵和 Brier 项。[技术报告](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)
