# Paper trail

This is a route through original papers, not a completeness contest. Read the question, method, assumptions, evidence, and limitations before inheriting the headline.

## Representation and architecture

| Work | Read it for |
|---|---|
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | neural BPE motivation and merge procedure |
| [SentencePiece](https://arxiv.org/abs/1808.06226) | raw-text subword training and language-independent tooling |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | scaled dot-product attention and the original Transformer |
| [RoFormer](https://arxiv.org/abs/2104.09864) | rotary position embeddings |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | gated feed-forward variants including SwiGLU experiments |
| [GQA](https://arxiv.org/abs/2305.13245) | grouped-query attention and quality/speed trade-offs |

## Data and scaling

| Work | Read it for |
|---|---|
| [The Pile](https://arxiv.org/abs/2101.00027) | diverse open corpus design and datasheet |
| [Dolma](https://arxiv.org/abs/2402.00159) | open corpus construction, tooling, and ablations |
| [FineWeb](https://arxiv.org/abs/2406.17557) | large-scale web-data filtering and evaluation |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | empirical loss scaling under its studied regime |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | compute allocation between parameters and tokens |

Scaling laws are fitted observations under specific data, architecture, and optimization choices—not physical constants.

## Mixture of experts

| Work | Read it for |
|---|---|
| [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538) | conditional computation, noisy gating, and balancing |
| [GShard](https://arxiv.org/abs/2006.16668) | scaling conditional computation with automatic sharding |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | top-1 routing, capacity, auxiliary loss, and scale |
| [ST-MoE](https://arxiv.org/abs/2202.08906) | stable training and transfer behavior |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | sparse decoder model report and per-token expert activation |
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | fine-grained and shared expert design |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | published architecture and auxiliary-loss-free routing approach |
| [OLMoE](https://arxiv.org/abs/2409.02060) | open MoE training artifacts and analysis |

## Post-training and reasoning

| Work | Read it for |
|---|---|
| [InstructGPT](https://arxiv.org/abs/2203.02155) | demonstration data, preference data, reward modeling, and PPO pipeline |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | model-generated critique/revision and AI feedback framework |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | preference optimization without an explicit learned reward model loop |
| [Self-Instruct](https://arxiv.org/abs/2212.10560) | synthetic instruction generation and filtering |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | process versus outcome supervision experiments for math |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | published RL-centered reasoning model development |

## Inference

| Work | Read it for |
|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) | exact attention with IO-aware tiling |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | improved work partitioning and parallelism |
| [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | paged KV-cache memory management and serving throughput |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | exact-distribution draft/verify decoding |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | post-training activation/weight quantization |

## Prompts, retrieval, and agents

| Work | Read it for |
|---|---|
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | coupling learned generation with retrieved evidence |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | demonstrations with intermediate reasoning on studied models/tasks |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | sampling multiple reasoning paths and aggregating answers |
| [ReAct](https://arxiv.org/abs/2210.03629) | interleaving reasoning-like traces and external actions |
| [Toolformer](https://arxiv.org/abs/2302.04761) | self-supervised API-use training |

These works report empirical results, not universal prompt laws. Re-test on the exact model, task, tools, and evaluation protocol you operate.

