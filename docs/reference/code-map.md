# Source-code map

Use this page as a guided index. Open one teaching implementation and one production implementation for the same mechanism. Default-branch links are for orientation; pin a release or commit for reproducible analysis.

## Model forward pass

| Question | Compact trail | Scaled trail |
|---|---|---|
| How does a decoder block fit together? | [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) | [torchtitan Llama model](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model.py) |
| How are multiple architectures configured? | [LitGPT `config.py`](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/config.py) | [Hugging Face model implementations](https://github.com/huggingface/transformers/tree/main/src/transformers/models) |
| How does this book's model work? | [`src/open_llm_lab/model.py`](https://github.com/ColtMercer/open-llm-engineering/blob/main/src/open_llm_lab/model.py) | compare with the two trails above |

Read in this order: config → embedding → one block → attention → FFN → final norm → language-model head → loss.

## Attention and position

| Mechanism | Source trail |
|---|---|
| Explicit causal attention | [companion `attention.py`](https://github.com/ColtMercer/open-llm-engineering/blob/main/src/open_llm_lab/attention.py) |
| Production Transformer attention abstraction | [Megatron-Core `attention.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py) |
| FlashAttention kernels and interface | [official FlashAttention repository](https://github.com/Dao-AILab/flash-attention) |
| Llama attention/rotary path | [torchtitan Llama model](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model.py) |

Ask whether the code is expressing the mathematical layer, selecting a kernel, managing a cache, or distributing tensors. Those concerns can make equivalent attention look unrelated.

## Mixture of experts

| Mechanism | Source trail |
|---|---|
| Teaching top-k route/combine | [companion `moe.py`](https://github.com/ColtMercer/open-llm-engineering/blob/main/src/open_llm_lab/moe.py) |
| Router | [Megatron-Core `router.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py) |
| MoE layer orchestration | [Megatron-Core `moe_layer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py) |
| Token dispatch and all-to-all | [Megatron-Core `token_dispatcher.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py) |
| Shared experts | [Megatron-Core `shared_experts.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py) |
| Open research model | [OLMoE repository trail](https://github.com/allenai/OLMoE) |

Trace a token index, not only a vector: router logits → selected expert IDs → permutation/dispatch → expert batch → weighted combine → inverse permutation → residual stream.

## Training systems

| Layer | Official codebase | Entry point |
|---|---|---|
| Small GPT training | [nanoGPT](https://github.com/karpathy/nanoGPT) | [`train.py`](https://github.com/karpathy/nanoGPT/blob/master/train.py) |
| Clean multi-model training | [LitGPT](https://github.com/Lightning-AI/litgpt) | project pretrain/finetune commands and `litgpt` package |
| PyTorch-native distributed LLM training | [torchtitan](https://github.com/pytorch/torchtitan) | [`torchtitan/train.py`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py) and [parallel dimensions](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py) |
| Large-scale Transformer/MoE training | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | [`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py) |
| Reproducible OLMo runs | [OLMo-core](https://github.com/allenai/OLMo-core) | `src/scripts/official/` and model cards |

## Data

| Task | Official codebase |
|---|---|
| OLMo/Dolma processing | [Dolma toolkit](https://github.com/allenai/dolma) |
| large-scale text processing | [DataTrove](https://github.com/huggingface/datatrove) |
| RedPajama processing | [RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data) |
| Common Crawl access | [Common Crawl examples](https://github.com/commoncrawl/cc-pyspark) and [index docs](https://commoncrawl.org/get-started) |
| tokenizer implementation | [SentencePiece](https://github.com/google/sentencepiece) and [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) |

## Post-training

| Task | Official codebase |
|---|---|
| SFT, DPO, reward modeling | [Hugging Face TRL](https://github.com/huggingface/trl) |
| open instruction and RLVR recipes | [Ai2 Open Instruct](https://github.com/allenai/open-instruct) |
| distributed RLHF | [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) |
| large-scale RL | [verl](https://github.com/volcengine/verl) |

Objective names are not enough to reproduce a run. Follow data formatting, chat template, masking, reference-model behavior, rollout generation, reward normalization, and distributed launch configuration.

## Inference

| Question | Official source |
|---|---|
| How are requests scheduled continuously? | [vLLM scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) |
| How are KV blocks managed? | [vLLM KV-cache manager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) |
| How does a serving engine combine kernels and radix/prefix reuse? | [SGLang](https://github.com/sgl-project/sglang) |
| How can quantized inference run broadly? | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| How does GPU-optimized deployment work? | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |

## Evaluation

- [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Ai2 OLMES](https://github.com/allenai/olmes)
- [Stanford HELM](https://github.com/stanford-crfm/helm)

Read the prompt templates, answer extraction, few-shot selection, normalization, and task version before comparing a score.

