# Source Ledger: Training, Post-Training, Inference, Prompting, and Agents

This ledger supports chapters 06–10. It contains original papers, official repositories, official standards, and first-party product documentation only.

**Verification date:** 2026-07-12.

## Reading rules

- **Paper:** original research report. A result applies to the reported experimental setup unless independently replicated.
- **Official repository:** implementation maintained by the named project or authors. Default-branch links move; pin a commit for reproducibility.
- **Official documentation:** current interface guidance from the provider or project. Treat it as versioned and potentially changing.
- **Standard/guidance:** first-party risk-management or security guidance, not an empirical performance guarantee.

“Extraction” below means the narrow claim used in this book. It is paraphrased; no long source text is reproduced.

## Pretraining, optimization, and scaling

| ID | Primary source | Type | Extracted claim used | Chapters |
|---|---|---|---|---|
| TR-01 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | paper | Decoder-only autoregressive pretraining predicts next tokens and can support in-context task behavior at scale. | 06.01 |
| TR-02 | [nanoGPT](https://github.com/karpathy/nanoGPT) | official repository | Compact GPT model, shifted cross-entropy, training loop, DDP, schedule, and sampling reference. Repository now describes itself as old/deprecated, so use it educationally. | 06.01, 06.02, 08.01 |
| TR-03 | [LitGPT](https://github.com/Lightning-AI/litgpt) | official repository | Readable pretraining, fine-tuning, evaluation, and inference recipes across open model families. | 06.01, 06.02, 07.01 |
| TR-04 | [TorchTitan](https://github.com/pytorch/torchtitan) | official repository | PyTorch-native training with composable FSDP, tensor, pipeline, and context parallelism plus distributed checkpoints. | 06.01–06.04 |
| TR-05 | [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) | paper | OLMo releases model, training, data, and evaluation artifacts to support open study of training trajectories. | 06.01, 06.02, 06.04 |
| TR-06 | [OLMo-core](https://github.com/allenai/OLMo-core) | official repository | Current OLMo training building blocks and official released-model scripts are public. | 06.01, 06.02, 06.04 |
| TR-07 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | paper | Loss followed approximate power-law trends with model size, data, and compute over the paper's studied range. | 06.02 |
| TR-08 | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | paper | The paper fit a different compute allocation and reported that many studied large models were undertrained relative to its fitted optimum. | 06.02 |
| TR-09 | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | paper | AdamW decouples weight decay from the adaptive loss-gradient update. | 06.02 |
| TR-10 | [ZeRO](https://arxiv.org/abs/1910.02054) | paper | Partitioning optimizer state, gradients, and parameters reduces replicated memory in data-parallel training. | 06.03 |
| TR-11 | [Megatron-LM paper](https://arxiv.org/abs/1909.08053) | paper | Intra-layer tensor parallelism enables large transformer training across accelerators. | 06.03 |
| TR-12 | [Megatron-LM and Megatron Core](https://github.com/NVIDIA/Megatron-LM) | official repository | Current code supports tensor, pipeline, data, context, and expert parallelism. | 06.03 |
| TR-13 | [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html) | official documentation | Activation checkpointing recomputes forward segments during backward and needs careful RNG handling. | 06.03 |
| TR-14 | [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) | official documentation | Distributed checkpointing saves/loads sharded state and can reshard across topology changes. | 06.03, 06.04 |
| TR-15 | [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) | official repository | Versionable tasks, prompts, model backends, sampling, and logged examples support reproducible evaluation. | 06.04, 09.04 |
| TR-16 | [OpenAI Evals](https://github.com/openai/evals) | official repository | Open framework and registry for model/system evaluations. | 06.04, 09.04, 10.03 |
| TR-17 | [PALOMA](https://arxiv.org/abs/2312.10523) | paper | Language-model fit should be measured across documented domains with contamination awareness. | 06.04 |

## Supervised and preference post-training

| ID | Primary source | Type | Extracted claim used | Chapters |
|---|---|---|---|---|
| PT-01 | [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) | paper | FLAN's multi-task instruction tuning improved zero-shot results on held-out task types in the reported setup. | 07.01 |
| PT-02 | [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) | paper | Primary description of SFT, pairwise reward modeling, and PPO-based RLHF for InstructGPT. | 07.01–07.03 |
| PT-03 | [TRL](https://github.com/huggingface/trl) | official repository | Current open trainers cover SFT, reward modeling, DPO, GRPO, and related methods. | 07.01–07.04 |
| PT-04 | [LoRA](https://arxiv.org/abs/2106.09685) | paper | Low-rank trainable updates can adapt a frozen base with far fewer trainable parameters in the studied tasks. | 07.01 |
| PT-05 | [QLoRA](https://arxiv.org/abs/2305.14314) | paper | Fine-tuning adapters through a quantized frozen base reduces training-memory requirements in the reported method. | 07.01 |
| PT-06 | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | paper | A KL-regularized preference objective can be optimized with a classification-style loss relative to a reference policy. | 07.02 |
| PT-07 | [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | paper | PPO alternates sampling with clipped surrogate-objective updates. | 07.02, 07.03 |
| PT-08 | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) | paper | Sampling candidate solutions and selecting with a learned verifier improved reported GSM8K performance. | 07.03 |
| PT-09 | [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | paper | Process supervision outperformed outcome supervision on the paper's studied MATH subset; PRM800K was released. | 07.03, 07.04 |
| PT-10 | [PRM800K](https://github.com/openai/prm800k) | official repository | Step-level human feedback artifacts for process-reward-model research are public. | 07.03, 07.04 |
| PT-11 | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | paper | Introduced the reported GRPO formulation for mathematical reasoning training. | 07.03 |
| PT-12 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | paper | Describes RL-only experiments, cold-start data, multi-stage reasoning training, and distillation. | 07.03, 07.04 |
| PT-13 | [DeepSeek-R1 repository](https://github.com/deepseek-ai/DeepSeek-R1) | official repository | Hosts the official paper, model usage notes, licenses, and released checkpoints; not a full frontier-training reproduction stack. | 07.03, 07.04 |
| PT-14 | [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | paper | Worked reasoning exemplars improved selected tasks for sufficiently large models in the paper. | 07.04, 09.04 |
| PT-15 | [Self-Consistency](https://arxiv.org/abs/2203.11171) | paper | Sampling multiple reasoning paths and aggregating answers improved selected benchmarks in the paper. | 07.04, 09.04 |
| PT-16 | [Measuring Faithfulness in Chain-of-Thought Reasoning](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning) | first-party research | Generated rationales were not consistently faithful to causal influences in the studied interventions. | 07.04, 09.04 |
| PT-17 | [Reasoning Models Don't Always Say What They Think](https://www.anthropic.com/research/reasoning-models-dont-say-think) | first-party research | Follow-up experiments found low disclosure of inserted hints in visible reasoning for studied reasoning models. | 07.04, 09.04 |

## Decoding and inference systems

| ID | Primary source | Type | Extracted claim used | Chapters |
|---|---|---|---|---|
| IN-01 | [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | paper | Introduced nucleus sampling and evaluated it against decoding baselines for open-ended generation. | 08.01 |
| IN-02 | [Transformers](https://github.com/huggingface/transformers) | official repository | Production generation utilities, logits processors, cache classes, and many architecture implementations. | 08.01, 08.02 |
| IN-03 | [GQA](https://arxiv.org/abs/2305.13245) | paper | Grouped-query attention uses fewer KV heads than query heads and reported a quality/speed compromise between MHA and MQA. | 08.02 |
| IN-04 | [FlashAttention](https://arxiv.org/abs/2205.14135) | paper | Exact tiled attention reduces memory traffic between GPU memory levels. | 08.02 |
| IN-05 | [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) | official repository | Official kernels and tests for FlashAttention. | 08.02 |
| IN-06 | [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) | paper | Block-managed KV cache reduces fragmentation and supports flexible sharing in vLLM. | 08.02, 08.03 |
| IN-07 | [vLLM](https://github.com/vllm-project/vllm) | official repository | Current high-throughput serving with continuous batching, prefix caching, quantization, and distributed inference. | 08.01–08.04 |
| IN-08 | [SGLang paper](https://arxiv.org/abs/2312.07104) | paper | Introduces RadixAttention and structured-language-model-program execution optimizations. | 08.02, 08.03 |
| IN-09 | [SGLang](https://github.com/sgl-project/sglang) | official repository | Current scheduler, radix cache, structured decoding, quantization, and speculative serving code. | 08.02–08.04 |
| IN-10 | [llama.cpp](https://github.com/ggml-org/llama.cpp) | official repository | Local C/C++ inference, GGUF, low-bit formats, heterogeneous hardware, and an HTTP server. | 08.03, 08.04 |
| IN-11 | [GPTQ](https://arxiv.org/abs/2210.17323) | paper | Layer-wise post-training weight quantization using approximate second-order information. | 08.04 |
| IN-12 | [AWQ](https://arxiv.org/abs/2306.00978) | paper | Activation-aware channel scaling protects salient weights in the reported low-bit method. | 08.04 |
| IN-13 | [SmoothQuant](https://arxiv.org/abs/2211.10438) | paper | Equivalent scaling moves quantization difficulty from activations toward weights for W8A8. | 08.04 |
| IN-14 | [TorchAO](https://github.com/pytorch/ao) | official repository | PyTorch-native PTQ, QAT, low-precision training, and serving integrations. | 06.02, 08.04 |
| IN-15 | [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192) | paper | An acceptance/correction algorithm can preserve a target model's sampling distribution while verifying draft tokens in parallel. | 08.04 |

## Prompting, structured output, tools, and retrieval

| ID | Primary source | Type | Extracted claim used | Chapters |
|---|---|---|---|---|
| PR-01 | [OpenAI Prompt Engineering](https://developers.openai.com/api/docs/guides/prompt-engineering) | official documentation | Current OpenAI-specific prompting interface and recommendations. | 09.01 |
| PR-02 | [Anthropic Prompt Engineering](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) | official documentation | Current Claude-specific prompt-development workflow begins with success criteria and evaluation. | 09.01, 09.04 |
| PR-03 | [Gemini Prompt Design Strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) | official documentation | Current Gemini-specific structure and parameter guidance; the page explicitly frames advice as iterative. | 09.01, 09.04 |
| PR-04 | [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208) | paper | Explores training models to prioritize higher-privilege instructions over conflicting lower-level content. | 09.01, 10.03 |
| PR-05 | [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) | official documentation | Supported schema-constrained output can enforce declared structure, not semantic truth. | 09.02 |
| PR-06 | [OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling) | official documentation | The model emits structured function requests and application code handles the execution loop. | 09.02 |
| PR-07 | [Anthropic Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) | official documentation | JSON output and strict tool-use features validate supported schemas. | 09.02 |
| PR-08 | [Anthropic Tool Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works) | official documentation | Client tools execute in the application; server tools execute at the provider; the agent loop is explicit. | 09.02, 10.01 |
| PR-09 | [Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling) | official documentation | Function declarations use structured schemas; the application executes client functions and returns results. | 09.02 |
| PR-10 | [Toolformer](https://arxiv.org/abs/2302.04761) | paper | The paper trained a model to decide which API to call, when, and with what arguments in its studied setup. | 09.02, 10.01 |
| PR-11 | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | paper | Combines parametric generation with retrieved non-parametric memory for knowledge-intensive tasks. | 09.03 |
| PR-12 | [FAISS](https://github.com/facebookresearch/faiss) | official repository | Official similarity-search and clustering implementation with CPU and GPU indexes. | 09.03 |
| PR-13 | [ColBERT](https://arxiv.org/abs/2004.12832) | paper | Late interaction retains token-level representations for passage ranking. | 09.03 |
| PR-14 | [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) | official author repository | Primary code for ColBERT and ColBERTv2 experiments. | 09.03 |
| PR-15 | [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/) | paper | Relevant-information position affected long-context performance in the evaluated models and tasks. | 09.03 |
| PR-16 | [Lost in the Middle repository](https://github.com/nelson-liu/lost-in-the-middle) | official author repository | Primary experimental code and data construction. | 09.03 |
| PR-17 | [OpenAI Reasoning Best Practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices) | official documentation | Current model-specific reasoning guidance; not generalized to unrelated models. | 09.04 |
| PR-18 | [Anthropic Interactive Prompting Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) | official repository | Executable exercises and examples maintained by Anthropic. | 09.01 |

## Agents, memory, multi-agent systems, and safety

| ID | Primary source | Type | Extracted claim used | Chapters |
|---|---|---|---|---|
| AG-01 | [ReAct](https://arxiv.org/abs/2210.03629) | paper | Interleaving model-generated reasoning and environment actions improved selected tasks over the paper's baselines. | 09.04, 10.01 |
| AG-02 | [ReAct repository](https://github.com/ysymyth/ReAct) | official author repository | Original notebooks, prompts, and environment wrappers. | 09.04, 10.01 |
| AG-03 | [tau-bench](https://arxiv.org/abs/2406.12045) | paper | Evaluates tool-agent-user interaction by final database state and repeated-trial reliability. | 10.01, 10.03 |
| AG-04 | [tau-bench repository](https://github.com/sierra-research/tau-bench) | official author repository | Primary benchmark environments and graders. | 10.01, 10.03 |
| AG-05 | [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | first-party engineering report | Distinguishes workflows from agents and recommends simple, composable patterns plus evaluation. | 10.01 |
| AG-06 | [OpenAI Practical Guide to Building Agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) | first-party guide | Agent foundations, orchestration, and guardrail guidance. | 10.01 |
| AG-07 | [Generative Agents](https://arxiv.org/abs/2304.03442) | paper | Reports a simulated-agent architecture combining an experience stream, retrieval, reflection, and planning. | 10.02 |
| AG-08 | [Generative Agents repository](https://github.com/joonspk-research/generative_agents) | official author repository | Primary simulation and memory/planning code. | 10.02 |
| AG-09 | [MemGPT](https://arxiv.org/abs/2310.08560) | paper | Explores OS-inspired virtual context management across memory tiers. | 10.02 |
| AG-10 | [Letta](https://github.com/letta-ai/letta) | official repository | Current open successor ecosystem to MemGPT for stateful agents. | 10.02 |
| AG-11 | [CAMEL](https://arxiv.org/abs/2303.17760) | paper | Studies role-playing communication among language-model agents. | 10.02 |
| AG-12 | [CAMEL repository](https://github.com/camel-ai/camel) | official repository | Current open framework from the CAMEL project. | 10.02 |
| AG-13 | [AutoGen](https://arxiv.org/abs/2308.08155) | paper | Describes programmable multi-agent conversation patterns. | 10.02 |
| AG-14 | [AutoGen repository](https://github.com/microsoft/autogen) | official repository | Historical/current implementation; the official README now states maintenance mode and points new users to its successor. | 10.02 |
| AG-15 | [How We Built Our Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) | first-party engineering report | Reports an orchestrator-worker system, gains on an internal breadth-first research eval, higher token use, and coordination costs. | 10.02 |
| AG-16 | [NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1) | official standard/guidance | Cross-sector generative-AI risk-management profile. | 10.03 |
| AG-17 | [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) | official security guidance | Identifies application risks including prompt injection, sensitive disclosure, supply chain, and excessive agency. | 10.03 |
| AG-18 | [OWASP LLM01: Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | official security guidance | Defines direct and indirect prompt-injection risk and defense-in-depth mitigations. | 10.03 |
| AG-19 | [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | first-party engineering report | Describes multi-turn environment evaluation, outcome graders, transcripts, and failure analysis. | 10.03 |
| AG-20 | [How We Contain Claude Across Products](https://www.anthropic.com/engineering/how-we-contain-claude) | first-party engineering report | Current first-party containment patterns aimed at limiting agent blast radius. | 10.03 |

## Source-code reading map

Use these entry points rather than browsing an entire repository at random:

| Topic | Start here | What to trace |
|---|---|---|
| next-token loss | [nanoGPT `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py) | labels shifted by one, causal attention, cross-entropy |
| compact training loop | [nanoGPT `train.py`](https://github.com/karpathy/nanoGPT/blob/master/train.py) | batch, accumulation, optimizer, schedule, evaluation, checkpoint |
| scalable trainer | [TorchTitan `train.py`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py) | initialization, data, model, parallel mesh, step, metrics |
| composable parallelism | [TorchTitan Llama parallelization](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize.py) | FSDP, TP, CP, activation checkpointing |
| industrial parallelism | [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) | tensor, pipeline, context, expert, distributed checkpoint |
| released training recipe | [OLMo-core official scripts](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official) | declared architecture, data, optimizer, schedule |
| SFT | [TRL `sft_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py) | templates, masks, dataset preprocessing, loss |
| DPO | [TRL `dpo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) | policy/reference log probabilities and pairwise loss |
| online reasoning RL | [TRL `grpo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) | rollout generation, rewards, group-relative update |
| decoding | [Transformers `generation/utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py) | generation loop, cache, stopping |
| logits filters | [Transformers `logits_process.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py) | temperature-adjacent filters, penalties, constraints |
| KV cache | [Transformers `cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py) | dynamic/static cache update and position semantics |
| high-throughput serving | [vLLM repository](https://github.com/vllm-project/vllm) | scheduler, cache blocks, attention, sampling, API |
| radix prefix cache | [SGLang memory cache](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache) | radix keys, eviction, KV pools |
| quantization | [TorchAO quantization](https://github.com/pytorch/ao/tree/main/torchao/quantization) | configs, tensor transforms, QAT, kernels |
| local GGUF quantization | [llama.cpp quantize guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) | conversion, quant type, importance matrix, validation |
| benchmark tasks | [lm-eval task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) | prompt rendering, dataset revision, metrics |
| agent trajectories | [ReAct repository](https://github.com/ysymyth/ReAct) | prompts, actions, environment observations |
| stateful agent eval | [tau-bench](https://github.com/sierra-research/tau-bench) | policy, tools, simulator, final-state grader |

## Claims deliberately excluded

The chapters do **not** claim:

- that a final model is reproducible merely because its weights and architecture are public;
- that one scaling-law exponent or token/parameter ratio applies to every future run;
- that DPO universally beats RLHF, or process supervision universally beats outcome supervision;
- that lower perplexity guarantees better factuality, safety, or instruction following;
- that a visible rationale is a faithful transcript of hidden computation;
- that prompt wording can directly select a named MoE expert;
- that schema-valid output is semantically correct or authorized;
- that long context guarantees robust use of middle-position evidence;
- that multi-agent systems inherently outperform a single agent at equal compute.
