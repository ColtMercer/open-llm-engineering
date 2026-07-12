# Glossary

Terms are defined for this book's context. Specific codebases may use narrower or conflicting names.

**Activated parameters** — Parameters used for one token's forward computation. In an MoE this can be much smaller than total parameters.

**Activation** — An intermediate tensor produced by a model operation, distinct from a learned parameter.

**Agent** — A system that repeatedly asks a model to choose or describe actions, executes allowed actions, observes results, updates state, and stops under explicit rules.

**All-to-all** — Collective communication in which each rank sends distinct data to every other rank; commonly used to dispatch tokens across expert-parallel devices.

**Attention head** — One projected query/key/value subspace within multi-head attention.

**Autoregressive** — Factorizing a sequence probability into next-element probabilities conditioned on earlier elements.

**Backpropagation** — Reverse-mode differentiation through a computation graph to obtain gradients of a scalar loss.

**Batch** — A collection of training examples or token sequences processed together. Always distinguish microbatch and global batch.

**BPE** — Byte-pair encoding; a family of tokenization methods that learns ordered merges of adjacent symbols.

**Capacity factor** — A multiplier determining allocated expert slots relative to expected average MoE load.

**Causal mask** — Attention constraint preventing a position from using future tokens during autoregressive training.

**Checkpoint** — Serialized training/model state. A weights-only checkpoint may be insufficient for exact resume.

**Context window** — Maximum token positions a configuration/runtime accepts. Useful retrieval and reasoning across that length must be evaluated separately.

**Cross-entropy** — Loss measuring negative log-probability placed on observed class labels, here next tokens.

**Data parallelism** — Replicating model computation while splitting batches, then coordinating gradients or parameter updates.

**DPO** — Direct Preference Optimization; an objective that uses preference pairs and a reference policy without the classic explicit reward-model-plus-PPO loop.

**Embedding** — Learned mapping from a discrete ID to a continuous vector; also used more broadly for vector representations.

**Expert** — In sparse Transformer MoE, usually a feed-forward subnetwork. The name does not guarantee a human-readable skill boundary.

**Expert parallelism** — Placing experts across ranks and routing token activations to their owners.

**Fine-tuning** — Continuing parameter updates on a new objective or dataset. Includes many methods; name the objective and trainable parameters.

**FLOP** — Floating-point operation. Counting conventions differ, so comparisons should state the convention.

**Gradient accumulation** — Summing/averaging gradients from multiple microbatches before an optimizer step.

**Grouped-query attention (GQA)** — Multiple query heads share fewer key/value heads, reducing KV-cache and projection cost.

**Hallucination** — A fluent output unsupported by evidence or inconsistent with reality; definitions and measurement protocols vary.

**Instruction tuning** — Supervised training on instruction/response or conversation-formatted examples.

**KV cache** — Stored attention keys and values from prior positions reused during autoregressive decoding.

**Logit** — Unnormalized score before softmax or another selection transform.

**Loss** — Scalar objective optimized during training. Lower loss on one distribution does not imply every desired behavior improves.

**LoRA** — Low-rank adaptation, which learns small low-rank updates around selected frozen weight matrices.

**MoE** — Mixture of experts; here, conditional computation that routes each token through a subset of feed-forward experts.

**Next-token prediction** — Objective of predicting each subsequent token from its preceding context.

**Open weights** — Downloadable learned parameters under stated terms. This does not imply open data or reproducible training.

**Optimizer** — Algorithm mapping gradients and state to parameter updates, such as AdamW.

**Packing** — Combining tokenized records into fixed-length sequences to reduce padding; requires boundary and loss-mask policy.

**Parameter** — Learned tensor element updated during training.

**Perplexity** — Exponential of average negative log-likelihood under a stated tokenization and evaluation convention.

**Pipeline parallelism** — Splitting sequential model stages across ranks and scheduling microbatches through them.

**Post-training** — Training stages after base pretraining that shape interaction, preferences, reasoning, safety, or domains.

**Prefill** — Inference phase computing activations/cache for the input context, often parallel across positions.

**Quantization** — Representing weights or activations with lower precision or discrete codes to reduce resource cost, with possible quality and kernel trade-offs.

**RAG** — Retrieval-augmented generation; an application pattern that supplies retrieved evidence in model context.

**Residual stream** — The main `[B,T,C]` representation carried through a Transformer via residual additions.

**RLHF** — Reinforcement learning from human feedback; a family of pipelines, not one objective.

**Router** — Learned or rule-based function assigning token representations to experts.

**SFT** — Supervised fine-tuning.

**Softmax** — Converts logits along an axis into positive values summing to one.

**Speculative decoding** — Drafting candidate tokens with a cheaper process and verifying them with the target model while preserving a specified target distribution under the algorithm's assumptions.

**Tensor parallelism** — Splitting individual tensor operations or parameters across ranks.

**Token** — Integer vocabulary item created by a tokenizer; not necessarily a word or character.

**Top-k routing** — Selecting the highest-scoring \(k\) experts for each token.

**Training token** — Token consumed by the training objective. Distinguish raw corpus size, retained encoded size, and sampled consumption.

**Transformer** — Architecture built from attention, position handling, feed-forward transformations, normalization, and residual paths.

**Vocabulary** — Fixed mapping between token IDs and byte/text pieces plus registered control tokens.

**Weight tying** — Reusing one parameter matrix for input embeddings and output vocabulary projection.

