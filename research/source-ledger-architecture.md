# Source ledger: transformer architecture and mixture of experts

Last verified: **2026-07-12**

This ledger records the primary evidence behind the transformer and
mixture-of-experts chapters. It is deliberately stricter than a bibliography.
Every model-specific number comes from an original paper, a model-owner
release, or a pinned source file. Third-party implementations are labeled as
implementations, not as evidence of how an unreleased training run worked.

## Evidence policy

- **Paper fact** means the authors reported it in the linked paper. It is not
  an independent replication.
- **Released-code fact** means the behavior is visible in the linked source at
  the pinned revision. It does not prove that identical code trained the model.
- **Model-card fact** means the model owner published it with the checkpoint.
- **Inference** means the conclusion follows from published interfaces or
  equations, but the source does not state the conclusion verbatim.
- Model names, parameter counts, and licenses refer to the named release only.
  They must not be generalized to later models in the same family.

## Transformer foundations

### T01 - The original Transformer

- Primary source: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Type: original paper
- Used for: scaled dot-product attention, multi-head attention, sinusoidal
  positions, residual connections, layer normalization, and the position-wise
  feed-forward network.
- Extraction: Section 3 defines
  `softmax(QK^T / sqrt(d_k))V`; Section 3.2.2 defines multiple learned
  projections (heads); Section 3.5 defines sinusoidal positions.
- Boundary: the paper describes an encoder-decoder translation model. A modern
  decoder-only causal LM is a descendant, not the exact 2017 architecture.

### T02 - Rotary position embedding

- Primary source: Su et al., [RoFormer: Enhanced Transformer with Rotary
  Position Embedding](https://arxiv.org/abs/2104.09864)
- Type: original paper
- Used for: rotating query and key coordinate pairs by position and explaining
  why their inner product depends on relative position.
- Boundary: RoPE is applied to queries and keys; it is not a learned position
  vector simply added to every token embedding.

### T03 - Attention with linear biases

- Primary source: Press et al., [Train Short, Test Long: Attention with Linear
  Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
- Type: original paper
- Used for: ALiBi as a contrasting positional method that adds a head-specific
  distance bias to attention scores.

### T04 - RMSNorm

- Primary source: Zhang and Sennrich, [Root Mean Square Layer
  Normalization](https://arxiv.org/abs/1910.07467)
- Type: original paper and linked author code
- Used for: the RMS-only normalization equation and the distinction from
  LayerNorm's re-centering operation.

### T05 - Gated feed-forward networks

- Primary source: Shazeer, [GLU Variants Improve
  Transformer](https://arxiv.org/abs/2002.05202)
- Type: original paper
- Used for: GLU-family FFNs and the SwiGLU equation.

### T06 - Multi-query and grouped-query attention

- Primary sources:
  - Shazeer, [Fast Transformer Decoding: One Write-Head is All You
    Need](https://arxiv.org/abs/1911.02150)
  - Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models
    from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Type: original papers
- Used for: distinguishing MHA, MQA, and GQA by the number of key/value heads.
- Boundary: fewer KV heads reduce KV-cache storage and bandwidth; they do not
  remove query heads or make attention linear in sequence length.

### T07 - Pre-normalization

- Primary source: Xiong et al., [On Layer Normalization in the Transformer
  Architecture](https://arxiv.org/abs/2002.04745)
- Type: original paper
- Used for: the pre-LN versus post-LN distinction and its training-stability
  motivation.

### T08 - Exact IO-aware attention

- Primary sources:
  - Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)
  - Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691)
  - Official repository: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- Type: original papers and author repository
- Used for: explaining that FlashAttention changes the execution algorithm and
  memory traffic while computing exact attention, rather than defining a new
  learned attention architecture.

### T09 - A readable released decoder block

- Primary source: Meta's released Llama 3 inference implementation at commit
  [`a0940f9`](https://github.com/meta-llama/llama3/tree/a0940f9cf7065d45bb6675660f80d305c041a754)
- Type: model-owner inference code
- Code trails:
  - [RMSNorm, lines 35-46](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L35-L46)
  - [RoPE, lines 49-76](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L49-L76)
  - [attention, lines 90-190](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L90-L190)
  - [SwiGLU FFN, lines 193-219](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L193-L219)
  - [pre-norm block, lines 222-248](https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L222-L248)
- Boundary: this repository is inference code for the released model family,
  not the complete pretraining system or dataset.

### T10 - Current PyTorch-native implementation trail

- Primary source: PyTorch
  [torchtitan](https://github.com/pytorch/torchtitan/tree/51c197c86d7c703da96f666d5a7dbd5432b4afbf)
  at commit `51c197c86d7c703da96f666d5a7dbd5432b4afbf`.
- Type: official framework implementation
- Code trails:
  - [PyTorch scaled dot-product attention call, lines
    342-403](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/common/attention.py#L342-L403)
  - [common SwiGLU FFN, lines
    34-77](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/common/feed_forward.py#L34-L77)
  - [Llama 3 block, lines
    19-50](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/llama3/model.py#L19-L50)
- Boundary: torchtitan evolves quickly, so all code links are commit-pinned.

## Mixture-of-experts foundations

### M01 - Sparsely gated MoE

- Primary source: Shazeer et al., [Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts
  Layer](https://arxiv.org/abs/1701.06538)
- Type: original paper
- Used for: learned sparse gates, expert importance/load objectives, noisy
  gating, and conditional capacity without proportional per-example compute.
- Boundary: the large language models in this paper place MoE layers between
  LSTMs. It predates Transformer MoE designs.

### M02 - GShard

- Primary source: Lepikhin et al., [GShard: Scaling Giant Models with
  Conditional Computation and Automatic
  Sharding](https://arxiv.org/abs/2006.16668)
- Type: original paper
- Extracted facts:
  - encoder-decoder Transformer for multilingual translation;
  - every other FFN is replaced by an MoE layer;
  - group-level top-2 gating;
  - finite per-group expert capacity and residual bypass for overflow;
  - probabilistic dispatch to the second expert;
  - a reported 600B-parameter model trained on 2,048 TPU v3 cores in four days.

### M03 - Switch Transformer

- Primary source: Fedus et al., [Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient
  Sparsity](https://arxiv.org/abs/2101.03961)
- Type: original paper
- Extracted facts:
  - top-1 token-choice routing;
  - expert capacity `(tokens / experts) * capacity_factor`;
  - overflow tokens skip the expert computation and continue through the
    residual path;
  - auxiliary balance loss `alpha * N * sum(f_i * P_i)`;
  - T5-derived encoder-decoder experiments.
- Boundary: "constant computational cost" is an asymptotic design claim, not a
  promise of equal wall-clock latency for every expert count and device layout.

### M04 - Router stability

- Primary source: Zoph et al., [ST-MoE: Designing Stable and Transferable
  Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- Type: original paper
- Used for: router z-loss and MoE stability/fine-tuning guidance.

### M05 - MegaBlocks and dropless execution

- Primary sources:
  - Gale et al., [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
  - Official Databricks repository at commit
    [`952db33`](https://github.com/databricks/megablocks/tree/952db33d6eac334d22c61e47a0d5d41446298784)
- Type: original paper and official implementation
- Used for: dropless MoE execution through block-sparse/grouped computation
  when experts receive unequal token counts.

## Released MoE architectures

### M06 - Mixtral 8x7B

- Primary source: Mistral AI, [Mixtral of
  Experts](https://arxiv.org/abs/2401.04088)
- Type: model-owner paper
- Extracted facts:
  - decoder-only;
  - every Transformer FFN is replaced by 8 SwiGLU experts;
  - top-2 routing independently for every token and layer;
  - 47B reported total and 13B active parameters;
  - 32K training context;
  - paper reports no obvious topic-based expert assignment and stronger
    syntactic than domain routing patterns.
- Official code trail: Mistral's inference library at commit `9eaeb91`:
  [top-k, normalized combine, lines
  16-32](https://github.com/mistralai/mistral-inference/blob/9eaeb91c17450e09021b6065a1d5cc69876507c8/src/mistral_inference/moe.py#L16-L32).
- License boundary: the paper states that the named base and instruct weights
  were released under Apache-2.0. That does not license unrelated later models.

### M07 - DeepSeekMoE

- Primary source: Dai et al., [DeepSeekMoE: Towards Ultimate Expert
  Specialization in Mixture-of-Experts Language
  Models](https://arxiv.org/abs/2401.06066)
- Type: model-owner paper
- Extracted facts:
  - fine-grained segmentation makes each expert smaller and activates more of
    them while holding expert compute roughly fixed;
  - shared experts are always active and intended to capture common knowledge;
  - DeepSeekMoE 16B reports 2 shared experts plus 64 routed experts, with 6
    routed experts active per token;
  - the first Transformer layer remains dense in that release.
- Boundary: the paper's specialization results are empirical for its tested
  checkpoints and do not establish a universal semantic role for expert IDs.

### M08 - DeepSeek-V3

- Primary source: DeepSeek-AI, [DeepSeek-V3 Technical
  Report](https://arxiv.org/abs/2412.19437)
- Type: model-owner technical report
- Extracted facts:
  - 671B total and 37B activated parameters;
  - the first three FFNs are dense;
  - each later MoE layer has 1 shared expert and 256 routed experts, 8 selected;
  - sigmoid affinities are normalized over selected experts;
  - a per-expert bias changes selection but not combine weights;
  - the main batch-level balancing method is auxiliary-loss-free;
  - a very small complementary sequence-wise auxiliary balance loss remains;
  - node-limited routing restricts a token to at most 4 nodes;
  - the report states no token dropping in training or inference;
  - training used 64-way expert parallelism across 8 nodes.
- Official code trails at commit `9b4e978`:
  - [671B configuration](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/configs/config_671B.json)
  - [sigmoid/bias/group-limited top-k router, lines
    535-598](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L535-L598)
  - [routed plus shared expert combine, lines
    636-693](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L636-L693)
- Critical terminology note: "auxiliary-loss-free" does **not** mean the total
  training objective contains no routing auxiliary term. It names the main
  batch-level load-balancing strategy; the complementary sequence-wise loss is
  explicitly reported.

### M09 - Qwen MoE releases

- Primary sources:
  - Qwen Team, [Qwen1.5-MoE release](https://qwenlm.github.io/blog/qwen-moe/)
  - Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
  - Qwen Team, [Global-batch load balance](https://qwenlm.github.io/blog/global-load-balance/)
- Type: model-owner release notes and technical report
- Extracted facts:
  - Qwen1.5-MoE-A2.7B reports 60 routed experts with 4 active plus the
    equivalent width of 4 always-active shared experts;
  - Qwen3-30B-A3B and Qwen3-235B-A22B use 128 experts with 8 active;
  - Qwen3 explicitly excludes shared experts and uses global-batch load
    balancing.
- Model-card trails:
  - [Qwen1.5-MoE configuration at revision
    `1a758c5`](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/config.json)
  - [Qwen3-30B-A3B configuration at revision
    `ad44e77`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)
- Boundary: the Qwen3 "thinking budget" is a post-training/inference control.
  The report does not identify it as a direct expert-routing control.

### M10 - DBRX

- Primary source: Databricks, [Introducing
  DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- Type: model-owner release
- Extracted facts:
  - decoder-only, 132B total and 36B active parameters;
  - 16 experts with top-4 routing;
  - RoPE, GLU, and GQA;
  - 40 layers and a reported 32K context;
  - pretrained on 12T tokens according to the release.
- Implementation trail: Hugging Face's independent official-library
  implementation at commit `63f32a8`, [DBRX router and expert combine, lines
  264-369](https://github.com/huggingface/transformers/blob/63f32a8782cb70da3365acab16f2b67947737985/src/transformers/models/dbrx/modeling_dbrx.py#L264-L369).
- Boundary: Hugging Face code is useful for inspection, but it is not evidence
  that Databricks used that exact implementation for pretraining.

### M11 - OLMoE-1B-7B

- Primary sources:
  - Muennighoff et al., [OLMoE: Open Mixture-of-Experts Language
    Models](https://arxiv.org/abs/2409.02060)
  - Official artifact hub: [allenai/OLMoE](https://github.com/allenai/OLMoE/tree/357454f4f647385839c0ff6b99a688dc7cd9c13f)
- Type: original paper and model-owner artifact repository
- Extracted facts:
  - 6.9B total and 1.3B active parameters, rounded in the model name;
  - 64 experts with top-8 token-choice routing;
  - dropless execution, no shared expert;
  - load-balancing loss weight 0.01 and router z-loss weight 0.001 during
    pretraining;
  - 5.1T pretraining tokens;
  - paper reports early router saturation plus domain and vocabulary
    specialization for this model;
  - the same paper finds little domain specialization in Mixtral, demonstrating
    that expert specialization is architecture- and training-run-dependent.
- Training code trail: [OLMo MoE block at commit `04a2da5`, lines
  674-740](https://github.com/allenai/OLMo/blob/04a2da53db172bd9a0450705592ed50888bdcaa7/olmo/model.py#L674-L740).
- Boundary: OLMoE's controlled experiments sometimes disagree with another
  model's design choice, such as shared experts. Report both results; do not
  turn either into a universal law.

## Expert-parallel implementation evidence

### M12 - Dispatch and combine in PyTorch

- Primary source: PyTorch torchtitan at commit `51c197c8`.
- Type: official framework source
- Code trails:
  - [token-choice top-k router, lines
    173-323](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/common/moe.py#L173-L323)
  - [MoE dispatch/expert/combine lifecycle, lines
    112-152](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/common/moe.py#L112-L152)
  - [all-to-all dispatcher, lines
    239-382](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/common/token_dispatcher.py#L239-L382)
  - [dispatch path, lines
    385-480](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/common/token_dispatcher.py#L385-L480)
  - [reverse combine path, lines
    572-665](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/common/token_dispatcher.py#L572-L665)
- Boundary: this is a current, readable implementation of the mechanism. Model
  owners may use different fused kernels and scheduling.

## Claim audit: prompting versus routing

| Claim | Status | Evidence and qualification |
|---|---|---|
| Routing occurs per token and per MoE layer. | Published fact | Mixtral Sections 2.1 and 5; released Mistral code M06. |
| The router consumes the token's current hidden state. | Published/code fact | Mixtral equation and Mistral `self.gate(inputs)`; DeepSeek-V3 `Gate.forward`. |
| A normal text prompt can name and force `expert 7`. | Unsupported | Released text-generation interfaces accept tokens, not expert IDs. No named-expert control is described for the compared releases. |
| Changing a prompt may change routing. | **Inference** | A prompt changes token IDs and contextual hidden states, which are router inputs. The effect is indirect and model/layer/token dependent. |
| A domain word reliably selects a domain expert. | Unsupported as a general rule | Mixtral reports little topic specialization; OLMoE reports stronger domain and vocabulary specialization. |
| A provider's reasoning-effort or thinking-mode setting selects more MoE experts. | Unsupported | Such controls may change generation behavior or token budget. The cited model reports do not define them as top-k overrides. |
| A model owner can force or mask expert routes by changing code. | Released-code fact | The top-k indices and router masks are explicit tensors in M06, M08, and M12. This is a model/runtime modification, not prompt engineering. |

## Known evidence gaps

- Exact proprietary training corpora and training stacks are not available for
  every named model. Architecture transparency must not be mistaken for full
  training reproducibility.
- Parameter counting conventions vary. Embeddings, routers, shared experts,
  and output heads may be included differently. Use each release's reported
  total/active pair and avoid reverse-engineering false precision.
- Router visualizations describe observed checkpoints and datasets. Expert IDs
  have no cross-layer or cross-model universal meaning.
- Wall-clock speed depends on batch shape, precision, memory bandwidth,
  kernels, expert placement, and network topology. Active-parameter count is
  not a latency benchmark.
