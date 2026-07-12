# Design a model from constraints

**Level:** Engineer · **Time:** 50 minutes

Architecture should follow an explicit objective and deployment envelope. “Use the newest block” is not a design process.

## Begin with a model charter

```text
Target behavior:       code completion for a private, permitted corpus
Languages/domains:     Python, TypeScript, internal APIs
Context requirement:   16k useful tokens
Training budget:       fixed accelerator-hours and storage
Serving target:        single-node, p95 latency constraint
Adaptation plan:       continued pretraining + instruction tuning
Evaluation:            held-out completion, repo-level tasks, security tests
License constraints:   organization-approved sources and dependencies
```

Every later choice should point back to this charter.

## Dense baseline before sparse ambition

A decoder-only dense model gives the cleanest baseline. Choose:

- vocabulary \(V\);
- layers \(L\);
- residual width \(C\);
- attention heads and key/value heads;
- FFN hidden width;
- position method and maximum trained context;
- normalization, activation, bias, and weight tying;
- precision and initialization.

Approximate parameter accounting is useful for design, but verify with actual code. A dense block is dominated by attention projections and the FFN. For standard multi-head attention with four \(C\times C\) projections and a gated FFN with three \(C\times F\) matrices:

\[
P_{block}\approx4C^2+3CF
\]

This excludes norms, biases, embeddings, and architecture-specific variations.

## When MoE is justified

Sparse MoE increases total parameter capacity while activating only a subset of expert FFNs per token. It also adds:

- router training and load-balance risk;
- expert-parallel communication;
- capacity/overflow policy;
- more complex checkpointing and serving;
- larger total weight memory even when compute per token is controlled.

Use it when the scale and system budget can exploit sparse capacity—not because “experts” sound modular. The [MoE section](../05-moe/01-why-sparse-models.md) separates total, active, and resident parameters.

## Context length is an end-to-end commitment

Long context affects:

- position method and training distribution;
- attention kernel and memory;
- data packing and document boundaries;
- evaluation at relevant lengths and tasks;
- serving cache capacity and admission policy;
- product truncation and retrieval strategy.

Increasing a config limit without appropriate training and evaluation creates an accepted length, not necessarily a useful length.

## Design matrix

| Decision | Benefit sought | Cost or risk | Required experiment |
|---|---|---|---|
| Larger \(V\) | shorter sequences | larger embeddings/head | multilingual/domain fertility + proxy training |
| Grouped-query attention | smaller KV cache | possible quality trade-off | long-context and serving comparison |
| Larger FFN | more transform capacity | parameters and FLOPs | iso-compute ablation |
| MoE top-2 | sparse capacity | routing and communication | dense-matched + active-FLOP-matched baseline |
| Longer context | more evidence | training/serving cost | length-controlled retrieval and recall eval |
| Weight tying | fewer parameters | coupling constraint | validation and downstream comparison |

## Tiny worked design

Suppose the learning goal is a laptop-readable model, not competitive capability:

```yaml
vocab_size: 256          # raw byte IDs
max_seq_len: 128
n_layers: 4
d_model: 128
n_heads: 4
d_hidden: 384
tie_embeddings: true
```

The configuration is deliberately small, CPU-runnable, and easy to inspect. It is not a miniature proof of scaling behavior. The companion `TinyGPT` implements this family.

## Ablation discipline

An architecture claim needs a matched comparison:

- same training tokens and data order where possible;
- same tokenizer and evaluation harness;
- controlled parameter or compute budget;
- multiple seeds when variance is material;
- training curves, not only final score;
- serving measurements if efficiency is claimed;
- negative results and instability records.

“Model B is better” is not an architecture conclusion when B also saw different data and more compute.

## Design review questions

1. Which behavior or system constraint motivates every non-default feature?
2. Which numbers are derived estimates versus measured?
3. Are total, active, and embedding parameters reported separately?
4. Is useful context measured, not only accepted context?
5. Can the training and serving stack implement the feature efficiently?
6. What is the simplest baseline that could falsify the idea?

