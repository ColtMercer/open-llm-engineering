# Learning paths

The book is ordered end to end, but different readers should take different first passes.

## The builder from zero

**Goal:** understand each transformation before touching distributed systems.

```mermaid
flowchart LR
    F[Foundations] --> T[Tokens]
    T --> A[Attention]
    A --> G[Tiny GPT lab]
    G --> D[Data]
    D --> P[Pretraining]
    P --> I[Inference]
```

Read chapters 1–4, run labs 1–3, then read chapters 3, 6, and 8. Treat MoE as an extension of the feed-forward sublayer after dense Transformers feel concrete.

**Exit test:** Given shapes `[B,T,C]`, `[C,3C]`, and `n_heads`, explain how tokens become Q/K/V, why the mask is triangular, and why logits have shape `[B,T,V]`.

## The code-first engineer

**Goal:** connect ordinary software constructs to model mechanics.

Start with [lab setup](labs/setup.md), then execute each lab. Use the [source-code map](reference/code-map.md) to compare the compact implementation with nanoGPT, LitGPT, Hugging Face Transformers, torchtitan, Megatron-Core, and vLLM.

**Exit test:** Trace a generated token from input bytes through token IDs, embedding lookup, attention, the residual stream, vocabulary projection, sampling, and cache update.

## The ML practitioner

**Goal:** develop system-level judgment about data, scaling, sparse models, post-training, and serving.

Read chapters 3–8 in order. Keep the [equation sheet](reference/equations.md) open. Finish with the end-to-end design and reproducibility blueprint.

**Exit test:** Design an ablation that distinguishes better routing from more total parameters, and specify data, compute, evaluation, and serving controls.

## The technical leader

**Goal:** evaluate claims, budgets, openness, and risks without implementing every layer.

Read:

1. [The complete pipeline](11-end-to-end/01-complete-pipeline.md)
2. [Open dataset atlas](03-data/02-open-dataset-atlas.md)
3. [Why sparse models](05-moe/01-why-sparse-models.md)
4. [Distributed training](06-training/03-distributed-training.md)
5. [Production safety](10-agents/03-production-safety.md)
6. [Open projects](reference/open-projects.md)

**Exit test:** For a model release, identify what is actually open, what remains irreproducible, what the license permits, and what it costs to train and serve.

## The research track

**Goal:** reproduce mechanisms and challenge claims.

Read the technical layer of each chapter, follow the original papers in the [paper trail](reference/papers.md), inspect the linked source at a pinned release or commit, then reproduce a small claim with controlled seeds and logged configuration.

**Exit test:** Turn a paper claim into a falsifiable experiment with an explicit baseline, independent variable, evaluation set, uncertainty, and failure interpretation.

## Depth markers

Chapters use these informal markers:

- **Foundation** — no ML prerequisite.
- **Builder** — assumes Python and basic algebra.
- **Engineer** — uses tensor shapes, memory costs, and implementation contracts.
- **Research** — expects papers, ablations, and distributed-system details.

Skipping a research subsection should never make the next foundation subsection unreadable.

