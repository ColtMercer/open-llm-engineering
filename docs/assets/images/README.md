# Visual asset provenance

The three PNG plates were generated specifically for Open LLM Engineering on 2026-07-12 with OpenAI's image-generation tool, then included without compositing or textual overlays. They are explanatory artwork; adjacent Mermaid diagrams, equations, and code define the exact mechanism.

## `llm-lifecycle.png`

**Alt text:** An end-to-end LLM lifecycle: source documents become token streams, pass through Transformer layers and routed experts, then become a checkpoint served inside a guarded application.

**Prompt summary:** A wordless 16:9 editorial illustration of diverse data sources, a token stream, a transparent Transformer, sparse expert routing, a checkpoint, server, and guarded tool-using application; midnight navy with violet, teal, and gold.

## `transformer-block.png`

**Alt text:** A decoder Transformer block with residual paths around causal multi-head attention and a gated feed-forward sublayer.

**Prompt summary:** A wordless 16:9 technical illustration tracing tokens through normalization, Q/K/V projections, triangular causal attention, head combination, residual addition, and a two-branch gated feed-forward network.

## `moe-routing.png`

**Alt text:** Colored token tiles pass through a router, take two of eight expert paths with capacity limits, recombine in original order, and rejoin a residual stream.

**Prompt summary:** A wordless 16:9 technical illustration of top-2 routing across eight FFN expert chambers, uneven load, capacity/overflow, balancing, weighted recombination, and a residual path.

The summaries above preserve the generation brief. No external logos or source figures were supplied as references.
