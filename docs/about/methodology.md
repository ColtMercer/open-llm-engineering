# Research methodology

This book is designed to be auditable. It is an explanation layer over public evidence, not a claim of inside knowledge.

## Evidence ladder

Sources are preferred in this order:

1. released source code at the relevant project or organization;
2. official dataset cards, model cards, configuration, logs, and documentation;
3. original research papers;
4. official technical reports or engineering posts;
5. secondary sources only for context or discovery.

Links should sit next to the claim they support. Research ledgers in `research/` record the major sources, the claim scope, and caveats found while writing.

## Claim labels

| Label | Meaning |
|---|---|
| **Published** | Directly stated or implemented in a primary source. |
| **Derived** | Calculated from published quantities; the calculation is shown. |
| **Inference** | A reasoned interpretation not directly confirmed by the publisher. |
| **Heuristic** | Practical advice whose usefulness depends on context. |
| **Unknown** | Material information was not published or could not be verified. |

Numbers without labels still need local context: model version, units, inclusion rules, and date.

## Version discipline

Web pages and repositories move. Stable paper identifiers and versioned dataset/model cards are preferred. Source-code links should target a release, tag, or commit when a line-level argument depends on the exact implementation. A link to a default branch is acceptable for orientation, not for proving an invariant that may change.

## Code extraction policy

The book explains external code in original prose and uses short identifiers or minimal excerpts only when needed to anchor navigation. It links readers to the licensed upstream file for the full implementation. Companion code in this repository is independently written for teaching and is intentionally smaller than the referenced production systems.

## Reproducibility standard

A serious training claim should make these reconstructable:

- source inventory and corpus version;
- filtering, deduplication, decontamination, mixing, and tokenizer configuration;
- model configuration and parameter accounting;
- optimizer, schedule, batch semantics, precision, and seeds;
- parallelism topology, software versions, and hardware class;
- checkpoint cadence and recovery behavior;
- evaluation prompts, decoding settings, harness revision, and raw results;
- known deviations, failures, and exclusions.

Publishing weights alone satisfies none of the process items above.

## What this project will not do

- infer a proprietary training corpus from model outputs;
- represent benchmark scores from unlike harnesses as directly comparable;
- imply that a prompt can explicitly select a hidden MoE expert unless an interface actually exposes routing;
- provide operational instructions for bypassing safeguards or mishandling restricted data;
- call a corpus legally safe merely because it is publicly downloadable.

## Corrections

Corrections should identify the exact sentence, a primary source, the relevant version, and the proposed replacement. When evidence conflicts, the book should preserve the disagreement rather than flatten it into false certainty.

