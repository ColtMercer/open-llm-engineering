# Contributing

Open LLM Engineering values corrections and reproducible explanations more than volume.

## Ground rules

1. Prefer original papers, official project documentation, source code, dataset cards, and model reports.
2. Put a link immediately after the claim it supports. Do not cite a search result or a secondary summary when a primary source exists.
3. Label estimates, practitioner heuristics, and inferences explicitly.
4. Use short excerpts only when necessary; explain code in your own words and link to the licensed source.
5. Include tensor shapes and units in technical explanations.
6. Never present a dataset as risk-free. Record its license or terms, provenance limits, sensitive-content risks, and takedown mechanism where published.
7. Every new Mermaid block must render, and every Python change needs a focused test.

## Local workflow

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,docs]'
python scripts/check_docs.py
ruff check .
ruff format --check .
pytest
mkdocs build --strict
```

## Chapter pattern

A substantial chapter should contain:

- a one-paragraph mental model;
- stated prerequisites and learning objectives;
- a concrete trace or worked example;
- equations with every symbol defined;
- implementation or source-code trails;
- failure modes and limits;
- a checkpoint and exercises;
- primary references near the claims they support.

## Content corrections

For factual corrections, include the primary source and quote no more than needed to locate the evidence. If a fact changed since publication, say which version or date supersedes it.
