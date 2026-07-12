# Lab setup

The labs are intentionally laptop-sized. They preserve shape and behavior contracts while omitting production optimizations.

## Install

```bash
git clone https://github.com/ColtMercer/open-llm-engineering.git
cd open-llm-engineering
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev,docs]'
pytest
```

Or use the checked-in `uv.lock`:

```bash
uv sync --extra dev --extra docs
uv run pytest
```

Python 3.10+ is supported. The default lab commands use CPU. PyTorch backend support differs by operating system and build; keep `--device cpu` until the baseline works.

## What is implemented

```text
src/open_llm_lab/
├── tokenizer.py   deterministic byte-level BPE
├── attention.py   explicit causal scaled dot-product attention
├── model.py       pre-norm decoder-only Transformer
├── moe.py         top-k sparse FFN routing with capacity statistics
└── training.py    seeded toy batch helpers
```

## Reproducibility envelope

The labs fix seeds and default to CPU, but exact floating-point values can still depend on PyTorch version and backend. The tests assert contracts—shape, mask, gate normalization, round trip—rather than brittle full-output snapshots.

## Safety and scale

No lab downloads remote code, model weights, or a large dataset. The included corpus is a few lines written for this project. To use external models or corpora, review their code, licenses, terms, provenance, and resource requirements first.

## Suggested sequence

1. [Tokenizer](01-tokenizer.md)
2. [Causal attention](02-attention.md)
3. [Tiny GPT](03-tiny-gpt.md)
4. [Tiny MoE](04-tiny-moe.md)
5. [Generation](05-generation.md)
