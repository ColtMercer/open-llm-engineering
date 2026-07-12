# Open-project atlas

No single adjective captures an AI release. Use this atlas to choose a project by the artifact you need, then verify the current license and version at the source.

**Checked:** July 2026. “Published” means an official project makes the artifact available or provides a reconstruction path; it does not promise affordable reproduction, unrestricted licensing, or absence of data risk.

## End-to-end research projects

| Project | Best reason to study it | Published artifact trail | Important boundary |
|---|---|---|---|
| [OLMo 3](https://allenai.org/blog/olmo3) | staged base, mid-training, long-context, instruction, thinking, and RL flows | [OLMo-core training scripts](https://github.com/allenai/OLMo-core), checkpoints, Dolma 3/Dolci data curriculum, evaluations and model-flow artifacts linked by Ai2 | inspect the license and availability of each individual artifact; stages have different purposes |
| [OLMo 2](https://github.com/allenai/OLMo) | reproducible dense pretraining with configs, data links, checkpoints, and logs | training/model/eval code, official configurations, staged checkpoints, data files, linked run logs | older OLMo repository and newer OLMo-core represent different generations of the stack |
| [OLMoE](https://allenai.org/olmo) | open sparse-MoE study | weights, training code, data, evaluations, logs, and checkpoints linked by the project | small research MoE is a mechanism case study, not a proxy for every frontier MoE |
| [Pythia](https://github.com/EleutherAI/pythia) | learning dynamics across sizes and steps | code, data reconstruction, controlled data order, and 154 checkpoints per main model described by the project | trained on versions of The Pile; study that corpus's provenance and availability separately |
| [BLOOM](https://huggingface.co/bigscience/bloom) | multilingual collaborative open science at large scale | model card, ROOTS data cards, training-code trails, optimizer/intermediate artifacts | weights use the BigScience RAIL license, which is not the same as a permissive software license |
| [LLM360 Amber](https://www.llm360.ai/) | release process and intermediate training artifacts | project links to model, data, code, configurations, metrics, and checkpoints | verify current mirrors and artifact-specific licenses before planning a reproduction |

## Educational and implementation projects

| Project | Use it to learn | Not intended to prove |
|---|---|---|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | a compact GPT-2-style model and training loop | an end-to-end modern frontier training stack |
| [build-nanogpt](https://github.com/karpathy/build-nanogpt) | a commit-by-commit construction trace | data governance or large-scale post-training |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | clean implementations across model families, pretraining, fine-tuning, and inference | exact provenance of checkpoints trained by third parties |
| [minbpe](https://github.com/karpathy/minbpe) | byte-level BPE mechanics | production multilingual tokenizer engineering |
| [llm.c](https://github.com/karpathy/llm.c) | GPT-2 training and inference close to C/CUDA kernels | a general-purpose framework for all current architectures |

## Build your component ledger

For any model family, fill this table with links, not checkmarks from memory:

| Component | Evidence | License/terms | Version | Status |
|---|---|---|---|---|
| Weights |  |  |  | available / partial / absent / unknown |
| Model code |  |  |  |  |
| Training code |  |  |  |  |
| Exact pretraining data |  |  |  |  |
| Data processing |  |  |  |  |
| Data mixture/order |  |  |  |  |
| Tokenizer training |  |  |  |  |
| Optimizer state |  |  |  |  |
| Intermediate checkpoints |  |  |  |  |
| Logs |  |  |  |  |
| Post-training data/code |  |  |  |  |
| Evaluation prompts/raw outputs |  |  |  |  |

## Four questions hidden by “open source model”

1. **Can I run it?** Weights, inference code, runtime support, hardware, license.
2. **Can I inspect it?** Architecture, implementation, tokenizer, documentation.
3. **Can I reproduce it?** Training data or sufficient data information, processing, training code, configuration, compute, state, and evals.
4. **Can I legally modify and redistribute it?** Every relevant license and dataset term.

The [OSI Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition) provides one formal standard for the freedoms and preferred form needed to modify an AI system. Project marketing terms do not override licenses.

