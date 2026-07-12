# Dataset and Open-Stack Reference

Last reviewed: **2026-07-12**

This is the compact link index for the data chapters. Sizes are publisher-reported and are not normalized across tokenizers. Access, cards and terms can change; pin an immutable revision at the start of an experiment.

## Legend

- **Direct:** ordinary files are publicly reachable.
- **Stream:** the publisher documents bounded/streaming access.
- **Gated:** sign-in or terms acceptance is required.
- **IDs:** repository contains identifiers/metadata; content access is separate.
- **Historical:** official reconstruction artifacts exist, but original complete hosting or components are brittle.

“Terms” names only the headline database/access layer. It is not a conclusion about every contained work.

## Raw and filtered web

| Dataset | Scale and scope | Access | Terms caveat | Primary trail |
|---|---|---|---|---|
| Common Crawl | Recurring web archives. June 2026: 2.10B pages, 354.59 TiB uncompressed. | Direct WARC/WAT/WET, indexes | [ToU](https://commoncrawl.org/terms-of-use); crawled content may have separate third-party rights/terms | [Get Started](https://commoncrawl.org/get-started) · [June 2026 release](https://commoncrawl.org/blog/june-2026-crawl-archive-now-available) · [formats](https://commoncrawl.org/blog/navigating-the-warc-file-format) |
| C4 | English TFDS: 806.87 GiB; 364.6M training rows. mC4 is a separate multilingual configuration. | Rebuild/Direct mirror/Stream mirror | Web-derived. Review Common Crawl and the selected distribution endpoint. | [TFDS card](https://www.tensorflow.org/datasets/catalog/c4) · [T5 preparation code](https://github.com/google-research/text-to-text-transfer-transformer#c4) · [paper](https://jmlr.org/papers/v21/20-074.html) · [AI2 mirror](https://huggingface.co/datasets/allenai/c4) |
| RefinedWeb | Public English extract: 968M pages, about 500–650B tokenizer-dependent tokens; paper describes a 5T-token internal corpus. | Direct/Stream | ODC-By 1.0 + Common Crawl ToU; individual-content rights remain | [card](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) · [paper](https://arxiv.org/abs/2306.01116) |
| FineWeb | Maintained card: more than 18.5T GPT-2-tokenizer tokens of filtered English Common Crawl. | Stream; per-crawl and 10B/100B/350B-token sample configs | ODC-By 1.0 + Common Crawl ToU | [card](https://huggingface.co/datasets/HuggingFaceFW/fineweb) · [DataTrove](https://github.com/huggingface/datatrove) · [pipeline](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py) · [paper](https://openreview.net/forum?id=n6SCkn2QaG) |
| FineWeb-Edu | Default score-3: 1.3T tokens; score-2: 5.4T. English educationally scored FineWeb. | Stream; per-crawl/sample configs | ODC-By/Common Crawl layers; learned classifier defines selection | [card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) · [paper](https://arxiv.org/abs/2406.17557) · [annotation set](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations) |

## Mixed-domain English corpora

| Dataset | Scale and composition | Access | Terms caveat | Primary trail |
|---|---|---|---|---|
| Dolma v1 | v1.5 ~3T tokens; v1.7 2.3085T available OLMo tokens. Web, papers, code, social, books, reference. | Version URL manifests; 10B-token v1.6 sample | Database ODC-By; toolkit Apache-2.0; contents may have other rights | [card](https://huggingface.co/datasets/allenai/dolma) · [toolkit](https://github.com/allenai/dolma) · [paper/datasheet](https://arxiv.org/abs/2402.00159) |
| Dolma 3 | ~9.3T pool; 5.9T pretraining, 100B midtraining and 50B long-context mixes for OLMo 3. | Direct, versioned mixes | Data ODC-By; reconstruction code Apache-2.0. The 7B reproduction card documents science-PDF redactions. | [6T mix](https://huggingface.co/datasets/allenai/dolma3_mix-6T) · [reconstruction](https://github.com/allenai/dolma3) · [OLMo 3 release](https://allenai.org/blog/olmo3) · [paper](https://allenai.org/papers/olmo3) |
| RedPajama v1 | About 1.2T tokens across web, C4, GitHub, books, arXiv, Wikipedia and Stack Exchange. | Direct | Source-specific terms; pipeline license is not blanket content license | [card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) · [`rp_v1` code](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1) |
| RedPajama v2 | 84 Common Crawl snapshots; 100B+ raw docs; 30B annotated; 20.8B deduped head/middle docs and ~30.4T estimated tokens across five languages. | Small sample custom loader, URL lists, streaming by snapshot | Data follows Common Crawl ToU; code Apache-2.0 | [card/schema](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) · [pipeline](https://github.com/togethercomputer/RedPajama-Data) · [paper](https://arxiv.org/abs/2411.12372) |
| SlimPajama | 627B tokens; cleaned and MinHashLSH-deduplicated RedPajama v1; 895 GB compressed. | Direct/Stream | Card directs user to each constituent source’s license | [card/schema](https://huggingface.co/datasets/cerebras/SlimPajama-627B) · [code](https://github.com/Cerebras/modelzoo/tree/main/src/cerebras/modelzoo/data_preparation/nlp/slimpajama) · [publisher article](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) |
| The Pile | 825 GiB, 22 English components. | Historical | No uniform component license; Books3/manual hosting caveat | [paper](https://arxiv.org/abs/2101.00027) · [dataset card](https://huggingface.co/datasets/EleutherAI/pile) · [replication code](https://github.com/EleutherAI/the-pile) |

## Multilingual corpora

| Dataset | Scale and scope | Access | Terms caveat | Primary trail |
|---|---|---|---|---|
| ROOTS | 1.6 TB, 498 datasets, 46 natural and 13 programming languages; BLOOM corpus. | Gated, component-by-component | Ethical Charter + per-component licenses/terms | [paper](https://arxiv.org/abs/2303.03915) · [data organization](https://huggingface.co/bigscience-data) · [corpus explorer](https://huggingface.co/spaces/bigscience/BigScienceCorpus) · [processing code](https://github.com/bigscience-workshop/data-preparation) · [charter](https://huggingface.co/spaces/bigscience/ethical-charter) |
| CulturaX | 6.3T tokens, 167 languages, 16 TB Parquet; cleaned mC4 + OSCAR. | Gated, per-language | Terms follow mC4 and OSCAR; card warns sensitive information may remain | [card/schema](https://huggingface.co/datasets/uonlp/CulturaX) · [paper](https://aclanthology.org/2024.lrec-main.377/) · [KenLM models](https://huggingface.co/uonlp/kenlm) |
| RedPajama v2 | English, German, French, Spanish and Italian web data with quality signals. | See mixed-domain table | Common Crawl ToU | [card](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) |

## Code corpus

| Dataset | Scale and scope | Access | Terms caveat | Primary trail |
|---|---|---|---|---|
| The Stack v2 | 67.5 TB; 3.28B unique files; 104.2M GitHub repositories; 658 programming/markup languages. Full training subset reported at ~900B tokens. | Gated IDs; separate Software Heritage content agreement | Original per-file licenses, attribution, Software Heritage principles and removal updates apply | [card/schema](https://huggingface.co/datasets/bigcode/the-stack-v2) · [curation code](https://github.com/bigcode-project/the-stack-v2) · [paper](https://arxiv.org/abs/2402.19173) · [governance card](https://huggingface.co/datasets/bigcode/governance-card) |

## Verified schema snapshots

| Dataset | Core published fields |
|---|---|
| C4 | `text`, `url`, `content-type`, `content-length`, `timestamp` |
| FineWeb | `text`, `id`, `dump`, `url`, `date`, `file_path`, `language`, `language_score`, `token_count` |
| FineWeb-Edu | FineWeb fields + `score`, `int_score` |
| RefinedWeb | `content`, `url`, `timestamp`, `dump`, `segment`, `image_urls` |
| Dolma 3 7B mix | `id`, `text`, `metadata`, `source`, `version`, `created`, `added`, `doc`, `attributes` |
| RedPajama v2 document | `url`, `date_download`, `digest`, lengths, domain/title, `raw_content`, `cc_segment`, line IDs, language/score, perplexity, bucket |
| RedPajama v2 signal | `id`, `id_int`, source metadata, named `quality_signals` represented as `(start,end,score)` spans |
| SlimPajama | `text`, `meta.redpajama_set_name` |
| CulturaX | `text`, `timestamp`, `url`, `source` |
| The Stack v2 | SWH IDs, path/repository/revision, detected licenses, dates, language/encoding, generated/vendor flags and quality metadata |
| ROOTS | Component-specific; read each card rather than assuming one schema |
| Common Crawl | WARC records; derived WAT metadata and WET text, not a single JSON training schema |

## Bounded access recipes

### Metadata only

```python
from huggingface_hub import HfApi

info = HfApi().dataset_info("HuggingFaceFW/fineweb", files_metadata=False)
print(info.id, info.sha, info.card_data.get("license"))
```

### FineWeb sample

```python
from datasets import load_dataset

rows = load_dataset(
    "HuggingFaceFW/fineweb",
    "sample-10BT",
    split="train",
    streaming=True,
)
for row in rows.take(3):
    print(row["id"], row["url"])
```

### RefinedWeb

```python
from datasets import load_dataset

rows = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
for row in rows.take(3):
    print(row["url"], row["dump"])
```

### SlimPajama validation

```python
from datasets import load_dataset

rows = load_dataset(
    "cerebras/SlimPajama-627B", split="validation", streaming=True
)
for row in rows.take(3):
    print(row["meta"]["redpajama_set_name"])
```

### RedPajama v2 small sample

```python
from datasets import load_dataset

# Custom loader: inspect and pin its revision before execution.
sample = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample")
print(sample)
```

### Gated data

After reviewing/accepting the applicable terms and authenticating:

```python
from datasets import load_dataset

culturax = load_dataset(
    "uonlp/CulturaX", "en", split="train", streaming=True, token=True
)

stack_ids = load_dataset(
    "bigcode/the-stack-v2", "Python", split="train", streaming=True, token=True
)
```

Never turn a metadata probe into an unbounded `list(dataset)` call.

## Projects with substantial end-to-end artifact trails

| Project | Data | Training/data code | Intermediate artifacts | Evaluation/logs | Key caveat |
|---|---|---|---|---|---|
| OLMo 3 | [Dolma 3 mixes](https://github.com/allenai/dolma3) + Dolci post-training data | [OLMo-core](https://github.com/allenai/OLMo-core), [Open Instruct](https://github.com/allenai/open-instruct) | Model-card revisions and checkpoint manifests | [OLMES](https://github.com/allenai/olmes), linked W&B reports | Large compute; 7B data card documents post-run redactions; some reconstruction paths are internal |
| OLMo 2 | [OLMo mix / Dolmino links](https://github.com/allenai/OLMo) | OLMo repository; 32B in OLMo-core | Frequent checkpoints | Evaluation code/log links in project | Original repository now directs current work to OLMo-core |
| Pythia | Pile and [pretokenized ordered data](https://github.com/EleutherAI/pythia#reproducing-training) | [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), exact configs | 154 checkpoints per main run; optimizer states for selected points | [project/paper artifacts](https://github.com/EleutherAI/pythia) | Inherits Pile component issues; read documented initialization and naming errata |
| Amber | [AmberDatasets](https://huggingface.co/datasets/LLM360/AmberDatasets), full sequence | [training source](https://github.com/LLM360/amber-train), [data prep](https://github.com/LLM360/amber-data-prep) | 360 checkpoints | Linked W&B logs/evals | Upstream data rights remain source-specific |
| BLOOM | [ROOTS components](https://huggingface.co/bigscience-data) | [176B run repository](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml), Megatron-DeepSpeed fork | [intermediate checkpoints](https://huggingface.co/bigscience/bloom-intermediate) | Training logs and chronicles linked by model card | BigScience BLOOM RAIL model license; heterogeneous ROOTS terms |
| OpenLLaMA | RedPajama v1 or v2-specific mixed sources | [EasyLM + project instructions](https://github.com/openlm-research/open_llama) | Several published progress checkpoints | Evaluation tables in repository | Less complete ordered-data/log/optimizer trail than OLMo/Pythia/Amber; v1/v2 differ |
| RedPajama-INCITE 7B | [RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | Model card documents trainer/hardware configuration | 11 checkpoints from 240B to 1T tokens | Benchmark links in card | Not a complete exact replay bundle |

Primary model links:

- [OLMo 3 32B model card](https://huggingface.co/allenai/Olmo-3-1125-32B)
- [Pythia paper](https://arxiv.org/abs/2304.01373)
- [Amber model card](https://huggingface.co/LLM360/Amber) and [LLM360 paper](https://arxiv.org/abs/2312.06550)
- [BLOOM model card](https://huggingface.co/bigscience/bloom) and [paper](https://arxiv.org/abs/2211.05100)
- [OpenLLaMA project](https://github.com/openlm-research/open_llama)
- [RedPajama-INCITE 7B Base card](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base)

## Data tooling and primary methods

| Need | Project/source |
|---|---|
| General scalable readers, filters, writers, stats and dedupe | [DataTrove](https://github.com/huggingface/datatrove) |
| Tag, exact/paragraph dedupe, mix, stats and tokenize | [Dolma toolkit](https://github.com/allenai/dolma) |
| Web quality signals and multiple MinHash signatures | [RedPajama v2 pipeline](https://github.com/togethercomputer/RedPajama-Data) |
| Exact and substring deduplication study/code | [ACL paper](https://aclanthology.org/2022.acl-long.577/) · [code](https://github.com/google-research/deduplicate-text-datasets) |
| Multilingual source-specific preprocessing | [BigScience data-preparation](https://github.com/bigscience-workshop/data-preparation) |
| Code corpus curation | [The Stack v2 repository](https://github.com/bigcode-project/the-stack-v2) |
| PDF linearization with Dolma-compatible provenance | [olmOCR](https://github.com/allenai/olmocr) |
| Dataset documentation | [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) · [Hub dataset cards](https://huggingface.co/docs/hub/datasets-cards) |
| Database-license boundary | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/index.html) |
| Open-source AI artifact definition | [OSI Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition) · [FAQ](https://opensource.org/ai/faq) |

## Terminology guardrails

| Phrase | Use it only when… |
|---|---|
| Open weights | parameters are downloadable under stated terms; says nothing by itself about data/training code |
| Open training code | actual trainer and exact configuration are available under an open-source license |
| Open data | data can be used, modified and reshared under applicable open terms; public visibility alone is insufficient |
| Public data | people can inspect/obtain it while it remains available; redistribution rights may be unclear or restricted |
| Fully reproducible | exact inputs/order, code/environment, randomness, hardware-relevant settings and compute are sufficient and replay has been demonstrated |
| Reproducible recipe | enough instructions/code exist to build a similar system; no claim of identical weights |
| Fully open | accompanied by an explicit checklist for data information, code, parameters, intermediate artifacts and terms |

For claim-by-claim evidence and caveats, see the
[research source ledger](https://github.com/ColtMercer/open-llm-engineering/blob/main/research/source-ledger-data.md).
