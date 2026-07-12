# Foundations and tokenization source ledger

Checked for the 2026 first edition. This ledger records primary anchors; chapter links carry the actual claims.

| Topic | Primary source | Used for | Caveat |
|---|---|---|---|
| Open source AI | [OSI Open Source AI Definition 1.0](https://opensource.org/ai/open-source-ai-definition) | Component-level openness framing | A definition does not replace each artifact's license review. |
| Transformer | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | Scaled dot-product attention and original architecture | Modern decoder-only LLMs differ from the original encoder-decoder. |
| Neural BPE | [Sennrich et al., 2015](https://arxiv.org/abs/1508.07909) | Iterative pair merges for subword vocabularies | GPT-style byte-level variants add different preprocessing. |
| SentencePiece | [Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226) | Training from raw text and Unigram/BPE implementation | A library supports multiple algorithms; it is not one algorithm. |
| Byte-level BPE | [OpenAI GPT-2 encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py) | Reversible byte mapping and BPE mechanics | Repository default branch links can move. |
| Educational GPT | [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) | Compact decoder-only model trail | Teaching-scale clarity, not a distributed training stack. |
| PyTorch tensors/autograd | [PyTorch documentation](https://pytorch.org/docs/stable/index.html) | Tensor and gradient API semantics | Behavior can vary by PyTorch version and backend. |

