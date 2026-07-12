# Lab 3: overfit a tiny GPT

```bash
python labs/03_tiny_gpt.py --steps 80
```

The model uses byte IDs directly, learned position embeddings, pre-normalized causal attention, SwiGLU feed-forward layers, residual connections, and a tied language-model head. It trains on the included tiny corpus and then samples bytes.

This is an **overfitting demonstration**, not evidence of useful generalization. The corpus is repeated so loss falls in seconds on a CPU.

## Follow one batch

```text
tokens [B,T+1]
  ├─ inputs  = tokens[:, :-1] [B,T]
  └─ targets = tokens[:,  1:] [B,T]
inputs -> logits [B,T,256]
logits + targets -> scalar cross-entropy
```

## Experiments

1. Set `n_layers=0`; embeddings and output head form a weak baseline.
2. Untie embeddings and compare parameter count.
3. Increase sequence length and time one step.
4. Hold out one sentence and compare train versus validation loss.
5. Save model, optimizer, step, generator, and data state; try an exact resume.

