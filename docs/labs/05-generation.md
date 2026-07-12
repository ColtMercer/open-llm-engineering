# Lab 5: decoding changes outputs, not knowledge

```bash
python labs/05_generation.py --draws 1000
```

The lab samples repeatedly from one fixed logit vector. Greedy decoding always chooses the maximum. Lower temperature sharpens the distribution; higher temperature flattens it; top-k removes all but the highest-k candidates before sampling.

## Key distinction

The logits are fixed. Decoding changes **which candidate is selected**, not the model weights or the evidence in context. A more random answer is not new knowledge, and a deterministic answer is not guaranteed truth.

## Experiments

1. Add a tied maximum and inspect greedy tie behavior.
2. Compare top-k with nucleus/top-p sampling.
3. Estimate empirical entropy from the draw counts.
4. Repeat with several seeds and confidence intervals.
5. Construct a distribution where the correct token is ranked second; explain what each decoding rule can and cannot repair.

