# Lab 2: inspect a causal attention matrix

```bash
python labs/02_attention.py
```

The printed matrix has query positions on rows and key positions on columns. Check two invariants:

1. each row sums to approximately one;
2. every cell above the main diagonal is zero.

```text
query 0 may see key 0
query 1 may see keys 0..1
query 2 may see keys 0..2
```

## Experiments

- Set every query and key to ones; allowed positions should receive uniform weights.
- Make one key vector align strongly with a later query and watch its weight grow where visible.
- Pass an `allowed_mask` that removes one earlier key.
- Intentionally mask an entire row and inspect the guard against an undefined softmax.

Then compare the explicit code with PyTorch's optimized `scaled_dot_product_attention`. Fused kernels avoid materializing some intermediate tensors but should respect the same mathematical contract.

