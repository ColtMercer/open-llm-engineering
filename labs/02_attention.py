#!/usr/bin/env python3
"""Print a causal attention matrix and verify its invariants."""

from __future__ import annotations

import torch

from open_llm_lab.attention import scaled_dot_product_attention


def main() -> None:
    torch.manual_seed(11)
    query = torch.randn(1, 1, 5, 4)
    key = torch.randn(1, 1, 5, 4)
    value = torch.arange(20, dtype=torch.float32).view(1, 1, 5, 4)
    output, weights = scaled_dot_product_attention(query, key, value)

    torch.set_printoptions(precision=3, sci_mode=False)
    print("attention weights [query position, key position]")
    print(weights[0, 0])
    print("\nrow sums:", weights[0, 0].sum(dim=-1))
    print("future mass:", weights[0, 0].triu(diagonal=1).sum().item())
    print("output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
