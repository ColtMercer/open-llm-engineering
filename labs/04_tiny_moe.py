#!/usr/bin/env python3
"""Inspect top-k routing, gate weights, load, and capacity overflow."""

from __future__ import annotations

import argparse

import torch

from open_llm_lab.moe import SparseMoE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(29)

    labels = ["math", "poem", "router", "bonjour", "python", "history"]
    token_vectors = torch.randn(1, len(labels), 12)
    layer = SparseMoE(
        d_model=12,
        d_hidden=24,
        n_experts=4,
        top_k=2,
        capacity_factor=args.capacity_factor,
    )
    output, stats = layer(token_vectors)

    print("token      selected experts    normalized gates")
    for index, label in enumerate(labels):
        experts = stats.expert_indices[0, index].tolist()
        gates = [round(value, 3) for value in stats.gates[0, index].tolist()]
        print(f"{label:<10} {str(experts):<19} {gates}")
    print("\nrequested per expert:", stats.requested_per_expert.tolist())
    print("accepted per expert: ", stats.accepted_per_expert.tolist())
    print("capacity per expert: ", stats.capacity)
    print("dropped assignments: ", stats.dropped_assignments)
    print("auxiliary loss:      ", round(stats.auxiliary_loss.item(), 4))
    print("output shape:        ", tuple(output.shape))
    print("\nThe labels are for us; this untrained router sees only numeric vectors.")


if __name__ == "__main__":
    main()
