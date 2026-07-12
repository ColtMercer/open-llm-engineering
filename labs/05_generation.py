#!/usr/bin/env python3
"""Compare greedy, temperature, and top-k decoding on fixed logits."""

from __future__ import annotations

import argparse

import torch


def sample(
    logits: torch.Tensor,
    *,
    draws: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax().repeat(draws)
    scaled = logits / temperature
    if top_k is not None:
        values, _ = scaled.topk(top_k)
        scaled = scaled.masked_fill(scaled < values[-1], float("-inf"))
    generator = torch.Generator().manual_seed(seed)
    return torch.multinomial(
        torch.softmax(scaled, dim=-1),
        draws,
        replacement=True,
        generator=generator,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=1_000)
    args = parser.parse_args()
    vocabulary = ["blue", "gray", "clear", "falling", "banana"]
    logits = torch.tensor([3.2, 2.1, 1.7, 0.8, -1.0])

    for name, temperature, top_k in [
        ("greedy", 0.0, None),
        ("temperature=0.7", 0.7, None),
        ("temperature=1.3", 1.3, None),
        ("top-k=2", 1.0, 2),
    ]:
        draws = sample(
            logits,
            draws=args.draws,
            temperature=temperature,
            top_k=top_k,
            seed=41,
        )
        counts = torch.bincount(draws, minlength=len(vocabulary))
        frequencies = {
            token: round(count.item() / args.draws, 3)
            for token, count in zip(vocabulary, counts, strict=True)
        }
        print(f"{name:<18} {frequencies}")


if __name__ == "__main__":
    main()
