"""Tiny training helpers shared by the executable labs."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_next_token_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    if len(tokens) <= sequence_length:
        raise ValueError("corpus must be longer than sequence_length")
    starts = torch.randint(0, len(tokens) - sequence_length, (batch_size,), generator=generator)
    inputs = torch.stack([tokens[start : start + sequence_length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + sequence_length + 1] for start in starts])
    return inputs, targets
