"""Causal attention with explicit tensor contracts."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attended values and attention weights.

    ``query``, ``key``, and ``value`` have shape ``[batch, heads, time, head_dim]``.
    ``allowed_mask`` is boolean and broadcastable to ``[batch, heads, q_time, kv_time]``;
    true entries are visible. This explicit implementation favors inspection over speed.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B,H,T,D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("batch, head, and feature dimensions must agree")

    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    visible = torch.ones((query.shape[-2], key.shape[-2]), dtype=torch.bool, device=query.device)
    if causal:
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("this teaching causal path expects equal query and key lengths")
        visible = visible.tril()
    if allowed_mask is not None:
        if allowed_mask.dtype is not torch.bool:
            raise TypeError("allowed_mask must be boolean, where True means visible")
        visible = visible & allowed_mask

    scores = scores.masked_fill(~visible, float("-inf"))
    if torch.isinf(scores).all(dim=-1).any():
        raise ValueError("at least one query has no visible key")
    weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return weights @ value, weights


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention operating on ``[B,T,C]`` activations."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, channels = x.shape
        if channels != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {channels}")
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)

        attended, weights = scaled_dot_product_attention(
            split_heads(q),
            split_heads(k),
            split_heads(v),
            causal=True,
            dropout_p=self.dropout,
            training=self.training,
        )
        joined = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.output(joined), weights
