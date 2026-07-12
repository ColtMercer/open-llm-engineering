"""A small decoder-only Transformer for shape-level learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CausalSelfAttention


@dataclass(frozen=True)
class TinyGPTConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_hidden: int = 384
    dropout: float = 0.0
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.d_hidden, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.attention_norm(x))
        x = x + attended
        return x + self.ffn(self.ffn_norm(x))


class TinyGPT(nn.Module):
    """A learned-position, pre-norm, decoder-only language model."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B,T]")
        _, time = token_ids.shape
        if time > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                if temperature < 0:
                    raise ValueError("temperature cannot be negative")
                next_logits = next_logits / temperature
                if top_k is not None:
                    if top_k < 1:
                        raise ValueError("top_k must be positive")
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                    cutoff = values[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
