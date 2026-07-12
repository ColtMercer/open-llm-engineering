"""A compact sparse mixture-of-experts feed-forward layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    """SwiGLU expert mapping ``d_model -> d_hidden -> d_model``."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


@dataclass
class RouterResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    expert_indices: torch.Tensor
    gates: torch.Tensor
    auxiliary_loss: torch.Tensor


class Router(nn.Module):
    """Learned top-k token router."""

    def __init__(self, d_model: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.n_experts = n_experts
        self.top_k = top_k
        self.projection = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, tokens: torch.Tensor) -> RouterResult:
        logits = self.projection(tokens)
        probabilities = torch.softmax(logits.float(), dim=-1).to(tokens.dtype)
        gates, expert_indices = probabilities.topk(self.top_k, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Switch-style differentiable importance times top-1 assignment frequency.
        importance = probabilities.float().mean(dim=0)
        top_one = F.one_hot(expert_indices[:, 0], self.n_experts).float().mean(dim=0)
        auxiliary_loss = self.n_experts * torch.sum(importance * top_one)
        return RouterResult(logits, probabilities, expert_indices, gates, auxiliary_loss)


@dataclass
class MoEStats:
    accepted_per_expert: torch.Tensor
    requested_per_expert: torch.Tensor
    dropped_assignments: int
    capacity: int | None
    auxiliary_loss: torch.Tensor
    expert_indices: torch.Tensor
    gates: torch.Tensor


class SparseMoE(nn.Module):
    """Route each token to ``top_k`` feed-forward experts.

    Capacity, when enabled, is applied per expert in flattened token/slot order.
    Overflow assignments contribute zero. In a Transformer block the surrounding
    residual path still carries the token; production systems use more optimized
    dispatch and may implement different overflow policies.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_experts: int,
        *,
        top_k: int = 2,
        capacity_factor: float | None = None,
    ) -> None:
        super().__init__()
        if capacity_factor is not None and capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = Router(d_model, n_experts, top_k)
        self.experts = nn.ModuleList(Expert(d_model, d_hidden) for _ in range(n_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoEStats]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected [B,T,{self.d_model}] input")
        flat = x.reshape(-1, self.d_model)
        routed = self.router(flat)
        output = torch.zeros_like(flat)
        requested = torch.bincount(routed.expert_indices.reshape(-1), minlength=self.n_experts)
        accepted = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)

        capacity = None
        if self.capacity_factor is not None:
            expected = flat.shape[0] * self.top_k / self.n_experts
            capacity = max(1, math.ceil(self.capacity_factor * expected))

        dropped = 0
        for expert_id, expert in enumerate(self.experts):
            token_index, slot_index = torch.where(routed.expert_indices == expert_id)
            if capacity is not None and token_index.numel() > capacity:
                dropped += token_index.numel() - capacity
                token_index = token_index[:capacity]
                slot_index = slot_index[:capacity]
            if token_index.numel() == 0:
                continue
            expert_output = expert(flat[token_index])
            weighted = expert_output * routed.gates[token_index, slot_index].unsqueeze(-1)
            output.index_add_(0, token_index, weighted)
            accepted[expert_id] = token_index.numel()

        stats = MoEStats(
            accepted_per_expert=accepted,
            requested_per_expert=requested,
            dropped_assignments=int(dropped),
            capacity=capacity,
            auxiliary_loss=routed.auxiliary_loss,
            expert_indices=routed.expert_indices.reshape(*x.shape[:2], self.top_k),
            gates=routed.gates.reshape(*x.shape[:2], self.top_k),
        )
        return output.reshape_as(x), stats
