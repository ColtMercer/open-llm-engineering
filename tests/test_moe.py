import torch

from open_llm_lab.moe import SparseMoE


def test_sparse_moe_shapes_and_gate_normalization() -> None:
    torch.manual_seed(3)
    layer = SparseMoE(d_model=8, d_hidden=16, n_experts=4, top_k=2)
    output, stats = layer(torch.randn(2, 5, 8))

    assert output.shape == (2, 5, 8)
    assert stats.expert_indices.shape == (2, 5, 2)
    assert torch.allclose(stats.gates.sum(dim=-1), torch.ones(2, 5))
    assert int(stats.accepted_per_expert.sum()) == 20
    assert stats.dropped_assignments == 0


def test_capacity_reports_overflow() -> None:
    layer = SparseMoE(d_model=4, d_hidden=8, n_experts=2, top_k=1, capacity_factor=0.5)
    with torch.no_grad():
        layer.router.projection.weight.zero_()  # top-k tie consistently picks one expert
    _, stats = layer(torch.ones(1, 8, 4))

    assert stats.capacity == 2
    assert stats.dropped_assignments == 6
