import torch

from open_llm_lab.attention import CausalSelfAttention, scaled_dot_product_attention


def test_causal_mask_blocks_future_positions() -> None:
    q = torch.ones(1, 1, 3, 2)
    k = torch.ones(1, 1, 3, 2)
    v = torch.tensor([[[[1.0, 0.0], [3.0, 0.0], [9.0, 0.0]]]])

    output, weights = scaled_dot_product_attention(q, k, v)

    assert torch.allclose(output[0, 0, 0], v[0, 0, 0])
    assert weights[0, 0, 0, 1:].sum() == 0
    assert weights[0, 0, 1, 2] == 0


def test_multihead_shape_contract() -> None:
    layer = CausalSelfAttention(d_model=12, n_heads=3)
    output, weights = layer(torch.randn(2, 5, 12))

    assert output.shape == (2, 5, 12)
    assert weights.shape == (2, 3, 5, 5)
