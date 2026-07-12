import torch

from open_llm_lab.model import TinyGPT, TinyGPTConfig


def test_model_forward_loss_and_generation() -> None:
    torch.manual_seed(5)
    config = TinyGPTConfig(
        vocab_size=32, max_seq_len=8, d_model=16, n_heads=4, n_layers=2, d_hidden=32
    )
    model = TinyGPT(config)
    inputs = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))

    logits, loss = model(inputs, targets)
    generated = model.generate(inputs[:, :2], 3, temperature=0)

    assert logits.shape == (2, 6, 32)
    assert loss is not None and loss.ndim == 0
    assert generated.shape == (2, 5)
