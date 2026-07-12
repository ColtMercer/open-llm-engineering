#!/usr/bin/env python3
"""Overfit a tiny decoder-only Transformer on a tiny local corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from open_llm_lab.model import TinyGPT, TinyGPTConfig
from open_llm_lab.training import random_next_token_batch, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    seed_everything(17)

    text = Path(__file__).with_name("data").joinpath("tiny_corpus.txt").read_text() * 20
    tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    device = torch.device(args.device)
    model = TinyGPT(
        TinyGPTConfig(
            vocab_size=256,
            max_seq_len=48,
            d_model=48,
            n_heads=4,
            n_layers=2,
            d_hidden=128,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch_generator = torch.Generator().manual_seed(17)

    for step in range(args.steps):
        inputs, targets = random_next_token_batch(
            tokens, batch_size=8, sequence_length=48, generator=batch_generator
        )
        _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0:
            print(f"step={step + 1:03d} loss={loss.item():.4f}")

    prompt = torch.tensor([list(b"Tokens ")], dtype=torch.long, device=device)
    sample_generator = torch.Generator(device=device).manual_seed(23)
    generated = model.generate(prompt, 100, temperature=0.75, top_k=32, generator=sample_generator)
    print("\n" + bytes(generated[0].cpu().tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
