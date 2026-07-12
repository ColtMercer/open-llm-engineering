#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer and inspect one encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_llm_lab.tokenizer import BytePairTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Tokens turn text into reusable pieces 🌧️")
    parser.add_argument("--vocab-size", type=int, default=300)
    args = parser.parse_args()

    corpus = Path(__file__).with_name("data").joinpath("tiny_corpus.txt").read_text()
    tokenizer = BytePairTokenizer.train(
        [corpus, args.text] * 12,
        vocab_size=args.vocab_size,
        min_frequency=2,
    )
    ids = tokenizer.encode(args.text)
    report = {
        "text": args.text,
        "utf8_bytes": len(args.text.encode("utf-8")),
        "learned_vocabulary_size": tokenizer.vocab_size,
        "token_count": len(ids),
        "token_ids": ids,
        "pieces": tokenizer.pieces(ids),
        "round_trip": tokenizer.decode(ids),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
