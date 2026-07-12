"""A deliberately small byte-pair tokenizer for teaching.

It starts from the 256 byte values, learns deterministic pair merges, and keeps
document boundaries separate while counting pairs. It is not optimized for
large corpora and has no special-token protocol; production tokenizers need
both performance engineering and a richer, versioned artifact contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def _merge_pair(ids: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of ``pair`` from left to right."""
    merged: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


@dataclass(frozen=True)
class Merge:
    left: int
    right: int
    token_id: int


class BytePairTokenizer:
    """A deterministic byte-level BPE tokenizer.

    IDs 0 through 255 always represent their matching byte value. Learned IDs
    are assigned in merge order starting at 256.
    """

    def __init__(self, merges: Iterable[Merge] = ()) -> None:
        self.merges = tuple(merges)
        self._merge_ranks = {
            (merge.left, merge.right): (rank, merge.token_id)
            for rank, merge in enumerate(self.merges)
        }
        self._vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for merge in self.merges:
            if merge.left not in self._vocab or merge.right not in self._vocab:
                raise ValueError("merge references an ID that has not been constructed")
            self._vocab[merge.token_id] = self._vocab[merge.left] + self._vocab[merge.right]

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 300,
        min_frequency: int = 2,
    ) -> BytePairTokenizer:
        """Learn merges from independent text records.

        Pair ties are resolved lexicographically so identical inputs and
        settings produce identical merge lists.
        """
        if vocab_size < 256:
            raise ValueError("byte-level BPE needs at least 256 vocabulary entries")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        records = [list(text.encode("utf-8")) for text in texts]
        merges: list[Merge] = []

        while 256 + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for record in records:
                counts.update(zip(record, record[1:], strict=False))
            eligible = [(pair, count) for pair, count in counts.items() if count >= min_frequency]
            if not eligible:
                break
            best_pair, _ = min(eligible, key=lambda item: (-item[1], item[0]))
            new_id = 256 + len(merges)
            merges.append(Merge(*best_pair, token_id=new_id))
            records = [_merge_pair(record, best_pair, new_id) for record in records]

        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode_bytes(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) > 1:
            candidates = {
                pair: self._merge_ranks[pair]
                for pair in zip(ids, ids[1:], strict=False)
                if pair in self._merge_ranks
            }
            if not candidates:
                break
            pair, (_, new_id) = min(candidates.items(), key=lambda item: item[1][0])
            ids = _merge_pair(ids, pair, new_id)
        return ids

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        try:
            return b"".join(self._vocab[token_id] for token_id in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token ID: {exc.args[0]}") from exc

    def decode(self, ids: Iterable[int], *, errors: str = "strict") -> str:
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def pieces(self, ids: Iterable[int]) -> list[str]:
        """Return escaped teaching views; pieces need not be valid standalone UTF-8."""
        return [
            self._vocab[token_id].decode("utf-8", errors="backslashreplace") for token_id in ids
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "byte_bpe",
            "version": 1,
            "merges": [
                {"left": merge.left, "right": merge.right, "token_id": merge.token_id}
                for merge in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BytePairTokenizer:
        if payload.get("type") != "byte_bpe" or payload.get("version") != 1:
            raise ValueError("unsupported tokenizer artifact")
        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("merges must be a list")
        merges = []
        for item in raw_merges:
            if not isinstance(item, dict):
                raise ValueError("each merge must be an object")
            merges.append(Merge(int(item["left"]), int(item["right"]), int(item["token_id"])))
        return cls(merges)
