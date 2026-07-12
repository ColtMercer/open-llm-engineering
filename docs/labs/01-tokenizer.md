# Lab 1: train a byte-level BPE tokenizer

This lab starts with 256 byte IDs, counts adjacent pairs, learns deterministic merges, encodes a string, and verifies exact decoding.

```bash
python labs/01_tokenizer.py --text 'naïve 🌧️ network'
```

Inspect:

- UTF-8 byte count versus token count;
- learned IDs at or above 256;
- pieces that are not independently valid Unicode;
- exact round trip;
- how the result changes with `--vocab-size`.

## Trace the implementation

Read `BytePairTokenizer.train` first, then `_merge_pair`, then `encode_bytes`. Notice that records remain separate during pair counting: the lab never learns a token spanning two documents.

## Experiments

1. Train without repeating the query text. Which pieces disappear?
2. Compare English, Arabic, Hindi, emoji, Python, and a UUID.
3. Serialize with `to_dict`, reconstruct with `from_dict`, and assert identical IDs.
4. Change the tie rule and observe why the artifact—not just “BPE”—defines compatibility.

The implementation is \(O(\text{merges}\times\text{corpus length})\) or worse in practice. Production libraries use efficient representations and parallel preprocessing.

