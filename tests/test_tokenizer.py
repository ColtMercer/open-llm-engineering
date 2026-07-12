from open_llm_lab.tokenizer import BytePairTokenizer


def test_byte_bpe_round_trip_and_compression() -> None:
    tokenizer = BytePairTokenizer.train(["banana bandana " * 20], vocab_size=280)
    text = "banana bandana 🌧️"
    encoded = tokenizer.encode(text)

    assert tokenizer.decode(encoded) == text
    assert len(encoded) < len(text.encode("utf-8"))


def test_tokenizer_artifact_round_trip() -> None:
    original = BytePairTokenizer.train(["repeat repeat repeat"], vocab_size=266)
    restored = BytePairTokenizer.from_dict(original.to_dict())

    assert restored.encode("repeat") == original.encode("repeat")
    assert restored.decode(restored.encode("café")) == "café"
