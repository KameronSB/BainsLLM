from __future__ import annotations

from bainsllm.data.char_tokenizer import CharTokenizer


def test_char_tokenizer_roundtrip() -> None:
    text = "hello world"
    tokenizer = CharTokenizer.from_text(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_char_tokenizer_vocab_size() -> None:
    text = "banana"
    tokenizer = CharTokenizer.from_text(text)

    assert tokenizer.vocab_size == len(set(text))