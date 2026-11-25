from bainsllm.data.byte_bpe import ByteBPETokenizer, merge_pair


def test_banana():
    tok = ByteBPETokenizer.train(["banana banana"], vocab_size=260)

    assert tok.merges == ((97, 110), (98, 257), (257, 97))
    assert tok.encode("banana banana") == [258, 259, 32, 258, 259]
    assert tok.decode([258, 259]) == "banana"

    # Three adjacent a's allow one non-overlapping replacement.
    assert merge_pair([97, 97, 97], (97, 97), 257) == [257, 97]


def test_round_trip_and_reload(tmp_path):
    tok = ByteBPETokenizer.train(["banana banana"], vocab_size=260)

    path = tmp_path / "tokenizer.json"
    tok.save(path)
    restored = ByteBPETokenizer.load(path)

    text = "Ç — £2\n\nbanana 🐦"
    ids = restored.encode(text)

    assert ids == tok.encode(text)
    assert restored.decode(ids + [restored.eos_id]) == text