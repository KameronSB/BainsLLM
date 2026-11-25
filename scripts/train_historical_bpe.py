import json
from pathlib import Path

from bainsllm.config import load_config
from bainsllm.data.byte_bpe import ByteBPETokenizer


def documents(path):
    with path.open(encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            yield row["document_id"], row["text"]


def main():
    cfg = load_config("configs/tiny.yaml").data
    if cfg.tokenizer != "byte_bpe":
        raise ValueError("This script expects data.tokenizer to be byte_bpe")

    data_dir = Path("data/raw") / cfg.dataset_name
    train_texts = (text for _, text in documents(data_dir / "train.jsonl"))
    tokenizer = ByteBPETokenizer.train(train_texts, vocab_size=cfg.vocab_size)

    output = (
        Path("data/processed")
        / cfg.dataset_name
        / f"byte_bpe-{cfg.vocab_size}"
        / "tokenizer.json"
    )
    tokenizer.save(output)

    saved_tokenizer = ByteBPETokenizer.load(output)
    books = 0
    tokens = 0

    for doc_id, text in documents(data_dir / "validation.jsonl"):
        ids = saved_tokenizer.encode(text)
        if saved_tokenizer.decode(ids) != text:
            raise ValueError(f"Round-trip failed for {doc_id}")
        books += 1
        tokens += len(ids)

    print(f"Saved: {output}")
    print(f"Vocabulary: {saved_tokenizer.vocab_size:,} IDs")
    print(f"Validation: {books} books, {tokens:,} text tokens")


if __name__ == "__main__":
    main()