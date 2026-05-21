from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """
    Simple character-level tokenizer.

    This is the baseline tokenizer for early model testing.
    It maps every unique character in the training text to an integer ID.
    """

    def __init__(self, chars: list[str]) -> None:
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"Character {missing!r} is not in the tokenizer vocabulary.") from exc

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"Token ID {missing!r} is not in the tokenizer vocabulary.") from exc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "type": "char",
            "chars": self.chars,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("type") != "char":
            raise ValueError(f"Expected tokenizer type 'char', got {payload.get('type')!r}")

        return cls(payload["chars"])