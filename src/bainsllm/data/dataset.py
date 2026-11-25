from __future__ import annotations

from dataclasses import dataclass

import torch

from bainsllm.data.char_tokenizer import CharTokenizer
from bainsllm.data.byte_bpe import ByteBPETokenizer


@dataclass
class TokenDataset:
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    tokenizer: CharTokenizer | ByteBPETokenizer

    @classmethod
    def from_text(
        cls,
        text: str,
        train_split: float = 0.9,
        *,
        kind: str = "byte_bpe",
        vocab_size: int = 2000,
    ) -> "TokenDataset":
        if not 0.0 < train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")

        cut = int(len(text) * train_split)
        if cut == 0 or cut == len(text):
            raise ValueError("Both splits need some text")

        train_text = text[:cut]
        val_text = text[cut:]

        if kind == "byte_bpe":
            tokenizer = ByteBPETokenizer.train([train_text], vocab_size)
        elif kind == "char":
            tokenizer = CharTokenizer.from_text(train_text)
        else:
            raise ValueError(f"Unknown tokenizer: {kind}")

        return cls(
            train_ids=torch.tensor(tokenizer.encode(train_text), dtype=torch.long),
            val_ids=torch.tensor(tokenizer.encode(val_text), dtype=torch.long),
            tokenizer=tokenizer,
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_ids
        elif split == "val":
            data = self.val_ids
        else:
            raise ValueError("split must be either 'train' or 'val'.")

        if len(data) <= block_size:
            raise ValueError(
                f"Dataset split is too small for block_size={block_size}. "
                f"Split length is {len(data)}."
            )

        starts = torch.randint(
            low=0,
            high=len(data) - block_size,
            size=(batch_size,),
        )

        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])

        return x.to(device), y.to(device)