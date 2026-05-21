from __future__ import annotations

from dataclasses import dataclass

import torch

from bainsllm.data.char_tokenizer import CharTokenizer


@dataclass
class TokenDataset:
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    tokenizer: CharTokenizer

    @classmethod
    def from_text(
        cls,
        text: str,
        train_split: float = 0.9,
    ) -> "TokenDataset":
        if not 0.0 < train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1.")

        tokenizer = CharTokenizer.from_text(text)
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        split_idx = int(len(ids) * train_split)

        train_ids = ids[:split_idx]
        val_ids = ids[split_idx:]

        if len(train_ids) == 0:
            raise ValueError("Training split is empty.")

        if len(val_ids) == 0:
            raise ValueError("Validation split is empty.")

        return cls(
            train_ids=train_ids,
            val_ids=val_ids,
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