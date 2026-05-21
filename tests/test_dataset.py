from __future__ import annotations

import torch

from bainsllm.data.dataset import TokenDataset


def test_dataset_batch_shapes() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    dataset = TokenDataset.from_text(text, train_split=0.9)

    batch_size = 4
    block_size = 8

    x, y = dataset.get_batch(
        split="train",
        batch_size=batch_size,
        block_size=block_size,
    )

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_targets_are_shifted_by_one() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    dataset = TokenDataset.from_text(text, train_split=0.9)

    x, y = dataset.get_batch(
        split="train",
        batch_size=16,
        block_size=8,
    )

    assert torch.equal(y[:, :-1], x[:, 1:])