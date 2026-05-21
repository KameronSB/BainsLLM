from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    n_embd: int
    dropout: float
    use_rope: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    eval_interval: int
    eval_steps: int
    grad_clip: float
    device: str = "cuda"


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    tokenizer: str
    train_split: float = 0.9


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig


def load_config(path: str | Path) -> Config:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**raw["data"]),
    )