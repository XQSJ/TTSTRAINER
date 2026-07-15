from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    metadata: str
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    batch_size: int = 8
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 192
    language_embedding_size: int = 32
    encoder_layers: int = 4
    encoder_heads: int = 4
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 2e-4
    seed: int = 1337
    output_dir: str = "runs/baseline"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    config = Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
    if config.model.hidden_size % config.model.encoder_heads:
        raise ValueError("model.hidden_size must be divisible by model.encoder_heads")
    if config.data.sample_rate <= 0 or config.data.n_mels <= 0:
        raise ValueError("sample_rate and n_mels must be positive")
    return config
