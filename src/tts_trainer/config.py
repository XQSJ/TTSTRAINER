"""基线训练的 JSON 配置加载与校验。 / JSON config loading and validation for baseline training."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """数据加载与音频特征参数。 / Data loading and audio feature parameters."""
    metadata: str
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    batch_size: int = 8
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """声学模型结构超参。 / Acoustic model architecture hyperparameters."""
    hidden_size: int = 192
    language_embedding_size: int = 32
    encoder_layers: int = 4
    encoder_heads: int = 4
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """训练循环超参。 / Training loop hyperparameters."""
    epochs: int = 100
    learning_rate: float = 2e-4
    seed: int = 1337
    output_dir: str = "runs/baseline"


@dataclass(frozen=True)
class Config:
    """三段式顶层配置。 / Top-level three-section config."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    """读取并校验 JSON 训练配置。 / Read and validate a JSON training config."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    config = Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
    # 注意力头必须整除隐藏维度，否则 Transformer 初始化会失败。 / Heads must divide hidden size or the Transformer fails to init.
    if config.model.hidden_size % config.model.encoder_heads:
        raise ValueError("model.hidden_size must be divisible by model.encoder_heads")
    if config.data.sample_rate <= 0 or config.data.n_mels <= 0:
        raise ValueError("sample_rate and n_mels must be positive")
    return config
