from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..project_config import load_project_config


@dataclass(frozen=True)
class VitsConfig:
    vocab_size: int
    num_languages: int = 7
    num_speakers: int = 1
    spec_channels: int = 513
    hidden_channels: int = 128
    latent_channels: int = 128
    conditioning_channels: int = 128
    language_embedding_channels: int = 32
    speaker_embedding_channels: int = 64
    text_encoder_layers: int = 4
    text_encoder_heads: int = 4
    flow_layers: int = 4
    decoder_initial_channels: int = 256
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    segment_frames: int = 32

    def __post_init__(self):
        if self.hidden_channels % self.text_encoder_heads:
            raise ValueError("hidden_channels must be divisible by text_encoder_heads")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("upsample rates and kernels must have equal length")
        if any(kernel < rate or (kernel - rate) % 2 for rate, kernel in zip(self.upsample_rates, self.upsample_kernel_sizes)):
            raise ValueError("each upsample kernel must produce an exact integer-length expansion")
        if min(self.vocab_size, self.num_languages, self.num_speakers) <= 0:
            raise ValueError("vocabulary, language and speaker counts must be positive")

    @property
    def hop_length(self) -> int:
        result = 1
        for rate in self.upsample_rates:
            result *= rate
        return result

    def to_dict(self) -> dict:
        return asdict(self)


def load_vits_config(path: str | Path, *, vocab_size: int | None = None) -> VitsConfig:
    raw = load_project_config(path)
    model = raw.get("model", raw)
    if vocab_size is not None:
        model["vocab_size"] = vocab_size
    for key in ("upsample_rates", "upsample_kernel_sizes"):
        if key in model:
            model[key] = tuple(model[key])
    return VitsConfig(**model)
