"""MultilingualVITS 模型的结构与超参数配置。 / Structure and hyperparameters for MultilingualVITS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..project_config import load_project_config


@dataclass(frozen=True)
class VitsConfig:
    """不可变 VITS 模型配置，字段与序列化 checkpoint 一一对应。 / Immutable VITS model config mirrored in checkpoints."""
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
    # Missing in format-4 checkpoints means the original log-normal predictor.
    # format-4 未包含该字段时，自动使用原有的对数正态时长预测器。
    duration_predictor_type: str = "stochastic_lognormal"
    duration_predictor_channels: int = 64
    duration_predictor_flow_layers: int = 2
    decoder_initial_channels: int = 256
    decoder_resblock_kernel_sizes: tuple[int, ...] = (3,)
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    segment_frames: int = 32

    def __post_init__(self):
        """校验字段合法性，防止延迟到建模期才失败。 / Validate fields eagerly at construction."""
        duration_types = {
            "stochastic_lognormal", "stochastic_mobile", "stochastic_quality",
        }
        if self.duration_predictor_type not in duration_types:
            raise ValueError(
                "duration_predictor_type must be stochastic_lognormal, "
                "stochastic_mobile, or stochastic_quality"
            )
        if self.duration_predictor_channels <= 0:
            raise ValueError("duration_predictor_channels must be positive")
        if self.duration_predictor_flow_layers < 2:
            raise ValueError("duration_predictor_flow_layers must be at least 2")
        if self.hidden_channels % self.text_encoder_heads:
            raise ValueError("hidden_channels must be divisible by text_encoder_heads")
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("upsample rates and kernels must have equal length")
        if not self.decoder_resblock_kernel_sizes or any(
            kernel < 3 or kernel % 2 == 0 for kernel in self.decoder_resblock_kernel_sizes
        ):
            raise ValueError("decoder resblock kernels must be non-empty odd integers >= 3")
        if any(kernel < rate or (kernel - rate) % 2 for rate, kernel in zip(self.upsample_rates, self.upsample_kernel_sizes)):
            raise ValueError("each upsample kernel must produce an exact integer-length expansion")
        if min(self.vocab_size, self.num_languages, self.num_speakers) <= 0:
            raise ValueError("vocabulary, language and speaker counts must be positive")

    @property
    def hop_length(self) -> int:
        """帧移 = 所有上采样率之积，须与音频前端保持一致。 / Hop length equals the product of upsample rates."""
        result = 1
        for rate in self.upsample_rates:
            result *= rate
        return result

    def to_dict(self) -> dict:
        """序列化为可写入 checkpoint 的字典。 / Serialize into a checkpoint-safe dict."""
        return asdict(self)


def load_vits_config(path: str | Path, *, vocab_size: int | None = None) -> VitsConfig:
    """从项目 YAML 读取并构造 VitsConfig。 / Build a VitsConfig from project YAML."""
    raw = load_project_config(path)
    # 兼容顶层或 model: 子节两种写法 / Accept either flat YAML or a nested model: section
    model = raw.get("model", raw)
    if vocab_size is not None:
        # 运行期词表可能与配置中不一致，以实际为准 / Actual vocabulary overrides the file value
        model["vocab_size"] = vocab_size
    # YAML 列表需转回元组以匹配 dataclass 类型 / YAML lists must become tuples to match field types
    for key in ("decoder_resblock_kernel_sizes", "upsample_rates", "upsample_kernel_sizes"):
        if key in model:
            model[key] = tuple(model[key])
    return VitsConfig(**model)
