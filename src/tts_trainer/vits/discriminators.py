"""MPD/MSD 风格的 GAN 判别器组合。 / MPD/MSD-style GAN discriminators."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PeriodDiscriminator(nn.Module):
    """多周期判别器：把波形重排为二维后捕捉周期伪影。 / Multi-period discriminator."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        channels = (1, 32, 128, 256, 512)
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], (5, 1), (3, 1), padding=(2, 0))
            for i in range(len(channels) - 1)
        )
        self.post = nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0))

    def forward(self, audio: torch.Tensor):
        """返回逐样本得分与各层中间特征。 / Return per-sample score plus layer features."""
        remainder = audio.shape[-1] % self.period
        if remainder:
            # 补齐到 period 的整数倍，reshape 才不丢样本 / Pad to a whole period so reshape loses nothing
            audio = F.pad(audio, (0, self.period - remainder), mode="reflect")
        # (B, T) -> (B, 1, T/period, period)，二维卷积即可检测基频伪影 / Reshape exposes periodic artifacts to Conv2d
        hidden = audio.view(audio.shape[0], 1, -1, self.period)
        features = []
        for conv in self.convs:
            hidden = F.leaky_relu(conv(hidden), 0.1); features.append(hidden)
        score = self.post(hidden); features.append(score)
        return score.flatten(1), features


class ScaleDiscriminator(nn.Module):
    """多尺度判别器：直接在时域逐层下采样打分。 / Scale discriminator over raw waveform."""

    def __init__(self):
        super().__init__()
        channels = (1, 32, 128, 256, 512, 512)
        self.convs = nn.ModuleList()
        for index in range(len(channels) - 1):
            self.convs.append(nn.Conv1d(channels[index], channels[index + 1], 15, stride=2, padding=7))
        self.post = nn.Conv1d(channels[-1], 1, 3, padding=1)

    def forward(self, audio: torch.Tensor):
        hidden = audio; features = []
        for conv in self.convs:
            hidden = F.leaky_relu(conv(hidden), 0.1); features.append(hidden)
        score = self.post(hidden); features.append(score)
        return score.flatten(1), features


class VitsDiscriminator(nn.Module):
    """MSD + 多个 MPD 的组合判别器。 / Combined MSD plus multiple MPD heads."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([ScaleDiscriminator(), *(PeriodDiscriminator(p) for p in periods)])

    def forward(self, audio: torch.Tensor):
        return [discriminator(audio) for discriminator in self.discriminators]
