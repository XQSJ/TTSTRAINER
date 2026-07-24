from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sequence_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = lengths.max()
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class SelfAttentionBlock(nn.Module):
    """Transformer block implemented with dynamic-shape ONNX primitives."""
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_channels = channels // heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels, channels)
        self.norm_attention = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels),
        )
        self.norm_feed_forward = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_channels)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_channels)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, channels)
        x = self.norm_attention(x + self.output(context))
        x = self.norm_feed_forward(x + self.feed_forward(x))
        return x * valid_mask.unsqueeze(-1).to(x.dtype)


class GlobalConditioning(nn.Module):
    def __init__(self, num_languages: int, num_speakers: int, language_channels: int,
                 speaker_channels: int, output_channels: int):
        super().__init__()
        self.language_embedding = nn.Embedding(num_languages, language_channels)
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_channels)
        self.projection = nn.Sequential(
            nn.Linear(language_channels + speaker_channels, output_channels),
            nn.SiLU(),
            nn.Linear(output_channels, output_channels),
        )

    def forward(self, language_ids: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        condition = torch.cat((self.language_embedding(language_ids), self.speaker_embedding(speaker_ids)), dim=-1)
        return self.projection(condition).unsqueeze(-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_channels: int, latent_channels: int,
                 condition_channels: int, layers: int, heads: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_channels, padding_idx=0)
        # VITS scales embeddings by sqrt(hidden), so their initialization must
        # use hidden**-0.5. PyTorch's default N(0, 1) initialization would make
        # the first attention layer see values roughly sqrt(hidden) too large.
        nn.init.normal_(self.embedding.weight, 0.0, hidden_channels ** -0.5)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        self.condition = nn.Linear(condition_channels, hidden_channels)
        self.encoder = nn.ModuleList(SelfAttentionBlock(hidden_channels, heads) for _ in range(layers))
        self.projection = nn.Conv1d(hidden_channels, latent_channels * 2, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, g: torch.Tensor):
        mask_bool = sequence_mask(lengths, tokens.shape[1])
        x = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
        x = x + sinusoidal_position_encoding(
            tokens.shape[1], self.embedding.embedding_dim,
            device=x.device, dtype=x.dtype,
        ).unsqueeze(0)
        x = x + self.condition(g.squeeze(-1)).unsqueeze(1)
        for layer in self.encoder:
            x = layer(x, mask_bool)
        x = x.transpose(1, 2)
        mask = mask_bool.unsqueeze(1).to(x.dtype)
        stats = self.projection(x) * mask
        mean, log_scale = stats.chunk(2, dim=1)
        return x * mask, mean, log_scale.clamp(-7.0, 2.0), mask


class ConvStack(nn.Module):
    def __init__(self, channels: int, condition_channels: int, layers: int = 4):
        super().__init__()
        self.condition = nn.Conv1d(condition_channels, channels, 1)
        self.layers = nn.ModuleList()
        for index in range(layers):
            dilation = 2 ** index
            self.layers.append(nn.Conv1d(channels, channels * 2, 3, padding=dilation, dilation=dilation))
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = x + self.condition(g)
        for layer in self.layers:
            gate, filter_ = layer(x * mask).chunk(2, dim=1)
            x = x + torch.sigmoid(gate) * torch.tanh(filter_)
        return self.output(x * mask) * mask


class PosteriorEncoder(nn.Module):
    def __init__(self, spec_channels: int, hidden_channels: int, latent_channels: int, condition_channels: int):
        super().__init__()
        self.pre = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.stack = ConvStack(hidden_channels, condition_channels, 6)
        self.projection = nn.Conv1d(hidden_channels, latent_channels * 2, 1)

    def forward(self, spectrogram: torch.Tensor, lengths: torch.Tensor, g: torch.Tensor):
        mask = sequence_mask(lengths, spectrogram.shape[2]).unsqueeze(1).to(spectrogram.dtype)
        hidden = self.stack(self.pre(spectrogram) * mask, mask, g)
        mean, log_scale = self.projection(hidden).chunk(2, dim=1)
        log_scale = log_scale.clamp(-7.0, 2.0)
        latent = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * mask
        return latent, mean * mask, log_scale * mask, mask


class DurationPredictor(nn.Module):
    def __init__(self, hidden_channels: int, condition_channels: int):
        super().__init__()
        self.condition = nn.Conv1d(condition_channels, hidden_channels, 1)
        self.convs = nn.ModuleList((
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
        ))
        self.norms = nn.ModuleList((nn.LayerNorm(hidden_channels), nn.LayerNorm(hidden_channels)))
        self.projection = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        hidden = x.detach() + self.condition(g.detach())
        for conv, norm in zip(self.convs, self.norms):
            hidden = conv(hidden * mask).transpose(1, 2)
            hidden = F.silu(norm(hidden)).transpose(1, 2)
        return self.projection(hidden * mask) * mask


def sinusoidal_position_encoding(length: int, channels: int, *, device, dtype) -> torch.Tensor:
    """Return an ONNX-friendly absolute position signal [length, channels]."""
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    pair_count = (channels + 1) // 2
    frequencies = torch.exp(
        torch.arange(pair_count, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(pair_count - 1, 1))
    ).unsqueeze(0)
    angles = positions * frequencies
    encoded = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(1)
    return encoded[:, :channels].to(dtype=dtype)


class AdditiveCoupling(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, condition_channels: int):
        super().__init__()
        if channels % 2:
            raise ValueError("flow channels must be even")
        half = channels // 2
        self.pre = nn.Conv1d(half, hidden_channels, 1)
        self.stack = ConvStack(hidden_channels, condition_channels, 4)
        # The VITS KL loss used by this project assumes a volume-preserving
        # flow. Match the reference mean-only residual coupling instead of
        # learning an affine scale whose log-determinant is never optimized.
        self.projection = nn.Conv1d(hidden_channels, half, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor, reverse: bool = False):
        first, second = x.chunk(2, dim=1)
        mean = self.projection(self.stack(self.pre(first) * mask, mask, g)) * mask
        if reverse:
            second = (second - mean) * mask
            logdet = None
        else:
            second = (mean + second) * mask
            logdet = x.new_zeros(x.shape[0])
        return torch.cat((first, second), dim=1), logdet


class ResidualCouplingFlow(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, condition_channels: int, layers: int):
        super().__init__()
        self.flows = nn.ModuleList(
            AdditiveCoupling(channels, hidden_channels, condition_channels)
            for _ in range(layers)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor, reverse: bool = False):
        flows = reversed(self.flows) if reverse else self.flows
        total_logdet = None if reverse else x.new_zeros(x.shape[0])
        for flow in flows:
            if reverse:
                x = torch.flip(x, (1,))
                x, _ = flow(x, mask, g, reverse=True)
            else:
                x, logdet = flow(x, mask, g)
                total_logdet = total_logdet + logdet
                x = torch.flip(x, (1,))
        return x * mask, total_logdet


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in (1, 3, 5):
            padding = (kernel_size * dilation - dilation) // 2
            self.convs.append(nn.Conv1d(
                channels, channels, kernel_size, padding=padding, dilation=dilation,
            ))
            self.convs.append(nn.Conv1d(
                channels, channels, kernel_size, padding=(kernel_size - 1) // 2,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for first, second in zip(self.convs[::2], self.convs[1::2]):
            residual = F.leaky_relu(x, 0.1)
            residual = second(F.leaky_relu(first(residual), 0.1))
            x = x + residual
        return x


class WaveformDecoder(nn.Module):
    def __init__(self, latent_channels: int, condition_channels: int, initial_channels: int,
                 upsample_rates: tuple[int, ...], upsample_kernels: tuple[int, ...],
                 resblock_kernels: tuple[int, ...] = (3,)):
        super().__init__()
        self.pre = nn.Conv1d(latent_channels, initial_channels, 7, padding=3)
        self.condition = nn.Conv1d(condition_channels, initial_channels, 1)
        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = initial_channels
        for rate, kernel in zip(upsample_rates, upsample_kernels):
            next_channels = channels // 2
            self.upsamples.append(nn.ConvTranspose1d(channels, next_channels, kernel, stride=rate,
                                                     padding=(kernel - rate) // 2))
            if len(resblock_kernels) == 1:
                # Preserve compact-model checkpoint keys created before quality presets existed.
                self.resblocks.append(ResBlock(next_channels, resblock_kernels[0]))
            else:
                self.resblocks.append(nn.ModuleList(
                    ResBlock(next_channels, kernel) for kernel in resblock_kernels
                ))
            channels = next_channels
        self.post = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, latent: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        hidden = self.pre(latent) + self.condition(g)
        for upsample, resblock in zip(self.upsamples, self.resblocks):
            hidden = upsample(F.leaky_relu(hidden, 0.1))
            if isinstance(resblock, nn.ModuleList):
                hidden = sum(block(hidden) for block in resblock) / len(resblock)
            else:
                hidden = resblock(hidden)
        return torch.tanh(self.post(F.leaky_relu(hidden, 0.1)))


@torch.no_grad()
def maximum_path(value: torch.Tensor, text_lengths: torch.Tensor, spec_lengths: torch.Tensor) -> torch.Tensor:
    """Monotonic alignment DP. `value` is [B, T_audio, T_text]."""
    batch_size, max_audio, max_text = value.shape
    negative = value.new_tensor(torch.finfo(value.dtype).min)
    if bool(torch.any(spec_lengths < text_lengths)):
        failed = int(torch.nonzero(spec_lengths < text_lengths, as_tuple=False)[0, 0])
        raise ValueError(
            f"audio frames ({int(spec_lengths[failed])}) must be >= "
            f"text tokens ({int(text_lengths[failed])}) for MAS"
        )

    # The old implementation launched one tiny GPU operation for every
    # audio-frame/text-token cell. Vectorize the text dimension and keep only
    # the unavoidable audio-frame recurrence on device.
    scores = value.new_full((batch_size, max_audio, max_text), negative)
    decisions = torch.zeros(
        (batch_size, max_audio, max_text), dtype=torch.bool, device=value.device,
    )
    scores[:, 0, 0] = value[:, 0, 0]
    text_positions = torch.arange(max_text, device=value.device).unsqueeze(0)
    for audio_index in range(1, max_audio):
        stay = scores[:, audio_index - 1]
        move = F.pad(stay[:, :-1], (1, 0), value=float(negative))
        choose_move = move >= stay
        candidate = value[:, audio_index] + torch.maximum(stay, move)
        valid = (
            (audio_index < spec_lengths.unsqueeze(1))
            & (text_positions < text_lengths.unsqueeze(1))
            & (text_positions <= audio_index)
            & (
                text_positions
                >= text_lengths.unsqueeze(1)
                - (spec_lengths.unsqueeze(1) - audio_index)
            )
        )
        scores[:, audio_index] = torch.where(valid, candidate, negative)
        decisions[:, audio_index] = choose_move & valid

    # One device-to-host transfer replaces thousands of scalar synchronizations.
    decisions_cpu = decisions.cpu()
    path_cpu = torch.zeros(
        (batch_size, max_audio, max_text), dtype=value.dtype, device="cpu",
    )
    for batch in range(batch_size):
        text_index = int(text_lengths[batch]) - 1
        for audio_index in range(int(spec_lengths[batch]) - 1, -1, -1):
            path_cpu[batch, audio_index, text_index] = 1
            if text_index and audio_index and decisions_cpu[batch, audio_index, text_index]:
                text_index -= 1
    return path_cpu.to(value.device)


@torch.no_grad()
def duration_path(durations: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Convert integer durations [B, 1, T_text] to [B, T_audio, T_text]."""
    cumulative = durations.cumsum(dim=2)
    positions = torch.arange(max_frames, device=durations.device).view(1, 1, 1, -1)
    cumulative_mask = positions < cumulative.unsqueeze(-1)
    previous = F.pad(cumulative_mask[:, :, :-1], (0, 0, 1, 0))
    return (cumulative_mask & ~previous).squeeze(1).transpose(1, 2).to(torch.float32)


def slice_latent(latent: torch.Tensor, lengths: torch.Tensor, segment_frames: int):
    if segment_frames <= 0:
        return latent, latent.new_zeros(latent.shape[0], dtype=torch.long)
    starts = []
    for batch in range(latent.shape[0]):
        max_start = max(int(lengths[batch]) - segment_frames, 0)
        start = int(torch.randint(max_start + 1, (1,), device=latent.device).item()) if max_start else 0
        starts.append(start)
    starts_tensor = torch.tensor(starts, device=latent.device)
    return slice_latent_at(latent, starts_tensor, segment_frames), starts_tensor


def slice_latent_at(latent: torch.Tensor, starts: torch.Tensor,
                    segment_frames: int) -> torch.Tensor:
    slices = []
    for batch, start_value in enumerate(starts):
        start = int(start_value.item())
        segment = latent[batch:batch + 1, :, start:start + segment_frames]
        slices.append(F.pad(segment, (0, segment_frames - segment.shape[-1])))
    return torch.cat(slices)
