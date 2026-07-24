from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import VitsConfig
from .modules import (DurationPredictor, GlobalConditioning, PosteriorEncoder,
                      ResidualCouplingFlow, TextEncoder, WaveformDecoder,
                      duration_path, maximum_path, slice_latent,
                      slice_latent_at)


@dataclass
class VitsTrainingOutput:
    audio: torch.Tensor
    attention: torch.Tensor
    duration_loss: torch.Tensor
    latent: torch.Tensor
    latent_prior: torch.Tensor
    prior_mean: torch.Tensor
    prior_log_scale: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_log_scale: torch.Tensor
    audio_mask: torch.Tensor
    slice_starts: torch.Tensor


class MultilingualVITS(nn.Module):
    """Trainable multilingual, multi-speaker VITS generator.

    Speaker and language identities are separate conditions. Even a one-speaker
    first release keeps both pathways so later checkpoints remain extensible.
    """
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.conditioning = GlobalConditioning(
            config.num_languages, config.num_speakers,
            config.language_embedding_channels, config.speaker_embedding_channels,
            config.conditioning_channels,
        )
        self.text_encoder = TextEncoder(
            config.vocab_size, config.hidden_channels, config.latent_channels,
            config.conditioning_channels, config.text_encoder_layers, config.text_encoder_heads,
        )
        self.posterior_encoder = PosteriorEncoder(
            config.spec_channels, config.hidden_channels, config.latent_channels,
            config.conditioning_channels,
        )
        self.duration_predictor = DurationPredictor(config.hidden_channels, config.conditioning_channels)
        self.flow = ResidualCouplingFlow(
            config.latent_channels, config.hidden_channels, config.conditioning_channels, config.flow_layers,
        )
        self.decoder = WaveformDecoder(
            config.latent_channels, config.conditioning_channels, config.decoder_initial_channels,
            config.upsample_rates, config.upsample_kernel_sizes,
            config.decoder_resblock_kernel_sizes,
        )

    def forward(self, tokens: torch.Tensor, text_lengths: torch.Tensor, spectrogram: torch.Tensor,
                spec_lengths: torch.Tensor, language_ids: torch.Tensor,
                speaker_ids: torch.Tensor) -> VitsTrainingOutput:
        g = self.conditioning(language_ids, speaker_ids)
        text_hidden, text_mean, text_log_scale, text_mask = self.text_encoder(tokens, text_lengths, g)
        latent, posterior_mean, posterior_log_scale, audio_mask = self.posterior_encoder(spectrogram, spec_lengths, g)
        latent_prior, _ = self.flow(latent, audio_mask, g)

        with torch.no_grad():
            difference = latent_prior.unsqueeze(3) - text_mean.unsqueeze(2)
            inv_variance = torch.exp(-2.0 * text_log_scale).unsqueeze(2)
            scores = (-0.5 * difference.square() * inv_variance - text_log_scale.unsqueeze(2)).sum(1)
            attention = maximum_path(scores, text_lengths, spec_lengths)
            durations = attention.sum(1).unsqueeze(1)

        # MAS is a discrete search and must not be differentiated, but the
        # aligned text-prior statistics must remain in the autograd graph.
        # Detaching these projections leaves the text encoder completely
        # untrained: teacher-forced reconstruction can improve while
        # text-only inference remains random noise.
        expanded_mean = torch.matmul(attention, text_mean.transpose(1, 2)).transpose(1, 2)
        expanded_log_scale = torch.matmul(attention, text_log_scale.transpose(1, 2)).transpose(1, 2)

        predicted_log_duration = self.duration_predictor(text_hidden, text_mask, g)
        target_log_duration = torch.log(durations + 1e-6) * text_mask
        duration_loss = ((predicted_log_duration - target_log_duration).square() * text_mask).sum() / text_mask.sum()
        segment, starts = slice_latent(latent, spec_lengths, self.config.segment_frames)
        audio = self.decoder(segment, g)
        return VitsTrainingOutput(
            audio, attention, duration_loss, latent, latent_prior,
            expanded_mean, expanded_log_scale, posterior_mean, posterior_log_scale,
            audio_mask, starts,
        )

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, text_lengths: torch.Tensor, language_ids: torch.Tensor,
              speaker_ids: torch.Tensor, noise_scale: float = 0.667,
              length_scale: float = 1.0, max_frames: int = 4000):
        g = self.conditioning(language_ids, speaker_ids)
        text_hidden, mean, log_scale, text_mask = self.text_encoder(tokens, text_lengths, g)
        log_duration = self.duration_predictor(text_hidden, text_mask, g)
        durations = torch.ceil(torch.exp(log_duration) * text_mask * length_scale).long().clamp_min(0)
        frame_lengths = durations.sum((1, 2)).clamp_min(1).clamp_max(max_frames)
        frames = int(frame_lengths.max().item())
        attention = duration_path(durations, frames)
        expanded_mean = torch.matmul(attention, mean.transpose(1, 2)).transpose(1, 2)
        expanded_log_scale = torch.matmul(attention, log_scale.transpose(1, 2)).transpose(1, 2)
        audio_mask = (torch.arange(frames, device=tokens.device).unsqueeze(0) < frame_lengths.unsqueeze(1)).unsqueeze(1)
        audio_mask = audio_mask.to(expanded_mean.dtype)
        latent_prior = (expanded_mean + torch.randn_like(expanded_mean) * torch.exp(expanded_log_scale) * noise_scale) * audio_mask
        latent, _ = self.flow(latent_prior, audio_mask, g, reverse=True)
        return self.decoder(latent, g), frame_lengths, attention

    @torch.no_grad()
    def decode_aligned_prior(self, prior_mean: torch.Tensor, audio_mask: torch.Tensor,
                             language_ids: torch.Tensor, speaker_ids: torch.Tensor,
                             starts: torch.Tensor | None = None) -> torch.Tensor:
        """Decode the text prior under an oracle MAS alignment.

        This isolates text-prior/flow quality from duration prediction. It is
        used only for validation diagnostics and checkpoint selection.
        """
        g = self.conditioning(language_ids, speaker_ids)
        latent, _ = self.flow(prior_mean * audio_mask, audio_mask, g, reverse=True)
        if starts is not None:
            latent = slice_latent_at(latent, starts, self.config.segment_frames)
        return self.decoder(latent, g)

    def infer_deploy(self, tokens: torch.Tensor, text_lengths: torch.Tensor,
                     language_ids: torch.Tensor, speaker_ids: torch.Tensor,
                     scales: torch.Tensor, max_frames: int = 4000):
        """Tensor-only inference path suitable for ONNX export.

        scales follows Piper order: noise_scale, length_scale, duration_scale.
        Our deterministic duration predictor uses the third value as a smooth
        duration-logit scale; 1.0 preserves the trained prediction.
        """
        g = self.conditioning(language_ids, speaker_ids)
        text_hidden, mean, log_scale, text_mask = self.text_encoder(tokens, text_lengths, g)
        log_duration = self.duration_predictor(text_hidden, text_mask, g)
        durations = torch.ceil(
            torch.exp(log_duration * scales[2]) * text_mask * scales[1]
        ).to(torch.long).clamp_min(0)
        frame_lengths = durations.sum((1, 2)).clamp_min(1).clamp_max(max_frames)
        frames = frame_lengths.max()
        attention = duration_path(durations, frames)
        expanded_mean = torch.matmul(attention, mean.transpose(1, 2)).transpose(1, 2)
        expanded_log_scale = torch.matmul(attention, log_scale.transpose(1, 2)).transpose(1, 2)
        positions = torch.arange(frames, device=tokens.device).unsqueeze(0)
        audio_mask = (positions < frame_lengths.unsqueeze(1)).unsqueeze(1).to(expanded_mean.dtype)
        latent_prior = (
            expanded_mean + torch.randn_like(expanded_mean) * torch.exp(expanded_log_scale) * scales[0]
        ) * audio_mask
        latent, _ = self.flow(latent_prior, audio_mask, g, reverse=True)
        return self.decoder(latent, g)
