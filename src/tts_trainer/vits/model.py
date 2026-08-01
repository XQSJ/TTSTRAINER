from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import VitsConfig
from .modules import (GlobalConditioning, PosteriorEncoder,
                      FlowStochasticDurationPredictor, ResidualCouplingFlow,
                      StochasticDurationPredictor,
                      TextEncoder, WaveformDecoder, duration_path,
                      maximum_path, slice_latent, slice_latent_at)


def integer_durations(
    log_duration: torch.Tensor,
    text_mask: torch.Tensor,
    length_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Convert predicted durations without systematically lengthening speech.

    ``ceil`` turns a prediction just above one frame into two frames. Mobile
    Piper sequences contain many one-frame blank/phoneme tokens, so that bias
    can almost double a sentence. Nearest-integer conversion preserves the
    learned MAS duration and still assigns every valid token at least one frame.
    """
    valid = text_mask.to(torch.bool)
    rounded = torch.round(torch.exp(log_duration) * length_scale).to(torch.long)
    return torch.where(valid, rounded.clamp_min(1), torch.zeros_like(rounded))


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


@dataclass
class VitsRefinementOutput:
    """冻结声码器时的文本先验训练输出。 / Text-prior refinement outputs."""

    audio: torch.Tensor
    duration_loss: torch.Tensor
    duration_mean_loss: torch.Tensor
    latent_prior: torch.Tensor
    prior_mean: torch.Tensor
    prior_log_scale: torch.Tensor
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
        if config.duration_predictor_type == "stochastic_lognormal":
            self.duration_predictor = StochasticDurationPredictor(
                config.hidden_channels, config.conditioning_channels,
            )
        else:
            self.duration_predictor = FlowStochasticDurationPredictor(
                config.hidden_channels, config.conditioning_channels,
                config.duration_predictor_channels,
                config.duration_predictor_flow_layers,
            )
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

        duration_loss = self.duration_predictor.loss(
            text_hidden, text_mask, g, durations,
        )
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
              length_scale: float = 1.0, duration_noise_scale: float = 0.35,
              max_frames: int = 4000):
        g = self.conditioning(language_ids, speaker_ids)
        text_hidden, mean, log_scale, text_mask = self.text_encoder(tokens, text_lengths, g)
        log_duration = self.duration_predictor.sample(
            text_hidden, text_mask, g, duration_noise_scale,
        )
        durations = integer_durations(log_duration, text_mask, length_scale)
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
        """在 MAS 真值对齐下解码文本先验。 / Decode with oracle MAS alignment.

        仅用于隔离文本先验/Flow 与时长预测问题，不参与反向传播。
        This isolates text-prior/flow quality from duration prediction and is
        validation-only: it never participates in backpropagation.
        """
        g = self.conditioning(language_ids, speaker_ids)
        latent, _ = self.flow(prior_mean * audio_mask, audio_mask, g, reverse=True)
        if starts is not None:
            latent = slice_latent_at(latent, starts, self.config.segment_frames)
        return self.decoder(latent, g)

    def decode_posterior(self, latent: torch.Tensor, audio_mask: torch.Tensor,
                         language_ids: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        """解码完整 posterior 供诊断。 / Decode a full posterior for diagnostics."""
        g = self.conditioning(language_ids, speaker_ids)
        return self.decoder(latent * audio_mask, g)

    def forward_text_refinement(
        self, tokens: torch.Tensor, text_lengths: torch.Tensor,
        spectrogram: torch.Tensor, spec_lengths: torch.Tensor,
        language_ids: torch.Tensor, speaker_ids: torch.Tensor,
    ) -> VitsRefinementOutput:
        """用固定声学主链训练文本先验。 / Refine text prior against frozen acoustics.

        调用方必须冻结 conditioning、posterior_encoder 和 decoder。Decoder 仍参与
        可微分前向传播，使 Mel 梯度只流向 Text Encoder 与逆 Flow。
        The caller must freeze conditioning, posterior_encoder, and decoder.
        Decoder operations stay differentiable with respect to their input, so
        Mel gradients reach only the text encoder and inverse flow.
        """
        frozen = (self.conditioning, self.posterior_encoder, self.decoder)
        if any(parameter.requires_grad for module in frozen for parameter in module.parameters()):
            raise RuntimeError(
                "text-prior refinement requires frozen conditioning, "
                "posterior_encoder, and decoder"
            )
        with torch.no_grad():
            g = self.conditioning(language_ids, speaker_ids)
            latent, _, posterior_log_scale, audio_mask = self.posterior_encoder(
                spectrogram, spec_lengths, g,
            )
        text_hidden, text_mean, text_log_scale, text_mask = self.text_encoder(
            tokens, text_lengths, g,
        )
        latent_prior, _ = self.flow(latent, audio_mask, g)
        with torch.no_grad():
            difference = latent_prior.detach().unsqueeze(3) - text_mean.detach().unsqueeze(2)
            inv_variance = torch.exp(-2.0 * text_log_scale.detach()).unsqueeze(2)
            scores = (
                -0.5 * difference.square() * inv_variance
                - text_log_scale.detach().unsqueeze(2)
            ).sum(1)
            attention = maximum_path(scores, text_lengths, spec_lengths)
            durations = attention.sum(1).unsqueeze(1)
        expanded_mean = torch.matmul(
            attention, text_mean.transpose(1, 2),
        ).transpose(1, 2)
        expanded_log_scale = torch.matmul(
            attention, text_log_scale.transpose(1, 2),
        ).transpose(1, 2)
        duration_loss = self.duration_predictor.loss(
            text_hidden, text_mask, g, durations,
        )
        duration_mean_loss = self.duration_predictor.mean_loss(
            text_hidden, text_mask, g, durations,
        )
        _, starts = slice_latent(
            latent, spec_lengths, self.config.segment_frames,
        )
        aligned_latent, _ = self.flow(
            expanded_mean * audio_mask, audio_mask, g, reverse=True,
        )
        aligned_latent = slice_latent_at(
            aligned_latent, starts, self.config.segment_frames,
        )
        audio = self.decoder(aligned_latent, g)
        return VitsRefinementOutput(
            audio=audio,
            duration_loss=duration_loss,
            duration_mean_loss=duration_mean_loss,
            latent_prior=latent_prior,
            prior_mean=expanded_mean,
            prior_log_scale=expanded_log_scale,
            posterior_log_scale=posterior_log_scale,
            audio_mask=audio_mask,
            slice_starts=starts,
        )

    def infer_deploy(self, tokens: torch.Tensor, text_lengths: torch.Tensor,
                     language_ids: torch.Tensor, speaker_ids: torch.Tensor,
                     scales: torch.Tensor, max_frames: int = 4000):
        """Tensor-only inference path suitable for ONNX export.

        scales follows Piper order: noise_scale, length_scale,
        duration_noise_scale. The third value controls stochastic timing:
        0.0 is deterministic and larger values add duration variation.
        """
        g = self.conditioning(language_ids, speaker_ids)
        text_hidden, mean, log_scale, text_mask = self.text_encoder(tokens, text_lengths, g)
        log_duration = self.duration_predictor.sample(
            text_hidden, text_mask, g, scales[2],
        )
        durations = integer_durations(log_duration, text_mask, scales[1])
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
