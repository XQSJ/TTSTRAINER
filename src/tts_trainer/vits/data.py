"""VITS 数据集、采样与批组装。 / Datasets, sampling and batch assembly for VITS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from torch.nn import functional as F

from ..manifest import Item
from ..text import Vocabulary


@dataclass(frozen=True)
class AudioConfig:
    """音频前端参数（STFT/Mel），决定帧与样本的换算。 / Audio frontend parameters mapping frames to samples."""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    mel_power: float = 2.0


class LengthBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """按长度分桶组批以减少 padding。 / Bucket similarly sized audio into batches to cut padding.

    Keep weighted sampling while grouping similarly sized audio.
    """

    def __init__(self, weights, lengths, batch_size: int, *, pool_batches: int = 20):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        self.pool_batches = int(pool_batches)
        if len(self.weights) != len(self.lengths) or not self.lengths:
            raise ValueError("weights and lengths must have the same non-zero length")
        if self.batch_size <= 0 or self.pool_batches <= 0:
            raise ValueError("batch_size and pool_batches must be positive")

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Sampling first preserves the existing language/speaker probability
        # mass. Local sorting reduces padding without globally ordering epochs.
        sampled = torch.multinomial(
            self.weights, len(self.lengths), replacement=True,
        ).tolist()
        pool_size = self.batch_size * self.pool_batches
        batches = []
        for start in range(0, len(sampled), pool_size):
            # 池内按音频长度排序，相邻样本进同批 / Sort within pool so batch members are near-equal length
            pool = sampled[start:start + pool_size]
            pool.sort(key=self.lengths.__getitem__)
            batches.extend(
                pool[offset:offset + self.batch_size]
                for offset in range(0, len(pool), self.batch_size)
            )
        if len(batches) > 1:
            # 打乱批次顺序，避免 epoch 内出现长度趋势 / Shuffle batch order to avoid length trends
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[index] for index in order]
        yield from batches


def audio_sample_lengths(items: list[Item]) -> list[int]:
    """只读头部获取样本数，供长度感知组批。 / Read inexpensive audio headers for length-aware batching."""
    return [int(sf.info(str(item.audio)).frames) for item in items]


def inspect_alignment_item(
    item: Item, vocabulary: Vocabulary, audio_config: AudioConfig,
    *, piper_compatible: bool = False,
) -> dict:
    """预检 MAS 硬约束（帧数≥token 数）。 / Check the hard MAS requirement before a sample reaches a batch."""
    info = sf.info(str(item.audio))
    sample_count = int(info.frames)
    # center=False 的 STFT 帧数公式 / Frame count for non-centered STFT
    audio_frames = (
        0 if sample_count < audio_config.n_fft
        else 1 + (sample_count - audio_config.n_fft) // audio_config.hop_length
    )
    text_tokens = len(
        vocabulary.encode_item(item, piper_compatible=piper_compatible)
    )
    return {
        "audio": str(item.audio),
        "text": item.text,
        "language": item.language,
        "speaker": item.speaker,
        "audio_frames": audio_frames,
        "text_tokens": text_tokens,
        "frame_deficit": max(0, text_tokens - audio_frames),
        "passed": audio_frames >= text_tokens,
    }


class VitsDataset(torch.utils.data.Dataset):
    """逐条加载音频并在线计算线性谱。 / Loads audio rows and computes spectrograms on the fly."""

    def __init__(self, items: list[Item], vocabulary: Vocabulary,
                 speaker_map: dict[str, int], language_map: dict[str, int],
                 audio_config: AudioConfig, *, piper_compatible: bool = False):
        self.items = items
        self.vocabulary = vocabulary
        self.speaker_map = speaker_map
        self.language_map = language_map
        self.audio_config = audio_config
        self.piper_compatible = piper_compatible

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        samples, sample_rate = sf.read(str(item.audio), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.T.copy())
        if sample_rate != self.audio_config.sample_rate:
            raise ValueError(f"{item.audio}: sample rate {sample_rate}, expected {self.audio_config.sample_rate}")
        if waveform.shape[0] != 1:
            raise ValueError(f"{item.audio}: expected mono audio")
        # center=False 保证帧数与 MAS 预检公式一致 / center=False matches the MAS pre-check frame count
        spectrogram = torch.stft(
            waveform.squeeze(0), n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length, win_length=self.audio_config.win_length,
            window=torch.hann_window(self.audio_config.win_length), center=False, return_complex=True,
        ).abs()
        return {
            "tokens": torch.tensor(
                self.vocabulary.encode_item(
                    item, piper_compatible=self.piper_compatible,
                ),
                dtype=torch.long,
            ),
            "spectrogram": spectrogram,
            "waveform": waveform,
            "language_id": self.language_map[item.language],
            "speaker_id": self.speaker_map[item.speaker],
        }


def collate_vits(batch):
    """把变长样本补零组批并记录真实长度。 / Pad variable-length rows into tensors with true lengths."""
    batch_size = len(batch)
    max_text = max(row["tokens"].shape[0] for row in batch)
    max_spec = max(row["spectrogram"].shape[1] for row in batch)
    max_audio = max(row["waveform"].shape[1] for row in batch)
    spec_channels = batch[0]["spectrogram"].shape[0]
    tokens = torch.zeros(batch_size, max_text, dtype=torch.long)
    spectrograms = torch.zeros(batch_size, spec_channels, max_spec)
    waveforms = torch.zeros(batch_size, 1, max_audio)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    spec_lengths = torch.zeros(batch_size, dtype=torch.long)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    for index, row in enumerate(batch):
        text_length = row["tokens"].shape[0]
        spec_length = row["spectrogram"].shape[1]
        audio_length = row["waveform"].shape[1]
        tokens[index, :text_length] = row["tokens"]
        spectrograms[index, :, :spec_length] = row["spectrogram"]
        waveforms[index, :, :audio_length] = row["waveform"]
        text_lengths[index] = text_length
        spec_lengths[index] = spec_length
        audio_lengths[index] = audio_length
    return {
        "tokens": tokens, "text_lengths": text_lengths,
        "spectrograms": spectrograms, "spec_lengths": spec_lengths,
        "waveforms": waveforms, "audio_lengths": audio_lengths,
        "language_ids": torch.tensor([row["language_id"] for row in batch]),
        "speaker_ids": torch.tensor([row["speaker_id"] for row in batch]),
    }


def slice_waveforms(waveforms: torch.Tensor, starts: torch.Tensor,
                    segment_frames: int, hop_length: int) -> torch.Tensor:
    """按帧起点切出与 latent 段对齐的真实音频段。 / Slice ground-truth audio to match latent segments."""
    segment_samples = segment_frames * hop_length
    result = []
    for batch, frame_start in enumerate(starts):
        sample_start = int(frame_start.item()) * hop_length
        segment = waveforms[batch:batch + 1, :, sample_start:sample_start + segment_samples]
        # 尾部不足一段时补零，保证所有段等长 / Zero-pad tail so every segment has equal length
        result.append(F.pad(segment, (0, segment_samples - segment.shape[-1])))
    return torch.cat(result)
