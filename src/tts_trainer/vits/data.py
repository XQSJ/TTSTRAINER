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
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80


class VitsDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[Item], vocabulary: Vocabulary,
                 speaker_map: dict[str, int], language_map: dict[str, int],
                 audio_config: AudioConfig):
        self.items = items
        self.vocabulary = vocabulary
        self.speaker_map = speaker_map
        self.language_map = language_map
        self.audio_config = audio_config

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
        spectrogram = torch.stft(
            waveform.squeeze(0), n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length, win_length=self.audio_config.win_length,
            window=torch.hann_window(self.audio_config.win_length), center=False, return_complex=True,
        ).abs()
        return {
            "tokens": torch.tensor(self.vocabulary.encode_item(item), dtype=torch.long),
            "spectrogram": spectrogram,
            "waveform": waveform,
            "language_id": self.language_map[item.language],
            "speaker_id": self.speaker_map[item.speaker],
        }


def collate_vits(batch):
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
    segment_samples = segment_frames * hop_length
    result = []
    for batch, frame_start in enumerate(starts):
        sample_start = int(frame_start.item()) * hop_length
        segment = waveforms[batch:batch + 1, :, sample_start:sample_start + segment_samples]
        result.append(F.pad(segment, (0, segment_samples - segment.shape[-1])))
    return torch.cat(result)
