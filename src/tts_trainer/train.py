from __future__ import annotations

import random
from pathlib import Path

from .config import load_config
from .manifest import read_manifest, validate_manifest
from .model import build_model
from .optional import require_training_dependencies
from .text import Vocabulary


def train(config_path: str) -> Path:
    torch, torchaudio = require_training_dependencies()
    config = load_config(config_path)
    report = validate_manifest(config.data.metadata, config.data.sample_rate)
    items = list(report.items)
    language_map = {language: index for index, language in enumerate(sorted(report.language_counts))}
    vocab = Vocabulary.build(items)
    output = Path(config.training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocab.save(output / "vocab.json")
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return len(items)
        def __getitem__(self, index):
            item = items[index]
            import soundfile as sf
            samples, sr = sf.read(str(item.audio), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(samples.T.copy())
            if sr != config.data.sample_rate: raise RuntimeError(f"unexpected sample rate: {sr}")
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=config.data.n_fft,
                hop_length=config.data.hop_length, n_mels=config.data.n_mels,
            )(waveform).clamp_min(1e-5).log()
            return torch.tensor(vocab.encode_item(item)), language_map[item.language], mel

    def collate(batch):
        # Baseline aligns text positions to uniformly resampled Mel frames. M2
        # replaces this with explicit duration prediction/alignment.
        max_text = max(len(row[0]) for row in batch)
        tokens = torch.zeros(len(batch), max_text, dtype=torch.long)
        targets = torch.zeros(len(batch), config.data.n_mels, max_text)
        mask = torch.ones(len(batch), max_text, dtype=torch.bool)
        langs = torch.tensor([row[1] for row in batch])
        for i, (ids, _, mel) in enumerate(batch):
            length = len(ids); tokens[i, :length] = ids; mask[i, :length] = False
            targets[i, :, :length] = torch.nn.functional.interpolate(mel.unsqueeze(0), size=length, mode="linear").squeeze(0)
        return tokens, langs, mask, targets

    counts = report.language_counts
    weights = [1.0 / counts[item.language] for item in items]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(items), replacement=True)
    loader = torch.utils.data.DataLoader(Dataset(), batch_size=config.data.batch_size, sampler=sampler,
                                         num_workers=config.data.num_workers, collate_fn=collate)
    model = build_model(
        len(vocab.tokens), config.data.n_mels, config.model,
        num_languages=len(language_map),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    best = float("inf")
    for epoch in range(1, config.training.epochs + 1):
        model.train(); total = 0.0
        for tokens, langs, mask, target in loader:
            prediction = model(tokens, langs, mask)
            valid = (~mask).unsqueeze(1).expand_as(prediction)
            loss = torch.nn.functional.l1_loss(prediction[valid], target[valid])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item()
        mean = total / max(len(loader), 1)
        checkpoint = {"model": model.state_dict(), "config": config, "vocab_size": len(vocab.tokens),
                      "language_map": language_map, "epoch": epoch, "loss": mean}
        torch.save(checkpoint, output / "last.pt")
        if mean < best:
            best = mean; torch.save(checkpoint, output / "best.pt")
        print(f"epoch={epoch} loss={mean:.5f}")
    return output / "best.pt"
