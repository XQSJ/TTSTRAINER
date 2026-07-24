from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
import soundfile as sf
from torch.nn import functional as F

from ..manifest import Item, format_phonemes
from ..logging_utils import TerminalProgress, format_duration, progress_bar
from .data import slice_waveforms
from .losses import kl_loss


logger = logging.getLogger(__name__)


def _item_key(item: Item, seed: int) -> str:
    identity = "\0".join((
        str(seed), str(item.audio.resolve()), item.text, item.language, item.speaker,
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def split_train_validation(
    items: list[Item], *, fraction: float, seed: int,
    minimum_per_profile: int = 1, maximum_per_profile: int | None = None,
) -> tuple[list[Item], list[Item], dict]:
    """Deterministically split every language/speaker profile.

    A profile always keeps at least one training row. Profiles with only one
    item cannot contribute validation data and are reported explicitly.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation.fraction must be between 0 and 1")
    if minimum_per_profile < 0:
        raise ValueError("validation.minimum_per_profile must not be negative")
    if maximum_per_profile is not None and maximum_per_profile < 1:
        raise ValueError("validation.maximum_per_profile must be at least 1 or null")

    groups: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        groups[(item.language, item.speaker)].append(item)

    train_items: list[Item] = []
    validation_items: list[Item] = []
    profiles = []
    for (language, speaker), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda item: _item_key(item, seed))
        if len(ordered) < 2:
            validation_count = 0
        else:
            validation_count = max(minimum_per_profile, round(len(ordered) * fraction))
            if maximum_per_profile is not None:
                validation_count = min(validation_count, maximum_per_profile)
            validation_count = min(validation_count, len(ordered) - 1)
        validation_items.extend(ordered[:validation_count])
        train_items.extend(ordered[validation_count:])
        profiles.append({
            "language": language,
            "speaker": speaker,
            "total": len(ordered),
            "train": len(ordered) - validation_count,
            "validation": validation_count,
        })

    report = {
        "format": 1,
        "strategy": "deterministic-language-speaker-stratified-v1",
        "seed": seed,
        "fraction": fraction,
        "minimum_per_profile": minimum_per_profile,
        "maximum_per_profile": maximum_per_profile,
        "train_items": len(train_items),
        "validation_items": len(validation_items),
        "train_fingerprint": hashlib.sha256(
            "\n".join(sorted(_item_key(item, seed) for item in train_items)).encode("ascii")
        ).hexdigest(),
        "validation_fingerprint": hashlib.sha256(
            "\n".join(sorted(_item_key(item, seed) for item in validation_items)).encode("ascii")
        ).hexdigest(),
        "profiles_without_validation": [
            {"language": row["language"], "speaker": row["speaker"]}
            for row in profiles if row["validation"] == 0
        ],
        "profiles": profiles,
    }
    return train_items, validation_items, report


def _write_items(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["audio", "text", "language", "speaker", "phonemes"],
        )
        writer.writeheader()
        for item in items:
            writer.writerow({
                "audio": str(item.audio),
                "text": item.text,
                "language": item.language,
                "speaker": item.speaker,
                "phonemes": format_phonemes(item.phonemes) if item.phonemes else "",
            })


def save_split_artifacts(run_dir: str | Path, train_items: list[Item],
                         validation_items: list[Item], report: dict) -> Path:
    destination = Path(run_dir) / "splits"
    _write_items(destination / "train.csv", train_items)
    _write_items(destination / "validation.csv", validation_items)
    report_path = destination / "split-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


@torch.no_grad()
def evaluate_validation(generator, loader, mel_transform, audio_config, model_config,
                        device: torch.device, *, seed: int = 1337,
                        preview_dir: str | Path | None = None) -> dict[str, float]:
    """Evaluate posterior reconstruction and the actual text-prior pathway."""
    was_training = generator.training
    generator.eval()
    totals = defaultdict(float)
    examples = 0
    total_batches = len(loader)
    interval = max(1, total_batches // 10)
    started = time.monotonic()
    live_progress = TerminalProgress("VALIDATION", total_batches)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for batch_index, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = generator(
                batch["tokens"], batch["text_lengths"], batch["spectrograms"],
                batch["spec_lengths"], batch["language_ids"], batch["speaker_ids"],
            )
            real_audio = slice_waveforms(
                batch["waveforms"], output.slice_starts,
                model_config.segment_frames, audio_config.hop_length,
            )
            mel_real = torch.log(mel_transform(real_audio.squeeze(1)).clamp_min(1e-5))
            mel_fake = torch.log(mel_transform(output.audio.squeeze(1)).clamp_min(1e-5))
            mel = F.l1_loss(mel_fake, mel_real)
            aligned_prior_audio = generator.decode_aligned_prior(
                output.prior_mean, output.audio_mask,
                batch["language_ids"], batch["speaker_ids"], output.slice_starts,
            )
            mel_prior = torch.log(
                mel_transform(aligned_prior_audio.squeeze(1)).clamp_min(1e-5)
            )
            prior_mel = F.l1_loss(mel_prior, mel_real)
            kl = kl_loss(
                output.latent_prior, output.posterior_log_scale,
                output.prior_mean, output.prior_log_scale, output.audio_mask,
            )
            batch_size = int(batch["tokens"].shape[0])
            examples += batch_size
            totals["mel"] += float(mel.item()) * batch_size
            totals["prior_mel"] += float(prior_mel.item()) * batch_size
            totals["duration"] += float(output.duration_loss.item()) * batch_size
            totals["kl"] += float(kl.item()) * batch_size
            totals["generated_peak"] += float(output.audio.abs().amax().item()) * batch_size
            totals["generated_rms"] += float(output.audio.square().mean().sqrt().item()) * batch_size
            totals["generated_clipping_ratio"] += float(
                (output.audio.abs() >= 0.999).to(torch.float32).mean().item()
            ) * batch_size
            if preview_dir is not None and batch_index == 1:
                preview = Path(preview_dir)
                preview.mkdir(parents=True, exist_ok=True)
                text_length = int(batch["text_lengths"][0].item())
                target_frames = int(batch["spec_lengths"][0].item())
                target_samples = int(batch["audio_lengths"][0].item())
                inferred, inferred_frames, _ = generator.infer(
                    batch["tokens"][0:1, :text_length],
                    batch["text_lengths"][0:1],
                    batch["language_ids"][0:1], batch["speaker_ids"][0:1],
                    noise_scale=0.0,
                    max_frames=min(max(target_frames * 2, text_length), 4000),
                )
                full_prior = generator.decode_aligned_prior(
                    output.prior_mean[0:1, :, :target_frames],
                    output.audio_mask[0:1, :, :target_frames],
                    batch["language_ids"][0:1], batch["speaker_ids"][0:1],
                )
                audio_files = {
                    "target.wav": batch["waveforms"][0, 0, :target_samples],
                    "posterior-reconstruction.wav": output.audio[0, 0],
                    "aligned-text-prior.wav": full_prior[0, 0, :target_frames * audio_config.hop_length],
                    "text-only-inference.wav": inferred[0, 0],
                }
                for filename, samples in audio_files.items():
                    sf.write(
                        preview / filename, samples.detach().cpu().numpy(),
                        audio_config.sample_rate, subtype="PCM_16",
                    )
                (preview / "diagnostics.json").write_text(json.dumps({
                    "format": 1,
                    "target_frames": target_frames,
                    "inferred_frames": int(inferred_frames[0].item()),
                    "duration_ratio": float(inferred_frames[0].item()) / max(target_frames, 1),
                    "posterior_mel": float(mel.item()),
                    "aligned_prior_mel": float(prior_mel.item()),
                    "files": list(audio_files),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(
                    "VALIDATION AUDIO | posterior_mel=%.4f | prior_mel=%.4f | "
                    "duration_ratio=%.3f | output=%s",
                    mel.item(), prior_mel.item(),
                    float(inferred_frames[0].item()) / max(target_frames, 1), preview,
                    extra={"tts_style": "success"},
                )
            live_progress.update(batch_index, f"items={examples} mel={mel.item():.4f}")
            if batch_index % interval == 0 or batch_index == total_batches:
                live_progress.clear()
                elapsed = time.monotonic() - started
                rate = batch_index / max(elapsed, 1e-9)
                logger.info(
                    "VALIDATION %s %6.2f%% | batches=%d/%d | items=%d | "
                    "mel=%.4f | prior_mel=%.4f | ETA=%s",
                    progress_bar(batch_index, total_batches),
                    100.0 * batch_index / max(total_batches, 1),
                    batch_index, total_batches, examples, mel.item(), prior_mel.item(),
                    format_duration((total_batches - batch_index) / rate),
                    extra={"tts_style": "progress"},
                )
                live_progress.update(batch_index, f"items={examples} mel={mel.item():.4f}")
    live_progress.close()
    if was_training:
        generator.train()
    if examples == 0:
        raise ValueError("validation loader contains no examples")
    metrics = {key: value / examples for key, value in totals.items()}
    metrics["total"] = 45.0 * metrics["prior_mel"] + metrics["duration"] + metrics["kl"]
    metrics["items"] = float(examples)
    return metrics
