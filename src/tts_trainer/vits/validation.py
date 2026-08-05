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


def _profile_path_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ) or "unknown"


@torch.no_grad()
def _evaluate_profile_preview(
    generator, batch, output, index: int, mel_transform, audio_config,
    *, posterior_segment_mel: float, prior_segment_mel: float,
    destinations: list[Path], language: str, speaker: str,
) -> dict[str, float]:
    """评估一个固定完整样本并可保存试听。 / Evaluate one full profile item."""
    text_length = int(batch["text_lengths"][index].item())
    target_frames = int(batch["spec_lengths"][index].item())
    target_samples = int(batch["audio_lengths"][index].item())
    language_ids = batch["language_ids"][index:index + 1]
    speaker_ids = batch["speaker_ids"][index:index + 1]
    audio_mask = output.audio_mask[index:index + 1, :, :target_frames]
    full_posterior = generator.decode_posterior(
        output.latent[index:index + 1, :, :target_frames],
        audio_mask, language_ids, speaker_ids,
    )
    full_posterior_mean = generator.decode_posterior(
        output.posterior_mean[index:index + 1, :, :target_frames],
        audio_mask, language_ids, speaker_ids,
    )
    comparison_samples = target_frames * audio_config.hop_length
    comparison_target = batch["waveforms"][
        index:index + 1, 0, :comparison_samples
    ]
    comparison_mel = torch.log(
        mel_transform(comparison_target.float()).clamp_min(1e-5)
    )
    posterior_mean_mel = F.l1_loss(
        torch.log(
            mel_transform(
                full_posterior_mean[0:1, 0, :comparison_samples].float()
            ).clamp_min(1e-5)
        ),
        comparison_mel,
    )
    posterior_sampled_mel = F.l1_loss(
        torch.log(
            mel_transform(
                full_posterior[0:1, 0, :comparison_samples].float()
            ).clamp_min(1e-5)
        ),
        comparison_mel,
    )
    metrics = {
        "posterior_mean_full_mel": float(posterior_mean_mel.item()),
        "posterior_sampled_full_mel": float(posterior_sampled_mel.item()),
    }
    if not destinations:
        return metrics

    inferred, inferred_frames, _ = generator.infer(
        batch["tokens"][index:index + 1, :text_length],
        batch["text_lengths"][index:index + 1],
        language_ids, speaker_ids, noise_scale=0.667,
        duration_noise_scale=0.35,
        max_frames=min(max(target_frames * 2, text_length), 4000),
    )
    deterministic, deterministic_frames, _ = generator.infer(
        batch["tokens"][index:index + 1, :text_length],
        batch["text_lengths"][index:index + 1],
        language_ids, speaker_ids, noise_scale=0.0,
        duration_noise_scale=0.0,
        max_frames=min(max(target_frames * 2, text_length), 4000),
    )
    full_prior = generator.decode_aligned_prior(
        output.prior_mean[index:index + 1, :, :target_frames],
        audio_mask, language_ids, speaker_ids,
    )
    audio_files = {
        "target.wav": batch["waveforms"][index, 0, :target_samples],
        "posterior-reconstruction.wav": full_posterior_mean[
            0, 0, :comparison_samples
        ],
        "posterior-sampled-reconstruction.wav": full_posterior[
            0, 0, :comparison_samples
        ],
        "aligned-text-prior.wav": full_prior[0, 0, :comparison_samples],
        "text-only-inference.wav": inferred[0, 0],
        "text-only-deterministic.wav": deterministic[0, 0],
    }
    diagnostics = {
        "format": 3,
        "language": language,
        "speaker": speaker,
        "target_frames": target_frames,
        "inferred_frames": int(inferred_frames[0].item()),
        "duration_ratio": float(inferred_frames[0].item()) / max(target_frames, 1),
        "deterministic_frames": int(deterministic_frames[0].item()),
        "deterministic_duration_ratio": (
            float(deterministic_frames[0].item()) / max(target_frames, 1)
        ),
        "inference_scales": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "duration_noise_scale": 0.35,
        },
        "posterior_mel": posterior_segment_mel,
        "posterior_segment_sampled_mel": posterior_segment_mel,
        **metrics,
        "posterior_scale_mean": float(
            output.posterior_log_scale[
                index, :, :target_frames
            ].exp().mean().item()
        ),
        "posterior_scale_max": float(
            output.posterior_log_scale[
                index, :, :target_frames
            ].exp().max().item()
        ),
        "aligned_prior_mel": prior_segment_mel,
        "files": list(audio_files),
    }
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for filename, samples in audio_files.items():
            sf.write(
                destination / filename, samples.detach().float().cpu().numpy(),
                audio_config.sample_rate, subtype="PCM_16",
            )
        (destination / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return metrics


@torch.no_grad()
def evaluate_validation(generator, loader, mel_transform, audio_config, model_config,
                        device: torch.device, *, seed: int = 1337,
                        preview_dir: str | Path | None = None,
                        language_map: dict[str, int] | None = None,
                        speaker_map: dict[str, int] | None = None) -> dict:
    """Evaluate posterior reconstruction and the actual text-prior pathway."""
    was_training = generator.training
    generator.eval()
    totals = defaultdict(float)
    profile_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    profile_full_metrics: dict[str, dict[str, float]] = {}
    language_names = {value: key for key, value in (language_map or {}).items()}
    speaker_names = {value: key for key, value in (speaker_map or {}).items()}
    legacy_preview_written = False
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
            mel_per_item = (mel_fake - mel_real).abs().mean((1, 2))
            prior_per_item = (mel_prior - mel_real).abs().mean((1, 2))
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
            language_ids = batch["language_ids"].detach().cpu().tolist()
            speaker_ids = batch["speaker_ids"].detach().cpu().tolist()
            for index, (language_id, speaker_id) in enumerate(zip(
                language_ids, speaker_ids,
            )):
                language = language_names.get(int(language_id), str(language_id))
                speaker = speaker_names.get(int(speaker_id), str(speaker_id))
                profile_name = f"{language}/{speaker}"
                profile_totals[profile_name]["mel"] += float(
                    mel_per_item[index].item()
                )
                profile_totals[profile_name]["prior_mel"] += float(
                    prior_per_item[index].item()
                )
                profile_totals[profile_name]["items"] += 1.0
                if profile_name in profile_full_metrics:
                    continue
                destinations: list[Path] = []
                if preview_dir is not None:
                    root = Path(preview_dir)
                    destinations.append(
                        root / "profiles"
                        / _profile_path_component(language)
                        / _profile_path_component(speaker)
                    )
                    if not legacy_preview_written:
                        destinations.append(root)
                        legacy_preview_written = True
                profile_full_metrics[profile_name] = _evaluate_profile_preview(
                    generator, batch, output, index, mel_transform, audio_config,
                    posterior_segment_mel=float(mel_per_item[index].item()),
                    prior_segment_mel=float(prior_per_item[index].item()),
                    destinations=destinations, language=language, speaker=speaker,
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
    metrics["combined_mel"] = metrics["mel"] + metrics["prior_mel"]
    # total 保持对应标准阶段的 VITS 目标；refinement 阶段使用 combined_mel
    # 单独选择兼顾声学主链和文本先验的 checkpoint。
    # Keep total aligned with the standard-stage VITS objective. Refinement
    # uses combined_mel separately to balance acoustics and the text prior.
    metrics["total"] = 45.0 * metrics["mel"] + metrics["duration"] + metrics["kl"]
    metrics["items"] = float(examples)
    metrics["profiles"] = {}
    for profile_name, values in sorted(profile_totals.items()):
        count = max(values["items"], 1.0)
        profile = {
            "mel": values["mel"] / count,
            "prior_mel": values["prior_mel"] / count,
            "items": values["items"],
            **profile_full_metrics.get(profile_name, {}),
        }
        metrics["profiles"][profile_name] = profile
        logger.info(
            "VALIDATION PROFILE | profile=%s | items=%d | mel=%.4f | "
            "prior_mel=%.4f | posterior_mean_full_mel=%s | "
            "posterior_sampled_full_mel=%s",
            profile_name, int(profile["items"]), profile["mel"],
            profile["prior_mel"],
            (
                f"{profile['posterior_mean_full_mel']:.4f}"
                if "posterior_mean_full_mel" in profile else "missing"
            ),
            (
                f"{profile['posterior_sampled_full_mel']:.4f}"
                if "posterior_sampled_full_mel" in profile else "missing"
            ),
            extra={"tts_style": "section"},
        )
    return metrics
