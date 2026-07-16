from __future__ import annotations

import random
import json
import logging
import math
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import torchaudio
from torch.nn import functional as F

from ..checkpoints import CHECKPOINT_FORMAT, load_training_checkpoint, save_training_checkpoint
from ..experiments import prepare_experiment, resolve_experiment
from ..frontend import (FrontendContract, frontend_contract_from_config,
                        frontend_lock_path, load_frontend_contract,
                        build_frontend_conformance)
from ..manifest import validate_manifest
from ..logging_utils import configure_logging
from ..quality import run_audio_quality_gate
from ..semantic_quality import run_semantic_quality_gate
from ..text import Vocabulary
from .config import load_vits_config
from .data import AudioConfig, VitsDataset, collate_vits, slice_waveforms
from .discriminators import VitsDiscriminator
from .losses import (discriminator_loss, feature_matching_loss,
                     generator_adversarial_loss, kl_loss)
from .model import MultilingualVITS
from .validation import (evaluate_validation, save_split_artifacts,
                         split_train_validation)


logger = logging.getLogger(__name__)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _checkpoint_metadata(path: Path) -> dict:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format"] != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    return metadata


def _extend_id_map(existing: dict[str, int], values: set[str]) -> dict[str, int]:
    result = dict(existing)
    for value in sorted(values - set(result)):
        result[value] = len(result)
    return result


def _vocabulary_for_initialization(items, mode: str, previous: dict | None) -> Vocabulary:
    discovered = Vocabulary.build(items)
    if previous is None:
        return discovered
    old_tokens = list(previous["tokens"])
    additions = [token for token in discovered.tokens if token not in old_tokens]
    if mode == "resume" and additions:
        raise ValueError(f"resume data contains tokens absent from checkpoint: {additions!r}")
    return Vocabulary([*old_tokens, *additions])


def _load_expanded_generator(generator: MultilingualVITS, checkpoint: Path) -> None:
    state = torch.load(checkpoint / "training-state.pt", map_location="cpu", weights_only=False)["generator"]
    current = generator.state_dict()
    expandable = {"conditioning.speaker_embedding.weight", "text_encoder.embedding.weight"}
    for name, old_value in state.items():
        if name not in current:
            raise ValueError(f"checkpoint parameter missing from current model: {name}")
        new_value = current[name]
        if old_value.shape == new_value.shape:
            current[name] = old_value
        elif name in expandable and old_value.ndim == new_value.ndim \
                and old_value.shape[1:] == new_value.shape[1:] \
                and old_value.shape[0] <= new_value.shape[0]:
            new_value[:old_value.shape[0]].copy_(old_value)
            current[name] = new_value
        else:
            raise ValueError(
                f"architecture mismatch for {name}: checkpoint {tuple(old_value.shape)} vs current {tuple(new_value.shape)}"
            )
    generator.load_state_dict(current)


def _optimizer_to(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _resolve_frontend_contract(raw: dict, metadata: Path, languages: tuple[str, ...],
                               previous: dict | None) -> dict:
    declared = frontend_contract_from_config(
        raw.get("frontend"), languages,
        language_registry=raw.get("language_registry"),
    )
    lock = frontend_lock_path(metadata)
    current = load_frontend_contract(lock) if lock.is_file() else declared
    if set(current.languages) != set(languages):
        raise ValueError("frontend contract languages differ from experiment.languages")
    if current.declaration_key() != declared.declaration_key():
        raise ValueError(
            "frontend.lock.json differs from the configured provider/voices; "
            "re-run phonemize or restore the matching frontend config"
        )
    previous_raw = previous.get("frontend") if previous else None
    if previous_raw:
        old = FrontendContract.from_dict(previous_raw)
        if lock.is_file() and old.compatibility_key() != current.compatibility_key():
            raise ValueError("frontend contract differs from the checkpoint; start a new model or re-phonemize compatibly")
        if not lock.is_file():
            # A resumed run may point at the already frozen metadata while its
            # adjacent lock file was not copied. The checkpoint is the stronger
            # source of truth, but only after its declarable routing still
            # matches the current user config.
            if old.declaration_key() != declared.declaration_key():
                raise ValueError(
                    "checkpoint frontend differs from the configured provider/voices; "
                    "restore the matching config or start a new model"
                )
            current = old
    return current.to_dict()


def train_vits(config_path: str, metadata_path: str | None = None,
               output_dir: str | None = None, *, device_name: str | None = None,
               max_steps: int | None = None):
    raw, layout = resolve_experiment(
        config_path, metadata_override=metadata_path,
        output_override=output_dir, device_override=device_name,
    )
    configure_logging(raw.get("logging", {}).get("level", "INFO"))
    prepare_experiment(layout, raw, config_path)
    logger.info("training setup model=%s languages=%s", layout.name, ",".join(layout.languages))
    audio_config = AudioConfig(**raw["audio"])
    require_phonemes = raw.get("frontend", {}).get("require_phonemes", True)
    report = validate_manifest(layout.metadata, audio_config.sample_rate,
                               require_single_speaker=False,
                               require_phonemes=require_phonemes,
                               supported_languages=layout.language_specs)
    items = list(report.items)
    previous = _checkpoint_metadata(layout.initialization_checkpoint) if layout.initialization_checkpoint else None
    frontend_contract = _resolve_frontend_contract(raw, layout.metadata, layout.languages, previous)
    language_map = {language: index for index, language in enumerate(layout.languages)}
    data_languages = {item.language for item in items}
    outside = sorted(data_languages - set(language_map))
    if outside:
        raise ValueError(f"metadata contains languages not enabled by experiment.languages: {', '.join(outside)}")
    missing = sorted(set(language_map) - data_languages)
    if missing:
        raise ValueError(f"metadata has no samples for configured languages: {', '.join(missing)}")
    if previous is not None and previous["language_map"] != language_map:
        raise ValueError(
            "configured languages or their order differ from the checkpoint; "
            "keep experiment.languages unchanged when resuming or expanding speakers"
        )
    logger.info("language map=%s", language_map)
    current_speakers = {item.speaker for item in items}
    if previous is None:
        speaker_map = {speaker: index for index, speaker in enumerate(sorted(current_speakers))}
    else:
        speaker_map = _extend_id_map(previous["speaker_map"], current_speakers)
        if layout.initialization_mode == "resume" and set(speaker_map) != set(previous["speaker_map"]):
            raise ValueError("resume cannot add speakers; use expand_speakers")
        missing_old = set(previous["speaker_map"]) - current_speakers
        if layout.initialization_mode == "expand_speakers" and missing_old:
            warnings.warn(
                "old speakers are absent from the new metadata and may be forgotten: " + ", ".join(sorted(missing_old)),
                stacklevel=2,
            )
    vocabulary = _vocabulary_for_initialization(items, layout.initialization_mode, previous)
    frontend_conformance = (
        build_frontend_conformance(items, vocabulary, language_map)
        if all(item.phonemes for item in items) else None
    )
    quality_config = raw.get("quality", {})
    quality_summary = None
    if quality_config.get("enabled", False):
        logger.info("audio quality gate status=started items=%d", len(items))
        quality_report = run_audio_quality_gate(
            items, quality_config, layout.run_dir / "quality" / "audio-quality-report.json",
        )
        quality_summary = {"signal": {
            key: quality_report[key]
            for key in ("provider", "items", "passed", "failed", "failure_counts")
        }}
        logger.info(
            "audio quality gate status=completed passed=%d failed=%d",
            quality_report["passed"], quality_report["failed"],
        )
        semantic_config = quality_config.get("semantic", {})
        if semantic_config.get("enabled", False):
            logger.info("semantic quality gate status=started items=%d", len(items))
            semantic_report = run_semantic_quality_gate(
                items, semantic_config,
                layout.run_dir / "quality" / "semantic-quality-report.json",
                reference_root=layout.dataset_dir / "references",
            )
            quality_summary["semantic"] = {
                key: semantic_report[key]
                for key in ("provider", "items", "passed", "failed", "failure_counts")
            }
            logger.info(
                "semantic quality gate status=completed passed=%d failed=%d",
                semantic_report["passed"], semantic_report["failed"],
            )

    validation_config = raw.get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", False))
    split_report = None
    if validation_enabled:
        train_items, validation_items, split_report = split_train_validation(
            items,
            fraction=float(validation_config.get("fraction", 0.05)),
            seed=int(validation_config.get("seed", raw["training"].get("seed", 1337))),
            minimum_per_profile=int(validation_config.get("minimum_per_profile", 1)),
            maximum_per_profile=validation_config.get("maximum_per_profile", 100),
        )
        if validation_config.get("require_every_profile", True) \
                and split_report["profiles_without_validation"]:
            missing_profiles = ", ".join(
                f"{row['language']}/{row['speaker']}"
                for row in split_report["profiles_without_validation"]
            )
            raise ValueError(
                "validation requires at least two samples for every language/speaker profile; "
                f"profiles without validation: {missing_profiles}"
            )
        if not validation_items:
            raise ValueError(
                "validation is enabled but no validation rows can be selected; "
                "provide more data or set validation.enabled=false for a smoke test"
            )
        if previous and layout.initialization_mode == "resume" and previous.get("data_split"):
            old_split = previous["data_split"]
            for key in ("train_fingerprint", "validation_fingerprint"):
                if old_split.get(key) != split_report.get(key):
                    raise ValueError(
                        "validation split differs from the resumed checkpoint; "
                        "restore the same data and validation settings"
                    )
        save_split_artifacts(layout.run_dir, train_items, validation_items, split_report)
        logger.info(
            "dataset split train=%d validation=%d profiles=%d",
            len(train_items), len(validation_items), len(split_report["profiles"]),
        )
    else:
        train_items = items
        validation_items = []
    config = load_vits_config(config_path, vocab_size=len(vocabulary.tokens))
    config = replace(config, num_languages=len(language_map), num_speakers=len(speaker_map),
                     spec_channels=audio_config.n_fft // 2 + 1)
    logger.info(
        "dataset samples=%d speakers=%d vocabulary=%d device=%s",
        len(items), len(speaker_map), len(vocabulary.tokens), layout.device,
    )
    if config.hop_length != audio_config.hop_length:
        raise ValueError(f"decoder hop length {config.hop_length} != audio hop length {audio_config.hop_length}")

    seed = int(raw["training"].get("seed", 1337))
    random.seed(seed); torch.manual_seed(seed)
    device = select_device(layout.device)
    logger.info("selected device=%s", device)
    dataset = VitsDataset(train_items, vocabulary, speaker_map, language_map, audio_config)
    language_counts = Counter(item.language for item in train_items)
    speaker_counts = Counter(item.speaker for item in train_items)
    weights = [
        1.0 / (language_counts[item.language] * speaker_counts[item.speaker]) ** 0.5
        for item in train_items
    ]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_items), replacement=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=raw["training"]["batch_size"], sampler=sampler,
        num_workers=raw["training"].get("num_workers", 0), collate_fn=collate_vits,
    )
    validation_loader = None
    if validation_items:
        validation_loader = torch.utils.data.DataLoader(
            VitsDataset(validation_items, vocabulary, speaker_map, language_map, audio_config),
            batch_size=int(validation_config.get("batch_size", raw["training"]["batch_size"])),
            shuffle=False, num_workers=raw["training"].get("num_workers", 0),
            collate_fn=collate_vits,
        )
    generator = MultilingualVITS(config).to(device)
    discriminator = VitsDiscriminator().to(device)
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=raw["training"]["learning_rate_generator"], betas=(0.8, 0.99))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=raw["training"]["learning_rate_discriminator"], betas=(0.8, 0.99))
    start_epoch = 1
    global_step = 0
    if layout.initialization_mode == "resume":
        restored = load_training_checkpoint(
            layout.initialization_checkpoint, generator=generator, discriminator=discriminator,
            optimizer_g=optimizer_g, optimizer_d=optimizer_d,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        _optimizer_to(optimizer_g, device); _optimizer_to(optimizer_d, device)
    elif layout.initialization_mode == "expand_speakers":
        _load_expanded_generator(generator, layout.initialization_checkpoint)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_config.sample_rate, n_fft=audio_config.n_fft,
        win_length=audio_config.win_length, hop_length=audio_config.hop_length,
        n_mels=audio_config.n_mels, center=False,
    ).to(device)
    destination = layout.checkpoints_dir
    vocabulary.save(layout.run_dir / "vocab.json")
    if start_epoch > raw["training"]["epochs"]:
        raise ValueError(
            f"checkpoint already reached epoch {start_epoch - 1}; set training.epochs to at least {start_epoch}"
        )
    log_every = int(raw["training"].get("log_every_steps", 10))
    if log_every < 1:
        raise ValueError("training.log_every_steps must be at least 1")
    selection_metric = str(validation_config.get("metric", "mel"))
    if selection_metric not in {"mel", "duration", "kl", "total"}:
        raise ValueError("validation.metric must be mel, duration, kl, or total")
    previous_selection = previous.get("selection") if previous else None
    best_value = float("inf")
    best_epoch = None
    if previous_selection and previous_selection.get("metric") == selection_metric:
        best_value = float(previous_selection.get("best_value", best_value))
        best_epoch = previous_selection.get("best_epoch")
    evaluation_every = int(validation_config.get("every_epochs", 1))
    if evaluation_every < 1:
        raise ValueError("validation.every_epochs must be at least 1")

    for epoch in range(start_epoch, raw["training"]["epochs"] + 1):
        logger.info("epoch=%d status=started", epoch)
        epoch_totals = Counter()
        epoch_steps = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = generator(
                batch["tokens"], batch["text_lengths"], batch["spectrograms"],
                batch["spec_lengths"], batch["language_ids"], batch["speaker_ids"],
            )
            real_audio = slice_waveforms(batch["waveforms"], output.slice_starts,
                                         config.segment_frames, audio_config.hop_length)

            optimizer_d.zero_grad(set_to_none=True)
            real_d = discriminator(real_audio)
            fake_d = discriminator(output.audio.detach())
            loss_d = discriminator_loss(real_d, fake_d)
            loss_d.backward(); optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            for parameter in discriminator.parameters(): parameter.requires_grad_(False)
            with torch.no_grad(): real_features = discriminator(real_audio)
            fake_features = discriminator(output.audio)
            mel_real = torch.log(mel_transform(real_audio.squeeze(1)).clamp_min(1e-5))
            mel_fake = torch.log(mel_transform(output.audio.squeeze(1)).clamp_min(1e-5))
            loss_mel = F.l1_loss(mel_fake, mel_real)
            loss_kl = kl_loss(output.latent_prior, output.posterior_log_scale,
                              output.prior_mean, output.prior_log_scale, output.audio_mask)
            loss_g = (
                45.0 * loss_mel + output.duration_loss + loss_kl
                + generator_adversarial_loss(fake_features)
                + 2.0 * feature_matching_loss(real_features, fake_features)
            )
            loss_g.backward(); torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0); optimizer_g.step()
            for parameter in discriminator.parameters(): parameter.requires_grad_(True)
            global_step += 1
            epoch_steps += 1
            epoch_totals.update({
                "generator": float(loss_g.item()),
                "discriminator": float(loss_d.item()),
                "mel": float(loss_mel.item()),
                "duration": float(output.duration_loss.item()),
                "kl": float(loss_kl.item()),
            })
            if global_step == 1 or global_step % log_every == 0:
                logger.info(
                    "epoch=%d step=%d generator=%.4f discriminator=%.4f mel=%.4f",
                    epoch, global_step, loss_g.item(), loss_d.item(), loss_mel.item(),
                )
            checkpoint_every = raw["training"].get("checkpoint_every_steps", 5000)
            if global_step % checkpoint_every == 0:
                logger.info("checkpoint step=%d status=saving", global_step)
                save_training_checkpoint(
                    destination / f"step-{global_step:09d}", generator=generator,
                    discriminator=discriminator, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                    epoch=epoch, global_step=global_step, config=config,
                    language_map=language_map, speaker_map=speaker_map, tokens=vocabulary.tokens,
                    frontend=frontend_contract,
                    frontend_conformance=frontend_conformance,
                    selection={
                        "metric": selection_metric,
                        "best_value": best_value if math.isfinite(best_value) else None,
                        "best_epoch": best_epoch,
                    },
                    data_split=split_report,
                    quality_summary=quality_summary,
                    metrics={"generator": loss_g.item(), "discriminator": loss_d.item(), "mel": loss_mel.item()},
                )
            if max_steps is not None and global_step >= max_steps:
                break
        train_metrics = {
            key: value / max(epoch_steps, 1) for key, value in epoch_totals.items()
        }
        validation_metrics = None
        should_evaluate = validation_loader is not None and (
            epoch == start_epoch or epoch % evaluation_every == 0
            or (max_steps is not None and global_step >= max_steps)
        )
        if should_evaluate:
            logger.info("validation epoch=%d status=started", epoch)
            validation_metrics = evaluate_validation(
                generator, validation_loader, mel_transform, audio_config, config, device,
                seed=int(validation_config.get("seed", seed)),
            )
            current_value = float(validation_metrics[selection_metric])
            logger.info(
                "validation epoch=%d mel=%.4f duration=%.4f kl=%.4f total=%.4f",
                epoch, validation_metrics["mel"], validation_metrics["duration"],
                validation_metrics["kl"], validation_metrics["total"],
            )
            if current_value < best_value:
                best_value = current_value
                best_epoch = epoch
                selection = {
                    "metric": selection_metric,
                    "best_value": best_value,
                    "best_epoch": best_epoch,
                    "mode": "min",
                }
                save_training_checkpoint(
                    destination / "best", generator=generator, discriminator=discriminator,
                    optimizer_g=optimizer_g, optimizer_d=optimizer_d, epoch=epoch,
                    global_step=global_step, config=config, language_map=language_map,
                    speaker_map=speaker_map, tokens=vocabulary.tokens,
                    frontend=frontend_contract,
                    frontend_conformance=frontend_conformance,
                    selection=selection, data_split=split_report,
                    quality_summary=quality_summary,
                    metrics={"train": train_metrics, "validation": validation_metrics},
                )
                logger.info(
                    "best checkpoint updated epoch=%d metric=%s value=%.6f path=%s",
                    epoch, selection_metric, best_value, destination / "best",
                )
        selection = {
            "metric": selection_metric,
            "best_value": best_value if math.isfinite(best_value) else None,
            "best_epoch": best_epoch,
            "mode": "min",
        } if validation_enabled else {"enabled": False}
        save_training_checkpoint(
            destination / "last", generator=generator, discriminator=discriminator,
            optimizer_g=optimizer_g, optimizer_d=optimizer_d, epoch=epoch,
            global_step=global_step, config=config, language_map=language_map,
            speaker_map=speaker_map, tokens=vocabulary.tokens, frontend=frontend_contract,
            frontend_conformance=frontend_conformance,
            selection=selection, data_split=split_report,
            quality_summary=quality_summary,
            metrics={"train": train_metrics, "validation": validation_metrics},
        )
        logger.info("epoch=%d status=completed step=%d checkpoint=%s", epoch, global_step, destination / "last")
        if max_steps is not None and global_step >= max_steps: break
    logger.info("training completed checkpoint=%s", destination / "last")
    return destination / "last"
