from __future__ import annotations

import random
import json
import logging
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
                        frontend_lock_path, load_frontend_contract)
from ..manifest import validate_manifest
from ..logging_utils import configure_logging
from ..text import Vocabulary
from .config import load_vits_config
from .data import AudioConfig, VitsDataset, collate_vits, slice_waveforms
from .discriminators import VitsDiscriminator
from .losses import (discriminator_loss, feature_matching_loss,
                     generator_adversarial_loss, kl_loss)
from .model import MultilingualVITS


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
    if current.compatibility_key() != declared.compatibility_key():
        raise ValueError(
            "frontend.lock.json differs from the configured provider/voices; "
            "re-run phonemize or restore the matching frontend config"
        )
    previous_raw = previous.get("frontend") if previous else None
    if previous_raw:
        old = FrontendContract.from_dict(previous_raw)
        if old.compatibility_key() != current.compatibility_key():
            raise ValueError("frontend contract differs from the checkpoint; start a new model or re-phonemize compatibly")
        if current.engine_version is None:
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
    dataset = VitsDataset(items, vocabulary, speaker_map, language_map, audio_config)
    language_counts = Counter(item.language for item in items)
    speaker_counts = Counter(item.speaker for item in items)
    weights = [1.0 / (language_counts[item.language] * speaker_counts[item.speaker]) ** 0.5 for item in items]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(items), replacement=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=raw["training"]["batch_size"], sampler=sampler,
        num_workers=raw["training"].get("num_workers", 0), collate_fn=collate_vits,
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
    for epoch in range(start_epoch, raw["training"]["epochs"] + 1):
        logger.info("epoch=%d status=started", epoch)
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
                    metrics={"generator": loss_g.item(), "discriminator": loss_d.item(), "mel": loss_mel.item()},
                )
            if max_steps is not None and global_step >= max_steps:
                break
        save_training_checkpoint(
            destination / "last", generator=generator, discriminator=discriminator,
            optimizer_g=optimizer_g, optimizer_d=optimizer_d, epoch=epoch,
            global_step=global_step, config=config, language_map=language_map,
            speaker_map=speaker_map, tokens=vocabulary.tokens, frontend=frontend_contract,
        )
        logger.info("epoch=%d status=completed step=%d checkpoint=%s", epoch, global_step, destination / "last")
        if max_steps is not None and global_step >= max_steps: break
    logger.info("training completed checkpoint=%s", destination / "last")
    return destination / "last"
