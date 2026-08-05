from __future__ import annotations

import json
import logging
import math
import random
import time
import warnings
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from statistics import median

import torch
import torchaudio
from torch.nn import functional as F

from ..checkpoints import (inherit_resume_best_checkpoint,
                           load_training_checkpoint, require_checkpoint_format,
                           require_warm_start_checkpoint_format,
                           save_training_checkpoint)
from ..experiments import prepare_experiment, resolve_experiment
from ..frontend import (FrontendContract, frontend_contract_from_config,
                        frontend_lock_path, load_frontend_contract,
                        build_frontend_conformance)
from ..frontend.contract import (MOBILE_DIRECT_TOKEN_ENCODING,
                                 PIPER_TOKEN_ENCODING)
from ..manifest import validate_manifest
from ..logging_utils import (TerminalProgress, configure_logging_from_config,
                             format_duration, progress_bar)
from ..quality import run_audio_quality_gate
from ..semantic_quality import run_semantic_quality_gate
from ..text import Vocabulary
from .config import load_vits_config
from .data import (AudioConfig, VitsDataset, collate_vits,
                   inspect_alignment_item, slice_waveforms)
from .discriminators import VitsDiscriminator
from .losses import (discriminator_loss, feature_matching_loss,
                     generator_adversarial_loss, kl_loss)
from .model import MultilingualVITS
from .validation import (evaluate_validation, save_split_artifacts,
                         split_train_validation)


logger = logging.getLogger(__name__)


def duration_predictor_settings(config: dict | object) -> dict:
    """Resolve duration architecture, including format-4 legacy defaults."""
    if not isinstance(config, dict):
        config = config.to_dict()
    return {
        "duration_predictor_type": config.get(
            "duration_predictor_type", "stochastic_lognormal",
        ),
        "duration_predictor_channels": int(config.get(
            "duration_predictor_channels", 64,
        )),
        "duration_predictor_flow_layers": int(config.get(
            "duration_predictor_flow_layers", 2,
        )),
    }


def preserve_checkpoint_duration_architecture(config, previous: dict | None, mode: str):
    """Resume with the checkpoint predictor even if preset defaults evolved.

    preset defaults may safely improve new experiments. Resume/refinement and
    speaker expansion must nevertheless reproduce the exact saved graph. Use
    warm_start to deliberately replace the predictor.
    """
    if previous is None or mode not in {
        "resume", "refine_text_prior", "expand_speakers",
    }:
        return config
    checkpoint_settings = duration_predictor_settings(previous.get("config", {}))
    current_settings = duration_predictor_settings(config)
    if checkpoint_settings != current_settings:
        logger.warning(
            "DURATION COMPATIBILITY | mode=%s | configured=%s | checkpoint=%s | "
            "action=use-checkpoint-architecture | use warm_start to upgrade",
            mode, current_settings["duration_predictor_type"],
            checkpoint_settings["duration_predictor_type"],
        )
    return replace(config, **checkpoint_settings)


def profile_balancing_weights(items) -> list[float]:
    """让每个语言×音色组合获得相同采样质量。 / Balance every lang×voice pair."""
    counts = Counter((item.language, item.speaker) for item in items)
    return [
        1.0 / counts[(item.language, item.speaker)]
        for item in items
    ]


def reject_removed_prior_options(training: dict) -> None:
    """拒绝会恢复旧回归的配置。 / Reject options that revive the regression."""
    removed = sorted({
        "aligned_prior_mel_weight",
        "aligned_prior_mel_start_steps",
        "aligned_prior_mel_warmup_steps",
    } & set(training))
    if removed:
        raise ValueError(
            "这些训练参数已移除，因为会破坏 posterior 重建 / "
            "removed training options would damage posterior reconstruction: "
            + ", ".join(removed)
            + "; 请删除后使用标准 VITS 损失 / delete them and use the "
            "standard VITS objective"
        )


def resolve_refinement_config(training: dict) -> dict:
    """校验文本先验强化参数。 / Validate text-prior refinement settings."""
    raw = dict(training.get("text_prior_refinement") or {})
    stage = str(training.get("stage", "auto")).lower()
    if stage not in {"auto", "standard"}:
        raise ValueError(
            "training.stage must be auto or standard; use "
            "experiment.initialization.mode=refine_text_prior to explicitly "
            "start text-prior refinement"
        )
    config = {
        "stage": stage,
        "enabled": bool(raw.get("enabled", False)) and stage == "auto",
        "start_steps": int(raw.get("start_steps", 10_000)),
        "posterior_mel_threshold": raw.get("posterior_mel_threshold", 0.5),
        "full_posterior_mel_threshold": raw.get(
            "full_posterior_mel_threshold", None,
        ),
        "require_all_profiles": bool(raw.get("require_all_profiles", True)),
        "consecutive_passes": int(raw.get("consecutive_passes", 1)),
        "learning_rate": float(raw.get("learning_rate", 1e-4)),
        "lr_decay": float(raw.get("lr_decay", training.get("lr_decay", 0.999875))),
        "mel_weight": float(raw.get("mel_weight", 1.0)),
        "mel_warmup_steps": int(raw.get("mel_warmup_steps", 5_000)),
        "kl_weight": float(raw.get("kl_weight", 1.0)),
        "duration_nll_weight": float(raw.get("duration_nll_weight", 1.0)),
        "duration_mean_weight": float(raw.get("duration_mean_weight", 1.0)),
    }
    threshold = config["posterior_mel_threshold"]
    if threshold is not None:
        threshold = float(threshold)
        if threshold <= 0.0:
            raise ValueError(
                "training.text_prior_refinement.posterior_mel_threshold "
                "must be positive or null"
            )
        config["posterior_mel_threshold"] = threshold
    full_threshold = config["full_posterior_mel_threshold"]
    if full_threshold is not None:
        full_threshold = float(full_threshold)
        if full_threshold <= 0.0:
            raise ValueError(
                "training.text_prior_refinement.full_posterior_mel_threshold "
                "must be positive or null"
            )
        config["full_posterior_mel_threshold"] = full_threshold
    if config["start_steps"] < 0 or config["mel_warmup_steps"] < 0:
        raise ValueError(
            "text-prior refinement step counts must not be negative"
        )
    if config["learning_rate"] <= 0.0:
        raise ValueError("text-prior refinement learning_rate must be positive")
    if config["consecutive_passes"] < 1:
        raise ValueError(
            "text-prior refinement consecutive_passes must be at least 1"
        )
    if not 0.0 < config["lr_decay"] <= 1.0:
        raise ValueError("text-prior refinement lr_decay must be in (0, 1]")
    for key in (
        "mel_weight", "kl_weight", "duration_nll_weight",
        "duration_mean_weight",
    ):
        if config[key] < 0.0:
            raise ValueError(f"text-prior refinement {key} must not be negative")
    if not any(config[key] > 0.0 for key in (
        "mel_weight", "kl_weight", "duration_nll_weight",
        "duration_mean_weight",
    )):
        raise ValueError("text-prior refinement must enable at least one loss")
    return config


def acoustic_refinement_gate(
    metrics: dict | None,
    config: dict,
) -> tuple[bool, list[str]]:
    """要求所有语言×音色声学指标达标。 / Gate on every language×voice."""
    threshold = config.get("posterior_mel_threshold")
    full_threshold = config.get("full_posterior_mel_threshold")
    if threshold is None and full_threshold is None:
        return True, []
    if metrics is None:
        return False, ["validation metrics unavailable"]

    failures: list[str] = []
    profiles = metrics.get("profiles") or {}
    if config.get("require_all_profiles", True):
        if not profiles:
            return False, ["per-profile validation metrics unavailable"]
        for name, profile in sorted(profiles.items()):
            if threshold is not None and float(profile["mel"]) > threshold:
                failures.append(
                    f"{name}:mel={float(profile['mel']):.4f}>{threshold:.4f}"
                )
            for key in (
                "posterior_mean_full_mel", "posterior_sampled_full_mel",
            ):
                value = profile.get(key)
                if (
                    full_threshold is not None
                    and (value is None or float(value) > full_threshold)
                ):
                    rendered = "missing" if value is None else f"{float(value):.4f}"
                    failures.append(
                        f"{name}:{key}={rendered}>{full_threshold:.4f}"
                    )
    elif threshold is not None and float(metrics["mel"]) > threshold:
        failures.append(
            f"overall:mel={float(metrics['mel']):.4f}>{threshold:.4f}"
        )
    return not failures, failures


def configure_training_stage(generator, discriminator, stage: str) -> None:
    """切换可训练模块且不改变最终模型结构。 / Select trainable modules."""
    if stage not in {"standard", "text_prior_refinement"}:
        raise ValueError(f"unknown training stage: {stage}")
    for parameter in generator.parameters():
        parameter.requires_grad_(stage == "standard")
    for parameter in discriminator.parameters():
        parameter.requires_grad_(stage == "standard")
    if stage == "text_prior_refinement":
        for module in (
            generator.text_encoder, generator.flow,
            generator.duration_predictor,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def refinement_mel_weight(config: dict, refinement_steps: int) -> float:
    """平滑加入安全 Mel 监督。 / Warm in the safe Mel supervision."""
    target = config["mel_weight"]
    warmup = config["mel_warmup_steps"]
    if target <= 0.0:
        return 0.0
    if warmup <= 0:
        return target
    return target * min(1.0, max(refinement_steps, 0) / warmup)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def resolve_mixed_precision(training: dict, device: torch.device) -> tuple[str, torch.dtype | None]:
    """Select CUDA AMP without changing checkpoint parameter precision."""
    requested = str(training.get("mixed_precision", "auto")).lower()
    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(
            "training.mixed_precision must be auto, bf16, fp16, or fp32"
        )
    if device.type != "cuda":
        if requested in {"bf16", "fp16"}:
            logger.warning(
                "MIXED PRECISION DISABLED | requested=%s | device=%s | using=fp32",
                requested, device,
            )
        return "fp32", None
    if requested == "fp32":
        return "fp32", None
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if requested == "bf16" and not bf16_supported:
        raise ValueError(
            "training.mixed_precision=bf16 requires a CUDA device with BF16 support; "
            "use auto or fp16"
        )
    if requested == "bf16" or (requested == "auto" and bf16_supported):
        return "bf16", torch.bfloat16
    if requested == "fp16":
        return "fp16", torch.float16
    # 自动模式宁可回退 FP32，也不在不支持 BF16 的设备上悄悄启用
    # 数值范围更窄的 FP16。 / Auto prefers a safe FP32 fallback over
    # silently selecting the narrower FP16 format.
    return "fp32", None


def _checkpoint_metadata(path: Path, *, warm_start: bool = False) -> dict:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if warm_start:
        require_warm_start_checkpoint_format(int(metadata["format"]))
    else:
        require_checkpoint_format(int(metadata["format"]))
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
    if mode in {"resume", "refine_text_prior"} and additions:
        raise ValueError(
            f"{mode} data contains tokens absent from checkpoint: {additions!r}"
        )
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


def _load_warm_start_generator(
    generator: MultilingualVITS, checkpoint: Path, excludes: tuple[str, ...],
    discriminator=None,
) -> dict:
    """Load compatible generator modules while resetting selected components.

    Format-3 checkpoints always reset the duration predictor because its
    deterministic one-channel head cannot represent the format-4 stochastic
    distribution. Format-4 checkpoints can additionally exclude explicit
    module prefixes from the public configuration.
    """
    state_file = torch.load(
        checkpoint / "training-state.pt", map_location="cpu", weights_only=False,
    )
    checkpoint_format = int(state_file["format"])
    require_warm_start_checkpoint_format(checkpoint_format)
    metadata = json.loads(
        (checkpoint / "metadata.json").read_text(encoding="utf-8"),
    )
    old_duration = duration_predictor_settings(metadata.get("config", {}))
    new_duration = duration_predictor_settings(generator.config)
    forced = (
        ("duration_predictor",)
        if checkpoint_format == 3 or old_duration != new_duration else ()
    )
    prefixes = tuple(dict.fromkeys((*forced, *excludes)))

    def excluded(name: str) -> bool:
        return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)

    old_state = state_file["generator"]
    current = generator.state_dict()
    expandable = {
        "conditioning.speaker_embedding.weight",
        "text_encoder.embedding.weight",
    }
    loaded = []
    skipped = []
    for name, old_value in old_state.items():
        if excluded(name):
            skipped.append(name)
            continue
        if name not in current:
            raise ValueError(f"warm-start parameter is absent from current model: {name}")
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
                f"warm-start architecture mismatch for {name}: checkpoint "
                f"{tuple(old_value.shape)} vs current {tuple(new_value.shape)}"
            )
        loaded.append(name)
    missing = [name for name in current if name not in old_state and not excluded(name)]
    if missing:
        raise ValueError(
            "warm-start checkpoint is missing non-excluded parameters: "
            + ", ".join(missing[:10])
        )
    generator.load_state_dict(current)
    discriminator_loaded = False
    if discriminator is not None and state_file.get("discriminator") is not None:
        discriminator.load_state_dict(state_file["discriminator"])
        discriminator_loaded = True
    return {
        "checkpoint_format": checkpoint_format,
        "loaded_tensors": len(loaded),
        "skipped_tensors": len(skipped),
        "excluded_modules": list(prefixes),
        "discriminator_loaded": discriminator_loaded,
        "duration_predictor_from": old_duration["duration_predictor_type"],
        "duration_predictor_to": new_duration["duration_predictor_type"],
    }


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


def _semantic_reference_root(dataset_dir: Path) -> Path:
    """Return the shared voice reference directory recorded for this dataset."""
    fallback = dataset_dir / "references"
    record_path = dataset_dir / "dataset.json"
    if not record_path.is_file():
        return fallback
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning(
            "SEMANTIC QUALITY | cannot read dataset record=%s | fallback=%s | error=%s",
            record_path, fallback, error,
        )
        return fallback
    voice_dataset = record.get("voice_dataset")
    if not voice_dataset:
        return fallback
    shared = Path(voice_dataset).expanduser().resolve()
    if not shared.exists() and (shared.parent / "voice.json").is_file():
        shared = shared.parent
    return shared / "references"


def train_vits(config_path: str, metadata_path: str | None = None,
               output_dir: str | None = None, *, device_name: str | None = None,
               max_steps: int | None = None):
    raw, layout = resolve_experiment(
        config_path, metadata_override=metadata_path,
        output_override=output_dir, device_override=device_name,
    )
    configure_logging_from_config(raw)
    # 2026-07-30 的辅助 prior Mel 会从随机 text prior 反向更新共享 Decoder，
    # 实测会破坏 posterior 重建，因此旧配置必须显式失败，不能静默忽略。
    # The 2026-07-30 auxiliary prior-Mel loss backpropagated from a random text
    # prior into the shared decoder and damaged posterior reconstruction.
    reject_removed_prior_options(raw["training"])
    prepare_experiment(layout, raw, config_path)
    logger.info("training setup model=%s languages=%s", layout.name, ",".join(layout.languages))
    audio_config = AudioConfig(**raw["audio"])
    require_phonemes = raw.get("frontend", {}).get("require_phonemes", True)
    report = validate_manifest(layout.metadata, audio_config.sample_rate,
                               require_single_speaker=False,
                               require_phonemes=require_phonemes,
                               supported_languages=layout.language_specs)
    items = list(report.items)
    previous = (
        _checkpoint_metadata(
            layout.initialization_checkpoint,
            warm_start=layout.initialization_mode == "warm_start",
        )
        if layout.initialization_checkpoint else None
    )
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
        if layout.initialization_mode in {"resume", "refine_text_prior"} \
                and set(speaker_map) != set(previous["speaker_map"]):
            raise ValueError(
                f"{layout.initialization_mode} cannot add speakers; "
                "use expand_speakers"
            )
        missing_old = set(previous["speaker_map"]) - current_speakers
        if layout.initialization_mode == "expand_speakers" and missing_old:
            warnings.warn(
                "old speakers are absent from the new metadata and may be forgotten: " + ", ".join(sorted(missing_old)),
                stacklevel=2,
            )
    # 先为 MAS 门禁建立宽松词表；过滤不兼容样本后再按初始化规则重建最终词表。
    # Build a permissive vocabulary for the MAS gate, then rebuild the final
    # vocabulary after rejected rows are removed.
    discovered = Vocabulary.build(items)
    if previous is None:
        vocabulary = discovered
    else:
        old_tokens = list(previous["tokens"])
        vocabulary = Vocabulary([
            *old_tokens,
            *(token for token in discovered.tokens if token not in old_tokens),
        ])
    piper_compatible = (
        frontend_contract.get("token_encoding")
        == PIPER_TOKEN_ENCODING
    )
    token_encoding = frontend_contract.get("token_encoding")
    logger.info(
        "frontend training contract provider=%s token_encoding=%s "
        "insert_piper_pads=%s mobile_wire_adapter=%s",
        frontend_contract.get("provider"), token_encoding,
        piper_compatible,
        (
            "strip-piper-pads-v1"
            if token_encoding == MOBILE_DIRECT_TOKEN_ENCODING else "none"
        ),
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
            reference_root = _semantic_reference_root(layout.dataset_dir)
            logger.info(
                "semantic quality gate status=started items=%d reference_root=%s",
                len(items), reference_root,
            )
            semantic_report = run_semantic_quality_gate(
                items, semantic_config,
                layout.run_dir / "quality" / "semantic-quality-report.json",
                reference_root=reference_root,
            )
            quality_summary["semantic"] = {
                key: semantic_report[key]
                for key in ("provider", "items", "passed", "failed", "failure_counts")
            }
            logger.info(
                "semantic quality gate status=completed passed=%d failed=%d",
                semantic_report["passed"], semantic_report["failed"],
            )

    alignment_config = quality_config.get("alignment", {})
    if alignment_config.get("enabled", True):
        logger.info("alignment quality gate status=started items=%d", len(items))
        alignment_results = [
            inspect_alignment_item(
                item, vocabulary, audio_config,
                piper_compatible=piper_compatible,
            )
            for item in items
        ]
        rejected = [
            result for result in alignment_results if not result["passed"]
        ]
        alignment_report = {
            "format": 1,
            "provider": "mas-alignment-capacity-v1",
            "items": len(alignment_results),
            "passed": len(alignment_results) - len(rejected),
            "failed": len(rejected),
            "piper_compatible": piper_compatible,
            "results": alignment_results,
        }
        alignment_target = (
            layout.run_dir / "quality" / "alignment-quality-report.json"
        )
        alignment_target.parent.mkdir(parents=True, exist_ok=True)
        alignment_target.write_text(
            json.dumps(alignment_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "alignment quality gate status=completed passed=%d failed=%d "
            "report=%s",
            alignment_report["passed"], alignment_report["failed"],
            alignment_target,
        )
        if rejected and alignment_config.get("fail_on_error", False):
            first = rejected[0]
            raise ValueError(
                "alignment quality gate rejected "
                f"{len(rejected)}/{len(items)} items; first={first['audio']} "
                f"audio_frames={first['audio_frames']} "
                f"text_tokens={first['text_tokens']}; see {alignment_target}"
            )
        if rejected:
            rejected_audio = {result["audio"] for result in rejected}
            items = [
                item for item in items if str(item.audio) not in rejected_audio
            ]
            logger.warning(
                "ALIGNMENT FILTER | excluded=%d | remaining=%d | reason=MAS "
                "requires audio_frames>=text_tokens",
                len(rejected), len(items),
            )
            remaining_languages = {item.language for item in items}
            remaining_speakers = {item.speaker for item in items}
            if remaining_languages != set(language_map):
                raise ValueError(
                    "alignment filtering removed every sample for languages: "
                    + ", ".join(sorted(set(language_map) - remaining_languages))
                )
            if remaining_speakers != current_speakers:
                raise ValueError(
                    "alignment filtering removed every sample for speakers: "
                    + ", ".join(sorted(current_speakers - remaining_speakers))
                )
        if quality_summary is None:
            quality_summary = {}
        quality_summary["alignment"] = {
            key: alignment_report[key]
            for key in ("provider", "items", "passed", "failed")
        }

    vocabulary = _vocabulary_for_initialization(
        items, layout.initialization_mode, previous,
    )
    frontend_conformance = (
        build_frontend_conformance(
            items, vocabulary, language_map,
            piper_compatible=piper_compatible,
        )
        if all(item.phonemes for item in items) else None
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
        if previous and layout.initialization_mode in {
            "resume", "refine_text_prior",
        } and previous.get("data_split"):
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
    config = preserve_checkpoint_duration_architecture(
        config, previous, layout.initialization_mode,
    )
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
    precision_name, autocast_dtype = resolve_mixed_precision(raw["training"], device)
    amp_enabled = autocast_dtype is not None
    # FP16 needs dynamic loss scaling. BF16 has FP32-like exponent range and
    # should not be scaled. Model/optimizer master weights remain FP32 in both.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=precision_name == "fp16",
    )
    logger.info(
        "MIXED PRECISION | configured=%s | active=%s | loss_scaler=%s | "
        "checkpoint_weights=fp32",
        raw["training"].get("mixed_precision", "auto"), precision_name,
        "dynamic" if scaler.is_enabled() else "disabled",
    )
    dataset = VitsDataset(
        train_items, vocabulary, speaker_map, language_map, audio_config,
        piper_compatible=piper_compatible,
    )
    profile_counts = Counter(
        (item.language, item.speaker) for item in train_items
    )
    weights = profile_balancing_weights(train_items)
    logger.info(
        "sampling strategy=equal-language-speaker-profile profiles=%d "
        "minimum_samples=%d maximum_samples=%d",
        len(profile_counts), min(profile_counts.values()),
        max(profile_counts.values()),
    )
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_items), replacement=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=raw["training"]["batch_size"], sampler=sampler,
        num_workers=raw["training"].get("num_workers", 0), collate_fn=collate_vits,
    )
    validation_loader = None
    if validation_items:
        validation_loader = torch.utils.data.DataLoader(
            VitsDataset(
                validation_items, vocabulary, speaker_map, language_map,
                audio_config, piper_compatible=piper_compatible,
            ),
            batch_size=int(validation_config.get("batch_size", raw["training"]["batch_size"])),
            shuffle=False, num_workers=raw["training"].get("num_workers", 0),
            collate_fn=collate_vits,
        )
    generator = MultilingualVITS(config).to(device)
    discriminator = VitsDiscriminator().to(device)
    generator_parameters = sum(parameter.numel() for parameter in generator.parameters())
    discriminator_parameters = sum(parameter.numel() for parameter in discriminator.parameters())
    duration_channels = (
        config.hidden_channels
        if config.duration_predictor_type == "stochastic_lognormal"
        else config.duration_predictor_channels
    )
    duration_flow_layers = (
        0 if config.duration_predictor_type == "stochastic_lognormal"
        else config.duration_predictor_flow_layers
    )
    logger.info(
        "model initialized generator_parameters=%d discriminator_parameters=%d "
        "duration_predictor=%s duration_channels=%d duration_flow_layers=%d "
        "train_batches_per_epoch=%d "
        "validation_batches=%d",
        generator_parameters, discriminator_parameters,
        config.duration_predictor_type, duration_channels,
        duration_flow_layers, len(loader),
        len(validation_loader) if validation_loader is not None else 0,
    )
    if generator_parameters < 10_000_000:
        logger.warning(
            "COMPACT MODEL | generator_parameters=%d | intended for pipeline/mobile-size "
            "validation; use configs/internal/quality_pipeline_defaults.json when sound "
            "quality matters",
            generator_parameters,
        )
    if raw.get("generation", {}).get("enabled", False) and not raw.get(
        "quality", {},
    ).get("semantic", {}).get("enabled", False):
        logger.warning(
            "SYNTHETIC DATA WITHOUT SEMANTIC GATE | ASR and speaker consistency checks are "
            "disabled; pronunciation errors and voice drift can be learned by the student"
        )
    if device.type == "cuda":
        logger.info(
            "GPU MEMORY | allocated=%.2f GiB | reserved=%.2f GiB | device=%s",
            torch.cuda.memory_allocated(device) / (1024 ** 3),
            torch.cuda.memory_reserved(device) / (1024 ** 3), device,
        )
    optimizer_g = torch.optim.AdamW(
        generator.parameters(), lr=raw["training"]["learning_rate_generator"],
        betas=(0.8, 0.99), eps=1e-9,
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(), lr=raw["training"]["learning_rate_discriminator"],
        betas=(0.8, 0.99), eps=1e-9,
    )
    refinement_config = resolve_refinement_config(raw["training"])
    if layout.initialization_mode == "refine_text_prior":
        if refinement_config["stage"] == "standard":
            raise ValueError(
                "training.stage=standard conflicts with "
                "initialization.mode=refine_text_prior"
            )
        refinement_config["enabled"] = True
    refinement_parameters = [
        parameter
        for module in (
            generator.text_encoder, generator.flow,
            generator.duration_predictor,
        )
        for parameter in module.parameters()
    ]
    optimizer_refinement = torch.optim.AdamW(
        refinement_parameters,
        lr=refinement_config["learning_rate"],
        betas=(0.8, 0.99), eps=1e-9,
    )
    lr_decay = float(raw["training"].get("lr_decay", 0.999875))
    if not 0.0 < lr_decay <= 1.0:
        raise ValueError("training.lr_decay must be in (0, 1]")
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=lr_decay)
    scheduler_refinement = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_refinement, gamma=refinement_config["lr_decay"],
    )
    start_epoch = 1
    global_step = 0
    refinement_active = layout.initialization_mode == "refine_text_prior"
    refinement_steps = 0
    refinement_ready_streak = 0
    refinement_activated_at_step = 0 if refinement_active else None
    refinement_activated_at_epoch = 1 if refinement_active else None
    if layout.initialization_mode == "resume":
        restored = load_training_checkpoint(
            layout.initialization_checkpoint, generator=generator, discriminator=discriminator,
            optimizer_g=optimizer_g, optimizer_d=optimizer_d,
            scheduler_g=scheduler_g, scheduler_d=scheduler_d,
            optimizer_refinement=optimizer_refinement,
            scheduler_refinement=scheduler_refinement,
            scaler=scaler,
        )
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
        restored_phase = restored.get("training_phase") or {}
        refinement_active = (
            restored_phase.get("stage") == "text_prior_refinement"
        )
        if refinement_active:
            if refinement_config["stage"] == "standard":
                raise ValueError(
                    "cannot resume a text_prior_refinement checkpoint with "
                    "training.stage=standard; use warm_start to unfreeze the "
                    "complete acoustic model"
                )
            # Checkpoint stage is authoritative: silently returning to full GAN
            # training would unfreeze the acoustic path that refinement was
            # deliberately protecting.
            # checkpoint 阶段优先，避免 resume 时悄悄解冻并破坏声学主链。
            refinement_config["enabled"] = True
        refinement_steps = int(restored_phase.get("refinement_steps", 0))
        refinement_ready_streak = int(
            restored_phase.get("acoustic_ready_streak", 0)
        )
        refinement_activated_at_step = restored_phase.get("activated_at_step")
        refinement_activated_at_epoch = restored_phase.get("activated_at_epoch")
        for optimizer in (optimizer_g, optimizer_d, optimizer_refinement):
            _optimizer_to(optimizer, device)
    elif layout.initialization_mode in {"warm_start", "refine_text_prior"}:
        warm_start_report = _load_warm_start_generator(
            generator, layout.initialization_checkpoint,
            (
                layout.initialization_exclude
                if layout.initialization_mode == "warm_start" else ()
            ),
            discriminator=discriminator,
        )
        logger.info(
            "%s | checkpoint=%s | format=%d | loaded_tensors=%d | "
            "skipped_tensors=%d | discriminator_loaded=%s | excluded=%s | "
            "duration=%s->%s | epoch_reset=1 | step_reset=0",
            (
                "TEXT PRIOR REFINEMENT START"
                if layout.initialization_mode == "refine_text_prior"
                else "WARM START"
            ),
            layout.initialization_checkpoint,
            warm_start_report["checkpoint_format"],
            warm_start_report["loaded_tensors"],
            warm_start_report["skipped_tensors"],
            warm_start_report["discriminator_loaded"],
            ",".join(warm_start_report["excluded_modules"]) or "none",
            warm_start_report["duration_predictor_from"],
            warm_start_report["duration_predictor_to"],
            extra={"tts_style": "success"},
        )
    elif layout.initialization_mode == "expand_speakers":
        _load_expanded_generator(generator, layout.initialization_checkpoint)
    training_stage = (
        "text_prior_refinement" if refinement_active else "standard"
    )
    if (
        refinement_config["enabled"]
        and not refinement_active
        and refinement_config["posterior_mel_threshold"] is not None
        and validation_loader is None
    ):
        raise ValueError(
            "automatic text-prior refinement needs validation.enabled=true; "
            "enable validation or set posterior_mel_threshold=null explicitly"
        )
    configure_training_stage(generator, discriminator, training_stage)
    logger.info(
        "TRAINING STAGE | stage=%s | policy=%s | refinement_enabled=%s | "
        "start_steps=%d | posterior_mel_threshold=%s | "
        "full_posterior_mel_threshold=%s | require_all_profiles=%s | "
        "consecutive_passes=%d | acoustic_passes=%d | refinement_steps=%d",
        training_stage, refinement_config["stage"], refinement_config["enabled"],
        refinement_config["start_steps"],
        refinement_config["posterior_mel_threshold"],
        refinement_config["full_posterior_mel_threshold"],
        refinement_config["require_all_profiles"],
        refinement_config["consecutive_passes"], refinement_ready_streak,
        refinement_steps,
        extra={"tts_style": "section"},
    )
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_config.sample_rate, n_fft=audio_config.n_fft,
        win_length=audio_config.win_length, hop_length=audio_config.hop_length,
        n_mels=audio_config.n_mels, center=False, power=audio_config.mel_power,
    ).to(device)
    logger.info(
        "loss setup mel_power=%.1f segment_frames=%d segment_samples=%d",
        audio_config.mel_power, config.segment_frames,
        config.segment_frames * audio_config.hop_length,
    )
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
    if selection_metric not in {
        "mel", "prior_mel", "combined_mel", "duration", "kl", "total",
    }:
        raise ValueError(
            "validation.metric must be mel, prior_mel, combined_mel, "
            "duration, kl, or total"
        )
    previous_selection = (
        previous.get("selection")
        if previous and layout.initialization_mode == "resume" else None
    )
    if previous_selection:
        inherited_best = inherit_resume_best_checkpoint(
            layout.initialization_checkpoint,
            destination / "best",
            previous_selection,
        )
        if inherited_best is not None:
            logger.info(
                "BEST CHECKPOINT INHERITED | epoch=%s | metric=%s | "
                "value=%s | path=%s",
                previous_selection.get("best_epoch"),
                previous_selection.get("metric"),
                previous_selection.get("best_value"), inherited_best,
                extra={"tts_style": "success"},
            )
        else:
            logger.warning(
                "BEST CHECKPOINT NOT FOUND | checkpoint=%s | best_epoch=%s | "
                "the resumed run will create best after validation improves",
                layout.initialization_checkpoint,
                previous_selection.get("best_epoch"),
            )
    best_value = float("inf")
    best_epoch = None
    if previous_selection and previous_selection.get("metric") == selection_metric:
        best_value = float(previous_selection.get("best_value", best_value))
        best_epoch = previous_selection.get("best_epoch")
    evaluation_every = int(validation_config.get("every_epochs", 1))
    if evaluation_every < 1:
        raise ValueError("validation.every_epochs must be at least 1")
    checkpoint_every_epochs = int(raw["training"].get("checkpoint_every_epochs", 1))
    if checkpoint_every_epochs < 1:
        raise ValueError("training.checkpoint_every_epochs must be at least 1")
    # Smoke tests should visibly advance instead of appearing frozen between
    # step 1 and the configured logging interval. Full training still honors
    # training.log_every_steps to avoid flooding long-running server logs.
    effective_log_every = log_every
    if max_steps is not None:
        effective_log_every = min(log_every, max(1, max_steps // 10))
    run_start_step = global_step
    remaining_epochs = int(raw["training"]["epochs"]) - start_epoch + 1
    planned_run_steps = (
        int(max_steps) if max_steps is not None
        else max(remaining_epochs, 0) * len(loader)
    )
    if planned_run_steps < 1:
        raise ValueError("training plan must contain at least one remaining step")
    target_global_step = run_start_step + planned_run_steps
    training_started = time.monotonic()
    last_log_time = training_started
    last_log_step = global_step
    recent_step_times = deque(maxlen=20)
    live_progress = TerminalProgress(
        "TRAIN", planned_run_steps,
        enabled=raw.get("logging", {}).get("live_progress"),
    )
    logger.info(
        "training plan epochs=%d start_epoch=%d remaining_epochs=%d "
        "initial_global_step=%d run_steps=%d target_global_step=%d max_steps=%s "
        "log_every_steps=%d "
        "batch_size=%d checkpoint_every_steps=%d checkpoint_every_epochs=%d "
        "validation_every_epochs=%d",
        raw["training"]["epochs"], start_epoch, remaining_epochs,
        run_start_step, planned_run_steps, target_global_step,
        max_steps if max_steps is not None else "unlimited",
        effective_log_every, raw["training"]["batch_size"],
        raw["training"].get("checkpoint_every_steps", 5000),
        checkpoint_every_epochs, evaluation_every,
    )
    if planned_run_steps >= 100_000:
        logger.warning(
            "LARGE TRAINING PLAN | run_steps=%d | use the rolling ETA after warm-up to "
            "decide whether training.epochs should be reduced",
            planned_run_steps,
        )
    elif (
        max_steps is None
        and layout.initialization_mode in {"scratch", "warm_start"}
        and planned_run_steps < 50_000
    ):
        logger.warning(
            "SHORT TRAINING PLAN | run_steps=%d | often insufficient for a VITS model "
            "trained from scratch; inspect validation audio before accepting the export",
            planned_run_steps,
        )
    live_progress.update(0, "warming up first batches")

    def training_phase_snapshot() -> dict:
        return {
            "stage": training_stage,
            "mixed_precision": precision_name,
            "refinement_steps": refinement_steps,
            "acoustic_ready_streak": refinement_ready_streak,
            "activated_at_step": refinement_activated_at_step,
            "activated_at_epoch": refinement_activated_at_epoch,
            "config": refinement_config,
        }

    for epoch in range(start_epoch, raw["training"]["epochs"] + 1):
        live_progress.clear()
        logger.info(
            "EPOCH START | epoch=%d/%d | batches=%d | global_step=%d",
            epoch, raw["training"]["epochs"], len(loader), global_step,
        )
        epoch_totals = Counter()
        epoch_steps = 0
        epoch_standard_steps = 0
        epoch_refinement_steps = 0
        for batch_index, batch in enumerate(loader, 1):
            step_started = time.monotonic()
            batch = {key: value.to(device) for key, value in batch.items()}
            if training_stage == "standard":
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype,
                    enabled=amp_enabled,
                ):
                    output = generator(
                        batch["tokens"], batch["text_lengths"],
                        batch["spectrograms"], batch["spec_lengths"],
                        batch["language_ids"], batch["speaker_ids"],
                    )
                real_audio = slice_waveforms(
                    batch["waveforms"], output.slice_starts,
                    config.segment_frames, audio_config.hop_length,
                )
                optimizer_d.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype,
                    enabled=amp_enabled,
                ):
                    real_d = discriminator(real_audio)
                    fake_d = discriminator(output.audio.detach())
                    loss_d = discriminator_loss(real_d, fake_d)
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)

                optimizer_g.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype,
                    enabled=amp_enabled,
                ):
                    with torch.no_grad():
                        real_features = discriminator(real_audio)
                    fake_features = discriminator(output.audio)
                    adversarial = generator_adversarial_loss(fake_features)
                    feature_matching = feature_matching_loss(
                        real_features, fake_features,
                    )
                # Spectral/log/probability reductions stay FP32 even when the
                # convolution-heavy networks use BF16/FP16 Tensor Cores.
                with torch.autocast(device_type=device.type, enabled=False):
                    mel_real = torch.log(
                        mel_transform(real_audio.squeeze(1).float()).clamp_min(1e-5)
                    )
                    mel_fake = torch.log(
                        mel_transform(output.audio.squeeze(1).float()).clamp_min(1e-5)
                    )
                    loss_mel = F.l1_loss(mel_fake, mel_real)
                    loss_kl = kl_loss(
                        output.latent_prior.float(),
                        output.posterior_log_scale.float(),
                        output.prior_mean.float(), output.prior_log_scale.float(),
                        output.audio_mask.float(),
                    )
                    loss_duration = output.duration_loss.float()
                    loss_duration_mean = loss_mel.new_zeros(())
                    effective_prior_weight = 0.0
                    loss_g = (
                        45.0 * loss_mel + loss_duration + loss_kl
                        + adversarial.float() + 2.0 * feature_matching.float()
                    )
                scaler.scale(loss_g).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
                scaler.step(optimizer_g)
                scaler.update()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                posterior_scale = output.posterior_log_scale.exp().mean()
                current_lr = optimizer_g.param_groups[0]["lr"]
                epoch_standard_steps += 1
            else:
                generator.zero_grad(set_to_none=True)
                optimizer_refinement.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype,
                    enabled=amp_enabled,
                ):
                    output = generator.forward_text_refinement(
                        batch["tokens"], batch["text_lengths"],
                        batch["spectrograms"], batch["spec_lengths"],
                        batch["language_ids"], batch["speaker_ids"],
                    )
                real_audio = slice_waveforms(
                    batch["waveforms"], output.slice_starts,
                    config.segment_frames, audio_config.hop_length,
                )
                with torch.autocast(device_type=device.type, enabled=False):
                    mel_real = torch.log(
                        mel_transform(real_audio.squeeze(1).float()).clamp_min(1e-5)
                    )
                    mel_fake = torch.log(
                        mel_transform(output.audio.squeeze(1).float()).clamp_min(1e-5)
                    )
                    loss_mel = F.l1_loss(mel_fake, mel_real)
                    loss_kl = kl_loss(
                        output.latent_prior.float(),
                        output.posterior_log_scale.float(),
                        output.prior_mean.float(), output.prior_log_scale.float(),
                        output.audio_mask.float(),
                    )
                    loss_duration = output.duration_loss.float()
                    loss_duration_mean = output.duration_mean_loss.float()
                    effective_prior_weight = refinement_mel_weight(
                        refinement_config, refinement_steps + 1,
                    )
                    loss_g = (
                        effective_prior_weight * loss_mel
                        + refinement_config["kl_weight"] * loss_kl
                        + refinement_config["duration_nll_weight"]
                        * loss_duration
                        + refinement_config["duration_mean_weight"]
                        * loss_duration_mean
                    )
                scaler.scale(loss_g).backward()
                scaler.unscale_(optimizer_refinement)
                torch.nn.utils.clip_grad_norm_(
                    refinement_parameters, 5.0,
                )
                scaler.step(optimizer_refinement)
                scaler.update()
                loss_d = loss_g.new_zeros(())
                posterior_scale = output.posterior_log_scale.exp().mean()
                current_lr = optimizer_refinement.param_groups[0]["lr"]
                refinement_steps += 1
                epoch_refinement_steps += 1
            global_step += 1
            run_step = global_step - run_start_step
            epoch_steps += 1
            recent_step_times.append(time.monotonic() - step_started)
            epoch_totals.update({
                "generator": float(loss_g.item()),
                "discriminator": float(loss_d.item()),
                "mel": float(loss_mel.item()),
                "duration": float(loss_duration.item()),
                "duration_mean": float(loss_duration_mean.item()),
                "kl": float(loss_kl.item()),
            })
            rolling_step_time = median(recent_step_times)
            remaining_steps = max(planned_run_steps - run_step, 0)
            eta = (
                format_duration(remaining_steps * rolling_step_time)
                if len(recent_step_times) >= 3 else "warming-up"
            )
            live_detail = (
                f"epoch={epoch}/{raw['training']['epochs']} "
                f"stage={training_stage} batch={batch_index}/{len(loader)} "
                f"ETA={eta} mel={loss_mel.item():.4f}"
            )
            live_progress.update(run_step, live_detail)
            if run_step == 1 or global_step % effective_log_every == 0:
                live_progress.clear()
                now = time.monotonic()
                steps_since_log = max(global_step - last_log_step, 1)
                seconds_per_step = (now - last_log_time) / steps_since_log
                overall_progress = min(
                    run_step / max(planned_run_steps, 1) * 100.0, 100.0,
                )
                logger.info(
                    "TRAIN %s %6.2f%% | epoch=%d/%d | batch=%d/%d | "
                    "stage=%s | "
                    "run_step=%d/%d | global_step=%d/%d | "
                    "step_time=%.2fs | rolling=%.2fs | speed=%.2f steps/min | ETA=%s | "
                    "generator=%.4f | discriminator=%.4f | mel=%.4f | "
                    "posterior_scale=%.3f | "
                    "duration=%.4f | duration_mean=%.4f | kl=%.4f | "
                    "prior_weight=%.4f | lr=%.8f",
                    progress_bar(run_step, planned_run_steps), overall_progress,
                    epoch, raw["training"]["epochs"],
                    batch_index, len(loader), training_stage,
                    run_step, planned_run_steps,
                    global_step, target_global_step,
                    seconds_per_step, rolling_step_time, 60.0 / max(rolling_step_time, 1e-9),
                    eta, loss_g.item(), loss_d.item(), loss_mel.item(),
                    posterior_scale.item(), loss_duration.item(),
                    loss_duration_mean.item(), loss_kl.item(),
                    effective_prior_weight, current_lr,
                    extra={"tts_style": "progress"},
                )
                last_log_time = now
                last_log_step = global_step
                live_progress.update(run_step, live_detail)
            checkpoint_every = raw["training"].get("checkpoint_every_steps", 5000)
            if global_step % checkpoint_every == 0:
                live_progress.clear()
                logger.info("CHECKPOINT SAVE | step=%d", global_step)
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
                    audio=audio_config,
                    scheduler_g=scheduler_g, scheduler_d=scheduler_d,
                    scaler=scaler,
                    optimizer_refinement=optimizer_refinement,
                    scheduler_refinement=scheduler_refinement,
                    training_phase=training_phase_snapshot(),
                    metrics={
                        "stage": training_stage,
                        "generator": loss_g.item(),
                        "discriminator": loss_d.item(),
                        "mel": loss_mel.item(),
                        "duration": loss_duration.item(),
                        "duration_mean": loss_duration_mean.item(),
                        "kl": loss_kl.item(),
                        "prior_weight": effective_prior_weight,
                    },
                )
                logger.info(
                    "CHECKPOINT SAVED | step=%d | path=%s",
                    global_step, destination / f"step-{global_step:09d}",
                    extra={"tts_style": "success"},
                )
                live_progress.update(run_step, live_detail)
            if max_steps is not None and run_step >= max_steps:
                break
        train_metrics = {
            key: value / max(epoch_steps, 1) for key, value in epoch_totals.items()
        }
        if epoch_standard_steps:
            scheduler_g.step()
            scheduler_d.step()
        if epoch_refinement_steps:
            scheduler_refinement.step()
        logger.info(
            "LEARNING RATE | epoch=%d | stage=%s | generator=%.8f | "
            "discriminator=%.8f | refinement=%.8f",
            epoch, training_stage, scheduler_g.get_last_lr()[0],
            scheduler_d.get_last_lr()[0],
            scheduler_refinement.get_last_lr()[0],
        )
        validation_metrics = None
        should_evaluate = validation_loader is not None and (
            epoch == start_epoch or epoch % evaluation_every == 0
            or (
                max_steps is not None
                and global_step - run_start_step >= max_steps
            )
        )
        if should_evaluate:
            live_progress.clear()
            logger.info(
                "VALIDATION START | epoch=%d | batches=%d",
                epoch, len(validation_loader),
            )
            preview_every = int(validation_config.get("preview_every_epochs", 10))
            if preview_every < 1:
                raise ValueError("validation.preview_every_epochs must be at least 1")
            preview_dir = (
                layout.run_dir / "validation-audio" / f"epoch-{epoch:04d}"
                if epoch == start_epoch or epoch % preview_every == 0
                or (
                    max_steps is not None
                    and global_step - run_start_step >= max_steps
                )
                else None
            )
            validation_metrics = evaluate_validation(
                generator, validation_loader, mel_transform, audio_config, config, device,
                seed=int(validation_config.get("seed", seed)),
                preview_dir=preview_dir,
                language_map=language_map,
                speaker_map=speaker_map,
            )
            current_value = float(validation_metrics[selection_metric])
            logger.info(
                "VALIDATION DONE | epoch=%d | mel=%.4f | prior_mel=%.4f | "
                "combined_mel=%.4f | duration=%.4f | kl=%.4f | total=%.4f",
                epoch, validation_metrics["mel"], validation_metrics["prior_mel"],
                validation_metrics["combined_mel"], validation_metrics["duration"],
                validation_metrics["kl"],
                validation_metrics["total"],
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
                logger.info(
                    "BEST CHECKPOINT SAVE START | epoch=%d | metric=%s | value=%.6f",
                    epoch, selection_metric, best_value,
                )
                save_training_checkpoint(
                    destination / "best", generator=generator, discriminator=discriminator,
                    optimizer_g=optimizer_g, optimizer_d=optimizer_d, epoch=epoch,
                    global_step=global_step, config=config, language_map=language_map,
                    speaker_map=speaker_map, tokens=vocabulary.tokens,
                    frontend=frontend_contract,
                    frontend_conformance=frontend_conformance,
                    selection=selection, data_split=split_report,
                    quality_summary=quality_summary,
                    audio=audio_config,
                    scheduler_g=scheduler_g, scheduler_d=scheduler_d,
                    scaler=scaler,
                    optimizer_refinement=optimizer_refinement,
                    scheduler_refinement=scheduler_refinement,
                    training_phase=training_phase_snapshot(),
                    metrics={"train": train_metrics, "validation": validation_metrics},
                )
                logger.info(
                    "BEST CHECKPOINT | epoch=%d | metric=%s | value=%.6f | path=%s",
                    epoch, selection_metric, best_value, destination / "best",
                    extra={"tts_style": "success"},
                )
        if (
            training_stage == "standard"
            and refinement_config["enabled"]
            and global_step >= refinement_config["start_steps"]
        ):
            posterior_ready, gate_failures = acoustic_refinement_gate(
                validation_metrics, refinement_config,
            )
            if posterior_ready:
                refinement_ready_streak += 1
            elif validation_metrics is not None:
                refinement_ready_streak = 0
            required_passes = refinement_config["consecutive_passes"]
            if posterior_ready and refinement_ready_streak >= required_passes:
                training_stage = "text_prior_refinement"
                refinement_active = True
                refinement_activated_at_step = global_step
                refinement_activated_at_epoch = epoch
                configure_training_stage(
                    generator, discriminator, training_stage,
                )
                logger.info(
                    "TEXT PRIOR REFINEMENT ACTIVATED | epoch=%d | step=%d | "
                    "acoustic_passes=%d/%d | profiles=%d | frozen="
                    "conditioning,posterior_encoder,decoder,discriminator | "
                    "trainable=text_encoder,flow,duration_predictor",
                    epoch, global_step, refinement_ready_streak,
                    required_passes,
                    len((validation_metrics or {}).get("profiles") or {}),
                    extra={"tts_style": "success"},
                )
            elif validation_metrics is not None:
                logger.info(
                    "TEXT PRIOR REFINEMENT WAIT | epoch=%d | step=%d | "
                    "acoustic_passes=%d/%d | failures=%s",
                    epoch, global_step, refinement_ready_streak,
                    required_passes,
                    "; ".join(gate_failures[:8]) or "waiting for stable passes",
                )
        selection = {
            "metric": selection_metric,
            "best_value": best_value if math.isfinite(best_value) else None,
            "best_epoch": best_epoch,
            "mode": "min",
        } if validation_enabled else {"enabled": False}
        reached_max_steps = max_steps is not None \
            and global_step - run_start_step >= max_steps
        final_epoch = epoch == raw["training"]["epochs"]
        should_save_last = (
            epoch == start_epoch or epoch % checkpoint_every_epochs == 0
            or reached_max_steps or final_epoch
        )
        live_progress.clear()
        if should_save_last:
            logger.info(
                "LAST CHECKPOINT SAVE START | epoch=%d | step=%d | path=%s",
                epoch, global_step, destination / "last",
            )
            save_training_checkpoint(
                destination / "last", generator=generator, discriminator=discriminator,
                optimizer_g=optimizer_g, optimizer_d=optimizer_d, epoch=epoch,
                global_step=global_step, config=config, language_map=language_map,
                speaker_map=speaker_map, tokens=vocabulary.tokens, frontend=frontend_contract,
                frontend_conformance=frontend_conformance,
                selection=selection, data_split=split_report,
                quality_summary=quality_summary,
                audio=audio_config,
                scheduler_g=scheduler_g, scheduler_d=scheduler_d,
                scaler=scaler,
                optimizer_refinement=optimizer_refinement,
                scheduler_refinement=scheduler_refinement,
                training_phase=training_phase_snapshot(),
                metrics={"train": train_metrics, "validation": validation_metrics},
            )
            logger.info(
                "LAST CHECKPOINT SAVED | epoch=%d/%d | step=%d | total_elapsed=%s | checkpoint=%s",
                epoch, raw["training"]["epochs"], global_step,
                format_duration(time.monotonic() - training_started), destination / "last",
                extra={"tts_style": "success"},
            )
            live_progress.update(
                global_step - run_start_step, f"epoch={epoch} checkpoint saved",
            )
        else:
            logger.info(
                "EPOCH DONE | epoch=%d/%d | step=%d | checkpoint=skipped "
                "(saved every %d epochs)",
                epoch, raw["training"]["epochs"], global_step,
                checkpoint_every_epochs,
            )
            live_progress.update(
                global_step - run_start_step, f"epoch={epoch} completed",
            )
        if reached_max_steps: break
    live_progress.close()
    logger.info(
        "TRAINING DONE | steps=%d | stage=%s | refinement_steps=%d | "
        "elapsed=%s | checkpoint=%s",
        global_step, training_stage, refinement_steps,
        format_duration(time.monotonic() - training_started),
        destination / "last",
        extra={"tts_style": "success"},
    )
    return destination / "last"
