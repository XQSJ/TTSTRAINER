"""配置驱动的全流程总编排：数据 → 前端 → 训练 → 导出。

run_pipeline 按以下阶段顺序执行（括号内为对应模块与函数）：

    preflight        环境与配置自检
        │
    generate_texts   LLM 生成语料（text_generation.generate_texts）
        │            ── 可选阶段；多音色（multi-speaker）时跳过
    generate_samples Qwen3-TTS 教师蒸馏语音样本
        │            （sample_generation.generate_samples）
    phonemize        G2P 音素化 + 契约冻结
        │            （frontend.phonemize_manifest）
    validate         元数据/WAV/质量门校验
    train            VITS GAN 训练（vits.trainer.train_vits）
        │
    export           Piper 形状 ONNX 导出
                   （vits.exporter.export_vits_onnx）

跳过条件：
- task=prepare 时，phonemize 及之后阶段全部跳过（只准备数据）。
- generate_texts 受 pipeline 配置开关控制，且多音色数据集不适用。
- 各阶段均可由 pipeline 配置独立开关。

English: run_pipeline is the config-driven orchestrator wiring every
stage (preflight → generate_texts → generate_samples → phonemize →
validate → train → export); task=prepare stops before phonemize.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import csv


def _english_corpus_texts(layout) -> list[str]:
    """读取数据集清单中英文语料的 text 列，供导出补全 OOV 词典。 / Read the English corpus text column from the dataset manifest for the OOV dictionary supplement."""
    manifest = Path(layout.metadata)
    if not manifest.is_file():
        return []
    texts: list[str] = []
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("language", "")).strip().lower() == "en":
                text = str(row.get("text", "")).strip()
                if text:
                    texts.append(text)
    return texts


def _gate_custom_words(config_path) -> dict:
    """读取定制词确认闸门的定稿产物；未配置时为空（零影响）。 / Load the
    confirmation gate's verified output; empty when unconfigured."""
    from .custom_words import load_custom_words, has_custom_words, load_verified
    if not has_custom_words(load_custom_words(config_path)):
        return {}
    verified = load_verified(config_path)
    if not verified:
        # 配置了定制词但闸门未跑/未确认任何词：明确拒绝导出，防止自动读音
        # 漏进生产。 / Custom words exist but the gate never confirmed any:
        # refuse to export so no unconfirmed reading slips into production.
        raise RuntimeError(
            "custom_words.json exists but custom_words_verified.json is empty; "
            "run: python -m tts_trainer custom-words --config " + str(config_path)
        )
    return verified

from .experiments import prepare_experiment, resolve_experiment
from .frontend import frontend_from_config, phonemize_manifest
from .language_check import check_language_support
from .logging_utils import (configure_logging_from_config, format_duration,
                            log_section)
from .manifest import validate_manifest
from .sample_generation import generate_samples
from .text_generation import generate_texts, validate_text_generation_config
from .vits.exporter import export_vits_onnx, validate_onnx_runtime
from .vits.trainer import train_vits


logger = logging.getLogger(__name__)


def run_pipeline(config_path: str | Path, *, max_steps: int | None = None) -> Path:
    """执行配置驱动的完整工作流。 / Run the configured dataset → frontend → train → export workflow."""
    raw, layout = resolve_experiment(config_path)
    configure_logging_from_config(raw)
    prepare_experiment(layout, raw, config_path)
    stages = raw.get("pipeline", {})
    generation = raw.get("generation", {})
    text_generation = raw.get("text_generation", {})
    task = raw.get("task", "train")
    multiple_voices = bool(generation.get("voices"))
    # 组装启用阶段列表；task=prepare 跳过 phonemize 及其后的训练阶段。
    # Assemble the stage list; task=prepare skips phonemize and all training stages.
    active_stages = ["preflight"]
    if stages.get("generate_texts", True) \
            and text_generation.get("enabled", False) and not multiple_voices:
        # 多音色配置的文本已按音色管理，文本生成阶段不适用。 / Multi-voice configs manage per-voice texts already.
        active_stages.append("generate_texts")
    if stages.get("generate_samples", True) and generation.get("enabled", True):
        active_stages.append("generate_samples")
    if task == "train":
        for name in ("phonemize", "validate", "train", "export"):
            if stages.get(name, True):
                active_stages.append(name)
    stage_numbers = {name: index for index, name in enumerate(active_stages, 1)}
    pipeline_started = time.monotonic()
    log_section(
        logger,
        "TTS TRAINING PIPELINE",
        f"Model: {layout.name}\n"
        f"Task: {task}\n"
        f"Languages: {', '.join(layout.languages)}\n"
        f"Stages: {' | '.join(active_stages)}",
    )

    def stage_started(name: str, description: str) -> float:
        """打印阶段横幅并返回计时起点。 / Log the stage banner and return the timer origin."""
        log_section(
            logger,
            f"STAGE {stage_numbers[name]}/{len(active_stages)}  {name.upper()}",
            description,
        )
        return time.monotonic()

    def stage_completed(name: str, started: float, detail: str) -> None:
        """记录阶段耗时与结果摘要。 / Log elapsed time and the stage detail."""
        elapsed = time.monotonic() - started
        logger.info(
            "STAGE DONE %d/%d | %s | elapsed=%s | %s",
            stage_numbers[name], len(active_stages), name,
            format_duration(elapsed), detail,
            extra={"tts_style": "success"},
        )

    logger.info("pipeline plan total_stages=%d stages=%s", len(active_stages), ",".join(active_stages))
    if stages.get("generate_texts", True) \
            and text_generation.get("enabled", False) and not multiple_voices:
        validate_text_generation_config(text_generation)
    stage_time = stage_started("preflight", "check language, teacher and G2P readiness")
    statuses = check_language_support(
        raw, layout,
        run_smoke=bool(stages.get("phonemize", True)),
        require_teacher=bool(
            stages.get("generate_samples", True)
            and generation.get("enabled", True)
            and (generation.get("voice") or generation.get("voices"))
        ),
    )
    for status in statuses:
        if status.ready:
            logger.info(
                "language ready code=%s teacher=%s g2p=%s:%s preview=%s",
                status.code, status.teacher, status.frontend, status.voice,
                status.phoneme_preview,
            )
        else:
            logger.error("language failed code=%s error=%s", status.code, status.error)
    failed = [status for status in statuses if not status.ready]
    if failed:
        raise RuntimeError("language preflight failed: " + "; ".join(
            f"{status.code}: {status.error}" for status in failed
        ))
    stage_completed("preflight", stage_time, f"languages={len(statuses)} ready={len(statuses)}")
    report = {
        "name": layout.name,
        "task": task,
        "config": str(Path(config_path).resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    text_manifest = Path(generation.get("text_manifest") or layout.dataset_dir / "texts.csv")
    if stages.get("generate_texts", True) \
            and text_generation.get("enabled", False) and not multiple_voices:
        stage_time = stage_started("generate_texts", "prepare or reuse multilingual training texts")
        # generate_texts 内部自行复用已生成文本，实现断点续跑。 / generate_texts reuses existing texts for resumability.
        text_manifest = generate_texts(config_path)
        report["stages"]["generate_texts"] = str(text_manifest.resolve())
        stage_completed("generate_texts", stage_time, f"output={text_manifest}")
    else:
        report["stages"]["generate_texts"] = "skipped"

    raw_metadata = Path(generation.get("raw_metadata") or layout.dataset_dir / "metadata.csv")
    if stages.get("generate_samples", True) and generation.get("enabled", True):
        stage_time = stage_started("generate_samples", "generate or reuse teacher WAV samples")
        # 多音色时文本由各音色自带，不传统一清单。 / Multi-voice runs carry per-voice texts, no shared manifest.
        raw_metadata = generate_samples(
            config_path,
            text_manifest_path=None if multiple_voices else text_manifest,
        )
        report["stages"]["generate_samples"] = str(raw_metadata.resolve())
        stage_completed("generate_samples", stage_time, f"output={raw_metadata}")
    else:
        report["stages"]["generate_samples"] = "skipped"

    if task == "train" and stages.get("phonemize", True):
        stage_time = stage_started("phonemize", "normalize text and convert it to language-specific phonemes")
        # 前端由训练配置解析，保证音素与训练时一致。 / The frontend is resolved from the training config for phoneme consistency.
        frontend = frontend_from_config(
            raw.get("frontend"), languages=layout.languages,
            language_registry=raw.get("language_registry"),
        )
        phonemize_manifest(raw_metadata, layout.metadata, frontend)
        report["stages"]["phonemize"] = str(layout.metadata.resolve())
        stage_completed("phonemize", stage_time, f"output={layout.metadata}")
    else:
        report["stages"]["phonemize"] = "skipped"

    if task == "train" and stages.get("validate", True):
        stage_time = stage_started("validate", "validate audio, metadata, languages and phonemes")
        validation = validate_manifest(
            layout.metadata,
            int(raw["audio"]["sample_rate"]),
            require_single_speaker=False,
            require_phonemes=bool(raw.get("frontend", {}).get("require_phonemes", True)),
            supported_languages=layout.language_specs,
        )
        # 双向核对：多余语言和缺失语言都视为配置错误。 / Check both directions: extra and missing languages fail.
        outside = sorted({item.language for item in validation.items} - set(layout.languages))
        if outside:
            raise ValueError(
                "metadata contains languages not enabled by experiment.languages: " + ", ".join(outside)
            )
        missing = sorted(set(layout.languages) - {item.language for item in validation.items})
        if missing:
            raise ValueError("metadata has no samples for configured languages: " + ", ".join(missing))
        report["stages"]["validate"] = {
            "samples": len(validation.items),
            "enabled_languages": list(layout.languages),
            "languages": validation.language_counts,
        }
        stage_completed(
            "validate", stage_time,
            f"samples={len(validation.items)} counts={validation.language_counts}",
        )
    else:
        report["stages"]["validate"] = "skipped"

    checkpoint = layout.checkpoints_dir / "last"
    if task == "train" and stages.get("train", True):
        stage_time = stage_started("train", "quality gate, dataset split and VITS optimization")
        # 训练内部处理质量门控、数据切分与断点续训。 / Training itself handles quality gating, splitting and resume.
        checkpoint = train_vits(str(config_path), max_steps=max_steps)
        report["stages"]["train"] = str(checkpoint.resolve())
        stage_completed("train", stage_time, f"checkpoint={checkpoint}")
    else:
        report["stages"]["train"] = "skipped"

    if task == "train" and stages.get("export", True):
        stage_time = stage_started("export", "load checkpoint, export ONNX and validate runtime")
        requested_checkpoint = raw.get("validation", {}).get("export_checkpoint", "last")
        if requested_checkpoint not in {"best", "last"}:
            raise ValueError("validation.export_checkpoint must be best or last")
        preferred = layout.checkpoints_dir / requested_checkpoint
        if preferred.is_dir():
            checkpoint = preferred
        elif requested_checkpoint == "best":
            # best 缺失时回退 last 并告警。 / Fall back to last with a warning when best is missing.
            logger.warning("best checkpoint is unavailable; exporting last checkpoint")
        model = export_vits_onnx(checkpoint, layout.artifacts_dir,
                                 sample_rate=int(raw["audio"]["sample_rate"]),
                                 corpus_texts=_english_corpus_texts(layout),
                                 verified_custom_words=_gate_custom_words(config_path))
        report["stages"]["export"] = str(model.resolve())
        report["stages"]["export_checkpoint"] = str(checkpoint.resolve())
        if stages.get("validate_onnx", True):
            report["stages"]["validate_onnx"] = list(validate_onnx_runtime(model))
        stage_completed("export", stage_time, f"model={model}")
    else:
        report["stages"]["export"] = "skipped"

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    # 汇总报告记录每个阶段的输出或 skipped，供断点与审计使用。 / The report records per-stage outputs or skips for audit/resume.
    destination = layout.run_dir / "pipeline-report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_section(
        logger,
        "PIPELINE COMPLETED",
        f"Elapsed: {format_duration(time.monotonic() - pipeline_started)}\n"
        f"Report: {destination}",
        success=True,
    )
    return destination
