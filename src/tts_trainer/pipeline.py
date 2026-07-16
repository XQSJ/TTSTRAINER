from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .experiments import prepare_experiment, resolve_experiment
from .frontend import frontend_from_config, phonemize_manifest
from .language_check import check_language_support
from .logging_utils import configure_logging
from .manifest import validate_manifest
from .sample_generation import generate_samples
from .text_generation import generate_texts
from .vits.exporter import export_vits_onnx, validate_onnx_runtime
from .vits.trainer import train_vits


logger = logging.getLogger(__name__)


def run_pipeline(config_path: str | Path, *, max_steps: int | None = None) -> Path:
    """Run the configured dataset → frontend → train → export workflow."""
    raw, layout = resolve_experiment(config_path)
    configure_logging(raw.get("logging", {}).get("level", "INFO"))
    prepare_experiment(layout, raw, config_path)
    logger.info("pipeline start model=%s languages=%s", layout.name, ",".join(layout.languages))
    stages = raw.get("pipeline", {})
    generation = raw.get("generation", {})
    text_generation = raw.get("text_generation", {})
    statuses = check_language_support(
        raw, layout,
        run_smoke=bool(stages.get("phonemize", True)),
        require_teacher=bool(stages.get("generate_samples", True) and generation.get("enabled", True)),
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
    report = {
        "name": layout.name,
        "config": str(Path(config_path).resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    text_manifest = Path(generation.get("text_manifest") or layout.dataset_dir / "texts.csv")
    if stages.get("generate_texts", True) and text_generation.get("enabled", False):
        logger.info("stage=generate_texts status=started")
        text_manifest = generate_texts(config_path)
        report["stages"]["generate_texts"] = str(text_manifest.resolve())
        logger.info("stage=generate_texts status=completed output=%s", text_manifest)
    else:
        report["stages"]["generate_texts"] = "skipped"

    raw_metadata = Path(generation.get("raw_metadata") or layout.dataset_dir / "metadata.csv")
    if stages.get("generate_samples", True) and generation.get("enabled", True):
        logger.info("stage=generate_samples status=started")
        raw_metadata = generate_samples(config_path, text_manifest_path=text_manifest)
        report["stages"]["generate_samples"] = str(raw_metadata.resolve())
        logger.info("stage=generate_samples status=completed output=%s", raw_metadata)
    else:
        report["stages"]["generate_samples"] = "skipped"

    if stages.get("phonemize", True):
        logger.info("stage=phonemize status=started")
        frontend = frontend_from_config(
            raw.get("frontend"), languages=layout.languages,
            language_registry=raw.get("language_registry"),
        )
        phonemize_manifest(raw_metadata, layout.metadata, frontend)
        report["stages"]["phonemize"] = str(layout.metadata.resolve())
        logger.info("stage=phonemize status=completed output=%s", layout.metadata)
    else:
        report["stages"]["phonemize"] = "skipped"

    if stages.get("validate", True):
        logger.info("stage=validate status=started")
        validation = validate_manifest(
            layout.metadata,
            int(raw["audio"]["sample_rate"]),
            require_single_speaker=False,
            require_phonemes=bool(raw.get("frontend", {}).get("require_phonemes", True)),
            supported_languages=layout.language_specs,
        )
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
        logger.info("stage=validate status=completed samples=%d counts=%s", len(validation.items), validation.language_counts)
    else:
        report["stages"]["validate"] = "skipped"

    checkpoint = layout.checkpoints_dir / "last"
    if stages.get("train", True):
        logger.info("stage=train status=started")
        checkpoint = train_vits(str(config_path), max_steps=max_steps)
        report["stages"]["train"] = str(checkpoint.resolve())
        logger.info("stage=train status=completed checkpoint=%s", checkpoint)
    else:
        report["stages"]["train"] = "skipped"

    if stages.get("export", True):
        logger.info("stage=export status=started")
        requested_checkpoint = raw.get("validation", {}).get("export_checkpoint", "best")
        if requested_checkpoint not in {"best", "last"}:
            raise ValueError("validation.export_checkpoint must be best or last")
        preferred = layout.checkpoints_dir / requested_checkpoint
        if preferred.is_dir():
            checkpoint = preferred
        elif requested_checkpoint == "best":
            logger.warning("best checkpoint is unavailable; exporting last checkpoint")
        model = export_vits_onnx(checkpoint, layout.artifacts_dir,
                                 sample_rate=int(raw["audio"]["sample_rate"]))
        report["stages"]["export"] = str(model.resolve())
        report["stages"]["export_checkpoint"] = str(checkpoint.resolve())
        if stages.get("validate_onnx", True):
            report["stages"]["validate_onnx"] = list(validate_onnx_runtime(model))
        logger.info("stage=export status=completed model=%s", model)
    else:
        report["stages"]["export"] = "skipped"

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    destination = layout.run_dir / "pipeline-report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("pipeline completed report=%s", destination)
    return destination
