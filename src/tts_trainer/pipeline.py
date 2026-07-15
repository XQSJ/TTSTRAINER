from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .experiments import prepare_experiment, resolve_experiment
from .frontend import EspeakFrontend, phonemize_manifest
from .manifest import validate_manifest
from .sample_generation import generate_samples
from .vits.exporter import export_vits_onnx, validate_onnx_runtime
from .vits.trainer import train_vits


def run_pipeline(config_path: str | Path, *, max_steps: int | None = None) -> Path:
    """Run the configured dataset → frontend → train → export workflow."""
    raw, layout = resolve_experiment(config_path)
    prepare_experiment(layout, raw, config_path)
    stages = raw.get("pipeline", {})
    generation = raw.get("generation", {})
    report = {
        "name": layout.name,
        "config": str(Path(config_path).resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    raw_metadata = Path(generation.get("raw_metadata") or layout.dataset_dir / "metadata.csv")
    if stages.get("generate_samples", True) and generation.get("enabled", True):
        raw_metadata = generate_samples(config_path)
        report["stages"]["generate_samples"] = str(raw_metadata.resolve())
    else:
        report["stages"]["generate_samples"] = "skipped"

    if stages.get("phonemize", True):
        frontend = EspeakFrontend()
        phonemize_manifest(raw_metadata, layout.metadata, frontend)
        report["stages"]["phonemize"] = str(layout.metadata.resolve())
    else:
        report["stages"]["phonemize"] = "skipped"

    if stages.get("validate", True):
        validation = validate_manifest(
            layout.metadata,
            int(raw["audio"]["sample_rate"]),
            require_single_speaker=False,
            require_phonemes=bool(raw.get("frontend", {}).get("require_phonemes", True)),
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
    else:
        report["stages"]["validate"] = "skipped"

    checkpoint = layout.checkpoints_dir / "last"
    if stages.get("train", True):
        checkpoint = train_vits(str(config_path), max_steps=max_steps)
        report["stages"]["train"] = str(checkpoint.resolve())
    else:
        report["stages"]["train"] = "skipped"

    if stages.get("export", True):
        model = export_vits_onnx(checkpoint, layout.artifacts_dir,
                                 sample_rate=int(raw["audio"]["sample_rate"]))
        report["stages"]["export"] = str(model.resolve())
        if stages.get("validate_onnx", True):
            report["stages"]["validate_onnx"] = list(validate_onnx_runtime(model))
    else:
        report["stages"]["export"] = "skipped"

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    destination = layout.run_dir / "pipeline-report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
