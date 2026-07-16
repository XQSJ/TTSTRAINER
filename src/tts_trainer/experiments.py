from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .languages import (DEFAULT_TRAINING_LANGUAGES, LanguageSpec,
                        language_specs_for, resolve_language_registry)
from .project_config import load_project_config


MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ExperimentLayout:
    name: str
    languages: tuple[str, ...]
    language_registry: dict[str, LanguageSpec]
    language_specs: dict[str, LanguageSpec]
    dataset_dir: Path
    metadata: Path
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    device: str
    initialization_mode: str
    initialization_checkpoint: Path | None


def validate_model_name(name: str) -> str:
    if not MODEL_NAME.fullmatch(name):
        raise ValueError("model name must contain only letters, numbers, '.', '_' and '-', and cannot start with punctuation")
    return name


def validate_languages(values, registry: dict[str, LanguageSpec] | None = None) -> tuple[str, ...]:
    registry = registry or resolve_language_registry()
    if values is None:
        values = list(DEFAULT_TRAINING_LANGUAGES)
    if not isinstance(values, list) or not values:
        raise ValueError("experiment.languages must be a non-empty JSON array")
    languages = tuple(str(value).strip().lower() for value in values)
    if len(set(languages)) != len(languages):
        raise ValueError("experiment.languages must not contain duplicates")
    unsupported = sorted(set(languages) - set(registry))
    if unsupported:
        raise ValueError(
            f"unregistered configured languages: {', '.join(unsupported)}; "
            "add them under language_registry"
        )
    return languages


def resolve_experiment(config_path: str | Path, *, metadata_override: str | None = None,
                       output_override: str | None = None,
                       device_override: str | None = None) -> tuple[dict, ExperimentLayout]:
    raw = load_project_config(config_path)
    experiment = raw.get("experiment", {})
    name = validate_model_name(experiment.get("name") or Path(config_path).stem)
    registry = resolve_language_registry(raw.get("language_registry"))
    languages = validate_languages(experiment.get("languages"), registry)
    specs = language_specs_for(registry, languages)
    dataset_dir = Path(experiment.get("dataset_root", "datasets")) / name
    metadata_value = metadata_override or experiment.get("metadata") or dataset_dir / "metadata.phonemes.csv"
    run_dir = Path(output_override) if output_override else Path(experiment.get("run_root", "runs")) / name
    artifacts_dir = Path(experiment.get("artifact_root", "artifacts")) / name
    initialization = experiment.get("initialization", {"mode": "scratch"})
    mode = initialization.get("mode", "scratch")
    if mode not in {"scratch", "resume", "expand_speakers"}:
        raise ValueError("experiment.initialization.mode must be scratch, resume, or expand_speakers")
    checkpoint_value = initialization.get("checkpoint")
    checkpoint = Path(checkpoint_value) if checkpoint_value else None
    if mode != "scratch" and checkpoint is None:
        raise ValueError(f"initialization mode {mode} requires a checkpoint")
    layout = ExperimentLayout(
        name=name,
        languages=languages,
        language_registry=registry,
        language_specs=specs,
        dataset_dir=dataset_dir,
        metadata=Path(metadata_value),
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        logs_dir=run_dir / "logs",
        artifacts_dir=artifacts_dir,
        device=device_override or experiment.get("device", "auto"),
        initialization_mode=mode,
        initialization_checkpoint=checkpoint,
    )
    return raw, layout


def prepare_experiment(layout: ExperimentLayout, resolved_config: dict, config_path: str | Path) -> None:
    layout.dataset_dir.mkdir(parents=True, exist_ok=True)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    layout.artifacts_dir.mkdir(parents=True, exist_ok=True)
    recorded_config = deepcopy(resolved_config)
    recorded_config.setdefault("experiment", {})["languages"] = list(layout.languages)
    if "model" in recorded_config:
        recorded_config["model"]["num_languages"] = len(layout.languages)
    (layout.run_dir / "resolved-config.json").write_text(
        json.dumps(recorded_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "name": layout.name,
        "languages": list(layout.languages),
        "language_registry": {code: spec.to_dict() for code, spec in layout.language_specs.items()},
        "source_config": str(Path(config_path).resolve()),
        "dataset_dir": str(layout.dataset_dir.resolve()),
        "metadata": str(layout.metadata.resolve()),
        "run_dir": str(layout.run_dir.resolve()),
        "checkpoints_dir": str(layout.checkpoints_dir.resolve()),
        "logs_dir": str(layout.logs_dir.resolve()),
        "artifacts_dir": str(layout.artifacts_dir.resolve()),
        "device": layout.device,
        "initialization": {
            "mode": layout.initialization_mode,
            "checkpoint": str(layout.initialization_checkpoint.resolve()) if layout.initialization_checkpoint else None,
        },
    }
    (layout.run_dir / "run-layout.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
