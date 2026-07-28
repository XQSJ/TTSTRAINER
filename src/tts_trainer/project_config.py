from __future__ import annotations

import json
from pathlib import Path


PRESET_FILES = {
    "compact": "configs/internal/pipeline_defaults.json",
    "quality": "configs/internal/quality_pipeline_defaults.json",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_dataset_config(raw: dict) -> dict:
    """Expand the small public `dataset` block into internal pipeline settings."""
    dataset = raw.get("dataset")
    if dataset is None:
        result = dict(raw)
        task = str(result.get("task", "train")).strip().lower()
        if task not in {"prepare", "train"}:
            raise ValueError("task must be prepare or train")
        result["task"] = task
        return result
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a JSON object")
    result = dict(raw)
    text = dataset.get("text", {})
    if not isinstance(text, dict):
        raise ValueError("dataset.text must be a JSON object")
    text_override = dict(text)
    if "sentences_per_language" in dataset:
        text_override["sentences_per_language"] = dataset["sentences_per_language"]
    text_override.setdefault(
        "enabled",
        bool(dataset.get("voice") or dataset.get("voices"))
        or not bool(dataset.get("speakers")),
    )
    result["text_generation"] = _deep_merge(
        result.get("text_generation", {}), text_override,
    )

    generation_override = {}
    if "voice" in dataset and "voices" in dataset:
        raise ValueError("dataset cannot define both voice and voices")
    if "voice" in dataset:
        generation_override["voice"] = dataset["voice"]
    if "voices" in dataset:
        voices = dataset["voices"]
        if not isinstance(voices, dict) or not voices:
            raise ValueError("dataset.voices must be a non-empty object keyed by voice ID")
        normalized_voices = {}
        for voice_id, settings in voices.items():
            voice_id = str(voice_id).strip()
            if not voice_id:
                raise ValueError("dataset.voices contains an empty voice ID")
            if settings is None:
                settings = {}
            if not isinstance(settings, dict):
                raise ValueError(
                    f"dataset.voices.{voice_id} must be a JSON object"
                )
            if "regenerate_audio" in settings \
                    and not isinstance(settings["regenerate_audio"], bool):
                raise ValueError(
                    f"dataset.voices.{voice_id}.regenerate_audio must be true or false"
                )
            strategy = settings.get("reference_strategy")
            if strategy is not None and strategy not in {
                "shared", "per_language", "cascade",
            }:
                raise ValueError(
                    f"dataset.voices.{voice_id}.reference_strategy must be "
                    "shared, per_language, or cascade"
                )
            configured_id = str(settings.get("id") or voice_id).strip()
            if configured_id != voice_id:
                raise ValueError(
                    f"dataset.voices.{voice_id}.id must match its object key"
                )
            normalized_voices[voice_id] = {**settings, "id": voice_id}
        generation_override["voices"] = normalized_voices
    if "speakers" in dataset:
        generation_override["speaker_assignments"] = dataset["speakers"]
    if "include" in dataset:
        generation_override["include_metadata"] = dataset["include"]
    if "enabled" in dataset:
        generation_override["enabled"] = bool(dataset["enabled"])
    result["generation"] = _deep_merge(
        result.get("generation", {}), generation_override,
    )
    task = str(result.get("task", "train")).strip().lower()
    if task not in {"prepare", "train"}:
        raise ValueError("task must be prepare or train")
    result["task"] = task
    if task == "prepare" and dataset.get("speakers"):
        raise ValueError(
            "task=prepare does not use dataset.speakers; assign speakers in a train config"
        )
    if task == "prepare" and not (dataset.get("voice") or dataset.get("voices")):
        raise ValueError(
            "task=prepare requires dataset.voices (or legacy dataset.voice)"
        )
    if task == "train" and dataset.get("voices") and not dataset.get("speakers"):
        raise ValueError(
            "task=train with dataset.voices requires dataset.speakers"
        )
    return result


def _preset_path(source: Path, preset: str) -> Path:
    relative = PRESET_FILES.get(preset)
    if relative is None:
        choices = ", ".join(sorted(PRESET_FILES))
        raise ValueError(f"unknown config preset {preset!r}; choose one of: {choices}")
    candidates = []
    for root in (source.parent, *source.parents, Path.cwd(), Path(__file__).resolve().parents[2]):
        candidate = (root / relative).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"cannot locate files for config preset {preset!r}; run from the tts-trainer project"
    )


def load_project_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load JSON configuration with a public preset or expert `extends` inheritance."""
    source = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if source in seen:
        chain = " -> ".join(str(item) for item in (*seen, source))
        raise ValueError(f"circular config inheritance: {chain}")
    seen.add(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    preset = raw.pop("preset", None)
    parent = raw.pop("extends", None)
    if preset is not None and parent is not None:
        raise ValueError("config cannot define both preset and extends")
    if preset is not None:
        parent_path = _preset_path(source, str(preset))
    elif parent is not None:
        parent_path = (source.parent / parent).resolve()
    else:
        return _normalize_dataset_config(raw)
    return _normalize_dataset_config(
        _deep_merge(load_project_config(parent_path, seen), raw),
    )
