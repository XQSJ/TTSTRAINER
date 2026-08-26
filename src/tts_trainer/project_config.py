"""项目配置加载：preset/extends 继承与 dataset 简写展开。 / Load configs with preset/extends inheritance and dataset sugar."""

from __future__ import annotations

import json
from pathlib import Path


# 公开 preset 到内部默认配置文件的映射。 / Public preset names mapped to internal default files.
PRESET_FILES = {
    "compact": "configs/internal/pipeline_defaults.json",
    "quality": "configs/internal/quality_pipeline_defaults.json",
    "mobile": "configs/internal/mobile_pipeline_defaults.json",
    "mobile_routed": "configs/internal/mobile_routed_pipeline_defaults.json",
    "quality_commercial": "configs/internal/quality_commercial_pipeline_defaults.json",
    "mobile_commercial": "configs/internal/mobile_commercial_pipeline_defaults.json",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归深合并，override 覆盖同名叶子值。 / Recursively merge dicts; override wins at leaves."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_dataset_config(raw: dict) -> dict:
    """把公开的精简 dataset 块展开为内部配置。 / Expand the small public `dataset` block into internal pipeline settings."""
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
        # 顶层短语法等价于 dataset.text.sentences_per_language。 / Top-level shorthand mirrors dataset.text.sentences_per_language.
        text_override["sentences_per_language"] = dataset["sentences_per_language"]
    # 未显式关闭时默认启用文本生成：有声线或未用现成 speaker 数据。 / Default on: a voice is set or no pre-made speakers are reused.
    text_override.setdefault(
        "enabled",
        bool(dataset.get("voice") or dataset.get("voices"))
        or not bool(dataset.get("speakers")),
    )
    result["text_generation"] = _deep_merge(
        result.get("text_generation", {}), text_override,
    )

    generation_override = {}
    # 单 voice 与多 voices 互斥，避免歧义覆盖。 / Single voice and voices map are mutually exclusive.
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
            regenerate = settings.get("regenerate")
            if regenerate is not None:
                if not isinstance(regenerate, dict):
                    raise ValueError(
                        f"dataset.voices.{voice_id}.regenerate must be a JSON object"
                    )
                unknown = sorted(
                    set(regenerate) - {"audio", "references", "languages"}
                )
                if unknown:
                    raise ValueError(
                        f"dataset.voices.{voice_id}.regenerate has unknown fields: "
                        + ", ".join(unknown)
                    )
                for field in ("audio", "references"):
                    if field in regenerate \
                            and not isinstance(regenerate[field], bool):
                        raise ValueError(
                            f"dataset.voices.{voice_id}.regenerate.{field} "
                            "must be true or false"
                        )
                selected_languages = regenerate.get("languages", "all")
                if selected_languages != "all" and not (
                    isinstance(selected_languages, list)
                    and selected_languages
                    and all(
                        isinstance(language, str) and language.strip()
                        for language in selected_languages
                    )
                ):
                    raise ValueError(
                        f"dataset.voices.{voice_id}.regenerate.languages must "
                        'be "all" or a non-empty array'
                    )
                if regenerate.get("references", False) \
                        and not regenerate.get("audio", False):
                    raise ValueError(
                        f"dataset.voices.{voice_id}.regenerate.references=true "
                        "requires audio=true"
                    )
                if settings.get("regenerate_audio", False):
                    raise ValueError(
                        f"dataset.voices.{voice_id} cannot combine regenerate "
                        "with regenerate_audio=true"
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
    # prepare 阶段只产数据，train 阶段消费数据；下方校验各自的必填项。 / prepare only produces data; train consumes it — validate each side's required fields.
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
    """定位 preset 对应的内部默认配置文件。 / Locate the internal default file for a preset."""
    relative = PRESET_FILES.get(preset)
    if relative is None:
        choices = ", ".join(sorted(PRESET_FILES))
        raise ValueError(f"unknown config preset {preset!r}; choose one of: {choices}")
    # 依次向上搜索配置目录、CWD 与安装源码树。 / Search upward, then CWD, then the installed source tree.
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
    """加载 JSON 配置并解析 preset 或专家 extends 继承。 / Load JSON configuration with a public preset or expert `extends` inheritance."""
    source = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    # 环形继承检测。 / Detect circular inheritance chains.
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
