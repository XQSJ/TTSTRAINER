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
        return raw
    return _deep_merge(load_project_config(parent_path, seen), raw)
