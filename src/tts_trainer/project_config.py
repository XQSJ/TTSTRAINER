from __future__ import annotations

import json
from pathlib import Path


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_project_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load JSON configuration with optional relative `extends` inheritance."""
    source = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if source in seen:
        chain = " -> ".join(str(item) for item in (*seen, source))
        raise ValueError(f"circular config inheritance: {chain}")
    seen.add(source)
    raw = json.loads(source.read_text(encoding="utf-8"))
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (source.parent / parent).resolve()
    return _deep_merge(load_project_config(parent_path, seen), raw)
