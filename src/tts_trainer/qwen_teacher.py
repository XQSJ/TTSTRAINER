from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from .model_registry import ensure_model, require_local_model


QWEN_TTS_VERSION = "0.1.1"
QWEN_INSTALL_HINT = "pip install -e '.[qwen]'"


def configure_qwen_runtime(runtime_mode: str = "installed", source_path: str | Path | None = None) -> dict:
    """Validate and configure the Qwen Python runtime without downloading weights."""
    if runtime_mode not in {"installed", "source"}:
        raise ValueError("generation.qwen_runtime must be installed or source")
    if runtime_mode == "source":
        if not source_path:
            raise RuntimeError("generation.qwen_source_path is required when qwen_runtime=source")
        source = Path(source_path).expanduser().resolve()
        if not (source / "qwen_tts" / "__init__.py").is_file():
            raise RuntimeError(f"Qwen source path does not contain qwen_tts: {source}")
        value = str(source)
        if value not in sys.path:
            sys.path.insert(0, value)
        return {"ready": True, "mode": "source", "source": value, "install_hint": QWEN_INSTALL_HINT}
    if importlib.util.find_spec("qwen_tts") is None:
        raise RuntimeError(
            f"qwen-tts runtime is not installed. Install the official package with: {QWEN_INSTALL_HINT}"
        )
    return {"ready": True, "mode": "installed", "source": "qwen-tts", "install_hint": QWEN_INSTALL_HINT}


def inspect_qwen_runtime(runtime_mode: str = "installed", source_path: str | Path | None = None) -> dict:
    try:
        return configure_qwen_runtime(runtime_mode, source_path)
    except (RuntimeError, ValueError) as exc:
        return {
            "ready": False,
            "mode": runtime_mode,
            "source": str(source_path) if source_path else None,
            "error": str(exc),
            "install_hint": QWEN_INSTALL_HINT,
        }


def load_qwen_teacher(model_key: str = "base-1.7b", *, download_if_missing: bool = True,
                      runtime_mode: str = "installed", source_path: str | Path | None = None,
                      **kwargs):
    """Load Qwen-TTS exclusively from the project-local model directory.

    Network access is disabled before `from_pretrained`, so Transformers cannot
    silently fetch a second copy into the user's global Hugging Face cache.
    """
    configure_qwen_runtime(runtime_mode, source_path)
    path = ensure_model(model_key) if download_if_missing else require_local_model(model_key)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            f"qwen-tts or one of its runtime dependencies could not be imported. Reinstall with: {QWEN_INSTALL_HINT}"
        ) from exc
    return Qwen3TTSModel.from_pretrained(str(path), **kwargs)
