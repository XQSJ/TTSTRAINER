from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "models" / "qwen"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str

    @property
    def directory_name(self) -> str:
        return self.repo_id.rsplit("/", 1)[-1]


MODEL_SPECS = {
    "voice-design-1.7b": ModelSpec("voice-design-1.7b", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "base-1.7b": ModelSpec("base-1.7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    "base-0.6b": ModelSpec("base-0.6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
}
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
)


@dataclass(frozen=True)
class ModelStatus:
    spec: ModelSpec
    path: Path
    ready: bool
    missing: tuple[str, ...]
    size_bytes: int


def models_root() -> Path:
    override = os.environ.get("TTS_TRAINER_MODELS_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_MODELS_ROOT


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unknown model {key!r}; choose from: {', '.join(MODEL_SPECS)}") from exc


def model_path(key: str, root: Path | None = None) -> Path:
    spec = get_spec(key)
    return (root or models_root()) / spec.directory_name


def inspect_model(key: str, root: Path | None = None) -> ModelStatus:
    spec = get_spec(key)
    path = model_path(key, root)
    missing = tuple(name for name in REQUIRED_FILES if not (path / name).is_file())
    size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.exists() else 0
    return ModelStatus(spec, path, not missing, missing, size)


@contextmanager
def _download_lock(root: Path, key: str):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{key}.download.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another process is downloading {key}; lock: {lock}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def ensure_model(key: str, root: Path | None = None, *, allow_download: bool = True) -> Path:
    destination_root = root or models_root()
    status = inspect_model(key, destination_root)
    if status.ready:
        logger.info("model ready key=%s path=%s size_bytes=%d", key, status.path, status.size_bytes)
        return status.path
    if not allow_download:
        raise FileNotFoundError(
            f"model {key} is incomplete at {status.path}; missing: {', '.join(status.missing)}. "
            f"Run: tts-trainer models ensure {key}"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download models") from exc
    with _download_lock(destination_root, key):
        status = inspect_model(key, destination_root)
        if not status.ready:
            logger.info(
                "model missing key=%s path=%s missing=%s; download starting",
                key, status.path, ",".join(status.missing),
            )
            status.path.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=status.spec.repo_id, local_dir=status.path)
        completed = inspect_model(key, destination_root)
        if not completed.ready:
            raise RuntimeError(f"download finished but model is incomplete; missing: {', '.join(completed.missing)}")
        marker = {
            "repo_id": completed.spec.repo_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": completed.size_bytes,
        }
        (completed.path / ".download-complete.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
        logger.info("model download completed key=%s path=%s size_bytes=%d", key, completed.path, completed.size_bytes)
        return completed.path


def require_local_model(key: str, root: Path | None = None) -> Path:
    """Resolve a model without ever accessing the network."""
    return ensure_model(key, root, allow_download=False)
