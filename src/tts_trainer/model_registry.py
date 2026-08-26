"""Qwen 基础模型的本地注册、检查与下载。 / Local registry, inspection and download of Qwen base models."""
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
    """模型短键到 Hugging Face 仓库的映射。 / Maps a short model key to a Hugging Face repo."""
    key: str
    repo_id: str

    @property
    def directory_name(self) -> str:
        """本地目录名取仓库名末段。 / Local directory name is the repo's last path segment."""
        return self.repo_id.rsplit("/", 1)[-1]


# 可用的 Qwen 基座模型清单。 / Available Qwen base models.
MODEL_SPECS = {
    "voice-design-1.7b": ModelSpec("voice-design-1.7b", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    "base-1.7b": ModelSpec("base-1.7b", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    "base-0.6b": ModelSpec("base-0.6b", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
}
# 判定模型完整所需的最小文件集合。 / Minimum files for a model to count as complete.
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
)


@dataclass(frozen=True)
class ModelStatus:
    """单个模型的本地就绪状态。 / Local readiness status of one model."""
    spec: ModelSpec
    path: Path
    ready: bool
    missing: tuple[str, ...]
    size_bytes: int


def models_root() -> Path:
    """返回模型根目录，环境变量优先。 / Return the models root; the env override wins."""
    override = os.environ.get("TTS_TRAINER_MODELS_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_MODELS_ROOT


def get_spec(key: str) -> ModelSpec:
    """按键查模型规格，未知键报错。 / Look up a spec by key; unknown keys fail."""
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unknown model {key!r}; choose from: {', '.join(MODEL_SPECS)}") from exc


def model_path(key: str, root: Path | None = None) -> Path:
    """返回模型的本地存放路径。 / Return the model's local directory."""
    spec = get_spec(key)
    return (root or models_root()) / spec.directory_name


def inspect_model(key: str, root: Path | None = None) -> ModelStatus:
    """检查模型文件是否齐全并统计体积。 / Check required files and total size."""
    spec = get_spec(key)
    path = model_path(key, root)
    missing = tuple(name for name in REQUIRED_FILES if not (path / name).is_file())
    size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) if path.exists() else 0
    return ModelStatus(spec, path, not missing, missing, size)


@contextmanager
def _download_lock(root: Path, key: str):
    """用 O_EXCL 原子锁防止并发下载同一模型。 / Atomic O_EXCL lock preventing concurrent downloads."""
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
    """确保模型就绪，必要时从 HF 下载。 / Ensure a model is ready, downloading from HF if needed."""
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
        # 双重检查：拿到锁后可能别的进程已完成下载。 / Re-check after acquiring the lock; another process may have finished.
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
    """只使用本地缓存解析模型，绝不联网。 / Resolve a model without ever accessing the network."""
    return ensure_model(key, root, allow_download=False)
