"""质量评测模型（ASR/说话人）的本地注册与下载。 / Local registry and download of quality-eval models (ASR/speaker)."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY_MODELS_ROOT = PROJECT_ROOT / "models" / "quality"


@dataclass(frozen=True)
class QualityModelSpec:
    """质量模型短键到 HF 仓库的映射。 / Maps a quality-model key to an HF repo."""
    key: str
    repo_id: str
    directory_name: str
    required_files: tuple[str, ...]


# 支持的质量评测模型清单。 / Supported quality-eval models.
QUALITY_MODEL_SPECS = {
    "asr-small": QualityModelSpec(
        "asr-small", "Systran/faster-whisper-small", "faster-whisper-small",
        ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
    ),
    "speaker-ecapa": QualityModelSpec(
        "speaker-ecapa", "speechbrain/spkrec-ecapa-voxceleb", "spkrec-ecapa-voxceleb",
        ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt"),
    ),
}


@dataclass(frozen=True)
class QualityModelStatus:
    """单个质量模型的本地就绪状态。 / Local readiness status of one quality model."""
    spec: QualityModelSpec
    path: Path
    ready: bool
    missing: tuple[str, ...]
    size_bytes: int


def quality_models_root() -> Path:
    """返回质量模型根目录，环境变量优先。 / Return the quality models root; the env override wins."""
    override = os.environ.get("TTS_TRAINER_QUALITY_MODELS_DIR")
    return Path(override).expanduser().resolve() if override else DEFAULT_QUALITY_MODELS_ROOT


def get_quality_model_spec(key: str) -> QualityModelSpec:
    """按键查规格，未知键报错。 / Look up a spec by key; unknown keys fail."""
    try:
        return QUALITY_MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown quality model {key!r}; choose from: {', '.join(QUALITY_MODEL_SPECS)}"
        ) from exc


def quality_model_path(key: str, root: Path | None = None) -> Path:
    """返回质量模型的本地存放路径。 / Return the model's local directory."""
    spec = get_quality_model_spec(key)
    return (root or quality_models_root()) / spec.directory_name


def inspect_quality_model(key: str, root: Path | None = None) -> QualityModelStatus:
    """检查模型文件是否齐全并统计体积。 / Check required files and total size."""
    spec = get_quality_model_spec(key)
    path = quality_model_path(key, root)
    missing = tuple(name for name in spec.required_files if not (path / name).is_file())
    size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) \
        if path.exists() else 0
    return QualityModelStatus(spec, path, not missing, missing, size)


@contextmanager
def _download_lock(root: Path, key: str):
    """用 O_EXCL 原子锁防止并发下载同一模型。 / Atomic O_EXCL lock preventing concurrent downloads."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{key}.download.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another process is downloading {key}: {lock}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def ensure_quality_model(key: str, root: Path | None = None,
                         *, allow_download: bool = True) -> Path:
    """确保质量模型就绪，必要时从 HF 下载。 / Ensure a quality model is ready, downloading if needed."""
    destination_root = root or quality_models_root()
    status = inspect_quality_model(key, destination_root)
    if status.ready:
        return status.path
    if not allow_download:
        raise FileNotFoundError(
            f"quality model {key} is incomplete at {status.path}; "
            f"missing: {', '.join(status.missing)}. Run: tts-trainer quality-models ensure {key}"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download quality models") from exc
    with _download_lock(destination_root, key):
        status = inspect_quality_model(key, destination_root)
        if not status.ready:
            status.path.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=status.spec.repo_id, local_dir=status.path)
        completed = inspect_quality_model(key, destination_root)
        if not completed.ready:
            raise RuntimeError(
                "quality model download is incomplete; missing: "
                + ", ".join(completed.missing)
            )
        marker = {
            "repo_id": completed.spec.repo_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": completed.size_bytes,
        }
        (completed.path / ".download-complete.json").write_text(
            json.dumps(marker, indent=2), encoding="utf-8",
        )
        return completed.path
