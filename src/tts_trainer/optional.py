"""可选训练依赖的懒加载入口。 / Lazy entry for optional training dependencies."""


def require_training_dependencies():
    """导入 torch/torchaudio，缺失时给出安装指引。 / Import torch/torchaudio, or fail with install hint."""
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("training dependencies are missing; run: .venv/bin/pip install -e '.[export,dev]'") from exc
    return torch, torchaudio
