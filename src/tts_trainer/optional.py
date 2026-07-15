def require_training_dependencies():
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("training dependencies are missing; run: .venv/bin/pip install -e '.[export,dev]'") from exc
    return torch, torchaudio
