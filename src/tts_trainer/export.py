from __future__ import annotations

from pathlib import Path

from .config import load_config
from .model import build_model
from .optional import require_training_dependencies


def export_onnx(config_path: str, checkpoint_path: str, output_path: str) -> Path:
    torch, _ = require_training_dependencies()
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ONNX dependency missing; install project with the export extra") from exc
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["vocab_size"], config.data.n_mels, config.model)
    model.load_state_dict(checkpoint["model"]); model.eval()
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    tokens = torch.tensor([[2, 3]], dtype=torch.long)
    language = torch.tensor([0], dtype=torch.long)
    torch.onnx.export(model, (tokens, language), str(target), input_names=["tokens", "language_id"],
                      output_names=["mel"], dynamic_axes={"tokens": {1: "text_length"}, "mel": {2: "text_length"}},
                      opset_version=17)
    return target
