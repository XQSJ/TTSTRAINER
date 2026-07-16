from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import torch
from torch import nn

from ..checkpoints import CHECKPOINT_FORMAT
from ..frontend import frontend_contract_from_config
from ..frontend.conformance import save_frontend_conformance
from .config import VitsConfig
from .model import MultilingualVITS


logger = logging.getLogger(__name__)


class PiperInferenceWrapper(nn.Module):
    """Expose standard Piper inputs while retaining two internal conditions.

    sid is a composite profile id:
      speaker_id = sid // num_languages
      language_id = sid % num_languages
    """
    def __init__(self, model: MultilingualVITS):
        super().__init__()
        self.model = model
        self.num_languages = model.config.num_languages

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor,
                scales: torch.Tensor, sid: torch.Tensor):
        sid = sid.to(torch.long)
        language_ids = torch.remainder(sid, self.num_languages)
        speaker_ids = torch.div(sid, self.num_languages, rounding_mode="floor")
        return self.model.infer_deploy(input, input_lengths, language_ids, speaker_ids, scales)


def _config_from_metadata(raw: dict) -> VitsConfig:
    config = dict(raw["config"])
    config["upsample_rates"] = tuple(config["upsample_rates"])
    config["upsample_kernel_sizes"] = tuple(config["upsample_kernel_sizes"])
    return VitsConfig(**config)


def voice_profiles(speaker_map: dict[str, int], language_map: dict[str, int]) -> list[dict]:
    profiles = []
    language_count = len(language_map)
    for speaker, speaker_id in sorted(speaker_map.items(), key=lambda item: item[1]):
        for language, language_id in sorted(language_map.items(), key=lambda item: item[1]):
            profiles.append({
                "sid": speaker_id * language_count + language_id,
                "speaker": speaker,
                "speaker_id": speaker_id,
                "language": language,
                "language_id": language_id,
            })
    return profiles


def export_vits_onnx(checkpoint_dir: str | Path, output_dir: str | Path,
                     *, sample_rate: int = 22050, opset: int = 17) -> Path:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export requires: pip install -e '.[export]'") from exc
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format"] != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    config = _config_from_metadata(metadata)
    logger.info("ONNX export step=1/5 action=load_checkpoint path=%s", checkpoint_dir)
    generator = MultilingualVITS(config)
    state = torch.load(checkpoint_dir / "training-state.pt", map_location="cpu", weights_only=False)
    generator.load_state_dict(state["generator"])
    wrapper = PiperInferenceWrapper(generator.eval())
    target = output_dir / "model.onnx"
    tokens = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    lengths = torch.tensor([4], dtype=torch.long)
    scales = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.long)
    logger.info("ONNX export step=2/5 action=build_graph opset=%d output=%s", opset, target)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        torch.onnx.export(
            wrapper, (tokens, lengths, scales, sid), str(target),
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["output"], opset_version=opset, do_constant_folding=True,
            dynamic_axes={"input": {0: "batch", 1: "text_length"},
                          "input_lengths": {0: "batch"}, "sid": {0: "batch"},
                          "output": {0: "batch", 2: "audio_length"}},
            dynamo=False,
        )
    logger.info("ONNX export step=3/5 action=check_model size_bytes=%d", target.stat().st_size)
    model = onnx.load(str(target)); onnx.checker.check_model(model)
    profiles = voice_profiles(metadata["speaker_map"], metadata["language_map"])
    frontend = metadata.get("frontend") or frontend_contract_from_config(
        {}, tuple(metadata["language_map"])
    ).to_dict()
    deployment = {
        "format": 1,
        "model_type": "multilingual-vits-piper-shaped",
        "sample_rate": sample_rate,
        "hop_length": config.hop_length,
        "inputs": ["input", "input_lengths", "scales", "sid"],
        "scales_default": [0.667, 1.0, 1.0],
        "sid_formula": "speaker_id * num_languages + language_id",
        "frontend": frontend,
        "frontend_note": "application supplies matching phoneme ids; stock sherpa multilingual switching requires an adapter",
        "num_languages": config.num_languages,
        "num_speakers": config.num_speakers,
        "voice_profiles": profiles,
    }
    (output_dir / "model.onnx.json").write_text(json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "frontend.json").write_text(json.dumps(frontend, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "tokens.json").write_text(json.dumps({"tokens": metadata["tokens"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    tokens_text = "".join(f"{token} {index}\n" for index, token in enumerate(metadata["tokens"]))
    (output_dir / "tokens.txt").write_text(tokens_text, encoding="utf-8")
    conformance = metadata.get("frontend_conformance")
    if conformance:
        save_frontend_conformance(conformance, output_dir / "frontend.conformance.json")
    logger.info(
        "ONNX export step=4/5 action=write_resources profiles=%d directory=%s",
        len(profiles), output_dir,
    )
    logger.info("ONNX export step=5/5 action=completed model=%s", target)
    return target


def validate_onnx_runtime(model_path: str | Path) -> tuple[int, ...]:
    import numpy as np
    import onnxruntime as ort
    logger.info("ONNX runtime validation status=started provider=CPUExecutionProvider model=%s", model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {
        "input": np.asarray([[2, 3]], dtype=np.int64),
        "input_lengths": np.asarray([2], dtype=np.int64),
        "scales": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        "sid": np.asarray([0], dtype=np.int64),
    })[0]
    if output.ndim != 3 or output.shape[1] != 1 or output.shape[2] <= 0:
        raise RuntimeError(f"unexpected ONNX output shape: {output.shape}")
    logger.info("ONNX runtime validation status=completed output_shape=%s", tuple(output.shape))
    return tuple(output.shape)
