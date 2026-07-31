from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import torch
from torch import nn

from ..checkpoints import require_checkpoint_format
from ..frontend import frontend_contract_from_config
from ..frontend.contract import (DIRECT_TOKEN_ENCODING,
                                 LEGACY_PIPER_TOKEN_ENCODING,
                                 MOBILE_DIRECT_TOKEN_ENCODING,
                                 PIPER_TOKEN_ENCODING)
from ..frontend.conformance import save_frontend_conformance
from .config import VitsConfig
from .model import MultilingualVITS


logger = logging.getLogger(__name__)
SHERPA_ONNX_ANDROID_VERSION = "1.13.4"


def require_mobile_blank_semantics(metadata: dict) -> None:
    frontend = metadata.get("frontend") or {}
    if frontend.get("token_encoding") == LEGACY_PIPER_TOKEN_ENCODING:
        raise ValueError(
            "this mobile checkpoint uses the legacy Piper sequence that omitted "
            "the blank immediately after BOS; start a new preset=mobile model "
            "from scratch with the corrected v2 frontend contract"
        )
    if (
        frontend.get("token_encoding")
        == PIPER_TOKEN_ENCODING
        and not bool(metadata.get("learned_blank_token", False))
    ):
        raise ValueError(
            "this mobile checkpoint was trained while Piper's valid blank token "
            "was frozen as batch padding; update TTSTRAINER, resume or warm-start "
            "the checkpoint so token 0 can learn, then export the newly saved "
            "checkpoint"
        )


class PiperInferenceWrapper(nn.Module):
    """Expose standard Piper inputs while retaining two internal conditions.

    sid is a composite profile id:
      speaker_id = sid // num_languages
      language_id = sid % num_languages
    """
    def __init__(
        self, model: MultilingualVITS, *,
        insert_pad_after_bos: bool = False,
        strip_piper_pads: bool = False,
    ):
        super().__init__()
        if insert_pad_after_bos and strip_piper_pads:
            raise ValueError(
                "insert_pad_after_bos and strip_piper_pads are mutually exclusive"
            )
        self.model = model
        self.num_languages = model.config.num_languages
        self.insert_pad_after_bos = insert_pad_after_bos
        self.strip_piper_pads = strip_piper_pads

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor,
                scales: torch.Tensor, sid: torch.Tensor):
        if self.insert_pad_after_bos:
            # sherpa-onnx 1.13.4 emits the historical
            # BOS,(phone,PAD)*,EOS wire sequence. Normalize it inside the
            # exported graph to the canonical BOS,PAD,(phone,PAD)*,EOS
            # sequence used to train the mobile model.
            pad = torch.zeros_like(input[:, :1])
            input = torch.cat((input[:, :1], pad, input[:, 1:]), dim=1)
            input_lengths = input_lengths + 1
        elif self.strip_piper_pads:
            # sherpa invokes a VITS graph with one sentence at a time. Remove
            # every Piper transport PAD (token id 0) so the core text encoder
            # receives the compact BOS,(phoneme)*,EOS sequence used in
            # training. This accepts both sherpa-onnx 1.13.4's historical wire
            # sequence and Piper's corrected sequence with a PAD after BOS.
            positions = torch.arange(
                input.shape[1], device=input.device,
            ).unsqueeze(0)
            valid = positions < input_lengths.unsqueeze(1)
            keep = valid & input.ne(0)
            input = torch.masked_select(input, keep).unsqueeze(0)
            input_lengths = keep.sum(dim=1)
        sid = sid.to(torch.long)
        language_ids = torch.remainder(sid, self.num_languages)
        speaker_ids = torch.div(sid, self.num_languages, rounding_mode="floor")
        return self.model.infer_deploy(input, input_lengths, language_ids, speaker_ids, scales)


def _config_from_metadata(raw: dict) -> VitsConfig:
    config = dict(raw["config"])
    for key in ("decoder_resblock_kernel_sizes", "upsample_rates", "upsample_kernel_sizes"):
        if key in config:
            config[key] = tuple(config[key])
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


def _replace_onnx_metadata(model, values: dict[str, object]) -> None:
    preserved = {
        item.key: item.value for item in model.metadata_props
        if item.key not in values
    }
    del model.metadata_props[:]
    for key, value in {**preserved, **values}.items():
        item = model.metadata_props.add()
        item.key = str(key)
        item.value = str(value)


def _find_espeak_data_dir() -> Path:
    configured = os.environ.get("ESPEAK_DATA_PATH")
    candidates = [Path(configured).expanduser()] if configured else []
    executable = shutil.which("espeak-ng") or shutil.which("espeak")
    if executable:
        result = subprocess.run(
            [executable, "--version"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        match = re.search(r"Data at:\s*(.+?)\s*$", result.stdout)
        if match:
            candidates.append(Path(match.group(1)))
    candidates.extend((
        Path("/usr/share/espeak-ng-data"),
        Path("/usr/local/share/espeak-ng-data"),
        Path("/opt/homebrew/share/espeak-ng-data"),
    ))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "mobile text export requires espeak-ng-data; install eSpeak NG or set "
        "ESPEAK_DATA_PATH to its data directory"
    )


def _export_sherpa_android_text_package(
    onnx, model, output_dir: Path, frontend: dict, profiles: list[dict],
    *, sample_rate: int, tokens: list[str],
) -> dict:
    """Write language-specific metadata wrappers for sherpa's eSpeak frontend."""
    supported_encodings = {
        PIPER_TOKEN_ENCODING,
        MOBILE_DIRECT_TOKEN_ENCODING,
    }
    if frontend.get("provider") != "espeak-ng" or frontend.get(
        "token_encoding",
    ) not in supported_encodings:
        return {
            "supported": False,
            "reason": (
                "model was not trained with the mobile eSpeak/Piper token "
                "contract; retrain with preset=mobile"
            ),
        }

    android_root = output_dir / "android_text"
    android_root.mkdir(parents=True, exist_ok=True)
    token_lines = []
    for token_id, token in enumerate(tokens):
        if token == "<unk>":
            continue
        if len(token) != 1:
            raise ValueError(
                "mobile eSpeak export requires Unicode-codepoint tokens; "
                f"found {token!r}"
            )
        token_lines.append(
            f"{token_id}\n" if token == " " else f"{token} {token_id}\n"
        )
    (android_root / "tokens.txt").write_text(
        "".join(token_lines), encoding="utf-8",
    )
    profile_count = len(profiles)
    languages = {}
    first_model = True
    for language, profile in frontend["languages"].items():
        voice = profile["voice"]
        _replace_onnx_metadata(model, {
            "model_type": "vits",
            "comment": "piper;ttstrainer-mobile",
            "sample_rate": sample_rate,
            "n_speakers": profile_count,
            "language": language,
            "voice": voice,
            "has_espeak": 1,
            "add_blank": 0,
            "version": 1,
        })
        relative = Path("android_text") / f"model-{language}.onnx"
        target = output_dir / relative
        if first_model:
            onnx.external_data_helper.convert_model_to_external_data(
                model,
                all_tensors_to_one_file=True,
                location="model.weights",
                size_threshold=0,
                convert_attribute=False,
            )
            onnx.save_model(model, target)
            first_model = False
        else:
            # The first save clears raw tensor data from this in-memory proto.
            # Later metadata wrappers retain references to the same file.
            onnx.save_model(model, target)
        languages[language] = {
            "voice": voice,
            "model": str(relative).replace("\\", "/"),
        }

    data_source = _find_espeak_data_dir()
    data_target = android_root / "espeak-ng-data"
    shutil.copytree(data_source, data_target, dirs_exist_ok=True)
    return {
        "supported": True,
        "runtime": "sherpa-onnx",
        "runtime_version": SHERPA_ONNX_ANDROID_VERSION,
        "tokens": "android_text/tokens.txt",
        "data_dir": "android_text/espeak-ng-data",
        "languages": languages,
        "note": (
            "Use the language-specific ONNX wrapper; all wrappers contain the "
            "same trained weights and differ only in eSpeak voice metadata."
        ),
    }


def export_vits_onnx(checkpoint_dir: str | Path, output_dir: str | Path,
                     *, sample_rate: int = 22050, opset: int = 17) -> Path:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export requires: pip install -e '.[export]'") from exc
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    require_checkpoint_format(int(metadata["format"]))
    require_mobile_blank_semantics(metadata)
    frontend = metadata.get("frontend") or frontend_contract_from_config(
        {}, tuple(metadata["language_map"])
    ).to_dict()
    mobile_piper = frontend.get("token_encoding") == PIPER_TOKEN_ENCODING
    mobile_direct = (
        frontend.get("token_encoding") == MOBILE_DIRECT_TOKEN_ENCODING
    )
    config = _config_from_metadata(metadata)
    logger.info("ONNX export step=1/5 action=load_checkpoint path=%s", checkpoint_dir)
    generator = MultilingualVITS(config)
    state = torch.load(checkpoint_dir / "training-state.pt", map_location="cpu", weights_only=False)
    generator.load_state_dict(state["generator"])
    wrapper = PiperInferenceWrapper(
        generator.eval(),
        insert_pad_after_bos=mobile_piper,
        strip_piper_pads=mobile_direct,
    )
    target = output_dir / "model.onnx"
    tokens = torch.tensor([[2, 4, 5, 3]], dtype=torch.long)
    lengths = torch.tensor([4], dtype=torch.long)
    scales = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.long)
    logger.info("ONNX export step=2/5 action=build_graph opset=%d output=%s", opset, target)
    dynamic_axes = {
        "input": {1: "text_length"},
        "output": {2: "audio_length"},
    }
    if not mobile_direct:
        dynamic_axes["input"][0] = "batch"
        dynamic_axes["input_lengths"] = {0: "batch"}
        dynamic_axes["sid"] = {0: "batch"}
        dynamic_axes["output"][0] = "batch"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        torch.onnx.export(
            wrapper, (tokens, lengths, scales, sid), str(target),
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["output"], opset_version=opset, do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    logger.info("ONNX export step=3/5 action=check_model size_bytes=%d", target.stat().st_size)
    model = onnx.load(str(target)); onnx.checker.check_model(model)
    profiles = voice_profiles(metadata["speaker_map"], metadata["language_map"])
    text_input = _export_sherpa_android_text_package(
        onnx, model, output_dir, frontend, profiles, sample_rate=sample_rate,
        tokens=metadata["tokens"],
    )
    deployment = {
        "format": 2,
        "model_type": "multilingual-vits-piper-shaped",
        "sample_rate": sample_rate,
        "hop_length": config.hop_length,
        "inputs": ["input", "input_lengths", "scales", "sid"],
        "scales_default": [0.667, 1.0, 0.35],
        "scales": ["noise_scale", "length_scale", "duration_noise_scale"],
        "duration_predictor": {
            "type": "stochastic-log-normal",
            "deterministic_value": 0.0,
            "recommended_range": [0.0, 0.6],
        },
        "sid_formula": "speaker_id * num_languages + language_id",
        "frontend": frontend,
        "model_token_encoding": frontend.get(
            "token_encoding", DIRECT_TOKEN_ENCODING,
        ),
        "wire_token_encoding": (
            LEGACY_PIPER_TOKEN_ENCODING
            if mobile_piper or mobile_direct else frontend.get(
                "token_encoding", DIRECT_TOKEN_ENCODING,
            )
        ),
        "input_adapter": (
            "insert-pad-after-bos-v1"
            if mobile_piper else (
                "strip-piper-pads-v1" if mobile_direct else "none"
            )
        ),
        "frontend_note": "application supplies matching phoneme ids; stock sherpa multilingual switching requires an adapter",
        "text_input": text_input,
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
        "scales": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "sid": np.asarray([0], dtype=np.int64),
    })[0]
    if output.ndim != 3 or output.shape[1] != 1 or output.shape[2] <= 0:
        raise RuntimeError(f"unexpected ONNX output shape: {output.shape}")
    logger.info("ONNX runtime validation status=completed output_shape=%s", tuple(output.shape))
    return tuple(output.shape)
