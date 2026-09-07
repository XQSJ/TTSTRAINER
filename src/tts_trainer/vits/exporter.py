"""VITS 检查点导出为 Piper 形状的 ONNX，并做部署资源打包与一致性验证。 / Export a VITS checkpoint to a Piper-shaped ONNX graph with deployment resources and parity validation."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..checkpoints import require_checkpoint_format
from ..frontend import export_frontend_packs, frontend_contract_from_config
from ..frontend.contract import (DIRECT_TOKEN_ENCODING,
                                 LEGACY_PIPER_TOKEN_ENCODING,
                                 MOBILE_DIRECT_TOKEN_ENCODING,
                                 PIPER_TOKEN_ENCODING)
from ..frontend.conformance import save_frontend_conformance
from ..frontend.resources import (
    build_english_cmudict_json,
    inspect_openjtalk_dictionary,
)
from .composable import export_composable_bundle
from .config import VitsConfig
from .model import MultilingualVITS


logger = logging.getLogger(__name__)
# sherpa-onnx Android 运行时的目标版本，决定线上 token 序列格式。 /
# Target sherpa-onnx Android runtime version; it dictates the wire token sequence.
SHERPA_ONNX_ANDROID_VERSION = "1.13.4"


def require_mobile_blank_semantics(metadata: dict) -> None:
    """拒绝 blank 语义不符合移动端契约的检查点。 / Reject checkpoints whose blank semantics violate the mobile contract."""
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
    """对外暴露标准 Piper 四输入，内部保留语言/音色两路条件。 / Expose standard Piper inputs while retaining two internal conditions.

    sid 是复合音色档案 id：
      speaker_id = sid // num_languages
      language_id = sid % num_languages

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
        # 拆解复合 sid：模数取语言，整除取音色。 / Decompose composite sid: modulo gives language, floor-division gives speaker.
        language_ids = torch.remainder(sid, self.num_languages)
        speaker_ids = torch.div(sid, self.num_languages, rounding_mode="floor")
        return self.model.infer_deploy(input, input_lengths, language_ids, speaker_ids, scales)


def _config_from_metadata(raw: dict) -> VitsConfig:
    """从 metadata.json 还原 VitsConfig，列表字段需转回元组。 / Rebuild a VitsConfig from metadata.json; list fields must be restored to tuples."""
    config = dict(raw["config"])
    for key in ("decoder_resblock_kernel_sizes", "upsample_rates", "upsample_kernel_sizes"):
        if key in config:
            config[key] = tuple(config[key])
    return VitsConfig(**config)


def voice_profiles(speaker_map: dict[str, int], language_map: dict[str, int]) -> list[dict]:
    """枚举全部音色×语言组合并生成复合 sid 档案。 / Enumerate every voice×language pair into composite-sid profiles."""
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


def _representative_wire_input(tokens: list[str], frontend: dict) -> torch.Tensor:
    """构造真实非空部署输入。 / Build a real non-empty runtime wire input."""
    if len(tokens) < 5:
        raise ValueError("checkpoint vocabulary is missing required special tokens")
    # 选一个非特殊 token 的音素 id，保证验证输入非空。 / Pick a phoneme id beyond the special tokens so the probe input is non-empty.
    unit_id = 5 if len(tokens) > 5 else 4
    token_encoding = frontend.get("token_encoding", DIRECT_TOKEN_ENCODING)
    if token_encoding in {PIPER_TOKEN_ENCODING, MOBILE_DIRECT_TOKEN_ENCODING}:
        # sherpa-onnx 1.13.4 传输格式 / wire format:
        # BOS,(phone,PAD)*,EOS.
        values = [1, unit_id, 0, 2]
    else:
        values = [1, unit_id, 2]
    return torch.tensor([values], dtype=torch.long)


def _replace_onnx_metadata(model, values: dict[str, object]) -> None:
    """按 key 覆写 ONNX 元数据，未提及的条目原样保留。 / Overwrite ONNX metadata by key while preserving untouched entries."""
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
    """定位可随包分发的 espeak-ng-data 目录。 / Locate an espeak-ng-data directory suitable for bundling."""
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


def _find_pypinyin_data_dir() -> Path:
    """返回 Python 中文前端实际使用的词典目录。 / Return the exact dictionaries used by the Python Mandarin frontend."""
    spec = importlib.util.find_spec("pypinyin")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "Chinese mobile export requires pypinyin; install the asian extra: "
            "pip install -e '.[asian]'"
        )
    directory = Path(spec.origin).resolve().parent
    missing = [
        name for name in ("pinyin_dict.json", "phrases_dict.json")
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "pypinyin data required by the Android Mandarin frontend is "
            "missing: " + ", ".join(missing)
        )
    return directory


def english_cmudict_dir_for_export() -> Path:
    """生成并返回英语词典目录（内含 cmudict_data.json）。 / Build and return the English dictionary directory holding cmudict_data.json."""
    # loadCmuDict 通过 findG2pDictFile 在 dict_dir 中按文件名查找，目录里
    # 只需这一个文件；内容与 g2p-en 训练查询逐词一致。
    # loadCmuDict resolves the file by name inside dict_dir via
    # findG2pDictFile, so the directory holds exactly this one file; its
    # entries match the g2p-en training lookups word for word.
    target = build_english_cmudict_json()
    directory = target.parent / "cmudict-export"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, directory / "cmudict_data.json")
    return directory


def _export_sherpa_android_text_package(
    onnx, model, output_dir: Path, frontend: dict, profiles: list[dict],
    *, sample_rate: int, tokens: list[str],
) -> dict:
    """为 sherpa 的 eSpeak 前端生成按语言区分的元数据封装模型。 / Write language-specific metadata wrappers for sherpa's eSpeak frontend."""
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
            # <unk> 与 sherpa 的音素查表逻辑冲突，必须跳过。 / <unk> clashes with sherpa's phoneme lookup and must be skipped.
            continue
        if len(token) != 1:
            raise ValueError(
                "mobile eSpeak export requires Unicode-codepoint tokens; "
                f"found {token!r}"
            )
        # 空格 token 需要特殊行格式：只写 id，避免行内出现裸空格。 /
        # The space token needs a special line format: id only, no bare space.
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
    """把训练检查点导出为 Piper 兼容 ONNX 及全部部署资源。 / Export a training checkpoint to a Piper-compatible ONNX plus all deployment resources.

    流程共 5 步：加载检查点、构建图、数值一致性校验、写前端/部署资源、收尾。 /
    Five steps: load checkpoint, build graph, parity check, write frontend/deployment resources, finish.
    """
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
    checkpoint_audio = metadata.get("audio") or {}
    checkpoint_sample_rate = checkpoint_audio.get("sample_rate")
    if (
        checkpoint_sample_rate is not None
        and int(checkpoint_sample_rate) != int(sample_rate)
    ):
        raise ValueError(
            "export sample rate does not match checkpoint: "
            f"requested {sample_rate}, checkpoint {checkpoint_sample_rate}"
        )
    mobile_piper = frontend.get("token_encoding") == PIPER_TOKEN_ENCODING
    mobile_direct = (
        frontend.get("token_encoding") == MOBILE_DIRECT_TOKEN_ENCODING
    )
    config = _config_from_metadata(metadata)
    if config.num_languages != len(metadata["language_map"]):
        raise ValueError(
            "checkpoint language map does not match model architecture: "
            f"map={len(metadata['language_map'])}, model={config.num_languages}"
        )
    if config.num_speakers != len(metadata["speaker_map"]):
        raise ValueError(
            "checkpoint speaker map does not match model architecture: "
            f"map={len(metadata['speaker_map'])}, model={config.num_speakers}"
        )
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
    tokens = _representative_wire_input(metadata["tokens"], frontend)
    lengths = torch.tensor([tokens.shape[1]], dtype=torch.long)
    # scales=[noise,length,duration_noise]；导出前先取 PyTorch 参考输出用于一致性比对。 /
    # scales=[noise,length,duration_noise]; capture a PyTorch reference output for parity comparison.
    scales = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        reference_output = wrapper(tokens, lengths, scales, sid).cpu().numpy()
    logger.info("ONNX export step=2/5 action=build_graph opset=%d output=%s", opset, target)
    # 文本长度与音频长度必须动态；mobile-direct 因逐元素 PAD 剥离只支持 batch=1。 /
    # Text/audio lengths must be dynamic; mobile-direct supports only batch=1 due to the element-wise PAD stripping.
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
        # 随机上采样只执行一步，常量折叠告警无实际影响，直接忽略。 /
        # The stochastic upsampling runs a single step; the constant-folding warning is harmless noise.
        warnings.filterwarnings(
            "ignore", message="Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        torch.onnx.export(
            # dynamo=False：走传统 TorchScript 导出路径，保证算子级兼容 ORT 1.22。 /
            # dynamo=False: legacy TorchScript export path for operator-level ORT 1.22 compatibility.
            wrapper, (tokens, lengths, scales, sid), str(target),
            input_names=["input", "input_lengths", "scales", "sid"],
            output_names=["output"], opset_version=opset, do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    logger.info("ONNX export step=3/5 action=check_model size_bytes=%d", target.stat().st_size)
    model = onnx.load(str(target)); onnx.checker.check_model(model)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX parity validation requires: pip install -e '.[export]'"
        ) from exc
    session = ort.InferenceSession(
        str(target), providers=["CPUExecutionProvider"],
    )
    runtime_output = session.run(None, {
        "input": tokens.numpy(),
        "input_lengths": lengths.numpy(),
        "scales": scales.numpy(),
        "sid": sid.numpy(),
    })[0]
    if runtime_output.shape != reference_output.shape:
        raise RuntimeError(
            "ONNX parity validation shape mismatch: "
            f"PyTorch {reference_output.shape}, ONNX {runtime_output.shape}"
        )
    maximum_error = float(np.max(np.abs(runtime_output - reference_output)))
    # 容差 2e-4 覆盖 float32 图在 CPU 上的算子级数值偏差。 /
    # Tolerance 2e-4 covers float32 op-level divergence on CPU.
    if not np.allclose(
        runtime_output, reference_output, atol=2e-4, rtol=2e-4,
    ):
        raise RuntimeError(
            "ONNX output differs from PyTorch inference; "
            f"maximum_absolute_error={maximum_error:.6g}"
        )
    logger.info(
        "ONNX export parity status=passed maximum_absolute_error=%.6g "
        "test_input_length=%d",
        maximum_error, tokens.shape[1],
    )
    with target.open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    profiles = voice_profiles(metadata["speaker_map"], metadata["language_map"])
    conformance = metadata.get("frontend_conformance")
    espeak_data_dir = None
    if any(
        profile.get("provider") == "espeak-ng"
        for profile in frontend.get("languages", {}).values()
    ):
        espeak_data_dir = _find_espeak_data_dir()
    frontend_packs = export_frontend_packs(
        output_dir, frontend, conformance, metadata["language_map"],
        model_sha256=model_sha256,
        espeak_data_dir=espeak_data_dir,
    )
    frontend_resources = {}
    if espeak_data_dir is not None:
        frontend_resources["espeak-ng"] = espeak_data_dir
    if any(
        language == "zh" and profile.get("provider") == "piper-plus-g2p"
        for language, profile in frontend.get("languages", {}).items()
    ):
        # Export the same pypinyin database that produced the frozen training
        # tokens. The Android native frontend accepts this JSON schema but
        # expects Piper's filenames inside its dictionary directory.
        # 导出训练时实际使用的 pypinyin 数据；Android 原生前端兼容该 JSON
        # 格式，但要求目录内使用 Piper 约定的文件名。
        frontend_resources["piper-plus-g2p:zh"] = _find_pypinyin_data_dir()
    if any(
        language == "en" and profile.get("provider") == "piper-plus-g2p"
        for language, profile in frontend.get("languages", {}).items()
    ):
        # g2p-en reads the nltk cmudict; the Android native English backend
        # otherwise falls back to the copy compiled into libpiper_plus.so.
        # Ship the training-side dictionary in the language pack so both ends
        # resolve every word identically.
        # g2p-en 训练读取 nltk cmudict；Android 原生英语后端在缺少外部词典时
        # 回退到编译进 libpiper_plus.so 的内嵌副本。把训练侧词典随语言包
        # 分发，保证两端逐词一致。
        frontend_resources["piper-plus-g2p:en"] = english_cmudict_dir_for_export()
    # 日语无论走 openjtalk 还是 piper-plus-g2p 都依赖同一套 OpenJTalk 词典。 /
    # Japanese needs the same OpenJTalk dictionary whether via openjtalk or piper-plus-g2p.
    needs_openjtalk_dictionary = any(
        profile.get("provider") == "openjtalk"
        or (
            language == "ja"
            and profile.get("provider") == "piper-plus-g2p"
        )
        for language, profile in frontend.get("languages", {}).items()
    )
    if needs_openjtalk_dictionary:
        openjtalk = inspect_openjtalk_dictionary()
        if not openjtalk.ready:
            raise FileNotFoundError(
                "mobile export requires the OpenJTalk dictionary inside the "
                "Japanese language pack; run: tts-trainer frontends ensure openjtalk"
            )
        android_required = (
            "sys.dic", "matrix.bin", "char.bin",
            "left-id.def", "right-id.def", "unk.dic",
        )
        # Android 原生 OpenJTalk 只加载这六个文件，缺失即无法部署。 /
        # The Android native OpenJTalk loads exactly these six files; any gap blocks deployment.
        missing = [
            name for name in android_required
            if not (openjtalk.path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "OpenJTalk dictionary cannot be deployed to Android; missing: "
                + ", ".join(missing)
            )
        if any(
            profile.get("provider") == "openjtalk"
            for profile in frontend.get("languages", {}).values()
        ):
            frontend_resources["openjtalk"] = openjtalk.path
        if frontend.get("languages", {}).get("ja", {}).get("provider") \
                == "piper-plus-g2p":
            frontend_resources["piper-plus-g2p:ja"] = openjtalk.path
    composable = export_composable_bundle(
        output_dir,
        generator,
        metadata,
        frontend,
        conformance,
        tokens,
        lengths,
        scales,
        sample_rate=sample_rate,
        opset=opset,
        insert_pad_after_bos=mobile_piper,
        strip_piper_pads=mobile_direct,
        frontend_resources=frontend_resources,
    )
    text_input = _export_sherpa_android_text_package(
        onnx, model, output_dir, frontend, profiles, sample_rate=sample_rate,
        tokens=metadata["tokens"],
    )
    # deployment.json 是 Android 端加载模型的唯一契约来源。 /
    # deployment.json is the single contract the Android side loads the model from.
    deployment = {
        "format": 2,
        "model_type": "multilingual-vits-piper-shaped",
        "model_sha256": model_sha256,
        "sample_rate": sample_rate,
        "hop_length": config.hop_length,
        "inputs": ["input", "input_lengths", "scales", "sid"],
        "scales_default": [0.667, 1.0, 0.35],
        "scales": ["noise_scale", "length_scale", "duration_noise_scale"],
        "duration_predictor": {
            "type": config.duration_predictor_type.replace("_", "-"),
            "channels": (
                config.hidden_channels
                if config.duration_predictor_type == "stochastic_lognormal"
                else config.duration_predictor_channels
            ),
            "flow_layers": (
                0 if config.duration_predictor_type == "stochastic_lognormal"
                else config.duration_predictor_flow_layers
            ),
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
        "frontend_packs": frontend_packs,
        "composable": composable,
        "num_languages": config.num_languages,
        "num_speakers": config.num_speakers,
        "voice_profiles": profiles,
        "export_validation": {
            "pytorch_onnx_parity": True,
            "maximum_absolute_error": maximum_error,
            "input_length": int(tokens.shape[1]),
        },
    }
    (output_dir / "model.onnx.json").write_text(json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "frontend.json").write_text(json.dumps(frontend, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "tokens.json").write_text(json.dumps({"tokens": metadata["tokens"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    tokens_text = "".join(f"{token} {index}\n" for index, token in enumerate(metadata["tokens"]))
    (output_dir / "tokens.txt").write_text(tokens_text, encoding="utf-8")
    if conformance:
        save_frontend_conformance(conformance, output_dir / "frontend.conformance.json")
    logger.info(
        "ONNX export step=4/5 action=write_resources profiles=%d directory=%s",
        len(profiles), output_dir,
    )
    logger.info("ONNX export step=5/5 action=completed model=%s", target)
    return target


def validate_onnx_runtime(model_path: str | Path) -> tuple[int, ...]:
    """用 onnxruntime 对导出模型做冒烟推理校验并返回输出形状。 / Smoke-test the exported model with onnxruntime and return the output shape."""
    import onnxruntime as ort
    model_path = Path(model_path)
    deployment = json.loads(
        (model_path.parent / "model.onnx.json").read_text(encoding="utf-8")
    )
    tokens = json.loads(
        (model_path.parent / "tokens.json").read_text(encoding="utf-8")
    )["tokens"]
    test_input = _representative_wire_input(
        tokens, deployment.get("frontend") or {},
    ).numpy()
    profiles = deployment.get("voice_profiles") or []
    sid = int(profiles[0]["sid"]) if profiles else 0
    logger.info("ONNX runtime validation status=started provider=CPUExecutionProvider model=%s", model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {
        "input": test_input,
        "input_lengths": np.asarray([test_input.shape[1]], dtype=np.int64),
        "scales": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "sid": np.asarray([sid], dtype=np.int64),
    })[0]
    if output.ndim != 3 or output.shape[1] != 1 or output.shape[2] <= 0:
        raise RuntimeError(f"unexpected ONNX output shape: {output.shape}")
    if not np.isfinite(output).all():
        # NaN/Inf 通常意味着图里残留了训练态算子或数值不稳定路径。 /
        # NaN/Inf usually means a training-mode op leaked into the graph or an unstable numeric path.
        raise RuntimeError("ONNX output contains NaN or infinity")
    peak = float(np.max(np.abs(output)))
    if peak > 1.001:
        raise RuntimeError(f"ONNX output exceeds waveform range: peak={peak}")
    logger.info(
        "ONNX runtime validation status=completed output_shape=%s "
        "input_length=%d sid=%d peak=%.6f",
        tuple(output.shape), test_input.shape[1], sid, peak,
    )
    return tuple(output.shape)
