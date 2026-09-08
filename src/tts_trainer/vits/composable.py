"""可组合导出：不含嵌入表的核心 ONNX 加独立安装的语言包/音色包。 / Composable export: an embedding-free core ONNX plus independently installable language and voice packs."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..frontend.contract import MOBILE_DIRECT_TOKEN_ENCODING
from .model import MultilingualVITS


logger = logging.getLogger(__name__)
# 可组合清单格式版本，端侧据此校验包兼容性。 / Composable manifest format version; the device side checks pack compatibility against it.
COMPOSABLE_FORMAT = 1


class ComposableInferenceWrapper(nn.Module):
    """导出核心图：语言/音色表外置到包中，以 embedding 向量为输入。 / Export a core graph whose language and speaker tables live in packs."""

    def __init__(
        self,
        model: MultilingualVITS,
        *,
        insert_pad_after_bos: bool = False,
        strip_piper_pads: bool = False,
    ):
        super().__init__()
        if insert_pad_after_bos and strip_piper_pads:
            raise ValueError(
                "insert_pad_after_bos and strip_piper_pads are mutually exclusive"
            )
        self.model = model
        self.insert_pad_after_bos = insert_pad_after_bos
        self.strip_piper_pads = strip_piper_pads

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        scales: torch.Tensor,
        language_embedding: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ):
        if self.insert_pad_after_bos:
            # 在 BOS 后插入一个 PAD，对齐 sherpa 1.13.4 的历史线上序列。 /
            # Insert one PAD after BOS to match sherpa 1.13.4's historical wire sequence.
            pad = torch.zeros_like(input[:, :1])
            input = torch.cat((input[:, :1], pad, input[:, 1:]), dim=1)
            input_lengths = input_lengths + 1
        elif self.strip_piper_pads:
            # 用位置掩码剔除所有传输 PAD(token 0)，得到训练用的紧凑序列。 /
            # Mask out every transport PAD (token 0) to get the compact training sequence.
            positions = torch.arange(
                input.shape[1], device=input.device,
            ).unsqueeze(0)
            valid = positions < input_lengths.unsqueeze(1)
            keep = valid & input.ne(0)
            input = torch.masked_select(input, keep).unsqueeze(0)
            input_lengths = keep.sum(dim=1)
        return self.model.infer_deploy_embeddings(
            input,
            input_lengths,
            language_embedding,
            speaker_embedding,
            scales,
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tree_identity(path: Path) -> tuple[str, int]:
    """对资源树按相对路径名+文件内容做哈希，返回 (摘要, 总字节数)。 / Hash relative names and file contents for a deployable resource tree; returns (digest, total bytes)."""
    digest = hashlib.sha256()
    total = 0
    for source in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = source.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
                digest.update(chunk)
    return digest.hexdigest(), total


def _write_vector(path: Path, vector: torch.Tensor) -> dict:
    """把嵌入向量写成小端 float32 裸文件并返回清单描述。 / Write an embedding vector as a little-endian float32 raw file and return its manifest entry."""
    # 强制小端 float32，保证跨端（Android ARM）字节序一致。 / Force LE float32 so Android/ARM reads identical bytes.
    values = vector.detach().cpu().numpy().astype("<f4", copy=False)
    path.write_bytes(values.tobytes(order="C"))
    return {
        "path": path.name,
        "dtype": "float32-le",
        "shape": [int(values.shape[0])],
        "sha256": _sha256(path),
    }


def _write_pack_archive(pack_dir: Path, target: Path) -> None:
    """把包目录以确定性顺序压缩为 zip。 / Zip a pack directory into an archive in deterministic order."""
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for source in sorted(pack_dir.rglob("*")):
            if source.is_file():
                archive.write(source, source.relative_to(pack_dir))


def _write_android_pinyin_data(source: Path, destination: Path) -> None:
    """把 pypinyin 声调标记转换为 Android 原生 TONE3 格式。 / Convert pypinyin tone marks to the native Android TONE3 format."""
    try:
        from pypinyin.contrib.tone_convert import to_tone3
    except ImportError as exc:
        raise RuntimeError(
            "Chinese Android export requires pypinyin tone conversion"
        ) from exc

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, str):
            # 单字词典可能包含多个逗号分隔的读音，必须逐个转换。
            # Single-character entries may contain comma-separated readings.
            return ",".join(
                to_tone3(item, neutral_tone_with_five=True)
                for item in value.split(",")
            )
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    destination.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("pinyin_dict.json", "pinyin_single.json"),
        ("phrases_dict.json", "pinyin_phrases.json"),
    ):
        payload = json.loads((source / source_name).read_text(encoding="utf-8"))
        (destination / target_name).write_text(
            json.dumps(convert(payload), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )


def export_composable_bundle(
    output_dir: str | Path,
    model: MultilingualVITS,
    metadata: dict,
    frontend: dict,
    conformance: dict | None,
    representative_input: torch.Tensor,
    input_lengths: torch.Tensor,
    scales: torch.Tensor,
    *,
    sample_rate: int,
    opset: int,
    insert_pad_after_bos: bool,
    strip_piper_pads: bool,
    frontend_resources: dict[str, Path] | None = None,
) -> dict:
    """导出声学核心与可独立安装的语言包、音色包。 / Export core + independently installable language and voice packs.

    核心 ONNX 不再包含 language/speaker embedding 表。语言包和音色包只
    保存各自向量与契约，并用 core_sha256 严格绑定对应声学核心。
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "composable ONNX export requires: pip install -e '.[export]'"
        ) from exc

    root = Path(output_dir) / "composable"
    core_dir = root / "core"
    language_root = root / "languages"
    voice_root = root / "voices"
    package_root = root / "packages"
    for directory in (core_dir, language_root, voice_root, package_root):
        directory.mkdir(parents=True, exist_ok=True)

    wrapper = ComposableInferenceWrapper(
        model.eval(),
        insert_pad_after_bos=insert_pad_after_bos,
        strip_piper_pads=strip_piper_pads,
    )
    # 用第 0 行嵌入作为导出与校验的代表性输入；运行时由包提供真实向量。 /
    # Row-0 embeddings serve as the representative export/parity input; packs supply real vectors at runtime.
    language_vector = model.conditioning.language_embedding.weight[0:1]
    speaker_vector = model.conditioning.speaker_embedding.weight[0:1]
    with torch.no_grad():
        reference = wrapper(
            representative_input,
            input_lengths,
            scales,
            language_vector,
            speaker_vector,
        ).cpu().numpy()

    target = core_dir / "model.onnx"
    # 文本/音频长度动态；strip_piper_pads 的逐元素掩码只支持 batch=1。 /
    # Dynamic text/audio lengths; the strip_piper_pads element-wise mask supports only batch=1.
    dynamic_axes = {
        "input": {1: "text_length"},
        "output": {2: "audio_length"},
    }
    if not strip_piper_pads:
        dynamic_axes.update({
            "input": {0: "batch", 1: "text_length"},
            "input_lengths": {0: "batch"},
            "language_embedding": {0: "batch"},
            "speaker_embedding": {0: "batch"},
            "output": {0: "batch", 2: "audio_length"},
        })
    with warnings.catch_warnings():
        # 单步随机上采样触发的常量折叠告警无实际影响。 /
        # Constant-folding warning from the single-step stochastic upsampling is harmless.
        warnings.filterwarnings(
            "ignore",
            message="Constant folding - Only steps=1 can be constant folded.*",
            category=UserWarning,
        )
        torch.onnx.export(
            wrapper,
            (
                representative_input,
                input_lengths,
                scales,
                language_vector,
                speaker_vector,
            ),
            str(target),
            input_names=[
                # dynamo=False 保持传统 TorchScript 路径，兼容 ORT 1.22。 /
                # dynamo=False keeps the legacy TorchScript path for ORT 1.22.
                "input",
                "input_lengths",
                "scales",
                "language_embedding",
                "speaker_embedding",
            ],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    onnx_model = onnx.load(str(target))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(target), providers=["CPUExecutionProvider"],
    )
    runtime = session.run(None, {
        "input": representative_input.numpy(),
        "input_lengths": input_lengths.numpy(),
        "scales": scales.numpy(),
        "language_embedding": language_vector.detach().numpy(),
        "speaker_embedding": speaker_vector.detach().numpy(),
    })[0]
    maximum_error = float(np.max(np.abs(runtime - reference)))
    if runtime.shape != reference.shape or not np.allclose(
        runtime, reference, atol=2e-4, rtol=2e-4,
    ):
        raise RuntimeError(
            "composable ONNX differs from PyTorch inference; "
            f"maximum_absolute_error={maximum_error:.6g}"
        )

    # core 哈希是所有包的兼容性锚点：不匹配即拒绝加载。 /
    # The core hash is the compatibility anchor for every pack: mismatch means refuse to load.
    core_sha256 = _sha256(target)
    tokens_payload = {"tokens": metadata["tokens"]}
    (core_dir / "tokens.json").write_text(
        json.dumps(tokens_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    core_manifest = {
        "format": COMPOSABLE_FORMAT,
        "type": "tts-acoustic-core",
        "layout": "composable-vits-v1",
        "core_id": core_sha256,
        "model": "model.onnx",
        "model_sha256": core_sha256,
        "tokens": "tokens.json",
        "sample_rate": int(sample_rate),
        "inputs": [
            "input",
            "input_lengths",
            "scales",
            "language_embedding",
            "speaker_embedding",
        ],
        "language_embedding_dim": int(
            model.config.language_embedding_channels
        ),
        "speaker_embedding_dim": int(
            model.config.speaker_embedding_channels
        ),
        "token_encoding": frontend.get("token_encoding"),
        "input_adapter": (
            "insert-pad-after-bos-v1"
            if insert_pad_after_bos else (
                "strip-piper-pads-v1" if strip_piper_pads else "none"
            )
        ),
        "scales_default": [0.667, 1.0, 0.35],
        "export_validation": {
            "pytorch_onnx_parity": True,
            "maximum_absolute_error": maximum_error,
        },
    }
    (core_dir / "manifest.json").write_text(
        json.dumps(core_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cases = list((conformance or {}).get("cases") or [])
    frontend_resources = frontend_resources or {}
    language_entries = []
    for language, language_id in sorted(
        metadata["language_map"].items(), key=lambda item: item[1],
    ):
        pack_dir = language_root / language
        shutil.rmtree(pack_dir, ignore_errors=True)
        pack_dir.mkdir(parents=True, exist_ok=True)
        # 语言包内嵌该语言的嵌入向量，供核心图运行时查用。 /
        # The language pack carries this language's embedding vector for runtime lookup.
        embedding = _write_vector(
            pack_dir / "embedding.f32",
            model.conditioning.language_embedding.weight[language_id],
        )
        language_cases = [
            dict(case) for case in cases
            if case.get("language") == language
        ]
        (pack_dir / "conformance.json").write_text(
            json.dumps({
                "format": int((conformance or {}).get("format", 1)),
                "languages": [language],
                "cases": language_cases,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        frontend_profile = dict(frontend["languages"][language])
        provider = frontend_profile["provider"]
        # 按 provider 把运行时资源(espeak 数据/词典/拼音库)复制进语言包。 /
        # Copy runtime resources (espeak data/dictionaries/pinyin DBs) into the pack by provider.
        runtime_resource = None
        if provider == "espeak-ng":
            source = frontend_resources.get(provider)
            if source is None or not source.is_dir():
                raise FileNotFoundError(
                    f"language {language} requires espeak-ng-data in its "
                    "deployable language pack"
                )
            relative = Path("runtime") / "espeak-ng-data"
            destination = pack_dir / relative
            shutil.copytree(source, destination, dirs_exist_ok=True)
            resource_sha256, resource_bytes = _tree_identity(destination)
            runtime_resource = {
                "id": "espeak-ng-data",
                "delivery": "language-pack",
                "path": relative.as_posix(),
                "sha256": resource_sha256,
                "bytes": resource_bytes,
            }
        elif provider == "openjtalk":
            source = frontend_resources.get(provider)
            if source is None or not source.is_dir():
                raise FileNotFoundError(
                    f"language {language} requires an OpenJTalk dictionary in "
                    "its deployable language pack"
                )
            relative = Path("runtime") / "open_jtalk_dic"
            destination = pack_dir / relative
            shutil.copytree(source, destination, dirs_exist_ok=True)
            resource_sha256, resource_bytes = _tree_identity(destination)
            runtime_resource = {
                "id": "openjtalk-dictionary",
                "delivery": "language-pack",
                "path": relative.as_posix(),
                "sha256": resource_sha256,
                "bytes": resource_bytes,
            }
        elif provider == "piper-plus-g2p":
            source = frontend_resources.get(f"{provider}:{language}")
            if language == "zh":
                if source is None or not source.is_dir():
                    raise FileNotFoundError(
                        "Chinese language pack requires pypinyin dictionaries "
                        "for Android Piper Plus G2P"
                    )
                relative = Path("runtime") / "piper-plus-g2p-data"
                destination = pack_dir / relative
                _write_android_pinyin_data(source, destination)
                resource_sha256, resource_bytes = _tree_identity(destination)
                runtime_resource = {
                    "id": "piper-plus-g2p-data",
                    "delivery": "language-pack",
                    "path": relative.as_posix(),
                    "sha256": resource_sha256,
                    "bytes": resource_bytes,
                }
            elif language == "ja":
                if source is None or not source.is_dir():
                    raise FileNotFoundError(
                        "Japanese Piper Plus language pack requires an "
                        "OpenJTalk dictionary"
                    )
                relative = Path("runtime") / "open_jtalk_dic"
                destination = pack_dir / relative
                shutil.copytree(source, destination, dirs_exist_ok=True)
                resource_sha256, resource_bytes = _tree_identity(destination)
                runtime_resource = {
                    "id": "openjtalk-dictionary",
                    "delivery": "language-pack",
                    "path": relative.as_posix(),
                    "sha256": resource_sha256,
                    "bytes": resource_bytes,
                }
            elif language == "en":
                # g2p-en trains against the nltk cmudict; ship that dictionary
                # so the native English backend never falls back to its
                # embedded copy and drifts from the frozen training tokens.
                # g2p-en 训练读取 nltk cmudict；随包分发该词典，避免原生
                # 英语后端回退到内嵌副本而偏离训练 token。
                if source is None or not (source / "cmudict_data.json").is_file():
                    raise FileNotFoundError(
                        "English Piper Plus language pack requires "
                        "cmudict_data.json"
                    )
                relative = Path("runtime") / "cmudict"
                destination = pack_dir / relative
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    source / "cmudict_data.json",
                    destination / "cmudict_data.json",
                )
                resource_sha256, resource_bytes = _tree_identity(destination)
                runtime_resource = {
                    "id": "cmudict",
                    "delivery": "language-pack",
                    "path": relative.as_posix(),
                    "sha256": resource_sha256,
                    "bytes": resource_bytes,
                }
            else:
                # 其他语言没有可打包资源，由宿主应用自带 G2P。 /
                # No bundleable resource for other languages; the host app ships its own G2P.
                runtime_resource = {
                    "id": "piper-plus-g2p",
                    "delivery": "application-runtime",
                }

        # 语言包清单：前端契约 + 一致性用例 + core 绑定。 /
        # Language-pack manifest: frontend contract + conformance cases + core binding.
        manifest = {
            "format": COMPOSABLE_FORMAT,
            "type": "tts-language-pack",
            "id": language,
            "language": language,
            "source_language_id": int(language_id),
            "compatible_core_sha256": core_sha256,
            "embedding": embedding,
            "frontend": frontend_profile,
            "normalization": frontend.get("normalization"),
            "token_encoding": frontend.get("token_encoding"),
            "conformance": "conformance.json",
        }
        if runtime_resource is not None:
            manifest["runtime_resource"] = runtime_resource
        (pack_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        archive = package_root / f"language-{language}.zip"
        _write_pack_archive(pack_dir, archive)
        language_entries.append({
            "id": language,
            "provider": manifest["frontend"]["provider"],
            "package": f"packages/{archive.name}",
            "sha256": _sha256(archive),
            "bytes": archive.stat().st_size,
        })

    voice_entries = []
    for speaker, speaker_id in sorted(
        metadata["speaker_map"].items(), key=lambda item: item[1],
    ):
        pack_dir = voice_root / speaker
        shutil.rmtree(pack_dir, ignore_errors=True)
        pack_dir.mkdir(parents=True, exist_ok=True)
        # 音色包只需一个说话人向量与默认 scales。 / A voice pack needs just one speaker vector plus default scales.
        embedding = _write_vector(
            pack_dir / "embedding.f32",
            model.conditioning.speaker_embedding.weight[speaker_id],
        )
        manifest = {
            "format": COMPOSABLE_FORMAT,
            "type": "tts-voice-pack",
            "id": speaker,
            "speaker": speaker,
            "source_speaker_id": int(speaker_id),
            "compatible_core_sha256": core_sha256,
            "embedding": embedding,
            "defaults": {
                "noise_scale": 0.667,
                "length_scale": 1.0,
                "duration_noise_scale": 0.35,
            },
        }
        (pack_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        archive = package_root / f"voice-{speaker}.zip"
        _write_pack_archive(pack_dir, archive)
        voice_entries.append({
            "id": speaker,
            "package": f"packages/{archive.name}",
            "sha256": _sha256(archive),
            "bytes": archive.stat().st_size,
        })

    # 总目录：列出核心与全部包及其校验和，是 Android 端发现资源的入口。 /
    # Top catalog: lists the core and every pack with checksums; the Android resource-discovery entry point.
    catalog = {
        "format": COMPOSABLE_FORMAT,
        "layout": "composable-vits-v1",
        "core": {
            "id": core_sha256,
            "manifest": "core/manifest.json",
            "model": "core/model.onnx",
            "tokens": "core/tokens.json",
        },
        "languages": language_entries,
        "voices": voice_entries,
        "compatibility": (
            "A pack is loadable only when compatible_core_sha256 exactly "
            "matches the bundled core. Quantization tooling must preserve or "
            "rewrite this identity explicitly."
        ),
    }
    (root / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "COMPOSABLE EXPORT | core=%s | languages=%d | voices=%d | directory=%s",
        core_sha256[:12], len(language_entries), len(voice_entries), root,
    )
    return {
        "format": COMPOSABLE_FORMAT,
        "layout": "composable-vits-v1",
        "catalog": "composable/catalog.json",
        "core": "composable/core/model.onnx",
        "core_sha256": core_sha256,
        "language_packs": len(language_entries),
        "voice_packs": len(voice_entries),
    }
