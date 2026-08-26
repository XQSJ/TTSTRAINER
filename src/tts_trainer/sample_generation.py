"""调用 Qwen3-TTS 教师模型生成语音样本，为 VITS 学生模型蒸馏训练数据。 / Drive the Qwen3-TTS teacher to synthesize speech samples for student-model distillation.

管理共享音色数据集缓存、参考音频策略与最终 metadata 组装。 / Manages the shared voice-dataset cache, reference-audio strategies, and final metadata assembly.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import threading
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .experiments import prepare_experiment, resolve_experiment
from .languages import resolve_language_registry
from .logging_utils import (configure_logging_from_config, format_duration,
                            log_section)
from .manifest import read_manifest
from .qwen_teacher import load_qwen_teacher
from .text_generation import generate_texts, text_corpus_path


logger = logging.getLogger(__name__)
# 公开音色 ID 的合法字符集（用于目录名与配置校验）。 / Valid charset for public voice IDs (used as directory names and validated in configs).
VOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class GenerationText:
    """一条待合成的文本及其语言。 / One text to synthesize with its language."""

    text: str
    language: str


@dataclass(frozen=True)
class GenerationJob:
    """单条音频生成任务：文本、候选序号与输出路径。 / One audio job: text, candidate index, and output path."""

    item: GenerationText
    candidate: int
    output: Path


@dataclass(frozen=True)
class RegenerationPlan:
    """按语言选择性失效缓存的计划。 / Per-language selective cache invalidation plan."""

    audio_languages: frozenset[str]
    reference_languages: frozenset[str]


class _BatchHeartbeat:
    """在同步 Qwen 调用耗时过久时周期性打点，而不是误报停滞。 / Report a slow synchronous Qwen call periodically without pretending it has stalled."""

    def __init__(self, *, batch_number: int, language: str, interval: float):
        self.batch_number = batch_number
        self.language = language
        self.interval = max(float(interval), 1.0)
        self.started = time.monotonic()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        # 后台心跳线程：间隔到期仍未停止就告警一次。 / Heartbeat thread: warn once per interval while the batch is still running.
        while not self.stopped.wait(self.interval):
            logger.warning(
                "AUDIO BATCH STILL RUNNING | batch=%d | language=%s | "
                "elapsed=%s | Qwen has not returned yet",
                self.batch_number, self.language,
                format_duration(time.monotonic() - self.started),
            )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stopped.set()
        self.thread.join(timeout=1)


def _file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256。 / Stream-compute a file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regeneration_plan(
    voice: dict, generation: dict, languages: tuple[str, ...],
) -> RegenerationPlan:
    """解析按语言选择性失效缓存的计划，同时保留旧字段行为。 / Resolve selective cache invalidation while preserving legacy behavior."""
    settings = voice.get("regenerate")
    # 旧版整音色重生成开关，与新版 regenerate 结构互斥。 / Legacy whole-voice regenerate switch; mutually exclusive with the newer regenerate object.
    legacy_audio = bool(
        voice.get("regenerate_audio", generation.get("overwrite", False))
    )
    if settings is None:
        selected = frozenset(languages) if legacy_audio else frozenset()
        return RegenerationPlan(selected, frozenset())
    if legacy_audio:
        raise ValueError(
            "dataset.voice.regenerate cannot be combined with "
            "regenerate_audio=true or generation.overwrite=true"
        )
    if not isinstance(settings, dict):
        raise ValueError("dataset.voice.regenerate must be a JSON object")
    unknown = sorted(set(settings) - {"audio", "references", "languages"})
    if unknown:
        raise ValueError(
            "dataset.voice.regenerate has unknown fields: "
            + ", ".join(unknown)
        )
    audio = settings.get("audio", False)
    references = settings.get("references", False)
    if not isinstance(audio, bool) or not isinstance(references, bool):
        raise ValueError(
            "dataset.voice.regenerate.audio and references must be true or false"
        )
    configured_languages = settings.get("languages", "all")
    if configured_languages == "all":
        selected = frozenset(languages)
    elif isinstance(configured_languages, list) and configured_languages:
        selected = frozenset(
            str(language).strip().lower()
            for language in configured_languages
        )
        unsupported = sorted(selected - set(languages))
        if unsupported:
            raise ValueError(
                "dataset.voice.regenerate.languages must be enabled in "
                "experiment.languages: " + ", ".join(unsupported)
            )
    else:
        raise ValueError(
            'dataset.voice.regenerate.languages must be "all" or a non-empty array'
        )
    if references and not audio:
        # 参考重生成必须连带音频重生成，避免新旧参考混入同一批训练 WAV。 / Reference regen requires audio regen so old training WAVs never mix with a new reference.
        raise ValueError(
            "reference regeneration also requires regenerate.audio=true so old "
            "training WAVs are not mixed with a new language reference"
        )
    return RegenerationPlan(
        selected if audio else frozenset(),
        selected if references else frozenset(),
    )


def _invalidate_language_references(
    references: Path, voice: dict, languages: frozenset[str],
) -> None:
    """只删除派生的语言参考，绝不替换音色主锚点。 / Remove only derived language references; never replace the voice anchor."""
    if not languages:
        return
    if str(voice.get("mode") or "") != "design":
        raise ValueError(
            "reference regeneration is only available for mode=design; use a "
            "new voice_id when changing an uploaded clone reference"
        )
    strategy = str(
        voice.get("reference_strategy", "shared")
    ).strip().lower()
    if strategy == "shared":
        raise ValueError(
            "shared strategy has only the immutable master reference; use a new "
            "voice_id to replace it, or regenerate audio while preserving it"
        )
    # cascade 策略产物名为 localized-*，其余为 designed-*。 / Cascade artifacts are named localized-*, others designed-*.
    prefix = "localized" if strategy == "cascade" else "designed"
    removed = []
    for language in sorted(languages):
        for suffix in (".wav", ".txt"):
            target = references / f"{prefix}-{language}{suffix}"
            if target.is_file():
                target.unlink()
                removed.append(target.name)
    logger.warning(
        "REFERENCE REGENERATION | strategy=%s | languages=%s | removed=%s | "
        "master_reference=preserved",
        strategy, ",".join(sorted(languages)),
        ",".join(removed) if removed else "none",
    )


def _voice_identity(raw: dict, generation: dict, voice: dict) -> dict:
    """返回一个公开音色 ID 拥有的不可变设置（用于身份锁）。 / Return the immutable settings owned by one public voice ID (used for the identity lock)."""
    mode = str(voice.get("mode") or "")
    reference_strategy = str(
        voice.get("reference_strategy", "shared")
    ).strip().lower()
    reference_identity = None
    if mode == "clone" and voice.get("reference_audio"):
        # 克隆模式以参考音频内容哈希作为身份指纹，防止悄悄换源。 / Clone mode fingerprints the reference audio by content hash so it cannot be silently swapped.
        reference_path = Path(voice["reference_audio"]).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"reference audio does not exist: {reference_path}")
        reference_identity = {
            "sha256": _file_sha256(reference_path),
            "suffix": reference_path.suffix.lower(),
        }
    identity = {
        "format": 1,
        "mode": mode,
        "prompt": str(voice.get("prompt") or "").strip() or None,
        "reference_text": (
            str(voice.get("reference_text") or "").strip() or None
            if reference_strategy in {"shared", "cascade"} else None
        ),
        "reference_language": (
            str(voice.get("reference_language", "en")).strip().lower()
            if mode == "design"
            and reference_strategy in {"shared", "cascade"} else None
        ),
        "reference_audio": reference_identity,
        "x_vector_only_mode": bool(voice.get("x_vector_only_mode", False)),
        "models": generation.get("models", {}),
        "generation_kwargs": generation.get("generation_kwargs", {}),
        "audio": {
            "sample_rate": int(raw["audio"]["sample_rate"]),
            "postprocess": generation.get("audio_postprocess", {}),
        },
    }
    if mode == "design" and reference_strategy != "shared":
        identity["reference_strategy"] = reference_strategy
    if voice.get("reference_texts"):
        identity["reference_texts"] = voice["reference_texts"]
    return identity


def _identity_digest(identity: dict) -> str:
    """对身份字典做规范化 JSON 后取 SHA-256。 / Canonical-JSON the identity dict and hash it with SHA-256."""
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _move_tree(source: Path, destination: Path) -> None:
    """迁移旧目录树且不覆盖内容不同的缓存文件。 / Move a legacy tree without replacing a different cached file."""
    if not source.is_dir():
        return
    if not destination.exists():
        source.replace(destination)
        return
    # 逐文件合并：同名且哈希一致才允许并存，否则视为冲突报错。 / Merge file-by-file: same-name files must hash-identical, otherwise conflict.
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if _file_sha256(path) != _file_sha256(target):
                raise RuntimeError(
                    f"voice cache migration conflict: {path} and {target} differ"
                )
            path.unlink()
        else:
            path.replace(target)
    for path in sorted(
        (item for item in source.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts), reverse=True,
    ):
        path.rmdir()
    source.rmdir()


def _canonicalize_reference_names(references: Path) -> None:
    """从共享参考路径中去掉模型内部 speaker 标签。 / Remove the model-internal speaker label from shared reference paths."""
    if not references.is_dir():
        return
    designed = references / "designed.wav"
    legacy_designs = sorted(references.glob("*.designed.wav"))
    if not designed.is_file() and len(legacy_designs) == 1:
        legacy_designs[0].replace(designed)
    uploaded = sorted(references.glob("*.uploaded.*"))
    if len(uploaded) == 1:
        suffix = "".join(uploaded[0].suffixes[1:]) or uploaded[0].suffix or ".wav"
        canonical = references / f"uploaded{suffix}"
        if not canonical.is_file():
            uploaded[0].replace(canonical)


def _write_legacy_voice_alias(legacy: Path, destination: Path, previous: dict) -> None:
    """WAV 迁移到音色 ID 根目录后，保留旧 metadata 路径可用。 / Keep old metadata paths working after moving WAVs to the voice-ID root."""
    legacy.mkdir(parents=True, exist_ok=True)
    for name in ("references", "wavs"):
        target = destination / name
        if target.exists():
            (legacy / name).symlink_to(Path("..") / name, target_is_directory=True)
    compatibility = dict(previous)
    compatibility["deprecated_storage"] = True
    compatibility["migrated_to"] = str(destination.resolve())
    (legacy / "voice.json").write_text(
        json.dumps(compatibility, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _voice_dataset(raw: dict, layout, generation: dict,
                   voice: dict) -> tuple[str, Path, dict]:
    """解析由公开 dataset.voice.id 拥有的追加式共享音色数据集。 / Resolve the append-only shared dataset owned by public dataset.voice.id."""
    voice_id = str(voice.get("id") or "").strip()
    if not voice_id:
        raise ValueError(
            "dataset.voice.id is required and is the only key used for shared "
            "voice dataset storage"
        )
    if not VOICE_ID.fullmatch(voice_id):
        raise ValueError(
            "dataset.voice.id must contain only letters, numbers, '.', '_' and '-', "
            "and cannot start with punctuation"
        )
    identity = _voice_identity(raw, generation, voice)
    digest = _identity_digest(identity)
    root = Path(generation.get("voice_dataset_root") or layout.dataset_dir.parent / "voices")
    destination = root / voice_id
    destination.mkdir(parents=True, exist_ok=True)
    record = destination / "voice.json"

    # Versions before the voice-ID storage contract used
    # <voice_id>/<identity-prefix>/. Move the matching cache in place once so
    # users keep every already-generated WAV without retaining an internal ID.
    legacy = destination / digest[:12]
    legacy_record = legacy / "voice.json"
    migrated_legacy = False
    if not record.is_file() and legacy_record.is_file():
        previous = json.loads(legacy_record.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise RuntimeError(
                f"legacy voice cache identity mismatch for voice_id={voice_id!r}: {legacy}"
            )
        _move_tree(legacy / "references", destination / "references")
        _move_tree(legacy / "wavs", destination / "wavs")
        legacy_record.unlink()
        legacy.rmdir()
        _write_legacy_voice_alias(legacy, destination, previous)
        migrated_legacy = True
        logger.info(
            "VOICE CACHE MIGRATED | voice_id=%s | old=%s | new=%s",
            voice_id, legacy, destination,
            extra={"tts_style": "success"},
        )

    if record.is_file():
        # 身份锁：同一 voice_id 不允许换用不同的音色设置。 / Identity lock: one voice_id may never switch to different voice settings.
        existing = json.loads(record.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise ValueError(
                f"voice_id {voice_id!r} is already locked to different voice settings at "
                f"{record}; keep the original prompt/reference/Qwen settings or choose a "
                "new dataset.voice.id"
            )
    else:
        legacy_records = sorted(
            path for path in destination.glob("*/voice.json")
            if path.parent != legacy
        )
        if legacy_records and not migrated_legacy:
            raise ValueError(
                f"voice_id {voice_id!r} already has a legacy cache with different voice "
                f"settings at {legacy_records[0]}; keep those settings or choose a new "
                "dataset.voice.id"
            )
        unmanaged = [] if migrated_legacy else [
            path for path in (destination / "references", destination / "wavs")
            if path.exists()
        ]
        if unmanaged:
            raise RuntimeError(
                f"voice_id {voice_id!r} contains audio without a voice.json identity lock: "
                f"{destination}; move it aside or restore its voice.json before generating"
            )
        # 先写临时文件再原子替换，避免中途崩溃留下半个身份锁。 / Write to a temp file then atomically replace so a crash never leaves a half-written lock.
        temporary = record.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "format": 2,
            "voice_id": voice_id,
            "identity_sha256": digest,
            "identity": identity,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(record)
    _canonicalize_reference_names(destination / "references")
    return voice_id, destination, identity


def _sample_filename(item: GenerationText, candidate: int,
                     teacher_language: str) -> str:
    """由文本/语言/候选/教师语言的规范化 JSON 哈希出内容寻址文件名。 / Hash canonical JSON of text/lang/candidate/teacher-language into a content-addressed filename."""
    encoded = json.dumps({
        "language": item.language,
        "text": item.text,
        "candidate": candidate,
        "teacher_language": teacher_language,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24] + ".wav"


def read_generation_texts(path: str | Path, supported_languages=None) -> list[GenerationText]:
    """读取生成文本清单 CSV（text/language 两列）并做严格校验。 / Read the generation-text manifest CSV (text/language columns) with strict validation."""
    source = Path(path)
    supported = None if supported_languages is None else set(supported_languages)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = {"text", "language"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"generation text manifest missing columns: {', '.join(sorted(missing))}")
        result = []
        for line, row in enumerate(reader, start=2):
            text = row["text"].strip()
            language = row["language"].strip().lower()
            if not text:
                raise ValueError(f"generation text manifest line {line}: empty text")
            if supported is not None and language not in supported:
                raise ValueError(f"generation text manifest line {line}: unsupported language {language!r}")
            result.append(GenerationText(text, language))
    if not result:
        raise ValueError("generation text manifest contains no samples")
    return result


def _runtime_kwargs(config: dict, inherited_device: str = "auto") -> tuple[str, dict]:
    """推断教师模型运行设备/精度/注意力实现。 / Infer device, dtype, and attention implementation for the teacher runtime."""
    runtime = config.get("runtime", {})
    requested = runtime.get("device", "auto")
    # 顶层实验的设备选择可作为 auto 的兜底。 / The experiment-level device acts as the fallback for auto.
    if requested == "auto" and inherited_device != "auto":
        requested = inherited_device
    if requested == "auto":
        # auto 优先级：CUDA > MPS > CPU。 / auto priority: CUDA > MPS > CPU.
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested

    dtype_name = runtime.get("dtype", "auto")
    if dtype_name == "auto":
        # auto 精度：CUDA 上 bf16（不支持则 fp16），其余 fp32。 / auto dtype: bf16 on CUDA (fp16 fallback), fp32 elsewhere.
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else \
            torch.float16 if device.startswith("cuda") else torch.float32
    else:
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported Qwen dtype: {dtype_name}") from exc

    attention = runtime.get("attention", "auto")
    if attention == "auto":
        # 仅当 CUDA 且已安装 flash_attn 时才启用 FA2，否则回退 SDPA。 / Enable FA2 only on CUDA with flash_attn installed; otherwise fall back to SDPA.
        attention = "flash_attention_2" if device.startswith("cuda") and importlib.util.find_spec("flash_attn") else "sdpa"
    kwargs = {"device_map": device, "dtype": dtype}
    if attention not in {None, "default"}:
        kwargs["attn_implementation"] = attention
    return device, kwargs


def _release_device_memory(device: str) -> None:
    """回收 Python 对象并清空 CUDA 缓存，释放教师模型显存。 / Collect Python objects and flush the CUDA cache to free teacher-model memory."""
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_runtime_language_support(model, required: set[str], model_name: str) -> None:
    """核对运行时教师模型上报的语言覆盖必需集合。 / Cross-check the teacher's runtime-reported languages against the required set."""
    getter = getattr(model, "get_supported_languages", None)
    if not callable(getter):
        logger.info("teacher=%s runtime language query unavailable; using validated registry", model_name)
        return
    supported = {str(value) for value in getter()}
    logger.info("teacher=%s runtime languages=%s", model_name, ",".join(sorted(supported)))
    supported_folded = {value.casefold() for value in supported}
    missing = sorted(value for value in required if value.casefold() not in supported_folded)
    if missing:
        raise RuntimeError(f"teacher {model_name} does not report required languages: {', '.join(missing)}")


def _write_training_wav(path: Path, waveform, source_rate: int, target_rate: int) -> None:
    """写出 PCM_16 训练 WAV，必要时重采样到目标采样率。 / Write a PCM_16 training WAV, resampling to the target rate when needed."""
    samples = np.asarray(waveform, dtype=np.float32).squeeze()
    if samples.ndim != 1:
        raise ValueError(f"Qwen returned a non-mono waveform with shape {samples.shape}")
    if source_rate != target_rate:
        # 教师输出采样率与项目目标不一致时用 torchaudio 重采样。 / Resample with torchaudio when the teacher rate differs from the project target.
        tensor = torch.from_numpy(samples)
        samples = torchaudio.functional.resample(tensor, source_rate, target_rate).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, target_rate, subtype="PCM_16", format="WAV")


def _postprocess_training_wav(path: Path, config: dict) -> dict | None:
    """就地裁掉项目生成 WAV 两端过长的静音。 / Trim excessive edge silence from a project-generated WAV, in place."""
    if not config.get("enabled", True) or not config.get("trim_edge_silence", True):
        return None
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32).squeeze()
    if samples.ndim != 1:
        raise ValueError(f"generated WAV must be mono: {path}")
    # dBFS 阈值换算成线性幅值。 / Convert the dBFS threshold to a linear amplitude.
    threshold = 10.0 ** (float(config.get("silence_threshold_dbfs", -45.0)) / 20.0)
    active = np.flatnonzero(np.abs(samples) > threshold)
    if not active.size:
        return None
    padding = max(0, round(float(config.get("keep_edge_silence_seconds", 0.15)) * sample_rate))
    start = max(0, int(active[0]) - padding)
    stop = min(len(samples), int(active[-1]) + padding + 1)
    if start == 0 and stop == len(samples):
        return None
    trimmed = samples[start:stop]
    # 先写 .trim.tmp 再原子替换，避免半写状态。 / Write to .trim.tmp then atomically replace to avoid a half-written file.
    temporary = path.with_name(path.name + ".trim.tmp")
    sf.write(temporary, trimmed, sample_rate, subtype="PCM_16", format="WAV")
    temporary.replace(path)
    return {
        "audio": str(path),
        "before_seconds": len(samples) / sample_rate,
        "after_seconds": len(trimmed) / sample_rate,
        "removed_leading_seconds": start / sample_rate,
        "removed_trailing_seconds": (len(samples) - stop) / sample_rate,
    }


def _copy_reference(source: Path, destination: Path) -> Path:
    """把上传的参考音频复制进音色数据集（同路径则跳过）。 / Copy an uploaded reference into the voice dataset (no-op when paths already match)."""
    if not source.is_file():
        raise FileNotFoundError(f"reference audio does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _qwen_reference_input(value):
    """Qwen 只接受字符串路径，不接受 pathlib.Path。 / Qwen accepts string paths, not pathlib.Path objects."""
    return str(value) if isinstance(value, Path) else value


def _checkpoint_dataset_metadata(layout) -> Path | None:
    """找到 resume/expand 检查点对应的数据集 metadata。 / Find the raw dataset belonging to a resume/expand checkpoint."""
    checkpoint = layout.initialization_checkpoint
    if checkpoint is None:
        return None
    run_layout_path = checkpoint.parent.parent / "run-layout.json"
    if not run_layout_path.is_file():
        return None
    try:
        previous = json.loads(run_layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    dataset_dir = Path(str(previous.get("dataset_dir") or ""))
    raw_metadata = dataset_dir / "metadata.csv"
    if raw_metadata.is_file():
        return raw_metadata
    phoneme_metadata = Path(str(previous.get("metadata") or ""))
    return phoneme_metadata if phoneme_metadata.is_file() else None


def _read_voice_manifest(path: Path) -> list[dict]:
    """读取音色数据集的 speaker-free manifest.csv。 / Read the voice dataset's speaker-free manifest.csv."""
    if not path.is_file():
        raise FileNotFoundError(
            f"voice dataset manifest is missing: {path}; generate this voice first"
        )
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"audio", "text", "language"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"voice manifest missing columns: {', '.join(sorted(missing))}"
            )
        return [{
            "audio": (path.parent / row["audio"].strip()).resolve(),
            "text": row["text"].strip(),
            "language": row["language"].strip().lower(),
        } for row in reader]


def _migrate_voice_manifest(voice_root: Path, datasets_root: Path,
                            voice_id: str) -> Path | None:
    """从旧的模型级 metadata 重建新的 speaker-free 索引。 / Build the new speaker-free index from older model-local metadata."""
    rows = []
    seen = set()
    for record_path in datasets_root.glob("*/dataset.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("voice_id") != voice_id:
            continue
        metadata = Path(str(record.get("metadata") or ""))
        if not metadata.is_file():
            continue
        for item in read_manifest(metadata):
            try:
                item.audio.resolve().relative_to(voice_root.resolve())
            except ValueError:
                continue
            key = (str(item.audio.resolve()), item.text, item.language)
            # 以 (音频, 文本, 语言) 三元组去重合并旧记录。 / Dedupe legacy rows by the (audio, text, language) triple.
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "audio": item.audio.resolve(),
                "text": item.text,
                "language": item.language,
            })
    if not rows:
        return None
    manifest = voice_root / "manifest.csv"
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "audio": os.path.relpath(row["audio"], voice_root.resolve()),
                "text": row["text"],
                "language": row["language"],
            })
    temporary.replace(manifest)
    logger.info(
        "VOICE MANIFEST MIGRATED | voice_id=%s | samples=%d | path=%s",
        voice_id, len(rows), manifest,
        extra={"tts_style": "success"},
    )
    return manifest


def _sync_voice_manifest(voice_dataset: Path, jobs: list[GenerationJob]) -> Path:
    """把 speaker-free 追加式索引持久化到共享音色 WAV 旁。 / Persist a speaker-free append-only index beside the shared voice WAVs."""
    manifest = voice_dataset / "manifest.csv"
    rows = _read_voice_manifest(manifest) if manifest.is_file() else []
    seen = {
        (str(row["audio"]), row["text"], row["language"])
        for row in rows
    }
    for job in jobs:
        # 追加式合并：已有 (音频, 文本, 语言) 的任务不重复写入。 / Append-only merge: skip jobs whose (audio, text, language) already exists.
        key = (str(job.output.resolve()), job.item.text, job.item.language)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "audio": job.output.resolve(),
            "text": job.item.text,
            "language": job.item.language,
        })
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "audio": os.path.relpath(
                    Path(row["audio"]).resolve(), voice_dataset.resolve(),
                ),
                "text": row["text"],
                "language": row["language"],
            })
    temporary.replace(manifest)
    return manifest


def _speaker_assignments(generation: dict) -> dict[str, str]:
    """返回「模型 speaker 标签 -> 公开音色 ID」映射。 / Return model speaker label -> public voice ID."""
    value = generation.get("speaker_assignments") or {}
    if not isinstance(value, dict):
        raise ValueError("dataset.speakers must be an object mapping speaker names to voice IDs")
    result = {}
    for speaker, voice_id in value.items():
        speaker = str(speaker).strip()
        voice_id = str(voice_id).strip()
        if not speaker:
            raise ValueError("dataset.speakers contains an empty speaker name")
        if not VOICE_ID.fullmatch(voice_id):
            raise ValueError(f"dataset.speakers has invalid voice ID: {voice_id!r}")
        if voice_id in result.values():
            raise ValueError(
                f"voice ID {voice_id!r} is assigned more than once in dataset.speakers"
            )
        result[speaker] = voice_id
    return result


def _assigned_voice_rows(
    assignments: dict[str, str], layout, generation: dict,
    text_generation: dict,
) -> list[dict]:
    """挑选共享音色 WAV 并打上模型本地 speaker 标签。 / Select shared voice WAVs and assign model-local speaker labels."""
    root = Path(generation.get("voice_dataset_root") or layout.dataset_dir.parent / "voices")
    candidates = int(generation.get("candidates_per_text", 1))
    target = (
        int(text_generation.get("sentences_per_language", 100)) * candidates
        if "sentences_per_language" in text_generation else None
    )
    selected = []
    counts = Counter()
    for speaker, voice_id in assignments.items():
        manifest = root / voice_id / "manifest.csv"
        # manifest 缺失时先尝试从旧模型目录迁移生成。 / When the manifest is missing, try migrating it from older model-local storage first.
        if not manifest.is_file():
            _migrate_voice_manifest(root / voice_id, layout.dataset_dir.parent, voice_id)
        for row in _read_voice_manifest(manifest):
            language = row["language"]
            if language not in layout.languages:
                continue
            profile = (speaker, language)
            # 按 speaker×语言封顶，超出目标配额的缓存样本跳过。 / Cap per speaker×language; cached rows beyond the target quota are skipped.
            if target is not None and counts[profile] >= target:
                continue
            counts[profile] += 1
            selected.append({**row, "speaker": speaker})
        # 每个 speaker×语言都必须凑够目标份数，否则训练分布不均。 / Every speaker×language must reach its target quota, otherwise training data is unbalanced.
        missing = [
            language for language in layout.languages
            if counts[(speaker, language)] < (target or 1)
        ]
        if missing:
            raise ValueError(
                f"voice ID {voice_id!r} does not have enough cached data for speaker "
                f"{speaker!r}: {', '.join(missing)}; generate or extend that voice first"
            )
    logger.info(
        "MODEL SPEAKER ASSIGNMENTS | assignments=%s | samples=%d",
        assignments, len(selected),
        extra={"tts_style": "success"},
    )
    return selected


def _write_model_metadata(output: Path, rows: list[dict]) -> Path:
    """写出含 speaker 列的模型 metadata CSV。 / Write the model metadata CSV with the speaker column."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["audio", "text", "language", "speaker"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "audio": os.path.relpath(
                    Path(row["audio"]).resolve(), output.parent.resolve(),
                ),
                "text": row["text"],
                "language": row["language"],
                "speaker": row["speaker"],
            })
    temporary.replace(output)
    return output


def _checkpoint_speaker_rows(
    layout, generation: dict, text_generation: dict,
    replaced_speakers: set[str],
) -> list[dict]:
    """从检查点复用未被替换 speaker 的历史样本行。 / Reuse checkpoint rows for speakers that are not being replaced."""
    if layout.initialization_mode not in {
        "resume", "warm_start", "expand_speakers", "refine_text_prior",
    }:
        return []
    metadata = _checkpoint_dataset_metadata(layout)
    if metadata is None:
        return []
    candidates = int(generation.get("candidates_per_text", 1))
    target = (
        int(text_generation.get("sentences_per_language", 100)) * candidates
        if "sentences_per_language" in text_generation else None
    )
    rows = []
    counts = Counter()
    for item in read_manifest(metadata):
        if item.language not in layout.languages or item.speaker in replaced_speakers:
            continue
        profile = (item.speaker, item.language)
        if target is not None and counts[profile] >= target:
            continue
        counts[profile] += 1
        rows.append({
            "audio": item.audio.resolve(),
            "text": item.text,
            "language": item.language,
            "speaker": item.speaker,
        })
    if rows:
        logger.info(
            "CHECKPOINT SPEAKERS REUSED | source=%s | speakers=%s | samples=%d",
            metadata, ",".join(sorted({row["speaker"] for row in rows})), len(rows),
            extra={"tts_style": "success"},
        )
    return rows


def _generate_samples_single(
    config_path: str | Path, *,
    text_manifest_path: str | Path | None = None,
    model_loader: Callable = load_qwen_teacher,
) -> Path:
    """用官方 Qwen 教师运行时生成一个具名 VITS 数据集。 / Generate a named VITS dataset using the official Qwen teacher runtime.

    两种音色模式遵循 Qwen3-TTS 官方 README：
    - design：VoiceDesign 先造一条参考音频，再由 Base 克隆它生成全部样本。
    - clone：Base 用上传的参考音频 + 转写文本构造可复用的克隆 prompt。

    Voice modes follow the official Qwen3-TTS README:
    - design: VoiceDesign creates one reference, then Base clones it for all rows.
    - clone: Base creates a reusable prompt from uploaded reference audio + transcript.
    """
    raw, layout = resolve_experiment(config_path)
    configure_logging_from_config(raw)
    prepare_experiment(layout, raw, config_path)
    registry = resolve_language_registry(raw.get("language_registry"))
    generation = raw.get("generation", {})
    if not generation.get("enabled", True):
        raise ValueError("sample generation is disabled in this config")
    log_section(
        logger,
        "QWEN AUDIO DATASET",
        f"Model: {layout.name}\nLanguages: {', '.join(layout.languages)}",
    )

    text_generation = raw.get("text_generation", {})
    voice = generation.get("voice") or {}
    voice_id_hint = str(voice.get("id") or "").strip() or None
    output_metadata = Path(
        generation.get("raw_metadata") or layout.dataset_dir / "metadata.csv"
    )
    assignments = _speaker_assignments(generation)
    if not voice:
        # 纯组装路径：只做 speaker 指派与 metadata 拼装，不调用教师模型。 / Assembly-only path: speaker assignment and metadata merge without invoking the teacher.
        if not assignments:
            raise ValueError(
                "dataset must define voice for generation or speakers for model assembly"
            )
        rows = _checkpoint_speaker_rows(
            layout, generation, text_generation, set(assignments),
        )
        rows.extend(_assigned_voice_rows(
            assignments, layout, generation, text_generation,
        ))
        _write_model_metadata(output_metadata, rows)
        (layout.dataset_dir / "dataset.json").write_text(json.dumps({
            "format": 3,
            "model": layout.name,
            "metadata": str(output_metadata.resolve()),
            "speaker_assignments": assignments,
            "samples": len(rows),
            "audio_storage": "shared-voice-reference",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "METADATA BUILD DONE | samples=%d | speakers=%d | output=%s",
            len(rows), len(assignments), output_metadata,
            extra={"tts_style": "success"},
        )
        return output_metadata
    generated_default = None
    if text_generation.get("enabled", False):
        # 未显式给 manifest 时自动生成或复用文本语料。 / Auto-generate or reuse the text corpus when no manifest is supplied.
        if text_manifest_path is None and not generation.get("text_manifest"):
            logger.info(
                "TEXT AUTO PREPARE | source=config | action=generate_or_reuse",
                extra={"tts_style": "success"},
            )
            generated_default = generate_texts(config_path)
        else:
            generated_default = text_corpus_path(
                text_generation, layout, voice_id=voice_id_hint,
            )
    text_manifest = Path(
        text_manifest_path or generation.get("text_manifest")
        or generated_default or layout.dataset_dir / "texts.csv"
    )
    all_texts = read_generation_texts(text_manifest, registry)
    texts = [item for item in all_texts if item.language in layout.languages]
    if text_generation.get("enabled", False):
        # 按语言截断到 sentences_per_language，保持各语言配额一致。 / Cap per language at sentences_per_language to keep quotas balanced.
        target = int(text_generation.get("sentences_per_language", 100))
        selected_counts = {}
        selected_texts = []
        for item in texts:
            count = selected_counts.get(item.language, 0)
            if count >= target:
                continue
            selected_texts.append(item)
            selected_counts[item.language] = count + 1
        texts = selected_texts
    missing_text_languages = sorted(set(layout.languages) - {item.language for item in texts})
    if missing_text_languages:
        raise ValueError(
            "generation text manifest has no rows for configured languages: "
            + ", ".join(missing_text_languages)
        )
    logger.info("text manifest=%s selected=%d", text_manifest, len(texts))
    teacher_languages = {}
    # 每个实验语言必须能映射到 Qwen 教师语言标签。 / Every experiment language must map to a Qwen teacher language tag.
    for language, spec in layout.language_specs.items():
        if spec.teacher_provider != "qwen" or not spec.teacher_language:
            raise ValueError(
                f"language {language} has no Qwen teacher mapping; disable generation "
                "and supply your own metadata, or configure a supported teacher"
            )
        teacher_languages[language] = spec.teacher_language
    mode = voice.get("mode")
    if mode not in {"design", "clone"}:
        raise ValueError("dataset.voice.mode must be design or clone")

    candidates = int(generation.get("candidates_per_text", 1))
    if candidates < 1:
        raise ValueError("generation.candidates_per_text must be at least 1")
    voice_id, voice_dataset, _ = _voice_dataset(
        raw, layout, generation, voice,
    )
    regeneration = _regeneration_plan(
        voice, generation, layout.languages,
    )
    _invalidate_language_references(
        voice_dataset / "references",
        voice,
        regeneration.reference_languages,
    )
    assigned_label = next(
        (label for label, assigned_voice in assignments.items()
         if assigned_voice == voice_id),
        None,
    )
    # Legacy voice.speaker remains accepted, but public configs assign speaker
    # labels at model assembly time through dataset.speakers.
    speaker = str(voice.get("speaker") or assigned_label or voice_id).strip()
    logger.info(
        "VOICE DATASET | voice_id=%s | path=%s",
        voice_id, voice_dataset,
        extra={"tts_style": "success"},
    )
    wav_root = voice_dataset / "wavs"
    legacy_wav_root = layout.dataset_dir / "wavs" / speaker
    jobs = []
    job_outputs = set()
    migrated = 0
    for index, item in enumerate(texts, start=1):
        for candidate in range(1, candidates + 1):
            output = wav_root / item.language / _sample_filename(
                item, candidate, teacher_languages[item.language],
            )
            if output in job_outputs:
                continue
            job_outputs.add(output)
            legacy = legacy_wav_root / f"{item.language}_{index:06d}_c{candidate:02d}.wav"
            regenerate_audio = (
                item.language in regeneration.audio_languages
            )
            if (
                not output.is_file()
                and legacy.is_file()
                and not regenerate_audio
            ):
                # 旧模型级命名缓存可直接搬进音色内容寻址布局。 / Legacy model-named caches can be copied straight into the content-addressed layout.
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy, output)
                migrated += 1
            jobs.append(GenerationJob(item, candidate, output))
    if migrated:
        logger.info(
            "legacy model audio migrated voice_id=%s files=%d source=%s destination=%s",
            voice_id, migrated, legacy_wav_root, wav_root,
        )
    if regeneration.audio_languages:
        selected_files = sum(
            job.item.language in regeneration.audio_languages
            for job in jobs
        )
        logger.warning(
            "AUDIO REGENERATION ENABLED | voice_id=%s | languages=%s | "
            "selected_files=%d | references=%s",
            voice_id, ",".join(sorted(regeneration.audio_languages)),
            selected_files,
            "regenerate" if regeneration.reference_languages else "preserve",
        )
    pending = [
        job for job in jobs
        if (
            job.item.language in regeneration.audio_languages
            or not job.output.is_file()
        )
    ]
    # 缓存命中即跳过，只重生成缺失或显式失效的任务。 / Cache hits are skipped; only missing or explicitly invalidated jobs are regenerated.
    cached_count = len(jobs) - len(pending)
    logger.info(
        "AUDIO PLAN | total=%d | pending=%d | cached=%d | output=%s",
        len(jobs), len(pending), cached_count, wav_root,
    )

    if pending:
        device, load_kwargs = _runtime_kwargs(generation, layout.device)
        common = {
            "download_if_missing": bool(generation.get("auto_download_models", True)),
            "runtime_mode": generation.get("qwen_runtime", "installed"),
            "source_path": generation.get("qwen_source_path"),
            **load_kwargs,
        }
        model_keys = generation.get("models", {})
        references = voice_dataset / "references"
        reference_text = voice.get("reference_text", "").strip()
        x_vector_only = bool(voice.get("x_vector_only_mode", False))
        reference_inputs = {}
        prompt_texts = {}

        if mode == "design":
            prompt = voice.get("prompt", "").strip()
            reference_strategy = str(
                voice.get("reference_strategy", "shared")
            ).strip().lower()
            if reference_strategy not in {
                "shared", "per_language", "cascade",
            }:
                raise ValueError(
                    "design reference_strategy must be shared, per_language, "
                    "or cascade"
                )
            reference_language = voice.get("reference_language", "en").lower()
            cascade_master_input = None
            cascade_master_text = None
            if not prompt:
                raise ValueError("design mode requires dataset.voice.prompt")
            if reference_strategy in {"shared", "cascade"}:
                if not reference_text:
                    raise ValueError(
                        f"{reference_strategy} design mode requires "
                        "dataset.voice.reference_text"
                    )
                if reference_language not in registry:
                    raise ValueError(
                        f"unsupported design reference language: {reference_language}"
                    )
                reference_spec = registry[reference_language]
                if reference_spec.teacher_provider != "qwen" \
                        or not reference_spec.teacher_language:
                    raise ValueError(
                        f"reference language {reference_language} has no Qwen teacher mapping"
                    )
                reference_audio = references / "designed.wav"
                legacy_reference = (
                    layout.dataset_dir / "references" / f"{speaker}.designed.wav"
                )
                if not reference_audio.is_file() and legacy_reference.is_file():
                    # 迁移旧模型目录里的 designed 参考而不是重新设计。 / Migrate the legacy designed reference instead of re-designing it.
                    reference_audio.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(legacy_reference, reference_audio)
                    logger.info(
                        "legacy designed reference migrated source=%s destination=%s",
                        legacy_reference, reference_audio,
                    )
                if not reference_audio.is_file():
                    # 参考音频缺失才加载 VoiceDesign 模型，用完立即释放。 / Load VoiceDesign only when the reference is missing; release it right after use.
                    design_model = model_loader(
                        model_keys.get("voice_design", "voice-design-1.7b"),
                        **common,
                    )
                    _log_runtime_language_support(
                        design_model, {reference_spec.teacher_language},
                        model_keys.get("voice_design", "voice-design-1.7b"),
                    )
                    logger.info(
                        "creating shared designed reference voice_id=%s language=%s "
                        "teacher_language=%s",
                        voice_id, reference_language,
                        reference_spec.teacher_language,
                    )
                    ref_wavs, ref_rate = design_model.generate_voice_design(
                        text=reference_text,
                        language=reference_spec.teacher_language,
                        instruct=prompt,
                        **generation.get("generation_kwargs", {}),
                    )
                    reference_audio.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        reference_audio, np.asarray(ref_wavs[0]).squeeze(),
                        ref_rate, subtype="PCM_16", format="WAV",
                    )
                    del design_model
                    _release_device_memory(device)
                if reference_strategy == "shared":
                    # shared：单条参考通配全部语言（键 "*"）。 / shared: one reference serves all languages via the "*" key.
                    reference_inputs["*"] = reference_audio
                    prompt_texts["*"] = reference_text
                else:
                    # cascade：此参考作为主锚点，稍后派生各语言版本。 / cascade: this reference is the master anchor for per-language derivatives.
                    cascade_master_input = reference_audio
                    cascade_master_text = reference_text
            if reference_strategy == "per_language":
                configured_reference_texts = voice.get("reference_texts") or {}
                if not isinstance(configured_reference_texts, dict):
                    raise ValueError(
                        "dataset.voice.reference_texts must be an object keyed by language"
                    )
                # 只为仍有待生成任务的语言制作参考，节省模型调用。 / Only build references for languages that still have pending jobs.
                target_languages = [
                    language for language in layout.languages
                    if any(job.item.language == language for job in pending)
                ]
                design_model = None
                for language in target_languages:
                    reference_spec = registry[language]
                    requested_reference_text = str(
                        configured_reference_texts.get(language) or next(
                            job.item.text for job in jobs
                            if job.item.language == language
                        )
                    ).strip()
                    reference_audio = references / f"designed-{language}.wav"
                    reference_transcript = (
                        references / f"designed-{language}.txt"
                    )
                    if reference_audio.is_file():
                        # 缓存命中必须带转写文本，否则音画不一致无法恢复。 / A cache hit must carry its transcript, otherwise text-audio pairing is unrecoverable.
                        if not reference_transcript.is_file():
                            raise RuntimeError(
                                "language-specific reference audio is missing its "
                                f"transcript: {reference_transcript}; remove "
                                f"{reference_audio} and rerun"
                            )
                        language_text = reference_transcript.read_text(
                            encoding="utf-8",
                        ).strip()
                        reference_inputs[language] = reference_audio
                    else:
                        language_text = requested_reference_text
                        # 惰性加载设计模型：首个缺失参考出现时才加载。 / Lazy-load the design model on the first missing reference.
                        if design_model is None:
                            design_model = model_loader(
                                model_keys.get(
                                    "voice_design", "voice-design-1.7b",
                                ),
                                **common,
                            )
                            _log_runtime_language_support(
                                design_model,
                                {
                                    teacher_languages[value]
                                    for value in target_languages
                                },
                                model_keys.get(
                                    "voice_design", "voice-design-1.7b",
                                ),
                            )
                        logger.info(
                            "creating language-specific designed reference "
                            "voice_id=%s language=%s teacher_language=%s",
                            voice_id, language,
                            reference_spec.teacher_language,
                        )
                        ref_wavs, ref_rate = design_model.generate_voice_design(
                            text=language_text,
                            language=reference_spec.teacher_language,
                            instruct=prompt,
                            **generation.get("generation_kwargs", {}),
                        )
                        reference_audio.parent.mkdir(
                            parents=True, exist_ok=True,
                        )
                        sf.write(
                            reference_audio,
                            np.asarray(ref_wavs[0]).squeeze(),
                            ref_rate, subtype="PCM_16", format="WAV",
                        )
                        reference_transcript.write_text(
                            language_text, encoding="utf-8",
                        )
                        # 直接复用刚生成的波形元组，避免再读一次磁盘。 / Reuse the freshly generated waveform tuple instead of re-reading disk.
                        reference_inputs[language] = (ref_wavs[0], ref_rate)
                    prompt_texts[language] = language_text
                if design_model is not None:
                    del design_model
                    _release_device_memory(device)
        else:
            # clone 模式：上传参考音频 + 精确转写构成克隆 prompt。 / clone mode: uploaded reference audio + exact transcript form the clone prompt.
            reference_value = voice.get("reference_audio")
            if not reference_value:
                raise ValueError("clone mode requires dataset.voice.reference_audio")
            if not reference_text and not x_vector_only:
                raise ValueError("clone mode requires the exact reference_text unless x_vector_only_mode is true")
            uploaded = Path(reference_value).expanduser()
            reference_inputs["*"] = _copy_reference(
                uploaded, references / f"uploaded{uploaded.suffix or '.wav'}",
            )
            prompt_texts["*"] = reference_text

        clone_model = model_loader(model_keys.get("voice_clone", "base-1.7b"), **common)
        _log_runtime_language_support(
            clone_model, set(teacher_languages.values()),
            model_keys.get("voice_clone", "base-1.7b"),
        )
        if mode == "design" and reference_strategy == "cascade":
            configured_reference_texts = voice.get("reference_texts") or {}
            if not isinstance(configured_reference_texts, dict):
                raise ValueError(
                    "dataset.voice.reference_texts must be an object keyed by language"
                )
            target_languages = [
                language for language in layout.languages
                if any(job.item.language == language for job in pending)
            ]
            # 主语言克隆 prompt 惰性创建一次，供其余语言参考生成复用。 / The master clone prompt is lazily built once and reused for all target languages.
            master_prompt = None
            logger.info(
                "CASCADE REFERENCES | voice_id=%s | master_language=%s | "
                "target_languages=%s",
                voice_id, reference_language, ",".join(target_languages),
            )
            for language in target_languages:
                requested_reference_text = str(
                    configured_reference_texts.get(language) or (
                        cascade_master_text
                        if language == reference_language
                        else next(
                            job.item.text for job in jobs
                            if job.item.language == language
                        )
                    )
                ).strip()
                localized_audio = references / f"localized-{language}.wav"
                localized_transcript = references / f"localized-{language}.txt"
                if localized_audio.is_file():
                    # 缓存的本地化参考必须能还原其转写文本。 / A cached localized reference must be able to restore its transcript.
                    if not localized_transcript.is_file():
                        raise RuntimeError(
                            "cascade reference audio is missing its transcript: "
                            f"{localized_transcript}; remove {localized_audio} "
                            "and rerun"
                        )
                    localized_text = localized_transcript.read_text(
                        encoding="utf-8",
                    ).strip()
                    logger.info(
                        "CASCADE REFERENCE CACHE HIT | voice_id=%s | language=%s "
                        "| audio=%s",
                        voice_id, language, localized_audio,
                    )
                elif (
                    language == reference_language
                    and requested_reference_text == cascade_master_text
                ):
                    # 主语言且文本一致：直接复制主参考，无需再合成。 / Master language with identical text: copy the master reference instead of synthesizing.
                    localized_audio.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cascade_master_input, localized_audio)
                    localized_transcript.write_text(
                        cascade_master_text, encoding="utf-8",
                    )
                    localized_text = cascade_master_text
                    logger.info(
                        "CASCADE REFERENCE MASTER REUSED | voice_id=%s | "
                        "language=%s | audio=%s",
                        voice_id, language, localized_audio,
                    )
                else:
                    if master_prompt is None:
                        logger.info(
                            "CASCADE MASTER PROMPT | voice_id=%s | "
                            "language=%s | status=creating",
                            voice_id, reference_language,
                        )
                        master_prompt = clone_model.create_voice_clone_prompt(
                            ref_audio=_qwen_reference_input(
                                cascade_master_input,
                            ),
                            ref_text=cascade_master_text,
                            x_vector_only_mode=x_vector_only,
                        )
                    teacher_language = teacher_languages[language]
                    logger.info(
                        "CASCADE REFERENCE GENERATION | voice_id=%s | "
                        "language=%s | teacher_language=%s",
                        voice_id, language, teacher_language,
                    )
                    localized_wavs, localized_rate = (
                        clone_model.generate_voice_clone(
                            text=[requested_reference_text],
                            language=[teacher_language],
                            voice_clone_prompt=master_prompt,
                            **generation.get("generation_kwargs", {}),
                        )
                    )
                    if len(localized_wavs) != 1:
                        raise RuntimeError(
                            "Qwen returned an unexpected number of cascade "
                            f"reference waveforms: {len(localized_wavs)}"
                        )
                    localized_audio.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        localized_audio,
                        np.asarray(localized_wavs[0]).squeeze(),
                        localized_rate, subtype="PCM_16", format="WAV",
                    )
                    localized_transcript.write_text(
                        requested_reference_text, encoding="utf-8",
                    )
                    localized_text = requested_reference_text
                reference_inputs[language] = localized_audio
                prompt_texts[language] = localized_text
            logger.info(
                "CASCADE REFERENCES READY | voice_id=%s | references=%d | "
                "directory=%s",
                voice_id, len(reference_inputs), references,
                extra={"tts_style": "success"},
            )
        clone_prompts = {}
        # 每条参考只构建一次可复用的克隆 prompt（Qwen 的耗时步骤）。 / Build one reusable clone prompt per reference — the expensive step in Qwen.
        for language_key, reference_input in reference_inputs.items():
            logger.info(
                "creating reusable clone prompt voice_id=%s mode=%s language=%s",
                voice_id, mode,
                "shared" if language_key == "*" else language_key,
            )
            clone_prompts[language_key] = clone_model.create_voice_clone_prompt(
                ref_audio=_qwen_reference_input(reference_input),
                ref_text=prompt_texts[language_key] or None,
                x_vector_only_mode=x_vector_only,
            )
        batch_size = int(generation.get("batch_size", 4))
        if batch_size < 1:
            raise ValueError("generation.batch_size must be at least 1")
        generation_kwargs = generation.get("generation_kwargs", {})
        target_rate = int(raw["audio"]["sample_rate"])
        progress_interval = max(
            1, int(raw.get("logging", {}).get("sample_progress_every_batches", 1)),
        )
        slow_batch_seconds = float(
            raw.get("logging", {}).get("sample_slow_batch_seconds", 60),
        )
        generation_started = time.monotonic()
        pending_by_language = {
            language: [
                job for job in pending if job.item.language == language
            ]
            for language in layout.languages
        }
        # 按语言分组后切批：同批共享一条教师语言标签与克隆 prompt。 / Group by language then slice into batches: each batch shares one teacher tag and clone prompt.
        batches = [
            (language, language_jobs[start:start + batch_size])
            for language, language_jobs in pending_by_language.items()
            for start in range(0, len(language_jobs), batch_size)
        ]
        total_batches = len(batches)
        completed_new = 0
        for batch_number, (language, batch) in enumerate(batches, 1):
            batch_started = time.monotonic()
            teacher_language = teacher_languages[language]
            logger.info(
                "AUDIO BATCH START | batch=%d/%d | language=%s | "
                "teacher_language=%s | items=%d",
                batch_number, total_batches, language, teacher_language,
                len(batch),
            )
            with _BatchHeartbeat(
                batch_number=batch_number,
                language=language,
                interval=slow_batch_seconds,
            ):
                wavs, sample_rate = clone_model.generate_voice_clone(
                    text=[job.item.text for job in batch],
                    language=[teacher_language] * len(batch),
                    # 优先用语言专属 prompt，shared 场景回退到 "*"。 / Prefer the language-specific prompt, falling back to "*" in shared mode.
                    voice_clone_prompt=clone_prompts.get(
                        language, clone_prompts.get("*"),
                    ),
                    **generation_kwargs,
                )
            if len(wavs) != len(batch):
                raise RuntimeError(f"Qwen returned {len(wavs)} waveforms for a batch of {len(batch)}")
            for job, waveform in zip(batch, wavs):
                _write_training_wav(job.output, waveform, sample_rate, target_rate)
            completed_new += len(batch)
            if batch_number % progress_interval == 0 or completed_new == len(pending):
                elapsed = time.monotonic() - generation_started
                rate = completed_new / max(elapsed, 1e-9)
                remaining = len(pending) - completed_new
                overall_completed = cached_count + completed_new
                percent = 100.0 * overall_completed / max(len(jobs), 1)
                logger.info(
                    "AUDIO %6.2f%% | completed=%d/%d | new=%d/%d | cached=%d | "
                    "batch=%d/%d (%s) | batch_time=%s | speed=%.1f/min | ETA=%s",
                    percent, overall_completed, len(jobs), completed_new, len(pending),
                    cached_count, batch_number, total_batches, language,
                    format_duration(time.monotonic() - batch_started), rate * 60,
                    format_duration(remaining / rate),
                    extra={"tts_style": "progress"},
                )
        logger.info(
            "AUDIO GENERATION DONE | generated=%d | cached=%d | total=%d | elapsed=%s",
            len(pending), cached_count, len(jobs),
            format_duration(time.monotonic() - generation_started),
            extra={"tts_style": "success"},
        )
        logger.info("AUDIO MODEL RELEASE | status=started | device=%s", device)
        # 生成完毕立即释放教师模型，把显存留给后续训练。 / Release the teacher right after generation so training gets the memory back.
        del clone_model
        _release_device_memory(device)
        logger.info(
            "AUDIO MODEL RELEASE | status=completed | device=%s",
            device,
            extra={"tts_style": "success"},
        )

    postprocess_config = generation.get("audio_postprocess", {})
    postprocess_enabled = bool(
        postprocess_config.get("enabled", True)
        and postprocess_config.get("trim_edge_silence", True)
    )
    trimmed = []
    if postprocess_enabled:
        postprocess_started = time.monotonic()
        postprocess_interval = max(
            1, int(raw.get("logging", {}).get("sample_postprocess_every_files", 200)),
        )
        logger.info(
            "AUDIO POSTPROCESS START | total=%d | progress_every_files=%d | action=trim_edge_silence",
            len(jobs), postprocess_interval,
        )
        for index, job in enumerate(jobs, 1):
            result = _postprocess_training_wav(job.output, postprocess_config)
            if result is not None:
                trimmed.append(result)
            if index % postprocess_interval == 0 or index == len(jobs):
                elapsed = time.monotonic() - postprocess_started
                rate = index / max(elapsed, 1e-9)
                remaining = len(jobs) - index
                logger.info(
                    "AUDIO POSTPROCESS %6.2f%% | checked=%d/%d | trimmed=%d | speed=%.1f/s | ETA=%s",
                    100.0 * index / max(len(jobs), 1), index, len(jobs),
                    len(trimmed), rate, format_duration(remaining / rate),
                    extra={"tts_style": "progress"},
                )
        logger.info(
            "AUDIO POSTPROCESS DONE | checked=%d | trimmed=%d | elapsed=%s",
            len(jobs), len(trimmed),
            format_duration(time.monotonic() - postprocess_started),
            extra={"tts_style": "success"},
        )
    else:
        logger.info(
            "AUDIO POSTPROCESS SKIPPED | total=%d | reason=disabled",
            len(jobs),
        )
    if postprocess_enabled:
        report_path = voice_dataset / "audio-postprocess-report.json"
        report_path.write_text(json.dumps({
            "format": 1,
            "provider": "edge-silence-trim-v1",
            "checked": len(jobs),
            "trimmed": len(trimmed),
            "settings": postprocess_config,
            "results": trimmed,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    voice_manifest = _sync_voice_manifest(voice_dataset, jobs)
    logger.info(
        "VOICE MANIFEST | voice_id=%s | path=%s | speaker_free=true",
        voice_id, voice_manifest,
        extra={"tts_style": "success"},
    )
    if raw.get("task", "train") == "prepare":
        # prepare 任务到音色清单即止，不组装模型 metadata。 / prepare tasks stop at the voice manifest without assembling model metadata.
        logger.info(
            "VOICE PREPARE DONE | voice_id=%s | samples=%d | manifest=%s | "
            "model_metadata=skipped",
            voice_id, len(jobs), voice_manifest,
            extra={"tts_style": "success"},
        )
        return voice_manifest
    included_manifests = [
        (Path(value), False) for value in generation.get("include_metadata", [])
    ]
    previous_metadata = (
        _checkpoint_dataset_metadata(layout)
        if layout.initialization_mode in {
            "resume", "expand_speakers", "refine_text_prior",
        } else None
    )
    if previous_metadata is not None and all(
        path.resolve() != previous_metadata.resolve()
        for path, _ in included_manifests
    ):
        # 检查点 metadata 自动并入（标记 automatic，稍后按配额截断）。 / Auto-include checkpoint metadata (flagged automatic for later quota capping).
        included_manifests.append((previous_metadata, True))
        logger.info(
            "METADATA AUTO REUSE | mode=%s | source=%s | current_speaker=%s",
            layout.initialization_mode, previous_metadata, speaker,
            extra={"tts_style": "success"},
        )
    logger.info(
        "METADATA BUILD START | generated_jobs=%d | included_manifests=%d | output=%s",
        len(jobs), len(included_manifests), output_metadata,
    )
    rows = []
    seen = set()
    included_counts = Counter()
    skipped_languages = Counter()
    target_per_profile = (
        int(text_generation.get("sentences_per_language", 100)) * candidates
        if text_generation.get("enabled", False) else None
    )
    for included_path, automatic in included_manifests:
        for item in read_manifest(included_path):
            if item.language not in layout.languages:
                skipped_languages[item.language] += 1
                continue
            # The currently configured voice is rebuilt from its append-only
            # cache. Automatic checkpoint reuse only carries the other voices.
            if automatic and item.speaker == speaker:
                continue
            profile = (item.speaker, item.language)
            if automatic and target_per_profile is not None \
                    and included_counts[profile] >= target_per_profile:
                continue
            key = (str(item.audio), item.text, item.language, item.speaker)
            # 四元组去重：同一音频绝不重复进入 metadata。 / Dedupe by the 4-tuple so one audio file never enters metadata twice.
            if key in seen:
                continue
            seen.add(key)
            included_counts[profile] += 1
            rows.append({
                "audio": os.path.relpath(item.audio.resolve(), output_metadata.parent.resolve()),
                "text": item.text,
                "language": item.language,
                "speaker": item.speaker,
            })
    if skipped_languages:
        logger.info(
            "METADATA FILTER | skipped_disabled_languages=%s",
            dict(sorted(skipped_languages.items())),
        )
    # 当前行优先：有 speaker 指派时走共享缓存选取，否则直接用本音色任务。 / Current voice first: pick from the shared cache when assignments exist, else use this run's jobs.
    current_rows = (
        _assigned_voice_rows(assignments, layout, generation, text_generation)
        if assignments else [{
            "audio": job.output.resolve(),
            "text": job.item.text,
            "language": job.item.language,
            "speaker": speaker,
        } for job in jobs]
    )
    for item in current_rows:
        key = (
            str(item["audio"]), item["text"], item["language"], item["speaker"],
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "audio": os.path.relpath(
                Path(item["audio"]).resolve(), output_metadata.parent.resolve(),
            ),
            "text": item["text"],
            "language": item["language"],
            "speaker": item["speaker"],
        })
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_metadata.with_suffix(output_metadata.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_metadata)
    (layout.dataset_dir / "dataset.json").write_text(json.dumps({
        "format": 2,
        "model": layout.name,
        "metadata": str(output_metadata.resolve()),
        "text_manifest": str(text_manifest.resolve()),
        "voice_id": voice_id,
        "voice_dataset": str(voice_dataset.resolve()),
        "speaker_label": speaker,
        "speaker_assignments": assignments,
        "samples": len(rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "METADATA BUILD DONE | samples=%d | output=%s",
        len(rows), output_metadata,
        extra={"tts_style": "success"},
    )
    return output_metadata


def generate_samples(config_path: str | Path, *, text_manifest_path: str | Path | None = None,
                     model_loader: Callable = load_qwen_teacher) -> Path:
    """先逐个准备声明的公开音色，再按需组装模型 speaker。 / Prepare every declared public voice, then assemble model speakers if requested."""
    raw, layout = resolve_experiment(config_path)
    voices = raw.get("generation", {}).get("voices") or {}
    if not voices:
        # 单音色/无 voices 配置时直接走单次生成路径。 / With no voices map, go straight to the single-run generation path.
        return _generate_samples_single(
            config_path, text_manifest_path=text_manifest_path,
            model_loader=model_loader,
        )

    prepare_experiment(layout, raw, config_path)
    job_root = layout.run_dir / "voice-jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "MULTI VOICE PLAN | task=%s | voices=%d | ids=%s",
        raw.get("task", "train"), len(voices), ",".join(voices),
        extra={"tts_style": "success"},
    )
    for index, (voice_id, voice) in enumerate(voices.items(), 1):
        # 每个音色拆成独立 prepare 子任务配置，逐个生成或复用。 / Split each voice into its own prepare sub-config and generate/reuse in turn.
        job = deepcopy(raw)
        job.pop("dataset", None)
        job["task"] = "prepare"
        job_generation = job.setdefault("generation", {})
        job_generation.pop("voices", None)
        job_generation["voice"] = dict(voice)
        job_generation["speaker_assignments"] = {}
        job_generation["raw_metadata"] = str(
            job_root / f"{index:02d}-{voice_id}.metadata.csv"
        )
        job_path = job_root / f"{index:02d}-{voice_id}.json"
        job_path.write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.info(
            "MULTI VOICE %d/%d | voice_id=%s | action=generate_or_reuse",
            index, len(voices), voice_id,
        )
        _generate_samples_single(
            job_path, text_manifest_path=text_manifest_path,
            model_loader=model_loader,
        )

    if raw.get("task", "train") == "prepare":
        # prepare 到各音色就绪为止，写汇总文件而不组装模型。 / prepare ends once every voice is ready; write the summary without model assembly.
        prepare_experiment(layout, raw, config_path)
        summary = layout.run_dir / "prepared-voices.json"
        summary.write_text(json.dumps({
            "format": 1,
            "task": "prepare",
            "voices": {
                voice_id: {
                    "directory": str(
                        (layout.dataset_dir.parent / "voices" / voice_id).resolve()
                    ),
                    "manifest": str(
                        (
                            layout.dataset_dir.parent / "voices"
                            / voice_id / "manifest.csv"
                        ).resolve()
                    ),
                }
                for voice_id in voices
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "MULTI VOICE PREPARE DONE | voices=%d | summary=%s | output_root=%s",
            len(voices), summary, layout.dataset_dir.parent / "voices",
            extra={"tts_style": "success"},
        )
        return summary

    # 组装阶段：去掉 voice 定义、关掉文本生成，仅做 metadata 合并。 / Assembly stage: drop voice defs, disable text generation, merge metadata only.
    assembly = deepcopy(raw)
    assembly.pop("dataset", None)
    assembly_generation = assembly.setdefault("generation", {})
    assembly_generation.pop("voices", None)
    assembly_generation.pop("voice", None)
    assembly.setdefault("text_generation", {})["enabled"] = False
    assembly_path = job_root / "assemble-model.json"
    assembly_path.write_text(
        json.dumps(assembly, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    result = _generate_samples_single(
        assembly_path, model_loader=model_loader,
    )
    prepare_experiment(layout, raw, config_path)
    return result
