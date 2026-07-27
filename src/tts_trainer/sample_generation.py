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
VOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class GenerationText:
    text: str
    language: str


@dataclass(frozen=True)
class GenerationJob:
    item: GenerationText
    candidate: int
    output: Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _voice_identity(raw: dict, generation: dict, voice: dict) -> dict:
    """Return the immutable settings owned by one public voice ID."""
    mode = str(voice.get("mode") or "")
    reference_identity = None
    if mode == "clone" and voice.get("reference_audio"):
        reference_path = Path(voice["reference_audio"]).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"reference audio does not exist: {reference_path}")
        reference_identity = {
            "sha256": _file_sha256(reference_path),
            "suffix": reference_path.suffix.lower(),
        }
    return {
        "format": 1,
        "mode": mode,
        "prompt": str(voice.get("prompt") or "").strip() or None,
        "reference_text": str(voice.get("reference_text") or "").strip() or None,
        "reference_language": (
            str(voice.get("reference_language", "en")).strip().lower()
            if mode == "design" else None
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


def _identity_digest(identity: dict) -> str:
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _move_tree(source: Path, destination: Path) -> None:
    """Move a legacy tree without replacing a different cached file."""
    if not source.is_dir():
        return
    if not destination.exists():
        source.replace(destination)
        return
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
    """Remove the model-internal speaker label from shared reference paths."""
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
    """Keep old metadata paths working after moving WAVs to the voice-ID root."""
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
    """Resolve the append-only shared dataset owned by public dataset.voice.id."""
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
    encoded = json.dumps({
        "language": item.language,
        "text": item.text,
        "candidate": candidate,
        "teacher_language": teacher_language,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24] + ".wav"


def read_generation_texts(path: str | Path, supported_languages=None) -> list[GenerationText]:
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
    runtime = config.get("runtime", {})
    requested = runtime.get("device", "auto")
    if requested == "auto" and inherited_device != "auto":
        requested = inherited_device
    if requested == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = requested

    dtype_name = runtime.get("dtype", "auto")
    if dtype_name == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else \
            torch.float16 if device.startswith("cuda") else torch.float32
    else:
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported Qwen dtype: {dtype_name}") from exc

    attention = runtime.get("attention", "auto")
    if attention == "auto":
        attention = "flash_attention_2" if device.startswith("cuda") and importlib.util.find_spec("flash_attn") else "sdpa"
    kwargs = {"device_map": device, "dtype": dtype}
    if attention not in {None, "default"}:
        kwargs["attn_implementation"] = attention
    return device, kwargs


def _release_device_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_runtime_language_support(model, required: set[str], model_name: str) -> None:
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
    samples = np.asarray(waveform, dtype=np.float32).squeeze()
    if samples.ndim != 1:
        raise ValueError(f"Qwen returned a non-mono waveform with shape {samples.shape}")
    if source_rate != target_rate:
        tensor = torch.from_numpy(samples)
        samples = torchaudio.functional.resample(tensor, source_rate, target_rate).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, target_rate, subtype="PCM_16", format="WAV")


def _postprocess_training_wav(path: Path, config: dict) -> dict | None:
    """Trim excessive edge silence from a project-generated WAV, in place."""
    if not config.get("enabled", True) or not config.get("trim_edge_silence", True):
        return None
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32).squeeze()
    if samples.ndim != 1:
        raise ValueError(f"generated WAV must be mono: {path}")
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
    if not source.is_file():
        raise FileNotFoundError(f"reference audio does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _qwen_reference_input(value):
    """Qwen accepts string paths, not pathlib.Path objects."""
    return str(value) if isinstance(value, Path) else value


def _checkpoint_dataset_metadata(layout) -> Path | None:
    """Find the raw dataset belonging to a resume/expand checkpoint."""
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
    """Build the new speaker-free index from older model-local metadata."""
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
    """Persist a speaker-free append-only index beside the shared voice WAVs."""
    manifest = voice_dataset / "manifest.csv"
    rows = _read_voice_manifest(manifest) if manifest.is_file() else []
    seen = {
        (str(row["audio"]), row["text"], row["language"])
        for row in rows
    }
    for job in jobs:
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
    """Return model speaker label -> public voice ID."""
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
    """Select shared voice WAVs and assign model-local speaker labels."""
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
        if not manifest.is_file():
            _migrate_voice_manifest(root / voice_id, layout.dataset_dir.parent, voice_id)
        for row in _read_voice_manifest(manifest):
            language = row["language"]
            if language not in layout.languages:
                continue
            profile = (speaker, language)
            if target is not None and counts[profile] >= target:
                continue
            counts[profile] += 1
            selected.append({**row, "speaker": speaker})
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
    if layout.initialization_mode not in {"resume", "warm_start", "expand_speakers"}:
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
    """Generate a named VITS dataset using the official Qwen teacher runtime.

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
            if not output.is_file() and legacy.is_file() and not generation.get("overwrite", False):
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy, output)
                migrated += 1
            jobs.append(GenerationJob(item, candidate, output))
    if migrated:
        logger.info(
            "legacy model audio migrated voice_id=%s files=%d source=%s destination=%s",
            voice_id, migrated, legacy_wav_root, wav_root,
        )
    overwrite = bool(generation.get("overwrite", False))
    if overwrite:
        logger.warning(
            "generation.overwrite=true will regenerate selected shared audio voice_id=%s",
            voice_id,
        )
    pending = jobs if overwrite else [job for job in jobs if not job.output.is_file()]
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

        if mode == "design":
            prompt = voice.get("prompt", "").strip()
            reference_language = voice.get("reference_language", "en").lower()
            if not prompt or not reference_text:
                raise ValueError("design mode requires dataset.voice.prompt and reference_text")
            if reference_language not in registry:
                raise ValueError(f"unsupported design reference language: {reference_language}")
            reference_spec = registry[reference_language]
            if reference_spec.teacher_provider != "qwen" or not reference_spec.teacher_language:
                raise ValueError(f"reference language {reference_language} has no Qwen teacher mapping")
            reference_audio = references / "designed.wav"
            legacy_reference = layout.dataset_dir / "references" / f"{speaker}.designed.wav"
            if not reference_audio.is_file() and legacy_reference.is_file():
                reference_audio.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_reference, reference_audio)
                logger.info(
                    "legacy designed reference migrated source=%s destination=%s",
                    legacy_reference, reference_audio,
                )
            if reference_audio.is_file():
                reference_input = reference_audio
            else:
                design_model = model_loader(model_keys.get("voice_design", "voice-design-1.7b"), **common)
                _log_runtime_language_support(
                    design_model, {reference_spec.teacher_language},
                    model_keys.get("voice_design", "voice-design-1.7b"),
                )
                logger.info("creating designed reference voice speaker=%s language=%s", speaker, reference_language)
                ref_wavs, ref_rate = design_model.generate_voice_design(
                    text=reference_text,
                    language=reference_spec.teacher_language,
                    instruct=prompt,
                    **generation.get("generation_kwargs", {}),
                )
                reference_audio.parent.mkdir(parents=True, exist_ok=True)
                sf.write(reference_audio, np.asarray(ref_wavs[0]).squeeze(), ref_rate,
                         subtype="PCM_16", format="WAV")
                reference_input = (ref_wavs[0], ref_rate)
                del design_model
                _release_device_memory(device)
        else:
            reference_value = voice.get("reference_audio")
            if not reference_value:
                raise ValueError("clone mode requires dataset.voice.reference_audio")
            if not reference_text and not x_vector_only:
                raise ValueError("clone mode requires the exact reference_text unless x_vector_only_mode is true")
            uploaded = Path(reference_value).expanduser()
            reference_input = _copy_reference(
                uploaded, references / f"uploaded{uploaded.suffix or '.wav'}",
            )

        clone_model = model_loader(model_keys.get("voice_clone", "base-1.7b"), **common)
        _log_runtime_language_support(
            clone_model, set(teacher_languages.values()),
            model_keys.get("voice_clone", "base-1.7b"),
        )
        logger.info("creating reusable clone prompt speaker=%s mode=%s", speaker, mode)
        clone_prompt = clone_model.create_voice_clone_prompt(
            ref_audio=_qwen_reference_input(reference_input),
            ref_text=reference_text or None,
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
        generation_started = time.monotonic()
        total_batches = (len(pending) + batch_size - 1) // batch_size
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            batch_number = start // batch_size + 1
            batch_started = time.monotonic()
            batch_languages = ",".join(sorted({job.item.language for job in batch}))
            logger.debug(
                "audio batch started batch=%d/%d range=%d-%d languages=%s",
                batch_number, total_batches, start + 1, start + len(batch),
                batch_languages,
            )
            wavs, sample_rate = clone_model.generate_voice_clone(
                text=[job.item.text for job in batch],
                language=[teacher_languages[job.item.language] for job in batch],
                voice_clone_prompt=clone_prompt,
                **generation_kwargs,
            )
            if len(wavs) != len(batch):
                raise RuntimeError(f"Qwen returned {len(wavs)} waveforms for a batch of {len(batch)}")
            for job, waveform in zip(batch, wavs):
                _write_training_wav(job.output, waveform, sample_rate, target_rate)
            completed_new = start + len(batch)
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
                    cached_count, batch_number, total_batches, batch_languages,
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
    if trimmed:
        report_path = layout.dataset_dir / "audio-postprocess-report.json"
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
    included_manifests = [
        (Path(value), False) for value in generation.get("include_metadata", [])
    ]
    previous_metadata = (
        _checkpoint_dataset_metadata(layout)
        if layout.initialization_mode in {"resume", "expand_speakers"} else None
    )
    if previous_metadata is not None and all(
        path.resolve() != previous_metadata.resolve()
        for path, _ in included_manifests
    ):
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
    """Prepare every declared public voice, then assemble model speakers if requested."""
    raw, layout = resolve_experiment(config_path)
    voices = raw.get("generation", {}).get("voices") or {}
    if not voices:
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
