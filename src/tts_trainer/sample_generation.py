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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .experiments import prepare_experiment, resolve_experiment
from .languages import resolve_language_registry
from .logging_utils import configure_logging
from .manifest import read_manifest
from .qwen_teacher import load_qwen_teacher
from .text_generation import text_corpus_path


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


def _voice_dataset(raw: dict, layout, generation: dict,
                   voice: dict) -> tuple[str, str, Path, dict]:
    """Resolve a model-independent, revision-safe voice dataset directory."""
    voice_id = str(voice.get("id") or voice.get("speaker") or "voice_01").strip()
    if not VOICE_ID.fullmatch(voice_id):
        raise ValueError(
            "generation.voice.id must contain only letters, numbers, '.', '_' and '-', "
            "and cannot start with punctuation"
        )
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
    identity = {
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
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    revision = hashlib.sha256(encoded).hexdigest()[:12]
    root = Path(generation.get("voice_dataset_root") or layout.dataset_dir.parent / "voices")
    destination = root / voice_id / revision
    destination.mkdir(parents=True, exist_ok=True)
    record = destination / "voice.json"
    if not record.is_file():
        temporary = record.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "format": 1,
            "voice_id": voice_id,
            "revision": revision,
            "identity": identity,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(record)
    return voice_id, revision, destination, identity


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


def _runtime_kwargs(config: dict) -> tuple[str, dict]:
    runtime = config.get("runtime", {})
    requested = runtime.get("device", "auto")
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


def generate_samples(config_path: str | Path, *, text_manifest_path: str | Path | None = None,
                     model_loader: Callable = load_qwen_teacher) -> Path:
    """Generate a named VITS dataset using the official Qwen teacher runtime.

    Voice modes follow the official Qwen3-TTS README:
    - design: VoiceDesign creates one reference, then Base clones it for all rows.
    - clone: Base creates a reusable prompt from uploaded reference audio + transcript.
    """
    raw, layout = resolve_experiment(config_path)
    configure_logging(raw.get("logging", {}).get("level", "INFO"))
    prepare_experiment(layout, raw, config_path)
    registry = resolve_language_registry(raw.get("language_registry"))
    logger.info("model=%s languages=%s", layout.name, ",".join(layout.languages))
    generation = raw.get("generation", {})
    if not generation.get("enabled", True):
        raise ValueError("sample generation is disabled in this config")

    text_generation = raw.get("text_generation", {})
    generated_default = None
    if text_generation.get("enabled", False):
        generated_default = text_corpus_path(text_generation, layout)
    text_manifest = Path(
        text_manifest_path or generation.get("text_manifest")
        or generated_default or layout.dataset_dir / "texts.csv"
    )
    output_metadata = Path(generation.get("raw_metadata") or layout.dataset_dir / "metadata.csv")
    all_texts = read_generation_texts(text_manifest, registry)
    texts = [item for item in all_texts if item.language in layout.languages]
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
    voice = generation.get("voice") or {}
    mode = voice.get("mode")
    if mode not in {"design", "clone"}:
        raise ValueError("generation.voice.mode must be design or clone")
    speaker = voice.get("speaker", "voice_01").strip()
    if not speaker:
        raise ValueError("generation.voice.speaker must not be empty")

    candidates = int(generation.get("candidates_per_text", 1))
    if candidates < 1:
        raise ValueError("generation.candidates_per_text must be at least 1")
    voice_id, voice_revision, voice_dataset, _ = _voice_dataset(
        raw, layout, generation, voice,
    )
    logger.info(
        "voice dataset ready voice_id=%s revision=%s path=%s speaker_label=%s",
        voice_id, voice_revision, voice_dataset, speaker,
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
            "generation.overwrite=true will regenerate shared voice audio voice_id=%s revision=%s",
            voice_id, voice_revision,
        )
    pending = jobs if overwrite else [job for job in jobs if not job.output.is_file()]
    logger.info("generation jobs total=%d pending=%d cached=%d", len(jobs), len(pending), len(jobs) - len(pending))

    if pending:
        device, load_kwargs = _runtime_kwargs(generation)
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
                raise ValueError("design mode requires generation.voice.prompt and reference_text")
            if reference_language not in registry:
                raise ValueError(f"unsupported design reference language: {reference_language}")
            reference_spec = registry[reference_language]
            if reference_spec.teacher_provider != "qwen" or not reference_spec.teacher_language:
                raise ValueError(f"reference language {reference_language} has no Qwen teacher mapping")
            reference_audio = references / f"{speaker}.designed.wav"
            legacy_reference = layout.dataset_dir / "references" / f"{speaker}.designed.wav"
            if not reference_audio.is_file() and legacy_reference.is_file() and not overwrite:
                reference_audio.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_reference, reference_audio)
                logger.info(
                    "legacy designed reference migrated source=%s destination=%s",
                    legacy_reference, reference_audio,
                )
            if reference_audio.is_file() and not overwrite:
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
                raise ValueError("clone mode requires generation.voice.reference_audio")
            if not reference_text and not x_vector_only:
                raise ValueError("clone mode requires the exact reference_text unless x_vector_only_mode is true")
            uploaded = Path(reference_value).expanduser()
            reference_input = _copy_reference(uploaded, references / f"{speaker}.uploaded{uploaded.suffix or '.wav'}")

        clone_model = model_loader(model_keys.get("voice_clone", "base-1.7b"), **common)
        _log_runtime_language_support(
            clone_model, set(teacher_languages.values()),
            model_keys.get("voice_clone", "base-1.7b"),
        )
        logger.info("creating reusable clone prompt speaker=%s mode=%s", speaker, mode)
        clone_prompt = clone_model.create_voice_clone_prompt(
            ref_audio=reference_input,
            ref_text=reference_text or None,
            x_vector_only_mode=x_vector_only,
        )
        batch_size = int(generation.get("batch_size", 4))
        if batch_size < 1:
            raise ValueError("generation.batch_size must be at least 1")
        generation_kwargs = generation.get("generation_kwargs", {})
        target_rate = int(raw["audio"]["sample_rate"])
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            logger.info("generating samples %d-%d/%d", start + 1, start + len(batch), len(pending))
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
        del clone_model
        _release_device_memory(device)

    postprocess_config = generation.get("audio_postprocess", {})
    trimmed = [
        result for job in jobs
        if (result := _postprocess_training_wav(job.output, postprocess_config)) is not None
    ]
    logger.info(
        "audio postprocess status=completed checked=%d trimmed=%d",
        len(jobs), len(trimmed),
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

    rows = []
    seen = set()
    for included_path in generation.get("include_metadata", []):
        for item in read_manifest(included_path):
            if item.language not in layout.languages:
                raise ValueError(
                    f"included metadata contains language {item.language!r} not enabled by experiment.languages"
                )
            key = (str(item.audio), item.text, item.language, item.speaker)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "audio": os.path.relpath(item.audio.resolve(), output_metadata.parent.resolve()),
                "text": item.text,
                "language": item.language,
                "speaker": item.speaker,
            })
    for job in jobs:
        key = (str(job.output.resolve()), job.item.text, job.item.language, speaker)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "audio": os.path.relpath(job.output.resolve(), output_metadata.parent.resolve()),
            "text": job.item.text,
            "language": job.item.language,
            "speaker": speaker,
        })
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_metadata.with_suffix(output_metadata.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_metadata)
    (layout.dataset_dir / "dataset.json").write_text(json.dumps({
        "format": 1,
        "model": layout.name,
        "metadata": str(output_metadata.resolve()),
        "text_manifest": str(text_manifest.resolve()),
        "voice_id": voice_id,
        "voice_revision": voice_revision,
        "voice_dataset": str(voice_dataset.resolve()),
        "speaker_label": speaker,
        "samples": len(rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("training metadata written=%s samples=%d", output_metadata, len(rows))
    return output_metadata
