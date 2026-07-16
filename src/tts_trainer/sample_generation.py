from __future__ import annotations

import csv
import gc
import importlib.util
import logging
import os
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


@dataclass(frozen=True)
class GenerationText:
    text: str
    language: str


@dataclass(frozen=True)
class GenerationJob:
    item: GenerationText
    candidate: int
    output: Path


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
    wav_root = layout.dataset_dir / "wavs" / speaker
    jobs = [
        GenerationJob(item, candidate, wav_root / f"{item.language}_{index:06d}_c{candidate:02d}.wav")
        for index, item in enumerate(texts, start=1)
        for candidate in range(1, candidates + 1)
    ]
    overwrite = bool(generation.get("overwrite", False))
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
        references = layout.dataset_dir / "references"
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
    logger.info("training metadata written=%s samples=%d", output_metadata, len(rows))
    return output_metadata
