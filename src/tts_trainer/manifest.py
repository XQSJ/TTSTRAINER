"""训练数据 CSV 清单的读取与校验。 / Reading and validating training CSV manifests."""
from __future__ import annotations

import csv
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .languages import resolve_language_registry


def parse_phonemes(value: str) -> tuple[str, ...] | None:
    """把 CSV 中的音素串解析为元组，`<space>` 还原为空格。 / Parse a CSV phoneme string; `<space>` maps back to a space token."""
    tokens = tuple(" " if token == "<space>" else token for token in value.split())
    return tokens or None


def format_phonemes(tokens: tuple[str, ...]) -> str:
    """把音素元组写回 CSV 文本形式。 / Serialize phoneme tokens back to CSV text form."""
    return " ".join("<space>" if token == " " else token for token in tokens)


@dataclass(frozen=True)
class Item:
    """单条训练样本。 / One training sample."""
    audio: Path
    text: str
    language: str
    speaker: str
    phonemes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ValidationReport:
    """清单校验结果汇总。 / Summary of manifest validation."""
    items: tuple[Item, ...]
    language_counts: dict[str, int]
    sample_rates: tuple[int, ...]


def read_manifest(path: str | Path) -> list[Item]:
    """读取 CSV 清单为样本列表。 / Read a CSV manifest into items."""
    manifest = Path(path)
    # utf-8-sig 兼容带 BOM 的 Excel 导出文件。 / utf-8-sig tolerates BOM from Excel exports.
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"audio", "text", "language", "speaker"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata missing columns: {', '.join(sorted(missing))}")
        return [
            Item(
                audio=(manifest.parent / row["audio"].strip()).resolve(),
                text=row["text"].strip(),
                language=row["language"].strip().lower(),
                speaker=row["speaker"].strip(),
                phonemes=parse_phonemes(row.get("phonemes", "")),
            )
            for row in reader
        ]


def validate_manifest(path: str | Path, expected_sample_rate: int | None = None,
                      *, require_single_speaker: bool = True,
                      require_phonemes: bool = False,
                      supported_languages=None) -> ValidationReport:
    """校验清单完整性与音频格式。 / Validate manifest completeness and audio format."""
    items = read_manifest(path)
    if not items:
        raise ValueError("metadata contains no samples")
    errors: list[str] = []
    rates: set[int] = set()
    speakers = {item.speaker for item in items}
    if "" in speakers:
        errors.append("speaker must not be empty")
    elif require_single_speaker and len(speakers) != 1:
        errors.append(f"expected exactly one speaker, got {sorted(speakers)!r}")
    supported = set(supported_languages or resolve_language_registry())
    for index, item in enumerate(items, start=2):
        if item.language not in supported:
            errors.append(f"line {index}: unsupported language {item.language!r}")
        if not item.text:
            errors.append(f"line {index}: empty text")
        if require_phonemes and not item.phonemes:
            errors.append(f"line {index}: missing frozen phonemes")
        if not item.audio.is_file():
            errors.append(f"line {index}: missing audio {item.audio}")
            continue
        try:
            with wave.open(str(item.audio), "rb") as wav:
                if wav.getnchannels() != 1:
                    errors.append(f"line {index}: audio must be mono")
                if wav.getsampwidth() != 2:
                    errors.append(f"line {index}: audio must be 16-bit PCM")
                rate = wav.getframerate()
                rates.add(rate)
                if expected_sample_rate and rate != expected_sample_rate:
                    errors.append(f"line {index}: sample rate {rate}, expected {expected_sample_rate}")
        except wave.Error as exc:
            errors.append(f"line {index}: invalid PCM WAV ({exc})")
    if errors:
        raise ValueError("manifest validation failed:\n- " + "\n- ".join(errors))
    counts = Counter(item.language for item in items)
    return ValidationReport(tuple(items), dict(sorted(counts.items())), tuple(sorted(rates)))
