"""eSpeak-NG G2P 前端（GPL）：通过子进程调用 espeak-ng 输出 IPA 音素。 / eSpeak-NG G2P frontend (GPL): shells out to espeak-ng for IPA phonemes."""

from __future__ import annotations

import csv
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..languages import resolve_language_registry
from ..logging_utils import TerminalProgress, format_duration, progress_bar
from ..manifest import format_phonemes, read_manifest
from ..text import normalize
from .contract import (DEFAULT_ESPEAK_VOICES, FrontendContract,
                       frontend_lock_path, save_frontend_contract)


ESPEAK_VOICES = DEFAULT_ESPEAK_VOICES
# \u96f6\u5bbd\u5b57\u7b26\u4f1a\u6df7\u5165 IPA \u8f93\u51fa\u6c61\u67d3 token \u96c6\uff0c\u5fc5\u987b\u5254\u9664\u3002
# Zero-width characters pollute the IPA token stream and must be stripped.
ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")
logger = logging.getLogger(__name__)


def parse_espeak_ipa(output: str) -> tuple[str, ...]:
    """把 eSpeak IPA 解析为 Piper 兼容的 UTF-8 码点 token。 / Parse eSpeak IPA into Piper-compatible UTF-8 codepoint tokens."""
    output = ZERO_WIDTH.sub("", output.strip())
    output = re.sub(r"\([a-z][a-z-]*\)", "", output)  # 移除语言切换标记 / drop language-switch markers
    # piper-phonemize maps individual UTF-8 codepoints, not whole IPA phones.
    # Collapse all sentence/word whitespace to the standard Piper space token.
    normalized = re.sub(r"\s+", " ", output.replace("|", "")).strip()
    return tuple(normalized)


class EspeakFrontend:
    """以子进程方式驱动 espeak-ng 的音素化前端。 / Phonemization frontend that drives espeak-ng via subprocess."""

    def __init__(self, executable: str | None = None, voices: dict[str, str] | None = None,
                 *, allow_language_switches: bool = False):
        self.executable = executable or shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.executable:
            raise RuntimeError("espeak-ng is not installed")
        self.voices = dict(voices or ESPEAK_VOICES)
        self.allow_language_switches = allow_language_switches

    def version(self) -> str:
        result = subprocess.run([self.executable, "--version"], check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # The trailing data-directory path is machine-specific and is not part
        # of the phonemizer version contract.
        return result.stdout.splitlines()[0].split("  Data at:", 1)[0].strip()

    def contract(self, languages: tuple[str, ...] | list[str]) -> FrontendContract:
        """按当前 espeak 版本生成这些语言的冻结契约。 / Build a frozen contract for these languages at the current espeak version."""
        missing = set(languages) - set(self.voices)
        if missing:
            raise ValueError(f"missing eSpeak voices for: {', '.join(sorted(missing))}")
        return FrontendContract(
            provider="espeak-ng",
            engine_version=self.version(),
            languages={language: {
                "provider": "espeak-ng", "voice": self.voices[language],
            } for language in languages},
        )

    def phonemize(self, text: str, language: str) -> tuple[str, ...]:
        """音素化文本，并拒绝未回归目标语言的语言切换。 / Phonemize text, rejecting language switches that do not return to the target language."""
        if language not in self.voices:
            raise ValueError(f"unsupported language: {language}")
        result = subprocess.run(
            [self.executable, "-q", "--ipa=1", "--sep=|", "-v", self.voices[language],
             normalize(text, language)],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        cleaned = ZERO_WIDTH.sub("", result.stdout)
        markers = re.findall(r"\(([a-z][a-z-]*)\)", cleaned)
        expected = {
            language, language.split("-", 1)[0], self.voices[language],
            self.voices[language].split("-", 1)[0],
        }
        # 只有「切走后从未切回」的标记才算失控的多语言混读。
        # Only switches that never switch back count as unbalanced drift.
        unbalanced = []
        for index, value in enumerate(markers):
            if value in expected:
                continue
            if not any(later in expected for later in markers[index + 1:]):
                unbalanced.append(value)
        if unbalanced and not self.allow_language_switches:
            raise ValueError(
                f"eSpeak did not return to {language} after switching to: "
                f"{', '.join(sorted(set(unbalanced)))}; use a language-specific frontend "
                "instead of accepting fallback pronunciation"
            )
        # eSpeak emits language-switch markers such as `(en)` as control
        # annotations; they are not phonemes and must never enter the token set.
        phones = parse_espeak_ipa(cleaned)
        if not phones:
            raise ValueError(f"phonemizer produced no tokens for {text!r}")
        return phones


def phonemize_manifest(source: str | Path, destination: str | Path,
                       frontend: EspeakFrontend | None = None,
                       *, lock_path: str | Path | None = None) -> Path:
    """对整个 manifest 批量音素化并写出处契约锁文件。 / Phonemize a whole manifest and write the frontend contract lock alongside it."""
    source = Path(source); destination = Path(destination)
    frontend = frontend or EspeakFrontend()
    items = read_manifest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    languages = tuple(dict.fromkeys(item.language for item in items))
    total = len(items)
    interval = max(1, total // 20)
    started = time.monotonic()
    live_progress = TerminalProgress("PHONEMIZE", total)
    logger.info("PHONEMIZE START | total=%d | output=%s", total, destination)
    # 先写临时文件再原子替换，避免中断留下半成品清单。
    # Write to a temporary file and atomically replace, so interrupts never leave a partial manifest.
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker", "phonemes"])
        writer.writeheader()
        for index, item in enumerate(items, 1):
            # 已冻结的音素直接复用，避免重复跑子进程。
            # Reuse frozen phonemes instead of re-running the subprocess.
            phones = item.phonemes or frontend.phonemize(item.text, item.language)
            try:
                audio = item.audio.relative_to(destination.parent)
            except ValueError:
                audio = item.audio
            writer.writerow({
                "audio": str(audio), "text": item.text, "language": item.language,
                "speaker": item.speaker, "phonemes": format_phonemes(phones),
            })
            live_progress.update(index, f"language={item.language}")
            if index % interval == 0 or index == total:
                live_progress.clear()
                elapsed = time.monotonic() - started
                rate = index / max(elapsed, 1e-9)
                logger.info(
                    "PHONEMIZE %s %6.2f%% | completed=%d/%d | speed=%.1f/s | ETA=%s",
                    progress_bar(index, total), 100.0 * index / max(total, 1),
                    index, total, rate, format_duration((total - index) / rate),
                    extra={"tts_style": "progress"},
                )
                live_progress.update(index, f"language={item.language}")
    live_progress.close()
    temporary.replace(destination)
    if hasattr(frontend, "contract"):
        save_frontend_contract(frontend.contract(languages), lock_path or frontend_lock_path(destination))
    logger.info(
        "PHONEMIZE DONE | completed=%d | elapsed=%s | output=%s",
        total, format_duration(time.monotonic() - started), destination,
        extra={"tts_style": "success"},
    )
    return destination


def espeak_frontend_from_config(config: dict | None = None, *, languages=None,
                                language_registry: dict | None = None) -> EspeakFrontend:
    """从训练配置构建单一 espeak 前端。 / Build a single espeak frontend from training config."""
    config = config or {}
    provider = config.get("provider", "espeak-ng")
    if provider != "espeak-ng":
        raise ValueError(f"unsupported frontend provider: {provider!r}; currently available: espeak-ng")
    specs = resolve_language_registry(language_registry)
    voices = {
        **ESPEAK_VOICES,
        **{code: spec.frontend_voice for code, spec in specs.items()},
        **config.get("voices", {}),
    }
    missing = set(languages or ()) - set(voices)
    if missing:
        raise ValueError(f"missing eSpeak voices for: {', '.join(sorted(missing))}")
    return EspeakFrontend(
        executable=config.get("executable"),
        voices=voices,
        allow_language_switches=not bool(config.get("strict_language_switches", True)),
    )
