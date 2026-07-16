from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path

from ..languages import resolve_language_registry
from ..manifest import format_phonemes, read_manifest
from ..text import normalize
from .contract import (DEFAULT_ESPEAK_VOICES, FrontendContract,
                       frontend_lock_path, save_frontend_contract)


ESPEAK_VOICES = DEFAULT_ESPEAK_VOICES
ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")


def parse_espeak_ipa(output: str) -> tuple[str, ...]:
    """Parse eSpeak IPA into Piper-compatible UTF-8 codepoint tokens."""
    output = ZERO_WIDTH.sub("", output.strip())
    output = re.sub(r"\([a-z][a-z-]*\)", "", output)
    # piper-phonemize maps individual UTF-8 codepoints, not whole IPA phones.
    # Collapse all sentence/word whitespace to the standard Piper space token.
    normalized = re.sub(r"\s+", " ", output.replace("|", "")).strip()
    return tuple(normalized)


class EspeakFrontend:
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
        if language not in self.voices:
            raise ValueError(f"unsupported language: {language}")
        result = subprocess.run(
            [self.executable, "-q", "--ipa=1", "--sep=|", "-v", self.voices[language],
             normalize(text, language)],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        cleaned = ZERO_WIDTH.sub("", result.stdout)
        switches = sorted({
            value for value in re.findall(r"\(([a-z][a-z-]*)\)", cleaned)
            if value != language and value != self.voices[language]
        })
        if switches and not self.allow_language_switches:
            raise ValueError(
                f"eSpeak changed language while phonemizing {language}: {', '.join(switches)}; "
                "use a language-specific frontend instead of accepting fallback pronunciation"
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
    source = Path(source); destination = Path(destination)
    frontend = frontend or EspeakFrontend()
    items = read_manifest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    languages = tuple(dict.fromkeys(item.language for item in items))
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio", "text", "language", "speaker", "phonemes"])
        writer.writeheader()
        for item in items:
            phones = item.phonemes or frontend.phonemize(item.text, item.language)
            try:
                audio = item.audio.relative_to(destination.parent)
            except ValueError:
                audio = item.audio
            writer.writerow({
                "audio": str(audio), "text": item.text, "language": item.language,
                "speaker": item.speaker, "phonemes": format_phonemes(phones),
            })
    if hasattr(frontend, "contract"):
        save_frontend_contract(frontend.contract(languages), lock_path or frontend_lock_path(destination))
    return destination


def espeak_frontend_from_config(config: dict | None = None, *, languages=None,
                                language_registry: dict | None = None) -> EspeakFrontend:
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
