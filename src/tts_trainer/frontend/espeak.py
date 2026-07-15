from __future__ import annotations

import csv
import re
import shutil
import subprocess
from pathlib import Path

from ..constants import LANGUAGES
from ..manifest import format_phonemes, read_manifest
from ..text import normalize


ESPEAK_VOICES = {
    "zh": "cmn",
    "en": "en-us",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr-fr",
    "es": "es",
    "pt": "pt-br",
}
ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")


def parse_espeak_ipa(output: str) -> tuple[str, ...]:
    """Parse eSpeak IPA into Piper-compatible UTF-8 codepoint tokens."""
    output = ZERO_WIDTH.sub("", output.strip())
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
        missing = set(LANGUAGES) - set(self.voices)
        if missing:
            raise ValueError(f"missing eSpeak voices for: {', '.join(sorted(missing))}")

    def version(self) -> str:
        result = subprocess.run([self.executable, "--version"], check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return result.stdout.splitlines()[0].strip()

    def phonemize(self, text: str, language: str) -> tuple[str, ...]:
        if language not in self.voices:
            raise ValueError(f"unsupported language: {language}")
        result = subprocess.run(
            [self.executable, "-q", "--ipa=1", "--sep=|", "-v", self.voices[language],
             normalize(text, language)],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        cleaned = ZERO_WIDTH.sub("", result.stdout)
        switches = sorted(set(re.findall(r"\(([a-z][a-z-]*)\)", cleaned)))
        if switches and not self.allow_language_switches:
            raise ValueError(
                f"eSpeak changed language while phonemizing {language}: {', '.join(switches)}; "
                "use a language-specific frontend instead of accepting fallback pronunciation"
            )
        phones = parse_espeak_ipa(result.stdout)
        if not phones:
            raise ValueError(f"phonemizer produced no tokens for {text!r}")
        return phones


def phonemize_manifest(source: str | Path, destination: str | Path,
                       frontend: EspeakFrontend | None = None) -> Path:
    source = Path(source); destination = Path(destination)
    frontend = frontend or EspeakFrontend()
    items = read_manifest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    return destination
