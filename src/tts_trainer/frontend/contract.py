from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


FRONTEND_CONTRACT_FORMAT = 1
NORMALIZATION_CONTRACT = "unicode-nfkc-collapse-whitespace-v1"
TOKEN_CONTRACT = "piper-utf8-codepoints-v1"
DEFAULT_ESPEAK_VOICES = {
    "zh": "cmn",
    "en": "en-us",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr-fr",
    "es": "es",
    "pt": "pt-br",
}


@dataclass(frozen=True)
class FrontendContract:
    provider: str
    languages: dict[str, dict[str, str]]
    engine_version: str | None = None
    format: int = FRONTEND_CONTRACT_FORMAT
    normalization: str = NORMALIZATION_CONTRACT
    tokens: str = TOKEN_CONTRACT

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "provider": self.provider,
            "normalization": self.normalization,
            "tokens": self.tokens,
            "engine_version": self.engine_version,
            "languages": self.languages,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "FrontendContract":
        if int(raw.get("format", 0)) != FRONTEND_CONTRACT_FORMAT:
            raise ValueError("unsupported frontend contract format")
        languages = raw.get("languages")
        if not isinstance(languages, dict) or not languages:
            raise ValueError("frontend contract must contain languages")
        return cls(
            provider=str(raw["provider"]),
            languages={str(key): dict(value) for key, value in languages.items()},
            engine_version=raw.get("engine_version"),
            normalization=str(raw.get("normalization", NORMALIZATION_CONTRACT)),
            tokens=str(raw.get("tokens", TOKEN_CONTRACT)),
        )

    def compatibility_key(self) -> tuple:
        """Return the token-semantic contract, excluding the diagnostic engine version."""
        return (
            self.format,
            self.provider,
            self.normalization,
            self.tokens,
            json.dumps(self.languages, ensure_ascii=False, sort_keys=True),
        )


def frontend_lock_path(metadata_path: str | Path) -> Path:
    return Path(metadata_path).with_name("frontend.lock.json")


def save_frontend_contract(contract: FrontendContract, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(contract.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_frontend_contract(path: str | Path) -> FrontendContract:
    return FrontendContract.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def frontend_contract_from_config(config: dict | None, languages,
                                  *, engine_version: str | None = None) -> FrontendContract:
    config = config or {}
    provider = config.get("provider", "espeak-ng")
    if provider != "espeak-ng":
        raise ValueError(f"unsupported frontend provider: {provider!r}; currently available: espeak-ng")
    voices = {**DEFAULT_ESPEAK_VOICES, **config.get("voices", {})}
    missing = set(languages) - set(voices)
    if missing:
        raise ValueError(f"missing eSpeak voices for: {', '.join(sorted(missing))}")
    return FrontendContract(
        provider=provider,
        engine_version=engine_version,
        languages={language: {"voice": voices[language]} for language in languages},
    )
