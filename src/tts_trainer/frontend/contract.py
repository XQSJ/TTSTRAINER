from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..languages import resolve_language_registry


FRONTEND_CONTRACT_FORMAT = 1
NORMALIZATION_CONTRACT = "unicode-nfkc-collapse-whitespace-v1"
TOKEN_CONTRACT = "routed-phoneme-units-v1"
DEFAULT_ESPEAK_VOICES = {
    code: spec.frontend_voice for code, spec in resolve_language_registry().items()
    if spec.frontend_provider == "espeak-ng"
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
        """Return the exact frozen frontend contract, including engine versions."""
        return (
            self.format,
            self.provider,
            self.normalization,
            self.tokens,
            self.engine_version,
            json.dumps(self.languages, ensure_ascii=False, sort_keys=True),
        )

    def declaration_key(self) -> tuple:
        """Return config-declarable semantics without machine-detected versions."""
        languages = {
            language: {key: value for key, value in profile.items() if key != "engine_version"}
            for language, profile in self.languages.items()
        }
        return (
            self.format,
            self.provider,
            self.normalization,
            self.tokens,
            json.dumps(languages, ensure_ascii=False, sort_keys=True),
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
                                  *, engine_version: str | None = None,
                                  language_registry: dict | None = None) -> FrontendContract:
    config = config or {}
    provider = config.get("provider", "language-router")
    if provider not in {"language-router", "espeak-ng"}:
        raise ValueError(
            f"unsupported frontend provider: {provider!r}; currently available: language-router"
        )
    registry = resolve_language_registry(language_registry)
    if provider == "espeak-ng":
        routed = sorted(
            language for language in languages
            if registry[language].frontend_provider != "espeak-ng"
        )
        if routed:
            raise ValueError(
                "frontend.provider=espeak-ng cannot serve routed languages: "
                + ", ".join(routed)
                + "; use frontend.provider=language-router"
            )
    registry_voices = {
        code: spec.frontend_voice for code, spec in registry.items()
        if spec.frontend_provider == "espeak-ng"
    }
    voices = {**DEFAULT_ESPEAK_VOICES, **registry_voices, **config.get("voices", {})}
    missing = {
        language for language in languages
        if language not in registry or (
            registry[language].frontend_provider == "espeak-ng" and language not in voices
        )
    }
    if missing:
        raise ValueError(f"missing frontend profiles for: {', '.join(sorted(missing))}")
    profiles = {}
    for language in languages:
        spec = registry[language]
        profile = {"provider": spec.frontend_provider, **spec.frontend_profile}
        if spec.frontend_provider == "espeak-ng":
            profile["voice"] = voices[language]
        elif spec.frontend_provider == "openjtalk":
            user_dictionary = config.get("openjtalk", {}).get("user_dictionary")
            if user_dictionary:
                path = Path(user_dictionary).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Open JTalk user dictionary not found: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                profile["dictionary"] = f"user:{path.name}:sha256:{digest}"
        profiles[language] = profile
    return FrontendContract(
        provider="language-router",
        engine_version=engine_version,
        languages=profiles,
    )
