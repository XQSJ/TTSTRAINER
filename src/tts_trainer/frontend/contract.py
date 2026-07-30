from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..languages import resolve_language_registry


FRONTEND_CONTRACT_FORMAT = 1
NORMALIZATION_CONTRACT = "unicode-nfkc-collapse-whitespace-v1"
TOKEN_CONTRACT = "routed-phoneme-units-v1"
DIRECT_TOKEN_ENCODING = "bos-phonemes-eos-v1"
# Piper's canonical phonemes_to_ids sequence is:
#   BOS, PAD, (phoneme, PAD)*, EOS
# Version 1 of TTSTRAINER (and sherpa-onnx 1.13.4) omitted the PAD after BOS.
# Keep the old name solely so old mobile checkpoints can be rejected clearly.
LEGACY_PIPER_TOKEN_ENCODING = "piper-bos-phoneme-pad-eos-v1"
PIPER_TOKEN_ENCODING = "piper-bos-pad-phoneme-pad-eos-v2"
MOBILE_ESPEAK_VOICES = {
    "zh": "cmn",
    "en": "en-us",
    "ja": "ja",
    "ko": "ko",
    "de": "de",
    "fr": "fr-fr",
    "ru": "ru",
    "pt": "pt-br",
    "es": "es",
    "it": "it",
}
DEFAULT_ESPEAK_VOICES = dict(MOBILE_ESPEAK_VOICES)
DEFAULT_ESPEAK_VOICES.update({
    code: spec.frontend_voice for code, spec in resolve_language_registry().items()
    if spec.frontend_provider == "espeak-ng"
})


@dataclass(frozen=True)
class FrontendContract:
    provider: str
    languages: dict[str, dict[str, str]]
    engine_version: str | None = None
    format: int = FRONTEND_CONTRACT_FORMAT
    normalization: str = NORMALIZATION_CONTRACT
    tokens: str = TOKEN_CONTRACT
    token_encoding: str = DIRECT_TOKEN_ENCODING

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "provider": self.provider,
            "normalization": self.normalization,
            "tokens": self.tokens,
            "token_encoding": self.token_encoding,
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
            token_encoding=str(raw.get("token_encoding", DIRECT_TOKEN_ENCODING)),
        )

    def compatibility_key(self) -> tuple:
        """Return the exact frozen frontend contract, including engine versions."""
        return (
            self.format,
            self.provider,
            self.normalization,
            self.tokens,
            self.token_encoding,
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
            self.token_encoding,
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
    registry_voices = {
        code: spec.frontend_voice for code, spec in registry.items()
        if spec.frontend_provider == "espeak-ng"
    }
    voices = {**DEFAULT_ESPEAK_VOICES, **registry_voices, **config.get("voices", {})}
    missing = {
        language for language in languages
        if language not in registry or (provider == "espeak-ng" and language not in voices)
        or (
            provider == "language-router"
            and registry[language].frontend_provider == "espeak-ng"
            and language not in voices
        )
    }
    if missing:
        raise ValueError(f"missing frontend profiles for: {', '.join(sorted(missing))}")
    profiles = {}
    for language in languages:
        spec = registry[language]
        if provider == "espeak-ng":
            profile = {"provider": "espeak-ng", "voice": voices[language]}
        else:
            profile = {"provider": spec.frontend_provider, **spec.frontend_profile}
        if profile["provider"] == "espeak-ng":
            profile["voice"] = voices[language]
        elif profile["provider"] == "openjtalk":
            user_dictionary = config.get("openjtalk", {}).get("user_dictionary")
            if user_dictionary:
                path = Path(user_dictionary).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Open JTalk user dictionary not found: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                profile["dictionary"] = f"user:{path.name}:sha256:{digest}"
        profiles[language] = profile
    return FrontendContract(
        provider=provider,
        engine_version=engine_version,
        languages=profiles,
        token_encoding=(
            PIPER_TOKEN_ENCODING if provider == "espeak-ng"
            and bool(config.get("piper_compatible", False))
            else DIRECT_TOKEN_ENCODING
        ),
    )
