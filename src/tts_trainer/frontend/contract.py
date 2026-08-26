"""前端契约：冻结归一化、token 编码与各语言 G2P 档案，保证训练与部署一致。 / Frontend contract: freezes normalization, token encoding and per-language G2P profiles so training matches deployment."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..languages import resolve_language_registry


FRONTEND_CONTRACT_FORMAT = 1  # 契约文件格式版本号 / contract file format version
NORMALIZATION_CONTRACT = "unicode-nfkc-collapse-whitespace-v1"  # 文本归一化契约标识 / text normalization contract id
TOKEN_CONTRACT = "routed-phoneme-units-v1"  # 音素单元契约标识 / phoneme-unit contract id
DIRECT_TOKEN_ENCODING = "bos-phonemes-eos-v1"  # 默认编码：BOS+音素+EOS / default encoding: BOS+phonemes+EOS
# Piper 官方序列 / canonical phonemes_to_ids sequence:
#   BOS, PAD, (phoneme, PAD)*, EOS
# TTSTRAINER v1 和 sherpa-onnx 1.13.4 的 wire 输入缺少 BOS 后的 PAD。
# The v1/sherpa-onnx 1.13.4 wire input omitted the PAD after BOS.
LEGACY_PIPER_TOKEN_ENCODING = "piper-bos-phoneme-pad-eos-v1"
PIPER_TOKEN_ENCODING = "piper-bos-pad-phoneme-pad-eos-v2"
# Mobile v3 将 Piper 传输 PAD 与 VITS 训练序列分离；ONNX 适配器先删除 PAD。
# Mobile v3 separates Piper wire PADs from VITS training; the ONNX adapter
# strips PADs before the text encoder sees the compact sequence.
MOBILE_DIRECT_TOKEN_ENCODING = "mobile-espeak-bos-phonemes-eos-v3"
# Android 端 espeak 语音映射表；移动端与训练端必须使用同一 voice 才能保证音素一致。
# Android-side espeak voice table; mobile must use identical voices for phoneme parity.
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
# 默认 voice 集以移动端映射为基底，再用语言注册表覆盖。
# Default voices start from the mobile table, then registry entries override.
DEFAULT_ESPEAK_VOICES = dict(MOBILE_ESPEAK_VOICES)
DEFAULT_ESPEAK_VOICES.update({
    code: spec.frontend_voice for code, spec in resolve_language_registry().items()
    if spec.frontend_provider == "espeak-ng"
})


@dataclass(frozen=True)
class FrontendContract:
    """训练与部署之间冻结的前端语义契约。 / The frozen frontend semantics shared between training and deployment."""

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
        """返回包含引擎版本的完全冻结契约键。 / Return the exact frozen frontend contract, including engine versions."""
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
        """返回仅含配置可声明语义的键（剥离机器检测的引擎版本）。 / Return config-declarable semantics without machine-detected versions."""
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
    """返回与 manifest 同目录的前端锁文件路径。 / Return the frontend lock path next to the manifest."""
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
    """从训练配置推导前端契约（含 voice 合并与用户词典指纹）。 / Derive the frontend contract from training config, merging voices and hashing user dictionaries."""
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
    # 每个训练语言都必须有前端档案；espeak 语言还必须有可用 voice。
    # Every training language needs a profile; espeak languages also need a voice.
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
    if bool(config.get("mobile_direct", False)) and bool(
        config.get("piper_compatible", False)
    ):
        raise ValueError(
            "frontend.mobile_direct and frontend.piper_compatible are mutually exclusive"
        )
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
            # 用户词典以「文件名+sha256 前 16 位」写入契约，内容变化即改变契约键。
            # User dictionaries enter the contract as name+sha256 prefix, so content changes alter the key.
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
            # 两个互斥的兼容开关分别映射到移动端直连与 Piper 序列编码。
            # Two mutually exclusive switches map to mobile-direct or Piper encodings.
            MOBILE_DIRECT_TOKEN_ENCODING
            if provider == "espeak-ng" and bool(config.get("mobile_direct", False))
            else (
                PIPER_TOKEN_ENCODING
                if provider == "espeak-ng"
                and bool(config.get("piper_compatible", False))
                else DIRECT_TOKEN_ENCODING
            )
        ),
    )
