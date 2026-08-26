"""前端路由：按语言/模型声明选择 G2P 前端，并共享同一 token 空间。 / Frontend routing: picks the G2P frontend per language/model declaration while sharing one token space."""

from __future__ import annotations

from ..languages import resolve_language_registry
from .contract import FrontendContract, frontend_contract_from_config
from .contract import MOBILE_DIRECT_TOKEN_ENCODING, PIPER_TOKEN_ENCODING
from .espeak import EspeakFrontend
from .openjtalk import OpenJTalkFrontend
from .piper_plus import PiperPlusFrontend


class FrontendRouter:
    """按语言路由到各自配置的 G2P，同时共享同一 token 空间。 / Route each language to its configured G2P while sharing one token space."""

    def __init__(self, routes: dict[str, object], declared: FrontendContract):
        self.routes = dict(routes)
        self.declared = declared
        # 各 provider 声明的资源标识（voice/profile/dictionary），用于报告与契约导出。
        # Declared resource identifier per provider (voice/profile/dictionary).
        self.voices = {
            language: profile.get(
                "voice", profile.get("profile", profile.get("dictionary", profile["provider"]))
            )
            for language, profile in declared.languages.items()
        }

    def frontend_for(self, language: str):
        try:
            return self.routes[language]
        except KeyError as exc:
            raise ValueError(f"unsupported frontend language: {language}") from exc

    def provider_for(self, language: str) -> str:
        return self.declared.languages[language]["provider"]

    def version_for(self, language: str) -> str:
        return self.frontend_for(language).version()

    def version(self) -> str:
        versions = {
            self.provider_for(language): self.version_for(language)
            for language in self.routes
        }
        return "; ".join(f"{provider}={version}" for provider, version in sorted(versions.items()))

    def phonemize(self, text: str, language: str) -> tuple[str, ...]:
        return self.frontend_for(language).phonemize(text, language)

    def contract(self, languages) -> FrontendContract:
        """导出指定语言的冻结前端契约（含实测引擎版本与资源 ID）。 / Export the frozen frontend contract for the given languages, with detected engine versions and resource IDs."""
        profiles = {}
        for language in languages:
            profile = dict(self.declared.languages[language])
            frontend = self.frontend_for(language)
            profile["engine_version"] = frontend.version()
            # 各 provider 的资源指纹字段名不同，契约里统一到对应键。
            # Each provider uses a different resource fingerprint key.
            if isinstance(frontend, OpenJTalkFrontend):
                profile["dictionary"] = frontend.dictionary_id()
            elif isinstance(frontend, PiperPlusFrontend):
                profile["resource"] = frontend.resource_id()
            profiles[language] = profile
        return FrontendContract(
            provider=self.declared.provider,
            languages=profiles,
            token_encoding=self.declared.token_encoding,
        )


def frontend_from_config(config: dict | None = None, *, languages=None,
                         language_registry: dict | None = None) -> FrontendRouter:
    """按训练配置构建路由器，同一 provider 的前端实例按语言复用。 / Build a router from training config; one provider instance is reused across its languages."""
    config = config or {}
    languages = tuple(languages or ())
    registry = resolve_language_registry(language_registry)
    declared = frontend_contract_from_config(config, languages, language_registry=language_registry)
    voices = {
        code: spec.frontend_voice for code, spec in registry.items()
        if spec.frontend_provider == "espeak-ng"
    }
    voices.update(config.get("voices", {}))
    espeak = None
    openjtalk = None
    piper_plus = {}
    routes = {}
    for language in languages:
        profile = declared.languages[language]
        provider = profile["provider"]
        if provider == "espeak-ng":
            if espeak is None:
                espeak = EspeakFrontend(
                    executable=config.get("executable"),
                    voices=voices,
                    allow_language_switches=not bool(config.get("strict_language_switches", True)),
                )
            routes[language] = espeak
        elif provider == "openjtalk":
            if openjtalk is None:
                openjtalk_config = config.get("openjtalk", {})
                openjtalk = OpenJTalkFrontend(
                    user_dictionary=openjtalk_config.get("user_dictionary"),
                    dictionary_root=openjtalk_config.get("dictionary_root"),
                    auto_download_dictionary=bool(
                        openjtalk_config.get("auto_download_dictionary", True)
                    ),
                )
            routes[language] = openjtalk
        elif provider == "piper-plus-g2p":
            piper_config = config.get("piper_plus", {})
            piper_plus[language] = PiperPlusFrontend(
                language,
                resource_root=piper_config.get("resource_root"),
                auto_download_resources=bool(
                    piper_config.get("auto_download_resources", True)
                ),
            )
            routes[language] = piper_plus[language]
        else:
            # LanguageSpec 校验应使该分支不可达；仍兜底以防注册表被绕过。
            # LanguageSpec validation should make this unreachable.
            raise ValueError(f"unsupported frontend provider: {provider}")
    return FrontendRouter(routes, declared)


def frontend_from_contract(contract: FrontendContract, config: dict | None = None) -> FrontendRouter:
    """从已导出的前端契约重建运行时路由器。 / Recreate a runtime router from an exported frontend contract."""
    config = dict(config or {})
    # 用户自定义 Open JTalk 词典无法从契约恢复，必须由调用方再次提供路径。
    # Custom Open JTalk dictionaries cannot be recovered from the contract.
    user_languages = [
        language for language, profile in contract.languages.items()
        if str(profile.get("dictionary", "")).startswith("user:")
    ]
    if user_languages and not config.get("openjtalk", {}).get("user_dictionary"):
        raise RuntimeError(
            "export uses an Open JTalk user dictionary for "
            + ", ".join(user_languages)
            + "; supply frontend.openjtalk.user_dictionary"
        )
    # 把契约语言档案反向翻译成 frontend_from_config 可消费的注册表项。
    # Translate contract language profiles back into registry entries.
    registry = {
        language: {
            "name": language,
            "teacher": None,
            "frontend": {
                key: value for key, value in profile.items()
                if key != "engine_version"  # 机器检测的版本不属于声明语义 / machine-detected, not declarable
            },
            "smoke_text": language,
        }
        for language, profile in contract.languages.items()
    }
    voices = {
        language: profile["voice"]
        for language, profile in contract.languages.items()
        if profile.get("provider", "espeak-ng") == "espeak-ng"
    }
    resolved_config = dict(config)
    # 由契约的 token 编码反推兼容开关，保证重建结果与导出时语义一致。
    # Derive compatibility switches from the contract's token encoding.
    resolved_config["provider"] = contract.provider
    resolved_config["piper_compatible"] = (
        contract.token_encoding == PIPER_TOKEN_ENCODING
    )
    resolved_config["mobile_direct"] = (
        contract.token_encoding == MOBILE_DIRECT_TOKEN_ENCODING
    )
    resolved_config["voices"] = {**voices, **config.get("voices", {})}
    return frontend_from_config(
        resolved_config,
        languages=tuple(contract.languages),
        language_registry=registry,
    )
