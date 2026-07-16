from __future__ import annotations

from ..languages import resolve_language_registry
from .contract import FrontendContract, frontend_contract_from_config
from .espeak import EspeakFrontend
from .openjtalk import OpenJTalkFrontend
from .piper_plus import PiperPlusFrontend


class FrontendRouter:
    """Route each language to its configured G2P while sharing one token space."""

    def __init__(self, routes: dict[str, object], declared: FrontendContract):
        self.routes = dict(routes)
        self.declared = declared
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
        profiles = {}
        for language in languages:
            profile = dict(self.declared.languages[language])
            frontend = self.frontend_for(language)
            profile["engine_version"] = frontend.version()
            if isinstance(frontend, OpenJTalkFrontend):
                profile["dictionary"] = frontend.dictionary_id()
            elif isinstance(frontend, PiperPlusFrontend):
                profile["resource"] = frontend.resource_id()
            profiles[language] = profile
        return FrontendContract(provider="language-router", languages=profiles)


def frontend_from_config(config: dict | None = None, *, languages=None,
                         language_registry: dict | None = None) -> FrontendRouter:
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
        spec = registry[language]
        if spec.frontend_provider == "espeak-ng":
            if espeak is None:
                espeak = EspeakFrontend(
                    executable=config.get("executable"),
                    voices=voices,
                    allow_language_switches=not bool(config.get("strict_language_switches", True)),
                )
            routes[language] = espeak
        elif spec.frontend_provider == "openjtalk":
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
        elif spec.frontend_provider == "piper-plus-g2p":
            piper_config = config.get("piper_plus", {})
            piper_plus[language] = PiperPlusFrontend(
                language,
                resource_root=piper_config.get("resource_root"),
                auto_download_resources=bool(
                    piper_config.get("auto_download_resources", True)
                ),
            )
            routes[language] = piper_plus[language]
        else:  # LanguageSpec validation should make this unreachable.
            raise ValueError(f"unsupported frontend provider: {spec.frontend_provider}")
    return FrontendRouter(routes, declared)


def frontend_from_contract(contract: FrontendContract, config: dict | None = None) -> FrontendRouter:
    """Recreate a runtime router from an exported frontend contract."""
    config = dict(config or {})
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
    registry = {
        language: {
            "name": language,
            "teacher": None,
            "frontend": {
                key: value for key, value in profile.items()
                if key != "engine_version"
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
    resolved_config["provider"] = "language-router"
    resolved_config["voices"] = {**voices, **config.get("voices", {})}
    return frontend_from_config(
        resolved_config,
        languages=tuple(contract.languages),
        language_registry=registry,
    )
