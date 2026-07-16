from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_TRAINING_LANGUAGES = ("zh", "en", "ja", "ko", "fr", "es", "pt")
QWEN_SUPPORTED_LANGUAGE_NAMES = {
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
}
LANGUAGE_CODE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    teacher_provider: str | None
    teacher_language: str | None
    frontend_provider: str
    frontend_profile: dict[str, str]
    smoke_text: str

    @property
    def frontend_voice(self) -> str:
        """Compatibility label used by logs and older callers."""
        return self.frontend_profile.get("voice", self.frontend_profile.get("dictionary", ""))

    @classmethod
    def from_dict(cls, code: str, raw: dict) -> "LanguageSpec":
        code = code.strip().lower()
        if not LANGUAGE_CODE.fullmatch(code):
            raise ValueError(f"invalid language code: {code!r}")
        teacher = raw.get("teacher")
        teacher_provider = None if teacher is None else str(teacher.get("provider", "qwen"))
        teacher_language = None if teacher is None else str(teacher.get("language", "")).strip()
        if teacher_provider == "qwen" and teacher_language not in QWEN_SUPPORTED_LANGUAGE_NAMES:
            raise ValueError(
                f"language {code}: unsupported Qwen language name {teacher_language!r}; "
                f"choose one of {sorted(QWEN_SUPPORTED_LANGUAGE_NAMES)!r} or set teacher to null"
            )
        if teacher_provider not in {None, "qwen"}:
            raise ValueError(f"language {code}: unsupported teacher provider {teacher_provider!r}")
        frontend = raw.get("frontend") or {}
        provider = str(frontend.get("provider", "espeak-ng"))
        if provider not in {"espeak-ng", "openjtalk", "piper-plus-g2p"}:
            raise ValueError(f"language {code}: unsupported frontend provider {provider!r}")
        profile = {
            str(key): str(value).strip()
            for key, value in frontend.items()
            if key != "provider" and value is not None and str(value).strip()
        }
        if provider == "espeak-ng" and not profile.get("voice"):
            raise ValueError(f"language {code}: frontend.voice must not be empty")
        if provider == "openjtalk":
            profile.setdefault("dictionary", "open_jtalk_dic_utf_8-1.11")
        if provider == "piper-plus-g2p":
            if code not in {"zh", "ko"}:
                raise ValueError(
                    f"language {code}: piper-plus-g2p is currently routed only for zh and ko"
                )
            defaults = {
                "zh": {"profile": "mandarin-ipa-v1", "resource": "pypinyin-rules-v1"},
                "ko": {"profile": "korean-ipa-v1", "resource": "nltk-cmudict-v1"},
            }[code]
            for key, value in defaults.items():
                profile.setdefault(key, value)
        smoke_text = str(raw.get("smoke_text", "")).strip()
        if not smoke_text:
            raise ValueError(f"language {code}: smoke_text must not be empty")
        return cls(
            code=code,
            name=str(raw.get("name", code)).strip() or code,
            teacher_provider=teacher_provider,
            teacher_language=teacher_language or None,
            frontend_provider=provider,
            frontend_profile=profile,
            smoke_text=smoke_text,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "teacher": None if self.teacher_provider is None else {
                "provider": self.teacher_provider,
                "language": self.teacher_language,
            },
            "frontend": {
                "provider": self.frontend_provider,
                **self.frontend_profile,
            },
            "smoke_text": self.smoke_text,
        }


BUILTIN_LANGUAGE_REGISTRY_RAW = {
    "zh": {"name": "Chinese", "teacher": {"provider": "qwen", "language": "Chinese"},
           "frontend": {"provider": "piper-plus-g2p", "profile": "mandarin-ipa-v1", "resource": "pypinyin-rules-v1"}, "smoke_text": "你好，欢迎使用语音系统。"},
    "en": {"name": "English", "teacher": {"provider": "qwen", "language": "English"},
           "frontend": {"provider": "espeak-ng", "voice": "en-us"}, "smoke_text": "Hello, welcome to the speech system."},
    "ja": {"name": "Japanese", "teacher": {"provider": "qwen", "language": "Japanese"},
           "frontend": {"provider": "openjtalk", "dictionary": "open_jtalk_dic_utf_8-1.11"}, "smoke_text": "こんにちは、音声システムへようこそ。"},
    "ko": {"name": "Korean", "teacher": {"provider": "qwen", "language": "Korean"},
           "frontend": {"provider": "piper-plus-g2p", "profile": "korean-ipa-v1", "resource": "nltk-cmudict-v1"}, "smoke_text": "안녕하세요. 음성 시스템에 오신 것을 환영합니다."},
    "de": {"name": "German", "teacher": {"provider": "qwen", "language": "German"},
           "frontend": {"provider": "espeak-ng", "voice": "de"}, "smoke_text": "Guten Morgen, willkommen beim Sprachsystem."},
    "fr": {"name": "French", "teacher": {"provider": "qwen", "language": "French"},
           "frontend": {"provider": "espeak-ng", "voice": "fr-fr"}, "smoke_text": "Bonjour et bienvenue dans le système vocal."},
    "ru": {"name": "Russian", "teacher": {"provider": "qwen", "language": "Russian"},
           "frontend": {"provider": "espeak-ng", "voice": "ru"}, "smoke_text": "Здравствуйте, добро пожаловать в голосовую систему."},
    "pt": {"name": "Portuguese", "teacher": {"provider": "qwen", "language": "Portuguese"},
           "frontend": {"provider": "espeak-ng", "voice": "pt-br"}, "smoke_text": "Olá, bem-vindo ao sistema de voz."},
    "es": {"name": "Spanish", "teacher": {"provider": "qwen", "language": "Spanish"},
           "frontend": {"provider": "espeak-ng", "voice": "es"}, "smoke_text": "Hola, bienvenido al sistema de voz."},
    "it": {"name": "Italian", "teacher": {"provider": "qwen", "language": "Italian"},
           "frontend": {"provider": "espeak-ng", "voice": "it"}, "smoke_text": "Buongiorno, benvenuto nel sistema vocale."},
}


def resolve_language_registry(overrides: dict | None = None) -> dict[str, LanguageSpec]:
    raw = {code: dict(value) for code, value in BUILTIN_LANGUAGE_REGISTRY_RAW.items()}
    for code, value in (overrides or {}).items():
        if value is None:
            raw.pop(code, None)
        else:
            previous = raw.get(code, {})
            merged = {**previous, **value}
            if "teacher" not in value and "teacher" in previous:
                merged["teacher"] = previous["teacher"]
            if "frontend" in previous or "frontend" in value:
                merged["frontend"] = {**previous.get("frontend", {}), **value.get("frontend", {})}
            raw[code] = merged
    return {code: LanguageSpec.from_dict(code, value) for code, value in raw.items()}


def language_specs_for(registry: dict[str, LanguageSpec], languages) -> dict[str, LanguageSpec]:
    missing = sorted(set(languages) - set(registry))
    if missing:
        raise ValueError(
            "languages are not registered: " + ", ".join(missing)
            + "; add entries under language_registry or choose a built-in language"
        )
    return {code: registry[code] for code in languages}
