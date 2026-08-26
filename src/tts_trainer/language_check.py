"""语言支持检查：教师映射与前端冒烟。 / Language support checks: teacher mapping and frontend smoke test."""
from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

from .experiments import ExperimentLayout
from .frontend import frontend_from_config


# eSpeak 对未知 CJK 字符的英文念白音素签名。 / Phone signatures of eSpeak's English reading of unknown CJK codepoints.
_UNICODE_FALLBACK_PHONES = {
    "chinese-letter": "tʃaɪnizletə",
    "japanese-letter": "dʒapənizletə",
}


def detect_unicode_name_fallback(language: str, phonemes) -> str | None:
    """检测 eSpeak 把不支持的中日韩字符读成 Unicode 英文名称。 / Detect eSpeak spelling an unsupported CJK codepoint by Unicode name.

    eSpeak 的日语嗓音会把未知汉字念成英文短语 "Chinese letter"。该输出虽然是合法 IPA，
    但仅靠供应商与语言切换检查无法发现；若拿去训练，会让许多不同字符坍缩到同一音素序列，
    破坏文本到先验的映射，必须在训练前失败。

    eSpeak's Japanese voice can pronounce unknown kanji as the English phrase
    "Chinese letter". That output is syntactically valid IPA, so provider and
    language-switch checks alone cannot catch it. Training on it destroys the
    text-to-prior mapping because many different characters collapse to the
    same phone sequence.
    """
    if language not in {"zh", "ja", "ko"}:
        return None
    compact = "".join(str(phone) for phone in phonemes)
    # 去掉变音符号与韵律标记再做子串匹配。 / Strip diacritics and prosody marks before substring matching.
    compact = "".join(
        character for character in unicodedata.normalize("NFD", compact)
        if not unicodedata.combining(character)
        and character not in {" ", "|", "ˈ", "ˌ", "ː"}
    ).lower()
    for name, signature in _UNICODE_FALLBACK_PHONES.items():
        if signature in compact:
            return name
    return None


@dataclass(frozen=True)
class LanguageStatus:
    """单语言的就绪检查结果。 / Readiness check result for one language."""
    code: str
    name: str
    selected: bool
    teacher: str
    teacher_ready: bool
    frontend: str
    voice: str
    frontend_version: str | None
    phoneme_preview: str | None
    ready: bool
    error: str | None

    def to_dict(self) -> dict:
        """序列化为普通字典。 / Serialize to a plain dict."""
        return asdict(self)


def check_language_support(raw: dict, layout: ExperimentLayout, codes=None,
                           *, run_smoke: bool = True,
                           require_teacher: bool | None = None) -> list[LanguageStatus]:
    """逐语言检查教师映射与前端音素化。 / Check teacher mapping and frontend phonemization per language."""
    requested = tuple(codes or layout.languages)
    unknown = sorted(set(requested) - set(layout.language_registry))
    if unknown:
        raise ValueError("languages are not selected or registered in this experiment: " + ", ".join(unknown))
    if run_smoke:
        frontend = frontend_from_config(
            raw.get("frontend"), languages=requested,
            language_registry=raw.get("language_registry"),
        )
        frontend_error = None
    else:
        frontend = None
        frontend_error = None
    generation_enabled = bool(raw.get("generation", {}).get("enabled", True)) \
        if require_teacher is None else require_teacher
    statuses = []
    for code in requested:
        spec = layout.language_registry[code]
        teacher_ready = not generation_enabled or (
            spec.teacher_provider == "qwen" and bool(spec.teacher_language)
        )
        teacher = "external-data" if not generation_enabled else (
            f"{spec.teacher_provider}:{spec.teacher_language}"
            if spec.teacher_provider and spec.teacher_language else "missing"
        )
        preview = None
        frontend_version = None
        error = frontend_error
        if error is None and run_smoke:
            # 冒烟：音素化样例文本并拦截 Unicode 名称回退念白。 / Smoke test: phonemize sample text and trap Unicode-name fallback speech.
            try:
                frontend_version = frontend.version_for(code)
                phonemes = frontend.phonemize(spec.smoke_text, code)
                fallback = detect_unicode_name_fallback(code, phonemes)
                if fallback:
                    raise ValueError(
                        f"{code} frontend pronounced text as {fallback}; "
                        "use the language-router frontend instead of accepting "
                        "Unicode-name fallback pronunciation"
                    )
                preview = " ".join(phonemes)[:100]
            except Exception as exc:
                error = str(exc)
        if not teacher_ready and error is None:
            error = "Qwen sample generation is enabled but this language has no Qwen teacher mapping"
        statuses.append(LanguageStatus(
            code=code,
            name=spec.name,
            selected=code in layout.languages,
            teacher=teacher,
            teacher_ready=teacher_ready,
            frontend=spec.frontend_provider,
            voice=frontend.voices[code] if frontend else spec.frontend_voice,
            frontend_version=frontend_version,
            phoneme_preview=preview,
            ready=teacher_ready and error is None,
            error=error,
        ))
    return statuses


def format_language_statuses(statuses: list[LanguageStatus]) -> str:
    """把检查结果渲染为对齐的文本表格。 / Render statuses as an aligned text table."""
    header = f"{'CODE':<7} {'TEACHER':<24} {'G2P PROFILE':<26} {'STATUS':<8} DETAILS"
    rows = [header, "-" * len(header)]
    for row in statuses:
        status = "ready" if row.ready else "failed"
        details = row.phoneme_preview or row.error or "declaration only"
        rows.append(f"{row.code:<7} {row.teacher:<24} {row.voice:<26} {status:<8} {details}")
    return "\n".join(rows)
