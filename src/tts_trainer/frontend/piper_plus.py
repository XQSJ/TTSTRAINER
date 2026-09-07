"""Piper Plus 商用前端（MIT）：无 GPL 依赖的多语言 G2P，用于商业部署链路。 / Piper Plus commercial frontend (MIT): GPL-free multilingual G2P for commercial deployment."""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import io
import os
from pathlib import Path

from .resources import ensure_korean_cmudict


# 每个语言一个前端实例；此集合为 Piper Plus 后端实际支持的语言范围。
# One frontend instance per language; the set the Piper Plus backend actually covers.
SUPPORTED_LANGUAGES = {"ja", "en", "zh", "ko", "es", "fr", "pt", "sv"}


class PiperPlusFrontend:
    """使用冻结 Piper Plus token 语义的无 GPL 多语言 G2P。 / GPL-free multilingual G2P using frozen Piper Plus token semantics."""

    def __init__(self, language: str, *, resource_root: Path | None = None,
                 auto_download_resources: bool = True):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Piper Plus frontend does not support {language!r}")
        self.language = language
        self.resource_root = Path(resource_root).expanduser().resolve() if resource_root else None
        self.auto_download_resources = auto_download_resources
        self._phonemizer = None

    def _prepare_korean(self) -> None:
        """为韩语准备 g2pk2/mecab/NLTK 数据环境。 / Prepare the g2pk2/mecab/NLTK data environment for Korean."""
        if importlib.util.find_spec("g2pk2") is None:
            raise RuntimeError(
                "Korean G2P requires the asian dependencies. "
                "Install: pip install 'tts-trainer[asian]'"
            )
        if importlib.util.find_spec("mecab") is None:
            raise RuntimeError(
                "Korean G2P requires python-mecab-ko. "
                "Install: pip install 'tts-trainer[asian]'"
            )
        data_root = ensure_korean_cmudict(
            self.resource_root, allow_download=self.auto_download_resources,
        )
        value = str(data_root)
        # 把项目本地的 NLTK 数据目录挂到查找路径最前，避免污染全局安装。
        # Put the project-local NLTK data dir first so global installs stay untouched.
        existing = os.environ.get("NLTK_DATA")
        paths = existing.split(os.pathsep) if existing else []
        if value not in paths:
            os.environ["NLTK_DATA"] = os.pathsep.join([value, *paths])
        # NLTK computes its path list at import time. Update it when another
        # dependency imported NLTK before this frontend was initialized.
        if importlib.util.find_spec("nltk") is not None:
            nltk = importlib.import_module("nltk")
            if value not in nltk.data.path:
                nltk.data.path.insert(0, value)

    def _load(self):
        """惰性加载 phonemizer，缺失商业依赖时给出安装指引。 / Lazily load the phonemizer with install hints for missing commercial deps."""
        if self._phonemizer is not None:
            return self._phonemizer
        if importlib.util.find_spec("piper_plus_g2p") is None:
            raise RuntimeError(
                f"{self.language} G2P requires Piper Plus. "
                "Install: pip install 'tts-trainer[commercial]'"
            )
        if self.language == "ko":
            self._prepare_korean()
        module = importlib.import_module("piper_plus_g2p")
        # g2pk2 prints dependency status from its constructor; normal training
        # logs should contain actionable project logs only.
        with contextlib.redirect_stdout(io.StringIO()):
            self._phonemizer = module.get_phonemizer(self.language)
        return self._phonemizer

    def phonemize(self, text: str, language: str) -> tuple[str, ...]:
        """音素化文本；每个实例只服务自己的语言。 / Phonemize text; each instance serves only its own language."""
        if language != self.language:
            raise ValueError(
                f"Piper Plus {self.language} frontend cannot phonemize {language!r}"
            )
        with contextlib.redirect_stdout(io.StringIO()):
            return tuple(self._load().phonemize(text))

    def version(self) -> str:
        """汇总本语言所有后端依赖的版本号，供契约冻结。 / Aggregate every backend dependency version for contract freezing."""
        def installed_version(distribution: str) -> str:
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Piper Plus {self.language} backend is incomplete: missing "
                    f"{distribution}; install: pip install 'tts-trainer[commercial]'"
                ) from exc

        versions = [f"piper-plus-g2p {installed_version('piper-plus-g2p')}"]
        # 各语言实际参与 G2P 的核心依赖，版本变化都可能改变音素输出。
        # The per-language G2P dependencies whose versions can change phoneme output.
        dependencies = {
            "ja": ("pyopenjtalk-plus",),
            "en": ("g2p-en",),
            "zh": ("pypinyin",),
            "ko": ("g2pk2",),
        }.get(self.language, ())
        for dependency in dependencies:
            versions.append(f"{dependency} {installed_version(dependency)}")
        if self.language == "ko":
            versions.append(f"python-mecab-ko {installed_version('python-mecab-ko')}")
        return "; ".join(versions)

    def resource_id(self) -> str:
        """返回各语言外部资源指纹，供契约与资源包引用。 / Return the per-language external resource fingerprint for contracts and packs."""
        # 必须与 languages.py 中 piper-plus-g2p 的注册表 resource 声明逐字一致，
        # 否则 frontend.lock.json 与训练配置的 declaration_key 永不相等。
        # Must match the registry's declared resource in languages.py verbatim,
        # or the lock file's declaration_key never equals the configured one.
        return {
            "ko": "nltk-cmudict-v1",
            "zh": "pypinyin-rules-v1",
            "ja": "openjtalk-dictionary-v1",
            "en": "g2p-en-v1",
            "es": "piper-plus-rules-v1",
            "fr": "piper-plus-rules-v1",
            "pt": "piper-plus-rules-v1",
            "sv": "piper-plus-rules-v1",
        }.get(self.language, "piper-plus-rules-v1")
