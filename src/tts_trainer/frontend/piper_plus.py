from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import io
import os
from pathlib import Path

from .resources import ensure_korean_cmudict


SUPPORTED_LANGUAGES = {"zh", "ko"}


class PiperPlusFrontend:
    """Production Mandarin/Korean G2P using Piper Plus token semantics."""

    def __init__(self, language: str, *, resource_root: Path | None = None,
                 auto_download_resources: bool = True):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Piper Plus frontend does not support {language!r}")
        self.language = language
        self.resource_root = Path(resource_root).expanduser().resolve() if resource_root else None
        self.auto_download_resources = auto_download_resources
        self._phonemizer = None

    def _prepare_korean(self) -> None:
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
        if self._phonemizer is not None:
            return self._phonemizer
        if importlib.util.find_spec("piper_plus_g2p") is None:
            raise RuntimeError(
                "Mandarin/Korean G2P requires Piper Plus. "
                "Install: pip install 'tts-trainer[asian]'"
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
        if language != self.language:
            raise ValueError(
                f"Piper Plus {self.language} frontend cannot phonemize {language!r}"
            )
        with contextlib.redirect_stdout(io.StringIO()):
            return tuple(self._load().phonemize(text))

    def version(self) -> str:
        versions = [f"piper-plus-g2p {importlib.metadata.version('piper-plus-g2p')}"]
        dependency = "pypinyin" if self.language == "zh" else "g2pk2"
        versions.append(f"{dependency} {importlib.metadata.version(dependency)}")
        if self.language == "ko":
            versions.append(f"python-mecab-ko {importlib.metadata.version('python-mecab-ko')}")
        return "; ".join(versions)

    def resource_id(self) -> str:
        if self.language == "ko":
            return "nltk-cmudict-v1"
        return "pypinyin-rules-v1"
