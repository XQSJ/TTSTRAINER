from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path

from ..text import normalize
from .resources import OPENJTALK_DICTIONARY_NAME, ensure_openjtalk_dictionary


class OpenJTalkFrontend:
    """Japanese text frontend backed by pyopenjtalk/Open JTalk.

    The dependency is loaded lazily so non-Japanese experiments do not need it.
    `g2p(..., join=False)` returns stable phone units such as ``ch`` and ``N``;
    these are intentionally not split into Unicode codepoints.
    """

    def __init__(self, *, user_dictionary: str | Path | None = None,
                 dictionary_root: str | Path | None = None,
                 auto_download_dictionary: bool = True):
        self.user_dictionary = Path(user_dictionary).expanduser().resolve() if user_dictionary else None
        self.dictionary_root = Path(dictionary_root).expanduser().resolve() if dictionary_root else None
        self.auto_download_dictionary = auto_download_dictionary
        self._module = None
        self._dictionary_applied = False

    def _load(self):
        if self._module is None:
            if importlib.util.find_spec("pyopenjtalk") is None:
                raise RuntimeError(
                    "Japanese G2P requires pyopenjtalk; install with "
                    "pip install -e '.[japanese]' (CMake and a C/C++ compiler may be required)"
                )
            dictionary = ensure_openjtalk_dictionary(
                self.dictionary_root, allow_download=self.auto_download_dictionary,
            )
            # pyopenjtalk otherwise downloads into site-packages on first use.
            # Set the project-local directory before importing the package.
            import os
            os.environ["OPEN_JTALK_DICT_DIR"] = str(dictionary)
            try:
                self._module = importlib.import_module("pyopenjtalk")
            except ImportError as exc:
                raise RuntimeError(
                    "Japanese G2P requires pyopenjtalk; install with "
                    "pip install -e '.[japanese]' (CMake and a C/C++ compiler may be required)"
                ) from exc
            self._module.OPEN_JTALK_DICT_DIR = str(dictionary).encode("utf-8")
        if self.user_dictionary and not self._dictionary_applied:
            if not self.user_dictionary.is_file():
                raise FileNotFoundError(f"Open JTalk user dictionary not found: {self.user_dictionary}")
            self._module.update_global_jtalk_with_user_dict(str(self.user_dictionary))
            self._dictionary_applied = True
        return self._module

    def version(self) -> str:
        self._load()
        try:
            version = importlib.metadata.version("pyopenjtalk")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(self._module, "__version__", "unknown")
        return f"pyopenjtalk {version}"

    def dictionary_id(self) -> str:
        if self.user_dictionary is None:
            return OPENJTALK_DICTIONARY_NAME
        digest = hashlib.sha256(self.user_dictionary.read_bytes()).hexdigest()[:16]
        return f"user:{self.user_dictionary.name}:sha256:{digest}"

    def phonemize(self, text: str, language: str) -> tuple[str, ...]:
        if language != "ja":
            raise ValueError(f"Open JTalk only supports ja, got {language!r}")
        module = self._load()
        phones = tuple(str(phone) for phone in module.g2p(
            normalize(text, language), kana=False, join=False,
        ) if str(phone).strip())
        if not phones:
            raise ValueError(f"Open JTalk produced no phonemes for {text!r}")
        return phones
