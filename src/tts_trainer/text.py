from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .constants import LANGUAGES
from .manifest import Item

PAD, BOS, EOS, SPACE, UNK = "_", "^", "$", " ", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, SPACE, UNK]


def normalize(text: str, language: str) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    text = unicodedata.normalize("NFKC", text).strip()
    return re.sub(r"\s+", " ", text)


class Vocabulary:
    def __init__(self, tokens: list[str]):
        if tokens[:4] != [PAD, BOS, EOS, SPACE]:
            raise ValueError("Piper vocabulary must start with _, ^, $, and space")
        self.tokens = tokens
        self.ids = {token: index for index, token in enumerate(tokens)}

    @classmethod
    def build(cls, items: list[Item]) -> "Vocabulary":
        units = sorted({unit for item in items for unit in cls.units_for_item(item)} - set(SPECIAL_TOKENS))
        return cls([*SPECIAL_TOKENS, *units])

    @staticmethod
    def units_for_item(item: Item) -> tuple[str, ...]:
        return item.phonemes or tuple(normalize(item.text, item.language))

    def encode(self, text: str, language: str, phonemes: tuple[str, ...] | None = None) -> list[int]:
        units = phonemes or tuple(normalize(text, language))
        return [self.ids[BOS], *(self.ids.get(unit, self.ids[UNK]) for unit in units), self.ids[EOS]]

    def encode_item(self, item: Item) -> list[int]:
        return self.encode(item.text, item.language, item.phonemes)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["tokens"])
