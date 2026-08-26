"""文本规范化与音素词表。 / Text normalization and phoneme vocabulary."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .manifest import Item

# 特殊符号约定与 Piper 保持一致，保证导出的词表可直接部署。
PAD, BOS, EOS, SPACE, UNK = "_", "^", "$", " ", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, SPACE, UNK]


def normalize(text: str, language: str) -> str:
    """NFKC 规范化并压缩空白。 / NFKC-normalize and collapse whitespace."""
    if not language or not isinstance(language, str):
        raise ValueError("language must be a non-empty string")
    text = unicodedata.normalize("NFKC", text).strip()
    return re.sub(r"\s+", " ", text)


class Vocabulary:
    """文本/音素到 id 的编码词表。 / Token-to-id encoding vocabulary."""

    def __init__(self, tokens: list[str]):
        # Piper 词表前四位固定为 _ ^ $ 空格，导出时必须满足。 / Piper vocab requires the fixed prefix _, ^, $, space.
        if tokens[:4] != [PAD, BOS, EOS, SPACE]:
            raise ValueError("Piper vocabulary must start with _, ^, $, and space")
        self.tokens = tokens
        self.ids = {token: index for index, token in enumerate(tokens)}

    @classmethod
    def build(cls, items: list[Item]) -> "Vocabulary":
        """从样本集合构建词表，特殊符号排最前。 / Build a vocabulary from items with special tokens first."""
        units = sorted({unit for item in items for unit in cls.units_for_item(item)} - set(SPECIAL_TOKENS))
        return cls([*SPECIAL_TOKENS, *units])

    @staticmethod
    def units_for_item(item: Item) -> tuple[str, ...]:
        # 冻结音素优先，缺失时回退到规范化文本。 / Prefer frozen phonemes; fall back to normalized text.
        return item.phonemes or tuple(normalize(item.text, item.language))

    def encode(self, text: str, language: str, phonemes: tuple[str, ...] | None = None,
               *, piper_compatible: bool = False) -> list[int]:
        """编码为 id 序列，可选 Piper 兼容的 blank 交错。 / Encode to ids, optionally interleaving blanks for Piper."""
        units = phonemes or tuple(normalize(text, language))
        encoded = [self.ids.get(unit, self.ids[UNK]) for unit in units]
        if piper_compatible:
            encoded = [
                value
                for token_id in encoded
                for value in (token_id, self.ids[PAD])
            ]
            # Piper 会像普通符号一样给 BOS 加 blank；训练与部署必须一致。
            # Piper frames BOS like any other symbol, so training and
            # deployment must agree on this leading blank.
            return [
                self.ids[BOS], self.ids[PAD], *encoded, self.ids[EOS],
            ]
        return [self.ids[BOS], *encoded, self.ids[EOS]]

    def encode_item(self, item: Item, *, piper_compatible: bool = False) -> list[int]:
        """编码单个样本。 / Encode a single item."""
        return self.encode(
            item.text, item.language, item.phonemes,
            piper_compatible=piper_compatible,
        )

    def save(self, path: str | Path) -> None:
        """把词表写成 JSON。 / Persist the vocabulary as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """从 JSON 恢复词表。 / Restore the vocabulary from JSON."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["tokens"])
