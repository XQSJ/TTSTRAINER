"""前端一致性：冻结代表性样本的音素与 token ID，保证训练期与 Android 端一致。 / Frontend conformance: freezes phonemes and token IDs of representative cases so training matches the Android side."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..manifest import Item
from ..text import Vocabulary


FRONTEND_CONFORMANCE_FORMAT = 1  # 一致性文件格式版本 / conformance file format version


def build_frontend_conformance(items: list[Item], vocabulary: Vocabulary,
                               language_map: dict[str, int],
                               *, cases_per_language: int = 3,
                               piper_compatible: bool = False) -> dict:
    """冻结代表性 文本->音素->token ID 样本，供移动端 QA 使用。 / Freeze representative text -> phoneme -> token-ID cases for mobile QA."""
    if cases_per_language < 1:
        raise ValueError("cases_per_language must be at least 1")
    counts = Counter()
    cases = []
    for item in items:
        if not item.phonemes or counts[item.language] >= cases_per_language:
            continue
        cases.append({
            "language": item.language,
            "language_id": language_map[item.language],
            "text": item.text,
            "phonemes": list(item.phonemes),
            "token_ids": vocabulary.encode_item(
                item, piper_compatible=piper_compatible,
            ),
        })
        counts[item.language] += 1
    missing = sorted(set(language_map) - set(counts))
    if missing:
        # 每个训练语言都必须有冻结样本，否则移动端无法验证该语言。
        # Every training language needs frozen cases, else mobile cannot verify it.
        raise ValueError(
            "cannot build frontend conformance without frozen phonemes for: "
            + ", ".join(missing)
        )
    return {
        "format": FRONTEND_CONFORMANCE_FORMAT,
        "cases_per_language": cases_per_language,
        "languages": list(language_map),
        "piper_compatible": piper_compatible,
        "cases": cases,
    }


def save_frontend_conformance(conformance: dict, path: str | Path) -> Path:
    """把一致性数据写入 JSON 文件。 / Write the conformance data to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(conformance, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_frontend_conformance(path: str | Path) -> dict:
    """读取并校验一致性 JSON 文件。 / Load and validate a conformance JSON file."""
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(result.get("format", 0)) != FRONTEND_CONFORMANCE_FORMAT:
        raise ValueError("unsupported frontend conformance format")
    if not isinstance(result.get("cases"), list) or not result["cases"]:
        raise ValueError("frontend conformance contains no cases")
    return result


def verify_frontend_conformance(conformance: dict, frontend,
                                vocabulary: Vocabulary) -> list[dict]:
    """返回不一致记录；空列表表示前端完全一致。 / Return mismatch records; an empty list means exact frontend parity."""
    mismatches = []
    for case in conformance["cases"]:
        actual_phonemes = frontend.phonemize(case["text"], case["language"])
        actual_ids = vocabulary.encode(
            case["text"], case["language"], actual_phonemes,
            piper_compatible=bool(conformance.get("piper_compatible", False)),
        )
        expected_phonemes = tuple(case["phonemes"])
        expected_ids = list(case["token_ids"])
        if actual_phonemes != expected_phonemes or actual_ids != expected_ids:
            mismatches.append({
                "language": case["language"],
                "text": case["text"],
                "expected_phonemes": list(expected_phonemes),
                "actual_phonemes": list(actual_phonemes),
                "expected_token_ids": expected_ids,
                "actual_token_ids": actual_ids,
            })
    return mismatches
