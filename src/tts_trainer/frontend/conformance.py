from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..manifest import Item
from ..text import Vocabulary


FRONTEND_CONFORMANCE_FORMAT = 1


def build_frontend_conformance(items: list[Item], vocabulary: Vocabulary,
                               language_map: dict[str, int],
                               *, cases_per_language: int = 3) -> dict:
    """Freeze representative text -> phoneme -> token-ID cases for mobile QA."""
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
            "token_ids": vocabulary.encode_item(item),
        })
        counts[item.language] += 1
    missing = sorted(set(language_map) - set(counts))
    if missing:
        raise ValueError(
            "cannot build frontend conformance without frozen phonemes for: "
            + ", ".join(missing)
        )
    return {
        "format": FRONTEND_CONFORMANCE_FORMAT,
        "cases_per_language": cases_per_language,
        "languages": list(language_map),
        "cases": cases,
    }


def save_frontend_conformance(conformance: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(conformance, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_frontend_conformance(path: str | Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(result.get("format", 0)) != FRONTEND_CONFORMANCE_FORMAT:
        raise ValueError("unsupported frontend conformance format")
    if not isinstance(result.get("cases"), list) or not result["cases"]:
        raise ValueError("frontend conformance contains no cases")
    return result


def verify_frontend_conformance(conformance: dict, frontend,
                                vocabulary: Vocabulary) -> list[dict]:
    """Return mismatch records; an empty list means exact frontend parity."""
    mismatches = []
    for case in conformance["cases"]:
        actual_phonemes = frontend.phonemize(case["text"], case["language"])
        actual_ids = vocabulary.encode(case["text"], case["language"], actual_phonemes)
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
