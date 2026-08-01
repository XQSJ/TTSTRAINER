from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# Boundaries are intentionally language-agnostic.  The frontend still decides
# how every resulting piece is normalized and phonemized for its language.
_STRONG_BOUNDARIES = frozenset("。！？!?；;\n")
_CLAUSE_BOUNDARIES = frozenset("，,、：:")
_CLOSING_MARKS = frozenset("\"'”’）》】」』〕〉）]")


@dataclass(frozen=True)
class TextChunk:
    """One independently phonemized inference request and its following pause."""

    text: str
    units: tuple[str, ...]
    pause_kind: str


def _is_period_boundary(text: str, index: int) -> bool:
    """Treat a period as punctuation, except inside a decimal number."""
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    return not (previous.isdigit() and following.isdigit())


def _boundary_kind(text: str) -> str:
    meaningful = text.rstrip()
    while meaningful and meaningful[-1] in _CLOSING_MARKS:
        meaningful = meaningful[:-1].rstrip()
    if not meaningful:
        return "none"
    mark = meaningful[-1]
    if mark in _STRONG_BOUNDARIES or mark == ".":
        return "sentence"
    if mark in _CLAUSE_BOUNDARIES:
        return "clause"
    return "none"


def _punctuation_pieces(text: str) -> list[str]:
    """Keep original whitespace while exposing natural sentence boundaries."""
    pieces: list[str] = []
    start = 0
    for index, character in enumerate(text):
        boundary = character in _STRONG_BOUNDARIES or character in _CLAUSE_BOUNDARIES
        boundary = boundary or (character == "." and _is_period_boundary(text, index))
        if boundary:
            pieces.append(text[start:index + 1])
            start = index + 1
    if start < len(text):
        pieces.append(text[start:])
    return [piece for piece in pieces if piece.strip()]


def _fallback_pieces(text: str) -> list[str]:
    """Prefer word boundaries; CJK or a single long word falls back to codepoints."""
    words: list[str] = []
    start = 0
    inside_word = False
    for index, character in enumerate(text):
        if not character.isspace() and not inside_word:
            start = index
            inside_word = True
        elif character.isspace() and inside_word:
            words.append(text[start:index + 1])
            inside_word = False
    if inside_word:
        words.append(text[start:])
    if len(words) > 1:
        return words
    return list(text)


def chunk_text(
    text: str,
    language: str,
    phonemize: Callable[[str, str], tuple[str, ...]],
    *,
    max_phoneme_tokens: int = 90,
) -> list[TextChunk]:
    """Split text by punctuation and enforce a phoneme-token inference budget.

    Text length is a poor proxy across languages: one Chinese character or one
    accented Latin letter may expand into several frontend units.  Each proposed
    chunk is therefore measured with the exact frontend used by the model.
    """
    normalized = text.strip()
    if not normalized:
        raise ValueError("text must not be empty")
    if max_phoneme_tokens < 8:
        raise ValueError("max_phoneme_tokens must be at least 8")

    cache: dict[str, tuple[str, ...]] = {}

    def units_for(value: str) -> tuple[str, ...]:
        key = value.strip()
        if key not in cache:
            cache[key] = tuple(phonemize(key, language))
        return cache[key]

    def fit_oversized(value: str) -> list[str]:
        fitted: list[str] = []
        if any(character.isspace() for character in value.strip()):
            pending = _fallback_pieces(value)
            current = ""
            for atom in pending:
                candidate = current + atom
                if current and len(units_for(candidate)) > max_phoneme_tokens:
                    fitted.append(current)
                    current = atom
                else:
                    current = candidate
                # One word can still exceed the budget. Fall through to the
                # codepoint splitter instead of emitting an oversized request.
                if current and len(units_for(current)) > max_phoneme_tokens:
                    fitted.extend(fit_codepoints(current))
                    current = ""
            if current.strip():
                fitted.append(current)
        else:
            fitted.extend(fit_codepoints(value))
        # Do not emit a punctuation-only request when a CJK sentence happens
        # to fill the budget exactly before its final mark.  Move enough of
        # the previous text into the final piece to keep both requests useful.
        if len(fitted) > 1 and not any(
            character.isalnum() for character in fitted[-1]
        ):
            tail = fitted.pop()
            previous = fitted.pop()
            target_tail_tokens = max(2, max_phoneme_tokens // 3)
            while previous and len(units_for(tail)) < target_tail_tokens:
                tail = previous[-1] + tail
                previous = previous[:-1]
            if previous.strip():
                fitted.append(previous)
            fitted.append(tail)
        return fitted

    def fit_codepoints(value: str) -> list[str]:
        """Binary-search bounded prefixes to avoid one G2P process per character."""
        fitted: list[str] = []
        remaining = value
        while remaining:
            if len(units_for(remaining)) <= max_phoneme_tokens:
                fitted.append(remaining)
                break
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if len(units_for(remaining[:middle])) <= max_phoneme_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            # A single codepoint may expand beyond the budget. Keep it so the
            # caller never loses text, even though this one chunk is oversized.
            split = max(best, 1)
            fitted.append(remaining[:split])
            remaining = remaining[split:]
        return fitted

    bounded: list[str] = []
    for piece in _punctuation_pieces(normalized):
        if len(units_for(piece)) <= max_phoneme_tokens:
            bounded.append(piece)
        else:
            bounded.extend(fit_oversized(piece))

    merged: list[str] = []
    current = ""
    for piece in bounded:
        candidate = current + piece
        if current and len(units_for(candidate)) > max_phoneme_tokens:
            merged.append(current.strip())
            current = piece
        else:
            current = candidate
    if current.strip():
        merged.append(current.strip())

    result = [
        TextChunk(value, units_for(value), _boundary_kind(value))
        for value in merged
    ]
    empty = [chunk.text for chunk in result if not chunk.units]
    if empty:
        raise ValueError(f"text produced no phonemes: {empty!r}")
    return result
