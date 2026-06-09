from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata


@dataclass(frozen=True)
class TextComparisonResult:
    score: float
    normalized_reference: str
    normalized_recognized: str
    missing_tokens: tuple[str, ...]
    extra_tokens: tuple[str, ...]
    is_match: bool
    weak_category: str
    messages: tuple[str, ...]


def normalize_korean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[^0-9a-z가-힣\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def compare_texts(reference_text: str, recognized_text: str) -> TextComparisonResult:
    normalized_reference = normalize_korean_text(reference_text)
    normalized_recognized = normalize_korean_text(recognized_text)
    compact_reference = normalized_reference.replace(" ", "")
    compact_recognized = normalized_recognized.replace(" ", "")

    if compact_reference and compact_reference == compact_recognized:
        return TextComparisonResult(
            score=100.0,
            normalized_reference=normalized_reference,
            normalized_recognized=normalized_recognized,
            missing_tokens=(),
            extra_tokens=(),
            is_match=True,
            weak_category="없음",
            messages=(
                f"인식된 문장: {recognized_text or '(인식 실패)'}",
                "문장 일치도는 100.0점입니다.",
                "정답 문장과 인식 문장이 전반적으로 잘 맞습니다.",
            ),
        )

    char_score = SequenceMatcher(None, normalized_reference, normalized_recognized).ratio() * 100
    compact_score = SequenceMatcher(None, compact_reference, compact_recognized).ratio() * 100
    ref_tokens = tuple(normalized_reference.split())
    rec_tokens = tuple(normalized_recognized.split())
    missing_tokens = _missing_tokens(ref_tokens, rec_tokens)
    extra_tokens = _missing_tokens(rec_tokens, ref_tokens)
    missing_tokens, extra_tokens = _cancel_joined_phrase_variants(missing_tokens, extra_tokens)
    weak_category = _weak_category(missing_tokens, extra_tokens)

    penalty = min(25.0, (len(missing_tokens) + len(extra_tokens)) * 4.0)
    score = max(0.0, min(100.0, max(char_score, compact_score) - penalty))
    score = round(score, 1)
    is_match = score >= 85.0

    messages = [
        f"인식된 문장: {recognized_text or '(인식 실패)'}",
        f"문장 일치도는 {score:.1f}점입니다.",
    ]
    if missing_tokens:
        messages.append(f"빠진 것으로 보이는 표현: {', '.join(missing_tokens)}")
    if extra_tokens:
        messages.append(f"다르게 인식된 표현: {', '.join(extra_tokens)}")
    if not missing_tokens and not extra_tokens:
        messages.append("정답 문장과 인식 문장이 전반적으로 잘 맞습니다.")

    return TextComparisonResult(
        score=score,
        normalized_reference=normalized_reference,
        normalized_recognized=normalized_recognized,
        missing_tokens=missing_tokens,
        extra_tokens=extra_tokens,
        is_match=is_match,
        weak_category=weak_category,
        messages=tuple(messages),
    )


def _missing_tokens(source_tokens: tuple[str, ...], target_tokens: tuple[str, ...]) -> tuple[str, ...]:
    remaining = list(target_tokens)
    missing = []
    for token in source_tokens:
        if token in remaining:
            remaining.remove(token)
        else:
            missing.append(token)
    return tuple(missing)


def _cancel_joined_phrase_variants(
    missing_tokens: tuple[str, ...],
    extra_tokens: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing = list(missing_tokens)
    extra = list(extra_tokens)

    changed = True
    while changed:
        changed = False
        for extra_token in list(extra):
            span = _matching_joined_span(missing, extra_token)
            if span is None:
                continue
            start, end = span
            del missing[start:end]
            extra.remove(extra_token)
            changed = True
            break

    return tuple(missing), tuple(extra)


def _matching_joined_span(tokens: list[str], candidate: str) -> tuple[int, int] | None:
    compact_candidate = candidate.replace(" ", "")
    if not compact_candidate:
        return None

    for start in range(len(tokens)):
        for end in range(start + 2, min(len(tokens), start + 5) + 1):
            compact_span = "".join(tokens[start:end])
            ratio = SequenceMatcher(None, compact_span, compact_candidate).ratio()
            if ratio >= 0.78:
                return start, end
    return None


def _weak_category(
    missing_tokens: tuple[str, ...],
    extra_tokens: tuple[str, ...],
) -> str:
    return "발음" if missing_tokens or extra_tokens else "없음"
