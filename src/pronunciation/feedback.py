from dataclasses import dataclass

from src.pronunciation.similarity import SegmentScore


@dataclass(frozen=True)
class FeedbackResult:
    sentence_text: str
    score: float
    weak_region: str
    duration_delta: float
    messages: tuple[str, ...]


def build_feedback(
    sentence_text: str,
    score: float,
    reference_duration: float,
    user_duration: float,
    segments: list[SegmentScore] | tuple[SegmentScore, ...],
) -> FeedbackResult:
    duration_delta = float(user_duration - reference_duration)
    weak_segment = min(segments, key=lambda segment: segment.score) if segments else None
    messages = [
        _duration_message(reference_duration, user_duration),
        _score_message(score),
    ]

    if weak_segment is not None:
        messages.append(
            f"{weak_segment.label} 구간의 유사도가 가장 낮습니다. 이 부분을 기준 발음과 다시 비교해 보세요."
        )

    return FeedbackResult(
        sentence_text=sentence_text,
        score=round(float(score), 1),
        weak_region=_weak_category(reference_duration, user_duration, score, weak_segment),
        duration_delta=round(duration_delta, 3),
        messages=tuple(messages),
    )


def _weak_category(
    reference_duration: float,
    user_duration: float,
    score: float,
    weak_segment: SegmentScore | None,
) -> str:
    if reference_duration > 0:
        duration_ratio = (user_duration - reference_duration) / reference_duration
        if abs(duration_ratio) > 0.15:
            return "속도"
    if weak_segment is not None and score < 95:
        return "발음"
    return "없음"


def _duration_message(reference_duration: float, user_duration: float) -> str:
    if reference_duration <= 0:
        return "기준 발화 길이를 계산할 수 없습니다."

    ratio = (user_duration - reference_duration) / reference_duration
    if ratio > 0.15:
        return "전체 발화 길이는 기준보다 깁니다."
    if ratio < -0.15:
        return "전체 발화 길이는 기준보다 짧습니다."
    return "전체 발화 길이는 기준과 비슷합니다."


def _score_message(score: float) -> str:
    if score >= 90:
        return "기준 발음과 전반적으로 매우 유사합니다."
    if score >= 75:
        return "기준 발음과 대체로 유사하지만 일부 구간을 다듬을 수 있습니다."
    if score >= 55:
        return "기준 발음과 차이가 있어 속도와 모음 명료도를 다시 확인하는 것이 좋습니다."
    return "기준 발음과 차이가 큽니다. 문장을 천천히 다시 읽어 보세요."
