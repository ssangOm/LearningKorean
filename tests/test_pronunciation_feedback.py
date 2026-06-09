from src.pronunciation.feedback import build_feedback
from src.pronunciation.similarity import SegmentScore


def test_feedback_reports_speed_category_when_duration_differs():
    feedback = build_feedback(
        sentence_text="오늘 공기가 맑아요",
        score=82.5,
        reference_duration=2.0,
        user_duration=2.6,
        segments=[
            SegmentScore(label="초반", score=91.0),
            SegmentScore(label="중반", score=63.0),
            SegmentScore(label="후반", score=85.0),
        ],
    )

    assert feedback.score == 82.5
    assert feedback.weak_region == "속도"
    assert "기준보다 깁니다" in feedback.messages[0]
    assert any("중반" in message for message in feedback.messages)


def test_feedback_reports_pronunciation_category_when_duration_is_similar():
    feedback = build_feedback(
        sentence_text="오늘 공기가 맑아요",
        score=82.5,
        reference_duration=2.0,
        user_duration=2.05,
        segments=[
            SegmentScore(label="초반", score=91.0),
            SegmentScore(label="중반", score=63.0),
            SegmentScore(label="후반", score=85.0),
        ],
    )

    assert feedback.weak_region == "발음"
