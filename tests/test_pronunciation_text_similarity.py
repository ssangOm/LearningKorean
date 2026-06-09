from src.pronunciation.text_similarity import compare_texts, normalize_korean_text


def test_normalize_korean_text_removes_punctuation_and_extra_spaces():
    assert normalize_korean_text("그럼,  지금 좀 보여 드리죠.") == "그럼 지금 좀 보여 드리죠"


def test_compare_texts_scores_exact_match_high():
    result = compare_texts("그럼 지금 좀 보여 드리죠.", "그럼 지금 좀 보여 드리죠")

    assert result.score == 100.0
    assert result.is_match
    assert result.weak_category == "없음"
    assert result.missing_tokens == ()
    assert result.extra_tokens == ()


def test_compare_texts_does_not_penalize_korean_spacing_only_difference():
    result = compare_texts("저 식당 음식이 정말 맛있나 봐요.", "저 식당 음식이 정말 맛있나봐요")

    assert result.score == 100.0
    assert result.is_match
    assert result.missing_tokens == ()
    assert result.extra_tokens == ()


def test_compare_texts_ignores_joined_phrases_from_fast_speech():
    result = compare_texts(
        "요즘은 주말에도 여권을 신청할 수 있는 곳이 있어요.",
        "요즘은 주말에도 여권을 신청할수있는 곳이 있어요",
    )

    assert result.score >= 95
    assert result.is_match
    assert result.missing_tokens == ()
    assert result.extra_tokens == ()


def test_compare_texts_does_not_split_fast_joined_phrase_into_many_errors():
    result = compare_texts("그럼 지금 좀 보여 드리죠.", "그럼 지금 좀 보여드려죠")

    assert result.score >= 85
    assert "보여" not in result.missing_tokens
    assert "드리죠" not in result.missing_tokens
    assert "보여드려죠" not in result.extra_tokens


def test_compare_texts_reports_missing_and_extra_tokens():
    result = compare_texts("그럼 지금 좀 보여 드리죠.", "그럼 지금 보여 드려요")

    assert 50 <= result.score < 100
    assert not result.is_match
    assert result.weak_category == "발음"
    assert "좀" in result.missing_tokens
    assert "드려요" in result.extra_tokens
    assert result.messages


def test_compare_texts_reports_pronunciation_category_for_sentence_ending_error():
    result = compare_texts("그럼 지금 좀 보여 드리죠.", "그럼 지금 좀 보여 드려요")

    assert result.weak_category == "발음"
