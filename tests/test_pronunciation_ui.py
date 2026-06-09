import subprocess
import sys
from pathlib import Path

import pandas as pd

from pronunciation_ui import (
    CONVERSATION_NEW_PROMPT_BUTTON_TEXT,
    DEFAULT_UI_ASR_MODEL_SIZE,
    DETAIL_PROMPT_MIN_HEIGHT,
    METRIC_CARD_MIN_HEIGHT,
    INITIAL_CONVERSATION_PROMPT_TEXT,
    PRIMARY_FONT_SIZE,
    RESULT_LINE_MIN_HEIGHT,
    SENTENCE_DETAIL_PAGE_INDEX,
    SENTENCE_LIST_FONT_SIZE,
    SENTENCE_LIST_MIN_WIDTH,
    SENTENCE_LIST_PAGE_INDEX,
    EvaluationWorker,
    build_stylesheet,
    build_recording_path,
    choose_reference_manifest,
    format_ai_coaching,
    format_conversation_feedback,
    format_result_summary,
    format_sentence_list_item_label,
    format_sentence_option_label,
    is_new_conversation_prompt,
    load_sentence_options,
    should_auto_start_conversation_prompt,
)
from src.pronunciation.feedback import FeedbackResult
from src.pronunciation.evaluate import EvaluationResult
from src.pronunciation.similarity import SimilarityResult


def test_load_sentence_options_reads_unique_manifest_sentences(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"sentence_id": "sentence_001", "text": "오늘 공기가 맑아요", "path": "a.wav"},
            {"sentence_id": "sentence_001", "text": "오늘 공기가 맑아요", "path": "b.wav"},
            {"sentence_id": "sentence_002", "text": "감기에 걸렸어요", "path": "c.wav"},
        ]
    ).to_csv(manifest_path, index=False)

    assert load_sentence_options(manifest_path) == [
        ("sentence_001", "오늘 공기가 맑아요"),
        ("sentence_002", "감기에 걸렸어요"),
    ]


def test_choose_reference_manifest_prefers_human_manifest(tmp_path):
    human_manifest = tmp_path / "human_reference_manifest.csv"
    fallback_manifest = tmp_path / "reference_manifest.csv"
    fallback_manifest.write_text("sentence_id,text,path\nsentence_001,old,a.wav\n", encoding="utf-8")
    human_manifest.write_text("sentence_id,text,path\naa0,new,b.wav\n", encoding="utf-8")

    assert choose_reference_manifest(human_manifest, fallback_manifest) == human_manifest


def test_choose_reference_manifest_falls_back_to_tts_manifest(tmp_path):
    human_manifest = tmp_path / "human_reference_manifest.csv"
    fallback_manifest = tmp_path / "reference_manifest.csv"
    fallback_manifest.write_text("sentence_id,text,path\nsentence_001,old,a.wav\n", encoding="utf-8")

    assert choose_reference_manifest(human_manifest, fallback_manifest) == fallback_manifest


def test_build_recording_path_uses_sentence_id_and_output_directory(tmp_path):
    path = build_recording_path("sentence_003", tmp_path)

    assert path == tmp_path / "ui_sentence_003.wav"


def test_format_result_summary_contains_scores_without_local_feedback():
    result = EvaluationResult(
        file="recorded.wav",
        target_sentence_id="sentence_001",
        closest_sentence_id="sentence_001",
        closest_text="오늘 공기가 맑아요",
        score=88.4,
        weak_region="발음",
        similarity=SimilarityResult(
            score=88.4,
            normalized_distance=1.2,
            path_length=10,
            reference_duration=3.0,
            user_duration=3.2,
            segments=[],
        ),
        feedback=FeedbackResult(
            sentence_text="오늘 공기가 맑아요",
            score=88.4,
            weak_region="발음",
            duration_delta=0.2,
            messages=("전체 발화 길이는 기준과 비슷합니다.", "기준 발음과 대체로 유사합니다."),
        ),
        recognized_text="오늘 공기가 맑아요",
        text_score=91.2,
    )

    summary = format_result_summary(result)

    assert "점수: 88.4 / 100" in summary
    assert "텍스트 일치도: 91.2 / 100" in summary
    assert "인식된 문장: 오늘 공기가 맑아요" in summary
    assert "개선 항목" not in summary
    assert "정답 문장: sentence_001 - 오늘 공기가 맑아요" in summary
    assert "피드백" not in summary
    assert "전체 발화 길이는 기준과 비슷합니다." not in summary


def test_format_result_summary_does_not_embed_ai_coaching():
    result = EvaluationResult(
        file="recorded.wav",
        target_sentence_id="sentence_001",
        closest_sentence_id="sentence_001",
        closest_text="답안을 확인하세요",
        score=91.0,
        weak_region="문장",
        similarity=SimilarityResult(
            score=91.0,
            normalized_distance=1.2,
            path_length=10,
            reference_duration=3.0,
            user_duration=3.2,
            segments=[],
        ),
        feedback=FeedbackResult(
            sentence_text="답안을 확인하세요",
            score=91.0,
            weak_region="문장",
            duration_delta=0.2,
            messages=("인식된 문장: 다반을 확인하세요",),
        ),
        recognized_text="다반을 확인하세요",
        text_score=91.0,
    )

    summary = format_result_summary(result, ai_coaching="표준 발음상 자연스러운 변화입니다.")

    assert "AI 코칭" not in summary
    assert "표준 발음상 자연스러운 변화입니다." not in summary


def test_format_ai_coaching_returns_fixed_panel_text():
    assert format_ai_coaching(None) == "AI 코칭 결과가 여기에 표시됩니다."
    assert format_ai_coaching("**표준 발음**상 자연스러운 변화입니다.") == "표준 발음상 자연스러운 변화입니다."


def test_format_sentence_option_label_keeps_combo_compact():
    assert format_sentence_option_label(0) == "문장 01"
    assert format_sentence_option_label(11) == "문장 12"


def test_format_sentence_list_item_label_shows_number_and_text():
    assert format_sentence_list_item_label(0, "aa0", "저 식당 음식이 정말 맛있나 봐요.") == (
        "01  저 식당 음식이 정말 맛있나 봐요."
    )


def test_evaluation_worker_uses_fixed_medium_asr_model(tmp_path):
    worker = EvaluationWorker("aa0", tmp_path / "out.wav", tmp_path / "manifest.csv")

    assert DEFAULT_UI_ASR_MODEL_SIZE == "medium"
    assert worker.asr_model_size == "medium"


def test_format_conversation_feedback_returns_fixed_panel_text():
    assert format_conversation_feedback(None) == "대화 평가가 여기에 표시됩니다."
    assert format_conversation_feedback("**문맥**에 맞는 답변입니다.") == "문맥에 맞는 답변입니다."
    assert "gemini" not in format_conversation_feedback("Gemini 응답입니다.").lower()


def test_conversation_ui_text_hides_provider_name():
    assert "Gemini" not in INITIAL_CONVERSATION_PROMPT_TEXT
    assert "Gemini" not in CONVERSATION_NEW_PROMPT_BUTTON_TEXT


def test_conversation_prompt_auto_starts_only_when_empty():
    assert should_auto_start_conversation_prompt(None, None) is True
    assert should_auto_start_conversation_prompt("오늘 뭐 했어요?", None) is False
    assert should_auto_start_conversation_prompt(None, object()) is False


def test_conversation_prompt_duplicate_check_ignores_spacing():
    assert is_new_conversation_prompt("오늘 뭐 했어요?", [" 오늘   뭐 했어요? "]) is False
    assert is_new_conversation_prompt("주말에 뭐 했어요?", ["오늘 뭐 했어요?"]) is True


def test_stylesheet_uses_larger_base_font_size():
    stylesheet = build_stylesheet()

    assert PRIMARY_FONT_SIZE >= 16
    assert f"font-size: {PRIMARY_FONT_SIZE}px;" in stylesheet
    assert "QPushButton" in stylesheet


def test_sentence_list_uses_wider_cells_with_smaller_font():
    stylesheet = build_stylesheet()

    assert SENTENCE_LIST_MIN_WIDTH >= 560
    assert SENTENCE_LIST_FONT_SIZE < PRIMARY_FONT_SIZE
    assert f"font-size: {SENTENCE_LIST_FONT_SIZE}px;" in stylesheet


def test_sentence_practice_uses_separate_list_and_detail_pages():
    assert SENTENCE_LIST_PAGE_INDEX == 0
    assert SENTENCE_DETAIL_PAGE_INDEX == 1


def test_detail_page_widgets_have_minimum_heights_to_prevent_clipping():
    stylesheet = build_stylesheet()

    assert DETAIL_PROMPT_MIN_HEIGHT >= 110
    assert METRIC_CARD_MIN_HEIGHT >= 100
    assert RESULT_LINE_MIN_HEIGHT >= 48
    assert f"min-height: {DETAIL_PROMPT_MIN_HEIGHT}px;" in stylesheet
    assert f"min-height: {METRIC_CARD_MIN_HEIGHT}px;" in stylesheet
    assert f"min-height: {RESULT_LINE_MIN_HEIGHT}px;" in stylesheet


def test_ui_script_check_command_runs_without_opening_window():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "pronunciation_ui.py", "--check"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ui_runtime_ok" in result.stdout
