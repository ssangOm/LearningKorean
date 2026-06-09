from types import SimpleNamespace

import pytest

from src.pronunciation.ai_coach import (
    AiCoachError,
    AiCoachConfig,
    CONVERSATION_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    build_conversation_prompt_request,
    build_conversation_evaluation_prompt,
    build_coaching_prompt,
    clean_model_text,
    generate_ai_coaching,
    generate_conversation_feedback,
    generate_conversation_prompt,
    load_api_key,
)
from src.pronunciation.evaluate import EvaluationResult
from src.pronunciation.feedback import FeedbackResult
from src.pronunciation.similarity import SimilarityResult
from src.pronunciation.text_similarity import compare_texts


def _evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        file="recorded.wav",
        target_sentence_id="aa0",
        closest_sentence_id="aa0",
        closest_text="답안을 확인하세요.",
        score=82.5,
        weak_region="문장",
        similarity=SimilarityResult(
            score=82.5,
            normalized_distance=1.0,
            path_length=10,
            reference_duration=2.0,
            user_duration=2.1,
            segments=[],
        ),
        feedback=FeedbackResult(
            sentence_text="답안을 확인하세요.",
            score=82.5,
            weak_region="문장",
            duration_delta=0.1,
            messages=(
                "인식된 문장: 다반을 확인하세요",
                "문장 일치도는 82.5점입니다.",
                "발화 길이 참고: 전체 발화 길이는 기준과 비슷합니다.",
            ),
        ),
        recognized_text="다반을 확인하세요",
        text_score=82.5,
        text_comparison=compare_texts("답안을 확인하세요.", "다반을 확인하세요"),
    )


class FakeModels:
    def __init__(self) -> None:
        self.request = None

    def generate_content(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(text="AI 코칭 결과입니다.")


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()


def test_build_coaching_prompt_contains_pronunciation_context():
    prompt = build_coaching_prompt(_evaluation_result())

    assert "정답 문장: 답안을 확인하세요." in prompt
    assert "인식 문장: 다반을 확인하세요" in prompt
    assert "점수: 82.5 / 100" in prompt
    assert "빠진 표현" in prompt
    assert "다르게 인식된 표현" in prompt
    assert "표준 발음" in prompt


def test_build_coaching_prompt_requires_gemini_to_say_feedback_summary():
    prompt = build_coaching_prompt(_evaluation_result())

    assert "반드시 아래 피드백 항목을 자연스럽게 포함하세요" in prompt
    assert "- 인식된 문장: 다반을 확인하세요" in prompt
    assert "- 문장 일치도는 82.5점입니다." in prompt
    assert "- 발화 길이 참고: 전체 발화 길이는 기준과 비슷합니다." in prompt
    assert "출력은 '피드백' 제목으로 시작하세요" in prompt


def test_generate_ai_coaching_uses_client_without_changing_score():
    client = FakeClient()

    coaching = generate_ai_coaching(_evaluation_result(), client=client, api_key="test-key")

    assert coaching == "AI 코칭 결과입니다."
    assert client.models.request["model"]
    assert "점수를 새로 만들지 마세요" in client.models.request["config"]["system_instruction"]
    assert client.models.request["config"]["max_output_tokens"] >= 1400
    assert "답안을 확인하세요" in client.models.request["contents"]


def test_ai_coach_default_allows_long_feedback():
    assert AiCoachConfig().max_output_tokens >= 1400


def test_system_instructions_reject_markdown_bold_markup():
    assert "Markdown" in SYSTEM_INSTRUCTION
    assert "**" in SYSTEM_INSTRUCTION
    assert "Markdown" in CONVERSATION_SYSTEM_INSTRUCTION
    assert "**" in CONVERSATION_SYSTEM_INSTRUCTION


def test_clean_model_text_removes_markdown_bold_markup():
    assert clean_model_text("**대화 평가**\n- **문맥**: 자연스럽습니다.") == (
        "대화 평가\n- 문맥: 자연스럽습니다."
    )


def test_clean_model_text_hides_provider_name():
    assert "gemini" not in clean_model_text("Gemini가 만든 질문입니다.").lower()


def test_build_conversation_prompt_request_avoids_previous_questions():
    prompt = build_conversation_prompt_request(previous_prompts=["오늘 점심에 무엇을 먹었나요?"])

    assert "오늘 점심에 무엇을 먹었나요?" in prompt
    assert "절대 반복하지 마세요" in prompt


def test_build_conversation_evaluation_prompt_checks_contextual_answer():
    prompt = build_conversation_evaluation_prompt(
        gemini_prompt="주말에 보통 무엇을 하나요?",
        recognized_response="저는 친구랑 영화를 봐요.",
    )

    assert "AI 질문: 주말에 보통 무엇을 하나요?" in prompt
    assert "사용자 답변: 저는 친구랑 영화를 봐요." in prompt
    assert "문맥에 맞는 답변인지" in prompt
    assert "발음/전달력" in prompt


def test_generate_conversation_prompt_uses_gemini_client():
    client = FakeClient()

    prompt = generate_conversation_prompt(
        previous_prompts=["오늘 점심에 무엇을 먹었나요?"],
        client=client,
        api_key="test-key",
    )

    assert prompt == "AI 코칭 결과입니다."
    assert client.models.request["model"]
    assert "한국어 회화 연습 질문" in client.models.request["contents"]
    assert "오늘 점심에 무엇을 먹었나요?" in client.models.request["contents"]


def test_generate_conversation_feedback_uses_user_response():
    client = FakeClient()

    feedback = generate_conversation_feedback(
        "주말에 보통 무엇을 하나요?",
        "저는 친구랑 영화를 봐요.",
        client=client,
        api_key="test-key",
    )

    assert feedback == "AI 코칭 결과입니다."
    assert "주말에 보통 무엇을 하나요?" in client.models.request["contents"]
    assert "저는 친구랑 영화를 봐요." in client.models.request["contents"]
    assert "대화 평가" in client.models.request["config"]["system_instruction"]


def test_load_api_key_reads_project_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local secrets",
                'GEMINI_API_KEY="file-api-key"',
            ]
        ),
        encoding="utf-8",
    )

    assert load_api_key(env_file=env_file) == "file-api-key"


def test_generate_ai_coaching_requires_api_key_without_injected_client(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(AiCoachError, match="AI API"):
        generate_ai_coaching(_evaluation_result(), env_file="missing.env")
