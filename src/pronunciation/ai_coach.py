from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from src.pronunciation.evaluate import EvaluationResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class AiCoachError(RuntimeError):
    pass


@dataclass(frozen=True)
class AiCoachConfig:
    model: str = DEFAULT_GEMINI_MODEL
    temperature: float = 0.2
    max_output_tokens: int = 1600


SYSTEM_INSTRUCTION = """
당신은 한국어 발음 평가 보조 코치입니다.
점수를 새로 만들지 마세요.
입력으로 주어진 정답 문장, 인식 문장, 기존 점수, 빠진 표현, 다르게 인식된 표현만 근거로 설명하세요.
표준 발음상 자연스러운 변화라면 감점 사유가 아니라고 분명히 말하세요.
출력은 반드시 '피드백' 제목으로 시작하고, 한국어 bullet로 작성하세요.
Markdown 문법, 별표 강조, **굵게** 표기를 절대 사용하지 마세요.
""".strip()


CONVERSATION_SYSTEM_INSTRUCTION = """
당신은 한국어 회화 발음/문맥 평가 코치입니다.
사용자가 말한 답변을 새로 만들어내지 말고, 입력된 AI 질문과 사용자 답변만 근거로 판단하세요.
출력은 반드시 '대화 평가' 제목으로 시작하고, 한국어 bullet로 작성하세요.
문맥 적합성, 답변 완성도, 발음/전달력, 다음 답변 예시를 짧고 구체적으로 포함하세요.
Markdown 문법, 별표 강조, **굵게** 표기를 절대 사용하지 마세요.
""".strip()


def build_coaching_prompt(result: EvaluationResult) -> str:
    comparison = result.text_comparison
    missing = ", ".join(comparison.missing_tokens) if comparison and comparison.missing_tokens else "없음"
    extra = ", ".join(comparison.extra_tokens) if comparison and comparison.extra_tokens else "없음"
    normalized_reference = comparison.normalized_reference if comparison else result.closest_text
    normalized_recognized = comparison.normalized_recognized if comparison else result.recognized_text
    feedback = "\n".join(f"- {message}" for message in result.feedback.messages)

    return f"""
다음은 한국어 발음 평가 결과입니다.

정답 문장: {result.closest_text}
인식 문장: {result.recognized_text or "(인식 실패)"}
정규화된 정답: {normalized_reference}
정규화된 인식: {normalized_recognized}
점수: {result.score:.1f} / 100
텍스트 일치도: {result.text_score:.1f} / 100
빠진 표현: {missing}
다르게 인식된 표현: {extra}
기존 피드백:
{feedback}

요청:
- 출력은 '피드백' 제목으로 시작하세요.
- 반드시 아래 피드백 항목을 자연스럽게 포함하세요.
{feedback}
- 표준 발음 규칙 관점에서 감점이 타당한지 설명해 주세요.
- 예: "답안"이 "다반"처럼 인식되는 경우에는 표준 발음상 자연스러운 연음/받침 변화인지 판단해 주세요.
- 사용자가 다음 녹음에서 바로 고칠 수 있는 구체적인 연습 팁을 제안해 주세요.
""".strip()


def build_conversation_prompt_request(previous_prompts: list[str] | tuple[str, ...] | None = None) -> str:
    previous_section = _previous_prompt_section(previous_prompts)
    return f"""
한국어 회화 연습 질문을 하나만 만들어 주세요.

조건:
- 일상 대화에서 자연스럽게 답할 수 있는 한 문장 질문이어야 합니다.
- 사용자가 1~2문장으로 답변하기 적당해야 합니다.
- 너무 어려운 전문 주제는 피하세요.
- 이전 질문과 같은 질문을 절대 반복하지 마세요.
- 출력은 질문 문장만 작성하세요.
{previous_section}
""".strip()


def build_conversation_evaluation_prompt(gemini_prompt: str, recognized_response: str) -> str:
    return f"""
다음은 한국어 회화 연습 결과입니다.

AI 질문: {gemini_prompt}
사용자 답변: {recognized_response or "(인식 실패)"}

요청:
- 사용자 답변이 AI 질문의 문맥에 맞는 답변인지 판단해 주세요.
- 답변이 너무 짧거나 엉뚱하면 왜 부족한지 설명해 주세요.
- 발음/전달력 관점에서 인식된 문장이 자연스러운지 설명해 주세요.
- 더 자연스러운 한국어 답변 예시를 1개 제안해 주세요.
""".strip()


def generate_ai_coaching(
    result: EvaluationResult,
    *,
    api_key: str | None = None,
    env_file: str | Path = DEFAULT_ENV_FILE,
    client=None,
    config: AiCoachConfig = AiCoachConfig(),
) -> str:
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    resolved_api_key = resolved_api_key or load_api_key(env_file=env_file)
    types_module = None

    if client is None:
        if not resolved_api_key:
            raise AiCoachError("AI API 키가 설정되어 있지 않습니다. .env 파일에 API 키를 넣어 주세요.")
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise AiCoachError(
                "`google-genai` 패키지가 없습니다. `pip install -r requirements.txt`를 실행해 주세요."
            ) from exc

        client = genai.Client(api_key=resolved_api_key)
        types_module = types

    response = client.models.generate_content(
        model=config.model,
        contents=build_coaching_prompt(result),
        config=_generation_config(config, types_module),
    )
    return _response_text(response)


def generate_conversation_prompt(
    *,
    previous_prompts: list[str] | tuple[str, ...] | None = None,
    api_key: str | None = None,
    env_file: str | Path = DEFAULT_ENV_FILE,
    client=None,
    config: AiCoachConfig = AiCoachConfig(),
) -> str:
    client, types_module = _resolve_client(api_key, env_file, client)
    response = client.models.generate_content(
        model=config.model,
        contents=build_conversation_prompt_request(previous_prompts),
        config=_generation_config(config, types_module, system_instruction=CONVERSATION_SYSTEM_INSTRUCTION),
    )
    return _response_text(response)


def generate_conversation_feedback(
    gemini_prompt: str,
    recognized_response: str,
    *,
    api_key: str | None = None,
    env_file: str | Path = DEFAULT_ENV_FILE,
    client=None,
    config: AiCoachConfig = AiCoachConfig(),
) -> str:
    client, types_module = _resolve_client(api_key, env_file, client)
    response = client.models.generate_content(
        model=config.model,
        contents=build_conversation_evaluation_prompt(gemini_prompt, recognized_response),
        config=_generation_config(config, types_module, system_instruction=CONVERSATION_SYSTEM_INSTRUCTION),
    )
    return _response_text(response)


def _resolve_client(api_key: str | None, env_file: str | Path, client):
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    resolved_api_key = resolved_api_key or load_api_key(env_file=env_file)
    types_module = None

    if client is not None:
        return client, types_module

    if not resolved_api_key:
        raise AiCoachError("AI API 키가 설정되어 있지 않습니다. .env 파일에 API 키를 넣어 주세요.")
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise AiCoachError(
            "`google-genai` 패키지가 없습니다. `pip install -r requirements.txt`를 실행해 주세요."
        ) from exc

    return genai.Client(api_key=resolved_api_key), types


def _response_text(response) -> str:
    text = clean_model_text(str(getattr(response, "text", "") or ""))
    if not text:
        raise AiCoachError("AI가 빈 응답을 반환했습니다.")
    return text


def clean_model_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = re.sub("gemini", "AI", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _previous_prompt_section(previous_prompts: list[str] | tuple[str, ...] | None) -> str:
    prompts = [prompt.strip() for prompt in previous_prompts or [] if prompt and prompt.strip()]
    if not prompts:
        return ""
    joined = "\n".join(f"- {prompt}" for prompt in prompts[-8:])
    return f"""

이전 질문:
{joined}
위 질문들은 절대 반복하지 마세요.
""".rstrip()


def _generation_config(config: AiCoachConfig, types_module=None, system_instruction: str = SYSTEM_INSTRUCTION):
    values = {
        "system_instruction": system_instruction,
        "max_output_tokens": config.max_output_tokens,
        "temperature": config.temperature,
    }
    if types_module is None:
        return values
    return types_module.GenerateContentConfig(**values)


def load_api_key(env_file: str | Path = DEFAULT_ENV_FILE) -> str | None:
    env_path = Path(env_file)
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
            continue
        cleaned = value.strip().strip("\"'")
        return cleaned or None

    return None
