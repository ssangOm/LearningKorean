from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol


DEFAULT_ASR_MODEL_SIZE = "medium"
DEFAULT_ASR_BEAM_SIZE = 5


class WhisperLikeModel(Protocol):
    def transcribe(
        self,
        path: str,
        language: str,
        beam_size: int,
        vad_filter: bool,
        word_timestamps: bool,
    ):
        ...


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    model_size: str


ModelLoader = Callable[[str], WhisperLikeModel]


def transcribe_audio(
    file_path: str | Path,
    model_size: str = DEFAULT_ASR_MODEL_SIZE,
    language: str = "ko",
    model_loader: ModelLoader | None = None,
) -> TranscriptionResult:
    model = model_loader(model_size) if model_loader else _load_model(model_size)
    segments, info = model.transcribe(
        str(file_path),
        language=language,
        beam_size=DEFAULT_ASR_BEAM_SIZE,
        vad_filter=True,
        word_timestamps=False,
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return TranscriptionResult(
        text=_compact_spaces(text),
        language=str(getattr(info, "language", language)),
        model_size=model_size,
    )


@lru_cache(maxsize=4)
def _load_model(model_size: str) -> WhisperLikeModel:
    try:
        return _create_whisper_model(model_size, local_files_only=True)
    except Exception as exc:
        if "`faster-whisper`" in str(exc):
            raise
        raise RuntimeError(
            f"Whisper {model_size!r} 모델이 아직 로컬에 준비되지 않았습니다. "
            "`python -m src.pronunciation.asr --download`를 먼저 실행해 모델을 받아 주세요."
        ) from exc


def download_asr_model(model_size: str = DEFAULT_ASR_MODEL_SIZE) -> WhisperLikeModel:
    model = _create_whisper_model(model_size, local_files_only=False)
    _load_model.cache_clear()
    return model


def _create_whisper_model(model_size: str, *, local_files_only: bool) -> WhisperLikeModel:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "로컬 음성 인식을 사용하려면 `faster-whisper`가 필요합니다. "
            "`pip install -r requirements.txt`를 다시 실행해 주세요."
        ) from exc

    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        local_files_only=local_files_only,
    )


def _compact_spaces(text: str) -> str:
    return " ".join(text.split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or inspect local Whisper ASR models.")
    parser.add_argument("--download", action="store_true", help="download the configured Whisper model")
    parser.add_argument("--model", default=DEFAULT_ASR_MODEL_SIZE, help="Whisper model size to prepare")
    args = parser.parse_args(argv)

    if args.download:
        print(f"Downloading Whisper model: {args.model}")
        download_asr_model(args.model)
        print(f"Whisper model ready: {args.model}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
