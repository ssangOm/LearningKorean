import sys
from types import SimpleNamespace

import pytest

from src.pronunciation.asr import (
    DEFAULT_ASR_BEAM_SIZE,
    DEFAULT_ASR_MODEL_SIZE,
    _load_model,
    download_asr_model,
    transcribe_audio,
)


class FakeWhisperModel:
    def transcribe(self, path, language, beam_size, vad_filter, word_timestamps):
        assert path == "sample.wav"
        assert language == "ko"
        assert beam_size == DEFAULT_ASR_BEAM_SIZE
        assert vad_filter is True
        assert word_timestamps is False
        segments = [
            SimpleNamespace(text=" 그럼 지금 "),
            SimpleNamespace(text="좀 보여 드리죠. "),
        ]
        return segments, SimpleNamespace(language="ko")


def test_transcribe_audio_returns_joined_text_from_model_loader():
    result = transcribe_audio("sample.wav", model_loader=lambda _model_size: FakeWhisperModel())

    assert result.text == "그럼 지금 좀 보여 드리죠."
    assert result.language == "ko"
    assert result.model_size == DEFAULT_ASR_MODEL_SIZE


def test_default_asr_model_prefers_accuracy_over_small_model_speed():
    assert DEFAULT_ASR_MODEL_SIZE == "medium"
    assert DEFAULT_ASR_BEAM_SIZE >= 3


def test_load_model_uses_local_cache_only(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_size, **kwargs):
            calls.append((model_size, kwargs))

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    _load_model.cache_clear()

    _load_model("medium")

    assert calls == [
        (
            "medium",
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
            },
        )
    ]
    _load_model.cache_clear()


def test_load_model_missing_cache_has_actionable_error(monkeypatch):
    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("model.bin missing")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    _load_model.cache_clear()

    with pytest.raises(RuntimeError, match="--download"):
        _load_model("medium")

    _load_model.cache_clear()


def test_download_asr_model_allows_network_download(monkeypatch):
    calls = []

    def fake_create_model(model_size, *, local_files_only):
        calls.append((model_size, local_files_only))
        return object()

    monkeypatch.setattr("src.pronunciation.asr._create_whisper_model", fake_create_model)

    download_asr_model("medium")

    assert calls == [("medium", False)]
