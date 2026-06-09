import numpy as np
import pandas as pd
import soundfile as sf

from src.pronunciation.evaluate import evaluate_file
from src.pronunciation.asr import TranscriptionResult


def _write_tone(path, frequency, sr=16000, duration=0.45):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, signal, sr)


def _write_silence(path, sr=16000, duration=0.45):
    signal = np.zeros(int(sr * duration))
    sf.write(path, signal, sr)


def _write_noise(path, sr=16000, duration=0.45):
    rng = np.random.default_rng(7)
    signal = 0.05 * rng.normal(size=int(sr * duration))
    sf.write(path, signal, sr)


def test_evaluate_file_selects_closest_reference(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    ref_a = tmp_path / "sentence_001_ref.wav"
    ref_b = tmp_path / "sentence_002_ref.wav"
    user = tmp_path / "user.wav"
    _write_tone(ref_a, 440)
    _write_tone(ref_b, 880)
    _write_tone(user, 440)
    pd.DataFrame(
        [
            {
                "sentence_id": "sentence_001",
                "text": "오늘 공기가 맑아요",
                "voice": "test",
                "condition": "clean",
                "path": str(ref_a),
            },
            {
                "sentence_id": "sentence_002",
                "text": "감기에 걸렸어요",
                "voice": "test",
                "condition": "clean",
                "path": str(ref_b),
            },
        ]
    ).to_csv(manifest_path, index=False)

    result = evaluate_file(user, manifest_path=manifest_path)

    assert result.closest_sentence_id == "sentence_001"
    assert result.score > 95
    assert result.feedback.messages


def test_evaluate_file_penalizes_recording_that_matches_another_sentence(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    target_ref = tmp_path / "sentence_001_ref.wav"
    other_ref = tmp_path / "sentence_002_ref.wav"
    user = tmp_path / "user.wav"
    _write_tone(target_ref, 440)
    _write_tone(other_ref, 880)
    _write_tone(user, 880)
    pd.DataFrame(
        [
            {
                "sentence_id": "sentence_001",
                "text": "오늘 공기가 맑아요",
                "voice": "test",
                "condition": "clean",
                "path": str(target_ref),
            },
            {
                "sentence_id": "sentence_002",
                "text": "감기에 걸렸어요",
                "voice": "test",
                "condition": "clean",
                "path": str(other_ref),
            },
        ]
    ).to_csv(manifest_path, index=False)

    result = evaluate_file(user, sentence_id="sentence_001", manifest_path=manifest_path)

    assert result.closest_sentence_id == "sentence_002"
    assert result.score <= 60
    assert any("선택한 문장과 다른 발화" in message for message in result.feedback.messages)


def test_evaluate_file_rejects_silent_recording(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    reference = tmp_path / "reference.wav"
    user = tmp_path / "silent.wav"
    _write_tone(reference, 440)
    _write_silence(user)
    pd.DataFrame(
        [
            {
                "sentence_id": "sentence_001",
                "text": "오늘 공기가 맑아요",
                "voice": "test",
                "condition": "clean",
                "path": str(reference),
            },
        ]
    ).to_csv(manifest_path, index=False)

    result = evaluate_file(user, sentence_id="sentence_001", manifest_path=manifest_path)

    assert result.score <= 20
    assert result.weak_region == "녹음 품질"
    assert any("음성이 거의 감지되지 않습니다" in message for message in result.feedback.messages)


def test_evaluate_file_rejects_noise_recording(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    reference = tmp_path / "reference.wav"
    user = tmp_path / "noise.wav"
    _write_tone(reference, 440)
    _write_noise(user)
    pd.DataFrame(
        [
            {
                "sentence_id": "sentence_001",
                "text": "오늘 공기가 맑아요",
                "voice": "test",
                "condition": "clean",
                "path": str(reference),
            },
        ]
    ).to_csv(manifest_path, index=False)

    result = evaluate_file(user, sentence_id="sentence_001", manifest_path=manifest_path)

    assert result.score <= 40
    assert result.weak_region == "녹음 품질"
    assert any("잡음에 가까운 녹음" in message for message in result.feedback.messages)


def test_evaluate_file_uses_asr_text_score_when_enabled(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    reference = tmp_path / "reference.wav"
    user = tmp_path / "user.wav"
    _write_tone(reference, 440)
    _write_tone(user, 440)
    pd.DataFrame(
        [
            {
                "sentence_id": "aa0",
                "text": "그럼 지금 좀 보여 드리죠.",
                "voice": "test",
                "condition": "clean",
                "path": str(reference),
            },
        ]
    ).to_csv(manifest_path, index=False)

    def fake_transcriber(_path):
        return TranscriptionResult(text="그럼 지금 보여 드려요", language="ko", model_size="fake")

    result = evaluate_file(
        user,
        sentence_id="aa0",
        manifest_path=manifest_path,
        use_asr=True,
        transcriber=fake_transcriber,
    )

    assert result.recognized_text == "그럼 지금 보여 드려요"
    assert result.text_score < 100
    assert result.score == result.text_score
    assert any("인식된 문장" in message for message in result.feedback.messages)


def test_evaluate_file_passes_asr_model_size_to_default_transcriber(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    reference = tmp_path / "reference.wav"
    user = tmp_path / "user.wav"
    _write_tone(reference, 440)
    _write_tone(user, 440)
    pd.DataFrame(
        [
            {
                "sentence_id": "aa0",
                "text": "그럼 지금 좀 보여 드리죠.",
                "voice": "test",
                "condition": "clean",
                "path": str(reference),
            },
        ]
    ).to_csv(manifest_path, index=False)
    calls = []

    def fake_transcribe_audio(path, model_size):
        calls.append((path, model_size))
        return TranscriptionResult(text="그럼 지금 좀 보여 드리죠", language="ko", model_size=model_size)

    monkeypatch.setattr("src.pronunciation.evaluate.transcribe_audio", fake_transcribe_audio)

    result = evaluate_file(
        user,
        sentence_id="aa0",
        manifest_path=manifest_path,
        use_asr=True,
        asr_model_size="medium",
    )

    assert result.recognized_text == "그럼 지금 좀 보여 드리죠"
    assert calls == [(user, "medium")]


def test_evaluate_file_uses_pronunciation_weak_category_when_asr_text_differs(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    reference = tmp_path / "reference.wav"
    user = tmp_path / "user.wav"
    _write_tone(reference, 440)
    _write_tone(user, 440)
    pd.DataFrame(
        [
            {
                "sentence_id": "aa0",
                "text": "그럼 지금 좀 보여 드리죠.",
                "voice": "test",
                "condition": "clean",
                "path": str(reference),
            },
        ]
    ).to_csv(manifest_path, index=False)

    def fake_transcriber(_path):
        return TranscriptionResult(text="그럼 지금 좀 보여 드려요", language="ko", model_size="fake")

    result = evaluate_file(
        user,
        sentence_id="aa0",
        manifest_path=manifest_path,
        use_asr=True,
        transcriber=fake_transcriber,
    )

    assert result.weak_region == "발음"


def test_evaluate_file_uses_speed_weak_category_when_duration_differs(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    reference = tmp_path / "reference.wav"
    user = tmp_path / "user.wav"
    _write_tone(reference, 440, duration=0.45)
    _write_tone(user, 440, duration=0.9)
    pd.DataFrame(
        [
            {
                "sentence_id": "aa0",
                "text": "그럼 지금 좀 보여 드리죠.",
                "voice": "test",
                "condition": "clean",
                "path": str(reference),
            },
        ]
    ).to_csv(manifest_path, index=False)

    def fake_transcriber(_path):
        return TranscriptionResult(text="그럼 지금 좀 보여 드리죠", language="ko", model_size="fake")

    result = evaluate_file(
        user,
        sentence_id="aa0",
        manifest_path=manifest_path,
        use_asr=True,
        transcriber=fake_transcriber,
    )

    assert result.weak_region == "속도"
