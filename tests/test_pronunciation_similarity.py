import numpy as np
import soundfile as sf

from src.pronunciation.similarity import compare_audio_files, distance_to_score


def _write_tone(path, frequency, sr=16000, duration=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, signal, sr)


def test_identical_audio_scores_higher_than_different_audio(tmp_path):
    first = tmp_path / "first.wav"
    same = tmp_path / "same.wav"
    different = tmp_path / "different.wav"
    _write_tone(first, 440)
    _write_tone(same, 440)
    _write_tone(different, 880)

    same_result = compare_audio_files(first, same)
    different_result = compare_audio_files(first, different)

    assert same_result.score > different_result.score
    assert same_result.score > 95
    assert 0 <= different_result.score <= 100
    assert same_result.normalized_distance < different_result.normalized_distance


def test_distance_to_score_spreads_realistic_dtw_distances():
    assert distance_to_score(0.0) == 100.0
    assert distance_to_score(0.2) < 90
    assert distance_to_score(0.4) < 65
    assert distance_to_score(0.7) < 30
