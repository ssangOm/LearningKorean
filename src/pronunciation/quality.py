from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.pronunciation.config import HOP_LENGTH, N_FFT, SAMPLE_RATE


@dataclass(frozen=True)
class QualityAssessment:
    is_valid: bool
    score: float
    duration: float
    message: str


def assess_recording_quality(file_path: str | Path) -> QualityAssessment:
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    duration = float(len(signal) / sr) if sr else 0.0
    if signal.size == 0:
        return QualityAssessment(
            is_valid=False,
            score=5.0,
            duration=duration,
            message="음성이 거의 감지되지 않습니다. 마이크 입력과 녹음 권한을 확인해 주세요.",
        )

    peak = float(np.max(np.abs(signal)))
    rms = librosa.feature.rms(y=signal, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms_max = float(np.max(rms)) if rms.size else 0.0
    if peak < 0.005 or rms_max < 0.002:
        return QualityAssessment(
            is_valid=False,
            score=5.0,
            duration=duration,
            message="음성이 거의 감지되지 않습니다. 문장을 조금 더 크게 읽어 주세요.",
        )

    zcr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=signal,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    flatness_mean = float(np.mean(flatness)) if flatness.size else 0.0
    centroid_std = float(np.std(centroid)) if centroid.size else 0.0

    if zcr_mean > 0.35 and flatness_mean > 0.5 and centroid_std < 500:
        return QualityAssessment(
            is_valid=False,
            score=25.0,
            duration=duration,
            message="잡음에 가까운 녹음으로 판단됩니다. 주변 소음을 줄이고 다시 녹음해 주세요.",
        )

    return QualityAssessment(
        is_valid=True,
        score=100.0,
        duration=duration,
        message="녹음 품질이 평가 가능한 범위입니다.",
    )
