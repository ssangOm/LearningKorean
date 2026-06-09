from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.pronunciation.config import HOP_LENGTH, N_FFT, N_MFCC, SAMPLE_RATE


@dataclass(frozen=True)
class SegmentScore:
    label: str
    score: float
    average_distance: float = 0.0


@dataclass(frozen=True)
class SimilarityResult:
    score: float
    normalized_distance: float
    path_length: int
    reference_duration: float
    user_duration: float
    segments: tuple[SegmentScore, ...]


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    signal, sample_rate = librosa.load(path, sr=sr, mono=True)
    return preprocess_signal(signal), sample_rate


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return signal.astype(np.float32)

    signal = signal.astype(np.float32)
    signal, _ = librosa.effects.trim(signal, top_db=35)
    if signal.size == 0:
        return signal.astype(np.float32)

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float32)


def extract_mfcc(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if signal.size < N_FFT:
        signal = np.pad(signal, (0, N_FFT - signal.size))
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return librosa.util.normalize(mfcc, axis=1)


def compare_audio_files(reference_path: str | Path, user_path: str | Path) -> SimilarityResult:
    reference_signal, sr = load_audio(reference_path)
    user_signal, _ = load_audio(user_path, sr=sr)
    return compare_audio_arrays(reference_signal, user_signal, sr=sr)


def compare_audio_arrays(
    reference_signal: np.ndarray,
    user_signal: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> SimilarityResult:
    reference_mfcc = extract_mfcc(reference_signal, sr=sr)
    user_mfcc = extract_mfcc(user_signal, sr=sr)
    distance, path = _dtw_distance(reference_mfcc, user_mfcc)
    normalized_distance = distance / max(len(path), 1) / np.sqrt(reference_mfcc.shape[0])
    score = distance_to_score(normalized_distance)
    segments = _segment_scores(reference_mfcc, user_mfcc, path)

    return SimilarityResult(
        score=score,
        normalized_distance=float(normalized_distance),
        path_length=int(len(path)),
        reference_duration=float(len(reference_signal) / sr),
        user_duration=float(len(user_signal) / sr),
        segments=segments,
    )


def distance_to_score(normalized_distance: float) -> float:
    distance = max(float(normalized_distance), 0.0)
    points = (
        (0.0, 100.0),
        (0.05, 95.0),
        (0.25, 80.0),
        (0.55, 35.0),
        (1.20, 5.0),
    )
    if distance >= points[-1][0]:
        return points[-1][1]

    for (left_distance, left_score), (right_distance, right_score) in zip(
        points,
        points[1:],
        strict=True,
    ):
        if left_distance <= distance <= right_distance:
            ratio = (distance - left_distance) / (right_distance - left_distance)
            score = left_score + ratio * (right_score - left_score)
            return float(np.clip(score, 0.0, 100.0))

    return 100.0


def _dtw_distance(reference_mfcc: np.ndarray, user_mfcc: np.ndarray) -> tuple[float, np.ndarray]:
    accumulated_cost, path = librosa.sequence.dtw(
        X=reference_mfcc,
        Y=user_mfcc,
        metric="euclidean",
    )
    return float(accumulated_cost[-1, -1]), path


def _segment_scores(
    reference_mfcc: np.ndarray,
    user_mfcc: np.ndarray,
    path: np.ndarray,
) -> tuple[SegmentScore, ...]:
    labels = ("초반", "중반", "후반")
    buckets: list[list[float]] = [[], [], []]
    max_user_frame = max(user_mfcc.shape[1] - 1, 1)

    for reference_frame, user_frame in path:
        bucket = min(int((user_frame / max_user_frame) * 3), 2)
        frame_distance = float(
            np.linalg.norm(reference_mfcc[:, reference_frame] - user_mfcc[:, user_frame])
            / np.sqrt(reference_mfcc.shape[0])
        )
        buckets[bucket].append(frame_distance)

    segment_scores = []
    for label, distances in zip(labels, buckets, strict=True):
        average_distance = float(np.mean(distances)) if distances else 0.0
        segment_scores.append(
            SegmentScore(
                label=label,
                score=distance_to_score(average_distance),
                average_distance=average_distance,
            )
        )
    return tuple(segment_scores)
