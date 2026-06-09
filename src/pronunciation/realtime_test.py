import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf

from src.pronunciation.config import REFERENCE_MANIFEST_PATH, SAMPLE_RATE, USER_TEST_DIR
from src.pronunciation.asr import DEFAULT_ASR_MODEL_SIZE
from src.pronunciation.evaluate import EvaluationResult, evaluate_file, write_results_csv


@dataclass(frozen=True)
class AutoRecordConfig:
    frame_duration: float = 0.08
    start_threshold: float = 0.012
    silence_threshold: float = 0.007
    speech_start_duration: float = 0.16
    min_record_duration: float = 0.7
    silence_duration: float = 0.75
    no_speech_timeout: float = 5.0
    max_duration: float = 12.0
    calibration_duration: float = 0.35
    speech_threshold_multiplier: float = 4.0
    silence_threshold_multiplier: float = 2.2
    pre_roll_duration: float = 0.24
    post_roll_duration: float = 0.28


AutoRecorder = Callable[[str | Path, int, AutoRecordConfig], None]


def get_prompt_text(
    sentence_id: str,
    manifest_path: str | Path = REFERENCE_MANIFEST_PATH,
) -> str:
    manifest = pd.read_csv(manifest_path)
    matches = manifest[manifest["sentence_id"] == sentence_id]
    if matches.empty:
        raise ValueError(f"No prompt found for sentence_id={sentence_id!r}")
    return str(matches.iloc[0]["text"])


def auto_record_stop_reason(
    rms_values: list[float],
    config: AutoRecordConfig = AutoRecordConfig(),
) -> str | None:
    if not rms_values:
        return None

    speech_threshold, silence_threshold = _effective_thresholds(rms_values, config)
    speech_frames_required = max(1, round(config.speech_start_duration / config.frame_duration))
    silence_frames_required = max(1, round(config.silence_duration / config.frame_duration))
    min_frames_required = max(1, round(config.min_record_duration / config.frame_duration))
    no_speech_frames_allowed = max(1, round(config.no_speech_timeout / config.frame_duration))
    max_frames_allowed = max(1, round(config.max_duration / config.frame_duration))

    started = False
    consecutive_speech_frames = 0
    consecutive_silence_frames = 0

    for index, rms in enumerate(rms_values, start=1):
        if not started:
            if rms >= speech_threshold:
                consecutive_speech_frames += 1
                if consecutive_speech_frames >= speech_frames_required:
                    started = True
                    consecutive_silence_frames = 0
            else:
                consecutive_speech_frames = 0

            if not started and index >= no_speech_frames_allowed:
                return "no_speech"
            continue

        if rms <= silence_threshold:
            consecutive_silence_frames += 1
        else:
            consecutive_silence_frames = 0

        if index >= min_frames_required and consecutive_silence_frames >= silence_frames_required:
            return "silence"

    if len(rms_values) >= max_frames_allowed:
        return "max_duration"

    return None


def record_microphone_until_silence(
    output_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    config: AutoRecordConfig = AutoRecordConfig(),
) -> None:
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Real-time microphone recording requires the optional package `sounddevice`. "
            "Install it with `pip install sounddevice` or `pip install -r requirements.txt`."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames_per_chunk = max(1, int(sample_rate * config.frame_duration))
    chunks: list[np.ndarray] = []
    rms_values: list[float] = []

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while True:
            chunk, _overflowed = stream.read(frames_per_chunk)
            mono = np.asarray(chunk[:, 0], dtype=np.float32).copy()
            chunks.append(mono)
            rms_values.append(_rms(mono))

            if auto_record_stop_reason(rms_values, config) is not None:
                break

    signal = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    signal = _trim_to_detected_speech(signal, rms_values, sample_rate, config)
    sf.write(output_path, signal, sample_rate)


def run_realtime_test(
    sentence_id: str,
    output_path: str | Path | None = None,
    manifest_path: str | Path = REFERENCE_MANIFEST_PATH,
    recorder: AutoRecorder = record_microphone_until_silence,
    auto_config: AutoRecordConfig = AutoRecordConfig(),
    use_asr: bool = True,
    asr_model_size: str = DEFAULT_ASR_MODEL_SIZE,
) -> EvaluationResult:
    output_path = Path(output_path) if output_path else _default_recording_path(sentence_id)
    recorder(output_path, SAMPLE_RATE, auto_config)
    return evaluate_file(
        output_path,
        sentence_id=sentence_id,
        manifest_path=manifest_path,
        use_asr=use_asr,
        asr_model_size=asr_model_size,
    )


def _default_recording_path(sentence_id: str) -> Path:
    USER_TEST_DIR.mkdir(parents=True, exist_ok=True)
    return USER_TEST_DIR / f"realtime_{sentence_id}.wav"


def _print_result(result: EvaluationResult, output_path: str | Path) -> None:
    print(f"Recorded file: {output_path}")
    print(f"Closest reference: {result.closest_sentence_id} - {result.closest_text}")
    if result.recognized_text:
        print(f"Recognized text: {result.recognized_text}")
        print(f"Text score: {result.text_score:.1f} / 100")
    print(f"Score: {result.score:.1f} / 100")
    print(f"Improvement category: {result.weak_region}")
    print("Feedback:")
    for message in result.feedback.messages:
        print(f"- {message}")


def _effective_thresholds(
    rms_values: list[float],
    config: AutoRecordConfig,
) -> tuple[float, float]:
    calibration_frames = max(1, round(config.calibration_duration / config.frame_duration))
    calibration_values = np.asarray(rms_values[:calibration_frames], dtype=np.float32)
    if calibration_values.size == 0:
        noise_floor = 0.0
    else:
        sorted_values = np.sort(calibration_values)
        lower_count = max(1, int(np.ceil(sorted_values.size * 0.6)))
        noise_floor = float(np.median(sorted_values[:lower_count]))

    speech_threshold = max(config.start_threshold, noise_floor * config.speech_threshold_multiplier)
    silence_threshold = max(config.silence_threshold, noise_floor * config.silence_threshold_multiplier)
    return speech_threshold, silence_threshold


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float32))))


def _trim_to_detected_speech(
    signal: np.ndarray,
    rms_values: list[float],
    sample_rate: int,
    config: AutoRecordConfig,
) -> np.ndarray:
    if signal.size == 0 or not rms_values:
        return signal

    speech_threshold, _silence_threshold = _effective_thresholds(rms_values, config)
    speech_indices = [index for index, rms in enumerate(rms_values) if rms >= speech_threshold]
    if not speech_indices:
        return signal

    frames_per_chunk = max(1, int(sample_rate * config.frame_duration))
    pre_roll_chunks = max(0, round(config.pre_roll_duration / config.frame_duration))
    post_roll_chunks = max(0, round(config.post_roll_duration / config.frame_duration))
    start_chunk = max(0, speech_indices[0] - pre_roll_chunks)
    end_chunk = min(len(rms_values), speech_indices[-1] + post_roll_chunks + 1)
    start_sample = start_chunk * frames_per_chunk
    end_sample = min(signal.size, end_chunk * frames_per_chunk)
    return signal[start_sample:end_sample]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record microphone audio and test pronunciation live.")
    parser.add_argument("--sentence-id", required=True)
    parser.add_argument("--manifest", default=str(REFERENCE_MANIFEST_PATH))
    parser.add_argument("--recording-output")
    parser.add_argument("--result-output", default="outputs/pronunciation/realtime_result.csv")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL_SIZE, help="faster-whisper model size")
    args = parser.parse_args()

    prompt = get_prompt_text(args.sentence_id, args.manifest)
    output_path = Path(args.recording_output) if args.recording_output else _default_recording_path(args.sentence_id)
    print(f"Read this sentence: {prompt}")
    print("Recording automatically. Start reading, then pause when you are done...")
    result = run_realtime_test(
        sentence_id=args.sentence_id,
        output_path=output_path,
        manifest_path=args.manifest,
        asr_model_size=args.asr_model,
    )
    write_results_csv([result], output_path=args.result_output)
    _print_result(result, output_path)
    print(f"Wrote result CSV: {args.result_output}")


if __name__ == "__main__":
    main()
