import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf

from src.pronunciation.realtime_test import (
    AutoRecordConfig,
    auto_record_stop_reason,
    get_prompt_text,
    run_realtime_test,
)


def _write_tone(path: Path, frequency: float, sr: int = 16000, duration: float = 0.45) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, signal, sr)


def test_get_prompt_text_reads_manifest_sentence(tmp_path):
    reference_path = tmp_path / "reference.wav"
    _write_tone(reference_path, 440)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "sentence_id": "sentence_001",
                "text": "오늘 공기가 맑아요",
                "path": str(reference_path),
            }
        ]
    ).to_csv(manifest_path, index=False)

    assert get_prompt_text("sentence_001", manifest_path) == "오늘 공기가 맑아요"


def test_run_realtime_test_records_then_evaluates(tmp_path):
    reference_path = tmp_path / "reference.wav"
    recorded_path = tmp_path / "recorded.wav"
    manifest_path = tmp_path / "manifest.csv"
    _write_tone(reference_path, 440)
    pd.DataFrame(
        [
            {
                "sentence_id": "sentence_001",
                "text": "오늘 공기가 맑아요",
                "path": str(reference_path),
            }
        ]
    ).to_csv(manifest_path, index=False)

    def fake_recorder(output_path, sample_rate, auto_config):
        _write_tone(Path(output_path), 440, sr=sample_rate, duration=0.45)

    result = run_realtime_test(
        sentence_id="sentence_001",
        output_path=recorded_path,
        manifest_path=manifest_path,
        recorder=fake_recorder,
        use_asr=False,
    )

    assert recorded_path.exists()
    assert result.closest_sentence_id == "sentence_001"
    assert result.score > 95


def test_auto_record_stop_reason_waits_for_speech_then_stops_on_silence():
    config = AutoRecordConfig(
        frame_duration=0.1,
        start_threshold=0.02,
        silence_threshold=0.01,
        speech_start_duration=0.1,
        min_record_duration=0.3,
        silence_duration=0.3,
        no_speech_timeout=2.0,
        max_duration=3.0,
        calibration_duration=0.1,
        speech_threshold_multiplier=1.0,
        silence_threshold_multiplier=1.0,
    )
    rms_values = [0.004, 0.03, 0.04, 0.035, 0.006, 0.005, 0.004]

    assert auto_record_stop_reason(rms_values, config) == "silence"


def test_auto_record_stop_reason_times_out_when_no_speech_is_detected():
    config = AutoRecordConfig(
        frame_duration=0.1,
        start_threshold=0.02,
        silence_threshold=0.01,
        no_speech_timeout=0.5,
        max_duration=3.0,
        calibration_duration=0.1,
        speech_threshold_multiplier=1.0,
        silence_threshold_multiplier=1.0,
    )

    assert auto_record_stop_reason([0.002, 0.003, 0.002, 0.003, 0.002], config) == "no_speech"


def test_realtime_script_can_run_directly_with_help():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "pronunciation" / "realtime_test.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--sentence-id" in result.stdout
    assert "--duration" not in result.stdout
