import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.pronunciation.config import EVALUATION_RESULTS_PATH, REFERENCE_MANIFEST_PATH
from src.pronunciation.asr import DEFAULT_ASR_MODEL_SIZE, TranscriptionResult, transcribe_audio
from src.pronunciation.feedback import FeedbackResult, build_feedback
from src.pronunciation.quality import QualityAssessment, assess_recording_quality
from src.pronunciation.similarity import SimilarityResult, compare_audio_files
from src.pronunciation.text_similarity import TextComparisonResult, compare_texts


MISMATCH_DISTANCE_MARGIN = 0.08
Transcriber = Callable[[str | Path], TranscriptionResult]


@dataclass(frozen=True)
class EvaluationResult:
    file: str
    target_sentence_id: str | None
    closest_sentence_id: str
    closest_text: str
    score: float
    weak_region: str
    similarity: SimilarityResult
    feedback: FeedbackResult
    recognized_text: str = ""
    text_score: float = 0.0
    text_comparison: TextComparisonResult | None = None


def evaluate_file(
    file_path: str | Path,
    sentence_id: str | None = None,
    manifest_path: str | Path = REFERENCE_MANIFEST_PATH,
    use_asr: bool = False,
    asr_model_size: str = DEFAULT_ASR_MODEL_SIZE,
    transcriber: Transcriber | None = None,
) -> EvaluationResult:
    manifest = pd.read_csv(manifest_path)
    if sentence_id and manifest[manifest["sentence_id"] == sentence_id].empty:
        raise ValueError(f"No references found for sentence_id={sentence_id!r}")

    quality = assess_recording_quality(file_path)
    if not quality.is_valid:
        return _quality_failure_result(
            file_path=file_path,
            sentence_id=sentence_id,
            manifest=manifest,
            quality=quality,
        )

    scored_references = _score_references(manifest, file_path)
    if not scored_references:
        raise ValueError("No reference audio found in manifest.")

    best_row, best_similarity = scored_references[0]
    target_row = best_row
    target_similarity = best_similarity
    mismatch_detected = False

    if sentence_id:
        target_candidates = [
            (row, similarity)
            for row, similarity in scored_references
            if str(row["sentence_id"]) == sentence_id
        ]
        target_row, target_similarity = target_candidates[0]
        mismatch_detected = _is_sentence_mismatch(
            target_sentence_id=sentence_id,
            target_similarity=target_similarity,
            best_row=best_row,
            best_similarity=best_similarity,
        )
        if not mismatch_detected:
            best_row = target_row
            best_similarity = target_similarity

    feedback = build_feedback(
        sentence_text=str(target_row["text"]),
        score=(
            _mismatch_score(target_similarity, best_similarity)
            if mismatch_detected
            else best_similarity.score
        ),
        reference_duration=target_similarity.reference_duration,
        user_duration=target_similarity.user_duration,
        segments=target_similarity.segments,
    )
    if mismatch_detected:
        feedback = _append_mismatch_feedback(feedback, best_row)

    transcription = None
    text_comparison = None
    if use_asr:
        transcription = (
            transcriber(file_path)
            if transcriber
            else transcribe_audio(file_path, model_size=asr_model_size)
        )
        text_comparison = compare_texts(str(target_row["text"]), transcription.text)
        feedback = _asr_feedback(
            sentence_text=str(target_row["text"]),
            audio_feedback=feedback,
            text_comparison=text_comparison,
        )
        best_row = target_row

    return EvaluationResult(
        file=str(file_path),
        target_sentence_id=sentence_id,
        closest_sentence_id=str(best_row["sentence_id"]),
        closest_text=str(best_row["text"]),
        score=round(feedback.score, 1),
        weak_region=feedback.weak_region,
        similarity=target_similarity if mismatch_detected else best_similarity,
        feedback=feedback,
        recognized_text=transcription.text if transcription else "",
        text_score=text_comparison.score if text_comparison else 0.0,
        text_comparison=text_comparison,
    )


def _score_references(
    manifest: pd.DataFrame,
    file_path: str | Path,
) -> list[tuple[pd.Series, SimilarityResult]]:
    scored_references = [
        (row, compare_audio_files(row["path"], file_path))
        for _, row in manifest.iterrows()
    ]
    return sorted(
        scored_references,
        key=lambda item: item[1].normalized_distance,
    )


def _is_sentence_mismatch(
    target_sentence_id: str,
    target_similarity: SimilarityResult,
    best_row: pd.Series,
    best_similarity: SimilarityResult,
) -> bool:
    if str(best_row["sentence_id"]) == target_sentence_id:
        return False

    distance_gap = target_similarity.normalized_distance - best_similarity.normalized_distance
    return distance_gap >= MISMATCH_DISTANCE_MARGIN


def _mismatch_score(
    target_similarity: SimilarityResult,
    best_similarity: SimilarityResult,
) -> float:
    distance_gap = target_similarity.normalized_distance - best_similarity.normalized_distance
    capped_score = 60.0 - distance_gap * 80.0
    return max(15.0, min(target_similarity.score, capped_score))


def _append_mismatch_feedback(
    feedback: FeedbackResult,
    closest_row: pd.Series,
) -> FeedbackResult:
    message = (
        "선택한 문장과 다른 발화일 가능성이 큽니다. "
        f"현재 녹음은 '{closest_row['text']}' 문장에 더 가깝게 감지되었습니다."
    )
    return FeedbackResult(
        sentence_text=feedback.sentence_text,
        score=feedback.score,
        weak_region=feedback.weak_region,
        duration_delta=feedback.duration_delta,
        messages=(*feedback.messages, message),
    )


def _asr_feedback(
    sentence_text: str,
    audio_feedback: FeedbackResult,
    text_comparison: TextComparisonResult,
) -> FeedbackResult:
    messages = (
        *text_comparison.messages,
        f"발화 길이 참고: {audio_feedback.messages[0]}",
    )
    return FeedbackResult(
        sentence_text=sentence_text,
        score=text_comparison.score,
        weak_region=_asr_weak_category(audio_feedback, text_comparison),
        duration_delta=audio_feedback.duration_delta,
        messages=messages,
    )


def _asr_weak_category(
    audio_feedback: FeedbackResult,
    text_comparison: TextComparisonResult,
) -> str:
    if _has_speed_issue(audio_feedback):
        return "속도"
    if text_comparison.weak_category != "없음" or text_comparison.score < 100:
        return "발음"
    if audio_feedback.score < 90:
        return "발음"
    return "없음"


def _has_speed_issue(feedback: FeedbackResult) -> bool:
    return bool(
        feedback.messages
        and (
            "기준보다 깁니다" in feedback.messages[0]
            or "기준보다 짧습니다" in feedback.messages[0]
        )
    )


def _quality_failure_result(
    file_path: str | Path,
    sentence_id: str | None,
    manifest: pd.DataFrame,
    quality: QualityAssessment,
) -> EvaluationResult:
    candidates = manifest[manifest["sentence_id"] == sentence_id] if sentence_id else manifest
    row = candidates.iloc[0] if not candidates.empty else manifest.iloc[0]
    similarity = SimilarityResult(
        score=quality.score,
        normalized_distance=float("inf"),
        path_length=0,
        reference_duration=0.0,
        user_duration=quality.duration,
        segments=[],
    )
    feedback = FeedbackResult(
        sentence_text=str(row["text"]),
        score=quality.score,
        weak_region="녹음 품질",
        duration_delta=round(quality.duration, 3),
        messages=(quality.message,),
    )
    return EvaluationResult(
        file=str(file_path),
        target_sentence_id=sentence_id,
        closest_sentence_id=str(row["sentence_id"]),
        closest_text=str(row["text"]),
        score=round(quality.score, 1),
        weak_region=feedback.weak_region,
        similarity=similarity,
        feedback=feedback,
    )


def evaluate_directory(
    input_dir: str | Path,
    sentence_id: str | None = None,
    manifest_path: str | Path = REFERENCE_MANIFEST_PATH,
    use_asr: bool = False,
    asr_model_size: str = DEFAULT_ASR_MODEL_SIZE,
) -> list[EvaluationResult]:
    paths = sorted(Path(input_dir).glob("*.wav"))
    results = []
    for path in paths:
        target_id = sentence_id or infer_sentence_id(path.name)
        results.append(
            evaluate_file(
                path,
                sentence_id=target_id,
                manifest_path=manifest_path,
                use_asr=use_asr,
                asr_model_size=asr_model_size,
            )
        )
    return results


def write_results_csv(
    results: list[EvaluationResult],
    output_path: str | Path = EVALUATION_RESULTS_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "file": result.file,
            "target_sentence_id": result.target_sentence_id or "",
            "closest_sentence_id": result.closest_sentence_id,
            "closest_text": result.closest_text,
            "score": result.score,
            "recognized_text": result.recognized_text,
            "text_score": result.text_score,
            "weak_region": result.weak_region,
            "duration_delta": result.feedback.duration_delta,
            "feedback": " ".join(result.feedback.messages),
        }
        for result in results
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def infer_sentence_id(filename: str) -> str | None:
    match = re.search(r"sentence_\d{3}", filename)
    return match.group(0) if match else None


def _print_result(result: EvaluationResult) -> None:
    print(f"Target sentence: {result.target_sentence_id or '(auto)'}")
    print(f"Closest reference: {result.closest_sentence_id} - {result.closest_text}")
    print(f"Score: {result.score:.1f} / 100")
    print(f"Improvement category: {result.weak_region}")
    print("Feedback:")
    for message in result.feedback.messages:
        print(f"- {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Korean pronunciation similarity.")
    parser.add_argument("--sentence-id")
    parser.add_argument("--file")
    parser.add_argument("--input-dir")
    parser.add_argument("--manifest", default=str(REFERENCE_MANIFEST_PATH))
    parser.add_argument("--output", default=str(EVALUATION_RESULTS_PATH))
    parser.add_argument("--asr", action="store_true", help="use local Whisper ASR text matching as the main score")
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL_SIZE, help="faster-whisper model size")
    args = parser.parse_args()

    if not args.file and not args.input_dir:
        parser.error("One of --file or --input-dir is required.")

    if args.file:
        result = evaluate_file(
            args.file,
            sentence_id=args.sentence_id,
            manifest_path=args.manifest,
            use_asr=args.asr,
            asr_model_size=args.asr_model,
        )
        write_results_csv([result], output_path=args.output)
        _print_result(result)
        print(f"Wrote results: {args.output}")
        return

    results = evaluate_directory(
        args.input_dir,
        sentence_id=args.sentence_id,
        manifest_path=args.manifest,
        use_asr=args.asr,
        asr_model_size=args.asr_model,
    )
    output_path = write_results_csv(results, output_path=args.output)
    for result in results:
        _print_result(result)
        print()
    print(f"Wrote results: {output_path}")


if __name__ == "__main__":
    main()
