from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceSentence:
    sentence_id: str
    text: str
    purpose: str


DATA_DIR = Path("data/pronunciation")
REFERENCE_DIR = DATA_DIR / "references"
USER_TEST_DIR = DATA_DIR / "user_tests"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = Path("outputs/pronunciation")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REFERENCE_MANIFEST_PATH = OUTPUT_DIR / "reference_manifest.csv"
EVALUATION_RESULTS_PATH = OUTPUT_DIR / "evaluation_results.csv"
PORTFOLIO_REPORT_PATH = REPORT_DIR / "pronunciation_similarity_summary.md"

SAMPLE_RATE = 16000
N_MFCC = 20
N_MELS = 64
HOP_LENGTH = 256
N_FFT = 512
SCORE_ALPHA = 0.075

DEFAULT_TTS_VOICES = (
    "ko-KR-SunHiNeural",
    "ko-KR-InJoonNeural",
)

DEFAULT_CONDITIONS = ("clean", "mild", "room")

SENTENCES = (
    ReferenceSentence("sentence_001", "오늘 공기가 맑아요", "공기/고기 유사 발음"),
    ReferenceSentence("sentence_002", "감기에 걸렸어요", "감기/경기 유사 발음"),
    ReferenceSentence("sentence_003", "거기 의자에 앉으세요", "위치 명령"),
    ReferenceSentence("sentence_004", "물을 조금 마실게요", "헬스케어/일상"),
    ReferenceSentence("sentence_005", "약을 먹을 시간이에요", "헬스케어"),
    ReferenceSentence("sentence_006", "천천히 다시 말해 주세요", "음성 인터페이스"),
    ReferenceSentence("sentence_007", "오른쪽 버튼을 눌러 주세요", "명령 문장"),
    ReferenceSentence("sentence_008", "숨을 크게 쉬어 보세요", "헬스케어"),
    ReferenceSentence("sentence_009", "지금 상태를 알려 주세요", "상태 입력"),
    ReferenceSentence("sentence_010", "도움이 필요해요", "도움 요청"),
)


def get_sentence(sentence_id: str) -> ReferenceSentence:
    for sentence in SENTENCES:
        if sentence.sentence_id == sentence_id:
            return sentence
    raise KeyError(f"Unknown sentence_id: {sentence_id}")


def ensure_project_dirs() -> None:
    for path in (
        REFERENCE_DIR,
        USER_TEST_DIR,
        PROCESSED_DIR,
        OUTPUT_DIR,
        REPORT_DIR,
        MODEL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
