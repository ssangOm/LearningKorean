# LearningKorean

한국어 문장 발음 연습을 위한 로컬 데스크톱 프로그램입니다. 사용자가 문장을 읽으면 로컬 Whisper ASR로 음성을 인식하고, 정답 문장과 비교해 점수와 피드백을 제공합니다. 선택적으로 Gemini API를 연결하면 더 자연스러운 한국어 코칭과 대화 평가를 생성합니다.

## 주요 기능

- 문장 목록 선택 후 녹음 평가
- 고정 녹음 시간이 아닌 음성 시작/침묵 기반 자동 녹음
- `faster-whisper` 기반 로컬 한국어 음성 인식
- 정답 문장과 인식 문장의 텍스트 일치도 계산
- MFCC + DTW 기반 기준 음성과 사용자 음성 유사도 계산
- 녹음 품질 검사
- AI 코칭 및 대화 연습 기능
- PySide6 기반 로컬 UI

## 프로젝트 구조

```text
pronunciation_ui.py              # 데스크톱 UI
src/pronunciation/
  asr.py                         # faster-whisper 음성 인식
  evaluate.py                    # 파일 단위 평가
  realtime_test.py               # 마이크 자동 녹음 평가
  similarity.py                  # MFCC + DTW 음향 유사도
  text_similarity.py             # 문장 텍스트 유사도
  feedback.py                    # 기본 피드백 생성
  quality.py                     # 녹음 품질 검사
  ai_coach.py                    # AI 코칭/대화 평가
tests/                           # pytest 테스트
outputs/pronunciation/*.csv      # 작은 결과/manifest CSV
reports/                         # 프로젝트 결과 요약
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS에서 마이크 녹음이 안 되면 PortAudio가 필요할 수 있습니다.

```bash
brew install portaudio
```

## Whisper 모델 준비

현재 UI는 한국어 문장 인식을 위해 `medium` 모델을 기본으로 사용합니다.

```bash
python -m src.pronunciation.asr --download
```

## 실행

```bash
python pronunciation_ui.py
```

창을 열지 않고 런타임만 확인하려면:

```bash
python pronunciation_ui.py --check
```

## AI 코칭 설정

AI 코칭 기능은 API 키가 있을 때만 동작합니다. 프로젝트 루트에 `.env` 파일을 만들고 아래처럼 설정합니다.

```text
GEMINI_API_KEY="your_api_key_here"
```

`.env` 파일은 GitHub에 올리지 않습니다.

## 테스트

```bash
python -m pytest
```

## 데이터 안내

개인 녹음 파일, Deeply 원본 음성 데이터셋, Whisper 모델 파일, `.env`는 저장소에 포함하지 않습니다. 필요한 경우 `LOCAL_TESTING.md`를 참고해 로컬에서 데이터셋과 모델을 준비하세요.

