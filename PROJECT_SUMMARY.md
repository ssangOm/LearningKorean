# 1HealthCare Project 정리

## 1. 프로젝트 개요

`1HealthCare Project`는 기존의 짧은 단어 분류 문제를 확장해서, 한국어 문장 발음을 평가하는 로컬 발음 코칭 프로그램으로 구성한 프로젝트이다.

핵심 목표는 사용자가 한국어 문장을 읽으면 시스템이 음성을 인식하고, 정답 문장과 비교하여 점수와 피드백을 제공하는 것이다. 단순히 단어 하나를 맞히는 분류 문제가 아니라, 문장 단위 발음 평가와 회화 답변 평가까지 포함한다.

프로젝트 한 줄 설명:

```text
로컬 Whisper ASR과 MFCC/DTW 기반 음성 유사도, 그리고 AI 코칭을 결합한 한국어 문장 발음 평가 프로그램
```

## 2. 최종 기능

현재 구현된 주요 기능은 다음과 같다.

| 기능 | 설명 |
| --- | --- |
| 문장 선택 | 기준 문장 목록에서 읽을 문장을 선택 |
| 자동 녹음 | 고정 초 단위가 아니라 음성 시작/침묵 감지를 기반으로 자동 종료 |
| 로컬 음성 인식 | `faster-whisper` 기반 Korean ASR 사용 |
| 문장 일치도 평가 | 정답 문장과 인식 문장을 비교하여 0~100점 산출 |
| 발음/속도 피드백 | 빠진 표현, 다르게 인식된 표현, 발화 길이 등을 바탕으로 피드백 생성 |
| AI 코칭 | `.env`의 API 키를 사용해 자연어 피드백 생성 |
| 대화 연습 | AI가 질문을 만들고, 사용자의 답변이 문맥에 맞는지 평가 |
| 기준 데이터 자동 선택 | 실제 사람 녹음 manifest가 있으면 우선 사용하고, 없으면 TTS reference로 fallback |
| Audio Transformer 실험 | Deeply 반복 문장 데이터로 log-mel Transformer 분류 모델 학습 |

## 3. 프로젝트 구조

```text
1HealthCare Project/
  pronunciation_ui.py                         # PySide6 데스크톱 UI
  LOCAL_TESTING.md                            # 로컬 실행 가이드
  requirements.txt                            # 필요한 Python 패키지
  .env                                        # API 키 설정 파일, 문서에는 키를 기록하지 않음

  src/pronunciation/
    config.py                                 # 기준 문장, 경로, 샘플레이트 설정
    reference_dataset.py                      # TTS 기준 발음 데이터 생성
    deeply_dataset.py                         # Deeply 공개 데이터셋 manifest/reference 생성
    taps_dataset.py                           # TAPS 데이터셋 처리
    similarity.py                             # MFCC + DTW 음성 유사도 계산
    text_similarity.py                        # 정답/인식 문장 텍스트 유사도 계산
    feedback.py                               # 점수, 속도, 발음 피드백 생성
    quality.py                                # 무음/잡음 등 녹음 품질 검사
    asr.py                                    # faster-whisper 로컬 음성 인식
    ai_coach.py                               # AI 발음 코칭 및 대화 평가
    realtime_test.py                          # CLI 실시간 녹음 평가
    evaluate.py                               # 파일/폴더 단위 발음 평가
    transformer.py                            # Audio Transformer 모델
    train_transformer.py                      # Transformer 학습 CLI
    reports.py                                # 결과 리포트 생성

  data/pronunciation/
    references/                               # TTS 기준 발음 wav
    user_tests/                               # 사용자 녹음 wav

  data/datasets/deeply/
    KoreanReadSpeechCorpus.tar.gz             # Deeply Korean read speech corpus
    extracted/Dataset/                        # 압축 해제된 공개 데이터셋

  outputs/pronunciation/
    reference_manifest.csv                    # TTS reference manifest
    human_reference_manifest.csv              # 실제 사람 녹음 기반 reference manifest
    deeply_manifest.csv                       # Deeply 전체 manifest
    evaluation_results.csv                    # TTS demo 평가 결과
    deeply_repeated_script_results.csv        # 실제 반복 문장 평가 결과
    transformer_metrics.csv                   # Transformer 학습 로그
    realtime_result.csv                       # UI/CLI 실시간 평가 결과

  models/
    pronunciation_transformer.pt              # 학습된 Transformer checkpoint

  reports/
    pronunciation_similarity_summary.md       # TTS reference 기반 평가 요약
    deeply_repeated_script_summary.md         # 실제 사람 녹음 평가 요약
    transformer_training_summary.md           # Transformer 학습 요약
    portfolio_pronunciation_summary.md        # 포트폴리오용 전체 요약
```

## 4. 데이터 구성

### 4.1 TTS 기준 문장 데이터

`src/pronunciation/config.py`에는 초기 기준 문장 10개가 정의되어 있다.

예시:

| ID | 문장 | 목적 |
| --- | --- | --- |
| `sentence_001` | 오늘 공기가 맑아요 | 공기/고기 유사 발음 |
| `sentence_002` | 감기에 걸렸어요 | 감기/경기 유사 발음 |
| `sentence_005` | 약을 먹을 시간이에요 | 헬스케어 |
| `sentence_008` | 숨을 크게 쉬어 보세요 | 헬스케어 |
| `sentence_010` | 도움이 필요해요 | 도움 요청 |

TTS reference는 다음 조합으로 생성된다.

```text
10문장 x 2개 voice x 3개 조건(clean, mild, room) = 60개 reference clip
```

현재 파일:

```text
outputs/pronunciation/reference_manifest.csv
```

현재 row 수:

```text
60
```

### 4.2 실제 사람 녹음 기반 Reference

프로젝트는 Deeply Korean read speech corpus 공개 샘플을 활용한다. 같은 script를 여러 번 읽은 실제 한국어 녹음이 있어서, TTS보다 실제 발음 평가 구조에 더 적합하다.

현재 산출물:

| 파일 | row 수 | 의미 |
| --- | ---: | --- |
| `outputs/pronunciation/deeply_manifest.csv` | 1,697 | Deeply 전체 음성 manifest |
| `outputs/pronunciation/human_reference_manifest.csv` | 320 | UI가 우선 사용하는 실제 사람 기준 발음 manifest |
| `outputs/pronunciation/deeply_repeated_script_results.csv` | 40 | 같은 script 반복 녹음 기반 평가 결과 |

`pronunciation_ui.py`는 `human_reference_manifest.csv`가 있으면 이를 우선 사용한다. 없으면 TTS 기반 `reference_manifest.csv`를 사용한다.

## 5. 평가 방식

평가는 크게 두 축으로 구성되어 있다.

### 5.1 음향 유사도: MFCC + DTW

`src/pronunciation/similarity.py`에서 처리한다.

흐름:

```text
reference audio
→ 16 kHz mono load
→ silence trim
→ amplitude normalization
→ MFCC sequence 추출

user audio
→ 같은 전처리
→ MFCC sequence 추출

reference MFCC + user MFCC
→ DTW alignment
→ normalized distance
→ 0~100 score 변환
→ 초반/중반/후반 구간별 점수 계산
```

이 방식의 장점은 발화 속도가 조금 달라도 DTW가 시간축을 정렬해 준다는 점이다. 그래서 사용자가 문장을 조금 빠르게 또는 느리게 읽어도 비교가 가능하다.

### 5.2 텍스트 일치도: Whisper ASR + 문자열 비교

`src/pronunciation/asr.py`와 `src/pronunciation/text_similarity.py`에서 처리한다.

흐름:

```text
사용자 녹음
→ faster-whisper medium 모델로 한국어 인식
→ 정답 문장과 인식 문장 정규화
→ 띄어쓰기 제거 비교 + 문자 유사도 비교
→ 빠진 표현 / 다르게 인식된 표현 추출
→ 텍스트 일치도 점수 계산
```

현재 UI의 핵심 점수는 실제 사용성이 더 좋은 텍스트 일치도 중심이다. 음향 유사도는 보조적으로 발화 길이와 구간 판단에 사용된다.

## 6. AI 코칭

`src/pronunciation/ai_coach.py`에서 처리한다.

사용 모델:

```text
gemini-2.5-flash
```

API 키 설정:

```text
.env
GEMINI_API_KEY="your_api_key_here"
```

AI 코칭은 새 점수를 만들지 않는다. 이미 계산된 정답 문장, 인식 문장, 점수, 빠진 표현, 다르게 인식된 표현을 근거로 자연어 피드백만 생성한다.

출력 정책:

- 별표 강조나 Markdown 문법 제거
- `Gemini`라는 문구는 UI에 노출하지 않음
- 표준 발음상 자연스러운 변화는 감점 사유가 아님을 설명
- 사용자가 다음 녹음에서 바로 고칠 수 있는 팁 제공

대화 연습에서는 AI가 질문을 만들고, 사용자의 답변이 문맥에 맞는지 평가한다.

## 7. UI 구성

실행 파일:

```bash
python pronunciation_ui.py
```

UI는 PySide6 기반 로컬 데스크톱 앱이다.

### 7.1 문장 연습 탭

구성:

1. 문장 목록 페이지
2. 문장 선택 후 평가 페이지로 이동
3. `녹음 및 평가` 버튼
4. 자동 녹음
5. 로컬 Whisper 인식
6. 점수와 텍스트 일치도 표시
7. AI 코칭 표시
8. 인식된 문장과 정답 문장 표시

### 7.2 대화 연습 탭

구성:

1. 탭 진입 시 질문 자동 생성
2. `새 질문` 버튼으로 다른 질문 생성
3. 사용자가 답변 녹음
4. Whisper로 답변 인식
5. AI가 문맥 적합성, 답변 완성도, 발음/전달력, 더 자연스러운 답변 예시 평가

## 8. 자동 녹음 방식

`src/pronunciation/realtime_test.py`의 `record_microphone_until_silence()`가 담당한다.

고정된 녹음 시간이 아니라 RMS 기반으로 음성 시작과 침묵을 감지한다.

주요 설정:

| 설정 | 기본값 | 의미 |
| --- | ---: | --- |
| `frame_duration` | 0.08초 | RMS 계산 단위 |
| `min_record_duration` | 0.7초 | 최소 녹음 시간 |
| `silence_duration` | 0.75초 | 이만큼 침묵이면 종료 |
| `no_speech_timeout` | 5.0초 | 음성이 없으면 중단 |
| `max_duration` | 12.0초 | 최대 녹음 시간 |
| `pre_roll_duration` | 0.24초 | 시작 전 여유 구간 |
| `post_roll_duration` | 0.28초 | 종료 후 여유 구간 |

## 9. Transformer 실험

`src/pronunciation/transformer.py`와 `src/pronunciation/train_transformer.py`에서 처리한다.

목적은 실시간 UI의 주 평가 엔진을 대체하는 것이 아니라, 포트폴리오용으로 최신 딥러닝 구조를 적용한 실험을 추가하는 것이다.

구조:

```text
audio
→ log-mel spectrogram
→ Linear projection
→ CLS token
→ sinusoidal positional encoding
→ TransformerEncoder
→ classification head
```

현재 결과:

| 항목 | 값 |
| --- | ---: |
| 클래스 수 | 40 |
| Train clips | 280 |
| Validation clips | 80 |
| Best validation accuracy | 85.0% |
| Best epoch | 22 |
| Checkpoint | `models/pronunciation_transformer.pt` |

## 10. 현재 결과 요약

### 10.1 TTS Demo 평가

파일:

```text
outputs/pronunciation/evaluation_results.csv
```

결과:

| 항목 | 값 |
| --- | ---: |
| 평가 샘플 | 10 |
| 평균 점수 | 98.5 |
| 최소 점수 | 98.3 |
| 최대 점수 | 98.8 |

주의: 이 결과는 TTS 기반 demo user clip이라 실제 사용자 성능으로 주장하기에는 약하다.

### 10.2 Deeply 실제 녹음 반복 문장 평가

파일:

```text
outputs/pronunciation/deeply_repeated_script_results.csv
```

결과:

| 항목 | 값 |
| --- | ---: |
| 평가 pair | 40 |
| 평균 점수 | 72.7 |
| 최소 점수 | 60.8 |
| 최대 점수 | 80.4 |

이 결과는 실제 한국어 녹음 기반이라 포트폴리오 설명에서 더 설득력이 있다. 다만 현재 데이터는 같은 corpus 안에서 reference와 user sample을 나누는 방식이므로, 완전히 독립적인 사용자 테스트는 아니다.

### 10.3 Transformer 학습

파일:

```text
outputs/pronunciation/transformer_metrics.csv
reports/transformer_training_summary.md
```

결과:

```text
Best validation accuracy: 85.0%
```

## 11. 실행 방법

### 11.1 환경 설정

```bash
cd "/Users/ssangom/Downloads/1HealthCare Project"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS에서 마이크 녹음을 쓰려면 PortAudio가 필요할 수 있다.

```bash
brew install portaudio
```

### 11.2 Whisper 모델 다운로드

현재 기본 ASR 모델은 `medium`으로 고정되어 있다.

```bash
python -m src.pronunciation.asr --download
```

다운로드가 끝나기 전에는 UI에서 모델 준비 안내가 뜰 수 있다.

### 11.3 UI 실행

```bash
python pronunciation_ui.py
```

윈도우를 열지 않고 런타임 체크만 할 때:

```bash
python pronunciation_ui.py --check
```

### 11.4 파일 기반 평가

```bash
python -m src.pronunciation.evaluate \
  --asr \
  --sentence-id aa0 \
  --manifest outputs/pronunciation/human_reference_manifest.csv \
  --file data/pronunciation/user_tests/ui_aa0.wav
```

### 11.5 CLI 실시간 녹음 평가

```bash
python -m src.pronunciation.realtime_test \
  --sentence-id aa0 \
  --manifest outputs/pronunciation/human_reference_manifest.csv
```

## 12. 테스트

테스트 파일은 `tests/` 아래에 기능별로 분리되어 있다.

주요 테스트 범위:

- ASR 모델 다운로드/로딩 처리
- AI 코칭 prompt와 출력 정리
- 문장 텍스트 유사도
- 녹음 자동 종료 조건
- MFCC/DTW 유사도
- 평가 결과 CSV
- UI 문장 목록/대화 prompt 동작
- Deeply/TAPS dataset manifest 생성
- Transformer 학습 루틴

실행:

```bash
python -m pytest -q
```

## 13. 포트폴리오에서 강조할 점

발표/포트폴리오에서는 다음 흐름으로 설명하는 것이 좋다.

1. 처음에는 비슷한 단어 분류 문제였지만, 실제 서비스 목적에는 문장 발음 평가가 더 적합하다고 판단했다.
2. 그래서 문제를 `단어 분류`에서 `문장 발음 유사도 평가`로 재정의했다.
3. 기준 발음과 사용자 발음을 MFCC sequence로 변환하고, DTW로 시간축을 정렬해 비교했다.
4. 로컬 Whisper ASR을 추가해 실제 사용자가 읽은 문장을 텍스트로 검증했다.
5. 단순 점수뿐 아니라 AI 코칭으로 사용자가 이해할 수 있는 피드백을 제공했다.
6. TTS reference에서 시작했지만, 실제 사람 녹음 기반 Deeply dataset도 추가해 합성음 의존도를 줄였다.
7. 추가로 Audio Transformer를 학습해 최신 딥러닝 구조를 적용한 실험도 포함했다.

포트폴리오용 핵심 문장:

```text
이 프로젝트는 한국어 문장 발음을 로컬에서 인식하고, 정답 문장과의 유사도를 평가한 뒤, 사용자가 이해할 수 있는 AI 피드백까지 제공하는 발음 코칭 시스템입니다.
```

## 14. 현재 한계

| 한계 | 설명 |
| --- | --- |
| Whisper medium 속도 | CPU 환경에서는 첫 실행과 인식 시간이 길 수 있음 |
| 외부 사용자 검증 부족 | 실제 사용자의 독립 테스트셋은 아직 충분하지 않음 |
| 발음 세부 분석 한계 | 음소 단위 오류 검출보다는 문장/단어 단위 피드백 중심 |
| AI 코칭 의존성 | 자연어 피드백은 API 키와 네트워크가 필요 |
| Transformer 역할 | 현재는 실시간 평가 엔진보다 포트폴리오용 학습 실험에 가까움 |

## 15. 다음 작업 제안

우선순위가 높은 순서:

1. 실제 사용자 녹음 30~50개를 별도 test set으로 모아 `human_reference_manifest.csv` 기준 평가
2. UI에서 평가 결과를 날짜별로 저장하고 이전 결과와 비교
3. Whisper 인식 시간 단축을 위해 `small`/`medium` 선택 옵션 또는 사전 로딩 상태 표시 개선
4. Deeply/TAPS 데이터를 기준으로 더 안정적인 reference selection 기준 정리
5. Transformer embedding을 발음 유사도 점수와 결합하는 실험
6. 포트폴리오용 PPT/README에 현재 수치와 시연 흐름 반영

