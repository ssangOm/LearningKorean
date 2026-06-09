# Korean Pronunciation Similarity Coach

## Project Overview

| Item | Value |
| --- | --- |
| Goal | 한국어 문장 발음 유사도 평가 및 구간별 피드백 |
| Input | 사용자 문장 녹음 WAV |
| Output | 0~100 유사도 점수, 가장 가까운 기준 문장, 약한 구간, 한국어 피드백 |
| Method | voice-only preprocessing, MFCC + DTW baseline |
| Reference clips | 60 clips / 10 sentences |
| Evaluated samples | 10 files |

## Baseline Result

| Metric | Value |
| --- | ---: |
| Average score | 98.5 |
| Best score | 98.8 |

Dataset note: 현재 평가는 포트폴리오 데모용으로 생성한 변형 TTS user clips 기준이다. 실제 사용자 녹음 WAV를 `data/pronunciation/user_tests/`에 넣으면 같은 CLI로 재평가할 수 있다.

![Score distribution](../outputs/pronunciation/score_distribution.png)

## Example Feedback

| file                    | target_sentence_id   | closest_sentence_id   |   score | weak_region   |
|:------------------------|:---------------------|:----------------------|--------:|:--------------|
| sentence_001_user01.wav | sentence_001         | sentence_001          |    98.4 | 중반            |
| sentence_002_user01.wav | sentence_002         | sentence_002          |    98.4 | 후반            |
| sentence_003_user01.wav | sentence_003         | sentence_003          |    98.4 | 후반            |
| sentence_004_user01.wav | sentence_004         | sentence_004          |    98.8 | 초반            |
| sentence_005_user01.wav | sentence_005         | sentence_005          |    98.3 | 중반            |
| sentence_006_user01.wav | sentence_006         | sentence_006          |    98.4 | 중반            |
| sentence_007_user01.wav | sentence_007         | sentence_007          |    98.5 | 후반            |
| sentence_008_user01.wav | sentence_008         | sentence_008          |    98.4 | 후반            |

## Insight

한국어 문장 발음을 기준 발음과 직접 비교하는 유사도 평가 문제로 정의했다. 기준 발음과 사용자 발음을 같은 전처리로 정규화한 뒤 MFCC sequence를 만들고, DTW alignment로 길이 차이를 흡수하면서 비교한다. 이 구조는 소량의 reference audio만으로도 점수와 약한 구간 피드백을 만들 수 있어 포트폴리오 데모에 적합하다.

## Next Step

Audio Transformer prototype은 log-mel spectrogram을 입력으로 받아 문장 ID embedding을 학습하는 확장 모델로 연결한다. 현재 MVP에서는 MFCC + DTW를 신뢰 가능한 baseline으로 사용한다.
