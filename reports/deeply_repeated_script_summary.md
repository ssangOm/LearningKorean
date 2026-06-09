# Deeply Korean Repeated Script Pronunciation Evaluation

## Dataset

| Item | Value |
| --- | ---: |
| Source | Deeply Korean read speech corpus public sample / OpenSLR SLR97 |
| Imported clips | 1697 |
| Speakers | 1 |
| Unique script IDs | 245 |
| Repeated scripts | 226 |

## Evaluation Setup

For each repeated `text_code`, one clean/reference-style recording is selected first, then another recording of the same script is evaluated as the user sample. The same MFCC + DTW similarity baseline from the pronunciation coach is used.

## Results

| Metric | Value |
| --- | ---: |
| Evaluated pairs | 40 |
| Average score | 72.7 |
| Min score | 60.8 |
| Max score | 80.4 |

![Deeply score distribution](../outputs/pronunciation/deeply_score_distribution.png)

## Example Pairs

| text_code   | reference_speaker   | user_speaker   | location_name   |   score | weak_region   | text                                                  |
|:------------|:--------------------|:---------------|:----------------|--------:|:--------------|:------------------------------------------------------|
| aa0         | a                   | a              | AnechoicChamber |    62.7 | 중반            | 저 식당 음식이 정말 맛있나 봐요.                                   |
| aa1         | a                   | a              | AnechoicChamber |    71.1 | 중반            | 아, 저기요. 삼계탕만 파는 식당인데 항상 사람들이 많아요.                     |
| aa2         | a                   | a              | AnechoicChamber |    69.5 | 후반            | 우리 회사 근처에 저런 유명한 식당이 있었네요. 다음에 삼계탕 한번 먹으러 가야겠어요.      |
| aa3         | a                   | a              | AnechoicChamber |    71.1 | 중반            | 저 식당은 그날 준비한 걸 다 팔면 문을 닫아요. 그러니까 늦게 가면 못 드실 수도 있어요.   |
| ab0         | a                   | a              | AnechoicChamber |    71.8 | 중반            | 여권을 만들어야 하는데요. 회사 일이 늦게 끝나서 갈 시간이 없어요.                |
| ab1         | a                   | a              | AnechoicChamber |    76.9 | 초반            | 요즘은 주말에도 여권을 신청할 수 있는 곳이 있어요. 저도 주말에 거기 가서 여권을 만들었어요. |
| ab2         | a                   | a              | AnechoicChamber |    74.8 | 초반            | 그래요? 어디로 가면 돼요?                                       |
| ab3         | a                   | a              | AnechoicChamber |    73.1 | 중반            | 만들어 주는 데가 여러 곳 있어요. 인터넷에서 찾아보고 가까운 곳으로 가세요.           |

## Portfolio Note

This result uses real Korean speaker recordings of repeated scripts, so it directly supports the pronunciation-similarity framing. It replaces synthetic user clips with same-script repeated recordings and keeps the scoring engine unchanged.
