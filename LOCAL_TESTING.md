# Local Testing Guide

## 1. Setup

```bash
cd "1HealthCare Project"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For real-time microphone recording, `sounddevice` also needs PortAudio.

Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2
```

macOS:

```bash
brew install portaudio
```

## 2. Run Tests

```bash
python -m pytest -q
```

Expected result in this environment: `19 passed`.

Current expected result after the human-reference update: `32 passed`.

## 3. Real Human Reference Dataset

Download and extract the Deeply Korean read speech corpus sample from OpenSLR SLR97:

```bash
mkdir -p data/datasets/deeply data/datasets/deeply/extracted/Dataset
curl -L --fail https://www.openslr.org/resources/97/KoreanReadSpeechCorpus.tar.gz \
  -o data/datasets/deeply/KoreanReadSpeechCorpus.tar.gz
tar -xzf data/datasets/deeply/KoreanReadSpeechCorpus.tar.gz \
  -C data/datasets/deeply/extracted/Dataset
```

Create the dataset manifests:

```bash
python -m src.pronunciation.deeply_dataset manifest \
  --root data/datasets/deeply/extracted \
  --output outputs/pronunciation/deeply_manifest.csv

python -m src.pronunciation.deeply_dataset reference \
  --manifest outputs/pronunciation/deeply_manifest.csv \
  --output outputs/pronunciation/human_reference_manifest.csv \
  --min-repetitions 2 \
  --max-scripts 80 \
  --max-references-per-script 4
```

The desktop UI automatically prefers `outputs/pronunciation/human_reference_manifest.csv` when it exists, and falls back to the old TTS manifest only when the human manifest is absent.

## 4. File-Based Pronunciation Test

```bash
python -m src.pronunciation.evaluate \
  --sentence-id aa0 \
  --manifest outputs/pronunciation/human_reference_manifest.csv \
  --file data/pronunciation/user_tests/ui_aa0.wav
```

## 5. Real-Time Microphone Test

Desktop UI:

```bash
python pronunciation_ui.py
```

The UI has two tabs: sentence practice and Gemini conversation practice. Sentence practice uses a left-side sentence list, records from the microphone with automatic speech-end detection, and shows the score, text match score, and Korean feedback.
The current UI uses local Whisper ASR through `faster-whisper`, so the main score is based on recognized Korean text vs. the target sentence. The default model is fixed to `medium` because smaller models were too weak for Korean sentence recognition.
Download the model before using the UI:

```bash
python -m src.pronunciation.asr --download
```

The download can take several minutes because the `medium` model is much larger than `tiny` or `base`. The UI does not download the model silently; if the model is missing, it shows a setup message instead of appearing to hang.
After a sentence evaluation finishes, the UI calls Gemini to generate the visible feedback panel from the target sentence, recognized sentence, score, and mismatch terms. In conversation practice, Gemini first creates a Korean question, then evaluates whether the user's recognized answer fits the conversational context.

Gemini setup:

```bash
# edit .env
GEMINI_API_KEY="your_api_key_here"
```

If `GEMINI_API_KEY` is not set in `.env` or the shell environment, the core local pronunciation evaluation still works and the AI coaching button shows a setup message.

Runtime check without opening the window:

```bash
python pronunciation_ui.py --check
```

CLI:

```bash
python -m src.pronunciation.realtime_test \
  --sentence-id aa0 \
  --manifest outputs/pronunciation/human_reference_manifest.csv
```

File-based ASR evaluation:

```bash
python -m src.pronunciation.evaluate \
  --asr \
  --sentence-id aa0 \
  --manifest outputs/pronunciation/human_reference_manifest.csv \
  --file data/pronunciation/user_tests/ui_aa0.wav
```

The command prints the Korean sentence to read, records microphone audio, compares it with the reference clips, and writes:

- `data/pronunciation/user_tests/realtime_aa0.wav`
- `outputs/pronunciation/realtime_result.csv`

## 6. Transformer Training Result

The trained checkpoint is already included:

- `models/pronunciation_transformer.pt`
- `outputs/pronunciation/transformer_metrics.csv`
- `reports/transformer_training_summary.md`
- `outputs/pronunciation/transformer_training_curve.png`

Training summary: 40 repeated Korean script classes, 280 train clips, 80 validation clips, best validation accuracy 85.0%.
