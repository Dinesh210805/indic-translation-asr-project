# Task 2 — Runbook

Step-by-step guide for setting up, running, testing, and evaluating the Tamil ASR + Transliteration system.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| pip | latest | `pip install --upgrade pip` |
| ffmpeg | any | required by librosa for MP3 decoding |
| Docker + Compose | optional | for containerised run |

Install ffmpeg if missing:
```bash
# Windows (winget)
winget install --id Gyan.FFmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## 1. Environment Setup

```bash
cd task2_asr_transliteration

# Create and activate venv (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Copy the environment template and fill in keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_...          # required for Groq Cloud model
HF_TOKEN=hf_...               # only if HuggingFace model is gated
MDC_API_KEY=...               # only for Common Voice download
GRADIO_PORT=7860              # optional, default 7860
```

Get keys:
- Groq: `https://console.groq.com`
- HuggingFace: `https://huggingface.co/settings/tokens`
- Mozilla Data Collective: `https://mozilladatacollective.com/profile/credentials`

---

## 2. Run Locally

```bash
python -m app.main
```

Open `http://localhost:7860` in your browser.

**Using the UI:**
1. Upload a WAV/MP3 file **or** click the microphone to record
2. Select a Whisper model from the dropdown
3. Select a transliteration scheme (ITRANS is the most readable)
4. Click **Transcribe + Transliterate**
5. Tamil transcript appears on the left, romanized output on the right
6. Download the JSON output with the file button below

---

## 3. Run with Docker

```bash
docker compose up --build
# Open http://localhost:7860
```

Model weights are cached in `./models_cache/` (mounted volume) so they survive container restarts.

To stop:
```bash
docker compose down
```

---

## 4. Run Tests

```bash
pytest tests/ -v --cov=app --cov=models --cov-report=term-missing
```

All tests run without GPU, API keys, or real audio files — external calls are mocked.

Expected output: all tests pass, coverage ≥ 80%.

---

## 5. Download Common Voice Clips

Common Voice 25.0 Tamil is available through the Mozilla Data Collective.

**One-time setup (required):**
1. Create an account at `https://mozilladatacollective.com`
2. Open the dataset page: `https://mozilladatacollective.com/datasets/cmn2gfvyp01geo107izoftfki`
3. Sign in and accept the terms / click the download button on the page
4. Copy your API key from `https://mozilladatacollective.com/profile/credentials`
5. Add `MDC_API_KEY=your_key` to `.env`

Then run:

```bash
python sample_inputs/download_common_voice.py
```

This streams the dataset tar.gz, extracts the test split, converts 50 MP3 clips to 16 kHz mono WAV, and writes:
- `sample_inputs/common_voice/*.wav` — audio clips (gitignored)
- `sample_inputs/common_voice/index.csv` — ground-truth sentences (committed)

---

## 6. Run WER Evaluation

After downloading clips:

```bash
python evaluation/evaluate.py
```

This runs all 50 clips through all three models and writes:
- `evaluation/results.csv` — per-clip transcript + WER for each model
- `evaluation/RESULTS_REPORT.md` — markdown summary with average WER table

**Expected runtimes (CPU):**
- whisper-small × 50 clips: ~3 min
- whisper-medium × 50 clips: ~8 min
- Groq Cloud × 50 clips: ~2 min (network-bound, 30 req/min rate limit)

---

## 7. Personal Recording Workflow

1. Read `sample_inputs/RECORDING_GUIDE.md` for 16 Tamil sentences across 4 difficulty levels
2. Record each sentence and save to `sample_inputs/personal/` as `p01_greeting.wav`, `p02_weather.wav`, etc.
3. Test each file via the Gradio UI
4. Fill in the **Personal Recording Results** table in `evaluation/RESULTS_REPORT.md`

---

## 8. Deploy to HuggingFace Spaces

1. Create a new Space at `https://huggingface.co/spaces` (SDK: Gradio, Python 3.11)
2. Push the contents of `task2_asr_transliteration/` to the Space repo
3. Add `GROQ_API_KEY` as a Space secret (Settings → Variables and secrets)
4. The Space entry point is `app.py` at the root — no changes needed

The app will start automatically. First run downloads Whisper model weights (~500 MB for small, ~1.5 GB for medium).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `AttributeError: module 'sanscript' has no attribute 'ISO15919'` | Use scheme `ISO` instead — `ISO15919` was renamed in indic-transliteration v2.3+ |
| `GROQ_API_KEY not set` | Add `GROQ_API_KEY=gsk_...` to `.env` and restart |
| `403 Forbidden` from download script | Accept dataset terms at the MDC dataset page first |
| Gradio `theme` UserWarning | Cosmetic only — app still works; fixed in `main.py` by passing `theme` to `launch()` |
| `Out of memory` with whisper-medium | Use whisper-small or the Groq Cloud model (requires no local GPU) |
| First run is slow | Model weights are being downloaded — subsequent runs use the cache |
