---
title: Tamil ASR Transliteration
emoji: 🎙️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: mit
short_description: Tamil speech → Tamil script → Romanized (5 schemes)
---

# Task 2 — Tamil ASR + Transliteration

Real-time Tamil speech recognition with romanized transliteration output.  
Supports local Whisper models (CPU/GPU) **and** Groq Cloud (`whisper-large-v3-turbo`) through a single Gradio UI.

---

## Features

- **Dual-backend ASR** — local HuggingFace Whisper (small / medium) or Groq Cloud API with zero local GPU
- **5 transliteration schemes** — ITRANS · ISO · IAST · HK · SLP1
- **Long-audio chunking** — automatic 30-second chunks with 5-second stride overlap for local models
- **JSON output** — every transcription run saves a structured `.json` file in `outputs/`
- **Gradio UI** — upload file or use microphone; runs locally or deploys to HuggingFace Spaces

---

## Models

| Model | Backend | Params | VRAM | Notes |
|---|---|---|---|---|
| `whisper-small (Local)` | HuggingFace | 244 M | ~1.5 GB | Fast; good for clear speech |
| `whisper-medium (Local)` | HuggingFace | 769 M | ~3 GB | Higher accuracy; slower |
| `whisper-large-v3-turbo ⚡ Groq Cloud` | Groq API | 1.5 B (distilled) | 0 | Best accuracy; requires `GROQ_API_KEY` |

Local models are downloaded automatically from HuggingFace on first run and cached in `models_cache/`.

---

## Quick Start

### 1. Install dependencies

```bash
cd task2_asr_transliteration
pip install -r requirements.txt
```

Python 3.10+ required. A virtual environment is recommended.

### 2. Configure environment

```bash
cp .env.example .env
# Add GROQ_API_KEY=gsk_... if you want the Groq Cloud model
# Add HF_TOKEN=hf_...   if models require gated HF access
```

### 3. Launch the UI

```bash
python -m app.main
# Open http://localhost:7860
```

### 4. Run with Docker

Reviewers can build and run the container with either:

```bash
# Option A — docker compose (uses .env automatically)
docker compose up --build

# Option B — plain docker (the submission guide style)
docker build -t asr-system .
docker run -p 7860:7860 --env-file .env asr-system
```

Then open http://localhost:7860. The container runs as non-root (UID 1000) so the same
image is compatible with Hugging Face Spaces Docker SDK.

### 5. Deploy to Hugging Face Spaces

This folder is already configured as a Gradio Space — the YAML header at the top of this
README, plus the root-level `app.py`, are everything Spaces needs.

```bash
# One-time: install + login
pip install huggingface_hub
huggingface-cli login

# Create the Space (Gradio SDK)
huggingface-cli repo create tamil-asr-transliteration --type space --space_sdk gradio

# Push from inside task2_asr_transliteration/
git init && git remote add space https://huggingface.co/spaces/<your-username>/tamil-asr-transliteration
git add app.py requirements.txt README.md app/ models/
git commit -m "Deploy Tamil ASR Space"
git push space main
```

In the Space's **Settings → Variables and secrets**, add:
- `GROQ_API_KEY` (only if you want the Groq Cloud backend)
- `HF_TOKEN` (only if using gated models)

> **Note on free tier:** HF Spaces free tier is CPU-only (2 vCPU / 16 GB RAM). Local
> Whisper Small/Medium will run but transcription takes ~30–60 s per clip. For
> sub-second inference use the **Groq Cloud** backend, which runs on Groq's LPUs
> via API regardless of the Space hardware.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | For Groq model only | API key from [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | For gated HF models | HuggingFace access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `MDC_API_KEY` | For Common Voice download | Mozilla Data Collective key from [mozilladatacollective.com/profile/credentials](https://mozilladatacollective.com/profile/credentials) |
| `GRADIO_PORT` | No (default `7860`) | Port for the Gradio server |

---

## Transliteration Schemes

| Scheme | Example (வணக்கம்) | Description |
|---|---|---|
| `ITRANS` | `vaNakkam` | ASCII-friendly; widely used in Indian language computing |
| `ISO` | `vaṇakkam` | ISO 15919 international standard |
| `IAST` | `vaṇakkam` | International Alphabet of Sanskrit Transliteration (scholarly) |
| `HK` | `vaNakkam` | Harvard-Kyoto convention |
| `SLP1` | `vaNakkam` | Sanskrit Library Phonetic encoding |

---

## Project Structure

```
task2_asr_transliteration/
├── app/
│   ├── asr_pipeline.py      # Dual-backend ASR: local HF Whisper + Groq Cloud
│   ├── buffer_manager.py    # Overlapping chunk segmentation for long audio
│   ├── interface.py         # Gradio Blocks UI + process_audio() orchestrator
│   ├── transliteration.py   # Tamil Unicode → Latin (5 schemes)
│   ├── utils.py             # load_audio(), save_output_json()
│   └── main.py              # CLI entry point (local dev)
├── models/
│   └── model_config.py      # MODEL_CONFIGS, SAMPLE_RATE, CHUNK_SECONDS constants
├── tests/
│   ├── test_asr_pipeline.py
│   ├── test_buffer_manager.py
│   ├── test_integration.py
│   └── test_transliteration.py
├── sample_inputs/
│   ├── download_common_voice.py  # Download Common Voice 25.0 Tamil test clips
│   ├── RECORDING_GUIDE.md        # 16 Tamil sentences for personal recordings
│   └── README.md
├── evaluation/
│   ├── evaluate.py          # Batch WER evaluation across all models
│   └── RESULTS_REPORT.md    # WER summary table (populated after evaluation run)
├── outputs/                 # Generated JSON transcription files (gitignored)
├── app.py                   # HuggingFace Spaces entry point
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running Tests

```bash
# From task2_asr_transliteration/
pytest tests/ -v --cov=app --cov=models --cov-report=term-missing
```

No GPU, no API keys, no real audio files needed — all external I/O is mocked.

---

## Batch Evaluation (Common Voice)

```bash
# Step 1 — Accept dataset terms at:
#   https://mozilladatacollective.com/datasets/cmn2gfvyp01geo107izoftfki
# Step 2 — Add MDC_API_KEY to your .env

# Step 3 — Download 50 test clips (~15 MB of WAV after conversion)
python sample_inputs/download_common_voice.py

# Step 4 — Run WER evaluation across all three models
python evaluation/evaluate.py

# Results written to:
#   evaluation/results.csv          per-clip predictions
#   evaluation/RESULTS_REPORT.md    model WER summary table
```

---

## HuggingFace Spaces Deployment

1. Push `task2_asr_transliteration/` contents to a public HF repo
2. Set `GROQ_API_KEY` as a Space secret (Settings → Variables and secrets)
3. The root `app.py` is the Spaces entry point — no edits needed

---

## Personal Recordings

See [`sample_inputs/RECORDING_GUIDE.md`](sample_inputs/RECORDING_GUIDE.md) for 16 Tamil sentences
across four difficulty sets (everyday speech → code-mixed → long utterances).

Record them, save to `sample_inputs/personal/`, then test via the Gradio UI.
