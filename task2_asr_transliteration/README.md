# Task 2 — Tamil ASR + Transliteration

Real-time Tamil speech recognition with romanized transliteration output.  
Supports local Whisper models (CPU/GPU) **and** Groq Cloud (`whisper-large-v3-turbo`) via a single Gradio UI.

---

## Features

- **Dual-backend ASR**: local HuggingFace Whisper (small / medium) or Groq Cloud API
- **5 transliteration schemes**: ITRANS · ISO15919 · IAST · HK · HUNTERIAN
- **Long-audio support**: automatic chunking (30s + 5s stride overlap) for local models
- **JSON download**: every run produces a structured JSON output file
- **Gradio UI**: runs locally or deploys to HuggingFace Spaces in one click

---

## Models

| Model | Backend | Params | VRAM | Notes |
|---|---|---|---|---|
| whisper-small (Local) | HuggingFace | 244M | ~1.5 GB | Fast, good for clear speech |
| whisper-medium (Local) | HuggingFace | 769M | ~3 GB | Higher accuracy |
| whisper-large-v3-turbo ⚡ Groq Cloud | Groq API | 1.5B (distilled) | 0 | Best accuracy, requires `GROQ_API_KEY` |

---

## Quick Start

### 1. Clone & install

```bash
cd task2_asr_transliteration
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add GROQ_API_KEY if you want the cloud model
```

### 3. Run

```bash
python -m app.main
# Open http://localhost:7860
```

### 4. Run with Docker

```bash
docker compose up --build
# Open http://localhost:7860
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Only for Groq model | API key from [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | Only for gated HF models | HuggingFace access token |
| `GRADIO_PORT` | No (default: 7860) | Port to serve the UI on |

---

## Project Structure

```
task2_asr_transliteration/
├── app/
│   ├── asr_pipeline.py      # Dual-backend transcription (local + Groq)
│   ├── buffer_manager.py    # Overlapping audio chunking for long files
│   ├── interface.py         # Gradio UI + process_audio() pipeline
│   ├── transliteration.py   # Tamil → Latin (5 schemes)
│   ├── utils.py             # load_audio, save_output_json
│   └── main.py              # Entry point
├── models/
│   └── model_config.py      # MODEL_CONFIGS dict + scheme/sample-rate constants
├── tests/
│   ├── test_asr_pipeline.py
│   ├── test_buffer_manager.py
│   ├── test_integration.py
│   └── test_transliteration.py
├── sample_inputs/           # Test WAV files (add your own)
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tests

```bash
# From task2_asr_transliteration/
pytest tests/ -v --cov=app --cov=models --cov-report=term-missing
```

No GPU, no API keys, no real audio files needed — all external calls are mocked.

---

## HuggingFace Spaces Deployment

1. Push to a public HF repo
2. Set `GROQ_API_KEY` as a Space secret (Settings → Variables and secrets)
3. Remove `server_name="0.0.0.0"` from `main.py` before pushing (Spaces manages routing)
