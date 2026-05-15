# Project 2 Blueprint v1.0: ASR + Transliteration System
## Full Implementation Guide — Docker + Gradio + Whisper + Indic-Transliteration

---

## WHAT THIS BLUEPRINT COVERS

- Every library documented with architecture, quirks, and exact API patterns
- Complete code for every file in the project
- Docker + docker-compose configuration
- Buffer queue design for chunked audio
- **Dual backend ASR**: local Whisper (small/medium) + Groq Cloud (whisper-large-v3-turbo)
- Gradio UI wiring (port 7860, audio upload, ⚡ Groq Cloud badge)
- HuggingFace Spaces deployment (free 1-click, best for evaluator demos)
- Test suite structure
- Execution order and known failure modes
- Environment variable handling (HF_TOKEN + GROQ_API_KEY — never committed)

---

## SYSTEM OVERVIEW

```
Audio Input (wav/mp3/ogg)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Gradio UI — Model Selector Dropdown                 │
│  "whisper-small (Local)"         → backend="local"  │
│  "whisper-medium (Local)"        → backend="local"  │
│  "whisper-large-v3-turbo ⚡ Groq" → backend="groq"  │
└───────────┬──────────────────────┬───────────────────┘
            │ backend="local"      │ backend="groq"
            ▼                      ▼
┌───────────────────┐   ┌───────────────────────────────┐
│  Buffer Manager   │   │  Groq Cloud API               │
│  queue.Queue      │   │  whisper-large-v3-turbo (1.5B) │
│  30s chunks       │   │  Sends raw file path directly  │
│  5s stride        │   │  Handles long audio server-side│
│  numpy float32    │   │  Requires GROQ_API_KEY         │
└────────┬──────────┘   └───────────┬───────────────────┘
         │                          │
         ▼                          │
┌──────────────────┐                │
│  Local Whisper   │                │
│  HF transformers │                │
│  pipeline()      │                │
└────────┬─────────┘                │
         │                          │
         └─────────────┬────────────┘
                       │  Tamil Unicode string
                       ▼
          ┌──────────────────────────────┐
          │  Transliteration Engine      │
          │  indic-transliteration       │
          │  Tamil → ITRANS/ISO15919/etc │
          └──────────┬───────────────────┘
                     │  Romanized string
                     ▼
          ┌──────────────────────────────┐
          │  Gradio UI output (port 7860)│
          │  Tamil transcript + Roman    │
          │  JSON download button        │
          └──────────────────────────────┘
```

---

## COMPONENT DOCUMENTATION (READ BEFORE CODING)

---

### COMPONENT 1 — openai/whisper-small (ASR Model)

**HuggingFace:** https://huggingface.co/openai/whisper-small
**Paper:** https://arxiv.org/abs/2212.04356 (Radford et al., 2022 — "Robust Speech Recognition via Large-Scale Weak Supervision")
**Architecture docs:** https://huggingface.co/docs/transformers/model_doc/whisper
**OpenAI blog:** https://openai.com/research/whisper

#### What It Is
Whisper is an encoder-decoder transformer trained by OpenAI on 680,000 hours of multilingual
speech. The training data is sourced from the internet with automatic quality filtering.
It supports 99 languages including Tamil (`ta`).

`whisper-small` has 244M parameters (~244MB), `whisper-medium` has 769M parameters (~769MB).
Both are fully supported on CPU (slower) and GPU (fast).

#### Architecture
- **Encoder**: 80-channel log-Mel spectrogram → positional encoding → transformer encoder
  - Converts raw waveform to a fixed 30-second spectrogram window
  - Audio longer than 30 seconds is internally chunked
- **Decoder**: Autoregressive transformer decoder → generates token IDs → BPE text output
- **Tokenizer**: Byte-level BPE with 51,865 tokens covering all supported languages
- **Input**: 16kHz mono audio, automatically resampled if needed
- **Language**: Explicitly set `language="ta"` to force Tamil output (avoids mis-detection)
- **Task**: Set `task="transcribe"` (not "translate" — we want Tamil text, not English)

#### Key Quirks
1. **Always set `language="ta"` and `task="transcribe"`** explicitly. Without this, Whisper
   may auto-detect the language and fall back to English transcription on noisy audio.
2. **Input sampling rate must be 16kHz**. `librosa.load(path, sr=16000)` handles this.
   If you use `torchaudio` or `soundfile`, always resample to 16000.
3. **return_timestamps=True** is needed if you want word-level timestamps. For basic
   transcription without timestamps, omit it — it adds overhead.
4. **Long audio chunking**: The `WhisperProcessor` + `pipeline()` approach handles long audio
   automatically via `chunk_length_s=30`. If using the model directly, you must chunk manually.
5. **GPU memory**: whisper-small requires ~1.5GB VRAM, whisper-medium ~3GB.
   On CPU, small takes ~10x real-time, medium ~30x real-time.

#### Two API Patterns

**Pattern A — pipeline() (recommended for this project)**
```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
    chunk_length_s=30,        # handles audio longer than 30s
    stride_length_s=5,        # overlap to avoid boundary artifacts
    generate_kwargs={
        "language": "ta",
        "task": "transcribe",
    }
)

result = asr("path/to/audio.wav")
transcript = result["text"]
```

**Pattern B — manual model + processor (needed when intercepting chunks)**
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa, torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()

audio, sr = librosa.load("audio.wav", sr=16000)  # MUST be 16kHz
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
forced_bos = processor.get_decoder_prompt_ids(language="ta", task="transcribe")
with torch.no_grad():
    ids = model.generate(
        inputs["input_features"],
        forced_decoder_ids=forced_bos,
        max_new_tokens=444,   # ~30 seconds of speech
    )
transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
```

#### Model Sizes and Expected Quality

| Model | Params | VRAM | WER Tamil (approx) | Best for |
|-------|--------|------|---------------------|---------|
| whisper-tiny | 39M | ~0.5GB | High error | Testing only |
| whisper-small | 244M | ~1.5GB | Medium error | Default for this project |
| whisper-medium | 769M | ~3GB | Low error | Better quality option |
| whisper-large-v3 | 1.5B | ~6GB | Very low error | Production, not needed here |

#### HuggingFace Cache Location
Models cache to `~/.cache/huggingface/hub/` by default.
Inside Docker, set `TRANSFORMERS_CACHE=/app/models` so the model survives container restarts
when you bind-mount the models directory.

---

### COMPONENT 2 — openai/whisper-medium (alternate ASR Model)

**HuggingFace:** https://huggingface.co/openai/whisper-medium
Same architecture as whisper-small, 3x the parameters.
Use identical API — only change `"openai/whisper-small"` → `"openai/whisper-medium"`.
Load conditionally based on user selection in Gradio UI dropdown.

---

### COMPONENT 3 — indic-transliteration Library

**PyPI:** https://pypi.org/project/indic-transliteration/
**GitHub:** https://github.com/sanskrit-coders/indic_transliteration
**Docs:** https://indic-transliteration.readthedocs.io/

#### What It Is
`indic-transliteration` is a Python library for converting between Indic scripts and
romanization schemes. It supports 20+ languages and 10+ transliteration schemes.
For this project we use it with Tamil (`TAMIL` or `DEVANAGARI` script detection +
`HK`, `IAST`, `ITRANS`, `ISO15919`, `SLP1`, or `HUNTERIAN` romanization).

#### Installation
```
pip install indic-transliteration
```

#### Script and Scheme Constants
```python
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Script constants (source script)
sanscript.TAMIL          # Tamil Unicode script
sanscript.DEVANAGARI     # Devanagari (Hindi, Sanskrit)
sanscript.TELUGU
sanscript.KANNADA
sanscript.BENGALI

# Romanization scheme constants (target)
sanscript.IAST           # International Alphabet of Sanskrit Transliteration
sanscript.ISO15919       # ISO 15919:2001 — standard for official use
sanscript.ITRANS         # Popular ASCII scheme, common in email/SMS
sanscript.HK             # Harvard-Kyoto ASCII scheme
sanscript.SLP1           # Sanskrit Library Phonological notation
sanscript.HUNTERIAN      # Hunterian — traditional colonial romanization
sanscript.VELTHUIS       # TeX-style
```

#### Core API — Tamil → Latin (Romanization)
```python
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

tamil_text = "வணக்கம்"    # "vanakkam" (hello)

# Option 1: ITRANS (ASCII-safe, common)
latin_itrans = transliterate(tamil_text, sanscript.TAMIL, sanscript.ITRANS)
# → "vaNakkam"

# Option 2: ISO 15919 (standard for this project)
latin_iso = transliterate(tamil_text, sanscript.TAMIL, sanscript.ISO15919)
# → "vaṇakkam"

# Option 3: IAST (diacritics, scholarly)
latin_iast = transliterate(tamil_text, sanscript.TAMIL, sanscript.IAST)
# → "vaṇakkam"
```

#### Core API — Latin → Tamil (Reverse Transliteration)
```python
# ITRANS romanization → Tamil script
roman_text = "vaNakkam"
tamil_back = transliterate(roman_text, sanscript.ITRANS, sanscript.TAMIL)
# → "வணக்கம்"

# ISO 15919 → Tamil (less reliable due to diacritics parsing)
iso_text = "vaṇakkam"
tamil_from_iso = transliterate(iso_text, sanscript.ISO15919, sanscript.TAMIL)
```

#### Key Quirks
1. **ITRANS is the most reliable** for round-trip Tamil → Latin → Tamil conversion.
   ISO 15919 has diacritic ambiguities that can fail on re-import.
2. **No model weights** — this is pure rule-based string substitution. Zero GPU requirement,
   sub-millisecond latency per word. Safe to call synchronously.
3. **Tamil vowel marks (matras) are handled correctly** — the library encodes the full
   Unicode composition rules for Tamil.
4. **Mixed script input** — if the ASR outputs some English words mixed with Tamil
   (code-switching), `transliterate()` will pass non-Tamil characters through unchanged.
   This is correct behavior, not a bug.
5. **Empty string safety** — `transliterate("", ...)` returns `""` safely. Always check for
   empty transcript before calling (Whisper may return `""` on silent audio or music).

#### All Supported Tamil Transliteration Schemes

| Scheme Constant | Output Example for "வணக்கம்" | Use case |
|----------------|-------------------------------|---------|
| ITRANS | vaNakkam | ASCII email/chat |
| ISO15919 | vaṇakkam | Academic/formal |
| IAST | vaṇakkam | Sanskrit/Indic scholars |
| HK | vaNakkam | Harvard-Kyoto, some academia |
| HUNTERIAN | wanakkam | Colonial-era romanization |

---

### COMPONENT 4 — queue.Queue (Buffer Manager)

**Python stdlib docs:** https://docs.python.org/3/library/queue.html

#### What It Is
Python's thread-safe FIFO queue. Used here to:
1. Accept incoming audio chunks from Gradio's async file upload
2. Hold chunks while GPU processes earlier ones
3. Prevent memory overflow on very long audio files (>5 minutes)
4. Enable future streaming extension (producer/consumer pattern)

#### Why It Is Needed
Gradio's audio component delivers a complete file path, not a stream. But for robustness:
- Very long audio (>10 min) cannot fit in GPU memory as one 30-second spectrogram window
- The buffer queue lets the pipeline process one 30s chunk at a time and join results
- The architecture is also streaming-ready if we add a microphone input later

#### API Pattern for This Project
```python
import queue
import numpy as np

# Create bounded queue (prevents unbounded memory growth)
audio_buffer = queue.Queue(maxsize=50)  # 50 chunks × 30s = 25 minutes max

def enqueue_chunks(audio_array: np.ndarray, chunk_size: int = 16000 * 30) -> int:
    """Split audio into 30-second chunks and enqueue them."""
    total = len(audio_array)
    n_chunks = 0
    for start in range(0, total, chunk_size):
        chunk = audio_array[start:start + chunk_size]
        if len(chunk) < 1600:   # skip chunks shorter than 0.1s (silence/noise)
            continue
        audio_buffer.put(chunk, block=True, timeout=5)
        n_chunks += 1
    return n_chunks

def process_queue(asr_fn) -> str:
    """Drain queue through ASR, concatenate results."""
    transcripts = []
    while not audio_buffer.empty():
        chunk = audio_buffer.get(block=True, timeout=1)
        text = asr_fn(chunk)
        transcripts.append(text.strip())
        audio_buffer.task_done()
    return " ".join(t for t in transcripts if t)
```

#### Key Quirks
1. **maxsize=50** sets a hard cap. If 50 chunks are queued and not consumed, `put()` blocks.
   For a Gradio app processing one file at a time, this never triggers — but it prevents bugs.
2. **task_done()** must be called after each `get()` if you use `queue.join()` for
   synchronization. We call it for hygiene even though we don't `join()`.
3. **Thread safety**: `queue.Queue` is fully thread-safe. Gradio runs each request in a
   separate thread, so the queue must be re-created per request (not shared globally).
   See implementation in `buffer_manager.py`.

---

### COMPONENT 5 — Gradio (UI Framework)

**PyPI:** https://pypi.org/project/gradio/
**Docs:** https://www.gradio.app/docs
**GitHub:** https://github.com/gradio-app/gradio

#### What It Is
Gradio is a Python library for building ML demos with a web UI. It creates a local web server
(default port 7860) with pre-built UI components for file upload, audio recording, text display,
and more. No HTML/CSS/JS required.

#### Installation
```
pip install gradio>=4.0.0
```

#### Core Blocks API (used in this project)
```python
import gradio as gr

with gr.Blocks(title="Tamil ASR + Transliteration") as demo:
    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],  # both upload and mic record
            type="filepath",                    # returns path string to temp file
            label="Upload Tamil Audio",
        )
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=["whisper-small", "whisper-medium"],
            value="whisper-small",
            label="Whisper Model",
        )
        scheme_selector = gr.Dropdown(
            choices=["ITRANS", "ISO15919", "IAST", "HK", "HUNTERIAN"],
            value="ITRANS",
            label="Transliteration Scheme",
        )
    with gr.Row():
        run_btn = gr.Button("Transcribe + Transliterate", variant="primary")
    with gr.Row():
        transcript_out = gr.Textbox(label="Tamil Transcript", lines=6, interactive=False)
        transliterated_out = gr.Textbox(label="Romanized (Transliterated)", lines=6, interactive=False)
    with gr.Row():
        download_btn = gr.File(label="Download Output JSON")

    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, model_selector, scheme_selector],
        outputs=[transcript_out, transliterated_out, download_btn],
    )

demo.launch(server_port=7860, server_name="0.0.0.0")  # 0.0.0.0 required inside Docker
```

#### Key Quirks
1. **`server_name="0.0.0.0"`** is MANDATORY inside Docker. Without it, Gradio binds to
   `localhost` inside the container and is unreachable from the host.
2. **`type="filepath"`** in `gr.Audio` returns a temp file path string. The temp file
   exists only for the duration of the request. Load it before the response returns.
3. **Gradio 4.x vs 3.x API**: `gr.Audio(sources=[...])` is the Gradio 4 API. In Gradio 3,
   it was `gr.Audio(source="upload")`. Pin to `gradio>=4.0.0` in requirements.
4. **`share=False`** (default) — do not set `share=True` in Docker as it requires outbound
   tunnel to Gradio's servers. Keep local.
5. **Queue for concurrent requests**: Gradio 4+ has a built-in queue. Add `demo.queue()` before
   `demo.launch()` if multiple users will use the app simultaneously.

---

### COMPONENT 6 — Groq Cloud Whisper API

**Groq Console:** https://console.groq.com/
**SDK PyPI:** https://pypi.org/project/groq/
**Supported models:** https://console.groq.com/docs/speech-text
**API reference:** https://console.groq.com/docs/openai

#### What It Is
Groq Cloud is an AI inference platform that runs Whisper models on custom LPU™ hardware.
The key model for this project is `whisper-large-v3-turbo` — a 1.5B parameter distilled
version of Whisper Large v3. Groq's LPU delivers near-instant inference: a 30-second audio
clip typically transcribes in under 1 second.

This is the **third model option** in the UI alongside the two local models. It lets users
test the largest and most accurate Whisper variant without local GPU requirements.

#### Why Use Groq Here
1. **whisper-large-v3-turbo** is not practical to run locally (6GB VRAM, slow on CPU)
2. Groq's free tier is generous — enough for a demo
3. Adds a compelling "best-of-class cloud inference" option to the submission
4. The `⚡ Groq Cloud` label in the UI makes it visually clear it's different from local models
5. Same Whisper family, so evaluators see apples-to-apples quality comparison

#### Installation
```bash
pip install groq>=0.4.0
```

#### API Pattern — Exact Code for This Project
```python
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def transcribe_with_groq(audio_file_path: str) -> str:
    """Transcribe Tamil audio via Groq Cloud whisper-large-v3-turbo.

    Args:
        audio_file_path: path to audio file (WAV/MP3/OGG/FLAC), max 100MB

    Returns:
        Tamil transcript string
    """
    with open(audio_file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_file_path), f),
            model="whisper-large-v3-turbo",
            language="ta",           # force Tamil — same rule as local Whisper
            response_format="text",  # returns plain string, not JSON object
        )
    return result.strip() if result else ""
```

#### Key Quirks
1. **Takes a file path, not a numpy array.** Unlike the local `pipeline()` which accepts
   `{"array": np_array, "sampling_rate": 16000}`, the Groq API takes an opened file handle.
   This means `load_audio()` + `AudioBufferManager` are NOT needed for the Groq backend.
   Pass the original Gradio temp file path directly to `transcribe_with_groq()`.
2. **Skip the buffer manager for Groq.** Groq handles audio up to 100MB server-side.
   Chunking is unnecessary and would complicate the code. Just pass the full file.
3. **`response_format="text"`** returns a plain string (not a dict). With `response_format="json"`
   it returns an object with `.text` attribute. Use `"text"` for simplicity.
4. **`language="ta"`** is still required — Groq passes this to Whisper's decoder.
   Without it, short Tamil audio may be mis-detected as another language.
5. **GROQ_API_KEY env var** — get from https://console.groq.com/keys. Free tier available.
   Add to `.env` (never commit). Pattern: `gsk_...` (starts with gsk_, not hf_).
6. **Error handling** — `groq.APIStatusError` is raised on auth failure or quota exhaustion.
   Catch it and return a user-friendly message in `process_audio()`.
7. **No model download.** Using Groq backend requires no local model weights. The `HF_TOKEN`
   is still needed for the local models, but Groq runs entirely in the cloud.

#### Authentication
```python
# In .env (never commit):
GROQ_API_KEY=gsk_your_key_here

# In code:
import os
from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Raises groq.AuthenticationError if key is missing or invalid
```

#### Model Comparison

| Model | Backend | Params | VRAM | Speed | Tamil Quality |
|-------|---------|--------|------|-------|--------------|
| whisper-small | Local HF | 244M | 1.5GB | ~10x realtime CPU | Good |
| whisper-medium | Local HF | 769M | 3.0GB | ~30x realtime CPU | Better |
| whisper-large-v3-turbo | ⚡ Groq Cloud | 1.5B | N/A (cloud) | <1s per clip | Best |

---

### COMPONENT 7 — UI Framework Decision (Gradio)

#### What the PDF Says

The assignment PDF (page 5) states:
> "Interactive Interface: Gradio is **highly recommended** for its minimal code,
> Hugging Face friendliness, and audio upload support."

"Highly recommended" — not mandatory. But in practice, Gradio is the right choice here.

#### Why Gradio (Not Streamlit or FastAPI)

| Criterion | Gradio | Streamlit | FastAPI |
|-----------|--------|-----------|---------|
| PDF recommendation | **Explicit** "highly recommended" | Not mentioned | Not mentioned |
| Audio upload component | Built-in `gr.Audio` | Requires 3rd-party widget | Manual HTML |
| Mic recording | Built-in | No native support | No native support |
| HuggingFace Spaces | **Native 1-click deploy** | Supported but not native | Not supported |
| Code needed | ~40 lines | ~60 lines | ~200+ lines |
| Evaluator familiarity | High (ML standard) | Medium | Low (no demo UI) |

**Decision: Gradio.** The combination of native audio support, HF Spaces compatibility,
and explicit PDF endorsement makes it the only sensible choice.

#### HuggingFace Spaces Deployment (Free)

Spaces is the fastest way to get a live demo that evaluators can click. Free tier is
sufficient for a Gradio app this size.

**Steps to deploy:**

1. Create a Space at https://huggingface.co/new-space
   - SDK: **Gradio**
   - Hardware: **CPU Basic** (free) — sufficient for local whisper-small
   - Visibility: Public

2. Add secrets in Space Settings → Repository secrets:
   ```
   HF_TOKEN = hf_your_token_here
   GROQ_API_KEY = gsk_your_key_here
   ```
   (Secrets are injected as env vars at runtime — same pattern as `.env`)

3. Push code to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USER/tamil-asr
   git push space main
   ```
   Or simply upload the files via the Space's web UI.

4. Spaces expects either:
   - `app.py` at repo root that calls `demo.launch()`, OR
   - A `README.md` with `sdk: gradio` frontmatter pointing to the entry file

   **For this project:** create `app.py` at `task2_asr_transliteration/` root that imports and launches:
   ```python
   # app.py (HuggingFace Spaces entry point)
   import sys, os
   sys.path.insert(0, os.path.dirname(__file__))
   from app.interface import build_ui
   demo = build_ui()
   demo.queue()
   demo.launch()  # No server_name needed on Spaces — HF handles routing
   ```

5. Spaces auto-rebuilds on every `git push`. View logs in the Space's "Logs" tab.

**Important**: On Spaces, do NOT set `server_name="0.0.0.0"` — Spaces injects its own
proxy. Use the default `demo.launch()` with no arguments, or `demo.launch(server_name="0.0.0.0")`
only for Docker local runs.

#### Port 7860

The PDF's Docker run command `docker run -p 7860:7860 asr-system` confirms port 7860 is
the expected port. Gradio's default is 7860, which is why the PDF uses it. Keep this port.

---

## REPOSITORY STRUCTURE

```
task2_asr_transliteration/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Entry point — launches Gradio
│   ├── asr_pipeline.py          # Whisper ASR logic
│   ├── transliteration.py       # indic-transliteration wrapper
│   ├── buffer_manager.py        # queue.Queue chunking logic
│   ├── interface.py             # Gradio Blocks UI definition
│   └── utils.py                 # Audio loading, JSON export, helpers
├── models/
│   └── model_config.py          # Model name constants and config dict
├── static/                      # Reserved for custom CSS/JS if needed
├── templates/                   # Reserved for custom HTML if needed
├── sample_inputs/               # Sample Tamil audio files for testing
│   └── README.md                # Instructions to get sample audio
├── outputs/                     # JSON outputs written here
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   ├── test_asr_pipeline.py
│   ├── test_transliteration.py
│   ├── test_buffer_manager.py
│   └── test_integration.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example                 # Template — never commit .env
├── README.md
└── Project2_Blueprint_v1.md     # This file
```

---

## FILE IMPLEMENTATIONS (FULL CODE)

---

### `models/model_config.py`

```python
MODEL_CONFIGS = {
    "whisper-small (Local)": {
        "backend": "local",
        "hf_id": "openai/whisper-small",
        "params": "244M",
        "vram_gb": 1.5,
        "description": "Fast, good quality for clear Tamil speech",
    },
    "whisper-medium (Local)": {
        "backend": "local",
        "hf_id": "openai/whisper-medium",
        "params": "769M",
        "vram_gb": 3.0,
        "description": "Higher accuracy, slower inference",
    },
    "whisper-large-v3-turbo ⚡ Groq Cloud": {
        "backend": "groq",
        "groq_id": "whisper-large-v3-turbo",
        "params": "1.5B (distilled)",
        "vram_gb": 0,           # runs on Groq Cloud — no local VRAM needed
        "description": "Highest accuracy via Groq Cloud API — requires GROQ_API_KEY",
    },
}

TRANSLITERATION_SCHEMES = ["ITRANS", "ISO15919", "IAST", "HK", "HUNTERIAN"]
DEFAULT_MODEL = "whisper-small (Local)"
DEFAULT_SCHEME = "ITRANS"
SAMPLE_RATE = 16000         # Whisper requires 16kHz
CHUNK_SECONDS = 30          # 30-second audio windows (Whisper's native window)
STRIDE_SECONDS = 5          # Overlap to prevent boundary word drops
MIN_CHUNK_SAMPLES = 1600    # 0.1s minimum chunk — skip pure silence
```

---

### `app/buffer_manager.py`

```python
import queue
import numpy as np
from models.model_config import SAMPLE_RATE, CHUNK_SECONDS, MIN_CHUNK_SAMPLES


class AudioBufferManager:
    """Thread-local buffer queue for chunked audio processing.

    One instance per request — never share across Gradio requests.
    """

    def __init__(self, maxsize: int = 50):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.chunk_samples = SAMPLE_RATE * CHUNK_SECONDS

    def enqueue(self, audio: np.ndarray) -> int:
        """Split audio array into 30-second chunks and enqueue. Returns chunk count."""
        n = 0
        for start in range(0, len(audio), self.chunk_samples):
            chunk = audio[start : start + self.chunk_samples]
            if len(chunk) < MIN_CHUNK_SAMPLES:
                continue
            self._q.put(chunk, block=True, timeout=10)
            n += 1
        return n

    def drain(self) -> list[np.ndarray]:
        """Return all queued chunks as a list, clearing the queue."""
        chunks = []
        while not self._q.empty():
            try:
                chunk = self._q.get(block=False)
                chunks.append(chunk)
                self._q.task_done()
            except queue.Empty:
                break
        return chunks

    @property
    def size(self) -> int:
        return self._q.qsize()
```

---

### `app/asr_pipeline.py`

```python
import os
import logging
import numpy as np
import torch
from transformers import pipeline as hf_pipeline
from groq import Groq
from models.model_config import MODEL_CONFIGS, SAMPLE_RATE, CHUNK_SECONDS, STRIDE_SECONDS

logger = logging.getLogger(__name__)

_loaded_models: dict = {}       # cache local HF pipelines — one per model name
_groq_client: Groq | None = None  # lazily initialised Groq client


def _get_device() -> int:
    """Return device index: 0 for first GPU, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


def _get_groq_client() -> Groq:
    """Return (or lazily create) the Groq API client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to .env to use the Groq Cloud model."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def load_asr_model(model_name: str):
    """Load local Whisper pipeline, caching by model name to avoid reload.

    Only called for backend='local' models. Groq backend has no local model.
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]
    if cfg["backend"] != "local":
        raise ValueError(f"load_asr_model() called on non-local model: {model_name}")

    hf_id = cfg["hf_id"]
    device = _get_device()
    logger.info("Loading %s on %s ...", hf_id, "GPU" if device == 0 else "CPU")

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=hf_id,
        device=device,
        chunk_length_s=CHUNK_SECONDS,
        stride_length_s=(STRIDE_SECONDS, STRIDE_SECONDS),
        generate_kwargs={
            "language": "ta",
            "task": "transcribe",
        },
        token=os.getenv("HF_TOKEN"),   # HF_TOKEN from .env
    )
    _loaded_models[model_name] = asr
    logger.info("Model %s loaded successfully", model_name)
    return asr


def transcribe_audio(audio: np.ndarray, model_name: str) -> str:
    """Transcribe a numpy float32 array using a local Whisper model.

    Only for backend='local'. For Groq, use transcribe_with_groq().
    """
    if audio is None or len(audio) == 0:
        return ""

    asr = load_asr_model(model_name)
    result = asr({"array": audio.astype(np.float32), "sampling_rate": SAMPLE_RATE})
    return result.get("text", "").strip()


def transcribe_chunks(chunks: list[np.ndarray], model_name: str) -> str:
    """Transcribe a list of audio chunks (local backend) and join results."""
    if not chunks:
        return ""
    transcripts = []
    for i, chunk in enumerate(chunks):
        logger.debug("Transcribing chunk %d/%d (%d samples)", i + 1, len(chunks), len(chunk))
        text = transcribe_audio(chunk, model_name)
        if text:
            transcripts.append(text)
    return " ".join(transcripts)


def transcribe_with_groq(audio_file_path: str) -> str:
    """Transcribe Tamil audio via Groq Cloud (whisper-large-v3-turbo).

    Args:
        audio_file_path: path to audio file — Gradio temp file path is fine.
                         Max 100MB. Groq handles chunking server-side.

    Returns:
        Tamil transcript string.

    Raises:
        EnvironmentError: if GROQ_API_KEY is not set
        groq.APIStatusError: on Groq API errors (auth, quota, etc.)
    """
    client = _get_groq_client()
    logger.info("Sending %s to Groq Cloud (whisper-large-v3-turbo) ...", audio_file_path)

    with open(audio_file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_file_path), f),
            model="whisper-large-v3-turbo",
            language="ta",           # force Tamil
            response_format="text",  # returns plain string, not JSON dict
        )

    transcript = result.strip() if result else ""
    logger.info("Groq transcription complete: %d chars", len(transcript))
    return transcript
```

---

### `app/transliteration.py`

```python
import logging
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate

logger = logging.getLogger(__name__)

SCHEME_MAP = {
    "ITRANS": sanscript.ITRANS,
    "ISO15919": sanscript.ISO15919,
    "IAST": sanscript.IAST,
    "HK": sanscript.HK,
    "HUNTERIAN": sanscript.HUNTERIAN,
}


def transliterate_tamil_to_latin(text: str, scheme: str = "ITRANS") -> str:
    """Convert Tamil Unicode text to Roman script using the given scheme.

    Args:
        text:   Tamil Unicode string (from Whisper transcript)
        scheme: one of ITRANS, ISO15919, IAST, HK, HUNTERIAN

    Returns:
        Romanized string. Non-Tamil characters (e.g., English loanwords) pass through unchanged.
    """
    if not text or not text.strip():
        return ""

    if scheme not in SCHEME_MAP:
        raise ValueError(f"Unknown scheme: {scheme}. Valid: {list(SCHEME_MAP.keys())}")

    try:
        result = _transliterate(text, sanscript.TAMIL, SCHEME_MAP[scheme])
        return result
    except Exception as exc:
        logger.error("Transliteration failed for scheme %s: %s", scheme, exc)
        return text     # fallback: return original on failure


def transliterate_latin_to_tamil(text: str, scheme: str = "ITRANS") -> str:
    """Convert romanized text back to Tamil Unicode script.

    Args:
        text:   Romanized string in the given scheme
        scheme: the romanization scheme used for input

    Returns:
        Tamil Unicode string.
    """
    if not text or not text.strip():
        return ""

    if scheme not in SCHEME_MAP:
        raise ValueError(f"Unknown scheme: {scheme}. Valid: {list(SCHEME_MAP.keys())}")

    try:
        return _transliterate(text, SCHEME_MAP[scheme], sanscript.TAMIL)
    except Exception as exc:
        logger.error("Reverse transliteration failed: %s", exc)
        return text
```

---

### `app/utils.py`

```python
import json
import os
import tempfile
import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load any audio file and resample to target sample rate.

    Handles: .wav, .mp3, .ogg, .flac, .m4a
    Always returns: float32 numpy array at target_sr Hz, mono channel.
    """
    if not file_path:
        raise ValueError("No audio file path provided")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    logger.debug("Loaded %s: %.1f seconds at %dHz", file_path, len(audio) / target_sr, target_sr)
    return audio.astype(np.float32)


def save_output_json(
    audio_path: str,
    model_name: str,
    scheme: str,
    transcript: str,
    transliterated: str,
    output_dir: str = "outputs",
) -> str:
    """Save pipeline results to a JSON file. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_{model_name}_{scheme}.json")

    payload = {
        "audio_file": os.path.basename(audio_path),
        "model": model_name,
        "transliteration_scheme": scheme,
        "transcript_tamil": transcript,
        "transliterated_roman": transliterated,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Output saved: %s", out_path)
    return out_path


def get_audio_duration(audio: np.ndarray, sr: int = 16000) -> float:
    """Return duration in seconds."""
    return len(audio) / sr
```

---

### `app/interface.py`

```python
import logging
import gradio as gr
from app.asr_pipeline import transcribe_chunks, transcribe_with_groq
from app.transliteration import transliterate_tamil_to_latin
from app.buffer_manager import AudioBufferManager
from app.utils import load_audio, save_output_json
from models.model_config import MODEL_CONFIGS, TRANSLITERATION_SCHEMES, DEFAULT_MODEL, DEFAULT_SCHEME

logger = logging.getLogger(__name__)


def process_audio(audio_path: str, model_name: str, scheme: str):
    """Main pipeline function wired to Gradio button.

    Routes to Groq Cloud or local HF pipeline based on MODEL_CONFIGS[model_name]["backend"].

    Args:
        audio_path:  temp file path from gr.Audio component
        model_name:  key in MODEL_CONFIGS (e.g. "whisper-small (Local)")
        scheme:      transliteration scheme (e.g. "ITRANS")

    Returns:
        (transcript: str, transliterated: str, json_file_path: str | None)
    """
    if audio_path is None:
        return "No audio provided.", "", None

    cfg = MODEL_CONFIGS.get(model_name, {})
    backend = cfg.get("backend", "local")

    try:
        if backend == "groq":
            # Groq Cloud path — raw file, no numpy loading, no buffer
            logger.info("Using Groq Cloud backend for model=%s", model_name)
            transcript = transcribe_with_groq(audio_path)
        else:
            # Local HF transformers path — load numpy, chunk, transcribe
            try:
                audio = load_audio(audio_path)
            except Exception as e:
                logger.error("Audio load failed: %s", e)
                return f"Error loading audio: {e}", "", None

            buffer = AudioBufferManager()
            n_chunks = buffer.enqueue(audio)
            logger.info("Enqueued %d chunks for model=%s", n_chunks, model_name)
            chunks = buffer.drain()
            transcript = transcribe_chunks(chunks, model_name)

    except EnvironmentError as e:
        return f"Configuration error: {e}", "", None
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return f"Transcription error: {e}", "", None

    if not transcript:
        return "Could not transcribe audio (silent or unsupported format).", "", None

    transliterated = transliterate_tamil_to_latin(transcript, scheme)
    json_path = save_output_json(audio_path, model_name, scheme, transcript, transliterated)

    return transcript, transliterated, json_path


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(
        title="Tamil ASR + Transliteration",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "## Tamil ASR + Transliteration\n"
            "Upload Tamil audio → get Tamil transcript + romanized transliteration.\n\n"
            "> **⚡ Groq Cloud** option uses `whisper-large-v3-turbo` via Groq's inference API — "
            "no local GPU needed, fastest inference, requires `GROQ_API_KEY` in `.env`."
        )

        with gr.Row():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Tamil Audio (WAV / MP3 / OGG / FLAC)",
            )

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=DEFAULT_MODEL,
                label="Whisper Model",
                info="Local models run on your machine. ⚡ Groq Cloud model requires GROQ_API_KEY.",
            )
            scheme_selector = gr.Dropdown(
                choices=TRANSLITERATION_SCHEMES,
                value=DEFAULT_SCHEME,
                label="Transliteration Scheme",
            )

        run_btn = gr.Button("Transcribe + Transliterate", variant="primary")

        with gr.Row():
            transcript_out = gr.Textbox(
                label="Tamil Transcript", lines=6, interactive=False
            )
            transliterated_out = gr.Textbox(
                label="Romanized Output", lines=6, interactive=False
            )

        download_file = gr.File(label="Download JSON Output")

        run_btn.click(
            fn=process_audio,
            inputs=[audio_input, model_selector, scheme_selector],
            outputs=[transcript_out, transliterated_out, download_file],
        )

        gr.Markdown(
            "**Schemes:** ITRANS (ASCII) · ISO15919 (standard) · IAST (scholarly) "
            "· HK (Harvard-Kyoto) · HUNTERIAN (colonial romanization)"
        )

    return demo
```

---

### `app/main.py`

```python
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from app.interface import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.queue()   # enable request queuing for concurrent users
    demo.launch(
        server_name="0.0.0.0",   # REQUIRED inside Docker
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
```

---

### `app/__init__.py`

```python
# app package
```

---

### `tests/__init__.py`

```python
# tests package
```

---

### `tests/test_buffer_manager.py`

```python
import numpy as np
import pytest
from app.buffer_manager import AudioBufferManager


def make_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def test_enqueue_single_chunk():
    buf = AudioBufferManager()
    audio = make_audio(10)   # 10s — fits in one 30s chunk
    n = buf.enqueue(audio)
    assert n == 1
    assert buf.size == 1


def test_enqueue_splits_long_audio():
    buf = AudioBufferManager()
    audio = make_audio(75)   # 75s → 3 chunks (30+30+15)
    n = buf.enqueue(audio)
    assert n == 3
    assert buf.size == 3


def test_drain_returns_all_chunks():
    buf = AudioBufferManager()
    audio = make_audio(75)
    buf.enqueue(audio)
    chunks = buf.drain()
    assert len(chunks) == 3
    assert buf.size == 0


def test_short_chunk_skipped():
    buf = AudioBufferManager()
    audio = make_audio(0.05)   # 50ms — below MIN_CHUNK_SAMPLES threshold
    n = buf.enqueue(audio)
    assert n == 0


def test_empty_audio():
    buf = AudioBufferManager()
    audio = np.array([], dtype=np.float32)
    n = buf.enqueue(audio)
    assert n == 0
    chunks = buf.drain()
    assert chunks == []
```

---

### `tests/test_transliteration.py`

```python
import pytest
from app.transliteration import transliterate_tamil_to_latin, transliterate_latin_to_tamil


def test_basic_tamil_to_itrans():
    result = transliterate_tamil_to_latin("வணக்கம்", "ITRANS")
    assert isinstance(result, str)
    assert len(result) > 0


def test_basic_tamil_to_iso15919():
    result = transliterate_tamil_to_latin("வணக்கம்", "ISO15919")
    assert isinstance(result, str)


def test_empty_string_returns_empty():
    assert transliterate_tamil_to_latin("", "ITRANS") == ""
    assert transliterate_tamil_to_latin("   ", "ITRANS") == ""


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        transliterate_tamil_to_latin("வணக்கம்", "FAKE_SCHEME")


def test_all_schemes_produce_output():
    text = "தமிழ்"
    for scheme in ["ITRANS", "ISO15919", "IAST", "HK", "HUNTERIAN"]:
        result = transliterate_tamil_to_latin(text, scheme)
        assert isinstance(result, str) and len(result) > 0, f"Scheme {scheme} returned empty"


def test_reverse_itrans_roundtrip():
    original = "வணக்கம்"
    roman = transliterate_tamil_to_latin(original, "ITRANS")
    back = transliterate_latin_to_tamil(roman, "ITRANS")
    # Round-trip may not be perfect but should not be empty
    assert isinstance(back, str) and len(back) > 0


def test_english_passthrough():
    # English chars should pass through unchanged
    result = transliterate_tamil_to_latin("hello", "ITRANS")
    assert "hello" in result.lower()
```

---

### `tests/test_asr_pipeline.py`

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.asr_pipeline import transcribe_audio, transcribe_chunks


@patch("app.asr_pipeline.load_asr_model")
def test_transcribe_audio_calls_pipeline(mock_load):
    mock_asr = MagicMock(return_value={"text": "வணக்கம்"})
    mock_load.return_value = mock_asr

    audio = np.zeros(16000, dtype=np.float32)
    result = transcribe_audio(audio, "whisper-small")
    assert result == "வணக்கம்"
    mock_asr.assert_called_once()


@patch("app.asr_pipeline.load_asr_model")
def test_transcribe_empty_audio_returns_empty(mock_load):
    result = transcribe_audio(np.array([]), "whisper-small")
    assert result == ""
    mock_load.assert_not_called()


@patch("app.asr_pipeline.load_asr_model")
def test_transcribe_none_returns_empty(mock_load):
    result = transcribe_audio(None, "whisper-small")
    assert result == ""


@patch("app.asr_pipeline.transcribe_audio")
def test_transcribe_chunks_joins_results(mock_transcribe):
    mock_transcribe.side_effect = ["வணக்கம்", "நன்றி"]
    chunks = [np.zeros(16000), np.zeros(16000)]
    result = transcribe_chunks(chunks, "whisper-small")
    assert "வணக்கம்" in result
    assert "நன்றி" in result


@patch("app.asr_pipeline.transcribe_audio")
def test_transcribe_chunks_empty_list(mock_transcribe):
    result = transcribe_chunks([], "whisper-small")
    assert result == ""
    mock_transcribe.assert_not_called()
```

---

### `tests/test_integration.py`

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.interface import process_audio


def test_process_audio_no_file():
    transcript, roman, json_path = process_audio(None, "whisper-small", "ITRANS")
    assert "No audio" in transcript
    assert json_path is None


@patch("app.interface.load_audio")
@patch("app.interface.transcribe_chunks")
@patch("app.interface.transliterate_tamil_to_latin")
@patch("app.interface.save_output_json")
def test_process_audio_full_pipeline(mock_save, mock_translit, mock_transcribe, mock_load):
    mock_load.return_value = np.zeros(16000, dtype=np.float32)
    mock_transcribe.return_value = "வணக்கம்"
    mock_translit.return_value = "vaNakkam"
    mock_save.return_value = "/tmp/test.json"

    transcript, roman, json_path = process_audio("/tmp/fake.wav", "whisper-small", "ITRANS")
    assert transcript == "வணக்கம்"
    assert roman == "vaNakkam"
    assert json_path == "/tmp/test.json"
```

---

### `requirements.txt`

```
# ASR — local HF pipeline
transformers>=4.36.0
torch>=2.1.0
torchaudio>=2.1.0
accelerate>=0.24.0

# ASR — Groq Cloud API
groq>=0.4.0

# Audio loading
librosa>=0.10.1
soundfile>=0.12.1

# Transliteration
indic-transliteration>=2.3.40

# UI
gradio>=4.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
numpy>=1.24.0
python-dotenv>=1.0.0
```

---

### `Dockerfile`

```dockerfile
FROM python:3.10-slim

# System dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create output dir
RUN mkdir -p outputs

# HuggingFace cache inside container
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache

# Gradio port
EXPOSE 7860

CMD ["python", "app/main.py"]
```

---

### `docker-compose.yml`

```yaml
version: "3.9"

services:
  asr-app:
    build: .
    container_name: tamil-asr-app
    ports:
      - "7860:7860"
    volumes:
      - ./outputs:/app/outputs           # persist JSON outputs
      - ./models_cache:/app/models_cache # persist downloaded model weights
      - ./sample_inputs:/app/sample_inputs
    env_file:
      - .env                             # loads HF_TOKEN + GROQ_API_KEY — never committed
    environment:
      - GRADIO_PORT=7860
      # GROQ_API_KEY is injected from .env via env_file above.
      # Explicitly listed here so docker-compose config shows it as expected.
      - GROQ_API_KEY=${GROQ_API_KEY:-}
    restart: unless-stopped
```

---

### `.env.example`

```
# Copy this to .env and fill in your tokens
# NEVER commit .env to git — it is in .gitignore

# HuggingFace token (required for model downloads)
HF_TOKEN=hf_your_token_here

# Groq Cloud API key (required only for whisper-large-v3-turbo ⚡ Groq Cloud option)
# Get a free key at: https://console.groq.com
GROQ_API_KEY=gsk_your_key_here
```

---

### `sample_inputs/README.md`

````markdown
# Sample Inputs

Place Tamil audio files here for testing.

## Getting sample Tamil audio

### Option 1 — Generate with gTTS (quick test)
```python
from gtts import gTTS
tts = gTTS("வணக்கம், இது ஒரு சோதனை ஒலி.", lang="ta")
tts.save("sample_inputs/test_tamil.mp3")
```

### Option 2 — Mozilla Common Voice (Tamil)
Download from: https://commonvoice.mozilla.org/ta/datasets
License: CC0. Recommended: any clips from `cv-corpus-*-ta.tar.gz`

### Option 3 — OpenSLR Tamil (SLR65)
Dataset: https://www.openslr.org/65/
500+ Tamil speakers, clean read speech.

## Required format
- Sample rate: any (librosa auto-resamples to 16kHz)
- Channels: mono or stereo (auto-converted to mono)
- Format: WAV, MP3, OGG, FLAC
- Duration: any (buffer manager handles chunking)
````

---

### `README.md` (content)

````markdown
# Task 2 — Tamil ASR + Transliteration System

Automatic speech recognition for Tamil audio with romanized transliteration output.

## Architecture

```
Audio → (local) Buffer Queue → Whisper (HF transformers) ─┐
      → (Groq)  raw file     → Groq Cloud API            ─┤
                                                           ↓
                                           Tamil Transcript → Transliteration → Gradio UI
```

## Models

| Model | Backend | Size | Use case |
|-------|---------|------|---------|
| openai/whisper-small | Local | 244M | Default — fast, no API key needed |
| openai/whisper-medium | Local | 769M | Higher accuracy, no API key needed |
| whisper-large-v3-turbo ⚡ | Groq Cloud | 1.5B (distilled) | Highest accuracy, requires `GROQ_API_KEY` |

Transliteration: `indic-transliteration` library — Tamil script → ITRANS / ISO 15919 / IAST / HK / HUNTERIAN.

## Quick Start (Docker)

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env — add HF_TOKEN (required) and GROQ_API_KEY (optional, for ⚡ Groq model)

# 2. Build and run
docker-compose up --build

# 3. Open browser
open http://localhost:7860
```

## Quick Start (Local)

```bash
cd task2_asr_transliteration
pip install -r requirements.txt
cp .env.example .env   # fill in tokens
python app/main.py
```

## Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token (for model download) | Yes |
| `GROQ_API_KEY` | Groq Cloud API key (for ⚡ Groq model) | Only for Groq backend |
| `GRADIO_PORT` | Port for Gradio web UI | No (default: 7860) |
| `TRANSFORMERS_CACHE` | Path to cache downloaded models | No |

## Output Format

Each run saves a JSON file to `outputs/`:
```json
{
  "audio_file": "sample.wav",
  "model": "whisper-small (Local)",
  "transliteration_scheme": "ITRANS",
  "transcript_tamil": "வணக்கம்",
  "transliterated_roman": "vaNakkam"
}
```
````

---

## EXECUTION ORDER

```
Step 1 — Local environment setup
  pip install -r requirements.txt
  cp .env.example .env   # add HF_TOKEN

Step 2 — Run tests (mocked — no model download needed)
  pytest tests/ -v

Step 3 — Run locally (will download models on first run ~500MB)
  python app/main.py
  → open http://localhost:7860

Step 4 — Upload a Tamil audio file
  → Select whisper-small (default)
  → Select transliteration scheme (ITRANS default)
  → Click "Transcribe + Transliterate"
  → Tamil transcript appears on left, romanized on right
  → Download JSON output if needed

Step 5 — Docker build and run
  docker-compose up --build
  → First run downloads models inside container (~500MB)
  → Models cached in ./models_cache volume (survives rebuild)
  → open http://localhost:7860

Step 6 — Test with multiple audio files from sample_inputs/
```

---

## KNOWN FAILURE MODES AND FIXES

### Whisper returns empty string or garbage
- **Cause**: Audio below 0.5s, pure silence, or non-speech content
- **Fix**: Check `len(audio) > 1600` before sending to model. Add a user-facing warning if output is empty.
- **Cause 2**: Language auto-detection chose wrong language. 
- **Fix 2**: Always pass `generate_kwargs={"language": "ta", "task": "transcribe"}` — never let Whisper guess.

### Gradio unreachable from browser in Docker
- **Cause**: `server_name` defaults to `localhost` (127.0.0.1) — inaccessible from host
- **Fix**: Always set `server_name="0.0.0.0"` in `demo.launch()`

### librosa install fails on Python 3.12
- **Cause**: Some older `librosa` versions have `numba` dependency issues on 3.12
- **Fix**: Pin `librosa>=0.10.1` and `numba>=0.58.0` in requirements.txt, or use Python 3.10 (already in Dockerfile)

### HuggingFace 401 unauthorized on model download
- **Cause**: `HF_TOKEN` not set or empty in `.env`
- **Fix**: `cp .env.example .env` and add your token. Whisper models are public so this is usually only triggered by token validation changes.

### `ModuleNotFoundError: indic_transliteration`
- **Cause**: Package name on PyPI is `indic-transliteration` (hyphen), import is `indic_transliteration` (underscore). Both correct.
- **Fix**: `pip install indic-transliteration` (with hyphen)

### Large audio OOM on CPU
- **Cause**: Very long audio (>60 min) exceeds system RAM when held as full numpy array
- **Fix**: The `AudioBufferManager` already chunked it to 30s pieces. If still OOM, reduce `CHUNK_SECONDS` to 15.

### Docker rebuild redownloads models every time
- **Cause**: `models_cache` volume not mounted, or mounted to wrong path
- **Fix**: Confirm `docker-compose.yml` has `./models_cache:/app/models_cache` and `ENV TRANSFORMERS_CACHE=/app/models_cache` in Dockerfile

---

## SUBMISSION CHECKLIST

### Repository root
- [ ] `task2_asr_transliteration/` directory exists
- [ ] `requirements.txt` — all deps listed (transformers, groq, librosa, gradio, indic-transliteration, pytest)
- [ ] `Dockerfile` — uses `python:3.10-slim`, installs `ffmpeg` + `libsndfile1`
- [ ] `docker-compose.yml` — port 7860, volumes for outputs + model cache, `GROQ_API_KEY` env var
- [ ] `.env.example` — has `HF_TOKEN=` and `GROQ_API_KEY=` placeholders (never commit `.env`)
- [ ] `.gitignore` includes `.env` (already confirmed)

### App code
- [ ] `app/main.py` — launches Gradio with `server_name="0.0.0.0"`
- [ ] `app/asr_pipeline.py` — local path: `pipeline()` with `language="ta"`, `task="transcribe"`; Groq path: `transcribe_with_groq()` using `groq.Groq` client
- [ ] `app/asr_pipeline.py` — `transcribe_with_groq()` present, uses `whisper-large-v3-turbo`, `language="ta"`, `response_format="text"`
- [ ] `app/transliteration.py` — wraps `indic-transliteration`, all 5 schemes supported
- [ ] `app/buffer_manager.py` — `queue.Queue`, 30s chunking, `drain()` method
- [ ] `app/interface.py` — `process_audio()` branches on `MODEL_CONFIGS[model_name]["backend"]`, Groq path skips buffer
- [ ] `app/interface.py` — dropdown choices show ⚡ Groq Cloud label; info callout explains GROQ_API_KEY requirement
- [ ] `app/utils.py` — `load_audio()` with librosa, `save_output_json()`
- [ ] `models/model_config.py` — `backend` field on every entry; `whisper-large-v3-turbo ⚡ Groq Cloud` entry present

### Tests
- [ ] `tests/test_buffer_manager.py` — chunk count, drain, short clip skip, empty audio
- [ ] `tests/test_transliteration.py` — all 5 schemes, empty string, unknown scheme error, roundtrip
- [ ] `tests/test_asr_pipeline.py` — mocked pipeline calls, empty/None input
- [ ] `tests/test_integration.py` — full pipeline mock (local), None file case
- [ ] All tests pass: `pytest tests/ -v`

### Functionality
- [ ] App launches locally: `python app/main.py` → http://localhost:7860
- [ ] App launches in Docker: `docker-compose up` → http://localhost:7860
- [ ] Audio upload works (WAV file)
- [ ] Tamil transcript appears
- [ ] Romanized transliteration appears
- [ ] JSON download works
- [ ] Model dropdown switches between whisper-small and whisper-medium
- [ ] Transliteration scheme dropdown changes output format
- [ ] `outputs/` directory populated after each run

### Documentation
- [ ] `README.md` — architecture, quick start, environment variables, output format
- [ ] `sample_inputs/README.md` — how to get Tamil audio for testing
