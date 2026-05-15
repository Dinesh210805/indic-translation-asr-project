# sample_inputs

Test audio for the Tamil ASR pipeline.

---

## Directory Layout

```
sample_inputs/
├── common_voice/          # Downloaded Common Voice 25.0 Tamil clips (gitignored)
│   ├── index.csv          # Ground-truth sentences per clip (committed)
│   └── *.wav              # 16 kHz mono WAV clips (not committed)
├── personal/              # Your own recordings (gitignored)
│   └── p01_greeting.wav   # Naming convention from RECORDING_GUIDE.md
├── download_common_voice.py   # Script to fetch Common Voice test clips
└── RECORDING_GUIDE.md         # 16 Tamil sentences to record yourself
```

---

## Getting Test Clips

### Option A — Common Voice (standard benchmark)

```bash
# 1. Accept terms at:
#    https://mozilladatacollective.com/datasets/cmn2gfvyp01geo107izoftfki
# 2. Add MDC_API_KEY to your .env
# 3. Run:
python sample_inputs/download_common_voice.py
```

Downloads up to 50 clips from the test split, resamples to 16 kHz mono WAV,
and writes `common_voice/index.csv` with ground-truth sentences.

### Option B — Personal recordings

Read `RECORDING_GUIDE.md`, record the sentences, and save files to `sample_inputs/personal/`
using the naming convention `p01_greeting.wav`, `p02_weather.wav`, etc.

---

## Audio Format Requirements

| Property | Requirement |
|---|---|
| Format | WAV (preferred), MP3, OGG, FLAC accepted |
| Sample rate | Any — resampled to 16 kHz automatically |
| Channels | Any — mixed to mono automatically |
| Duration | Any — chunked in 30-second windows for local models |

---

## Git Policy

Audio files are excluded from git (`.gitignore` covers `*.wav`, `*.mp3`, `*.ogg`, `*.flac`).  
Only `index.csv` and this README are committed.
