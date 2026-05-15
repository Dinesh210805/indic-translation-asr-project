# sample_inputs

Place Tamil audio files here for quick testing via the UI or CLI.

Supported formats: WAV · MP3 · OGG · FLAC

Recommended:
- 16 kHz mono WAV for best compatibility with local Whisper models
- Any sample rate accepted — `librosa` resamples to 16 kHz automatically

This directory is mounted read-only inside Docker (`./sample_inputs:/app/sample_inputs:ro`).

Audio files are excluded from git via `.gitignore` (`*.wav`, `*.mp3`, etc.).
Add your own test clips here; they will not be committed.
