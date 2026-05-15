"""
Download Mozilla Common Voice Tamil (cv-corpus) clips from HuggingFace.

Usage:
    python download_common_voice.py

Requirements:
    pip install datasets soundfile

Authentication:
    Set HF_TOKEN in your .env file (or environment). Required to access
    mozilla-foundation/common_voice_17_0 (gated dataset — accept terms at
    https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 first).

Output:
    sample_inputs/common_voice/*.wav      — audio clips at 16 kHz mono
    sample_inputs/common_voice/index.csv  — clip filename + ground-truth sentence
"""

import os
import csv
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    sys.exit(
        "ERROR: HF_TOKEN not set.\n"
        "Add HF_TOKEN=hf_... to your root .env file.\n"
        "Also accept dataset terms at: "
        "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0"
    )

try:
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np
except ImportError:
    sys.exit("Run: pip install datasets soundfile numpy")

DATASET_ID = "mozilla-foundation/common_voice_17_0"
LANGUAGE = "ta"
SPLIT = "test"           # 'test' is smaller and fully validated
MAX_CLIPS = 50           # change to None to download everything
TARGET_SR = 16000        # Whisper expects 16 kHz
OUT_DIR = Path(__file__).parent / "common_voice"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def resample(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio_array
    try:
        import librosa
        return librosa.resample(audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        log.warning("librosa not installed — skipping resample (orig_sr=%d)", orig_sr)
        return audio_array


def main():
    log.info("Loading dataset %s / %s / split=%s ...", DATASET_ID, LANGUAGE, SPLIT)
    ds = load_dataset(
        DATASET_ID,
        LANGUAGE,
        split=SPLIT,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    total = len(ds)
    limit = min(MAX_CLIPS, total) if MAX_CLIPS else total
    log.info("Dataset has %d clips. Downloading %d.", total, limit)

    index_path = OUT_DIR / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "sentence", "speaker_id", "split"])

        for i, row in enumerate(ds):
            if MAX_CLIPS and i >= MAX_CLIPS:
                break

            sentence = row.get("sentence", "")
            speaker_id = row.get("client_id", "unknown")[:8]
            audio_obj = row["audio"]

            array = np.array(audio_obj["array"], dtype=np.float32)
            orig_sr = audio_obj["sampling_rate"]
            array = resample(array, orig_sr, TARGET_SR)

            fname = f"cv_{LANGUAGE}_{i:04d}.wav"
            out_path = OUT_DIR / fname
            sf.write(str(out_path), array, TARGET_SR)

            writer.writerow([fname, sentence, speaker_id, SPLIT])
            log.info("[%d/%d] Saved %s | %s", i + 1, limit, fname, sentence[:60])

    log.info("Done. %d clips saved to %s", limit, OUT_DIR)
    log.info("Index written to %s", index_path)


if __name__ == "__main__":
    main()
