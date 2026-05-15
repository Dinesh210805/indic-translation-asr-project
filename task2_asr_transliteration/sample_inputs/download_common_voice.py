"""
Download Mozilla Common Voice 25.0 Tamil clips via Mozilla Data Collective API.

Usage:
    python download_common_voice.py

Requirements:
    pip install requests soundfile numpy

Authentication:
    Set MDC_API_KEY in your root .env file.
    Get your key at: https://mozilladatacollective.com/profile/credentials

What this does:
    1. Calls POST /datasets/:id/download to get a presigned URL (valid 12 hrs)
    2. Streams the tar.gz to a temp file
    3. Extracts only the test split (test.tsv + matching clips/)
    4. Resamples each MP3 to 16kHz mono WAV
    5. Writes sample_inputs/common_voice/index.csv with ground-truth sentences

Output:
    sample_inputs/common_voice/*.wav
    sample_inputs/common_voice/index.csv
"""

import os
import io
import csv
import sys
import tarfile
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Load .env from project root (two levels up from sample_inputs/)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

try:
    import requests
except ImportError:
    sys.exit("Run: pip install requests")

MDC_API_KEY = os.getenv("MDC_API_KEY")
if not MDC_API_KEY:
    sys.exit(
        "ERROR: MDC_API_KEY not set.\n"
        "Add MDC_API_KEY=your_key to your root .env file.\n"
        "Get your key at: https://mozilladatacollective.com/profile/credentials"
    )

DATASET_ID   = "cmn2gfvyp01geo107izoftfki"   # Common Voice 25.0 Tamil
API_BASE     = "https://mozilladatacollective.com/api"
MAX_CLIPS    = 50     # set to None to extract everything from the test split
OUT_DIR      = Path(__file__).parent / "common_voice"

OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"Authorization": f"Bearer {MDC_API_KEY}"}


def get_presigned_url() -> str:
    log.info("Requesting presigned download URL for dataset %s ...", DATASET_ID)
    resp = requests.post(
        f"{API_BASE}/datasets/{DATASET_ID}/download",
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    url = data.get("url") or data.get("downloadUrl") or data.get("presignedUrl")
    if not url:
        sys.exit(f"Could not find download URL in response: {data}")
    log.info("Got presigned URL (valid 12 hours).")
    return url


def stream_tar_and_extract(url: str):
    """Stream the tar.gz and extract test.tsv + test clips without writing full 8 GB."""
    log.info("Streaming tar.gz from presigned URL ...")

    tsv_rows = []        # rows from test.tsv
    clip_bytes = {}      # {clip_filename: bytes}

    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()

        # Wrap the streaming response in a file-like object for tarfile
        class StreamWrapper(io.RawIOBase):
            def __init__(self, iterator):
                self._iter = iterator
                self._buf = b""

            def readinto(self, b):
                while not self._buf:
                    try:
                        self._buf = next(self._iter)
                    except StopIteration:
                        return 0
                n = min(len(b), len(self._buf))
                b[:n] = self._buf[:n]
                self._buf = self._buf[n:]
                return n

        stream = io.BufferedReader(StreamWrapper(resp.iter_content(chunk_size=1 << 20)))

        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                name = member.name

                # Extract test.tsv (ground truth)
                if name.endswith("/test.tsv") or name == "test.tsv":
                    log.info("Found test.tsv: %s", name)
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8")
                        reader = csv.DictReader(io.StringIO(content), delimiter="\t")
                        tsv_rows = list(reader)
                    log.info("Loaded %d rows from test.tsv", len(tsv_rows))

                # Extract MP3 clips that are in test.tsv
                elif "/clips/" in name and name.endswith(".mp3"):
                    clip_name = Path(name).name
                    if MAX_CLIPS and len(clip_bytes) >= MAX_CLIPS:
                        continue
                    f = tar.extractfile(member)
                    if f:
                        clip_bytes[clip_name] = f.read()

                if MAX_CLIPS and len(clip_bytes) >= MAX_CLIPS and tsv_rows:
                    log.info("Reached MAX_CLIPS=%d, stopping extraction.", MAX_CLIPS)
                    break

    return tsv_rows, clip_bytes


def convert_and_save(tsv_rows: list, clip_bytes: dict):
    try:
        import soundfile as sf
        import numpy as np
        import librosa
    except ImportError:
        sys.exit("Run: pip install soundfile numpy librosa")

    # Build a lookup: clip filename -> sentence
    sentence_map = {row["path"]: row["sentence"] for row in tsv_rows if "path" in row}

    index_path = OUT_DIR / "index.csv"
    saved = 0

    with open(index_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "sentence", "original_mp3"])

        for mp3_name, mp3_data in clip_bytes.items():
            sentence = sentence_map.get(mp3_name, "")
            if not sentence:
                continue

            wav_name = mp3_name.replace(".mp3", ".wav")
            out_path = OUT_DIR / wav_name

            try:
                audio, orig_sr = librosa.load(io.BytesIO(mp3_data), sr=16000, mono=True)
                sf.write(str(out_path), audio, 16000)
                writer.writerow([wav_name, sentence, mp3_name])
                saved += 1
                log.info("[%d] Saved %s | %s", saved, wav_name, sentence[:60])
            except Exception as e:
                log.warning("Failed to convert %s: %s", mp3_name, e)

    log.info("Done. %d clips saved to %s", saved, OUT_DIR)
    log.info("Index: %s", index_path)


def main():
    url = get_presigned_url()
    tsv_rows, clip_bytes = stream_tar_and_extract(url)

    if not clip_bytes:
        sys.exit("No clips extracted. Check your API key and dataset ID.")

    convert_and_save(tsv_rows, clip_bytes)


if __name__ == "__main__":
    main()
