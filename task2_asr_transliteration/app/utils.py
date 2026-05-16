import json
import os
import logging
import time
import numpy as np
import librosa

logger = logging.getLogger(__name__)


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load any audio file and resample to target sample rate.

    Handles: .wav, .mp3, .ogg, .flac, .m4a
    Returns: float32 numpy array at target_sr Hz, mono channel.
    """
    if not file_path:
        raise ValueError("No audio file path provided")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    size_kb = os.path.getsize(file_path) / 1024
    logger.info("Loading audio: %s (%.1f KB)", file_path, size_kb)

    t0 = time.perf_counter()
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    duration = len(audio) / target_sr
    logger.info(
        "Loaded %s — %.2fs @ %dHz, %d samples, dtype=%s, loaded in %.2fs",
        os.path.basename(file_path), duration, target_sr, len(audio), audio.dtype,
        time.perf_counter() - t0,
    )
    if duration < 0.1:
        logger.warning(
            "Audio is very short (%.3fs). If you recorded via the mic, the recording may not have captured anything — "
            "check browser mic permissions and re-record.", duration,
        )
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
    safe_model = model_name.replace(" ", "_").replace("⚡", "groq")
    out_path = os.path.join(output_dir, f"{base_name}_{safe_model}_{scheme}.json")

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
