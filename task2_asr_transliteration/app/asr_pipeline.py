import os
import time
import logging
import numpy as np
import torch
from transformers import pipeline as hf_pipeline
from groq import Groq
from models.model_config import MODEL_CONFIGS, SAMPLE_RATE, CHUNK_SECONDS, STRIDE_SECONDS

logger = logging.getLogger(__name__)

_loaded_models: dict = {}
_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Lazily initialize the Groq client (raises EnvironmentError if key missing)."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to your .env file to use the Groq Cloud model."
            )
        logger.info("Initialising Groq client (key=%s***)", api_key[:6])
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def load_asr_model(model_name: str):
    """Load and cache a local HF Whisper pipeline.

    Raises ValueError for non-local (Groq) model names — Groq requires no local loading.
    """
    if model_name in _loaded_models:
        logger.info("Reusing cached pipeline for %s", model_name)
        return _loaded_models[model_name]

    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}. Valid: {list(MODEL_CONFIGS.keys())}")

    if cfg["backend"] != "local":
        raise ValueError(
            f"load_asr_model() called on non-local model '{model_name}'. "
            "Use transcribe_with_groq() for Groq Cloud models."
        )

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(
        "Loading %s (device=%s, dtype=%s) — this may take minutes on first run while weights download.",
        cfg["hf_id"], "cuda:0" if device == 0 else "cpu", dtype,
    )

    t0 = time.perf_counter()
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=cfg["hf_id"],
        device=device,
        torch_dtype=dtype,
        chunk_length_s=CHUNK_SECONDS,
        stride_length_s=(STRIDE_SECONDS, STRIDE_SECONDS),
        token=os.getenv("HF_TOKEN"),
    )
    elapsed = time.perf_counter() - t0
    _loaded_models[model_name] = asr
    logger.info("Model %s loaded in %.1fs", cfg["hf_id"], elapsed)
    return asr


def transcribe_audio(audio: np.ndarray, model_name: str) -> str:
    """Transcribe a numpy float32 array using a local Whisper model.

    Returns empty string for empty/None input without loading the model.
    """
    if audio is None or len(audio) == 0:
        logger.warning("transcribe_audio called with empty/None audio — returning empty string")
        return ""

    logger.debug(
        "transcribe_audio: %d samples (%.2fs at %dHz), dtype=%s",
        len(audio), len(audio) / SAMPLE_RATE, SAMPLE_RATE, audio.dtype,
    )
    asr = load_asr_model(model_name)
    t0 = time.perf_counter()
    result = asr(
        {"array": audio, "sampling_rate": SAMPLE_RATE},
        generate_kwargs={"language": "ta", "task": "transcribe"},
    )
    text = result.get("text", "").strip()
    logger.info(
        "transcribe_audio done in %.2fs — %d chars", time.perf_counter() - t0, len(text),
    )
    return text


def transcribe_chunks(chunks: list[np.ndarray], model_name: str) -> str:
    """Transcribe a list of audio chunks (local backend) and join results."""
    if not chunks:
        logger.warning("transcribe_chunks called with empty chunk list")
        return ""

    logger.info("Transcribing %d chunks with %s", len(chunks), model_name)
    t0 = time.perf_counter()
    parts = []
    for i, chunk in enumerate(chunks):
        text = transcribe_audio(chunk, model_name)
        if text:
            parts.append(text)
        logger.info("  chunk %d/%d: %d chars", i + 1, len(chunks), len(text))

    joined = " ".join(parts)
    logger.info(
        "transcribe_chunks done in %.2fs — %d chars total",
        time.perf_counter() - t0, len(joined),
    )
    return joined


def transcribe_with_groq(audio_file_path: str) -> str:
    """Send a raw audio file to Groq Cloud (whisper-large-v3-turbo) for transcription.

    Groq handles chunking server-side — no buffer needed. Max file size: 100MB.

    Raises:
        EnvironmentError: if GROQ_API_KEY is not set
        groq.APIStatusError: on Groq API errors (auth failure, quota exceeded, etc.)
    """
    client = _get_groq_client()
    file_size = os.path.getsize(audio_file_path)
    logger.info(
        "Sending %s (%.2f KB) to Groq Cloud whisper-large-v3-turbo ...",
        audio_file_path, file_size / 1024,
    )

    t0 = time.perf_counter()
    with open(audio_file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_file_path), f),
            model="whisper-large-v3-turbo",
            language="ta",
            response_format="text",
        )

    transcript = result.strip() if result else ""
    logger.info(
        "Groq transcription complete in %.2fs — %d chars",
        time.perf_counter() - t0, len(transcript),
    )
    return transcript
