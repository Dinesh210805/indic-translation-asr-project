import os
import logging
import numpy as np
import torch
from transformers import pipeline as hf_pipeline
from groq import Groq
from models.model_config import MODEL_CONFIGS, SAMPLE_RATE

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
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def load_asr_model(model_name: str):
    """Load and cache a local HF Whisper pipeline.

    Raises ValueError for non-local (Groq) model names — Groq requires no local loading.
    """
    if model_name in _loaded_models:
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
    logger.info("Loading %s on device=%d ...", cfg["hf_id"], device)

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=cfg["hf_id"],
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    _loaded_models[model_name] = asr
    logger.info("Model %s loaded.", cfg["hf_id"])
    return asr


def transcribe_audio(audio: np.ndarray, model_name: str) -> str:
    """Transcribe a numpy float32 array using a local Whisper model.

    Returns empty string for empty/None input without loading the model.
    """
    if audio is None or len(audio) == 0:
        return ""

    asr = load_asr_model(model_name)
    result = asr(
        {"array": audio, "sampling_rate": SAMPLE_RATE},
        generate_kwargs={"language": "ta", "task": "transcribe"},
    )
    return result.get("text", "").strip()


def transcribe_chunks(chunks: list[np.ndarray], model_name: str) -> str:
    """Transcribe a list of audio chunks (local backend) and join results."""
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks):
        text = transcribe_audio(chunk, model_name)
        if text:
            parts.append(text)
        logger.debug("Chunk %d/%d: %d chars", i + 1, len(chunks), len(text))

    return " ".join(parts)


def transcribe_with_groq(audio_file_path: str) -> str:
    """Send a raw audio file to Groq Cloud (whisper-large-v3-turbo) for transcription.

    Groq handles chunking server-side — no buffer needed. Max file size: 100MB.

    Raises:
        EnvironmentError: if GROQ_API_KEY is not set
        groq.APIStatusError: on Groq API errors (auth failure, quota exceeded, etc.)
    """
    client = _get_groq_client()
    logger.info("Sending %s to Groq Cloud (whisper-large-v3-turbo) ...", audio_file_path)

    with open(audio_file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_file_path), f),
            model="whisper-large-v3-turbo",
            language="ta",
            response_format="text",
        )

    transcript = result.strip() if result else ""
    logger.info("Groq transcription complete: %d chars", len(transcript))
    return transcript
