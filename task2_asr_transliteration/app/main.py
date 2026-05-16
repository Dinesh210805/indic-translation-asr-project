import logging
import os
import sys
from pathlib import Path

# Pin HF cache to a project-local folder BEFORE importing transformers/torch.
# Without this, weights go to ~/.cache/huggingface/hub. Doing it here keeps
# everything inside the repo directory and persists across machines/venvs.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_CACHE = _PROJECT_ROOT / "models_cache"
_MODELS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_MODELS_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(_MODELS_CACHE / "hub")
os.environ["HF_HUB_CACHE"] = str(_MODELS_CACHE / "hub")

import gradio as gr
import torch
from dotenv import load_dotenv

load_dotenv()  # load .env before any module imports that read env vars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
# Quiet noisy third-party loggers while keeping our app verbose
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

log = logging.getLogger("app.startup")


def _log_environment() -> None:
    """Print a one-time environment snapshot so cold-start issues are easy to diagnose."""
    log.info("=" * 70)
    log.info("Tamil ASR + Transliteration — startup")
    log.info("=" * 70)
    log.info("Python:       %s", sys.version.split()[0])
    log.info("PyTorch:      %s", torch.__version__)
    log.info("CUDA avail:   %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("CUDA device:  %s", torch.cuda.get_device_name(0))
    log.info("GROQ_API_KEY: %s", "set" if os.getenv("GROQ_API_KEY") else "NOT SET")
    log.info("HF_TOKEN:     %s", "set" if os.getenv("HF_TOKEN") else "not set (only needed for gated models)")

    # Check HF cache state — biggest contributor to "page unresponsive" on first run
    hf_hub_cache = _MODELS_CACHE / "hub"
    log.info("Model cache: %s", hf_hub_cache)
    if hf_hub_cache.exists():
        cached = [p.name for p in hf_hub_cache.iterdir() if p.name.startswith("models--openai--whisper")]
        if cached:
            log.info("Whisper cache: %s", ", ".join(cached))
        else:
            log.warning(
                "Whisper cache: EMPTY at %s — first transcription will trigger a multi-GB download "
                "(~500MB for small, ~1.5GB for medium). Browser may show 'page unresponsive' during download. "
                "Watch this terminal for progress. To pre-download, run:\n"
                "  python -m app.prefetch_models",
                hf_hub_cache,
            )
    else:
        log.warning("Cache dir was just created — models will download on first transcribe")
    log.info("=" * 70)


from app.interface import build_ui, get_theme, CUSTOM_CSS  # noqa: E402 — must come after load_dotenv()


if __name__ == "__main__":
    _log_environment()
    log.info("Building Gradio UI ...")
    demo = build_ui()
    demo.queue()
    port = int(os.getenv("GRADIO_PORT", "7860"))
    log.info("Launching Gradio on http://0.0.0.0:%d", port)
    demo.launch(
        server_name="0.0.0.0",  # REQUIRED inside Docker
        server_port=port,
        share=False,
        show_error=True,
        theme=get_theme(),
        css=CUSTOM_CSS,
    )
