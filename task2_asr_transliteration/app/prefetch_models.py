"""Pre-download Whisper weights into the project-local models_cache/ folder.

Run once before launching the UI to avoid the cold-start "page unresponsive" freeze:
    python -m app.prefetch_models                 # small + medium
    python -m app.prefetch_models small           # just small
    python -m app.prefetch_models small medium    # explicit list
"""
import logging
import os
import sys
import time
from pathlib import Path

# Pin cache to project-local folder BEFORE importing huggingface_hub / transformers.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_CACHE = _PROJECT_ROOT / "models_cache"
_MODELS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_MODELS_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(_MODELS_CACHE / "hub")
os.environ["HF_HUB_CACHE"] = str(_MODELS_CACHE / "hub")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("prefetch")

from huggingface_hub import snapshot_download  # noqa: E402

WHISPER_REPOS = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
}


def prefetch(names: list[str]) -> None:
    log.info("Cache target: %s", _MODELS_CACHE / "hub")
    for name in names:
        repo_id = WHISPER_REPOS.get(name)
        if repo_id is None:
            log.error("Unknown model %r — choose from: %s", name, list(WHISPER_REPOS))
            continue
        log.info("Downloading %s ...", repo_id)
        t0 = time.perf_counter()
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(_MODELS_CACHE / "hub"),
            token=os.getenv("HF_TOKEN"),
        )
        log.info("Done in %.1fs → %s", time.perf_counter() - t0, path)


if __name__ == "__main__":
    args = sys.argv[1:] or ["small", "medium"]
    prefetch(args)
    log.info("All requested models are cached. Launch the app with: python -m app.main")
