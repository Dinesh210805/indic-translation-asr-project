FROM python:3.11-slim

# System deps: ffmpeg for audio decoding, libsndfile for librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requires UID 1000; also better practice locally)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/user/app/models_cache \
    TRANSFORMERS_CACHE=/home/user/app/models_cache/hub \
    HF_HUB_CACHE=/home/user/app/models_cache/hub

WORKDIR $HOME/app

# Install Python deps first (cached layer)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application source
COPY --chown=user:user app/ ./app/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user app.py .

# Pre-create writable dirs
RUN mkdir -p models_cache/hub outputs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:7860/ || exit 1

# Use app.py (the HF Spaces entry point) so the same image works locally and on Spaces
CMD ["python", "app.py"]
