FROM python:3.11-slim

# System deps: ffmpeg for audio decoding, build tools for torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY models/ ./models/

ENV PYTHONPATH=/app
ENV GRADIO_PORT=7860
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache

EXPOSE 7860

CMD ["python", "-m", "app.main"]
