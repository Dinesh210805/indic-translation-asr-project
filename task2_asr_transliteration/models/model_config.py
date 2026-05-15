SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
STRIDE_SECONDS = 5

MODEL_CONFIGS = {
    "whisper-small (Local)": {
        "backend": "local",
        "hf_id": "openai/whisper-small",
        "params": "244M",
        "vram_gb": 1.5,
        "description": "Fast, good quality for clear Tamil speech",
    },
    "whisper-medium (Local)": {
        "backend": "local",
        "hf_id": "openai/whisper-medium",
        "params": "769M",
        "vram_gb": 3.0,
        "description": "Higher accuracy, slower inference",
    },
    "whisper-large-v3-turbo ⚡ Groq Cloud": {
        "backend": "groq",
        "groq_id": "whisper-large-v3-turbo",
        "params": "1.5B (distilled)",
        "vram_gb": 0,
        "description": "Highest accuracy via Groq Cloud API — requires GROQ_API_KEY",
    },
}

DEFAULT_MODEL = "whisper-small (Local)"

TRANSLITERATION_SCHEMES = ["ITRANS", "ISO", "IAST", "HK", "SLP1"]
DEFAULT_SCHEME = "ITRANS"
