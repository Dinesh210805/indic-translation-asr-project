import logging
import gradio as gr
from app.asr_pipeline import transcribe_chunks, transcribe_with_groq
from app.transliteration import transliterate_tamil_to_latin
from app.buffer_manager import AudioBufferManager
from app.utils import load_audio, save_output_json
from models.model_config import MODEL_CONFIGS, TRANSLITERATION_SCHEMES, DEFAULT_MODEL, DEFAULT_SCHEME

logger = logging.getLogger(__name__)


def process_audio(audio_path: str, model_name: str, scheme: str):
    """Main pipeline function wired to Gradio button.

    Routes to Groq Cloud or local HF pipeline based on MODEL_CONFIGS[model_name]["backend"].

    Returns:
        (transcript: str, transliterated: str, json_file_path: str | None)
    """
    if audio_path is None:
        return "No audio provided.", "", None

    cfg = MODEL_CONFIGS.get(model_name, {})
    backend = cfg.get("backend", "local")

    try:
        if backend == "groq":
            logger.info("Using Groq Cloud backend for model=%s", model_name)
            transcript = transcribe_with_groq(audio_path)
        else:
            try:
                audio = load_audio(audio_path)
            except Exception as e:
                logger.error("Audio load failed: %s", e)
                return f"Error loading audio: {e}", "", None

            buffer = AudioBufferManager()
            n_chunks = buffer.enqueue(audio)
            logger.info("Enqueued %d chunks for model=%s", n_chunks, model_name)
            chunks = buffer.drain()
            transcript = transcribe_chunks(chunks, model_name)

    except EnvironmentError as e:
        return f"Configuration error: {e}", "", None
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return f"Transcription error: {e}", "", None

    if not transcript:
        return "Could not transcribe audio (silent or unsupported format).", "", None

    transliterated = transliterate_tamil_to_latin(transcript, scheme)
    json_path = save_output_json(audio_path, model_name, scheme, transcript, transliterated)

    return transcript, transliterated, json_path


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(
        title="Tamil ASR + Transliteration",
    ) as demo:
        gr.Markdown(
            "## Tamil ASR + Transliteration\n"
            "Upload Tamil audio → get Tamil transcript + romanized transliteration.\n\n"
            "> **⚡ Groq Cloud** option uses `whisper-large-v3-turbo` via Groq's inference API — "
            "no local GPU needed, fastest inference. Requires `GROQ_API_KEY` in `.env`."
        )

        with gr.Row():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Tamil Audio (WAV / MP3 / OGG / FLAC)",
            )

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=DEFAULT_MODEL,
                label="Whisper Model",
                info="Local models run on your machine. ⚡ Groq Cloud model requires GROQ_API_KEY.",
            )
            scheme_selector = gr.Dropdown(
                choices=TRANSLITERATION_SCHEMES,
                value=DEFAULT_SCHEME,
                label="Transliteration Scheme",
            )

        run_btn = gr.Button("Transcribe + Transliterate", variant="primary")

        with gr.Row():
            transcript_out = gr.Textbox(
                label="Tamil Transcript", lines=6, interactive=False
            )
            transliterated_out = gr.Textbox(
                label="Romanized Output", lines=6, interactive=False
            )

        download_file = gr.File(label="Download JSON Output")

        run_btn.click(
            fn=process_audio,
            inputs=[audio_input, model_selector, scheme_selector],
            outputs=[transcript_out, transliterated_out, download_file],
        )

        gr.Markdown(
            "**Schemes:** ITRANS (ASCII-friendly) · ISO (ISO 15919 standard) · IAST (scholarly) "
            "· HK (Harvard-Kyoto) · SLP1 (Sanskrit Library)"
        )

    return demo
