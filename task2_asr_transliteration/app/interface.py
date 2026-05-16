import html
import logging
import os
import time
import uuid

import gradio as gr

from app.asr_pipeline import (
    load_asr_model,
    transcribe_audio,
    transcribe_with_groq,
)
from app.transliteration import transliterate_tamil_to_latin, transliterate_latin_to_tamil
from app.buffer_manager import AudioBufferManager
from app.utils import load_audio, save_output_json
from app.visualizations import (
    waveform_html,
    chunked_waveform_html,
    mel_spectrogram_html,
    flow_diagram_html,
    whisper_arch_html,
    glyph_mapping_html,
    token_stream_html,
    json_preview_html,
)
from models.model_config import (
    MODEL_CONFIGS,
    TRANSLITERATION_SCHEMES,
    DEFAULT_MODEL,
    DEFAULT_SCHEME,
    SAMPLE_RATE,  
    CHUNK_SECONDS,
    STRIDE_SECONDS,
)

logger = logging.getLogger(__name__)


CUSTOM_CSS = """
/* ─── reset & frame ─────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: 'Inter','Segoe UI',system-ui,-apple-system,sans-serif !important;
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(255,138,76,.12), transparent 60%),
        radial-gradient(900px 500px at 110% 10%, rgba(120,90,255,.10), transparent 60%),
        #0b0d12 !important;
    color: #e6e8ee !important;
    padding: 30px 26px 60px !important;
}
footer, .show-api { display: none !important; }

.gradio-container .block {
    background: rgba(255,255,255,.03) !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    border-radius: 16px !important;
}
.gradio-container .form { background: transparent !important; border: none !important; }
.gradio-container label > span, .gradio-container .label, .gradio-container .block .label-wrap {
    color: #c9cdd9 !important; font-weight: 500 !important;
    font-size: 12px !important; text-transform: uppercase; letter-spacing: .08em;
}
.gradio-container input[type=text], .gradio-container textarea, .gradio-container select {
    background: #14171f !important; color: #e6e8ee !important;
    border: 1px solid rgba(255,255,255,.08) !important; border-radius: 10px !important;
}

/* ─── hero ──────────────────────────────────────────────────────────── */
.hero {
    padding: 24px 30px; margin-bottom: 18px; border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,138,76,.10), rgba(120,90,255,.08));
    border: 1px solid rgba(255,255,255,.08);
    position: relative; overflow: hidden;
}
.hero::after {
    content: "தமிழ்"; position: absolute; right: -10px; bottom: -42px;
    font-size: 180px; font-weight: 900; line-height: 1;
    color: rgba(255,255,255,.035); letter-spacing: -8px; pointer-events: none;
}
.hero h1 {
    margin: 0 0 6px !important; font-size: 32px !important; font-weight: 700 !important;
    background: linear-gradient(120deg,#ffb38a,#c9b8ff 55%,#8ad7ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #9aa1b1 !important; font-size: 14px; max-width: 720px; margin: 0; }

/* ─── primary button ────────────────────────────────────────────────── */
.run-btn button {
    background: linear-gradient(120deg,#ff8a4c,#ff5c92 60%,#c062ff) !important;
    border: none !important; color: #fff !important;
    font-weight: 600 !important; font-size: 15px !important;
    height: 50px !important; border-radius: 12px !important;
    box-shadow: 0 8px 24px -8px rgba(255,92,146,.55) !important;
}
.run-btn button:hover { transform: translateY(-1px); }

/* ─── flow strip (top SVG pipe) ─────────────────────────────────────── */
.flow-strip {
    padding: 18px 22px 22px; margin: 14px 0 18px;
    background: rgba(255,255,255,.025); border: 1px solid rgba(255,255,255,.07);
    border-radius: 18px;
}
.flow-strip h3 {
    margin: 0 0 6px; font-size: 13px; color: #9aa1b1; font-weight: 500;
    text-transform: uppercase; letter-spacing: .12em;
}

/* ─── stage stack — all cards remain visible ───────────────────────── */
.stage-stack {
    display: flex; flex-direction: column; gap: 14px;
}

/* ─── stage detail card ─────────────────────────────────────────────── */
.stage-card {
    padding: 18px 22px; border-radius: 16px;
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.07);
    transition: all .35s ease;
    position: relative;
}
.stage-card.active {
    background: linear-gradient(180deg, rgba(255,138,76,.10), rgba(120,90,255,.06));
    border-color: rgba(255,138,76,.45);
    box-shadow: 0 0 0 1px rgba(255,138,76,.18), 0 12px 32px -10px rgba(255,138,76,.35);
}
.stage-card.active::before {
    content: ""; position: absolute; left: -1px; top: 14%; bottom: 14%; width: 3px;
    background: linear-gradient(180deg, #ff8a4c, #c062ff);
    border-radius: 3px; box-shadow: 0 0 12px rgba(255,138,76,.6);
}
.stage-card.done {
    background: rgba(126,229,163,.03);
    border-color: rgba(126,229,163,.18);
}
.stage-card.done::before {
    content: "✓"; position: absolute; right: 16px; top: 16px;
    width: 22px; height: 22px; border-radius: 50%;
    background: rgba(126,229,163,.18); color: #7ee5a3;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
}
.stage-card.error {
    background: rgba(255,92,92,.05);
    border-color: rgba(255,92,92,.30);
}
.stage-card .head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 12px; gap: 12px;
}
.stage-card h2 {
    margin: 0; font-size: 18px; color: #fff; font-weight: 700;
    letter-spacing: -0.005em;
}
.stage-card .timing {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #ffb38a; background: rgba(255,138,76,.10);
    padding: 3px 10px; border-radius: 999px;
    border: 1px solid rgba(255,138,76,.25);
}
.stage-card .explain {
    color: #b8c0d0; font-size: 13.5px; line-height: 1.6;
    margin: 0 0 14px; max-width: 920px;
}
.stage-card .meta {
    display: flex; flex-wrap: wrap; gap: 6px 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
    color: #c9cdd9; margin-bottom: 12px;
}
.stage-card .meta span {
    padding: 3px 9px; background: rgba(255,255,255,.04);
    border-radius: 6px; border: 1px solid rgba(255,255,255,.06);
}
.stage-card .meta b { color: #ffb38a; font-weight: 600; margin-right: 4px; }

/* ─── whisper architecture — detailed ──────────────────────────────── */
.wa-root {
    padding: 8px 2px 4px;
}
.wa-meta {
    display: flex; flex-wrap: wrap; gap: 6px 10px;
    margin: 0 0 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
    color: #c9cdd9;
}
.wa-meta span {
    padding: 3px 9px; background: rgba(138,215,255,.06);
    border: 1px solid rgba(138,215,255,.16);
    border-radius: 6px;
}
.wa-meta b { color: #8ad7ff; font-weight: 600; margin-right: 5px; }

.wa-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
    position: relative;
}
.wa-col { display: flex; flex-direction: column; gap: 6px; }
.wa-col-head {
    font-size: 11px; color: #c062ff; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 4px;
}

.wa-stage {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px;
    padding: 12px 14px;
    transition: all .25s ease;
}
.wa-stage.wa-active {
    background: linear-gradient(135deg, rgba(255,138,76,.12), rgba(192,98,255,.08));
    border-color: rgba(255,138,76,.50);
    box-shadow: 0 0 0 1px rgba(255,138,76,.20), 0 6px 22px -6px rgba(255,138,76,.40);
}
.wa-stage.wa-big { padding: 14px 16px; }
.wa-title {
    font-size: 13.5px; font-weight: 600; color: #fff; margin-bottom: 4px;
}
.wa-shape {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: #8ad7ff; margin-bottom: 6px;
    padding: 3px 8px; background: rgba(138,215,255,.06);
    border-radius: 6px; display: inline-block;
}
.wa-desc {
    color: #9aa1b1; font-size: 12px; line-height: 1.5;
    margin-top: 4px;
}
.wa-arrow {
    color: #5b6275; font-size: 11px; padding: 4px 4px 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    border-left: 2px solid rgba(255,255,255,.06);
    margin-left: 14px;
}

.wa-stack {
    display: flex; flex-wrap: wrap; gap: 3px;
    margin: 8px 0; align-items: center;
}
.wa-tx {
    width: 14px; height: 22px; border-radius: 3px;
    box-shadow: 0 0 0 1px rgba(0,0,0,.3) inset;
    opacity: 0.85;
}
.wa-more {
    color: #c9cdd9; font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 8px;
}

.wa-block-detail {
    background: rgba(0,0,0,.25);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 11.5px;
    line-height: 1.6;
    color: #b8c0d0;
    margin: 6px 0;
    border: 1px solid rgba(255,255,255,.05);
}
.wa-block-detail > div:first-child {
    color: #c9cdd9; font-weight: 600; margin-bottom: 2px;
    font-family: 'Inter', system-ui;
}
.wa-sub {
    font-family: 'JetBrains Mono', monospace;
    color: #9aa1b1;
}
.wa-sub.wa-cross {
    color: #8ad7ff;
    background: rgba(138,215,255,.06);
    padding: 2px 6px; border-radius: 4px;
    border-left: 2px solid #8ad7ff;
}
.wa-sub b { color: #ffd9b8; }

.wa-bridge {
    width: 100%; height: 50px; margin-top: -6px;
}
.wa-spacer { flex: 1; }

/* ─── glyph mapping ─────────────────────────────────────────────────── */
.mapping {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(78px, 1fr));
    gap: 10px; padding: 8px 4px;
}
.mapping .pair {
    display: flex; flex-direction: column; align-items: center;
    padding: 10px 8px; border-radius: 12px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    opacity: 0; transform: translateY(6px);
    animation: pairIn .35s ease forwards;
}
.mapping .src {
    font-family: 'Noto Sans Tamil','Latha','Nirmala UI', system-ui;
    font-size: 22px; color: #ffd9b8;
}
.mapping .arr { color: #5b6275; font-size: 14px; margin: 2px 0; }
.mapping .dst {
    font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #8ad7ff;
}
.mapping.empty { color: #7d8395; font-size: 13px; padding: 24px; text-align: center; }
@keyframes pairIn { to { opacity: 1; transform: translateY(0); } }

/* ─── token stream ──────────────────────────────────────────────────── */
.tokens {
    font-family: 'Noto Sans Tamil', 'Latha', system-ui;
    font-size: 22px; line-height: 1.7; color: #ffd9b8;
    padding: 12px; border-radius: 10px;
    background: rgba(0,0,0,.30); border: 1px solid rgba(255,255,255,.06);
    min-height: 56px;
}
.tokens .tok {
    display: inline-block; opacity: 0; transform: translateY(4px);
    animation: tokIn .25s ease forwards;
}
.tokens .cursor { color: #ffb38a; animation: blink 1s infinite; }
@keyframes tokIn   { to { opacity: 1; transform: translateY(0); } }
@keyframes blink   { 50% { opacity: 0; } }

/* ─── JSON preview ──────────────────────────────────────────────────── */
.json-card {
    border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(255,255,255,.08);
    background: #0d1018;
}
.json-head {
    display: flex; align-items: center; gap: 6px;
    padding: 8px 12px; background: #14171f;
    border-bottom: 1px solid rgba(255,255,255,.06);
}
.json-head .dot { width: 10px; height: 10px; border-radius: 50%; }
.json-head .r { background: #ff5f56; }
.json-head .y { background: #ffbd2e; }
.json-head .g { background: #27c93f; }
.json-head .fname {
    margin-left: 10px; color: #9aa1b1; font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
}
.json-body {
    margin: 0; padding: 14px 16px; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; line-height: 1.55; color: #c9cdd9;
    overflow-x: auto; max-height: 280px;
}

/* ─── output strip ──────────────────────────────────────────────────── */
.tamil-output textarea {
    font-family: 'Noto Sans Tamil','Latha','Nirmala UI',system-ui !important;
    font-size: 19px !important; line-height: 1.7 !important; color: #ffd9b8 !important;
}
.roman-output textarea {
    font-family: 'JetBrains Mono','Fira Code',ui-monospace,monospace !important;
    font-size: 14px !important; line-height: 1.7 !important; color: #b8e0ff !important;
}

/* ─── bonus playground ──────────────────────────────────────────────── */
.bonus-head {
    margin: 30px 0 14px; padding: 18px 24px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(138,215,255,.08), rgba(192,98,255,.05));
    border: 1px solid rgba(138,215,255,.18);
}
.bonus-head h2 {
    margin: 0 0 4px; font-size: 18px; font-weight: 700; color: #fff;
    background: linear-gradient(120deg,#8ad7ff,#c9b8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.bonus-head p { margin: 0; color: #9aa1b1; font-size: 13px; line-height: 1.55; }
.bonus-head .tag {
    display: inline-block; padding: 2px 9px; margin-right: 8px;
    border-radius: 999px; font-size: 11px; font-weight: 600;
    background: rgba(138,215,255,.18); color: #8ad7ff;
    border: 1px solid rgba(138,215,255,.30); letter-spacing: 0.05em;
}

/* ─── footer ────────────────────────────────────────────────────────── */
.foot {
    margin-top: 22px; padding: 14px 20px; border-radius: 14px;
    background: rgba(255,255,255,.02); border: 1px solid rgba(255,255,255,.06);
    color: #7d8395; font-size: 12px;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}
.foot kbd {
    background: #14171f; border: 1px solid rgba(255,255,255,.10);
    padding: 2px 7px; border-radius: 6px; font-size: 11px; color: #c9cdd9;
    font-family: 'JetBrains Mono', monospace;
}
"""


HERO_HTML = """
<div class="hero">
  <h1>Tamil Speech → Roman Text</h1>
  <p>Watch your audio flow through the entire pipeline — see the actual waveform, the spectrogram Whisper looks at, the chunks, the tokens streaming out, and every glyph mapped to Roman script. Built for understanding, not just transcription.</p>
</div>
"""


FOOTER_HTML = """
<div class="foot">
  <div>Models cached in <kbd>models_cache/</kbd> · Logs in <kbd>app.log</kbd></div>
  <div>Whisper · Gradio · indic-transliteration · matplotlib</div>
</div>
"""


STAGE_TEXT = {
    "idle": (
        "Ready",
        "Upload audio or record from the mic, pick a model and a transliteration scheme, then press <b>Run pipeline</b>. The seven stages below will animate as the audio flows through.",
    ),
    "receive": (
        "1 · Audio received",
        "The browser uploaded your file to the Python backend. We can already plot the raw waveform — every peak is a moment where the air pressure (the sound) crossed zero amplitude. Loud parts have tall peaks; silence is flat.",
    ),
    "decode": (
        "2 · Decoded & normalized",
        "<b>librosa</b> (via ffmpeg) decompressed the audio, downmixed stereo to mono, and resampled to <b>16 kHz</b> — the exact rate Whisper was trained on. The waveform now contains exactly 16,000 floating-point samples per second of audio.",
    ),
    "chunk": (
        "3 · Chunked through the buffer queue",
        "Whisper can only see <b>30 seconds at a time</b>. We slide a 30 s window across the waveform, stepping forward by 25 s so consecutive chunks overlap by 5 s. The overlap is critical — it stops words at the boundary from being cut in half. Each chunk goes onto a <code>queue.Queue()</code>.",
    ),
    "model": (
        "4 · ASR model loaded",
        "Whisper's neural network is now ready. The audio (as a mel-spectrogram) flows through an <b>encoder</b> that turns it into rich audio embeddings, then a <b>decoder</b> that generates Tamil text tokens autoregressively — looking at the audio via cross-attention at every step.",
    ),
    "transcribe": (
        "5 · Transcribing — audio → Tamil tokens",
        "What you see is the actual <b>mel-spectrogram</b> Whisper looks at: time runs left to right, mel-frequency bins go bottom to top, color intensity is sound energy. The decoder reads this image and emits one Tamil token at a time — these tokens stream in below.",
    ),
    "translit": (
        "6 · Tamil → Roman script",
        "<b>indic-transliteration</b> walks the Tamil string character by character, looking up each glyph in the chosen scheme's table and emitting the Roman equivalent. This is <em>not</em> translation — only script conversion. The meaning stays Tamil.",
    ),
    "save": (
        "7 · Output saved",
        "All pipeline results are serialized into a single JSON file you can download below. It contains the audio filename, the model used, the transliteration scheme, the Tamil transcript, and the romanized output.",
    ),
}


def _stage_card(title: str, timing: str, explain: str, meta: list, body: str,
                state: str = "active") -> str:
    """Render a single stage card. `state` ∈ {'active', 'done', 'error'}."""
    timing_html = f'<span class="timing">{html.escape(timing)}</span>' if timing else ""
    meta_html = "".join(
        f'<span><b>{html.escape(str(k))}</b>{html.escape(str(v))}</span>'
        for k, v in meta
    )
    meta_block = f'<div class="meta">{meta_html}</div>' if meta else ""
    return f"""
    <div class="stage-card {state}">
      <div class="head"><h2>{title}</h2>{timing_html}</div>
      <div class="explain">{explain}</div>
      {meta_block}
      {body}
    </div>
    """


def _idle_visual() -> str:
    title, explain = STAGE_TEXT["idle"]
    body = ('<div style="padding:30px 12px;text-align:center;color:#7d8395;font-size:13px">'
            '🎧 waiting for audio …</div>')
    return _stage_card(title, "", explain, [], body)


# ─── Main pipeline generator ────────────────────────────────────────────
def process_audio(audio_input, model_name: str, scheme: str):
    """Generator yielding (flow_svg, stage_html, transcript, romanized, json_path).

    audio_input: (sample_rate, np.ndarray) tuple from gr.Audio(type="numpy"),
                 or None if nothing was provided.
    """
    req_id = uuid.uuid4().hex[:8]
    logger.info("[req=%s] start — model=%s scheme=%s", req_id, model_name, scheme)
    done: list[str] = []
    cards: dict[str, str] = {}  # ordered: stage_key -> rendered card HTML

    def stack_html() -> str:
        if not cards:
            return _idle_visual()
        return '<div class="stage-stack">' + "".join(cards.values()) + '</div>'

    def update(stage_key: str, title: str, timing: str, explain: str,
               meta: list, body: str, state: str = "active") -> None:
        """Replace this stage's card in the running stack. Demote previous active to 'done'."""
        for k, v in cards.items():
            if 'stage-card active' in v and k != stage_key:
                cards[k] = v.replace('stage-card active', 'stage-card done', 1)
        cards[stage_key] = _stage_card(title, timing, explain, meta, body, state)

    def y(current, transcript: str = "", roman: str = "", json_path=None):
        flow = ('<div class="flow-strip"><h3>Pipeline flow</h3>'
                + flow_diagram_html(current, done) + '</div>')
        return flow, stack_html(), transcript, roman, json_path

    # Initial / idle
    yield y(None)

    # ── Normalize input ─────────────────────────────────────────────────
    # gr.Audio(type="numpy") always delivers (sample_rate, np.ndarray) or None.
    # We convert to a WAV temp file immediately so the rest of the pipeline
    # works with a plain filepath regardless of source (mic or upload).
    audio_path: str | None = None

    if audio_input is None:
        msg = "No audio provided. Upload a file or record via the microphone — press ⏹ Stop first, then Run."
        title, explain = STAGE_TEXT["receive"]
        body = f'<div style="color:#ff8585;font-size:14px;padding:20px">⚠ {html.escape(msg)}</div>'
        update("receive", title, "", explain, [], body, state="error")
        yield y(None, msg)
        return

    try:
        import soundfile as sf  # bundled with gradio
        import tempfile as _tmp
        import numpy as _np

        sr_in, data = audio_input
        data = _np.asarray(data)

        if data.ndim > 1:            # stereo → mono
            data = data.mean(axis=-1)
        if data.dtype.kind in "iu":  # int PCM → float32
            data = data.astype(_np.float32) / _np.iinfo(data.dtype).max
        else:
            data = data.astype(_np.float32)

        if len(data) < 100 or float(_np.max(_np.abs(data))) < 1e-6:
            msg = ("Recording appears silent or empty — make sure the browser "
                   "granted microphone permission, speak into the mic, press ⏹ Stop, "
                   "wait for the waveform to appear, then click Run.")
            title, explain = STAGE_TEXT["receive"]
            body = f'<div style="color:#ff8585;font-size:14px;padding:20px">⚠ {html.escape(msg)}</div>'
            update("receive", title, "", explain, [], body, state="error")
            yield y(None, msg)
            return

        tmp = _tmp.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, data, sr_in)
        audio_path = tmp.name
        logger.info("[req=%s] input → WAV: %s  (%d samples @ %d Hz)",
                    req_id, audio_path, len(data), sr_in)

    except Exception as e:
        logger.exception("[req=%s] failed to process audio input", req_id)
        msg = f"Could not read audio: {e}"
        title, explain = STAGE_TEXT["receive"]
        body = f'<div style="color:#ff8585;font-size:14px;padding:20px">⚠ {html.escape(msg)}</div>'
        update("receive", title, "", explain, [], body, state="error")
        yield y(None, msg)
        return

    backend = MODEL_CONFIGS.get(model_name, {}).get("backend", "local")

    # ── Stage 1: receive ────────────────────────────────────────────────
    t0 = time.perf_counter()
    size_kb = os.path.getsize(audio_path) / 1024
    ext = (os.path.splitext(audio_path)[1] or "").lstrip(".") or "?"
    title, explain = STAGE_TEXT["receive"]
    update("receive", title, "loading…", explain, [
        ("file ", os.path.basename(audio_path)[:36]),
        ("size ", f"{size_kb:.1f} KB"),
        ("type ", ext),
    ], "")
    yield y("receive")

    # ── Stage 2: decode & normalize ─────────────────────────────────────
    try:
        audio = load_audio(audio_path)
    except Exception as e:
        logger.exception("[req=%s] decode failed", req_id)
        title, explain = STAGE_TEXT["decode"]
        body = f'<div style="color:#ff8585;padding:20px">⚠ Decode failed: {html.escape(str(e))}</div>'
        update("decode", title, "", explain, [], body, state="error")
        yield y(None, f"Error loading audio: {e}")
        return

    done.append("receive")
    duration = len(audio) / SAMPLE_RATE
    title, explain = STAGE_TEXT["decode"]
    update("decode", title, f"{(time.perf_counter()-t0)*1000:.0f} ms", explain, [
        ("samples ", f"{len(audio):,}"),
        ("rate ", f"{SAMPLE_RATE} Hz"),
        ("duration ", f"{duration:.2f} s"),
        ("dtype ", "float32"),
    ], waveform_html(audio))
    yield y("decode")

    # ── Stage 3: chunk ──────────────────────────────────────────────────
    done.append("decode")
    t0 = time.perf_counter()
    title, explain = STAGE_TEXT["chunk"]
    if backend == "groq":
        body = ('<div style="color:#8ad7ff;background:rgba(138,215,255,.06);'
                'border:1px solid rgba(138,215,255,.20);padding:14px 18px;border-radius:10px;'
                'font-size:13px">ℹ Skipped — Groq Cloud chunks server-side. '
                'The full audio file goes straight to <code>whisper-large-v3-turbo</code>.</div>')
        update("chunk", title, "—", explain, [("backend ", "Groq Cloud (skip)")], body)
        chunks = []
        n_chunks = 1
    else:
        buf = AudioBufferManager()
        n_chunks = buf.enqueue(audio)
        chunks = buf.drain()
        update("chunk", title, f"{(time.perf_counter()-t0)*1000:.0f} ms", explain, [
            ("chunks ", str(n_chunks)),
            ("window ", f"{CHUNK_SECONDS} s"),
            ("overlap ", f"{STRIDE_SECONDS} s"),
            ("step ", f"{CHUNK_SECONDS-STRIDE_SECONDS} s"),
        ], chunked_waveform_html(audio, n_chunks))
    yield y("chunk")

    # ── Stage 4: load model ─────────────────────────────────────────────
    done.append("chunk")
    t0 = time.perf_counter()
    title, explain = STAGE_TEXT["model"]
    if backend == "groq":
        update("model", title, "remote", explain, [
            ("backend ", "Groq Cloud"),
            ("model ", "whisper-large-v3-turbo"),
            ("hardware ", "LPU (Groq)"),
        ], whisper_arch_html(model_name="whisper-large-v3-turbo", active="encoder"))
    else:
        try:
            load_asr_model(model_name)
        except Exception as e:
            logger.exception("[req=%s] model load failed", req_id)
            body = f'<div style="color:#ff8585;padding:20px">⚠ {html.escape(str(e))}</div>'
            update("model", title, "", explain, [], body, state="error")
            yield y(None, f"Model load failed: {e}")
            return
        hf_id = MODEL_CONFIGS[model_name].get("hf_id", "?")
        update("model", title, f"{time.perf_counter()-t0:.2f} s", explain, [
            ("backend ", "Local"),
            ("model ", hf_id.split("/")[-1]),
            ("device ", "CPU"),
        ], whisper_arch_html(model_name=hf_id.split("/")[-1], active="encoder"))
    yield y("model")

    # ── Stage 5: transcribe ─────────────────────────────────────────────
    done.append("model")
    t0 = time.perf_counter()
    title, explain = STAGE_TEXT["transcribe"]
    accumulated_tokens: list[str] = []

    try:
        if backend == "groq":
            body = mel_spectrogram_html(audio) + token_stream_html([])
            update("transcribe", title, "calling Groq…", explain,
                   [("backend ", "groq cloud")], body)
            yield y("transcribe")

            transcript = transcribe_with_groq(audio_path)
            accumulated_tokens = transcript.split()
            body = mel_spectrogram_html(audio) + token_stream_html(accumulated_tokens[:40])
            update("transcribe", title, f"{time.perf_counter()-t0:.2f} s", explain, [
                ("chars ", str(len(transcript))),
                ("tokens ", str(len(accumulated_tokens))),
                ("backend ", "groq"),
            ], body)
            yield y("transcribe", transcript)
        else:
            parts = []
            for i, chunk in enumerate(chunks):
                body = (chunked_waveform_html(audio, n_chunks, active_chunk=i) +
                        mel_spectrogram_html(chunk) +
                        token_stream_html(accumulated_tokens[:40]))
                update("transcribe", title,
                       f"chunk {i+1}/{len(chunks)} · {time.perf_counter()-t0:.1f}s",
                       explain, [
                           ("chunk ", f"{i+1}/{len(chunks)}"),
                           ("model ", MODEL_CONFIGS[model_name]["hf_id"].split("/")[-1]),
                       ], body)
                yield y("transcribe")

                text = transcribe_audio(chunk, model_name)
                if text:
                    parts.append(text)
                    accumulated_tokens.extend(text.split())
            transcript = " ".join(parts)
            body = (chunked_waveform_html(audio, n_chunks) +
                    mel_spectrogram_html(audio) +
                    token_stream_html(accumulated_tokens[:40]))
            update("transcribe", title, f"{time.perf_counter()-t0:.2f} s", explain, [
                ("chars ", str(len(transcript))),
                ("tokens ", str(len(accumulated_tokens))),
                ("chunks ", str(len(chunks))),
            ], body)
            yield y("transcribe", transcript)
    except EnvironmentError as e:
        body = f'<div style="color:#ff8585;padding:20px">⚠ {html.escape(str(e))}</div>'
        update("transcribe", title, "", explain, [], body, state="error")
        yield y(None, f"Configuration error: {e}")
        return
    except Exception as e:
        logger.exception("[req=%s] transcribe failed", req_id)
        body = f'<div style="color:#ff8585;padding:20px">⚠ {html.escape(str(e))}</div>'
        update("transcribe", title, "", explain, [], body, state="error")
        yield y(None, f"Transcription error: {e}")
        return

    if not transcript:
        body = '<div style="color:#ff8585;padding:20px">⚠ empty transcript — audio may be silent or unsupported</div>'
        update("transcribe", title, "", explain, [], body, state="error")
        yield y(None, "Could not transcribe audio (silent or unsupported).")
        return

    # ── Stage 6: transliterate ──────────────────────────────────────────
    done.append("transcribe")
    t0 = time.perf_counter()
    title, explain = STAGE_TEXT["translit"]
    transliterated = transliterate_tamil_to_latin(transcript, scheme)
    update("translit", title, f"{(time.perf_counter()-t0)*1000:.0f} ms", explain, [
        ("scheme ", scheme),
        ("in_chars ", str(len(transcript))),
        ("out_chars ", str(len(transliterated))),
    ], glyph_mapping_html(transcript, transliterated))
    yield y("translit", transcript, transliterated)

    # ── Stage 7: save ───────────────────────────────────────────────────
    done.append("translit")
    t0 = time.perf_counter()
    title, explain = STAGE_TEXT["save"]
    json_path = save_output_json(audio_path, model_name, scheme, transcript, transliterated)
    payload = {
        "audio_file": os.path.basename(audio_path),
        "model": model_name,
        "transliteration_scheme": scheme,
        "transcript_tamil": transcript[:80] + ("…" if len(transcript) > 80 else ""),
        "transliterated_roman": transliterated[:80] + ("…" if len(transliterated) > 80 else ""),
    }
    update("save", title, f"{(time.perf_counter()-t0)*1000:.0f} ms", explain, [
        ("path ", os.path.basename(json_path)),
        ("dir ", "outputs/"),
    ], json_preview_html(payload), state="done")
    done.append("save")
    # Promote the last active card to done as well
    for k in cards:
        cards[k] = cards[k].replace('stage-card active', 'stage-card done')
    yield y(None, transcript, transliterated, json_path)

    logger.info("[req=%s] done", req_id)


def reverse_transliterate(roman_text: str, scheme: str) -> str:
    """Convert romanized Tamil (e.g. 'vaNakkam') back into Tamil script."""
    if not roman_text or not roman_text.strip():
        return ""
    try:
        return transliterate_latin_to_tamil(roman_text, scheme)
    except Exception as e:
        logger.exception("Reverse transliteration failed")
        return f"Error: {e}"


def _model_label(name: str) -> str:
    cfg = MODEL_CONFIGS[name]
    return f"{cfg.get('params', '?')} params · {cfg.get('description', '')}"


def get_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#0b0d12",
        body_text_color="#e6e8ee",
        background_fill_primary="rgba(255,255,255,0.03)",
        background_fill_secondary="rgba(255,255,255,0.02)",
        border_color_primary="rgba(255,255,255,0.08)",
        block_radius="16px",
        block_shadow="none",
        button_primary_background_fill="linear-gradient(120deg,#ff8a4c,#c062ff)",
        button_primary_text_color="#fff",
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Tamil ASR — pipeline visualizer", analytics_enabled=False) as demo:
        gr.HTML(HERO_HTML)

        # ─── Controls row ────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=5):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="Audio input  ·  mic: press ⏺ Record, speak, press ⏹ Stop, then click Run",
                    interactive=True,
                )
            with gr.Column(scale=3):
                model_selector = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value=DEFAULT_MODEL,
                    label="Model",
                    info=_model_label(DEFAULT_MODEL),
                )
                scheme_selector = gr.Dropdown(
                    choices=TRANSLITERATION_SCHEMES,
                    value=DEFAULT_SCHEME,
                    label="Transliteration scheme",
                )
            with gr.Column(scale=2):
                run_btn = gr.Button("Run pipeline  →", variant="primary",
                                    elem_classes=["run-btn"])

        # ─── Live flow strip (animates as pipeline runs) ────────────
        flow_view = gr.HTML(
            '<div class="flow-strip"><h3>Pipeline flow</h3>'
            + flow_diagram_html(None, []) + '</div>'
        )

        # ─── Live stage detail card ─────────────────────────────────
        stage_view = gr.HTML(_idle_visual())

        # ─── Final outputs ───────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                transcript_out = gr.Textbox(
                    label="Tamil transcript", lines=4, interactive=False,
                    placeholder="தமிழ் transcript appears here …",
                    elem_classes=["tamil-output"],
                )
            with gr.Column():
                transliterated_out = gr.Textbox(
                    label="Romanized output", lines=4, interactive=False,
                    placeholder="Roman-script version appears here …",
                    elem_classes=["roman-output"],
                )
        download_file = gr.File(label="Download JSON output",
                                file_count="single", interactive=False)

        # ─── Bonus: Roman → Tamil playground ─────────────────────────
        gr.HTML(
            '<div class="bonus-head">'
            '  <h2><span class="tag">BONUS</span>Roman → Tamil script</h2>'
            '  <p>Type romanized Tamil (e.g. <code>vaNakkam</code> in ITRANS, or <code>vaṇakkam</code> in ISO/IAST) '
            '  and get the Tamil script back. This is the reverse direction of the main pipeline — same '
            '  <code>indic-transliteration</code> library, opposite mapping.</p>'
            '</div>'
        )
        with gr.Row():
            with gr.Column(scale=5):
                reverse_input = gr.Textbox(
                    label="Roman input",
                    placeholder="e.g. vaNakkam ulagam (ITRANS) or vaṇakkam ulakam (ISO)",
                    lines=3,
                    elem_classes=["roman-output"],
                )
                reverse_scheme = gr.Dropdown(
                    choices=TRANSLITERATION_SCHEMES,
                    value=DEFAULT_SCHEME,
                    label="Source scheme",
                    info="Which romanization convention is your input in?",
                )
                reverse_btn = gr.Button("→ Convert to Tamil", variant="primary",
                                        elem_classes=["run-btn"])
            with gr.Column(scale=5):
                reverse_output = gr.Textbox(
                    label="Tamil script",
                    placeholder="தமிழ் ஸ்கிரிப்ட் இங்கே ...",
                    lines=5,
                    interactive=False,
                    elem_classes=["tamil-output"],
                )

        reverse_btn.click(
            fn=reverse_transliterate,
            inputs=[reverse_input, reverse_scheme],
            outputs=[reverse_output],
        )
        # Also fire on Enter inside the textbox
        reverse_input.submit(
            fn=reverse_transliterate,
            inputs=[reverse_input, reverse_scheme],
            outputs=[reverse_output],
        )

        gr.HTML(FOOTER_HTML)

        run_btn.click(
            fn=process_audio,
            inputs=[audio_input, model_selector, scheme_selector],
            outputs=[flow_view, stage_view, transcript_out, transliterated_out, download_file],
        )

    return demo
