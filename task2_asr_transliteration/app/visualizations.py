"""Visual helpers for the pipeline viewer.

All functions return either base64-encoded PNGs (for matplotlib plots) or
inline HTML/SVG ready to embed in a Gradio HTML component.
"""
import base64
import io
import html
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless backend — no GUI, no warnings
import matplotlib.pyplot as plt
import numpy as np

from models.model_config import SAMPLE_RATE, CHUNK_SECONDS, STRIDE_SECONDS


_BG = "#0b0d12"
_FG = "#e6e8ee"
_MUTED = "#7d8395"
_ACCENT = "#ff8a4c"
_ACCENT2 = "#c062ff"


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=_BG, edgecolor="none", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:100%;border-radius:10px;display:block">'


# ──────────────────────────────────────────────────────────────────────
# Waveform — basic and chunked
# ──────────────────────────────────────────────────────────────────────
def waveform_html(audio: np.ndarray, title: str = "Waveform") -> str:
    fig, ax = plt.subplots(figsize=(11, 2.4), facecolor=_BG)
    ax.set_facecolor(_BG)
    t = np.linspace(0, len(audio) / SAMPLE_RATE, len(audio))
    ax.fill_between(t, audio, -audio, color=_ACCENT, alpha=0.85, linewidth=0)
    _style_axes(ax)
    ax.set_xlim(0, len(audio) / SAMPLE_RATE)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("time (s)", color="#9aa1b1", fontsize=10)
    return _img(_to_b64(fig), title)


def chunked_waveform_html(audio: np.ndarray, n_chunks: int, active_chunk: int | None = None) -> str:
    """Waveform with translucent bands marking each 30 s chunk; overlap hatched."""
    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=_BG)
    ax.set_facecolor(_BG)
    duration = len(audio) / SAMPLE_RATE
    t = np.linspace(0, duration, len(audio))
    ax.fill_between(t, audio, -audio, color=_ACCENT, alpha=0.55, linewidth=0)

    step = CHUNK_SECONDS - STRIDE_SECONDS
    for i in range(n_chunks):
        s = i * step
        e = min(s + CHUNK_SECONDS, duration)
        color = _ACCENT2 if i == active_chunk else "#5b6275"
        alpha = 0.30 if i == active_chunk else 0.12
        ax.axvspan(s, e, alpha=alpha, color=color, zorder=1)
        # overlap zone shading
        if i < n_chunks - 1:
            overlap_start = s + step
            overlap_end = e
            ax.axvspan(overlap_start, overlap_end, alpha=0.20, facecolor="none",
                       edgecolor="#8ad7ff", hatch="//", linewidth=0)
        ax.text((s + e) / 2, 0.85, f"chunk {i + 1}", color=_FG, ha="center",
                fontsize=10, fontweight="bold")

    _style_axes(ax)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("time (s) — striped zones = 5 s overlap between chunks", color="#9aa1b1", fontsize=10)
    return _img(_to_b64(fig), "chunked waveform")


# ──────────────────────────────────────────────────────────────────────
# Mel-spectrogram (Whisper's actual input)
# ──────────────────────────────────────────────────────────────────────
def mel_spectrogram_html(audio: np.ndarray) -> str:
    import librosa  # local import — keeps cold start light if visualisations unused
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=_BG)
    ax.set_facecolor(_BG)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=80, fmax=8000, hop_length=160
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=SAMPLE_RATE, hop_length=160,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma", fmax=8000,
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    cbar.ax.tick_params(colors=_MUTED, labelsize=8)
    cbar.outline.set_edgecolor("#3a3f4d")
    _style_axes(ax)
    ax.set_xlabel("time (s)", color="#9aa1b1", fontsize=10)
    ax.set_ylabel("mel frequency", color="#9aa1b1", fontsize=10)
    return _img(_to_b64(fig), "mel spectrogram")


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3a3f4d")
    ax.spines["bottom"].set_color("#3a3f4d")
    ax.tick_params(colors=_MUTED, labelsize=8)


# ──────────────────────────────────────────────────────────────────────
# Flow diagram — the moving "audio packet" along a 7-stage pipe
# ──────────────────────────────────────────────────────────────────────
STAGES = [
    ("receive",    "①", "Receive"),
    ("decode",     "②", "Decode"),
    ("chunk",      "③", "Chunk"),
    ("model",      "④", "Model"),
    ("transcribe", "⑤", "Transcribe"),
    ("translit",   "⑥", "Romanize"),
    ("save",       "⑦", "Save"),
]


def flow_diagram_html(current: str | None, done: Iterable[str]) -> str:
    """SVG flow diagram with an animated packet riding the pipe to `current`.

    Args:
        current: key of currently active stage, or None when idle.
        done: iterable of stage keys already completed.
    """
    done_set = set(done)
    width = 1080
    height = 110
    pad_x = 60
    inner_w = width - 2 * pad_x
    step = inner_w / (len(STAGES) - 1)
    cy = 55

    # tube (background pipe)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;height:auto;display:block">']
    parts.append(f'<defs>'
                 f'<linearGradient id="pipeGrad" x1="0" x2="1" y1="0" y2="0">'
                 f'  <stop offset="0%" stop-color="#ff8a4c"/>'
                 f'  <stop offset="50%" stop-color="#ff5c92"/>'
                 f'  <stop offset="100%" stop-color="#c062ff"/>'
                 f'</linearGradient>'
                 f'<linearGradient id="filled" x1="0" x2="1" y1="0" y2="0">'
                 f'  <stop offset="0%" stop-color="#ff8a4c"/>'
                 f'  <stop offset="100%" stop-color="#c062ff"/>'
                 f'</linearGradient>'
                 f'<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
                 f'  <feGaussianBlur stdDeviation="3" result="b"/>'
                 f'  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
                 f'</filter>'
                 f'</defs>')
    # pipe rail
    parts.append(f'<line x1="{pad_x}" y1="{cy}" x2="{width-pad_x}" y2="{cy}" '
                 f'stroke="#2a2f3d" stroke-width="6" stroke-linecap="round"/>')

    # determine progress fraction (0..1) for filled section
    current_idx = next((i for i, (k, _, _) in enumerate(STAGES) if k == current), -1)
    if current_idx < 0 and done_set:
        # done but nothing active — fill to last done
        done_idxs = [i for i, (k, _, _) in enumerate(STAGES) if k in done_set]
        current_idx = max(done_idxs) if done_idxs else -1
    progress_x = pad_x + (current_idx * step if current_idx >= 0 else 0)

    # filled portion
    if current_idx >= 0:
        parts.append(f'<line x1="{pad_x}" y1="{cy}" x2="{progress_x}" y2="{cy}" '
                     f'stroke="url(#filled)" stroke-width="6" stroke-linecap="round"/>')

    # nodes
    for i, (key, glyph, label) in enumerate(STAGES):
        x = pad_x + i * step
        if key == current:
            fill, txt, glow = "url(#filled)", "#fff", 'filter="url(#glow)"'
        elif key in done_set:
            fill, txt, glow = "#7ee5a3", "#0b0d12", ""
        else:
            fill, txt, glow = "#1a1e29", "#7d8395", ""
        parts.append(
            f'<g {glow}>'
            f'<circle cx="{x}" cy="{cy}" r="18" fill="{fill}" stroke="#0b0d12" stroke-width="3"/>'
            f'<text x="{x}" y="{cy+5}" text-anchor="middle" font-size="15" font-weight="700" '
            f'      fill="{txt}" font-family="Inter, system-ui">{glyph}</text>'
            f'</g>'
            f'<text x="{x}" y="{cy+38}" text-anchor="middle" font-size="11" '
            f'      fill="#c9cdd9" font-family="Inter, system-ui" font-weight="500" '
            f'      letter-spacing="0.04em">{html.escape(label)}</text>'
        )

    # animated packet — only when something is active
    if current_idx >= 0 and current is not None:
        packet_x = progress_x
        parts.append(
            f'<g transform="translate({packet_x},{cy})">'
            f'  <circle r="7" fill="#fff" filter="url(#glow)">'
            f'    <animate attributeName="r" values="6;9;6" dur="0.9s" repeatCount="indefinite"/>'
            f'    <animate attributeName="opacity" values="0.6;1;0.6" dur="0.9s" repeatCount="indefinite"/>'
            f'  </circle>'
            f'</g>'
        )

    parts.append("</svg>")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Whisper architecture — full detailed diagram
# ──────────────────────────────────────────────────────────────────────
# Architecture specs sourced from the Whisper paper + HF model configs.
# Keys match the model_config.py hf_id suffix.
WHISPER_SPECS = {
    "whisper-small":  {"enc": 12, "dec": 12, "dim": 768,  "heads": 12, "ffn": 3072, "vocab": 51865, "params": "244 M"},
    "whisper-medium": {"enc": 24, "dec": 24, "dim": 1024, "heads": 16, "ffn": 4096, "vocab": 51865, "params": "769 M"},
    "whisper-large-v3-turbo": {"enc": 32, "dec": 4, "dim": 1280, "heads": 20, "ffn": 5120, "vocab": 51866, "params": "809 M"},
}


def whisper_arch_html(model_name: str = "whisper-small", active: str = "encoder",
                      chunk_seconds: int = 30) -> str:
    """Detailed Whisper architecture diagram with real specs and tensor shapes.

    Args:
        model_name: matches a key in WHISPER_SPECS (or its hf_id suffix).
        active: which logical step is currently running. One of
                'mel', 'encoder', 'cross', 'decoder', 'tokens'.
        chunk_seconds: chunk window length, used for sample-count math.
    """
    key = model_name.split("/")[-1]
    spec = WHISPER_SPECS.get(key, WHISPER_SPECS["whisper-small"])

    samples = SAMPLE_RATE * chunk_seconds          # 480,000 for 30s
    mel_frames = chunk_seconds * 100               # mel hop=10ms → 100 frames/s
    enc_tokens = mel_frames // 2                   # conv stem stride=2

    enc_depth = spec["enc"]
    dec_depth = spec["dec"]

    def stage_class(name: str) -> str:
        return "wa-stage" + (" wa-active" if active == name else "")

    # Block strip helper — render N small rectangles representing a transformer stack
    def stack_strip(n: int, color: str) -> str:
        # cap visual to 12 cells to keep layout tight, label overflow
        shown = min(n, 12)
        cells = "".join(
            f'<div class="wa-tx" style="background:{color}"></div>' for _ in range(shown)
        )
        more = f'<span class="wa-more">×{n}</span>' if n > 0 else ""
        return f'<div class="wa-stack">{cells}{more}</div>'

    # Self-contained styles: HF Spaces SSR doesn't always propagate CUSTOM_CSS into
    # inner gr.HTML content, so the rules travel with the markup.
    arch_css = """
<style>
.wa-root{padding:8px 2px 4px}
.wa-meta{display:flex;flex-wrap:wrap;gap:6px 10px;margin:0 0 14px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#c9cdd9}
.wa-meta span{padding:3px 9px;background:rgba(138,215,255,.06);border:1px solid rgba(138,215,255,.16);border-radius:6px}
.wa-meta b{color:#8ad7ff;font-weight:600;margin-right:5px}
.wa-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;position:relative}
.wa-col{display:flex;flex-direction:column;gap:6px}
.wa-col-head{font-size:11px;color:#c062ff;font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px}
.wa-stage{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px 14px;transition:all .25s ease}
.wa-stage.wa-active{background:linear-gradient(135deg,rgba(255,138,76,.12),rgba(192,98,255,.08));border-color:rgba(255,138,76,.50);box-shadow:0 0 0 1px rgba(255,138,76,.20),0 6px 22px -6px rgba(255,138,76,.40)}
.wa-stage.wa-big{padding:14px 16px}
.wa-title{font-size:13.5px;font-weight:600;color:#fff;margin-bottom:4px}
.wa-shape{font-family:'JetBrains Mono',monospace;font-size:11px;color:#8ad7ff;margin-bottom:6px;padding:3px 8px;background:rgba(138,215,255,.06);border-radius:6px;display:inline-block}
.wa-desc{color:#9aa1b1;font-size:12px;line-height:1.5;margin-top:4px}
.wa-arrow{color:#5b6275;font-size:11px;padding:4px 4px 4px 14px;font-family:'JetBrains Mono',monospace;border-left:2px solid rgba(255,255,255,.06);margin-left:14px}
.wa-stack{display:flex;flex-wrap:wrap;gap:3px;margin:8px 0;align-items:center}
.wa-tx{width:14px;height:22px;border-radius:3px;box-shadow:0 0 0 1px rgba(0,0,0,.3) inset;opacity:.85}
.wa-more{color:#c9cdd9;font-size:11px;font-family:'JetBrains Mono',monospace;margin-left:8px}
.wa-block-detail{background:rgba(0,0,0,.25);border-radius:8px;padding:10px 12px;font-size:11.5px;line-height:1.6;color:#b8c0d0;margin:6px 0;border:1px solid rgba(255,255,255,.05)}
.wa-block-detail>div:first-child{color:#c9cdd9;font-weight:600;margin-bottom:2px;font-family:'Inter',system-ui}
.wa-sub{font-family:'JetBrains Mono',monospace;color:#9aa1b1}
.wa-sub.wa-cross{color:#8ad7ff;background:rgba(138,215,255,.06);padding:2px 6px;border-radius:4px;border-left:2px solid #8ad7ff}
.wa-sub b{color:#ffd9b8}
.wa-bridge{width:100%;height:50px;margin-top:-6px}
.wa-spacer{flex:1}
</style>
"""
    html_out = arch_css + f"""
<div class="wa-root">

  <div class="wa-meta">
    <span><b>model</b> {key}</span>
    <span><b>params</b> {spec['params']}</span>
    <span><b>d_model</b> {spec['dim']}</span>
    <span><b>heads</b> {spec['heads']}</span>
    <span><b>encoder</b> {spec['enc']} blocks</span>
    <span><b>decoder</b> {spec['dec']} blocks</span>
    <span><b>vocab</b> {spec['vocab']:,}</span>
  </div>

  <div class="wa-grid">
    <!-- LEFT COLUMN: data shapes flowing down -->
    <div class="wa-col">

      <div class="{stage_class('mel')}">
        <div class="wa-title">🎙️ Audio waveform</div>
        <div class="wa-shape">shape: ({samples:,},)  ·  float32</div>
        <div class="wa-desc">{chunk_seconds} s of 16 kHz mono audio = {samples:,} samples. This is what came out of the chunk buffer.</div>
      </div>

      <div class="wa-arrow">↓ STFT (n_fft=400, hop=160) + mel filterbank</div>

      <div class="{stage_class('mel')}">
        <div class="wa-title">📊 Log-mel spectrogram</div>
        <div class="wa-shape">shape: (80, {mel_frames})  ·  float32</div>
        <div class="wa-desc">80 mel-frequency bins × {mel_frames} time frames (10 ms each). Magnitude in log dB — Whisper was trained on exactly this format.</div>
      </div>

      <div class="wa-arrow">↓ Conv1d stem (kernel=3, stride=2) × 2 + GELU</div>

      <div class="{stage_class('encoder')}">
        <div class="wa-title">🔵 Encoder input embeddings</div>
        <div class="wa-shape">shape: ({enc_tokens}, {spec['dim']})  ·  float32</div>
        <div class="wa-desc">Conv stem downsamples time by 2× and projects to d_model = {spec['dim']}. Sinusoidal positional encoding is added here.</div>
      </div>

      <div class="wa-arrow">↓ pass through encoder stack</div>

      <div class="{stage_class('encoder')} wa-big">
        <div class="wa-title">🔵 ENCODER — {enc_depth}× transformer blocks</div>
        {stack_strip(enc_depth, '#ff8a4c')}
        <div class="wa-block-detail">
          <div>each block:</div>
          <div class="wa-sub">→ LayerNorm</div>
          <div class="wa-sub">→ Multi-head self-attention ({spec['heads']} heads, d_k = {spec['dim']//spec['heads']})</div>
          <div class="wa-sub">→ Residual + LayerNorm</div>
          <div class="wa-sub">→ Feed-forward (d_ff = {spec['ffn']:,}, GELU)</div>
          <div class="wa-sub">→ Residual</div>
        </div>
        <div class="wa-desc">Self-attention lets every audio frame look at every other frame to build context — that's how Whisper handles coarticulation, prosody, and Tamil's long-distance vowel harmony.</div>
      </div>

      <div class="wa-arrow">↓ encoder output: ({enc_tokens}, {spec['dim']})</div>

      <div class="wa-spacer"></div>
    </div>

    <!-- RIGHT COLUMN: decoder side -->
    <div class="wa-col">
      <div class="wa-col-head">Decoder (autoregressive)</div>

      <div class="{stage_class('decoder')}">
        <div class="wa-title">🟣 Decoder input</div>
        <div class="wa-shape">tokens so far: [&lt;SOT&gt;, &lt;|ta|&gt;, &lt;|transcribe|&gt;, ...]</div>
        <div class="wa-desc">Decoder is autoregressive — at step t, it sees only the tokens it has already emitted (plus special tokens that prime it for Tamil transcription).</div>
      </div>

      <div class="wa-arrow">↓ token embeddings + positional encoding</div>

      <div class="{stage_class('decoder')} wa-big">
        <div class="wa-title">🟣 DECODER — {dec_depth}× transformer blocks</div>
        {stack_strip(dec_depth, '#c062ff')}
        <div class="wa-block-detail">
          <div>each block:</div>
          <div class="wa-sub">→ Masked self-attention (can only see past tokens)</div>
          <div class="wa-sub wa-cross">→ <b>Cross-attention</b> ⇽ encoder output</div>
          <div class="wa-sub">→ Feed-forward (d_ff = {spec['ffn']:,}, GELU)</div>
        </div>
        <div class="wa-desc">Cross-attention is where audio meets text — every Tamil token "queries" the encoded audio to decide what to emit next.</div>
      </div>

      <div class="wa-arrow">↓ hidden states</div>

      <div class="{stage_class('tokens')}">
        <div class="wa-title">📐 Output head — tied to token embeddings</div>
        <div class="wa-shape">Linear({spec['dim']} → {spec['vocab']:,}) + softmax</div>
        <div class="wa-desc">Project to vocabulary, softmax, sample (or argmax) one token. Feed it back into the decoder and repeat until &lt;EOT&gt;.</div>
      </div>

      <div class="wa-arrow">↓ next token</div>

      <div class="{stage_class('tokens')}">
        <div class="wa-title">🔤 Tamil text tokens</div>
        <div class="wa-shape">e.g. ['வ', 'ண', 'க்', 'கம்', ' ', 'உ', 'ல', 'க', 'ம்']</div>
        <div class="wa-desc">BPE detokenizer assembles tokens back into Tamil Unicode text. This is the transcript you see at the bottom of the UI.</div>
      </div>

    </div>
  </div>

  <!-- Cross-attention bridge -->
  <svg class="wa-bridge" viewBox="0 0 400 80" preserveAspectRatio="none">
    <defs>
      <linearGradient id="bridgeGrad" x1="0" x2="1">
        <stop offset="0%" stop-color="#ff8a4c"/>
        <stop offset="100%" stop-color="#c062ff"/>
      </linearGradient>
    </defs>
    <path d="M 20 20 Q 200 80 380 20" fill="none"
          stroke="url(#bridgeGrad)" stroke-width="2.5" stroke-dasharray="6 4">
      <animate attributeName="stroke-dashoffset" from="0" to="-20"
               dur="1.2s" repeatCount="indefinite"/>
    </path>
    <text x="200" y="65" text-anchor="middle" fill="#8ad7ff" font-size="11"
          font-family="Inter, system-ui">cross-attention</text>
  </svg>

</div>"""
    return html_out


# ──────────────────────────────────────────────────────────────────────
# Glyph mapping — Tamil → Roman, animated character pairs
# ──────────────────────────────────────────────────────────────────────
def glyph_mapping_html(tamil: str, roman: str, max_pairs: int = 18) -> str:
    """Render Tamil glyphs alongside their Roman counterparts with arrows.

    We don't attempt true glyph-level alignment (akshara segmentation is non-trivial);
    we show the leading characters with a sequential fade-in to make the mapping vivid.
    """
    # Take a representative slice so we don't overwhelm the UI on long transcripts.
    src_chars = [c for c in tamil if not c.isspace()][:max_pairs]
    empty_style = "color:#7d8395;font-size:13px;padding:24px;text-align:center"
    if not src_chars:
        return f'<div style="{empty_style}">— no characters —</div>'

    # Inline ALL styles — HF Spaces SSR strips/scopes external CSS for gr.HTML content.
    grid_style = (
        "display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(78px,1fr));"
        "gap:10px;padding:8px 4px;"
    )
    pair_style = (
        "display:flex;flex-direction:column;align-items:center;"
        "padding:10px 8px;border-radius:12px;"
        "background:rgba(255,255,255,.03);"
        "border:1px solid rgba(255,255,255,.07);"
    )
    src_style = (
        "font-family:'Noto Sans Tamil','Latha','Nirmala UI',system-ui;"
        "font-size:22px;color:#ffd9b8;"
    )
    arr_style = "color:#5b6275;font-size:14px;margin:2px 0"
    dst_style = "font-family:'JetBrains Mono',monospace;font-size:14px;color:#8ad7ff"

    ratio = max(1, len(roman) // max(1, len(src_chars)))
    pairs = []
    cursor = 0
    for ch in src_chars:
        slice_end = min(len(roman), cursor + ratio)
        dst = roman[cursor:slice_end].strip() or "·"
        cursor = slice_end
        pairs.append(
            f'<div style="{pair_style}">'
            f'<div style="{src_style}">{html.escape(ch)}</div>'
            f'<div style="{arr_style}">→</div>'
            f'<div style="{dst_style}">{html.escape(dst[:4])}</div>'
            f'</div>'
        )
    return f'<div style="{grid_style}">' + "".join(pairs) + "</div>"


# ──────────────────────────────────────────────────────────────────────
# Token stream — typewriter effect for Tamil tokens emerging from decoder
# ──────────────────────────────────────────────────────────────────────
def token_stream_html(tokens: list[str]) -> str:
    if not tokens:
        return '<div class="tokens"><span class="cursor">▎</span></div>'
    spans = []
    for i, tok in enumerate(tokens):
        d = i * 0.05
        spans.append(
            f'<span class="tok" style="animation-delay:{d:.2f}s">{html.escape(tok)}</span>'
        )
    return '<div class="tokens">' + " ".join(spans) + '<span class="cursor">▎</span></div>'


# ──────────────────────────────────────────────────────────────────────
# JSON output preview
# ──────────────────────────────────────────────────────────────────────
def json_preview_html(payload: dict) -> str:
    import json
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        '<div class="json-card">'
        f'  <div class="json-head"><span class="dot r"></span><span class="dot y"></span>'
        f'       <span class="dot g"></span><span class="fname">output.json</span></div>'
        f'  <pre class="json-body">{html.escape(pretty)}</pre>'
        '</div>'
    )
