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
# Whisper architecture — animated SVG
# ──────────────────────────────────────────────────────────────────────
def whisper_arch_html(active: str = "encoder") -> str:
    """A simplified architecture flow: mel → encoder → cross-attn → decoder → tokens."""
    blocks = [("mel-spec", "📊"), ("encoder", "⬛⬛⬛"), ("cross-attn", "⇆"),
              ("decoder", "⬛⬛⬛"), ("tokens", "abc")]
    cells = []
    for name, sym in blocks:
        cls = "block active" if name == active else "block"
        cells.append(
            f'<div class="{cls}">'
            f'  <div class="sym">{sym}</div>'
            f'  <div class="lbl">{name}</div>'
            f'</div>'
        )
        cells.append('<div class="arr">→</div>')
    cells.pop()  # drop trailing arrow
    return (
        '<div class="arch-wrap">'
        + "".join(cells) +
        '</div>'
    )


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
    # Slice romanised output proportionally
    if not src_chars:
        return '<div class="mapping empty">— no characters —</div>'
    ratio = max(1, len(roman) // max(1, len(src_chars)))
    pairs = []
    cursor = 0
    for i, ch in enumerate(src_chars):
        slice_end = min(len(roman), cursor + ratio)
        dst = roman[cursor:slice_end].strip() or "·"
        cursor = slice_end
        delay = i * 0.08
        pairs.append(
            f'<div class="pair" style="animation-delay:{delay:.2f}s">'
            f'  <div class="src">{html.escape(ch)}</div>'
            f'  <div class="arr">→</div>'
            f'  <div class="dst">{html.escape(dst[:4])}</div>'
            f'</div>'
        )
    return '<div class="mapping">' + "".join(pairs) + "</div>"


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
