"""
Batch evaluation of ASR models against Common Voice Tamil test clips.

Usage:
    # First download clips:
    python sample_inputs/download_common_voice.py

    # Then run evaluation:
    python evaluation/evaluate.py

Output:
    evaluation/results.csv        — per-clip predictions vs ground truth
    evaluation/RESULTS_REPORT.md  — summary table with WER per model
"""

import os
import sys
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from app.asr_pipeline import transcribe_audio, transcribe_with_groq
from app.transliteration import transliterate_tamil_to_latin
from app.utils import load_audio
from models.model_config import MODEL_CONFIGS

CLIPS_DIR  = Path(__file__).parent.parent / "sample_inputs" / "common_voice"
INDEX_CSV  = CLIPS_DIR / "index.csv"
OUT_DIR    = Path(__file__).parent
RESULTS_CSV = OUT_DIR / "results.csv"
REPORT_MD   = OUT_DIR / "RESULTS_REPORT.md"

MODELS_TO_EVAL = [
    "whisper-small (Local)",
    "whisper-medium (Local)",
    "whisper-large-v3-turbo ⚡ Groq Cloud",
]
SCHEME = "ITRANS"


def word_error_rate(ref: str, hyp: str) -> float:
    """Compute WER between reference and hypothesis strings."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0

    # Dynamic programming edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def transcribe(wav_path: str, model_name: str) -> str:
    cfg = MODEL_CONFIGS[model_name]
    if cfg["backend"] == "groq":
        return transcribe_with_groq(wav_path)
    audio = load_audio(wav_path)
    return transcribe_audio(audio, model_name)


def run_evaluation():
    if not INDEX_CSV.exists():
        sys.exit(
            f"index.csv not found at {INDEX_CSV}.\n"
            "Run: python sample_inputs/download_common_voice.py first."
        )

    with open(INDEX_CSV, encoding="utf-8") as f:
        clips = list(csv.DictReader(f))

    log.info("Evaluating %d clips across %d models ...", len(clips), len(MODELS_TO_EVAL))

    rows = []
    model_wers: dict[str, list[float]] = {m: [] for m in MODELS_TO_EVAL}

    for clip in clips:
        wav_path = str(CLIPS_DIR / clip["filename"])
        reference = clip["sentence"].strip()

        if not Path(wav_path).exists():
            log.warning("Clip not found, skipping: %s", wav_path)
            continue

        row = {"filename": clip["filename"], "reference": reference}

        for model_name in MODELS_TO_EVAL:
            try:
                hypothesis = transcribe(wav_path, model_name)
                wer = word_error_rate(reference, hypothesis)
                transliterated = transliterate_tamil_to_latin(hypothesis, SCHEME)
            except Exception as e:
                log.warning("[%s] %s — %s", model_name, clip["filename"], e)
                hypothesis, wer, transliterated = f"ERROR: {e}", 1.0, ""

            short_key = model_name.split()[0] + "_" + model_name.split()[1]
            row[f"{short_key}_transcript"] = hypothesis
            row[f"{short_key}_itrans"]     = transliterated
            row[f"{short_key}_wer"]        = f"{wer:.3f}"
            model_wers[model_name].append(wer)

        rows.append(row)
        log.info("Done: %s", clip["filename"])

    # Write results CSV
    if rows:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info("Results saved: %s", RESULTS_CSV)

    # Write markdown report
    write_report(rows, model_wers)


def write_report(rows: list, model_wers: dict):
    lines = [
        "# ASR Evaluation Report — Common Voice 25.0 Tamil",
        "",
        f"**Dataset:** Mozilla Common Voice 25.0 Tamil (test split)  ",
        f"**Clips evaluated:** {len(rows)}  ",
        f"**Transliteration scheme shown:** {SCHEME}  ",
        "",
        "---",
        "",
        "## Model Summary (Word Error Rate)",
        "",
        "| Model | Avg WER | Notes |",
        "|---|---|---|",
    ]

    for model_name, wers in model_wers.items():
        if wers:
            avg = sum(wers) / len(wers)
            notes = "Local CPU/GPU" if MODEL_CONFIGS[model_name]["backend"] == "local" else "Groq Cloud API"
            lines.append(f"| {model_name} | {avg:.1%} | {notes} |")

    lines += [
        "",
        "> Lower WER = better. Tamil ASR is challenging — expect 20–50% WER on local models.",
        "> Groq Cloud (whisper-large-v3-turbo) typically outperforms local small/medium.",
        "",
        "---",
        "",
        "## Sample Predictions",
        "",
        "| Clip | Reference | whisper-small | whisper-medium | Groq Cloud |",
        "|---|---|---|---|---|",
    ]

    for row in rows[:15]:  # show first 15
        ref = row["reference"][:50]
        sm  = row.get("whisper-small_transcript", "")[:50]
        md  = row.get("whisper-medium_transcript", "")[:50]
        gr  = row.get("whisper-large-v3-turbo_transcript", "")[:50]
        lines.append(f"| {row['filename']} | {ref} | {sm} | {md} | {gr} |")

    lines += [
        "",
        "---",
        "",
        "## Personal Recording Results",
        "",
        "*(Add results from your own recordings here after testing via the Gradio UI)*",
        "",
        "| File | Reference | whisper-small | Groq Cloud | Notes |",
        "|---|---|---|---|---|",
        "| p01_greeting.wav | வணக்கம், எப்படி இருக்கீங்க? | | | |",
        "| p06_work.wav | நான் கணினி அறிவியல் படிக்கிறேன். | | | |",
        "| p15_code_mix.wav | நான் Python programming கத்துக்கிட்டிருக்கேன். | | | |",
    ]

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report saved: %s", REPORT_MD)


if __name__ == "__main__":
    run_evaluation()
