# Evaluation

Batch Word Error Rate (WER) evaluation of all three ASR backends against
the Mozilla Common Voice 25.0 Tamil test split.

---

## Files

| File | Purpose |
|---|---|
| `evaluate.py` | Runs all three models over downloaded Common Voice clips and writes results |
| `RESULTS_REPORT.md` | Markdown summary with per-model average WER and sample predictions (auto-generated) |
| `results.csv` | Per-clip raw predictions vs ground truth (auto-generated, not committed) |

---

## How to Run

### Prerequisites

1. **Accept dataset terms** at  
   `https://mozilladatacollective.com/datasets/cmn2gfvyp01geo107izoftfki`

2. **Add `MDC_API_KEY`** to your root `.env`  
   Get it from `https://mozilladatacollective.com/profile/credentials`

3. **Download clips**

   ```bash
   # From task2_asr_transliteration/
   python sample_inputs/download_common_voice.py
   ```

   This saves up to 50 WAV clips + `sample_inputs/common_voice/index.csv`.

4. **Run evaluation**

   ```bash
   python evaluation/evaluate.py
   ```

   Expected runtime: ~5 min (local CPU) for 50 clips × 2 local models + 50 Groq API calls.

---

## Output Format

### `results.csv`

```
filename,reference,whisper-small_transcript,whisper-small_itrans,whisper-small_wer,
         whisper-medium_transcript,...,whisper-large-v3-turbo_transcript,...
```

### `RESULTS_REPORT.md`

- Model summary table with average WER per model
- First 15 sample predictions side-by-side
- Personal recording results table (filled in manually)

---

## WER Formula

```
WER = edit_distance(reference_words, hypothesis_words) / len(reference_words)
```

Lower is better. Typical ranges for Tamil ASR:

| Model size | Expected WER (Tamil) |
|---|---|
| whisper-small | 30–50% |
| whisper-medium | 20–40% |
| whisper-large-v3-turbo | 10–25% |

Tamil is morphologically rich and low-resource compared to English — higher WER is expected.

---

## Adding Personal Recording Results

After recording your own samples from `sample_inputs/RECORDING_GUIDE.md`:

1. Test each file via the Gradio UI at `http://localhost:7860`
2. Copy the transcripts into the **Personal Recording Results** table in `RESULTS_REPORT.md`
