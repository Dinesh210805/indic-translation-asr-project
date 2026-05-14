# Task 1 — Complete Run Guide

> End-to-end instructions: from zero to a submitted, fully executed Task 1.

---

## Overview

Task 1 runs **three Kaggle notebooks in strict order**.

```
Part A  →  sacrebleu_results.csv  →  Part B  →  engineered_features.csv  →  Part C
```

- **Part A** needs GPU (Tesla T4 or P100). Runs 5 translation models sequentially (~60–90 min).
- **Part B** is CPU-only. Loads Part A CSV, computes token metrics (~5 min).
- **Part C** is CPU-only. Loads Part B CSV, runs vocab analysis (~5 min).

No dataset needs to be downloaded manually — FLoRes-200 loads from HuggingFace automatically inside Part A.

---

## Prerequisites

### 1. Kaggle account
- Sign up at https://www.kaggle.com if you haven't
- Go to **Account → Settings → API** → click **Create New Token**
- This downloads `kaggle.json` — save it to `~/.kaggle/kaggle.json` (or `C:\Users\<you>\.kaggle\kaggle.json` on Windows)
- Run `chmod 600 ~/.kaggle/kaggle.json` on Mac/Linux

### 2. Kaggle Studio VS Code extension (already done per your setup)
- You have already installed it and signed in — nothing more needed

### 3. Git repo pushed to GitHub
The extension pushes notebooks to Kaggle from your local files. Your repo needs to be clean.

```powershell
git add -A
git commit -m "feat: complete Task 1 notebooks"
git push
```

---

## Step 1 — Run Part A (GPU required)

### Open the notebook in VS Code
Open `task1_translation_evaluation/part_a_batch_translation/part_a_translation_evaluation.ipynb`

### Push to Kaggle via the extension
In VS Code, click the Kaggle Studio extension icon → **Push Kernel**  
OR use the Kaggle CLI:

```powershell
kaggle kernels push -p task1_translation_evaluation/part_a_batch_translation
```

> The `kaggle.yml` at the root already has the correct config for all three notebooks.

### Critical settings to verify in Kaggle UI before running
After pushing, go to https://www.kaggle.com/code and open your `part-a-batch-translation` kernel:

| Setting | Required value |
|---------|---------------|
| Accelerator | **GPU T4** (or P100 — either works) |
| Internet | **On** (needed for model downloads) |
| Runtime | Python 3 |

Click **Run All** (or **Save Version → Run All Cells**).

### What Part A does
1. Installs `IndicTransToolkit` from GitHub
2. Downloads FLoRes-200 dataset from HuggingFace (~automatic, no manual step)
3. Loads and runs 5 models one at a time, clearing GPU memory between each:
   - Helsinki (~300MB) — fastest, ~3 min
   - mT5 (~580MB) — tokenization reference, ~5 min
   - NLLB-200 (~2.4GB) — ~15 min
   - IndicTrans2 (~4GB fp16) — ~20 min
   - MADLAD-400 (~6GB fp16) — ~25 min
4. Saves `sacrebleu_results.csv` with all 5 translations + BLEU scores
5. Saves 2 plots: `plots/parta_bleu_analysis.png`, color-coded results table

### Expected runtime
60–90 minutes total on T4/P100.

### Verify it worked
After the run completes, check the **Output** tab in Kaggle. You should see:
```
✓ FLoRes-200 loaded: 1011 sentence pairs
Using 100 sentences for evaluation
...
[5/5] google/madlad400-3b-mt
✓ All 5 models done
✓ Saved sacrebleu_results.csv — includes all 5 model outputs
Helsinki  : BLEU = XX.XX
NLLB-200  : BLEU = XX.XX
IndicTrans2: BLEU = XX.XX
MADLAD    : BLEU = XX.XX
Note: mT5 excluded — it is not fine-tuned for translation.
```

### Download the output CSV
In Kaggle, go to the kernel **Output** tab → download `sacrebleu_results.csv`.  
Save it to: `task1_translation_evaluation/part_a_batch_translation/sacrebleu_results.csv`

> This file is already gitignored via `data/raw/` rule — but the part_a folder path is **not** ignored, so you can commit it.

---

## Step 2 — Run Part B (CPU, no GPU needed)

### Add Part A output as a data source
Part B reads `../part_a_batch_translation/sacrebleu_results.csv`.  
On Kaggle, the simplest approach is to **add your Part A kernel as a data source** to Part B:

1. Open your `part-b-token-eda` kernel in Kaggle
2. Click **Add Data** → search for your Part A kernel output by name
3. The CSV will be available at `/kaggle/input/part-a-batch-translation/sacrebleu_results.csv`

**OR** (simpler approach): upload `sacrebleu_results.csv` as a Kaggle Dataset:
1. Go to https://www.kaggle.com/datasets → **New Dataset**
2. Upload `sacrebleu_results.csv`, name it `part-a-results`
3. In Part B kernel → Add Data → your new dataset

Then edit Cell 3 in Part B to point to the correct path:
```python
df_parta = pd.read_csv("/kaggle/input/part-a-results/sacrebleu_results.csv")
```

### Push and run
```powershell
kaggle kernels push -p task1_translation_evaluation/part_b_token_analysis
```

Settings: No GPU needed, Internet On (for tokenizer downloads).

### What Part B does
1. Downloads only tokenizers (no model weights) — much faster than Part A
2. Computes 6 metrics per sentence per model (500 rows total)
3. Saves `token_counts.csv` and `engineered_features.csv`
4. Generates 4 plots: radar chart, bubble chart, violin plot, heatmap

### Expected runtime
5–10 minutes.

### Download output
Download from Kaggle Output tab:
- `token_counts.csv` → save to `task1_translation_evaluation/part_b_token_analysis/`
- `engineered_features.csv` → same folder
- `plots/` folder → same folder

---

## Step 3 — Run Part C (CPU, no GPU needed)

### Add Part B output as a data source
Same process as Step 2 — add `engineered_features.csv` from Part B output (or upload as dataset).

Edit Cell 1 in Part C to use the correct path:
```python
token_df = pd.read_csv("/kaggle/input/part-b-results/token_counts.csv")
```

### Push and run
```powershell
kaggle kernels push -p task1_translation_evaluation/part_c_indic_token_behavior
```

### What Part C does
1. Loads tokenizers only (no model weights)
2. Classifies Tamil words as known/fragmented/unknown per model
3. Renders HTML token span visualization
4. Generates 3 plots: donut coverage, memory footprint bar chart, chars/token bar chart

### Expected runtime
5 minutes.

### Download output
Download plots from Kaggle Output tab → save to `task1_translation_evaluation/part_c_indic_token_behavior/plots/`

---

## Step 4 — Fill in Real Numbers

After all three notebooks run, open the `observations.md` files and replace the "TBD after run" / placeholder lines with actual numbers from the Kaggle output.

### Part A — `part_a_batch_translation/observations.md`
**Already complete.** Actual corpus BLEU results:
```
MADLAD-400  : 29.58
IndicTrans2 : 27.75
NLLB-200    : 24.17
Helsinki    :  8.26
mT5         : excluded (tokenizer only)
```
See `REPORT.md` in the same folder for the full post-run analysis.

### Part B — `part_b_token_analysis/observations.md`
After running, the expansion ratio table prints to the console:
```
model
Helsinki       3.21
IndicTrans2    1.84
MADLAD         2.15
NLLB-200       2.03
mT5            2.67
```
Paste those numbers into the observations.

### Part C — `part_c_indic_token_behavior/observations.md`
The vocab coverage percentages print to the console:
```
IndicTrans2    : known=72%  frag=28%  unk=0%
NLLB-200       : known=45%  frag=55%  unk=0%
...
```
Paste those into the donut chart observations.

---

## Step 5 — Commit Everything

Once all runs are done and observations filled:

```powershell
git add task1_translation_evaluation/
git commit -m "feat: add Task 1 executed outputs and observations"
git push
```

Files to include:
```
task1_translation_evaluation/part_a_batch_translation/sacrebleu_results.csv
task1_translation_evaluation/part_a_batch_translation/observations.md
task1_translation_evaluation/part_b_token_analysis/token_counts.csv
task1_translation_evaluation/part_b_token_analysis/engineered_features.csv
task1_translation_evaluation/part_b_token_analysis/observations.md
task1_translation_evaluation/part_b_token_analysis/plots/*.png
task1_translation_evaluation/part_c_indic_token_behavior/tokenization_comparison.csv
task1_translation_evaluation/part_c_indic_token_behavior/tamil_token_patterns.csv
task1_translation_evaluation/part_c_indic_token_behavior/observations.md
task1_translation_evaluation/part_c_indic_token_behavior/plots/*.png
```

---

## Submission Checklist

Before submitting, verify every item below.

### Repository
- [ ] `LICENSE` file exists at root (MIT)
- [ ] `README.md` at root — describes project, models, dataset, how to run
- [ ] `requirements.txt` at root — all dependencies listed
- [ ] `.gitignore` at root — `data/raw/`, `*.pt`, `*.bin`, `*.safetensors` excluded

### Part A
- [ ] `part_a_translation_evaluation.ipynb` — all cells present, no errors in output
- [ ] `sacrebleu_results.csv` — 100 rows, columns: `english`, `reference_tamil`, `pred_Helsinki`, `pred_mT5`, `pred_NLLB-200`, `pred_IndicTrans2`, `pred_MADLAD`, `bleu_Helsinki`, `bleu_NLLB-200`, `bleu_IndicTrans2`, `bleu_MADLAD`
- [ ] `plots/parta_bleu_analysis.png` — BLEU bar + KDE density chart
- [ ] `observations.md` — actual BLEU numbers filled in (not placeholders)

### Part B
- [ ] `part_b_token_eda.ipynb` — all cells present, no errors in output
- [ ] `token_counts.csv` — 500 rows (100 sentences × 5 models)
- [ ] `engineered_features.csv` — includes `log_expansion`, `efficiency_score`, `fragmentation_class`
- [ ] `plots/partb_radar_chart.png`
- [ ] `plots/partb_bubble_chart.png`
- [ ] `plots/partb_violin_expansion.png`
- [ ] `plots/partb_heatmap.png`
- [ ] `observations.md` — actual expansion ratios filled in

### Part C
- [ ] `part_c_indic_token_analysis.ipynb` — all cells present, no errors in output
- [ ] `plots/partc_donut_coverage.png`
- [ ] `plots/partc_memory_footprint.png`
- [ ] `plots/partc_chars_per_token.png`
- [ ] `observations.md` — actual vocab coverage % filled in

### task1_translation_evaluation/
- [ ] `README.md` — describes all three parts and data flow

---

## Troubleshooting

### Part A: CUDA out of memory
MADLAD-400 (3B, ~6GB fp16) is the most likely cause.  
Fix: make sure `clear_memory()` is called after every previous model. Also try reducing `batch_size=4` in `translate_madlad()`.

### Part A: FLoRes-200 dataset key error
If you see `KeyError: 'sentence_eng_Latn'`, the dataset version changed.  
The notebook has a runtime key detection fallback:
```python
eng_key = "sentence_eng_Latn" if "sentence_eng_Latn" in sample else "sentence"
```
If both fail, run `print(list(dataset[0].keys()))` and update the key names.

### Part A: IndicTransToolkit install fails
Make sure the install cell runs successfully:
```python
!pip install indic-trans git+https://github.com/VarunGumma/IndicTransToolkit.git -q
```
If it errors on git clone, check that Internet is turned ON in the Kaggle kernel settings.

### Part B/C: FileNotFoundError on CSV
The notebooks use relative paths like `../part_a_batch_translation/sacrebleu_results.csv`.  
On Kaggle, relative paths don't work — use the full `/kaggle/input/...` path for your data source.  
Update the `pd.read_csv(...)` path in Cell 3 (Part B) and Cell 1 (Part C) to match your Kaggle data source path.

### Kernel push fails (CLI)
```powershell
# Verify you are authenticated
kaggle config view
# Should show your username

# Re-authenticate if needed
kaggle config set -n username -v YOUR_USERNAME
```

---

## Data Flow Diagram

```
HuggingFace
openlanguagedata/flores_plus
eng_Latn-tam_Taml
        │
        ▼
┌─────────────────────────────────────────┐
│  Part A  (GPU P100, ~90 min)            │
│  5 models run sequentially              │
│  Helsinki → mT5 → NLLB → IT2 → MADLAD  │
└──────────────┬──────────────────────────┘
               │  sacrebleu_results.csv
               │  (100 rows × 5 translations + BLEU)
               ▼
┌─────────────────────────────────────────┐
│  Part B  (CPU, ~10 min)                 │
│  Tokenizers only — no model weights     │
│  Computes 6 metrics per sentence        │
└──────────────┬──────────────────────────┘
               │  token_counts.csv
               │  engineered_features.csv
               ▼
┌─────────────────────────────────────────┐
│  Part C  (CPU, ~5 min)                  │
│  Tokenizers only                        │
│  Vocab coverage, memory, chars/token    │
└─────────────────────────────────────────┘
```

---

## Quick Reference

| Item | Value |
|------|-------|
| Dataset | `openlanguagedata/flores_plus`, split `eng_Latn-tam_Taml`, first 100 devtest rows |
| Part A GPU | T4 (15.6GB VRAM) or P100 (16GB VRAM) |
| Part A runtime | ~60–90 min |
| Part B/C runtime | ~5–10 min each, CPU only |
| IndicTrans2 task prefix | none — uses `forced_bos_token_id` |
| MADLAD task prefix | `<2ta>` — **must** be prepended |
| NLLB language codes | `eng_Latn` (src), `tam_Taml` (tgt) |
| mT5 BLEU | **excluded** — not a translation model |
| mT5 in tokenization | **included** in Parts B and C |
