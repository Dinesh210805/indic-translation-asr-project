# Indic Translation & ASR Evaluation

End-to-end evaluation suite for English → Tamil machine translation using five open-source models on the FLoRes-200 benchmark. Part A runs on Kaggle GPU (Tesla T4); Parts B and C are CPU-only and can run locally.

## Project Structure

```
indic-translation-asr-project/
├── task1_translation_evaluation/
│   ├── part_a_batch_translation/      # Batch translation + sacreBLEU scoring
│   ├── part_b_token_analysis/         # Token-level EDA and feature engineering
│   └── part_c_indic_token_behavior/   # Indic vocab coverage and memory analysis
├── data/
│   └── raw/                           # Raw datasets (gitignored)
├── kaggle.yml                         # Kaggle kernel push config
├── requirements.txt
└── LICENSE
```

## Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| IndicTrans2 (`ai4bharat/indictrans2-en-indic-1B`) | 1B | Indic-specific; uses IndicProcessor |
| NLLB-200 (`facebook/nllb-200-distilled-600M`) | 600M | Multilingual; src/tgt lang codes required |
| Helsinki MarianMT (`Helsinki-NLP/opus-mt-en-dra`) | ~74M | EN→Dravidian multilingual (ta/kn/ml/te) |
| MADLAD-400 (`google/madlad400-3b-mt`) | 3B | Requires `<2ta>` task prefix |
| mT5-base (`google/mt5-base`) | 580M | Tokenization analysis only — not a translation model |

## Dataset

**FLoRes-200** — `openlanguagedata/flores_plus`, split `eng_Latn-tam_Taml`, first 100 sentences (devtest split).

## How to Run

The pipeline runs in three sequential steps. Part A requires a GPU and runs on Kaggle. Parts B and C are CPU-only and can run either on Kaggle or locally.

### Part A — Kaggle GPU (required)
GPU is needed to load and run the 600M–3B parameter translation models.
```
kaggle kernels push -p task1_translation_evaluation/part_a_batch_translation
```
Download `sacrebleu_results.csv` from the kernel output before running Part B.

### Part B — local or Kaggle CPU
```bash
# local
cd task1_translation_evaluation/part_b_token_analysis
jupyter notebook part_b_token_eda.ipynb

# or Kaggle
kaggle kernels push -p task1_translation_evaluation/part_b_token_analysis
```
Requires `sacrebleu_results.csv` in the `part_a_batch_translation/` directory.

### Part C — local or Kaggle CPU
```bash
# local
cd task1_translation_evaluation/part_c_indic_token_behavior
jupyter notebook part_c_indic_token_analysis.ipynb

# or Kaggle
kaggle kernels push -p task1_translation_evaluation/part_c_indic_token_behavior
```
Requires `token_counts.csv` from Part B output.

You can also use the Kaggle Studio VS Code extension with `kaggle.yml` at the repo root for all three parts.

## Output Artifacts

| File | Written by | Read by |
|------|-----------|---------|
| `part_a_batch_translation/sacrebleu_results.csv` | Part A | Part B |
| `part_b_token_analysis/token_counts.csv` | Part B | Part C |
| `part_b_token_analysis/engineered_features.csv` | Part B | — |
| `part_c_indic_token_behavior/tamil_token_patterns.csv` | Part C | — |
| `part_c_indic_token_behavior/tokenization_comparison.csv` | Part C | — |
| `part_*/plots/*.png` | each part | — |
| `part_a_batch_translation/REPORT.md` | — | human reference |
| `part_b_token_analysis/REPORT.md` | — | human reference |
| `part_c_indic_token_behavior/REPORT.md` | — | human reference |
| `part_*/observations.md` | — | human reference |

## Requirements

See `requirements.txt`. Key packages: `transformers`, `sacrebleu`, `datasets`, `sentencepiece`, `torch`.

Install IndicTrans2 toolkit separately:
```bash
pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git
```

## License

MIT — see `LICENSE`.
