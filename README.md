# Indic Translation & ASR Evaluation

End-to-end evaluation suite for English → Tamil machine translation using five open-source models on the FLoRes-200 benchmark. Built for Kaggle GPU P100.

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
| Helsinki MarianMT (`Helsinki-NLP/opus-mt-en-ta`) | ~74M | Compact EN→TA specialist |
| MADLAD-400 (`google/madlad400-3b-mt`) | 3B | Requires `<2ta>` task prefix |
| mT5-base (`google/mt5-base`) | 580M | Tokenization analysis only — not a translation model |

## Dataset

**FLoRes-200** — `facebook/flores`, split `eng_Latn-tam_Taml`, first 100 sentences (dev split).

## How to Run

Notebooks are designed to run sequentially on Kaggle:

1. **Part A** — GPU enabled (P100), internet enabled
   ```
   kaggle kernels push -p task1_translation_evaluation/part_a_batch_translation
   ```
2. **Part B** — CPU, internet enabled (tokenizer downloads only)
   ```
   kaggle kernels push -p task1_translation_evaluation/part_b_token_analysis
   ```
3. **Part C** — CPU, internet enabled (tokenizer downloads only)
   ```
   kaggle kernels push -p task1_translation_evaluation/part_c_indic_token_behavior
   ```

Or use the Kaggle Studio VS Code extension with `kaggle.yml` at the repo root.

## Output Artifacts

| File | Written by | Read by |
|------|-----------|---------|
| `part_a_batch_translation/sacrebleu_results.csv` | Part A | Part B, Part C |
| `part_b_token_analysis/token_counts.csv` | Part B | — |
| `part_b_token_analysis/engineered_features.csv` | Part B | Part C |
| `part_*/plots/*.png` | each part | — |

## Requirements

See `requirements.txt`. Key packages: `transformers`, `sacrebleu`, `datasets`, `sentencepiece`, `torch`.

Install IndicTrans2 toolkit separately:
```bash
pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git
```

## License

MIT — see `LICENSE`.
