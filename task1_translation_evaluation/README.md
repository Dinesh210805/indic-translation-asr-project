# Task 1 — Translation Evaluation

Three-part evaluation of English → Tamil machine translation models on FLoRes-200.

## Parts

### Part A · Batch Translation (`part_a_batch_translation/`)
- Loads FLoRes-200 (first 100 sentences, `eng_Latn-tam_Taml` split)
- Runs inference with all 5 models sequentially (clears GPU memory between each)
- Computes corpus-level and sentence-level sacreBLEU scores (mT5 excluded — not an MT model)
- **Output**: `sacrebleu_results.csv` — source, references, all 5 predictions, per-sentence BLEU

### Part B · Token-Level EDA (`part_b_token_analysis/`)
- Loads `sacrebleu_results.csv` from Part A
- Loads tokenizers only (no model weights)
- Computes: expansion ratio, avg chars/token, subword fragmentation, unknown token rate
- Feature engineering: log_expansion, efficiency_score, fragmentation_class
- **Visualisations**: radar chart, bubble chart, violin plot, heatmap
- **Output**: `token_counts.csv`, `engineered_features.csv`

### Part C · Indic Token Behavior (`part_c_indic_token_behavior/`)
- Loads `engineered_features.csv` from Part B
- Analyses Tamil vocabulary coverage (known / fragmented / unknown words)
- Renders HTML token span visualiser for subword segmentation
- Computes memory footprint proxy (O(n²) attention scaling)
- **Visualisations**: donut charts (coverage), memory footprint bar chart, chars/token bar chart

## Data Flow

```
FLoRes-200
    │
    ▼
Part A ──► sacrebleu_results.csv
                │
                ▼
            Part B ──► token_counts.csv
                       engineered_features.csv
                               │
                               ▼
                           Part C ──► plots/
```

## Execution Order

Run **Part A first** (GPU required), then Part B, then Part C. Parts B and C depend on CSV output from the previous part. Each is a self-contained Kaggle notebook.

## Observations

- `part_a_batch_translation/observations.md` — BLEU scores, model ranking, key findings
- `part_b_token_analysis/observations.md` — tokenisation metrics, fragmentation analysis
- `part_c_indic_token_behavior/observations.md` — vocab coverage, memory footprint, chars/token
