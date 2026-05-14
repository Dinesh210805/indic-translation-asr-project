# Observations — Part A · Batch Translation Evaluation

## Dataset
- **Source**: FLoRes-200 (`facebook/flores`, `eng_Latn-tam_Taml` split)
- **Sentences**: First 100 English sentences with Tamil references
- **Platform**: Kaggle GPU P100 (16 GB VRAM)

## Models Evaluated
| Model | Type | Parameters | BLEU (approx.) |
|-------|------|-----------|----------------|
| IndicTrans2 | Encoder-decoder (IndicTrans) | ~1B | TBD after run |
| NLLB-200 | mBART-style multilingual | 600M distilled | TBD after run |
| Helsinki MarianMT | Marian encoder-decoder | ~74M | TBD after run |
| MADLAD-400 | T5-based multilingual | 3B | TBD after run |
| mT5-base | T5 pre-training (no fine-tune) | 580M | excluded (not MT) |

## Key Findings
- **Best BLEU**: IndicTrans2 expected to outperform — the only model specifically trained on Indic language pairs with IndicProcessor pre/post-processing
- **Lowest BLEU**: mT5 intentionally excluded from BLEU; among MT models Helsinki is expected to be weakest given its compact 74M parameter count
- **MADLAD quirk**: Requires `<2ta>` task prefix to direct the model to Tamil; omitting it causes random-language output

## Surprising Result
- *(To be filled after Kaggle run)* Initial expectation: MADLAD-400 (3B) outperforms NLLB-600M due to scale. Actual ranking may differ — architecture and training data quality can matter more than raw parameter count for low-resource Indic pairs.

## Model Recommendation
**IndicTrans2** for production Tamil translation. Reasons:
1. Trained specifically on Indic language pairs (not generic multilingual)
2. Uses IndicProcessor for script normalization and detokenization
3. Achieves highest BLEU on FLoRes-200 benchmark per published results

## Limitations
- Only 100 sentences evaluated (FLoRes-200 dev split subset) — results may not generalise across domains
- All models run with default decoding (greedy / beam=4); no hyperparameter tuning
- MADLAD 3B inference is slow on P100 — may time out on longer sentence sets
- mT5 BLEU excluded; its raw outputs are unintelligible Tamil text (pre-training artefacts, not translations)
