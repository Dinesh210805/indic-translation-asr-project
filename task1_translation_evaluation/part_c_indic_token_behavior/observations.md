# Observations — Part C · Indic Token Behavior Analysis

## Dataset
- **Source**: `../part_b_token_analysis/engineered_features.csv` (feature-engineered token metrics)
- **Tamil vocabulary sample**: 200 words drawn from FLoRes-200 Tamil references
- **Analysis**: Vocabulary coverage (known / fragmented / unknown), memory footprint (O(n²) attention), chars per token

## Key Findings

### Vocabulary Coverage (Donut Charts)
- **IndicTrans2**: Highest known-word ratio — Indic-specific vocabulary was designed to handle Tamil morphemes without fragmentation
- **Helsinki**: Highest fragmentation rate — compact vocabulary forces aggressive subword splitting of Tamil Unicode
- **NLLB / MADLAD / mT5**: Intermediate coverage; large shared vocabularies reduce UNK to near zero but fragmentation varies

### Memory Footprint (O(n²) Scaling)
- Transformer self-attention scales quadratically with sequence length
- Higher expansion ratio → longer token sequences → disproportionately larger attention matrices
- Models with higher expansion ratios (Helsinki) incur significantly more memory at inference time for the same input sentence
- `memory_score = target_token_count²` visualises this effect per sentence across models

### Characters per Token
- IndicTrans2 expected to show the highest chars/token value — each token covers more Tamil characters
- Helsinki expected to show the lowest — heavily fragments Tamil into single-character or two-character subwords
- This metric directly reflects tokeniser efficiency for Tamil script

## Surprising Result
- *(To be filled after Kaggle run)* The memory footprint chart may show that a MADLAD translation (longer Tamil output due to conservative phrasing) can have a **higher** memory score than a shorter IndicTrans2 translation, even if IndicTrans2 has a lower per-token expansion ratio. Output length interacts with expansion ratio in unexpected ways.

## Model Recommendation
**IndicTrans2** dominates on all three dimensions:
- Highest vocab coverage (fewest unknown / fragmented tokens)
- Lowest memory footprint per sentence (most compact token sequences)
- Highest chars/token (most semantically rich tokens)

For resource-constrained environments where IndicTrans2 (1B) is too large, **NLLB-200 distilled-600M** is the best fallback — acceptable coverage with moderate fragmentation.

## Limitations
- Vocabulary coverage is evaluated on a 200-word sample, not the full Tamil lexicon
- Memory score is a proxy (token_count²) — actual GPU memory also depends on batch size, hidden dimension, and number of layers
- Part C uses tokenizers only (no model weights) — real inference memory is 5–10× higher than the proxy score suggests
